"""Cross-agent delegate-tool shims.

Holds the four ``delegate_*`` tool-runner shims that let an agent
dispatch a turn against another agent (``/claude``, ``/codex``,
``/antigravity``, ``/delegate <name>``). Each shim loads the named child
agent through :func:`factory.load_agent`, opens / resumes a session
keyed off the parent sid, drives one turn via :meth:`Agent.run_turn`,
and assembles a :class:`ToolResult` whose ``content`` is the same
shape the legacy delegate runner emitted (so model transcript replay
is byte-stable across the cutover).

Renamed from ``legacy_compat`` at 2026-05-15 once the sidecar /
session-record migration helpers and the tool-call inline-marker
backfill that justified the old name were deleted — the remaining
content is the live cross-agent dispatch path, not legacy support.
"""

from __future__ import annotations

import asyncio
import re
import threading
from collections.abc import Callable
from typing import Any

from sharedagents_agent.resume_brief import (
    build_resume_brief,
    extract_next_step,
    render_resume_brief,
    write_resume_brief_sidecar,
)
from sharedagents_core import oscompat

_TASK_TURN_LOCKS: dict[tuple[int, str], list[Any]] = {}
_TASK_TURN_LOCKS_GUARD = threading.Lock()


def _retain_task_turn_lock(path: Any) -> tuple[tuple[int, str], asyncio.Lock]:
    """Retain one event-loop-local lock for a child task turn."""
    key = (id(asyncio.get_running_loop()), str(path))
    with _TASK_TURN_LOCKS_GUARD:
        entry = _TASK_TURN_LOCKS.get(key)
        if entry is None:
            entry = [asyncio.Lock(), 0]
            _TASK_TURN_LOCKS[key] = entry
        entry[1] += 1
        return key, entry[0]


def _drop_task_turn_lock(
    key: tuple[int, str],
    lock: asyncio.Lock,
) -> None:
    """Release a registry reference and remove an unused lock."""
    with _TASK_TURN_LOCKS_GUARD:
        entry = _TASK_TURN_LOCKS.get(key)
        if entry is None or entry[0] is not lock:
            return
        entry[1] -= 1
        if entry[1] <= 0:
            _TASK_TURN_LOCKS.pop(key, None)


async def _acquire_task_turn_lock(
    path: Any,
) -> tuple[int, tuple[int, str], asyncio.Lock]:
    """Acquire same-process serialization, then the cross-process file lock."""
    import os

    key, lock = _retain_task_turn_lock(path)
    acquired = False
    fd = -1
    try:
        await lock.acquire()
        acquired = True
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        while True:
            try:
                oscompat.file_lock(fd, nonblocking=True)
                break
            except BlockingIOError:
                # A detached process owns this task turn. Poll without blocking
                # a worker thread so cancellation immediately reaches the
                # cleanup below and never leaves a future flock acquisition
                # running against a closed/reused fd.
                await asyncio.sleep(0.05)
        return fd, key, lock
    except BaseException:
        if fd >= 0:
            os.close(fd)
        if acquired:
            lock.release()
        _drop_task_turn_lock(key, lock)
        raise


def _release_task_turn_lock(
    fd: int,
    key: tuple[int, str],
    lock: asyncio.Lock,
) -> None:
    """Release the file and process-local locks without leaking the registry."""
    import os

    try:
        oscompat.file_unlock(fd)
    finally:
        try:
            os.close(fd)
        finally:
            lock.release()
            _drop_task_turn_lock(key, lock)


def _resolve_model_override(child, model, subagent_type):
    """Honor a subagent ``model`` override only when it resolves to one of the
    child's *configured* models (alias or id) or is a raw endpoint URL;
    otherwise drop it so the child uses its own default.

    A delegating parent — especially a small local model — can hallucinate a
    ``model`` override the child's backend rejects (observed: main passing
    "sonnet" to a deepseek-backed coder → HTTP 400 → wasteful duplicate retry).
    Codex OAuth rows are folded into ``cfg.models`` once at config load /
    reload, so this path only reads that list — it does not rediscover.
    """
    if not model:
        return None
    if str(model).strip().lower().startswith(("http://", "https://")):
        return model
    known: set[str] = set()
    cfg = getattr(child, "cfg", None)
    for _m in getattr(cfg, "models", None) or []:
        get = _m.get if hasattr(_m, "get") else (lambda k, _o=_m: getattr(_o, k, None))
        if get("alias"):
            known.add(get("alias"))
        if get("id"):
            known.add(get("id"))
    if model not in known:
        import logging as _lg

        _lg.getLogger("sharedagents.delegate").info(
            "dropping unresolvable model override %r for subagent %r (not in child config) — using child default",
            model,
            subagent_type,
        )
        return None
    return model


def _parent_session_model_and_agent(
    sessions_dir: Any,
    parent_sid: str,
) -> tuple[str, str]:
    """Read a parent session's persisted ``model`` + ``agent_name``.

    Same unified ``sessions.db`` as :func:`_parent_cost_attribution` — works
    across agents. Returns ``("", "")`` when the parent row is missing.
    """
    if not parent_sid:
        return "", ""
    try:
        from pathlib import Path

        from sharedagents_session.session_index import SessionIndex, connect_sessions_db

        sdir = Path(sessions_dir)
        SessionIndex(sdir).tail(parent_sid)
        con = connect_sessions_db(sdir / "sessions.db")
        try:
            row = con.execute(
                "SELECT model, agent_name FROM sessions_meta WHERE session_hash = ?",
                (parent_sid,),
            ).fetchone()
        finally:
            con.close()
        if not row:
            return "", ""
        d = dict(row)
        return (
            str(d.get("model") or "").strip(),
            str(d.get("agent_name") or "").strip(),
        )
    except Exception:  # noqa: BLE001
        return "", ""


def _delegating_agent_name(sessions_dir: Any, parent_sid: str) -> str:
    """The agent issuing this delegation, as it is named in its session.

    The gateway stamps ``SHAREDAGENTS_AGENT_NAME`` with the acting agent on every
    tool runner it spawns (and the detached background runner inherits it),
    so it is the agent active in the parent session at dispatch time. An
    in-process dispatch without that env falls back to the parent session's
    recorded agent. ``""`` when neither is known (no author is stamped).
    """
    import os

    name = (os.environ.get("SHAREDAGENTS_AGENT_NAME") or "").strip()
    if name:
        return name
    return _parent_session_model_and_agent(sessions_dir, parent_sid)[1]


def _topmost_ancestor_sid(sessions_dir: Any, sid: str) -> str:
    """Walk ``parent_session_id`` to the operator-facing root session."""
    if not sid:
        return ""
    try:
        from pathlib import Path

        from sharedagents_session.session_index import SessionIndex, connect_sessions_db

        sdir = Path(sessions_dir)
        SessionIndex(sdir).tail(sid)
        con = connect_sessions_db(sdir / "sessions.db")
        try:
            seen: set[str] = set()
            cur = sid
            top = sid
            while cur and cur not in seen:
                seen.add(cur)
                top = cur
                row = con.execute(
                    "SELECT parent_session_id FROM sessions_meta WHERE session_hash = ?",
                    (cur,),
                ).fetchone()
                if not row:
                    break
                cur = str(dict(row).get("parent_session_id") or "").strip()
            return top
        finally:
            con.close()
    except Exception:  # noqa: BLE001
        return sid


def _operator_session_for_model_inherit(
    sessions_dir: Any,
    parent_sid: str,
) -> str:
    """Session whose model selection governs the whole delegation tree.

    Prefer the persisted cost-root (true operator session). Fall back to
    walking ``parent_session_id`` so nested explore→general still sees
    Jarvis's ``/model`` override even when a mid-level child row still
    carries that child's own default model.
    """
    if not parent_sid:
        return ""
    cost_root, _, _ = _parent_cost_attribution(sessions_dir, parent_sid)
    if cost_root:
        return cost_root
    return _topmost_ancestor_sid(sessions_dir, parent_sid) or parent_sid


#: Engine labels a CLI session may carry in ``sessions_meta.model`` — the
#: engine's own display name (e.g. the codex CLI engine stamps "codex" on
#: sessions created before the wire-id stamp fix). A label is NEVER a real
#: model id and NEVER an operator override, so it must not propagate to
#: delegated children (where the codex spawn gate would refuse it). See
#: ``_is_agent_default_model``.
_CLI_ENGINE_LABELS: dict[str, str] = {
    "claude_cli": "claude",
    "codex_cli": "codex",
    "antigravity_cli": "antigravity",
    "zcode_cli": "zcode",
}


def _is_agent_default_model(cfg: Any, session_model: str) -> bool:
    """True when ``session_model`` is the agent's configured default.

    Compares against ``upstream_model``, ``upstream_alias``, and the matching
    ``models[]`` row so wire-id and alias forms both count as "on default".
    Empty session model is treated as default (no operator override).

    A session stamped with the ENGINE LABEL (the
    pre-fix codex stamp "codex") counts as default too — the label is the
    engine's own display name, not a model the operator picked, so it must
    never propagate as an override to children.
    """
    sm = str(session_model or "").strip()
    if not sm:
        return True
    _engine = str(getattr(cfg, "engine", "") or "").strip()
    if _engine in _CLI_ENGINE_LABELS and sm == _CLI_ENGINE_LABELS[_engine]:
        return True
    default_id = str(getattr(cfg, "upstream_model", "") or "").strip()
    default_alias = str(getattr(cfg, "upstream_alias", "") or "").strip()
    if sm == default_id or (default_alias and sm == default_alias):
        return True
    for _m in getattr(cfg, "models", None) or []:
        get = _m.get if hasattr(_m, "get") else (lambda k, _o=_m: getattr(_o, k, None))
        alias = str(get("alias") or "").strip()
        mid = str(get("id") or "").strip()
        if default_alias and alias == default_alias and sm in (alias, mid):
            return True
        if default_id and mid == default_id and sm in (alias, mid):
            return True
    return False


def _is_native_local_engine(child: Any) -> bool:
    """True when the child runs the in-process ``local_llm`` engine.

    Only native local agents inherit the parent session's ``/model``
    override. External engines (``codex_cli``, ``claude_cli``, ACP
    variants, …) keep their own default — a foreign id like
    ``deepseek-v4-flash`` lives in the shared global ``models[]``, so
    :func:`_resolve_model_override` would otherwise pass it through and
    the spawn gate / ChatGPT backend would refuse it (Jarvis→codex
    deepseek incident).
    """
    try:
        kind = getattr(getattr(child, "engine", None), "kind", None)
        value = getattr(kind, "value", kind)
        return str(value or "") == "local_llm"
    except Exception:  # noqa: BLE001
        return False


def _child_pins_own_model(child: Any) -> bool:
    """True when the OPERATOR gave this child its own ``upstream_model``.

    ``AgentConfig`` stamps ``_upstream_model_is_per_agent`` only when the key is
    present in the child's OWN ``config.json`` (``agent.py``), never when the
    value was merged down from the global deployment config — so this is
    precisely "someone deliberately chose a model for this agent", not "this
    agent happens to have a model".

    Used to stop parent-model inheritance for such a child. A per-agent choice
    is durable and deliberate; a parent's ``/model`` override is transient to
    that one session. Letting the latter win would drag a pinned child onto the
    calling agent's model — fatal for the plan/build/review loop, whose whole
    premise is that the reviewer does NOT run on the builder's model.
    ``_resolve_model_override`` cannot catch this: it only checks the override
    is one of the child's configured rows, not that the child chose one.
    """
    try:
        return bool(getattr(getattr(child, "cfg", None), "_upstream_model_is_per_agent", False))
    except Exception:  # noqa: BLE001 — inheritance must never break delegation
        return False


def _inherit_parent_model_if_overridden(
    child: Any,
    parent_sid: str,
) -> str | None:
    """Propagate the operator session's model when it is a non-default override.

    Rule: if the operator-facing parent (cost-root / topmost ancestor) is
    still on that agent's configured default, return ``None`` so the child
    keeps *its own* default. If the operator switched that session
    off-default (``/model`` / picker), return that model so every **native
    local** subagent in the tree — including nested grandchildren — uses
    it for this session only. External-engine children always return
    ``None`` (their profile default / spawn pin wins). Never mutates
    agent config defaults.

    A child the operator pinned to its own model also returns ``None`` —
    see :func:`_child_pins_own_model`.

    Fail-open to ``None`` on any lookup error so a missing parent never
    blocks delegation. Explicit ``Agent(..., model=X)`` still wins at the
    call site before this helper runs.
    """
    if not parent_sid:
        return None
    # External engines never inherit a parent /model override — only
    # native local_llm children do. See ``_is_native_local_engine``.
    if not _is_native_local_engine(child):
        return None
    # A child with an operator-chosen model keeps it (reviewer independence).
    if _child_pins_own_model(child):
        return None
    sessions_dir = getattr(child, "sessions_dir", None)
    if sessions_dir is None:
        return None
    source_sid = _operator_session_for_model_inherit(sessions_dir, parent_sid)
    parent_model, parent_agent = _parent_session_model_and_agent(
        sessions_dir,
        source_sid,
    )
    if not parent_model or not parent_agent:
        return None
    try:
        from sharedagents_agent.agent import AgentConfig

        parent_cfg = AgentConfig.load(parent_agent)
        if _is_agent_default_model(parent_cfg, parent_model):
            return None
        return parent_model
    except Exception:  # noqa: BLE001
        return None


def _child_engine_kind(args: dict[str, Any], profile_name: str) -> str:
    """Engine id for a delegated sibling (``opencode_acp``, ``cursor_acp``, …)."""
    import json
    from pathlib import Path

    name = str(args.get("name") or profile_name or "").strip()
    if not name:
        return ""
    try:
        from sharedagents_agent.agent import agents_root

        cfg_path = agents_root() / name / "config.json"
    except Exception:  # noqa: BLE001
        cfg_path = Path.home() / ".sharedagents" / "agents" / name / "config.json"
    try:
        payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("engine") or "").strip()


def _engine_advertises_native_ask(engine: str) -> bool:
    """True when the engine's published ACP catalogue includes a native ``ask``."""
    if not engine:
        return True
    from sharedagents_agent.acp_mode_dict import ACP_MODE_CATALOGUE

    rows = ACP_MODE_CATALOGUE.get(engine, ())
    return any(str(row.get("id") or "").strip().lower() == "ask" for row in rows)


def _resolve_subagent_mode(args: dict[str, Any], profile_name: str) -> str:
    """The permission mode a delegated subagent runs its turn in.

    ``plan`` is a USER-FACING mode only. A delegated subagent always
    EXECUTES its task (research, edits, shell) on behalf of the plan — it must
    never inherit plan-mode read-only gating, which would deny its mcp__web__search,
    research scripts, etc. A subagent's safety comes from its tool allowlist +
    the non-overridable FORBIDDEN floor, not from plan mode. So any plan-mode
    source (an explicit ``mode="plan"`` arg, or a profile that defaults to
    ``mode="plan"`` such as a hidden util) coerces to ``auto``.

    ``ask`` is also user-facing. Engines that do not advertise a native
    ``ask`` (OpenCode: ``plan`` + ``build``) raise
    ``unsupported OpenCode mode: ask``. Clamp to that engine's default from
    ``default_requested_mode_for_engine`` (OpenCode ``auto`` → native
    ``build``). Cursor keeps inherited ``ask`` because its catalogue has it.
    """
    import os

    from sharedagents_agent.acp_mode_dict import default_requested_mode_for_engine

    mode = str(args.get("mode") or os.environ.get("SHAREDAGENTS_PERMISSION_MODE") or "auto")
    engine = _child_engine_kind(args, profile_name)
    if profile_name:
        from sharedagents_agent.agent_profiles import resolve_profile

        prof = resolve_profile(profile_name)
        # Native roster: the profile owns the mode (all auto). External
        # engines use the catalogue default — a stale profile ask/plan
        # must not overwrite the inherited or omitted mode.
        if (
            prof is not None
            and prof.name
            in {
                "jarvis",
                "plan",
                "build",
                "review",
                "explore",
                "general",
                "research",
            }
            or prof is not None
            and not engine
        ):
            mode = prof.mode
    if mode == "plan":
        return "auto"
    if mode == "ask" and engine and _engine_advertises_native_ask(engine):
        return "ask"
    if engine and mode not in advertised_requested_modes_for(engine):
        return default_requested_mode_for_engine(engine)
    if mode == "ask":
        return "auto"
    return mode


def advertised_requested_modes_for(engine: str) -> tuple[str, ...]:
    from sharedagents_agent.acp_mode_dict import advertised_requested_modes

    return advertised_requested_modes(engine)


_CAPTURE_PATTERN_TOOL = "mcp__delegate__capture_pattern"


def _delegation_success_hint(
    *,
    ok: bool,
    body: str,
    subagent_type: str,
    prompt: str,
    grants: Any = None,
) -> str:
    """Optional footer nudging the orchestrator to capture reusable prompt patterns.

    Never for a Web user's turn that cannot call the tool.
    """
    if not ok or len(body.strip()) < 200:
        return ""
    if grants is not None:
        from sharedagents_core.access_grants import filter_tool_names

        if not filter_tool_names({_CAPTURE_PATTERN_TOOL}, grants):
            return ""
    st = (subagent_type or "").strip().lower()
    # Don't nudge the orchestrator itself (jarvis) or the hidden runtime utilities.
    # (`main` retired 2026-07-21 — jarvis is the sole orchestrator.)
    if st in ("", "jarvis", "compaction", "title", "summary", "teacher"):
        return ""
    return (
        "\n\n---\n"
        "[hint] Delegation succeeded. If this prompt shape is reusable, call "
        "`mcp__delegate__capture_pattern(subagent_type=..., title=..., "
        "prompt_template=...)` — the runtime merges near-duplicates."
    )


def _parent_live_goal_id(sessions_dir: Any, parent_sid: str) -> str:
    """The goal a new child of ``parent_sid`` would be charged to, when the
    caller did not thread one."""
    from sharedagents_goal.goal_service import charged_goal_id

    p_root, p_goal, p_own_goal = _parent_cost_attribution(sessions_dir, parent_sid)
    return charged_goal_id(
        session_hash=parent_sid,
        cost_root_session_id=p_root,
        cost_ancestor_goal_id=p_goal,
        goal_id=p_own_goal,
    )


def _parent_cost_attribution(sessions_dir: Any, parent_sid: str) -> tuple[str, str, str]:
    """Read a parent session's PERSISTED cost attribution from the shared
    sessions.db: ``(cost_root_session_id, cost_ancestor_goal_id, goal_id)``.

    CROSS-AGENT IS FINE (main→claude/codex): ``sessions_dir`` is the ONE
    ``sessions_root()`` shared by EVERY agent (per-agent session dirs were retired
    in the 2026-05-26 layout; ``Agent.sessions_dir`` returns the unified path,
    attribution keyed by ``sessions_meta.agent_name``). So the parent row is in
    this db regardless of which agent owns it. We read it directly rather than via
    ``Agent.get_session`` (whose cross-agent guard would reject a different-agent
    parent). A nested grandchild thus inherits the TRUE root + ancestor goal, not
    its immediate parent — for EVERY delegation path (single/batch/async/gateway),
    no per-spawn-path cost threading required. (Reviewer note: findings that
    assume a PER-AGENT sessions.db are stale — see
    test_cost_attribution_resolves_parent_across_agents.)
    """
    if not parent_sid:
        return "", "", ""
    try:
        from pathlib import Path

        from sharedagents_session.session_index import SessionIndex, connect_sessions_db

        sdir = Path(sessions_dir)
        SessionIndex(sdir).tail(parent_sid)  # fold latest header writes
        con = connect_sessions_db(sdir / "sessions.db")
        try:
            row = con.execute(
                "SELECT cost_root_session_id, cost_ancestor_goal_id, goal_id FROM sessions_meta WHERE session_hash = ?",
                (parent_sid,),
            ).fetchone()
        finally:
            con.close()
        if not row:
            return "", "", ""
        d = dict(row)
        return (
            str(d.get("cost_root_session_id") or ""),
            str(d.get("cost_ancestor_goal_id") or ""),
            str(d.get("goal_id") or ""),
        )
    except Exception:  # noqa: BLE001
        return "", "", ""


def _persist_child_cost_attribution(child, child_sid, parent_sid, args, sess=None):
    """Compute + PERSIST a delegate child's cost attribution (root session +
    ancestor goal) so its spend rolls up to the parent's TRUE root/goal.

    Runs for a newly-created AND a REUSED child: the slash/direct delegate path
    reuses a persisted sidecar child session, so one created under an OLDER goal
    would otherwise keep stale (or missing) pointers and self-root / charge the
    wrong goal. ``sess`` is the in-memory session on the
    create path (set the fields there too so its first turn carries them); None
    on reuse (persist via the header writer only — the child reloads from disk).
    Prefer explicitly-threaded ``_sharedagents_cost_*`` args (exact for nested
    delegation); else inherit the parent's PERSISTED attribution so a grandchild
    attributes to the TRUE root, not its immediate parent.
    """
    try:
        cost_root = str(args.get("_sharedagents_cost_root_session_id") or "")
        cost_goal = str(args.get("_sharedagents_cost_ancestor_goal_id") or "")
        if (not cost_root or not cost_goal) and parent_sid:
            p_root, p_goal, p_own_goal = _parent_cost_attribution(
                child.sessions_dir,
                parent_sid,
            )
            if not cost_root:
                cost_root = p_root or parent_sid
            if not cost_goal:
                from sharedagents_goal.goal_service import charged_goal_id

                cost_goal = charged_goal_id(
                    session_hash=parent_sid,
                    cost_root_session_id=p_root,
                    cost_ancestor_goal_id=p_goal,
                    goal_id=p_own_goal,
                )
            if not cost_goal:
                from sharedagents_goal.goal_registry import get_goal_registry

                cost_goal = get_goal_registry().goal_id_for_root_session(cost_root or parent_sid) or ""
        if cost_root or cost_goal:
            if sess is not None:
                sess.cost_root_session_id = cost_root
                sess.cost_ancestor_goal_id = cost_goal
            # Persist via the writer + flush so the child's own run_turn (which
            # reloads the session) carries the attribution before its first turn.
            from sharedagents_session.session_writer import (
                close_session_writer,
                get_session_writer,
            )

            w = get_session_writer(
                child.sessions_dir,
                child_sid,
                engine=child.engine.kind.value,
            )
            w.header_update(
                patch={
                    "cost_root_session_id": cost_root,
                    "cost_ancestor_goal_id": cost_goal,
                }
            )
            close_session_writer(child.sessions_dir, child_sid)
    except Exception:  # noqa: BLE001
        pass


def _orchestration_depth_refusal(short_name: str, reservation) -> str | None:
    """Return a refusal message when this NEW delegation would create a level
    beyond the orchestration cap (``MAX_DELEGATION_DEPTH``), else ``None``.

    THE single, engine-agnostic 3-level enforcement, shared by every reservation
    site (the in-process chokepoint ``_drive_child_agent_turn`` AND the detached
    background path ``_maybe_background``). It closes the gap where the cap was
    enforced ONLY by the ``local_llm`` tool-strip (``_runtime_allowed_tools``):
    a sharedagents agent driven by a CLI/ACP engine (claude/codex/antigravity) has no
    tool-strip removing its ``Agent``/``delegate_*`` tools, so a leaf could still
    initiate a 4th level — which the sandbox broker PERMITS (its RESOURCE cap
    only refuses past ``DEPTH``=3, a 5th level). We do NOT weaken that resource
    booking (spec §6.4); we only refuse the new delegation itself.

    ``reservation is None`` (a precounted child TOP turn) is never refused here —
    that turn was already depth-checked by the parent that spawned it. Otherwise
    the child's would-be depth is the broker-computed ``reservation.agent_depth``
    AND at least ``env_depth + 1`` (the delegating agent's own depth, seeded on
    spawn via ``SHAREDAGENTS_SANDBOX_AGENT_DEPTH`` — the SAME source the local_llm
    strip reads); we take the deeper so an in-process reservation whose lineage
    under-counts can never dodge the cap.
    """
    if reservation is None:
        return None
    import os as _os

    from sharedagents_agent import sandbox_broker

    try:
        _env_depth = int(_os.environ.get("SHAREDAGENTS_SANDBOX_AGENT_DEPTH", "0") or "0")
    except (TypeError, ValueError):
        _env_depth = 0
    child_depth = max(int(getattr(reservation, "agent_depth", 0)), _env_depth + 1)
    if child_depth <= sandbox_broker.MAX_DELEGATION_DEPTH:
        return None
    return (
        f"[delegate {short_name}: refused — delegation-depth cap "
        f"({sandbox_broker.MAX_DELEGATION_DEPTH + 1} levels) reached. This agent "
        f"is a LEAF (delegation depth {child_depth - 1}); it must do the work "
        f"itself instead of spawning a further level.]"
    )


def _delegate_parent_sid(args: dict[str, Any]) -> str:
    """Resolve the trusted parent session threaded by the gateway."""
    import os

    return str(args.get("_sharedagents_parent_session_id") or os.environ.get("SHAREDAGENTS_SESSION_ID") or "")


def _delegate_sidecar_path(child: Any, parent_sid: str, call_id: str) -> Any:
    """Per-call Agent sidecar; per-engine sidecar for named slash commands."""
    base = child.delegate_sidecar_path(parent_sid, child.engine.kind)
    if not call_id:
        return base
    safe_cid = call_id.replace("/", "_").replace(".", "_")
    tail = base.name.split(".", 1)[1] if "." in base.name else base.name
    return base.with_name(f"{parent_sid}.subagent.{safe_cid}.{tail}")


def _task_metadata_path(child: Any, child_sid: str) -> Any:
    from pathlib import Path

    root = Path(child.sessions_dir).resolve()
    path = (root / f"{child_sid}.agent_task.json").resolve()
    if path.parent != root:
        raise ValueError("task metadata path escapes sessions root")
    return path


def _valid_task_id(value: str) -> bool:
    """Canonical sharedagents session-id syntax; rejects paths before any I/O."""
    import re

    return bool(re.fullmatch(r"\d{8}T\d{12}-[0-9a-f]{12}", value or ""))


def _detect_git_context(cwd: str) -> tuple[str, str]:
    """Best-effort (worktree root, branch) for the resume brief."""
    import subprocess

    base = cwd or ""
    if not base:
        import os as _os

        base = _os.getcwd()
    try:
        out = subprocess.run(
            ["git", "-C", base, "rev-parse", "--show-toplevel", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if out.returncode == 0:
            lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
            return (lines[0] if lines else "", lines[1] if len(lines) > 1 else "")
    except Exception:  # noqa: BLE001
        pass
    return "", ""


def _child_turn_id(child: Any, child_sid: str):
    """The child session's latest physical turn id (ordinal as fallback)."""
    try:
        session = child.get_session(child_sid)
        turns = getattr(session, "turns", None) or []
        if turns:
            last = turns[-1]
            if isinstance(last, dict) and last.get("physical_turn_id"):
                return last["physical_turn_id"]
            return len(turns)
    except Exception:  # noqa: BLE001
        pass
    return "?"


def _build_delegate_resume_brief(
    child: Any,
    child_sid: str,
    *,
    body: str,
    metrics: dict[str, Any],
    cwd: str,
    reason: str = "round_cap",
) -> dict[str, Any]:
    """Structured handoff for a round-capped or cost-budget child outcome."""
    worktree, branch = _detect_git_context(cwd)
    remaining: list[str] = []
    try:
        remaining = list(child.engine._incomplete_todos(child_sid))
    except Exception:  # noqa: BLE001
        remaining = []
    rounds = 0
    try:
        rounds = int((metrics or {}).get("rounds") or 0)
    except (TypeError, ValueError):  # noqa: BLE001
        rounds = 0
    wrap_up = str((metrics or {}).get("wrap_up_text") or body or "")[:4000]
    return build_resume_brief(
        session_id=child_sid,
        turn_id=_child_turn_id(child, child_sid),
        worktree=worktree,
        branch=branch,
        final_summary=wrap_up,
        remaining_todos=remaining,
        last_round_idx=max(0, rounds - 1),
        next_step=extract_next_step(wrap_up, remaining),
        reason=reason or "round_cap",
    )


def _delegate_flock(path: Any):
    """Cross-process exclusive lock using the platform's file-lock backend."""
    import contextlib
    import os
    from pathlib import Path

    @contextlib.contextmanager
    def _held():
        lock_path = Path(str(path) + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            oscompat.file_lock(fd)
            yield
        finally:
            try:
                oscompat.file_unlock(fd)
            finally:
                os.close(fd)

    return _held()


def _resolve_or_create_child_session_locked(
    child: Any,
    args: dict[str, Any],
    parent_sid: str,
) -> tuple[str, str]:
    """Resolve one Agent child or create a fresh one.

    A public ``task_id`` is accepted only on the first-party Agent path and only
    when its durable metadata proves the same parent and compatible target.
    Runtime retries still key through the trusted call-id sidecar.  Named slash
    commands have neither marker nor call id and retain their historical
    per-engine sticky session.

    Returns ``(child_sid, diagnostic)``.
    """
    import json as _json

    is_agent_tool = bool(args.get("_sharedagents_agent_tool"))
    call_id = str(args.get("_sharedagents_call_id") or "").strip()
    use_retry_sidecar = (not is_agent_tool) or bool(call_id)
    requested = str(
        args.get("_sharedagents_task_id_resolved") or (args.get("task_id") if is_agent_tool else "") or ""
    ).strip()
    profile_name = str(args.get("_agent_profile") or "").strip().lower()
    expected_meta = {
        "parent_sid": parent_sid,
        "agent": str(getattr(child, "name", "") or ""),
        "profile": profile_name,
        "engine": str(getattr(child.engine.kind, "value", child.engine.kind)),
    }

    child_sid = ""
    sidecar = _delegate_sidecar_path(child, parent_sid, call_id)
    if requested:
        if not _valid_task_id(requested):
            return "", (f"[Agent: task_id {requested!r} is not a canonical sharedagents session id]")
        if not parent_sid:
            return "", (f"[Agent: task_id {requested!r} cannot be continued without a parent session id]")
        sess = child.get_session(requested)
        if sess is None:
            return "", (f"[Agent: task_id {requested!r} is unknown, unreadable, or belongs to an incompatible target]")
        if str(getattr(sess, "parent_session_id", "") or "") != parent_sid:
            return "", (
                f"[Agent: task_id {requested!r} belongs to a different parent "
                "session; refusing cross-session continuation]"
            )
        try:
            meta = _json.loads(_task_metadata_path(child, requested).read_text(encoding="utf-8"))
        except (OSError, _json.JSONDecodeError):
            return "", (
                f"[Agent: task_id {requested!r} has no readable Agent task "
                "metadata; refusing an ambiguous continuation]"
            )
        if any(str(meta.get(k) or "") != str(v or "") for k, v in expected_meta.items()):
            return "", (f"[Agent: task_id {requested!r} is incompatible with the requested target/profile]")
        child_sid = requested
    elif parent_sid and use_retry_sidecar and sidecar.exists():
        try:
            data = _json.loads(sidecar.read_text(encoding="utf-8"))
            child_sid = str(data.get("sharedagents_child_session_id") or data.get("sharedagents_session_id") or "")
        except (OSError, _json.JSONDecodeError):
            child_sid = ""
        if child_sid and child.get_session(child_sid) is None:
            child_sid = ""

    sess = None
    if not child_sid:
        # Resolve the session-scoped model before the immutable creation
        # header is written.  ``run_turn(model=...)`` still applies the same
        # override below, but doing that only after creation leaves a short
        # (and permanently archived) window where the child header advertises
        # its configured default even though its very first wire request uses
        # the parent's selected model.  SDLC traces use that first header for
        # physical-route audits, so the header and request must agree.
        initial_model = args.get("model") or None
        subagent_type = str(args.get("subagent_type") or profile_name or getattr(child, "name", "") or "")
        if not initial_model:
            initial_model = _inherit_parent_model_if_overridden(
                child,
                parent_sid,
            )
        initial_model = _resolve_model_override(
            child,
            initial_model,
            subagent_type,
        )
        sess = child.create_session(
            workspace_cwd=args.get("cwd") or "",
            workspace_origin=(f"delegate:{parent_sid}" if parent_sid else "delegate:"),
            parent_session_id=parent_sid,
            model=initial_model or "",
        )
        child_sid = sess.hash

    _persist_child_cost_attribution(child, child_sid, parent_sid, args, sess)

    if parent_sid and use_retry_sidecar:
        try:
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            sidecar.write_text(
                _json.dumps(
                    {
                        "sharedagents_session_id": parent_sid,
                        "sharedagents_child_session_id": child_sid,
                        "engine": child.engine.kind.value,
                        "subagent_call_id": call_id,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            if is_agent_tool:
                return "", (
                    f"[Agent: could not persist the retry-idempotency mapping for task_id {child_sid!r}: {exc}]"
                )
    if is_agent_tool:
        try:
            task_meta = dict(expected_meta)
            task_meta["task_id"] = child_sid
            path = _task_metadata_path(child, child_sid)
            path.write_text(
                _json.dumps(task_meta, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            return "", (f"[Agent: could not persist metadata for task_id {child_sid!r}: {exc}]")
    if is_agent_tool and parent_sid and call_id:
        # A call may resume an existing child or belong to a fan-out. Publish
        # the exact edge before execution so the WebUI can link its card
        # without guessing from the agent name or the order of spawn events.
        try:
            from sharedagents_session.session_writer import get_session_writer

            get_session_writer(child.sessions_dir, parent_sid).lifecycle(
                kind="subagent_linked",
                payload={
                    "call_id": call_id,
                    "child_session_id": child_sid,
                    "child_agent_name": profile_name or expected_meta["agent"],
                },
            )
        except Exception:  # noqa: BLE001 — advisory, like subagent_spawned
            pass
    return child_sid, ""


def _resolve_or_create_child_session(
    child: Any,
    args: dict[str, Any],
    parent_sid: str,
) -> tuple[str, str]:
    """Interprocess-serialized resolve/create/persist for one tool call."""
    requested = str(
        args.get("_sharedagents_task_id_resolved")
        or (args.get("task_id") if args.get("_sharedagents_agent_tool") else "")
        or ""
    ).strip()
    if requested and not _valid_task_id(requested):
        return "", (f"[Agent: task_id {requested!r} is not a canonical sharedagents session id]")
    call_id = str(args.get("_sharedagents_call_id") or "").strip()
    sidecar = _delegate_sidecar_path(child, parent_sid, call_id)
    with _delegate_flock(sidecar):
        return _resolve_or_create_child_session_locked(child, args, parent_sid)


def _provider_exhaustion_refusal(short_name: str) -> str | None:
    """Defense in depth: refuse a provider marked exhausted by routing.

    Local/custom agents without a subscription digest entry are unaffected.
    Never silently substitute the suggested alternative or a different model.
    """
    from sharedagents_web.subscription_limits import PROVIDER_AGENTS, routing_blocker, routing_digest

    target = str(short_name or "").strip().lower()
    if target not in PROVIDER_AGENTS.values():
        return None
    try:
        digest = routing_digest()
    except Exception as exc:  # noqa: BLE001 — no dispatch on an unreadable provider verdict
        return f"[delegate {target}: routing availability check failed: {exc}]"
    row = next((row for row in digest.get("providers", []) if row.get("agent") == target), None)
    if row is None:
        return None
    blocker = row.get("dispatch_blocker") or routing_blocker(row)
    if not blocker:
        return None
    alternative = next((agent for agent in digest.get("available_agents", []) if agent != target), None)
    return f"[delegate {target}: routing refused: {blocker}. Next eligible alternative: {alternative or 'none'}]"


async def _drive_child_agent_turn(
    short_name: str,
    args: dict[str, Any],
    *,
    on_delta: Callable[[str], None] | None = None,
    on_event: Callable[[dict], None] | None = None,
    child_session_cb: Callable[[str], None] | None = None,
) -> Any:
    """Pre-spawn gated entry to the common delegated-agent driver.

       This is the SINGLE in-process chokepoint through which every delegated
       agent turn passes (the four ``shim_run_delegate_*`` shims and the detached
       background runner). Per spec §3.1 the counting unit is the agent, so we
       reserve exactly ONE AGENT slot here — before any engine subprocess is
       spawned and depth-checked.
       A child-process top turn arrives pre-counted (marked by its spawning
       parent, spec §6.3) and is skipped, never double-counted. On refusal we
       surface :class:`SpawnRefused` as the tool result and NEVER drive the turn.
       An agent already at ``MAX_DELEGATION_DEPTH`` is refused a NEW delegation here
    — engine-agnostically, closing the gap where a CLI/ACP
       parent had no local_llm tool-strip to cap it.
    """
    from sharedagents_agent import sandbox_broker
    from sharedagents_agent.tools.runners import ToolResult

    refusal = await asyncio.to_thread(_provider_exhaustion_refusal, short_name)
    if refusal:
        return ToolResult(content=refusal, ok=False, outcome="refused")
    try:
        _res = sandbox_broker.reserve_agent_or_precounted(args, kind="subagent")
    except sandbox_broker.SpawnRefused as _sr:
        return ToolResult(content=str(_sr), ok=False, outcome="refused")

    # Orchestration depth cap — the shared, engine-agnostic 3-level guard (see
    # ``_orchestration_depth_refusal``). Fires ONLY on a genuinely-new delegation
    # (fresh reservation); a precounted child TOP turn (``_res is None``) was
    # already depth-checked by the parent that spawned it.
    _depth_refusal = _orchestration_depth_refusal(short_name, _res)
    if _depth_refusal is not None:
        if _res is not None:
            _res.release()
        return ToolResult(content=_depth_refusal, ok=False, outcome="refused")

    # Durable budget admission. The session cap is enforced by the ledger,
    # but concurrent dispatchers read the same "remaining" figure before
    # either child spends, so a wave can over-commit it. Reserve this child's
    # stated ``budget_usd`` under the parent's lock; a batch reserves its
    # whole wave once and marks its entries so the leaves do not double-count.
    from sharedagents_agent.agents import budget_reservations

    _budget_parent = _delegate_parent_sid(args)
    _budget_usd = budget_reservations.parse_budget_usd(args.get("budget_usd"))
    _budget_res = None
    if _budget_parent and _budget_usd > 0 and not args.get("_sharedagents_budget_reserved"):
        _budget_res = budget_reservations.reserve(
            _budget_parent,
            _budget_usd,
            key=str(args.get("_sharedagents_call_id") or args.get("task_id") or ""),
        )
        if _budget_res is None:
            if _res is not None:
                _res.release()
            return ToolResult(
                content=budget_reservations.refusal_message(
                    _budget_parent,
                    _budget_usd,
                ),
                ok=False,
                outcome="refused",
            )

    try:
        return await _drive_child_agent_turn_inner(
            short_name,
            args,
            on_delta=on_delta,
            on_event=on_event,
            child_session_cb=child_session_cb,
        )
    finally:
        if _budget_res is not None:
            _budget_res.release()
        if _res is not None:
            _res.release()


def _touch_parent_session(sessions_dir: Any, parent_sid: str) -> None:
    """Keep parent session_state_updated_at fresh in sessions_meta while waiting on child turn."""
    if not parent_sid:
        return
    try:
        import time
        from pathlib import Path

        from sharedagents_session.session_index import connect_sessions_db

        db_path = Path(sessions_dir) / "sessions.db" if Path(sessions_dir).is_dir() else Path(sessions_dir)
        con = connect_sessions_db(db_path)
        try:
            con.execute(
                "UPDATE sessions_meta SET session_state_updated_at = ? WHERE session_hash = ?",
                (int(time.time() * 1000), str(parent_sid)),
            )
            con.commit()
        finally:
            con.close()
    except Exception:  # noqa: BLE001
        pass


async def _drive_child_agent_turn_inner(
    short_name: str,
    args: dict[str, Any],
    *,
    on_delta: Callable[[str], None] | None = None,
    on_event: Callable[[dict], None] | None = None,
    child_session_cb: Callable[[str], None] | None = None,
) -> Any:
    """Common driver for the four ``shim_run_delegate_*`` shims.

    Loads the named child agent, opens (or resumes) a session keyed
    off the parent sid, dispatches one turn, and assembles a
    :class:`ToolResult` whose ``content`` is the same shape the
    delegate runner emits.

    Hosts the single shared pre-flight delegation gate
    (:func:`sharedagents_agent.delegation_gate.evaluate_preflight`): every delegate
    route funnels through here, so an expensive/third-party delegate turn is
    blocked when its checklist is incomplete or its outbound payload
    contains a secret, and no route bypasses that gate.
    """
    import asyncio
    import os

    # Subagent-minimal boot: when the parent dispatcher spawned us as
    # a dedicated per-subagent ``sharedagents_agent.tool_runner`` subprocess
    # (env ``SHAREDAGENTS_SUBAGENT_MINIMAL=1`` set by
    # ``_spawn_subagent_runner``), skip the heavy gateway-side side
    # effects: MCP server discovery, skill-index rebuild, memory
    # provider init. ``Agent.run_turn`` lazy-loads each of these only
    # when the child config asks for them, so default subagents boot
    # in well under a second instead of paying the full gateway cost.
    # Surfaced as an env hint rather than an arg so the in-process
    # gather fallback ``SHAREDAGENTS_SUBAGENT_INPROCESS=1`` doesn't need
    # to thread the flag through.
    if os.environ.get("SHAREDAGENTS_SUBAGENT_MINIMAL") == "1":
        os.environ.setdefault("SHAREDAGENTS_SKIP_MCP_DISCOVERY", "1")
        os.environ.setdefault("SHAREDAGENTS_SKIP_SKILL_INDEX_REBUILD", "1")
        os.environ.setdefault("SHAREDAGENTS_SKIP_MEMORY_PROVIDER", "1")
    from sharedagents_agent.tools.runners import ToolResult

    from . import factory
    from .engine import TextDelta, ToolCallResult, ToolCallStarted, TurnDone, TurnError

    try:
        child = factory.load_agent(short_name)
    except FileNotFoundError:
        return ToolResult(
            content=f"[delegate {short_name}: no agent scaffolded — install the CLI and retry]",
            ok=False,
            outcome="refused",
        )
    except Exception as exc:  # noqa: BLE001
        return ToolResult(
            content=f"[delegate {short_name}: load_agent failed: {exc}]",
            ok=False,
            outcome="refused",
        )

    # OpenCode-style subagent profile (explore / plan / compaction / …).
    profile_name = str(args.get("_agent_profile") or "").strip()
    if profile_name:
        from sharedagents_agent.agent_profiles import apply_profile_to_config, resolve_profile

        prof = resolve_profile(profile_name)
        if prof is not None:
            apply_profile_to_config(child.cfg, prof)

    available, reason = child.engine.is_available()
    if not available:
        return ToolResult(
            content=f"[delegate {short_name}: {reason}]",
            ok=False,
            outcome="refused",
        )

    # A Web user's turn delegates only within their grants:
    # the target agent must be granted, and an external engine (which brings
    # its own tools and skills, outside the folder sandbox) needs All tools,
    # All skills and All folders. Decided from
    # the parent session's binding BEFORE any child session exists.
    from sharedagents_core.access_grants import delegation_refusal
    from sharedagents_session.access_binding import session_access_grants

    _access_parent = _delegate_parent_sid(args)
    _access_grants = session_access_grants(child.sessions_dir, _access_parent) if _access_parent else None
    if _access_grants is not None:
        _child_kind = str(getattr(getattr(child.engine, "kind", None), "value", "") or "")
        _access_refusal = delegation_refusal(
            _access_grants,
            short_name,
            external=_child_kind != "local_llm",
        )
        if _access_refusal:
            return ToolResult(content=_access_refusal, ok=False, outcome="refused")

    # Optionally block delegation to EXTERNAL CLI agents (claude_cli / codex_cli
    # / antigravity_cli) and steer to a sharedagents sibling. Off by default (prod
    # parity); the dockerized goal-test harness sets
    # SHAREDAGENTS_DISABLE_EXTERNAL_DELEGATES=1 so goal runs never error on a missing
    # or unauthenticated external CLI (the "error 25s" subagent failures).
    if os.environ.get("SHAREDAGENTS_DISABLE_EXTERNAL_DELEGATES") == "1":
        try:
            _kind = child.engine.kind.value
        except Exception:  # noqa: BLE001
            _kind = ""
        if isinstance(_kind, str) and _kind.endswith("_cli"):
            return ToolResult(
                content=(
                    f"[delegate {short_name}: external CLI agents are disabled "
                    f"here — delegate to a sharedagents sibling instead (explore, "
                    f"research, plan, general), which run the local "
                    f"engine.]"
                ),
                ok=False,
                outcome="refused",
            )

    # --- Pre-flight delegation gate --
    #
    # THE single shared enforcement point. Every delegation route (the four
    # shim_run_delegate_* shims + the detached background runner) funnels
    # through this driver, so gating here — and nowhere else — gives
    # "one gate, no bypass": there is no checklist-free path
    # to an expensive delegate. It fires ONLY for EXPENSIVE (frontier /
    # third-party) engines, classified by the child's RESOLVED engine kind
    # (available here, immediately after ``factory.load_agent``), which closes
    # the ``subagent_type=general-purpose`` + external-engine-sibling alias hole.
    # Local in-process subagents (local_llm) are the cheap side of the arbitrage
    # and are never gated or scrubbed. Runs BEFORE any session is created or
    # ``run_turn`` fires; on refusal we surface a structured failed ToolResult
    # (like the external-CLI block above) and NEVER drive the turn — the caller
    # can repair the checklist / payload and retry in-context.
    #   * An incomplete checklist (goal / constraints /
    #     acceptance_criteria / budget_usd) blocks the call.
    #   * A secret in the outbound payload hard-blocks it (no override,
    #     respecting the scrubber's base64 data: URI exemption).
    from sharedagents_agent import delegation_gate

    try:
        _engine_kind = child.engine.kind
    except Exception:  # noqa: BLE001
        _engine_kind = None
    _preflight = delegation_gate.evaluate_preflight(_engine_kind, args)
    if not _preflight.ok:
        return ToolResult(content=_preflight.reason, ok=False, outcome="refused")

    # OpenCode-style task lifecycle. The helper creates children with
    # parent_session_id=parent_sid and workspace_origin=f"delegate:{parent_sid}".
    # Agent calls are fresh per distinct call,
    # retry-idempotent per call id, and can explicitly continue one compatible
    # task_id. Named slash commands carry no Agent marker/call id and therefore
    # retain their historical per-engine sticky child.
    parent_sid = _delegate_parent_sid(args)
    try:
        from sharedagents_goal.goal_service import remaining_for_session_hash

        if parent_sid and remaining_for_session_hash(parent_sid) <= 0:
            return ToolResult(
                content=(
                    "[sharedagents budget] session cost budget exhausted — "
                    "new child refused. Ask the user to authorize more "
                    "with `/budget add`."
                ),
                ok=False,
                outcome="refused",
            )
        # A goal that reached its own budget admits nothing more, even while
        # the children it already started keep it running.
        from sharedagents_goal.goal_service import goal_child_refusal

        _goal_refusal = goal_child_refusal(
            str(args.get("_sharedagents_cost_ancestor_goal_id") or "")
            or _parent_live_goal_id(child.sessions_dir, parent_sid)
        )
        if _goal_refusal:
            return ToolResult(content=_goal_refusal, ok=False, outcome="refused")
    except Exception:  # noqa: BLE001
        pass
    child_sid, task_error = _resolve_or_create_child_session(
        child,
        args,
        parent_sid,
    )
    if task_error:
        return ToolResult(content=task_error, ok=False, outcome="refused")
    subagent_call_id = str(args.get("_sharedagents_call_id") or "").strip()

    # Surface the resolved child session id to the caller BEFORE the turn runs,
    # so a background job records its child_sid before the child can spawn a
    # grandchild — closing the cascade-cancel race where killing this session
    # couldn't find its own runner yet.
    if child_session_cb is not None and child_sid:
        try:
            child_session_cb(child_sid)
        except Exception:  # noqa: BLE001
            pass

    # Recursive-cancel registry. When THIS turn runs as a DEDICATED subagent
    # subprocess (spawned by ``_spawn_subagent_runner`` with
    # ``SHAREDAGENTS_SUBAGENT_MINIMAL=1``), record our own pid against the child
    # session id in the durable engine-pid sidecar. That lets an ancestor's
    # hard ``/abort`` SIGKILL this subagent's WHOLE process subtree by walking
    # the ``sessions_meta`` parent→child registry (``_cancel_child_sessions``)
    # — independent of live ``/proc`` ancestry. Without it, a grandchild
    # subagent whose nested ``BatchAgent`` launcher died first is reparented
    # to init and escapes the top-level ``_kill_tree`` descendant walk, so it
    # keeps running (and billing) after the operator aborted.
    #
    # GATED on ``SHAREDAGENTS_SUBAGENT_MINIMAL``: the in-process slash-command path
    # (``/claude`` etc.) runs ``_drive_child_agent_turn`` INSIDE the gateway,
    # where ``os.getpid()`` is the GATEWAY's pid — recording (and later
    # killing) that would take down the whole agent. ``pgid=None`` skips the
    # ``killpg`` sweep because a subagent ``tool_runner`` SHARES its parent's
    # group; the precise ``/proc`` descendant walk reaps our subtree instead.
    _recorded_pidfile = False
    if child_sid and os.environ.get("SHAREDAGENTS_SUBAGENT_MINIMAL") == "1":
        try:
            from sharedagents_agent.agents import engine_pidfile

            engine_pidfile.record(
                child.sessions_dir,
                child_sid,
                os.getpid(),
                pgid=None,
            )
            _recorded_pidfile = True
        except Exception:  # noqa: BLE001
            pass

    prompt = str(args.get("prompt") or args.get("task") or "")
    # Subagents never run in plan mode — see ``_resolve_subagent_mode``.
    mode = _resolve_subagent_mode(args, profile_name)
    cwd = args.get("cwd") or None
    model = args.get("model") or None
    # Where ``model`` came from — the codex spawn gate's refusal copy is
    # source-aware. "explicit" = Agent(model=...)
    # or a direct runner call; "inherited" = propagated from the parent
    # session's off-default override below.
    model_source = "explicit"
    subagent_type = str(args.get("subagent_type") or profile_name or short_name or "")

    # Session-scoped parent-model inheritance: when the operator switched the
    # PARENT off its configured default for this session, propagate that model
    # to NATIVE local_llm children only. External engines (codex/claude/ACP/…)
    # keep their own default — never inherit a foreign parent id. When the
    # parent is still on its default, leave ``model`` empty so the child keeps
    # *its own* default. Explicit ``args.model`` wins.
    if not model:
        model = _inherit_parent_model_if_overridden(child, parent_sid)
        if model:
            model_source = "inherited"

    # Honor the override ONLY when it resolves to one of the child's configured
    # models (alias or id) or is a raw endpoint URL; otherwise drop it and let
    # the child use its own default. Prevents the bogus-model retry loop.
    model = _resolve_model_override(child, model, subagent_type)

    # The child runs on a model granted to the user (the
    # resolved override, else the child's own default), and keeps the
    # parent's grants for its whole life — even after the parent moves on.
    if _access_grants is not None and not _access_grants.is_all("models"):
        from sharedagents_agent.discovery.cache_lock import expand_model_names

        _child_model = str(model or getattr(child.cfg, "upstream_model", "") or "").strip()
        if not _child_model or not _access_grants.allows(
            "models",
            *expand_model_names(_child_model),
        ):
            return ToolResult(
                content=(f"[access] model {_child_model or '(agent default)'!r} is not granted to this user"),
                ok=False,
            )
    # The child's turn runs under its caller's grants: the parent turn's stamp
    # (explicitly unrestricted for an unrestricted parent, so an older
    # restriction is lifted) rides into ``child.run_turn`` and is bound there
    # under the child's own lock — never written from out here, where it
    # could change a turn already running in the child.
    _child_access = None
    if parent_sid:
        from sharedagents_core.access_grants import AccessStamp, Grants
        from sharedagents_session.access_binding import session_access_stamp

        _parent_stamp = session_access_stamp(child.sessions_dir, parent_sid)
        _child_access = (
            _parent_stamp
            if _parent_stamp is not None and _parent_stamp.restricted
            else AccessStamp(_parent_stamp.email if _parent_stamp else "", Grants.all())
        )

    # --- Flag sensitive-but-sendable egress PRE-CALL -------
    # Positions / PII may accompany a genuinely-needed delegate payload, but
    # the owner must be flagged that sensitive data is about to cross the
    # local boundary BEFORE an EXPENSIVE (third-party) turn fires. This is a
    # VISIBILITY flag, not a block (contrast the secret hard-block in
    # the pre-flight gate above): the turn still proceeds. Fires the structured
    # warning through on_event (owner-visible surfaces) AND durably archives it
    # to the PARENT session log — both strictly before
    # child.run_turn below. Local/co-located engines never leave the boundary
    # and are never flagged (the arbitrage exemption).
    try:
        _sensitive_classes = (
            delegation_gate.sensitive_egress_classes(args) if delegation_gate.is_expensive(_engine_kind) else ()
        )
    except Exception:  # noqa: BLE001
        _sensitive_classes = ()
    if _sensitive_classes:
        _egress_flag = delegation_gate.sensitive_egress_flag_event(
            _sensitive_classes,
            short_name,
        )
        if parent_sid:
            try:
                from sharedagents_session.session_writer import get_session_writer

                get_session_writer(
                    child.sessions_dir,
                    parent_sid,
                ).record_sensitive_egress_flag(dict(_egress_flag))
            except Exception:  # noqa: BLE001
                pass
        if on_event is not None:
            try:
                _r = on_event(dict(_egress_flag))
                if hasattr(_r, "__await__"):
                    await _r
            except Exception:  # noqa: BLE001
                pass

    buf: list[str] = []
    ok = True
    last_error = ""
    completion_reason = "stop"
    turn_metrics: dict[str, Any] = {}
    # True once the child streamed prose on a channel that is NOT the answer
    # channel (today: ``thinking``). Distinguishes "the child said nothing" from
    # "the child spoke, but not where we look for the answer".
    saw_nonanswer_text = False

    # The child's events are RELAYED upward as they stream (TextDelta -> on_delta,
    # ToolCall* -> on_event, below), so the user sees the subagent working live and
    # receives its output incrementally. There is DELIBERATELY no idle / wall-clock
    # cap here: a long-running or momentarily-quiet subagent is NEVER auto-abandoned
    # — the user should always be able to receive the full response (or steer the
    # subagent to finish). The ONLY thing that ends the wait early is the USER: the
    # parent turn cancels the asyncio task driving this delegation
    # (``CancelledError``) on ``/abort`` or a mid-turn steer message — see
    # ``local_llm.LocalLLMEngine._execute_call_singleton`` (~L7524), which
    # ``driver.cancel()``s us. We catch that, promptly + cleanly ``aclose()`` the
    # child (so its engine/tool STOPS immediately instead of leaking to the GC),
    # and return whatever the child accumulated so far — an interrupt yields the
    # partial, never a hang. (A bare ``async for`` propagates the cancellation but
    # leaves the child generator for the event-loop finalizer to close later, so
    # the child keeps running after the user asked to stop — the steerability gap
    # this closes.)
    # Serialize the complete task turn across processes. In-process engine
    # locks do not protect two detached runners (or a foreground continuation)
    # from concurrently driving the same CLI/ACP session.
    _turn_lock_path = child.sessions_dir / f".{child_sid}.agent_turn"
    (
        _turn_lock_fd,
        _local_turn_lock_key,
        _local_turn_lock,
    ) = await _acquire_task_turn_lock(
        _turn_lock_path,
    )
    _turn = None
    # Wrap the WHOLE iteration + callback body: cancellation can land while we
    # await a child event (``__anext__``) OR while we await an on_delta/on_event
    # callback below. The callback try/except only catches ``Exception`` (not the
    # BaseException-derived ``CancelledError``), so without this outer guard a
    # cancel during a blocked callback would escape the loop and skip aclose(),
    # orphaning the child generator + its session lock until the GC.
    try:
        # A few embedders (and older third-party Agent doubles) expose the
        # pre-model-source ``run_turn`` signature. Keep the new source
        # attribution when the engine accepts it, while avoiding a
        # signature TypeError before the child gets a turn at all.
        import inspect

        _run_turn_params = inspect.signature(child.run_turn).parameters
        _run_turn_kwargs = {
            "mode": mode,
            "cwd": cwd,
            "model": model,
        }
        _supports_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in _run_turn_params.values())
        if "model_source" in _run_turn_params or _supports_var_kw:
            _run_turn_kwargs["model_source"] = model_source if model else None
        if "source" in _run_turn_params or _supports_var_kw:
            _run_turn_kwargs["source"] = "delegate"
        # The delegating agent is the author of the child's prompt, so the
        # child chat names it on the user bubble instead of a generic label.
        if parent_sid and ("actors" in _run_turn_params or _supports_var_kw):
            from sharedagents_session.actors import agent_actor

            _delegator = agent_actor(
                _delegating_agent_name(child.sessions_dir, parent_sid),
            )
            if _delegator is not None:
                _run_turn_kwargs["actors"] = [_delegator]
        if _child_access is not None and child_sid:
            if "access" in _run_turn_params or _supports_var_kw:
                _run_turn_kwargs["access"] = _child_access
            else:  # an embedder's Agent without per-turn grants
                from sharedagents_session.access_binding import copy_session_access

                copy_session_access(child.sessions_dir, parent_sid, child_sid)
        _turn = child.run_turn(child_sid, prompt, **_run_turn_kwargs)
        _it = _turn.__aiter__()
        _last_touch = 0.0
        if parent_sid:
            _touch_parent_session(child.sessions_dir, parent_sid)
            try:
                _last_touch = asyncio.get_event_loop().time()
            except Exception:
                _last_touch = 0.0
        while True:
            try:
                _now = asyncio.get_event_loop().time()
                if parent_sid and _now - _last_touch > 15.0:
                    _last_touch = _now
                    _touch_parent_session(child.sessions_dir, parent_sid)
            except Exception:
                pass
            try:
                ev = await _it.__anext__()
            except StopAsyncIteration:
                break
            if isinstance(ev, TextDelta):
                if ev.channel != "tool_delta" and ev.text:
                    # ``buf`` becomes the RESULT the parent consumes — for a
                    # background delegation it is literally the parent's next
                    # turn prompt. Only the child's ``content`` (its answer)
                    # belongs there: a child's ``thinking`` is its private
                    # reasoning, and folding it in shipped the whole trace into
                    # the parent's context AND rendered it in the parent chat
                    # (measured: 1468 of 4771 payload chars were the
                    # child's reasoning, 1687 chars of preamble before its
                    # report even began). The live ``on_delta`` forward is
                    # deliberately unfiltered — progress rendering stays as-is.
                    #
                    # ALLOW-list, paired with the answerless check after the loop.
                    # A deny-list would leak a future private channel; a bare
                    # allow-list would silently return an EMPTY result if a new
                    # channel carried the answer. Together they give both: only
                    # ``content`` is the answer, and producing no answer at all is
                    # a LOUD failure rather than a successful empty delegation.
                    if ev.channel == "content":
                        buf.append(ev.text)
                    else:
                        saw_nonanswer_text = True
                    if on_delta is not None:
                        try:
                            res = on_delta(ev.text)
                            if hasattr(res, "__await__"):
                                await res
                        except Exception:  # noqa: BLE001
                            pass
            elif isinstance(ev, ToolCallStarted):
                if on_event is not None:
                    try:
                        res = on_event(
                            {
                                "kind": "tool_use",
                                "id": ev.call_id,
                                "name": ev.name,
                                "input": dict(ev.arguments or {}),
                                # Forwarding-boundary marker: this is a delegated CHILD's
                                # inner tool call surfaced for live display only, so every
                                # interface can gate it (delivery.is_subagent_event). The
                                # child's real answer rides the parent delegate tool_result.
                                "subagent": True,
                            }
                        )
                        if hasattr(res, "__await__"):
                            await res
                    except Exception:  # noqa: BLE001
                        pass
            elif isinstance(ev, ToolCallResult):
                if on_event is not None:
                    try:
                        res = on_event(
                            {
                                "kind": "tool_result",
                                "tool_use_id": ev.call_id,
                                "ok": ev.ok,
                                "text": ev.content,
                                # Forwarding-boundary marker (see tool_use above).
                                "subagent": True,
                            }
                        )
                        if hasattr(res, "__await__"):
                            await res
                    except Exception:  # noqa: BLE001
                        pass
            elif isinstance(ev, TurnError):
                last_error = ev.message
            elif isinstance(ev, TurnDone):
                ok = ev.state == "ok"
                completion_reason = getattr(ev, "completion_reason", "stop") or "stop"
                turn_metrics = dict(ev.metrics or {})
                break
    except asyncio.CancelledError:
        ok = False
        last_error = (
            f"[delegate {short_name}: interrupted by the user before the "
            f"'{subagent_type or short_name}' agent finished — returning its "
            f"partial output. Re-ask, refine the request, or steer the "
            f"subagent to continue.]"
        )
        # Promptly + cleanly stop the child: throw GeneratorExit into its turn
        # so the engine/tool is torn down now (``Agent.run_turn`` re-raises
        # GeneratorExit → its finallys release the session lock / close the
        # engine stream), rather than orphaning it for the GC. Swallow any
        # exception (incl. a re-delivered CancelledError) from the close.
        # Do NOT re-raise: the owner requirement is that an interrupt yields the
        # accumulated partial; the singleton caller re-raises its own
        # CancelledError, so structured cancellation is still preserved upstream.
        _aclose = getattr(_turn, "aclose", None)
        if _aclose is not None:
            try:
                await _aclose()
            except (Exception, asyncio.CancelledError):  # noqa: BLE001
                pass
    finally:
        # Exhausting through TurnDone breaks before requesting the generator's
        # next item, so its ``finally`` (engine/session cleanup) has not
        # necessarily run yet. Close it while both task locks are still held;
        # otherwise a queued continuation can enter run_turn before the prior
        # turn has actually released its live resources.
        _aclose = getattr(_turn, "aclose", None)
        if _aclose is not None:
            try:
                await _aclose()
            except (Exception, asyncio.CancelledError):  # noqa: BLE001
                pass
        _release_task_turn_lock(
            _turn_lock_fd,
            _local_turn_lock_key,
            _local_turn_lock,
        )

    # Turn finished cleanly: drop our engine-pid sidecar so a later abort of
    # an unrelated session can't match a stale entry. (A crash / hard-kill
    # exit can't run this — but ``kill_session_engine``'s start-time identity
    # guard makes a stale sidecar harmless, never signalling a recycled pid.)
    if _recorded_pidfile and child_sid:
        try:
            from sharedagents_agent.agents import engine_pidfile

            engine_pidfile.clear(child.sessions_dir, child_sid)
        except Exception:  # noqa: BLE001
            pass

    args["_sharedagents_metrics_out"] = turn_metrics
    body = "".join(buf)

    # Round-cap AND cost-budget wrap-ups are INCOMPLETE everywhere — never
    # look completed to the parent, background job finalizer, or cost
    # rollup. Deliver a machine-readable resume brief (result + atomic
    # sidecar). Engine never auto-resumes.
    _capped_incomplete = completion_reason in ("round_cap", "budget")
    _resume_brief: dict[str, Any] | None = None
    if _capped_incomplete:
        ok = False
        try:
            _resume_brief = _build_delegate_resume_brief(
                child,
                child_sid,
                body=body,
                metrics=turn_metrics,
                cwd=cwd,
                reason=completion_reason,
            )
            write_resume_brief_sidecar(_resume_brief)
        except Exception:  # noqa: BLE001
            try:
                _resume_brief = build_resume_brief(
                    session_id=child_sid,
                    turn_id="?",
                    final_summary=body,
                    next_step=extract_next_step(body, ()),
                    reason=completion_reason,
                )
                write_resume_brief_sidecar(_resume_brief)
            except Exception:  # noqa: BLE001
                _resume_brief = None
        if completion_reason == "budget":
            cap_note = (
                f"[delegate {short_name}: the '{subagent_type or short_name}' "
                "child hit the session cost budget with work still in "
                "flight — this task is INCOMPLETE, not completed. Its resume "
                "brief (below / sidecar) records what remains; authorize more "
                "budget to continue, or treat the wrap-up as the checkpoint.]"
            )
        else:
            cap_note = (
                f"[delegate {short_name}: the '{subagent_type or short_name}' child "
                "hit its per-turn step budget (round_cap) with work still in "
                "flight — this task is INCOMPLETE, not completed. Its resume "
                "brief (below / sidecar) records what remains; continue with an "
                "explicit next turn carrying the brief, raise the cap, or "
                "re-delegate.]"
            )
        body = cap_note + (("\n\n" + body) if body.strip() else "")

    # A child that finished while producing no answer text at all is an ERROR —
    # never a successful empty delegation. If it streamed prose on another
    # channel (e.g. thinking), distinguish harness channel mismatch from a child
    # that failed to produce any deliverable.
    if not body.strip():
        ok = False
        if saw_nonanswer_text:
            last_error = (
                "[delegate " + short_name + ": the child produced no 'content' "
                "output despite streaming text on another channel — the engine's "
                "answer channel is not being read. This is a harness bug, not a "
                "subagent failure.]"
            )
        elif not last_error:
            last_error = (
                f"[delegate {short_name}: subagent completed without returning "
                f"an answer or content output. The task did not return any deliverable.]"
            )
    if not ok and last_error and last_error not in body:
        body = (body + "\n\n" + last_error).strip()

    if _resume_brief is not None:
        body = (body + "\n\n" + render_resume_brief(_resume_brief)).strip()

    body += _delegation_success_hint(
        ok=ok,
        body=body,
        subagent_type=subagent_type,
        prompt=prompt,
        grants=_access_grants,
    )

    # Read the child engine's OWN CLI session id (claude/codex/antigravity
    # resume token) that the child engine wrote into its
    # ``<child_sid>.<engine>.json`` state during run_turn.
    engine_sid = ""
    if child_sid:
        try:
            state = child.engine.read_state(child.sessions_dir, child_sid)
            engine_sid = str(getattr(state, "engine_session_id", "") or "")
        except Exception:  # noqa: BLE001
            engine_sid = ""
    # Surface the link back to the direct-dispatch caller
    # (LocalLLMEngine._run_direct_tool_call) so the PARENT session also
    # records the sharedagents↔engine session id in its log + sessions.db.
    if isinstance(args, dict):
        args["_sharedagents_engine_session_id_out"] = engine_sid
        args["_sharedagents_child_session_id_out"] = child_sid

    # Persist the full mapping parent_sid → child_sid → engine_session_id
    # in one sidecar the operator can correlate. Best-effort — never fail
    # the turn over a bookkeeping write.
    if parent_sid and child_sid and (not args.get("_sharedagents_agent_tool") or subagent_call_id):
        try:
            import json as _json

            sidecar = _delegate_sidecar_path(child, parent_sid, subagent_call_id)
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            sidecar.write_text(
                _json.dumps(
                    {
                        "sharedagents_session_id": parent_sid,
                        "sharedagents_child_session_id": child_sid,
                        "engine": child.engine.kind.value,
                        "engine_session_id": engine_sid,
                        "subagent_call_id": subagent_call_id,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            pass

    if args.get("_sharedagents_agent_tool"):
        state = "capped" if _capped_incomplete else "completed" if ok else "failed"
        _reason_attr = f' completion_reason="{completion_reason}"' if completion_reason != "stop" else ""
        body = f'<task id="{child_sid}" state="{state}"{_reason_attr}>\n{body}\n</task>'
    diagnostic = str(args.get("_sharedagents_background_config_diagnostic") or "").strip()
    if diagnostic:
        body = f"{diagnostic}\n\n{body}".strip()
    return ToolResult(content=body, ok=ok)


# --- background (async) delegation ----------------------------------------
#
# When ``background=true`` the shim does NOT run the child turn inline. It
# records a job, spawns a DETACHED runner (its own session/process group so it
# outlives the parent turn — even a gateway restart), and returns a job handle
# immediately. The runner (``async_delegate_runner``) self-reports a
# ``subagent_result`` to the parent session log. Default (background absent) is
# the unchanged synchronous path below.


def _spawn_detached_runner(spec: dict[str, Any], *, reservation=None) -> int:
    """Spawn the detached async-delegate runner; return its pid.

    When ``reservation`` is supplied (spec §6 / addendum), the child inherits
    that reservation's delegated sub-budget via env so the SECOND fan-out path
    (``background=true``) is capped identically to the subagent-runner path —
    its nested spawns draw from the grant, and can never escape the session
    budget with a fresh empty broker.
    """
    import json as _json
    import os
    import subprocess
    import sys
    from pathlib import Path

    # Guarantee the child imports the SAME sharedagents_core as us, even when this
    # package is run from a source tree that isn't pip-installed on the child's
    # default path (tests, dev). Harmless when it IS installed.
    import sharedagents_core

    src_root = str(Path(sharedagents_core.__file__).resolve().parents[1])
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_root + (os.pathsep + existing if existing else "")
    if reservation is not None:
        env = reservation.child_env(env)
    # The parent turn's pinned grants ride along.
    from sharedagents_session.access_binding import apply_access_pin_env

    apply_access_pin_env(env, str(spec.get("parent_sid") or ""))
    proc = subprocess.Popen(
        [sys.executable, "-m", "sharedagents_agent.agents.async_delegate_runner"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=not oscompat.IS_WINDOWS,
        env=env,
        **oscompat.detach_kwargs(),
    )
    if proc.stdin is not None:
        proc.stdin.write(_json.dumps(spec).encode("utf-8"))
        proc.stdin.close()
    return proc.pid


def profile_allows_background_delegation() -> bool:
    """May the profile issuing this call delegate asynchronously?

    True unless the acting agent is pinned foreground-only. Resolution is
    CONFIG-DRIVEN (per-agent ``delegation_mode`` in config.json, then the
    applied profile's default — see ``background_subagents``): build/plan
    scaffold as foreground-only, jarvis as background-only. An UNKNOWN
    identity keeps the historical behaviour — background stays available — so
    a missing environment variable degrades to today's semantics instead of
    silently serialising every delegation in the fleet.
    """
    from sharedagents_agent.background_subagents import policed_delegation_mode

    return policed_delegation_mode() != "foreground"


def _existing_live_background_job(
    parent_sid: str,
    short_name: str,
    args: dict[str, Any],
):
    """Return an exact call replay already in flight.

    Distinct prompts are intentionally allowed to run concurrently.  Fuzzy
    prompt matching is unsafe here because two legitimate Jarvis explorations
    often share words such as "source", "tweets", or "Dashboard".  The
    model-facing live-task guard handles explicit ``task_id`` polling; this
    helper only protects a literal replay of the same tool call.
    """
    if not parent_sid or args.get("_sharedagents_batch"):
        return None
    from . import async_jobs

    call_id = str(args.get("_sharedagents_call_id") or "").strip()
    try:
        existing = async_jobs.open_jobs_for(parent_sid)
    except Exception:  # noqa: BLE001 — dedupe is a safety enhancement
        return None
    for job in existing:
        if str(job.get("agent") or "") != str(short_name or ""):
            continue
        if call_id and str(job.get("call_id") or "") == call_id:
            return job, "same_call_id"
    return None


def _deduplicated_background_result(job: dict[str, Any], reason: str):
    """Build a model-visible handle without starting another child process."""
    import json as _json

    from sharedagents_agent.tools.runners import ToolResult

    payload = {
        "status": "already_started",
        "deduplicated": True,
        "reason": reason,
        "job_id": str(job.get("job_id") or ""),
        "agent": str(job.get("agent") or ""),
        "task": str(job.get("task") or "")[:120],
    }
    if job.get("child_sid"):
        payload["task_id"] = str(job["child_sid"])
    return ToolResult(
        content=_json.dumps(payload),
        ok=True,
        background_handle=True,
    )


def _background_execution_extra(args: dict[str, Any]) -> dict[str, Any]:
    """Serialize turn options that must survive detached delegation.

    Background jobs cross a process boundary and the runner reconstructs its
    call from only this dictionary. Omitting an execution contract here
    silently changes background behavior even when the synchronous path
    honors it.
    """
    return {
        k: args[k]
        for k in (
            "cwd",
            "model",
            "mode",
            "_agent_profile",
            "goal",
            "constraints",
            "acceptance_criteria",
            "budget_usd",
            "_sharedagents_agent_tool",
        )
        if args.get(k) is not None
    }


async def _maybe_background(short_name: str, args: dict[str, Any]) -> Any:
    """Return a job-handle ToolResult for a background call, else None."""
    if args.get("_sharedagents_agent_tool"):
        # First-party Agent lifecycle semantics (per-agent background default +
        # public task_id). Delegation mode is CONFIG-DRIVEN (per-agent
        # ``delegation_mode`` in config.json, then the applied profile's
        # default — jarvis ⇒ background, build/plan ⇒ foreground), overriding
        # an explicit boolean; unpoliced profiles keep the explicit value,
        # then the .env default.
        from sharedagents_agent.background_subagents import (
            apply_delegation_background_to_args,
        )

        diagnostic = apply_delegation_background_to_args(args)
        wants_background = bool(args.get("background"))
        if diagnostic:
            args["_sharedagents_background_config_diagnostic"] = diagnostic
    else:
        # Named slash/direct delegate sessions are a separate interaction.
        wants_background = args.get("background") is True if "background" in args else False
    # Foreground-pinned callers (build/plan scaffold default) were already
    # coerced by ``resolve_background_wants`` above — the historical
    # profile-level guard is subsumed.
    # Compaction is foreground-mandatory: the child must finish in this turn so
    # its output/summary can be used immediately. A background handle would
    # leave the parent waiting or uncompacted.
    if str(short_name or "").strip().lower() in ("compaction", "compact"):
        wants_background = False
    if not wants_background:
        return None
    refusal = await asyncio.to_thread(_provider_exhaustion_refusal, short_name)
    if refusal:
        from sharedagents_agent.tools.runners import ToolResult

        return ToolResult(content=refusal, ok=False, outcome="refused")
    import json as _json
    import os

    from sharedagents_agent.tools.runners import ToolResult

    from . import async_jobs

    parent_sid = _delegate_parent_sid(args)
    if not parent_sid:
        # Without a parent session there's nowhere to deliver the result;
        # refuse rather than silently dropping it.
        return ToolResult(
            content=("[background delegation needs a parent session id but none was resolved]"),
            ok=False,
            outcome="refused",
        )
    # Wave selection: a batch entry's `_sharedagents_wave_id` (or a caller's
    # explicit `wave_id`) joins that in-flight wave, the mid-flight edit for
    # same-goal work. Refuse an unknown, foreign or closed wave here, before a
    # reservation or a child session exists; create() re-checks under its locks.
    join_wave = async_jobs.requested_wave_id(args.get("_sharedagents_wave_id") or args.get("wave_id"))
    if join_wave:
        try:
            async_jobs.ensure_wave_joinable(parent_sid, join_wave)
        except ValueError as exc:
            return ToolResult(content=f"[Agent: {exc}]", ok=False, outcome="refused")
    # Kept in the durable record for backward compatibility only. There is no
    # longer a model-facing notify/resume choice: each terminal wave wakes its parent.
    on_complete = "resume" if args.get("on_complete") == "resume" else "notify"

    # --- Pre-flight delegation gate BEFORE backgrounding -----------------------
    # The synchronous path gates
    # inside `_drive_child_agent_turn`, but `background=true` spawns a DETACHED
    # runner and returns ok=True to the parent WITHOUT ever driving that gate
    # in-process — the parent saw a clean "started" while the child re-ran the
    # gate in its own process, invisible to the parent. Gate HERE, before the
    # reservation and spawn, so a backgrounded delegate is gated IDENTICALLY to a
    # synchronous one: an incomplete checklist or a secret in the call args
    # refuses up front and NEVER spawns the detached runner. Engine kind is
    # resolved the same way the synchronous driver resolves it (load_agent →
    # engine.kind); on resolution failure we fall through to `None`, which
    # `evaluate_preflight` treats as non-expensive (local) — matching the
    # synchronous path's identical `except` fallback, so background is never MORE
    # permissive than synchronous.
    from sharedagents_agent import delegation_gate

    _bg_child = None
    try:
        from . import factory

        _bg_child = factory.load_agent(short_name)
        _profile_name = str(args.get("_agent_profile") or "").strip()
        if _profile_name:
            from sharedagents_agent.agent_profiles import (
                apply_profile_to_config,
                resolve_profile,
            )

            _prof = resolve_profile(_profile_name)
            if _prof is not None:
                apply_profile_to_config(_bg_child.cfg, _prof)
        _bg_engine_kind = _bg_child.engine.kind
    except Exception as exc:  # noqa: BLE001
        if args.get("_sharedagents_agent_tool"):
            return ToolResult(
                content=f"[delegate {short_name}: load_agent failed: {exc}]",
                ok=False,
                outcome="refused",
            )
        # Legacy explicit-background shims keep their historical submit-now
        # behavior; the detached runner surfaces any eventual load failure.
        _bg_engine_kind = None
    if _bg_child is not None:
        available, reason = _bg_child.engine.is_available()
        if not available:
            return ToolResult(content=f"[delegate {short_name}: {reason}]", ok=False, outcome="refused")
    _bg_preflight = delegation_gate.evaluate_preflight(_bg_engine_kind, args)
    if not _bg_preflight.ok:
        return ToolResult(content=_bg_preflight.reason, ok=False, outcome="refused")

    task = ""
    for k in ("task", "prompt", "message", "input"):
        v = args.get(k)
        if isinstance(v, str) and v.strip():
            task = v
            break
    duplicate = _existing_live_background_job(parent_sid, short_name, args)
    if duplicate is not None:
        job, reason = duplicate
        return _deduplicated_background_result(job, reason)
    call_id = str(args.get("_sharedagents_call_id") or "")
    # Forward execution options so the detached child honors the SAME cwd /
    # model / permission mode the synchronous path would (the runner rebuilds
    # args only from the spec). The four pre-flight checklist fields
    # (goal / constraints / acceptance_criteria / budget_usd) MUST ride along
    # too: the detached runner re-runs the shared gate in its OWN process, and
    # dropping them here made the backgrounded child fail the checklist that the
    # synchronous path passes — so a backgrounded delegate now carries the
    # identical checklist.
    extra = _background_execution_extra(args)
    # Pre-spawn AGENT reservation for the SECOND fan-out path (spec addendum):
    # a background delegation crosses a process boundary into a
    # detached runner with a fresh broker, so it must be gated + budgeted
    # exactly like the subagent-runner path. Book the whole subtree up front;
    # on refusal, NEVER spawn. Released when the gateway next observes the job
    # terminal (see :func:`_reconcile_bg_reservations`).
    from sharedagents_agent import sandbox_broker

    _reconcile_bg_reservations()
    try:
        _res = sandbox_broker.reserve_agent(kind="subagent", cross_process=True)
    except sandbox_broker.SpawnRefused as _sr:
        return ToolResult(content=str(_sr), ok=False, outcome="refused")
    # Orchestration depth cap: a BACKGROUND delegation crosses
    # into a detached runner that runs its top turn ``_precounted`` (so the
    # chokepoint check there is skipped) — enforce the SAME shared cap here, at
    # this second reservation site, so backgrounding is never a depth-cap bypass.
    _bg_depth_refusal = _orchestration_depth_refusal(short_name, _res)
    if _bg_depth_refusal is not None:
        _res.release()
        return ToolResult(content=_bg_depth_refusal, ok=False, outcome="refused")
    try:
        # Create/validate the Agent task BEFORE returning the handle, matching
        # OpenCode's public contract: the immediate response already contains
        # the exact reusable child task_id. The detached runner is pinned to it.
        child_sid = ""
        if args.get("_sharedagents_agent_tool"):
            child_sid, task_error = _resolve_or_create_child_session(
                _bg_child,
                args,
                parent_sid,
            )
            if task_error:
                _res.release()
                return ToolResult(content=task_error, ok=False, outcome="refused")
            extra["_sharedagents_task_id_resolved"] = child_sid
        # Without a wave to join this dispatch is a solo wave: open AND seal
        # its manifest around the single registration so claims are never
        # fenced by a live dispatcher and the receipt key can pin on delivery.
        wave_id = join_wave or async_jobs.create_wave(
            parent_sid,
            origin=str(args.get("_sharedagents_wave_origin") or ""),
        )
        try:
            rec = async_jobs.create(
                parent_sid=parent_sid,
                agent=short_name,
                task=task,
                on_complete=on_complete,
                call_id=call_id,
                pid=0,
                wave_id=wave_id,
            )
        except ValueError as exc:
            if not join_wave:
                raise
            # The wave's delivery began after the early check: no job exists.
            _res.release()
            return ToolResult(content=f"[Agent: {exc}]", ok=False, outcome="refused")
        if not join_wave:
            async_jobs.seal_wave(parent_sid, wave_id)
        if child_sid:
            async_jobs.update(rec["job_id"], child_sid=child_sid)
        spec = {
            "job_id": rec["job_id"],
            "parent_sid": parent_sid,
            "agent": short_name,
            "task": task,
            "on_complete": on_complete,
            "call_id": call_id,
            "args_extra": extra,
            "_precounted": True,
            # Pin the resume target to THIS process's own control bind so the
            # detached runner resumes the parent on the gateway the parent
            # actually lives on — never the hardcoded prod :8100 default (a
            # dev/usersim runner must not leak a resume to prod). Published by
            # the gateway at boot; empty ⇒ the runner falls through env/config
            # and, failing all, refuses rather than defaulting to prod.
            "gateway_url": os.environ.get("SHAREDAGENTS_SELF_GATEWAY_URL") or "",
        }
        pid = _spawn_detached_runner(spec, reservation=_res)
        async_jobs.update(rec["job_id"], pid=int(pid))
    except Exception as exc:  # noqa: BLE001 — no runner owns this failure
        _res.release()
        if "rec" in locals():
            async_jobs.fail_submission(rec["job_id"], str(exc))
        raise
    _BG_RESERVATIONS[rec["job_id"]] = _res
    handle = {
        "status": "started",
        "job_id": rec["job_id"],
        "agent": short_name,
        "task": task[:120],
        "wave_id": rec["wave_id"],
    }
    if child_sid:
        handle["task_id"] = child_sid
    # A coerced mode (profile policy overriding the model's explicit boolean)
    # is visible on the handle itself, not only in logs.
    _diag = str(args.get("_sharedagents_background_config_diagnostic") or "").strip()
    if _diag:
        handle["note"] = _diag
    if not args.get("_sharedagents_agent_tool"):
        handle["on_complete"] = on_complete
    return ToolResult(
        content=_json.dumps(handle),
        ok=True,
        background_handle=True,
    )


# Gateway-held reservations for detached background jobs, keyed by job_id. The
# detached runner finalizes in its OWN process, so the gateway releases the
# session slot lazily when it next observes the job terminal (crash-safe: a
# gateway restart drops the map, releasing everything).
_BG_RESERVATIONS: dict[str, Any] = {}


def _reconcile_bg_reservations() -> None:
    """Release gateway-held background reservations whose job is terminal/gone."""
    if not _BG_RESERVATIONS:
        return
    from . import async_jobs

    for jid in list(_BG_RESERVATIONS):
        try:
            rec = async_jobs.read(jid)
        except Exception:  # noqa: BLE001
            rec = None
        if rec is None or rec.get("status") != "running":
            res = _BG_RESERVATIONS.pop(jid, None)
            if res is not None:
                try:
                    res.release()
                except Exception:  # noqa: BLE001
                    pass


def _attach_background_diagnostic(result: Any, args: dict[str, Any]) -> Any:
    """Attach an invalid per-agent default diagnostic to any foreground result."""
    diagnostic = str(args.get("_sharedagents_background_config_diagnostic") or "").strip()
    content = str(getattr(result, "content", "") or "")
    if not diagnostic or diagnostic in content:
        return result
    from sharedagents_agent.tools.runners import ToolResult

    return ToolResult(
        content=f"{diagnostic}\n\n{content}".strip(),
        ok=bool(getattr(result, "ok", False)),
        truncated=bool(getattr(result, "truncated", False)),
        outcome=getattr(result, "outcome", None),
        member_outcomes=getattr(result, "member_outcomes", ()),
        background_handle=bool(getattr(result, "background_handle", False)),
    )


_TASK_ID_ATTR_RE = re.compile(r'<task\s+id="([^"]+)"')


def _emit_sync_member_done(
    short_name: str,
    args: dict[str, Any],
    result: Any,
) -> None:
    """Publish one foreground delegation member's terminal state.

    A synchronous fan-out (``AgentSync`` / ``Agent`` with ``runs``) returns ONE
    tool result after ``asyncio.gather``, so the parent log used to carry only
    the spawn-time ``subagent_linked`` edge. The web delegation card therefore
    kept EVERY member row on "Running" until the slowest sibling finished,
    even when one child had completed minutes earlier.

    Emit a durable ``subagent_done`` edge for this member as soon as its turn
    returns; the batch tool result remains the authoritative aggregate.
    Background children never reach this wrapper — the detached runner owns
    them and already emits the terminal ``subagent_result``.
    """
    try:
        if not args.get("_sharedagents_agent_tool"):
            return
        parent_sid = _delegate_parent_sid(args)
        call_id = str(args.get("_sharedagents_call_id") or "").strip()
        if not parent_sid or not call_id:
            return
        child_sid = str(args.get("_sharedagents_child_session_id_out") or "").strip()
        if not child_sid:
            match = _TASK_ID_ATTR_RE.search(str(getattr(result, "content", "") or ""))
            child_sid = match.group(1) if match else ""
        ok = bool(getattr(result, "ok", False))
        outcome = str(getattr(result, "outcome", "") or "").strip().lower()
        status = outcome if outcome in ("cancelled", "aborted", "refused") else "done" if ok else "failed"
        from sharedagents_core.layout import sessions_root
        from sharedagents_session.session_writer import get_session_writer

        get_session_writer(sessions_root(), parent_sid).lifecycle(
            kind="subagent_done",
            payload={
                "call_id": call_id,
                "child_session_id": child_sid,
                "child_agent_name": short_name,
                "status": status,
                "ok": ok,
            },
        )
    except Exception:  # noqa: BLE001 — advisory, like subagent_linked
        pass


async def shim_run_delegate_cli_async(
    short_name: str,
    args: dict[str, Any],
    *,
    on_delta: Callable[[str], None] | None = None,
    on_event: Callable[[dict], None] | None = None,
) -> Any:
    """Drive one external CLI delegate turn for ``claude`` / ``codex`` / ``antigravity``."""
    from sharedagents_agent.external_registry import CLI_DELEGATE_SHORT_NAMES

    name = str(short_name or "").strip().lower()
    if name not in CLI_DELEGATE_SHORT_NAMES:
        from sharedagents_agent.tools.runners import ToolResult

        return ToolResult(
            content=f"[delegate: unknown CLI agent {short_name!r}]",
            ok=False,
        )
    bg = await _maybe_background(name, args)
    if bg is not None:
        return bg
    result = await _drive_child_agent_turn(
        name,
        args,
        on_delta=on_delta,
        on_event=on_event,
    )
    _emit_sync_member_done(name, args, result)
    return _attach_background_diagnostic(result, args)


def delegate_shim_for_tool(tool_name: str) -> Callable[..., Any] | None:
    """Resolve a direct-dispatch delegate tool through the shared CLI shim."""
    from functools import partial

    from sharedagents_agent.external_registry import CLI_DELEGATE_SHORT_NAMES

    tool = str(tool_name or "").strip().lower()
    if tool == "delegate_agent":
        return shim_run_delegate_agent_async
    if not tool.startswith("delegate_"):
        return None
    short_name = tool.removeprefix("delegate_")
    if short_name not in CLI_DELEGATE_SHORT_NAMES:
        return None
    return partial(shim_run_delegate_cli_async, short_name)


async def shim_run_delegate_agent_async(
    args: dict[str, Any],
    *,
    on_delta: Callable[[str], None] | None = None,
    on_event: Callable[[dict], None] | None = None,
) -> Any:
    """Dispatch one turn against a sibling sharedagents agent.

    Works uniformly across engine kinds because the work is performed
    by ``Agent.run_turn`` on the target child agent.
    """
    # Accept every alias smaller-context LLMs typically reach for —
    # the canonical schema parameter is ``name`` but deepseek-v4-pro /
    # qwen / kimi often emit ``agent_id`` / ``agent_name`` / ``agent``
    # / ``target`` on the first attempt. Round-tripping a 4-round
    # tool-error recovery just to relearn the field name is wasteful
    # when the intent is unambiguous; normalise here so the FIRST
    # call lands.
    target = str(
        args.get("name")
        or args.get("agent")
        or args.get("agent_id")
        or args.get("agent_name")
        or args.get("target")
        or args.get("to")
        or ""
    )
    if not target:
        from sharedagents_agent.tools.runners import ToolResult

        return ToolResult(
            content=(
                "[delegate_agent: missing target agent — pass `name` "
                "(canonical) or any of agent / agent_id / agent_name "
                "/ target / to. Available agents: see ~/.sharedagents/agents/]"
            ),
            ok=False,
        )
    # SHARED delegation boundary guard: the live orchestrator
    # (jarvis) and hidden runtime utilities (compaction/title/summary/teacher) are never
    # valid delegation TARGETS. The Agent-tool wrapper guards its own path, but
    # this coroutine is ALSO reachable directly from the ACP / external-CLI /
    # local-engine delegation markers — so the refusal must live here too, or a
    # non-canonical route could run the live orchestrator. Literal check on the
    # target IDENTITY (a worker legitimately named e.g. ``main`` is NOT jarvis and
    # stays allowed); alias→profile dispatch is guarded in _shim_agent_canonical.
    _tgt = target.strip().lower()
    if _tgt == "jarvis" or _tgt in ("compaction", "title", "summary", "teacher"):
        from sharedagents_agent.tools.runners import ToolResult

        return ToolResult(
            content=(
                f"[delegate_agent: `{target}` is not a delegation target — the "
                "orchestrator (jarvis) and hidden runtime agents "
                "(compaction/title/summary/teacher) can never be delegated to.]"
            ),
            ok=False,
        )
    # Normalise back to the canonical key + accept ``prompt`` /
    # ``message`` as task synonyms (registry says ``task``, but
    # delegate_claude / delegate_codex use ``prompt`` and the LLM
    # carries that habit over).
    args = dict(args)
    args["name"] = target
    if "task" not in args:
        for k in ("prompt", "message", "input"):
            v = args.get(k)
            if isinstance(v, str) and v.strip():
                args["task"] = v
                break
    bg = await _maybe_background(target, args)
    if bg is not None:
        return bg
    result = await _drive_child_agent_turn(
        target,
        args,
        on_delta=on_delta,
        on_event=on_event,
    )
    _emit_sync_member_done(target, args, result)
    return _attach_background_diagnostic(result, args)
