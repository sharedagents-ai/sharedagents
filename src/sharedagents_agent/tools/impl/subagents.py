"""Extracted tool-runner family: subagents (de-monolith of runners.py).

Behavior-preserving split — code moved verbatim.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sharedagents_agent import sandbox_broker
from sharedagents_agent.tools._log import log_event
from sharedagents_core import oscompat

from ._utils import (
    _ROUTE_TRACE,
    ToolResult,
    _kill_tree,
)


# Lazy import — the shim package imports from this module, so we
# cannot import at module load time without a cycle.
def _shim_cli(short_name: str, args, *, on_delta=None, on_event=None):
    from sharedagents_agent.agents.delegate_shims import shim_run_delegate_cli_async

    return shim_run_delegate_cli_async(
        short_name,
        args,
        on_delta=on_delta,
        on_event=on_event,
    )


def _shim_claude(args, *, on_delta=None, on_event=None):
    return _shim_cli("claude", args, on_delta=on_delta, on_event=on_event)


def _shim_codex(args, *, on_delta=None, on_event=None):
    return _shim_cli("codex", args, on_delta=on_delta, on_event=on_event)


def _shim_antigravity(args, *, on_delta=None, on_event=None):
    return _shim_cli("antigravity", args, on_delta=on_delta, on_event=on_event)


def _shim_zcode(args, *, on_delta=None, on_event=None):
    return _shim_cli("zcode", args, on_delta=on_delta, on_event=on_event)


def _shim_agent(args, *, on_delta=None, on_event=None):
    from sharedagents_agent.agents.delegate_shims import shim_run_delegate_agent_async

    return shim_run_delegate_agent_async(args, on_delta=on_delta, on_event=on_event)


def _calling_agent_name() -> str:
    """The profile whose turn is issuing this Agent call.

    Mirrors the write gate's resolution order (``$SHAREDAGENTS_AGENT_PROFILE`` then
    ``$SHAREDAGENTS_AGENT_NAME``) so the delegation roster and the write roots are
    always decided about the same identity.
    """
    import os as _os

    return (_os.environ.get("SHAREDAGENTS_AGENT_PROFILE") or "").strip() or (
        _os.environ.get("SHAREDAGENTS_AGENT_NAME") or ""
    ).strip()


def _delegation_refusal_for_caller(target: str):
    """Why the calling profile may not spawn *target*, or ``None`` if it may.

    A child runs under its OWN profile's grants, so an unrestricted roster lets
    a narrow parent act through a wider child. Unknown caller ⟹ unrestricted:
    identity here is best-effort and must not break delegation fleet-wide.
    """
    from sharedagents_agent.agent_profiles import delegation_target_refusal

    return delegation_target_refusal(_calling_agent_name(), target)


def _shim_agent_sync(args, *, on_delta=None, on_event=None):
    """``AgentSync``: force inline wait, then the canonical dispatcher."""
    if not isinstance(args, dict):
        args = {}
    args = dict(args)
    args["background"] = False
    args["_sharedagents_agent_variant"] = "sync"
    return _shim_agent_canonical(args, on_delta=on_delta, on_event=on_event)


def _shim_agent_async(args, *, on_delta=None, on_event=None):
    """``AgentAsync``: force detached handles, then the canonical dispatcher."""
    if not isinstance(args, dict):
        args = {}
    args = dict(args)
    args["background"] = True
    args["_sharedagents_agent_variant"] = "async"
    return _shim_agent_canonical(args, on_delta=on_delta, on_event=on_event)


_READONLY_LOOKUP_SEATS = frozenset({"explore", "research", "general"})


def _scope_readonly_lookup(args: dict[str, Any], subagent_type: str) -> dict[str, Any]:
    """Scope a read-only lookup to the parent session's ``workspace_cwd``.

    The model routinely copies the prompt's ``source folder:`` line (the parent
    of all repos) into ``cwd``, turning a one-repo lookup into a whole-tree
    sweep. For the read-only seats only, rewrite a missing or source-folder
    cwd to the parent session's workspace. An explicit narrower cwd is
    preserved, and build/plan are never rewritten.
    """
    if subagent_type not in _READONLY_LOOKUP_SEATS:
        return args
    parent = str(args.get("_sharedagents_parent_session_id") or os.environ.get("SHAREDAGENTS_SESSION_ID") or "").strip()
    if not parent:
        return args
    try:
        from sharedagents_core.layout import sessions_root
        from sharedagents_session.session_index import connect_sessions_db

        con = connect_sessions_db(Path(sessions_root()) / "sessions.db")
        try:
            row = con.execute(
                "SELECT workspace_cwd FROM sessions_meta WHERE session_hash=?",
                (parent,),
            ).fetchone()
        finally:
            con.close()
        parent_cwd = str((dict(row).get("workspace_cwd") if row is not None else "") or "").strip()
    except Exception:  # noqa: BLE001 — scoping is advisory, never fatal
        return args
    if not parent_cwd:
        return args
    source = os.environ.get("SOURCE_FOLDER", "").strip().rstrip("/")
    requested = str(args.get("cwd") or "").strip().rstrip("/")
    parent_norm = parent_cwd.rstrip("/")
    if not requested or source and requested == source and parent_norm != source:
        args["cwd"] = parent_cwd
    return args


def _shim_agent_canonical(args, *, on_delta=None, on_event=None):
    """Dispatcher for first-party ``AgentSync`` / ``AgentAsync``.

    The model sees ``AgentSync`` (wait inline) or ``AgentAsync`` (detach)
    — never a ``background`` argument and never a tool named ``Agent``.
    The runner routes to the appropriate internal shim:

      - ``general-purpose`` → ``shim_run_delegate_agent_async`` on
        the sibling sharedagents agent named by ``name`` (defaults to
        ``general``; `main` retired).
      - ``claude`` / ``codex`` / ``antigravity`` → the shared CLI shim.

    Lifecycle CONTROL: ``action=list|abort|steer`` branches
    before dispatch and does not require ``subagent_type``.

    Param normalisation:
      - the canonical ``prompt`` is forwarded as-is to claude / codex
        / antigravity shims; the agent shim wants ``task`` so we mirror
        ``prompt`` → ``task`` if absent.
      - ``description`` lives in the args dict (consumed by the TUI
        subagent-box for the header label) but is not part of the
        underlying shim's contract; safely ignored.
    """
    if not isinstance(args, dict):
        args = {}
    args = dict(args)
    # Distinguish first-party Agent lifecycle semantics (per-agent background
    # default + public task_id) from named slash-command delegate sessions.
    args["_sharedagents_agent_tool"] = True
    # CONTROL verbs (list / abort / steer) — before fan-out or dispatch.
    _action = str(args.get("action") or "dispatch").strip().lower()
    if _action in ("list", "abort", "steer"):
        if _action == "list":
            return run_check_delegations_async(
                args,
                on_delta=on_delta,
                on_event=on_event,
            )
        if _action == "abort":
            return run_cancel_delegation_async(
                args,
                on_delta=on_delta,
                on_event=on_event,
            )
        return run_steer_delegation_async(
            args,
            on_delta=on_delta,
            on_event=on_event,
        )
    if _action not in ("", "dispatch"):

        async def _bad_action():
            from sharedagents_agent.tools.runners import ToolResult

            return ToolResult(
                content=(f"[Agent: unknown action {_action!r} — use dispatch, list, abort, or steer]"),
                ok=False,
                outcome="refused",
            )

        return _bad_action()
    # A model-facing ``task_id`` can be a polling/continuation mistake.  Check
    # live durable jobs before routing to a shim so repeated status calls never
    # create another detached runner for the same child session.
    _status_guard = _agent_task_status_guard(args)
    if _status_guard is not None:

        async def _status_result():
            return _status_guard

        return _status_result()
    # A fresh spawn with no prompt creates a child whose ENTIRE user turn is
    # empty: it answers "No response requested." and the caller pays a full
    # spawn plus a round trip to discover it (measured: 26 of 293 delegated
    # turns in one week, across review/build/hermes/zcode/general). Refuse
    # before any reservation or process spawn. ``task_id`` continuations are
    # validated by the status guard above; ``runs`` entries carry their own
    # prompt and are validated per entry.
    if not str(args.get("task_id") or "").strip() and not _has_runs(args.get("runs")):
        _task_text = next(
            (
                str(args.get(key)).strip()
                for key in ("prompt", "task", "message", "input")
                if isinstance(args.get(key), str) and str(args.get(key)).strip()
            ),
            "",
        )
        if not _task_text:

            async def _missing_prompt():
                from sharedagents_agent.tools.runners import ToolResult

                return ToolResult(
                    content=(
                        "[Agent: missing `prompt` — the child cannot see this "
                        "conversation, so a spawn without one is an empty turn "
                        'that answers "No response requested." Pass the full '
                        "task text in `prompt`.]"
                    ),
                    ok=False,
                    outcome="refused",
                )

            return _missing_prompt()
    # Merged fan-out (formerly the ``BatchAgent`` tool): an ``Agent`` call
    # carrying a non-empty ``runs`` list dispatches every entry concurrently
    # via ``_shim_batch_agent``. ``runs`` may arrive as a native list or a
    # JSON-encoded string (weak models); the batch runner normalises both.
    # Per-entry dispatch re-enters this function WITHOUT a ``runs`` key, so
    # there is no recursion.
    # Route to the batch runner when the caller genuinely uses the fan-out field:
    # a NON-EMPTY list (fan-out), a native dict (one entry — parity with the
    # JSON-encoded-object form weak models also emit), or a non-empty JSON string.
    # An EMPTY list is NOT a batch request: `Agent(runs=[], subagent_type="code")`
    # is an empty fan-out that falls through to a normal single dispatch of
    # `code`. An empty string / None likewise falls
    # through.
    _runs = args.get("runs")
    if isinstance(_runs, dict) or (isinstance(_runs, list) and _runs) or (isinstance(_runs, str) and _runs.strip()):
        return _shim_batch_agent(args, on_delta=on_delta, on_event=on_event)
    subagent_type = str(args.get("subagent_type") or "").strip().lower()
    # Web research needs no reroute: `general` (mode=auto, carries the mcp__web__*
    # web tools) handles it natively now that the dedicated `research` profile is
    # retired. Native research publishes through its ontology tools.
    # Single-phase specialist reroute (the generalisation of the web-research hook
    # above). When the orchestrator falls back to a GENERIC agent (explore/general)
    # for a task that unambiguously asks for ONE specialist deliverable
    # (write a spec, run tests, deploy, review a diff, …), reroute to that
    # specialist. Keyed on the ORIGINAL user task threaded in as
    # ``_sharedagents_user_task`` by the turn runtime — NOT the orchestrator's paraphrase
    # to the subagent, which strips the intent (measured: intent survives in the
    # task 84/95 vs the paraphrase 5/95). No-op when the field is absent, so this
    # is inert until the runtime feeds it (backward-compatible). Measured lift on
    # 511 simulated tasks: 41.5% → 61.8% routing accuracy, no regression on
    # explore/general/research/code (those match no pattern).
    _user_task = str(args.get("_sharedagents_user_task") or "")
    if _user_task and subagent_type in ("general-purpose", "gp"):
        pass  # named-sibling route owns its own name arg; don't second-guess it
    elif _user_task:
        from sharedagents_agent.subagent_policy import GENERIC_ROUTE_TARGETS, specialist_intent

        if subagent_type in GENERIC_ROUTE_TARGETS:
            # PRIMARY: the high-precision regex intent detector (verb + object).
            # FALLBACK: the embedding semantic router's prediction, computed
            # upstream in the async engine and threaded in as ``_sharedagents_route_hint``
            # (this shim is sync and can't await an embedder) with generic-protection
            # + margin so it only fires for a confident specialist. Regex-first was
            # measured cleaner than embedding-first on the 511-task benchmark (the
            # embedding alone confuses spec/plan/requirements on surface words).
            # Both only fire for a task that clearly wants ONE specialist, so
            # explore-first for a real multi-step CHANGE is untouched.
            _want = specialist_intent(_user_task)
            _via = "regex"
            if not _want:
                _want = str(args.get("_sharedagents_route_hint") or "").strip().lower() or None
                _via = "embed"
            # The reroute may never WIDEN authority. It rewrites the model's
            # target using the parent's own user task, so for a restricted
            # parent it used to pick the wrong side: `plan` asked for
            # `explore` and its planning task was rewritten to `build` —
            # five explore requests became five build children in one
            # 2026-07-25 session, handing read-only plan a child that writes
            # the whole repo. Honour the caller's roster and keep the model's
            # own (narrower) choice when the reroute is not permitted for it.
            if _want and _delegation_refusal_for_caller(_want):
                if _ROUTE_TRACE:
                    import sys as _sys

                    print(
                        f"[route-reroute-blocked] {subagent_type}->{_want} "
                        f"not permitted for caller; keeping {subagent_type}",
                        file=_sys.stderr,
                        flush=True,
                    )
                _want = None
            if _want and _want != subagent_type and _want not in ("explore", "general"):
                if _ROUTE_TRACE:
                    import sys as _sys

                    print(
                        f"[route-reroute] {subagent_type}->{_want} via={_via} task={_user_task[:70]!r}",
                        file=_sys.stderr,
                        flush=True,
                    )
                subagent_type = _want
                args["subagent_type"] = _want
    if not subagent_type:
        from sharedagents_agent.agent_profiles import delegation_roster_text
        from sharedagents_agent.tools.runners import ToolResult

        async def _err_async():
            return ToolResult(
                content=(
                    "[Agent: missing `subagent_type` — pick the one whose job "
                    "matches the task:\n" + delegation_roster_text() + "]"
                ),
                ok=False,
                outcome="refused",
            )

        return _err_async()
    # Authority chokepoint. Every dispatch path below — external engine, sibling
    # shim, named worker — passes through here, so a target the caller's profile
    # does not permit is refused ONCE, whether the model asked for it or the
    # reroute produced it. Refusing is the whole point: a child does not inherit
    # the parent's limits, so an unchecked spawn is a privilege escalation, not
    # a routing mistake.
    _refusal = _delegation_refusal_for_caller(subagent_type)
    if _refusal:
        from sharedagents_agent.tools.runners import ToolResult

        async def _refuse_async():
            return ToolResult(content=f"[Agent: {_refusal}]", ok=False, outcome="refused")

        return _refuse_async()
    if _ROUTE_TRACE:
        import sys as _sys

        print(f"[route-dispatch] -> {subagent_type}", file=_sys.stderr, flush=True)
    # Mirror canonical ``prompt`` → ``task`` for the agent-sibling
    # shim (the canonical Claude-Code Agent uses ``prompt``; sharedagents's
    # delegate_agent shim historically uses ``task``).
    if "prompt" in args and "task" not in args:
        args["task"] = args["prompt"]
    args = _scope_readonly_lookup(args, subagent_type)
    if subagent_type in ("claude", "delegate_claude"):
        return _shim_cli("claude", args, on_delta=on_delta, on_event=on_event)
    if subagent_type in ("codex", "delegate_codex"):
        return _shim_cli("codex", args, on_delta=on_delta, on_event=on_event)
    if subagent_type in ("antigravity", "delegate_antigravity"):
        return _shim_cli("antigravity", args, on_delta=on_delta, on_event=on_event)
    if subagent_type in ("zcode", "delegate_zcode"):
        return _shim_cli("zcode", args, on_delta=on_delta, on_event=on_event)
    # ---- shared delegation-target validation ----
    from sharedagents_agent.agent import agents_root
    from sharedagents_agent.agent_profiles import delegation_roster_text, resolve_profile
    from sharedagents_agent.agents.delegate_shims import shim_run_delegate_agent_async
    from sharedagents_agent.tools.runners import ToolResult

    # The orchestrator delegates; it is never a delegation TARGET. Hidden runtime
    # profiles (compaction/title/summary) are internal-only. (`main` was retired
    # 2026-07-21 — jarvis is the sole orchestrator.)
    _NON_DELEGATABLE_NAMES = {"jarvis"}

    def _is_hidden(nm: str) -> bool:
        pr = resolve_profile((nm or "").strip().lower())
        return pr is not None and pr.opencode_mode == "hidden"

    def _not_a_target(nm: str) -> bool:
        # LITERAL check — used for a caller-supplied worker IDENTITY (the ``name``
        # arg). A retired name like ``main`` is a legitimate arbitrary worker
        # identity; only the live orchestrator (jarvis) and hidden runtimes are
        # forbidden as identities. Profile-DISPATCH aliasing is guarded separately
        # (``_profile_not_a_target`` below) so ``subagent_type=main`` -> jarvis
        # can't reach the live orchestrator.
        nm = (nm or "").strip().lower()
        return nm in _NON_DELEGATABLE_NAMES or _is_hidden(nm)

    def _profile_not_a_target(nm: str) -> bool:
        # ALIAS-RESOLVING check — for ``subagent_type`` (a profile to dispatch).
        # Resolves aliases (``main`` -> ``jarvis``) to the CANONICAL profile so an
        # alias can never dispatch to the live orchestrator / a hidden runtime.
        pr = resolve_profile((nm or "").strip().lower())
        return pr is not None and (pr.name in _NON_DELEGATABLE_NAMES or pr.opencode_mode == "hidden")

    def _reject(nm: str, reason: str):
        async def _a():
            return ToolResult(
                content=(f"[Agent: `{nm}` {reason}. Pick a roster agent —\n" + delegation_roster_text() + "]"),
                ok=False,
                outcome="refused",
            )

        return _a()

    # Named-sibling route: ``general-purpose`` / ``gp`` select the target agent by
    # the ``name`` arg (default ``general`` — the generic worker profile; `main`
    # was retired). Validate the chosen name so ``general-purpose``+
    # ``name=jarvis|compaction`` can't bypass the orchestrator/hidden refusal.
    if subagent_type in ("general-purpose", "gp"):
        # Default to ``general`` (the generic worker — Claude Code's
        # "general-purpose" convention). An empty / whitespace-only name is
        # "absent" → also defaults to general. The orchestrator (jarvis) and any
        # HIDDEN runtime agent (compaction/title/summary) can never be a target —
        # refuse them even by explicit ``name``.
        target = (args.get("name") or "").strip() or "general"
        if _not_a_target(target):
            return _reject(target, "is not a delegation target (orchestrator / hidden runtime agent)")
        args["name"] = target
        return shim_run_delegate_agent_async(args, on_delta=on_delta, on_event=on_event)

    # Direct dispatch to an orchestrator or hidden profile → refuse. Alias-resolve
    # too, so ``subagent_type=main`` (aliased to jarvis) can't reach the live
    # orchestrator by slipping past the literal name check.
    if _not_a_target(subagent_type) or _profile_not_a_target(subagent_type):
        return _reject(subagent_type, "is not a delegation target (orchestrator / hidden runtime agent)")

    # Known profile sibling (explore / plan / build / general / research / …).
    _prof = resolve_profile(subagent_type)
    if _prof is not None:
        if (agents_root() / _prof.name / "config.json").exists():
            args["name"] = _prof.name
        else:
            # No scaffolded agent for this profile: run its ROLE (_agent_profile)
            # on a generic mount. Honor a caller-supplied worker name, but NEVER
            # an orchestrator (jarvis) or hidden name — that would smuggle a
            # forbidden target in as the session identity. `general` is the
            # generic default host (the built-in `main` host was retired
            # 2026-07-21).
            req_name = (args.get("name") or "").strip()
            if req_name and _not_a_target(req_name):
                return _reject(req_name, "is not a delegation target (orchestrator / hidden runtime agent)")
            args["_agent_profile"] = _prof.name
            args["name"] = req_name or "general"
        args["mode"] = _prof.mode
        args.setdefault("subagent_type", _prof.name)
        return shim_run_delegate_agent_async(args, on_delta=on_delta, on_event=on_event)

    # Unknown subagent_type → a custom on-disk sibling selected by ``name`` (e.g.
    # ``opencode``/``cursor`` external engines, or a user-created ``scraper``). An
    # empty / whitespace-only name is "absent" for routing (round-7 codex: an empty
    # name previously bypassed the typo guard and dispatched an empty target). If
    # there is neither a real ``name`` nor an agent dir of that name, it is a
    # typo/unknown value → surface the roster rather than silently spawning it.
    req_name = (args.get("name") or "").strip()
    target = req_name or subagent_type
    if _not_a_target(target):
        return _reject(target, "is not a delegation target")
    if not req_name and not (agents_root() / subagent_type / "config.json").exists():
        return _reject(subagent_type, "is not a known agent or roster subagent_type")
    args["name"] = req_name or subagent_type
    return shim_run_delegate_agent_async(args, on_delta=on_delta, on_event=on_event)


# --- per-subagent subprocess fan-out -------------------------------------
#
# ``_spawn_subagent_runner`` runs ONE child ``python -m
# sharedagents_agent.tool_runner`` process per subagent entry and multiplexes
# its JSONL stdout (deltas + tool_events + final record) back to the
# caller via the supplied ``on_delta`` / ``on_event`` callbacks. This
# is the primitive ``_shim_batch_agent`` uses to get true OS-level
# parallelism across subagent entries — versus the prior in-process
# ``asyncio.gather`` over ``_shim_agent_canonical``, which:
#   - crash-coupled every entry (one subagent OOM took the batch)
#   - gave no CPU parallelism for prompt-render / large-JSON-parse
#   - couldn't be SIGTERM'd from the TUI (no PID to signal)
# Wire shape is byte-identical to ``_default_run_tool`` (acp_bridge.py
# L968), so a child subagent run looks the same as any other tool
# dispatch on the wire — feed ``{"tool": "AgentSync", "args": entry}`` on
# stdin, read JSONL events on stdout, terminate on a ``"type": "final"``
# record. Cancel path: parent CancelledError → SIGTERM → 3 s grace →
# SIGKILL, so the ``ps`` table reflects the kill within seconds and the
# TUI's ``/restart`` actually stops subagent work in flight.
#
# ``SHAREDAGENTS_SUBAGENT_INPROCESS=1`` short-circuits this helper in
# ``_shim_batch_agent`` back to the in-process gather so a regression
# escape exists.


async def _spawn_subagent_runner(
    entry: dict[str, Any],
    *,
    on_delta: Callable[[str], Any] | None = None,
    on_event: Callable[[dict], Any] | None = None,
    reservation: sandbox_broker.AgentReservation | None = None,
) -> ToolResult:
    """Run ONE ``AgentSync`` tool call in a dedicated OS process.

    See the module-level commentary above for design rationale and
    wire shape. Returns a :class:`ToolResult` whose ``content`` is the
    subagent turn's final body (or a diagnostic when the runner
    crashed before emitting a final record). Crash-isolated.

    ``reservation`` is a slot the caller already booked for this entry (the
    batch runner books it so an entry can wait for one); it is released here
    exactly like one booked below.
    """
    # Pre-spawn AGENT reservation (spec §5.1): book the slot BEFORE
    # the child OS process exists. cross_process=True atomically earmarks this
    # subagent's ENTIRE subtree entitlement against the session budget (spec
    # §6.4) so the child broker, initialised from the grant we hand
    # it in the env, can never overspend it. On refusal we NEVER spawn.
    _res = reservation
    if _res is None:
        try:
            _res = sandbox_broker.reserve_agent(kind="subagent", cross_process=True)
        except sandbox_broker.SpawnRefused as _sr:
            return ToolResult(content=str(_sr), ok=False)

    request = {"tool": "AgentSync", "args": dict(entry)}
    # The child's top turn is THIS already-counted subagent; mark it so the
    # child does not re-reserve it (spec §6.3 — the grant covers descendants).
    request["args"]["_sharedagents_precounted"] = True
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    # Hint to the child's ``_drive_child_agent_turn`` that this boot
    # should skip MCP discovery / skill-index rebuild / memory provider
    # init — the parent has already paid those costs once for the
    # gateway process and the subagent only needs to dispatch a single
    # Agent.run_turn.
    env["SHAREDAGENTS_SUBAGENT_MINIMAL"] = "1"
    # The parent turn's pinned grants ride along.
    from sharedagents_session.access_binding import apply_access_pin_env

    apply_access_pin_env(
        env, str(entry.get("_sharedagents_parent_session_id") or os.environ.get("SHAREDAGENTS_SESSION_ID") or "")
    )
    # Hand the child broker its delegated sub-budget (spec §6.3).
    env = _res.child_env(env)
    # Not folder-confined: this runs sharedagents orchestration (the child's agent
    # config, its session log, its model's credentials). The child's own host
    # tools are confined through its turn's pin.
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "sharedagents_agent.tool_runner",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            limit=16 * 1024 * 1024,
        )
    except Exception as _spawn_exc:  # noqa: BLE001 — nothing spawned: give the slot back
        _res.release()
        return ToolResult(
            content=f"[subagent runner: spawn failed: {_spawn_exc!r}]",
            ok=False,
        )
    # Bind the child's pid to the reservation's lease for the capability-kill
    # model, then guarantee the whole grant is returned however the child exits.
    _res.bind(proc.pid)
    try:
        return await _spawn_subagent_runner_body(
            proc,
            request,
            on_delta=on_delta,
            on_event=on_event,
        )
    finally:
        _res.release()


async def _spawn_subagent_runner_body(
    proc,
    request: dict,
    *,
    on_delta: Callable[[str], Any] | None = None,
    on_event: Callable[[dict], Any] | None = None,
) -> ToolResult:
    """Stream one subagent runner child's JSONL stdout back to the caller.

    Split out of :func:`_spawn_subagent_runner` so the reservation
    (spec §5.1) wraps the entire lifetime in a single ``finally``.
    """
    if proc.stdin is None:
        return ToolResult(
            content="[subagent runner: stdin pipe is None]",
            ok=False,
        )
    proc.stdin.write(json.dumps(request).encode("utf-8"))
    await proc.stdin.drain()
    proc.stdin.close()

    final: dict | None = None
    try:
        if proc.stdout is None:
            return ToolResult(
                content="[subagent runner: stdout pipe is None]",
                ok=False,
            )
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            try:
                obj = json.loads(line.decode("utf-8", "replace"))
            except json.JSONDecodeError:
                continue
            t = obj.get("type")
            if t == "delta":
                if on_delta is not None:
                    txt = obj.get("text") or ""
                    if txt:
                        try:
                            r = on_delta(txt)
                            if hasattr(r, "__await__"):
                                await r
                        except Exception:  # noqa: BLE001
                            pass
                continue
            if t == "tool_event":
                if on_event is not None:
                    try:
                        r = on_event(
                            {k: v for k, v in obj.items() if k != "type"},
                        )
                        if hasattr(r, "__await__"):
                            await r
                    except Exception:  # noqa: BLE001
                        pass
                continue
            # Only a ``final`` record (or the legacy untyped shape) is the
            # result; any other typed record is skipped, never taken for it.
            if final is None and t in (None, "final"):
                final = obj
    except asyncio.CancelledError:
        # Hard cancel (parent /abort / second ESC — graceful stop is
        # cooperative and never cancels the task, so reaching here means a
        # force kill). SIGKILL the subagent's ENTIRE tree on this tick: a
        # bare ``proc.terminate()``/``proc.kill()`` signals ONLY this
        # ``tool_runner`` proc and orphans the delegate CLI (claude/codex)
        # it spawned in its OWN session — exactly the subagent that must
        # die. ``_kill_tree`` reaches those grandchildren via the /proc
        # descendant walk, and CRUCIALLY does NOT killpg this proc's group
        # (the subagent runner is spawned WITHOUT start_new_session, so it
        # shares the GATEWAY's group — a killpg here would kill the
        # gateway). No grace window: the operator asked to kill NOW.
        _kill_tree(proc, oscompat.FORCE_KILL)
        raise

    try:
        await proc.wait()
    except Exception:  # noqa: BLE001
        pass

    if final is None:
        head = ""
        if proc.stderr is not None:
            try:
                err = await proc.stderr.read()
                lines = err.decode("utf-8", "replace").strip().splitlines()
                head = (lines or [""])[-1][:200]
            except Exception:  # noqa: BLE001
                pass
        return ToolResult(
            content=(f"[subagent runner crashed (rc={proc.returncode}): {head or '<no stderr>'}]"),
            ok=False,
        )
    return ToolResult(
        content=str(final.get("content") or ""),
        ok=bool(final.get("ok", False)),
        truncated=bool(final.get("truncated", False)),
        background_handle=bool(final.get("background_handle", False)),
        outcome=final.get("outcome"),
        member_outcomes=tuple(final.get("member_outcomes") or ()),
    )


def _batch_entry_attribution(
    entry: dict,
    idx: int,
    batch_cid: str,
    batch_parent: str,
) -> dict:
    """Stamp delegate attribution onto one BatchAgent entry.

    The batch wrapper owns ``_sharedagents_call_id`` (one per entry,
    derived as ``"<batch_cid>.<i>"``) and
    ``_sharedagents_parent_session_id`` (the dispatching session's sid).
    A model-supplied value in the entry's dict is REPLACED — not
    preserved — so an entry can't spoof a different session as its
    parent. When the batch has no trusted ``batch_parent`` (CLI
    invocation outside a session, unit tests), any per-entry
    parent_session_id is stripped instead of accepted. Same
    authority rule as the bridge's
    ``_inject_subagent_attribution``.
    """
    e = dict(entry)
    # The wave origin is runtime-owned and set only on the top-level call;
    # an entry (including a nested batch) never supplies its own.
    e.pop("_sharedagents_wave_origin", None)
    e["_sharedagents_call_id"] = f"{batch_cid}.{idx}"
    if batch_parent:
        e["_sharedagents_parent_session_id"] = batch_parent
    else:
        e.pop("_sharedagents_parent_session_id", None)
    return e


async def _shim_batch_agent(args, *, on_delta=None, on_event=None):
    """BatchAgent runner — fan out N ``Agent``-equivalent calls concurrently.

    The model emits ONE BatchAgent tool_call with ``runs=[{…}, {…}, …]``
    and the runner dispatches every entry through ``_shim_agent_canonical``
    in parallel via ``asyncio.gather``. This sidesteps the small-model
    failure mode where qwen3.6-27b-fp8 (and similar) can't reliably
    emit N tool_calls in one assistant message even when the registry
    description begs them to.

    Per-entry behaviour is byte-identical to a standalone ``Agent``
    call — same session persistence, same ``tool_use`` event streaming,
    same TUI subagent box. We just multiplex through one parent
    ToolResult whose body concatenates each run's output with a
    ``=== run <i> [<subagent_type>:<name>] ===`` header so the model
    can parse them apart.

    Streaming: ``on_delta`` deltas from each entry are prefixed with
    ``[run <i>] `` so the parent transcript stays interleaved-but-
    attributable. ``on_event`` events get ``batch_idx`` injected so
    the TUI subagent panel can distinguish them.
    """
    from sharedagents_agent.tools.runners import ToolResult

    if not isinstance(args, dict):
        args = {}
    runs = args.get("runs")
    # Weak / local models (e.g. qwen3.6-27b-fp8) routinely emit a
    # structured argument as a JSON-*encoded string* — i.e.
    # ``runs: "[{...}]"`` instead of the native array ``runs: [{...}]``.
    # The native-array path is unchanged; we only attempt a decode when
    # ``runs`` arrives as a non-empty string so a perfectly valid fan-out
    # request isn't rejected over a serialisation quirk.
    if isinstance(runs, str):
        stripped = runs.strip()
        if stripped:
            try:
                decoded = json.loads(stripped)
            except (ValueError, TypeError):
                decoded = None
            if isinstance(decoded, list):
                runs = decoded
            elif isinstance(decoded, dict):
                runs = [decoded]
    # A native dict is a single entry — mirror the JSON-encoded-object path above
    # so ``runs={...}`` and ``runs='{...}'`` behave identically (round-6 antigravity).
    elif isinstance(runs, dict):
        runs = [runs]
    if not isinstance(runs, list) or not runs:
        return ToolResult(
            content=(
                "[BatchAgent: missing or empty `runs` list — pass "
                "`runs: [{subagent_type, description, prompt}, …]` "
                "with at least one entry]"
            ),
            ok=False,
            outcome="refused",
        )
    # Top-level fields (cwd, model, checklist fields, subagent_type/prompt
    # defaults, …) are shared defaults for every entry; an entry's own value
    # wins. A continuation id belongs to one child, so it never fans out.
    # Attribution is stamped per entry afterwards by ``_entry_args``.
    shared = {k: v for k, v in args.items() if k not in ("runs", "task_id")}

    # Validate every entry up-front so we don't half-spawn before
    # surfacing a schema error. Mirror _shim_agent_canonical's required
    # field — ``subagent_type`` — and reject anything else as a clean
    # error rather than dispatching garbage.
    for i, entry in enumerate(runs):
        # Same serialisation quirk one level down: some models JSON-encode
        # each entry too (``runs: ["{...}", "{...}"]``). Decode a
        # stringified entry back to a dict before the type check and write
        # it back so the dispatch loop below sees the parsed object.
        if isinstance(entry, str):
            try:
                decoded_entry = json.loads(entry)
            except (ValueError, TypeError):
                decoded_entry = None
            if isinstance(decoded_entry, dict):
                entry = decoded_entry
                runs[i] = entry
        if not isinstance(entry, dict):
            return ToolResult(
                content=(f"[BatchAgent: runs[{i}] must be an object, got {type(entry).__name__}]"),
                ok=False,
                outcome="refused",
            )
        entry = {**shared, **entry}
        runs[i] = entry
        if not str(entry.get("subagent_type") or "").strip():
            from sharedagents_agent.agent_profiles import delegation_roster_text

            return ToolResult(
                content=(
                    f"[BatchAgent: runs[{i}] missing `subagent_type` — pick the "
                    "one whose job matches the task:\n" + delegation_roster_text() + "]"
                ),
                ok=False,
                outcome="refused",
            )

    # Per-entry callbacks that prefix the delta / tag the event so the
    # interleaved output stays attributable to its source run.
    def _make_on_delta(idx):
        if on_delta is None:
            return None

        async def _wrapped(text):
            try:
                res = on_delta(f"[run {idx}] {text}")
                if hasattr(res, "__await__"):
                    await res
            except Exception:  # noqa: BLE001
                pass

        return _wrapped

    def _make_on_event(idx):
        if on_event is None:
            return None

        async def _wrapped(ev):
            try:
                tagged = dict(ev or {})
                tagged["batch_idx"] = idx
                res = on_event(tagged)
                if hasattr(res, "__await__"):
                    await res
            except Exception:  # noqa: BLE001
                pass

        return _wrapped

    # Schedule all entries concurrently — ONE OS PROCESS PER ENTRY,
    # not an in-process coroutine. ``_spawn_subagent_runner`` re-enters
    # ``python -m sharedagents_agent.tool_runner`` per entry and multiplexes
    # its JSONL stdout back to the caller via the per-entry
    # ``_make_on_delta`` / ``_make_on_event`` callbacks. Wire shape is
    # byte-identical to the gateway-side ``_default_run_tool`` spawn,
    # so the child subagent run looks the same as any other tool
    # dispatch on the wire (the child runner itself ends up calling
    # ``_shim_agent_canonical`` once inside its own process).
    #
    # Trade-offs vs the prior in-process gather:
    #   + true CPU parallelism for any subagent CPU-bound work
    #     (prompt-template render, large JSON parse on tool results)
    #   + crash-isolation: one subagent OOM / segfault / wedge returns
    #     a failed ToolResult for its slot instead of killing the batch
    #   + SIGTERM-able: ESC / /restart / /new propagates into every
    #     subagent OS process via ``_spawn_subagent_runner``'s cancel
    #     path (SIGTERM → 3 s grace → SIGKILL)
    #   − adds ~150 ms boot overhead per entry (one fork+exec each)
    #     plus the ``SHAREDAGENTS_SUBAGENT_MINIMAL=1`` lean-boot fast path
    #     inside ``_drive_child_agent_turn`` — cheap vs a subagent turn
    # ``SHAREDAGENTS_SUBAGENT_INPROCESS=1`` short-circuits back to the
    # in-process gather so a regression escape exists (e.g. when the
    # subprocess pattern misbehaves under heavy fan-out, an operator
    # can fall back to today's behaviour without redeploying).
    inprocess = os.environ.get("SHAREDAGENTS_SUBAGENT_INPROCESS") == "1"
    # A session per subagent: the batch is ONE tool call with ONE call
    # id, but each entry is an independent subagent that must get its
    # OWN child session. Derive a unique per-entry call id from the
    # batch's so ``_drive_child_agent_turn`` keys a fresh session per
    # entry (without this, every entry of the same engine collided on —
    # and concurrently raced — one shared session). The parent sid is
    # propagated so each child back-links to the dispatching session.
    batch_cid = str(args.get("_sharedagents_call_id") or "batch")
    batch_parent = str(args.get("_sharedagents_parent_session_id") or os.environ.get("SHAREDAGENTS_SESSION_ID") or "")

    from sharedagents_agent.agents import async_jobs

    # Open before dispatch: early finishers wait until registration is sealed.
    wave_id = ""

    def _entry_args(i: int, entry: dict) -> dict:
        item = _batch_entry_attribution(entry, i, batch_cid, batch_parent)
        item["_sharedagents_agent_tool"] = True
        # Batch siblings are an intentional fan-out, even when their prompts
        # share the same codebase vocabulary. Keep the live-task dedupe guard
        # for accidental sequential redispatches out of this explicit batch.
        item["_sharedagents_batch"] = True
        item["_sharedagents_wave_id"] = wave_id
        # A top-level per-call override governs the fan-out unless an entry
        # supplies its own explicit override.
        if "background" not in item and "background" in args:
            item["background"] = args["background"]
        return item

    def _entry_wants_background(item: dict) -> bool:
        # Profile policy (jarvis background-only; build/plan foreground-only)
        # pins the mode ahead of the entry's explicit value and the .env
        # default — same resolution the single-dispatch path uses.
        from sharedagents_agent.background_subagents import (
            apply_delegation_background_to_args,
        )

        apply_delegation_background_to_args(item)
        return bool(item.get("background"))

    # Wrap each dispatch so we record its own wall-clock. The batch is the
    # goal-spawn primitive ("spawn N agents, each answers inside the period"),
    # so per-run + total elapsed is the signal the orchestrator needs to tell
    # WHICH agent overran and whether the batch fit the budget. The wrapper
    # swallows the exception into the returned pair so gather never re-raises
    # (parity with the prior return_exceptions=True).
    async def _timed(coro):
        start = time.monotonic()
        try:
            r = await coro
        except BaseException as exc:  # noqa: BLE001 — mirror gather(return_exceptions)
            return exc, time.monotonic() - start
        return r, time.monotonic() - start

    # Resolve the per-entry dispatch functions through the ``runners``
    # re-export so tests that ``monkeypatch.setattr(runners, "_shim_agent_canonical"
    # / "_spawn_subagent_runner", …)`` still intercept every batch entry after the
    # de-monolith split (local shadow keeps the call sites below unchanged).
    import sharedagents_agent.tools.runners as _runners

    _shim_agent_canonical = getattr(_runners, "_shim_agent_canonical", globals()["_shim_agent_canonical"])
    _spawn_subagent_runner = getattr(_runners, "_spawn_subagent_runner", globals()["_spawn_subagent_runner"])

    # A per-process entry books its whole subtree up front, so the session
    # budget admits only a few entries at once. An entry
    # that does not fit waits for a sibling OF THIS BATCH to return its slot
    # instead of coming back refused. The budget still bounds what runs at
    # once. When no sibling holds a slot the budget is held elsewhere and
    # nothing here can free it, so the refusal is returned as before.
    slot_freed = asyncio.Condition()
    holding = 0
    queued_s: dict[int, float] = {}

    async def _spawn_when_admitted(i: int, entry_args: dict) -> ToolResult:
        nonlocal holding
        asked = time.monotonic()
        waited = False
        async with slot_freed:
            while True:
                try:
                    reservation = sandbox_broker.reserve_agent(
                        kind="subagent",
                        cross_process=True,
                    )
                    break
                except sandbox_broker.SpawnRefused as refused:
                    if refused.reason != "session_total" or not holding:
                        return ToolResult(content=str(refused), ok=False, outcome="refused")
                    waited = True
                    await slot_freed.wait()
            holding += 1
        if waited:
            queued_s[i] = time.monotonic() - asked
        try:
            return await _spawn_subagent_runner(
                entry_args,
                on_delta=_make_on_delta(i),
                on_event=_make_on_event(i),
                reservation=reservation,
            )
        finally:
            # The runner returns the slot itself; releasing again is a no-op
            # and covers a cancel that lands before it took ownership.
            reservation.release()
            holding -= 1
            async with slot_freed:
                slot_freed.notify_all()

    entries = [_entry_args(i, entry) for i, entry in enumerate(runs)]
    background_entries = [_entry_wants_background(entry) for entry in entries]
    if any(background_entries):
        # A nested batch (a run that is itself a batch) joins its enclosing
        # wave instead of opening a second one for the same request.
        _join = async_jobs.requested_wave_id(args.get("_sharedagents_wave_id") or args.get("wave_id"))
        if _join:
            # Mid-flight edit for the same goal: append these children to the
            # caller's in-flight wave instead of opening a new one. An unknown,
            # foreign or closed wave refuses the whole batch before any entry
            # is dispatched.
            try:
                async_jobs.ensure_wave_joinable(batch_parent, _join)
            except ValueError as exc:
                return ToolResult(content=f"[BatchAgent: {exc}]", ok=False, outcome="refused")
            wave_id = _join
        else:
            wave_id = async_jobs.create_wave(
                batch_parent,
                origin=str(args.get("_sharedagents_wave_origin") or ""),
            )
        for entry in entries:
            entry["_sharedagents_wave_id"] = wave_id
    # A wave is admitted as ONE unit: either every stated child budget fits
    # under the parent's remaining budget (after live reservations) or the
    # whole batch is refused before any child spawns. Foreground leaves are
    # covered by the group reservation held until ``gather`` returns; a
    # BACKGROUND leaf outlives this batch, so it reserves its own budget when
    # its detached runner starts (the early check below still refuses an
    # over-budget wave up front).
    from sharedagents_agent.agents import budget_reservations

    _budget_amounts = [budget_reservations.parse_budget_usd(e.get("budget_usd")) for e in entries]
    _budget_group = None
    if any(_budget_amounts):
        if not budget_reservations.can_admit(
            batch_parent,
            sum(_budget_amounts),
        ):
            return ToolResult(
                content=budget_reservations.refusal_message(
                    batch_parent,
                    sum(_budget_amounts),
                ),
                ok=False,
                outcome="refused",
            )
        _foreground_amounts = [amount if not background_entries[i] else 0.0 for i, amount in enumerate(_budget_amounts)]
        if any(_foreground_amounts):
            _budget_group = budget_reservations.reserve_many(
                batch_parent,
                _foreground_amounts,
                keys=[f"{batch_cid}:{i}" for i in range(len(entries))],
            )
            if _budget_group is None:
                return ToolResult(
                    content=budget_reservations.refusal_message(
                        batch_parent,
                        sum(_foreground_amounts),
                    ),
                    ok=False,
                    outcome="refused",
                )
            for i, entry in enumerate(entries):
                if not background_entries[i]:
                    entry["_sharedagents_budget_reserved"] = True
    coros = []
    for i, entry_args in enumerate(entries):
        # A background entry must dispatch in this process. Sending it through
        # _spawn_subagent_runner would pre-reserve one Agent slot and then make
        # _maybe_background reserve a second slot before spawning the detached
        # runner (nested/double-counted semantics).
        if inprocess or background_entries[i]:
            coros.append(
                _shim_agent_canonical(
                    entry_args,
                    on_delta=_make_on_delta(i),
                    on_event=_make_on_event(i),
                )
            )
        else:
            coros.append(_spawn_when_admitted(i, entry_args))
    batch_start = time.monotonic()
    try:
        pairs = await asyncio.gather(*(_timed(c) for c in coros))
    finally:
        if _budget_group:
            for _budget_res in _budget_group:
                _budget_res.release()
        # A nested batch leaves sealing to the enclosing batch: sealing here
        # would let a fast sibling deliver the wave before the outer batch
        # finished registering its other runs.
        _nested = bool(async_jobs.requested_wave_id(args.get("_sharedagents_wave_id")))
        member_ids = async_jobs.seal_wave(batch_parent, wave_id) if wave_id and not _nested else []
        # A child can finish before sealing. Actively deliver that wave now;
        # the live reconciler provides recovery if this dispatcher exits.
        from sharedagents_agent.agents.async_delegate_runner import deliver_terminal_job

        for jid in member_ids:
            await deliver_terminal_job(jid)
    total_elapsed = time.monotonic() - batch_start

    # Assemble one combined ToolResult body in CALL ORDER (asyncio.gather
    # preserves order regardless of completion timing).
    chunks: list[str] = []
    overall_ok = True
    for i, (entry, (res, elapsed)) in enumerate(zip(runs, pairs)):
        st = str(entry.get("subagent_type") or "?")
        nm = str(entry.get("name") or "")
        # A queued entry reports its own run time, and its wait separately, so
        # the per-run figure still says WHICH agent was slow.
        queued = queued_s.get(i)
        took = f"{max(0.0, elapsed - queued):.2f}s · queued {queued:.2f}s" if queued is not None else f"{elapsed:.2f}s"
        header = f"=== run {i} [{st}{':' + nm if nm else ''}] · {took} ==="
        if isinstance(res, BaseException):
            overall_ok = False
            chunks.append(f"{header}\n[BatchAgent: run {i} crashed: {res!r}]")
            continue
        body = getattr(res, "content", "") or ""
        if not getattr(res, "ok", True):
            overall_ok = False
        chunks.append(f"{header}\n{body}")
    n = len(runs)
    chunks.append(f"=== batch timing: {n} run{'s' if n != 1 else ''}, total {total_elapsed:.2f}s ===")
    return ToolResult(
        content="\n\n".join(chunks),
        ok=overall_ok,
        outcome="refused" if all(getattr(res, "outcome", None) == "refused" for res, _ in pairs) else None,
        member_outcomes=tuple(
            getattr(res, "outcome", None)
            or (
                "running"
                if getattr(res, "background_handle", False)
                else "done"
                if getattr(res, "ok", False)
                else "failed"
            )
            for res, _ in pairs
        ),
    )


def _lineage_set(own: str) -> set:
    """Caller session + descendants (empty when ``own`` is empty)."""
    if not own:
        return set()
    from sharedagents_agent.agents import async_jobs

    return {own} | async_jobs._descendant_sessions(own)


def _native_control_scope_error(args: dict[str, Any], action: str) -> ToolResult | None:
    """Fail closed when a model-facing Agent control lost session identity."""
    if args.get("_sharedagents_agent_tool") and not str(os.environ.get("SHAREDAGENTS_SESSION_ID") or "").strip():
        return ToolResult(
            content=(
                f"[Agent {action} refused: caller session identity is missing; "
                "native controls never inherit unscoped operator authority]"
            ),
            ok=False,
        )
    return None


def _has_runs(raw: Any) -> bool:
    """True when ``runs`` is a genuine fan-out request (not ``None``/``[]``).

    Mirrors the routing rule below: a native dict, a non-empty list, or a
    non-empty JSON string. The JSON-empty string forms weak models emit
    (``"[]"``, ``"{}"``) are not a fan-out.
    """
    if isinstance(raw, dict):
        return bool(raw)
    if isinstance(raw, list):
        return bool(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return False
        try:
            return bool(json.loads(text))
        except ValueError:
            # Not JSON at all: not a fan-out either. Treat it as absent so the
            # prompt guard applies — a garbage ``runs`` with no prompt is a
            # malformed dispatch, not an excuse to spawn an empty child.
            return False
    return False


def _agent_task_status_guard(args: dict[str, Any]):
    """Prevent a model-facing continuation from spawning a second live turn.

    The canonical ``Agent`` surface is asynchronous for Jarvis.  A small model
    can nevertheless interpret the returned ``task_id`` as a polling handle
    and call ``Agent(task_id=...)`` repeatedly.  Before this guard each call
    entered the background shim again, creating concurrent jobs against the
    same child session.  A live task is a status result; steering it requires
    the explicit ``action=steer`` control path.  Terminal tasks retain the
    existing continuation contract when a real prompt is supplied.

    This is intentionally limited to the model-facing canonical tool.  The
    lower-level delegate shim still supports explicit task continuation for
    trusted callers and for the synchronous foreground path.
    """
    if not args.get("_sharedagents_agent_tool"):
        return None
    task_id = str(args.get("task_id") or "").strip()
    if not task_id:
        return None

    from sharedagents_agent.agents import async_jobs
    from sharedagents_agent.tools.runners import ToolResult

    own = str(os.environ.get("SHAREDAGENTS_SESSION_ID") or "").strip()
    if not own:
        return ToolResult(
            content=(
                "[Agent task continuation refused: caller session identity "
                "is missing; use the runtime-provided task handle]"
            ),
            ok=False,
        )

    lineage = _lineage_set(own)
    matches = []
    outside = False
    try:
        jobs = async_jobs.list_jobs()
    except Exception:  # noqa: BLE001 — status must never break dispatch
        jobs = []
    for job in jobs:
        jid = str(job.get("job_id") or "")
        child_sid = str(job.get("child_sid") or "")
        if task_id not in (jid, child_sid):
            continue
        parent_sid = str(job.get("parent_sid") or "")
        if parent_sid not in lineage:
            outside = True
            continue
        matches.append(job)

    if outside and not matches:
        return ToolResult(
            content=(f"[Agent task_id {task_id} refused: task is outside the caller's descendant lineage]"),
            ok=False,
        )

    has_prompt = any(
        isinstance(args.get(key), str) and args.get(key).strip() for key in ("prompt", "task", "message", "input")
    )
    if matches:
        # There should be one job per child, but choose the newest record if a
        # legacy duplicate already exists so polling remains deterministic.
        job = sorted(
            matches,
            key=lambda item: float(item.get("created_at") or 0),
        )[-1]
        status = str(job.get("status") or "unknown")
        jid = str(job.get("job_id") or task_id)
        child_sid = str(job.get("child_sid") or task_id)
        if status not in async_jobs.TERMINAL:
            from sharedagents_core.injection_guard import wrap_untrusted

            task_text = wrap_untrusted(
                str(job.get("task") or ""),
                source="agent:background-status",
            )
            suffix = (
                ' Use Agent(action="steer", task_id=..., prompt=...) '
                "to change a live child; do not redispatch or poll it."
            )
            if has_prompt:
                suffix = (
                    " The supplied prompt was not dispatched because the "
                    'child is live; use action="steer" for guidance.'
                )
            return ToolResult(
                content=(f"[Agent task status: {status}; job_id={jid}; task_id={child_sid}; task={task_text}]{suffix}"),
                ok=True,
            )
        if not has_prompt:
            return ToolResult(
                content=(
                    f"[Agent task status: {status}; job_id={jid}; "
                    f"task_id={child_sid}. Supply a prompt to continue a "
                    "terminal child, or omit task_id for a fresh task.]"
                ),
                ok=True,
            )
        # A terminal child with an explicit prompt follows the existing,
        # validated continuation path below.
        return None

    # A task_id without a prompt is never a fresh dispatch. Returning a visible
    # failure makes malformed polling safe even when the durable job record has
    # already been compacted or removed.
    if not has_prompt:
        return ToolResult(
            content=(
                f"[Agent task_id {task_id} not found; no child was created. "
                "Provide a prompt for a validated continuation or omit "
                "task_id for a fresh dispatch.]"
            ),
            ok=False,
        )
    return None


async def _deliver_explicit_cancellations(
    args: dict[str, Any],
    job_ids: list[str],
) -> None:
    """Close cancelled child turns, then preserve each delivery behavior."""
    if not job_ids:
        return
    from sharedagents_agent.agents import async_jobs

    _finalize_cancelled_job_sessions(job_ids)
    # The cascade also cancels jobs spawned BY the aborted children. Their
    # parent session was itself aborted here: delivering to it would revive it.
    records = {jid: async_jobs.read(jid) or {} for jid in job_ids}
    aborted_sessions = {str(r.get("child_sid") or "") for r in records.values()} - {""}
    orphaned = [jid for jid, r in records.items() if r.get("parent_sid") in aborted_sessions]
    if orphaned:
        async_jobs.silence_deliveries(orphaned, reason="cancelled (parent abort)")
    from sharedagents_agent.agents.async_delegate_runner import deliver_terminal_job

    await asyncio.gather(*(deliver_terminal_job(job_id) for job_id in job_ids if job_id not in orphaned))


def _finalize_cancelled_job_sessions(job_ids: list[str]) -> None:
    """Append the terminal event a killed detached runner cannot emit itself.

    FIX (2026-08-11 'waiting' wedge): a mid-tool-call child whose
    ``live_turn_idx`` is NULL was skipped (``if turn_idx is None: continue``)
    and stayed ghost-live forever. The turn event now falls back to
    ``last_turn_idx`` (the in-flight turn that never closed, mirroring
    ``reset_stale_sessions``) and ``mark_session_terminal`` guarantees the
    row leaves the live set even when no event is possible.
    """
    from sharedagents_agent.agents import async_jobs
    from sharedagents_core.layout import sessions_root
    from sharedagents_session.session_index import (
        connect_sessions_db,
        mark_session_terminal,
    )
    from sharedagents_session.session_reader import turn_engine
    from sharedagents_session.session_writer import (
        close_session_writer,
        get_session_writer,
    )

    sessions_dir = sessions_root()
    db = sessions_dir / "sessions.db"
    if not db.exists():
        return

    child_sids = {str((async_jobs.read(job_id) or {}).get("child_sid") or "") for job_id in job_ids}
    child_sids.discard("")
    from sharedagents_session.session_activity import (
        ABORT_LIVE_TURN_STATES,
        TREE_BUSY_STATES,
    )

    live_states = set(ABORT_LIVE_TURN_STATES)
    busy_states = live_states | set(TREE_BUSY_STATES)
    for sid in child_sids:
        try:
            con = connect_sessions_db(db)
            try:
                row = con.execute(
                    "SELECT session_state, live_turn_idx, last_turn_idx, engine "
                    "FROM sessions_meta WHERE session_hash=?",
                    (sid,),
                ).fetchone()
            finally:
                con.close()
            # A finished (idle/error/closed) child is left as it is; a busy
            # one — including ``waiting`` on its own children — is folded.
            if row is None or row["session_state"] not in busy_states:
                continue
            turn_idx = row["live_turn_idx"]
            if turn_idx is None:
                turn_idx = row["last_turn_idx"]
            if turn_idx is not None and row["session_state"] in live_states:
                # The engine that ran the turn, not the row's header engine
                # (see ``turn_engine``); per event, so a cached writer is not
                # repointed.
                engine = turn_engine(sessions_dir, sid, int(turn_idx)) or str(row["engine"] or "")
                writer = get_session_writer(sessions_dir, sid)
                try:
                    writer.turn_ended(
                        turn_idx=int(turn_idx),
                        aborted=True,
                        metrics={"abort_reason": "explicit_agent_abort"},
                        engine=engine,
                    )
                finally:
                    close_session_writer(sessions_dir, sid)
            # Unconditional terminal backstop: the row must leave the live
            # set even when no turn event could be emitted.
            mark_session_terminal(sessions_dir, sid, state="error")
        except Exception as exc:  # noqa: BLE001 — abort still owns the stop
            log_event(
                "AGENT_ABORT_SESSION_FINALIZE_FAILED",
                level="ERROR",
                session_id=sid,
                error=repr(exc),
            )
            continue


#: Delegation targets that are routing hints, not agents with a config of
#: their own (``general-purpose`` = a custom sibling named via ``name``).
_ROSTER_PSEUDO_TARGETS = frozenset({"general-purpose"})


def _row_label(row: dict[str, Any]) -> str:
    for key in ("display", "title", "label", "id", "alias"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _roster_models(
    names: list[str],
    providers: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """``({agent: default model row}, configured agents)`` from saved config only.

    The picker summary names each agent's default model from its configuration
    and saved catalogues without asking any provider. An agent with no summary
    falls back to the routing digest's configured model; one with neither is
    not configured on this installation.
    """
    summaries: dict[str, dict[str, Any]] = {}
    try:
        from sharedagents_agent.discovery import picker

        for summary in picker.picker_summary([n for n in names if n not in _ROSTER_PSEUDO_TARGETS]):
            summaries[str(summary.get("name") or "")] = summary
    except Exception:  # noqa: BLE001 — the roster degrades to names + roles
        pass
    models: dict[str, dict[str, Any]] = {}
    configured: set[str] = set()
    for name in names:
        summary = summaries.get(name)
        if summary is not None:
            configured.add(name)
        row = dict((summary or {}).get("default") or {})
        if not row.get("id"):
            pinned = str((providers.get(name) or {}).get("configured_model") or "").strip()
            row = {"id": pinned, "display": pinned} if pinned else {}
        if name in providers:
            configured.add(name)
        if row.get("id"):
            models[name] = row
    return models, configured


# The routing digest's dispatch verdicts (``subscription_limits._DISPATCHABLE_STATES``).
_DISPATCHABLE_PROVIDER_STATES = ("spendable", "balance")


def _provider_status(prov: dict[str, Any]) -> str:
    """Status text for one routing-digest row.

    ``score`` is never shown as pace: it is the share of allowance LEFT (0 for
    an offline engine), so ``pace=0.0%`` made an engine that cannot run look
    like the least-used subscription to a lowest-pace router.
    """
    blocker = prov.get("dispatch_blocker")
    if blocker:
        return f"blocked: {blocker}"
    state = prov.get("state")
    if state and state not in _DISPATCHABLE_PROVIDER_STATES:
        return f"not dispatchable: {state}"
    pace = prov.get("ranking_pace")
    if type(pace) in (int, float) and math.isfinite(pace):
        return f"pace={pace:.1f}%"
    score = prov.get("score")
    if prov.get("score_unit") == "percent" and type(score) in (int, float) and math.isfinite(score):
        return f"pace unknown, {score:.0f}% left"
    usd = prov.get("remaining_usd")
    if prov.get("score_unit") == "usd" and type(usd) in (int, float) and math.isfinite(usd):
        return f"pace unknown, balance ${usd:.2f}"
    return "pace unknown"


def _delegation_options_summary() -> str:
    """One line per delegation target: model, public score, live status, role.

    Everything is read at runtime from this installation, never written here:
    the targets from ``agent_profiles.delegation_roster()`` (built-ins plus
    custom on-disk agents) and any extra agent the routing digest reports;
    each agent's default model from its saved configuration; pace and
    blockers from the routing digest; and the intelligence index from the
    public benchmark catalogue already on disk. Unknown values are omitted.
    """
    lines = ["\nAvailable delegation options:"]
    providers: dict[str, dict[str, Any]] = {}
    try:
        from sharedagents_web import subscription_limits

        digest = subscription_limits.routing_digest()
        providers = {p.get("agent"): p for p in digest.get("providers", []) if p.get("agent")}
    except Exception:
        pass

    try:
        from sharedagents_agent.agent_profiles import delegation_roster

        roster = [(str(n), str(d or "")) for n, d in delegation_roster()]
    except Exception:  # noqa: BLE001
        roster = []
    listed = {name for name, _ in roster}
    roster += [(agent, "") for agent in sorted(providers) if agent not in listed]

    models, configured = _roster_models([name for name, _ in roster], providers)
    scores: dict[str, float] = {}
    try:
        from sharedagents_web.model_benchmarks import intelligence_by_model

        scores = intelligence_by_model(models.values())
    except Exception:  # noqa: BLE001 — a score is optional display data
        pass

    for agent, role in roster:
        row = models.get(agent) or {}
        label = _row_label(row)
        parts = [f"- {agent} ({label})" if label else f"- {agent}"]
        score = scores.get(str(row.get("id") or ""))
        if isinstance(score, (int, float)):
            parts.append(f"Intel: {score:.1f}")
        prov = providers.get(agent)
        if prov:
            parts.append(f"Status: {_provider_status(prov)}")
        elif agent not in configured and agent not in _ROSTER_PSEUDO_TARGETS:
            parts.append("Status: not configured")
        if role:
            parts.append(role)
        lines.append(" | ".join(parts))
    return "\n".join(lines)


async def run_check_delegations_async(
    args: dict[str, Any],
    *,
    on_delta=None,
    on_event=None,
) -> ToolResult:
    """List background (async) delegations and their status.

        When ``SHAREDAGENTS_SESSION_ID`` is set, only jobs in the caller's lineage
        (self + descendants) are shown — the model-facing Agent list surface
    . An unscoped operator context still sees every job.
    """
    scope_error = _native_control_scope_error(args, "list")
    if scope_error is not None:
        return scope_error
    # Derived status surface over the durable async-job registry.
    from sharedagents_agent.agents import async_jobs

    # Reclaim session budget held for any background job that has since finished
    # (spec §6 — the detached runner finalizes in its own process).
    from sharedagents_agent.agents.delegate_shims import _reconcile_bg_reservations

    _reconcile_bg_reservations()
    jobs = async_jobs.list_jobs()
    own = str(os.environ.get("SHAREDAGENTS_SESSION_ID") or "")
    if own:
        lineage = _lineage_set(own)
        jobs = [j for j in jobs if str(j.get("parent_sid") or "") in lineage]
    from sharedagents_core.injection_guard import wrap_untrusted

    lines = []
    if not jobs:
        lines.append("No background delegations.")
    else:
        for j in jobs:
            tid = str(j.get("child_sid") or j.get("task_id") or "")
            tid_bit = f" task_id={tid}" if tid else ""
            if j.get("wave_id"):
                members = async_jobs.wave_jobs(j)
                done = sum(m.get("status") in async_jobs.TERMINAL for m in members)
                tid_bit += f" wave_id={j['wave_id']} wave={done}/{len(members)} terminal"
            task_text = wrap_untrusted(
                str(j.get("task", ""))[:240],
                source="agent:background-list",
            )
            lines.append(f"{j['job_id']} [{j.get('status')}] {j.get('agent')}{tid_bit}: {task_text}")
    lines.append(_delegation_options_summary())
    return ToolResult(content="\n".join(lines), ok=True)


async def run_cancel_delegation_async(
    args: dict[str, Any],
    *,
    on_delta=None,
    on_event=None,
) -> ToolResult:
    """Cancel a background delegation by ``job_id`` (kills that job + the
    subtree it spawned), by public ``task_id`` (child session), or every job
    under a ``session_id`` subtree.

    Native Agent cancellation only walks DOWN the caller's descendant session
    tree: when the caller has a session id
    (``SHAREDAGENTS_SESSION_ID``), a target is honored only if it is a descendant;
    self, ancestor, and sibling aborts are refused. Legacy operator/MCP calls
    remain allowed to target self and, when unscoped, any session.
    """
    scope_error = _native_control_scope_error(args, "abort")
    if scope_error is not None:
        return scope_error
    from sharedagents_agent.agents import async_jobs
    from sharedagents_agent.agents.delegate_shims import _reconcile_bg_reservations

    _reconcile_bg_reservations()
    wave_id = str(args.get("wave_id") or "")
    if wave_id:
        own = str(os.environ.get("SHAREDAGENTS_SESSION_ID") or "")
        lineage = _lineage_set(own) if own else None
        members = [j for j in async_jobs.list_jobs() if j.get("wave_id") == wave_id]
        if not members or any(lineage is not None and j.get("parent_sid") not in lineage for j in members):
            return ToolResult(content="[Agent abort: wave not found in your lineage]", ok=False)
        # Fence any dispatch still being registered before cancelling members.
        # create() shares the parent lock and refuses registration after seal;
        # closing (seal + receipt-key pin) also freezes membership so a late
        # same-goal append cannot outlive the abort.
        member_ids = async_jobs.close_wave(str(members[0]["parent_sid"]), wave_id)
        if member_ids:
            members = [r for r in map(async_jobs.read, member_ids) if r] or members
        cancelled = []
        for member in members:
            cancelled.extend(async_jobs.cancel_job_ids(member["job_id"]))
        await _deliver_explicit_cancellations(args, list(dict.fromkeys(cancelled)))
        # Already-terminal members may own a still-undelivered wave.
        from sharedagents_agent.agents.async_delegate_runner import deliver_terminal_job

        await deliver_terminal_job(members[0]["job_id"])
        return ToolResult(content=f"cancelled wave {wave_id}: {len(cancelled)} job(s)", ok=True)
    jid = str(args.get("job_id") or "")
    tid = str(args.get("task_id") or "")
    sid = str(args.get("session_id") or args.get("sid") or "")
    # Models sometimes pass the job handle as task_id — accept job_* there.
    if not jid and tid.startswith("job_"):
        jid, tid = tid, ""
    # Public task_id (child session) → cancel that subtree. Prefer explicit
    # job_id when both are supplied.
    if not jid and tid and not sid:
        sid = tid
    if jid:
        # The lineage guard must bind the
        # job_id path too, not just session_id. A scoped worker (has
        # SHAREDAGENTS_SESSION_ID) may cancel a job only if the job's owning session
        # (parent_sid) is its own or a descendant — never an ancestor's or a
        # sibling's job. An unscoped operator context is unrestricted, as below.
        own = str(os.environ.get("SHAREDAGENTS_SESSION_ID") or "")
        if own:
            rec = async_jobs.read(jid)
            job_sid = str((rec or {}).get("parent_sid") or "")
            lineage = _lineage_set(own)
            if rec is not None and job_sid not in lineage:
                return ToolResult(
                    content=(
                        f"[cancel refused: job {jid} (session "
                        f"{job_sid or 'unknown'}) is not "
                        f"in your lineage (self+descendants of {own}); a worker "
                        f"cannot cancel an ancestor or sibling job]"
                    ),
                    ok=False,
                )
        existed = async_jobs.read(jid) is not None
        cancelled = async_jobs.cancel_job_ids(jid) if existed else []
        await _deliver_explicit_cancellations(args, cancelled)
        return ToolResult(
            content=(
                f"cancelled {len(cancelled)} job(s): {', '.join(cancelled)}" if existed else f"[no such job: {jid}]"
            ),
            ok=existed,
        )
    if sid:
        own = str(os.environ.get("SHAREDAGENTS_SESSION_ID") or "")
        if args.get("_sharedagents_agent_tool") and sid == own:
            return ToolResult(
                content=(
                    "[Agent abort refused: native Agent cannot abort its own "
                    "session; pass a descendant task_id or job_id]"
                ),
                ok=False,
            )
        if own and sid != own and sid not in async_jobs._descendant_sessions(own):
            return ToolResult(
                content=(
                    f"[cancel refused: session {sid} is not in your lineage "
                    f"(self+descendants of {own}); a worker cannot cancel an "
                    f"ancestor or sibling session]"
                ),
                ok=False,
            )
        cancelled = async_jobs.cancel_subtree(sid)
        await _deliver_explicit_cancellations(args, cancelled)
        return ToolResult(
            content=f"cancelled {len(cancelled)} job(s) under session {sid}",
            ok=True,
        )
    return ToolResult(
        content="[Agent abort: pass task_id, job_id, or session_id]",
        ok=False,
    )


def _append_steer_queue_msg(sessions_dir: Path, sid: str, text: str) -> str:
    """Append ``queue_msg_added`` (source=steer) to an existing session JSONL.

    Reuses the durable follow-up queue already folded by ``fold_queue_state``
    — no separate inbox file.
    """
    import uuid

    from sharedagents_session.session_writer import (
        get_session_writer,
        release_session_writer,
    )

    qid = uuid.uuid4().hex[:12]
    payload = {
        "queue_id": qid,
        "text": text,
        "source": "steer",
        "author": "Agent",
    }
    # The steer carries the calling session's grants, so a restricted turn
    # never injects guidance into a wider-granted child.
    import os as _os

    from sharedagents_session.access_binding import session_access_stamp

    _caller = _os.environ.get("SHAREDAGENTS_SESSION_ID", "")
    _stamp = session_access_stamp(sessions_dir, _caller) if _caller else None
    if _stamp is not None and _stamp.restricted:
        payload["access"] = _stamp.to_obj()
    # The steering agent authors the injected message, like the delegated
    # prompt it follows up on.
    from sharedagents_agent.agents.delegate_shims import _delegating_agent_name
    from sharedagents_session.actors import agent_actor

    _who = agent_actor(_delegating_agent_name(sessions_dir, _caller))
    if _who is not None:
        payload["actor"] = _who
    w = get_session_writer(sessions_dir, sid, pin=True)
    try:
        w.lifecycle(kind="queue_msg_added", payload=payload)
    finally:
        release_session_writer(w)
    return qid


async def run_steer_delegation_async(
    args: dict[str, Any],
    *,
    on_delta=None,
    on_event=None,
) -> ToolResult:
    """Inject mid-flight guidance into a live descendant subagent.

    Writes a ``queue_msg_added`` (source=steer) into the child's existing
    session JSONL — the same follow-up queue frontends already use. The live
    child re-folds that log at its next safe round boundary. This deliberately
    does not signal the durable PID sidecar: for local children that sidecar
    can identify the whole tool runner, so signalling it would turn steer into
    an abort.
    """
    scope_error = _native_control_scope_error(args, "steer")
    if scope_error is not None:
        return scope_error
    from sharedagents_agent.agents import async_jobs
    from sharedagents_core.kernel.layout import sessions_root

    tid = str(args.get("task_id") or args.get("session_id") or "").strip()
    text = str(args.get("prompt") or args.get("text") or args.get("message") or "").strip()
    if not tid:
        return ToolResult(
            content="[Agent steer: pass task_id of the child to steer]",
            ok=False,
        )
    if not text:
        return ToolResult(
            content="[Agent steer: pass prompt with the steering guidance]",
            ok=False,
        )
    own = str(os.environ.get("SHAREDAGENTS_SESSION_ID") or "")
    lineage = _lineage_set(own) if own else None
    jobs = [
        j
        for j in async_jobs.open_jobs()
        if str(j.get("child_sid") or "") == tid and (lineage is None or str(j.get("parent_sid") or "") in lineage)
    ]
    if not jobs:
        # Distinguish a terminal child from an unknown/out-of-lineage child.
        # ``open_jobs`` intentionally omits terminal records, which made every
        # late steer look like a lineage bug and encouraged small models to
        # redispatch the work. Keep the detailed status scoped to the caller.
        terminal_matches = []
        outside_match = False
        try:
            all_jobs = async_jobs.list_jobs()
        except Exception:  # noqa: BLE001
            all_jobs = []
        for job in all_jobs:
            if str(job.get("child_sid") or "") != tid:
                continue
            if lineage is not None and str(job.get("parent_sid") or "") not in lineage:
                outside_match = True
                continue
            if str(job.get("status") or "") in async_jobs.TERMINAL:
                terminal_matches.append(job)
        if outside_match and not terminal_matches:
            return ToolResult(
                content=(f"[steer refused: task_id {tid} is outside the caller's descendant lineage]"),
                ok=False,
            )
        if terminal_matches:
            status = str(terminal_matches[-1].get("status") or "terminal")
            return ToolResult(
                content=(
                    f"[steer refused: task_id {tid} is terminal "
                    f"(status={status}); continue it with Agent(task_id=..., "
                    "prompt=...) or start a fresh task]"
                ),
                ok=False,
            )
        scope = f" in your lineage (self+descendants of {own})" if own else ""
        return ToolResult(
            content=f"[steer refused: task_id {tid} is not a live subagent{scope}]",
            ok=False,
        )
    root = sessions_root()
    if not (root / f"{tid}.jsonl").is_file():
        return ToolResult(
            content=f"[Agent steer: session log not found for task_id {tid}]",
            ok=False,
        )
    try:
        qid = _append_steer_queue_msg(root, tid, text)
    except Exception as exc:  # noqa: BLE001
        return ToolResult(
            content=f"[Agent steer: failed to write session queue: {exc}]",
            ok=False,
        )
    return ToolResult(
        content=(f"steered task_id={tid} queue_id={qid} delivery=next_safe_round"),
        ok=True,
    )


__all__ = [
    "_shim_claude",
    "_shim_codex",
    "_shim_antigravity",
    "_shim_zcode",
    "_shim_agent",
    "_shim_agent_canonical",
    "_shim_agent_sync",
    "_shim_agent_async",
    "_spawn_subagent_runner",
    "_spawn_subagent_runner_body",
    "_batch_entry_attribution",
    "_shim_batch_agent",
    "run_check_delegations_async",
    "run_cancel_delegation_async",
    "run_steer_delegation_async",
]
