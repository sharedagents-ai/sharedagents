"""Tool dispatch + parallel-batch eligibility for :class:`CoreBackend`.

Extracted from ``acp_bridge.py`` (behavior-preserving de-monolith). This
module owns the live tool-execution path the engine calls into:

* ``_default_run_tool`` — spawns ``sharedagents.tool_runner`` for one tool
  call in its own OS process (foundational killability), parses its
  JSONL stdout, and returns ``(ok, content, aborted, truncated)``. The
  ``question`` / ``budget`` / ``goal`` tools short-circuit before the
  subprocess.
* the parallel-dispatch gates (``_call_is_parallel_eligible`` +
  ``_state_file_key`` → ``_build_parallel_groups``);
* subagent attribution injection (``_inject_subagent_attribution``);
* the runner-crash describer (``_describe_runner_exit``).

Re-exported from ``sharedagents_gateway.acp_bridge`` for import stability —
``sharedagents_agent.agents.local_llm`` imports ``_default_run_tool`` /
``_build_parallel_groups`` / ``_call_is_parallel_eligible`` /
``_state_file_key`` LAZILY from that path on the live turn loop.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sharedagents_agent.tools import (
    Criticality,
    PermissionEvaluator,
    Tool,
    ToolKind,
)
from sharedagents_agent.turn_runtime import TurnState
from sharedagents_core import oscompat
from sharedagents_core.attachments import is_image_url_part

from .ask_user import _ask_user_outbound

if TYPE_CHECKING:
    from sharedagents_gateway.acp_bridge import CoreBackend

# Loop/cron tools run on the LIVE in-process agent (the loop registry and
# supervisor are in-process singletons, so a subprocess runner would act on a
# different registry). Short-circuited in ``_default_run_tool`` exactly like
# ``goal``/``budget``; ``CoreBackend`` dispatches to the shared
# ``sharedagents_loop.loop_tools`` runners.
_LOOP_TOOL_NAMES = frozenset({"cron_create", "cron_delete", "cron_list", "schedule_wakeup"})


# --- helpers -------------------------------------------------------------


def _describe_runner_exit(returncode: int | None, stderr_text: str = "") -> str:
    """Render a debuggable reason a ``tool_runner`` subprocess died without
    emitting a ``final`` result line.

    The ACP bridge previously surfaced only ``rc=<n>`` plus the *last* line
    of stderr, which made runner crashes effectively undebuggable:

      * a signal death showed up as a bare negative number (e.g. ``rc=-11``)
        with no name — the reader had to know ``-11 == SIGSEGV``;
      * a multi-line Python traceback was collapsed to its final line, so
        the failing frame / chained context was lost;
      * an exit code with no stderr (observed: ``rc=255``, ``<no stderr>``)
        gave the operator — and this self-improver — nothing to go on.

    This renders, in a single bounded string:

      * signal deaths as ``killed by signal N (SIGNAME)``;
      * normal exits as ``exited rc=N``;
      * ``no exit status`` when the process was never reaped;
      * a bounded multi-line *tail* of stderr (not just the last line),
        or an explicit ``<no stderr captured>`` marker.

    Pure / side-effect free so it is unit-testable in isolation.
    """
    if returncode is None:
        exit_desc = "no exit status (process not reaped)"
    elif returncode < 0:
        signum = -returncode
        try:
            name = signal.Signals(signum).name
        except (ValueError, KeyError):
            name = "?"
        exit_desc = f"killed by signal {signum} ({name})"
    else:
        exit_desc = f"exited rc={returncode}"

    tail = ""
    if stderr_text:
        lines = [ln for ln in stderr_text.strip().splitlines() if ln.strip()]
        if lines:
            tail = "\n".join(lines[-5:])
            if len(tail) > 600:
                tail = tail[-600:]
    stderr_desc = tail if tail else "<no stderr captured>"
    return f"{exit_desc} — stderr: {stderr_desc}"


def _state_file_key(call: Mapping[str, Any]) -> str | None:
    """Per-batch collision key for delegate-style tools. ``None`` means
    "no shared state file" (the call doesn't conflict with anything).
    """
    name = call.get("name")
    if name in ("delegate_claude", "delegate_codex", "delegate_antigravity"):
        # Single state file per session — N concurrent calls would race
        # on ``<sid>.delegate_<kind>.json``.
        return str(name)
    if name in ("delegate_agent", "acp_call"):
        target = (call.get("arguments") or {}).get("name") or ""
        target = target.strip() if isinstance(target, str) else ""
        return f"{name}:{target}" if target else str(name)
    return None


def _call_is_parallel_eligible(
    tool: Tool,
    call: Mapping[str, Any],
    evaluator: PermissionEvaluator,
) -> bool:
    """Eligibility for concurrent dispatch in a multi-call round."""
    if call.get("name") in _PARALLEL_UNSAFE_TOOLS:
        return False
    args = dict(call.get("arguments") or {})
    crit = evaluator._effective_criticality(tool, args)
    if crit == Criticality.READ_ONLY:
        return True
    # NETWORK tools (delegate_*, acp_call, web_fetch, …) are I/O-bound
    # and don't share local state. Per-batch state-file uniqueness is
    # checked separately in ``_build_parallel_groups``.
    return tool.kind == ToolKind.NETWORK


def _build_parallel_groups(
    entries: Sequence[Mapping[str, Any]],
) -> list[list[int]]:
    """Walk ``entries`` and return groups of consecutive indices safe
    to dispatch together.

    Each entry must carry ``parallel_safe: bool`` and ``state_key: str|None``.
    A group is either a singleton (any non-eligible call) or a maximal
    run of consecutive eligible calls whose ``state_key``s are pairwise
    distinct within the run. Order is preserved across groups so the
    OpenAI-shape tool message list still tracks the assistant's
    ``tool_calls`` order.
    """
    groups: list[list[int]] = []
    i = 0
    n = len(entries)
    while i < n:
        if not entries[i].get("parallel_safe", False):
            groups.append([i])
            i += 1
            continue
        batch = [i]
        keys: set[str] = set()
        sk = entries[i].get("state_key")
        if sk is not None:
            keys.add(sk)
        j = i + 1
        while j < n:
            nxt = entries[j]
            if not nxt.get("parallel_safe", False):
                break
            nk = nxt.get("state_key")
            if nk is not None and nk in keys:
                break
            batch.append(j)
            if nk is not None:
                keys.add(nk)
            j += 1
        groups.append(batch)
        i = j
    return groups


# --- parallel-dispatch eligibility --------------------------------------
#
# The dispatch loop runs read-only / state-file-distinct calls
# concurrently (one ``asyncio.create_subprocess_exec`` runner per call,
# multiplexed through one shared event queue). Two gates decide whether
# a call can join a parallel batch:
#
#   1. ``_call_is_parallel_eligible`` — semantic safety. Read-only
#      criticality (model claim AND heuristic agree, per the
#      evaluator's ``_effective_criticality``) means siblings can't
#      step on each other's state. ``NETWORK``-kind tools (delegates,
#      ``acp_call``, ``web_fetch``, …) are I/O-bound and isolate state
#      via per-call session files — also eligible. Bridge-state
#      mutators (``compact_context``, ``tool_search``) are permanently
#      excluded: they mutate the running ``messages`` list / promoted-
#      tool set and would race against any sibling that read either.
#
#   2. ``_state_file_key`` — sequential per-batch uniqueness for
#      delegate-style tools that share a persisted ``<sid>.delegate_*.json``.
#      Two ``delegate_claude`` calls in one round both read+write the
#      same state file, so they collide; the batcher splits them into
#      separate groups (sequentially executed). Different ``name=``
#      arguments to ``delegate_agent`` / ``acp_call`` get distinct keys
#      because their state files are namespaced by target name.

_PARALLEL_UNSAFE_TOOLS: frozenset = frozenset({"compact_context"})


# --- tool runner subprocess ---------------------------------------------


# Tool names that count as subagent / delegate dispatches. Calls of
# these names get the bridge-injected ``_sharedagents_call_id`` +
# ``_sharedagents_parent_session_id`` so ``delegate_shims`` / the child
# sharedagents session can link back to the parent.
_SUBAGENT_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "AgentSync",
        "AgentAsync",
        "BatchAgent",
        "delegate_claude",
        "delegate_codex",
        "delegate_antigravity",
        "delegate_agent",
    }
)


def _inject_subagent_attribution(
    call_args: dict[str, Any],
    call: Mapping[str, Any],
    session_id: str,
    sessions_dir: Any = None,
) -> None:
    """Overwrite delegate attribution fields with the bridge's values.

    A model-supplied ``_sharedagents_parent_session_id`` /
    ``_sharedagents_call_id`` in the tool arguments would otherwise win
    against the gateway's own — letting the prompt spoof another
    session as the delegate parent so the child's
    ``session_started`` event links to a sid the model chose.
    Overwriting (not ``setdefault``) makes the bridge the only
    authority for these fields.

    No-op for non-subagent tool calls so other tool args stay
    byte-stable.
    """
    if call.get("name") not in _SUBAGENT_TOOL_NAMES:
        return
    _cid = str(call.get("id") or "")
    if _cid:
        call_args["_sharedagents_call_id"] = _cid
    if session_id:
        call_args["_sharedagents_parent_session_id"] = session_id
        # Thread the TRUE cost root + ancestor goal from the parent session read
        # in ITS OWN (correct) dir — ``sessions_dir`` is the agent this bridge
        # serves, where ``session_id`` lives. A cross-agent child (main→claude)
        # can't resolve parent_sid in its OWN db and would self-root to depth-1.
        # Overwrite (authority rule) like parent_sid.
        if sessions_dir is not None:
            try:
                from sharedagents_agent.agents.delegate_shims import (
                    _parent_cost_attribution,
                )
                from sharedagents_goal.goal_service import charged_goal_id

                p_root, p_goal, p_own = _parent_cost_attribution(
                    sessions_dir,
                    session_id,
                )
                call_args["_sharedagents_cost_root_session_id"] = p_root or session_id
                call_args["_sharedagents_cost_ancestor_goal_id"] = charged_goal_id(
                    session_hash=session_id,
                    cost_root_session_id=p_root,
                    cost_ancestor_goal_id=p_goal,
                    goal_id=p_own,
                )
            except Exception:  # noqa: BLE001
                pass


def _session_workspace_cwd(backend: Any, session_id: str) -> str:
    """The folder the session works in, or ``""`` when it has none.

    The bridge keeps live sessions in ``_active_sessions``. A delegated agent
    runs outside the bridge, behind a backend that only exposes
    ``_session_for``; without that fallback its tool runner never learns the
    folder it was dispatched to and resolves the agent's write roots against
    the service's own working directory.
    """
    for lookup in (
        lambda: backend._active_sessions.get(session_id),
        lambda: backend._session_for(session_id),
    ):
        try:
            sess = lookup()
        except Exception:  # noqa: BLE001
            continue
        if sess is not None:
            return str(getattr(sess, "workspace_cwd", "") or "")
    return ""


async def _default_run_tool(
    backend: CoreBackend,
    session_id: str,
    call: Mapping[str, Any],
    turn: TurnState,
    *,
    on_delta: Callable[[str], Awaitable[None]] | None = None,
    on_event: Callable[[dict], Awaitable[None]] | None = None,
) -> (
    tuple[bool, str, bool, bool]
    | tuple[bool, str, bool, bool, bool, str | None, tuple]
    | tuple[bool, str, bool, bool, bool, str | None, tuple, tuple]
):
    """Spawn ``sharedagents.tool_runner`` for one tool call.

    Each call lives in its own OS process (foundational killability):
    ``turn.proc`` is updated to the runner so the existing
    :meth:`CoreBackend.cancel` SIGTERMs whichever subprocess is active
    AT THIS MOMENT — runner OR worker. The runner's stdout is parsed
    for ``{"type": "delta", "text": ...}`` lines (forwarded via
    ``on_delta`` when provided — Claude's in-band ``🔧 tool_use`` /
    ``↳ tool_result`` markers ride this channel) and a final
    ``{"type": "final", "ok": ..., "content": ..., "truncated": ...}``
    record. Returns ``(ok, content, aborted, truncated)``; when display metadata
    is present, appends ``(background_handle, outcome, member_outcomes)``, and
    when the tool returned images, also ``images`` (``image_url`` parts).

    ``on_event``, when wired, receives one ``{"kind": ...}`` dict per
    ``{"type": "tool_event", ...}`` runner-stdout record — today
    these come from ``delegate_claude`` surfacing its inner
    ``tool_use`` / ``tool_result`` blocks so the bridge can fan them
    out as additional ACP ``tool_call`` / ``tool_call_update``
    notifications and the TUI mounts native widgets for them.

    ``AskUserQuestion`` short-circuits before subprocess spawn: there's
    no runner implementation for it (the call has to round-trip
    through the connected frontend), so we hand it to
    :func:`_ask_user_outbound` and return the wire result directly.

    After a force-terminate, a subsequently-spawned Bash call has its
    idle and wall timeouts clamped downward to a bounded short ceiling.
    """
    if call.get("name") == "question":
        ok, content = await _ask_user_outbound(backend, session_id, call)
        return ok, content, False, False

    # goal_budget runs on the LIVE in-process agent (resume needs the live
    # supervisor), so it short-circuits the subprocess runner like AskUserQuestion.
    if call.get("name") == "budget":
        a = dict(call.get("arguments") or {})
        try:
            usd = float(a.get("usd") or 0)
        except (TypeError, ValueError):
            return False, "budget: usd must be a number", False, False
        out = backend.goal_budget(
            session_id,
            str(a.get("action") or ""),
            usd,
            str(a.get("goal_id") or ""),
        )
        return bool(out.get("ok")), str(out.get("message") or ""), False, False

    # The autonomous-goal tool runs on the LIVE agent too (create/status
    # updates reach the live supervisor via execute_goal_command).
    # TIER 2a — forward the model-emitted structured goal_id; previously it
    # was dropped here, so ``goal archive`` could never name its target and
    # the model's off-ramp silently no-op'd into a usage error.
    if call.get("name") == "goal":
        a = dict(call.get("arguments") or {})
        out = backend.goal(
            session_id,
            str(a.get("action") or "set"),
            str(a.get("text") or a.get("condition") or ""),
            str(a.get("goal_id") or ""),
            args=a,
        )
        return bool(out.get("ok")), str(out.get("message") or ""), False, False

    # The loop/cron tools run on the LIVE agent too — the
    # registry and supervisor are in-process singletons, so a subprocess
    # runner would schedule against a different registry.
    if call.get("name") in _LOOP_TOOL_NAMES:
        out = getattr(backend, str(call["name"]))(session_id, dict(call.get("arguments") or {}))
        return bool(out.get("ok")), str(out.get("message") or ""), False, False

    package_parent = str(Path(__file__).resolve().parent.parent)
    env = dict(os.environ)
    # The agent's own secrets (shared + per-agent ``secrets.env``, the same
    # ``api_keys`` its external CLI gets as spawn env) — per-agent values win
    # over shared ones. A restricted turn's sandbox drops them again
    # (``folder_sandbox.narrow_env``).
    env.update({str(k): str(v) for k, v in (getattr(backend.cfg, "api_keys", None) or {}).items()})
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = package_parent + (os.pathsep + existing if existing else "")
    env["PYTHONUNBUFFERED"] = "1"
    env["SHAREDAGENTS_SESSION_ID"] = session_id
    # ACP-supplied MCP servers live in the parent's in-memory registry;
    # the runner runs in THIS fresh subprocess and can't see it. Serialize
    # this session's ACP servers into env so the runner can overlay them
    # onto its config-loaded servers when resolving an mcp__<server>__<tool>
    # call. Per-subprocess env keeps it correctly session-scoped.
    try:
        from sharedagents_mcp import mcp_client as _mcp

        _acp_srv = _mcp.acp_session_servers(session_id)
        if _acp_srv:
            env[_mcp.ACP_MCP_SERVERS_ENV] = _mcp.encode_acp_servers_for_env(_acp_srv)
    except Exception:  # noqa: BLE001
        pass
    env["SHAREDAGENTS_SESSIONS_DIR"] = str(backend.cfg.sessions_dir)
    # The running turn's pinned grants: the runner's gates
    # (Skill, acp_call, steer, delegation) use them, not the shared binding
    # file a later turn may have rewritten.
    from sharedagents_session.access_binding import apply_access_pin_env

    apply_access_pin_env(env, session_id)
    env["SHAREDAGENTS_AGENT_HOME"] = str(backend.cfg.home)
    env["SHAREDAGENTS_AGENT_NAME"] = backend.cfg.name
    # The resolved agent PROFILE (falls back to the agent name). The tool_runner's
    # path-scoped write gate resolves the profile's write_roots
    # from this — the config name can differ from the profile, so stamp the
    # profile explicitly rather than relying on name==profile.
    # Prefer the EFFECTIVE applied profile (``_agent_profile_name``, set by
    # apply_profile_to_config) over the persisted ``agent_profile``: an
    # unscaffolded profile runs its ROLE on the ``general`` mount, so the persisted
    # ``agent_profile`` is still ``general`` (unrestricted writes) while the turn is
    # actually a ``plan``/etc. role — stamping the persisted value would let the
    # fallback escape the selected profile's write_roots.
    env["SHAREDAGENTS_AGENT_PROFILE"] = (
        str(getattr(backend.cfg, "_agent_profile_name", "") or "")
        or str(getattr(backend.cfg, "agent_profile", "") or "")
        or backend.cfg.name
    )
    # This gateway hosts multiple acting agents in one process.  Resolve the
    # experimental Agent default from THIS actor's home and overwrite any
    # process-global value before spawning its tool runner, otherwise Jarvis's
    # enabled .env would leak into Build/Plan turns.
    from sharedagents_agent.background_subagents import stamp_for_agent

    stamp_for_agent(env, Path(backend.cfg.home))
    env["SHAREDAGENTS_PERMISSION_MODE"] = backend.cfg.permission_mode or "auto"
    # Stamp the session's real workspace_origin for tool-runner context.
    _priv_origin = ""
    _ws_cwd = ""
    try:
        _priv_sess = backend._active_sessions.get(session_id)
        if _priv_sess is not None:
            _priv_origin = getattr(_priv_sess, "workspace_origin", "") or ""
    except Exception:  # noqa: BLE001
        _priv_origin = ""
    _ws_cwd = _session_workspace_cwd(backend, session_id)
    env["SHAREDAGENTS_WORKSPACE_ORIGIN"] = _priv_origin
    # The session's workspace root (the repo the agent operates in). The
    # path-scoped write gate resolves relative write_roots (e.g. "plan/") against
    # this. Empty when the session has no workspace record → the gate falls back
    # to the tool_runner's process cwd.
    if _ws_cwd:
        env["SHAREDAGENTS_WORKSPACE_CWD"] = _ws_cwd
    if backend.cfg.upstream_endpoint:
        env["SHAREDAGENTS_UPSTREAM_ENDPOINT"] = backend.cfg.upstream_endpoint
    if backend.cfg.upstream_model:
        env["SHAREDAGENTS_UPSTREAM_MODEL"] = backend.cfg.upstream_model

    # Fix B: post-force-terminate cap. If the bridge already fired
    # ``LOOP_FORCE_TERMINATE`` on this turn, clamp the EXEC tool's
    # idle + wall budget to 30s before handing args to the runner.
    # In normal operation the bridge's dispatch-level deny gate
    # (fix A) refuses these calls before we get here, so this branch
    # is dead code — but it's a cheap safety net for any future
    # refactor that loosens A or for any code path that bypasses it.
    # See acp_bridge.py: ``post_force_terminate`` near the prompt
    # loop's force-terminate block.
    call_args = dict(call.get("arguments") or {})
    if getattr(turn, "post_force_terminate", False) and call.get("name") == "bash":
        cap_s = 30.0
        # Translate Claude-style millisecond ``timeout`` (max 600000)
        # consistently with ``run_bash_async`` so a model-supplied
        # value gets compared on the same axis as our cap.
        raw_idle = call_args.get("timeout")
        if raw_idle is not None:
            try:
                v = float(raw_idle)
                if v > 3600:
                    v = v / 1000.0
                call_args["timeout"] = min(v, cap_s)
            except (TypeError, ValueError):
                call_args["timeout"] = cap_s
        else:
            call_args["timeout"] = cap_s
        raw_wall = call_args.get("wall_timeout")
        if raw_wall is not None:
            try:
                call_args["wall_timeout"] = min(float(raw_wall), cap_s)
            except (TypeError, ValueError):
                call_args["wall_timeout"] = cap_s
        else:
            call_args["wall_timeout"] = cap_s
    # A session per subagent: thread this tool call's id into the
    # subagent dispatch args so ``_drive_child_agent_turn`` keys the
    # child sharedagents session per CALL (one fresh session per subagent
    # invocation) instead of sharing one per (parent, engine). The
    # parent sid already rides the subprocess env (SHAREDAGENTS_SESSION_ID),
    # but stamp it explicitly too so the link survives any future
    # in-process dispatch path. Harmless for non-subagent tools but
    # scoped to the subagent family to keep other tool args byte-stable.
    _inject_subagent_attribution(
        call_args,
        call,
        session_id,
        backend.cfg.sessions_dir,
    )
    from sharedagents_agent.folder_sandbox import SandboxRefused, sandbox_argv, spawn_stamp

    stamp = spawn_stamp(session_id, str(backend.cfg.sessions_dir))
    if stamp is not None and stamp.restricted:
        env.pop("SHAREDAGENTS_ACP_MCP_SERVERS", None)
    request = {"tool": call["name"], "args": call_args}
    try:
        spawn_argv, spawn_env = sandbox_argv(
            [sys.executable, "-m", "sharedagents_agent.tool_runner"],
            env,
            stamp,
            tool=str(call.get("name") or ""),
        )
    except SandboxRefused as exc:
        return False, f"[access] {exc}", False, False
    proc = await asyncio.create_subprocess_exec(
        *spawn_argv,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=spawn_env,
        limit=16 * 1024 * 1024,
        # Own session / process group. A force-cancel SIGKILLs the whole
        # process group via ``os.killpg`` (TurnState.stop(force=True)); if
        # this subprocess shared the gateway's group, a hard ESC /
        # ``/abort`` would kill the GATEWAY itself (observed:
        # the gateway died and systemd restarted it). ``start_new_session``
        # makes it a session leader so killpg reaches only it and its
        # descendants (the bash/sleep/CLI it spawns).
        start_new_session=not oscompat.IS_WINDOWS,
        **oscompat.detach_kwargs(),
    )
    if proc.stdin is None:
        raise RuntimeError("tool_runner stdin pipe is None")
    proc.stdin.write(json.dumps(request).encode("utf-8"))
    await proc.stdin.drain()
    proc.stdin.close()
    # Re-point the live TurnState at the runner so ESC kills it
    # instantly (instead of trying to terminate the already-exited
    # worker proc). Also register on the runner_procs dict so parallel
    # batches with multiple in-flight runners all get SIGTERM'd by ESC,
    # not just whichever happened to land in turn.proc last.
    turn.proc = proc
    call_id = str(call.get("id") or "")
    if call_id:
        turn.register_runner(call_id, proc)

    final: dict | None = None
    try:
        if proc.stdout is None:
            raise RuntimeError("tool_runner stdout pipe is None")
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            try:
                obj = json.loads(line.decode("utf-8", "replace"))
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "delta":
                # Forward Claude/Codex's in-band streaming text +
                # tool-use markers if a callback was provided. Failures
                # in the callback don't kill the runner — log and move
                # on so a flaky frontend can't break the agent.
                if on_delta is not None:
                    text = obj.get("text") or ""
                    if text:
                        try:
                            await on_delta(text)
                        except Exception:  # noqa: BLE001
                            pass
                continue
            if obj.get("type") == "tool_event":
                # Structured inner-tool side-channel from streaming
                # wrappers (today: delegate_claude). The direct-
                # dispatch path uses this to fan out additional ACP
                # tool_call/tool_call_update notifications so the TUI
                # mounts native widgets for Claude's inner calls.
                if on_event is not None:
                    try:
                        await on_event({k: v for k, v in obj.items() if k != "type"})
                    except Exception:  # noqa: BLE001
                        pass
                continue
            # Only a ``final`` record (or the legacy untyped shape) is the
            # result; any other typed record is skipped, never taken for it.
            if final is None and obj.get("type") in (None, "final"):
                final = obj
    except asyncio.CancelledError:
        # SIGKILL the runner's ENTIRE descendant tree, not just the launcher.
        # A bare ``proc.kill()`` signals only this ``tool_runner`` process; any
        # subagent / shell it spawned in its OWN session (``start_new_session``
        # → a different pgid) is left orphaned to init and keeps running. This
        # is the in-process-cancel twin of ``TurnState.stop(force)`` and
        # ``_spawn_subagent_runner``'s cancel path — both already whole-tree.
        try:
            from sharedagents_agent.tools.runners import _kill_tree

            _kill_tree(proc, oscompat.FORCE_KILL)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        try:
            await proc.wait()
        except Exception:  # noqa: BLE001
            pass
        if call_id:
            turn.unregister_runner(call_id)
        raise

    try:
        await proc.wait()
    except Exception:  # noqa: BLE001
        pass
    if call_id:
        turn.unregister_runner(call_id)

    if final is None:
        # The runner exited without emitting a result line. ``proc.wait()``
        # above already returned, so the process is reaped and its stderr
        # pipe is at EOF — reading it here cannot block. Capture the full
        # text and let ``_describe_runner_exit`` interpret the exit (signal
        # vs. plain rc) and preserve a multi-line stderr tail so the crash
        # is actually debuggable instead of a bare ``rc=<n>: <no stderr>``.
        stderr_text = ""
        if proc.stderr is not None:
            try:
                err_bytes = await proc.stderr.read()
                stderr_text = err_bytes.decode("utf-8", "replace")
            except Exception:  # noqa: BLE001
                pass
        return (
            False,
            f"tool_runner crashed: {_describe_runner_exit(proc.returncode, stderr_text)}",
            False,
            False,
        )

    content = final.get("content", "")
    ok = bool(final.get("ok", False))
    truncated = bool(final.get("truncated", False))
    aborted = not ok and content == "aborted"
    raw_images = final.get("images")
    images = tuple(dict(p) for p in (raw_images if isinstance(raw_images, list) else []) if is_image_url_part(p))
    if images:
        return (
            ok,
            str(content),
            aborted,
            truncated,
            bool(final.get("background_handle")),
            final.get("outcome"),
            tuple(final.get("member_outcomes") or ()),
            images,
        )
    if final.get("outcome") or final.get("member_outcomes") or final.get("background_handle"):
        return (
            ok,
            str(content),
            aborted,
            truncated,
            bool(final.get("background_handle")),
            final.get("outcome"),
            tuple(final.get("member_outcomes") or ()),
        )
    return ok, str(content), aborted, truncated
