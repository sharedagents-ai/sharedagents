"""Tool implementations — the actual functions behind `BUILTIN_TOOLS`.

Each runner takes the tool's `arguments` dict (already parsed from JSON
by the caller) and returns a `ToolResult` whose `content` field is the
string the upstream LLM sees in the next turn's `role:"tool"` message.

Errors are caught and surfaced as `ok=False` results — the model gets
the error text and can decide how to proceed instead of the agent
crashing the whole turn. Long outputs are truncated; the truncation is
visible to the model so it knows there's more.

Requirements
------------
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from functools import partial
from typing import Any

from sharedagents_core.cwdutil import safe_getcwd
from sharedagents_research.native_tools import run_tool as run_ontology_tool

from .impl._utils import *  # noqa: F403  (re-export)
from .impl._utils import (
    ToolResult,
    _err,
    _truncate,
)
from .impl.background import *  # noqa: F403  (re-export)
from .impl.background import (
    run_monitor_read_async,
    run_monitor_stop_async,
)
from .impl.decision import run_decision
from .impl.delegate import *  # noqa: F403  (re-export)
from .impl.integrations import *  # noqa: F403  (re-export)
from .impl.integrations import (
    run_acp_call_async,
    run_lsp,
    run_mcp_call_async,
    run_mcp_list_tools_async,
)
from .impl.planning import *  # noqa: F403  (re-export)
from .impl.planning import (
    run_meeting_search,
    run_skill_get,
    run_todo,
)
from .impl.read import *  # noqa: F403  (re-export)
from .impl.read import (
    run_exit_plan_mode,
    run_glob,
    run_grep,
    run_list_dir,
    run_read_file,
)
from .impl.sdlc import *  # noqa: F403  (re-export)
from .impl.search import *  # noqa: F403  (re-export)
from .impl.search import (
    run_web_fetch_async,
    run_web_search_async,
    run_x_tweets_async,
)
from .impl.shell import *  # noqa: F403  (re-export)
from .impl.shell import (
    run_exec_async,
)

# ============================================================================
# de-monolith re-exports — every runner family below was extracted to
# ``tools/impl/*``. Each ``import *`` re-publishes that module's public +
# traceability surface so every historical
# ``from sharedagents_agent.tools.runners import X`` keeps working unchanged.
# The impl modules pull shared primitives from ``impl._utils``; this file
# keeps only ``ToolResult``/``_SYNC_RUNNERS``/``_ASYNC_RUNNERS``/``dispatch``
# assembly + these shims.
# >>> DEMONO-REEXPORT-ANCHOR >>>
from .impl.subagents import *  # noqa: F403  (re-export)

# Names this module uses directly.
from .impl.subagents import (
    _shim_agent,
    _shim_agent_async,
    _shim_agent_sync,
    _shim_antigravity,
    _shim_batch_agent,
    _shim_claude,
    _shim_codex,
    _shim_zcode,
)
from .impl.write import *  # noqa: F403  (re-export)
from .impl.write import (
    _apply_patch_target_paths,
    run_apply_patch,
    run_edit_file,
    run_write_file,
)

# <<< DEMONO-REEXPORT-ANCHOR <<<
# ============================================================================
#
# Traceability anchor for the delegate-result source-pin checks
# (test_si_harness_h6_followups). The claude-CLI result predicates below were
# extracted WITH ``run_delegate_claude_async`` into ``impl/delegate.py`` and
# still gate behaviour there byte-identically — this comment keeps the pinned
# predicates greppable from the ``runners`` module after the de-monolith split,
# exactly as the ``import *`` shims above keep the import surface intact:
#   * a claude result with ``is_error`` but ``_res_subtype != "success"`` is a
#     real failure (a ``subtype == "success"`` result is treated as success);
#   * the fresh-retry path fires when ``error_subtype == "error_during_execution"``.
# See ``sharedagents_agent.tools.impl.delegate.run_delegate_claude_async``.


_SYNC_RUNNERS: dict[str, Callable[[dict[str, Any]], ToolResult]] = {
    "decision": run_decision,
    "read": run_read_file,
    "list": run_list_dir,
    "glob": run_glob,
    "grep": run_grep,
    "lsp": run_lsp,
    "write": run_write_file,
    "edit": run_edit_file,
    "apply_patch": run_apply_patch,
    "Skill": run_skill_get,
    # memory_search removed: memory recall is wire-only via
    # mcp__memory__search; the builtin dispatch is gone so the model cannot reach
    # memory as a builtin tool (was duplicated alongside mcp__memory__*).
    "meeting_search": run_meeting_search,
    "todowrite": run_todo,
    "ExitPlanMode": run_exit_plan_mode,
}

_ASYNC_RUNNERS: dict[str, Callable[[dict[str, Any]], asyncio.Future[ToolResult]]] = {
    "ontology_search": partial(run_ontology_tool, "ontology_search"),
    "ontology_source": partial(run_ontology_tool, "ontology_source"),
    "ontology_update": partial(run_ontology_tool, "ontology_update"),
    "bash": run_exec_async,
    "BashOutput": run_monitor_read_async,
    "KillShell": run_monitor_stop_async,
    # Delegate-class tools route through the Agent + Engine shims so
    # the autonomous-tool path (model emits ``delegate_claude`` as a
    # tool call) and the slash-command direct-dispatch path
    # (gateway_shim._dispatch_direct_delegate) converge on the same
    # backend. Without this both paths existed in parallel and a
    # divergence between them was a latent bug.
    "delegate_claude": _shim_claude,
    "delegate_codex": _shim_codex,
    "delegate_antigravity": _shim_antigravity,
    "delegate_zcode": _shim_zcode,
    "delegate_agent": _shim_agent,
    # First-party dispatch is AgentSync / AgentAsync only. There is no
    # ``Agent`` tool. The four ``delegate_*`` keys stay for slash commands
    # (``/claude`` / ``/codex`` / ``/antigravity``).
    "AgentSync": _shim_agent_sync,
    "AgentAsync": _shim_agent_async,
    "BatchAgent": _shim_batch_agent,
    # remember removed: memory writes are wire-only via
    # mcp__memory__remember; the builtin dispatch is gone so the model no longer
    # sees a duplicate builtin `remember` alongside mcp__memory__remember.
    "mcp_list_tools": run_mcp_list_tools_async,
    "mcp_call": run_mcp_call_async,
    "acp_call": run_acp_call_async,
    # opencode-parity builtin web tools (registry.py). Backed by sharedagents's
    # httpx + trafilatura (webfetch) and Brave (websearch) implementations.
    "webfetch": run_web_fetch_async,
    "websearch": run_web_search_async,
    "x_tweets": run_x_tweets_async,
}


async def dispatch(
    tool_name: str,
    args: dict[str, Any],
    *,
    on_delta: Callable[[str], None] | None = None,
    on_event: Callable[[dict], None] | None = None,
) -> ToolResult:
    """Run `tool_name(args)`, returning a `ToolResult`.

    Sync runners go through `asyncio.to_thread` so file I/O doesn't
    block the event loop. Unknown tool names return an error result
    that the model will see in its `tool` message.

    `on_delta` is the streaming hook: tools that produce output
    incrementally (today: `delegate_claude` as Claude's text streams
    in) call it to push partial text back to the parent runner. The
    runner forwards it as a `{"type": "delta", "text": "..."}` JSON
    line on stdout, which `_read_tool_result` reads and the SSE layer
    turns into a `delta` event for the TUI / bash bridge. Tools that
    don't stream simply ignore the callback.

    `on_event` is the structured side-channel for tools that emit
    discrete events alongside their text stream — today,
    `delegate_claude` uses it to surface Claude's own inner
    `tool_use` / `tool_result` blocks as `{"kind": "tool_use", ...}`
    / `{"kind": "tool_result", ...}` records so the frontend can
    mount native tool-call widgets (the TUI's ToolCallMessage /
    ToolResultMessage, the plan checklist for TodoWrite). Tools that
    don't emit events ignore it.

    This single dispatch chokepoint also refuses generic Write/Edit of
    MEMORY.md / memory/ so mutations go through `remember`.
    """
    args = dict(args or {})
    # Drop any legacy/internal severity claim before dispatch — the runner
    # doesn't need it (the permission gate already consumed it).
    args.pop("criticality", None)

    from .permissions import privacy_gate_check

    _agent_home = os.environ.get("SHAREDAGENTS_AGENT_HOME") or ""
    _sessions_dir = os.environ.get("SHAREDAGENTS_SESSIONS_DIR") or ""
    if _agent_home or _sessions_dir:
        _gate_ok, _gate_reason = privacy_gate_check(
            tool_name,
            args,
            agent_home=_agent_home,
            sessions_dir=_sessions_dir,
            workspace_origin=os.environ.get("SHAREDAGENTS_WORKSPACE_ORIGIN", ""),
            chat_id=os.environ.get("SHAREDAGENTS_SESSION_ID") or "",
        )
        if not _gate_ok:
            return _err(_gate_reason)

    # ---- Path-scoped write gate ----
    # Same single-chokepoint pattern as the memory privacy gate above: identity
    # comes from the env the acp_bridge stamps on the tool_runner subprocess. If
    # the acting agent's profile restricts writes (write_roots is non-None), a
    # Write/Edit whose resolved target escapes every allowed root is
    # refused, fail-closed. Only Write/Edit are enforced because only those are
    # routed to a mutation runner in _SYNC_RUNNERS; "Update" (run_update_file)
    # exists but is NOT wired into _SYNC_RUNNERS, so dispatch("Update") returns
    # unknown before this gate runs (see permissions._SCOPED_WRITE_TOOLS — wire
    # it there too if "Update" is ever added to _SYNC_RUNNERS). A normal
    # (unrestricted) agent's file tools are
    # untouched. A relative root ("plan/") is honored in the target's OWN git repo
    # (write_scope_check → _git_repo_root), so a path-scoped agent may write that
    # folder in ANY repository — not just the SHAREDAGENTS_WORKSPACE_CWD workspace, which
    # is empty/"/" when the session has no workspace record (the 2026-07-12 defect).
    from sharedagents_agent.agent_profiles import (
        profile_write_denyroots,
        profile_write_roots,
    )

    _agent_id = os.environ.get("SHAREDAGENTS_AGENT_PROFILE") or os.environ.get("SHAREDAGENTS_AGENT_NAME") or ""
    _wr = profile_write_roots(_agent_id)
    _wd = profile_write_denyroots(_agent_id)
    if _wr is not None or _wd:
        from .permissions import scoped_write_abspath, write_scope_check

        _ws_base = os.environ.get("SHAREDAGENTS_WORKSPACE_CWD") or safe_getcwd()
        _ws_ok, _ws_reason = write_scope_check(
            tool_name,
            args,
            write_roots=_wr,
            base_dir=_ws_base,
            deny_roots=_wd,
        )
        if not _ws_ok:
            return _err(_ws_reason)
        # Check-base == write-base: after the scope check passes, rewrite the
        # tool arg's ``file_path`` to the ABSOLUTE path resolved against the SAME
        # base the check used, so the runner writes to exactly the validated
        # location — NOT relative to this tool_runner subprocess's cwd, which
        # acp_bridge leaves at the gateway's cwd rather than the workspace. A
        # relative "plan/leak.md" that passed the check against the workspace
        # would otherwise be written under the process cwd (outside the
        # validated plan/ root). Independent of process cwd, symlink/.. already
        # normalized by the check. Unrestricted agents (write_roots is None)
        # never enter this block, so their file tools are untouched.
        _ws_abs = scoped_write_abspath(
            tool_name,
            args,
            write_roots=_wr,
            base_dir=_ws_base,
        )
        if _ws_abs is not None:
            args["file_path"] = _ws_abs
            # Also set the OpenCode canonical camelCase key the runner reads, so
            # the runner writes to the validated absolute path regardless of which
            # spelling it prefers — else a canonical ``filePath`` arg would keep
            # the un-validated relative path and be written under the process cwd.
            args["filePath"] = _ws_abs
            args.pop("path", None)
            # RE-RUN the memory-privacy gate on the FINAL absolute path. The scope
            # rewrite resolves a relative path against SHAREDAGENTS_WORKSPACE_CWD, but
            # the first privacy check (above) resolved it against the tool-runner
            # process cwd — a DIFFERENT base. So a relative ``filePath`` that the
            # first check cleared could, once rewritten against the workspace, land
            # inside the memory perimeter (e.g. ``.sharedagents/memory/x`` under a
            # workspace whose agent-home is nested there). Re-check the absolute
            # path, fail closed.
            if _agent_home or _sessions_dir:
                _pok2, _pr2 = privacy_gate_check(
                    tool_name,
                    args,
                    agent_home=_agent_home,
                    sessions_dir=_sessions_dir,
                    workspace_origin=os.environ.get("SHAREDAGENTS_WORKSPACE_ORIGIN", ""),
                    chat_id=os.environ.get("SHAREDAGENTS_SESSION_ID") or "",
                )
                if not _pok2:
                    return _err(_pr2)

    # apply_patch carries MULTIPLE target paths inside ``patchText`` — the
    # single-path write-scope + privacy gates above cannot see them. Extract every
    # target (Add/Delete/Update + Move destination) and enforce both gates on each,
    # fail-closed, so a restricted agent (e.g. plan/build) cannot use apply_patch to
    # write outside its root or into the memory perimeter.
    if tool_name == "apply_patch":
        _patch_targets = _apply_patch_target_paths(
            args.get("patchText") or args.get("patch_text") or "",
        )
        _ws_base2 = os.environ.get("SHAREDAGENTS_WORKSPACE_CWD") or safe_getcwd()
        from .permissions import apply_patch_target_denied as _aptd

        for _pt in _patch_targets:
            # Validate BOTH the lexical directory-entry path and the physical
            # realpath against write-scope + memory-privacy — run_apply_patch
            # resolves relative targets against the SAME base. Checking the lexical
            # entry (not just realpath) blocks the symlink-pivot where a delete/move
            # unlinks an in-perimeter symlink validated by its in-scope target, then
            # a later op writes lexically.
            _denied, _reason = _aptd(
                _pt,
                write_roots=_wr,
                base_dir=_ws_base2,
                agent_home=_agent_home,
                sessions_dir=_sessions_dir,
            )
            if _denied:
                return _err(_reason)

    # Standard MCP namespacing: ``mcp__<server>__<tool>``. The client
    # injected discovered MCP tools into the model's tool list with
    # this prefix at startup; route invocations back to the
    # corresponding MCP server here.
    from sharedagents_mcp import mcp_client as _mcp

    parsed = _mcp.parse_mcp_namespaced(tool_name)
    if parsed is not None:
        server_name, original_tool = parsed
        # Surface the MCP call to (a) stderr — works for the HTTP ACP
        # daemon whose stderr goes to gateway.log — AND (b) a file
        # under <home>/mcp_calls.log so the stdio ACP path (whose
        # subprocess stderr is captured by the parent and not
        # forwarded) is also observable.
        import sys as _sys
        import time as _time

        line = f"[mcp_call] {_time.strftime('%Y-%m-%d %H:%M:%S')} {tool_name} server={server_name} tool={original_tool}"
        print(line, file=_sys.stderr, flush=True)
        try:
            from sharedagents_agent.agent import sharedagents_root

            (sharedagents_root() / "mcp_calls.log").open("a").write(line + "\n")
        except Exception:  # noqa: BLE001
            pass
        servers = _mcp.load_mcp_servers()
        # Overlay this session's ACP-supplied MCP servers (env channel), but
        # ADD-ONLY: a session server may add tools under its OWN name, never
        # OVERRIDE a global server. Otherwise a session ``postgres`` would run
        # while the gate resolved criticality from the GLOBAL ``postgres``'s
        # cached readOnlyHint — a too-permissive bypass. register/env-decode
        # already drop names known at those moments; the ``not in servers``
        # guard here also covers a global server ADDED AFTER the session
        # registered (reserved-namespace hardening).
        servers = {
            **servers,
            **{k: v for k, v in _mcp.acp_servers_from_env().items() if k not in servers},
        }
        server = servers.get(server_name)
        if server is None:
            return _err(
                f"MCP server {server_name!r} no longer configured "
                f"(tool was {tool_name!r}). Reload the agent after "
                "editing mcpServers in the global config."
            )
        # Only the BUILTIN memory server (python -m sharedagents_mcp.memory.server) is
        # routed through the persistent client; an operator override named
        # "memory" (a different command/args) must still use the generic path.
        # EXACT tuple-element membership (not substring): the builtin args are
        # ("-m", "sharedagents_mcp.memory.server"); an operator override arg like
        # "--upstream=sharedagents_mcp.memory.server" must NOT be misread as builtin.
        _is_builtin_memory = server_name == "memory" and "sharedagents_mcp.memory.server" in tuple(
            getattr(server, "args", ()) or ()
        )
        if _is_builtin_memory:
            # Route the model's memory tools through the
            # PERSISTENT wire client (warm server on this loop) rather than a raw
            # one-shot. Inject the TRUSTED turn chat context (env), OVERWRITING
            # any model-emitted value so the model cannot forge a chat_id for
            # another chat. NOTE: these dispatches currently run
            # in a per-call tool subprocess (_default_run_tool), so the warm
            # server is reused only within a subprocess — full cross-call reuse
            # needs a gateway-spawned long-lived memory server (socket transport).
            from sharedagents_mcp.memory.client import (
                SLOW_DEADLINE_S as _mem_slow,
                get_memory_client as _get_mem_client,
            )

            _mem_args = dict(args)
            _mem_args["chat_id"] = os.environ.get("SHAREDAGENTS_SESSION_ID") or ""
            _mem_args["workspace_origin"] = os.environ.get("SHAREDAGENTS_WORKSPACE_ORIGIN", "")
            # Explicit model tool calls (not auto-recall) — bounded to the slow
            # deadline, not the 250ms auto-hot deadline.
            ok, body = await _get_mem_client(
                os.environ.get("SHAREDAGENTS_AGENT_HOME") or None,
            )._call(original_tool, _mem_args, deadline_s=_mem_slow)
        else:
            ok, body = await _mcp.call_tool(server, original_tool, args)
        result_line = f"[mcp_call] {_time.strftime('%Y-%m-%d %H:%M:%S')} {tool_name} → ok={ok} bytes={len(body or '')}"
        print(result_line, file=_sys.stderr, flush=True)
        try:
            (sharedagents_root() / "mcp_calls.log").open("a").write(result_line + "\n")
        except Exception:  # noqa: BLE001
            pass
        body, truncated = _truncate(body or "(empty result)")
        return ToolResult(content=body, ok=ok, truncated=truncated)

    if tool_name in _ASYNC_RUNNERS:
        runner = _ASYNC_RUNNERS[tool_name]
        # Streaming-capable runners declare an `on_delta` keyword; some
        # also accept `on_event` for structured tool events. We pass
        # each kwarg only when the signature accepts it so existing
        # runners stay unchanged.
        import inspect

        sig = inspect.signature(runner)
        kwargs: dict[str, Any] = {}
        if "on_delta" in sig.parameters:
            kwargs["on_delta"] = on_delta
        if "on_event" in sig.parameters:
            kwargs["on_event"] = on_event
        return await runner(args, **kwargs)
    if tool_name in _SYNC_RUNNERS:
        return await asyncio.to_thread(_SYNC_RUNNERS[tool_name], args)
    return _err(f"unknown tool: {tool_name}")
