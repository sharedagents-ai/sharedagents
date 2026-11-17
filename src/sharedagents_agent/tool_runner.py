"""Per-tool-call subprocess wrapper.

One process per tool invocation. The agent runtime spawns this via
`python -m sharedagents.tool_runner`, writes the request as a JSON line on
stdin, and reads a stream of JSON lines on stdout (terminating with a
`final` record). The Process object is tracked on `TurnState.proc`
for the duration of the call, so ESC (which routes through
`TurnState.stop()` → `proc.terminate()`) instantly kills whatever the
tool is doing — a runaway `grep`, a filesystem walk, an HTTP fetch,
the `claude -p` subprocess inside `delegate_claude`.

This is the foundational rule: every long-running operation lives in
its own OS process. See `CLAUDE.md` and `src/README.md`.

Wire format (stdin):

    {"tool": "Grep", "args": {"pattern": "TODO", "path": "/home"}}

Wire format (stdout, one or more JSON lines, terminated by a final
record):

    {"type": "delta", "text": "..."}                          # streaming tools only (delegate_claude)
    {"type": "tool_event", "kind": "tool_use"|"tool_result", ...}  # structured side-channel
    {"type": "final", "ok": true, "content": "...", "truncated": false}

Backward-compat: tools that emit a single line of the legacy shape
(`{"ok": ..., "content": ..., "truncated": ...}` with no `"type"`)
still work — the readers take an untyped record as a final. A record with
any other `type` is never the result; the readers skip it.

Exit codes:

    0  request handled (`ok` field reports tool success/failure)
    1  bad request / runner crash before producing a result
    SIGTERM-handled: emits {"type": "final", "ok": false, "content": "aborted"} and exits 0
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from typing import Any

from sharedagents_agent.tools.runners import ToolResult, dispatch


def _emit(obj: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _emit_delta(text: str) -> None:
    """Streaming-capable tools (e.g. `delegate_claude`) call this via
    the `on_delta` callback to push incremental text. Each call emits
    one JSON line on stdout; the agent runtime forwards it as an SSE
    `delta` event so the TUI / bash bridge can render token-by-token.
    """
    if not text:
        return
    _emit({"type": "delta", "text": text})


def _emit_tool_event(event: dict[str, Any]) -> None:
    """Structured side-channel for inner tool events.

    Streaming-capable tools that wrap an outer agent (today:
    ``delegate_claude``) emit one of these per inner ``tool_use`` /
    ``tool_result`` block — ``{"kind": "tool_use", "id", "name",
    "input"}`` or ``{"kind": "tool_result", "tool_use_id", "ok",
    "text"}``. The parent runner parses it from stdout and translates
    it into ACP ``tool_call`` / ``tool_call_update`` notifications
    keyed off a fresh inner ``toolCallId``, so the TUI mounts a
    native tool-call widget for the inner call (and ``TodoWrite``
    tick the plan checklist) without any per-frontend special-casing.

    The envelope ``type`` is set LAST: an event that carries its own ``type``
    (the ``sensitive_egress_flag``) must not retype the
    record, or the reader takes it for the tool's result.
    """
    if not event:
        return
    _emit({**event, "type": "tool_event"})


async def _run() -> int:
    """Read the request, dispatch the tool, emit one JSON line, exit.

    SIGTERM/SIGINT handlers are wired into the running event loop and
    cancel the dispatch task. asyncio.CancelledError unwinds through
    `tools.runners` (where exec/web_fetch's finally blocks kill their
    in-flight child subprocess), so a SIGTERM here propagates to whatever
    real long-running thing the tool was doing — `sleep 30`, `grep -r`,
    a long HTTP fetch — within a fraction of a second.
    """
    loop = asyncio.get_running_loop()

    raw = sys.stdin.read()
    try:
        req = json.loads(raw)
    except json.JSONDecodeError as e:
        _emit({"ok": False, "content": f"bad request json: {e}"})
        return 1

    tool = req.get("tool")
    args = req.get("args") or {}
    if not tool or not isinstance(tool, str):
        _emit({"ok": False, "content": "missing tool name"})
        return 1
    if not isinstance(args, dict):
        _emit({"ok": False, "content": "args must be an object"})
        return 1

    # Run dispatch as a Task so we have a handle the signal handler can
    # cancel. asyncio.run(_run()) gives us the event-loop signal handler
    # API needed to wire SIGTERM → task.cancel().
    task: asyncio.Task[ToolResult] = asyncio.ensure_future(
        dispatch(tool, args, on_delta=_emit_delta, on_event=_emit_tool_event)
    )

    # Diagnostic: log to stderr if a signal lands. The parent captures
    # stderr but never reads it under normal flow — only useful when
    # debugging. Stays out of the way otherwise.
    def _cancel(signame: str = "?") -> None:
        sys.stderr.write(f"[tool_runner] SIG {signame} received, cancelling task\n")
        sys.stderr.flush()
        if not task.done():
            task.cancel()

    import functools

    for sig, name in ((signal.SIGTERM, "TERM"), (signal.SIGINT, "INT")):
        try:
            loop.add_signal_handler(sig, functools.partial(_cancel, name))
        except (NotImplementedError, RuntimeError):
            pass

    try:
        result: ToolResult = await task
    except asyncio.CancelledError:
        _emit({"type": "final", "ok": False, "content": "aborted"})
        # Hard exit: a sync tool dispatched via `asyncio.to_thread` keeps
        # the thread running even after the awaiting task is cancelled —
        # `asyncio.run`'s loop shutdown then blocks waiting for the
        # executor to drain. Skip the cooperative shutdown entirely so
        # SIGTERM is felt instantly. Foundational killability rule.
        os._exit(0)
    except Exception as e:  # noqa: BLE001
        _emit({"type": "final", "ok": False, "content": f"runner crash: {e}"})
        return 1

    _emit(
        {
            "type": "final",
            "ok": result.ok,
            "content": result.content,
            "truncated": result.truncated,
            "background_handle": result.background_handle,
            "outcome": result.outcome,
            "member_outcomes": result.member_outcomes,
            # Only present when the tool returned images, so every other
            # tool's final record stays byte-identical.
            **({"images": list(result.images)} if getattr(result, "images", ()) else {}),
        }
    )
    # Hard exit on SUCCESS for the same reason as the cancel path above: a
    # persistent MCP client left alive on this loop — notably the warm memory
    # wire client, whose owner task holds a stdio ClientSession
    # + spawned server child — makes ``asyncio.run``'s cooperative loop shutdown
    # BLOCK for tens of seconds tearing it down. The model then sees the memory
    # tool "take forever" and the turn stalls, even though the search itself
    # already returned in <1s. The result is fully emitted above, so the
    # subprocess has nothing left to do — skip the hanging shutdown entirely.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main() -> int:
    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        _emit({"type": "final", "ok": False, "content": "aborted"})
        return 0


if __name__ == "__main__":
    sys.exit(main())
