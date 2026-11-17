"""Extracted tool-runner family: background (de-monolith of runners.py).

Behavior-preserving split — code moved verbatim.
"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from typing import Any

from sharedagents_core import oscompat

from ._utils import (
    ToolResult,
    _background_usage_footer,
    _err,
    _truncate,
    monitor_timeout_s,
)

# ---- monitor (long-running background process registry) ----

# Mirrors Claude Code's ``Monitor`` tool. The model spawns a process
# via ``monitor_start``, reads new output via ``monitor_read``, and
# kills it via ``monitor_stop``. v1 is pull-based (the model polls);
# the push variant (each line wakes the agent via session-update
# notifications) is a follow-up that needs gateway integration.
#
# The registry is per-worker: every tool subprocess holds its own
# MonitorRegistry, so monitor ids are stable for the duration of a
# tool round but do NOT survive across rounds. Long-running watches
# need either the push variant or a session-scoped registry — both
# are tracked in the README's ``Architecture`` section.


async def run_monitor_start_async(args: dict[str, Any]) -> ToolResult:
    """Spawn a long-running command and return its monitor id.

    Mirrors Claude Code's ``Monitor`` schema (``command``, ``description``,
    ``timeout_ms``, ``persistent``). Legacy ``cwd`` arg still accepted.
    The monitor id is surfaced as ``bash_id`` so the rest of the
    Claude-style tool family (``BashOutput`` / ``KillShell``) can
    address it without operators having to remember the sharedagents
    legacy name.
    """
    from ..monitor import _registry

    command = args.get("command")
    cwd = args.get("cwd")
    description = args.get("description") or ""
    if not isinstance(command, str) or not command:
        return _err("command required")
    # Gateway self-protection — the SAME guard as run_exec_async. The
    # ``Monitor`` tool (and ``Bash(run_in_background=True)``, which routes
    # here) spawns a command in the background; without this an agent
    # could dodge the foreground guard with
    # ``Monitor(command="pkill -f 'sharedagents gateway serve'")``.
    # ``restart`` is NOT matched. See
    # ``sharedagents_agent.tools.safety.kills_sharedagents_gateway``.
    from ..safety import kills_sharedagents_gateway

    if kills_sharedagents_gateway(command):
        return _err(
            "refused: this command would terminate the sharedagents gateway "
            "the agent runs inside. Killing the gateway is never allowed; "
            "use `sharedagents gateway restart` to bounce it instead."
        )
    try:
        mon = await _registry().start(
            command,
            cwd=str(cwd) if isinstance(cwd, str) and cwd else None,
            timeout_s=monitor_timeout_s(args),
            persistent=bool(args.get("persistent")),
        )
    except Exception as e:  # noqa: BLE001
        return _err(f"spawn failed: {e}")
    desc_line = f"description: {description}\n" if description else ""
    mid = mon.monitor_id
    timeout_s = monitor_timeout_s(args)
    kind = str(args.get("_footer_kind") or "Monitor")
    return ToolResult(
        content=(
            f"bash_id: {mid}\n"
            f"{desc_line}"
            f"command: {mon.command}\n"
            f"pid: {mon.process.pid}"
            f"{_background_usage_footer(mid, timeout_s=timeout_s, kind=kind)}"
        ),
    )


async def run_monitor_read_async(args: dict[str, Any]) -> ToolResult:
    """Pop unread lines from a monitor's buffer.

    Reads from the per-monitor logfile (the spawned process writes
    directly there, so SIGPIPE never fires when launcher subprocesses
    cycle). The byte cursor is persisted in the sidecar entry, so
    each ``monitor_read`` call across separate tool rounds resumes
    cleanly without re-reading lines.
    """
    from ..monitor import (
        _registry,
        load_sidecar,
        read_log_lines,
        write_sidecar_entry,
    )

    # Accept Claude Code's ``bash_id`` AND legacy ``monitor_id``.
    mid = args.get("bash_id") or args.get("monitor_id")
    if not isinstance(mid, str) or not mid:
        return _err("bash_id required")
    max_lines = int(args.get("max_lines") or 100)
    timeout_s = float(args.get("timeout_s") or 2.0)
    # Optional Claude ``filter`` regex — drops lines that don't match.
    filter_pat = args.get("filter")
    filter_regex: re.Pattern[str] | None = None
    if isinstance(filter_pat, str) and filter_pat:
        try:
            filter_regex = re.compile(filter_pat)
        except re.error as exc:
            return _err(f"bad filter regex: {exc}")
    # In-proc fast path: if THIS subprocess started the monitor, the
    # registry has the buffered lines + asyncio.Event for tail-style
    # waits. Use it.
    try:
        result = await _registry().read(
            mid,
            max_lines=max_lines,
            timeout_s=timeout_s,
        )
    except LookupError:
        result = None
    if result is not None:
        lines = result["lines"]
        if filter_regex is not None:
            lines = [ln for ln in lines if filter_regex.search(ln["text"])]
        parts = [
            f"bash_id: {result['monitor_id']}",
            f"running: {result['running']}",
        ]
        if result["exit_code"] is not None:
            parts.append(f"exit_code: {result['exit_code']}")
        parts.append(f"new lines: {len(lines)}")
        if result["more_buffered"]:
            parts.append("(more buffered — call BashOutput again)")
        if lines:
            parts.append("--- output ---")
            for ln in lines:
                tag = "[stderr] " if ln["stream"] == "stderr" else ""
                parts.append(f"{tag}{ln['text'].rstrip()}")
        body, truncated = _truncate("\n".join(parts))
        return ToolResult(content=body, truncated=truncated)

    # Cross-subprocess: read from the on-disk logfile starting at the
    # cursor in the sidecar. Wait up to ``timeout_s`` for new bytes
    # to appear (or for the process to exit).
    sidecar = load_sidecar()
    match = next(
        (e for e in sidecar if e.get("monitor_id") == mid),
        None,
    )
    if match is None:
        # The sidecar entry may already be REAPED: a background process
        # that exits naturally has its entry removed by
        # MonitorRegistry._watch_exit (monitor.py) the moment the exit is
        # observed. That removal races the model's first BashOutput, so a
        # bg shell that finishes quickly (e.g. `cd missing/dir && ...`)
        # can leave the model holding a bash_id that sharedagents *just issued*
        # yet returns "unknown monitor_id". The per-monitor logfile is NOT
        # deleted on reap, so fall back to it: the finished shell's output
        # stays drainable instead of surfacing a confusing error. Only a
        # genuinely-never-issued id (no logfile) still errors.
        return _read_reaped_monitor_logfile(mid, max_lines, filter_regex)
    log_path = match.get("log_path")
    if not log_path:
        # Older sidecar entry without log_path — surface status only.
        running = bool(match.get("alive"))
        return ToolResult(
            content=(
                f"monitor_id: {mid}\nrunning: {running}\n"
                f"pid: {match.get('pid')}\n"
                "(no log_path recorded — likely an older monitor; "
                "use monitor_list / monitor_stop instead.)"
            )
        )
    log_path = Path(log_path)
    cursor = int(match.get("read_cursor") or 0)
    pid = int(match.get("pid") or 0)

    deadline = time.monotonic() + min(max(0.0, timeout_s), 30.0)
    lines: list[tuple[str, bool]] = []
    new_cursor = cursor
    while True:
        lines, new_cursor = read_log_lines(
            log_path,
            start_byte=cursor,
            max_lines=max_lines,
        )
        if lines:
            break
        if pid > 0 and not _pid_alive(pid):
            break
        if time.monotonic() >= deadline:
            break
        await asyncio.sleep(0.1)

    if new_cursor != cursor:
        # Persist the advance so the next call (this subprocess or
        # the next one) resumes from the right place.
        merged = {**match, "read_cursor": new_cursor}
        # Keep the value pristine — write_sidecar_entry already
        # strips ``alive``.
        write_sidecar_entry(merged)

    if filter_regex is not None:
        lines = [(t, e) for (t, e) in lines if filter_regex.search(t)]
    running = pid > 0 and _pid_alive(pid)
    parts = [
        f"bash_id: {mid}",
        f"running: {running}",
        f"pid: {pid}",
        f"new lines: {len(lines)}",
    ]
    if lines:
        parts.append("--- output ---")
        for text, is_err in lines:
            tag = "[stderr] " if is_err else ""
            parts.append(f"{tag}{text.rstrip()}")
    body, truncated = _truncate("\n".join(parts))
    return ToolResult(content=body, truncated=truncated)


def _read_reaped_monitor_logfile(
    mid: str,
    max_lines: int,
    filter_regex: re.Pattern[str] | None,
) -> ToolResult:
    """Drain a finished background shell's output after its sidecar entry
    was reaped on process exit.

    Called from ``run_monitor_read_async`` when neither the in-process
    registry nor the sidecar knows ``mid``. The per-monitor logfile
    (``<sid>.monitor-<mid>.log``) survives the sidecar reap, so a bg shell
    that already exited can still have its final output read instead of
    failing with ``unknown monitor_id`` for an id sharedagents itself issued.

    Returns the genuine ``unknown monitor_id`` error only when no logfile
    exists for the id — i.e. the id was never issued (a hallucinated
    bash_id like ``'live'``), not a just-finished monitor.
    """
    from ..monitor import _monitor_logfile_for, read_log_lines

    log_path = _monitor_logfile_for(mid)
    if log_path is None or not log_path.exists():
        return _err(f"unknown monitor_id {mid!r}")
    lines, _new_cursor = read_log_lines(
        log_path,
        start_byte=0,
        max_lines=max_lines,
    )
    if filter_regex is not None:
        lines = [(t, e) for (t, e) in lines if filter_regex.search(t)]
    parts = [
        f"bash_id: {mid}",
        "running: False",
        "(monitor already exited and was reaped; showing final logged output)",
        f"new lines: {len(lines)}",
    ]
    if lines:
        parts.append("--- output ---")
        for text, is_err in lines:
            tag = "[stderr] " if is_err else ""
            parts.append(f"{tag}{text.rstrip()}")
    body, truncated = _truncate("\n".join(parts))
    return ToolResult(content=body, truncated=truncated)


def _pid_alive(pid: int) -> bool:
    return oscompat.process_alive(pid, permission_denied_alive=False)


async def run_monitor_stop_async(args: dict[str, Any]) -> ToolResult:
    """SIGTERM the monitor's process group, escalate to SIGKILL."""
    from ..monitor import _registry

    # Accept Claude Code's ``shell_id`` AND legacy ``monitor_id``.
    mid = args.get("shell_id") or args.get("monitor_id")
    if not isinstance(mid, str) or not mid:
        return _err("shell_id required")
    try:
        result = await _registry().stop(mid)
    except LookupError as e:
        return _err(str(e))
    return ToolResult(
        content=(
            f"shell_id: {result['monitor_id']}\n"
            f"running: {result['running']}\n"
            f"exit_code: {result['exit_code']}\n"
            f"unread_lines: {result['unread_lines']}"
        ),
    )


def run_monitor_list(args: dict[str, Any]) -> ToolResult:  # noqa: ARG001
    """List every live monitor + its buffer state.

    Reads the session sidecar (`<sid>.monitors.json`) so monitors
    started in earlier tool subprocesses still show up. Each entry's
    `alive` flag is recomputed via `os.kill(pid, 0)` on every read,
    so a process that died after its subprocess exited shows as
    `exited` regardless of what the sidecar last recorded.
    """
    from ..monitor import _registry, load_sidecar

    in_proc = {r["monitor_id"]: r for r in _registry().list()}
    sidecar = load_sidecar()
    # Merge: in-process entries win for buffered/unread counts; sidecar
    # provides the cross-subprocess survivors.
    by_id: dict[str, dict[str, Any]] = {}
    for entry in sidecar:
        by_id[entry["monitor_id"]] = {
            "monitor_id": entry["monitor_id"],
            "command": entry.get("command", ""),
            "pid": entry.get("pid"),
            "alive": entry.get("alive", False),
            "started_at_unix": entry.get("started_at_unix"),
            "buffered_lines": 0,
            "unread_lines": 0,
            "scope": "sidecar",
        }
    for mid, row in in_proc.items():
        merged = by_id.get(mid, {})
        merged.update(
            {
                "monitor_id": mid,
                "command": row["command"],
                "alive": row["running"],
                "exit_code": row["exit_code"],
                "buffered_lines": row["buffered_lines"],
                "unread_lines": row["unread_lines"],
                "scope": "in_proc",
            }
        )
        by_id[mid] = merged
    rows = list(by_id.values())
    if not rows:
        return ToolResult(content="(no live monitors)")
    parts = [f"{len(rows)} monitor(s):"]
    for r in rows:
        if r.get("alive"):
            state = "running"
        elif r.get("exit_code") is not None:
            state = f"exited({r['exit_code']})"
        else:
            state = "exited"
        pid = r.get("pid")
        parts.append(f"  {r['monitor_id']}  {state}  pid={pid}  unread={r['unread_lines']}  cmd={r['command']!r}")
    return ToolResult(content="\n".join(parts))


__all__ = [
    "run_monitor_start_async",
    "run_monitor_read_async",
    "_read_reaped_monitor_logfile",
    "_pid_alive",
    "run_monitor_stop_async",
    "run_monitor_list",
]
