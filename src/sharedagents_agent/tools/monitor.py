"""Monitor: long-running background process registry.

Mirrors Claude Code's ``Monitor`` tool. The model spawns a process via
``monitor_start`` and reads new output line-by-line via ``monitor_read``
until the process exits or ``monitor_stop`` reaps it. ``monitor_list``
inspects the registry.

Why pull-based, not push-based:

* The push-based variant (each line becomes a session-update
  notification that wakes the agent) requires deeper integration
  with the worker / turn pipeline so notifications can land
  between rounds. v1 keeps the contract simple — the model
  polls. Once the model's planning loop is stable around the
  pull contract, the push variant is a strict superset.

* The pull contract composes with the gateway's cancel cascade
  for free: when the parent session ends, ``MonitorRegistry``
  receives ``shutdown_session`` and reaps every monitor it owns.

Killability:

* Each monitor runs in its own process group (``start_new_session``).
  ``stop`` SIGTERMs the group, escalates to SIGKILL after a grace
  period.
* The session-scoped registry kills every owned monitor on session
  close so an aborted parent never leaks background work.

The registry lives at process scope inside the single-process
gateway (one registry shared across every agent mounted under
``/acp/<name>``). Monitor IDs are opaque short strings — humans
should not be expected to type them.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sharedagents_core import oscompat

from ._log import log_event

# How many lines a monitor buffers in memory before dropping the head.
# 4 KB lines × 5 000 = ~20 MB worst case per monitor. The model can
# always raise the cap by reading more often.
_MAX_BUFFERED_LINES = 5_000

# Default per-call timeout for ``monitor_read``: how long to wait for
# at least one line if nothing is buffered yet. Bounded so a runaway
# poll cannot block a tool round indefinitely.
_DEFAULT_READ_TIMEOUT_S = 2.0
_MAX_READ_TIMEOUT_S = 30.0


@dataclass
class _Monitor:
    """One backgrounded process + its line buffer."""

    monitor_id: str
    command: str
    cwd: str | None
    process: Any  # asyncio.subprocess.Process
    started_at: float
    lines: list[tuple[float, str, bool]] = field(default_factory=list)
    """[(monotonic_ts, line, is_stderr)]. Rolling — head dropped past cap."""
    cursor: int = 0
    """Next index in ``lines`` ``monitor_read`` should yield from."""
    new_line: asyncio.Event = field(default_factory=asyncio.Event)
    drain_tasks: list[asyncio.Task[None]] = field(default_factory=list)
    closed: bool = False
    exit_code: int | None = None
    log_path: Path | None = None
    """Per-monitor on-disk logfile. Subprocess writes here directly so
    SIGPIPE doesn't fire when the spawning tool subprocess exits."""
    deadline_unix: float | None = None
    """Wall-clock kill time (UTC epoch). Set when ``timeout_s`` is armed."""
    deadline_task: asyncio.Task[None] | None = None
    deadline_process: Any = None
    """Windows stdlib deadline watcher; the shell remains the owned process."""

    def push(self, text: str, is_stderr: bool) -> None:
        self.lines.append((time.monotonic(), text, is_stderr))
        if len(self.lines) > _MAX_BUFFERED_LINES:
            # Drop head + slide cursor so old indices remain meaningful.
            drop = len(self.lines) - _MAX_BUFFERED_LINES
            self.lines = self.lines[drop:]
            self.cursor = max(0, self.cursor - drop)
            # Surface buffer-overflow events so a debugger can see when
            # output started getting lost. Once per overflow, not per
            # dropped line — otherwise a chatty process spams the log.
            log_event(
                "MONITOR_BUFFER_DROP",
                monitor_id=self.monitor_id,
                dropped=drop,
                cap=_MAX_BUFFERED_LINES,
            )
        self.new_line.set()


class MonitorRegistry:
    """Per-session registry of live monitors.

    Owned by the south orchestrator. ``shutdown`` reaps every
    monitor; the gateway invokes it when a parent session closes so
    backgrounded watches cannot outlive their owner.
    """

    def __init__(self) -> None:
        self._monitors: dict[str, _Monitor] = {}
        self._lock = asyncio.Lock()

    def list(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in self._monitors.values():
            out.append(
                {
                    "monitor_id": m.monitor_id,
                    "command": m.command,
                    "cwd": m.cwd,
                    "started_at": m.started_at,
                    "running": m.exit_code is None,
                    "exit_code": m.exit_code,
                    "buffered_lines": len(m.lines),
                    "unread_lines": max(0, len(m.lines) - m.cursor),
                }
            )
        return out

    async def start(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout_s: float | None = None,
        persistent: bool = False,
    ) -> _Monitor:
        """Spawn ``command`` in its own process group with stdout +
        stderr redirected to a per-monitor logfile.

        When ``timeout_s`` is set (and ``persistent`` is false), the
        POSIX uses GNU ``timeout`` plus an in-process backup reaper.
        Windows uses a stdlib watcher so the deadline survives the launcher's
        exit; termination is best effort and grandchildren may survive.
        """
        if not command:
            raise ValueError("command required")
        cwd_path: str | None = None
        if cwd:
            cwd_path = str(Path(cwd).expanduser())

        spawn_command = command
        deadline_unix: float | None = None
        if not persistent and timeout_s is not None and timeout_s > 0:
            secs = max(1, int(timeout_s))
            if not oscompat.IS_WINDOWS:
                spawn_command = f"timeout --foreground -s TERM -k 2 {secs}s bash -c {shlex.quote(command)}"
            deadline_unix = time.time() + secs

        mid = uuid.uuid4().hex[:12]

        # Per-monitor logfile. Sit alongside the sidecar so a session
        # cleanup hook can wipe both with one glob.
        log_path = _monitor_logfile_for(mid)
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # No session id (ad-hoc invocation outside an agent run):
            # tempfile fallback so the subprocess still survives the
            # spawning tool subprocess's exit.
            import tempfile

            log_path = Path(tempfile.mkstemp(prefix=f"sharedagents-mon-{mid}-", suffix=".log")[1])

        # Use plain subprocess.Popen, NOT asyncio. asyncio's
        # subprocess transport runs ``__del__`` on loop close which
        # can interfere with the child. Popen creates a fully
        # detached child the operator can hold via PID alone.
        #
        # `executable=` forces bash for `shell=True` so background
        # commands get the same bash semantics (`source`, `[[ … ]]`,
        # arrays) as the foreground exec path — otherwise Popen's
        # shell=True uses /bin/sh (dash) and `source` fails with exit
        # 127. None falls back to the platform default (/bin/sh).
        from sharedagents_agent.tools.runners import _bash_executable

        log_fp = open(log_path, "ab", buffering=0)
        deadline_process = None
        try:
            spawn_kwargs = dict(
                stdout=log_fp,
                stderr=log_fp,
                cwd=cwd_path,
                start_new_session=not oscompat.IS_WINDOWS,
                **oscompat.detach_kwargs(),
            )
            if oscompat.IS_WINDOWS:
                argv = oscompat.spawn_shell_argv(command)
                proc = subprocess.Popen(argv, **spawn_kwargs)
                if deadline_unix is not None:
                    try:
                        deadline_process = subprocess.Popen(
                            [sys.executable, "-m", "sharedagents_core.timed_shell", str(proc.pid), str(deadline_unix)],
                            **spawn_kwargs,
                        )
                    except OSError:
                        # Do not leave an unbounded shell if its timer cannot start.
                        with contextlib.suppress(OSError):
                            oscompat.kill_process(proc.pid, force=True)
                        raise
            else:
                proc = subprocess.Popen(  # noqa: S602
                    spawn_command,
                    shell=True,
                    executable=_bash_executable(),
                    **spawn_kwargs,
                )
        finally:
            log_fp.close()

        mon = _Monitor(
            monitor_id=mid,
            command=command,
            cwd=cwd_path,
            process=proc,
            started_at=time.monotonic(),
            deadline_unix=deadline_unix,
            deadline_process=deadline_process,
        )
        mon.log_path = log_path
        # Lease the background shell's process group (§10.4): a monitor is an
        # untrusted command exec; its stop/terminate kill is then capability-
        # validated (downward-only) via the broker, with raw killpg fallback.
        from sharedagents_agent import sandbox_broker as _sb

        _sb.lease_process(pgid=proc.pid, kind="monitor")

        # Drain task: tail the logfile and push lines to the in-memory
        # buffer. The drain is async-friendly (poll the file size) so
        # cancellation when this subprocess exits is graceful — the
        # actual file keeps growing regardless.
        mon.drain_tasks = [
            asyncio.create_task(
                self._drain_file(mon),
                name=f"monitor-{mid}-drain",
            ),
            asyncio.create_task(
                self._watch_exit(mon),
                name=f"monitor-{mid}-wait",
            ),
        ]
        if deadline_unix is not None and not oscompat.IS_WINDOWS:
            mon.deadline_task = asyncio.create_task(
                self._deadline_reaper(mon, timeout_s=float(timeout_s)),
                name=f"monitor-{mid}-deadline",
            )
            mon.drain_tasks.append(mon.deadline_task)
        async with self._lock:
            self._monitors[mid] = mon
        log_event(
            "MONITOR_START",
            monitor_id=mid,
            pid=proc.pid,
            command=command,
            cwd=cwd_path,
            log_path=str(log_path),
        )
        write_sidecar_entry(
            {
                "monitor_id": mid,
                "pid": proc.pid,
                "command": command,
                "cwd": cwd_path,
                "started_at_unix": time.time(),
                "deadline_unix": deadline_unix,
                "timeout_s": timeout_s if not persistent else None,
                "exit_code": None,
                "log_path": str(log_path),
                "read_cursor": 0,
            }
        )
        return mon

    async def _deadline_reaper(
        self,
        mon: _Monitor,
        *,
        timeout_s: float,
    ) -> None:
        """In-process backup: ``stop()`` when the advertised wall expires."""
        try:
            await asyncio.sleep(max(0.0, timeout_s))
            if mon.exit_code is None and not mon.closed:
                log_event(
                    "MONITOR_DEADLINE",
                    monitor_id=mon.monitor_id,
                    timeout_s=timeout_s,
                )
                await self.stop(mon.monitor_id, grace_s=2.0)
        except asyncio.CancelledError:
            return

    async def _drain_file(self, mon: _Monitor) -> None:
        """Tail the per-monitor logfile into the in-memory buffer.

        Polls every 100 ms — coarse but cheap. The file is the
        source of truth; this just keeps the in-process buffer warm
        for fast ``monitor_read`` calls in the same subprocess.
        """
        if mon.log_path is None:
            return
        path = mon.log_path
        cursor = 0
        while True:
            try:
                if path.exists():
                    size = path.stat().st_size
                    if size > cursor:
                        with path.open("rb") as f:
                            f.seek(cursor)
                            chunk = f.read(size - cursor)
                        cursor = size
                        # Split on newlines; trailing partial line is
                        # held until the next read by adjusting cursor.
                        text = chunk.decode("utf-8", "replace")
                        # Rewind cursor by the trailing partial-line
                        # bytes so the next iteration re-reads them.
                        last_nl = text.rfind("\n")
                        if last_nl == -1:
                            # No newline yet — no full lines this round.
                            cursor -= len(chunk)
                        else:
                            tail_partial = len(text) - last_nl - 1
                            cursor -= len(chunk[len(chunk) - tail_partial :]) if tail_partial else 0
                            for line in text[: last_nl + 1].splitlines(keepends=True):
                                mon.push(line, False)
                # Stop draining once the process has exited AND no new
                # bytes are appearing — saves a poll cycle on long-dead
                # monitors.
                if mon.exit_code is not None:
                    # One last drain pass already happened above; bail.
                    return
            except (OSError, ValueError):
                pass
            try:
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                return

    async def _watch_exit(self, mon: _Monitor) -> None:
        # ``Popen.wait()`` is sync. We can't ``await asyncio.to_thread``
        # it — the executor thread would block process shutdown for
        # the full lifetime of the monitor (a 10-minute ``sleep 600``
        # blocks for 10 minutes after the launcher subprocess tries
        # to exit). Poll ``Popen.poll()`` instead — it's
        # non-blocking, returns ``None`` while the child is alive
        # and the exit code once it's gone.
        try:
            while True:
                rc = mon.process.poll()
                if rc is not None:
                    mon.exit_code = rc
                    return
                await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            # Launcher subprocess is exiting; the child stays alive
            # because it's in its own session group. Don't observe
            # its eventual exit — the next subprocess that opens the
            # sidecar will recompute liveness via ``os.kill(pid, 0)``.
            return
        except Exception as exc:  # noqa: BLE001
            mon.exit_code = -1
            log_event(
                "MONITOR_WAIT_ERROR",
                monitor_id=mon.monitor_id,
                error=str(exc),
            )
        finally:
            mon.new_line.set()  # unblock any waiting reader
            # Distinguish a real natural exit (we observed mon.exit_code)
            # from a tool-subprocess teardown that cancelled _watch_exit
            # before proc.wait() returned. The cancelled path leaves
            # exit_code=None and the actual process keeps running because
            # it was started in its own process group; the sidecar entry
            # SHOULD remain so the next subprocess (or the TUI) can still
            # see the live PID.
            log_event(
                "MONITOR_EXIT",
                monitor_id=mon.monitor_id,
                exit_code=mon.exit_code,
                buffered_lines=len(mon.lines),
                unread=max(0, len(mon.lines) - mon.cursor),
                cancelled=mon.exit_code is None,
            )
            if mon.exit_code is not None:
                # Real exit — drop from sidecar so the TUI / monitor_list
                # stops listing it. Cancelled path keeps the entry; the
                # `alive` flag is recomputed from os.kill on every read.
                remove_sidecar_entry(mon.monitor_id)

    def get(self, monitor_id: str) -> _Monitor | None:
        return self._monitors.get(monitor_id)

    async def read(
        self,
        monitor_id: str,
        *,
        max_lines: int = 100,
        timeout_s: float = _DEFAULT_READ_TIMEOUT_S,
    ) -> dict[str, Any]:
        """Pop up to ``max_lines`` new lines from the buffer.

        If nothing new is buffered, waits up to ``timeout_s`` for the
        next line (or process exit). Returns a dict with the lines,
        the running flag, and the exit code if exited.
        """
        timeout_s = min(max(0.0, float(timeout_s)), _MAX_READ_TIMEOUT_S)
        max_lines = max(1, int(max_lines))
        mon = self._monitors.get(monitor_id)
        if mon is None:
            raise LookupError(f"unknown monitor_id {monitor_id!r}")

        # Wait for new lines if the buffer is empty.
        if mon.cursor >= len(mon.lines) and mon.exit_code is None:
            mon.new_line.clear()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(mon.new_line.wait(), timeout=timeout_s)

        slice_ = mon.lines[mon.cursor : mon.cursor + max_lines]
        mon.cursor += len(slice_)
        log_event(
            "MONITOR_READ",
            monitor_id=monitor_id,
            lines_returned=len(slice_),
            more_buffered=mon.cursor < len(mon.lines),
            running=mon.exit_code is None,
            exit_code=mon.exit_code,
            timeout_s=timeout_s,
        )
        return {
            "monitor_id": monitor_id,
            "running": mon.exit_code is None,
            "exit_code": mon.exit_code,
            "lines": [{"text": text, "stream": "stderr" if is_err else "stdout"} for _ts, text, is_err in slice_],
            "more_buffered": mon.cursor < len(mon.lines),
        }

    async def stop(self, monitor_id: str, *, grace_s: float = 2.0) -> dict[str, Any]:
        """SIGTERM the monitor's process group, escalate to SIGKILL.

        Returns the same shape as ``read`` so the caller sees the
        terminal exit code in one round-trip.

        Cross-subprocess path: when ``monitor_id`` isn't in the
        in-process registry (because monitor_start ran in a previous
        tool subprocess), look it up in the sidecar and SIGTERM the
        bare PID directly. The exit-code observation can only happen
        in the original subprocess that owned the proc handle, so the
        cross-subprocess stop returns ``exit_code=None``.
        """
        mon = self._monitors.get(monitor_id)
        if mon is None:
            return await self._stop_by_sidecar(monitor_id, grace_s=grace_s)
        already_exited = mon.exit_code is not None
        await self._terminate(mon, grace_s=grace_s)
        # Drain any tail output.
        for t in mon.drain_tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(t, timeout=1.0)
        mon.closed = True
        log_event(
            "MONITOR_STOP",
            monitor_id=monitor_id,
            already_exited=already_exited,
            exit_code=mon.exit_code,
            unread=max(0, len(mon.lines) - mon.cursor),
        )
        remove_sidecar_entry(monitor_id)
        return {
            "monitor_id": monitor_id,
            "running": False,
            "exit_code": mon.exit_code,
            "buffered_lines": len(mon.lines),
            "unread_lines": max(0, len(mon.lines) - mon.cursor),
        }

    async def _stop_by_sidecar(
        self,
        monitor_id: str,
        *,
        grace_s: float,
    ) -> dict[str, Any]:
        """SIGTERM by PID alone — the proc handle lives in another
        (now-gone) subprocess. Same SIGTERM→SIGKILL escalation."""
        entries = load_sidecar()
        match = next(
            (e for e in entries if e.get("monitor_id") == monitor_id),
            None,
        )
        if match is None:
            raise LookupError(f"unknown monitor_id {monitor_id!r}")
        pid = int(match.get("pid") or 0)
        if pid <= 0:
            remove_sidecar_entry(monitor_id)
            raise LookupError(f"monitor {monitor_id!r} has no recorded pid")
        was_alive = _is_pid_alive(pid)
        if was_alive:
            from sharedagents_agent import sandbox_broker as _sb

            if not _sb.try_capability_kill_pgid(pid, signal.SIGTERM):
                with contextlib.suppress(ProcessLookupError):
                    oscompat.kill_process(pid, force=False, group=True)
            # Bounded SIGTERM grace, then SIGKILL.
            deadline = time.monotonic() + grace_s
            while time.monotonic() < deadline and _is_pid_alive(pid):
                await asyncio.sleep(0.1)
            if _is_pid_alive(pid):
                log_event(
                    "MONITOR_SIGKILL_ESCALATE",
                    monitor_id=monitor_id,
                    pid=pid,
                    grace_s=grace_s,
                    cross_subprocess=True,
                )
                if not _sb.try_capability_kill_pgid(pid, oscompat.FORCE_KILL):
                    with contextlib.suppress(ProcessLookupError):
                        oscompat.kill_process(pid, force=True, group=True)
            _sb.release_pgid(pid)
        log_event(
            "MONITOR_STOP",
            monitor_id=monitor_id,
            cross_subprocess=True,
            was_alive=was_alive,
            pid=pid,
        )
        remove_sidecar_entry(monitor_id)
        return {
            "monitor_id": monitor_id,
            "running": False,
            "exit_code": None,
            "buffered_lines": 0,
            "unread_lines": 0,
        }

    async def _terminate(self, mon: _Monitor, *, grace_s: float) -> None:
        proc = mon.process
        if proc.returncode is not None:
            return
        from sharedagents_agent import sandbox_broker as _sb

        _pgid = oscompat.process_group(proc.pid)
        if not _sb.try_capability_kill_pgid(_pgid, signal.SIGTERM):
            with contextlib.suppress(ProcessLookupError):
                oscompat.kill_process(_pgid, force=False, group=True)
        # Popen.wait is sync; await it on a thread with a timeout
        # poll. We use small steps so cancellation is responsive.
        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return
            await asyncio.sleep(0.05)
        log_event(
            "MONITOR_SIGKILL_ESCALATE",
            monitor_id=mon.monitor_id,
            pid=proc.pid,
            grace_s=grace_s,
        )
        if not _sb.try_capability_kill_pgid(_pgid, oscompat.FORCE_KILL):
            with contextlib.suppress(ProcessLookupError):
                oscompat.kill_process(_pgid, force=True, group=True)
        _sb.release_pgid(_pgid)
        # Final 2 s grace for SIGKILL to land.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return
            await asyncio.sleep(0.05)

    async def shutdown(self) -> None:
        """Reap every live monitor. Called by the session-close hook
        (gateway cascade) so backgrounded watches never outlive their
        owner."""
        ids = list(self._monitors)
        if ids:
            log_event("MONITOR_SHUTDOWN_ALL", count=len(ids), monitors=list(ids))
        for mid in ids:
            with contextlib.suppress(Exception):
                await self.stop(mid, grace_s=1.0)
        self._monitors.clear()


# ---- runtime resolver -----------------------------------------------

# A single per-process registry. Sessions get their own MonitorRegistry
# in a richer integration (gateway-aware), but for the v1 pull-only
# contract a single registry per worker is enough — every tool round
# runs in the same worker subprocess, so monitor IDs are stable for
# the duration of a turn. Cross-turn persistence is a follow-up.
_GLOBAL_REGISTRY: MonitorRegistry | None = None


def _registry() -> MonitorRegistry:
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = MonitorRegistry()
    return _GLOBAL_REGISTRY


__all__ = [
    "MonitorRegistry",
    "_Monitor",
    "_registry",
    "monitors_sidecar_path",
    "load_sidecar",
    "write_sidecar_entry",
    "remove_sidecar_entry",
    "live_monitors_for_session",
]


# ---- session-scoped sidecar persistence -----------------------------
#
# v1 in-memory registry can't survive across tool subprocess
# boundaries (every tool call spawns a fresh subprocess; the next
# `monitor_read` would see an empty registry). The sidecar is the
# cross-process truth: every monitor_start writes an entry, every
# monitor_stop / process exit removes it, and the TUI / monitor_list
# read this file directly. The actual subprocess keeps running because
# it was spawned with ``start_new_session=True`` and was reparented to
# init when the spawning tool subprocess exited — so the PID survives
# even when the registry doesn't. The sidecar carries enough state
# (pid, command, started_at) for an external observer to reconstruct
# what's alive without needing the original registry object.
#
# File path: ``<SHAREDAGENTS_SESSIONS_DIR>/<SHAREDAGENTS_SESSION_ID>.monitors.json``
# Same naming convention as ``.todos.json`` and
# ``.delegate_agent.<target>.json``.
#
# Windows serializes the read/modify/replace through oscompat's lockfile.
# POSIX retains its existing atomic-replace behavior (no flock was present).


def _monitor_logfile_for(monitor_id: str) -> Path | None:
    """Per-monitor logfile path, sibling of the monitors sidecar.

    Format: ``<sessions_dir>/<sid>.monitor-<mid>.log``. Same naming
    convention as ``.todos.json`` / ``.delegate_agent.<target>.json``
    so a session-level cleanup can wipe everything with one glob.
    Returns ``None`` when env doesn't carry a session id.
    """
    sid = os.environ.get("SHAREDAGENTS_SESSION_ID")
    sdir = os.environ.get("SHAREDAGENTS_SESSIONS_DIR")
    if not sid or not sdir:
        return None
    return Path(sdir) / f"{sid}.monitor-{monitor_id}.log"


def read_log_lines(
    log_path: Path,
    *,
    start_byte: int,
    max_lines: int,
) -> tuple[list[tuple[str, bool]], int]:
    """Read up to ``max_lines`` from ``log_path`` starting at byte
    offset ``start_byte``. Returns ``(lines, new_cursor)`` where each
    line is ``(text, is_stderr)`` — but stderr is merged into the
    same file in v1, so ``is_stderr`` is always False here.

    The new cursor points at the first byte AFTER the last full line
    we returned, so the next call resumes cleanly even if the
    monitored process is mid-write.
    """
    if not log_path.exists():
        return [], start_byte
    try:
        size = log_path.stat().st_size
        if size <= start_byte:
            return [], start_byte
        with log_path.open("rb") as f:
            f.seek(start_byte)
            chunk = f.read(min(size - start_byte, 4 * 1024 * 1024))
    except OSError:
        return [], start_byte
    text = chunk.decode("utf-8", "replace")
    last_nl = text.rfind("\n")
    if last_nl == -1:
        # No full line yet — leave cursor unchanged.
        return [], start_byte
    full = text[: last_nl + 1]
    consumed_bytes = len(full.encode("utf-8"))
    new_cursor = start_byte + consumed_bytes
    line_strs = full.splitlines(keepends=True)[:max_lines]
    # If we didn't consume all lines (max_lines limited us), back the
    # cursor up to just after the last line we kept.
    if len(line_strs) < len(full.splitlines(keepends=True)):
        kept_bytes = sum(len(s.encode("utf-8")) for s in line_strs)
        new_cursor = start_byte + kept_bytes
    return [(s, False) for s in line_strs], new_cursor


def monitors_sidecar_path() -> Path | None:
    """Resolve `<sessions_dir>/<sid>.monitors.json` from the env.

    Returns `None` when the env doesn't carry a session id (e.g.
    ad-hoc invocation outside a real agent run). Callers must
    silently no-op so a missing sidecar never breaks a tool call.
    """
    sid = os.environ.get("SHAREDAGENTS_SESSION_ID")
    sdir = os.environ.get("SHAREDAGENTS_SESSIONS_DIR")
    if not sid or not sdir:
        return None
    return Path(sdir) / f"{sid}.monitors.json"


def _is_pid_alive(pid: int) -> bool:
    return oscompat.process_alive(pid, permission_denied_alive=False)


def load_sidecar(path: Path | None = None) -> list[dict[str, Any]]:
    """Read the sidecar; corrupt/missing → `[]`. Filters dead PIDs out."""
    if path is None:
        path = monitors_sidecar_path()
    if path is None or not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        pid = entry.get("pid")
        entry = {**entry, "alive": True} if isinstance(pid, int) and _is_pid_alive(pid) else {**entry, "alive": False}
        out.append(entry)
    return out


def _save_sidecar(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(entries, indent=2, ensure_ascii=False) + "\n"
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


@contextlib.contextmanager
def _sidecar_lock(path: Path):
    if oscompat.IS_WINDOWS:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix(path.suffix + ".lock").open("a+b") as fp, oscompat.file_lock(fp):
            yield
    else:
        yield


def write_sidecar_entry(entry: dict[str, Any]) -> None:
    """Upsert one monitor's row in the sidecar (matched by `monitor_id`)."""
    path = monitors_sidecar_path()
    if path is None:
        return
    try:
        with _sidecar_lock(path):
            try:
                existing = load_sidecar(path)
            except Exception:  # noqa: BLE001
                existing = []
            mid = entry.get("monitor_id")
            out = [e for e in existing if e.get("monitor_id") != mid]
            out.append({k: v for k, v in entry.items() if k != "alive"})
            _save_sidecar(path, out)
    except OSError:
        pass


def remove_sidecar_entry(monitor_id: str) -> None:
    path = monitors_sidecar_path()
    if path is None:
        return
    try:
        with _sidecar_lock(path):
            try:
                existing = load_sidecar(path)
            except Exception:  # noqa: BLE001
                return
            out = [{k: v for k, v in e.items() if k != "alive"} for e in existing if e.get("monitor_id") != monitor_id]
            _save_sidecar(path, out)
    except OSError:
        pass


def live_monitors_for_session(
    sessions_dir: Path,
    session_id: str,
) -> list[dict[str, Any]]:
    """TUI / monitor_list helper. Read sidecar for one session, return
    entries with a fresh `alive` flag.

    Differs from :func:`load_sidecar` only in that it accepts an
    explicit (sessions_dir, session_id) instead of reading the env —
    so the TUI can call it with the active session id without first
    materialising the same env vars the runner sets.
    """
    return load_sidecar(sessions_dir / f"{session_id}.monitors.json")
