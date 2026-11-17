"""Durable engine-subprocess PID registry — the state-independent abort backstop.

A delegated CLI engine subprocess (``claude -p`` / ``codex exec`` /
antigravity) is spawned with ``start_new_session=True``, so it LEADS its own
process group and SURVIVES the death of the orchestrator that spawned it
(orphaning to PID 1). When the orchestrator restarts, every in-memory
active-turn / active-proc map is gone — so a Stop / ``/abort`` for that session
has nothing in memory to kill and the orphaned engine keeps running, immune to
the operator's abort. That is the one remaining hole in the "kill any agent
instantly, at any time" guarantee.

This module persists the engine subprocess's ``pid``/``pgid`` to a small
per-session sidecar so a HARD abort can SIGKILL the whole process tree using
ONLY the session id, with zero in-memory state, even across a restart.

On POSIX, a PID-reuse guard (the process's ``/proc`` start-time) prevents
signalling a recycled PID. Windows can only check liveness and terminate the
saved immediate child: PID reuse and surviving grandchildren remain risks.
Bookkeeping is best-effort and must not fail the operator's abort.
"""

from __future__ import annotations

import json
import os
import signal
from pathlib import Path
from typing import Any, Union

from sharedagents_core import oscompat

_PathLike = Union[str, "os.PathLike[str]"]


def _pidfile(sessions_dir: _PathLike, sid: str) -> Path:
    return Path(sessions_dir) / f"{sid}.engine_pid.json"


def _proc_starttime(pid: int) -> int | None:
    """Field 22 (``starttime``, clock ticks since boot) of ``/proc/<pid>/stat``.

    ``None`` when the pid is gone or unreadable. The ``comm`` field (2) is
    paren-wrapped and may itself contain spaces / parens, so we split only
    AFTER the last ``)`` — fields then are ``state(3) ppid(4) … starttime(22)``,
    i.e. ``starttime`` sits at index ``22 - 3 = 19``.
    """
    if oscompat.IS_WINDOWS:
        return None  # No /proc birth identity on Windows.
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    rparen = data.rfind(b")")
    if rparen < 0:
        return None
    after = data[rparen + 2 :].split()
    try:
        return int(after[19])
    except (IndexError, ValueError):
        return None


def record(
    sessions_dir: _PathLike,
    sid: str,
    pid: int,
    *,
    cli: str = "",
    pgid: Any = "auto",
) -> None:
    """Persist the engine subprocess's pid/pgid/start-time. Best-effort.

    ``pgid`` controls the process-group recorded for the killpg sweep in
    :func:`kill_session_engine`:

    * ``"auto"`` (default) — resolve via ``os.getpgid(pid)``; correct for a
      session-LEADER process (spawned ``start_new_session=True``).
    * ``None`` — record NO pgid, so the kill sweep skips ``killpg`` and reaps
      the subtree purely via the precise ``/proc`` descendant walk. Use this
      for a process that SHARES its parent's group (e.g. a subagent
      ``tool_runner`` spawned WITHOUT a new session) where a ``killpg`` would
      over-reach into sibling / ancestor work.
    * an ``int`` — record that pgid verbatim.
    """
    try:
        if pgid == "auto":
            try:
                pgid_val: int | None = oscompat.process_group(pid)
            except OSError:
                pgid_val = pid
        elif pgid is None:
            pgid_val = None
        else:
            pgid_val = int(pgid)
        payload = {
            "pid": int(pid),
            "pgid": (int(pgid_val) if pgid_val is not None else None),
            "starttime": _proc_starttime(pid),
            "cli": cli,
        }
        p = _pidfile(sessions_dir, sid)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, p)
    except Exception:  # noqa: BLE001 — sidecar is best-effort observability
        pass


def clear(sessions_dir: _PathLike, sid: str) -> None:
    """Remove the sidecar. Best-effort, idempotent."""
    try:
        _pidfile(sessions_dir, sid).unlink()
    except OSError:
        pass


def is_alive(sessions_dir: _PathLike, sid: str) -> bool:
    """True for a live sidecar process (identity-verified only on POSIX).

    Same identity guard as :func:`kill_session_engine` (recorded ``starttime``
    must match the live pid's) but READ-ONLY: a recycled pid is never treated
    as live. A missing/corrupt sidecar or a dead process reads ``False``.
    Used by the stale-live reconcile sweep to prove a session's engine
    subprocess is really gone before folding the row terminal.
    """
    p = _pidfile(sessions_dir, sid)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    pid = data.get("pid")
    if not isinstance(pid, int):
        return False
    if oscompat.IS_WINDOWS:
        # Best effort: liveness only; a recycled Windows PID is not identified.
        return oscompat.process_alive(pid)
    live_start = _proc_starttime(pid)
    if live_start is None:
        return False
    rec_start = data.get("starttime")
    # Missing starttime is not a verified identity — a recycled pid must
    # never count as a live engine.
    if rec_start is None:
        return False
    return int(rec_start) == int(live_start)


def kill_session_engine(sessions_dir: _PathLike, sid: str) -> bool:
    """SIGKILL the recorded engine subprocess AND its whole descendant tree,
    using ONLY the durable sidecar (no in-memory state).

    Returns ``True`` iff a live process was signalled; Windows only verifies
    liveness and terminates the immediate child.
    Best-effort; never raises. Removes the sidecar on every terminal outcome
    (killed, already-dead, or pid-recycled).

    Identity guard: the recorded start-time must still match the live pid's
    start-time. A recycled pid is therefore never killed — the function bails
    out BEFORE sending any signal.
    """
    p = _pidfile(sessions_dir, sid)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    pid = data.get("pid")
    if not isinstance(pid, int):
        clear(sessions_dir, sid)
        return False

    if oscompat.IS_WINDOWS:
        # Best effort: saved child only, with no birth identity or tree sweep.
        if not oscompat.process_alive(pid):
            clear(sessions_dir, sid)
            return False
        try:
            oscompat.kill_process(pid, force=True)
        except OSError:
            return False
        clear(sessions_dir, sid)
        return True

    live_start = _proc_starttime(pid)
    if live_start is None:
        # Process already gone — stale sidecar.
        clear(sessions_dir, sid)
        return False
    rec_start = data.get("starttime")
    if rec_start is None or int(rec_start) != int(live_start):
        # No identity, or PID recycled — DO NOT signal it.
        clear(sessions_dir, sid)
        return False

    # Descendants FIRST (a killed parent can't reparent them past the sweep),
    # then the process group, then the leader. Mirrors runners._kill_tree but
    # works from a bare pid (no live ``Process`` object).
    try:
        from sharedagents_agent.tools.runners import _descendant_pids

        descendants = _descendant_pids(int(pid))
    except Exception:  # noqa: BLE001
        descendants = []
    for dpid in descendants:
        try:
            os.kill(dpid, signal.SIGKILL)
        except OSError:
            pass
    pgid = data.get("pgid")
    if isinstance(pgid, int):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            pass
    try:
        os.kill(int(pid), signal.SIGKILL)
    except OSError:
        pass
    clear(sessions_dir, sid)
    return True


__all__ = ["record", "clear", "is_alive", "kill_session_engine"]
