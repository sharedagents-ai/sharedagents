"""Which process wrote a turn, and is it still running?

Every ``turn_started`` event carries a ``runner`` stamp: the writer's pid and
the kernel start time of that pid (``/proc/<pid>/stat`` field 22, clock ticks
since boot). Within one kernel boot and PID namespace, the pair guards
against a recycled pid being read as the original writer.

Why this exists: two orchestrators can share one sessions directory. The
boot sweep used to close EVERY session in a live state, so each boot of one
stack aborted the other stack's running turns with ``orchestrator_restart``.
With the stamp, a sweep closes a turn only when the process that opened it
is gone.

Never raises. A turn with no stamp (older code, foreign writer) reads as
"unknown", and callers keep their previous behavior for it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Union

_PathLike = Union[str, "os.PathLike[str]"]


def proc_starttime(pid: int) -> int | None:
    """``starttime`` of a running ``pid``, or ``None`` if unavailable or dead.

    ``comm`` (field 2) may contain spaces and parentheses, so fields are
    split after the LAST ``)``; ``starttime`` is then at index 19.
    """
    try:
        with open(f"/proc/{int(pid)}/stat", "rb") as fh:
            raw = fh.read()
        tail = raw[raw.rindex(b")") + 2 :].split()
        # An unreaped process keeps its pid and start time, but cannot finish
        # a turn. Treat zombie/dead states as gone rather than pinning it live.
        if tail[0] in {b"Z", b"X", b"x"}:
            return None
        return int(tail[19])
    except (OSError, ValueError, IndexError):
        return None


_STAMP_CACHE: dict[int, dict[str, Any]] = {}


def runner_stamp() -> dict[str, Any]:
    """The stamp for the current process (cached per pid, fork-safe)."""
    pid = os.getpid()
    stamp = _STAMP_CACHE.get(pid)
    if stamp is None:
        stamp = {"pid": pid, "start": proc_starttime(pid)}
        _STAMP_CACHE.clear()
        _STAMP_CACHE[pid] = stamp
    return dict(stamp)


def runner_alive(stamp: Any) -> bool | None:
    """``True`` when the stamped process still runs, ``False`` when it is gone.

    ``None`` when the stamp is missing or malformed, or when the start time
    cannot be compared (no ``/proc``): liveness is unknown.
    """
    if not isinstance(stamp, dict):
        return None
    try:
        pid = int(stamp.get("pid"))
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return None
    recorded = stamp.get("start")
    live = proc_starttime(pid)
    if recorded is None:
        return None
    if live is None:
        # Recorded on a host with /proc, now unreadable: the pid is gone
        # (or /proc vanished, in which case nothing can be proven alive).
        return False if os.path.isdir("/proc") else None
    try:
        return int(recorded) == live
    except (TypeError, ValueError):
        return None


def turn_runner(jsonl: _PathLike, turn_idx: int) -> dict[str, Any] | None:
    """The ``runner`` stamp of the newest open turn, if it is ``turn_idx``.

    ``None`` when the log is unreadable, the turn is not found, or the
    event carries no stamp. Walk backwards only to the latest turn boundary,
    so old conversation history is not rescanned on every sweep.
    """
    from sharedagents_session.session_reader import decode_line, reverse_jsonl

    try:
        for ev, start, end in reverse_jsonl(Path(jsonl)):
            kind = ev.get("type")
            if kind not in {"turn_started", "turn_ended"}:
                continue
            if int(ev.get("turn_idx")) != int(turn_idx):
                if kind == "turn_started":
                    return None
                # An older turn can close late while a newer turn is open;
                # its terminal event says nothing about this turn's owner.
                continue
            if kind == "turn_ended":
                return None
            if ev.get("_content_truncated"):
                # Oversized prompts (e.g. inline images) are projected without
                # runner by reverse_jsonl. Decode just this boundary in full.
                with Path(jsonl).open("rb") as fp:
                    fp.seek(start)
                    ev = decode_line(fp.read(end - start).rstrip(b"\n")) or {}
            runner = ev.get("runner")
            return runner if isinstance(runner, dict) else None
    except (OSError, TypeError, ValueError, OverflowError):
        return None
    return None


def turn_runner_alive_elsewhere(jsonl: _PathLike, turn_idx: int) -> bool:
    """True only when the turn's writer is a DIFFERENT process that still runs.

    That process (another orchestrator, a detached delegate runner) owns the
    turn and will close it itself, so a sweep in this process must leave it.
    """
    stamp = turn_runner(jsonl, turn_idx)
    if runner_alive(stamp) is not True:
        return False
    return int(stamp.get("pid")) != os.getpid()
