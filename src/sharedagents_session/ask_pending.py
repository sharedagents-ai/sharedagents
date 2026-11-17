"""Detect unanswered operator interaction parked on a session JSONL.

A ``question`` / ``AskUserQuestion`` / ``ask_user`` / ``UserQuestion``
``tool_call`` whose structural ``tool_result_meta`` has not been written
yet means the turn is waiting on a human. A ``permission_request``
without its matching ``permission_resolved`` is the same. Neither has a
wall-clock expiry — the operator may answer the next day. Used by the
Tasks board, stale-live sweeper, and web activity bit.

The log is the durable source of truth: a parked interaction writes
nothing further, so ``session_state_updated_at`` freezes and the 300s
freshness gate would otherwise treat the runner as a ghost.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

_PathLike = str | os.PathLike[str]


def is_ask_tool_name(name: Any) -> bool:
    """True for any engine's spelling of the ask-user tool.

    One matcher for the whole product: ``sharedagents_agent.tools.widget_adapt``
    owns the alias table, so a foreign spelling (``AskUserQuestion``,
    ``askuser``, ``ask_user_question``) parks and detects exactly like the
    native ``question``. Imported lazily because this package sits below
    ``sharedagents_agent`` in the import graph.
    """
    from sharedagents_agent.tools.widget_adapt import is_ask_user_tool

    return is_ask_user_tool(name)


#: How much of the session JSONL tail to scan. An interaction and its
#: structural result are adjacent events, so a bounded tail read of the
#: file END is enough while the turn is parked.
ASK_LOG_TAIL_BYTES = 64 * 1024

#: Kept for callers / tests. 0 = no age window (the live default).
#: A positive value is only used when a caller passes ``timeout_s``.
DEFAULT_ASK_USER_TIMEOUT_S = 0.0

#: Shape of a real session id/hash. Anything else reaching a path join
#: would fabricate a bogus JSONL path.
_SID_RE = re.compile(r"[0-9A-Za-z_-]{6,128}")


def ask_user_timeout_s() -> float:
    raw = os.environ.get("SHAREDAGENTS_WEB_ASK_USER_TIMEOUT_S")
    try:
        return float(raw) if raw else 0.0
    except (TypeError, ValueError):
        return 0.0


def session_log_pending_interaction(
    path: _PathLike,
    *,
    timeout_s: float | None = None,
) -> str:
    """Return ``"question"`` / ``"permission"`` when input is pending.

    Parses the canonical on-disk codec via ``decode_line`` (every line is
    ``<8-hex CRC><space><compact JSON>``). The file is opened once,
    ``fstat``-ed, and read only through that snapshot's EOF.

    ``turn_ended`` / ``turn_started`` clear the pending signal: a later
    turn or a closer means the interaction is no longer parked.
    Questions and permissions wait for an explicit resolution or
    lifecycle boundary and never expire by age. ``timeout_s`` is ignored
    unless a caller passes a positive value (test-only age window).
    """
    from sharedagents_session.session_reader import decode_line

    p = Path(path)
    if not p.name.endswith(".jsonl"):
        return ""
    window = 0.0 if timeout_s is None else float(timeout_s)
    try:
        with p.open("rb") as stream:
            st = os.fstat(stream.fileno())
            age_s = time.time() - st.st_mtime

            size = st.st_size
            start = max(0, size - ASK_LOG_TAIL_BYTES)
            starts_on_line_boundary = start == 0
            if start:
                stream.seek(start - 1)
                starts_on_line_boundary = stream.read(1) == b"\n"
            stream.seek(start)
            raw_tail = stream.read(size - start)
    except OSError:
        return ""
    if not raw_tail:
        return ""

    if not starts_on_line_boundary:
        first_lf = raw_tail.find(b"\n")
        if first_lf < 0:
            return ""
        raw_tail = raw_tail[first_lf + 1 :]

    if raw_tail and not raw_tail.endswith(b"\n"):
        last_lf = raw_tail.rfind(b"\n")
        if last_lf < 0:
            return ""
        raw_tail = raw_tail[: last_lf + 1]

    pending_call_id = ""
    pending_permission_ids: set[str] = set()
    for raw_line in raw_tail.split(b"\n"):
        ev = decode_line(raw_line)
        if ev is None:
            continue
        etype = ev.get("type")
        if etype == "tool_call":
            pending_call_id = str(ev.get("call_id") or "") if is_ask_tool_name(ev.get("name")) else ""
        elif etype == "tool_result_meta":
            if pending_call_id and str(ev.get("call_id") or "") == pending_call_id:
                pending_call_id = ""
        elif etype == "permission_request":
            request_id = str(ev.get("request_id") or "")
            if request_id:
                pending_permission_ids.add(request_id)
        elif etype == "permission_resolved":
            pending_permission_ids.discard(str(ev.get("request_id") or ""))
        elif etype in ("turn_ended", "turn_started"):
            pending_call_id = ""
            pending_permission_ids.clear()
    if pending_permission_ids:
        return "permission"
    if pending_call_id:
        if window > 0 and age_s > window:
            return ""
        return "question"
    return ""


def session_log_pending_ask(
    path: _PathLike,
    *,
    timeout_s: float | None = None,
) -> bool:
    """True when this JSONL tail ends on an unanswered ask-family call."""
    return session_log_pending_interaction(path, timeout_s=timeout_s) == "question"


def session_log_pending_permission(path: _PathLike) -> bool:
    """True when a permission request awaits an explicit resolution."""
    return session_log_pending_interaction(path) == "permission"


def session_pending_interaction(sessions_dir: _PathLike, sid: str) -> str:
    """Return the pending interaction kind for ``sid``, or ``""``."""
    if not sid or not _SID_RE.fullmatch(sid):
        return ""
    return session_log_pending_interaction(Path(sessions_dir) / f"{sid}.jsonl")


def session_pending_ask(sessions_dir: _PathLike, sid: str) -> bool:
    """True when ``<sessions_dir>/<sid>.jsonl`` is parked on an ask."""
    return session_pending_interaction(sessions_dir, sid) == "question"


def session_pending_permission(sessions_dir: _PathLike, sid: str) -> bool:
    """True when ``<sessions_dir>/<sid>.jsonl`` awaits permission."""
    return session_pending_interaction(sessions_dir, sid) == "permission"
