"""Durable record of an operator Stop on a session log.

A Stop pressed by the user appends an ``operator_stop`` event to the session's
JSONL log. The record stands until the session's next user ``turn_started``: from
the moment of the Stop until the user prompts again, nothing automatic may
start a turn on the session — no background result, recovery sweep, retry
attempt, scheduled job or memory-flush turn.

Two modes:

* ``hard`` — ``/abort`` / the web Stop button. The whole session tree is
  stopped; background results from its children are dropped.
* ``graceful`` — ``/stop`` / the first ESC. The in-flight round finishes and
  delegated children may still deliver their results, but no maintenance or
  scheduled turn starts and a failed attempt is not retried.

The hard-Stop cascade's ``subagent_result`` records (``cancelled (parent
abort)``) on a parent log count as a hard stop too, so logs written before
this record existed keep their meaning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

OPERATOR_STOP_EVENT = "operator_stop"
STOP_MODE_HARD = "hard"
STOP_MODE_GRACEFUL = "graceful"

#: Error text ``async_jobs.silence_deliveries`` records on the parent log for
#: every background job a hard Stop cancelled.
PARENT_ABORT_RESULT_MARK = "parent abort"


def record_operator_stop(
    sessions_dir: Any,
    sid: str,
    *,
    mode: str = STOP_MODE_HARD,
) -> bool:
    """Append an ``operator_stop`` record to ``sid``'s log.

    Write it BEFORE killing anything: a process that sees its engine die
    reads this record to tell a user Stop from an external kill. Returns
    whether a record was written. Never raises.
    """
    if not sid:
        return False
    try:
        root = Path(sessions_dir)
        if not (root / f"{sid}.jsonl").exists():
            return False
        from sharedagents_session.session_writer import get_session_writer

        get_session_writer(root, sid).lifecycle(
            kind=OPERATOR_STOP_EVENT,
            payload={"mode": mode if mode == STOP_MODE_GRACEFUL else STOP_MODE_HARD},
        )
        return True
    except Exception:  # noqa: BLE001 — the Stop itself must not fail
        return False


def _stop_state(sessions_dir: Any, sid: str) -> tuple[str, int]:
    """The Stop standing on ``sid``: ``"hard"``, ``"graceful"`` or ``""``.

    A Stop stands when it is newer than the session's last user turn.
    The token tracks the latest stop even after a user resumes.
    A hard Stop outranks a graceful one recorded since the same turn. An
    unreadable log reads as no Stop (never blocks work by accident).
    """
    if not sid:
        return "", -1
    from sharedagents_session.session_reader import tail_jsonl

    last_turn = -1
    last_hard = -1
    last_graceful = -1
    try:
        for ev, off in tail_jsonl(Path(sessions_dir) / f"{sid}.jsonl", 0):
            etype = ev.get("type")
            if etype == "turn_started" and str(ev.get("source") or "user") == "user":
                last_turn = off
            elif etype == OPERATOR_STOP_EVENT:
                if str(ev.get("mode") or "") == STOP_MODE_GRACEFUL:
                    last_graceful = off
                else:
                    last_hard = off
            elif etype == "subagent_result":
                payload = ev.get("payload")
                err = (payload if isinstance(payload, dict) else ev).get("error")
                if PARENT_ABORT_RESULT_MARK in str(err or "").lower():
                    last_hard = off
    except Exception:  # noqa: BLE001
        return "", -1
    token = max(last_hard, last_graceful)
    if last_hard > last_turn:
        return STOP_MODE_HARD, token
    if last_graceful > last_turn:
        return STOP_MODE_GRACEFUL, token
    return "", token


def operator_stop_mode(sessions_dir: Any, sid: str) -> str:
    """The Stop standing since the last user turn; automatic turns never lift it."""
    return _stop_state(sessions_dir, sid)[0]


class OperatorStopped(Exception):
    """A Stop won the race with turn admission."""


@dataclass(frozen=True)
class TurnStopGuard:
    """A turn may resume an existing Stop, but cannot overtake a newer one.

    Capture before waiting for a session lock or composing a preamble. Check
    again under the session writer's flock when admitting ``turn_started``;
    a Stop from another process and admission then have one durable order.
    """

    token: int
    source: str
    retry: bool = False

    @classmethod
    def capture(cls, sessions_dir: Any, sid: str, source: str) -> TurnStopGuard:
        return cls(_stop_state(sessions_dir, sid)[1], source or "user")

    def check(self, sessions_dir: Any, sid: str) -> None:
        mode, token = _stop_state(sessions_dir, sid)
        may_resume = self.source == "user" and not self.retry
        may_deliver = mode == STOP_MODE_GRACEFUL and self.source == "background_task" and not self.retry
        if token != self.token or (mode and not may_resume and not may_deliver):
            raise OperatorStopped("session stopped before turn admission")


def operator_stopped(sessions_dir: Any, sid: str) -> bool:
    """Whether any user Stop (hard or graceful) stands on ``sid``."""
    return bool(operator_stop_mode(sessions_dir, sid))
