"""Shared pieces of a user Stop, used by every engine's cancel path.

The CLI-engine path (``gateway_shim.AgentACPBackend.cancel``) and the
local-LLM path (``acp_bridge.CoreBackend.cancel``) both call these, so a Stop
behaves the same whichever engine drives the session:

* :func:`begin_stop` records the Stop on the session (and, for a hard Stop,
  on every descendant) BEFORE anything is killed, and cancels the pending
  memory-flush timers.
* :func:`stop_descendants` kills each descendant's engine and folds its busy
  row out of the live set.
* :func:`purge_queues` drops the follow-up queues so nothing queued starts a
  turn after the Stop.

Every function is best-effort and never raises: bookkeeping must not fail the
user's Stop.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from sharedagents_session.operator_stop import (
    STOP_MODE_GRACEFUL,
    STOP_MODE_HARD,
    record_operator_stop,
)


def descendant_sessions(sessions_dir: Any, session_id: str) -> list[str]:
    """Every session below ``session_id`` in ``sessions_meta``, any state."""
    out: list[str] = []
    try:
        from sharedagents_session.session_index import connect_sessions_db

        db = Path(sessions_dir) / "sessions.db"
        if not session_id or not db.exists():
            return out
        seen = {session_id}
        con = connect_sessions_db(db)
        try:
            frontier = [session_id]
            while frontier:
                curr = frontier.pop()
                rows = con.execute(
                    "SELECT session_hash FROM sessions_meta WHERE parent_session_id=?",
                    (curr,),
                ).fetchall()
                for (child,) in rows:
                    if child and child not in seen:
                        seen.add(child)
                        out.append(child)
                        frontier.append(child)
        finally:
            con.close()
    except Exception:  # noqa: BLE001
        pass
    return out


def begin_stop(
    agent: Any,
    sessions_dir: Any,
    session_id: str,
    *,
    force: bool,
) -> list[str]:
    """Record a user Stop before any process is killed.

    A hard Stop records on the session and all its descendants; a graceful
    one only on the session (its children keep running). Returns the
    descendants found. Also cancels the memory-flush timers of every stopped
    session so no maintenance turn starts on them.
    """
    descendants = descendant_sessions(sessions_dir, session_id) if force else []
    mode = STOP_MODE_HARD if force else STOP_MODE_GRACEFUL
    record_operator_stop(sessions_dir, session_id, mode=mode)
    for sid in descendants:
        record_operator_stop(sessions_dir, sid, mode=STOP_MODE_HARD)
    cancel_memory_flush(agent, [session_id, *descendants])
    pause_automatic_work(agent, descendants)
    return descendants


def pause_automatic_work(agent: Any, session_ids: Iterable[str]) -> None:
    """Pause durable jobs/goals owned by these stopped sessions."""
    from sharedagents_goal.goal_service import stop_goals_for_session
    from sharedagents_loop.loop_service import stop_loops_for_session

    for sid in session_ids:
        for stop in (stop_goals_for_session, stop_loops_for_session):
            try:
                stop(agent, sid, reason="aborted")
            except Exception:  # noqa: BLE001
                pass


def cancel_memory_flush(agent: Any, session_ids: Iterable[str]) -> None:
    cancel = getattr(agent, "_cancel_pending_flush", None)
    if not callable(cancel):
        return
    for sid in session_ids:
        try:
            cancel(sid)
        except Exception:  # noqa: BLE001
            pass


def stop_descendants(
    sessions_dir: Any,
    session_id: str,
    known: Iterable[str] = (),
    *,
    agent: Any = None,
) -> list[str]:
    """Kill each descendant's engine and fold each still-busy row.

    Re-walks the tree so a child linked after :func:`begin_stop` is stopped
    too (and gets its Stop record before its engine is killed). Finished
    descendants are left as they are. Returns every descendant handled.
    """
    known_list = list(known)
    current = descendant_sessions(sessions_dir, session_id)
    for sid in current:
        if sid not in known_list:
            record_operator_stop(sessions_dir, sid, mode=STOP_MODE_HARD)
    every = list(dict.fromkeys([*known_list, *current]))
    if agent is not None:
        cancel_memory_flush(agent, every)
        pause_automatic_work(agent, every)
    try:
        from sharedagents_agent.agents import engine_pidfile
        from sharedagents_gateway.control_endpoint import _abort_child_session_write
    except Exception:  # noqa: BLE001
        return every
    for sid in every:
        try:
            engine_pidfile.kill_session_engine(Path(sessions_dir), sid)
        except Exception:  # noqa: BLE001
            pass
        try:
            _abort_child_session_write(sid)
        except Exception:  # noqa: BLE001
            pass
    return every


def purge_queues(agent: Any, session_ids: Iterable[str]) -> None:
    """Drop every pending follow-up item of the stopped sessions.

    ``background_purged`` retires a background result for good (the receipt
    sweep skips it); anything else is removed as ``user_deleted``.
    """
    from sharedagents_session.queue_remove import (
        QUEUE_REMOVE_BACKGROUND_PURGED,
        QUEUE_REMOVE_USER_DELETED,
    )

    queue_list = getattr(agent, "queue_list", None)
    queue_remove = getattr(agent, "queue_remove", None)
    if not callable(queue_list) or not callable(queue_remove):
        return
    for sid in dict.fromkeys(session_ids):
        try:
            for item in queue_list(sid):
                qid = item.get("queue_id")
                if not qid:
                    continue
                reason = (
                    QUEUE_REMOVE_BACKGROUND_PURGED
                    if str(item.get("source") or "") == "background_task"
                    else QUEUE_REMOVE_USER_DELETED
                )
                queue_remove(sid, qid, reason=reason)
        except Exception:  # noqa: BLE001
            pass
