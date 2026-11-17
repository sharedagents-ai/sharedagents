"""Hidden internal side-call children must not keep a parent ``waiting``.

A hidden internal side call (title generation, task recap) is bookkeeping,
not delegated work: it must never flip a finished parent to ``waiting``
(which the WebUI renders as an armed Stop on a session whose last turn
already succeeded). Mirrors the exclusion ``count_fresh_live_descendants``
applies for the activity snapshot / Tasks board.
"""

from __future__ import annotations

import pytest

from sharedagents_session.session_index import (
    EventProjector,
    SessionIndex,
    connect_sessions_db,
)


def _schema(tmp_path):
    root = tmp_path / "sessions"
    root.mkdir()
    SessionIndex(root)  # materialise the schema
    return root


def _insert(con, sid, *, parent="", state="idle", outcome="none", agent=None):
    con.execute(
        "INSERT INTO sessions_meta (session_hash, agent_name, parent_session_id, "
        "session_state, session_state_updated_at, created_at, session_last_outcome) "
        "VALUES (?,?,?,?,?,?,?)",
        (sid, agent or ("explore" if parent else "jarvis"), parent, state, 1, "2026-11-04T00:00:00Z", outcome),
    )


@pytest.mark.parametrize("agent", ["title", "summary"])
def test_side_call_child_does_not_count_as_parent_work(tmp_path, agent):
    root = _schema(tmp_path)
    con = connect_sessions_db(root / "sessions.db")
    try:
        _insert(con, "parent", state="idle")
        _insert(con, "recap", parent="parent", state="streaming", agent=agent)
        con.commit()
        assert EventProjector()._parent_still_has_work(con, "parent") is False
        assert (
            EventProjector()._parent_still_has_work(
                con,
                "parent",
                freshness_ms=0,
            )
            is False
        )
    finally:
        con.close()


@pytest.mark.parametrize("agent", ["title", "summary"])
def test_side_call_child_does_not_mark_parent_waiting(tmp_path, agent):
    root = _schema(tmp_path)
    con = connect_sessions_db(root / "sessions.db")
    try:
        _insert(con, "parent", state="idle")
        _insert(con, "recap", parent="parent", state="streaming", agent=agent)
        con.commit()
        EventProjector()._maybe_mark_parent_waiting(con, "recap", 2)
        con.commit()
        row = con.execute(
            "SELECT session_state FROM sessions_meta WHERE session_hash='parent'",
        ).fetchone()
        assert row[0] == "idle"
    finally:
        con.close()


def test_real_child_still_marks_parent_waiting(tmp_path):
    root = _schema(tmp_path)
    con = connect_sessions_db(root / "sessions.db")
    try:
        _insert(con, "parent", state="idle")
        _insert(con, "child", parent="parent", state="streaming")
        con.commit()
        EventProjector()._maybe_mark_parent_waiting(con, "child", 2)
        con.commit()
        row = con.execute(
            "SELECT session_state FROM sessions_meta WHERE session_hash='parent'",
        ).fetchone()
        assert row[0] == "waiting"
    finally:
        con.close()


def test_real_child_still_counts_beside_a_side_call(tmp_path):
    root = _schema(tmp_path)
    con = connect_sessions_db(root / "sessions.db")
    try:
        _insert(con, "parent", state="idle")
        _insert(con, "recap", parent="parent", state="streaming", agent="summary")
        _insert(con, "child", parent="parent", state="streaming")
        con.commit()
        assert EventProjector()._parent_still_has_work(con, "parent") is True
    finally:
        con.close()
