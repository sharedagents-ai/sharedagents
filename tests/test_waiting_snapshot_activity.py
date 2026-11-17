"""Gateway ``_session_tree_activity`` must surface open-job evidence.

A ``waiting`` row's own fresh stamp makes it ``busy`` before the async-job
probe would otherwise run — the probe must still run so the snapshot exposes
``open_job_count``; the composer's waiting-is-idle check needs that count to
distinguish a real waiting-on-a-job session from a wedged ``waiting`` row
(the phantom Stop button the WebUI showed on succeeded turns).
"""

from __future__ import annotations

import json
import sqlite3
import time

import pytest

from sharedagents_agent import orchestrator as orch


@pytest.fixture
def sessions(tmp_path, monkeypatch):
    monkeypatch.setenv("SHAREDAGENTS_HOME", str(tmp_path))
    monkeypatch.setenv("SHAREDAGENTS_AGENTS_HOME", str(tmp_path / "agents"))
    monkeypatch.delenv("SHAREDAGENTS_SESSIONS_DIR", raising=False)
    root = tmp_path / "sessions"
    root.mkdir()
    from sharedagents_session.session_index import SessionIndex

    SessionIndex(root)  # materialise the schema
    return root


def _insert(sessions, sid, *, agent="build", parent="", state="tool_running", updated_ms=0, live_turn_idx=None):
    con = sqlite3.connect(sessions / "sessions.db")
    try:
        con.execute(
            "INSERT INTO sessions_meta (session_hash, agent_name, "
            "parent_session_id, session_state, session_state_updated_at, "
            "live_turn_idx, last_turn_ts, created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (sid, agent, parent, state, int(updated_ms), live_turn_idx, int(updated_ms), "2026-11-17T00:00:00Z"),
        )
        con.commit()
    finally:
        con.close()


def _stale_ms() -> int:
    return int(time.time() * 1000) - (orch.LIVE_FRESHNESS_SECONDS + 300) * 1000


def test_waiting_row_reports_open_job_count(sessions, tmp_path):
    """Fresh ``waiting`` + open async job: jobs are the sole evidence of
    work, and the count must reach the snapshot so Stop stays armed."""
    from sharedagents_gateway.control_endpoint import _session_tree_activity

    fresh = int(time.time() * 1000) - 10_000
    _insert(sessions, "parentSession", agent="jarvis", state="waiting", updated_ms=fresh)
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    (jobs / "job_1.json").write_text(
        json.dumps({"job_id": "job_1", "parent_sid": "parentSession", "status": "running"}), encoding="utf-8"
    )
    activity = _session_tree_activity("parentSession")
    assert activity["session_state"] == "waiting"
    assert activity["open_job_count"] == 1
    assert activity["tree_busy"] is True
    assert activity["capabilities"]["can_stop"] is True


def test_wedged_waiting_row_reports_no_abortable_work(sessions):
    """A waiting row with no live descendants and no open jobs is a wedge:
    the snapshot must expose that so the composer renders Send, not Stop."""
    from sharedagents_gateway.control_endpoint import _session_tree_activity

    _insert(sessions, "parentSession", agent="jarvis", state="waiting", updated_ms=_stale_ms())
    activity = _session_tree_activity("parentSession")
    # Stale + no evidence folds to idle display state.
    assert activity["session_state"] == "idle"
    assert activity["active_descendant_count"] == 0
    assert activity["open_job_count"] == 0
    assert activity["tree_busy"] is False
    assert activity["capabilities"]["can_stop"] is False
