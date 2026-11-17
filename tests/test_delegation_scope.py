"""Read-only lookups are scoped to the parent session's workspace."""

from __future__ import annotations

from sharedagents_agent.tools.impl.subagents import _scope_readonly_lookup
from sharedagents_session.session_index import SessionIndex, connect_sessions_db


def _sessions(tmp_path, monkeypatch):
    root = tmp_path / "sessions"
    root.mkdir()
    SessionIndex(root)  # schema
    monkeypatch.setenv("SHAREDAGENTS_SESSIONS_DIR", str(root))
    monkeypatch.setenv("SOURCE_FOLDER", "/home/user/src")
    con = connect_sessions_db(root / "sessions.db")
    try:
        con.execute(
            "INSERT INTO sessions_meta (session_hash, agent_name, workspace_cwd, "
            "session_state, session_state_updated_at, created_at) "
            "VALUES (?,?,?,?,?,?)",
            ("parent", "jarvis", "/home/user/src/sharedagents", "idle", 1, "2026-11-04T00:00:00Z"),
        )
        con.commit()
    finally:
        con.close()
    return root


def test_source_folder_cwd_is_rewritten_for_readonly_lookup(tmp_path, monkeypatch):
    _sessions(tmp_path, monkeypatch)
    args = {"cwd": "/home/user/src", "_sharedagents_parent_session_id": "parent"}
    out = _scope_readonly_lookup(args, "explore")
    assert out["cwd"] == "/home/user/src/sharedagents"


def test_missing_cwd_defaults_to_the_session_workspace(tmp_path, monkeypatch):
    _sessions(tmp_path, monkeypatch)
    args = {"_sharedagents_parent_session_id": "parent"}
    out = _scope_readonly_lookup(args, "general")
    assert out["cwd"] == "/home/user/src/sharedagents"


def test_explicit_narrower_cwd_is_preserved(tmp_path, monkeypatch):
    _sessions(tmp_path, monkeypatch)
    args = {"cwd": "/home/user/src/other-repo", "_sharedagents_parent_session_id": "parent"}
    out = _scope_readonly_lookup(args, "explore")
    assert out["cwd"] == "/home/user/src/other-repo"


def test_build_is_never_rewritten(tmp_path, monkeypatch):
    _sessions(tmp_path, monkeypatch)
    args = {"cwd": "/home/user/src", "_sharedagents_parent_session_id": "parent"}
    out = _scope_readonly_lookup(args, "build")
    assert out["cwd"] == "/home/user/src"


def test_parent_workspace_equal_to_source_is_not_rewritten(tmp_path, monkeypatch):
    root = _sessions(tmp_path, monkeypatch)
    con = connect_sessions_db(root / "sessions.db")
    try:
        con.execute(
            "UPDATE sessions_meta SET workspace_cwd=? WHERE session_hash='parent'",
            ("/home/user/src",),
        )
        con.commit()
    finally:
        con.close()
    args = {"cwd": "/home/user/src", "_sharedagents_parent_session_id": "parent"}
    out = _scope_readonly_lookup(args, "explore")
    assert out["cwd"] == "/home/user/src"
