"""Session Hierarchy Tree & Live Status Queries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sharedagents_session.session_index import connect_sessions_db


def list_live_sessions(sessions_dir: Path | str) -> list[dict[str, Any]]:
    """Query live (non-idle/non-closed) sessions directly from SQLite in <2ms."""
    db_path = Path(sessions_dir) / "sessions.db"
    if not db_path.exists():
        return []

    con = connect_sessions_db(db_path)
    try:
        rows = con.execute(
            "SELECT session_hash, agent_name, model, session_state, "
            "live_turn_idx, live_started_at, updated_at "
            "FROM sessions_meta WHERE session_state NOT IN ('idle', 'closed') "
            "ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        con.close()


def session_tree(sessions_dir: Path | str, root_hash: str | None = None) -> list[dict[str, Any]]:
    """Fetch session hierarchy (parent/children relationships) from sessions_meta."""
    db_path = Path(sessions_dir) / "sessions.db"
    if not db_path.exists():
        return []

    con = connect_sessions_db(db_path)
    try:
        if root_hash:
            rows = con.execute(
                "SELECT session_hash, parent_session_id, agent_name, title, model, session_state, created_at "
                "FROM sessions_meta WHERE session_hash=? OR parent_session_id=? "
                "ORDER BY created_at ASC",
                (root_hash, root_hash),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT session_hash, parent_session_id, agent_name, title, model, session_state, created_at "
                "FROM sessions_meta ORDER BY created_at ASC"
            ).fetchall()
        return [dict(row) for row in rows]
    finally:
        con.close()
