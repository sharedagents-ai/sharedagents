"""SQLite registry for Dev-tab closed loops.

One row per object/branch. Reuse the row if the same repo+branch is started
again — that is the one-branch rule in storage.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sharedagents_core.layout import sharedagents_root

_CREATE = """
CREATE TABLE IF NOT EXISTS dev_loops (
    id TEXT PRIMARY KEY,
    object TEXT NOT NULL,
    done_when TEXT NOT NULL,
    repo TEXT NOT NULL,
    branch TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'feature',
    agent TEXT NOT NULL DEFAULT 'build',
    session_id TEXT NOT NULL DEFAULT '',
    pr_number INTEGER,
    pr_url TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL DEFAULT 'inner',
    inner_laps INTEGER NOT NULL DEFAULT 0,
    last_sha TEXT NOT NULL DEFAULT '',
    ci_status TEXT NOT NULL DEFAULT '',
    last_error TEXT NOT NULL DEFAULT '',
    last_continue_at TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    published_at TEXT NOT NULL DEFAULT '',
    issue_number INTEGER,
    issue_url TEXT NOT NULL DEFAULT ''
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_dev_loops_repo_branch
    ON dev_loops(repo, branch);
"""

_STATES = frozenset(
    {
        "inner",
        "pr_open",
        "ci_red",
        "ready",
        "merged",
        "blocked",
        "aborted",
    }
)

_lock = threading.Lock()
_registry: DevLoopRegistry | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_db_path() -> Path:
    return Path(sharedagents_root()) / "web" / "dev_loops.db"


class DevLoopRegistry:
    def __init__(self, db_path: Path | None = None):
        self.path = Path(db_path) if db_path else default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_CREATE)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(dev_loops)")}
        if "issue_number" not in cols:
            self._conn.execute("ALTER TABLE dev_loops ADD COLUMN issue_number INTEGER")
        if "issue_url" not in cols:
            self._conn.execute("ALTER TABLE dev_loops ADD COLUMN issue_url TEXT NOT NULL DEFAULT ''")
        try:
            self._conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_dev_loops_repo_issue "
                "ON dev_loops(repo, issue_number) WHERE issue_number IS NOT NULL"
            )
        except sqlite3.OperationalError:
            pass

    def close(self) -> None:
        self._conn.close()

    def get(self, loop_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM dev_loops WHERE id = ?",
            (loop_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_by_branch(self, repo: str, branch: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM dev_loops WHERE repo = ? AND branch = ?",
            (repo, branch),
        ).fetchone()
        return dict(row) if row else None

    def get_by_issue(self, repo: str, issue_number: int) -> dict[str, Any] | None:
        try:
            number = int(issue_number)
        except (TypeError, ValueError):
            return None
        if number <= 0:
            return None
        row = self._conn.execute(
            """SELECT * FROM dev_loops
               WHERE repo = ? AND issue_number = ?
               ORDER BY updated_at DESC""",
            (repo, number),
        ).fetchone()
        return dict(row) if row else None

    def list_loops(self, *, include_done: bool = True) -> list[dict[str, Any]]:
        sql = "SELECT * FROM dev_loops"
        if not include_done:
            sql += " WHERE state NOT IN ('merged', 'aborted')"
        sql += " ORDER BY updated_at DESC"
        return [dict(r) for r in self._conn.execute(sql).fetchall()]

    def upsert(
        self,
        *,
        object_name: str,
        done_when: str,
        repo: str,
        branch: str,
        kind: str = "feature",
        agent: str = "build",
        session_id: str = "",
    ) -> dict[str, Any]:
        existing = self.get_by_branch(repo, branch)
        now = _now()
        if existing:
            self._conn.execute(
                """UPDATE dev_loops SET object=?, done_when=?, kind=?, agent=?,
                   session_id=CASE WHEN ? = '' THEN session_id ELSE ? END,
                   state=CASE WHEN state IN ('merged','aborted') THEN 'inner' ELSE state END,
                   updated_at=?, last_error=''
                   WHERE id=?""",
                (
                    object_name,
                    done_when,
                    kind,
                    agent,
                    session_id,
                    session_id,
                    now,
                    existing["id"],
                ),
            )
            self._conn.commit()
            row = self.get(existing["id"])
            assert row is not None
            return row
        loop_id = "dev_" + uuid.uuid4().hex[:16]
        self._conn.execute(
            """INSERT INTO dev_loops (
                id, object, done_when, repo, branch, kind, agent, session_id,
                state, created_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?, 'inner', ?, ?)""",
            (
                loop_id,
                object_name,
                done_when,
                repo,
                branch,
                kind,
                agent,
                session_id,
                now,
                now,
            ),
        )
        self._conn.commit()
        row = self.get(loop_id)
        assert row is not None
        return row

    def bump_lap(self, loop_id: str) -> None:
        self._conn.execute(
            "UPDATE dev_loops SET inner_laps = inner_laps + 1, updated_at=? WHERE id=?",
            (_now(), loop_id),
        )
        self._conn.commit()

    def patch(self, loop_id: str, **fields: Any) -> dict[str, Any] | None:
        if not fields:
            return self.get(loop_id)
        allowed = {
            "object",
            "done_when",
            "repo",
            "branch",
            "kind",
            "agent",
            "session_id",
            "pr_number",
            "pr_url",
            "state",
            "inner_laps",
            "last_sha",
            "ci_status",
            "last_error",
            "last_continue_at",
            "published_at",
            "issue_number",
            "issue_url",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"bad fields {sorted(unknown)}")
        if "state" in fields and fields["state"] not in _STATES:
            raise ValueError(f"bad state {fields['state']}")
        fields = {**fields, "updated_at": _now()}
        cols = ", ".join(f"{k}=?" for k in fields)
        vals = list(fields.values()) + [loop_id]
        self._conn.execute(f"UPDATE dev_loops SET {cols} WHERE id=?", vals)
        self._conn.commit()
        return self.get(loop_id)

    def discard(self, loop_id: str) -> None:
        """Remove a row that never bound an issue. Not an abort."""
        self._conn.execute("DELETE FROM dev_loops WHERE id=?", (loop_id,))
        self._conn.commit()


def get_dev_registry(db_path: Path | None = None) -> DevLoopRegistry:
    global _registry
    with _lock:
        if _registry is None or (db_path and Path(db_path) != _registry.path):
            _registry = DevLoopRegistry(db_path)
        return _registry


def reset_dev_registry() -> None:
    global _registry
    with _lock:
        if _registry is not None:
            try:
                _registry.close()
            except Exception:  # noqa: BLE001
                pass
            _registry = None
