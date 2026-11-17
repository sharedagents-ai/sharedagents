"""Who else may open a conversation: private, or public to everyone / picked people.

Rows live in the Web SQLite database next to sessions and accounts. No row
means private (owner + admins). Keyed by session id, so a gateway-native
conversation (TUI, Telegram) can be shared exactly like a Web one.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

VIS_PRIVATE = "private"
VIS_PUBLIC = "public"
VISIBILITIES = (VIS_PRIVATE, VIS_PUBLIC)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS session_acl (
    session_id TEXT PRIMARY KEY,
    visibility TEXT NOT NULL DEFAULT 'private',
    share_all  INTEGER NOT NULL DEFAULT 0,
    updated_at INTEGER NOT NULL,
    updated_by TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS session_acl_members (
    session_id TEXT NOT NULL,
    email      TEXT NOT NULL,
    PRIMARY KEY (session_id, email)
);
CREATE INDEX IF NOT EXISTS idx_session_acl_members_email ON session_acl_members(email);
"""


def _norm(email: str) -> str:
    return str(email or "").strip().lower()


@dataclass(frozen=True)
class Sharing:
    visibility: str = VIS_PRIVATE
    share_all: bool = False
    members: tuple[str, ...] = ()
    updated_at: int = 0
    updated_by: str = ""

    def admits(self, email: str) -> bool:
        """Whether this sharing lets ``email`` (a non-owner) in."""
        if self.visibility != VIS_PUBLIC:
            return False
        return self.share_all or _norm(email) in self.members

    def to_dict(self) -> dict:
        return {
            "visibility": self.visibility,
            "share_all": self.share_all,
            "members": list(self.members),
            "updated_at": self.updated_at,
            "updated_by": self.updated_by,
        }


class SharingStore:
    """SQLite-backed per-session visibility (``session_acl`` + members)."""

    def __init__(self, db_path):
        self._path = Path(db_path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._reopen_gate = threading.Lock()
        self._inherited_conns: list[sqlite3.Connection] = []
        self._pid = os.getpid()
        self._open_db()

    def _open_db(self) -> None:
        self._db = sqlite3.connect(
            str(self._path),
            check_same_thread=False,
            isolation_level=None,
        )
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA busy_timeout=5000")
        self._db.executescript(_SCHEMA_SQL)

    def reopen_after_fork(self) -> None:
        """Abandon a connection inherited across Gunicorn's fork (see NativeAuthStore)."""
        with self._reopen_gate:
            if self._pid == os.getpid():
                return
            self._inherited_conns.append(self._db)
            self._open_db()
            self._pid = os.getpid()

    def _acquire(self) -> threading.RLock:
        if os.getpid() != self._pid:
            self.reopen_after_fork()
        return self._lock

    def _read_locked(self, sid: str) -> Sharing:
        row = self._db.execute(
            "SELECT visibility, share_all, updated_at, updated_by FROM session_acl WHERE session_id=?",
            (sid,),
        ).fetchone()
        if row is None:
            return Sharing()
        members = tuple(
            sorted(
                str(r["email"])
                for r in self._db.execute(
                    "SELECT email FROM session_acl_members WHERE session_id=?",
                    (sid,),
                )
            )
        )
        return Sharing(
            visibility=str(row["visibility"]),
            share_all=bool(row["share_all"]),
            members=members,
            updated_at=int(row["updated_at"] or 0),
            updated_by=str(row["updated_by"] or ""),
        )

    def get(self, sid: str) -> Sharing:
        sid = str(sid or "").strip()
        if not sid:
            return Sharing()
        with self._acquire():
            return self._read_locked(sid)

    def get_many(self, sids: Iterable[str]) -> dict[str, Sharing]:
        ids = sorted({str(s).strip() for s in sids or () if str(s or "").strip()})
        out = {sid: Sharing() for sid in ids}
        if not ids:
            return out
        with self._acquire():
            for start in range(0, len(ids), 500):
                chunk = ids[start : start + 500]
                ph = ",".join("?" * len(chunk))
                for row in self._db.execute(
                    f"SELECT session_id, visibility, share_all, updated_at, updated_by "
                    f"FROM session_acl WHERE session_id IN ({ph})",
                    chunk,
                ):
                    out[str(row["session_id"])] = Sharing(
                        visibility=str(row["visibility"]),
                        share_all=bool(row["share_all"]),
                        updated_at=int(row["updated_at"] or 0),
                        updated_by=str(row["updated_by"] or ""),
                    )
                members: dict[str, list[str]] = {}
                for row in self._db.execute(
                    f"SELECT session_id, email FROM session_acl_members WHERE session_id IN ({ph})",
                    chunk,
                ):
                    members.setdefault(str(row["session_id"]), []).append(str(row["email"]))
                for sid, emails in members.items():
                    cur = out[sid]
                    out[sid] = Sharing(
                        visibility=cur.visibility,
                        share_all=cur.share_all,
                        members=tuple(sorted(emails)),
                        updated_at=cur.updated_at,
                        updated_by=cur.updated_by,
                    )
        return out

    def set(
        self,
        sid: str,
        *,
        visibility: str,
        share_all: bool,
        members: Iterable[str],
        by: str,
    ) -> Sharing:
        """Replace a conversation's sharing. Private forgets everyone."""
        sid = str(sid or "").strip()
        if not sid:
            raise ValueError("session id required")
        if visibility not in VISIBILITIES:
            raise ValueError("visibility must be private or public")
        people = sorted({_norm(m) for m in (members or ()) if _norm(m)})
        if visibility == VIS_PRIVATE:
            share_all, people = False, []
        elif not share_all and not people:
            raise ValueError("a public conversation needs everyone or at least one person")
        with self._acquire():
            self._db.execute("BEGIN IMMEDIATE")
            try:
                self._db.execute(
                    "INSERT INTO session_acl(session_id, visibility, share_all, updated_at, updated_by) "
                    "VALUES(?,?,?,?,?) ON CONFLICT(session_id) DO UPDATE SET "
                    "visibility=excluded.visibility, share_all=excluded.share_all, "
                    "updated_at=excluded.updated_at, updated_by=excluded.updated_by",
                    (sid, visibility, 1 if share_all else 0, int(time.time()), _norm(by)),
                )
                self._db.execute("DELETE FROM session_acl_members WHERE session_id=?", (sid,))
                self._db.executemany(
                    "INSERT INTO session_acl_members(session_id, email) VALUES(?,?)",
                    [(sid, email) for email in people],
                )
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise
            return self._read_locked(sid)

    def delete(self, sid: str) -> None:
        sid = str(sid or "").strip()
        with self._acquire():
            self._db.execute("DELETE FROM session_acl_members WHERE session_id=?", (sid,))
            self._db.execute("DELETE FROM session_acl WHERE session_id=?", (sid,))

    def shared_ids_for(self, email: str) -> set[str]:
        """Every conversation public to ``email`` (everyone, or by name)."""
        email = _norm(email)
        with self._acquire():
            rows = self._db.execute(
                "SELECT a.session_id FROM session_acl a WHERE a.visibility=? AND ("
                "a.share_all=1 OR EXISTS (SELECT 1 FROM session_acl_members m "
                "WHERE m.session_id=a.session_id AND m.email=?))",
                (VIS_PUBLIC, email),
            ).fetchall()
        return {str(r["session_id"]) for r in rows}
