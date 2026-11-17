"""Cross-worker web state — live gateway registry and device presence.

SQLite-backed, in the same ``~/.sharedagents/web`` store as sessions/auth — no new
infrastructure, honouring the no-external-dependency posture even
internally.

Two settled behaviours live here:

  * **Live gateway switching.**  The active gateway is a DB
    row (``kv['active_gateway']``), not an import-time constant.  ``server``
    resolves it per request via ``current_gateway()``, so switching is a DB
    write that takes effect WITHOUT restarting the serving process — and a
    second connection (another gunicorn worker) sees the new value immediately.
    The ``SHAREDAGENTS_GATEWAY_URL`` env is only the first-boot SEED, not the
    authority.

  * **Seamless cross-device handoff.**  Each device upserts a
    ``presence`` row keyed on its native device token; ``continue_here`` finds
    another device's recently-active session so the second device can continue
    live mid-turn without picking from a session list.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path

# A device is considered "recently active" for the continue-here affordance if
# it reported presence within this window.
PRESENCE_STALE_SECS = 300

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS kv (
    k TEXT PRIMARY KEY,
    v TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS gateways (
    name     TEXT PRIMARY KEY,
    url      TEXT NOT NULL,
    added_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS presence (
    device_id     TEXT PRIMARY KEY,
    owner         TEXT NOT NULL,
    device_label  TEXT,
    active_sid    TEXT,
    last_event_at INTEGER NOT NULL,
    streaming     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_presence_owner ON presence(owner, last_event_at DESC);
"""

_ACTIVE_GATEWAY_KEY = "active_gateway"


def _now() -> int:
    return int(time.time())


class WebState:
    def __init__(self, db_path, *, seed_gateway: str | None = None):
        self._path = Path(db_path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._reopen_gate = threading.Lock()
        self._inherited_conns: list[sqlite3.Connection] = []
        self._pid = os.getpid()
        self._open_db()
        self._db.executescript(_SCHEMA_SQL)
        if seed_gateway:
            self.seed_active_gateway(seed_gateway)

    def _open_db(self) -> None:
        self._db = sqlite3.connect(
            str(self._path),
            check_same_thread=False,
            isolation_level=None,
        )
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA busy_timeout=5000")

    def reopen_after_fork(self) -> None:
        """Abandon a SQLite connection inherited across Gunicorn's fork.

        Do not ``close()`` the inherited handle (SQLAlchemy
        ``dispose(close=False)``). Pin it so GC cannot close it either.
        """
        with self._reopen_gate:
            if self._pid == os.getpid():
                return
            self._inherited_conns.append(self._db)
            self._open_db()
            self._db.executescript(_SCHEMA_SQL)
            self._pid = os.getpid()

    def _acquire(self) -> threading.RLock:
        if os.getpid() != self._pid:
            self.reopen_after_fork()
        return self._lock

    # -- kv -----------------------------------------------------------------

    def kv_get(self, key: str) -> str | None:
        with self._acquire():
            row = self._db.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
        return row["v"] if row else None

    def kv_set(self, key: str, value: str) -> None:
        with self._acquire():
            self._db.execute(
                "INSERT INTO kv(k, v) VALUES(?, ?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                (key, value),
            )
            self._db.commit()

    # -- gateway registry -----------------------------------------

    def seed_active_gateway(self, url: str) -> None:
        """Set the active gateway only if none is recorded yet (first boot)."""
        with self._acquire():
            row = self._db.execute("SELECT 1 FROM kv WHERE k=?", (_ACTIVE_GATEWAY_KEY,)).fetchone()
            if row is None and url:
                self._db.execute("INSERT INTO kv(k, v) VALUES(?, ?)", (_ACTIVE_GATEWAY_KEY, url))
                self._db.execute(
                    "INSERT OR IGNORE INTO gateways(name, url, added_at) VALUES(?, ?, ?)",
                    ("default", url, _now()),
                )
                self._db.commit()

    def get_active_gateway(self, default: str = "") -> str:
        val = self.kv_get(_ACTIVE_GATEWAY_KEY)
        return val if val else default

    def set_active_gateway(self, url: str, *, name: str | None = None) -> None:
        url = (url or "").strip()
        if not url:
            raise ValueError("gateway url required")
        with self._acquire():
            self._db.execute(
                "INSERT INTO kv(k, v) VALUES(?, ?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                (_ACTIVE_GATEWAY_KEY, url),
            )
            self._db.execute(
                "INSERT OR IGNORE INTO gateways(name, url, added_at) VALUES(?, ?, ?)",
                (name or url, url, _now()),
            )
            self._db.commit()

    def list_gateways(self):
        with self._acquire():
            rows = self._db.execute("SELECT name, url, added_at FROM gateways ORDER BY added_at").fetchall()
        return [dict(r) for r in rows]

    # -- presence / handoff ---------------------------------------

    def upsert_presence(
        self,
        device_id: str,
        owner: str,
        *,
        device_label: str | None = None,
        active_sid: str | None = None,
        streaming: bool = False,
    ) -> None:
        if not device_id or not owner:
            return
        with self._acquire():
            self._db.execute(
                "INSERT INTO presence(device_id, owner, device_label, active_sid, "
                "last_event_at, streaming) VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(device_id) DO UPDATE SET "
                "owner=excluded.owner, device_label=excluded.device_label, "
                "active_sid=excluded.active_sid, last_event_at=excluded.last_event_at, "
                "streaming=excluded.streaming",
                (device_id, owner, device_label, active_sid, _now(), 1 if streaming else 0),
            )
            self._db.commit()

    def continue_here(
        self,
        owner: str,
        *,
        exclude_device_id: str = "",
        current_sid: str | None = None,
        stale_secs: int = PRESENCE_STALE_SECS,
    ) -> dict | None:
        """Return the most-recently-active OTHER device's live session, if any.

        This is the data behind the "continue here" chip: an owner opening a
        second device is offered the session already live on their first device,
        with no session-list navigation.  Returns ``None`` when there is nothing
        to hand off to (no other recently-active device, or it is on the same
        session already being viewed).
        """
        cutoff = _now() - max(1, stale_secs)
        with self._acquire():
            rows = self._db.execute(
                "SELECT device_id, device_label, active_sid, last_event_at, "
                "streaming FROM presence WHERE owner=? AND last_event_at>=? "
                "AND active_sid IS NOT NULL AND active_sid<>'' "
                "ORDER BY last_event_at DESC",
                (owner, cutoff),
            ).fetchall()
        for r in rows:
            if r["device_id"] == exclude_device_id:
                continue
            if current_sid and r["active_sid"] == current_sid:
                continue
            # This is a "continue the turn already LIVE elsewhere" handoff —
            # not "this owner has some other tab open". Without this gate an idle
            # second device (e.g. a desktop parked on a blank "New Chat") keeps
            # its presence row fresh every 20s and falsely advertises itself as
            # "Live on another device" on every other device, forever. Only a
            # device mid-stream is a real handoff target.
            if not r["streaming"]:
                continue
            return dict(r)
        return None
