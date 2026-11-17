"""sharedagents_web.session_store — the SQLite-backed web chat SessionStore.

Extracted verbatim from ``server.py`` (Task 2 de-monolith split). The class is
re-exported from ``server`` so ``sharedagents_web.server.SessionStore`` and every
existing import keep resolving. Module-level constants/helpers/locks it depends
on still live in ``server`` and are referenced lazily via ``server.<name>`` so
there is no import-time cycle (this module is imported while ``server`` is still
executing its module body, but every referenced name is read at call time —
except the ``create`` default arg, which reads the already-bound
``server.DEFAULT_AGENT``).
"""

import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone

from sharedagents_core.ids import bare_id
from sharedagents_session.session import build_new_chat_title, new_session_id

DEFAULT_AGENT = "jarvis"


def _server():
    from sharedagents_web import server

    return server


def _owner_matches(stored: object, asked: object) -> bool:
    """The store never partitions by owner.

    Who may reach a conversation is decided in ONE place, before any store
    read: ``sharedagents_web.access`` (admins reach everything — a stored owner that
    differs from an admin is a historical artifact, which is what once hid 99
    sessions the loopback ``debug@localhost`` identity had adopted; users
    reach their own and shared conversations). Keeping the store unpartitioned
    means an admin can never be locked out by a stale ``owner`` column.
    """
    return True


class SessionStore:
    """Manages chat sessions, backed by a single SQLite database.

    Replaces the old per-owner file-per-session layout (SESSION_DIR/<owner>/)
    with a single WAL-mode SQLite DB at SESSION_DB_PATH. An in-memory dict
    cache sits in front of SQLite for O(1) lookups; writes go through to both.
    """

    MAX_SESSIONS = 100  # hard cap per user
    MAX_MESSAGES = 100  # max messages kept per session
    MAX_MESSAGE_BYTES = 50 * 1024  # 50 KB max per user message
    MAX_STREAM_BYTES = 100 * 1024  # 100 KB max accumulated assistant response

    _MIGRATION_MARKER = ".migrated_from_legacy"

    # ── schema ──────────────────────────────────────────────────────────────

    _SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS sessions (
        sid        TEXT NOT NULL PRIMARY KEY,
        owner      TEXT NOT NULL,
        updated_at INTEGER NOT NULL,       -- epoch ms for sortable indexing
        payload    TEXT NOT NULL            -- full session JSON blob
    );
    CREATE INDEX IF NOT EXISTS idx_sessions_owner_updated
        ON sessions(owner, updated_at DESC);
    CREATE TABLE IF NOT EXISTS principal_bindings (
        owner             TEXT NOT NULL PRIMARY KEY,
        gateway_principal TEXT NOT NULL UNIQUE,
        created_at        INTEGER NOT NULL
    );
    CREATE TABLE IF NOT EXISTS active_session_streams (
        sid            TEXT PRIMARY KEY,
        heartbeated_at REAL NOT NULL
    );
    """

    # ── lifecycle ───────────────────────────────────────────────────────────

    def __init__(self):
        self.sessions: dict[str, dict] = {}
        # In-process lock for concurrent append from request + background thread.
        # SQLite WAL handles cross-process/thread isolation; this RLock serialises
        # the read-modify-write of `sdata["messages"]` in add_message.
        self._lock = threading.RLock()
        self._reopen_gate = threading.Lock()
        self._inherited_conns: list[sqlite3.Connection] = []
        self._pid = os.getpid()
        self._open_db()
        self._db.executescript(self._SCHEMA_SQL)
        # Migrate legacy files if present, then load into cache
        self._migrate_from_files()
        self._load_all()
        # End the implicit read transaction from ``_load_all`` so later
        # SELECTs see WAL commits from sibling gunicorn workers.
        self._db.commit()

    def _open_db(self) -> None:
        """Open a process-private WAL connection.

        ``isolation_level=None`` is autocommit: SELECTs do not leave a
        deferred snapshot that a later ``BEGIN IMMEDIATE`` must upgrade
        (that upgrade deadlocks against a sibling worker already holding
        the reserved lock). Writes take an explicit ``BEGIN IMMEDIATE``.
        """
        _s = _server()
        _s.SESSION_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(
            str(_s.SESSION_DB_PATH),
            check_same_thread=False,
            isolation_level=None,
        )
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA busy_timeout=5000")

    def reopen_after_fork(self) -> None:
        """Abandon a SQLite connection inherited across Gunicorn's fork.

        ``store = SessionStore()`` runs in the master before workers are
        forked. The child must open its own connection. Do **not**
        ``close()`` the inherited handle — that is SQLAlchemy's
        ``Engine.dispose(close=True)`` bug: the child's close can drop
        POSIX / WAL-shm locks the parent still holds. Pin the old
        object so GC cannot close it either. Do not replace
        ``self._lock``; an unlocked inherited RLock is fine, and
        replacing it strands any waiter still parked on the old object.
        """
        with self._reopen_gate:
            if self._pid == os.getpid():
                return
            self._inherited_conns.append(self._db)
            self._open_db()
            self.sessions.clear()
            self._load_all()
            self._pid = os.getpid()

    def _acquire(self) -> threading.RLock:
        if os.getpid() != self._pid:
            self.reopen_after_fork()
        return self._lock

    # ── migration from legacy file layout ───────────────────────────────────

    def _migrate_from_files(self):
        """One-shot migration of legacy web session files into SQLite.

        Scans the dedicated legacy paths (``_SESSIONS_DIR_DEPRECATED`` and, for
        one backward-compat pass, the gateway sessions dir) for per-owner
        subdirectories containing session ``.json`` files, inserts each into
        SQLite via ``INSERT OR IGNORE``, and writes a marker file at
        ``SESSION_DB_PATH.parent / .migrated_from_legacy`` to prevent re-scan.

        Does NOT rename or mutate any directory it reads — the migration is
        idempotent and leaves the source files untouched. Only runs when the
        SQLite DB is empty AND the marker file is absent.
        """
        _s = _server()
        marker = _s.SESSION_DB_PATH.parent / self._MIGRATION_MARKER
        if marker.exists():
            return
        count = self._db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        if count > 0:
            marker.write_text("1\n")
            return
        inserted = 0
        total = 0

        # Scan the dedicated legacy dir (web-owned, safe to read)
        for legacy in (_s._SESSIONS_DIR_DEPRECATED, _s._GW_SESSIONS_DEPRECATED):
            if not legacy.exists():
                continue
            for owner_dir in sorted(legacy.iterdir()):
                if not owner_dir.is_dir():
                    continue
                owner_name = owner_dir.name
                if owner_name.startswith(".") or owner_name.endswith(self._MIGRATION_MARKER):
                    continue
                for fpath in sorted(owner_dir.glob("*.json")):
                    total += 1
                    try:
                        payload = fpath.read_text(encoding="utf-8")
                        sdata = json.loads(payload)
                        if not isinstance(sdata, dict):
                            continue
                        sid = sdata.get("id")
                        if not isinstance(sid, str) or not sid:
                            continue
                        owner = sdata.get("owner") or owner_name
                        ts = self._iso_to_epoch(sdata.get("updated_at", ""))
                        cur = self._db.execute(
                            "INSERT OR IGNORE INTO sessions(sid, owner, updated_at, payload) VALUES(?,?,?,?)",
                            (sid, owner, ts, payload),
                        )
                        if cur.rowcount:
                            inserted += 1
                    except Exception:
                        continue
        if inserted:
            self._db.commit()
            print(f"SessionStore: migrated {inserted}/{total} sessions into {_s.SESSION_DB_PATH}")
        marker.write_text(f"{inserted}\n")

    # ── persistence helpers ────────────────────────────────────────────────

    _log = logging.getLogger(__name__)

    @staticmethod
    def _normalize_sdata(sdata: dict) -> dict:
        """Collapse ``sharedagents/<agent>`` id prefixes stored in payloads."""
        if not isinstance(sdata, dict):
            return sdata
        for key in ("agent", "model"):
            value = sdata.get(key)
            if isinstance(value, str):
                sdata[key] = bare_id(value)
        return sdata

    def _session_from_payload(self, payload: str) -> dict | None:
        try:
            sdata = json.loads(payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(sdata, dict) or not sdata.get("id"):
            return None
        return self._normalize_sdata(sdata)

    def _clear_txn_locked(self) -> None:
        try:
            self._db.commit()
        except sqlite3.Error:
            self._db.rollback()

    def _begin_write_locked(self) -> None:
        """Take the WAL write lock so a sibling gunicorn worker cannot
        interleave a read-modify-write on the same sessions table."""
        self._clear_txn_locked()
        self._db.execute("BEGIN IMMEDIATE")

    def _fetch_row_locked(self, sid: str) -> dict | None:
        row = self._db.execute(
            "SELECT payload FROM sessions WHERE sid = ?",
            (sid,),
        ).fetchone()
        payload = row["payload"] if row else None
        if payload is None:
            self.sessions.pop(sid, None)
            return None
        sdata = self._session_from_payload(payload)
        if sdata is None:
            self.sessions.pop(sid, None)
            return None
        self.sessions[sid] = sdata
        return sdata

    def _load_from_db_locked(self, sid: str) -> dict | None:
        """Return the live cached row after a SQLite read.

        Other gunicorn workers write the same WAL file; the in-memory map is
        only a same-request mutate/_save scratch pad.
        """
        sdata = self._fetch_row_locked(sid)
        self._clear_txn_locked()
        return sdata

    def _iter_db_sessions_locked(self, *, release: bool = True):
        payloads = [
            row["payload"]
            for row in self._db.execute(
                "SELECT payload FROM sessions ORDER BY updated_at DESC",
            ).fetchall()
        ]
        if release:
            self._clear_txn_locked()
        for payload in payloads:
            sdata = self._session_from_payload(payload)
            if sdata is not None:
                yield sdata

    def _upsert_locked(self, sdata: dict) -> None:
        sid = sdata["id"]
        self.sessions[sid] = sdata
        owner = sdata.get("owner", "")
        ts_ms = self._iso_to_epoch(sdata.get("updated_at", ""))
        payload = json.dumps(sdata, ensure_ascii=False)
        self._db.execute(
            "INSERT OR REPLACE INTO sessions(sid, owner, updated_at, payload) VALUES(?,?,?,?)",
            (sid, owner, ts_ms, payload),
        )

    @staticmethod
    def _iso_to_epoch(iso_str: str) -> int:
        """Convert an ISO-8601 string to epoch milliseconds. Returns 0 on fail."""
        if not iso_str:
            return 0
        if isinstance(iso_str, (int, float)):
            return int(iso_str * 1000) if iso_str < 1e12 else int(iso_str)
        try:
            dt = datetime.fromisoformat(iso_str)
            return int(dt.timestamp() * 1000)
        except (TypeError, ValueError):
            return 0

    def _save(self, sdata: dict) -> bool:
        """Upsert session into SQLite AND update the in-memory cache.

        By updating ``self.sessions`` here (rather than expecting every caller
        to do it after mutation), we support callers that ``store.get()`` a
        session, mutate it, and ``_save()`` it — the cache stays consistent.

        Logs but does NOT propagate failures to callers — the session stays
        alive in-memory and will be recovered from the gateway log on restart.
        Returns True if the write succeeded, False if it was skipped/errored
        (the caller can log or degrade accordingly).
        """
        with self._acquire():
            try:
                self._begin_write_locked()
                self._upsert_locked(sdata)
                self._db.commit()
                return True
            except Exception:
                self._db.rollback()
                self._log.exception("SessionStore._save failed for sid=%s", sdata.get("id"))
                return False

    def update_fields(self, sid: str, **fields) -> dict | None:
        """Read-modify-write named fields so a stale get()+_save cannot
        clobber a sibling worker's ``add_message`` (lost-update).
        """
        if not sid or not fields:
            return None
        with self._acquire():
            self._begin_write_locked()
            try:
                sdata = self._fetch_row_locked(sid)
                if not sdata:
                    self._db.rollback()
                    return None
                sdata.update(fields)
                sdata["updated_at"] = datetime.now(timezone.utc).isoformat()
                self._upsert_locked(sdata)
                self._db.commit()
                return sdata
            except Exception:
                self._db.rollback()
                raise

    def _load_all(self):
        """Load all sessions from SQLite into the in-memory cache."""
        rows = self._db.execute("SELECT payload FROM sessions ORDER BY updated_at DESC").fetchall()
        for row in rows:
            try:
                sdata = self._normalize_sdata(json.loads(row["payload"]))
                sid = sdata.get("id")
                if sid:
                    msgs = sdata.get("messages", [])
                    if len(msgs) > self.MAX_MESSAGES:
                        sdata["messages"] = msgs[-self.MAX_MESSAGES :]
                    # Migrate the old generic placeholder in memory; fresh
                    # records are created with the timestamp form below.
                    if sdata.get("title") == "New Chat":
                        sdata["title"] = build_new_chat_title(
                            sdata.get("created_at"),
                        )
                    self.sessions[sid] = sdata
            except Exception:
                continue

    def _prune(self, owner: str, keep: int = None):
        """If the user has more than ``keep`` (default MAX_SESSIONS) sessions,
        delete the oldest untracked ones.

        Tasks-board rows deliberately outlive a single agent run and keep the
        Web session UUID as their click target.  Pruning that UUID leaves an
        apparently valid task which can only open a ``Session not found``
        page, even though its gateway transcript is still intact.  Tracked
        sessions therefore count toward the cap but are never eviction
        candidates.  If there are more tracked sessions than the configured
        cap, preservation wins and the store may temporarily exceed the cap.
        """
        if keep is None:
            keep = self.MAX_SESSIONS

        # Read the independent task registry before taking the session-store
        # lock.  This avoids a cross-database lock-order dependency with the
        # Tasks API, which reads its registry before consulting this store.
        try:
            from sharedagents_web.task_board import get_task_board_registry

            protected = {str(row.get("session_id") or "") for row in get_task_board_registry().list(owner=owner)}
            protected.discard("")
        except Exception:
            # A temporarily unavailable task registry must degrade toward
            # retention, not silently destroy the only Web UUID -> gateway
            # conversation binding a tracked task may have.
            self._log.exception(
                "SessionStore._prune skipped: cannot read tracked tasks for owner=%s",
                owner,
            )
            return

        with self._acquire():
            rows = self._db.execute(
                """SELECT sid FROM sessions
                   WHERE owner = ?
                   ORDER BY updated_at DESC, rowid DESC""",
                (owner,),
            ).fetchall()
            present_protected = {row["sid"] for row in rows if row["sid"] in protected}
            untracked_slots = max(0, keep - len(present_protected))
            to_remove = []
            kept_untracked = 0
            for row in rows:
                sid = row["sid"]
                if sid in present_protected:
                    continue
                if kept_untracked < untracked_slots:
                    kept_untracked += 1
                    continue
                to_remove.append(sid)
            if not to_remove:
                self._clear_txn_locked()
                return
            placeholders = ",".join("?" for _ in to_remove)
            self._db.execute(
                f"DELETE FROM sessions WHERE sid IN ({placeholders})",
                to_remove,
            )
            self._db.commit()
            for sid in to_remove:
                self.sessions.pop(sid, None)
                # Evict per-session gw lock on prune too
                with _server()._gw_locks_guard:
                    _server()._gw_session_locks.pop(sid, None)

    # ── public API ──────────────────────────────────────────────────────────

    def list_sessions(
        self,
        owner: str,
        agent_name: str = None,
        limit: int = 50,
        *,
        only=None,
    ):
        """List sessions, optionally filtered by agent.

        ``only`` (a predicate over the stored row) narrows the rows BEFORE the
        limit applies — a non-admin's own/shared conversations must not be
        pushed off the page by other people's newer ones.
        """
        limit = max(1, min(200, limit))
        results = []
        # Snapshot the sessions dict under the lock so iterating doesn't race
        # with concurrent creates/deletes.
        with self._acquire():
            snapshot = list(self._iter_db_sessions_locked())
        for sdata in snapshot:
            sid = sdata["id"]
            if not _owner_matches(sdata.get("owner"), owner):
                continue
            if only is not None and not only(sdata):
                continue
            if agent_name and sdata.get("agent") != agent_name:
                continue
            results.append(
                {
                    "id": sid,
                    "title": sdata.get("title", "New Chat"),
                    "session_summary": sdata.get("session_summary", "") or "",
                    "agent": sdata.get("agent", DEFAULT_AGENT),
                    "model": sdata.get("model", ""),
                    "engine_model": sdata.get("engine_model", ""),
                    "engine_model_alias": sdata.get("engine_model_alias", ""),
                    "revision": sdata.get("revision", 0),
                    "effective_mode": sdata.get("effective_mode") or {},
                    "origin": sdata.get("origin", "web"),
                    # Whose conversation this is (sidebar "shared by").
                    "owner": sdata.get("owner", ""),
                    # Carry the bound topic so /api/topics buckets a freshly-created
                    # per-topic chat correctly (whitelisted dict drops unlisted keys).
                    "topic": sdata.get("topic") or None,
                    "updated_at": sdata.get("updated_at", ""),
                    "message_count": len(sdata.get("messages", [])),
                }
            )
        results.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return results[:limit]

    def list_session_bindings(self, owner: str) -> list[dict]:
        """All retained owner-scoped ids needed for provenance recovery.

        This is intentionally not the sidebar projection: that API caps at 200
        and sorts by last activity, while a migration must see every retained
        binding and later order it by the gateway id's creation timestamp.
        Messages and other payload fields are never exposed.
        """
        with self._acquire():
            snapshot = list(self._iter_db_sessions_locked())
        out = []
        for sdata in snapshot:
            if not _owner_matches(sdata.get("owner"), owner):
                continue
            out.append(
                {
                    "id": sdata["id"],
                    "agent": sdata.get("agent", DEFAULT_AGENT),
                }
            )
        return out

    @staticmethod
    def _payload_looks_unused(sdata: dict | None) -> bool:
        """True when the SQLite payload itself has no prompt or turn."""
        if not sdata or sdata.get("origin") == "gateway":
            return False
        if sdata.get("submitted"):
            return False
        if sdata.get("intent_materialized"):
            return False
        if int(sdata.get("revision") or 0) > 0:
            return False
        if int(sdata.get("message_count") or 0) > 0:
            return False
        if sdata.get("session_summary"):
            return False
        messages = sdata.get("messages") or []
        for m in messages:
            if isinstance(m, dict) and m.get("role") in ("user", "assistant"):
                return False
        return not list(messages)

    @staticmethod
    def _gateway_turns_or_unknown(sid: str) -> bool:
        """True when jsonl has turns, or the check failed (keep the row)."""
        try:
            from sharedagents_core.layout import sessions_root
            from sharedagents_session.session_state import turn_count

            sid = str(sid or "")
            return bool(sid and (sessions_root() / f"{sid}.jsonl").is_file() and turn_count(sessions_root(), sid))
        except Exception:
            return True

    @staticmethod
    def is_unused_web_placeholder(sdata: dict | None) -> bool:
        """True when a web row has never had any prompt or turn run (no messages)."""
        if not SessionStore._payload_looks_unused(sdata):
            return False
        return not SessionStore._gateway_turns_or_unknown(str((sdata or {}).get("id") or ""))

    @staticmethod
    def display_agent(sdata: dict | None) -> str:
        """Agent the picker / stream header should show for this web row.

        Unused New Chat placeholders keep ``pending_agent`` (picker intent
        before the first turn). A bound conversation uses the committed
        ``agent`` — leftover ``pending_agent`` from an earlier switch must
        not win, or every ``event: state`` tick flips the chip backward.
        """
        row = sdata or {}
        if SessionStore.is_unused_web_placeholder(row):
            return str(row.get("pending_agent") or row.get("agent") or DEFAULT_AGENT)
        return str(row.get("agent") or row.get("pending_agent") or DEFAULT_AGENT)

    def _task_protected_sids(self, owner: str) -> set[str]:
        try:
            from sharedagents_web.task_board import get_task_board_registry

            protected = {str(row.get("session_id") or "") for row in get_task_board_registry().list(owner=owner)}
            protected.discard("")
            return protected
        except Exception:
            self._log.exception(
                "SessionStore: cannot read tracked tasks for owner=%s",
                owner,
            )
            return set()

    def register_stream(self, sid: str) -> None:
        key = str(sid or "").strip()
        if not key:
            return
        with self._acquire():
            try:
                self._db.execute(
                    "INSERT OR REPLACE INTO active_session_streams (sid, heartbeated_at) VALUES (?, ?)",
                    (key, time.time()),
                )
            except Exception:
                pass

    def unregister_stream(self, sid: str) -> None:
        key = str(sid or "").strip()
        if not key:
            return
        with self._acquire():
            try:
                self._db.execute(
                    "DELETE FROM active_session_streams WHERE sid = ?",
                    (key,),
                )
            except Exception:
                pass

    def is_stream_active(self, sid: str) -> bool:
        key = str(sid or "").strip()
        if not key:
            return False
        with self._acquire():
            try:
                row = self._db.execute(
                    "SELECT heartbeated_at FROM active_session_streams WHERE sid = ?",
                    (key,),
                ).fetchone()
                if row and (time.time() - float(row[0] or 0)) < 45.0:
                    return True
            except Exception:
                pass
        return False

    def reserve_web_chat(
        self,
        owner: str,
        agent: str = DEFAULT_AGENT,
        model: str = "",
        topic: str = "",
    ) -> tuple[dict, list[str]]:
        """Return one unused New Chat session, reusing an existing empty session if available."""
        topic_norm = (topic or "").strip() or None
        protected = self._task_protected_sids(owner)
        discarded: list[str] = []
        with self._acquire():
            self._begin_write_locked()
            try:
                unused = []
                me = str(owner or "").strip().lower()
                for sdata in self._iter_db_sessions_locked(release=False):
                    sid = sdata["id"]
                    # Strictly this person's placeholders: reusing (or
                    # discarding) someone else's New Chat would hand them a
                    # row they don't own.
                    if str(sdata.get("owner") or "").strip().lower() != me:
                        continue
                    if sid in protected:
                        continue
                    if not self.is_unused_web_placeholder(sdata):
                        continue
                    if self.is_stream_active(sid):
                        continue
                    unused.append(sdata)
                same_topic = [row for row in unused if (row.get("topic") or None) == topic_norm]
                keep = None
                if same_topic:
                    keep = max(
                        same_topic,
                        key=lambda row: str(row.get("updated_at") or ""),
                    )
                elif unused:
                    keep = max(
                        unused,
                        key=lambda row: str(row.get("updated_at") or ""),
                    )
                for row in unused:
                    sid = row["id"]
                    if keep is not None and sid == keep["id"]:
                        continue
                    discarded.append(sid)
                    self._db.execute("DELETE FROM sessions WHERE sid = ?", (sid,))
                    self.sessions.pop(sid, None)
                    with _server()._gw_locks_guard:
                        _server()._gw_session_locks.pop(sid, None)
                if keep is not None:
                    now = datetime.now(timezone.utc).isoformat()
                    keep["agent"] = agent
                    if agent != "cursor" and str(model or "").strip().lower() in {"auto", "ask", "plan"}:
                        keep["model"] = ""
                    else:
                        keep["model"] = model
                    keep["topic"] = topic_norm
                    keep["updated_at"] = now
                    self._upsert_locked(keep)
                    self._db.commit()
                    return keep, discarded
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
        if agent != "cursor" and str(model or "").strip().lower() in {"auto", "ask", "plan"}:
            model = ""
        sdata = self.create(owner, agent, model, topic_norm or "")
        return sdata, discarded

    def create(
        self,
        owner: str,
        agent: str = DEFAULT_AGENT,
        model: str = "",
        topic: str = "",
        workspace_cwd: str = "",
        *,
        session_id: str = "",
    ):
        if agent != "cursor" and str(model or "").strip().lower() in {"auto", "ask", "plan"}:
            model = ""
        sid = session_id or new_session_id()
        now = datetime.now(timezone.utc).isoformat()
        title = build_new_chat_title(now)
        with self._acquire():
            sdata = {
                "id": sid,
                "title": title,
                "agent": agent,
                "model": model,
                "owner": owner,
                "workspace_cwd": str(workspace_cwd or ""),
                "topic": (topic or "").strip() or None,
                "messages": [],
                "created_at": now,
                "updated_at": now,
            }
            self.sessions[sid] = sdata
            self._save(sdata)
        self._prune(owner)

        return sdata

    def bind_gateway_principal(
        self,
        owner: str,
        *,
        local_owner: str = "",
    ) -> str:
        """Return the gateway principal for an authenticated Web owner.

        sharedagents is SINGLE-USER: any credential pair that
        authenticates reaches the same sessions, so every authenticated owner
        is the local operator and resolves to ``local``.

        This replaces a multi-tenant model that no settled requirement ever
        described: one identity could hold ``local`` and every other got an
        opaque ``web:<hmac>`` principal barred from the operator's own
        conversations. It inverted the intent in practice — the loopback
        ``debug@localhost`` identity claimed ``local`` first, bindings were
        immutable, and the operator's OWN account was permanently issued
        ``web:a8bda82b…``. Because ``may_adopt_gateway_session`` admits only
        ``local``, every CLI / TUI / subagent session opened READ-ONLY for them:
        no typing, no stopping a subagent, no continuing a conversation.

        No per-owner row is required to answer this, and none is written: the
        ``principal_bindings`` table carries a UNIQUE constraint on
        ``gateway_principal``, so it cannot represent "every owner is local" at
        all. Existing rows are simply no longer consulted, which also repairs
        an installation already holding a stale non-local binding.
        """
        owner = str(owner or "").strip().lower()
        if not owner:
            raise ValueError("authenticated web owner required")
        return "local"

    def _gateway_binding_owners_locked(self) -> dict[str, str]:
        """Database-backed binding index; caller holds ``self._lock``."""
        bindings: dict[str, str] = {}
        rows = self._db.execute("SELECT owner, payload FROM sessions").fetchall()
        for row in rows:
            try:
                sdata = json.loads(row["payload"])
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            owner = str(row["owner"] or "")
            for gateway_id in {
                sdata.get("id"),
            }:
                if not gateway_id:
                    continue
                prior = bindings.get(gateway_id)
                if prior is not None and not _owner_matches(prior, owner):
                    # Quarantine conflicting historical bindings.  The
                    # sentinel matches no authenticated owner, so all adoption
                    # attempts fail closed until an operator repairs the data.
                    bindings[gateway_id] = "!conflict"
                else:
                    bindings[gateway_id] = owner
        return bindings

    def _gateway_binding_owner_locked(self, gateway_id: str) -> str | None:
        return self._gateway_binding_owners_locked().get(gateway_id)

    def gateway_binding_owners(self) -> dict[str, str]:
        """Return one canonical snapshot used by bounded session listings."""
        with self._acquire():
            bindings = self._gateway_binding_owners_locked()
            self._clear_txn_locked()
            return bindings

    def gateway_binding_owner(self, gateway_id: str) -> str | None:
        """Return the Web owner already bound to a public/gateway id.

        SQLite, not the process-local cache, is authoritative here because
        production uses multiple Web workers.  A stale worker must never
        adopt over a binding another worker just committed.
        """
        with self._acquire():
            owner = self._gateway_binding_owner_locked(gateway_id)
            self._clear_txn_locked()
            return owner

    def adopt_gateway(
        self,
        owner: str,
        hash_id: str,
        meta: dict,
        *,
        gateway_principal: str,
    ):
        """Bind a gateway-native session into this user's web store.

        The web record's id IS the gateway session hash. Idempotent:
        re-adopting refreshes title/agent/history. ``meta`` is a
        :func:`gw_session_history` dict.

        Prunes before inserting (room for one new session) and protects
        the adopted session from eviction by pruning first.
        """
        if gateway_principal != "local":
            raise PermissionError("only the local installation owner may adopt gateway sessions")
        now = datetime.now(timezone.utc).isoformat()
        # Prune BEFORE the insert: evict oldest pre-existing sessions down
        # to MAX_SESSIONS-1 so there's always room for the incoming one.
        self._prune(owner, keep=self.MAX_SESSIONS - 1)
        with self._acquire():
            self._begin_write_locked()
            try:
                bound_owner = self._gateway_binding_owner_locked(hash_id)
                if bound_owner is not None and not _owner_matches(bound_owner, owner):
                    raise PermissionError("gateway session belongs to another web owner")
                row = self._db.execute("SELECT payload FROM sessions WHERE sid=?", (hash_id,)).fetchone()
                try:
                    existing = json.loads(row["payload"]) if row is not None else {}
                except (TypeError, ValueError, json.JSONDecodeError):
                    existing = {}
                sdata = existing if isinstance(existing, dict) else {}
                adopted_title = meta.get("title") or sdata.get("title") or build_new_chat_title(now)
                if adopted_title == "New Chat":
                    adopted_title = build_new_chat_title(
                        sdata.get("created_at") or now,
                    )
                sdata.update(
                    {
                        "id": hash_id,
                        "title": adopted_title,
                        "agent": meta.get("agent") or sdata.get("agent") or DEFAULT_AGENT,
                        "model": meta.get("model") or sdata.get("model") or "",
                        "engine_model": meta.get("engine_model") or sdata.get("engine_model") or "",
                        "engine_model_alias": meta.get("engine_model_alias") or sdata.get("engine_model_alias") or "",
                        "owner": owner,
                        "origin": "gateway",
                        "messages": meta.get("messages") or [],
                        "created_at": sdata.get("created_at") or now,
                        "updated_at": _server()._iso(meta.get("updated_at")) or now,
                        "title_synced": meta.get("title") or "",
                    }
                )
                if len(sdata["messages"]) > self.MAX_MESSAGES:
                    sdata["messages"] = sdata["messages"][-self.MAX_MESSAGES :]
                self._db.execute(
                    "INSERT OR REPLACE INTO sessions(sid, owner, updated_at, payload) VALUES(?,?,?,?)",
                    (
                        hash_id,
                        owner,
                        self._iso_to_epoch(sdata.get("updated_at", "")),
                        json.dumps(sdata, ensure_ascii=False),
                    ),
                )
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
            self.sessions[hash_id] = sdata
        return sdata  # live reference — callers mutate in-place

    def get(self, sid: str, owner: str = None):
        """Get a session, optionally verifying ownership. Returns the live
        reference — callers can mutate it and then call ``_save()``. For a
        read-only snapshot see ``list_sessions``."""
        with self._acquire():
            sdata = self._load_from_db_locked(sid)
        if sdata and owner and not _owner_matches(sdata.get("owner"), owner):
            return None
        return sdata

    def delete(self, sid: str, owner: str = None):
        """Delete a session, optionally verifying ownership. Also evicts the
        per-session gw lock so the lock table doesn't grow unbounded."""
        with self._acquire():
            sdata = self._load_from_db_locked(sid)
            if sdata and owner and not _owner_matches(sdata.get("owner"), owner):
                return None
            if sdata is None:
                return None
            self._db.execute("DELETE FROM sessions WHERE sid = ?", (sid,))
            self._db.commit()
            self.sessions.pop(sid, None)
            # Evict the per-session gateway lock to prevent unbounded growth
            with _server()._gw_locks_guard:
                _server()._gw_session_locks.pop(sid, None)
            return sdata

    def add_message(self, sid: str, role: str, content: str, metadata: dict = None):
        """Append a message to a session. Locked for concurrent-safe append
        from request thread and background worker."""
        with self._acquire():
            self._begin_write_locked()
            try:
                sdata = self._fetch_row_locked(sid)
                if not sdata:
                    self._db.rollback()
                    return None
                msg = {
                    "role": role,
                    "content": content,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                if metadata:
                    msg["metadata"] = metadata
                sdata["messages"].append(msg)
                if len(sdata["messages"]) > self.MAX_MESSAGES:
                    sdata["messages"] = sdata["messages"][-self.MAX_MESSAGES :]
                sdata["updated_at"] = msg["timestamp"]
                if sdata.get("title") == "New Chat":
                    sdata["title"] = build_new_chat_title(sdata.get("created_at"))
                self._upsert_locked(sdata)
                self._db.commit()
                return msg
            except Exception:
                self._db.rollback()
                raise
