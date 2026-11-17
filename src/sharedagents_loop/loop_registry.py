"""Durable scheduled-loop registry — SQLite system of record for loops.

Mirrors the goals trio's registry conventions (``sharedagents_goal.goal_registry``):
self-healing ``_connect``, append-only ``loop_events`` audit log, run-history
(``loop_runs``), a cross-process supervisor lease with heartbeat, and a module
singleton. On top of loops it also owns the cron-job table
(``scheduled_jobs`` — the ``CronCreate``/``CronDelete``/``CronList`` surface)
and the one-shot ``wakeups`` table (the ``ScheduleWakeup`` surface), so the
entire scheduled-work family shares one DB and one 50-job ceiling.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import secrets
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sharedagents_core.layout import loops_db_path
from sharedagents_loop.cron import cron_next

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_STATUSES = frozenset({"running", "paused", "archived"})

STATUS_RUNNING = "running"
STATUS_PAUSED = "paused"
# Legacy value used only while migrating rows written by the initial release.
STATUS_EXPIRED = "expired"
STATUS_ARCHIVED = "archived"

#: Cross-process supervisor lease: heartbeat must arrive within this window.
_SUPERVISOR_LEASE_S = 120.0
#: How often the supervisor renews its lease while a loop is armed.
_LEASE_RENEW_INTERVAL_S = 20.0

#: Hard ceiling on concurrent scheduled work (loops + cron jobs).
MAX_SCHEDULED_JOBS = 50

#: Per-run loop cost cap the operator may set at create time.
MAX_LOOP_BUDGET_USD = 1000.0

#: Stored prompts are replayed unattended, so bound their durable size.
MAX_LOOP_PROMPT_BYTES = 25_000

#: Keepalive fallback window after a missed fire (reference: 1200 s).
KEEPALIVE_FALLBACK_S = 1200

#: Lease column names shared by the loops and jobs tables.
_LEASE_COLS = ("supervisor_owner_id", "supervisor_heartbeat_at")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS loops (
    loop_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    prompt_text TEXT NOT NULL,
    interval_s INTEGER,
    cron TEXT,
    human_schedule TEXT NOT NULL,
    status TEXT NOT NULL,
    created_by TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_run_at TEXT,
    next_run_at TEXT,
    run_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    terminal_reason TEXT,
    expires_at TEXT NOT NULL,
    supervisor_owner_id TEXT,
    supervisor_heartbeat_at TEXT,
    misses INTEGER NOT NULL DEFAULT 0,
    fresh_session_each_run INTEGER NOT NULL DEFAULT 0,
    budget_usd REAL,
    origin_session_id TEXT NOT NULL DEFAULT '',
    actor_email TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_loops_status ON loops(status);
CREATE INDEX IF NOT EXISTS idx_loops_agent_status ON loops(agent_name, status);
CREATE INDEX IF NOT EXISTS idx_loops_session ON loops(session_id);

CREATE TABLE IF NOT EXISTS loop_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    loop_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    ok INTEGER NOT NULL DEFAULT 0,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_loop_runs_loop ON loop_runs(loop_id);

CREATE TABLE IF NOT EXISTS loop_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    loop_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_loop_events_loop ON loop_events(loop_id);

CREATE TABLE IF NOT EXISTS scheduled_jobs (
    job_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    cron TEXT NOT NULL,
    human_schedule TEXT NOT NULL,
    prompt TEXT NOT NULL,
    recurring INTEGER NOT NULL DEFAULT 1,
    durable INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'running',
    created_by TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_fired_at TEXT,
    next_fire_at TEXT,
    expires_at TEXT,
    supervisor_owner_id TEXT,
    supervisor_heartbeat_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON scheduled_jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_session ON scheduled_jobs(session_id);

CREATE TABLE IF NOT EXISTS wakeups (
    wakeup_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    loop_id TEXT,
    scheduled_for TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    fired_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_wakeups_session_status
    ON wakeups(session_id, status);

CREATE TABLE IF NOT EXISTS loop_session_leases (
    session_id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    heartbeat_at TEXT NOT NULL
);
"""


def _ensure_loop_option_columns(conn: sqlite3.Connection) -> None:
    """Add per-run session/budget columns on pre-existing loop DBs."""
    cols = {str(row[1]) for row in conn.execute("PRAGMA table_info(loops)")}
    if "fresh_session_each_run" not in cols:
        conn.execute("ALTER TABLE loops ADD COLUMN fresh_session_each_run INTEGER NOT NULL DEFAULT 0")
    if "budget_usd" not in cols:
        conn.execute("ALTER TABLE loops ADD COLUMN budget_usd REAL")
    if "origin_session_id" not in cols:
        conn.execute("ALTER TABLE loops ADD COLUMN origin_session_id TEXT NOT NULL DEFAULT ''")
    if "actor_email" not in cols:
        try:
            conn.execute("ALTER TABLE loops ADD COLUMN actor_email TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError as exc:
            # Another web/orchestrator process can migrate after our PRAGMA.
            if "duplicate column name: actor_email" not in str(exc).lower():
                raise


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _iso_to_epoch(iso: str) -> float:
    if not iso:
        return 0.0
    try:
        return _dt.datetime.fromisoformat(iso).timestamp()
    except ValueError:
        return 0.0


def _epoch_to_iso(epoch: float) -> str:
    return _dt.datetime.fromtimestamp(
        epoch,
        tz=_dt.timezone.utc,
    ).isoformat(timespec="seconds")


def _add_iso(iso: str, seconds: float) -> str:
    if not iso:
        return iso
    return _epoch_to_iso(_iso_to_epoch(iso) + seconds)


def _iso_age_seconds(iso: str) -> float:
    if not iso:
        return float("inf")
    age = _iso_to_epoch(_now_iso()) - _iso_to_epoch(iso)
    return max(0.0, age)


def new_loop_id() -> str:
    """Fresh unique loop id: ``loop_<epoch>_<hex>``."""
    return f"loop_{int(_dt.datetime.now(_dt.timezone.utc).timestamp())}_{secrets.token_hex(4)}"


def new_job_id() -> str:
    """Fresh unique cron-job id: ``job_<epoch>_<hex>``."""
    return f"job_{int(_dt.datetime.now(_dt.timezone.utc).timestamp())}_{secrets.token_hex(4)}"


def new_wakeup_id() -> str:
    """Fresh unique wakeup id: ``wake_<epoch>_<hex>``."""
    return f"wake_{int(_dt.datetime.now(_dt.timezone.utc).timestamp())}_{secrets.token_hex(4)}"


def first_fire_jitter_s(loop_id: str, interval_s: int) -> float:
    """Deterministic per-loop first-fire jitter.

    ``delay = min(hash(id) * 0.5 * period, 1800000 ms)`` where a stable
    SHA-256 prefix is read as a 0..1 fraction.
    Spreads first fires so loops created together do not all land on
    the same minute.
    """
    digest = hashlib.sha256(str(loop_id).encode("utf-8")).digest()
    frac = int.from_bytes(digest[:8], "big") / ((1 << 64) - 1)
    return min(frac * 0.5 * float(interval_s), 1800.0)


# ---------------------------------------------------------------------------
# Record types
# ---------------------------------------------------------------------------


@dataclass
class LoopRecord:
    """One scheduled loop row (mirrors the GoalRecord conventions)."""

    loop_id: str
    session_id: str
    agent_name: str
    prompt_text: str
    human_schedule: str
    status: str
    created_at: str
    updated_at: str
    expires_at: str
    interval_s: int | None = None
    cron: str | None = None
    created_by: str | None = None
    last_run_at: str | None = None
    next_run_at: str | None = None
    run_count: int = 0
    last_error: str | None = None
    terminal_reason: str | None = None
    supervisor_owner_id: str | None = None
    supervisor_heartbeat_at: str | None = None
    misses: int = 0
    fresh_session_each_run: bool = False
    budget_usd: float | None = None
    origin_session_id: str = ""
    actor_email: str = ""

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> LoopRecord:
        keys = set(row.keys())
        budget_raw = row["budget_usd"] if "budget_usd" in keys else None
        budget = None
        if budget_raw is not None and str(budget_raw) != "":
            try:
                budget = float(budget_raw)
            except (TypeError, ValueError):
                budget = None
        fresh = False
        if "fresh_session_each_run" in keys:
            raw_fresh = row["fresh_session_each_run"]
            try:
                fresh = int(raw_fresh or 0) == 1
            except (TypeError, ValueError):
                fresh = False
        return cls(
            loop_id=str(row["loop_id"]),
            session_id=str(row["session_id"]),
            agent_name=str(row["agent_name"]),
            prompt_text=str(row["prompt_text"]),
            interval_s=row["interval_s"],
            cron=row["cron"],
            human_schedule=str(row["human_schedule"] or ""),
            status=str(row["status"]),
            created_by=row["created_by"],
            actor_email=str(row["actor_email"] or "") if "actor_email" in keys else "",
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            last_run_at=row["last_run_at"],
            next_run_at=row["next_run_at"],
            run_count=int(row["run_count"] or 0),
            last_error=row["last_error"],
            terminal_reason=row["terminal_reason"],
            expires_at=str(row["expires_at"] or ""),
            supervisor_owner_id=row["supervisor_owner_id"],
            supervisor_heartbeat_at=row["supervisor_heartbeat_at"],
            misses=int(row["misses"] or 0),
            fresh_session_each_run=fresh,
            budget_usd=budget,
            origin_session_id=(str(row["origin_session_id"] or "") if "origin_session_id" in keys else ""),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-able projection for the web API and command output."""
        return {
            "loop_id": self.loop_id,
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "prompt_text": self.prompt_text,
            "interval_s": self.interval_s,
            "cron": self.cron,
            "human_schedule": self.human_schedule,
            "period_label": self.human_schedule,
            "status": self.status,
            "created_by": self.created_by,
            "actor_email": str(self.actor_email or ""),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_run_at": self.last_run_at,
            "next_run_at": self.next_run_at,
            "run_count": self.run_count,
            "last_error": self.last_error,
            "terminal_reason": self.terminal_reason,
            "expires_at": self.expires_at,
            "misses": self.misses,
            "fresh_session_each_run": bool(self.fresh_session_each_run),
            "budget_usd": self.budget_usd,
            "origin_session_id": str(self.origin_session_id or ""),
        }


@dataclass
class LoopFilters:
    """List filters for :meth:`LoopRegistry.list_loops`."""

    status: str | None = None
    agent_name: str | None = None
    session_id: str | None = None


@dataclass
class TransitionResult:
    """Outcome of a loop status transition."""

    ok: bool
    message: str
    loop: LoopRecord | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "message": self.message,
            "loop": self.loop.to_dict() if self.loop else None,
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class LoopRegistry:
    """SQLite system of record for loops, cron jobs, and wakeups."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else loops_db_path()
        self._lock = threading.RLock()
        self._connect()

    # -- connection --------------------------------------------------------

    def _connect(self) -> None:
        """Open (or heal) the SQLite connection, creating schema as needed."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=15.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=15000")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(SCHEMA_SQL)
        _ensure_loop_option_columns(conn)
        # Keep the legacy columns for database/API compatibility, but remove
        # age limits from existing schedules as well as newly created ones.
        conn.execute("UPDATE loops SET expires_at = '' WHERE expires_at != ''")
        conn.execute("UPDATE scheduled_jobs SET expires_at = NULL WHERE expires_at IS NOT NULL")
        # 2026-09-07 (owner): loops are DELETED when they end — the board has
        # no archived state. Purge any
        # rows a previous archive/expire lifecycle left behind (children
        # first — loop_runs / loop_events / wakeups have no FK cascade).
        for _table in ("wakeups", "loop_events", "loop_runs"):
            conn.execute(
                f"DELETE FROM {_table} WHERE loop_id IN (SELECT loop_id FROM loops WHERE status IN (?, ?))",
                (STATUS_ARCHIVED, STATUS_EXPIRED),
            )
        conn.execute(
            "DELETE FROM loops WHERE status IN (?, ?)",
            (STATUS_ARCHIVED, STATUS_EXPIRED),
        )
        conn.commit()
        self._conn = conn

    # -- low-level helpers ---------------------------------------------------

    def _execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        with self._lock:
            try:
                cur = self._conn.execute(sql, params)
                self._conn.commit()
                return cur
            except sqlite3.ProgrammingError:
                self._connect()
                cur = self._conn.execute(sql, params)
                self._conn.commit()
                return cur

    def _query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            try:
                return list(self._conn.execute(sql, params))
            except sqlite3.ProgrammingError:
                self._connect()
                return list(self._conn.execute(sql, params))

    def _append_event(
        self,
        loop_id: str,
        event_type: str,
        payload: str | None = None,
    ) -> None:
        """Append-only audit event (sole writer; mirrors goal_events)."""
        self._execute(
            "INSERT INTO loop_events (loop_id, event_type, payload, created_at) VALUES (?, ?, ?, ?)",
            (loop_id, event_type, payload, _now_iso()),
        )

    # -- loops ----------------------------------------------------------------

    def create_loop(
        self,
        *,
        session_id: str,
        agent_name: str,
        prompt_text: str,
        human_schedule: str,
        interval_s: int | None = None,
        cron: str | None = None,
        created_by: str | None = None,
        actor_email: str = "",
        next_run_at: str | None = None,
        first_delay_s: float | None = None,
        fire_immediately: bool = False,
        fresh_session_each_run: bool = False,
        budget_usd: float | None = None,
    ) -> LoopRecord:
        """Create a running loop (interval- or cron-backed).

        ``fire_immediately`` (web create contract) anchors
        the FIRST run at the creation instant — no interval + jitter wait and
        no cron-next wait — so the operator sees the first run start on
        creation. Subsequent runs follow the normal interval/cron anchors.
        """
        prompt_text = str(prompt_text or "").strip()
        if not prompt_text:
            raise ValueError("prompt_text is required")
        if len(prompt_text.encode("utf-8")) > MAX_LOOP_PROMPT_BYTES:
            raise ValueError(f"prompt_text exceeds {MAX_LOOP_PROMPT_BYTES} bytes")
        if not session_id:
            raise ValueError("session_id is required")
        now = _now_iso()
        loop_id = new_loop_id()
        if cron:
            from sharedagents_loop.cron import parse_cron

            parse_cron(cron)
        if next_run_at is None:
            if fire_immediately:
                next_run_at = now
            elif interval_s is not None:
                delay = first_fire_jitter_s(loop_id, interval_s) if first_delay_s is None else first_delay_s
                next_run_at = _add_iso(now, float(interval_s) + delay)
            elif cron:
                next_fire = cron_next(cron, _dt.datetime.fromisoformat(now))
                if next_fire is None:
                    raise ValueError("cron expression has no reachable fire time")
                next_run_at = next_fire.isoformat(timespec="seconds")
        expires_at = ""
        # Opt-in only: True/1 store 1. "false", 2, or any other truthy
        # leftover must not enable per-run session minting.
        fresh_flag = 1 if fresh_session_each_run == 1 else 0
        budget_val: float | None = None
        if budget_usd is not None:
            budget_val = float(budget_usd)
            if not (budget_val > 0) or budget_val > MAX_LOOP_BUDGET_USD:
                raise ValueError(f"budget_usd must be > 0 and ≤ {MAX_LOOP_BUDGET_USD:g}")
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                if self._scheduled_count_conn() >= MAX_SCHEDULED_JOBS:
                    raise LookupError("scheduled-work limit reached (50 max) — cancel one first")
                self._conn.execute(
                    "INSERT INTO loops (loop_id, session_id, agent_name, "
                    "prompt_text, interval_s, cron, human_schedule, status, "
                    "created_by, created_at, updated_at, next_run_at, expires_at, "
                    "fresh_session_each_run, budget_usd, origin_session_id, actor_email) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        loop_id,
                        session_id,
                        agent_name,
                        prompt_text,
                        interval_s,
                        cron,
                        human_schedule,
                        STATUS_RUNNING,
                        created_by,
                        now,
                        now,
                        next_run_at,
                        expires_at,
                        fresh_flag,
                        budget_val,
                        session_id,
                        str(actor_email or "").strip().lower(),
                    ),
                )
                self._conn.commit()
            except Exception:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise
        self._append_event(loop_id, "created", human_schedule)
        return self.get(loop_id)  # type: ignore[return-value]

    def stamp_actor_if_empty(self, loop_id: str, actor_email: str) -> bool:
        """Record the WebUI user behind a chat-created loop; first writer wins.

        ``created_by`` keeps its provenance meaning ('chat'/'model'/email);
        ``actor_email`` is the human.
        """
        email = str(actor_email or "").strip().lower()
        if not email:
            return False
        with self._lock:
            cur = self._conn.execute(
                "UPDATE loops SET actor_email=? WHERE loop_id=? AND actor_email=''",
                (email, loop_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def get(self, loop_id: str) -> LoopRecord | None:
        rows = self._query(
            "SELECT * FROM loops WHERE loop_id = ?",
            (str(loop_id),),
        )
        return LoopRecord.from_row(rows[0]) if rows else None

    def ensure_origin_session_id(
        self,
        loop_id: str,
        inferred: str,
    ) -> str | None:
        """Persist the immutable loop origin once; later calls are no-ops.

        ``origin_session_id`` is the session the loop was first bound to.
        Fresh-session ticks parent to this id so they stay siblings instead
        of a linked list. Rebind must never overwrite it.
        """
        inferred = str(inferred or "").strip()
        rec = self.get(loop_id)
        if rec is None:
            return None
        stored = str(getattr(rec, "origin_session_id", "") or "").strip()
        if stored:
            return stored
        if not inferred:
            return ""
        now = _now_iso()
        self._execute(
            "UPDATE loops SET origin_session_id = ?, updated_at = ? "
            "WHERE loop_id = ? AND (origin_session_id IS NULL "
            "OR origin_session_id = '')",
            (inferred, now, loop_id),
        )
        rec = self.get(loop_id)
        return str(getattr(rec, "origin_session_id", "") or inferred) if rec else inferred

    def tick_origin_edges(self) -> dict[str, str]:
        """Immutable tick → origin map from rebound events + stored origin.

        Independent of the live ``fresh_session_each_run`` flag. A tick
        stays a tick after minting is disabled. Missing origin or a
        rebound-history read failure yields no edge (fail-closed).
        """
        edges: dict[str, str] = {}
        for rec in self.list_loops():
            origin = str(getattr(rec, "origin_session_id", "") or "").strip()
            if not origin:
                continue
            try:
                rebound = self.rebound_session_ids(rec.loop_id)
            except Exception:  # noqa: BLE001 — skip unverified edges
                continue
            for sid in rebound:
                sid = str(sid or "").strip()
                if sid and sid != origin:
                    edges[sid] = origin
        return edges

    def origin_and_tick_ids(self) -> tuple[set[str], set[str]]:
        """Loop origins and minted tick ids from durable rebound edges.

        A tick is a ``session_rebound`` payload paired with the stored
        ``origin_session_id``. The live fresh-session flag is ignored so
        historical ticks remain visible after minting is turned off.
        """
        edges = self.tick_origin_edges()
        return set(edges.values()), set(edges.keys())

    def rebound_session_ids(self, loop_id: str) -> list[str]:
        """Session ids minted by ``session_rebound`` events, oldest first."""
        rows = self._query(
            "SELECT payload FROM loop_events WHERE loop_id = ? AND event_type = 'session_rebound' ORDER BY event_id",
            (str(loop_id),),
        )
        out: list[str] = []
        for row in rows:
            sid = str(row["payload"] if not isinstance(row, tuple) else row[0] or "")
            sid = sid.strip()
            if sid:
                out.append(sid)
        return out

    def rebind_session(self, loop_id: str, session_id: str) -> LoopRecord | None:
        """Point a loop at a new bound session (per-run fresh session)."""
        sid = str(session_id or "").strip()
        if not sid:
            raise ValueError("session_id is required")
        now = _now_iso()
        self._execute(
            "UPDATE loops SET session_id = ?, updated_at = ? WHERE loop_id = ?",
            (sid, now, loop_id),
        )
        self._append_event(loop_id, "session_rebound", sid)
        return self.get(loop_id)

    def set_fresh_session_each_run(
        self,
        loop_id: str,
        enabled: bool,
    ) -> LoopRecord | None:
        """Turn per-run session minting on or off for an existing loop."""
        if self.get(loop_id) is None:
            return None
        flag = 1 if enabled == 1 else 0
        now = _now_iso()
        self._execute(
            "UPDATE loops SET fresh_session_each_run = ?, updated_at = ? WHERE loop_id = ?",
            (flag, now, loop_id),
        )
        self._append_event(
            loop_id,
            "fresh_session_updated",
            "1" if flag else "0",
        )
        return self.get(loop_id)

    def list_loops(self, filters: LoopFilters | None = None) -> list[LoopRecord]:
        filters = filters or LoopFilters()
        clauses: list[str] = []
        params: list[Any] = []
        if filters.status:
            clauses.append("status = ?")
            params.append(filters.status)
        if filters.agent_name:
            clauses.append("agent_name = ?")
            params.append(filters.agent_name)
        if filters.session_id:
            clauses.append("session_id = ?")
            params.append(filters.session_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._query(
            f"SELECT * FROM loops {where} ORDER BY created_at DESC, rowid DESC",
            tuple(params),
        )
        return [LoopRecord.from_row(r) for r in rows]

    def record_run(
        self,
        loop_id: str,
        *,
        started_at: str,
        finished_at: str | None = None,
        ok: bool = True,
        error: str | None = None,
        advance_schedule: bool = True,
    ) -> None:
        """Append a run-history row and advance the loop's anchors.

        ``next_run_at`` is anchored to the LAST run (no catch-up), pushed
        one interval for fixed-interval loops; model-paced loops (no
        interval) leave ``next_run_at`` alone so the supervisor re-arms
        from the model's wakeup.
        """
        self._execute(
            "INSERT INTO loop_runs (loop_id, started_at, finished_at, ok, error) VALUES (?, ?, ?, ?, ?)",
            (loop_id, started_at, finished_at, 1 if ok else 0, error),
        )
        loop = self.get(loop_id)
        if loop is None:
            return
        next_run_at = loop.next_run_at
        if advance_schedule and loop.interval_s is not None:
            anchor = finished_at or started_at
            next_run_at = _add_iso(anchor, float(loop.interval_s))
        elif advance_schedule and loop.cron:
            anchor = _dt.datetime.fromisoformat(finished_at or started_at)
            next_fire = cron_next(loop.cron, anchor)
            next_run_at = next_fire.isoformat(timespec="seconds") if next_fire else None
        self._execute(
            "UPDATE loops SET last_run_at = ?, next_run_at = ?, "
            "run_count = run_count + 1, last_error = ?, "
            "misses = CASE WHEN ? AND (interval_s IS NOT NULL OR cron IS NOT NULL) "
            "THEN 0 ELSE misses END, updated_at = ? WHERE loop_id = ?",
            (
                finished_at or started_at,
                next_run_at,
                None if ok else error,
                1 if ok else 0,
                _now_iso(),
                loop_id,
            ),
        )
        self._append_event(loop_id, "ran", "ok" if ok else "error")

    def consecutive_failures(self, loop_id: str, *, limit: int = 100) -> int:
        """Count newest failed attempts until the most recent success."""
        rows = self._query(
            "SELECT ok FROM loop_runs WHERE loop_id = ? ORDER BY run_id DESC LIMIT ?",
            (loop_id, max(1, int(limit))),
        )
        count = 0
        for row in rows:
            if bool(row["ok"]):
                break
            count += 1
        return count

    def mark_fire_skipped(self, loop_id: str) -> int:
        """A scheduled fire found the session busy: no catch-up, push one
        interval, count the miss. Returns the new consecutive-miss count."""
        loop = self.get(loop_id)
        if loop is None:
            return 0
        now = _now_iso()
        if loop.interval_s is not None:
            next_run_at = _add_iso(now, float(loop.interval_s))
        elif loop.cron:
            next_fire = cron_next(loop.cron, _dt.datetime.fromisoformat(now))
            next_run_at = next_fire.isoformat(timespec="seconds") if next_fire else None
        else:
            next_run_at = _add_iso(now, 600.0)
        self._execute(
            "UPDATE loops SET next_run_at = ?, misses = misses + 1, updated_at = ? WHERE loop_id = ?",
            (next_run_at, now, loop_id),
        )
        self._append_event(loop_id, "fire_skipped", f"misses={loop.misses + 1}")
        return loop.misses + 1

    def arm_keepalive(self, loop_id: str, *, count_miss: bool = True) -> None:
        """Keepalive fallback: re-check the session after ``KEEPALIVE_FALLBACK_S``.

        ``count_miss=True`` (the model-declined-to-reschedule path) also
        increments the consecutive-miss counter so the supervisor can end
        a model-paced loop after its second missed reschedule; busy-skips
        pass ``count_miss=False`` — the model never had a chance to
        reschedule, so a busy skip is not a decline.
        """
        loop = self.get(loop_id)
        if loop is None:
            return
        self._execute(
            "UPDATE loops SET next_run_at = ?, misses = misses + ?, updated_at = ? WHERE loop_id = ?",
            (_add_iso(_now_iso(), KEEPALIVE_FALLBACK_S), 1 if count_miss else 0, _now_iso(), loop_id),
        )
        self._append_event(loop_id, "keepalive_armed")

    def rearm_from_wakeup(self, loop_id: str, wakeup_time: str) -> None:
        """Model-paced re-arm: the model scheduled a wakeup for this loop."""
        self._execute(
            "UPDATE loops SET next_run_at = ?, misses = 0, updated_at = ? WHERE loop_id = ?",
            (wakeup_time, _now_iso(), loop_id),
        )

    def pause_loop(self, loop_id: str, *, reason: str | None = None) -> TransitionResult:
        loop = self.get(loop_id)
        if loop is None:
            return TransitionResult(False, f"loop not found: {loop_id}")
        if loop.status == STATUS_ARCHIVED or loop.status == STATUS_EXPIRED:
            return TransitionResult(
                False,
                f"cannot pause a {loop.status} loop",
                loop,
            )
        if loop.status == STATUS_PAUSED:
            return TransitionResult(True, "already paused", loop)
        self._execute(
            "UPDATE loops SET status = ?, last_error = ?, updated_at = ?, "
            "supervisor_owner_id = NULL, supervisor_heartbeat_at = NULL "
            "WHERE loop_id = ?",
            (STATUS_PAUSED, reason, _now_iso(), loop_id),
        )
        self._append_event(loop_id, "paused", reason)
        self.cancel_wakeups_for_loop(loop_id)
        return TransitionResult(True, "paused", self.get(loop_id))

    def resume_loop(self, loop_id: str) -> TransitionResult:
        loop = self.get(loop_id)
        if loop is None:
            return TransitionResult(False, f"loop not found: {loop_id}")
        if loop.status not in (STATUS_PAUSED,):
            if loop.status == STATUS_RUNNING:
                return TransitionResult(True, "already running", loop)
            return TransitionResult(
                False,
                f"cannot resume a {loop.status} loop",
                loop,
            )
        now = _now_iso()
        next_run_at = _add_iso(now, float(loop.interval_s)) if loop.interval_s is not None else None
        if loop.interval_s is None and loop.cron:
            next_fire = cron_next(loop.cron, _dt.datetime.fromisoformat(now))
            next_run_at = next_fire.isoformat(timespec="seconds") if next_fire else None
        resumed = False
        ok = False
        message = ""
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute(
                    "SELECT status FROM loops WHERE loop_id = ?",
                    (loop_id,),
                ).fetchone()
                status = str(row["status"]) if row is not None else ""
                if not status:
                    message = f"loop not found: {loop_id}"
                elif status == STATUS_RUNNING:
                    ok, message = True, "already running"
                elif status != STATUS_PAUSED:
                    message = f"cannot resume a {status} loop"
                elif self._scheduled_count_conn() >= MAX_SCHEDULED_JOBS:
                    message = "scheduled-work limit reached (50 max) — cancel one first"
                else:
                    self._conn.execute(
                        "UPDATE loops SET status = ?, next_run_at = ?, "
                        "misses = 0, last_error = NULL, updated_at = ? "
                        "WHERE loop_id = ? AND status = ?",
                        (
                            STATUS_RUNNING,
                            next_run_at,
                            now,
                            loop_id,
                            STATUS_PAUSED,
                        ),
                    )
                    resumed = True
                    ok, message = True, "resumed"
                self._conn.commit()
            except Exception:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise
        if resumed:
            self._append_event(loop_id, "resumed")
        return TransitionResult(ok, message, self.get(loop_id))

    def archive_loop(self, loop_id: str, *, reason: str | None = None) -> TransitionResult:
        loop = self.get(loop_id)
        if loop is None:
            return TransitionResult(False, f"loop not found: {loop_id}")
        if loop.status == STATUS_ARCHIVED:
            return TransitionResult(True, "already archived", loop)
        self._execute(
            "UPDATE loops SET status = ?, terminal_reason = ?, "
            "next_run_at = NULL, updated_at = ?, supervisor_owner_id = NULL, "
            "supervisor_heartbeat_at = NULL WHERE loop_id = ?",
            (STATUS_ARCHIVED, reason, _now_iso(), loop_id),
        )
        self._append_event(loop_id, "archived", reason)
        self.cancel_wakeups_for_loop(loop_id)
        return TransitionResult(True, "archived", self.get(loop_id))

    def end_model_no_reschedule(
        self,
        loop_id: str,
        *,
        reason: str = "model_stopped",
    ) -> TransitionResult:
        """End a model-paced loop after the 2nd missed reschedule."""
        loop = self.get(loop_id)
        if loop is None:
            return TransitionResult(False, f"loop not found: {loop_id}")
        if loop.status != STATUS_RUNNING:
            return TransitionResult(True, f"already {loop.status}", loop)
        self._execute(
            "UPDATE loops SET status = ?, terminal_reason = ?, next_run_at = NULL, updated_at = ? WHERE loop_id = ?",
            (STATUS_PAUSED, reason, _now_iso(), loop_id),
        )
        self._append_event(loop_id, "model_no_reschedule", reason)
        return TransitionResult(True, "ended", self.get(loop_id))

    def delete_loop(self, loop_id: str) -> bool:
        """Permanently delete a loop and its run/event/wakeup history.

        The board's terminal action (Delete, 2026-09-07 owner contract):
        nothing is archived anymore. Returns True when a row was removed.
        The supervisor drive task exits at its next checkpoint when the row
        it drives disappears (``_drive_one_loop`` re-reads each cycle).
        """
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                cur = self._conn.execute(
                    "DELETE FROM loops WHERE loop_id = ?",
                    (loop_id,),
                )
                if cur.rowcount == 0:
                    self._conn.commit()
                    return False
                self._conn.execute(
                    "DELETE FROM loop_runs WHERE loop_id = ?",
                    (loop_id,),
                )
                self._conn.execute(
                    "DELETE FROM loop_events WHERE loop_id = ?",
                    (loop_id,),
                )
                self._conn.execute(
                    "DELETE FROM wakeups WHERE loop_id = ?",
                    (loop_id,),
                )
                self._conn.commit()
                return True
            except Exception:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise

    # -- run history / events --------------------------------------------------

    def list_runs(self, loop_id: str, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._query(
            "SELECT * FROM loop_runs WHERE loop_id = ? ORDER BY started_at DESC LIMIT ?",
            (loop_id, int(limit)),
        )
        return [dict(r) for r in rows]

    def list_events(
        self,
        loop_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if loop_id:
            rows = self._query(
                "SELECT * FROM loop_events WHERE loop_id = ? ORDER BY event_id ASC LIMIT ?",
                (loop_id, int(limit)),
            )
        else:
            rows = self._query(
                "SELECT * FROM loop_events ORDER BY event_id DESC LIMIT ?",
                (int(limit),),
            )
        return [dict(r) for r in rows]

    # -- supervisor lease --------------------------------------------------------

    def try_claim_supervisor(self, loop_id: str, owner: str) -> bool:
        """Claim the drive lease for ``loop_id``. Only one owner at a time."""
        now = _now_iso()
        cutoff = _add_iso(now, -_SUPERVISOR_LEASE_S)
        cur = self._execute(
            "UPDATE loops SET supervisor_owner_id = ?, "
            "supervisor_heartbeat_at = ? WHERE loop_id = ? AND status = ? "
            "AND (supervisor_owner_id IS NULL OR supervisor_owner_id = ? "
            "OR supervisor_heartbeat_at IS NULL "
            "OR julianday(supervisor_heartbeat_at) IS NULL "
            "OR julianday(supervisor_heartbeat_at) <= julianday(?))",
            (owner, now, loop_id, STATUS_RUNNING, owner, cutoff),
        )
        return cur.rowcount > 0

    def touch_supervisor(self, loop_id: str, owner: str) -> bool:
        cur = self._execute(
            "UPDATE loops SET supervisor_heartbeat_at = ? WHERE loop_id = ? AND supervisor_owner_id = ? AND status = ?",
            (_now_iso(), loop_id, owner, STATUS_RUNNING),
        )
        return cur.rowcount > 0

    def release_supervisor(self, loop_id: str, owner: str) -> bool:
        cur = self._execute(
            "UPDATE loops SET supervisor_owner_id = NULL, "
            "supervisor_heartbeat_at = NULL "
            "WHERE loop_id = ? AND supervisor_owner_id = ?",
            (loop_id, owner),
        )
        return cur.rowcount > 0

    # -- scheduled jobs (CronCreate/CronDelete/CronList) ---------------------------

    def scheduled_count(self) -> int:
        """Running loops + running cron jobs (the shared 50-job ceiling)."""
        with self._lock:
            return self._scheduled_count_conn()

    def _scheduled_count_conn(self) -> int:
        row = self._conn.execute(
            "SELECT (SELECT COUNT(*) FROM loops WHERE status = ?) + "
            "(SELECT COUNT(*) FROM scheduled_jobs WHERE status = ?) AS n",
            (STATUS_RUNNING, STATUS_RUNNING),
        ).fetchone()
        return int(row["n"] if row else 0)

    def create_job(
        self,
        *,
        session_id: str,
        agent_name: str,
        cron: str,
        human_schedule: str,
        prompt: str,
        recurring: bool = True,
        durable: bool = False,
        created_by: str | None = None,
        next_fire_at: str | None = None,
    ) -> dict[str, Any]:
        """Create a cron job. Enforces the shared 50-job ceiling."""
        prompt = str(prompt or "").strip()
        if not prompt:
            raise ValueError("prompt is required")
        if len(prompt.encode("utf-8")) > MAX_LOOP_PROMPT_BYTES:
            raise ValueError(f"prompt exceeds {MAX_LOOP_PROMPT_BYTES} bytes")
        if not session_id:
            raise ValueError("session_id is required")
        from sharedagents_loop.cron import parse_cron

        parse_cron(cron)  # validate early
        now = _now_iso()
        job_id = new_job_id()
        if next_fire_at is None:
            after = _dt.datetime.fromisoformat(now)
            nxt = cron_next(cron, after)
            if nxt is None:
                raise ValueError("cron expression has no reachable fire time")
            next_fire_at = nxt.isoformat(timespec="seconds")
        expires_at = None
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                if self._scheduled_count_conn() >= MAX_SCHEDULED_JOBS:
                    raise LookupError("scheduled-work limit reached (50 max) — cancel one first")
                self._conn.execute(
                    "INSERT INTO scheduled_jobs (job_id, session_id, agent_name, cron, "
                    "human_schedule, prompt, recurring, durable, status, created_by, "
                    "created_at, updated_at, next_fire_at, expires_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        job_id,
                        session_id,
                        agent_name,
                        cron,
                        human_schedule,
                        prompt,
                        1 if recurring else 0,
                        1 if durable else 0,
                        STATUS_RUNNING,
                        created_by,
                        now,
                        now,
                        next_fire_at,
                        expires_at,
                    ),
                )
                self._conn.commit()
            except Exception:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise
        self._append_event(job_id, "job_created", human_schedule)
        job = self.get_job(job_id)
        assert job is not None
        return job

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        rows = self._query(
            "SELECT * FROM scheduled_jobs WHERE job_id = ?",
            (str(job_id),),
        )
        return dict(rows[0]) if rows else None

    def list_jobs(self, include_inactive: bool = False) -> list[dict[str, Any]]:
        where = "" if include_inactive else "WHERE status = ?"
        params: tuple[Any, ...] = (STATUS_RUNNING,) if not include_inactive else ()
        rows = self._query(
            f"SELECT * FROM scheduled_jobs {where} ORDER BY created_at DESC",
            params,
        )
        return [dict(r) for r in rows]

    def record_job_fire(
        self,
        job_id: str,
        fired_at: str | None = None,
    ) -> dict[str, Any] | None:
        """Advance a job's fire anchors. Recurring jobs get the next cron
        time; one-shot (non-recurring) jobs are deleted after firing."""
        job = self.get_job(job_id)
        if job is None:
            return None
        now = _now_iso()
        fired_at = fired_at or now
        if not job["recurring"]:
            self.delete_job(job_id, reason="one_shot_fired")
            return None
        after = _dt.datetime.fromisoformat(fired_at)
        nxt = cron_next(job["cron"], after)
        next_fire_at = nxt.isoformat(timespec="seconds") if nxt else None
        self._execute(
            "UPDATE scheduled_jobs SET last_fired_at = ?, next_fire_at = ?, updated_at = ? WHERE job_id = ?",
            (fired_at, next_fire_at, now, job_id),
        )
        return self.get_job(job_id)

    def delete_job(self, job_id: str, *, reason: str | None = None) -> bool:
        cur = self._execute(
            "DELETE FROM scheduled_jobs WHERE job_id = ?",
            (job_id,),
        )
        if cur.rowcount:
            self._append_event(job_id, "job_deleted", reason)
        return cur.rowcount > 0

    def pause_jobs_for_session(
        self,
        session_id: str,
        *,
        reason: str | None = None,
    ) -> list[str]:
        """Pause every running job bound to ``session_id``; returns their ids.

        A paused job is out of ``list_jobs()``, so it neither fires nor
        re-arms at boot, until :meth:`resume_jobs_for_session`.
        """
        rows = self._query(
            "SELECT job_id FROM scheduled_jobs WHERE session_id = ? AND status = ?",
            (str(session_id), STATUS_RUNNING),
        )
        ids = [str(r["job_id"]) for r in rows]
        if not ids:
            return []
        self._execute(
            "UPDATE scheduled_jobs SET status = ?, updated_at = ?, "
            "supervisor_owner_id = NULL, supervisor_heartbeat_at = NULL "
            "WHERE session_id = ? AND status = ?",
            (STATUS_PAUSED, _now_iso(), str(session_id), STATUS_RUNNING),
        )
        for jid in ids:
            self._append_event(jid, "job_paused", reason)
        return ids

    def resume_jobs_for_session(self, session_id: str) -> list[str]:
        """Resume every paused job bound to ``session_id``; returns their ids.

        The next fire is the next cron match after now: fires missed while
        paused are skipped, never caught up.
        """
        rows = self._query(
            "SELECT job_id, cron FROM scheduled_jobs WHERE session_id = ? AND status = ?",
            (str(session_id), STATUS_PAUSED),
        )
        resumed: list[str] = []
        now = _now_iso()
        for r in rows:
            jid = str(r["job_id"])
            try:
                nxt = cron_next(r["cron"], _dt.datetime.fromisoformat(now))
            except Exception:  # noqa: BLE001 — a bad row stays paused
                continue
            cur = self._execute(
                "UPDATE scheduled_jobs SET status = ?, next_fire_at = ?, updated_at = ? "
                "WHERE job_id = ? AND status = ?",
                (
                    STATUS_RUNNING,
                    nxt.isoformat(timespec="seconds") if nxt else None,
                    now,
                    jid,
                    STATUS_PAUSED,
                ),
            )
            if cur.rowcount:
                self._append_event(jid, "job_resumed", None)
                resumed.append(jid)
        return resumed

    def try_claim_job(self, job_id: str, owner: str) -> bool:
        now = _now_iso()
        cutoff = _add_iso(now, -_SUPERVISOR_LEASE_S)
        cur = self._execute(
            "UPDATE scheduled_jobs SET supervisor_owner_id = ?, "
            "supervisor_heartbeat_at = ? WHERE job_id = ? AND status = ? "
            "AND (supervisor_owner_id IS NULL OR supervisor_owner_id = ? "
            "OR supervisor_heartbeat_at IS NULL "
            "OR julianday(supervisor_heartbeat_at) IS NULL "
            "OR julianday(supervisor_heartbeat_at) <= julianday(?))",
            (owner, now, job_id, STATUS_RUNNING, owner, cutoff),
        )
        return cur.rowcount > 0

    def touch_job(self, job_id: str, owner: str) -> bool:
        cur = self._execute(
            "UPDATE scheduled_jobs SET supervisor_heartbeat_at = ? "
            "WHERE job_id = ? AND supervisor_owner_id = ? AND status = ?",
            (_now_iso(), job_id, owner, STATUS_RUNNING),
        )
        return cur.rowcount > 0

    def release_job(self, job_id: str, owner: str) -> bool:
        cur = self._execute(
            "UPDATE scheduled_jobs SET supervisor_owner_id = NULL, "
            "supervisor_heartbeat_at = NULL "
            "WHERE job_id = ? AND supervisor_owner_id = ?",
            (job_id, owner),
        )
        return cur.rowcount > 0

    # -- cross-process session turn lease ---------------------------------------

    def try_claim_session_turn(self, session_id: str, owner: str) -> bool:
        """Atomically reserve a session for one scheduled turn."""
        now = _now_iso()
        cutoff = _add_iso(now, -_SUPERVISOR_LEASE_S)
        cur = self._execute(
            "INSERT INTO loop_session_leases "
            "(session_id, owner_id, heartbeat_at) VALUES (?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "owner_id = excluded.owner_id, heartbeat_at = excluded.heartbeat_at "
            "WHERE loop_session_leases.owner_id = excluded.owner_id "
            "OR loop_session_leases.heartbeat_at IS NULL "
            "OR julianday(loop_session_leases.heartbeat_at) IS NULL "
            "OR julianday(loop_session_leases.heartbeat_at) <= julianday(?)",
            (session_id, owner, now, cutoff),
        )
        return cur.rowcount > 0

    def touch_session_turn(self, session_id: str, owner: str) -> bool:
        cur = self._execute(
            "UPDATE loop_session_leases SET heartbeat_at = ? WHERE session_id = ? AND owner_id = ?",
            (_now_iso(), session_id, owner),
        )
        return cur.rowcount > 0

    def release_session_turn(self, session_id: str, owner: str) -> bool:
        cur = self._execute(
            "DELETE FROM loop_session_leases WHERE session_id = ? AND owner_id = ?",
            (session_id, owner),
        )
        return cur.rowcount > 0

    # -- wakeups (ScheduleWakeup) -------------------------------------------------

    def add_wakeup(
        self,
        *,
        session_id: str,
        agent_name: str,
        scheduled_for: str,
        loop_id: str | None = None,
    ) -> str:
        wakeup_id = new_wakeup_id()
        self._execute(
            "INSERT INTO wakeups (wakeup_id, session_id, agent_name, loop_id, "
            "scheduled_for, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (wakeup_id, session_id, agent_name, loop_id, scheduled_for, "pending", _now_iso()),
        )
        return wakeup_id

    def list_pending_wakeups(
        self,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if session_id:
            rows = self._query(
                "SELECT * FROM wakeups WHERE session_id = ? AND status = 'pending' ORDER BY scheduled_for ASC",
                (session_id,),
            )
        else:
            rows = self._query(
                "SELECT * FROM wakeups WHERE status = 'pending' ORDER BY scheduled_for ASC",
            )
        return [dict(r) for r in rows]

    def take_wakeup(self, wakeup_id: str) -> dict[str, Any] | None:
        """Atomically mark a wakeup fired; returns its row (or None)."""
        rows = self._query(
            "SELECT * FROM wakeups WHERE wakeup_id = ? AND status = 'pending'",
            (wakeup_id,),
        )
        if not rows:
            return None
        self._execute(
            "UPDATE wakeups SET status = 'fired', fired_at = ? WHERE wakeup_id = ?",
            (_now_iso(), wakeup_id),
        )
        return dict(rows[0])

    def fire_due_wakeups_for_loop(self, loop_id: str, through: str) -> int:
        """Mark every due pending wakeup for a loop as consumed."""
        cur = self._execute(
            "UPDATE wakeups SET status = 'fired', fired_at = ? "
            "WHERE loop_id = ? AND status = 'pending' "
            "AND julianday(scheduled_for) <= julianday(?)",
            (_now_iso(), loop_id, through),
        )
        return cur.rowcount

    def cancel_wakeups_for_session(self, session_id: str) -> int:
        """Cancel every pending wakeup for a session. Returns count."""
        rows = self.list_pending_wakeups(session_id)
        if not rows:
            return 0
        self._execute(
            "UPDATE wakeups SET status = 'cancelled', fired_at = ? WHERE session_id = ? AND status = 'pending'",
            (_now_iso(), session_id),
        )
        for r in rows:
            self._append_event(
                r.get("loop_id") or "session",
                "wakeup_cancelled",
                session_id,
            )
        return len(rows)

    def cancel_wakeups_for_loop(self, loop_id: str) -> int:
        rows = self._query(
            "SELECT * FROM wakeups WHERE loop_id = ? AND status = 'pending'",
            (loop_id,),
        )
        if not rows:
            return 0
        self._execute(
            "UPDATE wakeups SET status = 'cancelled', fired_at = ? WHERE loop_id = ? AND status = 'pending'",
            (_now_iso(), loop_id),
        )
        return len(rows)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_registry: LoopRegistry | None = None
_registry_lock = threading.Lock()


def get_loop_registry(db_path: Path | None = None) -> LoopRegistry:
    """Process-wide singleton registry (mirrors the goals singleton)."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = LoopRegistry(db_path)
    return _registry
