"""sharedagents_web.task_board — sqlite registry backing the web "Tasks" tab.

A task is nothing more than "a session the operator explicitly chose to
track from the Tasks tab". Topic, agent, AND live state are never
duplicated here — they're read live from the session store / agent tree
at request time, so a rename or topic change elsewhere never goes stale,
and Completed/Working/Needs-input are always current (same as the real
`claude agents` cloud list: Completed is the run finishing cleanly, not a
manual flag). Untracking (removing a row) never touches the underlying
session — it ARCHIVES the row (``archived_at``
stamp) instead of deleting it, so the operator can bring it back with
the "show archived" toggle.

``label`` is the name the operator typed in the "New Task" form.
For a labelled task the parent session's
title is PINNED to this label (``POST …/sessions/{sid}/title`` with
``lock: true``, see :mod:`sharedagents_web.task_recap`), so the title agent never
spends a side call on that parent and can never overwrite the operator's own
name for the task. Child sessions are untouched and keep their own automatic
titles. A task created WITHOUT a label keeps today's behaviour exactly
(``session.title`` stays a placeholder until the title agent fills it).

Completion recaps live in the same row: the ``summary_*`` columns hold the
recap text, the completion verdict (``complete`` / ``follow_up``) and the
single-flight claim/backoff bookkeeping that lets any number of web workers
run :mod:`sharedagents_web.task_recap` without double-spending a side call.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sharedagents_core.kernel.layout import tasks_db_path

ARCHIVED_TASK_LIMIT = 20

#: Additive recap/title-pin columns, applied on every connect so an old
#: tasks.db self-heals exactly like ``label``/``archived_at`` do.
#: ``summary_stamp`` = the parent's own ``last_turn_ts`` (epoch ms) at recap
#: time; ``summary_attempt_at``/``summary_failures`` drive the retry backoff;
#: ``label_pinned_at`` throttles drift re-pins of a board-created task title.
_RECAP_COLUMNS: tuple[tuple[str, str], ...] = (
    ("summary", "summary TEXT"),
    ("summary_verdict", "summary_verdict TEXT"),
    ("summary_follow_up", "summary_follow_up TEXT"),
    ("summary_status", "summary_status TEXT"),
    ("summary_stamp", "summary_stamp INTEGER"),
    ("summary_at", "summary_at TEXT"),
    ("summary_attempt_at", "summary_attempt_at TEXT"),
    ("summary_failures", "summary_failures INTEGER NOT NULL DEFAULT 0"),
    ("summary_version", "summary_version INTEGER"),
    ("label_pinned_at", "label_pinned_at TEXT"),
    ("label_pin_confirmed", "label_pin_confirmed INTEGER NOT NULL DEFAULT 0"),
    ("observed_card", "observed_card TEXT"),
    ("observed_turn_stamp", "observed_turn_stamp INTEGER"),
    ("completion_stamp", "completion_stamp INTEGER"),
)


class TaskBoardRegistry:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path or tasks_db_path())
        self._lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        # Re-create the parent dir + table on every connect (not just at
        # construction) so the registry self-heals if SHAREDAGENTS_HOME or the db
        # file gets pruned/relocated out from under a long-running process —
        # the same gotcha goal_registry.py guards against.
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(self.db_path), timeout=30.0)
        con.row_factory = sqlite3.Row
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS tracked_sessions (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                owner TEXT,
                created_at TEXT NOT NULL,
                label TEXT,
                archived_at TEXT
            )
            """
        )
        try:
            con.execute("ALTER TABLE tracked_sessions ADD COLUMN label TEXT")
        except sqlite3.OperationalError:
            pass  # column already present — this is the common case
        try:
            con.execute("ALTER TABLE tracked_sessions ADD COLUMN archived_at TEXT")
        except sqlite3.OperationalError:
            pass  # column already present — this is the common case
        # Task-completion recap (Tasks board). Additive and self-healing:
        # ``summary_stamp`` is the parent's own ``last_turn_ts`` at recap
        # time, so a recap re-fires only after a NEW parent turn (the recap
        # child's own turns never move the parent's stamp); ``summary_status``
        # is the single-flight claim ('' | queued | running | done | failed).
        for _column, _ddl in _RECAP_COLUMNS:
            try:
                con.execute(f"ALTER TABLE tracked_sessions ADD COLUMN {_ddl}")
            except sqlite3.OperationalError:
                pass  # column already present — this is the common case
        # Older parsers stored truncated JSON as a successful unknown recap.
        # Retain the original text for diagnosis, but clear coverage so it can
        # be retried. summary_fields hides the invalid fragment meanwhile.
        from sharedagents_session.session_recap import parse_recap_reply

        broken = con.execute(
            "SELECT id, summary FROM tracked_sessions WHERE summary_status='done'"
            " AND summary_verdict='unknown'"
            " AND (ltrim(summary) LIKE '{%' OR ltrim(summary) LIKE '```%')"
        ).fetchall()
        for row in broken:
            if not parse_recap_reply(row["summary"])["summary"]:
                con.execute(
                    "UPDATE tracked_sessions SET summary_status='failed', summary_stamp=NULL,"
                    " summary_attempt_at=NULL, summary_failures=0 WHERE id=?"
                    " AND summary_status='done' AND summary=?",
                    (row["id"], row["summary"]),
                )
        # A recap judged by an older prompt/digest (RECAP_VERSION) may carry
        # a wrong verdict: re-run each active one once. Re-arm observation as
        # well as coverage: legacy rows can lack a matching completion edge.
        # A queued row with no observation resumes on the next live completed
        # observation; working/needs-input rows still wait for completion.
        # Archived history and running claims are left alone.
        from sharedagents_session.session_recap import RECAP_VERSION

        stale = " WHERE archived_at IS NULL AND summary_status='done' AND COALESCE(summary_version,0) < ?"
        if con.execute("SELECT 1 FROM tracked_sessions" + stale + " LIMIT 1", (RECAP_VERSION,)).fetchone():
            con.execute(
                "UPDATE tracked_sessions SET summary_status='queued', summary_stamp=NULL,"
                " summary_attempt_at=NULL, summary_failures=0, observed_card=NULL,"
                " observed_turn_stamp=NULL, completion_stamp=NULL" + stale,
                (RECAP_VERSION,),
            )
        # The Tasks page is owner-scoped and has two deliberately separate
        # access paths: every active row, newest-created first; and only the
        # most recently archived rows, newest archive first.  Partial indexes
        # keep both reads proportional to what the UI can display even after
        # years of archived history have accumulated.
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_tracked_sessions_owner_active_created "
            "ON tracked_sessions(owner, created_at DESC) "
            "WHERE archived_at IS NULL"
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_tracked_sessions_owner_archived_at "
            "ON tracked_sessions(owner, archived_at DESC) "
            "WHERE archived_at IS NOT NULL"
        )
        return con

    def add(
        self,
        session_id: str,
        owner: str | None = None,
        label: str | None = None,
    ) -> str:
        row_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as con:
            con.execute(
                "INSERT INTO tracked_sessions (id, session_id, owner, created_at, label, observed_card) "
                "VALUES (?, ?, ?, ?, ?, 'unstarted')",
                (row_id, session_id, owner, now, label or None),
            )
        return row_id

    def find_by_session(
        self,
        session_id: str,
        owner: str | None = None,
    ) -> dict | None:
        """Newest row for this chat — active first, then archived."""
        sid = str(session_id or "").strip()
        if not sid:
            return None
        with self._lock, self._connect() as con:
            if owner:
                row = con.execute(
                    "SELECT * FROM tracked_sessions WHERE session_id = ? AND owner = ?"
                    " ORDER BY (archived_at IS NOT NULL), created_at DESC LIMIT 1",
                    (sid, owner),
                ).fetchone()
            else:
                row = con.execute(
                    "SELECT * FROM tracked_sessions WHERE session_id = ?"
                    " ORDER BY (archived_at IS NOT NULL), created_at DESC LIMIT 1",
                    (sid,),
                ).fetchone()
        return dict(row) if row else None

    def ensure(
        self,
        session_id: str,
        owner: str | None = None,
        label: str | None = None,
    ) -> str:
        """Track ``session_id`` once. Unarchive and refresh the label if present."""
        existing = self.find_by_session(session_id, owner)
        if not existing:
            return self.add(session_id, owner=owner, label=label)
        sets = ["archived_at = NULL"]
        args: list[object] = []
        if label:
            sets.append("label = ?")
            args.append(label)
        args.append(existing["id"])
        with self._lock, self._connect() as con:
            if owner:
                con.execute(
                    f"UPDATE tracked_sessions SET {', '.join(sets)} WHERE id = ? AND owner = ?",
                    (*args, owner),
                )
            else:
                con.execute(
                    f"UPDATE tracked_sessions SET {', '.join(sets)} WHERE id = ?",
                    args,
                )
        return str(existing["id"])

    def list(
        self,
        owner: str | None = None,
        include_archived: bool = False,
        *,
        archived_limit: int = ARCHIVED_TASK_LIMIT,
    ) -> list[dict]:
        archived_limit = max(0, min(int(archived_limit), ARCHIVED_TASK_LIMIT))
        with self._lock, self._connect() as con:
            if owner:
                rows = con.execute(
                    "SELECT * FROM tracked_sessions WHERE owner = ? AND archived_at IS NULL ORDER BY created_at DESC",
                    (owner,),
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT * FROM tracked_sessions WHERE archived_at IS NULL ORDER BY created_at DESC"
                ).fetchall()
            if include_archived and archived_limit:
                if owner:
                    archived = con.execute(
                        "SELECT * FROM tracked_sessions WHERE owner = ?"
                        " AND archived_at IS NOT NULL ORDER BY archived_at DESC LIMIT ?",
                        (owner, archived_limit),
                    ).fetchall()
                else:
                    archived = con.execute(
                        "SELECT * FROM tracked_sessions"
                        " WHERE archived_at IS NOT NULL ORDER BY archived_at DESC LIMIT ?",
                        (archived_limit,),
                    ).fetchall()
                rows = rows + archived
        return [dict(r) for r in rows]

    def remove(self, row_id: str, owner: str | None = None) -> bool:
        """Soft-remove: stamp ``archived_at`` so the row (and its session
        link) survives; only the default board view hides it. The session
        itself is NEVER deleted."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as con:
            if owner:
                cur = con.execute(
                    "UPDATE tracked_sessions SET archived_at = ? WHERE id = ? AND owner = ?",
                    (now, row_id, owner),
                )
            else:
                cur = con.execute(
                    "UPDATE tracked_sessions SET archived_at = ? WHERE id = ?",
                    (now, row_id),
                )
        return cur.rowcount > 0

    # --- task-completion recap ------------------------------------------

    def observe_cards(self, observations: list[tuple[str, str, int | None]]) -> dict[str, dict]:
        """Persist completion edges; first sight of an existing idle row is a baseline."""
        result = {}
        with self._lock, self._connect() as con:
            con.commit()
            con.execute("BEGIN IMMEDIATE")
            for row_id, card, stamp in observations:
                saved = con.execute("SELECT * FROM tracked_sessions WHERE id=?", (row_id,)).fetchone()
                if saved is None:
                    continue
                row = dict(saved)
                completion = row["completion_stamp"]
                if card != "completed":
                    completion = None
                elif stamp is not None:
                    if row["observed_card"] is not None and (
                        row["observed_card"] != "completed" or row["observed_turn_stamp"] != stamp
                    ):
                        completion = stamp
                    elif row["observed_card"] is None and row["summary_status"] in {"queued", "running", "failed"}:
                        completion = stamp  # resume an already triggered attempt
                if (row["observed_card"], row["observed_turn_stamp"], row["completion_stamp"]) != (
                    card,
                    stamp,
                    completion,
                ):
                    con.execute(
                        "UPDATE tracked_sessions SET observed_card=?, observed_turn_stamp=?, completion_stamp=? WHERE id=?",
                        (card, stamp, completion, row_id),
                    )
                row.update(observed_card=card, observed_turn_stamp=stamp, completion_stamp=completion)
                result[row_id] = row
        return result

    def archive_completed(self, stamps: dict[str, int], *, owner: str) -> list[str]:
        """Archive selected owned rows whose completed recaps are unchanged."""
        if not owner:
            return []
        now = datetime.now(timezone.utc).isoformat()
        archived = []
        with self._lock, self._connect() as con:
            for row_id, stamp in stamps.items():
                result = con.execute(
                    "UPDATE tracked_sessions SET archived_at=? WHERE id=? AND owner=?"
                    " AND archived_at IS NULL AND summary_status='done'"
                    " AND summary_verdict='complete' AND summary_stamp=?",
                    (now, row_id, owner, stamp),
                )
                if result.rowcount:
                    archived.append(row_id)
        return archived

    def claim_summary(
        self,
        row_id: str,
        *,
        now_iso: str,
        stale_before_iso: str,
        parent_turn_stamp: int | None = None,
        retry_before_iso: str | None = None,
        backoff_before_iso: str | None = None,
        require_transition: bool = False,
    ) -> bool:
        """Atomically take the recap for ``row_id`` — the single-flight gate.

        CONDITIONAL UPDATE, so two web workers (or a poll racing the
        sweeper) can never both spend a side call: exactly one sees
        ``rowcount == 1``. A row left ``running`` by a worker that died is
        reclaimable once its attempt stamp is older than
        ``stale_before_iso``; a previous ``failed`` run is always retryable
        (the due rule owns the backoff).
        """
        # The poll's snapshot can be stale even after another worker FINISHED.
        # Check coverage and retry spacing in this same conditional UPDATE.
        due_sql = ""
        due_args: tuple = ()
        if parent_turn_stamp is not None:
            due_sql = (
                " AND (summary_stamp IS NULL OR summary_stamp < ?)"
                " AND (summary_status='running' OR summary_attempt_at IS NULL"
                "      OR summary_attempt_at='' OR julianday(summary_attempt_at) <="
                "      julianday(CASE WHEN COALESCE(summary_failures,0)>=3"
                "                    THEN ? ELSE ? END))"
            )
            due_args = (parent_turn_stamp, backoff_before_iso, retry_before_iso)
        if require_transition:
            due_sql += " AND observed_card='completed' AND completion_stamp=?"
            due_args += (parent_turn_stamp,)
        with self._lock, self._connect() as con:
            cur = con.execute(
                "UPDATE tracked_sessions SET summary_status='running',"
                " summary_attempt_at=?"
                " WHERE id=? AND archived_at IS NULL"
                "   AND (COALESCE(summary_status,'') <> 'running'"
                "        OR COALESCE(summary_attempt_at,'') = ''"
                "        OR julianday(summary_attempt_at) IS NULL"
                "        OR julianday(summary_attempt_at) < julianday(?))" + due_sql,
                (now_iso, row_id, stale_before_iso, *due_args),
            )
        return cur.rowcount == 1

    def finish_summary(
        self,
        row_id: str,
        *,
        summary: str,
        verdict: str,
        follow_up: str,
        stamp: int | None,
        now_iso: str,
        claim_at: str | None = None,
    ) -> None:
        """Persist a completed recap and its verdict, kept verbatim."""
        from sharedagents_session.session_recap import RECAP_VERSION

        with self._lock, self._connect() as con:
            con.execute(
                "UPDATE tracked_sessions SET summary=?, summary_verdict=?,"
                " summary_follow_up=?, summary_status='done', summary_stamp=?,"
                " summary_at=?, summary_failures=0, summary_version=? WHERE id=?"
                " AND (? IS NULL OR (summary_status='running' AND summary_attempt_at=?))",
                (
                    summary or None,
                    verdict or None,
                    follow_up or None,
                    int(stamp) if stamp is not None else None,
                    now_iso,
                    RECAP_VERSION,
                    row_id,
                    claim_at,
                    claim_at,
                ),
            )

    def fail_summary(self, row_id: str, *, claim_at: str | None = None) -> None:
        """Record a failed attempt: keeps the previous recap text visible."""
        with self._lock, self._connect() as con:
            con.execute(
                "UPDATE tracked_sessions SET summary_status='failed',"
                " summary_failures=COALESCE(summary_failures,0)+1"
                " WHERE id=?"
                " AND (? IS NULL OR (summary_status='running' AND summary_attempt_at=?))",
                (row_id, claim_at, claim_at),
            )

    def set_label_pinned(self, row_id: str, now_iso: str, *, confirmed: bool = False) -> None:
        """Stamp the last title-pin attempt (throttles drift re-pins)."""
        with self._lock, self._connect() as con:
            con.execute(
                "UPDATE tracked_sessions SET label_pinned_at=?, label_pin_confirmed=? WHERE id=?",
                (now_iso, int(confirmed), row_id),
            )


_registry: TaskBoardRegistry | None = None
_registry_lock = threading.Lock()


def get_task_board_registry() -> TaskBoardRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = TaskBoardRegistry()
    return _registry


def reset_task_board_registry_for_tests() -> None:
    """Test-only hook: drop the process-wide singleton so a test that
    monkeypatches ``SHAREDAGENTS_HOME`` gets a fresh registry on the new path."""
    global _registry
    with _registry_lock:
        _registry = None
