"""Durable goal registry — goals outlive sessions.

Source of truth for goal entities lives in ``~/.sharedagents/goals/goals.db``.
``sessions_meta.goal_id`` is a projection for chat/engine compatibility.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sharedagents_core.layout import goals_db_path

_LOG = logging.getLogger("sharedagents.goal_registry")

_SUPERVISOR_LEASE_S = 120.0
_SUPERVISOR_START_GRACE_S = 30.0
_LEASE_RENEW_INTERVAL_S = 20.0
#: Completed (met) goals are removed after one week unless deleted sooner.
MAX_COMPLETED_AGE_S = 7 * 24 * 3600

STATUS_RUNNING = "running"
STATUS_PAUSED = "paused"
STATUS_MET = "met"
STATUS_BLOCKED = "blocked"
STATUS_ARCHIVED = "archived"
_LIVE_STATUSES = (STATUS_RUNNING, STATUS_PAUSED, STATUS_BLOCKED)
#: ``blocked`` + this reason: the goal evaluator judged the condition
#: impossible in this session (Claude Code: "goal cleared, recorded as failed").
REASON_IMPOSSIBLE = "impossible"
#: Pauses the runtime itself takes (Claude Code semantics). The goal stays set;
#: the user's next message in its session resumes it.
REASON_NO_PROGRESS = "no_progress"
REASON_EVALUATOR_UNAVAILABLE = "evaluator_unavailable"
REASON_TURN_ERRORS = "turn_errors"
#: The evaluator found the goal waiting on the owner (approval, a production
#: step, a value only the owner has). Nonterminal: the goal stays set and the
#: owner's next message resumes it; no automatic turn is admitted.
REASON_WAITING_OWNER = "waiting_owner"
AUTO_PAUSE_REASONS = (
    REASON_NO_PROGRESS,
    REASON_EVALUATOR_UNAVAILABLE,
    REASON_TURN_ERRORS,
    REASON_WAITING_OWNER,
)
# Statuses a goal may leave for each settle target. Completing a paused goal
# starts no work, so ``met`` is reachable from ``paused`` as well; a pause or a
# block still starts from ``running`` only, so a late one never relabels a
# goal that already settled. ``blocked`` → ``met`` happens only through
# ``mark_met(owner_confirmed=True)``: the owner marking the goal met by hand.
_SETTLE_SOURCES = {
    STATUS_PAUSED: (STATUS_RUNNING,),
    STATUS_BLOCKED: (STATUS_RUNNING,),
    STATUS_MET: (STATUS_RUNNING, STATUS_PAUSED),
}


def _iso_age_seconds(ts: str | None) -> float:
    if not ts:
        return float("inf")
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - dt).total_seconds()
    except Exception:  # noqa: BLE001
        return float("inf")


def _is_goal_starting(
    status: str,
    started_at: str | None,
    turn_count: int,
) -> bool:
    """Whether a RUNNING binding is still inside its first-driver grace.

    The registry transition is authoritative during this bounded window: a
    supervisor heartbeat may not exist yet, and a session terminal can belong
    to the prior goal cycle. After the first turn/window, normal terminal and
    zombie reconciliation resumes.
    """
    return (
        str(status or "") == STATUS_RUNNING
        and int(turn_count or 0) == 0
        and _iso_age_seconds(started_at) <= _SUPERVISOR_START_GRACE_S
    )


#: Depth cap for :func:`session_work_live`'s recursive subtree walk. A real
#: delegation tree is a handful of levels; the cap bounds the read so a
#: pathological (or corrupt, e.g. self-referential) parent chain can never
#: turn a liveness probe into a long scan.
_WORK_LIVE_MAX_DEPTH = 25


def _subtree_live_row(con: sqlite3.Connection, root_session_id: str, cutoff_ms: int, live_states) -> Any | None:
    """One row (if any) from the root's descendant subtree that is freshly
    live — the shared ``sessions_meta`` query behind both the full
    :func:`session_work_live` probe (supervisor/CLI) and the cheap read-path
    probe (:func:`session_work_live_cheap`), so the two can never disagree
    about what "still working" means."""
    ph = ",".join("?" for _ in live_states)
    return con.execute(
        "WITH RECURSIVE subtree(sid, depth) AS ("
        "  SELECT ?, 0"
        "  UNION ALL"
        "  SELECT m.session_hash, s.depth + 1"
        "  FROM sessions_meta m JOIN subtree s"
        "    ON m.parent_session_id = s.sid"
        "  WHERE s.depth < ?"
        ")"
        f"SELECT m.session_hash FROM sessions_meta m "
        f"JOIN subtree s ON m.session_hash = s.sid "
        f"WHERE m.session_state IN ({ph}) "
        f"AND m.session_state_updated_at >= ? LIMIT 1",
        (root_session_id, _WORK_LIVE_MAX_DEPTH, *live_states, cutoff_ms),
    ).fetchone()


def session_work_live(root_session_id: str) -> tuple[bool, str]:
    """Whether a session subtree still has LIVE work, independent of the goal
    supervisor heartbeat.

    The goal/loop supervisor heartbeat dies the moment its drive task exits
    (a manual-turn skip, a clean ``finally`` release, a crash) — but the work
    it delegated may still be genuinely running: background Agent children,
    open async jobs, a parent session folded to ``waiting``. This probe
    answers "is the work still live?" from the session index, using the SAME
    signals the Agent Monitor uses:

    * a session in the root's descendant subtree whose ``session_state`` is
      one of ``orchestrator.LIVE_STATES`` (``prompting`` / ``prefill`` /
      ``reasoning`` / ``streaming`` / ``tool_running`` / ``compacting`` /
      ``ask`` / ``paused`` / ``waiting``) and whose
      state register was touched within
      ``orchestrator.LIVE_FRESHNESS_SECONDS`` (300s) — the projector keeps a
      parent with live descendants in ``waiting`` and re-stamps its freshness
      on every child event, so the ROOT itself is usually the hit;
    * open (non-terminal) async jobs spawned by the root — covers the race
      where the child's ``session_started`` has not been folded into the
      index yet (same semantics as ``EventProjector._parent_still_has_work``).

    Returns ``(live, source)``; source is ``"root_session"`` /
    ``"descendant_session"`` / ``"open_jobs"`` (informational only, for the
    web ``liveness_source`` field) or ``""`` when not live. NEVER raises — a
    liveness probe must not break a read path, so every failure degrades to
    ``(False, "")``.

    Shared by the goal outer-loop supervisor
    (``GoalSupervisor._wait_while_subtree_live`` — holds auto-turns while
    children run), the web reconciler (``supervised_live`` in
    ``sharedagents_web.routes.goals``), the CLI/TUI status reconciler
    (``goal_service.reconcile_goal_view_status``) and the Loops page
    (``sharedagents_web.routes.loops._loop_live``) so every surface agrees that a
    goal/loop whose supervisor drive task exited — or whose root turn ended
    after dispatching background work — is still "working" while its
    children run.
    """
    if not root_session_id:
        return False, ""
    try:
        from sharedagents_agent.agents.async_jobs import open_jobs_for
        from sharedagents_agent.orchestrator import (
            LIVE_FRESHNESS_SECONDS,
            LIVE_STATES,
        )
        from sharedagents_core.layout import sessions_root
        from sharedagents_session.session_index import SessionIndex
    except Exception:  # noqa: BLE001 — a liveness probe must never raise
        return False, ""
    try:
        # Refresh first: a child written by a foreign process (subprocess
        # tool runner, external CLI agent) is invisible until sessions.db is
        # folded forward. Cheap when every cursor is already at EOF.
        idx = SessionIndex(sessions_root())
        try:
            idx.refresh()
        except Exception:  # noqa: BLE001
            pass
        cutoff_ms = int(time.time() * 1000) - LIVE_FRESHNESS_SECONDS * 1000
        con = idx._connect()
        try:
            row = _subtree_live_row(con, root_session_id, cutoff_ms, LIVE_STATES)
        finally:
            con.close()
        if row is not None:
            live_sid = row["session_hash"] if not isinstance(row, tuple) else row[0]
            source = "root_session" if str(live_sid) == root_session_id else "descendant_session"
            return True, source
    except Exception:  # noqa: BLE001 — a liveness probe must never raise
        pass
    try:
        if open_jobs_for(str(root_session_id)):
            return True, "open_jobs"
    except Exception:  # noqa: BLE001
        pass
    return False, ""


#: Short TTL for the read-path liveness snapshot (:func:`session_work_live_cheap`).
#: The probe itself is a single indexed ``sessions_meta`` read; the TTL only
#: collapses the overlapping Goals-page / detail / chat-footer pollers onto one
#: probe per window. Bounded staleness is fine for a display signal — the
#: supervisor/CLI surfaces keep using the full :func:`session_work_live`.
_WORK_LIVE_CACHE_TTL_S = 10.0
#: Hard cap on cached entries: a long-lived web process can otherwise grow the
#: dict with every unique root session ever listed. Over the cap the cache
#: drops expired entries first, then the oldest by stamp — so the bound holds
#: even under a burst of FRESH roots (expired-only eviction would not).
_WORK_LIVE_CACHE_MAX = 8192
_work_live_cache: dict[str, tuple[bool, str, str, float]] = {}
_work_live_cache_lock = threading.Lock()


def session_work_live_cheap(root_session_id: str) -> tuple[bool, str, bool, str]:
    """Cheap bounded liveness probe for web read paths (list/detail).

    Same subtree semantics as :func:`session_work_live` — a session in the
    root's descendant subtree whose ``session_state`` is one of
    ``orchestrator.LIVE_STATES`` and freshly stamped — but WITHOUT the
    SessionIndex construction (``ensure_schema`` migrations/DDL plus two
    COUNT scans on a large sessions.db), WITHOUT the full JSONL refresh and
    WITHOUT the async-jobs glob/read scan. The gateway's ``SessionWriter``
    projects events into ``sessions_meta`` in-line, so the index is current
    in the steady state and a plain indexed read answers the same question
    at millisecond cost. Foreign-process writers (subprocess tool runner,
    external CLI agent) may lag until the next full refresh — a bounded
    staleness acceptable for a display signal.

    Also returns the root session's projected ``goal_terminal`` (the session
    header's last-cycle terminal, folded in-line by the projector) so the web
    reconciler can show a settled goal at its terminal without loading an
    Agent or reading the session JSONL — the cheap half of zombie healing;
    the write itself happens in the web process's background heal pass.

    Returns ``(live, source, determinate, terminal)``:
    ``determinate=False`` means the index could not be read (missing
    db/table/error) OR the root row is absent from a readable index
    (foreign/stale metadata) — the caller must NOT claim "not live" from
    that and marks the goal unknown/degraded instead. ``source`` is
    ``"root_session"`` / ``"descendant_session"`` or ``""`` when not live;
    ``terminal`` is the root's projected ``goal_terminal`` (surfaced even
    when the subtree is live) or ``""`` when none/unknown. NEVER raises — a
    liveness probe must not break a read path.
    """
    if not root_session_id:
        return False, "", True, ""
    now = time.monotonic()
    with _work_live_cache_lock:
        hit = _work_live_cache.get(root_session_id)
        if hit is not None and now - hit[3] < _WORK_LIVE_CACHE_TTL_S:
            return hit[0], hit[1], True, hit[2]
    live, source, determinate, terminal = _work_live_cheap_uncached(root_session_id)
    if determinate:
        with _work_live_cache_lock:
            _work_live_cache[root_session_id] = (live, source, terminal, time.monotonic())
            if len(_work_live_cache) > _WORK_LIVE_CACHE_MAX:
                # Enforce a TRUE bound: drop expired entries, then the oldest
                # by stamp, until the dict is at/below the cap — a burst of
                # fresh roots must not grow the cache unboundedly.
                _cutoff = time.monotonic() - _WORK_LIVE_CACHE_TTL_S
                for _key, _entry in list(_work_live_cache.items()):
                    if _entry[3] < _cutoff:
                        del _work_live_cache[_key]
                while len(_work_live_cache) > _WORK_LIVE_CACHE_MAX:
                    _oldest = min(
                        _work_live_cache.items(),
                        key=lambda _kv: _kv[1][3],
                    )[0]
                    del _work_live_cache[_oldest]
    return live, source, determinate, terminal


def _work_live_cheap_uncached(root_session_id: str) -> tuple[bool, str, bool, str]:
    try:
        from sharedagents_agent.orchestrator import (
            LIVE_FRESHNESS_SECONDS,
            LIVE_STATES,
        )
        from sharedagents_core.layout import sessions_root
        from sharedagents_session.session_index import connect_sessions_db
    except Exception:  # noqa: BLE001 — a liveness probe must never raise
        return False, "", False, ""
    db_path = sessions_root() / "sessions.db"
    if not db_path.exists():
        return False, "", False, ""
    try:
        con = connect_sessions_db(db_path)
    except Exception:  # noqa: BLE001
        return False, "", False, ""
    try:
        cutoff_ms = int(time.time() * 1000) - LIVE_FRESHNESS_SECONDS * 1000
        row = _subtree_live_row(con, root_session_id, cutoff_ms, LIVE_STATES)
        try:
            trow = con.execute(
                "SELECT goal_terminal FROM sessions_meta WHERE session_hash=?",
                (root_session_id,),
            ).fetchone()
        except Exception:  # noqa: BLE001
            return False, "", False, ""
        terminal = str(trow["goal_terminal"] or "") if trow is not None else ""
        if row is not None:
            live_sid = row["session_hash"] if not isinstance(row, tuple) else row[0]
            source = "root_session" if str(live_sid) == root_session_id else "descendant_session"
            return True, source, True, terminal
        if trow is None:
            # Readable index but the root row is ABSENT — foreign/stale
            # metadata. Never claim "not live" determinately from that; the
            # caller surfaces liveness_unknown instead.
            return False, "", False, ""
        return False, "", True, terminal
    except Exception:  # noqa: BLE001 — a liveness probe must never raise
        return False, "", False, ""
    finally:
        con.close()


#: Durable session states that mean "this session is over" from the goal
#: supervisor's point of view: explicitly closed by the operator/close path,
#: or its last turn ended in error/abort (the projector folds an aborted
#: ``turn_ended`` into state ``"error"``). Ordinary ``idle`` is NOT ended —
#: an idle-running root is exactly what restart recovery should reclaim.
#: (TIER 1d / E1)
SESSION_ENDED_STATES: frozenset[str] = frozenset({"closed", "error"})


def root_session_state(session_id: str) -> dict[str, Any] | None:
    """Read the root session's durable state register (``sessions_meta``).

    E1 — supervisor session awareness: gives the drive loop a real view of
    the root session lifecycle (state machine value, last activity, compaction
    count) rather than only the subtree-liveness probe. Returns a dict with
    ``session_state`` / ``session_state_updated_at`` / ``last_turn_ts`` /
    ``compaction_count`` / ``last_prompt_tokens``, or ``None`` when the row
    does not exist (a goal rooted on a vanished session). A transient
    register/SQLite failure returns a ``probe_error`` marker instead of
    ``None`` so recovery does not turn an infrastructure read failure into a
    durable goal pause. NEVER raises.
    """
    if not session_id:
        return None
    try:
        from sharedagents_core.layout import sessions_root
        from sharedagents_session.session_index import SessionIndex

        idx = SessionIndex(sessions_root())
        try:
            idx.refresh()
        except Exception:  # noqa: BLE001
            pass
        con = idx._connect()
        try:
            row = con.execute(
                "SELECT session_state, session_state_updated_at, last_turn_ts, "
                "compaction_count, last_prompt_tokens "
                "FROM sessions_meta WHERE session_hash = ?",
                (session_id,),
            ).fetchone()
        finally:
            con.close()
        if row is None:
            return None
        return {
            "session_state": str(row["session_state"] or ""),
            "session_state_updated_at": int(row["session_state_updated_at"] or 0),
            "last_turn_ts": int(row["last_turn_ts"] or 0),
            "compaction_count": int(row["compaction_count"] or 0),
            "last_prompt_tokens": int(row["last_prompt_tokens"] or 0),
        }
    except Exception as exc:  # noqa: BLE001 — a state probe must never raise
        # Missing rows are represented by None above. Keep probe failures
        # distinct: recovery must not pause a live goal merely because the
        # session index was temporarily unavailable or locked.
        _LOG.debug("root session state probe failed for %s: %s", session_id, exc)
        return {"session_state": "", "probe_error": True}


VALID_STATUSES = frozenset(
    {
        STATUS_RUNNING,
        STATUS_PAUSED,
        STATUS_MET,
        STATUS_BLOCKED,
        STATUS_ARCHIVED,
    }
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS goals (
    goal_id TEXT PRIMARY KEY,
    condition_text TEXT NOT NULL,
    normalized_condition_text TEXT NOT NULL,
    condition_hash TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    scope_key TEXT,
    status TEXT NOT NULL,
    root_session_id TEXT,
    created_from_session_id TEXT,
    created_by TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    paused_at TEXT,
    completed_at TEXT,
    archived_at TEXT,
    turn_count INTEGER NOT NULL DEFAULT 0,
    max_turns INTEGER NOT NULL DEFAULT 0,
    cost_pico INTEGER NOT NULL DEFAULT 0,
    max_cost_pico INTEGER NOT NULL DEFAULT 0,
    terminal_reason TEXT,
    terminal_summary TEXT,
    conflict_override INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    supervisor_owner_id TEXT,
    supervisor_heartbeat_at TEXT,
    -- Effective-progress ledger (TIER 1a): written by the goal supervisor
    -- after every supervised turn so the no-progress streak breaker can
    -- survive a supervisor restart. ``last_turn_fingerprint`` is the
    -- normalized hash of (tool name, action, success/error class); identical
    -- consecutive fingerprints are NOT progress (TIER 3b).
    no_effective_progress_streak INTEGER NOT NULL DEFAULT 0,
    last_effective_progress_at TEXT,
    last_turn_fingerprint TEXT,
    -- Goal evaluator state (JSON): last verdict + reason, evaluation count,
    -- runtime pause notice. See goal_evaluator.py.
    contract_json TEXT
);

CREATE TABLE IF NOT EXISTS goal_root_session_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    goal_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    bound_at TEXT NOT NULL,
    unbound_at TEXT,
    unbind_reason TEXT,
    FOREIGN KEY(goal_id) REFERENCES goals(goal_id)
);

CREATE TABLE IF NOT EXISTS goal_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    goal_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(goal_id) REFERENCES goals(goal_id)
);

CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status);
CREATE INDEX IF NOT EXISTS idx_goals_agent_status ON goals(agent_name, status);
CREATE INDEX IF NOT EXISTS idx_goals_condition_agent ON goals(agent_name, condition_hash);
CREATE INDEX IF NOT EXISTS idx_goals_scope_status ON goals(scope_key, status);
CREATE INDEX IF NOT EXISTS idx_goals_root_status ON goals(root_session_id, status);

CREATE UNIQUE INDEX IF NOT EXISTS idx_goals_one_running_root_session
ON goals(root_session_id)
WHERE status = 'running' AND root_session_id IS NOT NULL;

-- goal_events is append-only and grows large (the two biggest goals on the
-- operator box each carry ~250k rows); the detail/timeline read filters by
-- goal_id and orders by id DESC, so a composite index turns the per-open
-- whole-table SCAN (524k rows / 71MB measured 2026-09-09) into a per-goal
-- b-tree seek — every session click that hydrates a goal detail and every
-- polled /api/goals/<id>/events read pays it. Mirrors the loop registry's
-- idx_loop_events_loop shape.
CREATE INDEX IF NOT EXISTS idx_goal_events_goal
ON goal_events(goal_id, id DESC);
-- Same read shape on the root-session binding log (get_detail opens both).
CREATE INDEX IF NOT EXISTS idx_goal_root_history_goal
ON goal_root_session_history(goal_id, id);
"""


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_condition_text(text: str) -> str:
    raw = (text or "").strip()
    return re.sub(r"\s+", " ", raw)


def condition_hash(text: str) -> str:
    norm = normalize_condition_text(text).lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def new_goal_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"goal_{ts}_{secrets.token_hex(4)}"


def _load_contract(raw: Any) -> dict:
    if not raw:
        return {}
    try:
        val = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return val if isinstance(val, dict) else {}


@dataclass
class GoalConflict:
    goal_id: str
    agent_name: str
    status: str
    condition_text: str
    root_session_id: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "agent_name": self.agent_name,
            "status": self.status,
            "condition_text": self.condition_text,
            "root_session_id": self.root_session_id,
            "reason": self.reason,
        }


class GoalContractChangedError(RuntimeError):
    """A concurrent edit invalidated the checks or claim being committed."""


class GoalConflictError(Exception):
    def __init__(self, conflicts: list[GoalConflict]) -> None:
        self.conflicts = conflicts
        super().__init__(f"goal conflict: {len(conflicts)} existing goal(s)")


@dataclass
class GoalRecord:
    goal_id: str
    condition_text: str
    normalized_condition_text: str
    condition_hash: str
    agent_name: str
    scope_key: str | None
    status: str
    root_session_id: str | None
    created_from_session_id: str | None
    created_by: str | None
    created_at: str
    updated_at: str
    started_at: str | None
    paused_at: str | None
    completed_at: str | None
    archived_at: str | None
    turn_count: int
    # ``max_turns`` is a display/back-compat column ONLY — it never terminates
    # the outer loop (the loop is cost-terminated; see ``max_cost_pico`` below
    # and goal_supervisor._drive_loop). Retained for schema/UI back-compat.
    # An opt-in turn cap was rejected — owner override 2026-07-11
    # "only cost limit no turn limit".
    max_turns: int
    terminal_reason: str | None
    terminal_summary: str | None
    conflict_override: bool
    last_error: str | None
    supervisor_owner_id: str | None
    supervisor_heartbeat_at: str | None
    # Cost budget (USD as integer pico-USD). ``cost_pico`` is a display cache
    # reconcilable from the usage ledger; ``max_cost_pico`` is the budget that
    # terminates the outer loop (0 == unbounded). Defaults keep positional
    # construction (older call sites / tests) working.
    cost_pico: int = 0
    max_cost_pico: int = 0
    # TIER 1a — effective-progress ledger (written by the goal supervisor).
    no_effective_progress_streak: int = 0
    last_effective_progress_at: str | None = None
    last_turn_fingerprint: str | None = None
    # Goal evaluator state (``contract_json``): last verdict and reason,
    # evaluation count, the notice of a runtime pause. See goal_evaluator.py.
    contract: dict = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> GoalRecord:
        keys = set(row.keys())
        return cls(
            goal_id=row["goal_id"],
            condition_text=row["condition_text"],
            normalized_condition_text=row["normalized_condition_text"],
            condition_hash=row["condition_hash"],
            agent_name=row["agent_name"],
            scope_key=row["scope_key"],
            status=row["status"],
            root_session_id=row["root_session_id"],
            created_from_session_id=row["created_from_session_id"],
            created_by=row["created_by"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            started_at=row["started_at"],
            paused_at=row["paused_at"],
            completed_at=row["completed_at"],
            archived_at=row["archived_at"],
            turn_count=int(row["turn_count"] or 0),
            max_turns=int(row["max_turns"] or 0),
            cost_pico=int((row["cost_pico"] if "cost_pico" in keys else 0) or 0),
            max_cost_pico=int(
                (row["max_cost_pico"] if "max_cost_pico" in keys else 0) or 0,
            ),
            terminal_reason=row["terminal_reason"],
            terminal_summary=row["terminal_summary"],
            conflict_override=bool(row["conflict_override"]),
            last_error=row["last_error"],
            supervisor_owner_id=row["supervisor_owner_id"],
            supervisor_heartbeat_at=row["supervisor_heartbeat_at"],
            no_effective_progress_streak=int(
                (row["no_effective_progress_streak"] if "no_effective_progress_streak" in keys else 0) or 0,
            ),
            last_effective_progress_at=(
                row["last_effective_progress_at"] if "last_effective_progress_at" in keys else None
            ),
            last_turn_fingerprint=(row["last_turn_fingerprint"] if "last_turn_fingerprint" in keys else None),
            contract=_load_contract(
                row["contract_json"] if "contract_json" in keys else None,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "condition_text": self.condition_text,
            "condition_hash": self.condition_hash,
            "agent_name": self.agent_name,
            "scope_key": self.scope_key,
            "status": self.status,
            "root_session_id": self.root_session_id,
            "created_from_session_id": self.created_from_session_id,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "paused_at": self.paused_at,
            "completed_at": self.completed_at,
            "archived_at": self.archived_at,
            "turn_count": self.turn_count,
            "max_turns": self.max_turns,
            "cost_pico": self.cost_pico,
            "max_cost_pico": self.max_cost_pico,
            "cost_usd": self.cost_pico / 1_000_000_000_000,
            "max_cost_usd": self.max_cost_pico / 1_000_000_000_000,
            "terminal_reason": self.terminal_reason,
            "terminal_summary": self.terminal_summary,
            "conflict_override": self.conflict_override,
            "last_error": self.last_error,
            "supervisor_owner_id": self.supervisor_owner_id,
            "supervisor_heartbeat_at": self.supervisor_heartbeat_at,
            "last_activity_at": self.updated_at,
            "no_effective_progress_streak": self.no_effective_progress_streak,
            "last_effective_progress_at": self.last_effective_progress_at,
            "last_turn_fingerprint": self.last_turn_fingerprint,
            "contract": self.contract,
            "wait_label": self.wait_label,
            "last_check": str((self.contract or {}).get("last_check") or ""),
        }

    @property
    def wait_label(self) -> str:
        """Why a goal stopped, as the owner should read it ("" while it runs
        or when it was met). Frontends render this verbatim."""
        if self.status == STATUS_BLOCKED and self.terminal_reason == REASON_IMPOSSIBLE:
            why = (self.terminal_summary or "").strip()
            return f"not achievable: {why}" if why else "not achievable"
        if self.status == STATUS_PAUSED and self.terminal_reason in AUTO_PAUSE_REASONS:
            return str((self.contract or {}).get("notice") or "paused — send a message to continue")
        return ""


@dataclass
class GoalFilters:
    status: str | None = None
    agent_name: str | None = None
    exclude_goal_id: str | None = None
    root_session_id: str | None = None


class GoalRegistry:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path or goals_db_path())
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        # Re-ensure the store on every connect, not just at construction. A
        # GoalRegistry (notably the module-level ``_registry`` singleton) can
        # outlive the directory it was built against — e.g. a test's tmp
        # SHAREDAGENTS_HOME gets cleaned up, or the tree is pruned — after which
        # ``sqlite3.connect`` fails with "unable to open database file" and the
        # web ``/api/goals`` handler 500s. Recreate the parent dir here, and if
        # the DB file itself was lost with the tree, re-apply the schema so the
        # reopened (empty) DB has its tables — otherwise the next query would
        # merely trade the open error for "no such table". Makes connect fully
        # self-healing under an arbitrary/transient SHAREDAGENTS_HOME.
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        fresh = not self.db_path.exists()
        con = sqlite3.connect(str(self.db_path), timeout=30.0)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA busy_timeout=15000")
        if fresh:
            con.executescript(SCHEMA_SQL)
            self._migrate_columns(con)
            con.commit()
        return con

    def _ensure_schema(self) -> None:
        with self._connect() as con:
            con.executescript(SCHEMA_SQL)
            self._migrate_columns(con)
            con.execute("DROP INDEX IF EXISTS idx_goals_one_running_condition")
            con.execute("DROP INDEX IF EXISTS idx_goals_one_running_scope")
            self._purge_archived_locked(con)
            self._expire_due_locked(con)
            con.commit()

    @staticmethod
    def _migrate_columns(con: sqlite3.Connection) -> None:
        """Additive migrations for goal DBs created before a column existed.
        ``CREATE TABLE IF NOT EXISTS`` never alters an existing table, so the
        cost-budget columns must be added explicitly on older DBs."""
        cols = {r["name"] for r in con.execute("PRAGMA table_info(goals)").fetchall()}
        if "cost_pico" not in cols:
            con.execute("ALTER TABLE goals ADD COLUMN cost_pico INTEGER NOT NULL DEFAULT 0")
        if "max_cost_pico" not in cols:
            con.execute("ALTER TABLE goals ADD COLUMN max_cost_pico INTEGER NOT NULL DEFAULT 0")
        # TIER 1a — effective-progress ledger columns (additive/backward-compatible).
        if "no_effective_progress_streak" not in cols:
            con.execute("ALTER TABLE goals ADD COLUMN no_effective_progress_streak INTEGER NOT NULL DEFAULT 0")
        if "last_effective_progress_at" not in cols:
            con.execute("ALTER TABLE goals ADD COLUMN last_effective_progress_at TEXT")
        if "last_turn_fingerprint" not in cols:
            con.execute("ALTER TABLE goals ADD COLUMN last_turn_fingerprint TEXT")
        if "contract_json" not in cols:
            con.execute("ALTER TABLE goals ADD COLUMN contract_json TEXT")

    def _append_event(
        self,
        con: sqlite3.Connection,
        goal_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Append one immutable ``goal_events`` row — the sole writer of that
        table, emitting exactly one row per goal transition (``created``,
        ``paused``, ``met``, ``blocked``, ``archived``, …). No update/delete
        path exists, so the transition log stays append-only.
        """
        con.execute(
            "INSERT INTO goal_events (goal_id, event_type, payload_json, created_at) VALUES (?, ?, ?, ?)",
            (goal_id, event_type, json.dumps(payload or {}), _utcnow_iso()),
        )

    def get(self, goal_id: str) -> GoalRecord | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM goals WHERE goal_id = ?",
                (goal_id,),
            ).fetchone()
        return GoalRecord.from_row(row) if row else None

    def list_goals(self, filters: GoalFilters | None = None) -> list[GoalRecord]:
        self.expire_due()
        filters = filters or GoalFilters()
        clauses: list[str] = []
        params: list[Any] = []
        if filters.status:
            clauses.append("status = ?")
            params.append(filters.status)
        else:
            clauses.append("status != ?")
            params.append(STATUS_ARCHIVED)
        if filters.agent_name:
            clauses.append("agent_name = ?")
            params.append(filters.agent_name)
        if filters.exclude_goal_id:
            clauses.append("goal_id != ?")
            params.append(filters.exclude_goal_id)
        if filters.root_session_id:
            clauses.append("root_session_id = ?")
            params.append(filters.root_session_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as con:
            rows = con.execute(
                f"SELECT * FROM goals {where} ORDER BY updated_at DESC",
                params,
            ).fetchall()
        return [GoalRecord.from_row(r) for r in rows]

    def chat_goal_id_for_root(self, sid: str) -> str:
        """One indexed lookup for the chat widget — never scan the whole table."""
        sid = str(sid or "").strip()
        if not sid:
            return ""
        with self._connect() as con:
            row = con.execute(
                "SELECT goal_id FROM goals WHERE root_session_id = ? AND status = 'running' LIMIT 1",
                (sid,),
            ).fetchone()
            if row:
                return str(row["goal_id"] or "")
            row = con.execute(
                "SELECT goal_id FROM goals WHERE root_session_id = ? "
                "AND status IN ('paused', 'blocked') "
                "ORDER BY CASE status "
                "WHEN 'paused' THEN 0 WHEN 'blocked' THEN 1 "
                "ELSE 2 END, updated_at DESC LIMIT 1",
                (sid,),
            ).fetchone()
            return str(row["goal_id"] or "") if row else ""

    def list_events(self, goal_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as con:
            return self._events_for(con, goal_id, limit=limit)

    @staticmethod
    def _events_for(con: sqlite3.Connection, goal_id: str, *, limit: int) -> list[dict[str, Any]]:
        rows = con.execute(
            "SELECT * FROM goal_events WHERE goal_id = ? ORDER BY id DESC LIMIT ?",
            (goal_id, limit),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            try:
                payload = json.loads(row["payload_json"] or "{}")
            except json.JSONDecodeError:
                payload = {}
            out.append(
                {
                    "id": row["id"],
                    "goal_id": row["goal_id"],
                    "event_type": row["event_type"],
                    "payload": payload,
                    "created_at": row["created_at"],
                }
            )
        return out

    def get_detail(self, goal_id: str) -> tuple[GoalRecord, list[dict[str, Any]], list[dict[str, Any]]] | None:
        """Read a goal plus its recent events and root-session history on ONE
        connection (the web detail route used to open three). Returns
        ``(goal, events, root_history)`` or ``None`` when the goal is absent.
        Read-only."""
        with self._connect() as con:
            row = con.execute("SELECT * FROM goals WHERE goal_id = ?", (goal_id,)).fetchone()
            if row is None:
                return None
            goal = GoalRecord.from_row(row)
            events = self._events_for(con, goal_id, limit=50)
            history = self._root_history_for(con, goal_id)
        return goal, events, history

    def list_root_history(self, goal_id: str) -> list[dict[str, Any]]:
        with self._connect() as con:
            return self._root_history_for(con, goal_id)

    @staticmethod
    def _root_history_for(con: sqlite3.Connection, goal_id: str) -> list[dict[str, Any]]:
        rows = con.execute(
            "SELECT * FROM goal_root_session_history WHERE goal_id = ? ORDER BY id ASC",
            (goal_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def detect_conflicts(
        self,
        *,
        condition_text: str,
        agent_name: str,
        scope_key: str | None = None,
        exclude_goal_id: str | None = None,
        force: bool = False,
        strict_one_per_agent: bool = False,
    ) -> list[GoalConflict]:
        """Same-text / scope conflicts are not enforced."""
        return []

    def create_goal(
        self,
        *,
        condition_text: str,
        agent_name: str,
        root_session_id: str,
        created_by: str | None = None,
        created_from_session_id: str | None = None,
        max_turns: int = 0,
        max_cost_pico: int = 0,
        scope_key: str | None = None,
        force: bool = False,
        replace_existing: bool = True,
    ) -> GoalRecord:
        """Insert a new running goal on ``root_session_id``.

        One live goal per session: any running/paused/blocked goal already
        bound to this session is deleted first. Same condition text on
        another session is allowed. ``force`` is accepted for API compat and
        ignored. Model callers use ``replace_existing=False`` so an owner
        setting a goal concurrently wins without having their goal deleted.
        """
        norm = normalize_condition_text(condition_text)
        ch = condition_hash(condition_text)
        now = _utcnow_iso()
        goal_id = new_goal_id()
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            if not replace_existing:
                row = con.execute(
                    "SELECT * FROM goals WHERE root_session_id = ? AND status IN (?, ?, ?)",
                    (root_session_id, *_LIVE_STATUSES),
                ).fetchone()
                if row is not None:
                    raise GoalConflictError(
                        [
                            GoalConflict(
                                goal_id=row["goal_id"],
                                agent_name=row["agent_name"],
                                status=row["status"],
                                condition_text=row["condition_text"],
                                root_session_id=root_session_id,
                                reason="session_has_live_goal",
                            )
                        ]
                    )
            self._replace_live_on_session_locked(con, root_session_id)
            try:
                con.execute(
                    """
                    INSERT INTO goals (
                        goal_id, condition_text, normalized_condition_text,
                        condition_hash, agent_name, scope_key, status,
                        root_session_id, created_from_session_id, created_by,
                        created_at, updated_at, started_at, turn_count, max_turns,
                        max_cost_pico, conflict_override
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
                    """,
                    (
                        goal_id,
                        condition_text,
                        norm,
                        ch,
                        agent_name,
                        scope_key,
                        STATUS_RUNNING,
                        root_session_id,
                        created_from_session_id or root_session_id,
                        created_by,
                        now,
                        now,
                        now,
                        max_turns,
                        int(max_cost_pico or 0),
                        0,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                self._replace_live_on_session_locked(con, root_session_id)
                try:
                    con.execute(
                        """
                        INSERT INTO goals (
                            goal_id, condition_text, normalized_condition_text,
                            condition_hash, agent_name, scope_key, status,
                            root_session_id, created_from_session_id, created_by,
                            created_at, updated_at, started_at, turn_count, max_turns,
                            max_cost_pico, conflict_override
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
                        """,
                        (
                            goal_id,
                            condition_text,
                            norm,
                            ch,
                            agent_name,
                            scope_key,
                            STATUS_RUNNING,
                            root_session_id,
                            created_from_session_id or root_session_id,
                            created_by,
                            now,
                            now,
                            now,
                            max_turns,
                            int(max_cost_pico or 0),
                            0,
                        ),
                    )
                except sqlite3.IntegrityError as exc2:
                    con.rollback()
                    raise GoalConflictError(
                        [
                            GoalConflict(
                                goal_id="",
                                agent_name=agent_name,
                                status=STATUS_RUNNING,
                                condition_text=condition_text,
                                root_session_id=root_session_id,
                                reason=f"integrity:{exc2}",
                            )
                        ]
                    ) from exc
            con.execute(
                """
                INSERT INTO goal_root_session_history
                (goal_id, session_id, agent_name, bound_at)
                VALUES (?, ?, ?, ?)
                """,
                (goal_id, root_session_id, agent_name, now),
            )
            self._append_event(
                con,
                goal_id,
                "created",
                {
                    "root_session_id": root_session_id,
                    "force": force,
                },
            )
            con.commit()
        rec = self.get(goal_id)
        assert rec is not None
        return rec

    def bind_root_session(
        self,
        *,
        goal_id: str,
        new_root_session_id: str,
        agent_name: str,
        reason: str,
        force: bool = False,
        expected_pause_reasons: tuple[str, ...] = (),
    ) -> GoalRecord:
        """Rebind a goal onto a new root session, closing out the previous
        binding in the audit history rather than overwriting it. Any other
        live goal already on the target session is deleted first. ``force``
        is accepted for API compat and ignored.
        """
        now = _utcnow_iso()
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT * FROM goals WHERE goal_id = ?",
                (goal_id,),
            ).fetchone()
            if row is None:
                con.rollback()
                raise KeyError(f"goal not found: {goal_id}")
            goal = GoalRecord.from_row(row)
            if expected_pause_reasons and (
                goal.status != STATUS_PAUSED
                or goal.terminal_reason not in expected_pause_reasons
                or goal.root_session_id != new_root_session_id
            ):
                raise GoalContractChangedError("goal is no longer paused for this session")
            self._replace_live_on_session_locked(
                con,
                new_root_session_id,
                exclude_goal_id=goal_id,
            )
            con.execute(
                """
                UPDATE goal_root_session_history
                SET unbound_at = ?, unbind_reason = ?
                WHERE goal_id = ? AND unbound_at IS NULL
                """,
                (now, reason, goal_id),
            )
            con.execute(
                """
                INSERT INTO goal_root_session_history
                (goal_id, session_id, agent_name, bound_at)
                VALUES (?, ?, ?, ?)
                """,
                (goal_id, new_root_session_id, agent_name, now),
            )
            con.execute(
                """
                UPDATE goals
                SET root_session_id = ?, status = ?, updated_at = ?,
                    started_at = COALESCE(started_at, ?), paused_at = NULL,
                    terminal_reason = NULL, terminal_summary = NULL,
                    completed_at = NULL
                WHERE goal_id = ?
                """,
                (new_root_session_id, STATUS_RUNNING, now, now, goal_id),
            )
            # Rebinding, including resume onto the same root, invalidates
            # in-flight evaluations and cached verdicts from the old cycle.
            state = dict(goal.contract)
            state["evaluation_epoch"] = int(state.get("evaluation_epoch", 0)) + 1
            for key in ("verdict", "evaluated_turns", "evaluated_transcript_sha256", "notice"):
                state.pop(key, None)
            con.execute("UPDATE goals SET contract_json = ? WHERE goal_id = ?", (json.dumps(state), goal_id))
            self._append_event(
                con,
                goal_id,
                "root_rebound",
                {
                    "new_root_session_id": new_root_session_id,
                    "reason": reason,
                    "force": force,
                },
            )
            con.commit()
        rec = self.get(goal_id)
        assert rec is not None
        return rec

    def update_turn_count(self, goal_id: str, turn_count: int) -> None:
        now = _utcnow_iso()
        with self._connect() as con:
            con.execute(
                "UPDATE goals SET turn_count = ?, updated_at = ? WHERE goal_id = ?",
                (turn_count, now, goal_id),
            )
            con.commit()

    def goal_id_for_root_session(self, session_id: str) -> str | None:
        """The running goal bound to ``session_id`` as its root, if any. Used to
        derive a delegated child's ancestor goal from its parent session id."""
        if not session_id:
            return None
        with self._connect() as con:
            row = con.execute(
                "SELECT goal_id FROM goals WHERE root_session_id = ? AND status = ? ORDER BY updated_at DESC LIMIT 1",
                (session_id, STATUS_RUNNING),
            ).fetchone()
        return row["goal_id"] if row else None

    def update_cost_cache(self, goal_id: str, cost_pico: int) -> None:
        """Refresh the display cache of accumulated spend. Authoritative spend
        is the usage-ledger SUM; this is only for cheap status rendering."""
        now = _utcnow_iso()
        with self._connect() as con:
            con.execute(
                "UPDATE goals SET cost_pico = ?, updated_at = ? WHERE goal_id = ?",
                (int(cost_pico or 0), now, goal_id),
            )
            con.commit()

    def update_effective_progress(
        self,
        goal_id: str,
        *,
        streak: int,
        last_effective_progress_at: str | None = None,
        last_turn_fingerprint: str | None = None,
    ) -> None:
        """Persist the supervisor's effective-progress ledger on the goal row
        (TIER 1a): the consecutive no-progress streak, the last moment of
        genuine progress, and the normalized fingerprint of the last
        supervised turn. Survives a supervisor restart so the streak breaker
        is not reset by a crash/recovery cycle."""
        now = _utcnow_iso()
        with self._connect() as con:
            con.execute(
                "UPDATE goals SET no_effective_progress_streak = ?, "
                "last_effective_progress_at = COALESCE(?, last_effective_progress_at), "
                "last_turn_fingerprint = ?, "
                "updated_at = ? WHERE goal_id = ?",
                (
                    max(0, int(streak or 0)),
                    # None means this turn made no effective progress: retain
                    # the prior timestamp instead of redefining "effective"
                    # progress as mere supervisor activity.
                    last_effective_progress_at,
                    last_turn_fingerprint,
                    now,
                    goal_id,
                ),
            )
            con.commit()

    def set_max_cost_pico(self, goal_id: str, max_cost_pico: int) -> GoalRecord | None:
        """Set the goal's budget to an absolute pico-USD value."""
        now = _utcnow_iso()
        with self._connect() as con:
            con.execute(
                "UPDATE goals SET max_cost_pico = ?, updated_at = ? WHERE goal_id = ?",
                (int(max_cost_pico or 0), now, goal_id),
            )
            self._append_event(
                con,
                goal_id,
                "budget_set",
                {
                    "max_cost_pico": int(max_cost_pico or 0),
                },
            )
            con.commit()
        return self.get(goal_id)

    def add_max_cost_pico(self, goal_id: str, delta_pico: int) -> GoalRecord | None:
        """Top up the goal's budget by ``delta_pico`` (atomic)."""
        now = _utcnow_iso()
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            con.execute(
                "UPDATE goals SET max_cost_pico = max_cost_pico + ?, updated_at = ? WHERE goal_id = ?",
                (int(delta_pico or 0), now, goal_id),
            )
            self._append_event(
                con,
                goal_id,
                "budget_topup",
                {
                    "delta_pico": int(delta_pico or 0),
                },
            )
            con.commit()
        return self.get(goal_id)

    def try_claim_supervisor(
        self,
        goal_id: str,
        owner_id: str,
        *,
        lease_s: float = _SUPERVISOR_LEASE_S,
    ) -> bool:
        """Transactional lease — one live supervisor per goal across processes.

        MID-FLIGHT CRASH — see spec/as-built/goals.md Defects D1, D6)
        """
        now = _utcnow_iso()
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                """
                SELECT status, supervisor_owner_id, supervisor_heartbeat_at
                FROM goals WHERE goal_id = ?
                """,
                (goal_id,),
            ).fetchone()
            if row is None or row["status"] != STATUS_RUNNING:
                con.rollback()
                return False
            cur_owner = str(row["supervisor_owner_id"] or "")
            if cur_owner and cur_owner != owner_id and _iso_age_seconds(row["supervisor_heartbeat_at"]) < lease_s:
                con.rollback()
                return False
            con.execute(
                """
                UPDATE goals
                SET supervisor_owner_id = ?, supervisor_heartbeat_at = ?,
                    updated_at = ?
                WHERE goal_id = ?
                """,
                (owner_id, now, now, goal_id),
            )
            con.commit()
            return True

    def release_supervisor(self, goal_id: str, owner_id: str) -> None:
        """Release the supervisor lease, only if still held by ``owner_id``."""
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            con.execute(
                """
                UPDATE goals
                SET supervisor_owner_id = NULL, supervisor_heartbeat_at = NULL,
                    updated_at = ?
                WHERE goal_id = ? AND supervisor_owner_id = ?
                """,
                (_utcnow_iso(), goal_id, owner_id),
            )
            con.commit()

    def touch_supervisor(
        self,
        goal_id: str,
        *,
        owner_id: str,
    ) -> bool:
        """Renew the supervisor lease's heartbeat, only if still held by
        ``owner_id``.
        """
        now = _utcnow_iso()
        with self._connect() as con:
            cur = con.execute(
                """
                UPDATE goals
                SET supervisor_heartbeat_at = ?, updated_at = ?
                WHERE goal_id = ? AND supervisor_owner_id = ?
                """,
                (now, now, goal_id, owner_id),
            )
            con.commit()
            return int(cur.rowcount or 0) > 0

    def mutate_contract(
        self,
        goal_id: str,
        mutate: Any,
        *,
        event_type: str | None = None,
        payload: dict[str, Any] | None = None,
        expected_contract: dict[str, Any] | None = None,
    ) -> GoalRecord | None:
        """Read-modify-write the goal contract in ONE transaction.

        ``mutate(contract)`` edits the dict in place (or returns a new one).
        ``event_type`` appends one ``goal_events`` row with ``payload``.
        Returns the updated record, or None when the goal does not exist.
        With ``expected_contract``, refuse a concurrent change before writing.
        """
        now = _utcnow_iso()
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT contract_json FROM goals WHERE goal_id = ?",
                (goal_id,),
            ).fetchone()
            if row is None:
                con.rollback()
                return None
            contract = _load_contract(row["contract_json"])
            if expected_contract is not None and contract != expected_contract:
                raise GoalContractChangedError("goal checks changed; read them and retry")
            out = mutate(contract)
            if isinstance(out, dict):
                contract = out
            con.execute(
                "UPDATE goals SET contract_json = ?, updated_at = ? WHERE goal_id = ?",
                (json.dumps(contract), now, goal_id),
            )
            if event_type:
                self._append_event(con, goal_id, event_type, payload or {})
            con.commit()
        return self.get(goal_id)

    def commit_evaluator_update(
        self,
        expected: GoalRecord,
        *,
        owner_id: str,
        state_updates: dict[str, Any],
        outcome: str,
        reason: str = "",
        summary: str | None = None,
    ) -> GoalRecord | None:
        """Commit evaluator state and its lifecycle decision in one transaction.

        A model call runs outside the database lock. Its result belongs only
        to this running root, binding generation, transcript turn and lease.
        Owner commands and another supervisor always invalidate that result.
        """
        now = _utcnow_iso()
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT * FROM goals WHERE goal_id = ?", (expected.goal_id,)).fetchone()
            if row is None:
                raise GoalContractChangedError("goal was cleared during evaluation")
            current = GoalRecord.from_row(row)
            if (
                current.status != STATUS_RUNNING
                or current.root_session_id != expected.root_session_id
                or current.contract != expected.contract
                or current.turn_count != expected.turn_count
                or current.supervisor_owner_id != owner_id
            ):
                raise GoalContractChangedError("goal or supervisor changed during evaluation")
            if outcome == "cleared":
                self._delete_goal_locked(con, expected.goal_id)
            else:
                state = dict(current.contract)
                state.update(state_updates)
                if outcome != "paused":
                    state.pop("notice", None)
                con.execute(
                    "UPDATE goals SET contract_json = ?, updated_at = ? WHERE goal_id = ?",
                    (json.dumps(state), now, expected.goal_id),
                )
                if outcome in ("met", "impossible"):
                    status = STATUS_MET if outcome == "met" else STATUS_BLOCKED
                    con.execute(
                        "UPDATE goals SET status = ?, completed_at = ?, terminal_reason = ?, "
                        "terminal_summary = ? WHERE goal_id = ?",
                        (status, now, reason, summary, expected.goal_id),
                    )
                    self._append_event(con, expected.goal_id, status, {"reason": reason, "summary": summary})
                elif outcome == "paused":
                    con.execute(
                        "UPDATE goals SET status = ?, paused_at = ?, terminal_reason = ? WHERE goal_id = ?",
                        (STATUS_PAUSED, now, reason, expected.goal_id),
                    )
                    self._append_event(con, expected.goal_id, "paused", {"reason": reason})
                if "verdict" in state_updates:
                    self._append_event(
                        con,
                        expected.goal_id,
                        "goal_evaluated",
                        {
                            "verdict": state_updates["verdict"],
                            "reason": str(state_updates.get("last_check") or "")[:1000],
                        },
                    )
            con.commit()
        return self.get(expected.goal_id)

    def append_event(
        self,
        goal_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Append one ``goal_events`` row outside a status transition."""
        with self._connect() as con:
            self._append_event(con, goal_id, event_type, payload)
            con.commit()

    def pause_goal(self, goal_id: str, reason: str | None = None) -> GoalRecord:
        # TIER 3e — compare-and-set: only RUNNING → paused transitions
        # happen; a goal already paused (idempotent) OR already terminal
        # (met/blocked/archived — first terminal transition wins) writes
        # nothing and appends NO event, so a stale late pause can never
        # resurrect or relabel a finished goal.
        return self._terminal_transition(
            goal_id,
            STATUS_PAUSED,
            reason=reason,
            paused=True,
        )[0]

    def _terminal_transition(
        self,
        goal_id: str,
        new_status: str,
        *,
        reason: str | None = None,
        summary: str | None = None,
        paused: bool = False,
        extra_sources: tuple[str, ...] = (),
        expected_contract: dict[str, Any] | None = None,
    ) -> tuple[GoalRecord, bool]:
        """Compare-and-set a terminal transition inside ONE transaction.

        Returns ``(record, transitioned)`` where ``transitioned`` is True
        ONLY for the call that actually performed the write (the CAS
        winner). Concurrent callbacks — a completion claim, the owner's confirm
        and the turn finalizer can race on the same goal —
        all observe the same final record, but exactly one sees
        ``transitioned=True``: the atomic gate for the single "Goal
        completed" notice (a read-then-infer caller,
        ``before = get(); mark(); before.status == RUNNING``, can BOTH
        report a transition).

        TIER 3e — compare-and-set: the write happens only from a status
        ``_SETTLE_SOURCES`` allows for the target. RUNNING reaches any of
        them. PAUSED reaches ``met`` alone: the completion verdict for a
        goal that was paused first (budget, operator) is recorded here, in
        the same transaction as the status check, without a resume that
        would launch work. Everything else — a goal already
        met/blocked/archived, a pause or block of a paused goal — writes
        nothing, appends NO event, and reports ``transitioned=False``
        (idempotent — first terminal transition wins, a stale late
        pause/mark can never resurrect or relabel a finished goal).
        ``paused=True`` writes the paused columns/event; otherwise
        ``new_status`` (met/blocked) writes the terminal columns/event.
        """
        now = _utcnow_iso()
        target = STATUS_PAUSED if paused else new_status
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT status, contract_json FROM goals WHERE goal_id = ?",
                (goal_id,),
            ).fetchone()
            # Runtime claims only settle the running goal whose exact contract
            # was checked. Owner commands keep the ordinary idempotent CAS.
            if expected_contract is not None and (
                row is None
                or row["status"] != STATUS_RUNNING
                or _load_contract(row["contract_json"]) != expected_contract
            ):
                raise GoalContractChangedError("goal state or checks changed; read them and retry")
            allowed = _SETTLE_SOURCES.get(target, ()) + tuple(extra_sources)
            if row is None or row["status"] not in allowed:
                con.rollback()
                rec = self.get(goal_id)
                assert rec is not None
                return rec, False
            if paused:
                con.execute(
                    """
                    UPDATE goals
                    SET status = ?, paused_at = ?, updated_at = ?,
                        terminal_reason = ?
                    WHERE goal_id = ?
                    """,
                    (STATUS_PAUSED, now, now, reason or "paused", goal_id),
                )
                self._append_event(con, goal_id, "paused", {"reason": reason})
            else:
                con.execute(
                    """
                    UPDATE goals
                    SET status = ?, completed_at = ?, updated_at = ?,
                        terminal_reason = ?, terminal_summary = ?
                    WHERE goal_id = ?
                    """,
                    (new_status, now, now, reason, summary, goal_id),
                )
                payload = {"reason": reason, "summary": summary}
                if row["status"] != STATUS_RUNNING:
                    payload["from_status"] = row["status"]
                self._append_event(con, goal_id, new_status, payload)
            con.commit()
        rec = self.get(goal_id)
        assert rec is not None
        return rec, True

    def mark_met(
        self,
        goal_id: str,
        reason: str,
        summary: str | None = None,
        *,
        owner_confirmed: bool = False,
        expected_contract: dict[str, Any] | None = None,
    ) -> GoalRecord:
        """Transition a goal to the terminal ``met`` state with a summary.

        This is the only registry-level path to ``met`` — callers (the goal
        evaluator's ``ok`` verdict, the owner's ``/goal met``) must reach here
        explicitly; nothing in this method itself infers "met" from arbitrary
        text.

        TIER 3e — compare-and-set: a goal already ``met`` appends NO event
        and writes nothing (idempotent). A running goal and a paused goal
        both complete here; a blocked goal only with ``owner_confirmed``
        (the owner marking it met by hand); an archived goal never —
        the first terminal transition wins; later callbacks are no-ops.
        """
        return self._terminal_transition(
            goal_id,
            STATUS_MET,
            reason=reason,
            summary=summary,
            extra_sources=(STATUS_BLOCKED,) if owner_confirmed else (),
            expected_contract=expected_contract,
        )[0]

    def mark_blocked(
        self,
        goal_id: str,
        reason: str,
        summary: str | None = None,
        *,
        expected_contract: dict[str, Any] | None = None,
    ) -> GoalRecord:
        """Transition a goal to the terminal ``blocked`` state with a
        summary — reachable, unlike ``mark_met``, from a bare model-emitted
        text sentinel at the caller (see
        ``goal_supervisor._sync_terminal_to_registry``).

        TIER 3e — compare-and-set: a goal already ``blocked`` appends NO
        event; any other terminal state is left untouched (first terminal
        transition wins).
        """
        return self._terminal_transition(
            goal_id,
            STATUS_BLOCKED,
            reason=reason,
            summary=summary,
            expected_contract=expected_contract,
        )[0]

    def _delete_goal_locked(self, con: sqlite3.Connection, goal_id: str) -> bool:
        con.execute("DELETE FROM goal_events WHERE goal_id = ?", (goal_id,))
        con.execute(
            "DELETE FROM goal_root_session_history WHERE goal_id = ?",
            (goal_id,),
        )
        cur = con.execute("DELETE FROM goals WHERE goal_id = ?", (goal_id,))
        return cur.rowcount > 0

    def _replace_live_on_session_locked(
        self,
        con: sqlite3.Connection,
        root_session_id: str,
        *,
        exclude_goal_id: str | None = None,
    ) -> None:
        if not root_session_id:
            return
        placeholders = ",".join("?" * len(_LIVE_STATUSES))
        rows = con.execute(
            f"SELECT goal_id FROM goals WHERE root_session_id = ? AND status IN ({placeholders})",
            (root_session_id, *_LIVE_STATUSES),
        ).fetchall()
        for row in rows:
            gid = str(row["goal_id"] or "")
            if exclude_goal_id and gid == exclude_goal_id:
                continue
            self._delete_goal_locked(con, gid)

    def _purge_archived_locked(self, con: sqlite3.Connection) -> None:
        rows = con.execute(
            "SELECT goal_id FROM goals WHERE status = ?",
            (STATUS_ARCHIVED,),
        ).fetchall()
        for row in rows:
            self._delete_goal_locked(con, str(row["goal_id"]))

    def _expire_due_locked(self, con: sqlite3.Connection) -> list[GoalRecord]:
        rows = con.execute(
            "SELECT * FROM goals WHERE status = ?",
            (STATUS_MET,),
        ).fetchall()
        expired: list[GoalRecord] = []
        for row in rows:
            rec = GoalRecord.from_row(row)
            ts = rec.completed_at or rec.updated_at
            if _iso_age_seconds(ts) > MAX_COMPLETED_AGE_S:
                self._delete_goal_locked(con, rec.goal_id)
                expired.append(rec)
        return expired

    def expire_due(self) -> list[GoalRecord]:
        """Delete ``met`` goals older than ``MAX_COMPLETED_AGE_S``."""
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            expired = self._expire_due_locked(con)
            con.commit()
        return expired

    def delete_goal(self, goal_id: str) -> bool:
        """Permanently delete a goal and its events/root history."""
        if not goal_id:
            return False
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            ok = self._delete_goal_locked(con, goal_id)
            con.commit()
        return ok

    def _archive_goal(
        self,
        goal_id: str,
        reason: str | None = None,
    ) -> tuple[GoalRecord | None, bool]:
        """Manual delete alias. Returns ``(snapshot, deleted)``."""
        rec = self.get(goal_id)
        if rec is None:
            return None, False
        return rec, self.delete_goal(goal_id)

    def apply_terminal_if_unchanged(
        self,
        goal_id: str,
        *,
        expected_status: str,
        expected_root_session_id: str,
        expected_supervisor_heartbeat_at: str | None,
        terminal: str,
        reason: str,
        summary: str | None = None,
    ) -> GoalRecord | None:
        """Background-heal CAS: transition ``goal_id`` to ``terminal`` ONLY if
        the row is still exactly the record the caller observed — status, root
        binding, and supervisor heartbeat unchanged since the snapshot.

        The web reconciler's background heal pass (``_heal_stale_goals_once``)
        snapshots goals and reads the root session's projected ``goal_terminal``
        on a separate connection; between the two the goal can be resumed, its
        root can rebind to a new cycle, or a live descendant can appear. A
        blind ``mark_met``/``mark_blocked``/``pause_goal`` by id would mark a
        live cycle terminal. This compares the full evidence atomically under
        ``BEGIN IMMEDIATE``:

        * ``expected_status`` — a resumed or already-terminal goal is a no-op
          (first terminal wins, idempotent, no duplicate events);
        * ``expected_root_session_id`` — a rebind must never be healed from the
          old session's header;
        * ``expected_supervisor_heartbeat_at`` — a renewed lease means the
          supervisor is live again and owns the terminal transition.

        The caller performs the fresh cheap subtree-liveness check immediately
        before invoking this (the session-side evidence); this method locks the
        goal-side evidence. Returns the current record (``None`` only when the
        goal row is gone). Never raises on a mismatch — a no-op.
        """
        now = _utcnow_iso()
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT status, root_session_id, supervisor_heartbeat_at FROM goals WHERE goal_id = ?",
                (goal_id,),
            ).fetchone()
            if row is None:
                con.rollback()
                return None
            current_status = str(row["status"] or "")
            current_root = str(row["root_session_id"] or "")
            current_hb = row["supervisor_heartbeat_at"]
            # First-terminal-wins mirrors the primitives: met/blocked only from
            # running; paused from running/blocked (never a terminal override).
            eligible = (
                current_status == STATUS_RUNNING
                if terminal in (STATUS_MET, STATUS_BLOCKED)
                else current_status in (STATUS_RUNNING, STATUS_BLOCKED)
            )
            if (
                current_status != expected_status
                or current_root != expected_root_session_id
                or (current_hb or None) != (expected_supervisor_heartbeat_at or None)
                or not eligible
            ):
                # Evidence changed (resumed / rebound / lease renewed) or the
                # goal is already terminal — first terminal wins, no-op.
                con.rollback()
                return self.get(goal_id)
            if terminal == STATUS_MET:
                con.execute(
                    """
                    UPDATE goals
                    SET status = ?, completed_at = ?, updated_at = ?,
                        terminal_reason = ?, terminal_summary = ?
                    WHERE goal_id = ?
                    """,
                    (STATUS_MET, now, now, reason, summary, goal_id),
                )
                self._append_event(
                    con,
                    goal_id,
                    "met",
                    {
                        "reason": reason,
                        "summary": summary,
                    },
                )
            elif terminal == STATUS_BLOCKED:
                con.execute(
                    """
                    UPDATE goals
                    SET status = ?, completed_at = ?, updated_at = ?,
                        terminal_reason = ?, terminal_summary = ?
                    WHERE goal_id = ?
                    """,
                    (STATUS_BLOCKED, now, now, reason, summary, goal_id),
                )
                self._append_event(
                    con,
                    goal_id,
                    "blocked",
                    {
                        "reason": reason,
                        "summary": summary,
                    },
                )
            elif terminal == STATUS_PAUSED:
                con.execute(
                    """
                    UPDATE goals
                    SET status = ?, paused_at = ?, updated_at = ?,
                        terminal_reason = ?
                    WHERE goal_id = ?
                    """,
                    (STATUS_PAUSED, now, now, reason, goal_id),
                )
                self._append_event(con, goal_id, "paused", {"reason": reason})
            else:
                con.rollback()
                return self.get(goal_id)
            con.commit()
        return self.get(goal_id)

    def archive_goal(self, goal_id: str, reason: str | None = None) -> GoalRecord:
        """Manual delete alias. Raises if the id is missing."""
        rec, deleted = self._archive_goal(goal_id, reason=reason)
        if rec is None or not deleted:
            raise KeyError(f"goal not found: {goal_id}")
        return rec

    def migrate_from_session(
        self,
        *,
        session_id: str,
        agent_name: str,
        active_goal: str,
        goal_terminal: str,
        goal_turn_count: int,
        max_turns: int = 0,
        max_cost_pico: int = 0,
    ) -> GoalRecord:
        """Lazy migration: session-scoped goal → durable record (idempotent)."""
        status = STATUS_RUNNING
        if goal_terminal == "met":
            status = STATUS_MET
        elif goal_terminal == "blocked":
            status = STATUS_BLOCKED
        elif goal_terminal == "paused":
            status = STATUS_PAUSED

        norm = normalize_condition_text(active_goal)
        ch = condition_hash(active_goal)
        now = _utcnow_iso()
        goal_id = new_goal_id()
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            existing = con.execute(
                """
                SELECT goal_id FROM goals
                WHERE agent_name = ?
                  AND condition_text = ?
                  AND (root_session_id = ? OR created_from_session_id = ?)
                LIMIT 1
                """,
                (agent_name, active_goal, session_id, session_id),
            ).fetchone()
            if existing:
                con.rollback()
                got = self.get(existing["goal_id"])
                assert got is not None
                return got
            if status == STATUS_RUNNING:
                self._replace_live_on_session_locked(con, session_id)
            try:
                con.execute(
                    """
                    INSERT INTO goals (
                        goal_id, condition_text, normalized_condition_text,
                        condition_hash, agent_name, scope_key, status,
                        root_session_id, created_from_session_id,
                        created_at, updated_at, started_at, turn_count, max_turns,
                        max_cost_pico, terminal_reason
                    ) VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        goal_id,
                        active_goal,
                        norm,
                        ch,
                        agent_name,
                        status,
                        session_id,
                        session_id,
                        now,
                        now,
                        now,
                        goal_turn_count,
                        max_turns,
                        int(max_cost_pico or 0),
                        goal_terminal or None,
                    ),
                )
            except sqlite3.IntegrityError:
                con.rollback()
                row = con.execute(
                    """
                    SELECT goal_id FROM goals
                    WHERE root_session_id = ? AND status = 'running'
                    LIMIT 1
                    """,
                    (session_id,),
                ).fetchone()
                if row:
                    got = self.get(row["goal_id"])
                    assert got is not None
                    return got
                raise
            con.execute(
                """
                INSERT INTO goal_root_session_history
                (goal_id, session_id, agent_name, bound_at)
                VALUES (?, ?, ?, ?)
                """,
                (goal_id, session_id, agent_name, now),
            )
            self._append_event(
                con,
                goal_id,
                "migrated_from_session_goal",
                {
                    "session_id": session_id,
                },
            )
            con.commit()
        rec = self.get(goal_id)
        assert rec is not None
        return rec


_registry: GoalRegistry | None = None


def get_goal_registry(db_path: Path | None = None) -> GoalRegistry:
    global _registry
    if db_path is not None:
        return GoalRegistry(db_path)
    if _registry is None:
        _registry = GoalRegistry()
    return _registry


_GOAL_SESSION_PROJECTION_FIELDS = (
    "goal_id",
    "active_goal",
    "goal_terminal",
    "goal_turn_count",
)


def sync_session_goal_projection(session: Any, goal: GoalRecord | None) -> None:
    """Mirror durable goal state onto a session header projection."""
    if goal is None:
        session.goal_id = ""
        session.active_goal = ""
        session.goal_terminal = ""
        session.goal_turn_count = 0
    else:
        session.goal_id = goal.goal_id
        session.active_goal = goal.condition_text
        session.goal_turn_count = goal.turn_count

        if goal.status == STATUS_MET:
            # Keep id + condition + terminal so the chat widget can show
            # a completed badge. Archived still detaches.
            session.goal_terminal = "met"
        elif goal.status == STATUS_BLOCKED:
            session.goal_terminal = "blocked"
        elif goal.status == STATUS_PAUSED:
            session.goal_terminal = "paused"
        elif goal.status == STATUS_ARCHIVED:
            # Archived is a TERMINAL, detached state: the goal row survives
            # in the registry, but the live session projection is
            # cleared so the chat widget/API drop it immediately — the
            # durable goal_events row is the record (``/goal clear``
            # archives rather than pausing).
            session.goal_id = ""
            session.active_goal = ""
            session.goal_terminal = ""
            session.goal_turn_count = 0
        else:
            session.goal_terminal = ""

    try:
        save_fields = getattr(session, "save_header_fields", None)
        if callable(save_fields):
            save_fields(_GOAL_SESSION_PROJECTION_FIELDS)
        else:
            # Session-like test/back-compat objects keep the legacy save seam;
            # production Session supplies save_header_fields for the durable,
            # scoped forced patch.
            session.save()
    except Exception as exc:  # noqa: BLE001
        # Loud everywhere, no carve-out: this session-header projection is
        # best-effort and must NEVER block the authoritative registry write
        # (already committed by the caller) — so we do NOT re-raise. But the
        # projection failure is NOT exempt from uniform observability: route it
        # through the SAME structured error-capture channel every other
        # write-failure surface uses (``sharedagents_core.errors.log_error`` → the
        # SessionError-hook machinery that fires across surfaces),
        # rather than a lone ``logging.warning`` the dashboard/aggregation never
        # sees. Kept fail-safe: a capture failure can never turn a best-effort
        # projection into a raising path (fallback to a plain warning).
        try:
            from sharedagents_core import errors as _errors

            _errors.log_error(
                "goal:projection",
                exc,
                summary="goal projection save failed (registry write unaffected)",
                session_id=getattr(session, "hash", "") or None,
            )
        except Exception:  # noqa: BLE001
            import logging

            logging.getLogger("sharedagents.goal").warning(
                "goal projection save failed (registry write is unaffected): %s: %s",
                type(exc).__name__,
                exc,
            )
