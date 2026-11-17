"""Canonical session activity contract shared by projection and WebUI.

``sessions_meta.session_state`` predates the WebUI's need to distinguish a
running turn, a pending operator decision, descendant work, and the outcome of
the last completed turn.  Keep that column as a compatibility register, but
derive one typed, additive snapshot for new consumers.

The module is deliberately free of database and Flask dependencies.  It is
used by the event projector, the orchestrator tree, and gateway/UI adapters so
the same precedence and labels apply everywhere.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Single durable occupancy + pipeline enum. ``paused`` is accepted for
# historical rows only (nothing emits ``pause_resume`` today). ``ask`` is
# a real state: a parked operator question, not a UI overlay.
SESSION_STATES = (
    "idle",
    "prompting",
    "prefill",
    "reasoning",
    "streaming",
    "tool_running",
    "compacting",
    "ask",
    "retrying",
    "paused",
    "waiting",
    "error",
    "closed",
)
PHASES = (
    "idle",
    "prompting",
    "prefill",
    "reasoning",
    "streaming",
    "tool_running",
    "compacting",
    "ask",
    "retrying",
)
ATTENTION_STATES = ("none", "question", "permission", "operator_paused")
LIFECYCLE_STATES = ("open", "detached")
OUTCOMES = ("none", "succeeded", "cancelled", "interrupted", "failed")

# Compatibility alias — every writer/reader must accept the same set.
LEGACY_STATES = SESSION_STATES
# Hidden internal side-call utilities (title agent, task recap).  They mirror
# ``orchestrator.MONITOR_HIDDEN_AGENTS``: their children are invisible in the
# monitor, and they must not make a parent look tree-busy on the Tasks board
# either (a running side call would flip a Completed card to Working).
INTERNAL_SIDE_CALL_AGENTS = ("title", "summary")
OWN_LIVE_STATES = (
    "prompting",
    "prefill",
    "reasoning",
    "streaming",
    "tool_running",
    "compacting",
    "ask",
    "retrying",
)
TREE_BUSY_STATES = OWN_LIVE_STATES + ("paused", "waiting")
# Own-turn states that Stop/abort still has a live_turn_idx to close.
ABORT_LIVE_TURN_STATES = (
    "prompting",
    "prefill",
    "reasoning",
    "streaming",
    "tool_running",
    "compacting",
    "paused",
    "ask",
    "retrying",
)

_PHASE_BY_LEGACY = {
    "idle": "idle",
    "prompting": "prompting",
    "prefill": "prefill",
    "reasoning": "reasoning",
    "streaming": "streaming",
    "tool_running": "tool_running",
    "compacting": "compacting",
    "ask": "ask",
    "retrying": "retrying",
    "paused": "idle",
    "waiting": "idle",
    "error": "idle",
    "closed": "idle",
}

_STATUS = {
    "needs_input": ("Needs input", "attention"),
    "working": ("Working", "working"),
    "completed": ("Ready", "complete"),
    "detached": ("Closed", "muted"),
}


def _text(row: Mapping[str, Any], key: str, default: str = "") -> str:
    value = row.get(key, default)
    return str(default if value is None else value).strip()


def _int(row: Mapping[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def legacy_phase(
    state: str,
    stored_phase: str = "",
    *,
    live_turn_idx: Any = None,
) -> str:
    """Map the compatibility register to an own-turn phase.

    A leftover V2 live phase after the own turn closed (``live_turn_idx``
    NULL) must not look live. Compaction-resume may keep ``streaming`` only
    while the own turn is still open.
    """
    phase = stored_phase if stored_phase in PHASES else ""
    if state not in OWN_LIVE_STATES and live_turn_idx is None:
        return _PHASE_BY_LEGACY.get(state, "idle")
    # Rows created before the V2 migration receive the column default
    # ``session_phase='idle'``. Let the legacy live register win over that
    # migration default; an explicitly projected V2 phase (for example a
    # compaction-resume row carrying ``streaming``) still takes precedence.
    if phase == "idle" and state in OWN_LIVE_STATES:
        return state
    return phase or _PHASE_BY_LEGACY.get(state, "idle")


def outcome_for_legacy(state: str, stored_outcome: str = "") -> str:
    """Return a safe outcome for rows created before the V2 columns."""
    if stored_outcome in OUTCOMES:
        return stored_outcome
    if state == "error":
        return "failed"
    return "none"


def _pending_attention(pending_action: Any) -> str:
    if isinstance(pending_action, Mapping):
        kind = _text(pending_action, "kind").lower()
    else:
        kind = _text({"value": pending_action}, "value").lower()
    if kind in {"permission", "tool_permission", "approval"}:
        return "permission"
    if kind in {"question", "ask", "ask_user", "operator_question"}:
        return "question"
    return "none"


def row_stale_for_display(
    *,
    raw_stale: bool,
    own_live: bool,
    descendant_count: int,
) -> bool:
    """Keep the stale→Needs input rule except for closed-turn + live children."""
    return bool(raw_stale) and not (int(descendant_count or 0) > 0 and not own_live)


def count_fresh_live_descendants(
    con: Any,
    sid: str,
    *,
    live_states: tuple[str, ...] = TREE_BUSY_STATES,
    cutoff_ms: int,
    max_depth: int = 16,
) -> int:
    """Freshness-filtered live descendant count (any depth, cycle-capped).

    Elapsed time alone is not proof of death: a descendant whose turn is
    still OPEN (``live_turn_idx`` set — a single long silent tool call ages
    ``session_state_updated_at`` past the cutoff while genuinely running)
    counts as live regardless of the stamp age. Rows abandoned by a crash
    keep their open turn only until the stale-live resolver folds them
    terminal, so this stays bounded and never makes a dead row look live
    forever.

    Hidden internal side-call children (``title``, ``summary``) and their
    subtrees never count: a running title/recap side call is not operator
    work, and must not flip a Completed Tasks card to Working.
    """
    if not sid:
        return 0
    ph = ",".join("?" for _ in live_states)
    hidden = ",".join("?" for _ in INTERNAL_SIDE_CALL_AGENTS)
    row = con.execute(
        f"WITH RECURSIVE kids AS ("
        f"  SELECT session_hash, session_state, session_state_updated_at,"
        f"         live_turn_idx, 1 AS depth"
        f"  FROM sessions_meta WHERE parent_session_id=?"
        f"    AND LOWER(TRIM(COALESCE(agent_name, ''))) NOT IN ({hidden})"
        f"  UNION ALL"
        f"  SELECT c.session_hash, c.session_state, c.session_state_updated_at,"
        f"         c.live_turn_idx, kids.depth + 1"
        f"  FROM sessions_meta c"
        f"  JOIN kids ON c.parent_session_id = kids.session_hash"
        f"  WHERE kids.depth < ?"
        f"    AND LOWER(TRIM(COALESCE(c.agent_name, ''))) NOT IN ({hidden})"
        f") SELECT COUNT(*) FROM kids"
        f" WHERE session_state IN ({ph})"
        f"   AND (session_state_updated_at >= ? OR live_turn_idx IS NOT NULL)",
        (sid, *INTERNAL_SIDE_CALL_AGENTS, int(max_depth), *INTERNAL_SIDE_CALL_AGENTS, *live_states, int(cutoff_ms)),
    ).fetchone()
    if row is None:
        return 0
    return int(row[0] if not hasattr(row, "keys") else row[0])


def _display_status(
    *,
    attention: str,
    stale: bool,
    phase: str,
    tree_busy: bool,
    lifecycle: str,
    outcome: str,
    own_live: bool = False,
    descendant_count: int = 0,
) -> tuple[str, str, str]:
    """Return ``(code, label, tone)`` with one stable precedence order."""
    if attention in {"question", "permission"} or attention == "operator_paused":
        code = "needs_input"
    elif row_stale_for_display(
        raw_stale=stale,
        own_live=own_live,
        descendant_count=descendant_count,
    ):
        # Only a parent whose own turn is closed AND that still has live
        # descendants escapes the stale→Needs input rule. Bare
        # ``waiting`` is already tree_busy, so keying off tree_busy would
        # hide abandoned waits with zero children.
        code = "needs_input"
    elif phase != "idle" or tree_busy:
        code = "working"
    elif lifecycle == "detached":
        code = "detached"
    elif outcome in {"failed", "interrupted"}:
        code = "needs_input"
    else:
        code = "completed"
    label, tone = _STATUS[code]
    return code, label, tone


def derive_activity(
    row: Mapping[str, Any] | None,
    *,
    now_ms: int | None = None,
    freshness_seconds: int = 300,
    tree_busy: bool | None = None,
    stale: bool | None = None,
    pending_action: Any = None,
    descendant_count: int = 0,
    open_job_count: int = 0,
) -> dict[str, Any]:
    """Derive the canonical, JSON-safe activity snapshot for one row.

    ``row`` may be a legacy test fixture with only ``session_state``.  Missing
    V2 columns therefore fall back deterministically instead of making a UI
    endpoint fail while a database migration is in progress.
    """
    raw = _text(row or {}, "session_state", "idle") or "idle"
    if raw not in SESSION_STATES:
        raw = "idle"
    live_turn_idx = row.get("live_turn_idx") if row else None
    phase = legacy_phase(
        raw,
        _text(row or {}, "session_phase"),
        live_turn_idx=live_turn_idx,
    )
    attention = _text(row or {}, "session_attention", "none")
    if attention not in ATTENTION_STATES:
        attention = "none"
    if raw == "ask" and attention == "none":
        attention = "question"
    pending = _pending_attention(pending_action)
    if pending != "none":
        attention = pending
    lifecycle = _text(row or {}, "session_lifecycle", "")
    if lifecycle not in LIFECYCLE_STATES:
        lifecycle = "detached" if raw == "closed" else "open"
    outcome = outcome_for_legacy(raw, _text(row or {}, "session_last_outcome"))

    own_live = (live_turn_idx is not None and phase != "idle") or raw == "retrying"
    if tree_busy is None:
        tree_busy = raw in TREE_BUSY_STATES or own_live
    tree_busy = bool(tree_busy or descendant_count > 0 or open_job_count > 0)

    heartbeat = _int(
        row or {},
        "session_heartbeat_at",
        _int(row or {}, "session_state_updated_at", 0),
    )
    if stale is None:
        if freshness_seconds <= 0 or heartbeat <= 0 or now_ms is None:
            stale = False
        else:
            stale = raw in TREE_BUSY_STATES and heartbeat < (int(now_ms) - freshness_seconds * 1000)
    stale = bool(stale)

    code, label, tone = _display_status(
        attention=attention,
        stale=stale,
        phase=phase,
        tree_busy=tree_busy,
        lifecycle=lifecycle,
        outcome=outcome,
        own_live=bool(own_live),
        descendant_count=max(0, int(descendant_count)),
    )
    can_answer = attention in {"question", "permission"}
    return {
        "version": 2,
        "phase": phase,
        "attention": attention,
        "lifecycle": lifecycle,
        "last_outcome": outcome,
        "own_live": bool(own_live),
        "tree_busy": tree_busy,
        "stale": stale,
        "active_descendant_count": max(0, int(descendant_count)),
        "open_job_count": max(0, int(open_job_count)),
        "phase_changed_at": _int(row or {}, "session_phase_changed_at"),
        "heartbeat_at": heartbeat,
        "outcome_at": _int(row or {}, "session_outcome_at"),
        "status": code,
        "display": {"code": code, "label": label, "tone": tone},
        "capabilities": {
            "can_stop": tree_busy,
            "can_answer": can_answer,
            "can_send": not tree_busy or can_answer,
        },
    }


def legacy_state_for(activity: Mapping[str, Any]) -> str:
    """Compatibility projection for clients that still require one enum."""
    attention = _text(activity, "attention", "none")
    phase = _text(activity, "phase", "idle")
    outcome = _text(activity, "last_outcome", "none")
    lifecycle = _text(activity, "lifecycle", "open")
    if lifecycle == "detached":
        return "closed"
    if attention == "operator_paused":
        return "paused"
    if attention in {"question", "permission"}:
        return "ask"
    if phase in OWN_LIVE_STATES:
        return phase
    if bool(activity.get("tree_busy")):
        return "waiting"
    if outcome == "failed":
        return "error"
    return "idle"


def tasks_card(
    state: str,
    *,
    last_outcome: str = "",
    stale: bool = False,
    tree_busy: bool = False,
    own_live: bool = False,
    descendant_count: int = 0,
    has_row: bool = True,
    archived: bool = False,
) -> str:
    """Fold the one session enum onto the Tasks board's three cards.

    ``unstarted`` (no sessions_meta row) is a board concern, not a session
    state. ``paused`` historical rows stay Needs input. Abort still lands
    on ``error`` (Stop is not remapped to idle in this change).
    """
    if archived:
        return "archived"
    if not has_row or state in {"", "unstarted"}:
        return "needs_input"
    if state in {"ask", "paused", "error"}:
        return "needs_input"
    if row_stale_for_display(
        raw_stale=stale,
        own_live=own_live,
        descendant_count=descendant_count,
    ):
        return "needs_input"
    if state in TREE_BUSY_STATES or tree_busy:
        return "working"
    if state in {"idle", "closed"}:
        if last_outcome in {"failed", "interrupted"}:
            return "needs_input"
        return "completed"
    return "needs_input"


__all__ = [
    "ABORT_LIVE_TURN_STATES",
    "ATTENTION_STATES",
    "LEGACY_STATES",
    "LIFECYCLE_STATES",
    "OWN_LIVE_STATES",
    "OUTCOMES",
    "PHASES",
    "INTERNAL_SIDE_CALL_AGENTS",
    "SESSION_STATES",
    "TREE_BUSY_STATES",
    "count_fresh_live_descendants",
    "derive_activity",
    "legacy_phase",
    "legacy_state_for",
    "outcome_for_legacy",
    "row_stale_for_display",
    "tasks_card",
]
