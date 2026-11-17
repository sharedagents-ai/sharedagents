"""High-level goal operations — shared by CLI, TUI, web, and engine."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


def _finite_usd(value: Any) -> float | None:
    """Parse a USD amount, rejecting non-finite (nan/inf) values. Returns None
    if it isn't a finite number (callers surface a clear error)."""
    try:
        usd = float(value)
    except (TypeError, ValueError):
        return None
    return usd if math.isfinite(usd) else None


from sharedagents_goal.goal_registry import (  # noqa: E402
    STATUS_ARCHIVED,
    STATUS_MET,
    STATUS_PAUSED,
    STATUS_RUNNING,
    GoalConflictError,
    GoalFilters,
    GoalRecord,
    get_goal_registry,
    sync_session_goal_projection,
)
from sharedagents_model.usage_pricing import pico_to_usd, usd_to_pico  # noqa: E402


def _goal_cost_pico(goal_id: str) -> int:
    """Spend attributed to this goal: the ledger SUM of the rows carrying its id.

    This is the figure for the goal's OWN cap. Session-tree spend also holds
    work done before the goal existed and is limited by the session cap, a
    separate number — :func:`goal_budget_state` pairs each spend with its cap.
    """
    try:
        from sharedagents_model.usage_ledger import get_usage_ledger

        return int(get_usage_ledger().goal_cost_pico(goal_id))
    except Exception:  # noqa: BLE001
        return 0


def cost_snapshot_metrics(
    root_session_id: str,
    goal_id: str = "",
) -> dict[str, Any]:
    """Cumulative USD cost for a session (and its goal) for the live UI.

    THE single helper both the per-turn finalizer (``agent.py``, every engine)
    and ``command_runtime._goal_state`` (/status, session-open) call, so the web
    goal widget + session footer get realtime spend without either path knowing
    how cost is summed. Returns:

      • ``session_cost_usd``  — SUM of all turn costs rolled up to this root
        session (always present).
      • ``goal_cost_usd``     — SUM of the turn costs attributed to the
        active goal (the figure its own budget is compared with).
      • ``goal_max_cost_usd`` — the goal's USD cost budget. Never ``None``: a
        goal with no explicit budget floors to ``DEFAULT_BUDGET_USD`` so the UI
        never renders ∞.

    Best-effort — never raises into a turn or /status.
    """
    out: dict[str, Any] = {}
    try:
        from sharedagents_model.usage_ledger import get_usage_ledger

        led = get_usage_ledger()
        out["session_cost_usd"] = pico_to_usd(led.session_cost_pico(str(root_session_id or "")))
        out["session_max_cost_usd"] = lookup_session_budget_usd(str(root_session_id or ""))
        gid = str(goal_id or "")
        if gid:
            out["goal_cost_usd"] = pico_to_usd(_goal_cost_pico(gid))
            rec = get_goal_registry().get(gid)
            out["goal_max_cost_usd"] = _effective_goal_cap_usd(rec) if rec else DEFAULT_BUDGET_USD
    except Exception:  # noqa: BLE001
        pass
    return out


# The budget is NEVER unbounded by default. Absent an explicit
# owner budget, spend is capped at this floor (owner override 2026-07-12: "the default
# should be 25 usd never unbounded"). Single source of truth for the default, so goal
# creation and every budget display agree. Config key ``default_budget_usd`` may raise it.
DEFAULT_BUDGET_USD = 25.0


def resolve_budget_usd(budget_usd: float | None) -> float:
    """The effective budget for display/creation — never ``None``, never ≤ 0.

    A missing or non-positive budget (legacy ``0`` sentinel, or no goal at all) is
    floored to :data:`DEFAULT_BUDGET_USD` so the UI never renders ∞ and no session
    runs truly unbounded.
    """
    if budget_usd is None or budget_usd <= 0:
        return DEFAULT_BUDGET_USD
    return float(budget_usd)


def session_root_id(session: Any) -> str:
    """Root session id this session's spend rolls up to."""
    return str(getattr(session, "cost_root_session_id", "") or getattr(session, "hash", "") or "")


def charged_goal_id(
    *,
    session_hash: Any,
    cost_root_session_id: Any,
    cost_ancestor_goal_id: Any,
    goal_id: Any,
) -> str:
    """The goal a session's spend is charged to.

    A delegated child rolls up to the goal it was started under: its stored
    ancestor, fixed for the child's lifetime. A root session is charged to the
    goal bound to it NOW. Its stored ancestor is a copy of an earlier binding
    that the index never overwrites, so once the session gets a second goal it
    still names the first; it remains the fallback while no goal is bound.
    """
    own = str(goal_id or "")
    ancestor = str(cost_ancestor_goal_id or "")
    root = str(cost_root_session_id or "")
    if root and root != str(session_hash or ""):
        return ancestor or own
    return own or ancestor


def session_charged_goal_id(session: Any) -> str:
    """:func:`charged_goal_id` for a loaded session."""
    return charged_goal_id(
        session_hash=getattr(session, "hash", ""),
        cost_root_session_id=getattr(session, "cost_root_session_id", ""),
        cost_ancestor_goal_id=getattr(session, "cost_ancestor_goal_id", ""),
        goal_id=getattr(session, "goal_id", ""),
    )


def raw_session_budget_usd(session: Any) -> float | None:
    """Stored session-tree cap, or None when unset (caller floors to default)."""
    for key in ("session_budget_usd", "_prompt_budget_usd"):
        val = getattr(session, key, None)
        if val is None:
            continue
        try:
            n = float(val)
        except (TypeError, ValueError):
            continue
        if n > 0:
            return n
    return None


def _session_cost_pico(root_session_id: str) -> int:
    try:
        from sharedagents_model.usage_ledger import get_usage_ledger

        return int(get_usage_ledger().session_cost_pico(str(root_session_id or "")))
    except Exception:  # noqa: BLE001
        return 0


def remaining_session_budget_usd(session: Any, *, extra_pico: int = 0) -> float:
    """USD still available on this session TREE (root cap − root SUM)."""
    cap = tree_cap_usd(session)
    spent = pico_to_usd(_session_cost_pico(session_root_id(session)) + int(extra_pico or 0))
    return max(0.0, cap - spent)


def session_budget_snapshot(session: Any) -> dict[str, float]:
    """Return the effective tree spend/cap figures for a live session.

    The admission gate and the manual-turn notice must describe the same
    root-tree accounting.  Keeping this calculation beside
    :func:`remaining_session_budget_usd` prevents the notice from reporting a
    child-local cap or a stale goal-only spend.
    """
    cap = float(tree_cap_usd(session))
    spent = float(pico_to_usd(_session_cost_pico(session_root_id(session))))
    return {
        "spent_usd": max(0.0, spent),
        "budget_usd": max(0.0, cap),
        "remaining_usd": max(0.0, cap - spent),
    }


def manual_budget_system_prompt(session: Any) -> str:
    """Build the short system notice used for manual prompts after the cap.

    Automatic goal/loop/heartbeat work remains refused at the admission gate,
    but an operator can still ask a direct question.  The notice is framework
    authored and deliberately tells the model to answer briefly without tools,
    delegation, or a long-running task; it is not a user-provided instruction.
    """
    snap = session_budget_snapshot(session)
    return (
        "[sharedagents session budget notice]\n"
        f"The session cost limit has been reached: "
        f"${snap['spent_usd']:.4f} spent of ${snap['budget_usd']:.2f}; "
        f"${snap['remaining_usd']:.4f} remains.\n"
        "This is an operator's manual prompt and may receive a brief direct "
        "answer. Do not call tools, delegate, edit files, run commands, or "
        "start or continue a long-running task. If the request needs work, "
        "explain that the session must be topped up or restarted first. "
        "Answer only what can be stated safely and concisely."
    )


#: What the runtime writes into a turn's answer when the session budget stopped
#: it. The text stays in the conversation whatever the cap is afterwards.
SESSION_BUDGET_LIMIT_PHRASES = (
    "The session cost budget has reached its limit",
    "session cost budget reached",
)


# Lines that state THIS session's budget or spend. Deliberately narrow: other
# budgets ("hosting budget cap $20/month") and work items ("fix the budget
# limit validation") are durable facts and must survive.
#
# Session budget STATE (not instructions about it): "session budget
# exhausted", "session cost budget reached ($32.23/$25.00)", "owner raised the
# session budget to $100".
_SESSION_BUDGET_STATE_RE = re.compile(
    r"\b(?:session|cost|tree)[\s-]+(?:cost\s+)?budget\b[^.\n]*?"
    r"\b(?:is\s+|was\s+|has\s+been\s+)?(?:exhausted|reached|frozen|spent|"
    r"remains?|topped[\s-]?up)\b"
    r"|\b(?:raised|topped\s+up|increased|set)\b[^.\n]*\b(?:session|cost|tree)"
    r"[\s-]+budget\b[^.\n]*\bto\s+\$\s?\d",
    re.I,
)
_SESSION_CONTEXT_RE = re.compile(r"\b(?:session|tree)\b|\[sharedagents\]", re.I)
# The exact line the runtime appends to a turn's answer (turn_execution).
_RUNTIME_NOTICE_LINE_RE = re.compile(
    r"^\s*\[sharedagents\]\s+(?:" + "|".join(re.escape(p) for p in SESSION_BUDGET_LIMIT_PHRASES) + ")",
    re.I,
)
# Instructions and work items about a budget are durable; always keep them.
_INSTRUCTION_RE = re.compile(
    r"^\s*(?:(?:[-*+]|\d+[.)])\s+)?(?:\[[ xX]\]\s+)?(?:\*\*|__)?"
    r"(?:todo|next|pending)?:?(?:\*\*|__)?:?\s*"
    r"(?:never|do\s+not|don'?t|always|must|should|fix|ask|only|keep|avoid|todo|"
    r"pending|add|implement|test|check|write|update|verify)\b",
    re.I,
)
# Quoted text (code spans, quotations) names a phrase; it does not state it.
_QUOTED_RE = re.compile(r"`[^`\n]*`|\"[^\"\n]*\"|“[^”\n]*”")
# "$32.23 of $25.00 spent", "spent $32.23 of/against a $25.00 cap",
# "($32.23/$25.00)".
_SPEND_OF_CAP_RE = re.compile(
    r"\$\s?\d[\d,]*(?:\.\d+)?\s+of\s+\$\s?\d[\d,]*(?:\.\d+)?\s+spent"
    r"|\bspent\s+\$\s?\d[\d,]*(?:\.\d+)?\s+(?:of|against)\s+(?:a\s+|the\s+)?"
    r"\$\s?\d"
    r"|\(\s*\$\s?\d[\d,]*(?:\.\d+)?\s*/\s*\$\s?\d[\d,]*(?:\.\d+)?\s*\)",
    re.I,
)


def is_budget_state_line(line: str) -> bool:
    """True for a line that states this session's budget or its spend.

    Budget state is live data. Text that carries it forward (a compaction
    summary, a child's answer) goes stale as soon as the cap moves, and the
    model then believes the stale figure over the live one. Instructions
    ("never raise the session budget cap without approval") and work items
    ("fix the session budget limit validation") are durable and kept.
    """
    text = str(line or "")
    if not text.strip():
        return False
    if _INSTRUCTION_RE.search(text):
        return False
    text = _QUOTED_RE.sub(" ", text)
    # Only state about THIS session: hosting allowances, plan caps and other
    # budgets the work talks about are durable facts. The runtime's own notice
    # always carries its figures, so the spend pattern catches it.
    if not _SESSION_CONTEXT_RE.search(text):
        return False
    return bool(_SPEND_OF_CAP_RE.search(text) or _SESSION_BUDGET_STATE_RE.search(text))


def strip_budget_state(text: str) -> str:
    """Drop every line that states the session budget or its spend."""
    if not text:
        return text or ""
    kept = [ln for ln in str(text).splitlines() if not is_budget_state_line(ln)]
    return "\n".join(kept).strip("\n") if kept else ""


def strip_budget_limit_notice(text: str) -> str:
    """Remove the runtime's appended limit notice from a child's answer.

    The notice describes the session tree at the moment the child's turn
    ended. Relayed to the parent it reads as a standing fact about the
    parent's budget; the parent gets the live figures each turn instead.
    """
    if not text:
        return text or ""
    out = [ln for ln in str(text).splitlines() if not _RUNTIME_NOTICE_LINE_RE.search(ln)]
    return "\n".join(out).rstrip()


def live_budget_preamble_block(session: Any) -> dict[str, str] | None:
    """Per-turn live budget figures for engines that rebuild context each turn.

    Rides the rolling preamble tail (never the cached system head) so the
    model always holds the current numbers. Returns ``None`` when the cap is
    spent; the manual-turn notice covers that case.
    """
    try:
        snap = session_budget_snapshot(session)
    except Exception:  # noqa: BLE001 — display only
        return None
    if snap["remaining_usd"] <= 0:
        return None
    return {
        "role": "user",
        "content": (
            "[sharedagents live budget] Session tree spend: "
            f"${snap['spent_usd']:.2f} of ${snap['budget_usd']:.2f}; "
            f"${snap['remaining_usd']:.2f} remains. These live figures replace "
            "any budget number in earlier messages, summaries or subagent "
            "results. Work normally; do not ask for more budget while money "
            "remains."
        ),
    }


def budget_raised_system_prompt(session: Any) -> str:
    """Notice for the first turn after a spent session budget was raised.

    The previous turn's answer ends with the runtime's limit notice. Unless the
    model is told the cap has moved, that stale notice is the only budget fact
    it has: it keeps treating the budget as spent and asks the operator to
    authorize more instead of doing the work. Returns ``""`` when the last turn
    carries no limit notice or nothing remains to spend.
    """
    turns = getattr(session, "turns", None) or []
    last = turns[-1] if turns else None
    content = str(last.get("content") or "") if isinstance(last, dict) else ""
    if not any(phrase in content for phrase in SESSION_BUDGET_LIMIT_PHRASES):
        return ""
    snap = session_budget_snapshot(session)
    if snap["remaining_usd"] <= 0:
        return ""
    return (
        "[sharedagents session budget notice]\n"
        "The session budget has been raised since the limit notice in the "
        f"previous reply: ${snap['spent_usd']:.2f} spent of "
        f"${snap['budget_usd']:.2f}; ${snap['remaining_usd']:.2f} remains.\n"
        "That limit notice no longer applies. Carry out the request normally. "
        "Do not ask for more budget or change the budget unless the operator "
        "asks."
    )


def lookup_session_budget_usd(session_hash: str) -> float:
    """Effective tree cap for ``session_hash`` from ``sessions_meta``.

    Reads the root row's ``session_budget_usd`` (children inherit the root
    cap). Legacy rows that predate the session cap inherit the bound goal's
    stored budget before falling back to :data:`DEFAULT_BUDGET_USD`.
    """
    sh = str(session_hash or "")
    if not sh:
        return DEFAULT_BUDGET_USD
    try:
        from pathlib import Path

        from sharedagents_core.layout import sessions_root
        from sharedagents_session.session_index import connect_sessions_db

        con = connect_sessions_db(Path(sessions_root()) / "sessions.db")
        try:
            row = con.execute(
                "SELECT cost_root_session_id, session_budget_usd, goal_id, "
                "cost_ancestor_goal_id "
                "FROM sessions_meta WHERE session_hash = ?",
                (sh,),
            ).fetchone()
            d = dict(row) if row else {}
            root = str(d.get("cost_root_session_id") or "") or sh
            raw = d.get("session_budget_usd")
            goal_id = str(d.get("goal_id") or d.get("cost_ancestor_goal_id") or "")
            if root and root != sh:
                rrow = con.execute(
                    "SELECT session_budget_usd, goal_id, cost_ancestor_goal_id "
                    "FROM sessions_meta "
                    "WHERE session_hash = ?",
                    (root,),
                ).fetchone()
                if rrow is not None:
                    rd = dict(rrow)
                    raw = rd.get("session_budget_usd", raw)
                    goal_id = str(rd.get("goal_id") or rd.get("cost_ancestor_goal_id") or goal_id)
        finally:
            con.close()
        try:
            if raw is not None and float(raw) > 0:
                return resolve_budget_usd(float(raw))
        except (TypeError, ValueError):
            pass

        # Before ``session_budget_usd`` was added, the goal row was the only
        # durable cost cap. Preserve that cap when opening an old session so a
        # migration does not silently turn a historical $5/$50 goal into $25.
        try:
            reg = get_goal_registry()
            goal = reg.get(goal_id) if goal_id else reg.goal_id_for_root_session(root)
            if goal is not None and int(getattr(goal, "max_cost_pico", 0) or 0) > 0:
                return resolve_budget_usd(
                    pico_to_usd(int(goal.max_cost_pico)),
                )
        except Exception:  # noqa: BLE001
            pass
        return DEFAULT_BUDGET_USD
    except Exception:  # noqa: BLE001
        return DEFAULT_BUDGET_USD


def _effective_goal_cap_usd(goal: Any) -> float:
    """USD cap for this goal row alone — independent of the session tree cap."""
    goal_cap = int(getattr(goal, "max_cost_pico", 0) or 0)
    if goal_cap > 0:
        return resolve_budget_usd(pico_to_usd(goal_cap))
    return DEFAULT_BUDGET_USD


def _stored_session_budget_usd(session_hash: str) -> float | None:
    """Raw root ``session_budget_usd`` from ``sessions_meta``, or None if unset."""
    sh = str(session_hash or "")
    if not sh:
        return None
    try:
        from pathlib import Path

        from sharedagents_core.layout import sessions_root
        from sharedagents_session.session_index import connect_sessions_db

        con = connect_sessions_db(Path(sessions_root()) / "sessions.db")
        try:
            row = con.execute(
                "SELECT cost_root_session_id, session_budget_usd FROM sessions_meta WHERE session_hash = ?",
                (sh,),
            ).fetchone()
            d = dict(row) if row else {}
            root = str(d.get("cost_root_session_id") or "") or sh
            raw = d.get("session_budget_usd")
            if root and root != sh:
                rrow = con.execute(
                    "SELECT session_budget_usd FROM sessions_meta WHERE session_hash = ?",
                    (root,),
                ).fetchone()
                if rrow is not None:
                    raw = dict(rrow).get("session_budget_usd", raw)
        finally:
            con.close()
        if raw is None:
            return None
        n = float(raw)
        return n if n > 0 else None
    except Exception:  # noqa: BLE001
        return None


def tree_cap_usd(session: Any) -> float:
    """USD cap for the session tree — always the ROOT's stored cap.

    A child session's own ``session_budget_usd`` (usually 0) is ignored so
    every engine in the tree shares one number. The durable root row wins
    over a stale in-memory copy so a mid-turn ``/budget`` is visible to
    the admission gate.
    """
    root = session_root_id(session)
    sid = str(getattr(session, "hash", "") or "")
    durable = _stored_session_budget_usd(root or sid)
    if durable is not None:
        return resolve_budget_usd(durable)
    if root and sid and root != sid:
        return lookup_session_budget_usd(root)
    stored = raw_session_budget_usd(session)
    if stored is not None:
        return resolve_budget_usd(stored)
    return lookup_session_budget_usd(root or sid)


@dataclass(frozen=True)
class GoalBudgetState:
    """Both budget scopes of one goal, each spend paired with its own cap.

    ``goal_*`` is the spend attributed to the goal against the goal's cap;
    ``tree_*`` is the whole session tree against the session cap. The two are
    never cross-compared: a goal set in a session that already spent most of
    its session budget still starts at zero against its own cap.
    """

    goal_spent_pico: int
    goal_cap_pico: int
    tree_spent_pico: int
    tree_cap_pico: int

    def exhausted_scope(self, reserve_pico: int = 0) -> str:
        """``"goal"`` / ``"session"`` for the cap that ``reserve_pico`` more
        spend would reach (goal first), or ``""`` while both have room."""
        reserve = max(0, int(reserve_pico or 0))
        if 0 < self.goal_cap_pico <= self.goal_spent_pico + reserve:
            return "goal"
        if 0 < self.tree_cap_pico <= self.tree_spent_pico + reserve:
            return "session"
        return ""


def _cost_root_for_hash(session_hash: str) -> str:
    """Root session id that ``session_hash``'s spend rolls up to."""
    sh = str(session_hash or "")
    if not sh:
        return ""
    try:
        from pathlib import Path

        from sharedagents_core.layout import sessions_root
        from sharedagents_session.session_index import connect_sessions_db

        con = connect_sessions_db(Path(sessions_root()) / "sessions.db")
        try:
            row = con.execute(
                "SELECT cost_root_session_id FROM sessions_meta WHERE session_hash = ?",
                (sh,),
            ).fetchone()
        finally:
            con.close()
        if row is not None:
            return str(dict(row).get("cost_root_session_id") or "") or sh
    except Exception:  # noqa: BLE001
        pass
    return sh


def goal_budget_state(goal: Any, session: Any = None) -> GoalBudgetState:
    """The one budget reading every goal limiter uses.

    Turn finalization, the supervisor checkpoints and the goal top-ups all
    read this, so they agree on what was spent against which cap. ``session``
    is the goal's root session when the caller has it loaded; without it the
    session figures come from the durable rows.
    """
    goal_id = str(getattr(goal, "goal_id", "") or "")
    if session is not None:
        root = session_root_id(session)
        tree_cap = tree_cap_usd(session)
    else:
        bound = str(getattr(goal, "root_session_id", "") or "")
        root = _cost_root_for_hash(bound)
        tree_cap = lookup_session_budget_usd(bound)
    return GoalBudgetState(
        goal_spent_pico=_goal_cost_pico(goal_id),
        goal_cap_pico=int(usd_to_pico(_effective_goal_cap_usd(goal))),
        tree_spent_pico=_session_cost_pico(root),
        tree_cap_pico=int(usd_to_pico(tree_cap)),
    )


def goal_child_refusal(goal_id: str) -> str:
    """Why a running goal may not start another child, or ``""``.

    A goal that reached its own cap stays running until the work it already
    admitted stops; this is what keeps that window from admitting more. The
    session cap is refused separately, for every session with or without a
    goal. A goal that is not running limits nothing here: an operator's own
    prompt is not goal work.
    """
    gid = str(goal_id or "")
    if not gid:
        return ""
    try:
        goal = get_goal_registry().get(gid)
        if goal is None or goal.status != STATUS_RUNNING:
            return ""
        spent = _goal_cost_pico(gid)
        cap = int(usd_to_pico(_effective_goal_cap_usd(goal)))
        if 0 < cap <= spent:
            return (
                "[sharedagents budget] goal cost budget reached "
                f"(${pico_to_usd(spent):.4f} spent / ${pico_to_usd(cap):.2f} budget) — "
                "new child refused. Ask the user to authorize more with "
                "`/goal topup`."
            )
    except Exception:  # noqa: BLE001
        pass
    return ""


def remaining_for_session_hash(session_hash: str, *, extra_pico: int = 0) -> float:
    """USD still available for ``session_hash``'s tree. No Session object needed."""
    sh = str(session_hash or "")
    cap = lookup_session_budget_usd(sh)
    root = sh
    try:
        from pathlib import Path

        from sharedagents_core.layout import sessions_root
        from sharedagents_session.session_index import connect_sessions_db

        con = connect_sessions_db(Path(sessions_root()) / "sessions.db")
        try:
            row = con.execute(
                "SELECT cost_root_session_id FROM sessions_meta WHERE session_hash = ?",
                (sh,),
            ).fetchone()
            if row is not None:
                root = str(dict(row).get("cost_root_session_id") or "") or sh
        finally:
            con.close()
    except Exception:  # noqa: BLE001
        root = sh
    spent = pico_to_usd(_session_cost_pico(root) + int(extra_pico or 0))
    return max(0.0, cap - spent)


# Budget severity thresholds (green→amber→red). Single source of truth so
# the policy is tested here, not hidden in CSS: the frontend maps the returned
# ``severity`` to a brand semantic token (spec/brand-book.md).
_BUDGET_WARN_RATIO = 0.75  # ≥ 75% of budget → warning (amber)
_BUDGET_CRIT_RATIO = 0.90  # ≥ 90% (approaching/over a hard cap) → critical (red)


def budget_severity(spent_usd: float, budget_usd: float | None) -> str:
    """ "ok" | "warn" | "crit" for a spend/budget pair. Unbounded (no budget) is
    always healthy — there is no limit being approached, and never a /0.
    """
    if budget_usd is None or budget_usd <= 0:
        return "ok"
    ratio = float(spent_usd) / float(budget_usd)
    if ratio >= _BUDGET_CRIT_RATIO:
        return "crit"
    if ratio >= _BUDGET_WARN_RATIO:
        return "warn"
    return "ok"


def _budget_pair(spent_usd: float, budget_usd: float | None) -> dict[str, Any]:
    return {
        "spent_usd": spent_usd,
        "budget_usd": budget_usd,
        "severity": budget_severity(spent_usd, budget_usd),
    }


def stamp_budget_from_goal(
    budget: dict[str, Any] | None,
    goal: dict[str, Any] | None,
) -> dict[str, Any]:
    """Fill a $0 ledger view from the goal widget's persisted cost.

    Isolated / missing ``sessions.db`` leaves ``spent_usd`` at 0 while the
    goal row already has ``cost_usd``. The spend chip reads ``budget.total``.
    """
    view = dict(budget or {})
    total = dict(view.get("total") or {})
    try:
        spent = float(total.get("spent_usd") or 0)
    except (TypeError, ValueError):
        spent = 0.0
    if not isinstance(goal, dict):
        if total and "total" not in view:
            view["total"] = total
        return view

    goal_cost = _finite_usd(goal.get("cost_usd")) if goal.get("cost_usd") not in (None, "") else None
    try:
        cap = float(total.get("budget_usd")) if total.get("budget_usd") is not None else None
    except (TypeError, ValueError):
        cap = None
        total.pop("budget_usd", None)
    if spent == 0 and goal_cost:
        total["spent_usd"] = goal_cost
        total["severity"] = budget_severity(goal_cost, cap)
        view["total"] = total
    return view


def hierarchical_budget_view(session_hash: str) -> dict[str, Any]:
    """Downstream (this agent's own spend) + total (root-tree spend) budget view
    for the web chat header.

    The root session owns the budget (a session always has one, goal or not).
    Child turns roll up to that root. ``total`` is the whole tree vs the root
    session cap; ``downstream`` is this session's own turns. Shape:

        { "is_subagent": bool,
          "total":      { "spent_usd": float, "budget_usd": float|None },
          "downstream": { "spent_usd": float, "budget_usd": float|None } }  # iff is_subagent

    ``budget_usd`` is never ``None``/∞: it floors to ``DEFAULT_BUDGET_USD`` when no
    explicit budget is set. Best-effort — returns ``{}`` on any
    failure so the widget simply hides.

    ``delegate`` is always present: the expensive-delegate spend
    rolled up under the root vs the same root budget — the delegated
    portion of ``total``, so the footer indicator shows delegate spend inline.
    """
    sh = str(session_hash or "")
    if not sh:
        return {}
    try:
        from pathlib import Path

        from sharedagents_core.layout import sessions_root
        from sharedagents_model.usage_ledger import get_usage_ledger
        from sharedagents_session.session_index import connect_sessions_db

        sdir = Path(sessions_root())
        con = connect_sessions_db(sdir / "sessions.db")
        try:
            row = con.execute(
                "SELECT cost_root_session_id, cost_ancestor_goal_id, goal_id FROM sessions_meta WHERE session_hash = ?",
                (sh,),
            ).fetchone()
            d = dict(row) if row else {}
            root = str(d.get("cost_root_session_id") or "") or sh
        finally:
            con.close()

        led = get_usage_ledger()
        total_budget_usd = lookup_session_budget_usd(root)

        view: dict[str, Any] = {
            "is_subagent": bool(root and root != sh),
            "total": _budget_pair(pico_to_usd(led.session_cost_pico(root)), total_budget_usd),
        }
        # Surface the expensive-delegate portion of the tree
        # spend (recorded delegate spend) against the same root budget,
        # so the footer indicator shows delegate spend without a second
        # widget. Delegate spend = whole-tree total minus the root's OWN turns;
        # prefer the ledger's dedicated query, fall back to the arithmetic for
        # older ledger fakes so this never breaks the view.
        _deleg_fn = getattr(led, "session_delegate_cost_pico", None)
        if callable(_deleg_fn):
            _deleg_pico = _deleg_fn(root)
        else:
            _deleg_pico = max(0, led.session_cost_pico(root) - led.session_own_cost_pico(root))
        view["delegate"] = _budget_pair(pico_to_usd(_deleg_pico), total_budget_usd)
        if view["is_subagent"]:
            # MVP: a child inherits the root budget until a per-child allocation
            # is set — the requirement's "root can set budget to its child" clause
            # is the write-side follow-up; the display reads whatever is set.
            view["downstream"] = _budget_pair(pico_to_usd(led.session_own_cost_pico(sh)), total_budget_usd)
        return view
    except Exception:  # noqa: BLE001
        return {}


def split_budget(text: str) -> tuple[str, float | None]:
    """Pull an optional ``--budget <usd>`` (or ``--cost``) out of a /goal
    condition string. Returns (cleaned_condition, usd) or (text, None)."""
    m = re.search(r"--(?:budget|cost)\s+\$?(\d+(?:\.\d+)?)", text)
    if not m:
        return text.strip(), None
    try:
        usd = float(m.group(1))
    except ValueError:
        return text.strip(), None
    cleaned = (text[: m.start()] + text[m.end() :]).strip()
    return cleaned, usd


@dataclass
class GoalCommandResult:
    ok: bool
    message: str
    goal: GoalRecord | None = None
    conflicts: list[dict] | None = None


def _conflicts_payload(exc: GoalConflictError) -> list[dict]:
    return [c.to_dict() for c in exc.conflicts]


def ensure_session_goal_migrated(session: Any, agent_name: str) -> GoalRecord | None:
    """Lazy migration from legacy session-scoped active_goal."""
    active = str(getattr(session, "active_goal", "") or "").strip()
    goal_id = str(getattr(session, "goal_id", "") or "").strip()
    if not active or goal_id:
        return None
    reg = get_goal_registry()
    goal = reg.migrate_from_session(
        session_id=session.hash,
        agent_name=agent_name,
        active_goal=active,
        goal_terminal=str(getattr(session, "goal_terminal", "") or ""),
        goal_turn_count=int(getattr(session, "goal_turn_count", 0) or 0),
    )
    session.goal_id = goal.goal_id
    try:
        session.save()
    except Exception:  # noqa: BLE001
        pass
    return goal


def create_goal_for_session(
    agent: Any,
    session: Any,
    *,
    condition_text: str,
    max_turns: int | None = None,
    max_cost_usd: float | None = None,
    scope_key: str | None = None,
    force: bool = False,
    created_by: str | None = None,
    replace_existing: bool = True,
) -> GoalCommandResult:
    """Create a durable goal for ``session``, resolving its cost budget from
    ``agent.cfg.max_goal_cost_usd`` exactly once (only when the caller didn't
    pass one explicitly) — the budget is never re-derived from config again
    after this point.
    """
    reg = get_goal_registry()
    # ``mt`` is stored for schema/UI back-compat ONLY — it never terminates the
    # outer loop (the loop is cost-terminated; see goal_supervisor._drive_loop).
    # An opt-in turn cap was rejected — owner override 2026-07-11
    # "only cost limit no turn limit".
    mt = (
        max_turns
        if max_turns is not None
        else int(
            getattr(agent.cfg, "max_goal_turns", 0) or 0,
        )
    )
    # Goal budget is independent of the session-tree cap. An explicit
    # caller value wins; otherwise the config default. Never copy or
    # overwrite the session cap here — /budget owns that number.
    if max_cost_usd is not None:
        budget_usd = float(max_cost_usd)
    else:
        budget_usd = float(getattr(agent.cfg, "max_goal_cost_usd", DEFAULT_BUDGET_USD) or 0.0)
    budget_usd = resolve_budget_usd(budget_usd)
    max_cost_pico = usd_to_pico(budget_usd)
    try:
        goal = reg.create_goal(
            condition_text=condition_text,
            agent_name=agent.name,
            root_session_id=session.hash,
            created_from_session_id=session.hash,
            created_by=created_by,
            max_turns=mt,
            max_cost_pico=max_cost_pico,
            scope_key=scope_key,
            force=force,
            replace_existing=replace_existing,
        )
    except GoalConflictError as exc:
        return GoalCommandResult(
            ok=False,
            message="could not create goal for this session",
            conflicts=_conflicts_payload(exc),
        )

    sync_session_goal_projection(session, goal)
    sup = _supervisor(agent)
    if sup is not None:
        # Only start supervision here when we're actually on an event loop (the
        # gateway daemon). On the web/WSGI request thread there is NO running
        # loop, so ensure_started would merely CLAIM then release the supervisor
        # lease (it needs a loop to create the drive task) — churning the lease
        # and opening a window that makes the gateway's follow-up supervise kick
        # lose the claim and back off ~125s before the first turn. Skipping it
        # off-loop lets that gateway call (or the /goal command, which already
        # runs on the gateway) claim cleanly and start driving immediately.
        import asyncio

        try:
            asyncio.get_running_loop()
            _on_loop = True
        except RuntimeError:
            _on_loop = False
        if _on_loop:
            sup.ensure_started(goal.goal_id, source="create")
            # /goal set also submits the condition as a visible chat turn.
            # Yield the first auto-turn so that prompt is what the operator
            # sees. Supervision resumes on the following turn_done.
            sid = getattr(session, "hash", None) or getattr(session, "id", None)
            if sid:
                try:
                    sup.notify_manual_turn_start(str(sid))
                except Exception:  # noqa: BLE001
                    pass
    return GoalCommandResult(
        ok=True,
        message=f"goal created: {goal.goal_id}",
        goal=goal,
    )


def pause_goal(
    agent: Any,
    goal_id: str,
    *,
    reason: str | None = None,
) -> GoalCommandResult:
    reg = get_goal_registry()
    goal = reg.get(goal_id)
    if goal is None:
        return GoalCommandResult(ok=False, message=f"goal not found: {goal_id}")
    goal = reg.pause_goal(goal_id, reason=reason)
    sup = _supervisor(agent)
    if sup is not None:
        sup.cancel(goal_id, reason=reason or "paused")
    _sync_root_session(agent, goal)
    return GoalCommandResult(ok=True, message=f"goal paused: {goal_id}", goal=goal)


def resume_goal(
    agent: Any,
    goal_id: str,
    session: Any,
    *,
    force: bool = False,
) -> GoalCommandResult:
    reg = get_goal_registry()
    goal = reg.get(goal_id)
    if goal is None:
        return GoalCommandResult(ok=False, message=f"goal not found: {goal_id}")
    if goal.status == STATUS_ARCHIVED:
        return GoalCommandResult(ok=False, message="goal is archived")
    try:
        goal = reg.bind_root_session(
            goal_id=goal_id,
            new_root_session_id=session.hash,
            agent_name=agent.name,
            reason="resume",
            force=force,
        )
    except GoalConflictError as exc:
        return GoalCommandResult(
            ok=False,
            message="could not resume goal on this session",
            conflicts=_conflicts_payload(exc),
        )
    sync_session_goal_projection(session, goal)
    sup = _supervisor(agent)
    if sup is not None:
        sup.ensure_started(goal.goal_id, source="resume")
    return GoalCommandResult(ok=True, message=f"goal resumed: {goal_id}", goal=goal)


def restart_goal(
    agent: Any,
    goal_id: str,
    session: Any,
    *,
    force: bool = False,
) -> GoalCommandResult:
    reg = get_goal_registry()
    goal = reg.get(goal_id)
    if goal is None:
        return GoalCommandResult(ok=False, message=f"goal not found: {goal_id}")
    try:
        goal = reg.bind_root_session(
            goal_id=goal_id,
            new_root_session_id=session.hash,
            agent_name=agent.name,
            reason="restart",
            force=force,
        )
    except GoalConflictError as exc:
        return GoalCommandResult(
            ok=False,
            message="restart conflict",
            conflicts=_conflicts_payload(exc),
        )
    session.goal_turn_count = 0
    session.goal_terminal = ""
    reg.update_turn_count(goal.goal_id, 0)
    goal = reg.get(goal.goal_id)
    sync_session_goal_projection(session, goal)
    sup = _supervisor(agent)
    if sup is not None:
        sup.cancel(goal_id, reason="restart", pause=False, release_lease=False)
        sup.ensure_started(goal.goal_id, source="restart")
    return GoalCommandResult(
        ok=True,
        message=f"goal restarted on {session.hash}: {goal_id}",
        goal=goal,
    )


def _propagate_live_session_budget(agent: Any, root: str, session: Any, usd: float) -> None:
    """Copy the new cap onto every in-memory Session we can reach."""
    seen: set[int] = set()

    def _apply(obj: Any) -> None:
        if obj is None:
            return
        ident = id(obj)
        if ident in seen:
            return
        seen.add(ident)
        try:
            obj.session_budget_usd = usd
        except Exception:  # noqa: BLE001
            pass

    _apply(session)
    if not root:
        return
    for holder in (getattr(agent, "engine", None), agent):
        bag = getattr(holder, "_sessions", None)
        if isinstance(bag, dict):
            _apply(bag.get(root))


def _stamp_session_budget(agent: Any, session: Any, usd: float) -> bool:
    """Write the session-tree cap onto the root session and persist it.

    Delegation can cross agent ownership, so ``agent.get_session(root)`` may
    intentionally refuse the row. In that case update the root JSONL through
    the shared writer instead of silently leaving the child tree uncapped.
    """
    eff = resolve_budget_usd(usd)
    target = session
    root = session_root_id(session) if session is not None else ""
    getter = getattr(agent, "get_session", None) if agent is not None else None
    if callable(getter) and root and (session is None or root != str(getattr(session, "hash", "") or "")):
        try:
            loaded = getter(root)
        except Exception:  # noqa: BLE001
            loaded = None
        if loaded is not None:
            target = loaded
    if target is None:
        return False
    target.session_budget_usd = eff
    persisted = False
    try:
        saver = getattr(target, "save_header_fields", None)
        if callable(saver):
            saver(("session_budget_usd",))
        else:
            target.save()
        persisted = True
    except Exception:  # noqa: BLE001
        # If the target was loaded through a different Agent, its ownership
        # guard may reject ``save``. Fall through to the shared root writer.
        pass

    if not persisted:
        if not root:
            return False
        try:
            from pathlib import Path

            from sharedagents_core.layout import sessions_root
            from sharedagents_session.session_writer import (
                get_session_writer,
                release_session_writer,
            )

            storage = getattr(session, "_storage_root", None) or getattr(agent, "sessions_dir", None)
            root_dir = Path(storage) if storage is not None else Path(sessions_root())
            writer = get_session_writer(
                root_dir,
                root,
                engine=str(getattr(target, "engine", "") or ""),
                pin=True,
            )
            try:
                writer.header_update(patch={"session_budget_usd": eff})
            finally:
                release_session_writer(writer)
            persisted = True
        except Exception:  # noqa: BLE001
            return False
    if persisted:
        _propagate_live_session_budget(agent, root, session, eff)
        _propagate_live_session_budget(agent, root, target, eff)
    return persisted


def set_session_budget(
    agent: Any,
    session: Any,
    usd: float,
) -> GoalCommandResult:
    """Set this session tree's USD cap. Does not require a goal.

    ``0`` resets to the $25 default. The bound goal's cap is left alone —
    goal budget and session budget are independent.
    """
    if session is None:
        return GoalCommandResult(ok=False, message="no session")
    parsed = _finite_usd(usd)
    if parsed is None or parsed < 0:
        return GoalCommandResult(ok=False, message="budget must be a finite number >= 0")
    eff = resolve_budget_usd(parsed)
    if not _stamp_session_budget(agent, session, eff):
        return GoalCommandResult(
            ok=False,
            message="session budget could not be persisted",
        )
    gid = str(getattr(session, "goal_id", "") or getattr(session, "cost_ancestor_goal_id", "") or "")
    goal = get_goal_registry().get(gid) if gid else None
    spent = pico_to_usd(_session_cost_pico(session_root_id(session)))
    return GoalCommandResult(
        ok=True,
        message=f"session budget set to ${eff:.2f} (spent ${spent:.4f})",
        goal=goal,
    )


def topup_session_budget(
    agent: Any,
    session: Any,
    usd: float,
    *,
    force: bool = False,
) -> GoalCommandResult:
    """Raise this session tree's cap by ``usd`` and resume a budget-paused goal."""
    if session is None:
        return GoalCommandResult(ok=False, message="no session")
    parsed = _finite_usd(usd)
    if parsed is None or parsed <= 0:
        return GoalCommandResult(ok=False, message="topup amount must be a finite number > 0")
    current = tree_cap_usd(session)
    res = set_session_budget(agent, session, current + parsed)
    if not res.ok:
        return res
    gid = str(getattr(session, "goal_id", "") or getattr(session, "cost_ancestor_goal_id", "") or "")
    if not gid:
        return res
    rec = get_goal_registry().get(gid)
    if rec is None or rec.status != STATUS_PAUSED:
        return res
    if (rec.terminal_reason or "") != "cost_budget_exhausted":
        return res
    scope = goal_budget_state(rec, session).exhausted_scope()
    if scope == "goal":
        return GoalCommandResult(
            ok=False,
            message=(f"{res.message} — the goal's own budget is spent; raise it with /goal topup to resume"),
            goal=rec,
        )
    if scope:
        return GoalCommandResult(
            ok=False,
            message=(f"{res.message} — still over spend; top up more to resume"),
            goal=rec,
        )
    resume_goal(agent, gid, session, force=force)
    return GoalCommandResult(
        ok=True,
        message=f"{res.message} — resumed",
        goal=get_goal_registry().get(gid),
    )


def set_goal_budget(
    agent: Any,
    goal_id: str,
    usd: float,
) -> GoalCommandResult:
    """Set a goal's cost budget to an absolute USD value. ``0`` resets to the
    default ($25) — the budget is never unbounded.

    Does not change the session-tree cap. Use ``/budget`` for that.
    """
    _ = agent
    reg = get_goal_registry()
    goal = reg.get(goal_id)
    if goal is None:
        return GoalCommandResult(ok=False, message=f"goal not found: {goal_id}")
    if not math.isfinite(usd) or usd < 0:
        return GoalCommandResult(ok=False, message="budget must be a finite number >= 0")
    eff = resolve_budget_usd(usd)  # 0/negative → DEFAULT_BUDGET_USD, never the unbounded sentinel
    goal = reg.set_max_cost_pico(goal_id, usd_to_pico(eff))
    spent = pico_to_usd(_goal_cost_pico(goal_id))
    label = f"${eff:.2f}"
    return GoalCommandResult(
        ok=True,
        message=f"goal {goal_id} budget set to {label} (spent ${spent:.4f})",
        goal=goal,
    )


def topup_goal(
    agent: Any,
    goal_id: str,
    usd: float,
    session: Any,
    *,
    force: bool = False,
) -> GoalCommandResult:
    """Raise a goal's cost budget by ``usd`` and resume it (continue-with-more)."""
    reg = get_goal_registry()
    goal = reg.get(goal_id)
    if goal is None:
        return GoalCommandResult(ok=False, message=f"goal not found: {goal_id}")
    if not math.isfinite(usd) or usd <= 0:
        return GoalCommandResult(ok=False, message="topup amount must be a finite number > 0")

    goal = reg.add_max_cost_pico(goal_id, usd_to_pico(usd))
    spent_pico = _goal_cost_pico(goal_id)
    spent = pico_to_usd(spent_pico)
    new_budget = _effective_goal_cap_usd(goal) if goal else 0.0

    # Only resume a goal that was paused for budget exhaustion AND whose raised
    # budget now actually exceeds spend — otherwise resuming would immediately
    # re-pause (or revive a met/blocked/running goal it shouldn't touch).
    resumed = False
    if goal and goal.status == STATUS_PAUSED and (goal.terminal_reason or "") == "cost_budget_exhausted":
        scope = goal_budget_state(goal, session).exhausted_scope()
        if not scope:
            res = resume_goal(agent, goal_id, session, force=force)
            resumed = res.ok
            goal = reg.get(goal_id)
        elif scope == "goal":
            return GoalCommandResult(
                ok=False,
                message=(
                    f"goal {goal_id} budget now ${new_budget:.2f} but ${spent:.4f} "
                    "already spent — top up more to resume"
                ),
                goal=goal,
            )
        else:
            return GoalCommandResult(
                ok=False,
                message=(
                    f"goal {goal_id} budget now ${new_budget:.2f} "
                    f"(spent ${spent:.4f}) but the session budget is spent — "
                    "raise it with /budget add to resume"
                ),
                goal=goal,
            )
    tail = "resumed" if resumed else f"not resumed (status={goal.status})"
    return GoalCommandResult(
        ok=True,
        message=(f"goal {goal_id} budget +${usd:.2f} → ${new_budget:.2f} (spent ${spent:.4f}); {tail}"),
        goal=goal,
    )


def complete_goal(
    agent: Any,
    goal_id: str,
    *,
    summary: str | None = None,
) -> GoalCommandResult:
    """Mark a goal ``met`` and stop its supervisor."""
    reg = get_goal_registry()
    if reg.get(goal_id) is None:
        return GoalCommandResult(ok=False, message=f"goal not found: {goal_id}")
    goal = reg.mark_met(goal_id, reason="goal_met", summary=summary, owner_confirmed=True)
    if goal.status != STATUS_MET:
        return GoalCommandResult(
            ok=False,
            message=f"goal not marked met: {goal_id} (status={goal.status})",
            goal=goal,
        )
    sup = _supervisor(agent)
    if sup is not None:
        sup.cancel(goal_id, reason="met")
    _sync_root_session(agent, goal)
    return GoalCommandResult(ok=True, message=f"goal met: {goal_id}", goal=goal)


# ---- Claude Code /goal semantics -----------------------------------------
#
# After every turn the goal evaluator (goal_evaluator.py) judges the condition
# against the transcript; the supervisor acts on its verdict. The model has
# no way to end, pause or rewrite a goal. These helpers serve the owner side.


def live_goal_for_session(session: Any) -> GoalRecord | None:
    """The running / paused / blocked goal bound to this root session."""
    if session is None:
        return None
    root = str(
        getattr(session, "root_session_id", "") or getattr(session, "hash", "") or "",
    ).strip()
    reg = get_goal_registry()
    gid = (reg.goal_id_for_root_session(root) if root else "") or str(
        getattr(session, "goal_id", "") or "",
    ).strip()
    goal = reg.get(gid) if gid else None
    if goal is None or goal.status in (STATUS_MET, STATUS_ARCHIVED):
        return None
    return goal


def goal_status_text(goal: GoalRecord) -> str:
    """``/goal`` view (Claude Code: condition, last check, turns, spend)."""
    from datetime import datetime, timezone

    state = goal.contract or {}
    head = "Goal active" if goal.status == STATUS_RUNNING else f"Goal {goal.status}"
    lines = [f"{head}: {goal.condition_text}"]
    wait = goal.wait_label
    if wait:
        lines.append(wait)
    last = str(state.get("last_check") or "").strip()
    lines.append(f"Last check: {last or 'not yet evaluated'}")
    elapsed = ""
    started = goal.started_at or goal.created_at
    if started:
        try:
            dt = datetime.fromisoformat(str(started).replace("Z", "+00:00"))
            secs = int((datetime.now(timezone.utc) - dt).total_seconds())
            elapsed = f"{secs // 3600}h{(secs % 3600) // 60:02d}m" if secs >= 3600 else f"{secs // 60}m{secs % 60:02d}s"
        except ValueError:
            elapsed = ""
    spent = pico_to_usd(_goal_cost_pico(goal.goal_id))
    lines.append(
        f"{goal.goal_id} · elapsed {elapsed or '?'} · turns {goal.turn_count} · "
        f"checks {int(state.get('evaluations', 0) or 0)} · "
        f"cost ${spent:.4f}/${_effective_goal_cap_usd(goal):.2f}"
    )
    return "\n".join(lines)


def resume_goal_on_user_message(agent: Any, session: Any) -> bool:
    """The user's next message resumes a goal the runtime paused (no progress,
    evaluator or turn errors) — Claude Code keeps the goal set and continues
    after the next prompt. Supervision restarts at the end of this turn, which
    the evaluator then judges. Owner pauses and budget pauses are untouched."""
    from sharedagents_goal.goal_registry import AUTO_PAUSE_REASONS, STATUS_PAUSED

    if session is None:
        return False
    sid = str(getattr(session, "hash", "") or "")
    reg = get_goal_registry()
    gid = str(getattr(session, "goal_id", "") or "")
    goal = reg.get(gid) if gid else None
    if goal is None or goal.status != STATUS_PAUSED:
        return False
    if goal.terminal_reason not in AUTO_PAUSE_REASONS or goal.root_session_id != sid:
        return False
    try:
        goal = reg.bind_root_session(
            goal_id=gid,
            new_root_session_id=sid,
            agent_name=goal.agent_name,
            reason="user_message",
            expected_pause_reasons=AUTO_PAUSE_REASONS,
        )
    except Exception:  # noqa: BLE001
        return False
    reg.mutate_contract(gid, lambda k: k.pop("notice", None))
    reg.update_effective_progress(gid, streak=0)
    sync_session_goal_projection(session, goal)
    sup = _supervisor(agent)
    skip = getattr(sup, "_skip_next_auto", None)
    if isinstance(skip, set):
        # This manual turn is the resume; the loop must drive after it.
        skip.discard(sid)
    return True


def archive_goal(agent: Any, goal_id: str, reason: str | None = None) -> GoalCommandResult:
    """Manual delete (``archive`` is a kept alias). Clears the session projection."""
    reg = get_goal_registry()
    goal = reg.get(goal_id)
    if goal is None:
        return GoalCommandResult(ok=False, message=f"goal not found: {goal_id}")
    snapshot, deleted = reg._archive_goal(goal_id, reason=reason)
    if snapshot is None or not deleted:
        return GoalCommandResult(
            ok=False,
            message=f"goal already deleted: {goal_id}",
            goal=snapshot,
        )
    sup = _supervisor(agent)
    if sup is not None:
        sup.cancel(goal_id, reason="deleted")
    if snapshot.root_session_id:
        try:
            session = agent.get_session(snapshot.root_session_id)
            if session is not None:
                sync_session_goal_projection(session, None)
        except Exception:  # noqa: BLE001
            pass
    return GoalCommandResult(ok=True, message=f"goal deleted: {goal_id}", goal=snapshot)


def _resolve_target_goal_id(
    agent: Any,
    session: Any,
    explicit: str = "",
) -> tuple[str, str | None]:
    """TIER 2a — resolve the durable goal a ``/goal`` action must target.

    Precedence (authoritative-first, never best-effort projection alone):
    1. explicit goal id (the structured ``goal_id`` the model tool now
       carries end-to-end),
    2. the unique RUNNING durable goal bound to this root session
       (``goal_registry.goal_id_for_root_session`` — the registry is the
       system of record),
    3. the session projection ``goal_id`` as the legacy fallback.

    Ownership is validated (agent name + root session binding) before the
    caller is allowed to mutate, so a model cannot archive another agent's
    goal or a goal bound to a different root session. Returns ``(gid, None)``
    on success or ``(gid, error)`` when unresolvable/foreign.
    """
    reg = get_goal_registry()
    gid = (explicit or "").strip()
    if not gid:
        # TIER 2a — the unique RUNNING durable goal bound to this root
        # session is the authoritative answer, NOT the session
        # projection (which a lazy loader can leave split-brain empty).
        root = str(
            getattr(session, "root_session_id", "") or getattr(session, "hash", "") or "",
        ).strip()
        if root:
            gid = reg.goal_id_for_root_session(root) or ""
    if not gid:
        # Legacy fallback: the session projection's goal id.
        gid = str(getattr(session, "goal_id", "") or "").strip()
    if not gid:
        return "", "no goal to act on"
    goal = reg.get(gid)
    if goal is None:
        return gid, f"goal not found: {gid}"
    agent_name = str(getattr(agent, "name", "") or "")
    root = str(
        getattr(session, "root_session_id", "") or getattr(session, "hash", "") or "",
    ).strip()
    if goal.agent_name and agent_name and goal.agent_name != agent_name:
        return gid, f"goal {gid} belongs to agent {goal.agent_name}"
    if root and goal.root_session_id and goal.root_session_id != root:
        return gid, f"goal {gid} is bound to root session {goal.root_session_id}"
    return gid, None


def detach_session_goal(
    session: Any,
    agent: Any | None = None,
    goal_id: str = "",
) -> GoalCommandResult:
    """Delete the session's goal and clear the session projection (``/goal clear``).

    Manual delete — the durable row is removed, then the live session
    projection is cleared so the widget/API drop it immediately.
    """
    gid, err = _resolve_target_goal_id(agent, session, goal_id)
    if err:
        return GoalCommandResult(ok=False, message=err)
    if agent is not None:
        archive_goal(agent, gid, reason="session_clear")
    else:
        reg = get_goal_registry()
        if reg.get(gid) is not None:
            reg.delete_goal(gid)
    sync_session_goal_projection(session, None)
    return GoalCommandResult(ok=True, message=f"goal deleted and detached from session: {gid}")


def dismiss_met_goal_on_follow_up(session: Any) -> bool:
    """Hide a completed-goal linger when the user sends the next ordinary turn.

    The met badge stays on the session projection until this call so the
    chat widget can show ``✓ completed``. The following manual prompt
    detaches that projection (does not archive — the durable row is
    already ``met``) so header/SSE updates no longer resurrect the widget.
    """
    if session is None:
        return False
    if str(getattr(session, "goal_terminal", "") or "").lower() != "met":
        return False
    sync_session_goal_projection(session, None)
    return True


def list_goals(
    *,
    status: str | None = None,
    agent_name: str | None = None,
) -> list[GoalRecord]:
    reg = get_goal_registry()
    return reg.list_goals(GoalFilters(status=status, agent_name=agent_name))


#: ``/goal`` actions that are control verbs, not a new objective — frontends
#: that route control verbs differently from a SET read this one list.
GOAL_CONTROL_ACTIONS = frozenset(
    {
        "list",
        "status",
        "show",
        "pause",
        "resume",
        "restart",
        "met",
        "archive",
        "clear",
        "topup",
        "budget",
    }
)


def parse_goal_subcommand(rest: str) -> tuple[str, str, bool]:
    """Parse ``/goal`` args → (action, payload, force)."""
    text = (rest or "").strip()
    force = False
    if text.startswith("--force "):
        force = True
        text = text[len("--force ") :].strip()
    elif text == "--force":
        return "list", "", True

    if not text:
        return "list", "", False

    parts = text.split(None, 1)
    head = parts[0].lower()
    tail = parts[1].strip() if len(parts) > 1 else ""

    if head in ("clear", "stop", "reset", "off", "none", "done", "cancel"):
        return "clear", "", False
    if head in ("list", "status", "show"):
        return head, tail, force
    if head == "budget":
        # budget keeps strict parsing: its argument is an amount / ``add`` / a
        # goal id, and an invalid budget argument is a usage error — never a new
        # objective (see test_174 "budget foo").
        return head, tail, force
    if head in ("pause", "resume", "restart", "archive", "delete", "topup", "met"):
        # A status verb acts on a goal only when its payload
        # NAMES a goal (a ``goal_…`` id) — or, for ``pause``/``met``, is empty
        # (= the active goal). A status verb that merely HEADS free text is a
        # goal OBJECTIVE ("pause the ingest job when the queue drains") → a
        # SET, so routing every set through this shared parser never misparses it.
        first = tail.split(None, 1)[0] if tail else ""
        if head in ("pause", "met"):
            if not tail or first.startswith("goal_"):
                return head, tail, force
            return "set", text, force
        # resume/restart/archive/delete/topup need an explicit goal id; an
        # empty payload stays the command (its downstream usage error), a
        # goal_… id is the command, anything else is a set-objective.
        if head == "delete":
            if not tail or first.startswith("goal_"):
                return "archive", tail, force
            return "set", text, force
        if not tail or first.startswith("goal_"):
            return head, tail, force
        return "set", text, force
    return "set", text, force


def reconcile_goal_view_status(agent: Any, goal: Any) -> str:
    """Return the status to DISPLAY for ``goal``, reconciling the registry
    against reality on observation — the CLI/TUI cross-surface equivalent of the
    web ``_reconcile_goal_views`` path.

    A goal whose registry row still says ``running`` but whose supervisor
    heartbeat is stale is not supervised. If its root session recorded a
    terminal outcome (met / blocked / paused) we heal goals.db to that
    terminal — the registry is the system of record — and show it. Otherwise
    we check whether the root session subtree still has LIVE work (running
    children / open async jobs via ``session_work_live``): a supervisor drive
    task can exit while the work it delegated keeps running, and that goal is
    still genuinely working. A goal truly mid-flight at supervisor death
    (running, no recorded terminal, no live supervisor, no live subtree work)
    is surfaced honestly as ``unsupervised`` rather than presented as live.
    """
    from sharedagents_goal.goal_registry import (
        _SUPERVISOR_LEASE_S,
        STATUS_BLOCKED,
        STATUS_MET,
        STATUS_PAUSED,
        STATUS_RUNNING,
        _is_goal_starting,
        _iso_age_seconds,
        session_work_live,
    )

    status = str(getattr(goal, "status", "") or "")
    hb = getattr(goal, "supervisor_heartbeat_at", None)
    starting = not hb and _is_goal_starting(
        status,
        getattr(goal, "started_at", None) or getattr(goal, "created_at", None),
        int(getattr(goal, "turn_count", 0) or 0),
    )
    supervised = status == STATUS_RUNNING and (starting or (bool(hb) and _iso_age_seconds(hb) < _SUPERVISOR_LEASE_S))
    if (supervised and not starting) or status not in (STATUS_RUNNING, STATUS_BLOCKED):
        return status
    root = str(getattr(goal, "root_session_id", "") or "")
    if not root:
        return status
    sess = None
    getter = getattr(agent, "get_session", None)
    if callable(getter):
        try:
            sess = getter(root)
        except Exception:  # noqa: BLE001 — a view must never fail on reconcile
            sess = None
    terminal = str(getattr(sess, "goal_terminal", "") or "") if sess is not None else ""
    if terminal in (STATUS_MET, STATUS_BLOCKED, STATUS_PAUSED):
        if starting and terminal == STATUS_PAUSED:
            # A fresh RUNNING registry binding supersedes the previous goal
            # cycle's stale paused projection. Genuine met/blocked outcomes
            # still reconcile immediately, including inside the grace window.
            return status
        reg = get_goal_registry()
        gid = str(getattr(goal, "goal_id", "") or "")
        try:
            if terminal == STATUS_MET:
                reg.mark_met(gid, reason="goal_met_reconciled")
            elif terminal == STATUS_BLOCKED:
                reg.mark_blocked(gid, reason="goal_blocked_reconciled")
            else:
                reg.pause_goal(gid, reason="goal_paused_reconciled")
        except Exception:  # noqa: BLE001 — reconciliation must never break the view
            pass
        return terminal
    if starting:
        return status
    if status == STATUS_RUNNING:
        # No live supervisor and no recorded terminal. The drive task may have
        # exited cleanly (manual-turn skip / finally release) while the work it
        # delegated still runs — ask the session index whether the root subtree
        # has live children / open async jobs (the web reconciler's
        # supervised_live uses the same helper, so CLI and web agree).
        live, _src = session_work_live(root)
        if live:
            return status
        # Mid-flight zombie: surface it honestly, never as live.
        return "unsupervised"
    return status


def execute_goal_command(
    agent: Any,
    session: Any,
    rest: str,
    goal_id: str = "",
) -> GoalCommandResult:
    """The single shared entry point every ``/goal`` action routes through.

    CLI, TUI, web, and the ``goal`` tool all dispatch here, so conflict /
    budget / lease invariants are enforced identically across every interface.

    TIER 2a — ``goal_id`` is the structured explicit target id (the model
    tool now carries it end-to-end); every mutating/status action resolves
    through ``_resolve_target_goal_id`` (explicit > unique running durable
    goal for the root session > projection), with ownership validation.
    """
    action, payload, force = parse_goal_subcommand(rest)

    if action == "clear":
        return detach_session_goal(session, agent, goal_id)

    if action == "list":
        goals = list_goals(agent_name=agent.name)
        lines = [f"{g.goal_id} [{reconcile_goal_view_status(agent, g)}] {g.condition_text[:60]}" for g in goals[:20]]
        return GoalCommandResult(
            ok=True,
            message="\n".join(lines) if lines else "no goals",
        )

    if action in ("status", "show"):
        gid, err = _resolve_target_goal_id(agent, session, goal_id or payload)
        if err:
            if err == "no goal to act on":
                cap = tree_cap_usd(session)
                spent = pico_to_usd(_session_cost_pico(session_root_id(session)))
                return GoalCommandResult(
                    ok=True,
                    message=f"session budget ${spent:.4f}/${cap:.2f} (no active goal)",
                )
            return GoalCommandResult(ok=False, message=err)
        goal = get_goal_registry().get(gid)
        if goal is None:
            return GoalCommandResult(ok=False, message=f"goal not found: {gid}")
        return GoalCommandResult(ok=True, message=goal_status_text(goal), goal=goal)

    if action == "pause":
        gid, err = _resolve_target_goal_id(agent, session, goal_id or payload)
        if err:
            return GoalCommandResult(ok=False, message=err)
        return pause_goal(agent, gid)

    if action == "met":
        # The owner marking the goal met by hand.
        gid, err = _resolve_target_goal_id(agent, session, goal_id or payload)
        if err:
            return GoalCommandResult(ok=False, message=err)
        return complete_goal(agent, gid)

    if action == "resume":
        gid = goal_id or payload
        if not gid:
            return GoalCommandResult(ok=False, message="usage: /goal resume <goal_id>")
        return resume_goal(agent, gid, session, force=force)

    if action == "restart":
        gid = goal_id or payload
        if not gid:
            return GoalCommandResult(ok=False, message="usage: /goal restart <goal_id>")
        return restart_goal(agent, gid, session, force=force)

    if action == "archive":
        # TIER 2a — the structured explicit goal_id (from the model tool)
        # satisfies the archive target; the legacy bare-verb payload still
        # works for CLI users.
        gid, err = _resolve_target_goal_id(agent, session, goal_id or payload)
        if err:
            return GoalCommandResult(ok=False, message=err)
        return archive_goal(agent, gid)

    if action == "topup":
        parts = payload.split()
        if len(parts) < 2:
            return GoalCommandResult(ok=False, message="usage: /goal topup <goal_id> <usd>")
        gid, amt = parts[0], parts[1].lstrip("$")
        try:
            usd = float(amt)
        except ValueError:
            return GoalCommandResult(ok=False, message=f"bad amount: {amt}")
        return topup_goal(agent, gid, usd, session, force=force)

    if action == "budget":
        # /goal budget <usd>            → set THIS session's tree budget
        # /goal budget add <usd>        → top up THIS session's tree
        # /goal budget <goal_id> <usd>  → set a specific goal's budget (+ session)
        parts = payload.split()
        if not parts:
            return GoalCommandResult(ok=False, message="usage: /goal budget <usd>")
        if parts[0].lower() in ("add", "+", "topup"):
            if len(parts) != 2:
                return GoalCommandResult(ok=False, message="usage: /goal budget add <usd>")
            usd = _finite_usd(parts[1].lstrip("+$"))
            if usd is None:
                return GoalCommandResult(ok=False, message=f"bad amount: {parts[1]}")
            return topup_session_budget(agent, session, usd, force=force)
        if parts[0].startswith("goal_"):
            if len(parts) != 2:
                return GoalCommandResult(ok=False, message="usage: /goal budget <goal_id> <usd>")
            usd = _finite_usd(parts[1].lstrip("$"))
            if usd is None:
                return GoalCommandResult(ok=False, message=f"bad amount: {parts[1]}")
            return set_goal_budget(agent, parts[0], usd)
        if len(parts) != 1:
            return GoalCommandResult(ok=False, message="usage: /goal budget <usd>")
        usd = _finite_usd(parts[0].lstrip("$"))
        if usd is None:
            return GoalCommandResult(ok=False, message=f"bad amount: {parts[0]}")
        return set_session_budget(agent, session, usd)

    # set new goal — honor an inline `--budget <usd>`
    condition, budget_usd = split_budget(payload)
    return create_goal_for_session(
        agent,
        session,
        condition_text=condition,
        max_cost_usd=budget_usd,
        force=force,
    )


def _supervisor(agent: Any) -> Any | None:
    return getattr(agent, "goal_supervisor", None)


def stop_goals_for_session(
    agent: Any,
    session_id: str,
    *,
    reason: str = "aborted",
) -> None:
    """Halt the goal outer loop for every RUNNING goal rooted on ``session_id``.

    The operator-stop chokepoint: ``/abort`` and graceful ``/stop`` route here
    so a goal does not silently re-drive after its in-flight turn is killed.
    Delegates to :meth:`GoalSupervisor.cancel_for_session`, which cancels the
    re-drive task AND pauses the goal (resumable). Must run on the LIVE agent
    that owns the supervisor (gateway daemon), never an ephemeral copy.

    Never raises — a bookkeeping failure must not fail the operator's stop.
    """
    if not session_id:
        return
    sup = _supervisor(agent)
    if sup is None:
        return
    try:
        sup.cancel_for_session(session_id, reason=reason)
    except Exception:  # noqa: BLE001
        pass


def run_goal_action(agent: Any, goal_id: str, action: str) -> dict[str, Any]:
    """Run a goal action (``pause`` | ``archive`` | ``resume`` | ``restart``) on ``agent``.

    The gateway-side handler behind the web Pause/Archive/Resume/Restart buttons.
    Must be invoked on the LIVE agent that owns the supervisor so the re-drive
    task is truly controlled (not just the DB status flipped). Returns a JSON-able dict.
    """
    gid = str(goal_id or "").strip()
    if not gid:
        return {"ok": False, "message": "goal_id required"}
    act = str(action or "").strip().lower()
    if act == "pause":
        result = pause_goal(agent, gid)
    elif act in ("archive", "delete"):
        result = archive_goal(agent, gid)
    elif act in ("resume", "restart"):
        reg = get_goal_registry()
        goal = reg.get(gid)
        if goal is None:
            return {"ok": False, "message": f"goal not found: {gid}"}
        session_id = goal.root_session_id or ""
        if not session_id:
            return {"ok": False, "message": "no root session to resume/restart into"}
        session = agent.get_session(session_id)
        if session is None:
            return {"ok": False, "message": "no session to resume/restart into"}
        result = resume_goal(agent, gid, session) if act == "resume" else restart_goal(agent, gid, session)
    else:
        return {"ok": False, "message": f"unknown goal action: {act!r}"}
    return {
        "ok": result.ok,
        "message": result.message,
        "goal": result.goal.to_dict() if result.goal else None,
    }


def _sync_root_session(agent: Any, goal: GoalRecord) -> None:
    if not goal.root_session_id:
        return
    try:
        session = agent.get_session(goal.root_session_id)
        if session is not None:
            sync_session_goal_projection(session, goal)
    except Exception:  # noqa: BLE001
        pass
