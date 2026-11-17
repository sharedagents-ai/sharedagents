"""Shared session-header goal/cost hydration — TIER 2c.

Both session loaders must restore the goal/cost projection IDENTICALLY:

- the Agent loader (``sharedagents_agent.agents.session_lifecycle``) and
- the gateway's CoreBackend lazy loader (``sharedagents_gateway.acp_bridge``).

Historically only the Agent loader did; a ``goal`` tool action routed through
the gateway's lazily-loaded session saw an empty ``goal_id`` — the split-brain
that let ``/goal clear`` force-write an empty projection, after which the
supervisor copied a zero turn count back to goals.db (12 → 1) and later turns
were cost-ledgered with ``ancestor_goal_id = NULL`` (frozen goal spend).

This helper is the ONE restoration path both loaders call, and it reconciles
disagreements from the goal registry (the durable registry is the
system of record):

- header ``goal_id`` empty but a RUNNING durable goal exists for this root
  session → the projection is healed back to that goal (the lazy-loader
  split-brain case);
- registry row says running/paused/blocked → the bound projection is
  restored from that durable record; met/archived → the stale chat
  projection is detached;
- ``cost_ancestor_goal_id`` empty while a durable goal is bound → re-linked so
  the ledger SUM keeps charging the goal.

Registry access is lazy-imported inside the function so this module stays
cycle-free (sharedagents_goal imports sharedagents_session, never the reverse at
import time).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["hydrate_session_goal_fields"]


def hydrate_session_goal_fields(
    sess: Any,
    d: Mapping[str, Any],
    *,
    agent_name: str = "",
    sessions_dir: Any = None,
) -> None:
    """Restore the unified runtime sessions root, agent/engine identity, and
    the goal/cost projection fields onto a freshly built ``Session``.

    Must be called after the Session object is constructed from ``d`` and
    before the engine is attached. Best-effort: registry reconciliation
    failures never raise (a load must not fail on a reconcile miss).
    """
    # Preserve the unified runtime sessions root on the loaded object.
    # Session.save()/save_header_fields() are intentionally callable
    # without a root; without this binding, an isolated/dev session would
    # fall back to the operator's live ~/.sharedagents/sessions store.
    if sessions_dir is not None:
        try:
            sess._storage_root = sessions_dir
        except Exception:  # noqa: BLE001
            pass
    sess.agent_name = d.get("agent_name") or ""
    sess.engine = d.get("engine") or ""
    sess.parent_session_id = d.get("parent_session_id") or ""
    # Real engine model (claude-opus-4-…/composer-2.5/…), folded into
    # the header at a prior TurnDone. Rehydrate so a same-session later
    # turn (e.g. the web reading ``session.engine_model``) and a fresh
    # ``save()`` both see the persisted value, not the "" default.
    sess.engine_model = d.get("engine_model") or ""
    sess.system_prompt_sha256 = d.get("system_prompt_sha256") or ""
    sess.persist_reasoning_traces = bool(d.get("persist_reasoning_traces", 1))
    sess.persist_tool_result_bodies = bool(d.get("persist_tool_result_bodies", 1))
    sess.active_goal = d.get("active_goal") or ""
    sess.goal_id = d.get("goal_id") or ""
    sess.cost_root_session_id = d.get("cost_root_session_id") or ""
    sess.cost_ancestor_goal_id = d.get("cost_ancestor_goal_id") or ""
    try:
        sess.session_budget_usd = float(d.get("session_budget_usd") or 0)
    except (TypeError, ValueError):
        sess.session_budget_usd = 0.0
    sess.goal_turn_count = int(d.get("goal_turn_count") or 0)
    sess.goal_terminal = d.get("goal_terminal") or ""
    # Restore the conversation topic so a later header-mutating save()
    # re-emits the stored topic instead of clobbering it with the
    # "general" dataclass default.
    sess.topic = (d.get("topic") or "general").strip() or "general"

    # TIER 2c — reconcile disagreements from the goal registry (the
    # durable registry is the system of record). Lazy import keeps this
    # module import-cycle-free.
    try:
        from sharedagents_goal import goal_service as _gs
        from sharedagents_goal.goal_registry import get_goal_registry

        # Lazy migration: legacy session-scoped active_goal → durable record.
        if sess.active_goal and not sess.goal_id:
            try:
                migrated = _gs.ensure_session_goal_migrated(sess, agent_name)
                if migrated is not None:
                    sess.goal_id = migrated.goal_id
            except Exception:  # noqa: BLE001
                pass

        reg = get_goal_registry()
        gid = str(getattr(sess, "goal_id", "") or "").strip()
        if not gid:
            # Split-brain healing: a RUNNING durable goal for this root
            # session with an empty projection (the gateway lazy-loader
            # defect) is re-bound so subsequent goal tool actions and the
            # supervisor's copy-back see the real goal id.
            root = str(d.get("root_session_id") or "") or str(
                getattr(sess, "hash", "") or "",
            )
            if root:
                # ``goal_id_for_root_session`` returns the id string itself,
                # not a GoalRecord.
                gid = str(reg.goal_id_for_root_session(root) or "").strip()
        if gid:
            goal = reg.get(gid)
            if goal is not None:
                status = str(goal.status or "")
                if status in ("met", "archived"):
                    # Terminal portfolio rows do not remain attached to chat:
                    # hide the widget on reload exactly as the live transition
                    # does. The historical cost ancestor is intentionally left
                    # intact for ledger attribution.
                    sess.goal_id = ""
                    sess.active_goal = ""
                    sess.goal_terminal = ""
                    sess.goal_turn_count = 0
                else:
                    # The durable registry owns the complete bound projection,
                    # not only its status. This repairs missing/stale text and
                    # progress on both running and resumable paused sessions.
                    sess.goal_id = goal.goal_id
                    sess.active_goal = goal.condition_text
                    sess.goal_turn_count = int(goal.turn_count or 0)
                    sess.goal_terminal = status if status in ("blocked", "paused") else ""
                    # Re-link the cost ancestor so the ledger SUM keeps
                    # charging the goal even when the header lost the link.
                    if not getattr(sess, "cost_ancestor_goal_id", ""):
                        sess.cost_ancestor_goal_id = gid
    except Exception:  # noqa: BLE001 — reconciliation is best-effort
        pass
