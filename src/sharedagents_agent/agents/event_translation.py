"""Engine-event / metric translation helpers for :class:`Agent`.

Pure functions that translate an engine's per-turn outputs (metrics
dicts, cost-recording outcomes) into the shapes the turn loop persists.
Split out of ``agents/agent.py`` so the composition root keeps only the
turn orchestration and these free helpers can be unit-tested in
isolation. Re-exported from ``agents.agent`` for backward compatibility
(``from sharedagents_agent.agents.agent import _fold_engine_metrics`` etc.
still resolves).
"""

from __future__ import annotations

from typing import Any


def _cost_failure_should_surface(
    *,
    root_session_id: str,
    session_hash: str,
    ancestor_goal_id: str,
    has_cost_override: bool = False,
) -> bool:
    """Decide whether a cost-recording failure must SURFACE (raise) vs. be
    swallowed best-effort.

    Every root session is budgeted now, so a persistence failure would make the
    session appear to have spent nothing and allow the tree to run untracked.
    Surface failures for every attributed session (including direct sessions
    with no goal); retain the cross-root predicate for legacy callers that do
    not provide a root id.

    ``has_cost_override`` (a ``cost_usd_override`` metric is present) is passed
    from the call site but DELIBERATELY IGNORED — it is NOT a delegate signal.
    Ordinary multi-round local-LLM turns set that metric for cumulative per-turn
    pricing (``local_llm.py``) and direct external-CLI turns use it too, so the
    old ``"cost_usd_override" in metrics`` delegate criterion mis-classified an
    ordinary paid local turn as a delegate and raised a hard CostRecordingError
    on a transient unwritable ``usage.db``. Root-
    session divergence is the authoritative delegate signal.
    """
    del has_cost_override  # dropped criterion — see docstring
    budgeted = bool(root_session_id)
    delegate = bool(root_session_id and root_session_id != session_hash)
    return budgeted or delegate


def _fold_engine_metrics(audit: dict, ev_metrics: Any) -> dict:
    """Ensure the turn-audit dict carries per-turn metrics.

    ``local_llm`` / ``external_cli`` stash metrics in
    ``request.extra["_engine_audit"]``, but ACP engines (cursor/opencode)
    hand them on ``TurnDone.metrics`` instead. When the audit has no
    metrics of its own, fold in the event's metrics so the downstream
    cost-pricing block and the persisted ``turn_ended.metrics`` both see a
    populated dict (else: blank stats footer + no cost stamp on every ACP
    turn). Never clobbers a non-empty audit metrics dict.

    Two paths:

    * audit already HAS metrics (local_llm / external_cli) → returned
      UNCHANGED (same object); the downstream cost block mutates
      ``audit["metrics"]`` in place to stamp cost — the pre-existing,
      intended behavior.
    * audit LACKS metrics but the event carries them (ACP engines) → return
      a shallow copy whose ``metrics`` is a COPY of ``ev_metrics``, so the
      later in-place cost stamping never mutates the engine's
      ``TurnDone.metrics`` dict.
    """
    if audit.get("metrics"):
        return audit
    if isinstance(ev_metrics, dict) and ev_metrics:
        folded = dict(audit)
        folded["metrics"] = dict(ev_metrics)
        return folded
    return audit
