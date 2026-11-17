"""Replay the 2026-11-04 jarvis runs against the orchestration policy.

The fixtures are sanitized shapes (roles, costs, outcomes), not transcripts.
They pin four properties: authorized phases, bounded
cost, corrected assumptions, and task-level outcomes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sharedagents_agent.agents import budget_reservations as br
from sharedagents_agent.agents.local_llm.coordination import workflow_disposition

FIXTURE = Path(__file__).parent / "fixtures" / "orchestration_replay_2026-11-04.json"


@pytest.fixture(scope="module")
def replay():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_session_a_answered_before_building_and_blew_the_review_budget(replay):
    run = replay["session_a"]
    first = run["phases"][0]
    # Authorized phases: a confirmation is answered by the lookup; the first
    # child produced the answer before any planner/builder was spawned.
    assert first["role"] == "lookup"
    assert first["cost_usd"] == run["expected"]["first_answer_cost_usd"]
    assert first["cost_usd"] < run["total_cost_usd"] * 0.1
    # Bounded cost: the run consumed its whole cap and still delivered nothing.
    assert run["total_cost_usd"] >= run["budget_cap_usd"]
    assert run["expected"]["delivered"] is False
    # Review budget: three review rounds is one over the policy cap, so the
    # replay flags it instead of treating the loop as normal.
    assert run["expected"]["review_rounds"] > replay["review_budget"]["max_rounds"]


def test_session_b_corrected_the_premise_and_delivered(replay):
    run = replay["session_b"]
    diagnosis = run["phases"][0]
    assert diagnosis["role"] == "diagnose"
    assert diagnosis["cost_usd"] == run["expected"]["diagnosis_cost_usd"]
    # The diagnosis is what allowed the anchor to be corrected; without the
    # evidence-based revision rule the frozen todo would have kept building
    # the wrong premise.
    assert run["expected"]["anchor_corrected"] is True
    assert run["expected"]["delivered"] is True
    # The delivered run stayed under its cap.
    assert run["total_cost_usd"] < run["budget_cap_usd"]


def test_replay_workflow_disposition_separates_work_from_owner_waits():
    # A corrected plan is work; an approval gate is not.
    assert workflow_disposition(["fix the reconciliation defect"]) == "working"
    assert workflow_disposition(["deploy the fix [blocked: approval]"]) == "waiting_approval"
    assert workflow_disposition(["wait for the deploy [blocked: external]"]) == "waiting_external"


def test_replay_wave_budget_is_admitted_as_one_unit(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sharedagents_goal.goal_service.remaining_for_session_hash",
        lambda sid, **kw: 10.0,
    )
    # The A/B failure shape: two frontier children at $6 each against $10.
    assert br.reserve_many("replay", [6, 6], keys=["a", "b"], sessions_dir=tmp_path) is None
    # A single $6 child fits, and a second concurrent $6 child does not.
    first = br.reserve("replay", 6, key="a", sessions_dir=tmp_path)
    assert first is not None
    assert br.reserve("replay", 6, key="b", sessions_dir=tmp_path) is None
    first.release()
