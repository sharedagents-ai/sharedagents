"""Durable delegation budget reservations."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from sharedagents_agent.agents import budget_reservations as br


@pytest.fixture
def budget(monkeypatch):
    """Pin the session's remaining budget for deterministic admission."""
    remaining = {"usd": 10.0}
    monkeypatch.setattr(
        "sharedagents_goal.goal_service.remaining_for_session_hash",
        lambda sid, **kw: remaining["usd"],
    )
    return remaining


def test_parse_budget_usd_accepts_number_and_dollar_string():
    assert br.parse_budget_usd(6) == 6.0
    assert br.parse_budget_usd("$6.00") == 6.0
    assert br.parse_budget_usd(" 3.5 ") == 3.5
    assert br.parse_budget_usd("n/a") == 0.0
    assert br.parse_budget_usd(-4) == 0.0


def test_wave_reservation_is_all_or_none(tmp_path, budget):
    group = br.reserve_many("parent", [6, 6], keys=["a", "b"], sessions_dir=tmp_path)
    assert group is None
    assert br.active_reserved_usd("parent", sessions_dir=tmp_path) == 0.0

    first = br.reserve("parent", 6, key="a", sessions_dir=tmp_path)
    assert first is not None
    assert br.active_reserved_usd("parent", sessions_dir=tmp_path) == 6.0
    assert br.reserve("parent", 6, key="b", sessions_dir=tmp_path) is None

    first.release()
    assert br.active_reserved_usd("parent", sessions_dir=tmp_path) == 0.0
    second = br.reserve("parent", 6, key="b", sessions_dir=tmp_path)
    assert second is not None
    second.release()


def test_concurrent_reservations_cannot_over_admit(tmp_path, budget):
    def attempt(i):
        return br.reserve("parent", 3, key=f"r{i}", sessions_dir=tmp_path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(attempt, range(8)))
    admitted = [r for r in results if r is not None]
    assert len(admitted) == 3
    assert br.active_reserved_usd("parent", sessions_dir=tmp_path) == 9.0
    for reservation in admitted:
        reservation.release()
    assert br.active_reserved_usd("parent", sessions_dir=tmp_path) == 0.0


def test_stale_reservations_expire(tmp_path, budget):
    path = tmp_path / "parent.budget_reservations.json"
    path.write_text(
        json.dumps(
            {
                "old": {"usd": 99.0, "created_at": time.time() - br.DEFAULT_TTL_S - 1},
                "live": {"usd": 2.0, "created_at": time.time()},
            }
        ),
        encoding="utf-8",
    )
    assert br.active_reserved_usd("parent", sessions_dir=tmp_path) == 2.0
    # The stale claim does not block a new wave.
    reservation = br.reserve("parent", 8, key="new", sessions_dir=tmp_path)
    assert reservation is not None
    reservation.release()


def test_release_reconciles_against_actual_spend(tmp_path, budget):
    first = br.reserve("parent", 6, key="a", sessions_dir=tmp_path)
    assert first is not None
    # Actual spend lands in the ledger (simulated by lowering the remaining
    # figure); releasing the claim must expose exactly that, not the old
    # reservation.
    budget["usd"] = 4.0
    first.release()
    assert br.active_reserved_usd("parent", sessions_dir=tmp_path) == 0.0
    assert br.reserve("parent", 4, key="b", sessions_dir=tmp_path) is not None
    assert br.reserve("parent", 5, key="c", sessions_dir=tmp_path) is None


def test_refusal_message_names_the_shortfall(tmp_path, budget):
    br.reserve("parent", 7, key="held", sessions_dir=tmp_path)
    message = br.refusal_message("parent", 6, sessions_dir=tmp_path)
    assert "6.00" in message and "10.00" in message and "7.00" in message
    assert "budget add" in message


def test_can_admit_is_read_only_and_respects_live_claims(tmp_path, budget):
    assert br.can_admit("parent", 10, sessions_dir=tmp_path)
    held = br.reserve("parent", 4, key="held", sessions_dir=tmp_path)
    assert held is not None
    assert br.can_admit("parent", 6, sessions_dir=tmp_path)
    assert not br.can_admit("parent", 7, sessions_dir=tmp_path)
    held.release()
    assert br.can_admit("parent", 7, sessions_dir=tmp_path)
