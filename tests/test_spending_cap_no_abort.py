"""Tests verifying that spending cap does not abort a session, lets the turn end,
and informs the user afterwards.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from sharedagents_agent.agents.engine import TextDelta, TurnDone
from sharedagents_agent.agents.local_llm import LocalLLMEngine
from sharedagents_agent.agents.turn_execution import maybe_abort_external_on_tree_budget


def test_maybe_abort_external_on_tree_budget_does_not_abort():
    """Spending cap records budget exhaustion but never aborts the engine."""
    aborted_calls = []

    class FakeEngine:
        kind = SimpleNamespace(value="claude_cli")

        async def abort(self, turn_id: str, *, force: bool = False):
            aborted_calls.append((turn_id, force))

    engine = FakeEngine()
    sess = SimpleNamespace(hash="sess-1")
    extra = {}
    scheduled = []

    result = maybe_abort_external_on_tree_budget(
        engine=engine,
        session=sess,
        extra=extra,
        turn_id="turn-1",
        remaining_fn=lambda _s: 0.0,
        schedule=lambda abort_fn, tid: scheduled.append((tid, True)),
    )

    assert result is True
    assert extra.get("_budget_exhausted") is True
    assert extra.get("_budget_abort_scheduled") is None
    assert len(aborted_calls) == 0
    assert len(scheduled) == 0


_SPEND_USD = 30.0


class _BudgetSpendLocalLLMEngine(LocalLLMEngine):
    """Local-LLM engine that records spend over budget and completes turn."""

    __slots__ = ()

    async def run_turn(self, request, state, *, sessions_dir, turn_id, env=None):
        checkpoint = (request.extra or {}).get("_usage_checkpoint")
        if callable(checkpoint):
            checkpoint(
                {
                    "cost_usd_override": _SPEND_USD,
                    "model": "gemini-3.8-flash",
                    "provider": "google",
                }
            )
        yield TextDelta(text="Task completed successfully.", channel="content")
        yield TurnDone(state="ok")


def test_spending_cap_lets_turn_end_and_informs_user(tmp_path: Path, monkeypatch):
    """Spending cap allows the turn to finish cleanly and informs user afterwards."""
    from sharedagents_agent.agents import factory
    from sharedagents_goal.goal_service import set_session_budget
    from sharedagents_model import usage_ledger

    home = tmp_path / "sharedagents_home"
    agent_dir = home / "agents" / "test_agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "config.json").write_text(json.dumps({"name": "test_agent", "engine": "local_llm"}) + "\n")
    monkeypatch.setenv("SHAREDAGENTS_HOME", str(home))
    monkeypatch.setattr(usage_ledger, "_ledger", None)
    monkeypatch.setattr(usage_ledger, "_pricing_cache", {})

    agent = factory.load_agent("test_agent")
    monkeypatch.setattr(agent.__class__, "schedule_title_creation", lambda *args, **kwargs: None)
    agent.engine.__class__ = _BudgetSpendLocalLLMEngine

    session = agent.create_session()
    set_session_budget(agent, session, 25.0)

    async def drive():
        return [ev async for ev in agent.run_turn(session.hash, "execute review")]

    events = asyncio.run(drive())

    # Turn must NOT be aborted or errored
    dones = [ev for ev in events if isinstance(ev, TurnDone)]
    assert len(dones) == 1
    assert dones[0].state == "ok"

    # User must be informed afterwards
    texts = [ev.text for ev in events if isinstance(ev, TextDelta)]
    full_text = "".join(texts)
    assert "Task completed successfully." in full_text
    assert "[sharedagents] The session cost budget has reached its limit" in full_text
    assert "$30.00 of $25.00 spent" in full_text
    assert "Top up the session budget to continue" in full_text

    # Verify persisted turn content also contains the notice
    sess = agent.get_session(session.hash)
    turns = list(getattr(sess, "turns", None) or [])
    assert len(turns) >= 1
    assert "Task completed successfully." in turns[-1]["content"]
    assert "[sharedagents] The session cost budget has reached its limit" in turns[-1]["content"]
