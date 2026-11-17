"""Retry burst: spend and side effects.

On 2026-09-23 an antigravity session kept auto-retrying after its $25 session
cap was spent: budget admission ran once, before the retry loop, so four
retries ran with a frozen "$25.49 spent" notice. Every retry also re-sent the
operator's original text into the resumed conversation, so "merge to main
restart and clear this worktree" was re-issued twice after the first attempt
had already merged, pulled and removed the worktree.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from sharedagents_agent.agents.engine import TextDelta, ToolCallStarted, TurnDone, TurnError
from sharedagents_agent.agents.local_llm import LocalLLMEngine

REQUEST = "merge to main restart and clear this worktree"

_SCRIPT: dict = {
    "runs": [],
    "prompts": [],
    "system_prompts": [],
    "attempts": 0,
    "on_attempt": None,
}


class _ScriptedLocalLLMEngine(LocalLLMEngine):
    """Real local-LLM engine whose per-attempt ``run_turn`` is scripted."""

    __slots__ = ()

    async def run_turn(self, request, state, *, sessions_dir, turn_id, env=None):
        idx = _SCRIPT["attempts"]
        _SCRIPT["attempts"] += 1
        _SCRIPT["prompts"].append(request.prompt)
        _SCRIPT["system_prompts"].append(request.system_prompt or "")
        if callable(_SCRIPT["on_attempt"]):
            _SCRIPT["on_attempt"](idx)
        runs = _SCRIPT["runs"]
        for ev in runs[idx] if idx < len(runs) else runs[-1]:
            yield ev


def _failed(*progress, classification: str = "cli_error"):
    return [
        *progress,
        TurnError(message="upstream 503: service unavailable", classification=classification),
        TurnDone(state="error"),
    ]


def _ok():
    return [TextDelta(text="done", channel="content"), TurnDone(state="ok")]


@pytest.fixture
def agent(tmp_path: Path, monkeypatch):
    from sharedagents_agent.agents import factory
    from sharedagents_model import usage_ledger

    home = tmp_path / "sharedagents_home"
    agent_dir = home / "agents" / "rt"
    agent_dir.mkdir(parents=True)
    (agent_dir / "config.json").write_text(json.dumps({"name": "rt", "engine": "local_llm"}) + "\n")
    monkeypatch.setenv("SHAREDAGENTS_HOME", str(home))
    monkeypatch.setenv("SHAREDAGENTS_TURN_RETRY_INTERVAL_S", "0")
    monkeypatch.setenv("SHAREDAGENTS_TURN_RETRY_MAX", "5")
    monkeypatch.delenv("SHAREDAGENTS_TURN_RETRY_CLASSES", raising=False)
    monkeypatch.delenv("SHAREDAGENTS_TURN_RETRY_BLOCK_CLASSES", raising=False)
    monkeypatch.setattr(usage_ledger, "_ledger", None)
    monkeypatch.setattr(usage_ledger, "_pricing_cache", {})
    _SCRIPT.update(runs=[], prompts=[], system_prompts=[], attempts=0, on_attempt=None)

    loaded = factory.load_agent("rt")
    loaded.engine.__class__ = _ScriptedLocalLLMEngine
    return loaded


def _spend(session_hash: str, usd: float) -> None:
    """Record real ledger spend against the session tree (as a turn would)."""
    from sharedagents_model.usage_ledger import record_turn_cost

    record_turn_cost(
        session_id=session_hash,
        turn_id=f"spend-{time.time_ns()}",
        metrics={"cost_usd_override": usd, "model": "gemini-3.8-flash", "provider": "google"},
        root_session_id=session_hash,
    )


def _drive(agent, sid: str, prompt: str = REQUEST) -> list:
    async def main() -> list:
        return [ev async for ev in agent.run_turn(sid, prompt)]

    return asyncio.run(main())


def _texts(events) -> list[str]:
    return [ev.text for ev in events if isinstance(ev, TextDelta)]


def test_no_retry_once_the_failed_attempt_used_up_the_session_budget(agent):
    # Break: deciding to retry without re-reading the session tree's spend.
    session = agent.create_session()
    _SCRIPT.update(
        runs=[_failed(), _ok()],
        on_attempt=lambda idx: _spend(session.hash, 30.0) if idx == 0 else None,
    )

    events = _drive(agent, session.hash)

    assert _SCRIPT["attempts"] == 1
    dones = [ev for ev in events if isinstance(ev, TurnDone)]
    assert [d.state for d in dones] == ["error"]
    assert not any("retrying in" in t for t in _texts(events))
    assert any("budget" in t.lower() for t in _texts(events)), _texts(events)


def test_turn_admitted_over_budget_is_not_retried(agent):
    # Break: auto-retrying a turn that was only admitted as a bounded manual
    # answer because the cap was already spent.
    session = agent.create_session()
    _spend(session.hash, 30.0)
    _SCRIPT.update(runs=[_failed(), _ok()])

    events = _drive(agent, session.hash)

    assert _SCRIPT["attempts"] == 1
    assert [ev.state for ev in events if isinstance(ev, TurnDone)] == ["error"]


def test_retry_with_budget_left_still_happens(agent):
    # Break: over-correcting and dropping transient-error retries entirely.
    session = agent.create_session()
    _SCRIPT.update(runs=[_failed(), _ok()])

    events = _drive(agent, session.hash)

    assert _SCRIPT["attempts"] == 2
    assert [ev.state for ev in events if isinstance(ev, TurnDone)] == ["ok"]


def test_retry_rechecks_budget_after_the_delay(agent):
    # Break: reusing the memoized end-of-attempt budget verdict after another
    # session in the tree has spent the remaining budget during the delay.
    session = agent.create_session()
    _SCRIPT.update(runs=[_failed(), _ok()])

    async def main():
        events = []
        async for ev in agent.run_turn(session.hash, REQUEST):
            events.append(ev)
            if isinstance(ev, TextDelta) and "retrying in" in ev.text:
                _spend(session.hash, 30.0)
        return events

    events = asyncio.run(main())

    assert _SCRIPT["attempts"] == 1
    assert [ev.state for ev in events if isinstance(ev, TurnDone)] == ["error"]
    assert any("not retrying" in text for text in _texts(events))
    assert not agent.has_active_turn(session.hash)


def test_budget_stop_during_delay_finalizes_goal_bookkeeping(agent, monkeypatch):
    # Break: skipping goal bookkeeping for a supposedly intermediate attempt
    # when spending during the retry delay makes that attempt the final one.
    from sharedagents_agent.agents.agent import Agent

    session = agent.create_session()
    calls = []
    monkeypatch.setattr(
        Agent,
        "_maybe_goal_re_drive",
        lambda self, sid, sess, done, content, **kw: calls.append((done, content, kw)),
    )
    _SCRIPT.update(
        runs=[
            _failed(
                ToolCallStarted(call_id="t1", name="run_command", arguments={}),
                TextDelta(text="Partial", channel="content"),
                TextDelta(text=" ", channel="content"),
                TextDelta(text="answer", channel="content"),
            ),
            _ok(),
        ]
    )

    async def main():
        events = []
        async for ev in agent.run_turn(session.hash, REQUEST):
            events.append(ev)
            if isinstance(ev, TextDelta) and "retrying in" in ev.text:
                _spend(session.hash, 30.0)
        return events

    events = asyncio.run(main())

    assert _SCRIPT["attempts"] == 1
    assert [ev.state for ev in events if isinstance(ev, TurnDone)] == ["error"]
    assert len(calls) == 1
    done, content, kwargs = calls[0]
    assert done.state == "error"
    assert content == "Partial answer"
    assert kwargs["tool_call_count"] == 1
    assert not agent.has_active_turn(session.hash)


@pytest.mark.parametrize(
    "progress",
    [
        ToolCallStarted(call_id="t1", name="run_command", arguments={"CommandLine": "git pull"}),
        TextDelta(text="Merged PR #1519, pulling main now.", channel="content"),
    ],
    ids=["tool-call", "answer-text"],
)
def test_retry_after_the_attempt_acted_continues_instead_of_replaying(agent, progress):
    # Break: re-sending the operator's instruction after the failed attempt
    # already ran tools or answered — the model then repeats the actions.
    session = agent.create_session()
    _SCRIPT.update(runs=[_failed(progress), _ok()])

    _drive(agent, session.hash)

    assert _SCRIPT["attempts"] == 2
    assert _SCRIPT["prompts"][0] == REQUEST
    retry_prompt = _SCRIPT["prompts"][1]
    assert REQUEST not in retry_prompt
    assert "continue" in retry_prompt.lower()


def test_retry_before_anything_happened_replays_the_request(agent):
    # Break: replacing the request with a bare "continue" when the failed
    # attempt produced nothing, leaving the retry without the task.
    session = agent.create_session()
    _SCRIPT.update(runs=[_failed(TextDelta(text="thinking...", channel="thinking")), _ok()])

    _drive(agent, session.hash)

    assert _SCRIPT["prompts"] == [REQUEST, REQUEST]


# -- the model learns when a spent budget has been raised ------------------------
#
# On 2026-10-07 a turn hit the $25 session cap. The operator raised the cap to
# $100 and wrote "go ahead implement it"; the agent answered that the budget
# was exhausted and asked for $10 more, because the only budget fact in its
# context was the limit notice recorded at the end of the previous turn.


def _raise_budget(agent, session, usd: float) -> None:
    from sharedagents_goal.goal_service import set_session_budget

    assert set_session_budget(agent, session, usd).ok


def test_next_turn_is_told_when_a_spent_budget_was_raised(agent):
    session = agent.create_session()
    _SCRIPT.update(
        runs=[_ok(), _ok()],
        on_attempt=lambda idx: _spend(session.hash, 27.19) if idx == 0 else None,
    )

    first = _drive(agent, session.hash, "create the plan")
    assert any("reached its limit" in t for t in _texts(first)), _texts(first)
    assert "session budget" not in _SCRIPT["system_prompts"][0]

    _raise_budget(agent, session, 100.0)
    _drive(agent, session.hash, "go ahead implement it")

    notice = _SCRIPT["system_prompts"][1]
    assert "[sharedagents session budget notice]" in notice
    assert "raised" in notice
    assert "$100.00" in notice and "$27.19" in notice
    assert "Do not call tools" not in notice  # not the over-budget restriction


def test_budget_raise_is_announced_once(agent):
    session = agent.create_session()
    _SCRIPT.update(
        runs=[_ok(), _ok(), _ok()],
        on_attempt=lambda idx: _spend(session.hash, 27.19) if idx == 0 else None,
    )
    _drive(agent, session.hash, "create the plan")
    _raise_budget(agent, session, 100.0)
    _drive(agent, session.hash, "go ahead implement it")
    _drive(agent, session.hash, "continue")

    assert "[sharedagents session budget notice]" in _SCRIPT["system_prompts"][1]
    assert "[sharedagents session budget notice]" not in _SCRIPT["system_prompts"][2]


def test_still_spent_budget_keeps_the_restriction(agent):
    # Break: announcing a raise that did not happen.
    session = agent.create_session()
    _SCRIPT.update(
        runs=[_ok(), _ok()],
        on_attempt=lambda idx: _spend(session.hash, 27.19) if idx == 0 else None,
    )
    _drive(agent, session.hash, "create the plan")
    _drive(agent, session.hash, "go ahead implement it")

    notice = _SCRIPT["system_prompts"][1]
    assert "The session cost limit has been reached" in notice
    assert "raised" not in notice


def test_no_budget_notice_on_an_ordinary_turn(agent):
    session = agent.create_session()
    _SCRIPT.update(runs=[_ok(), _ok()])
    _drive(agent, session.hash, "hello")
    _drive(agent, session.hash, "again")

    assert all("session budget notice" not in sp for sp in _SCRIPT["system_prompts"])


@pytest.mark.parametrize(
    "message",
    [
        "Individual quota reached",
        "quota exceeded",
        "insufficient_quota",
        '{"code": -32602, "message": "Invalid params"}',
        "Invalid parameters: cwd",
        "invalid_parameters supplied to tool",
        "cwd must be an absolute path",
        "Schema validation failed: task.cwd is invalid",
        "schema_validation_error: cwd",
    ],
)
def test_permanent_engine_message_fails_once_and_wakes_failed_wave(agent, monkeypatch, message):
    """Real Agent retry loop inside the real async job finalizer; no model IO."""
    from types import SimpleNamespace

    from sharedagents_agent.agents import async_delegate_runner as runner, async_jobs

    session = agent.create_session()
    _SCRIPT.update(
        runs=[
            [TurnError(message=message, classification="cli_error"), TurnDone(state="error")],
            _ok(),
        ]
    )
    posts = []

    async def post(parent, text, gateway_url=None):
        posts.append(text)

    async def drive(name, args, child_session_cb=None):
        child_session_cb(session.hash)
        args["_sharedagents_child_session_id_out"] = session.hash
        events = [ev async for ev in agent.run_turn(session.hash, "same delegated goal")]
        assert [ev.state for ev in events if isinstance(ev, TurnDone)] == ["error"]
        return SimpleNamespace(ok=False, content=next(ev.message for ev in events if isinstance(ev, TurnError)))

    monkeypatch.setattr(runner, "_drive_child_agent_turn", drive)
    monkeypatch.setattr(runner, "_submit_parent_prompt", post)
    wid = async_jobs.create_wave("retry-parent")
    rec = async_jobs.create(
        parent_sid="retry-parent",
        agent="rt",
        task="same delegated goal",
        on_complete="notify",
        call_id="permanent",
        pid=0,
        wave_id=wid,
    )
    async_jobs.seal_wave("retry-parent", wid)
    asyncio.run(runner.run_job({"job_id": rec["job_id"], "parent_sid": "retry-parent", "agent": "rt"}))
    assert _SCRIPT["attempts"] == 1
    assert async_jobs.read(rec["job_id"])["status"] == "failed"
    assert len(posts) == 1
    start = posts[0].index('[{"job_id"')
    rows, _ = json.JSONDecoder().raw_decode(posts[0][start:])
    assert rows[0]["status"] == "failed" and rows[0]["content"] == message


@pytest.mark.parametrize(
    "message", ["timeout", "upstream HTTP 503", "connection reset by peer", "HTTP 429 too many requests"]
)
def test_transient_engine_messages_still_retry(agent, message):
    session = agent.create_session()
    _SCRIPT.update(
        runs=[
            [TurnError(message=message, classification="cli_error"), TurnDone(state="error")],
            _ok(),
        ]
    )
    events = _drive(agent, session.hash)
    assert _SCRIPT["attempts"] == 2
    assert [ev.state for ev in events if isinstance(ev, TurnDone)] == ["ok"]
