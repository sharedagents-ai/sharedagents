"""A past turn's reasoning is replayed once, after its tool exchange.

The turn records one reasoning text — the final round's. Putting it on the
tool-call message as well sent post-call reasoning before the call and paid
for it twice.
"""

from __future__ import annotations

from types import SimpleNamespace

from sharedagents_agent.agents.local_llm import injection

FINAL_THOUGHT = "The command printed hi; reply with it."


def _turn(content: str) -> dict:
    return {
        "user": "run echo hi",
        "content": content,
        "thinking": FINAL_THOUGHT,
        "started_at": "2026-11-01T11:00:00+00:00",
        "tool_calls": [{"id": "c1", "name": "bash", "arguments": {"command": "echo hi"}}],
        "tool_results": [{"id": "c1", "content": "hi"}],
    }


def _build(turn: dict) -> list[dict]:
    session = SimpleNamespace(
        turns=[turn],
        compaction_summary="",
        compaction_count=0,
        compaction_turn=0,
        pinned_facts=[],
        injected_messages=[],
        hash="t",
    )
    mixin = injection.InjectionMixin()
    mixin._injected_messages = {}
    return mixin._build_messages_simple(session, "next", system_prompt="sys", preamble=[])


def _assistants(msgs):
    return [m for m in msgs if m.get("role") == "assistant"]


def test_reasoning_rides_the_final_reply_only():
    calls, final = _assistants(_build(_turn("hi")))
    assert calls["tool_calls"] and calls["reasoning_content"] == ""
    assert final["content"] == "hi" and final["reasoning_content"] == FINAL_THOUGHT
    texts = [str(m.get("reasoning_content") or "") for m in _build(_turn("hi"))]
    assert sum(t == FINAL_THOUGHT for t in texts) == 1


def test_a_turn_without_a_reply_keeps_its_reasoning_on_the_call():
    (calls,) = _assistants(_build(_turn("")))
    assert calls["tool_calls"] and calls["reasoning_content"] == FINAL_THOUGHT
