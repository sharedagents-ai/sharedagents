"""The pre-egress flag is shown to the owner, not just logged.

The shared delegate chokepoint fires the flag through ``on_event`` before the
child turn. The local-LLM engine's event handlers only understood ``tool_use``
/ ``tool_result``, so on the ``AgentSync`` path the flag was archived in the
session log and never rendered. These tests drive the two real dispatchers
with a runner / shim that fires the flag the way it arrives off the wire and
assert the notice is emitted as assistant-message text, ahead of the result.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from sharedagents_agent import delegation_gate
from sharedagents_agent.agents import delegate_shims
from sharedagents_agent.agents.engine import TextDelta, ToolCallResult
from sharedagents_agent.agents.local_llm.tool_dispatch import ToolDispatchMixin
from sharedagents_agent.tools.runners import ToolResult

SID = "20261004T000000000000-bbbbbbbbbbbb"
TO_CODEX = "Personal data (PII) in this request is being sent to codex, a third-party agent."
TO_ANTIGRAVITY = "Personal data (PII) in this request is being sent to antigravity, a third-party agent."
TO_UNNAMED = "Personal data (PII) in this request is being sent to a third-party agent."
# Internal bookkeeping that must never reach the reader: a fan-out index, a
# log-style tag, a requirement id.
INTERNAL = ("[run", "[sensitive", "REQ-")


def _flag(agent="", **extra):
    """The flag as an ``on_event`` consumer receives it: no envelope ``type``."""
    ev = dict(delegation_gate.sensitive_egress_flag_event(("pii",), agent))
    ev.pop("type")
    ev.update(extra)
    return ev


class _Engine(ToolDispatchMixin):
    def __init__(self, runner=None):
        self._tool_runner = runner
        self.cfg = SimpleNamespace(auto_complete_todos=False)
        self._sessions = {}
        self._agent = self
        self._cancelled = {}

    def _log_event(self, event, **fields):
        pass


def _collect(events):
    async def run():
        return [ev async for ev in events]

    return asyncio.run(run())


def _notices(events):
    # The assistant-message channel: shown at every /tools level. Tool-stream
    # text (``tool_delta``) is hidden unless tools are set to ``stream``, which
    # is how the notice was invisible in the web UI at the default setting.
    return [
        ev.text
        for ev in events
        if isinstance(ev, TextDelta) and ev.channel == "content" and "is being sent to" in ev.text
    ]


def test_the_notice_is_a_plain_sentence_naming_the_receiving_agent():
    assert delegation_gate.sensitive_egress_notice_text(("pii",), "codex") == TO_CODEX
    assert delegation_gate.sensitive_egress_notice_text(("pii",)) == TO_UNNAMED
    for text in (TO_CODEX, TO_UNNAMED):
        assert not any(marker in text for marker in INTERNAL)


@pytest.mark.parametrize(
    "flag, expected",
    [
        (_flag("codex"), TO_CODEX),
        # An entry of a fan-out is told apart by WHO receives the data, never by
        # its position in the batch.
        (_flag("antigravity", batch_idx=2), TO_ANTIGRAVITY),
        (_flag(), TO_UNNAMED),
    ],
    ids=["single", "batch-entry", "unnamed"],
)
def test_a_tool_calls_egress_flag_is_shown_in_the_transcript(flag, expected):
    async def runner(session_id, call, turn, *, on_delta=None, on_event=None):
        await on_event(flag)
        return True, "the child's answer", False, False

    engine = _Engine(runner)
    session = SimpleNamespace(hash=SID)
    messages, record = [], {}
    events = _collect(
        engine._execute_call_singleton(
            {"id": "call-1", "name": "AgentSync", "arguments": {"subagent_type": "claude"}},
            session,
            0,
            None,
            messages,
            record,
        )
    )

    assert _notices(events) == ["\n" + expected + "\n"]
    kinds = [type(ev).__name__ for ev in events]
    # Shown before the call's result, and never folded into the result body.
    assert kinds.index("TextDelta") < kinds.index("ToolCallResult")
    result = next(ev for ev in events if isinstance(ev, ToolCallResult))
    assert "is being sent to" not in result.content


def test_a_slash_command_delegates_egress_flag_is_shown_in_the_transcript(monkeypatch):
    async def shim(args, *, on_delta=None, on_event=None):
        await on_event(_flag("claude"))
        return ToolResult(content="the child's answer", ok=True)

    monkeypatch.setattr(delegate_shims, "delegate_shim_for_tool", lambda name: shim)
    engine = _Engine()
    events = _collect(
        engine._run_direct_tool_call(
            SimpleNamespace(hash=SID),
            "turn0001xyz",
            {"name": "delegate_claude", "arguments": {"prompt": "hello"}},
        )
    )

    assert _notices(events) == ["\nPersonal data (PII) in this request is being sent to claude, a third-party agent.\n"]
    result = next(ev for ev in events if isinstance(ev, ToolCallResult))
    assert result.ok and "is being sent to" not in result.content


def test_the_delegate_chokepoint_names_the_agent_it_is_about_to_call(tmp_path, monkeypatch):
    from sharedagents_agent.agents import factory
    from sharedagents_agent.agents.engine import EngineKind, TurnDone

    class _CodexEngine:
        kind = EngineKind.CODEX_CLI

        def is_available(self):
            return True, ""

    class _Child:
        sessions_dir = tmp_path
        engine = _CodexEngine()
        cfg = SimpleNamespace(name="codex", engine="codex_cli", upstream_model="gpt")

        async def run_turn(self, sid, prompt, access=None, **_kw):
            yield TextDelta(text="ok", channel="content")
            yield TurnDone(state="ok")

    monkeypatch.setattr(factory, "load_agent", lambda name: _Child())
    monkeypatch.setattr(
        delegate_shims,
        "_resolve_or_create_child_session",
        lambda child, args, parent_sid: ("child", ""),
    )
    monkeypatch.delenv("SHAREDAGENTS_SESSION_ID", raising=False)
    seen: list[dict] = []
    result = asyncio.run(
        delegate_shims._drive_child_agent_turn_inner(
            "codex",
            {
                "prompt": "Reply to jane.doe@example.com",
                "goal": "Answer the ticket",
                "constraints": "Read-only",
                "acceptance_criteria": "A reply is drafted",
                "budget_usd": 1,
            },
            on_event=seen.append,
        )
    )

    assert result.ok, result.content
    flags = [ev for ev in seen if ev.get("kind") == "sensitive_egress_flag"]
    assert [ev.get("agent") for ev in flags] == ["codex"]
