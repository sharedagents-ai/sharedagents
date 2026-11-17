"""A side-channel event must never be taken for a tool's result.

Session ``20261003T094716540485-5ebb976bebe6``: two ``AgentSync``
batches to claude / codex / antigravity came back with EMPTY bodies and
``ok=False`` although every child had finished and answered. The pre-egress
sensitive-data flag is a dict with its own ``"type":
"sensitive_egress_flag"``; ``tool_runner._emit_tool_event`` spread it AFTER the
``"type": "tool_event"`` envelope, so the line left the runner typed
``sensitive_egress_flag``. Both stdout readers took the first record that was
neither ``delta`` nor ``tool_event`` as the result, so the flag — emitted before
the child turn — won and the real ``final`` record was dropped.

These tests pin both halves: the envelope type cannot be overridden by an
event, and a reader only accepts a ``final`` (or legacy untyped) record.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from sharedagents_agent import delegation_gate, tool_runner
from sharedagents_agent.tools.impl import subagents
from sharedagents_gateway import tool_execution

FINAL = {"type": "final", "ok": True, "content": "the child's answer", "truncated": False}
# What the runner put on the wire before the fix: the flag, typed as itself.
STRAY = dict(delegation_gate.sensitive_egress_flag_event(("pii",)))


class _Stream:
    def __init__(self, records):
        self._lines = [json.dumps(r).encode("utf-8") + b"\n" for r in records]

    async def readline(self):
        return self._lines.pop(0) if self._lines else b""

    async def read(self):
        return b""


class _Stdin:
    def write(self, data):
        pass

    async def drain(self):
        pass

    def close(self):
        pass


class _Proc:
    """A finished ``tool_runner`` child that wrote ``records`` to stdout."""

    def __init__(self, records):
        self.stdin = _Stdin()
        self.stdout = _Stream(records)
        self.stderr = _Stream([])
        self.returncode = 0
        self.pid = 4242

    async def wait(self):
        return 0


def _gateway_run(tmp_path, monkeypatch, records):
    """``_default_run_tool`` against a runner that wrote ``records``."""

    async def fake_exec(*_argv, **_kw):
        return _Proc(records)

    monkeypatch.setattr(tool_execution.asyncio, "create_subprocess_exec", fake_exec)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    backend = SimpleNamespace(
        cfg=SimpleNamespace(
            sessions_dir=tmp_path,
            home=home,
            name="rt",
            agent_profile="",
            permission_mode="auto",
            upstream_endpoint="",
            upstream_model="",
        ),
        _active_sessions={},
    )
    turn = SimpleNamespace(post_force_terminate=False)
    return asyncio.run(
        tool_execution._default_run_tool(
            backend,
            "s1",
            {"name": "todowrite", "arguments": {}},
            turn,
        )
    )


def test_an_events_own_type_cannot_replace_the_tool_event_envelope(capsys):
    tool_runner._emit_tool_event(delegation_gate.sensitive_egress_flag_event(("pii",)))
    record = json.loads(capsys.readouterr().out)
    assert record["type"] == "tool_event"
    # The event stays recognisable to the surfaces that render it.
    assert record["kind"] == "sensitive_egress_flag"
    assert record["classes"] == ["pii"]


def test_subagent_runner_result_is_the_final_record_not_an_earlier_typed_one():
    proc = _Proc([STRAY, {"type": "delta", "text": "working"}, FINAL])
    result = asyncio.run(subagents._spawn_subagent_runner_body(proc, {"tool": "AgentSync"}))
    assert result.ok is True
    assert result.content == "the child's answer"


def test_gateway_result_is_the_final_record_not_an_earlier_typed_one(tmp_path, monkeypatch):
    ok, content, aborted, truncated = _gateway_run(tmp_path, monkeypatch, [STRAY, FINAL])
    assert (ok, content, aborted, truncated) == (True, "the child's answer", False, False)


def test_a_legacy_untyped_record_is_still_the_result(tmp_path, monkeypatch):
    legacy = {"ok": True, "content": "legacy shape", "truncated": False}
    assert _gateway_run(tmp_path, monkeypatch, [legacy])[:2] == (True, "legacy shape")
    result = asyncio.run(
        subagents._spawn_subagent_runner_body(_Proc([legacy]), {"tool": "read"}),
    )
    assert (result.ok, result.content) == (True, "legacy shape")


def test_a_runner_that_only_wrote_a_typed_event_is_reported_as_a_crash():
    # No final record at all: the reader must say so, not return an empty result.
    result = asyncio.run(
        subagents._spawn_subagent_runner_body(_Proc([STRAY]), {"tool": "AgentSync"}),
    )
    assert result.ok is False
    assert "subagent runner crashed" in result.content


def test_refusal_metadata_survives_both_subprocess_readers(tmp_path, monkeypatch):
    record = {
        **FINAL,
        "ok": False,
        "content": "quota reason",
        "outcome": "refused",
        "member_outcomes": ["refused", "failed"],
    }
    result = asyncio.run(subagents._spawn_subagent_runner_body(_Proc([record]), {"tool": "AgentAsync"}))
    assert result.outcome == "refused"
    assert result.member_outcomes == ("refused", "failed")
    wire = _gateway_run(tmp_path, monkeypatch, [record])
    assert wire[0] is False and wire[1] == "quota reason"
    assert wire[5:] == ("refused", ("refused", "failed"))
