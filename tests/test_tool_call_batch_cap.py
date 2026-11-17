"""A runaway tool-call batch is stopped early instead of burning the output cap.

Observed, session 20260930T103640938302-ed7b75e05ad3 (``plan`` on
deepseek-flash): one response emitted 144 tool calls -- 100+ ``bash``, web
fetches and searches, 11 ``todowrite``, a 14k-char ``write``, ``question`` and
``ExitPlanMode`` three times -- the whole remaining investigation scripted
without waiting for a single result. It ran into the 32,000-token cap,
nothing ran, and the turn needed a nudge. In the week before, no native
deepseek-flash or qwen3.8 response used more than 11 calls (4,836 and 1,874
rounds).

The worker now stops reading a response once it starts call N+1
(``SHAREDAGENTS_MAX_TOOL_CALLS_PER_RESPONSE``, default 32; 0 disables). Closing the
stream stops the upstream; the complete calls among the first N run as an
ordinary tool round and the model continues from real results -- no nudge, no
burned budget. DeepSeek and vLLM stream calls one after another, so the first
N are normally complete; a call is kept only with a name and arguments that
parse to an object, so an unfinished one is never repaired and run.
"""

from __future__ import annotations

import io
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from sharedagents_agent.agents.engine import (
    EngineKind,
    EngineSessionState,
    TextDelta,
    ToolCallResult,
    TurnDone,
    TurnError,
    TurnRequest,
)
from sharedagents_agent.agents.local_llm import LocalLLMEngine
from sharedagents_gateway import worker
from sharedagents_session.session import Session

_REAL_CLIENT = httpx.Client


def _chunk(delta: dict, finish=None) -> dict:
    return {"choices": [{"index": 0, "finish_reason": finish, "delta": delta}]}


def _batch(n: int, *, finish: str = "tool_calls", last_args: str | None = None) -> list[dict]:
    """``n`` calls streamed the way DeepSeek/vLLM do: one after another."""
    chunks = [_chunk({"role": "assistant", "content": "Checking everything at once."})]
    for i in range(n):
        chunks.append(
            _chunk(
                {
                    "tool_calls": [
                        {
                            "index": i,
                            "id": f"call_{i}",
                            "type": "function",
                            "function": {"name": "bash", "arguments": ""},
                        }
                    ]
                }
            )
        )
        args = json.dumps({"command": f"echo {i}"})
        if last_args is not None and i == n - 1:
            args = last_args
        chunks.append(
            _chunk(
                {
                    "tool_calls": [
                        {
                            "index": i,
                            "function": {"arguments": args},
                        }
                    ]
                }
            )
        )
    chunks.append(
        {
            "choices": [{"index": 0, "finish_reason": finish, "delta": {}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 32000, "total_tokens": 32100},
        }
    )
    return chunks


def _run_worker(monkeypatch, capsys, tmp_path, chunks) -> tuple[list[dict], list[int]]:
    """Drive the real ``worker.main()``; return its events and how many SSE
    events the upstream got to send before the worker stopped reading."""
    served: list[int] = []

    def body():
        for index, c in enumerate(chunks):
            served.append(index)
            yield f"data: {json.dumps(c)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    def client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=body(),
            )
        )
        return _REAL_CLIENT(*args, **kwargs)

    monkeypatch.setenv("SHAREDAGENTS_HOME", str(tmp_path))
    monkeypatch.setattr(worker, "_install_term_handler", lambda: None)
    monkeypatch.setattr(worker.httpx, "Client", client)
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "endpoint": "https://api.deepseek.com/v1",
                    "model": "deepseek-flash",
                    "backend": "openai",
                    "messages": [{"role": "user", "content": "investigate"}],
                }
            )
        ),
    )
    assert worker.main() == 0
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    return events, served


def _calls_and_metrics(events):
    calls = [c for e in events if e["kind"] == "tool_calls" for c in e["calls"]]
    metrics = next(e for e in events if e["kind"] == "metrics")
    return calls, metrics


@pytest.fixture(autouse=True)
def _default_cap(monkeypatch):
    monkeypatch.delenv("SHAREDAGENTS_MAX_TOOL_CALLS_PER_RESPONSE", raising=False)


@pytest.mark.parametrize("finish", ["tool_calls", "length"])
def test_a_runaway_batch_runs_its_first_calls_and_stops(monkeypatch, capsys, tmp_path, finish):
    # The live shape ended at the 32k cap ("length"); a shorter runaway could
    # end "tool_calls". Either way only the first 32 complete calls run.
    chunks = _batch(144, finish=finish)
    events, served = _run_worker(monkeypatch, capsys, tmp_path, chunks)
    calls, metrics = _calls_and_metrics(events)
    assert [c["id"] for c in calls] == [f"call_{i}" for i in range(32)]
    assert all(c["arguments"] == {"command": f"echo {i}"} for i, c in enumerate(calls))
    assert metrics["finish_reason"] == "tool_calls"  # an ordinary tool round
    assert metrics["tool_calls_capped"] == 32
    assert metrics["truncated_tool_calls"] == []
    # No usage chunk was read, so tokens are estimated -- counting the calls'
    # arguments, which are nearly all of this response.
    args_chars = sum(len(json.dumps({"command": f"echo {i}"})) for i in range(32))
    assert metrics["tokens_estimated"] is True
    assert metrics["completion_tokens"] >= args_chars // 4
    # The worker stopped reading at call 33's first delta: the other ~220
    # events (and the tokens behind them) were never pulled.
    assert max(served) < 1 + 2 * 32 + 3


def test_a_normal_batch_is_untouched(monkeypatch, capsys, tmp_path):
    events, served = _run_worker(monkeypatch, capsys, tmp_path, _batch(11))
    calls, metrics = _calls_and_metrics(events)
    assert len(calls) == 11
    assert metrics["tool_calls_capped"] == 0
    assert max(served) == len(_batch(11)) - 1


@pytest.mark.parametrize("value,expected", [("4", 4), ("0", 40)])
def test_the_cap_is_set_by_the_environment(monkeypatch, capsys, tmp_path, value, expected):
    monkeypatch.setenv("SHAREDAGENTS_MAX_TOOL_CALLS_PER_RESPONSE", value)  # 0 disables
    events, _ = _run_worker(monkeypatch, capsys, tmp_path, _batch(40))
    calls, metrics = _calls_and_metrics(events)
    assert len(calls) == expected


def test_an_incomplete_call_at_the_cap_is_never_run(monkeypatch, capsys, tmp_path):
    # A provider that interleaves calls could leave one unfinished when the
    # cap trips; a half-written call must not be "repaired" and run.
    monkeypatch.setenv("SHAREDAGENTS_MAX_TOOL_CALLS_PER_RESPONSE", "3")
    # Calls 0-1 complete, call 2 unfinished, then call 3 starts (no finish
    # chunk in between).
    chunks = _batch(3, last_args='{"command": "rm -rf /tmp/pla')[:-1] + _batch(2)[1:]
    # Renumber the tail so call index 3 starts after the unfinished call 2.
    for c in chunks[7:]:
        for tc in c["choices"][0]["delta"].get("tool_calls", []):
            tc["index"] += 3
            if "id" in tc:
                tc["id"] = f"call_{tc['index']}"
    events, _ = _run_worker(monkeypatch, capsys, tmp_path, chunks)
    calls, metrics = _calls_and_metrics(events)
    assert [c["id"] for c in calls] == ["call_0", "call_1"]
    assert metrics["truncated_tool_calls"] == [
        {"name": "bash", "args_chars": len('{"command": "rm -rf /tmp/pla')},
    ]


# --- the engine treats a capped batch as an ordinary tool round -------------


@pytest.mark.asyncio
async def test_a_capped_batch_runs_and_the_turn_continues_without_a_nudge(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setenv("SHAREDAGENTS_MAX_TOOL_CALLS_PER_RESPONSE", "5")
    first, _ = _run_worker(monkeypatch, capsys, tmp_path, _batch(144, finish="length"))
    monkeypatch.setenv("SHAREDAGENTS_SESSIONS_DIR", str(tmp_path / "sessions"))
    cfg = SimpleNamespace(
        name="plan",
        upstream_model="deepseek-flash",
        upstream_endpoint="https://api.deepseek.com/v1",
        upstream_max_context=1_000_000,
        upstream_max_completion_tokens=None,
        compact_at_frac=0.85,
        compact_min_turns=3,
        models=[],
        sessions_dir=tmp_path / "sessions",
    )
    engine = LocalLLMEngine(cfg)
    session = Session(
        hash="tool-call-batch-cap",
        title="t",
        created_at="2026-09-30T11:20:00+00:00",
        system_prompt="t",
        model=cfg.upstream_model,
        endpoint=cfg.upstream_endpoint,
        context_window=1_000_000,
        system_mode="auto",
        agent_name="plan",
    )
    monkeypatch.setattr(session, "save", Mock())
    engine._sessions[session.hash] = session
    monkeypatch.setattr(engine, "_log_event", Mock())
    monkeypatch.setattr(engine, "_log_round", Mock())
    monkeypatch.setattr(engine, "_teacher_enabled", lambda *args: False)

    rounds = [
        first,
        [
            {"kind": "content", "text": "Five results in; continuing."},
            {
                "kind": "metrics",
                "finish_reason": "stop",
                "terminal_finish_reason_seen": True,
                "prompt_tokens": 200,
                "completion_tokens": 8,
            },
            {"kind": "done"},
        ],
    ]
    spawned: list[list[dict]] = []

    async def spawn(engine, session, messages, turn_id, env):
        spawned.append([dict(m) for m in messages])
        events = rounds[len(spawned) - 1]

        async def stream():
            for ev in events:
                yield ev

        return SimpleNamespace(proc=None, stderr_task=None), stream()

    dispatched: list[str] = []

    async def dispatch(calls, session, messages, *args, **kwargs):
        for call in calls:
            dispatched.append(call["id"])
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": "ok"})
            yield ToolCallResult(call_id=call["id"], content="ok", ok=True)

    monkeypatch.setattr(engine, "_worker_spawner", spawn)
    monkeypatch.setattr(engine, "_dispatch_tool_calls", dispatch)
    monkeypatch.setattr(
        engine,
        "_build_messages_simple",
        lambda *a, **kw: [
            {"role": "user", "content": "Investigate."},
        ],
    )
    request = TurnRequest(prompt="Investigate.", extra={"session": session})
    state = EngineSessionState(session.hash, EngineKind.LOCAL_LLM)
    events = [
        e
        async for e in engine.run_turn(
            request,
            state,
            sessions_dir=cfg.sessions_dir,
            turn_id="t",
        )
    ]

    assert not [e for e in events if isinstance(e, TurnError)]
    assert isinstance(events[-1], TurnDone)
    assert dispatched == [f"call_{i}" for i in range(5)]
    assert len(spawned) == 2
    # The next round sees exactly the calls that ran, named, with their results.
    assistant = [m for m in spawned[1] if m.get("role") == "assistant" and m.get("tool_calls")]
    assert len(assistant) == 1
    sent = assistant[0]["tool_calls"]
    assert [c["id"] for c in sent] == [f"call_{i}" for i in range(5)]
    assert all(c["function"]["name"] == "bash" for c in sent)
    results = [m["tool_call_id"] for m in spawned[1] if m.get("role") == "tool"]
    assert results == [f"call_{i}" for i in range(5)]
    assert not any(isinstance(e, TextDelta) and "nudge" in (e.text or "") for e in events)


# --- shapes a provider might stream --------------


def _tc(index, *, id=None, name=None, args=None) -> dict:
    tc: dict = {"index": index}
    if id is not None:
        tc["id"] = id
        tc["type"] = "function"
    fn = {}
    if name is not None:
        fn["name"] = name
    if args is not None:
        fn["arguments"] = args
    if fn:
        tc["function"] = fn
    return tc


def _stream(*deltas: list, finish="tool_calls") -> list[dict]:
    chunks = [_chunk({"role": "assistant", "content": "On it."})]
    chunks += [_chunk({"tool_calls": d}) for d in deltas]
    chunks.append({"choices": [{"index": 0, "finish_reason": finish, "delta": {}}]})
    return chunks


def test_an_announced_call_without_arguments_is_not_run(monkeypatch, capsys, tmp_path):
    # All headers first, arguments later: at the cap nothing has arguments yet,
    # and an empty buffer is not "{}".
    monkeypatch.setenv("SHAREDAGENTS_MAX_TOOL_CALLS_PER_RESPONSE", "2")
    chunks = _stream(
        [
            _tc(0, id="call_0", name="bash", args=""),
            _tc(1, id="call_1", name="bash", args=""),
            _tc(2, id="call_2", name="bash", args=""),
        ]
    )
    events, _ = _run_worker(monkeypatch, capsys, tmp_path, chunks)
    calls, metrics = _calls_and_metrics(events)
    assert calls == []
    assert metrics["finish_reason"] == "length"  # -> the truncated-call nudge
    assert [t["name"] for t in metrics["truncated_tool_calls"]] == ["bash", "bash"]


def test_an_explicitly_empty_object_is_a_complete_call(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("SHAREDAGENTS_MAX_TOOL_CALLS_PER_RESPONSE", "1")
    chunks = _stream(
        [_tc(0, id="call_0", name="list", args="{}")],
        [_tc(1, id="call_1", name="bash", args='{"command": "ls"}')],
    )
    events, _ = _run_worker(monkeypatch, capsys, tmp_path, chunks)
    calls, metrics = _calls_and_metrics(events)
    assert [(c["id"], c["name"], c["arguments"]) for c in calls] == [("call_0", "list", {})]
    assert metrics["truncated_tool_calls"] == []


def test_a_call_whose_name_never_arrived_is_not_run(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("SHAREDAGENTS_MAX_TOOL_CALLS_PER_RESPONSE", "2")
    chunks = _stream(
        [_tc(0, id="call_0", name="bash", args='{"command": "ls"}')],
        [_tc(1, id="call_1", args='{"command": "pwd"}')],  # name not yet sent
        [_tc(2, id="call_2", name="bash", args='{"command": "id"}')],
    )
    events, _ = _run_worker(monkeypatch, capsys, tmp_path, chunks)
    calls, metrics = _calls_and_metrics(events)
    assert [c["id"] for c in calls] == ["call_0"]
    assert metrics["truncated_tool_calls"] == [
        {"name": "", "args_chars": len('{"command": "pwd"}')},
    ]


def test_fragments_after_the_overflowing_index_still_finish_kept_calls(
    monkeypatch,
    capsys,
    tmp_path,
):
    # One chunk carries the header of call 2 (over the cap) AND the rest of
    # call 1's arguments: call 1 must still complete and run.
    monkeypatch.setenv("SHAREDAGENTS_MAX_TOOL_CALLS_PER_RESPONSE", "2")
    chunks = _stream(
        [_tc(0, id="call_0", name="bash", args='{"command": "ls"}')],
        [_tc(1, id="call_1", name="bash", args='{"command": ')],
        [_tc(2, id="call_2", name="bash", args=""), _tc(1, args='"pwd"}')],
    )
    events, _ = _run_worker(monkeypatch, capsys, tmp_path, chunks)
    calls, metrics = _calls_and_metrics(events)
    assert [(c["id"], c["arguments"]) for c in calls] == [
        ("call_0", {"command": "ls"}),
        ("call_1", {"command": "pwd"}),
    ]
    assert metrics["tool_calls_capped"] == 2
    assert metrics["truncated_tool_calls"] == []


@pytest.mark.parametrize("n,capped", [(32, 0), (33, 32)])
def test_the_cap_trips_only_past_n(monkeypatch, capsys, tmp_path, n, capped):
    events, _ = _run_worker(monkeypatch, capsys, tmp_path, _batch(n))
    calls, metrics = _calls_and_metrics(events)
    assert len(calls) == 32
    assert metrics["tool_calls_capped"] == capped


def test_estimated_tokens_split_arguments_out_of_reasoning(monkeypatch, capsys, tmp_path):
    # A capped round has no usage chunk: its arguments are output, not thinking.
    monkeypatch.setenv("SHAREDAGENTS_MAX_TOOL_CALLS_PER_RESPONSE", "4")
    chunks = _batch(10)
    chunks.insert(1, _chunk({"reasoning_content": "plan " * 20}))
    events, _ = _run_worker(monkeypatch, capsys, tmp_path, chunks)
    _, metrics = _calls_and_metrics(events)
    assert metrics["tokens_estimated"] is True
    assert metrics["content_tokens"] > metrics["reasoning_tokens"] > 0
