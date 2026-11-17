"""Worker tells the engine the upstream is alive while it has nothing to show.

A long tool-call argument produces no ``thinking``/``content`` event: the
worker buffers it and emits one ``tool_calls`` event at the end, and vLLM
sends no bytes at all for an array/object argument, so the vLLM router sends
SSE keep-alive comments instead. Without a liveness event the engine's idle
timeout (240s) ends a round that is making progress.
"""

from __future__ import annotations

import io
import json
import sys

import httpx

from sharedagents_gateway import worker
from sharedagents_vllm.proxy import KEEPALIVE_COMMENT


def _data(delta, finish=None):
    choice = {"index": 0, "delta": delta, "finish_reason": finish}
    return f"data: {json.dumps({'choices': [choice]})}\n\n".encode()


def _tool_delta(arguments, name=None):
    fn = {"arguments": arguments}
    if name:
        fn["name"] = name
    return _data({"tool_calls": [{"index": 0, "id": "call_1", "function": fn}]})


def _run(monkeypatch, capsys, frames):
    real_client = httpx.Client

    def handle(_request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=iter(frames),
        )

    def client(*args, **kwargs):
        return real_client(*args, transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.delenv("SHAREDAGENTS_DEBUG_DUMP", raising=False)
    monkeypatch.setattr(worker, "_install_term_handler", lambda: None)
    monkeypatch.setattr(worker.httpx, "Client", client)
    request = {
        "endpoint": "http://127.0.0.1:8000/v1",
        "model": "qwen3.8-27b-fp8",
        "messages": [{"role": "user", "content": "hi"}],
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request)))
    result = worker.main()
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    return result, [e["kind"] for e in events], events


_ARGS = '{"runs": [{"subagent_type": "codex", "prompt": "audit the network"}]}'


def test_router_keepalive_comment_becomes_a_liveness_event(monkeypatch, capsys):
    frames = [
        _data({"role": "assistant"}),
        KEEPALIVE_COMMENT,
        KEEPALIVE_COMMENT,
        _tool_delta(_ARGS, name="AgentSync"),
        _data({}, finish="tool_calls"),
        b"data: [DONE]\n\n",
    ]

    result, kinds, events = _run(monkeypatch, capsys, frames)

    assert result == 0
    assert kinds.count("keepalive") == 2
    assert kinds.index("keepalive") < kinds.index("tool_calls")
    (calls,) = [e for e in events if e["kind"] == "tool_calls"]
    assert calls["calls"][0]["arguments"] == json.loads(_ARGS)


def test_other_providers_keepalive_comments_stay_silent(monkeypatch, capsys):
    # DeepSeek-style comments must not hold the engine's idle timer open.
    frames = [
        _data({"content": "hello"}),
        b": keep-alive\n\n",
        b": keep-alive\n\n",
        _data({}, finish="stop"),
        b"data: [DONE]\n\n",
    ]

    result, kinds, _events = _run(monkeypatch, capsys, frames)

    assert result == 0 and "keepalive" not in kinds


def test_streamed_tool_arguments_emit_liveness_events(monkeypatch, capsys):
    monkeypatch.setattr(worker, "_TOOL_ARGS_KEEPALIVE_S", 0.0)
    pieces = [_ARGS[i : i + 8] for i in range(0, len(_ARGS), 8)]
    frames = [_tool_delta(pieces[0], name="AgentSync")]
    frames += [_tool_delta(piece) for piece in pieces[1:]]
    frames += [_data({}, finish="tool_calls"), b"data: [DONE]\n\n"]

    result, kinds, _events = _run(monkeypatch, capsys, frames)

    assert result == 0
    assert kinds.count("keepalive") == len(pieces)
    assert "tool_calls" in kinds


def test_tool_argument_liveness_is_throttled(monkeypatch, capsys):
    pieces = [_ARGS[i : i + 8] for i in range(0, len(_ARGS), 8)]
    frames = [_tool_delta(pieces[0], name="AgentSync")]
    frames += [_tool_delta(piece) for piece in pieces[1:]]
    frames += [_data({}, finish="tool_calls"), b"data: [DONE]\n\n"]

    result, kinds, _events = _run(monkeypatch, capsys, frames)

    assert result == 0 and "keepalive" not in kinds


def test_worker_matches_the_comment_the_router_sends():
    assert KEEPALIVE_COMMENT.decode().startswith(worker._MUX_KEEPALIVE_PREFIX)
