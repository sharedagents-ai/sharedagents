"""Worker fails fast when a provider queues a request behind SSE keep-alives.

DeepSeek holds an accepted request open under load and sends ``: keep-alive``
comments until it can start inference. Each comment resets httpx's per-read
timeout, and the worker dropped them silently, so the only guard left was the
engine's 240s idle timeout, paid again on every retry.
"""

from __future__ import annotations

import io
import json
import sys
import time

import httpx
import pytest

from sharedagents_gateway import worker

_CHUNK = {"choices": [{"delta": {"content": "done"}, "finish_reason": "stop"}]}


def _run(monkeypatch, capsys, body, *, stall_s):
    real_client = httpx.Client
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body(),
        )

    def client(*args, **kwargs):
        return real_client(*args, transport=httpx.MockTransport(handle), **kwargs)

    if stall_s is None:
        monkeypatch.delenv("SHAREDAGENTS_WORKER_QUEUE_STALL_S", raising=False)
    else:
        monkeypatch.setenv("SHAREDAGENTS_WORKER_QUEUE_STALL_S", str(stall_s))
    monkeypatch.delenv("SHAREDAGENTS_DEBUG_DUMP", raising=False)
    monkeypatch.setattr(worker, "_install_term_handler", lambda: None)
    monkeypatch.setattr(worker.httpx, "Client", client)
    request = {
        "endpoint": "https://api.deepseek.com/v1",
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "hi"}],
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request)))
    result = worker.main()
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    return result, events, requests


def _queued_then_answer(keepalives, pause=0.05):
    def body():
        for _ in range(keepalives):
            time.sleep(pause)
            yield b": keep-alive\n\n"
        yield f"data: {json.dumps(_CHUNK)}\n\ndata: [DONE]\n\n".encode()

    return body


def test_keepalive_only_stream_fails_fast_with_a_retryable_error(monkeypatch, capsys):
    result, events, _ = _run(monkeypatch, capsys, _queued_then_answer(20), stall_s=0.2)
    assert result == 1
    errors = [e for e in events if e["kind"] == "error"]
    assert len(errors) == 1
    message = errors[0]["message"]
    assert "keep-alive" in message and "upstream_unavailable" in message
    # The engine's round-retry / failover classifiers must both recognise it.
    from sharedagents_agent.agents.local_llm.state import (
        _is_endpoint_down,
        _is_transient_upstream,
    )

    assert _is_transient_upstream(message) and _is_endpoint_down(message)
    assert not any(e["kind"] in ("content", "done") for e in events)


def test_queue_that_clears_before_the_deadline_streams_normally(monkeypatch, capsys):
    result, events, _ = _run(monkeypatch, capsys, _queued_then_answer(2), stall_s=5)
    assert result == 0
    assert not any(e["kind"] == "error" for e in events)
    assert any(e["kind"] == "done" for e in events)


def test_keepalives_after_the_first_chunk_never_trip_the_guard(monkeypatch, capsys):
    def body():
        yield f"data: {json.dumps({'choices': [{'delta': {'content': 'a'}}]})}\n\n".encode()
        for _ in range(8):
            time.sleep(0.05)
            yield b": keep-alive\n\n"
        yield f"data: {json.dumps(_CHUNK)}\n\ndata: [DONE]\n\n".encode()

    result, events, _ = _run(monkeypatch, capsys, body, stall_s=0.1)
    assert result == 0
    assert not any(e["kind"] == "error" for e in events)


def test_guard_can_be_disabled(monkeypatch, capsys):
    result, events, _ = _run(monkeypatch, capsys, _queued_then_answer(8), stall_s=0)
    assert result == 0
    assert not any(e["kind"] == "error" for e in events)


def test_a_silent_stream_without_keepalives_is_not_affected(monkeypatch, capsys):
    """Local vLLM sends nothing while prefilling; only keep-alives mean 'queued'."""

    def body():
        time.sleep(0.3)
        yield f"data: {json.dumps(_CHUNK)}\n\ndata: [DONE]\n\n".encode()

    result, events, _ = _run(monkeypatch, capsys, body, stall_s=0.1)
    assert result == 0
    assert not any(e["kind"] == "error" for e in events)


@pytest.mark.parametrize("raw", ["abc", "-5", "nan"])
def test_bad_env_values_fall_back_to_the_default(monkeypatch, raw):
    monkeypatch.setenv("SHAREDAGENTS_WORKER_QUEUE_STALL_S", raw)
    assert worker._queue_stall_timeout_s() == worker._DEFAULT_QUEUE_STALL_S
