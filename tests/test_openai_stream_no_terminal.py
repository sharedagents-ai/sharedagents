"""Regression: a turn stream that ends without a terminal event must still
produce a clean SSE terminal chunk (no ``UnboundLocalError``)."""

from __future__ import annotations

import asyncio
import json

from sharedagents_gateway import openai_endpoint


def test_stream_without_terminal_event(monkeypatch) -> None:
    async def _empty_turn(**_kwargs):
        return
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(openai_endpoint, "_run_one_turn", _empty_turn)

    async def _collect() -> list[bytes]:
        return [
            c
            async for c in openai_endpoint._stream_chat_completion(
                manager=None,
                agent="jarvis",
                prompt_blocks=[],
                completion_id="cmpl-1",
                created=0,
            )
        ]

    chunks = asyncio.run(_collect())
    assert chunks[-1] == b"data: [DONE]\n\n"
    frames = [json.loads(c[len(b"data: ") :]) for c in chunks[:-1]]
    terminal = frames[-1]
    assert terminal["choices"][0]["finish_reason"] == "stop"
    assert terminal["model_alias"] == ""
