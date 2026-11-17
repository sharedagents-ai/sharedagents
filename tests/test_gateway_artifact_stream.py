"""The ``artifact_reference`` frame the gateway streams to channels.

Channels (WhatsApp, Telegram) scope an artifact to its turn and apply their
stale-replay cutoff to it, so the public frame must keep ``turn_idx`` and
``ts`` — while still never exposing the local path or locator.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

from sharedagents_gateway import control_endpoint as ce
from sharedagents_session.session_reader import encode_line

DIGEST = "ab" * 32
TS = "2026-09-25T12:00:00.000Z"
CANONICAL = {
    "type": "artifact_reference",
    "v": 1,
    "ts": TS,
    "turn_idx": 3,
    "physical_turn_id": "pt-1",
    "engine": "local",
    "path": "/home/user/.sharedagents/web/artifacts/sid/abab/report.pdf",
    "locator": "/home/user/.sharedagents/web/artifacts/sid/abab/report.pdf",
    "storage_scope": "artifact_store",
    "sha256": DIGEST,
    "media_type": "application/pdf",
    "display_name": "report.pdf",
    "size_bytes": 1234,
}


async def _first_artifact_frame(sid: str, query: dict) -> dict:
    response = await ce._stream_events_mode(SimpleNamespace(query_params=query), sid, None)
    body = response.body_iterator
    try:
        async for chunk in body:
            if not chunk.startswith("event: log"):
                continue
            frame = json.loads(chunk.split("data: ", 1)[1])
            if frame["event"]["type"] == "artifact_reference":
                return frame["event"]
    finally:
        await body.aclose()
    raise AssertionError("no artifact frame")


@pytest.mark.asyncio
async def test_stream_keeps_turn_and_time_but_never_the_path(tmp_path, monkeypatch):
    monkeypatch.setenv("SHAREDAGENTS_HOME", str(tmp_path))
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    monkeypatch.setenv("SHAREDAGENTS_SESSIONS_DIR", str(sessions))
    sid = "sess-artifact"
    started = {"type": "turn_started", "v": 1, "ts": TS, "turn_idx": 3, "user": "chart please"}
    (sessions / f"{sid}.jsonl").write_bytes(encode_line(started) + encode_line(CANONICAL))

    for query in (
        {"mode": "events", "from": "0", "reasoning": "off", "tools": "on"},  # WhatsApp
        {"mode": "events", "from": "0", "reasoning": "stream", "tools": "stream"},  # Telegram
    ):
        event = await asyncio.wait_for(_first_artifact_frame(sid, query), 10)
        assert event == {
            "type": "artifact_reference",
            "turn_idx": 3,
            "ts": TS,
            "sha256": DIGEST,
            "display_name": "report.pdf",
            "media_type": "application/pdf",
            "size_bytes": 1234,
        }
