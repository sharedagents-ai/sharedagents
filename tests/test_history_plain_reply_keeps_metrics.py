"""A plain one-bubble reply keeps its serving metrics in replayed history.

The web stats line under a reply (``alias/model``, tokens, speed) is rendered
from the assistant segment's ``metrics``. History used to drop the segments of
any reply with no thinking and no tool call, so that line showed while the
reply streamed and vanished on reload. Seen live with DeepSeek ``dpsk-pro``,
which streams no reasoning for short answers.
"""

from __future__ import annotations

from sharedagents_gateway.control_endpoint import _fold_turns_to_messages
from sharedagents_session.session_reader import fold_events_into_turns

_TURN = {"turn_idx": 0, "physical_turn_id": "t0", "round_idx": 0}
_METRICS = {"model": "deepseek-v4-pro", "model_alias": "dpsk-pro", "completion_tokens": 3}


def _events(*, thinking: str = ""):
    yield {"type": "turn_started", "user": "Reply with exactly: READY", **_TURN}
    yield {"type": "round_started", "model": "deepseek-v4-pro", **_TURN}
    if thinking:
        yield {"type": "thinking_delta", "delta": thinking, **_TURN}
    yield {"type": "text_delta", "delta": "READY", **_TURN}
    yield {"type": "round_ended", "finish_reason": "stop", "metrics": _METRICS, **_TURN}
    yield {"type": "turn_ended", "aborted": False, **_TURN}


def _assistant(events):
    messages = _fold_turns_to_messages(list(fold_events_into_turns(events)))
    return [m for m in messages if m["role"] == "assistant"][-1]


def test_plain_reply_keeps_serving_metrics():
    reply = _assistant(_events())
    assert reply["content"] == "READY"
    metrics = [s.get("metrics") for s in reply.get("segments") or [] if s.get("kind") == "assistant"]
    assert metrics and metrics[-1]["model_alias"] == "dpsk-pro"


def test_reply_with_thinking_still_keeps_segments():
    reply = _assistant(_events(thinking="short plan"))
    assert reply["segments"][0]["thinking"] == "short plan"
    assert reply["segments"][0]["metrics"]["model"] == "deepseek-v4-pro"
