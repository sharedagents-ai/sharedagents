"""Regression: an agent with no model configured must fail once, not retry.

The worker refuses with ``missing endpoint/model/messages`` before any
upstream request. It used to classify as a generic ``upstream error`` and
was auto-retried five times (an empty ERROR bubble every 5 s in the WebUI).
"""

from __future__ import annotations

from sharedagents_agent.agents.agent import _turn_error_is_retryable
from sharedagents_agent.turn_runtime import _classify_upstream_error


def test_missing_model_is_config_and_not_retryable(monkeypatch) -> None:
    monkeypatch.delenv("SHAREDAGENTS_TURN_RETRY_CLASSES", raising=False)
    monkeypatch.delenv("SHAREDAGENTS_TURN_RETRY_BLOCK_CLASSES", raising=False)
    for msg in (
        "missing endpoint/model/messages",
        "missing endpoint/model/messages: no model is configured for this agent; pick one with /model",
    ):
        cls = _classify_upstream_error(msg)
        assert cls == "config"
        assert _turn_error_is_retryable(cls) is False


def test_generic_upstream_error_still_retries(monkeypatch) -> None:
    monkeypatch.delenv("SHAREDAGENTS_TURN_RETRY_CLASSES", raising=False)
    monkeypatch.delenv("SHAREDAGENTS_TURN_RETRY_BLOCK_CLASSES", raising=False)
    cls = _classify_upstream_error("connection reset by peer")
    assert cls == "upstream error"
    assert _turn_error_is_retryable(cls) is True
