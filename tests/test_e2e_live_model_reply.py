"""End-to-end: a running WebUI returns a real model reply for each configured model.

Unit tests stub the engine, so they cannot catch a broken provider wiring (bad
endpoint, missing key, wrong alias, rejected effort). This test drives the same
HTTP API the browser uses -- login, create a chat, send a prompt, wait for the
turn -- against a live instance, and checks the reply and the model that served it.

Opt-in, because every turn spends real provider tokens or GPU time::

    SHAREDAGENTS_E2E_URL=http://localhost:8110 \\
    SHAREDAGENTS_E2E_EMAIL=you@example.com \\
    SHAREDAGENTS_E2E_PASSWORD_FILE=~/password.txt \\
    SHAREDAGENTS_E2E_MODELS=dpsk,qwen-local \\
    pytest -m e2e tests/test_e2e_live_model_reply.py

``SHAREDAGENTS_E2E_MODELS`` lists model aliases from the instance's config.
``SHAREDAGENTS_E2E_AGENT`` picks the agent (default ``jarvis``).
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import pytest

httpx = pytest.importorskip("httpx")

pytestmark = pytest.mark.e2e

_URL = os.environ.get("SHAREDAGENTS_E2E_URL", "").rstrip("/")
_EMAIL = os.environ.get("SHAREDAGENTS_E2E_EMAIL", "")
_PASSWORD_FILE = os.environ.get("SHAREDAGENTS_E2E_PASSWORD_FILE", "")
_MODELS = [m.strip() for m in os.environ.get("SHAREDAGENTS_E2E_MODELS", "").split(",") if m.strip()]
_AGENT = os.environ.get("SHAREDAGENTS_E2E_AGENT", "jarvis")
_TURN_TIMEOUT_S = float(os.environ.get("SHAREDAGENTS_E2E_TIMEOUT", "180"))

if not (_URL and _EMAIL and _PASSWORD_FILE and _MODELS):
    pytest.skip(
        "set SHAREDAGENTS_E2E_URL, _EMAIL, _PASSWORD_FILE and _MODELS to run live model e2e",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def client():
    password = Path(_PASSWORD_FILE).expanduser().read_text().strip()
    with httpx.Client(base_url=_URL, timeout=30) as c:
        resp = c.post("/api/login", json={"email": _EMAIL, "password": password})
        assert resp.status_code == 200, f"login failed: HTTP {resp.status_code}"
        yield c


def _send(client, sid: str, prompt: str, **extra) -> None:
    payload = {"prompt": prompt, "request_id": f"e2e-{uuid.uuid4().hex}", **extra}
    resp = client.post(f"/api/sessions/{sid}/prompt", json=payload)
    assert resp.status_code in (200, 202), resp.text
    assert resp.json().get("accepted") is True, resp.text


def _wait_for_reply(client, sid: str, n_assistant: int) -> dict:
    """Poll until the session is idle with at least ``n_assistant`` replies."""
    deadline = time.monotonic() + _TURN_TIMEOUT_S
    session: dict = {}
    while time.monotonic() < deadline:
        time.sleep(1.5)
        body = client.get(f"/api/sessions/{sid}").json()
        session = body.get("session", body)
        replies = [m for m in session.get("messages") or [] if m.get("role") == "assistant"]
        if len(replies) >= n_assistant and not session.get("active"):
            return session
    pytest.fail(f"no reply within {_TURN_TIMEOUT_S:.0f}s; last state: {session.get('state')!r}")


def _last_reply(session: dict) -> dict:
    return [m for m in session["messages"] if m.get("role") == "assistant"][-1]


def _served_by(reply: dict) -> tuple[str, str]:
    """Return ``(alias, provider model)`` from the reply's serving metrics."""
    for seg in reversed(reply.get("segments") or []):
        metrics = seg.get("metrics") or {}
        if metrics.get("model"):
            return str(metrics.get("model_alias") or ""), str(metrics["model"])
    return "", ""


@pytest.mark.parametrize("alias", _MODELS)
def test_model_replies_and_remembers_context(client, alias):
    resp = client.post("/api/sessions", json={"agent": _AGENT, "model": alias})
    assert resp.status_code in (200, 201), resp.text
    sid = resp.json()["session"]["id"]
    try:
        _send(
            client,
            sid,
            "Remember the code word PELICAN-7. Reply with exactly: READY",
            initial_agent=_AGENT,
            initial_model=alias,
        )
        session = _wait_for_reply(client, sid, 1)
        first = _last_reply(session)
        assert "READY" in (first.get("content") or "").upper(), first.get("content")
        served_alias, served_model = _served_by(first)
        assert served_model, "reply carries no serving-model metrics"
        # An unknown alias falls back to the default model; fail loudly instead.
        assert served_alias == alias, f"asked for {alias!r}, served by {served_alias!r}"

        _send(client, sid, "What was the code word? Reply with only the code word.")
        session = _wait_for_reply(client, sid, 2)
        follow_up = _last_reply(session)
        assert "PELICAN-7" in (follow_up.get("content") or "").upper(), follow_up.get("content")
    finally:
        client.delete(f"/api/sessions/{sid}")
