"""Steer is decided by the server, and a background result never opens a
second turn next to a running one.

The page read the session as idle while an ACP turn was still running,
so the web Steer button skipped
the cancel and the text went straight back into the queue. The idle came
from a subagent result that the gateway had admitted as a SECOND turn on the
same session; that turn's end cleared the session's live marker.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from sharedagents_gateway import control_endpoint as ce


class _Backend:
    def __init__(self, *, steers=False, has_steer=True):
        self.prompts: list = []
        self.queued: list = []
        self.steer_calls: list[str] = []
        self.removed: list = []
        self._steers = steers
        if not has_steer:
            self.steer_now = None  # an engine path with no steer hook

    async def steer_now(self, sid):  # noqa: F811 — replaced per instance
        self.steer_calls.append(sid)
        return {"ok": True, "steered": self._steers}

    async def prompt(self, sid, message_id, blocks):
        self.prompts.append(blocks)
        return
        yield

    async def queue_add(self, sid, text, *, source="", author="", actor=None, access=None):
        self.queued.append({"text": text, "source": source, "author": author})
        return {"ok": True, "queue_id": f"q{len(self.queued)}"}

    async def queue_remove(self, sid, queue_id, *, reason=None):
        self.removed.append((queue_id, reason))
        return {"ok": True}

    async def queue_list(self, sid):
        return {"pending": []}


class _Manager:
    def __init__(self, backends):
        self.backends = backends

    def get(self, name):
        return self.backends[name]


def _home(tmp_path, monkeypatch, sid):
    monkeypatch.setenv("SHAREDAGENTS_HOME", str(tmp_path))
    for key in ("SHAREDAGENTS_AGENTS_HOME", "SHAREDAGENTS_SESSIONS_DIR"):
        monkeypatch.delenv(key, raising=False)
    sessions = tmp_path / "sessions"
    sessions.mkdir(exist_ok=True)
    (sessions / f"{sid}.jsonl").write_bytes(b"")
    return sessions


def _queue(sessions, sid, queue_id, text):
    from sharedagents_session.session_writer import get_session_writer

    get_session_writer(sessions, sid).lifecycle(
        kind="queue_msg_added",
        payload={
            "queue_id": queue_id,
            "text": text,
            "source": "web",
            "author": "",
        },
    )


def _app(monkeypatch, backends, owner, **kw):
    monkeypatch.setattr(ce, "_owner_agent_for", lambda sid: owner)
    return ce.make_control_app(
        agent_names=list(backends),
        manager=_Manager(backends),
        bearer_token=None,
        **kw,
    )


def test_steer_reaches_the_owner_turn(tmp_path, monkeypatch):
    sid = "s-steer-owner"
    sessions = _home(tmp_path, monkeypatch, sid)
    _queue(sessions, sid, "q-1", "Where's the screenshot")
    devin = _Backend(steers=True)
    backends = {"jarvis": _Backend(), "devin": devin}
    with TestClient(_app(monkeypatch, backends, "devin")) as client:
        r = client.post(f"/orchestrator/sessions/{sid}/steer", json={"queue_ids": ["q-1"]})
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True, "steered": True, "agent": "devin"}
    assert devin.steer_calls == [sid]
    assert backends["jarvis"].steer_calls == []


def test_steer_finds_the_turn_on_another_backend(tmp_path, monkeypatch):
    """The session reads as one agent while another backend runs its turn."""
    sid = "s-steer-other"
    sessions = _home(tmp_path, monkeypatch, sid)
    _queue(sessions, sid, "q-1", "steer me")
    backends = {"jarvis": _Backend(steers=False), "devin": _Backend(steers=True)}
    with TestClient(_app(monkeypatch, backends, "jarvis")) as client:
        r = client.post(f"/orchestrator/sessions/{sid}/steer", json={"queue_ids": ["q-1"]})
    assert r.status_code == 200, r.text
    assert r.json()["steered"] is True and r.json()["agent"] == "devin"


def test_steer_of_a_delivered_item_is_not_pending(tmp_path, monkeypatch):
    sid = "s-steer-gone"
    _home(tmp_path, monkeypatch, sid)
    backends = {"devin": _Backend(steers=True)}
    with TestClient(_app(monkeypatch, backends, "devin")) as client:
        r = client.post(f"/orchestrator/sessions/{sid}/steer", json={"queue_ids": ["q-x"]})
    assert r.status_code == 409, r.text
    assert r.json()["error_code"] == "not_pending"
    assert backends["devin"].steer_calls == []


def test_unsteerable_turn_reports_the_server_busy_state(tmp_path, monkeypatch):
    """No engine can steer: the reply carries the SERVER's busy bit."""
    sid = "s-steer-busy"
    sessions = _home(tmp_path, monkeypatch, sid)
    _queue(sessions, sid, "q-1", "later")
    backends = {"claude": _Backend(has_steer=False)}
    live = {"value": True}
    monkeypatch.setattr(ce, "_durable_session_busy", lambda root, s: live["value"])
    with TestClient(_app(monkeypatch, backends, "claude")) as client:
        r = client.post(f"/orchestrator/sessions/{sid}/steer", json={"queue_ids": ["q-1"]})
        assert r.json() == {"ok": True, "steered": False, "busy": True}, r.text
        live["value"] = False
        r = client.post(f"/orchestrator/sessions/{sid}/steer", json={"queue_ids": ["q-1"]})
        assert r.json() == {"ok": True, "steered": False, "busy": False}, r.text


def test_north_gateway_proxies_steer_to_the_orchestrator(tmp_path, monkeypatch):
    sid = "s-steer-north"
    _home(tmp_path, monkeypatch, sid)
    calls = []

    async def proxy(ctx, method, path, *, json_body=None, principal="", timeout_s=30.0):
        calls.append((method, path, json_body, principal))
        return ce.JSONResponse({"ok": True, "steered": True, "agent": "devin"})

    monkeypatch.setattr(ce, "_proxy_command", proxy)
    backends = {"devin": _Backend(steers=False)}
    app = _app(monkeypatch, backends, "devin", orchestrator_url="https://example.com")
    with TestClient(app) as client:
        r = client.post(
            f"/orchestrator/sessions/{sid}/steer",
            json={"queue_ids": ["q-1"]},
            headers={"X-SharedAgents-Principal": "user@example.com"},
        )
    assert r.json()["steered"] is True
    assert calls == [
        (
            "POST",
            f"/control/v1/orchestrator/sessions/{sid}/steer",
            {"queue_ids": ["q-1"]},
            "user@example.com",
        )
    ]
    assert backends["devin"].steer_calls == []


def test_bridge_steer_routes_the_live_turn_to_its_engine():
    from sharedagents_gateway.acp_bridge import CoreBackend

    seen = []
    engine = SimpleNamespace(steer_now=lambda tid: seen.append(tid) or True)
    bridge = SimpleNamespace(
        _new_backend_active_turns={"s1": "turn-9"},
        _new_agent=SimpleNamespace(engine=engine),
    )
    assert asyncio.run(CoreBackend.steer_now(bridge, "s1"))["steered"] is True
    assert seen == ["turn-9"]
    res = asyncio.run(CoreBackend.steer_now(bridge, "s-idle"))
    assert res["steered"] is False and res["reason"] == "no_live_turn"
    bridge._new_agent = SimpleNamespace(engine=SimpleNamespace())
    res = asyncio.run(CoreBackend.steer_now(bridge, "s1"))
    assert res["steered"] is False and res["reason"] == "unsupported"


def test_background_result_waits_while_the_session_turn_is_open(tmp_path, monkeypatch):
    """No admission entry, but the log shows an open turn: queue, never start."""
    sid = "s-bg-live"
    _home(tmp_path, monkeypatch, sid)
    live = {"value": True}
    monkeypatch.setattr(
        ce,
        "_durable_session_busy",
        lambda root, s: live["value"] and s == sid,
    )
    backend = _Backend()
    with TestClient(_app(monkeypatch, {"jarvis": backend}, "jarvis")) as client:
        assert ce._SUBMIT_INFLIGHT.get(sid) is None
        r = client.post(
            f"/orchestrator/sessions/{sid}/prompt",
            json={
                "prompt": "<delegation_wave>result</delegation_wave>",
                "background_job_id": "job_" + "b" * 16,
            },
        )
        assert r.status_code == 202, r.text
        assert r.json()["queued"] is True, r.json()
        time.sleep(0.3)
        assert backend.prompts == []

    # The retry sweep runs, but leaves the payload queued for the open turn.
    pending = [
        {
            "queue_id": "q9",
            "consume_ids": ["q9"],
            "source": "background_task",
            "author": "job_" + "d" * 16,
            "text": "result",
        }
    ]

    async def sweep():
        assert ce._schedule_background_queue_recovery(backend, sid, pending) is True
        await asyncio.sleep(0.3)

    asyncio.run(sweep())
    assert backend.prompts == []
    assert ("q9", "background_consumed") not in backend.removed

    # Once the turn has closed, the same sweep delivers it as one turn.
    live["value"] = False
    asyncio.run(sweep())
    assert len(backend.prompts) == 1
    assert ("q9", "background_consumed") in backend.removed


def test_background_recovery_starts_once_the_turn_is_closed(tmp_path, monkeypatch):
    sid = "s-bg-idle"
    _home(tmp_path, monkeypatch, sid)
    monkeypatch.setattr(ce, "_durable_session_busy", lambda root, s: False)
    backend = _Backend()
    with TestClient(_app(monkeypatch, {"jarvis": backend}, "jarvis")) as client:
        r = client.post(
            f"/orchestrator/sessions/{sid}/prompt",
            json={
                "prompt": "<delegation_wave>result</delegation_wave>",
                "background_job_id": "job_" + "c" * 16,
            },
        )
        assert r.status_code == 202, r.text
        end = time.monotonic() + 5
        while not backend.prompts and time.monotonic() < end:
            time.sleep(0.02)
    assert len(backend.prompts) == 1


def test_agent_backend_steers_an_external_engine_turn():
    """The production backend: ACP turns live in ``_cli_active_turns``.

    Found by the live web check: the route reached ``AgentACPBackend``, which
    had no ``steer_now``, so a running Devin turn answered ``steered: false``.
    """
    from sharedagents_agent.agents.gateway_shim import AgentACPBackend

    seen = []
    backend = AgentACPBackend.__new__(AgentACPBackend)
    backend._cli_active_turns = {"s-acp": "turn-acp"}
    backend.agent = SimpleNamespace(
        engine=SimpleNamespace(steer_now=lambda tid: seen.append(tid) or True),
    )
    backend._core = None
    res = asyncio.run(AgentACPBackend.steer_now(backend, "s-acp"))
    assert res == {"ok": True, "steered": True, "reason": ""}
    assert seen == ["turn-acp"]
    res = asyncio.run(AgentACPBackend.steer_now(backend, "s-none"))
    assert res["steered"] is False and res["reason"] == "no_live_turn"


def test_agent_backend_hands_a_local_turn_to_the_core_bridge():
    from sharedagents_agent.agents.gateway_shim import AgentACPBackend

    async def core_steer(sid):
        return {"ok": True, "steered": False, "reason": "unsupported"}

    backend = AgentACPBackend.__new__(AgentACPBackend)
    backend._cli_active_turns = {}
    backend.agent = SimpleNamespace(engine=SimpleNamespace())
    backend._core = SimpleNamespace(steer_now=core_steer)
    res = asyncio.run(AgentACPBackend.steer_now(backend, "s-local"))
    assert res["reason"] == "unsupported" and res["steered"] is False


@pytest.mark.parametrize("restart", [False, True])
def test_real_live_marker_releases_background_recovery(tmp_path, monkeypatch, restart):
    from sharedagents_agent.orchestrator import reset_stale_sessions
    from sharedagents_session.session_writer import get_session_writer

    sid = "s-real-live-restart" if restart else "s-real-live-close"
    sessions = _home(tmp_path, monkeypatch, sid)
    writer = get_session_writer(sessions, sid, engine="acp")
    writer.session_started({"agent_name": "acp", "engine": "acp"})
    writer.turn_started(turn_idx=0, user="Start")
    assert ce._session_turn_marked_live(sid)
    backend = _Backend()
    pending = [
        {
            "queue_id": "q-result",
            "consume_ids": ["q-result"],
            "source": "background_task",
            "author": "job_" + "e" * 16,
            "text": "Background result",
        }
    ]

    async def sweep():
        assert ce._schedule_background_queue_recovery(backend, sid, pending)
        task = ce._BACKGROUND_QUEUE_RECOVERY_TASKS[sid]
        await asyncio.wait_for(asyncio.shield(task), 10)

    asyncio.run(sweep())
    assert not backend.prompts
    assert not backend.removed
    if restart:
        assert reset_stale_sessions() == 1
    else:
        writer.turn_ended(turn_idx=0)
    assert not ce._session_turn_marked_live(sid)
    asyncio.run(sweep())
    assert len(backend.prompts) == 1
    assert backend.removed == [("q-result", "background_consumed")]
