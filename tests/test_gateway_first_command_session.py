"""A slash command can be a new chat's first input (``/goal`` in a fresh web
chat). Break: the gateway answered "session not found" because
only the prompt path created sessions from the chat's first intent."""

import pytest
from starlette.testclient import TestClient

from sharedagents_gateway import control_endpoint as ce

SID = "20261028T120000000000-fresh0000000"


class _Agent:
    def __init__(self, sessions):
        self.sessions, self.created = sessions, []

    def create_session(self, *, session_id, workspace_cwd="", workspace_origin="", mode=None, model=""):
        self.created.append({"session_id": session_id, "workspace_origin": workspace_origin, "mode": mode})
        (self.sessions / f"{session_id}.jsonl").write_bytes(b"")


class _Backend:
    def __init__(self, agent):
        self.agent, self.commands = agent, []

    async def run_command(self, session_id, text, *, frontend="api", confirmed=False):
        self.commands.append((session_id, text))
        return {"ok": True, "kind": "text", "text": "goal created"}


class _Manager:
    def __init__(self, backend):
        self.backend = backend

    def get(self, name):
        return self.backend


def _app(tmp_path, monkeypatch, **kw):
    monkeypatch.setenv("SHAREDAGENTS_HOME", str(tmp_path))
    for key in ("SHAREDAGENTS_AGENTS_HOME", "SHAREDAGENTS_SESSIONS_DIR"):
        monkeypatch.delenv(key, raising=False)
    sessions = tmp_path / "sessions"
    sessions.mkdir(exist_ok=True)
    backend = _Backend(_Agent(sessions))
    app = ce.make_control_app(agent_names=["jarvis"], manager=_Manager(backend), bearer_token=None, **kw)
    return app, backend


def test_first_command_creates_the_session_then_runs(tmp_path, monkeypatch):
    app, backend = _app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        r = client.post(
            "/jarvis/command",
            json={
                "text": "/goal ship it",
                "sessionId": SID,
                "initial_agent": "jarvis",
                "workspace_origin": "web",
                "permission_mode": "auto",
            },
        )
    assert r.status_code == 200 and r.json()["ok"], r.text
    assert backend.agent.created == [{"session_id": SID, "workspace_origin": "web", "mode": "auto"}]
    assert backend.commands == [(SID, "/goal ship it")]


def test_without_first_intent_an_unknown_session_is_still_not_found(tmp_path, monkeypatch):
    app, backend = _app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        r = client.post("/jarvis/command", json={"text": "/goal ship it", "sessionId": SID})
    assert r.status_code == 404
    assert backend.agent.created == [] and backend.commands == []


def test_an_existing_session_is_never_recreated(tmp_path, monkeypatch):
    app, backend = _app(tmp_path, monkeypatch)
    (tmp_path / "sessions" / f"{SID}.jsonl").write_bytes(b"")
    with TestClient(app) as client:
        r = client.post("/jarvis/command", json={"text": "/goal", "sessionId": SID, "initial_agent": "jarvis"})
    assert r.status_code == 200
    assert backend.agent.created == [] and backend.commands == [(SID, "/goal")]


def test_north_gateway_hands_the_first_command_to_the_orchestrator(tmp_path, monkeypatch):
    proxied = []

    async def fake_proxy(ctx, method, path, *, json_body=None, principal="", timeout_s=30.0):
        proxied.append((method, path, json_body, principal))
        return ce.JSONResponse({"ok": True})

    monkeypatch.setattr(ce, "_proxy_command", fake_proxy)
    app, backend = _app(tmp_path, monkeypatch, orchestrator_url="http://orchestrator")
    body = {"text": "/goal ship it", "sessionId": SID, "initial_agent": "jarvis"}
    with TestClient(app) as client:
        r = client.post("/jarvis/command", json=body, headers={"X-SharedAgents-Principal": "web:alice"})
    assert r.status_code == 200
    assert proxied == [("POST", "/control/v1/jarvis/command", body, "web:alice")]
    assert backend.agent.created == [] and backend.commands == []


@pytest.mark.parametrize("sid", ["../escaped", "./../escaped", "absolute", "nested/escaped", "..", "bad\x00id"])
@pytest.mark.parametrize("north", [False, True])
def test_first_command_refuses_session_paths_before_creation(tmp_path, monkeypatch, sid, north):
    async def no_proxy(*args, **kwargs):
        pytest.fail("invalid session id was forwarded")

    monkeypatch.setattr(ce, "_proxy_command", no_proxy)
    app, backend = _app(tmp_path, monkeypatch, orchestrator_url="http://orchestrator" if north else "")
    if sid == "absolute":
        sid = str(tmp_path / "absolute")
    with TestClient(app) as client:
        r = client.post(
            "/jarvis/command",
            json={
                "text": "/goal ship it",
                "sessionId": sid,
                "initial_agent": "jarvis",
            },
        )
    assert r.status_code == 400, r.text
    assert backend.agent.created == [] and backend.commands == []
    assert not (tmp_path / "escaped.jsonl").exists()
    assert not (tmp_path / "absolute.jsonl").exists()
