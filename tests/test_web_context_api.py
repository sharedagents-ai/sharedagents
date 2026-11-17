"""Web facade of the Context sheet: access-checked like other session reads.

The owner reads everything; a person the conversation is shared with reads the
conversation but not the agent's instructions; anyone else is refused before
the gateway is asked.
"""

import os
import secrets

os.environ.setdefault("SHAREDAGENTS_WEB_SECRET", secrets.token_urlsafe(32))
os.environ.setdefault("SHAREDAGENTS_SDLC_NO_RECOVERY", "1")
for _k in ("SHAREDAGENTS_WEB_BASE_PATH", "SHAREDAGENTS_WEB_BASE_URL"):
    os.environ.pop(_k, None)

import pytest  # noqa: E402

from sharedagents_core.access_grants import Grants  # noqa: E402
from sharedagents_session.session_reader import encode_line  # noqa: E402
from sharedagents_web import access, server  # noqa: E402
from sharedagents_web.native_auth import NativeAuthStore  # noqa: E402
from sharedagents_web.session_sharing import SharingStore  # noqa: E402
from sharedagents_web.session_store import SessionStore  # noqa: E402

ALICE = "alice@example.com"
BOB = "bob@example.com"
CAROL = "carol@example.com"
PASSWORD = secrets.token_urlsafe(24)


class _Resp:
    status_code = 200

    def __init__(self, path, params):
        self._body = {"ok": True, "path": path, "params": dict(params or {}), "items": []}

    def json(self):
        return self._body


@pytest.fixture()
def env(tmp_path, monkeypatch):
    for key in (
        "SHAREDAGENTS_WEB_ADMINS",
        "SHAREDAGENTS_WEB_LOCAL_OPERATOR_EMAIL",
        "SHAREDAGENTS_WEB_DEBUG",
        "SHAREDAGENTS_WEB_TRUST_PROXY",
    ):
        monkeypatch.delenv(key, raising=False)
    db = tmp_path / "web.db"
    gw = tmp_path / "gw"
    gw.mkdir()
    monkeypatch.setattr(server, "SESSION_DB_PATH", db)
    monkeypatch.setattr(server, "_SESSIONS_DIR_DEPRECATED", tmp_path / "legacy-missing")
    monkeypatch.setattr(server, "_GW_SESSIONS_DEPRECATED", tmp_path / "gw-missing")
    monkeypatch.setattr(server, "GW_SESSIONS_DIR", gw)
    monkeypatch.setattr(server, "store", SessionStore())
    monkeypatch.setattr(server, "auth_store", NativeAuthStore(db))
    monkeypatch.setattr(server, "sharing_store", SharingStore(db))
    from sharedagents_web.routes import auth as auth_routes, sessions as sessions_routes

    monkeypatch.setattr(auth_routes, "auth_store", server.auth_store)
    monkeypatch.setattr(sessions_routes, "store", server.store)
    monkeypatch.setattr(sessions_routes, "GW_SESSIONS_DIR", gw)
    monkeypatch.setattr(server, "gw_root_sessions", lambda limit=50: [])
    calls = []

    def fake_gateway(method, path, **kw):
        calls.append((method, path, dict(kw.get("params") or {})))
        if "/context" in path:
            return _Resp(path, kw.get("params"))
        return None

    monkeypatch.setattr(sessions_routes, "gateway_request", fake_gateway)
    monkeypatch.setattr(server, "gateway_request", lambda *a, **k: None)
    monkeypatch.setattr(access, "agent_is_external", lambda name: False)
    grants = Grants.from_obj({"agents": ["jarvis"], "models": "*"})
    for who in (ALICE, BOB, CAROL):
        server.auth_store.create_user(who, PASSWORD, grants=grants)
    return tmp_path, calls


def _login(email):
    client = server.app.test_client()
    assert client.post("/api/login", json={"email": email, "password": PASSWORD}).status_code == 200
    return client


def _with_log(env_dir, sid):
    event = {"type": "session_started", "session_id": sid, "parent_session_id": ""}
    (env_dir / "gw" / f"{sid}.jsonl").write_bytes(encode_line(event))


def test_owner_reads_everything_and_may_compact(env):
    env_dir, calls = env
    sid = server.store.create(ALICE, "jarvis")["id"]
    _with_log(env_dir, sid)
    r = _login(ALICE).get(f"/api/sessions/{sid}/context?cursor=abc:50&limit=50")
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body["can_compact"] is True
    method, path, params = calls[-1]
    assert (method, path) == ("GET", f"/control/v1/orchestrator/sessions/{sid}/context")
    assert params == {"cursor": "abc:50", "limit": 50, "instructions": "1"}
    r = _login(ALICE).get(f"/api/sessions/{sid}/context/items/7?snapshot=abc&offset=0")
    assert r.status_code == 200
    assert calls[-1][1].endswith(f"/sessions/{sid}/context/items/7")
    assert calls[-1][2] == {"snapshot": "abc", "offset": 0, "instructions": "1"}


def test_a_stranger_is_refused_without_asking_the_gateway(env):
    env_dir, calls = env
    sid = server.store.create(ALICE, "jarvis")["id"]
    _with_log(env_dir, sid)
    carol = _login(CAROL)
    assert carol.get(f"/api/sessions/{sid}/context").status_code == 404
    assert carol.get(f"/api/sessions/{sid}/context/items/0").status_code == 404
    assert not [c for c in calls if "/context" in c[1]]


def test_a_shared_viewer_reads_the_conversation_but_not_the_instructions(env):
    env_dir, calls = env
    sid = server.store.create(ALICE, "jarvis")["id"]
    _with_log(env_dir, sid)
    server.sharing_store.set(sid, visibility="public", share_all=False, members=[BOB], by=ALICE)
    assert access.session_access(BOB, sid) >= access.VIEW
    r = _login(BOB).get(f"/api/sessions/{sid}/context")
    assert r.status_code == 200
    assert calls[-1][2]["instructions"] == "0"
    assert r.get_json()["can_compact"] is (access.session_access(BOB, sid) >= access.CHAT)


def test_an_owned_chat_without_a_log_is_empty_not_an_error(env):
    _env_dir, calls = env
    sid = server.store.create(ALICE, "jarvis")["id"]
    r = _login(ALICE).get(f"/api/sessions/{sid}/context")
    assert r.status_code == 200
    body = r.get_json()
    assert body["items"] == [] and body["empty_text"]
    assert not [c for c in calls if "/context" in c[1]]
