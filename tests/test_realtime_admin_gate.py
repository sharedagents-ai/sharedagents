"""Realtime voice drives agents directly, outside the Web's per-user grants,
so only admins may use it."""

from types import SimpleNamespace

import pytest

pytest.importorskip("aiohttp")

from sharedagents_web.native_auth import ROLE_ADMIN, NativeAuthStore, resolve_admin_source  # noqa: E402


@pytest.fixture()
def voice(tmp_path, monkeypatch):
    from sharedagents_realtime import server as rt

    for key in ("SHAREDAGENTS_WEB_ADMINS", "SHAREDAGENTS_WEB_LOCAL_OPERATOR_EMAIL", "SHAREDAGENTS_REALTIME_DEBUG"):
        monkeypatch.delenv(key, raising=False)
    db = tmp_path / "sessions.db"
    monkeypatch.setenv("SHAREDAGENTS_WEB_DB", str(db))
    monkeypatch.setattr(rt, "_NATIVE_AUTH_STORE", None)
    store = NativeAuthStore(db)
    store.create_user("admin@example.com", "pw", role=ROLE_ADMIN)
    store.create_user("user@example.com", "pw")
    return rt, store


def _request(token):
    from sharedagents_web.native_auth import COOKIE_NAME

    return SimpleNamespace(cookies={COOKIE_NAME: token} if token else {}, headers={})


def test_only_admins_pass_the_voice_gate(voice):
    rt, store = voice
    assert rt._native_token_ok(_request(store.issue_token("admin@example.com"))) is True
    assert rt._native_token_ok(_request(store.issue_token("user@example.com"))) is False
    assert rt._native_token_ok(_request("")) is False


def test_resolve_admin_source_matches_the_web_rules(tmp_path, monkeypatch):
    for key in ("SHAREDAGENTS_WEB_ADMINS", "SHAREDAGENTS_WEB_LOCAL_OPERATOR_EMAIL"):
        monkeypatch.delenv(key, raising=False)
    store = NativeAuthStore(tmp_path / "db")
    store.create_user("a@example.com", "pw")
    store.create_user("b@example.com", "pw")
    # Nobody is an admin yet: the oldest account is, so nobody is locked out.
    assert resolve_admin_source(store, "a@example.com") == "fallback"
    assert resolve_admin_source(store, "b@example.com") is None
    store.set_role("b@example.com", ROLE_ADMIN)
    assert resolve_admin_source(store, "b@example.com") == "role"
    assert resolve_admin_source(store, "a@example.com") is None
    monkeypatch.setenv("SHAREDAGENTS_WEB_ADMINS", "a@example.com")
    assert resolve_admin_source(store, "A@example.com") == "env"


def test_external_auth_login_is_also_admin_only(voice, monkeypatch):
    import asyncio

    rt, store = voice
    monkeypatch.setenv("SHAREDAGENTS_EXTERNAL_AUTH", "1")

    class _Resp:
        def __init__(self, email):
            self.status_code = 200
            self.headers = {"X-User-Email": email}

    class _Client:
        email = ""

        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, *a, **k):
            return _Resp(_Client.email)

    monkeypatch.setattr(rt.httpx, "AsyncClient", _Client)
    request = SimpleNamespace(cookies={}, headers={"Cookie": "dash=1"})
    _Client.email = "user@example.com"
    assert asyncio.run(rt._authorized(request)) is False
    _Client.email = "admin@example.com"
    assert asyncio.run(rt._authorized(request)) is True
