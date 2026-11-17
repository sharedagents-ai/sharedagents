"""External browser SSO must use the same mount/verifier as the outer proxy."""

from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
import requests
from flask import Response


@pytest.fixture
def web(tmp_path, monkeypatch):
    # Bind import-time state to a disposable home, never production credentials.
    monkeypatch.setenv("SHAREDAGENTS_HOME", str(tmp_path))
    monkeypatch.setenv("SHAREDAGENTS_WEB_SECRET", "isolated-sso-test-secret")
    for key in (
        "SHAREDAGENTS_AGENTS_HOME",
        "SHAREDAGENTS_WEB_DB",
        "SHAREDAGENTS_WEB_DEBUG",
        "SHAREDAGENTS_WEB_TRUST_PROXY",
        "SHAREDAGENTS_WEB_BASE_PATH",
        "SHAREDAGENTS_WEB_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SHAREDAGENTS_SDLC_NO_RECOVERY", "1")
    monkeypatch.setenv("SHAREDAGENTS_EXTERNAL_AUTH", "1")
    monkeypatch.setenv("SHAREDAGENTS_EXTERNAL_LOGIN_URL", "/login")
    from sharedagents_web import access, server

    monkeypatch.setattr(server, "BASE_PATH", "/sharedagentsweb")
    monkeypatch.setattr(server, "DASHBOARD_URL", "http://127.0.0.1:20247")
    monkeypatch.setattr(server, "verify_native_auth", lambda: (False, "No native session"))
    monkeypatch.setattr(access, "gate_request", lambda email: None)
    return server


@pytest.mark.parametrize("mount,expected", [("/sharedagentsweb", "/sharedagentsweb/"), ("", "/sharedagents/")])
def test_external_verifier_checks_public_mount_and_forwards_cookie(web, monkeypatch, mount, expected):
    monkeypatch.setattr(web, "BASE_PATH", mount)
    calls = []

    def verify(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(status_code=200, headers={"X-User-Email": "viewer@example.test"})

    monkeypatch.setattr(web.requests, "get", verify)
    with web.app.test_request_context("/", headers={"Cookie": "access_token_cookie=test"}):
        assert web.verify_dashboard_auth() == (True, {"email": "viewer@example.test", "name": "User"})
    url, options = calls[0]
    assert url == "http://127.0.0.1:20247/verify-token"
    assert options["headers"] == {"X-Original-URI": expected, "Cookie": "access_token_cookie=test"}
    assert options["timeout"] == 5


def test_signed_in_external_cookie_opens_page_without_native_login(web, monkeypatch):
    monkeypatch.setattr(
        web.requests,
        "get",
        lambda *a, **k: SimpleNamespace(status_code=200, headers={"X-User-Email": "viewer@example.test"}),
    )
    with web.app.test_request_context("/", headers={"Cookie": "access_token_cookie=test"}):
        reply = web.require_auth_page(lambda: Response("protected"))()
        assert reply.status_code == 200
        assert reply.data == b"protected"


@pytest.mark.parametrize("result", [401, 403, 500, "unreachable", "missing_identity"])
def test_external_rejection_redirects_browser_but_api_stays_fail_closed(web, monkeypatch, result):
    def verify(*args, **kwargs):
        if result == "unreachable":
            raise requests.ConnectionError("offline")
        return SimpleNamespace(status_code=200 if result == "missing_identity" else result, headers={})

    monkeypatch.setattr(web.requests, "get", verify)
    with web.app.test_request_context("/?session=abc&monitor=1", headers={"Cookie": "access_token_cookie=test"}):
        reply = web.require_auth_page(lambda: pytest.fail("unauthorized page"))()
        assert reply.status_code == 302
        assert urlsplit(reply.location).path == "/login"
        assert parse_qs(urlsplit(reply.location).query)["next"] == ["/sharedagentsweb/?session=abc&monitor=1"]
        denied = web.require_auth(lambda: pytest.fail("unauthorized API"))()
        assert web.app.make_response(denied).status_code == 401


def test_native_login_url_redirects_to_sso_without_return_loop(web):
    reply = web.app.test_client().get("/login")
    assert reply.status_code == 302
    assert parse_qs(urlsplit(reply.location).query)["next"] == ["/sharedagentsweb/"]
    assert urlsplit(reply.location).path == "/login"


@pytest.mark.parametrize("setting", ["disabled", "unconfigured"])
def test_standalone_native_login_is_unchanged(web, monkeypatch, setting):
    if setting == "disabled":
        monkeypatch.delenv("SHAREDAGENTS_EXTERNAL_AUTH")
    else:
        monkeypatch.delenv("SHAREDAGENTS_EXTERNAL_LOGIN_URL")
    client = web.app.test_client()
    assert client.get("/login").status_code == 200
    with web.app.test_request_context("/"):
        assert web.require_auth_page(lambda: pytest.fail("unauthorized page"))().location == "/sharedagentsweb/login"
