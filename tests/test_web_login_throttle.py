"""Brute-force throttle on POST /api/login: per IP, per account, proxy-safe."""

import os

os.environ.setdefault("SHAREDAGENTS_WEB_SECRET", "test-secret-login-throttle")
os.environ.setdefault("SHAREDAGENTS_SDLC_NO_RECOVERY", "1")
for _k in ("SHAREDAGENTS_WEB_BASE_PATH", "SHAREDAGENTS_WEB_BASE_URL"):
    os.environ.pop(_k, None)

import pytest  # noqa: E402

from sharedagents_web import server  # noqa: E402
from sharedagents_web.native_auth import NativeAuthStore  # noqa: E402
from sharedagents_web.routes import auth as auth_routes  # noqa: E402

USER = "user@example.com"
GOOD = "right-password"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("SHAREDAGENTS_WEB_TRUSTED_PROXIES", raising=False)
    monkeypatch.setenv("SHAREDAGENTS_WEB_TRUST_PROXY", "1")  # the Docker default must not trust XFF
    store = NativeAuthStore(tmp_path / "web.db")
    store.create_user(USER, GOOD)
    monkeypatch.setattr(server, "auth_store", store)
    monkeypatch.setattr(auth_routes, "auth_store", store)
    now = [1000.0]
    monkeypatch.setattr(auth_routes, "_clock", lambda: now[0])
    monkeypatch.setattr(auth_routes, "_login_failures", {})
    c = server.app.test_client()
    c.now = now
    return c


def _login(c, password, ip="192.0.2.10", email=USER, headers=None):
    return c.post(
        "/api/login",
        json={"email": email, "password": password},
        environ_base={"REMOTE_ADDR": ip},
        headers=headers or {},
    )


def test_failures_get_blocked_then_a_good_login_works_after_the_window(client):
    for _ in range(auth_routes.LOGIN_FREE_FAILURES):
        assert _login(client, "wrong").status_code == 401
    blocked = _login(client, GOOD)
    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) == auth_routes.LOGIN_BLOCK_BASE_S
    assert "token" not in blocked.get_data(as_text=True)
    client.now[0] += auth_routes.LOGIN_BLOCK_BASE_S + 1
    assert _login(client, GOOD).status_code == 200
    assert _login(client, GOOD).status_code == 200  # a success does not renew the block


def test_backoff_doubles_and_unknown_accounts_get_the_same_error(client):
    for _ in range(auth_routes.LOGIN_FREE_FAILURES):
        _login(client, "wrong")
    client.now[0] += auth_routes.LOGIN_BLOCK_BASE_S + 1
    assert _login(client, "wrong").status_code == 401
    assert int(_login(client, GOOD).headers["Retry-After"]) == 2 * auth_routes.LOGIN_BLOCK_BASE_S
    missing = _login(client, "wrong", ip="192.0.2.19", email="nobody@example.com")
    assert missing.status_code == 401
    assert missing.get_json() == {"error": "Invalid credentials"}


def test_ips_are_isolated_but_the_account_is_still_protected(client):
    for _ in range(auth_routes.LOGIN_FREE_FAILURES):
        _login(client, "wrong", ip="192.0.2.10", email="a@example.com")
    assert _login(client, "wrong", ip="192.0.2.10", email="b@example.com").status_code == 429
    assert _login(client, GOOD, ip="192.0.2.11").status_code == 200
    for i in range(auth_routes.LOGIN_FREE_FAILURES):
        _login(client, "wrong", ip=f"198.51.100.{i}")
    assert _login(client, GOOD, ip="192.0.2.12").status_code == 429


def test_spoofed_forwarded_for_is_ignored_unless_the_proxy_is_trusted(client, monkeypatch):
    for i in range(auth_routes.LOGIN_FREE_FAILURES):
        _login(client, "wrong", email=f"x{i}@example.com", headers={"X-Forwarded-For": f"203.0.113.{i}"})
    assert (
        _login(client, "wrong", email="y@example.com", headers={"X-Forwarded-For": "198.51.100.7"}).status_code == 429
    )

    monkeypatch.setenv("SHAREDAGENTS_WEB_TRUSTED_PROXIES", "198.51.100.250, 192.0.2.10")
    monkeypatch.setattr(auth_routes, "_login_failures", {})
    for i in range(auth_routes.LOGIN_FREE_FAILURES):
        _login(client, "wrong", email=f"x{i}@example.com", headers={"X-Forwarded-For": f"203.0.113.{i}, 192.0.2.1"})
    hdr = {"X-Forwarded-For": "203.0.113.200, 192.0.2.2"}
    assert _login(client, "wrong", email="y@example.com", headers=hdr).status_code == 401


def test_a_concurrent_burst_cannot_exceed_the_free_attempts(client):
    from concurrent.futures import ThreadPoolExecutor

    def attempt(_):
        return _login(server.app.test_client(), "wrong").status_code

    with ThreadPoolExecutor(max_workers=16) as pool:
        codes = list(pool.map(attempt, range(32)))
    assert codes.count(401) == auth_routes.LOGIN_FREE_FAILURES
    assert codes.count(429) == 32 - auth_routes.LOGIN_FREE_FAILURES


def test_retained_state_is_capped(client, monkeypatch):
    monkeypatch.setattr(auth_routes, "_LOGIN_MAX_KEYS", 6)
    for i in range(20):
        _login(client, "wrong", ip=f"198.51.100.{i}", email=f"u{i}@example.com")
    assert len(auth_routes._login_failures) <= 6


def test_overlapping_attempts_at_capacity_keep_the_cap(client, monkeypatch):
    monkeypatch.setattr(auth_routes, "_LOGIN_MAX_KEYS", 6)
    pairs = [(("ip", f"198.51.100.{i}"), ("account", f"c{i}@example.com")) for i in range(4)]
    started = [keys for keys in pairs if auth_routes._login_begin(keys) == 0]
    assert started
    for keys in started:
        auth_routes._login_end(keys, False)
    assert len(auth_routes._login_failures) <= 6


def test_capacity_eviction_never_drops_an_in_flight_attempt(client, monkeypatch):
    monkeypatch.setattr(auth_routes, "_LOGIN_MAX_KEYS", 4)
    first = (("ip", "198.51.100.1"), ("account", "d1@example.com"))
    second = (("ip", "198.51.100.2"), ("account", "d2@example.com"))
    assert auth_routes._login_begin(first) == 0
    assert auth_routes._login_begin(second) == 0
    assert auth_routes._login_begin((("ip", "198.51.100.3"), ("account", "d3@example.com"))) == 1
    assert all(k in auth_routes._login_failures for k in first + second)


def test_a_success_keeps_other_in_flight_checks_on_the_account(client):
    account = ("account", "e@example.com")
    slow = (("ip", "198.51.100.1"), account)
    fast = (("ip", "198.51.100.2"), account)
    assert auth_routes._login_begin(slow) == 0
    assert auth_routes._login_begin(fast) == 0
    auth_routes._login_end(fast, True)
    assert auth_routes._login_failures[account][3] == 1
    auth_routes._login_end(slow, False)
    assert auth_routes._login_failures[account][:1] == (1,)
    assert auth_routes._login_failures[account][3] == 0


def test_waitress_keeps_forwarded_for_from_listed_proxies(client, monkeypatch):
    import threading
    import urllib.error
    import urllib.request

    create_server = pytest.importorskip("waitress.server").create_server
    monkeypatch.setenv("SHAREDAGENTS_WEB_TRUSTED_PROXIES", "192.0.2.99, 127.0.0.1")
    srv = create_server(server.app, host="127.0.0.1", port=0, **server._waitress_proxy_kwargs())
    threading.Thread(target=srv.run, daemon=True).start()

    def status(hop, email):
        req = urllib.request.Request(
            f"http://127.0.0.1:{srv.effective_port}/api/login",
            data=b'{"email": "%s", "password": "wrong"}' % email.encode(),
            headers={"Content-Type": "application/json", "X-Forwarded-For": hop},
        )
        try:
            return urllib.request.urlopen(req, timeout=10).status
        except urllib.error.HTTPError as exc:
            return exc.code

    try:
        for i in range(auth_routes.LOGIN_FREE_FAILURES):
            assert status("203.0.113.1", f"w{i}@example.com") == 401
        assert status("203.0.113.1", "z@example.com") == 429
        assert status("203.0.113.2", "z@example.com") == 401
    finally:
        srv.close()
