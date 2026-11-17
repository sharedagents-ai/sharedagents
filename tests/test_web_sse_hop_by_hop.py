"""SSE responses must not set hop-by-hop headers (PEP 3333 / waitress).

Waitress rejects ``Connection`` inside ``start_response`` and turns that
AssertionError into HTTP 500. Gunicorn accepts the same header, so the Linux
suite never saw the Windows failure. The Flask test client does not enforce
the rule, so both the header check and a real waitress GET are required.
"""

import http.client
import json
import os
import re
import threading
from pathlib import Path

os.environ.setdefault("SHAREDAGENTS_WEB_SECRET", "test-secret-sse-hop")
os.environ.setdefault("SHAREDAGENTS_SDLC_NO_RECOVERY", "1")
for _k in ("SHAREDAGENTS_WEB_BASE_PATH", "SHAREDAGENTS_WEB_BASE_URL"):
    os.environ.pop(_k, None)

import pytest  # noqa: E402

from sharedagents_core.access_grants import Grants  # noqa: E402
from sharedagents_web import server  # noqa: E402
from sharedagents_web.native_auth import NativeAuthStore  # noqa: E402
from sharedagents_web.routes import ui_stream  # noqa: E402
from sharedagents_web.session_sharing import SharingStore  # noqa: E402
from sharedagents_web.session_store import SessionStore  # noqa: E402

ALICE = "alice@example.com"

# Names waitress refuses in start_response (waitress.task.hop_by_hop), plus
# Proxy-Connection, which is the same class of header.
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)

# App-set response headers the audit removes. ``Upgrade`` is included so a
# future WSGI Upgrade response is visible in this scan; WebSocket servers
# outside sharedagents_web are not this package.
_APP_HEADER_LITERAL = re.compile(
    r"""['"](?:Connection|Keep-Alive|Transfer-Encoding|Proxy-Connection|Upgrade)['"]\s*:""",
)


def _no_gateway(*_args, **_kwargs):
    return None


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
    store = SessionStore()
    auth = NativeAuthStore(db)
    monkeypatch.setattr(server, "store", store)
    monkeypatch.setattr(server, "auth_store", auth)
    monkeypatch.setattr(server, "sharing_store", SharingStore(db))
    from sharedagents_web.routes import auth as auth_routes, sessions as sessions_routes

    monkeypatch.setattr(auth_routes, "auth_store", auth)
    monkeypatch.setattr(sessions_routes, "store", store)
    monkeypatch.setattr(sessions_routes, "GW_SESSIONS_DIR", gw)
    monkeypatch.setattr(ui_stream, "store", store)
    monkeypatch.setattr(sessions_routes, "gateway_request", _no_gateway)
    monkeypatch.setattr(server, "gateway_request", _no_gateway)
    monkeypatch.setattr(ui_stream, "gateway_request", _no_gateway)
    auth.create_user(
        ALICE,
        "pw",
        grants=Grants.from_obj({"agents": ["jarvis"], "models": "*"}),
    )
    return tmp_path


def _hop_by_hop_present(header_items) -> list[str]:
    names = set()
    for key, _value in header_items:
        names.add(str(key).lower())
    return sorted(names & _HOP_BY_HOP)


def _wsgi_start_headers(client, path: str):
    """Headers the app passes to ``start_response`` — what waitress checks.

    The body is drained so ``stream_with_context`` exits inside its request
    context. Closing a still-suspended SSE generator raises outside that
    context and shows up as an unraisable exception.
    """
    captured: list[tuple[str, list[tuple[str, str]]]] = []
    original = server.app.wsgi_app

    def wrapper(environ, start_response):
        def spy(status, headers, exc_info=None):
            if environ.get("PATH_INFO", "") == path:
                captured.append((status, list(headers)))
            return start_response(status, headers, exc_info)

        return original(environ, spy)

    server.app.wsgi_app = wrapper
    try:
        resp = client.get(path, buffered=False)
        try:
            status = resp.status_code
            mimetype = (resp.mimetype or "").split(";", 1)[0].strip().lower()
            b"".join(resp.response)
        finally:
            resp.close()
    finally:
        server.app.wsgi_app = original
    assert captured, f"start_response was not called for {path}"
    status_line, headers = captured[-1]
    return status, mimetype, status_line, headers


def test_session_stream_headers_exclude_hop_by_hop(env, monkeypatch):
    """The session SSE route's WSGI headers contain no hop-by-hop name."""
    monkeypatch.setattr(ui_stream, "_STATE_MAX_SECONDS", 0.4)
    sid = server.store.create(ALICE, "jarvis")["id"]
    with server.app.test_client() as client:
        assert (
            client.post(
                "/api/login",
                json={"email": ALICE, "password": "pw"},
            ).status_code
            == 200
        )
        status, mimetype, status_line, headers = _wsgi_start_headers(
            client,
            f"/api/sessions/{sid}/stream",
        )
    assert status == 200, (status, status_line)
    assert mimetype == "text/event-stream", mimetype
    present = _hop_by_hop_present(headers)
    assert present == [], present


def test_sharedagents_web_sets_no_hop_by_hop_response_headers():
    """No module under sharedagents_web assigns Connection / Keep-Alive / etc."""
    root = Path(server.__file__).resolve().parent
    hits = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            if _APP_HEADER_LITERAL.search(line):
                hits.append(f"{path.relative_to(root)}:{lineno}:{line.strip()}")
    assert hits == [], hits


def _cookie_pair(conn):
    body = json.dumps({"email": ALICE, "password": "pw"}).encode()
    conn.request(
        "POST",
        "/api/login",
        body=body,
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    raw = resp.read()
    assert resp.status == 200, (resp.status, raw)
    set_cookie = resp.getheader("Set-Cookie") or ""
    pair = set_cookie.split(";", 1)[0]
    assert pair.startswith("sharedagents_session="), set_cookie
    return pair


def test_session_stream_under_waitress_is_event_stream(env, monkeypatch):
    """GET the real session stream through waitress, not the Flask test client."""
    pytest.importorskip("waitress")
    from waitress import create_server

    # The production mux stays open for minutes. Cap it so this process can
    # drain the chunked body and stop waitress after the response finishes.
    monkeypatch.setattr(ui_stream, "_STATE_MAX_SECONDS", 0.6)
    sid = server.store.create(ALICE, "jarvis")["id"]
    listener = create_server(
        server.app,
        host="127.0.0.1",
        port=0,
        threads=4,
        channel_timeout=30,
        asyncore_loop_timeout=0.1,
    )
    thread = threading.Thread(target=listener.run, name="waitress-sse-hop")
    thread.start()
    port = int(listener.effective_port)
    login = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    stream = None
    try:
        cookie = _cookie_pair(login)
        stream = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
        stream.request(
            "GET",
            f"/api/sessions/{sid}/stream",
            headers={"Cookie": cookie, "Accept": "text/event-stream"},
        )
        resp = stream.getresponse()
        status = resp.status
        ctype = (resp.getheader("Content-Type") or "").split(";", 1)[0].strip().lower()
        if status != 200:
            detail = resp.read(800)
            raise AssertionError(f"status={status} ctype={ctype!r} body={detail!r}")
        assert ctype == "text/event-stream", ctype
        body = resp.read()
        assert b"event: ready" in body, body[:200]
    finally:
        login.close()
        if stream is not None:
            stream.close()
        listener.close()
        thread.join(timeout=10)
    assert not thread.is_alive(), "waitress did not stop"
