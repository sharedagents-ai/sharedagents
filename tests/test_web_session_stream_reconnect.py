"""Gateway-reader death must not kill the browser session SSE mux.

A /cancel often resets the upstream HTTP tail. The old mux emitted ``gw_error``
and broke, so every open tab closed EventSource, waited 1s, and sometimes
auto-downgraded to polling — the tab that did not click Stop then missed
``turn_ended`` and the next ``turn_started``.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import types

os.environ.setdefault("SHAREDAGENTS_WEB_SECRET", "test-secret-stream-reconnect")
os.environ.setdefault("SHAREDAGENTS_SDLC_NO_RECOVERY", "1")
for _k in ("SHAREDAGENTS_WEB_BASE_PATH", "SHAREDAGENTS_WEB_BASE_URL"):
    os.environ.pop(_k, None)

import pytest  # noqa: E402

from sharedagents_session.session_reader import encode_line  # noqa: E402
from sharedagents_web import access, server  # noqa: E402
from sharedagents_web.native_auth import ROLE_ADMIN, NativeAuthStore  # noqa: E402
from sharedagents_web.routes import ui_stream  # noqa: E402
from sharedagents_web.session_sharing import SharingStore  # noqa: E402
from sharedagents_web.session_store import SessionStore  # noqa: E402

ADMIN = "admin@example.com"


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
    monkeypatch.setattr(ui_stream, "store", server.store)
    monkeypatch.setattr(server, "gateway_request", lambda *a, **k: None)
    monkeypatch.setattr(access, "agent_is_external", lambda name: False)
    server.auth_store.create_user(ADMIN, "pw", role=ROLE_ADMIN)
    monkeypatch.setattr(ui_stream, "_STATE_TICK_SECONDS", 0.05)
    monkeypatch.setattr(ui_stream, "_STATE_MAX_SECONDS", 2.5)
    monkeypatch.setattr(ui_stream, "_STATE_HEARTBEAT_SECONDS", 0.2)
    monkeypatch.setattr(ui_stream, "_models_catalogue", lambda owner: {})
    monkeypatch.setattr(
        ui_stream,
        "_state_snapshot",
        lambda owner, sid: {
            "ok": True,
            "session_id": sid,
            "live": {"session_state": "idle"},
        },
    )
    monkeypatch.setattr(
        ui_stream,
        "_state_or_heartbeat",
        lambda payload, sig: (
            ui_stream._emit_frame("state", payload),
            "sig",
        ),
    )
    return tmp_path


def test_polling_redacts_tool_images_without_mutating_the_session_log(env):
    sid = server.store.create(ADMIN, "jarvis")["id"]
    event = {
        "type": "tool_result",
        "call_id": "read-image",
        "content": "image read",
        "images": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}],
    }
    path = server.GW_SESSIONS_DIR / f"{sid}.jsonl"
    raw = encode_line(event)
    path.write_bytes(raw)
    client = server.app.test_client()
    assert client.post("/api/login", json={"email": ADMIN, "password": "pw"}).status_code == 200
    response = client.get(f"/api/sessions/{sid}/events?from=0&timeout=0")
    assert response.status_code == 200
    payload = response.get_json()
    assert len(payload["events"]) == 1
    delivered = payload["events"][0]
    assert "images" not in delivered["event"]
    assert delivered["event"]["content"] == "image read"
    assert delivered["offset"] == len(raw)
    assert path.read_bytes() == raw


class _FailThenTail:
    """First GET dies (cancel / reset); later GETs yield one log frame."""

    def __init__(self) -> None:
        self.calls = 0
        self.lock = threading.Lock()

    def __call__(self, *args, **kwargs):
        with self.lock:
            self.calls += 1
            n = self.calls
        if n == 1:
            raise ConnectionError("upstream reset after cancel")
        resp = _FakeSse()
        return resp


class _FakeSse:
    status_code = 200

    def iter_lines(self, chunk_size=64):
        yield b"event: log"
        yield (
            b'data: {"offset": 99, "event": {"type": "turn_started",'
            b' "physical_turn_id": "new-turn", "user": "continue"}}'
        )
        yield b""
        time.sleep(0.4)

    def close(self) -> None:
        return None


def _install_requests(monkeypatch, getter):
    """``_read_gateway`` imports requests locally — inject the module."""
    mod = types.ModuleType("requests")
    mod.get = getter
    monkeypatch.setitem(sys.modules, "requests", mod)
    return getter


def test_gateway_reader_error_reattaches_without_gw_error(env, monkeypatch):
    sid = server.store.create(ADMIN, "jarvis")["id"]
    fake = _FailThenTail()
    _install_requests(monkeypatch, fake)
    kinds: list[str] = []
    saw_continue = False
    with server.app.test_request_context("/"):
        stream = ui_stream.iter_session_stream(
            ADMIN,
            sid,
            "http://127.0.0.1:9/events?from=0",
            {},
            1.0,
        )
        deadline = time.monotonic() + 2.4
        try:
            for frame in stream:
                raw = bytes(frame)
                if raw.startswith(b"event: "):
                    kinds.append(raw.split(b"\n", 1)[0].decode("ascii", "replace"))
                if b"continue" in raw and b"turn_started" in raw:
                    saw_continue = True
                    break
                if time.monotonic() > deadline:
                    break
        finally:
            stream.close()
    assert "event: gw_error" not in kinds, kinds
    assert fake.calls >= 2, f"reader must retry after the first reset, calls={fake.calls}"
    assert saw_continue, f"reattached tail must forward the next turn, kinds={kinds}"


def test_gateway_reader_error_keeps_mux_emitting_state(env, monkeypatch):
    sid = server.store.create(ADMIN, "jarvis")["id"]
    fake = _FailThenTail()
    _install_requests(monkeypatch, fake)
    kinds: list[str] = []
    with server.app.test_request_context("/"):
        stream = ui_stream.iter_session_stream(
            ADMIN,
            sid,
            "http://127.0.0.1:9/events?from=0",
            {},
            1.0,
        )
        deadline = time.monotonic() + 1.5
        try:
            for frame in stream:
                raw = bytes(frame)
                if raw.startswith(b"event: "):
                    kinds.append(raw.split(b"\n", 1)[0].decode("ascii", "replace"))
                if kinds.count("event: state") + kinds.count("event: hb") >= 3:
                    break
                if time.monotonic() > deadline:
                    break
        finally:
            stream.close()
    assert kinds[0] == "event: ready"
    assert "event: gw_error" not in kinds
    assert any(k in {"event: state", "event: hb"} for k in kinds[1:]), kinds


class _SucceedingTail:
    def __init__(self) -> None:
        self.calls = 0
        self.lock = threading.Lock()

    def __call__(self, *args, **kwargs):
        with self.lock:
            self.calls += 1
        return _FakeSse()


def test_placeholder_mux_attaches_when_session_log_appears(env, monkeypatch):
    """Observer EventSource opened before the first prompt must pick up the JSONL."""
    sid = server.store.create(ADMIN, "jarvis")["id"]
    fake = _SucceedingTail()
    _install_requests(monkeypatch, fake)
    kinds: list[str] = []
    saw_continue = False
    wrote = False
    log_path = server.GW_SESSIONS_DIR / f"{sid}.jsonl"
    assert not log_path.exists()
    with server.app.test_request_context("/"):
        stream = ui_stream.iter_session_stream(
            ADMIN,
            sid,
            None,
            {},
            1.0,
            pending_gw_url="http://127.0.0.1:9/events?from=0",
        )
        deadline = time.monotonic() + 2.4
        try:
            for frame in stream:
                raw = bytes(frame)
                if raw.startswith(b"event: "):
                    kinds.append(raw.split(b"\n", 1)[0].decode("ascii", "replace"))
                if not wrote:
                    log_path.write_text("{}\n", encoding="utf-8")
                    wrote = True
                if b"continue" in raw and b"turn_started" in raw:
                    saw_continue = True
                    break
                if time.monotonic() > deadline:
                    break
        finally:
            stream.close()
    assert "event: gw_error" not in kinds, kinds
    assert fake.calls >= 1, f"placeholder mux must GET the gateway once the log exists, calls={fake.calls}"
    assert saw_continue, f"observer mux must forward the first turn, kinds={kinds}"


def test_stream_never_builds_catalogues_on_connect_or_ticks(env, monkeypatch):
    calls = []
    monkeypatch.setattr(ui_stream, "_models_catalogue", lambda *a: calls.append(a) or {})
    monkeypatch.setattr(ui_stream, "_STATE_MAX_SECONDS", 0.35)
    with server.app.test_request_context("/"):
        stream = ui_stream.iter_session_stream(ADMIN, "", "", {}, 1.0)
        ready = next(stream)
        ui_stream.broadcast_models_event("jarvis", [{"id": "unopened-provider-model"}])
        frames = [ready, *stream]
    assert calls == []
    assert not any(b"event: models" in frame for frame in frames)


def test_polling_bootstrap_never_builds_catalogues(env, monkeypatch):
    calls = []
    monkeypatch.setattr(ui_stream, "_models_catalogue", lambda *a: calls.append(a) or {})
    sid = server.store.create(ADMIN, "jarvis")["id"]
    client = server.app.test_client()
    client.post("/api/login", json={"email": ADMIN, "password": "pw"})
    for query in ("catalogue=1", "catalogue_rev=old"):
        payload = client.get(f"/api/sessions/{sid}/events?timeout=0&{query}").get_json()
        assert payload["ok"] is True
        assert payload["models_catalogue"] is None
    assert calls == []
