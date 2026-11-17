"""Conversations a non-admin sees: their own plus what is shared with them.

Admins keep the single-operator view (every Web row + gateway-native roots).
"""

import os

os.environ.setdefault("SHAREDAGENTS_WEB_SECRET", "test-secret-visibility")
os.environ.setdefault("SHAREDAGENTS_SDLC_NO_RECOVERY", "1")
# The server reads its mount point at import. A shell that exports the live
# one (an agent-hosted shell does) scopes the login cookie to that path, so the
# test client never sends it back and every authenticated request is a 401.
for _k in ("SHAREDAGENTS_WEB_BASE_PATH", "SHAREDAGENTS_WEB_BASE_URL"):
    os.environ.pop(_k, None)

import pytest  # noqa: E402

from sharedagents_core.access_grants import Grants  # noqa: E402
from sharedagents_web import access, server  # noqa: E402
from sharedagents_web.approvals_map import map_gateway_approvals  # noqa: E402
from sharedagents_web.native_auth import ROLE_ADMIN, NativeAuthStore  # noqa: E402
from sharedagents_web.routes import ui_stream  # noqa: E402
from sharedagents_web.session_sharing import VIS_PRIVATE, VIS_PUBLIC, SharingStore  # noqa: E402
from sharedagents_web.session_store import SessionStore  # noqa: E402

ADMIN = "admin@example.com"
ALICE = "alice@example.com"
BOB = "bob@example.com"


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
    from sharedagents_web.routes import auth as auth_routes, presence as presence_routes, sessions as sessions_routes

    monkeypatch.setattr(auth_routes, "auth_store", server.auth_store)
    monkeypatch.setattr(sessions_routes, "store", server.store)
    monkeypatch.setattr(sessions_routes, "GW_SESSIONS_DIR", gw)
    monkeypatch.setattr(presence_routes, "store", server.store)
    monkeypatch.setattr(ui_stream, "store", server.store)
    # No gateway: every proxied read fails soft.
    monkeypatch.setattr(sessions_routes, "gateway_request", lambda *a, **k: None)
    monkeypatch.setattr(server, "gateway_request", lambda *a, **k: None)
    # The container has no agent configs; name the external engines.
    monkeypatch.setattr(access, "agent_is_external", lambda name: name in {"codex", "claude"})
    server.auth_store.create_user(ADMIN, "pw", role=ROLE_ADMIN)
    server.auth_store.create_user(
        ALICE,
        "pw",
        grants=Grants.from_obj({"agents": ["jarvis"], "models": "*"}),
    )
    server.auth_store.create_user(BOB, "pw")
    return tmp_path


def _login(email):
    client = server.app.test_client()
    assert client.post("/api/login", json={"email": email, "password": "pw"}).status_code == 200
    return client


def _ids(rows):
    return {r["id"] for r in rows}


def test_user_sidebar_is_own_plus_shared_and_not_starved_by_newer_rows(env, monkeypatch):
    mine = server.store.create(ALICE, "jarvis")["id"]
    shared = server.store.create(ADMIN, "jarvis")["id"]
    server.sharing_store.set(shared, visibility=VIS_PUBLIC, share_all=False, members=[ALICE], by=ADMIN)
    newer = [server.store.create(ADMIN, "jarvis")["id"] for _ in range(25)]
    monkeypatch.setattr(
        server,
        "gw_root_sessions",
        lambda limit=50: [
            {"hash": "tui1", "title": "TUI work", "agent": "jarvis", "updated_at": "2099-01-01T00:00:00+00:00"},
        ],
    )
    rows = server._merged_sessions(ALICE, limit=20)
    assert _ids(rows) == {mine, shared}
    admin_rows = _ids(server._merged_sessions(ADMIN, limit=200))
    assert {mine, shared, "tui1", *newer} <= admin_rows


def test_shared_gateway_native_conversation_is_listed_for_its_member(env, monkeypatch):
    monkeypatch.setattr(
        server,
        "gw_root_sessions",
        lambda limit=50: [
            {"hash": "tui1", "title": "TUI work", "agent": "jarvis", "updated_at": "2099-01-01T00:00:00+00:00"},
        ],
    )
    assert "tui1" not in _ids(server._merged_sessions(ALICE))
    server.sharing_store.set("tui1", visibility=VIS_PUBLIC, share_all=True, members=[], by=ADMIN)
    rows = {r["id"]: r for r in server._merged_sessions(ALICE)}
    assert rows["tui1"]["title"] == "TUI work"
    # Shared but not among the gateway's newest roots: still listed.
    server.sharing_store.set("old-tui", visibility=VIS_PUBLIC, share_all=True, members=[], by=ADMIN)
    assert "old-tui" in _ids(server._merged_sessions(ALICE))


def test_boot_hides_a_foreign_conversation_and_reports_access(env):
    foreign = server.store.create(ADMIN, "jarvis")["id"]
    alice = _login(ALICE)
    body = alice.get(f"/api/boot?session={foreign}").get_json()
    assert body["session"] is None
    assert body["viewer"] == {"email": ALICE, "name": "alice", "role": "user", "can_chat": True}
    server.sharing_store.set(foreign, visibility=VIS_PUBLIC, share_all=False, members=[ALICE], by=ADMIN)
    session = alice.get(f"/api/boot?session={foreign}").get_json()["session"]
    assert session["access"] == "chat"
    assert session["sharing"]["visibility"] == VIS_PUBLIC
    assert session["owner_email"] == ADMIN
    assert not session.get("monitor_read_only")
    mine = server.store.create(ALICE, "jarvis")["id"]
    own = alice.get(f"/api/boot?session={mine}").get_json()["session"]
    assert own["access"] == "manage" and own["sharing"]["visibility"] == VIS_PRIVATE


def test_shared_conversation_on_an_ungranted_agent_is_view_only(env):
    sid = server.store.create(ADMIN, "codex")["id"]
    server.sharing_store.set(sid, visibility=VIS_PUBLIC, share_all=True, members=[], by=ADMIN)
    session = _login(ALICE).get(f"/api/sessions/{sid}").get_json()["session"]
    assert session["access"] == "view"
    assert session["monitor_read_only"] is True
    assert "codex" in session["read_only_reason"]
    # The same conversation is fully usable by an admin.
    admin_view = _login(ADMIN).get(f"/api/sessions/{sid}").get_json()["session"]
    assert admin_view["access"] == "manage" and not admin_view.get("monitor_read_only")


def test_view_only_shared_member_cannot_answer_or_cancel(env):
    sid = server.store.create(ADMIN, "codex")["id"]
    server.sharing_store.set(sid, visibility=VIS_PUBLIC, share_all=True, members=[], by=ADMIN)
    alice = _login(ALICE)
    assert alice.post(f"/api/sessions/{sid}/answer", json={"answers": {}}).status_code == 404
    assert alice.post(f"/api/sessions/{sid}/cancel").status_code == 404
    assert access.session_access(ALICE, sid) == access.VIEW


def test_only_admins_may_adopt_gateway_sessions(env):
    from sharedagents_web.principal import is_configured_local_operator, may_adopt_gateway_session

    assert may_adopt_gateway_session(ADMIN) is True
    assert may_adopt_gateway_session(ALICE) is False
    assert is_configured_local_operator(ADMIN) is True
    assert is_configured_local_operator(ALICE) is False


def test_approvals_only_for_reachable_conversations(env):
    mine = server.store.create(ALICE, "jarvis")["id"]
    theirs = server.store.create(ADMIN, "jarvis")["id"]
    raw = [{"id": "a1", "session_hash": mine}, {"id": "a2", "session_hash": theirs}]
    assert [a["id"] for a in map_gateway_approvals(raw, server.store, ALICE)] == ["a1"]
    got = map_gateway_approvals(raw, server.store, ALICE, extra_session_ids=(theirs,))
    assert [a["id"] for a in got] == ["a1"]
    assert {a["id"] for a in map_gateway_approvals(raw, server.store, ADMIN)} == {"a1", "a2"}
    # Shared with Alice, but tool approvals stay with the owner and admins.
    server.sharing_store.set(theirs, visibility=VIS_PUBLIC, share_all=True, members=[], by=ADMIN)
    assert [a["id"] for a in map_gateway_approvals(raw, server.store, ALICE)] == ["a1"]


def test_new_chat_placeholders_belong_to_their_person(env):
    admin_row, _ = server.store.reserve_web_chat(ADMIN, "jarvis")
    alice_row, discarded = server.store.reserve_web_chat(ALICE, "jarvis")
    assert alice_row["id"] != admin_row["id"]
    assert alice_row["owner"] == ALICE
    assert admin_row["id"] not in discarded
    assert server.store.get(admin_row["id"]) is not None
    again, _ = server.store.reserve_web_chat(ALICE, "jarvis")
    assert again["id"] == alice_row["id"]


def test_deleting_a_conversation_revokes_its_sharing(env):
    sid = server.store.create(ALICE, "jarvis")["id"]
    server.sharing_store.set(sid, visibility=VIS_PUBLIC, share_all=True, members=[], by=ALICE)
    r = _login(ALICE).delete(f"/api/sessions/{sid}?confirm=1")
    assert r.status_code == 200, r.get_data(as_text=True)
    assert server.sharing_store.get(sid).visibility == VIS_PRIVATE


def test_continue_here_never_offers_an_unreachable_conversation(env):
    foreign = server.store.create(ADMIN, "jarvis")["id"]
    server.web_state.upsert_presence("dev-a", ALICE, device_label="phone", active_sid=foreign, streaming=True)
    alice = _login(ALICE)
    body = alice.get("/api/presence/continue-here?device_id=dev-b").get_json()
    assert body["continue"] is None
    server.sharing_store.set(foreign, visibility=VIS_PUBLIC, share_all=True, members=[], by=ADMIN)
    body = alice.get("/api/presence/continue-here?device_id=dev-b").get_json()
    assert body["continue"]["sid"] == foreign


def test_stream_ticks_reauthorize_every_time(env):
    sid = server.store.create(ADMIN, "jarvis")["id"]
    with server.app.test_request_context("/"):
        assert ui_stream._session_hash_for(ALICE, sid) == ""
        server.sharing_store.set(sid, visibility=VIS_PUBLIC, share_all=True, members=[], by=ADMIN)
        assert ui_stream._session_hash_for(ALICE, sid) == sid
        server.sharing_store.set(sid, visibility=VIS_PRIVATE, share_all=False, members=[], by=ADMIN)
        assert ui_stream._session_hash_for(ALICE, sid) == ""
        assert ui_stream._session_hash_for(ADMIN, sid) == sid


def test_an_open_stream_ends_when_its_share_is_revoked(env, monkeypatch):
    import time as _time

    sid = server.store.create(ADMIN, "jarvis")["id"]
    server.sharing_store.set(sid, visibility=VIS_PUBLIC, share_all=True, members=[], by=ADMIN)
    monkeypatch.setattr(ui_stream, "_STATE_TICK_SECONDS", 0.05)
    with server.app.test_request_context("/"):
        stream = ui_stream.iter_session_stream(ALICE, sid, None, {}, 5.0)
        assert next(stream).startswith(b"event: ready")
        server.sharing_store.set(sid, visibility=VIS_PRIVATE, share_all=False, members=[], by=ADMIN)
        deadline = _time.monotonic() + 5
        frames = []
        for frame in stream:
            frames.append(frame)
            if frame.startswith(b"event: reconnect") or _time.monotonic() > deadline:
                break
        stream.close()
    assert frames and frames[-1].startswith(b"event: reconnect")


def test_users_only_see_topics_of_their_conversations(env, monkeypatch):
    from sharedagents_web.routes import topics as topics_routes

    monkeypatch.setattr(topics_routes, "all_canonical_names", lambda registry: ["Secret project"])
    monkeypatch.setattr(server, "load_web_topics", lambda: ["Board memo"])
    server.store.create(ALICE, "jarvis", topic="Mine")
    names = [t["topic"] for t in _login(ALICE).get("/api/topics").get_json()["topics"]]
    assert "Mine" in names and "Secret project" not in names and "Board memo" not in names
    admin_names = [t["topic"] for t in _login(ADMIN).get("/api/topics").get_json()["topics"]]
    assert "Secret project" in admin_names


def test_an_open_stream_ends_when_the_account_is_disabled(env, monkeypatch):
    import time as _time

    sid = server.store.create(ALICE, "jarvis")["id"]
    monkeypatch.setattr(ui_stream, "_STATE_TICK_SECONDS", 0.05)
    with server.app.test_request_context("/"):
        stream = ui_stream.iter_session_stream(ALICE, sid, None, {}, 5.0)
        assert next(stream).startswith(b"event: ready")
        server.auth_store.set_disabled(ALICE, True)
        deadline = _time.monotonic() + 5
        frames = []
        for frame in stream:
            frames.append(frame)
            if frame.startswith(b"event: reconnect") or _time.monotonic() > deadline:
                break
        stream.close()
    assert frames and frames[-1].startswith(b"event: reconnect")


def _read_stream(resp, seconds):
    import time as _time

    frames, end = [], _time.monotonic() + seconds
    for frame in resp.response:
        frames.append(bytes(frame))
        if frame.startswith(b"event: reconnect") or _time.monotonic() > end:
            break
    resp.close()
    return frames


def test_a_constantly_waking_stream_still_ends_on_sign_out(env, monkeypatch):
    # Codex round 2: every wake re-armed the tick that re-checked the login,
    # so a busy conversation kept a revoked login's stream open.
    import itertools

    wakes = itertools.count(1)
    monkeypatch.setattr(ui_stream, "take_state_wake", lambda sid: next(wakes))
    monkeypatch.setattr(ui_stream, "_STATE_TICK_SECONDS", 0.05)
    sid = server.store.create(ALICE, "jarvis")["id"]
    client = _login(ALICE)
    resp = client.get(f"/api/sessions/{sid}/stream", buffered=False)
    assert resp.status_code == 200
    first = next(iter(resp.response))
    assert first.startswith(b"event: ready")
    server.auth_store.revoke_all_tokens(ALICE)  # a password reset signs her out
    frames = _read_stream(resp, 5)
    assert frames and frames[-1].startswith(b"event: reconnect")


def test_an_external_auth_login_keeps_its_stream(env, monkeypatch):
    # The dashboard's bearer is not a native token: re-checking it against the
    # native store closed every external-auth stream on its first tick.
    monkeypatch.setenv("SHAREDAGENTS_EXTERNAL_AUTH", "1")
    monkeypatch.setattr(server, "verify_dashboard_auth", lambda: (True, {"email": ALICE, "name": "Alice"}))
    monkeypatch.setattr(ui_stream, "_STATE_TICK_SECONDS", 0.05)
    sid = server.store.create(ALICE, "jarvis")["id"]
    resp = server.app.test_client().get(
        f"/api/sessions/{sid}/stream",
        headers={"Authorization": "Bearer dashboard-token"},
        buffered=False,
    )
    assert resp.status_code == 200
    frames = _read_stream(resp, 1.6)
    assert len(frames) >= 3
    assert not any(f.startswith(b"event: reconnect") for f in frames)


def test_a_revoked_dashboard_login_ends_its_stream(env, monkeypatch):
    # Codex round 3: the dashboard credential itself is re-verified.
    verdict = {"ok": True}
    monkeypatch.setenv("SHAREDAGENTS_EXTERNAL_AUTH", "1")
    monkeypatch.setattr(
        server,
        "verify_dashboard_auth",
        lambda: (True, {"email": ALICE, "name": "Alice"}) if verdict["ok"] else (False, "Auth failed: 401"),
    )
    monkeypatch.setattr(ui_stream, "_STATE_TICK_SECONDS", 0.05)
    monkeypatch.setattr(ui_stream, "_DASHBOARD_RECHECK_SECONDS", 0.05, raising=False)
    sid = server.store.create(ALICE, "jarvis")["id"]
    resp = server.app.test_client().get(
        f"/api/sessions/{sid}/stream",
        headers={"Authorization": "Bearer dashboard-token"},
        buffered=False,
    )
    assert resp.status_code == 200
    assert next(iter(resp.response)).startswith(b"event: ready")
    verdict["ok"] = False  # revoked (or expired) at the dashboard
    frames = _read_stream(resp, 5)
    assert frames and frames[-1].startswith(b"event: reconnect")


def test_refusals_read_as_sentences_for_people(env):
    # The "[access]" tag is for logs and models; the browser shows people a sentence.
    sid = server.store.create(ADMIN, "codex")["id"]
    server.sharing_store.set(sid, visibility=VIS_PUBLIC, share_all=True, members=[], by=ADMIN)
    session = _login(ALICE).get(f"/api/sessions/{sid}").get_json()["session"]
    reason = session["read_only_reason"]
    assert reason.startswith("View only — ") and "[access]" not in reason
    assert "to you" in reason or "you" in reason
    mine = server.store.create(ALICE, "jarvis")["id"]
    r = _login(ALICE).post(f"/api/sessions/{mine}/agent", json={"target_agent": "codex"})
    assert r.status_code == 403
    body = r.get_json()
    assert "[access]" not in (body.get("error") or "") and "[access]" not in (body.get("message") or "")


def test_boot_says_when_a_person_has_no_agent_to_talk_to(env):
    bob_boot = _login(BOB).get("/api/boot").get_json()
    assert bob_boot["viewer"]["can_chat"] is False
    assert "admin" in bob_boot["viewer"]["chat_block_reason"].lower()
    alice_boot = _login(ALICE).get("/api/boot").get_json()
    assert alice_boot["viewer"]["can_chat"] is True
    assert _login(ADMIN).get("/api/boot").get_json()["viewer"]["can_chat"] is True


class _Resp:
    status_code = 200

    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


def test_switch_on_a_turnless_chat_the_gateway_already_holds_reaches_the_gateway(env, monkeypatch):
    """A task's chat exists on the gateway (pinned title) before its first turn.

    Recording the switch only as web intent left the gateway on the old agent,
    so the picker's follow-up ``/model <new agent's model>`` was checked
    against the old agent and failed with "unknown model".
    """
    from sharedagents_web.routes import sessions as sessions_routes

    sid = server.store.create(ADMIN, "jarvis")["id"]
    (env / "gw" / f"{sid}.jsonl").write_text(
        'h {"agent_name":"jarvis","engine":"local_llm","title":"databento bars"}\n'
    )
    calls = []

    def fake_gateway(method, path, **kw):
        calls.append((method, path, kw.get("json")))
        return _Resp({"active_agent": "codex", "revision": 1})

    monkeypatch.setattr(sessions_routes, "gateway_request", fake_gateway)
    r = _login(ADMIN).post(f"/api/sessions/{sid}/agent", json={"target_agent": "codex"})
    assert r.status_code == 200, r.get_json()
    assert [c[1] for c in calls] == [f"/control/v1/orchestrator/sessions/{sid}/agent"]
    assert calls[0][2]["target_agent"] == "codex"
    row = server.store.get(sid, ADMIN)
    assert row["agent"] == "codex" and not row.get("pending_agent")


def test_switch_on_a_new_chat_without_gateway_log_stays_web_intent(env, monkeypatch):
    from sharedagents_web.routes import sessions as sessions_routes

    sid = server.store.create(ADMIN, "jarvis")["id"]
    calls = []
    monkeypatch.setattr(
        sessions_routes,
        "gateway_request",
        lambda *a, **k: calls.append(a) or None,
    )
    r = _login(ADMIN).post(f"/api/sessions/{sid}/agent", json={"target_agent": "codex"})
    assert r.status_code == 200, r.get_json()
    assert calls == []
    assert server.store.get(sid, ADMIN)["pending_agent"] == "codex"


@pytest.mark.parametrize(
    "selection",
    [
        {"target_mode": "auto"},
        {"target_model": "deepseek-flash"},
    ],
)
def test_turnless_gateway_chat_accepts_selection_without_agent(env, monkeypatch, selection):
    from sharedagents_web.routes import sessions as sessions_routes

    sid = server.store.create(ADMIN, "jarvis")["id"]
    (env / "gw" / f"{sid}.jsonl").write_text('h {"agent_name":"jarvis","engine":"local_llm","title":"Task"}\n')
    calls = []

    def fake_gateway(method, path, **kw):
        payload = kw["json"]
        calls.append(payload)
        # The gateway switch contract requires an explicit agent, even when
        # the web caller only changes a model or mode.
        assert payload["target_agent"] == "jarvis"
        return _Resp({"active_agent": "jarvis", "revision": 1})

    monkeypatch.setattr(sessions_routes, "gateway_request", fake_gateway)
    response = _login(ADMIN).post(f"/api/sessions/{sid}/agent", json=selection)
    assert response.status_code == 200, response.get_json()
    assert len(calls) == 1
    for key, value in selection.items():
        assert calls[0][key] == value


@pytest.mark.parametrize("gateway_bound", [False, True])
def test_first_prompt_after_switch_uses_selected_agent(env, monkeypatch, gateway_bound):
    from sharedagents_web.routes import sessions as sessions_routes

    row = server.store.create(ADMIN, "jarvis")
    sid = row["id"]
    row["pending_mode"] = "ask"
    assert server.store._save(row)
    if gateway_bound:
        (env / "gw" / f"{sid}.jsonl").write_text('h {"agent_name":"jarvis","engine":"local_llm","title":"Task"}\n')
    calls = []

    def fake_gateway(method, path, **kw):
        calls.append((path, kw["json"]))
        if path.endswith("/agent"):
            return _Resp({"active_agent": "codex", "revision": 1})
        response = _Resp({"accepted": True, "hash": sid})
        response.status_code = 202
        return response

    monkeypatch.setattr(sessions_routes, "gateway_request", fake_gateway)
    client = _login(ADMIN)
    assert (
        client.post(
            f"/api/sessions/{sid}/agent",
            json={
                "target_agent": "codex",
                "target_mode": "ask",
            },
        ).status_code
        == 200
    )
    assert client.post(f"/api/sessions/{sid}/prompt", json={"prompt": "hello"}).status_code == 202
    path, body = calls[-1]
    assert path == f"/control/v1/orchestrator/sessions/{sid}/prompt"
    assert body["initial_agent"] == "codex"
    assert "permission_mode" not in body
    assert len(calls) == (2 if gateway_bound else 1)
