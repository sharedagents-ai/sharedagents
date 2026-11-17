"""Default-deny gate for non-admin Web accounts, and conversation access.

A user reaches only the endpoints ``access.USER_ENDPOINTS`` opens, and only
the conversations they own or that are shared with them (404 otherwise).
Admins keep the single-operator reach.
"""

import os

os.environ.setdefault("SHAREDAGENTS_WEB_SECRET", "test-secret-role-gate")
os.environ.setdefault("SHAREDAGENTS_SDLC_NO_RECOVERY", "1")
# The server reads its mount point at import. A shell that exports the live
# one (an agent-hosted shell does) scopes the login cookie to that path, so the
# test client never sends it back and every authenticated request is a 401.
for _k in ("SHAREDAGENTS_WEB_BASE_PATH", "SHAREDAGENTS_WEB_BASE_URL"):
    os.environ.pop(_k, None)

import pytest  # noqa: E402

from sharedagents_core.access_grants import Grants  # noqa: E402
from sharedagents_session.session_reader import encode_line  # noqa: E402
from sharedagents_web import access, server  # noqa: E402
from sharedagents_web.native_auth import ROLE_ADMIN, NativeAuthStore  # noqa: E402
from sharedagents_web.session_sharing import VIS_PUBLIC, SharingStore  # noqa: E402
from sharedagents_web.session_store import SessionStore  # noqa: E402

ADMIN = "admin@example.com"
ALICE = "alice@example.com"
BOB = "bob@example.com"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Fresh stores + gateway sessions dir; nothing touches the process home."""
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
    monkeypatch.setattr(sessions_routes, "GW_SESSIONS_DIR", gw, raising=False)
    # The container has no agent configs (an unknown agent counts as external).
    monkeypatch.setattr(access, "agent_is_external", lambda name: name in {"codex", "claude"})
    server.auth_store.create_user(ADMIN, "pw", role=ROLE_ADMIN)
    server.auth_store.create_user(
        ALICE,
        "pw",
        grants=Grants.from_obj({"agents": ["jarvis"], "models": "*"}),
    )
    server.auth_store.create_user(BOB, "pw")
    return tmp_path


def _gate(email, path, method="GET", json=None, query=None):
    with server.app.test_request_context(path, method=method, json=json, query_string=query):
        from flask import g

        g.user_email = email
        resp = access.gate_request(email)
        if resp is None:
            return None
        return resp.status_code


def _concrete(rule):
    path = rule.rule
    for arg in rule.arguments:
        conv = rule._converters.get(arg)
        value = "1" if type(conv).__name__ == "IntegerConverter" else "x1"
        path = path.replace(f"<{arg}>", value)
        for prefix in ("path", "int", "string"):
            path = path.replace(f"<{prefix}:{arg}>", value)
    return path


def _write_gateway_session(gw, sid, parent=""):
    event = {"type": "session_started", "session_id": sid, "parent_session_id": parent}
    (gw / f"{sid}.jsonl").write_bytes(encode_line(event))


def test_every_unlisted_endpoint_is_admin_only(env):
    checked = 0
    for rule in server.app.url_map.iter_rules():
        if rule.endpoint in access.USER_ENDPOINTS or rule.endpoint == "static":
            continue
        view = server.app.view_functions[rule.endpoint]
        if not getattr(view, "__wrapped__", None):  # public (login, favicon, …)
            continue
        for method in sorted((rule.methods or set()) - {"HEAD", "OPTIONS"}):
            path = _concrete(rule)
            got = _gate(BOB, path, method)
            expected = 302 if (method == "GET" and not path.startswith("/api/")) else 403
            assert got == expected, (rule.endpoint, method, path, got)
            assert _gate(ADMIN, path, method) is None, (rule.endpoint, method)
            checked += 1
    assert checked > 100  # Agent Builder, Channels, Dev, Goals, Loops, Usage, …


def test_admin_only_surfaces_named_in_the_spec(env):
    for path, method in [
        ("/agents", "GET"),
        ("/api/channels", "GET"),
        ("/api/settings/gateway", "POST"),
        ("/api/agents/defs/jarvis", "PATCH"),
        ("/api/loops", "POST"),
        ("/api/goals", "POST"),
        ("/api/chat", "POST"),
        ("/api/topics", "POST"),
        ("/api/usage", "GET"),
        ("/api/agents/monitor", "GET"),
        ("/api/agents/tree", "GET"),
    ]:
        assert _gate(BOB, path, method) in (302, 403), path


def test_session_doors_answer_404_to_strangers(env):
    sid = server.store.create(ALICE, "jarvis")["id"]
    doors = [
        (f"/api/sessions/{sid}", "GET", None, None),
        (f"/api/sessions/{sid}/stream", "GET", None, None),
        (f"/api/sessions/{sid}/events", "GET", None, None),
        (f"/api/sessions/{sid}/prompt", "POST", {"prompt": "hi"}, None),
        (f"/api/sessions/{sid}/queue", "POST", {"text": "hi"}, None),
        (f"/api/sessions/{sid}/cancel", "POST", {}, None),
        (f"/files/{sid}/a.png", "GET", None, None),
        (f"/api/budget/{sid}", "GET", None, None),
        ("/api/agents/footer", "GET", None, {"session": sid}),
        ("/api/agents/abort", "POST", {"session_id": sid}, None),
        ("/api/command", "POST", {"text": "/help", "session_id": sid}, None),
        (f"/api/sessions/{sid}", "DELETE", None, None),
    ]
    for path, method, body, query in doors:
        assert _gate(BOB, path, method, body, query) == 404, path
        assert _gate(ALICE, path, method, body, query) is None, path
        assert _gate(ADMIN, path, method, body, query) is None, path


def test_sharing_opens_a_conversation_but_not_its_management(env):
    server.auth_store.set_grants(BOB, Grants.from_obj({"agents": ["jarvis"], "models": "*"}))
    sid = server.store.create(ALICE, "jarvis")["id"]
    server.sharing_store.set(sid, visibility=VIS_PUBLIC, share_all=False, members=[BOB], by=ALICE)
    assert _gate(BOB, f"/api/sessions/{sid}") is None
    assert _gate(BOB, f"/api/sessions/{sid}/prompt", "POST", {"prompt": "x"}) is None
    assert _gate(BOB, f"/api/sessions/{sid}", "DELETE") == 404
    assert access.session_access(BOB, sid) == access.CHAT
    assert access.session_access(ALICE, sid) == access.MANAGE
    assert access.session_access("carol@example.com", sid) == access.NONE


@pytest.mark.parametrize("operation", ["steer", "prepare"])
def test_queue_steer_routes_allow_chat_users_only(env, operation):
    sid = server.store.create(ALICE, "jarvis")["id"]
    path = f"/api/sessions/{sid}/queue/{operation}"
    body = {"queue_ids": ["q-1"]}
    assert _gate(ALICE, path, "POST", body) is None
    assert _gate(BOB, path, "POST", body) == 404
    server.auth_store.set_grants(BOB, Grants.from_obj({"agents": ["jarvis"], "models": "*"}))
    server.sharing_store.set(sid, visibility=VIS_PUBLIC, share_all=False, members=[BOB], by=ALICE)
    assert _gate(BOB, path, "POST", body) is None
    server.auth_store.set_grants(BOB, Grants.none())
    assert access.session_access(BOB, sid) == access.VIEW
    assert _gate(BOB, path, "POST", body) == 404


def test_gateway_native_sessions_belong_to_admins_unless_shared(env):
    _write_gateway_session(env / "gw", "tui1")
    assert access.session_access(ADMIN, "tui1") == access.MANAGE
    assert access.session_access(ALICE, "tui1") == access.NONE
    # An adopted row keeps the operator's ownership, whoever adopted it.
    server.store.adopt_gateway(ALICE, "tui1", {"title": "t"}, gateway_principal="local")
    assert access.session_access(ALICE, "tui1") == access.NONE
    server.sharing_store.set("tui1", visibility=VIS_PUBLIC, share_all=True, members=[], by=ADMIN)
    assert access.session_access(ALICE, "tui1") == access.CHAT


def test_subagent_sessions_inherit_their_root_conversation(env):
    sid = server.store.create(ALICE, "jarvis")["id"]
    _write_gateway_session(env / "gw", "child1", parent=sid)
    _write_gateway_session(env / "gw", "grandchild1", parent="child1")
    assert access.session_access(ALICE, "grandchild1") == access.MANAGE  # the owner's tree
    assert access.session_access(BOB, "grandchild1") == access.NONE
    assert _gate(ALICE, "/api/sessions/grandchild1/stream") is None
    assert _gate(BOB, "/api/sessions/grandchild1/stream") == 404


def test_admin_sources_and_fallback(env, monkeypatch):
    assert access.admin_source(ADMIN) == "role"
    assert not access.is_admin(ALICE)
    monkeypatch.setenv("SHAREDAGENTS_WEB_ADMINS", ALICE.upper())
    assert access.admin_source(ALICE) == "env"
    monkeypatch.delenv("SHAREDAGENTS_WEB_ADMINS")
    monkeypatch.setenv("SHAREDAGENTS_WEB_LOCAL_OPERATOR_EMAIL", BOB)
    assert access.admin_source(BOB) == "operator"
    monkeypatch.delenv("SHAREDAGENTS_WEB_LOCAL_OPERATOR_EMAIL")
    monkeypatch.setattr(server, "_loopback_debug_bypass_ok", lambda: True)
    assert access.admin_source("debug@localhost") == "debug"
    monkeypatch.setattr(server, "_loopback_debug_bypass_ok", lambda: False)
    # With no admin left, the oldest enabled account is the admin (no lockout).
    server.auth_store.set_disabled(ADMIN, True)
    assert access.admin_source(ADMIN) is None
    assert access.admin_source(ALICE) == "fallback"
    assert not access.is_admin(BOB)


def test_grants_for_admin_is_all_and_disabled_is_none(env):
    assert access.grants_for(ADMIN) == Grants.all()
    assert access.grants_for(ALICE).allows("agents", "jarvis")
    assert access.grants_for(BOB) == Grants.none()
    server.auth_store.set_disabled(ALICE, True)
    assert access.grants_for(ALICE) == Grants.none()


def test_require_auth_runs_the_gate(env):
    client = server.app.test_client()
    assert client.post("/api/login", json={"email": BOB, "password": "pw"}).status_code == 200
    assert client.get("/api/settings/gateway").status_code == 403
    page = client.get("/usage")
    assert page.status_code == 302 and page.headers["Location"].endswith("/")
    admin = server.app.test_client()
    assert admin.post("/api/login", json={"email": ADMIN, "password": "pw"}).status_code == 200
    assert admin.get("/api/settings/gateway").status_code == 200


def test_only_owners_and_admins_decide_tool_permissions(env):
    sid = server.store.create(ALICE, "jarvis")["id"]
    server.sharing_store.set(sid, visibility=VIS_PUBLIC, share_all=False, members=[BOB], by=ALICE)
    body = {"request_id": "r1", "decision": "allow"}
    assert _gate(BOB, f"/api/sessions/{sid}/permission", "POST", body) == 404
    assert _gate(ALICE, f"/api/sessions/{sid}/permission", "POST", body) is None
    assert _gate(ADMIN, f"/api/sessions/{sid}/permission", "POST", body) is None


def test_every_user_endpoint_names_a_registered_route(env):
    registered = {rule.endpoint for rule in server.app.url_map.iter_rules()}
    missing = sorted(set(access.USER_ENDPOINTS) - registered)
    assert missing == []
