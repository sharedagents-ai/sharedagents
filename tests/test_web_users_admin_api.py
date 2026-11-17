"""The admin-only Users tab: accounts, roles, grants, and the user directory."""

import os
from pathlib import Path

os.environ.setdefault("SHAREDAGENTS_WEB_SECRET", "test-secret-users")
os.environ.setdefault("SHAREDAGENTS_SDLC_NO_RECOVERY", "1")
# The server reads its mount point at import. A shell that exports the live
# one (an agent-hosted shell does) scopes the login cookie to that path, so the
# test client never sends it back and every authenticated request is a 401.
for _k in ("SHAREDAGENTS_WEB_BASE_PATH", "SHAREDAGENTS_WEB_BASE_URL"):
    os.environ.pop(_k, None)

import pytest  # noqa: E402

from sharedagents_core.access_grants import Grants  # noqa: E402
from sharedagents_web import server  # noqa: E402
from sharedagents_web.native_auth import ROLE_ADMIN, NativeAuthStore  # noqa: E402
from sharedagents_web.session_sharing import SharingStore  # noqa: E402
from sharedagents_web.session_store import SessionStore  # noqa: E402

ADMIN = "admin@example.com"
ALICE = "alice@example.com"
CATALOGUE = {
    "agents": [{"name": "jarvis", "external": False, "conversational": True, "engine": "local_llm"}],
    "models": [{"id": "m1", "alias": "one", "display": "Model one", "provider": "local"}],
    "tools": [{"name": "read", "description": "Read a file", "group": "Files"}],
    "skills": [{"name": "pdf", "description": "PDF tools"}],
}


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
    monkeypatch.setattr(server, "SESSION_DB_PATH", db)
    monkeypatch.setattr(server, "_SESSIONS_DIR_DEPRECATED", tmp_path / "legacy-missing")
    monkeypatch.setattr(server, "_GW_SESSIONS_DEPRECATED", tmp_path / "gw-missing")
    monkeypatch.setattr(server, "store", SessionStore())
    monkeypatch.setattr(server, "auth_store", NativeAuthStore(db))
    monkeypatch.setattr(server, "sharing_store", SharingStore(db))
    from sharedagents_web import user_catalog
    from sharedagents_web.routes import auth as auth_routes

    monkeypatch.setattr(auth_routes, "auth_store", server.auth_store)
    monkeypatch.setattr(user_catalog, "catalog", lambda: CATALOGUE)
    server.auth_store.create_user(ADMIN, "pw", role=ROLE_ADMIN, display_name="Admin")
    server.auth_store.create_user(ALICE, "pw", display_name="Alice")
    return tmp_path


def _login(email, password="pw"):
    client = server.app.test_client()
    r = client.post("/api/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.get_data(as_text=True)
    return client


def test_users_page_and_api_are_admin_only(env):
    alice = _login(ALICE)
    assert alice.get("/users").status_code == 302
    for method, path in [
        ("get", "/api/admin/users"),
        ("get", "/api/admin/catalog"),
        ("post", "/api/admin/users"),
        ("patch", f"/api/admin/users/{ADMIN}"),
        ("post", f"/api/admin/users/{ADMIN}/password"),
    ]:
        assert getattr(alice, method)(path, json={}).status_code == 403, path
    admin = _login(ADMIN)
    page = admin.get("/users")
    assert page.status_code == 200 and b"users.js" in page.data


def test_listing_shows_roles_sources_and_grants(env, monkeypatch):
    monkeypatch.setenv("SHAREDAGENTS_WEB_ADMINS", "boss@example.com")
    server.auth_store.create_user("boss@example.com", "pw")
    rows = {u["email"]: u for u in _login(ADMIN).get("/api/admin/users").get_json()["users"]}
    assert rows[ADMIN]["role"] == "admin" and rows[ADMIN]["admin_source"] == "role"
    assert rows[ADMIN]["is_self"] is True
    assert rows["boss@example.com"]["locked"] is True
    assert rows[ALICE]["role"] == "user" and rows[ALICE]["grants"] == Grants.none().to_obj()
    assert rows[ALICE]["display_name"] == "Alice"
    assert "password_hash" not in rows[ALICE]


def test_admin_creates_a_user_with_no_grants_by_default(env):
    admin = _login(ADMIN)
    r = admin.post(
        "/api/admin/users",
        json={
            "email": " Bob@Example.com ",
            "display_name": "Bob",
            "password": "s3cret-pass",
        },
    )
    assert r.status_code == 201, r.get_data(as_text=True)
    user = r.get_json()["user"]
    assert user["email"] == "bob@example.com" and user["role"] == "user"
    assert user["grants"] == Grants.none().to_obj()
    assert _login("bob@example.com", "s3cret-pass")
    assert admin.post("/api/admin/users", json={"email": "bob@example.com", "password": "x" * 8}).status_code == 409
    for bad in (
        {"email": "no-at-sign", "password": "x" * 8},
        {"email": "c@example.com"},
        {"email": "c@example.com", "password": "short"},
        {"email": "c@example.com", "password": "x" * 8, "role": "root"},
    ):
        assert admin.post("/api/admin/users", json=bad).status_code == 400, bad


def test_admin_edits_role_grants_name_and_status(env):
    admin = _login(ADMIN)
    grants = {"agents": ["jarvis"], "models": "*", "tools": ["read"], "skills": []}
    r = admin.patch(f"/api/admin/users/{ALICE}", json={"grants": grants, "display_name": "Alice L."})
    assert r.status_code == 200, r.get_data(as_text=True)
    assert server.auth_store.account(ALICE)["grants"] == Grants.from_obj(grants)
    assert server.auth_store.display_name(ALICE) == "Alice L."
    alice = _login(ALICE)
    assert alice.get("/api/boot").status_code == 200
    assert admin.patch(f"/api/admin/users/{ALICE}", json={"disabled": True}).status_code == 200
    assert alice.get("/api/boot").status_code == 401  # every login revoked
    assert admin.patch(f"/api/admin/users/{ALICE}", json={"disabled": False, "role": "admin"}).status_code == 200
    assert server.auth_store.account(ALICE)["role"] == "admin"
    assert admin.patch("/api/admin/users/nobody@example.com", json={"role": "user"}).status_code == 404
    assert admin.patch(f"/api/admin/users/{ALICE}", json={"role": "god"}).status_code == 400


def test_the_last_admin_cannot_be_removed(env):
    admin = _login(ADMIN)
    r = admin.patch(f"/api/admin/users/{ADMIN}", json={"role": "user"})
    assert r.status_code == 409 and "admin" in r.get_json()["error"]
    assert admin.patch(f"/api/admin/users/{ADMIN}", json={"disabled": True}).status_code == 409
    admin.patch(f"/api/admin/users/{ALICE}", json={"role": "admin"})
    assert admin.patch(f"/api/admin/users/{ADMIN}", json={"role": "user"}).status_code == 200


def test_env_and_operator_admins_are_locked(env, monkeypatch):
    monkeypatch.setenv("SHAREDAGENTS_WEB_LOCAL_OPERATOR_EMAIL", ALICE)
    r = _login(ADMIN).patch(f"/api/admin/users/{ALICE}", json={"role": "user"})
    assert r.status_code == 409 and "SHAREDAGENTS_WEB_LOCAL_OPERATOR_EMAIL" in r.get_json()["error"]


def test_password_reset_signs_the_user_out(env):
    alice = _login(ALICE)
    admin = _login(ADMIN)
    assert admin.post(f"/api/admin/users/{ALICE}/password", json={"password": "short"}).status_code == 400
    assert admin.post(f"/api/admin/users/{ALICE}/password", json={"password": "new-password-1"}).status_code == 200
    assert alice.get("/api/boot").status_code == 401
    assert _login(ALICE, "new-password-1")


def test_admin_writes_refuse_cross_site_and_non_json(env):
    admin = _login(ADMIN)
    r = admin.patch(f"/api/admin/users/{ALICE}", json={"role": "admin"}, headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403
    r = admin.patch(f"/api/admin/users/{ALICE}", data="role=admin", content_type="application/x-www-form-urlencoded")
    assert r.status_code == 415
    assert server.auth_store.account(ALICE)["role"] == "user"


def test_admin_grants_non_overlapping_folders_and_rejects_bad_paths(env, tmp_path, monkeypatch):
    monkeypatch.setenv("SHAREDAGENTS_FOLDER_GRANT_ROOTS", str(tmp_path))
    allowed = tmp_path / "proj"
    allowed.mkdir()
    admin = _login(ADMIN)
    r = admin.patch(
        f"/api/admin/users/{ALICE}",
        json={
            "grants": {
                "agents": ["jarvis"],
                "models": "*",
                "tools": ["bash"],
                "skills": [],
                "folders": [{"path": str(allowed), "mode": "rw"}],
            },
        },
    )
    assert r.status_code == 200, r.get_data(as_text=True)
    stored = server.auth_store.account(ALICE)["grants"].to_obj()["folders"]
    assert stored == [{"path": str(allowed.resolve()), "mode": "rw"}]
    bad = admin.patch(
        f"/api/admin/users/{ALICE}",
        json={
            "grants": {"folders": [{"path": str(Path.home()), "mode": "rw"}]},
        },
    )
    assert bad.status_code == 400


def test_catalogue_is_served_to_admins(env):
    assert _login(ADMIN).get("/api/admin/catalog").get_json() == CATALOGUE


def test_directory_lists_enabled_people_for_the_share_picker(env):
    server.auth_store.create_user("gone@example.com", "pw")
    server.auth_store.set_disabled("gone@example.com", True)
    people = _login(ALICE).get("/api/users/directory").get_json()["users"]
    assert people == [
        {"email": ADMIN, "name": "Admin"},
        {"email": ALICE, "name": "Alice"},
    ]


def test_catalogue_builder_marks_external_agents_and_mcp_groups(monkeypatch):
    from sharedagents_web import user_catalog

    class _Cfg:
        def __init__(self, engine):
            self.engine = engine

    monkeypatch.setattr(user_catalog, "_agent_names", lambda: ["jarvis", "codex", "explore"])
    monkeypatch.setattr(user_catalog, "_agent_engine", lambda name: "codex_cli" if name == "codex" else "local_llm")
    monkeypatch.setattr(
        user_catalog,
        "_cached_models",
        lambda: {
            "jarvis": [{"id": "m1", "alias": "one", "display": "Model one", "provider": "local"}],
            "plan": [{"id": "m1", "alias": "one"}, {"id": "m2", "display": "Two", "provider": "openai"}],
        },
    )
    monkeypatch.setattr(
        user_catalog,
        "_configured_models",
        lambda: [
            {"id": "qwen-local", "alias": "qwen", "backend": "vllm"},
            {"alias": "m1"},  # an alias the cache already knows: not duplicated
        ],
    )
    monkeypatch.setattr(user_catalog, "_builtin_tools", lambda: [("read", "Read a file"), ("bash", "Run a command")])
    monkeypatch.setattr(user_catalog, "_mcp_server_names", lambda: ["tavily"])
    monkeypatch.setattr(user_catalog, "_skill_rows", lambda: [("pdf", "PDF tools")])
    cat = user_catalog.build_catalog()
    agents = {a["name"]: a for a in cat["agents"]}
    assert agents["codex"]["external"] is True and agents["jarvis"]["external"] is False
    assert agents["explore"]["conversational"] is False
    # Configured deployments are grantable even before any catalogue is cached.
    assert [m["id"] for m in cat["models"]] == ["m1", "m2", "qwen-local"]
    assert {
        "name": "mcp__tavily__*",
        "description": "Every tool of the tavily MCP server",
        "group": "MCP",
        "risk": "",
    } in cat["tools"]
    assert {"name": "bash", "description": "Run a command", "group": "Shell", "risk": "host"} in cat["tools"]
    assert cat["skills"] == [{"name": "pdf", "description": "PDF tools"}]
    assert "folder_roots" in cat


def test_last_admin_guard_lives_in_the_write_transaction(env):
    from sharedagents_web.native_auth import LastAdminError

    store = server.auth_store
    store.update_account(ALICE, role="admin")
    store.update_account(ADMIN, role="user")  # Alice is still an admin
    with pytest.raises(LastAdminError):
        store.update_account(ALICE, disabled=True)  # would leave nobody
    assert store.account(ALICE)["disabled"] is False
    assert store.account(ALICE)["role"] == "admin"
