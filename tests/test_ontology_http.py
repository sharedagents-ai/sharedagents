"""Real bearer authentication and workspace-scoped ontology review."""

import json
import os
import secrets
from pathlib import Path

import pytest

from sharedagents_research.ontology import open_ontology


@pytest.fixture
def api(tmp_path, monkeypatch):
    pytest.importorskip("SharedData.SearchIndex")
    from sharedagents_web import server
    from sharedagents_web.native_auth import ROLE_ADMIN, NativeAuthStore

    for key in (
        "SHAREDAGENTS_WEB_DEBUG",
        "SHAREDAGENTS_WEB_ADMINS",
        "SHAREDAGENTS_WEB_LOCAL_OPERATOR_EMAIL",
        "SHAREDAGENTS_ACCESS_PIN",
    ):
        monkeypatch.delenv(key, raising=False)
    home = tmp_path / "data"
    config = tmp_path / "ontology.json"
    config.write_text(
        json.dumps(
            {
                "workspaces": {
                    "team": {
                        "members": {
                            "editor@example.com": "editor",
                            "viewer@example.com": "viewer",
                            "reviewer@example.com": "reviewer",
                        }
                    }
                }
            }
        )
    )
    monkeypatch.setenv("SHAREDAGENTS_ONTOLOGY_ROOT", str(home))
    monkeypatch.setenv("SHAREDAGENTS_ONTOLOGY_CONFIG", str(config))
    monkeypatch.delenv("SHAREDAGENTS_ONTOLOGY_PYTHON", raising=False)
    auth = NativeAuthStore(tmp_path / "auth.sqlite")
    auth.create_user("admin@example.com", secrets.token_urlsafe(24), role=ROLE_ADMIN)
    tokens = {}
    for email in ("editor@example.com", "viewer@example.com", "reviewer@example.com", "other@example.com"):
        auth.create_user(email, secrets.token_urlsafe(24))
        tokens[email] = auth.issue_token(email)
    monkeypatch.setattr(server, "auth_store", auth)
    model = open_ontology(home, "workspace:team")
    obj = model.model_put("term", {"name": "Revenue", "description": "Reported sales"}, actor="test-user")
    return server.app.test_client(), tokens, model, obj


def call(api, method, email=None, **kwargs):
    client, tokens, _, _ = api
    if email:
        kwargs["headers"] = dict(kwargs.get("headers", {}), Authorization="Bearer " + tokens[email])
    return getattr(client, method)("/api/ontology?workspace=team", **kwargs)


def test_workspace_requires_authentication_and_membership(api):
    assert call(api, "get").status_code == 401
    assert call(api, "get", "other@example.com").status_code == 403
    response = call(api, "get", "viewer@example.com")
    assert response.status_code == 200 and response.json["value"][0]["data"]["name"] == "Revenue"
    assert call(api, "post", "viewer@example.com", json={"action": "edit"}).status_code == 403


def test_only_reviewers_can_certify_and_stale_edits_fail(api):
    _, _, model, obj = api
    body = {"action": "review", "id": obj["id"], "expected_revision": obj["revision"], "certified": True}
    assert call(api, "post", "editor@example.com", json=body).status_code == 403
    reviewed = call(api, "post", "reviewer@example.com", json=body)
    assert reviewed.status_code == 200
    assert reviewed.json["value"]["certification"]["reviewer"] == "user:reviewer@example.com"
    edit = {
        "action": "edit",
        "id": obj["id"],
        "kind": "term",
        "expected_revision": obj["revision"],
        "data": {"name": "Stale"},
    }
    assert call(api, "post", "editor@example.com", json=edit).status_code == 409
    assert model.model_get(obj["id"])["certification"]["status"] == "certified"


def test_cross_origin_changes_are_rejected(api):
    _, _, _, obj = api
    body = {
        "action": "edit",
        "id": obj["id"],
        "kind": "term",
        "expected_revision": obj["revision"],
        "data": {"name": "Changed"},
    }
    assert (
        call(api, "post", "editor@example.com", json=body, headers={"Origin": "https://example.com"}).status_code == 403
    )


@pytest.mark.asyncio
async def test_mcp_uses_real_bearer_auth_and_same_workspace_api(api, monkeypatch):
    import httpx

    from sharedagents_research import mcp

    client, tokens, _, obj = api
    path = os.environ["SHAREDAGENTS_ONTOLOGY_CONFIG"]

    config = json.loads(Path(path).read_text())
    config["default_workspaces"] = {"viewer@example.com": "team"}
    Path(path).write_text(json.dumps(config))
    monkeypatch.setenv("SHAREDAGENTS_RESEARCH_API_URL", "https://example.com")
    monkeypatch.setenv("SHAREDAGENTS_RESEARCH_API_TOKEN", tokens["viewer@example.com"])
    original = httpx.AsyncClient

    def transport(request):
        response = client.open(
            request.url.raw_path.decode(), method=request.method, headers=dict(request.headers), data=request.content
        )
        return httpx.Response(
            response.status_code, content=response.data, headers={"Content-Type": response.content_type}
        )

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(transport), **kwargs)
    )
    found = json.loads(await mcp.dispatch("search_ontology", {"query": "Revenue"}))
    assert found[0]["object"]["id"] == obj["id"]
    coverage = json.loads(await mcp.dispatch("check_answerable", {"required": ["Revenue", "Missing"]}))
    assert coverage["missing"] == ["Missing"]
    exported = json.loads(await mcp.dispatch("export_model", {"format": "native"}))
    assert exported["objects"][0]["id"] == obj["id"]
    monkeypatch.setenv("SHAREDAGENTS_RESEARCH_API_TOKEN", secrets.token_urlsafe(24))
    with pytest.raises(mcp.ToolError, match="401"):
        await mcp.dispatch("ontology_status", {})
