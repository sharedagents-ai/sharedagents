"""sharedagents_web.routes.agent_builder — the Agent Builder tab (page + API proxy).

Two cards (owner scope):
  1. **System prompt** — the COMPLETE system prompt the agent sends to the
     model, with each section tagged hardcoded / AGENTS.md / system /
     dynamic, and the AGENTS.md part editable + saveable.
  2. **Permissions** — the agent's effective permissions (tool set resolved
     by the backend's ``compose_tool_allowlist``, never re-derived in JS),
     with the operator-owned allowlist fields editable + saveable.

Every route is authenticated server-side (``@require_auth``).  Agent Builder
is available to authenticated users by default; the reusable admin whitelist
in :mod:`sharedagents_web.server` remains available for future restricted surfaces.
All data routes are thin proxies to the gateway's
``/control/v1/agents/defs*`` + ``/control/v1/agents/registry`` surface —
the web carries zero agent-definition logic.

gated server-side.
leave the backend in any Builder response.
"""

from __future__ import annotations

from flask import Blueprint, g, jsonify, render_template, request

from sharedagents_web.principal import gateway_principal_headers
from sharedagents_web.server import (
    BASE_PATH,
    REALTIME_URL,
    gateway_request,
    require_auth,
)

bp = Blueprint("agent_builder", __name__)


def _proxy(method: str, path: str, json_body: dict | None = None):
    """Forward to the gateway control surface and relay its envelope.

    Returns a Flask response: 502 when the gateway is unreachable, the
    gateway's own status + JSON body otherwise (including its 409 sha
    conflicts, which the UI re-merges against).
    """
    kwargs: dict = {"headers": gateway_principal_headers(g.user_email)}
    if json_body is not None:
        kwargs["json"] = json_body
    resp = gateway_request(method, path, **kwargs)
    if resp is None:
        return jsonify({"error": "Gateway unreachable"}), 502
    try:
        payload = resp.json()
    except ValueError:
        payload = {"error": "gateway returned non-JSON"}
    if resp.status_code >= 400:
        return jsonify(payload), resp.status_code
    return jsonify(payload)


@bp.route("/agents/builder")
@require_auth
def builder_page():
    """Serve the Agent Builder tab to authenticated users."""
    return render_template(
        "agent_builder.html",
        base=BASE_PATH,
        realtime_url=REALTIME_URL,
    )


@bp.route("/api/agents/defs")
@require_auth
def api_agent_defs_list():
    """Summary rows for every configured agent."""
    return _proxy("GET", "/control/v1/agents/defs")


@bp.route("/api/agents/defs/<agent>")
@require_auth
def api_agent_def_get(agent: str):
    """Full definition + effective permissions for one agent.

    A non-admin (the chat composer's saved-default read) gets only
    ``{name, model, effort}`` of an agent granted to them; anything else is
    404 — the full definition is operator configuration.
    """
    from sharedagents_web import access

    if access.is_admin(g.user_email):
        return _proxy("GET", f"/control/v1/agents/defs/{agent}")
    if access.agent_refusal(g.user_email, agent):
        return jsonify({"error": "unknown agent"}), 404
    resp = gateway_request(
        "GET",
        f"/control/v1/agents/defs/{agent}",
        headers=gateway_principal_headers(g.user_email),
    )
    if resp is None:
        return jsonify({"error": "Gateway unreachable"}), 502
    try:
        data = resp.json()
    except ValueError:
        data = {}
    if resp.status_code >= 400 or not isinstance(data, dict):
        return jsonify({"error": "unknown agent"}), 404
    return jsonify(
        {
            "name": str(data.get("name") or agent),
            "model": data.get("model"),
            "effort": data.get("effort"),
        }
    )


@bp.route("/api/agents/defs/<agent>", methods=["PATCH"])
@require_auth
def api_agent_def_patch(agent: str):
    """Raw-JSON-merge config patch (sha-guarded; gateway-owned)."""
    body = request.get_json(silent=True) or {}
    return _proxy("PATCH", f"/control/v1/agents/defs/{agent}", json_body=body)


@bp.route("/api/agents/defs/<agent>/validate", methods=["POST"])
@require_auth
def api_agent_def_validate(agent: str):
    """Dry-run validation of a patch body."""
    body = request.get_json(silent=True) or {}
    return _proxy("POST", f"/control/v1/agents/defs/{agent}/validate", json_body=body)


@bp.route("/api/agents/defs/<agent>/agents_md")
@require_auth
def api_agent_agents_md_get(agent: str):
    """The operator-editable AGENTS.md + sha."""
    return _proxy("GET", f"/control/v1/agents/defs/{agent}/agents_md")


@bp.route("/api/agents/defs/<agent>/agents_md", methods=["PUT"])
@require_auth
def api_agent_agents_md_put(agent: str):
    """Persist AGENTS.md (sha-guarded; 64 KiB cap; fence-marker rejection)."""
    body = request.get_json(silent=True) or {}
    return _proxy("PUT", f"/control/v1/agents/defs/{agent}/agents_md", json_body=body)


@bp.route("/api/agents/defs/<agent>/system_prompt")
@require_auth
def api_agent_system_prompt(agent: str):
    """The complete system prompt, split by source (hardcoded/AGENTS.md/…)."""
    return _proxy("GET", f"/control/v1/agents/defs/{agent}/system_prompt")


@bp.route("/api/agents/registry")
@require_auth
def api_agent_registry():
    """Builtin + MCP tool names (for the permissions card's reference list)."""
    return _proxy("GET", "/control/v1/agents/registry")
