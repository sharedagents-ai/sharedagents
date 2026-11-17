"""sharedagents_web.routes.secrets — the "Secrets" dialog's API (admin only).

A value typed into the dialog goes from the browser to the gateway's
``/control/v1/secrets`` surface and nowhere else: never ``store.add_message``,
never a chat completion, never a log line, never a response body. The gateway
owns validation and storage (:mod:`sharedagents_gateway.secrets_endpoint`); this
module only proxies, and relays a fixed ``{error, code}`` shape on failure so
nothing from the request can echo back.
"""

from __future__ import annotations

from urllib.parse import quote

from flask import Blueprint, g, jsonify, request

from sharedagents_web.principal import gateway_principal_headers
from sharedagents_web.server import gateway_request, require_admin, require_auth

bp = Blueprint("secrets", __name__)

_NO_STORE = {"Cache-Control": "no-store"}


def _write_guard():
    """Writes are JSON from this origin (CSRF defense in depth)."""
    if request.headers.get("Sec-Fetch-Site", "").lower() == "cross-site":
        return jsonify({"error": "cross-site request refused"}), 403
    if not request.is_json or not isinstance(request.get_json(silent=True), dict):
        return jsonify({"error": "JSON object body required"}), 415
    return None


def _relay(resp):
    if resp is None:
        return jsonify({"error": "Gateway unreachable"}), 502, _NO_STORE
    try:
        payload = resp.json()
    except ValueError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    if resp.status_code >= 400:
        body = {
            "error": str(payload.get("error") or "Request failed")[:300],
            "code": str(payload.get("code") or ""),
        }
        return jsonify(body), resp.status_code, _NO_STORE
    return jsonify(payload), 200, _NO_STORE


def _gw(method: str, path: str, **kwargs):
    headers = gateway_principal_headers(g.user_email)
    return gateway_request(method, path, headers=headers, timeout=30, **kwargs)


@bp.route("/api/secrets", methods=["GET"])
@require_auth
@require_admin
def api_secrets_list():
    """Stored names and their scope; values never leave the gateway."""
    return _relay(_gw("GET", "/control/v1/secrets"))


@bp.route("/api/secrets", methods=["PUT"])
@require_auth
@require_admin
def api_secrets_put():
    refused = _write_guard()
    if refused:
        return refused
    body = request.get_json(silent=True)
    agent = body.get("agent") or None
    return _relay(
        _gw(
            "PUT",
            "/control/v1/secrets",
            json={
                "name": body.get("name"),
                "value": body.get("value"),
                "agent": agent,
            },
        )
    )


@bp.route("/api/secrets", methods=["DELETE"])
@require_auth
@require_admin
def api_secrets_delete():
    refused = _write_guard()
    if refused:
        return refused
    body = request.get_json(silent=True)
    name = body.get("name")
    if not isinstance(name, str) or not name:
        return jsonify({"error": "name required"}), 400
    agent = body.get("agent") or ""
    path = f"/control/v1/secrets/{quote(name, safe='')}"
    params = {"agent": agent} if isinstance(agent, str) and agent else None
    return _relay(_gw("DELETE", path, params=params))
