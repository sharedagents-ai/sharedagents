"""sharedagents_web.routes.settings — Live gateway-switching settings route.

Extracted from server.py (Task 2 de-monolith). Handler bodies are
byte-identical to the originals; only the decorator target changed
(``@app.route`` -> ``@bp.route``). Registered by server.create_app().
"""

import requests
from flask import Blueprint, jsonify, request

from sharedagents_web.server import current_gateway, require_auth, web_state

bp = Blueprint("settings", __name__)


@bp.route("/api/settings/gateway", methods=["GET", "POST"])
@require_auth
def api_settings_gateway():
    """Read or switch the active gateway at RUNTIME, no process restart.

    POST {url}: probe the target first (short-timeout reachability check); only
    on success is the active gateway written to the web-state store, from which
    ``current_gateway()`` resolves per request. A failed probe leaves the prior
    connection fully usable.
    """
    if request.method == "GET":
        return jsonify({"gateway": current_gateway(), "gateways": web_state.list_gateways()})
    data = request.get_json(silent=True) or {}
    url = str(data.get("url") or "").strip().rstrip("/")
    if not url or not (url.startswith("http://") or url.startswith("https://")):
        return jsonify({"error": "a valid http(s) gateway url is required"}), 400
    # Probe reachability — any HTTP response (even 401) proves the gateway is up;
    # only a connection failure means unreachable.
    try:
        requests.get(f"{url}/", timeout=3)
    except requests.RequestException:
        return jsonify({"error": "gateway unreachable — switch aborted", "gateway": current_gateway()}), 502
    web_state.set_active_gateway(url)
    return jsonify({"ok": True, "gateway": current_gateway()})
