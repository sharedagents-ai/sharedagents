"""Private / public sharing of one conversation (the chat header's share control).

``GET`` — anyone who can open the conversation sees who it is shared with.
``PUT`` — its owner or an admin changes it (the access gate requires manage
access): ``{"visibility": "private"}`` or ``{"visibility": "public",
"share_all": true}`` or ``{"visibility": "public", "members": [emails]}``.
"""

from __future__ import annotations

from flask import Blueprint, g, jsonify, request

from sharedagents_web import access
from sharedagents_web.server import require_auth

bp = Blueprint("sharing", __name__)


def _server():
    from sharedagents_web import server

    return server


def _payload(sid: str) -> dict:
    srv = _server()
    out = access.sharing_payload(sid)
    owner = access.row_owner(srv.store.get(sid))
    out["owner_email"] = owner
    out["owner_name"] = srv.auth_store.display_name(owner) if owner else ""
    out["can_manage"] = access.session_access(g.user_email, sid) >= access.MANAGE
    return out


def _exists(sid: str) -> bool:
    srv = _server()
    return srv.store.get(sid) is not None or srv._gw_session_log_exists(sid)


@bp.route("/api/sessions/<sid>/sharing", methods=["GET", "PUT"])
@require_auth
def api_session_sharing(sid):
    if not _exists(sid):
        return access.not_found()
    if request.method == "GET":
        return jsonify(_payload(sid))

    if request.headers.get("Sec-Fetch-Site", "").lower() == "cross-site":
        return jsonify({"error": "cross-site request refused"}), 403
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 415
    # Whole conversations only: a sub-agent session is reached through its
    # parent (sharing one alone would expose its parent's tree metadata).
    from sharedagents_web.routes.sessions import _canonical_gateway_parent

    _valid, parent = _canonical_gateway_parent(sid)
    if parent:
        return jsonify({"error": "share the parent conversation instead"}), 400
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "body must be a JSON object"}), 400
    visibility = str(body.get("visibility") or "")
    share_all = body.get("share_all", False)
    if not isinstance(share_all, bool):
        return jsonify({"error": "share_all must be true or false"}), 400
    raw_members = body.get("members", [])
    if not isinstance(raw_members, list):
        return jsonify({"error": "members must be a list"}), 400
    srv = _server()
    owner = access.row_owner(srv.store.get(sid))
    members = []
    for raw in raw_members:
        email = str(raw or "").strip().lower()
        if not email or email == owner:
            continue  # the owner always has the conversation
        acct = srv.auth_store.account(email)
        if acct is None or acct["disabled"]:
            return jsonify({"error": f"no active account {email!r}"}), 400
        members.append(email)
    try:
        srv.sharing_store.set(
            sid,
            visibility=visibility,
            share_all=share_all,
            members=members,
            by=g.user_email,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    access.forget()
    return jsonify(_payload(sid))
