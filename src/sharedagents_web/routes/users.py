"""Users tab — admin-only accounts, roles and grants; the user directory.

Admins add accounts, set role (admin / user), enable or disable, reset the
password, rename, and grant each user the agents, models, tools and skills
they may use. Any signed-in person may read the directory (email + name of
enabled accounts) to pick people to share a conversation with.

Every route here except the directory is admin-only twice over: the access
gate denies unlisted endpoints to users, and ``require_admin`` re-checks.
"""

from __future__ import annotations

import re

from flask import Blueprint, g, jsonify, render_template, request

from sharedagents_core.access_grants import Grants
from sharedagents_core.folder_policy import FolderPolicyError, normalize_folder_grants
from sharedagents_web import access, user_catalog
from sharedagents_web.native_auth import ROLE_USER, ROLES, LastAdminError
from sharedagents_web.server import BASE_PATH, REALTIME_URL, require_admin, require_auth

bp = Blueprint("users", __name__)

_EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+")
MIN_PASSWORD_LEN = 8
_LOCKS = {
    "env": "SHAREDAGENTS_WEB_ADMINS",
    "operator": "SHAREDAGENTS_WEB_LOCAL_OPERATOR_EMAIL",
    "debug": "the loopback debug instance",
}


def _auth():
    from sharedagents_web import server

    return server.auth_store


def _me() -> str:
    return str(getattr(g, "user_email", "") or "").strip().lower()


def _row(acct: dict) -> dict:
    source = access.admin_source(acct["email"], fresh=True)
    return {
        "email": acct["email"],
        "display_name": acct["display_name"],
        "name": _auth().display_name(acct["email"]),
        "role": acct["role"],
        "is_admin": source is not None,
        "admin_source": source,
        "locked": source in _LOCKS,
        "lock_reason": _LOCKS.get(source or "", ""),
        "disabled": acct["disabled"],
        "created_at": acct["created_at"],
        "last_seen": acct.get("last_seen", 0),
        "grants": acct["grants"].to_obj(),
        "is_self": acct["email"] == _me(),
    }


def _grants_from_body(raw):
    """Parse grants, validating folder paths. Returns Grants or (response, status)."""
    if raw is None:
        return Grants.none()
    if not isinstance(raw, dict):
        return jsonify({"error": "grants must be an object"}), 400
    grants = Grants.from_obj(raw)
    if grants.folders is None:
        return grants
    try:
        from dataclasses import replace

        return replace(grants, folders=normalize_folder_grants(grants.folders))
    except FolderPolicyError as exc:
        return jsonify({"error": str(exc)}), 400


def _write_guard():
    """Admin writes are JSON from this origin (CSRF defense in depth)."""
    if request.headers.get("Sec-Fetch-Site", "").lower() == "cross-site":
        return jsonify({"error": "cross-site request refused"}), 403
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 415
    if not isinstance(request.get_json(silent=True), dict):
        return jsonify({"error": "body must be a JSON object"}), 400
    return None


def _clean_email(raw) -> str:
    return str(raw or "").strip().lower()


@bp.route("/users")
@require_auth
@require_admin
def users_page():
    return render_template("users.html", base=BASE_PATH, realtime_url=REALTIME_URL)


@bp.route("/api/admin/users", methods=["GET"])
@require_auth
@require_admin
def api_admin_users():
    return jsonify({"users": [_row(a) for a in _auth().list_accounts()]})


@bp.route("/api/admin/users", methods=["POST"])
@require_auth
@require_admin
def api_admin_create_user():
    refused = _write_guard()
    if refused:
        return refused
    body = request.get_json(silent=True) or {}
    email = _clean_email(body.get("email"))
    password = str(body.get("password") or "")
    role = str(body.get("role") or ROLE_USER)
    if not _EMAIL_RE.fullmatch(email):
        return jsonify({"error": "a valid email is required"}), 400
    if len(password) < MIN_PASSWORD_LEN:
        return jsonify({"error": f"password must be at least {MIN_PASSWORD_LEN} characters"}), 400
    if role not in ROLES:
        return jsonify({"error": "role must be admin or user"}), 400
    if _auth().account(email) is not None:
        return jsonify({"error": "an account with this email already exists"}), 409
    grants = _grants_from_body(body.get("grants")) if "grants" in body else None
    if isinstance(grants, tuple):
        return grants
    _auth().create_user(
        email,
        password,
        display_name=str(body.get("display_name") or ""),
        role=role,
        grants=grants,
    )
    access.forget()
    return jsonify({"user": _row(_auth().account(email))}), 201


@bp.route("/api/admin/users/<path:email>", methods=["PATCH"])
@require_auth
@require_admin
def api_admin_update_user(email):
    refused = _write_guard()
    if refused:
        return refused
    email = _clean_email(email)
    acct = _auth().account(email)
    if acct is None:
        return jsonify({"error": "no such account"}), 404
    body = request.get_json(silent=True) or {}
    role = body.get("role")
    if role is not None and role not in ROLES:
        return jsonify({"error": "role must be admin or user"}), 400
    disabled = body.get("disabled")
    if disabled is not None and not isinstance(disabled, bool):
        return jsonify({"error": "disabled must be true or false"}), 400
    grants = body.get("grants")
    if grants is not None and not isinstance(grants, dict):
        return jsonify({"error": "grants must be an object"}), 400
    source = access.admin_source(email, fresh=True)
    if source in _LOCKS and (role == ROLE_USER or disabled is True):
        return jsonify(
            {
                "error": f"this admin comes from {_LOCKS[source]}; change it there",
            }
        ), 409
    parsed = None
    if grants is not None:
        parsed = _grants_from_body(grants)
        if isinstance(parsed, tuple):
            return parsed
    store = _auth()
    try:
        store.update_account(
            email,
            role=role,
            disabled=disabled,
            grants=parsed,
            display_name=(str(body.get("display_name") or "") if "display_name" in body else None),
        )
    except LastAdminError as exc:
        return jsonify({"error": str(exc)}), 409
    access.forget()
    return jsonify({"user": _row(store.account(email))})


@bp.route("/api/admin/users/<path:email>/password", methods=["POST"])
@require_auth
@require_admin
def api_admin_reset_password(email):
    refused = _write_guard()
    if refused:
        return refused
    email = _clean_email(email)
    password = str((request.get_json(silent=True) or {}).get("password") or "")
    if len(password) < MIN_PASSWORD_LEN:
        return jsonify({"error": f"password must be at least {MIN_PASSWORD_LEN} characters"}), 400
    store = _auth()
    if not store.set_password(email, password):
        return jsonify({"error": "no such account"}), 404
    store.revoke_all_tokens(email)  # sign them out everywhere
    return jsonify({"ok": True})


@bp.route("/api/admin/catalog", methods=["GET"])
@require_auth
@require_admin
def api_admin_catalog():
    return jsonify(user_catalog.catalog())


@bp.route("/api/users/directory", methods=["GET"])
@require_auth
def api_user_directory():
    """Enabled accounts (email + name) for the chat's share picker."""
    store = _auth()
    people = [
        {"email": a["email"], "name": store.display_name(a["email"])}
        for a in store.list_accounts()
        if not a["disabled"]
    ]
    people.sort(key=lambda p: (p["name"].lower(), p["email"]))
    return jsonify({"users": people})
