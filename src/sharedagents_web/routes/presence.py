"""sharedagents_web.routes.presence — Presence + seamless cross-device handoff routes.

Extracted from server.py (Task 2 de-monolith). Handler bodies are
byte-identical to the originals; only the decorator target changed
(``@app.route`` -> ``@bp.route``). Registered by server.create_app().
"""

from flask import Blueprint, g, jsonify, request

from sharedagents_web.server import BASE_PATH, require_auth, store, web_state

bp = Blueprint("presence", __name__)


@bp.route("/api/presence", methods=["POST"])
@require_auth
def api_presence():
    """Upsert this device's presence.

    Piggybacked on activity the client already reports (session open, stream
    attach). Scoped to the authenticated owner so no device ever advertises
    presence for another operator (the auth boundary).
    """
    data = request.get_json(silent=True) or {}
    device_id = str(data.get("device_id") or "").strip()
    if not device_id:
        return jsonify({"error": "device_id required"}), 400
    web_state.upsert_presence(
        device_id,
        g.user_email,
        device_label=str(data.get("device_label") or "")[:120] or None,
        active_sid=str(data.get("active_sid") or "").strip() or None,
        streaming=bool(data.get("streaming")),
    )
    return jsonify({"ok": True})


@bp.route("/api/presence/continue-here")
@require_auth
def api_continue_here():
    """Offer the session live on the operator's OTHER device.

    This is the data behind the continue-here chip: the second device is handed
    its first device's live session directly, so handoff is never
    pick-from-session-list. Owner-scoped.
    """
    device_id = str(request.args.get("device_id") or "").strip()
    current_sid = str(request.args.get("current_sid") or "").strip() or None
    other = web_state.continue_here(
        g.user_email,
        exclude_device_id=device_id,
        current_sid=current_sid,
    )
    if not other:
        return jsonify({"continue": None})
    # Never offer a conversation this person can no longer open.
    from sharedagents_web import access

    if access.session_access(g.user_email, other["active_sid"]) == access.NONE:
        return jsonify({"continue": None})
    # Resolve the web session id + title for the deep link.
    sdata = store.get(other["active_sid"], g.user_email)
    title = (sdata or {}).get("title") or (sdata or {}).get("topic") or ""
    return jsonify(
        {
            "continue": {
                "sid": other["active_sid"],
                "device_label": other.get("device_label"),
                "streaming": bool(other.get("streaming")),
                "title": title,
            }
        }
    )


@bp.route("/api/sessions/<sid>/handoff")
@require_auth
def api_session_handoff(sid):
    """Return the live attach-point for a session so another device continues
    it mid-turn.

    The turn-inverted architecture already runs turns server-side and lets any
    viewer attach via ``mode=events&from=<offset>``; this endpoint hands the
    second device exactly what it needs to attach at the live tail (gateway
    hash + current message offset + streaming flag) — no session-list step.
    Owner-scoped: a session the caller does not own is 404, never leaked.
    """
    sdata = store.get(sid, g.user_email)
    if sdata is None:
        return jsonify({"error": "session not found"}), 404
    messages = sdata.get("messages") or []
    return jsonify(
        {
            "sid": sid,
            "session_id": sid,
            "offset": len(messages),
            "streaming": bool(sdata.get("streaming")),
            "title": sdata.get("title") or sdata.get("topic") or "",
            "stream_url": f"{BASE_PATH}/api/sessions/{sid}/stream?mode=events&from={len(messages)}",
        }
    )
