"""Channels tab — Telegram + WhatsApp send-allowlist CRUD."""

from flask import Blueprint, jsonify, render_template, request

from sharedagents_web import channel_allowlist, channel_qr
from sharedagents_web.server import BASE_PATH, REALTIME_URL, require_auth

bp = Blueprint("channels", __name__)


@bp.route("/channels")
@require_auth
def channels_page():
    return render_template(
        "channels.html",
        base=BASE_PATH,
        realtime_url=REALTIME_URL,
    )


@bp.route("/api/channels", methods=["GET"])
@require_auth
def api_channels_get():
    return jsonify(channel_allowlist.snapshot())


@bp.route("/api/channels/<channel>/allowlist", methods=["POST", "PATCH", "DELETE"])
@require_auth
def api_channels_allowlist(channel):
    try:
        if request.method == "DELETE":
            kind = request.args.get("kind") or ""
            ident = request.args.get("id") or ""
            return jsonify(channel_allowlist.remove(channel, kind, ident))
        body = request.get_json(silent=True) or {}
        kind = str(body.get("kind") or "")
        ident = str(body.get("id") or "")
        if request.method == "POST":
            return jsonify(
                channel_allowlist.add(
                    channel,
                    kind,
                    ident,
                    str(body.get("label") or ""),
                    reply=body.get("reply"),
                    mention=body.get("mention"),
                    read=body.get("read"),
                    send=body.get("send"),
                    agent=body.get("agent"),
                    model=body.get("model"),
                )
            )
        return jsonify(
            channel_allowlist.update(
                channel,
                kind,
                ident,
                label=body.get("label"),
                new_id=body.get("new_id"),
                reply=body.get("reply"),
                mention=body.get("mention"),
                read=body.get("read"),
                send=body.get("send"),
                agent=body.get("agent"),
                model=body.get("model"),
            )
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@bp.route("/api/channels/<channel>/qr", methods=["GET", "POST"])
@require_auth
def api_channels_qr(channel):
    if channel != "whatsapp":
        return jsonify({"error": "QR pairing is only for WhatsApp"}), 400
    if request.method == "GET":
        return jsonify(channel_qr.status())
    body = request.get_json(silent=True) or {}
    phone = str(body.get("phone") or "").strip() or None
    return jsonify(channel_qr.generate(phone))
