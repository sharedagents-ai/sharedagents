"""sharedagents_web.routes.topics — Topics page + web-local topic grouping routes.

Extracted from server.py (Task 2 de-monolith). Handler bodies are
byte-identical to the originals; only the decorator target changed
(``@app.route`` -> ``@bp.route``). Registered by server.create_app().
"""

from flask import Blueprint, g, jsonify, render_template, request

# Import the server MODULE (not copied names) for globals that tests patch on
# sharedagents_web.server (TOPICS_SCAN_LIMIT / load_forum_topics / load_web_topics):
# attribute access on _server is deferred to call time, so monkeypatches reach
# these relocated handlers (pre-split parity — the reads lived in server.py).
import sharedagents_web.server as _server
from sharedagents_web.server import (
    BASE_PATH,
    REALTIME_URL,
    _merged_sessions,
    add_web_topic,
    all_canonical_names,
    require_auth,
)

bp = Blueprint("topics", __name__)


def _topics_payload(sessions, *, compact: bool, include_all_names: bool = True) -> dict:
    """Build the Topics response from an already-fetched session snapshot.

    Keeping grouping separate from transport lets the Tasks bootstrap reuse
    its one combined gateway read instead of issuing another ``/api/topics``
    request that re-reads the same sessions.
    """
    scanned = len(sessions)
    registry = _server.load_forum_topics()
    buckets = {}
    for s in sessions:
        topic = (s.get("topic") or "general").strip() or "general"
        b = buckets.get(topic)
        if b is None:
            b = {"topic": topic, "count": 0, "sessions": [], "live": False, "waiting": False, "_last": ""}
            buckets[topic] = b
        b["sessions"].append(s)
        b["count"] += 1
        if s.get("live"):
            b["live"] = True
        if s.get("state") == "waiting":
            b["waiting"] = True
            b["live"] = True
        upd = s.get("updated_at") or ""
        if upd > b["_last"]:
            b["_last"] = upd
    seen_fold = {k.casefold() for k in buckets}
    # Every configured topic name (Telegram forum / web topics) is operator
    # data: a user only gets the topics of their own conversations
    extra_names = list(all_canonical_names(registry)) + _server.load_web_topics() if include_all_names else []
    for name in extra_names:
        if name.casefold() in seen_fold:
            continue
        seen_fold.add(name.casefold())
        buckets[name] = {
            "topic": name,
            "count": 0,
            "sessions": [],
            "live": False,
            "waiting": False,
            "_last": "",
        }
    active = sorted(
        [b for b in buckets.values() if b["_last"]],
        key=lambda b: b["_last"],
        reverse=True,
    )
    empty = sorted(
        [b for b in buckets.values() if not b["_last"]],
        key=lambda b: (b.get("topic") or "").casefold(),
    )
    topics = active + empty
    for b in topics:
        b["sessions"].sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        b.pop("_last", None)
    if compact:
        for b in topics:
            latest = b["sessions"][0] if b["sessions"] else None
            b["latest_id"] = (latest or {}).get("id") or ""
            b.pop("sessions", None)
    return {
        "topics": topics,
        "scanned": scanned,
        "capped": scanned >= _server.TOPICS_SCAN_LIMIT,
        "forum_topics_updated_at": registry.get("updated_at") or "",
        "compact": compact,
    }


def _topics_for(owner, agent=None, *, compact: bool, gateway_roots=None) -> dict:
    """The ``GET /api/topics`` payload for *owner*.

    ``gateway_roots`` lets a caller that already read the gateway's session
    roots (the chat boot) reuse that read instead of issuing another.
    """
    sessions = _merged_sessions(
        owner,
        agent,
        limit=_server.TOPICS_SCAN_LIMIT,
        gateway_roots=gateway_roots,
    )
    from sharedagents_web import access

    return _topics_payload(
        sessions,
        compact=compact,
        include_all_names=access.is_admin(owner),
    )


@bp.route("/api/topics", methods=["GET"])
@require_auth
def api_list_topics():
    """Group conversations by topic.

    Default (chat sidebar): compact rows ``{topic, count, live, waiting,
    latest_id}`` — no per-topic session arrays. Pass ``?full=1`` for the Topics
    page payload ``{topic, count, sessions: [...]}`` (same session shape as
    ``/api/sessions``). ``capped`` is true when the scan hit
    ``TOPICS_SCAN_LIMIT``.
    """
    agent = request.args.get("agent")
    # Chat sidebar / other-tab nav only need name, count, live, and the latest
    # session id. Embedding every conversation (TOPICS_SCAN_LIMIT=500) was a
    # ~70 kB download on every existing-chat enter. The Topics page still
    # requests the full session arrays with ?full=1.
    compact = str(request.args.get("full") or "").lower() not in ("1", "true", "yes")
    return jsonify(_topics_for(g.user_email, agent, compact=compact))


@bp.route("/api/topics", methods=["POST"])
@require_auth
def api_create_topic():
    """Create a web-local topic (no Telegram required).

    Body: ``{"name": "..."}``. Persists to the web-owned ``web_topics.json`` so
    the topic appears on the Topics page as an empty channel until a
    conversation is bound to it. Idempotent on an existing name (casefold).
    """
    data = request.get_json(silent=True) or {}
    name = data.get("name") or data.get("topic") or ""
    try:
        topic, created = add_web_topic(name)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True, "topic": topic, "created": created}), (201 if created else 200)


@bp.route("/topics")
@require_auth
def topics_page():
    """Serve the Topics page — conversations grouped by topic.

    Mirrors the Telegram forum-topic grouping the operator sees in the bot:
    every session carries a ``topic`` (default ``general``) and this page lists
    them topic-by-topic, each session deep-linking back to the chat view.
    """
    return render_template("topics.html", base=BASE_PATH, realtime_url=REALTIME_URL)
