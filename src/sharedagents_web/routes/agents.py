"""sharedagents_web.routes.agents — Agent-monitor tree page, filtered session poll, snapshot/stream.

Extracted from server.py (Task 2 de-monolith). Handler bodies are
byte-identical to the originals; only the decorator target changed
(``@app.route`` -> ``@bp.route``). Registered by server.create_app().
"""

import json
import time

from flask import Blueprint, Response, g, jsonify, render_template, request, stream_with_context

from sharedagents_web.principal import gateway_principal_headers
from sharedagents_web.server import (
    _TREE_STREAM_INTERVAL,
    _TREE_STREAM_MAX_SECONDS,
    BASE_PATH,
    REALTIME_URL,
    _fetch_tree,
    _parse_tree_limit,
    gateway_request,
    require_auth,
    store,
)

bp = Blueprint("agents", __name__)


@bp.route("/agents")
@require_auth
def agents_monitor():
    """Serve the live agent-monitoring tree page."""
    return render_template("agents.html", base=BASE_PATH, realtime_url=REALTIME_URL)


@bp.route("/api/agents/tree")
@require_auth
def api_agents_tree():
    """One-shot agent → session → subagent tree snapshot (proxy)."""
    include_all = request.args.get("all") in ("1", "true", "yes")
    refresh = request.args.get("refresh") in ("1", "true", "yes")
    limit = _parse_tree_limit(request.args.get("limit"))
    session_id = str(request.args.get("session") or "").strip()
    tree = _fetch_tree(
        include_all,
        limit=limit,
        refresh=refresh,
        session_id=session_id,
        headers=(gateway_principal_headers(g.user_email) if session_id else None),
    )
    if tree is None:
        return jsonify({"agents": [], "summary": [], "totals": {}, "error": "Gateway unreachable"}), 502
    return jsonify(tree)


@bp.route("/api/agents/footer")
@require_auth
def api_agents_footer():
    """Flat session lineage for the chat-docked Agent Monitor."""
    from sharedagents_web.agent_monitor_footer import footer_payload, unbound_footer_payload

    session_id = str(request.args.get("session") or "").strip()
    if not session_id:
        return jsonify({"rows": [], "subs": 0, "error": "session required"}), 400
    sdata = store.get(session_id, g.user_email)
    from sharedagents_web.agent_monitor_footer import (
        enrich_footer_with_history,
        stamp_footer_from_web_row,
    )
    from sharedagents_web.server import GW_SESSIONS_DIR, resolve_authoritative_session_title

    def _stamp(footer):
        return stamp_footer_from_web_row(
            footer,
            sdata,
            session_id,
            title=resolve_authoritative_session_title(
                session_id,
                str((sdata or {}).get("title") or ""),
            ),
        )

    if sdata and store.is_unused_web_placeholder(sdata):
        agent = store.display_agent(sdata)
        return jsonify(
            _stamp(
                unbound_footer_payload(
                    agent,
                    str(sdata.get("title") or ""),
                    str(sdata.get("model") or sdata.get("engine_model") or ""),
                )
            )
        )
    tree = _fetch_tree(
        False,
        session_id=session_id,
        headers=gateway_principal_headers(g.user_email),
        retry_unavailable=False,
    )

    if tree is None:
        if not sdata:
            return jsonify({"rows": [], "subs": 0, "error": "Gateway unreachable"}), 502
        footer = _stamp(
            enrich_footer_with_history(
                {"rows": [], "subs": 0, "error": "Gateway unreachable"},
                session_id,
                GW_SESSIONS_DIR,
            )
        )
        if footer.get("rows") or footer.get("subs"):
            return jsonify(footer)
        return jsonify(footer), 502
    return jsonify(
        _stamp(
            enrich_footer_with_history(
                footer_payload(tree, session_id),
                session_id,
                GW_SESSIONS_DIR,
            )
        )
    )


def _parse_max_age_sec(raw: str | None) -> int:
    try:
        return max(0, int(raw or 0))
    except (TypeError, ValueError):
        return 0


def _csv_values(args, key: str) -> set[str]:
    values: set[str] = set()
    for item in args.getlist(key) or ():
        values.update(part.strip() for part in str(item).split(",") if part.strip())
    return values


def _last_seen_ms(node: dict) -> int:
    ts = node.get("last_activity_at") or node.get("last_turn_ts") or node.get("state_updated_at") or 0
    try:
        ts = int(ts)
    except (TypeError, ValueError):
        return 0
    if ts <= 0:
        return 0
    return ts if ts > 1_000_000_000_000 else ts * 1000


def _fg_val(node: dict, dim: str) -> str:
    if dim == "agent":
        return str(node.get("agent") or "")
    if dim == "state":
        if node.get("stale"):
            return "stale"
        return str(node.get("state") or "idle")
    return str(node.get(dim) or "")


def _self_passes(
    node: dict,
    *,
    now_ms: int,
    max_age_sec: int,
    query: str,
    fg: dict[str, set[str]],
) -> bool:
    if max_age_sec:
        ms = _last_seen_ms(node)
        if not ms or (now_ms - ms) > max_age_sec * 1000:
            return False
    if query:
        hay = (f"{node.get('agent') or ''} {node.get('title') or ''} {node.get('hash') or ''}").lower()
        if query not in hay:
            return False
    return all(not (allowed and _fg_val(node, dim) not in allowed) for dim, allowed in fg.items())


def _node_kept(node: dict, **kw) -> bool:
    if _self_passes(node, **kw):
        return True
    return any(_node_kept(child, **kw) for child in (node.get("children") or []))


def _prune_node(node: dict, **kw) -> dict | None:
    if not _node_kept(node, **kw):
        return None
    out = dict(node)
    out["children"] = [
        pruned for child in (node.get("children") or []) if (pruned := _prune_node(child, **kw)) is not None
    ]
    return out


def filter_monitor_tree(
    tree: dict,
    *,
    max_age_sec: int = 0,
    query: str = "",
    agent: set[str] | None = None,
    model: set[str] | None = None,
    engine: set[str] | None = None,
    state: set[str] | None = None,
    now_ms: int | None = None,
) -> dict:
    """Prune a session tree with the Agent Monitor tab's filters."""
    kw = {
        "now_ms": int(now_ms if now_ms is not None else time.time() * 1000),
        "max_age_sec": int(max_age_sec or 0),
        "query": (query or "").strip().lower(),
        "fg": {
            "agent": set(agent or ()),
            "model": set(model or ()),
            "engine": set(engine or ()),
            "state": set(state or ()),
        },
    }
    agents = []
    for bucket in tree.get("agents") or []:
        kept_roots = [pruned for root in (bucket.get("roots") or []) if (pruned := _prune_node(root, **kw)) is not None]
        if not kept_roots:
            continue
        out = dict(bucket)
        out["roots"] = kept_roots
        agents.append(out)
    filtered = dict(tree)
    filtered["agents"] = agents
    return filtered


@bp.route("/api/agents/monitor")
@require_auth
def api_agents_monitor():
    """Filtered session tree for the Agent Monitor 1s poll.

    Same payload shape as ``GET /api/agents/tree`` (agents → roots →
    children), with the tab's session filters applied server-side:
    ``limit``, ``max_age_sec``, ``q``, ``agent``, ``model``, ``engine``,
    ``state``. This is a session monitor, not a goals overlay.
    """
    include_all = request.args.get("all") in ("1", "true", "yes")
    refresh = request.args.get("refresh") in ("1", "true", "yes")
    limit = _parse_tree_limit(request.args.get("limit"))
    tree = _fetch_tree(include_all, limit=limit, refresh=refresh)
    if tree is None:
        return jsonify({"agents": [], "summary": [], "totals": {}, "error": "Gateway unreachable"}), 502
    filtered = filter_monitor_tree(
        tree,
        max_age_sec=_parse_max_age_sec(request.args.get("max_age_sec")),
        query=str(request.args.get("q") or ""),
        agent=_csv_values(request.args, "agent"),
        model=_csv_values(request.args, "model"),
        engine=_csv_values(request.args, "engine"),
        state=_csv_values(request.args, "state"),
    )
    return jsonify(filtered)


@bp.route("/api/agents/abort", methods=["POST"])
@require_auth
def api_agents_abort():
    """Stop ONE delegated subagent session (non-cascading) via the gateway.

    The body carries the GATEWAY session hash of the child (``session_id`` or
    ``sid``) — the hash a monitor row already carries, so no web-store
    resolution/adoption is involved. Lineage authorization is enforced HERE,
    from gateway index data only: a non-admin caller must own a verified
    ancestor of the target (``_owned_gateway_ancestor``), so a sibling or a
    foreign-tree hash is rejected. An admin (``may_adopt_gateway_session``)
    may stop any subagent — the same single-user rule that already lets an
    admin read every gateway session, including TUI/Telegram trees no Web
    user is bound to. Identity never comes from browser query params
    (fail-closed).
    """
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        return jsonify({"error": "body must be a JSON object"}), 400
    sid = str(body.get("session_id") or body.get("sid") or "").strip()
    if not sid:
        return jsonify({"error": "'session_id' is required"}), 400

    from sharedagents_web.principal import may_adopt_gateway_session
    from sharedagents_web.routes.sessions import _owned_gateway_ancestor

    allowed = False
    if may_adopt_gateway_session(g.user_email):
        allowed = True
    else:
        try:
            ancestor = _owned_gateway_ancestor(sid, g.user_email)
            if ancestor:
                allowed = True
        except Exception:  # noqa: BLE001 — fail closed
            allowed = False
    if not allowed:
        return jsonify({"error": "Session not found"}), 403

    resp = gateway_request(
        "POST",
        f"/control/v1/orchestrator/sessions/{sid}/abort",
        json=body,
        headers=gateway_principal_headers(g.user_email),
    )
    if resp is None:
        return jsonify({"error": "Gateway unreachable"}), 502
    try:
        return jsonify(resp.json()), resp.status_code
    except ValueError:
        return jsonify({"error": "bad gateway response"}), 502


@bp.route("/api/agents/tree/stream")
@require_auth
def api_agents_tree_stream():
    """SSE stream of the live agent tree.

    Emits a ``snapshot`` event on connect and whenever the tree changes; a
    named ``hb`` heartbeat event otherwise (a real event, not a ``:`` comment,
    so the browser's EventSource delivers it as a liveness signal). A
    ``gw_error`` event signals the gateway is unreachable. Bounded lifetime —
    the browser reconnects transparently.
    """
    import time as _time

    include_all = request.args.get("all") in ("1", "true", "yes")
    refresh = request.args.get("refresh") in ("1", "true", "yes")
    limit = _parse_tree_limit(request.args.get("limit"))

    def generate():
        last_sig = None
        started = _time.monotonic()
        # Prime immediately so a fresh tab paints without waiting a tick.
        while _time.monotonic() - started < _TREE_STREAM_MAX_SECONDS:
            tree = _fetch_tree(include_all, limit=limit, refresh=refresh)
            if tree is None:
                yield f"event: gw_error\ndata: {json.dumps({'error': 'gateway_unreachable'})}\n\n"
            else:
                sig = json.dumps(tree, sort_keys=True, ensure_ascii=False)
                if sig != last_sig:
                    last_sig = sig
                    yield f"event: snapshot\ndata: {sig}\n\n"
                else:
                    # Named heartbeat (NOT a `:` comment) so the browser's
                    # EventSource actually delivers it — the client uses it as
                    # a liveness signal so the connection dot never falsely
                    # goes "stale" during quiescent (unchanged-tree) periods.
                    yield "event: hb\ndata: {}\n\n"
            _time.sleep(_TREE_STREAM_INTERVAL)
        # Tell the client to reconnect cleanly when we hit the lifetime cap.
        yield "event: reconnect\ndata: {}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
