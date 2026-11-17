"""sharedagents_web.routes.sessions — Session CRUD, streaming, prompt/answer/permission/queue/cancel, approvals.

Extracted from server.py (Task 2 de-monolith). Handler bodies are
byte-identical to the originals; only the decorator target changed
(``@app.route`` -> ``@bp.route``). Registered by server.create_app().
"""

import json
import logging
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)

from flask import Blueprint, Response, g, jsonify, request, session, stream_with_context  # noqa: E402

from sharedagents_core.ids import bare_id  # noqa: E402
from sharedagents_web.principal import (  # noqa: E402
    canonical_gateway_principal,
    gateway_principal_headers,
    is_configured_local_operator,
    may_adopt_gateway_session,
    turn_actor,
)
from sharedagents_web.server import (  # noqa: E402
    _SESSION_STREAM_READ_TIMEOUT,
    DEFAULT_AGENT,
    GW_SESSIONS_DIR,
    HISTORY_PAGE_DEFAULT,
    _gateway_auth_header,
    _gw_session_log_exists,
    _iso,
    _jsonl_eof,
    _merged_sessions,
    _page_history,
    current_gateway,
    gateway_request,
    gw_session_history,
    require_auth,
    resolve_authoritative_session_title,
    store,
)

bp = Blueprint("sessions", __name__)

# Fail-soft boot extras (health / approvals / state / capabilities) must never
# stall the chat page. The default gateway GET retries for ~25 s and times out
# at 120 s — that was the 10 s empty-chat boot. History stays on the local
# JSONL fast path; these calls just seed badges.
_BOOT_GW_TIMEOUT = 0.4
_BOOT_GW_KW = {"timeout": _BOOT_GW_TIMEOUT, "retry_unavailable": False}


def _stream_gateway_sid(sid: str, sdata: dict | None) -> str:
    """Return the SID whose single JSONL the live stream tails."""
    return str(sid or "").strip()


def _fast_session_header(sid: str) -> dict:
    """Session-header fields the gateway history response would carry.

    The fast local path folds the JSONL directly, so it must project the
    header itself — otherwise the GET merge below falls back to the web
    store's creation-time values and a page refresh silently reverts the
    picker to a stale model (the exact regression the gateway-side comment
    in ``control_endpoint.py`` documents for this field set: the web store's
    stale creation-time model winning the reconciliation).

    ``engine_model_alias`` is left empty: alias resolution needs the live
    agent config (gateway-owned), and a contextless read degrades to the raw
    id — exactly what ``_engine_model_alias_for`` does with no context.
    """
    if not sid:
        return {}
    try:
        from sharedagents_session.session_reader import load_session_dict

        d = load_session_dict(GW_SESSIONS_DIR, sid, fold=False) or {}
    except Exception:
        return {}
    # No sessions_meta row yet (e.g. the projection dir is missing or the
    # lane sid is stale): emit nothing so the GET merge keeps the web store
    # fallback instead of blanking fields with empty strings.
    # ``load_session_dict(fold=False)`` always injects ``turns`` and ``hash``
    # even with no row, so both sentinels must be subtracted — a real row
    # adds at least one more key.
    if not d or not (set(d) - {"hash", "turns"}):
        return {}
    from sharedagents_gateway.control_endpoint import (
        _as_json_dict,
        _session_tree_activity,
    )

    out = {
        "agent": d.get("agent_name") or "",
        "title": d.get("title") or "",
        "engine": d.get("engine") or "",
        "model": d.get("model") or "",
        "engine_model": d.get("engine_model") or "",
        "engine_model_alias": "",
        "updated_at": d.get("last_turn_ts") or d.get("updated_at") or "",
        "context_window": int(d.get("context_window") or 0),
        "last_prompt_tokens": int(d.get("last_prompt_tokens") or 0),
        "compaction_count": int(d.get("compaction_count") or 0),
        "compaction_summary": d.get("compaction_summary") or "",
        "compaction_last_stats": _as_json_dict(d.get("compaction_last_stats")),
    }
    try:
        activity = _session_tree_activity(sid)
        out["session_state"] = activity.get("session_state", "idle")
        out["tree_busy"] = activity.get("tree_busy", False)
        out["parent_session_id"] = activity.get("parent_session_id", "")
        out["subagents"] = activity.get("subagents", [])
        out["session_row"] = bool(activity.get("session_row"))
    except Exception:
        pass
    return out


def _fast_local_history(sid, before, limit):
    """Bounded local fold from the prod JSONL without waiting on the gateway.

    Newest and older pages both reverse-read one 8 MiB window and return
    lazy one-liner shells. Card expand seeks by ``@offset``. The live
    gateway is only a fallback when the local log is missing.

    ``tail_offset`` is the byte cursor for the same session stream the browser
    opens.
    """
    if not sid:
        return None
    from sharedagents_gateway.control_endpoint import (
        _delegating_agent_actors,
        _fold_turns_to_messages,
        _ui_parent_turn_active,
    )
    from sharedagents_session.session_reader import (
        lazy_history_messages,
        scan_history_page,
    )

    # Single-session ledger: exactly one file per session — <sid>.jsonl.
    phys = GW_SESSIONS_DIR / f"{sid}.jsonl"
    if not phys.is_file():
        return None
    path = phys
    stats: dict = {}
    turns: list = []
    try:
        turns, tail, active, cursor = scan_history_page(
            path,
            before_offset=before,
            limit=limit or HISTORY_PAGE_DEFAULT,
            stats=stats,
        )
        messages = _fold_turns_to_messages(
            turns,
            delegator=lambda: _delegating_agent_actors(GW_SESSIONS_DIR, sid),
        )
        stats["active"] = active
    except Exception:
        return None
    # Live cursor: open turn's turn_started, or this file's EOF when idle.
    # Never substitute a sibling log's size — that is the replay storm.
    # The gateway response carries the session header (model, agent, context
    # meter, sticky-Stop state, compaction widget). The fast path skips the
    # gateway, so it must project the header itself — otherwise the GET merge
    # below falls back to the web store's creation-time model/agent and a page
    # refresh silently reverts the picker (regression from the fast-path
    # addition: the stale web-store value won the reconciliation).
    # Header projection is best-effort: any unexpected failure degrades to
    # "no header" (the GET merge keeps the web-store fallback) instead of
    # turning a fast-page GET into a 500.
    header = {}
    try:
        header = _fast_session_header(sid)
        if header:
            # Mirror the native gateway endpoint's badge rule exactly
            # (control_endpoint.py): the most recent COMPLETED turn's model
            # wins over the header column — the column is empty for local-LLM
            # sessions, so the column-only read would leave the web store's
            # adopt-time engine_model winning after a /model switch.
            from sharedagents_gateway.control_endpoint import _history_engine_model

            header["engine_model"] = _history_engine_model(header, turns)
    except Exception:
        header = {}
    if not messages and not header:
        return None
    scan_active = bool(stats.get("open_turn_snapshot") or stats.get("active"))
    return {
        "ok": True,
        "hash": sid,
        "messages": lazy_history_messages(messages),
        "tail_offset": tail,
        "before_cursor": cursor,
        "active": _ui_parent_turn_active(
            scan_active,
            header.get("session_state"),
            session_known=bool(header.get("session_row")),
        ),
        "history_truncated": True,
        "tail_continuation": bool(stats.get("tail_continuation")),
        **header,
    }


def _paged_gateway_history(sid, before, limit):
    """Fetch a bounded gateway page (legacy one-arg test doubles stay valid)."""
    local = _fast_local_history(sid, before, limit)
    if local is not None:
        return local
    kwargs = {
        "before": before,
        "limit": limit or HISTORY_PAGE_DEFAULT,
    }
    try:
        return gw_session_history(sid, **kwargs)
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        return gw_session_history(sid)


def _tail_watermark(hist, gw):
    """The live-stream resume offset handed to the client, never null-for-a-log.

    The browser opens the SSE at ``tail_from_offset`` and treats a NON-NUMBER as
    "start from 0". A from=0 open is a full replay of the entire log over SSE,
    which lands on top of the bounded REST page and re-renders the whole
    session (measured on a 1.6 MB conversation: 1.2 MB / 1436 frames
    still climbing at 15 s, versus 0 KB at the real tail). That silently undoes
    the last-N-turns + scroll-up design.

    So when the gateway omits ``tail_offset``, fall back to the log's EOF. Only
    a session with NO log may return None — those are empty, so a from=0 open
    costs nothing.
    """
    raw = (hist or {}).get("tail_offset")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return raw
    return _jsonl_eof(gw) or None


def _trim_session_payload(session):
    """Bound structured-card bodies on the newest history page.

    Card shells remain complete and chronological; expanded reasoning/tool
    payloads carry the per-card limits applied by ``trim_history_messages``.
    Lazy summarize strips tool/reasoning bodies so reload stays a shell page.
    """
    from sharedagents_session.session_reader import lazy_history_messages

    if not isinstance(session, dict):
        return session
    messages = session.get("messages")
    if not messages:
        return session
    out = dict(session)
    out["messages"] = lazy_history_messages(messages)
    return out


def _owned_session(sid):
    """Return an already-owned web session without translating its SID."""
    sdata = store.get(sid, g.user_email)
    if sdata is None:
        return None, None
    return sdata, sid


def _canonical_gateway_parent(sid: str) -> tuple[bool, str]:
    """Read the parent edge from the immutable session-start header.

    ``sessions.db`` is an index and therefore only a preflight/candidate
    source.  Authorization is based on the JSONL header that created the
    session. A missing or malformed first event fails closed.
    """
    from sharedagents_session.session_reader import tail_jsonl

    path = GW_SESSIONS_DIR / f"{sid}.jsonl"
    try:
        for event, _offset in tail_jsonl(path, 0):
            if event.get("type") != "session_started":
                return False, ""
            return True, str(event.get("parent_session_id") or "")
    except (OSError, TypeError, ValueError):
        return False, ""
    return False, ""


def _gateway_lineage_binding(
    sid: str,
    max_depth: int = 64,
) -> tuple[bool, str, str]:
    """Return ``(valid, Web owner, durable principal)`` for a lineage.

    This distinguishes a genuinely unbound TUI/native tree from another Web
    user's delegated session. Every indexed edge is revalidated against
    its immutable JSONL header before any binding can influence the result.
    """
    from sharedagents_session.session_index import connect_sessions_db

    db_path = GW_SESSIONS_DIR / "sessions.db"
    if not sid or not db_path.exists():
        return False, "", ""
    current = sid
    seen: set[str] = set()
    try:
        con = connect_sessions_db(db_path)
        try:
            for _ in range(max_depth):
                if current in seen:
                    return False, "", ""
                seen.add(current)
                row = con.execute(
                    "SELECT parent_session_id FROM sessions_meta WHERE session_hash=?",
                    (current,),
                ).fetchone()
                if row is None:
                    return False, "", ""
                indexed_parent = str(row["parent_session_id"] or "")
                valid_header, parent = _canonical_gateway_parent(current)
                if not valid_header or parent != indexed_parent:
                    return False, "", ""
                direct_owner = store.gateway_binding_owner(current)
                if direct_owner:
                    return True, direct_owner, ""
                if not parent:
                    return True, "", ""
                current = parent
        finally:
            con.close()
    except Exception:
        return False, "", ""
    return False, "", ""


def _native_gateway_visible(sid: str, owner: str, *, min_level: int) -> bool:
    """Whether ``owner`` may use gateway-native ``sid`` without adopting it."""
    from sharedagents_web import access

    if (
        may_adopt_gateway_session(owner)
        or _configured_native_view_allowed(sid, owner)
        or _owned_gateway_ancestor(sid, owner)
    ):
        return _gw_session_log_exists(sid)
    if access.session_access(owner, sid) >= min_level:
        return _gw_session_log_exists(sid)
    return False


def _configured_native_view_allowed(sid: str, owner: str) -> bool:
    if not is_configured_local_operator(owner):
        return False
    valid, bound_owner, owner_principal = _gateway_lineage_binding(sid)
    if not valid or bound_owner not in ("", owner):
        return False
    if not owner_principal:
        return True
    return owner_principal in (
        "local",
        canonical_gateway_principal(owner),
    )


def _owned_gateway_ancestor(sid: str, owner: str, max_depth: int = 64) -> str:
    """Return an already-owned ancestor of ``sid`` without folding its log.

    This is the authorization preflight for a mini Agent Monitor child click.
    The SQLite index is used only to avoid probing arbitrary raw paths.  Every
    indexed parent edge is then checked against the canonical JSONL
    ``session_started`` header. Unknown, stale, unrelated, cyclic, or malformed
    chains fail closed before Web asks the gateway for history.
    """
    from sharedagents_session.session_index import connect_sessions_db

    db_path = GW_SESSIONS_DIR / "sessions.db"
    if not sid or not owner or not db_path.exists():
        return ""
    current = sid
    seen: set[str] = set()
    try:
        con = connect_sessions_db(db_path)
        try:
            for _ in range(max_depth):
                if current in seen:
                    return ""
                seen.add(current)
                row = con.execute(
                    "SELECT parent_session_id FROM sessions_meta WHERE session_hash=?",
                    (current,),
                ).fetchone()
                if row is None:
                    return ""
                indexed_parent = str(row["parent_session_id"] or "")
                valid_header, parent = _canonical_gateway_parent(current)
                if not valid_header or parent != indexed_parent:
                    return ""
                # Access follows the canonical delegation edge only.
                # ``cost_root_session_id`` is accounting metadata and may be
                # stale or corrupt; it is never an authorization shortcut.
                if parent and store.gateway_binding_owner(parent) == owner:
                    return parent
                if not parent:
                    return ""
                current = parent
        finally:
            con.close()
    except Exception:
        return ""
    return ""


def _gateway_json(resp):
    if resp is None:
        return jsonify(
            {
                "code": "gateway_unreachable",
                "message": "Gateway unreachable",
                "retryable": True,
            }
        ), 502
    try:
        return jsonify(resp.json()), resp.status_code
    except ValueError:
        return jsonify(
            {
                "code": "bad_gateway",
                "message": "Bad gateway response",
                "retryable": True,
            }
        ), 502


def _session_persist_error():
    return jsonify(
        {
            "code": "session_persist_failed",
            "message": "Authoritative session state was not persisted",
            "retryable": True,
        }
    ), 503


def _session_state_payload(sid, *, persist=True, boot=False):
    """Gateway session state for the bootstrap payload (fail-soft)."""
    sdata, owned_sid = _owned_session(sid)
    if sdata is None:
        return None
    kw = dict(_BOOT_GW_KW) if boot else {}
    resp = gateway_request(
        "GET",
        f"/control/v1/orchestrator/sessions/{owned_sid}/state",
        headers=gateway_principal_headers(g.user_email),
        **kw,
    )
    if resp is not None and resp.status_code == 200:
        try:
            payload = resp.json()
            if payload.get("active_agent"):
                sdata["agent"] = payload["active_agent"]
            if payload.get("revision") is not None:
                sdata["revision"] = payload["revision"]
            if persist:
                store._save(sdata)
            return payload
        except ValueError:
            pass
    return {}


@bp.route("/api/sessions/<sid>/state", methods=["GET"])
@require_auth
def api_session_state(sid):
    """Owned facade for the gateway's authoritative session state."""
    payload = _session_state_payload(sid)
    if payload is None:
        return jsonify(
            {
                "code": "session_forbidden",
                "message": "Session not found",
                "retryable": False,
            }
        ), 403
    return jsonify(payload), 200


def _agent_capabilities_payload(sid, *, boot=False):
    """Capability-honest agent/mode discovery for the bootstrap payload (fail-soft).

    Returns ``None`` when the session is not owned; ``{}`` on transient error.
    """
    sdata, owned_sid = _owned_session(sid)
    if sdata is None:
        return None
    kw = dict(_BOOT_GW_KW) if boot else {}
    resp = gateway_request(
        "GET",
        f"/control/v1/orchestrator/sessions/{owned_sid}/capabilities",
        headers=gateway_principal_headers(g.user_email),
        **kw,
    )
    if resp is not None and resp.status_code == 200:
        try:
            return resp.json()
        except ValueError:
            pass
    return {}


@bp.route("/api/sessions/<sid>/capabilities", methods=["GET"])
@require_auth
def api_agent_capabilities(sid):
    """Owned facade for capability-honest agent/mode discovery."""
    payload = _agent_capabilities_payload(sid)
    if payload is None:
        return jsonify(
            {
                "code": "session_forbidden",
                "message": "Session not found",
                "retryable": False,
            }
        ), 403
    return jsonify(payload), 200


_MODE_WORDS = frozenset({"auto", "ask", "plan"})


def _switch_refusal(target_agent, target_model):
    """Why this viewer may not switch to ``target_agent``/``target_model``."""
    from sharedagents_web import access

    if target_agent:
        refusal = access.agent_refusal(g.user_email, target_agent)
        if refusal:
            return refusal
    model = str(target_model or "").strip()
    if model and model.lower() not in _MODE_WORDS:
        return access.model_refusal(g.user_email, model)
    return None


def _agent_default_model(agent):
    """The saved default model of ``agent`` (the gateway def read), or ''."""
    resp = gateway_request(
        "GET",
        f"/control/v1/agents/defs/{agent}",
        headers=gateway_principal_headers(g.user_email),
        timeout=2,
        retry_unavailable=False,
    )
    if resp is None or resp.status_code != 200:
        return ""
    try:
        data = resp.json()
    except ValueError:
        return ""
    return str((data or {}).get("model") or "").strip() if isinstance(data, dict) else ""


def _live_session_model(sid):
    """The model the gateway session actually runs (its index header), or ''."""
    if not sid:
        return ""
    from sharedagents_web.server import _load_meta_headers

    try:
        return str((_load_meta_headers([sid]).get(sid) or {}).get("model") or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def _submit_refusal(row, initial_model="", sid=""):
    """Why this viewer may not send into conversation ``row``, else ``None``.

    The agent must be granted; with restricted model grants, so must the
    model the turn will run on. ``initial_model`` only counts while the
    gateway session does not exist yet (the gateway reads it at creation
    only); after that it is the session's live model, else the stored one,
    else the agent's saved default (fail closed when none resolves). The
    gateway re-checks the executing agent and model under the session lock.
    """
    from sharedagents_web import access

    row = row if isinstance(row, dict) else {}
    sid = str(sid or row.get("id") or "")
    if not row and sid:
        # A TUI/Telegram conversation (no Web row): judge its own agent and
        # model, from the gateway's session index; unknown = refused.
        from sharedagents_web.server import _load_meta_headers

        meta = _load_meta_headers([sid]).get(sid) or {}
        if not meta.get("agent"):
            meta = gw_session_history(sid) or {}
        if not meta.get("agent"):
            return "[access] this conversation's agent is unknown"
        row = {"agent": meta.get("agent"), "model": meta.get("model") or ""}
    agent = str(row.get("agent") or DEFAULT_AGENT).strip()
    refusal = access.agent_refusal(g.user_email, agent)
    if refusal:
        return refusal
    if access.grants_for(g.user_email).is_all("models"):
        return None
    started = bool(sid) and _gw_session_log_exists(sid)
    candidates = (
        (_live_session_model(sid), row.get("model"), row.get("engine_model"))
        if started
        else (initial_model, row.get("model"), row.get("engine_model"))
    )
    model = ""
    for candidate in candidates:
        candidate = str(candidate or "").strip()
        if candidate and candidate.lower() not in _MODE_WORDS:
            model = candidate
            break
    if not model:
        model = _agent_default_model(agent)
    if not model:
        return "choose a model before sending"
    return access.model_refusal(g.user_email, model)


def _refused(message):
    from sharedagents_web import access

    reason = access.human_reason(message)
    return jsonify({"error": reason[:1].upper() + reason[1:], "error_code": "not_granted"}), 403


def _viewer_chat_state(email) -> dict:
    """Whether ``email`` has an agent to talk to; the composer says so when not."""
    from sharedagents_web import access

    if access.is_admin(email) or _first_granted_agent():
        return {"can_chat": True}
    return {
        "can_chat": False,
        "chat_block_reason": "No agents are available to you yet — ask an admin to grant one.",
    }


def _first_granted_agent():
    """The first conversational agent this viewer may use, or ''."""
    from sharedagents_web import access
    from sharedagents_web.agent_visibility import is_non_conversational_agent

    names = access.grants_for(g.user_email).names("agents")
    if names is None:
        try:
            from sharedagents_agent.agent import list_agents

            candidates = [DEFAULT_AGENT] + sorted(list_agents())
        except Exception:  # noqa: BLE001
            candidates = [DEFAULT_AGENT]
    else:
        candidates = sorted(names)
    for name in candidates:
        if is_non_conversational_agent(name):
            continue
        if access.agent_refusal(g.user_email, name) is None:
            return name
    return ""


def _remember_picked_effort(agent, model, effort):
    """Record the picker's effort for ``agent``'s ``model`` (never raises)."""
    if not model or not str(effort or "").strip():
        return
    try:
        from sharedagents_agent.agents.session_lifecycle import remember_model_effort

        remember_model_effort(agent, model, str(effort))
    except Exception as exc:  # noqa: BLE001 — the selection itself still applies
        log.debug("remember picked effort failed: %s", type(exc).__name__)


@bp.route("/api/models/effort", methods=["POST"])
@require_auth
def api_remember_model_effort():
    """Remember the effort the picker applied with a model, outside a session.

    Save on a new chat has no gateway session to run ``/model`` on yet; the
    sheet reports ``{agent, model, effort}`` here and the server keys and
    stores it (``sharedagents_model.effort_prefs``). ``default`` forgets it.
    """
    from sharedagents_web import access

    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "object required"}), 400
    agent = str(data.get("agent") or "").strip()
    model = str(data.get("model") or "").strip()
    effort = str(data.get("effort") or "").strip()
    if not agent or not model or not effort:
        return jsonify({"ok": False, "error": "agent, model and effort required"}), 400
    if access.agent_refusal(g.user_email, agent) or access.model_refusal(g.user_email, model):
        return jsonify({"ok": False, "error": "not granted"}), 403
    from sharedagents_agent.discovery.picker import saved_models
    from sharedagents_command.commands import normalize_effort
    from sharedagents_model.models import derive_effort_capability

    row = next(
        (
            r
            for r in saved_models(agent)
            if model.lower() in {str(r.get(k) or "").strip().lower() for k in ("id", "alias", "wire_id", "model_id")}
        ),
        None,
    )
    level = normalize_effort(effort)
    if row is None or level is None or (level != "default" and level not in derive_effort_capability(row)["values"]):
        return jsonify({"ok": False, "error": "invalid model or effort"}), 400
    _remember_picked_effort(agent, model, effort)
    return jsonify({"ok": True})


@bp.route("/api/sessions/<sid>/agent", methods=["POST"])
@require_auth
def api_switch_agent(sid):
    """Atomically switch the agent assigned to an owned session."""
    sdata, owned_sid = _owned_session(sid)
    if sdata is None:
        return jsonify(
            {
                "code": "session_forbidden",
                "message": "Session not found",
                "retryable": False,
            }
        ), 403
    data = request.get_json(silent=True) or {}
    target_agent = str(data.get("target_agent") or "").strip()
    refusal = _switch_refusal(target_agent, data.get("target_model"))
    if refusal:
        from sharedagents_web import access

        refusal = access.human_reason(refusal)
        refusal = refusal[:1].upper() + refusal[1:]
        return jsonify(
            {
                "code": "not_granted",
                "message": refusal,
                "error": refusal,
                "retryable": False,
            }
        ), 403
    # New Chat has a web row but no gateway conversation yet. Persist the
    # picker intent and wake `event: tree` so the Agent Monitor footer
    # matches the composer without minting a lane. A turnless chat the
    # gateway already holds (a task's pinned-title session) is switched on
    # the gateway: its commands run on the gateway's active agent, so web-only
    # intent made the next `/model` resolve against the old agent.
    if (
        (target_agent or data.get("target_mode") or data.get("target_model") is not None)
        and store.is_unused_web_placeholder(sdata)
        and not _gw_session_log_exists(owned_sid)
    ):
        import re

        if not target_agent and data.get("target_model") is not None:
            tm = str(data.get("target_model") or "").strip()
            cur_agent = str(sdata.get("agent") or DEFAULT_AGENT)
            if cur_agent != "cursor" and tm.lower() in {"auto", "ask", "plan"}:
                tm = ""
            sdata["model"] = tm
        previous_agent = str(sdata.get("pending_agent") or sdata.get("agent") or DEFAULT_AGENT).strip()
        switching_agent = bool(target_agent and target_agent != previous_agent)
        if target_agent:
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", target_agent):
                return jsonify(
                    {
                        "code": "invalid_agent",
                        "message": "Invalid agent name",
                        "retryable": False,
                    }
                ), 400
            sdata["agent"] = target_agent
            sdata["pending_agent"] = target_agent
            # Atomically cleanse or update model when switching agents on an unused placeholder
            target_model = data.get("target_model")
            if target_model is not None:
                tm = str(target_model or "").strip()
                if target_agent != "cursor" and tm.lower() in {"auto", "ask", "plan"}:
                    tm = ""
                sdata["model"] = tm
            elif target_agent != "cursor" and str(sdata.get("model") or "").strip().lower() in {"auto", "ask", "plan"}:
                sdata["model"] = ""
        target_mode = data.get("target_mode")
        if switching_agent:
            # An agent change is not a mode selection. Drop both an explicit
            # mode sent with this request and any mode left on the placeholder;
            # the first turn must use the target agent's advertised default.
            sdata.pop("pending_mode", None)
        elif target_mode:
            sdata["pending_mode"] = target_mode
        if not store._save(sdata):
            return _session_persist_error()

        from sharedagents_web.routes.ui_stream import notify_state_changed

        notify_state_changed(sid)
        return jsonify(
            {
                "active_agent": sdata.get("agent") or target_agent or DEFAULT_AGENT,
                "agent": sdata.get("agent") or target_agent or DEFAULT_AGENT,
                "revision": int(sdata.get("revision") or 0),
            }
        ), 200
    # Model/mode-only selections used to stay on the placeholder path.
    # The gateway needs the conversation's agent explicitly when those
    # turnless conversations already have a session log.
    if not target_agent and (data.get("target_mode") or data.get("target_model") is not None):
        target_agent = str(sdata.get("agent") or DEFAULT_AGENT)
    payload = {
        "target_agent": target_agent,
        "target_mode": data.get("target_mode"),
        "target_model": data.get("target_model"),
        "expected_revision": data.get("expected_revision"),
        "request_id": data.get("request_id"),
        "frontend": "web",
    }
    from sharedagents_web.routes.ui_stream import mark_apply_inflight

    mark_apply_inflight(sdata)
    store._save(sdata)
    resp = gateway_request(
        "POST",
        f"/control/v1/orchestrator/sessions/{owned_sid}/agent",
        json=payload,
        headers=gateway_principal_headers(g.user_email),
    )
    if resp is not None and resp.status_code == 200:
        try:
            result = resp.json()
            if result.get("active_agent"):
                sdata["agent"] = result["active_agent"]
                sdata.pop("pending_agent", None)
            if result.get("revision") is not None:
                sdata["revision"] = result["revision"]
            if "model" in result:
                sdata["model"] = result.get("model") or ""
            elif data.get("target_model") is not None:
                sdata["model"] = str(data.get("target_model") or "").strip()
            if sdata.get("agent") != "cursor" and str(sdata.get("model") or "").strip().lower() in {
                "auto",
                "ask",
                "plan",
            }:
                sdata["model"] = ""
            mark_apply_inflight(sdata)
            if not store._save(sdata):
                return _session_persist_error()
            from sharedagents_web.routes.ui_stream import notify_state_changed

            notify_state_changed(sid)
            return jsonify(result), 200
        except ValueError:
            pass
    return _gateway_json(resp)


def _agent_descriptions() -> dict:
    """`{agent_name: FULL description}` for the picker cards.

    A card must never show a stub or an ellipsis. The
    description is therefore the agent's OWN text, resolved per agent as
    ``config.json`` description → first non-heading ``AGENTS.md`` paragraph →
    delegation-roster line (``agent_profiles.agent_display_descriptions``).

    The roster line is only the LAST fallback: for a custom on-disk agent it is
    cut to 69 chars + ``…`` by ``_custom_on_disk_roster_entries`` for the
    delegation tool-schema budget, and that cut must not reach a card (before
    this, a custom agent's long description rendered as its first 70 chars and
    a built-in external agent rendered a short stub instead of its own text).

    Best-effort: the picker degrades to bare names rather than failing if the
    profile module is unavailable.
    """
    try:
        from sharedagents_agent.agent_profiles import (
            agent_display_descriptions,
            delegation_roster,
            list_profiles,
        )

        out = dict(delegation_roster())
        for prof in list_profiles():
            out.setdefault(prof.name, getattr(prof, "description", "") or "")
        resolved = agent_display_descriptions()
        for name in out:
            full = (resolved.get(name) or {}).get("text") or ""
            if full:
                out[name] = full
        return out
    except Exception:  # noqa: BLE001 — a label is never worth a 500
        return {}


def _agent_origin_flags() -> dict:
    """``{agent: {"origin": …, "external": bool}}`` — ORIGIN first, engine second.

    The picker sections the roster
    Native → User created → External, and a **user-created agent that runs an
    external engine belongs to User created**, not External. Origin therefore
    wins over the engine; ``external`` only decides the third bucket.

    Every input is an EXISTING authoritative predicate — no invented lists:

    * user-created — ``agent_profiles._custom_on_disk_roster_entries()``: the
      live scanner for on-disk agent dirs that are NOT part of the shipped
      roster/profiles (``agent_profiles.py:645``). That is the same set the
      orchestrator already treats as operator specialists, so the picker and
      the delegation roster cannot drift.
    * external — ``external_registry.is_external_engine_kind()``, i.e. the
      ``EXTERNAL_ENGINE_VALUES`` set the runtime itself dispatches on
      (``external_registry.py:285``); the engine string is read from the same
      per-agent ``config.json`` field ``_agent_engines`` uses.
    * subscription — ``subscription_limits.PROVIDER_AGENTS``: the authoritative
      map of subscription provider rails to agents (``subscription_limits.py:48``).
      Subscription plan engines (e.g. MiMo, Codex, Claude) belong with the external
      engines in External, not User created.

    Fallback direction: when the origin scanner cannot classify a name the row
    stays **native** — a shipped agent is never promoted to "user created"
    (mis-promotion is the expensive error; an unknown agent shown under Native
    is merely conservative).
    """
    try:
        from sharedagents_agent.agent_profiles import _custom_on_disk_roster_entries

        user_created = {name for name, _ in _custom_on_disk_roster_entries()}
    except Exception:  # noqa: BLE001 — classification is decorative
        user_created = set()
    try:
        from sharedagents_agent.external_registry import is_external_engine_kind
    except Exception:  # noqa: BLE001
        is_external_engine_kind = None  # type: ignore[assignment]
    try:
        from sharedagents_web.subscription_limits import PROVIDER_AGENTS

        subscription_agents = set(PROVIDER_AGENTS.values())
    except Exception:  # noqa: BLE001
        subscription_agents = set()

    flags: dict = {}
    for name, engine in (_agent_engines() or {}).items():
        is_sub = name in subscription_agents
        try:
            external = bool(
                is_sub
                or (
                    is_external_engine_kind(engine)  # type: ignore[misc]
                    if is_external_engine_kind
                    else False
                )
            )
        except Exception:  # noqa: BLE001
            external = is_sub
        flags[name] = {
            "origin": "native" if is_sub else ("user_created" if name in user_created else "native"),
            "external": external,
        }
    return flags


def _agent_engines() -> dict:
    """`{agent_name: engine kind}` from each on-disk per-agent config.

    Reads only the per-agent ``engine`` field (not ``AgentConfig.load``) so a
    leftover global ``models[]`` cannot leak into classification. Missing or
    unreadable files default to ``local_llm``.
    """
    try:
        from sharedagents_agent.agent import agents_root

        root = agents_root()
    except Exception:  # noqa: BLE001 — grouping is decorative
        return {}
    out: dict[str, str] = {}
    try:
        for path in Path(root).glob("*/config.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(data, dict):
                continue
            name = str(data.get("name") or path.parent.name).strip()
            if not name:
                continue
            engine = str(data.get("engine") or "local_llm").strip() or "local_llm"
            out[name] = engine
    except Exception:  # noqa: BLE001
        return out
    return out


@bp.route("/api/models")
@require_auth
def api_models():
    """List available agents/models from the gateway."""
    resp = gateway_request(
        "GET",
        "/openai/v1/models",
        timeout=2,
        retry_unavailable=False,
    )
    if resp and resp.status_code == 200:
        from sharedagents_web import access

        usage_summary = _bootstrap_usage_summary()
        roster = _agent_roster(resp.json(), usage_summary=usage_summary)
        # A user's picker offers only their granted agents.
        if not access.is_admin(g.user_email):
            roster = [r for r in roster if access.agent_refusal(g.user_email, r["name"]) is None]
        return jsonify({"models": roster, "usage_summary": usage_summary})
    return jsonify({"models": [], "error": "Gateway unreachable"}), 502


def _agent_roster(data, usage_summary=None):
    """Agent picker rows from a gateway /openai/v1/models payload."""
    models = []
    descriptions = _agent_descriptions()
    engines = _agent_engines()
    origins = _agent_origin_flags()
    from sharedagents_web.agent_visibility import is_non_conversational_agent

    agent_paces = (usage_summary or {}).get("agent_paces") or {}
    provider_credits = (usage_summary or {}).get("provider_credits") or {}

    for m in data.get("data", []):
        model_id = m.get("id", "")
        agent_name = bare_id(model_id)
        if is_non_conversational_agent(agent_name):
            continue
        flags = origins.get(agent_name) or {}
        entry = {
            "id": model_id,
            "name": agent_name,
            "label": agent_name,
            # What the agent is FOR. Sourced from the delegation roster the
            # orchestrator itself delegates by, so the picker and the
            # orchestrator describe an agent with the same words.
            "description": descriptions.get(agent_name, ""),
            "engine": engines.get(agent_name, "local_llm"),
            # Sectioning inputs: ORIGIN first, engine second.
            "origin": flags.get("origin", "native"),
            "external": bool(flags.get("external", False)),
        }
        # Local models do not have subscription pace or prepaid API credits
        is_local_model = (
            agent_name in descriptions
            and engines.get(agent_name, "local_llm") == "local_llm"
            and not flags.get("external", False)
        )
        if not is_local_model:
            if agent_name in agent_paces:
                entry["pace"] = agent_paces[agent_name]
            if agent_name in provider_credits:
                entry["credits"] = provider_credits[agent_name]
        # External engines carry their CLI version and an auth alert so the
        # selector instructs "sign in" instead of failing the first turn.
        # Cache-only projection — /api/models never blocks on a CLI probe.
        if flags.get("external", False):
            try:
                from sharedagents_web import agent_health

                health = agent_health.picker_projection(agent_name)
            except Exception:  # noqa: BLE001 — a chip is never worth a 500
                health = {}
            if health:
                entry["version"] = health.get("version", "")
                entry["auth_state"] = health.get("auth_state", "")
                entry["auth_hint"] = health.get("auth_hint", "")
                entry["installed"] = bool(health.get("installed", True))
                entry["auto_auth"] = bool(health.get("auto_auth", False))
        models.append(entry)
    return models


def _picker_spec(agent):
    from sharedagents_agent.discovery.picker import engine_spec

    return engine_spec(agent)


def _with_effort(rows, agent="", sid=""):
    """Stamp capabilities, memory, and the resolved display level on rows.

    Session overrides apply only to that session's agent and current model.
    The browser renders ``effort_selected`` without resolving precedence.
    """
    from sharedagents_model import effort_prefs
    from sharedagents_model.models import derive_effort_capability

    for m in rows or []:
        if isinstance(m, dict) and "effort" not in m and "reasoning" not in m:
            m["effort"] = derive_effort_capability(m)
    try:
        from sharedagents_agent.agent import AgentConfig

        try:
            cfg = AgentConfig.load(agent, ambient_discovery=False)
        except (OSError, ValueError):
            cfg = None
        provider = effort_prefs.engine_provider(cfg, agent)
        effort_prefs.stamp_rows(rows, provider)
        header = {}
        if sid:
            from sharedagents_web import access

            if access.session_access(g.user_email, sid) >= access.VIEW:
                from sharedagents_session.session_reader import load_session_dict

                header = load_session_dict(GW_SESSIONS_DIR, sid, fold=False) or {}
        same_agent = (header.get("agent_name") or header.get("agent")) == agent
        for row in rows or []:
            if isinstance(row, dict):
                row["effort_selected"] = effort_prefs.selected_level_for(
                    row,
                    getattr(cfg, "effort", ""),
                    engine_provider=provider,
                    session_model=str(header.get("engine_model") or header.get("model") or "") if same_agent else "",
                    session_effort=str(header.get("effort") or "") if same_agent else "",
                    mode=(header.get("system_mode") if same_agent else None) or getattr(cfg, "permission_mode", "auto"),
                )
    except Exception as exc:  # noqa: BLE001 — a missing preference hides nothing
        log.debug("effort_pref stamp failed: %s", type(exc).__name__)
    return rows


def _picker_row_filter(email):
    """``row_filter`` for the picker module: ``None`` (everything) for admins."""
    from sharedagents_web import access

    if access.is_admin(email):
        return None
    return lambda agent, rows: access.filter_model_rows(email, agent, rows)


@bp.route("/api/agents/picker")
@require_auth
def api_agents_picker():
    """What the Agent & Model picker shows when it opens — and nothing more.

    Every agent the account may use (the ``/api/models`` roster), each with
    its DEFAULT model row and its provider tabs. A tab's ``status`` is
    ``ready`` or ``error`` (with ``error.code`` ``no_key``), decided from
    configuration only. No model provider is contacted: a provider's models
    are requested by ``POST /api/models/discover`` with ``provider`` when its
    tab is opened.

    ``?agent=<name>`` answers for that one agent (still access-checked).
    """
    started = time.perf_counter()
    from sharedagents_agent.discovery import picker
    from sharedagents_web import access

    wanted = bare_id(request.args.get("agent") or "")
    roster = []
    if wanted:
        if access.agent_refusal(g.user_email, wanted):
            return jsonify({"ok": True, "agents": [], "server_ms": 0})
        roster = [{"name": wanted}]
    else:
        resp = gateway_request(
            "GET",
            "/openai/v1/models",
            timeout=2,
            retry_unavailable=False,
        )
        if not resp or resp.status_code != 200:
            return jsonify({"ok": False, "agents": [], "error": "Gateway unreachable"}), 502
        roster = _agent_roster(resp.json())
        if not access.is_admin(g.user_email):
            roster = [r for r in roster if access.agent_refusal(g.user_email, r["name"]) is None]
    summaries = {
        s["name"]: s
        for s in picker.picker_summary(
            [r["name"] for r in roster],
            row_filter=_picker_row_filter(g.user_email),
        )
    }
    agents = []
    for row in roster:
        summary = summaries.get(row["name"]) or {
            "default": None,
            "defaults": {},
            "providers": [],
        }
        if summary.get("default"):
            _with_effort([summary["default"]], row["name"], str(request.args.get("session_id") or ""))
        agents.append(
            {
                **row,
                **{
                    "default": summary.get("default"),
                    "defaults": summary.get("defaults") or {},
                    "providers": summary.get("providers") or [],
                },
            }
        )
    elapsed = round((time.perf_counter() - started) * 1000.0, 1)
    return jsonify({"ok": True, "agents": agents, "server_ms": elapsed})


def _discover_provider(agent, provider, force, sid):
    """``POST /api/models/discover`` with ``provider``: that tab's rows only."""
    from sharedagents_agent.discovery import picker
    from sharedagents_web import access
    from sharedagents_web.routes.ui_stream import notify_state_changed

    started = time.perf_counter()
    agent = bare_id(agent)
    admin = access.is_admin(g.user_email)
    if not admin and access.agent_refusal(g.user_email, agent):
        return jsonify(
            {
                "ok": True,
                "agent": agent,
                "provider": provider,
                "models": [],
                "providers": [],
                "loaded": [],
                "error": None,
                "degraded": False,
            }
        )
    try:
        # A user is served the saved rows; only an admin's click asks the
        # provider itself (the same rule as the forced refresh).
        result = picker.provider_models(
            agent,
            provider,
            force=bool(force and admin),
            allow_live=admin,
            row_filter=_picker_row_filter(g.user_email),
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("provider_models(%s, %s) failed (%s)", agent, provider, type(exc).__name__)
        result = {
            "agent": agent,
            "provider": provider,
            "models": [],
            "providers": [],
            "loaded": [],
            "degraded": True,
            "error": {"code": "unreachable", "message": f"{provider}: the model list could not be loaded"},
        }
    _with_effort(result.get("models"), agent, sid)
    # No ``models`` broadcast: one provider's rows are a partial list and no
    # open tab may replace its catalogue with them. Other tabs load the saved
    # data on their next picker or provider request.
    if sid:
        try:
            notify_state_changed(sid)
        except Exception as exc:  # noqa: BLE001
            log.debug("notify_state_changed failed: %s", exc)
    result["ok"] = True
    result["server_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
    return jsonify(result)


@bp.route("/api/models/discover", methods=["POST"])
@require_auth
def api_discover_models():
    """Discover models for an agent with file locking and stream to webui.

    With ``provider`` (the chat picker): only that provider tab is loaded and
    only that provider is asked — see :func:`_discover_provider`. Without it
    (other pages' agent/model selects): the agent's whole list as before.

    On ANY error, falls back to an empty list [] flagged ``degraded`` so the
    webui and session never collapse and the picker keeps its good list.
    """
    agent = ""
    try:
        data = request.get_json(silent=True) or {}
        agent = bare_id(data.get("agent") or "")
        force = bool(data.get("force", False))
        sid = ""
        try:
            sid = str(data.get("session_id") or session.get("current_session_id") or "").strip()
        except Exception:
            pass

        if not agent:
            return jsonify({"ok": False, "error": "agent required", "models": []}), 400
        if "provider" in data:
            provider = str(data.get("provider") or "").strip()
            if not provider:
                return jsonify({"ok": False, "error": "provider required", "models": []}), 400
            return _discover_provider(agent, provider, force, sid)

        from sharedagents_agent.discovery import cache_lock
        from sharedagents_web import access
        from sharedagents_web.routes.ui_stream import broadcast_models_event, notify_state_changed

        # A user sees only granted agents' granted models and may not force
        # a live provider probe.
        admin = access.is_admin(g.user_email)
        if not admin:
            force = False
            if access.agent_refusal(g.user_email, agent):
                return jsonify({"ok": True, "agent": agent, "models": []})
        # ``degraded``: discovery failed and only the provider's static
        # fallback (or nothing) is served. The picker must not let such a
        # list replace a good one or drop the operator's pick.
        try:
            if admin:
                models, degraded = cache_lock.discover_agent_models_status(agent, force=force)
            else:
                from sharedagents_agent.discovery.picker import saved_models

                models, degraded = saved_models(agent), False
            if isinstance(models, list):
                from sharedagents_agent.discovery.picker import public_model_rows, stamp_tabs

                models = _with_effort(
                    public_model_rows(stamp_tabs(agent, models, _picker_spec(agent))),
                    agent,
                )
        except Exception as exc:
            log.warning("discover_agent_models(%s) failed (%s)", agent, type(exc).__name__)
            models, degraded = [], True

        # Every open tab applies a broadcast as its list, so only a real one goes.
        if not degraded:
            try:
                broadcast_models_event(agent, models)
            except Exception as exc:
                log.debug("broadcast_models_event failed: %s", exc)

        if sid:
            try:
                notify_state_changed(sid)
            except Exception as exc:
                log.debug("notify_state_changed failed: %s", exc)

        rows = models if isinstance(models, list) else []
        if not admin:
            rows = access.filter_model_rows(g.user_email, agent, rows)
        _with_effort(rows, agent, sid)
        return jsonify({"ok": True, "agent": agent, "models": rows, "degraded": bool(degraded)})
    except Exception as exc:
        log.warning("api_discover_models exception for %s (%s)", agent, type(exc).__name__)
        return jsonify({"ok": True, "agent": agent, "models": [], "degraded": True})


@bp.route("/api/sessions", methods=["GET"])
@require_auth
def api_list_sessions():
    """List chat sessions — web-store conversations PLUS gateway-native ones.

    The web's own conversations are merged with the gateway's top-level
    sessions so a conversation started in the TUI (which lives only in the
    gateway) is observable here too. Dedup is by the bound gateway hash: a web
    session already represented by the same SID masks that gateway row, so each
    conversation appears once. Gateway-only rows are tagged ``origin:gateway``.
    """
    agent = request.args.get("agent")
    try:
        limit = int(request.args.get("limit", 50))
    except (TypeError, ValueError):
        limit = 50
    return jsonify({"sessions": _session_list_rows(g.user_email, agent, limit)})


def _session_list_rows(owner, agent=None, limit=50, *, gateway_roots=None):
    """``GET /api/sessions`` rows for *owner*, newest activity first.

    ``gateway_roots`` lets the chat boot reuse its one gateway roots read.
    """
    sessions = _merged_sessions(owner, agent, limit=limit, gateway_roots=gateway_roots)
    sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return sessions[:limit]


@bp.route("/api/sessions", methods=["POST"])
@require_auth
def api_create_session():
    """Create a new chat session for the agent selected by the web control.

    Traceability:
    """
    from sharedagents_web.agent_visibility import conversational_agent_or_default

    data = request.get_json() or {}
    agent = conversational_agent_or_default(
        data.get("agent", DEFAULT_AGENT),
        DEFAULT_AGENT,
    )
    model = data.get("model", "")
    if agent != "cursor" and str(model or "").strip().lower() in {"auto", "ask", "plan"}:
        model = ""
    from sharedagents_web import access

    if not access.is_admin(g.user_email):
        # A stale picker preference must not open a chat on an agent/model
        # this person was not granted: fall back to a granted agent and the
        # agent's own default model.
        if access.agent_refusal(g.user_email, agent):
            agent = _first_granted_agent()
            if not agent:
                return _refused("no agent is granted to your account")
        if model and access.model_refusal(g.user_email, model):
            model = ""
    topic = data.get("topic", "")
    topic = str(topic).strip()[:120] if topic else ""  # coerce + cap length
    if data.get("reserve"):
        sdata, discarded = store.reserve_web_chat(
            g.user_email,
            agent,
            model,
            topic,
        )
    else:
        sdata = store.create(g.user_email, agent, model, topic)
        discarded = []
    # The page keeps this as the open conversation: without the access and
    # sharing fields its header hides the privacy control.
    sdata = _with_access(sdata, sdata["id"])
    return jsonify({"session": sdata, "discarded": discarded}), 201


@bp.route("/api/boot", methods=["GET"])
@require_auth
def api_boot():
    """The chat page's ONE boot read.

    Returns the session (``null`` when absent/unresolvable), gateway health,
    pending approvals, and the session state/capabilities the page seeds on
    open. ``session`` may be empty — the client then opens a New Chat draft.

    An untouched chat page makes two API calls: this read and the session
    stream (owner rule). So this read also carries what the page
    would otherwise fetch on its own:

    * ``agent_defaults`` — the SAVED default ``{effort, model}`` of the
      picker agent (``?agent=``) and of the resumed session's agent, as
      ``GET /api/agents/defs/<agent>`` reads them;
    * ``sidebar`` — only when ``?lists=conversations,topics`` says the left
      menu is already visible at load: the rows ``GET /api/sessions?limit=20``
      and ``GET /api/topics`` would return. A menu opened later (a click)
      still loads them lazily.

    The agent roster and model catalogues stay lazy: the Agent Selector
    loads them when the operator opens it.
    """
    from sharedagents_web import access

    sid = str(request.args.get("session") or "").strip()
    session = None
    # ``?session=`` is not a gated path argument: a conversation the viewer
    # cannot reach boots as "no session" (a New Chat draft), never a leak.
    if sid and access.session_access(g.user_email, sid) == access.NONE:
        sid = ""
    if sid:
        resp = _session_payload(sid)
        # _session_payload may return a (response, status) tuple for the
        # not-found / forbidden cases; a non-200 means "no session to open".
        if isinstance(resp, tuple):
            resp = resp[0]
        if resp.status_code == 200:
            session = (resp.get_json() or {}).get("session")

    def _soft(fn, fallback, *args, **kwargs):
        try:
            value = fn(*args, **kwargs)
            return fallback if value is None else value
        except Exception:
            log.exception("boot extra %s failed", getattr(fn, "__name__", fn))
            return fallback

    from sharedagents_web import server as _web_server

    viewer_email = str(g.user_email or "").strip().lower()
    payload = {
        "session": session,
        # Who is looking: the page hides operator-only
        # controls for a user. Never an access decision on its own.
        "viewer": {
            "email": viewer_email,
            "name": _web_server.auth_store.display_name(viewer_email),
            "role": "admin" if access.is_admin(viewer_email) else "user",
            **_viewer_chat_state(viewer_email),
        },
        "agents": [],
        "health": _soft(_bootstrap_health, {"healthy": False}),
        "approvals": _soft(_bootstrap_approvals, [], sid),
        "agent_defaults": _soft(
            _bootstrap_agent_defaults,
            {},
            request.args.get("agent"),
            session,
        ),
        "usage_summary": _soft(_bootstrap_usage_summary, None),
    }
    lists = _boot_lists(request.args.get("lists"))
    if lists:
        payload["sidebar"] = _soft(_bootstrap_sidebar, {}, g.user_email, lists)
    if session:
        payload["goal"] = session.get("goal")
        payload["budget"] = session.get("budget")
        payload["footer"] = session.get("footer")
        payload["session_state"] = _soft(
            _session_state_payload,
            {},
            sid,
            persist=False,
            boot=True,
        )
        payload["agent_capabilities"] = _soft(
            _agent_capabilities_payload,
            {},
            sid,
            boot=True,
        )
    return jsonify(payload)


@bp.route("/api/sessions/<sid>", methods=["GET"])
@require_auth
def api_get_session(sid):
    """Get a session and its messages.

    Three cases, all of which must surface the *recent history* a click in the
    sidebar (or a monitor deep-link) expects:

    1. A submitted web conversation: the gateway
       owns the transcript, so we refresh messages from the session log — this
       is what makes a TUI-continued or cross-frontend conversation *entirely
       observable* on the web (every turn shows, not just the web-typed ones).
    2. A plain web conversation (never bound): return the stored messages.
    3. A gateway-native hash NOT in the web store (e.g. a TUI session or a
       subagent session double-clicked in the monitor): synthesize a session
       view straight from the gateway history. It is adopted into the store on
       the first chat/command (see ``api_chat``), not merely by viewing it.
    """
    return _session_payload(sid)


#: Oldest cached reading whose pace the agent selector still shows: two pace
#: sampler slots, so one missed sample does not blank the tiles.
_SELECTOR_PACE_MAX_AGE_S = 1800.0


def _reading_pace(sl, card, now):
    """Pace of one cached card as of its own reading time, or ``{}``.

    The picker never probes vendors, it reads the cache the 15-minute pace
    sampler writes. Paced at the current time, that reading counts as stale
    after 10 minutes and the tiles lose their pace for a third of every slot.
    Pacing it at the moment it was read gives the pace that reading measured.
    A reading older than ``_SELECTOR_PACE_MAX_AGE_S`` has no pace. A limit
    that reset after the reading no longer holds what was read (a hit that
    has cleared, a pace taken just before its reset), so it is left out, as
    live pace leaves out a reset that already passed (``stale_reset``). Pure:
    nothing is written (no pace-history point per page load).
    """
    if str(card.get("id") or "") not in sl.PROVIDER_IDS:
        return {}
    read_at = sl._parse_iso_epoch(card.get("updated_at"))
    if read_at is None or not 0 <= now - read_at <= _SELECTOR_PACE_MAX_AGE_S:
        return {}
    windows = [
        window
        for window in card.get("windows") or []
        if not (isinstance(window, dict) and read_at < (sl._parse_iso_epoch(window.get("resets_at")) or 0.0) <= now)
    ]
    return sl.card_pace({**card, "windows": windows}, now=read_at)


def _agent_health_summary() -> dict:
    """Cache-only external-agent health map for the chat picker rows.

    Never probes: the boot payload and ``/api/models`` answer from the last
    scan; the first paint of a fresh server triggers the scan in the
    background and catches up on the next boot.
    """
    try:
        from sharedagents_web import agent_health

        return agent_health.picker_chips_map()
    except Exception:  # noqa: BLE001 — a chip is never worth a 500
        return {}


def _bootstrap_usage_summary():
    """Summary of subscription pace and API credits from cache, never blocking."""
    try:
        from sharedagents_web import subscription_limits as sl

        cache = sl._read_cache()
        if not cache:
            health = _agent_health_summary()
            return {"agent_health": health} if health else None
        providers = [p for p in cache.get("providers") or [] if isinstance(p, dict)]
        now = sl._pace_now()
        agent_paces = {}
        best_pace = None
        best_pace_agent = None
        total_api_usd = 0.0
        provider_credits = {}

        for p in providers:
            pid = p.get("id")
            agent = sl.PROVIDER_AGENTS.get(pid, pid)
            paced = _reading_pace(sl, p, now)
            pace = paced.get("pace_used_percent")
            status = paced.get("pace_status", "ok")
            hit = bool(paced.get("pace_limit_hit"))
            if pace is not None:
                pace_info = {
                    "pace": round(float(pace), 1),
                    "status": status,
                    "hit": hit,
                }
                agent_paces[agent] = pace_info
                if pid:
                    agent_paces[pid] = pace_info
                if best_pace is None or pace < best_pace:
                    best_pace = round(float(pace), 1)
                    best_pace_agent = agent
            for w in p.get("windows", []):
                wid = str(w.get("id") or "").lower()
                unit = str(w.get("unit") or "").upper()
                rem = w.get("remaining")
                if wid in ("credits", "balance", "available") and unit == "USD" and rem is not None:
                    total_api_usd += float(rem)
                    provider_credits[pid] = round(float(rem), 2)
                    if agent:
                        provider_credits[agent] = round(float(rem), 2)

        return {
            "best_pace": best_pace,
            "best_pace_agent": best_pace_agent,
            "agent_paces": agent_paces,
            "total_api_usd": round(total_api_usd, 2) if total_api_usd > 0 else None,
            "provider_credits": provider_credits,
            "agent_health": _agent_health_summary(),
        }
    except Exception:
        log.exception("failed to extract usage summary for boot")
        return None


def _bootstrap_health():
    """Gateway liveness for the conn badge, fail-soft."""
    resp = gateway_request("GET", "/", timeout=_BOOT_GW_TIMEOUT, retry_unavailable=False)
    return {"healthy": bool(resp is not None and resp.status_code == 200)}


def _boot_agent_names(requested, session):
    """Agents whose saved defaults the booting page renders: the picker's
    (``?agent=``) and the resumed session's own.

    ``?agent=`` is spliced into a gateway path, so only names the gateway
    itself accepts as agent names get through.
    """
    from sharedagents_agent.agent_editor import AgentEditorError, validate_agent_name

    names = []
    for raw in (requested, (session or {}).get("agent")):
        name = str(raw or "").strip()
        if not name or name in names:
            continue
        try:
            validate_agent_name(name)
        except AgentEditorError:
            continue
        names.append(name)
    # A user only gets defaults for agents granted to them.
    from sharedagents_web import access

    return [n for n in names if access.agent_refusal(g.user_email, n) is None]


def _bootstrap_agent_defaults(requested, session):
    """Saved default ``{effort, model}`` per agent the page renders, fail-soft.

    The same gateway read ``GET /api/agents/defs/<agent>`` proxies —
    ``read_definition``'s RESOLVED values (the per-agent override, else the
    inherited global model/effort or the engine's own default) — cut down to
    the two fields the composer seeds from. An agent the
    gateway does not answer for is left out; the page then shows the model
    row's own default, as it did after a failed def read.
    """
    names = _boot_agent_names(requested, session)
    if not names:
        return {}
    headers = gateway_principal_headers(g.user_email)
    out = {}
    for name in names:
        resp = gateway_request(
            "GET",
            f"/control/v1/agents/defs/{name}",
            headers=headers,
            **_BOOT_GW_KW,
        )
        if resp is None or resp.status_code != 200:
            continue
        try:
            data = resp.json()
        except ValueError:
            continue
        if isinstance(data, dict):
            out[name] = {"effort": data.get("effort"), "model": data.get("model")}
    return out


# The chat sidebar's lazy Conversations page: loadSessionList({limit: 20}).
_SIDEBAR_SESSIONS_LIMIT = 20
_BOOT_LISTS = ("conversations", "topics")


def _boot_lists(raw):
    """The sidebar lists a ``?lists=`` value asks for, in canonical order."""
    wanted = {part.strip() for part in str(raw or "").split(",")}
    return [name for name in _BOOT_LISTS if name in wanted]


def _bootstrap_sidebar(owner, lists):
    """The left menu's lists, for a page whose menu is visible at load.

    Exactly what the lazy loaders fetch — ``GET /api/sessions?limit=20`` and
    ``GET /api/topics`` — from ONE gateway roots read: ``gw_root_sessions`` is
    newest-first, so the 20-row page is a prefix of the topics scan.
    """
    from sharedagents_web import server as _server
    from sharedagents_web.routes.topics import _topics_for

    roots = None
    from sharedagents_web import access

    if access.is_admin(owner) or access.visible_shared_ids(owner):
        roots = _server.gw_root_sessions(
            limit=(_server.TOPICS_SCAN_LIMIT if "topics" in lists else _SIDEBAR_SESSIONS_LIMIT),
        )
    out = {}
    if "conversations" in lists:
        out["sessions"] = _session_list_rows(
            owner,
            None,
            _SIDEBAR_SESSIONS_LIMIT,
            gateway_roots=roots,
        )
    if "topics" in lists:
        out["topics"] = _topics_for(owner, compact=True, gateway_roots=roots)["topics"]
    return out


def _map_gateway_approvals(
    raw: list,
    owner: str,
    *,
    extra_session_ids: tuple[str, ...] | list[str] = (),
) -> list[dict]:
    from sharedagents_web.approvals_map import map_gateway_approvals

    return map_gateway_approvals(
        raw,
        store,
        owner,
        extra_session_ids=extra_session_ids,
    )


def _bootstrap_approvals(*extra_session_ids: str):
    """Pending decisions for the sidebar panel, fail-soft."""
    resp = gateway_request(
        "GET",
        "/control/v1/orchestrator/approvals",
        timeout=_BOOT_GW_TIMEOUT,
        retry_unavailable=False,
    )
    if resp is None or resp.status_code != 200:
        return []
    try:
        raw = resp.json().get("approvals") or []
    except ValueError:
        raw = []
    return _map_gateway_approvals(
        raw,
        g.user_email,
        extra_session_ids=extra_session_ids,
    )


def _with_access(payload, sid):
    """Stamp the viewer's access, the owner and the sharing onto a session.

    View-only access reuses the monitor read-only lock the composer already
    honours.
    """
    from sharedagents_web import access

    info = access.session_view(g.user_email, sid, payload)
    out = dict(payload)
    reason = info.pop("read_only_reason", "")
    out.update(info)
    out["monitor_read_only"] = info.get("access") == "view"
    if reason:
        out["read_only_reason"] = reason
    from sharedagents_agent.compaction import is_internal_agent

    if is_internal_agent(out.get("agent")):
        # compaction / title / summary: the operator reads the turn, never
        # continues it (the gateway refuses the write as well).
        out["monitor_read_only"] = True
        out["read_only_reason"] = f"{out.get('agent')} session — view only"
    return out


def _session_payload(sid):
    """Get a session and its messages.

    Three cases, all of which must surface the *recent history* a click in the
    sidebar (or a monitor deep-link) expects:

    1. A submitted web conversation: the gateway
       owns the transcript, so we refresh messages from the session log — this
       is what makes a TUI-continued or cross-frontend conversation *entirely
       observable* on the web (every turn shows, not just the web-typed ones).
    2. A plain web conversation (never bound): return the stored messages.
    3. A gateway-native hash NOT in the web store (e.g. a TUI session or a
       subagent session double-clicked in the monitor): synthesize a session
       view straight from the gateway history. It is adopted into the store on
       the first chat/command (see ``api_chat``), not merely by viewing it.
    """
    # gateway log, pages history, and admits a canonical delegated descendant
    # only when its ancestor is already bound to this Web owner.
    before = request.args.get("before", type=int)
    limit = request.args.get("limit", type=int)
    monitor_view = str(request.args.get("monitor") or "").lower() in {
        "1",
        "true",
        "yes",
    }
    # Look up the goal_id and detail for this session so the chat widget can pause/resume and show metrics.
    # Skip on reverse-history pages (?before=) — pagination must not scan the
    # whole goal registry on every scroll-up.
    goal_id = None
    goal = None
    budget = None
    if before is None:
        try:
            from sharedagents_goal.goal_registry import get_goal_registry

            goal_id = get_goal_registry().chat_goal_id_for_root(sid) or None
            if goal_id:
                from sharedagents_web.routes.goals import _web_goal_detail

                goal = _web_goal_detail(goal_id)
            from sharedagents_goal.goal_service import hierarchical_budget_view

            budget = hierarchical_budget_view(sid)
        except Exception:
            pass

    footer = None
    _sdata = None
    if before is None:
        try:
            from sharedagents_web.agent_monitor_footer import footer_payload, unbound_footer_payload
            from sharedagents_web.server import _fetch_tree

            _sdata = store.get(sid, g.user_email)
            if _sdata and store.is_unused_web_placeholder(_sdata):
                _agent = store.display_agent(_sdata)
                footer = unbound_footer_payload(
                    _agent,
                    str(_sdata.get("title") or ""),
                    str(_sdata.get("model") or _sdata.get("engine_model") or ""),
                )
            else:
                _tree = _fetch_tree(
                    False,
                    session_id=sid,
                    headers=gateway_principal_headers(g.user_email),
                    retry_unavailable=False,
                )
                footer = footer_payload(_tree, sid) if _tree is not None else {"rows": [], "subs": 0}
        except Exception:
            footer = {"rows": [], "subs": 0}
        try:
            from sharedagents_goal.goal_service import stamp_budget_from_goal

            budget = stamp_budget_from_goal(budget, goal)
        except Exception:
            pass
        try:
            from sharedagents_web.agent_monitor_footer import (
                enrich_footer_with_history,
                stamp_footer_from_web_row,
            )

            footer = enrich_footer_with_history(footer, sid, GW_SESSIONS_DIR)
            if _sdata:
                footer = stamp_footer_from_web_row(
                    footer,
                    _sdata,
                    sid,
                    title=resolve_authoritative_session_title(
                        sid,
                        str(_sdata.get("title") or ""),
                    ),
                )
        except Exception:
            pass

    # ``monitor=1`` opens a delegated session without changing ownership.
    sdata = None if monitor_view else store.get(sid, g.user_email)
    if sdata is None and not monitor_view:
        from sharedagents_web import access

        if access.session_access(g.user_email, sid) >= access.VIEW:
            sdata = store.get(sid)
    if sdata:
        gw = _stream_gateway_sid(sid, sdata)
        if gw:
            hist = _paged_gateway_history(gw, before, limit)
            if hist is not None:
                # An apply may have completed while history was being fetched.
                sdata = store.get(sid, g.user_email) or store.get(sid) or sdata
                merged = dict(sdata)
                merged["messages"] = hist.get("messages") or []
                # The gateway owns title lifecycle (including the temporary
                # timestamp and async 3–6-word refresh). Project it into the
                # WebUI response so a reload never falls back to stale SQLite
                # metadata while the gateway log is already authoritative.
                from sharedagents_session.session_title import is_default_title

                hist_title = str(hist.get("title") or "").strip()
                if hist_title and not is_default_title(hist_title):
                    merged["title"] = hist_title
                merged["history_total"] = hist.get("history_total")
                merged["before_cursor"] = hist.get("before_cursor")
                merged["history_truncated"] = bool(hist.get("history_truncated"))
                merged["tail_continuation"] = bool(hist.get("tail_continuation"))
                # Authoritative resume offset from the gateway's single-scan
                # fold (completed-turns page + in-flight-turn offset). No more
                # racy stat-fold-stat: an in-flight turn no longer forces a
                # full from=0 replay on a joining/monitor viewer.
                # Never null-for-a-log: a null watermark makes the client open
                # the stream at from=0 and replay the whole transcript over SSE
                # on top of the bounded page. See ``_tail_watermark``.
                merged["tail_from_offset"] = _tail_watermark(hist, gw)
                merged["active"] = bool(hist.get("active"))
                # Waiting parent / live descendants — sticky Stop on reload.
                merged["state"] = hist.get("session_state") or "idle"
                merged["tree_busy"] = bool(hist.get("tree_busy") or hist.get("session_state") == "waiting")
                merged["engine_model"] = hist.get("engine_model") or sdata.get("engine_model", "")
                merged["engine_model_alias"] = hist.get("engine_model_alias") or sdata.get("engine_model_alias", "")
                # Context budget for the footer (used / max) — straight from the
                # gateway session header; the web stays a pure renderer.
                merged["context_window"] = hist.get("context_window") or 0
                merged["last_prompt_tokens"] = hist.get("last_prompt_tokens") or 0
                # Manual-compaction stats for the footer widget — the gateway
                # header is the source of truth (persists across reloads).
                merged["compaction_count"] = hist.get("compaction_count") or 0
                merged["compaction_summary"] = hist.get("compaction_summary") or ""
                merged["compaction_last_stats"] = hist.get("compaction_last_stats") or {}
                # The gateway owns the session's real model — reconcile it so the
                # picker reflects the ACTUAL model, not a stale value the web store
                # captured at creation (which could disagree, e.g. stored "local"
                # while the gateway default ran "deepseek-v4-flash").
                from sharedagents_web.routes.ui_stream import _apply_inflight

                hist_agent = str(hist.get("agent") or hist.get("agent_name") or "").strip()
                hist_model = str(hist.get("model") or "").strip()
                row_changed = False
                inflight = _apply_inflight(sdata)
                if hist.get("model") and not inflight:
                    merged["model"] = hist.get("model")
                if hist_agent and not store.is_unused_web_placeholder(sdata) and not inflight:
                    merged["agent"] = hist_agent
                    if str(sdata.get("agent") or "").strip() != hist_agent:
                        sdata["agent"] = hist_agent
                        sdata.pop("pending_agent", None)
                        row_changed = True
                    if hist_model and str(sdata.get("model") or "").strip() != hist_model:
                        sdata["model"] = hist_model
                        row_changed = True
                else:
                    merged["agent"] = store.display_agent(sdata)
                if row_changed:
                    store._save(sdata)
                merged["goal_id"] = goal_id
                merged["goal"] = goal
                merged["budget"] = budget
                merged["footer"] = footer
                return jsonify({"session": _trim_session_payload(_with_access(merged, sid))})
        # Plain web conversation (no gateway log): page the stored messages; no
        # tail watermark → client falls back to a full (but small) replay.
        page, before_cursor, total = _page_history(sdata.get("messages", []), before, limit)
        out = dict(sdata)
        out["messages"] = page
        out["history_total"] = total
        out["before_cursor"] = before_cursor
        out["agent"] = store.display_agent(sdata)
        out["goal_id"] = goal_id
        out["goal"] = goal
        out["budget"] = budget
        out["footer"] = footer
        return jsonify({"session": _trim_session_payload(_with_access(out, sid))})

    # Not a web conversation — is it a gateway-native session (TUI / monitor)?
    #
    # A non-local Web principal must not enumerate arbitrary gateway sessions.
    # It may, however, view a delegated child whose canonical log
    # points back to a gateway id already bound to this same Web owner.  This
    # is the chat-docked Agent Monitor path: parent Web conversation -> child
    # subagent.  The lineage comes from the gateway session log, never from a
    # browser query parameter.
    from sharedagents_web import access

    local_adoption = may_adopt_gateway_session(g.user_email)
    configured_native_view = _configured_native_view_allowed(
        sid,
        g.user_email,
    )
    # Single-user: a binding recorded under a DIFFERENT
    # identity is a historical artifact, not an access boundary. The loopback
    # `debug@localhost` identity had adopted 99 gateway-native sessions, so the
    # operator opening their own CLI/TUI work got "Session not found".
    _bound_owner = store.gateway_binding_owner(sid)  # noqa: F841 — kept for logs
    verified_ancestor = ""
    if not local_adoption:
        verified_ancestor = _owned_gateway_ancestor(sid, g.user_email)
        shared = access.session_access(g.user_email, sid) >= access.VIEW and _gw_session_log_exists(sid)
        if not verified_ancestor and not configured_native_view and not shared:
            return jsonify({"error": "Session not found"}), 403
    hist = _paged_gateway_history(sid, before, limit)
    if hist is None:
        return jsonify({"error": "Session not found"}), 403
    if (
        local_adoption
        and not monitor_view
        and store.get(sid, g.user_email) is None
        and not str(hist.get("parent_session_id") or "")
    ):
        # Preserve the established local/TUI deep-link behavior: the machine
        # owner may adopt a native session and continue it. Only an isolated
        # Web user's delegated child is the read-only monitor view.
        try:
            store.adopt_gateway(
                g.user_email,
                sid,
                hist,
                gateway_principal=canonical_gateway_principal(g.user_email),
            )
        except PermissionError:
            return jsonify({"error": "Session not found"}), 403
        hist = _paged_gateway_history(sid, before, limit) or hist
    elif not local_adoption:
        valid_header, canonical_parent = _canonical_gateway_parent(sid)
        reported_parent = str(hist.get("parent_session_id") or "")
        if not valid_header or (reported_parent and reported_parent != canonical_parent):
            # The gateway response must not contradict the immutable log
            # header. ``cost_root_session_id`` is accounting data, never ACL.
            return jsonify({"error": "Session not found"}), 403
    return jsonify(
        {
            "session": _trim_session_payload(
                _with_access(
                    {
                        "id": sid,
                        "title": hist.get("title") or "Session",
                        "agent": hist.get("agent") or DEFAULT_AGENT,
                        "model": hist.get("model") or "",
                        "engine_model": hist.get("engine_model") or "",
                        "engine_model_alias": hist.get("engine_model_alias") or "",
                        "owner": g.user_email,
                        "origin": "gateway",
                        "messages": hist.get("messages") or [],
                        "history_total": hist.get("history_total"),
                        "before_cursor": hist.get("before_cursor"),
                        "history_truncated": bool(hist.get("history_truncated")),
                        "tail_continuation": bool(hist.get("tail_continuation")),
                        # Same guarantee as the bound-conversation branch above.
                        "tail_from_offset": _tail_watermark(hist, sid),
                        "active": bool(hist.get("active")),
                        "state": hist.get("session_state") or "idle",
                        "tree_busy": bool(hist.get("tree_busy") or hist.get("session_state") == "waiting"),
                        "updated_at": _iso(hist.get("updated_at")),
                        "context_window": hist.get("context_window") or 0,
                        "last_prompt_tokens": hist.get("last_prompt_tokens") or 0,
                        "goal_id": goal_id,
                        "goal": goal,
                        "budget": budget,
                        "footer": footer,
                        # The LOCAL operator's monitor view of a subagent IS writable: the
                        # gateway owns the continuation (queue-steer while the child's turn is
                        # live, fresh turn when idle), so the composer unlocks here. Only a
                        # NON-local principal's delegated-child view stays read-only — with or
                        # without the ``monitor=1`` flag, since adoption is exactly what they
                        # may not do.
                        "monitor_read_only": not local_adoption,
                    },
                    sid,
                )
            )
        }
    )


def _delete_confirmed() -> bool:
    """True iff this DELETE carries an explicit irreversibility confirmation.

    Accepts ``?confirm=1`` (or true/yes), a JSON body ``{"confirm": true}``, or
    an ``X-Confirm-Delete`` header — so the browser dialog AND a deliberate API
    client can both satisfy it, but neither can skip it.
    """
    truthy = {"1", "true", "yes"}
    if str(request.args.get("confirm", "")).lower() in truthy:
        return True
    if str(request.headers.get("X-Confirm-Delete", "")).lower() in truthy:
        return True
    body = request.get_json(silent=True) or {}
    return bool(body.get("confirm")) is True


def _authorized_card_log(sid: str):
    """Resolve a session JSONL only when the caller may read that session.

    Mirrors ``api_get_session`` ownership: owned web row, local adopt, or an
    already-owned gateway ancestor. Unknown hashes fail closed.
    """
    sdata = store.get(sid, g.user_email)
    if sdata:
        gw = _stream_gateway_sid(sid, sdata)
        if not gw:
            return None, False
    else:
        local_adoption = may_adopt_gateway_session(g.user_email)
        from sharedagents_web import access

        if not local_adoption and not _native_gateway_visible(sid, g.user_email, min_level=access.VIEW):
            return None, False
        if not _gw_session_log_exists(sid):
            return None, False
        gw = sid
    # Single-session ledger: exactly one file per session.
    phys = GW_SESSIONS_DIR / f"{gw}.jsonl"
    if not phys.is_file():
        return None, False
    return phys, False


@bp.route("/api/sessions/<sid>/cards/<call_id>", methods=["GET"])
@require_auth
def api_session_card(sid, call_id):
    """Lazy-load one historical tool card body on expand.

    Assistant reasoning ships inline on history / SSE / live payloads; this
    endpoint remains for tool cards (and older ``reason@…`` lazy shells).
    """
    from sharedagents_session.session_reader import (
        collect_card_events,
        fold_card_segment,
    )

    path, _ = _authorized_card_log(sid)
    if path is None:
        return jsonify({"error": "Session not found"}), 404
    events = collect_card_events(path, call_id)
    segment = fold_card_segment(events)
    if segment is None:
        return jsonify({"error": "Card not found"}), 404
    return jsonify({"call_id": call_id, "segment": segment})


def _context_access(sid: str):
    """``(readable, sees_instructions, can_compact, has_log)`` for the viewer.

    Readable exactly when the session's history is (``_authorized_card_log``).
    The agent's instructions (system prompt, preamble, tool definitions) are
    shown to the conversation's owner only; compaction needs chat access.
    """
    from sharedagents_web import access

    email = g.user_email
    sdata = store.get(sid)
    owned = sdata is not None and access.row_owner(sdata) == access._norm(email)
    path, _ = _authorized_card_log(sid)
    if path is None:
        return owned, owned, owned, False
    owner = (
        owned
        or may_adopt_gateway_session(email)
        or _configured_native_view_allowed(sid, email)
        or bool(_owned_gateway_ancestor(sid, email))
    )
    level = access.MANAGE if owner else access.session_access(email, sid)
    return True, level >= access.MANAGE, level >= access.CHAT, True


def _context_gateway_read(sid: str, path: str, params: dict):
    readable, instructions, can_compact, has_log = _context_access(sid)
    if not readable:
        return jsonify(
            {
                "code": "session_forbidden",
                "message": "Session not found",
                "retryable": False,
            }
        ), 404
    if not has_log:
        # An owned conversation that has not reached the gateway yet.
        return jsonify(
            {
                "ok": True,
                "session_id": sid,
                "snapshot": "",
                "items": [],
                "next_cursor": None,
                "notice": "",
                "notes": [],
                "totals": {"items": 0, "tokens": 0, "label": "0"},
                "empty_text": "Nothing has been sent to the model yet.",
                "can_compact": False,
            }
        ), 200
    params = {k: v for k, v in params.items() if v not in (None, "")}
    params["instructions"] = "1" if instructions else "0"
    resp = gateway_request(
        "GET",
        f"/control/v1/orchestrator/sessions/{sid}/{path}",
        params=params,
        headers=gateway_principal_headers(g.user_email),
        timeout=60,
        retry_unavailable=False,
    )
    if resp is None:
        return _gateway_json(resp)
    try:
        payload = resp.json()
    except ValueError:
        return _gateway_json(None)
    if isinstance(payload, dict) and payload.get("ok") and path == "context":
        payload["can_compact"] = bool(can_compact)
    return jsonify(payload), resp.status_code


@bp.route("/api/sessions/<sid>/context", methods=["GET"])
@require_auth
def api_session_context(sid):
    """One page of the context the session's model works with (read-only)."""
    return _context_gateway_read(
        sid,
        "context",
        {
            "cursor": request.args.get("cursor", ""),
            "limit": request.args.get("limit", type=int),
            "max_bytes": request.args.get("max_bytes", type=int),
        },
    )


@bp.route("/api/sessions/<sid>/context/items/<int:index>", methods=["GET"])
@require_auth
def api_session_context_item(sid, index):
    """A character range of one context item (``load full item``)."""
    return _context_gateway_read(
        sid,
        f"context/items/{index}",
        {
            "snapshot": request.args.get("snapshot", ""),
            "offset": request.args.get("offset", type=int),
            "length": request.args.get("length", type=int),
        },
    )


@bp.route("/api/sessions/<sid>", methods=["DELETE"])
@require_auth
def api_delete_session(sid):
    """Delete a session — only if owned by current user.

    Irreversible-command re-challenge: deleting a session is
    irreversible, so a DELETE that does not carry an explicit confirmation is
    NOT executed — the server replies 409 with ``confirmation_required`` plus the
    dialog content. A direct API caller bypassing the client dialog hits the
    identical re-challenge (the dialog cannot be skipped, only rendered
    differently). The confirm gate is keyed to *irreversibility*, not "sounds
    destructive": recoverable commands (e.g. goal archive, POST
    /api/goals/<id>/archive) keep their one-call path with no added step.
    """
    if not _delete_confirmed():
        return jsonify(
            {
                "error": "confirmation required",
                "confirmation_required": True,
                "irreversible": True,
                "object": f"conversation {sid}",
                "prompt": "Deleting this conversation is permanent and cannot be undone.",
            }
        ), 409
    deleted = store.delete(sid, g.user_email)
    if not deleted:
        return jsonify({"error": "Session not found"}), 403
    # A deleted conversation is shared with nobody.
    from sharedagents_web import server as _web_server

    _web_server.sharing_store.delete(sid)
    return jsonify({"deleted": True})


@bp.route("/api/sessions/<sid>/live", methods=["GET"])
@require_auth
def api_session_live(sid):
    """Cheap sticky-Stop bit: ``session_state`` + ``tree_busy`` for one session.

    Chat used to poll ``GET /api/agents/tree?all=1&limit=300`` every 3s to
    decide whether Stop stays armed. That is a 100KB+ DAG walk. This proxies
    the gateway's one-row ``/activity`` read instead.
    """
    activity_sid = sid
    resp = gateway_request(
        "GET",
        f"/control/v1/orchestrator/sessions/{activity_sid}/activity",
        timeout=5,
    )
    if resp is None or resp.status_code != 200:
        return jsonify(
            {
                "session_state": "idle",
                "tree_busy": False,
                "parent_session_id": "",
            }
        )
    try:
        data = resp.json()
    except ValueError:
        return jsonify(
            {
                "session_state": "idle",
                "tree_busy": False,
                "parent_session_id": "",
            }
        )
    if not isinstance(data, dict):
        return jsonify(
            {
                "session_state": "idle",
                "tree_busy": False,
                "parent_session_id": "",
            }
        )
    payload = {
        "session_state": data.get("session_state") or "idle",
        "tree_busy": bool(data.get("tree_busy")),
        "parent_session_id": str(data.get("parent_session_id") or ""),
        "subagents": list(data.get("subagents") or []) if isinstance(data.get("subagents"), list) else [],
    }
    # V2 is additive: old gateway responses and existing clients retain the
    # compact four-field shape, while newer gateways expose the authoritative
    # activity/display/capability contract through the same endpoint.
    for key in (
        "activity",
        "version",
        "phase",
        "attention",
        "lifecycle",
        "last_outcome",
        "own_live",
        "stale",
        "active_descendant_count",
        "open_job_count",
        "phase_changed_at",
        "heartbeat_at",
        "outcome_at",
        "status",
        "display",
        "capabilities",
    ):
        if key in data:
            payload[key] = data[key]
    return jsonify(payload)


@bp.route("/api/sessions/<sid>/stream", methods=["GET"])
@require_auth
def api_session_stream(sid):
    """SSE proxy of the gateway's live conversation stream for ONE session.

    The client opens this ONLY for the currently-selected conversation, so the
    web stays light — one warm stream per viewer, not per session. We resolve
    the current SID directly, or accept a gateway-native SID (TUI / monitor
    deep-link), enforce ownership
    (IDOR-safe), then byte-forward the gateway SSE. When there is no gateway
    session to stream yet (a brand-new web chat with no turns), the mux stays
    open on control frames and attaches the gateway tail once the JSONL exists
    so a second browser already watching the chat sees the first turn.
    """
    sdata = store.get(sid, g.user_email)
    if sdata is not None:
        gw = _stream_gateway_sid(sid, sdata)
    else:
        bound_owner = store.gateway_binding_owner(sid)
        from sharedagents_web import access

        if bound_owner not in (None, g.user_email) and access.session_access(g.user_email, sid) < access.VIEW:
            return jsonify({"error": "Session not found"}), 403
        authorized = _native_gateway_visible(
            sid,
            g.user_email,
            min_level=access.VIEW,
        )
        if not authorized:
            return jsonify({"error": "Session not found"}), 403
        # Gateway-native session the user opened but hasn't adopted yet.
        # Existence is a cheap stat of the session log — NOT gw_session_history,
        # which folds the whole <hash>.jsonl just to return a bool. On a large /
        # active session (e.g. switching to ``main``) that O(file) fold ran on
        # the SSE-open hot path IN ADDITION to the history GET's own fold and
        # held a (scarce) gthread the whole time it scanned.
        gw = sid

    # A web placeholder (or stale binding) whose resolved gateway hash has no
    # session log on disk has nothing to forward — streaming it to the gateway
    # 404s and the client reconnect-loops. Fall back to the local control-frame
    # stream (ready/state/reconnect) until the first prompt creates its log —
    # then the mux attaches ``pending_gw_url`` so other open browsers see it.
    hash_for_url = gw or sid
    if gw and not _gw_session_log_exists(gw):
        gw = ""

    resp_headers = {
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
    }

    qs = request.query_string.decode("latin-1")
    pending_gw_url = f"{current_gateway()}/control/v1/orchestrator/sessions/{hash_for_url}/stream"
    if qs:
        pending_gw_url += f"?{qs}"
    gateway_headers = {
        **_gateway_auth_header(),
        **gateway_principal_headers(g.user_email),
    }
    from sharedagents_web.routes.ui_stream import iter_session_stream

    owner = str(g.user_email)
    live_gw_url = pending_gw_url if gw else None
    return Response(
        stream_with_context(
            iter_session_stream(
                owner,
                sid,
                live_gw_url,
                gateway_headers,
                _SESSION_STREAM_READ_TIMEOUT,
                pending_gw_url=pending_gw_url if not gw else None,
            )
        ),
        mimetype="text/event-stream",
        headers=resp_headers,
    )


@bp.route("/api/sessions/<sid>/events", methods=["GET"])
@require_auth
def api_session_events(sid):
    """Cursor-based polling fallback when SSE is disabled or blocked.

    Used by ``SHAREDAGENTS_WEB_DISABLE_SSE=1`` and ``?transport=poll``.
    Supports short-poll (timeout=0) and long-poll (timeout > 0).
    """
    sdata = store.get(sid, g.user_email)
    if sdata is not None:
        gw = _stream_gateway_sid(sid, sdata)
    else:
        bound_owner = store.gateway_binding_owner(sid)
        from sharedagents_web import access

        if bound_owner not in (None, g.user_email) and access.session_access(g.user_email, sid) < access.VIEW:
            return jsonify({"error": "Session not found"}), 403
        authorized = _native_gateway_visible(
            sid,
            g.user_email,
            min_level=access.VIEW,
        )
        if not authorized:
            return jsonify({"error": "Session not found"}), 403
        gw = sid

    try:
        from_offset = max(0, int(request.args.get("from", 0)))
    except (ValueError, TypeError):
        from_offset = 0
    try:
        timeout = min(20.0, max(0.0, float(request.args.get("timeout", 0.0))))
    except (ValueError, TypeError):
        timeout = 0.0
    try:
        limit = min(500, max(1, int(request.args.get("limit", 200))))
    except (ValueError, TypeError):
        limit = 200

    from sharedagents_session.session_reader import tail_jsonl
    from sharedagents_web.routes.ui_stream import (
        _state_signature,
        _state_snapshot,
    )

    owner = str(g.user_email)
    from sharedagents_agent.delivery import redact

    # ``state`` is sent ONCE and thereafter only when it CHANGES — it is not an
    # opt-in flag. The client reports the signature of the snapshot it last
    # applied (``state_sig``); holding nothing bootstraps, and a later poll
    # re-sends only when that snapshot actually moved. Same rule as the SSE
    # tick, where an unchanged snapshot degrades to a heartbeat.
    #
    # (There used to be an ``include_state`` switch here. It was worse than
    # useless: the ``or from_offset == 0`` fallback meant it was ignored for
    # every idle session, whose offset never advances — so the full snapshot
    # rode every single poll while the client believed it had asked for it
    # once every 4s.)
    held_state_sig = str(request.args.get("state_sig") or "").strip()
    # Picker endpoints own model data for both SSE and polling clients.
    # A new Web session has no JSONL until its first prompt. Keep the initial
    # request long-polling for that file instead of returning immediately.
    log_path = GW_SESSIONS_DIR / f"{gw or sid}.jsonl"
    start_time = time.monotonic()
    events = []
    curr_offset = from_offset
    while True:
        try:
            for ev, end in tail_jsonl(log_path, curr_offset):
                events.append(
                    {
                        "offset": end,
                        "seq": len(events),
                        "spawnedBy": ev.get("spawned_by") or "",
                        "event": redact(ev),
                    }
                )
                curr_offset = end
                if len(events) >= limit:
                    break
        except OSError as exc:
            log.debug("poll read failed for session %s: %s", sid, exc)
        if events or timeout <= 0.0 or (time.monotonic() - start_time >= timeout):
            break
        time.sleep(0.2)
    state = _state_snapshot(owner, sid)
    state_sig = _state_signature(state)
    # A snapshot we cannot digest (empty signature) can never be PROVEN
    # unchanged, so it always sends — the same fail-open rule as the SSE tick.
    send_state = not held_state_sig or not state_sig or state_sig != held_state_sig
    return jsonify(
        {
            "ok": True,
            "session_id": sid,
            "from": from_offset,
            "next_offset": curr_offset,
            "events": events,
            "state": state if send_state else None,
            # Always reported, so the client can adopt it even on a poll that
            # carried no state (the value it already had, restated).
            "state_sig": state_sig,
            "models_catalogue": None,
        }
    )


def _resolve_gw_hash(sid):
    """Authorize a web SID for a gateway control proxy.

    Admins may adopt a TUI/Telegram hash into the Web store. A person the
    conversation is shared with may prompt that hash without adopting it
    (the HTTP gate already required CHAT). IDOR stays closed: a hash that
    is neither owned, adopted, nor shared is 403.

    On success returns ``(gw_hash, None)``; on failure ``(None, (response, status))``.
    """
    sdata = store.get(sid, g.user_email)
    if sdata is not None:
        return sid, None
    from sharedagents_web import access

    if may_adopt_gateway_session(g.user_email):
        bound_owner = store.gateway_binding_owner(sid)
        if bound_owner is not None and bound_owner != g.user_email:
            return None, (jsonify({"error": "Session not found"}), 403)
        hist = gw_session_history(sid)
        if hist is None:
            return None, (jsonify({"error": "Session not found"}), 403)
        try:
            store.adopt_gateway(
                g.user_email,
                sid,
                hist,
                gateway_principal=canonical_gateway_principal(g.user_email),
            )
        except PermissionError:
            return None, (jsonify({"error": "Session not found"}), 403)
        sdata = store.get(sid, g.user_email)
        if sdata is not None:
            return sid, None
        return None, (jsonify({"error": "Session not found"}), 403)
    if access.session_access(g.user_email, sid) >= access.CHAT and _gw_session_log_exists(sid):
        return sid, None
    return None, (jsonify({"error": "Session not found"}), 403)


def _proxy_session_command(gw, action, body):
    from uuid import uuid4

    from sharedagents_gateway.command_receipts import pending_receipt

    body = dict(body)
    body.setdefault("request_id", str(uuid4()))
    try:
        resp = gateway_request(
            "POST",
            f"/control/v1/orchestrator/sessions/{gw}/{action}",
            json=body,
            headers=gateway_principal_headers(g.user_email),
            report_read_timeout=True,
        )
    except requests.ReadTimeout:
        return jsonify(pending_receipt(body["request_id"])), 202
    if resp is None:
        return jsonify({"error": "Gateway unreachable"}), 502
    try:
        return jsonify(resp.json()), resp.status_code
    except ValueError:
        return jsonify({"error": "bad gateway response"}), 502


def _submitted_files(body):
    """Saved-file references a submit carries: ``(files, error_response)``.

    The page sends what the upload route returned (``path``/``name``/
    ``mime``/``size``); sizes are read from disk here. A regular user may
    only reference files inside their own downloads folder.
    """
    from sharedagents_core.attachment_context import normalize_files
    from sharedagents_web import access
    from sharedagents_web.uploads import upload_directory

    raw = body.pop("files", None)
    if raw in (None, []):
        return [], None
    if not isinstance(raw, list):
        return [], (jsonify({"error": "'files' must be a list"}), 400)
    files = normalize_files(raw, stat=False)
    if len(files) != len(raw):
        return [], (jsonify({"error": "invalid attachment reference"}), 400)
    root = None
    if not access.is_admin(g.user_email):
        try:
            root = upload_directory(g.user_email, admin=False)
        except (OSError, PermissionError, ValueError, RuntimeError):
            return [], (jsonify({"error": "invalid attachment reference"}), 400)
    for ref in files:
        try:
            path = Path(ref["path"]).resolve(strict=True)
            if root is not None and not path.is_relative_to(root):
                return [], (jsonify({"error": "attachment is outside your downloads folder"}), 403)
            if not path.is_file():
                raise ValueError("attachment must be a regular file")
            ref["path"] = str(path)
            ref["size"] = path.stat().st_size
        except (OSError, RuntimeError, ValueError):
            return [], (jsonify({"error": "invalid attachment reference"}), 400)
    return files, None


def _prompt_with_files(prompt, files):
    """The submit ``prompt`` as ACP blocks with ``files`` on block 0.

    Client-supplied references inside blocks are dropped: only the
    server-checked ``files`` field may name saved originals.
    """
    from sharedagents_core.attachment_context import strip_files_meta, with_files_meta

    if isinstance(prompt, str):
        blocks = [{"type": "text", "text": prompt}] if (prompt or files) else []
    elif isinstance(prompt, dict):
        blocks = strip_files_meta([prompt])
    elif isinstance(prompt, list):
        blocks = strip_files_meta(prompt)
    else:
        blocks = []
    if not files:
        return prompt if isinstance(prompt, str) or not blocks else blocks
    if not blocks:
        blocks = [{"type": "text", "text": ""}]
    blocks[0] = with_files_meta(blocks[0], files)
    return blocks


@bp.route("/api/sessions/<sid>/prompt", methods=["POST"])
@require_auth
def api_session_prompt(sid):
    """Submit a prompt to a session's gateway turn (spec §9.1).

    The passive submit half of the inverted WebUI: resolve the bound gateway
    hash (same auth / IDOR checks as the stream route), proxy the body to the
    orchestrator submit route, and return its ``202 {accepted, hash,
    start_offset}`` so the browser subscribes to the event stream from exactly
    this turn's first byte. The gateway owns the turn — nothing streams back
    here.
    """
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        return jsonify({"error": "body must be a JSON object"}), 400
    from sharedagents_web import access

    # Saved uploads travel as references; the backend (not the page) builds
    # the model-facing attachment context from them at the turn boundary.
    files, files_error = _submitted_files(body)
    if files_error is not None:
        return files_error
    body["prompt"] = _prompt_with_files(body.get("prompt"), files)
    admin = access.is_admin(g.user_email)
    if not admin:
        # A user's turn runs where the conversation says, never where the
        # request says, and never skips the model.
        if access.has_direct_dispatch(body.get("prompt"), body.get("attachments")):
            return _refused("this prompt form is not available to your account")
        for key in ("initial_agent", "workspace_cwd", "workspace_origin"):
            body.pop(key, None)
    sdata = store.get(sid, g.user_email)
    if sdata is not None:
        gw = sid
        body.setdefault("initial_agent", sdata.get("agent") or DEFAULT_AGENT)
        body.setdefault("workspace_cwd", sdata.get("workspace_cwd") or "")
        body.setdefault("workspace_origin", "web")
    else:
        gw, adoption_error = _resolve_gw_hash(sid)
        if adoption_error is not None:
            return adoption_error
    if not admin:
        refusal = _submit_refusal(
            sdata if sdata is not None else store.get(gw),
            body.get("initial_model"),
            gw,
        )
        if refusal:
            return _refused(refusal)
    if sdata is not None:
        save_needed = False
        initial_model = str(body.get("initial_model") or "").strip()
        if initial_model and not admin and _gw_session_log_exists(gw):
            # The gateway reads initial_model at creation only: never let it
            # rewrite a running conversation's recorded model.
            initial_model = ""
        if initial_model:
            eff_agent = str(sdata.get("agent") or DEFAULT_AGENT)
            if eff_agent != "cursor" and initial_model.lower() in {"auto", "ask", "plan"}:
                initial_model = ""
            if initial_model and sdata.get("model") != initial_model:
                sdata["model"] = initial_model
                save_needed = True
        if not sdata.get("submitted"):
            sdata["submitted"] = True
            save_needed = True
        if save_needed and not store._save(sdata):
            return jsonify({"error": "Session state could not be persisted"}), 503
    # This route IS the authenticated web transport: every request here is the
    # ``web`` frontend (ask-capable via permission cards + answer POST). Stamp
    # the frontend from the transport so a caller-supplied ``frontend`` body
    # cannot claim a more-capable origin than the transport
    # (mirrors ``/api/command``, which pins ``"web"``). A body ``frontend`` may only narrow.
    if isinstance(body, dict):
        # The author rides the same server stamp as ``frontend``: it is the
        # authenticated SUBMITTER (not the session owner) and a client-sent
        # ``actor`` is overwritten, never trusted.
        body = {**body, "frontend": "web", "actor": turn_actor(g.user_email)}
        # The submitter's grants ride the same server stamp: the gateway
        # enforces tools, skills and delegation from it; a client value is
        # overwritten, and any access stamp inside client blocks is dropped.
        # ``None`` = an admin (unrestricted).
        body["access_policy"] = access.access_policy_for(g.user_email)
        if isinstance(body.get("prompt"), (list, dict)):
            from sharedagents_core.access_grants import strip_access_meta

            raw_prompt = body["prompt"]
            body["prompt"] = strip_access_meta([raw_prompt] if isinstance(raw_prompt, dict) else raw_prompt)
        # Only internal gateway clients can deliver background receipts. That
        # path deliberately omits human actors, so a browser cannot select it.
        body.pop("background_job_id", None)
    return _proxy_session_command(gw, "prompt", body)


@bp.route("/api/sessions/<sid>/answer", methods=["POST"])
@require_auth
def api_session_answer(sid):
    """Deliver a human's AskUserQuestion answer to a running gateway turn.

    The browser POSTs ``{"callId": ..., "answers": [{question, selected, notes}]}``
    when the user submits the interactive question card; we resolve the bound
    gateway hash (same auth / IDOR checks as /prompt) and relay it to the
    orchestrator answer route, which completes the future the turn awaits.
    """
    gw, err = _resolve_gw_hash(sid)
    if err is not None:
        return err
    body = request.get_json(silent=True) or {}
    resp = gateway_request(
        "POST",
        f"/control/v1/orchestrator/sessions/{gw}/answer",
        json=body,
        headers=gateway_principal_headers(g.user_email),
    )
    if resp is None:
        return jsonify({"error": "Gateway unreachable"}), 502
    try:
        payload = resp.json()
    except ValueError:
        return jsonify({"error": "bad gateway response"}), 502
    if resp.status_code < 400:
        try:
            from sharedagents_web.routes.dev import fold_chat_answers_into_run

            fold_chat_answers_into_run(sid, body)
        except Exception:  # noqa: BLE001 — Chat answer must still succeed
            log.warning("sdlc: could not fold Chat answers into Dev for %s", sid, exc_info=True)
    return jsonify(payload), resp.status_code


@bp.route("/api/sessions/<sid>/permission", methods=["POST"])
@require_auth
def api_session_permission(sid):
    """Resolve an interactive tool-permission request for a running gateway turn

    The browser POSTs ``{"requestId": ..., "outcome": {"outcome":
    "selected"|"cancelled", "optionId": ...}}`` when the operator approves/denies
    a tool. We resolve the bound gateway hash (the SAME auth / IDOR check as
    /prompt and /answer via ``_resolve_gw_hash``; an unauthenticated /
    different-session client is rejected the same way) and
    relay it to the orchestrator permission route, which completes the future the
    turn awaits. The panel is a view, not a second channel: actioning goes through
    this same authenticated, session-bound route.
    """
    gw, err = _resolve_gw_hash(sid)
    if err is not None:
        return err
    body = request.get_json(silent=True) or {}
    resp = gateway_request(
        "POST",
        f"/control/v1/orchestrator/sessions/{gw}/permission",
        json=body,
        headers=gateway_principal_headers(g.user_email),
    )
    if resp is None:
        return jsonify({"error": "Gateway unreachable"}), 502
    try:
        return jsonify(resp.json()), resp.status_code
    except ValueError:
        return jsonify({"error": "bad gateway response"}), 502


@bp.route("/api/sessions/<sid>/queue", methods=["GET"])
@require_auth
def api_session_queue_list(sid):
    """List pending follow-up queue items for a session."""
    gw, err = _resolve_gw_hash(sid)
    if err is not None:
        return err
    resp = gateway_request(
        "GET",
        f"/control/v1/orchestrator/sessions/{gw}/queue",
        headers=gateway_principal_headers(g.user_email),
    )
    if resp is None:
        return jsonify({"error": "Gateway unreachable"}), 502
    try:
        payload = resp.json()
    except ValueError:
        return jsonify({"error": "bad gateway response"}), 502
    if isinstance(payload, dict) and isinstance(payload.get("pending"), list):
        from sharedagents_core.attachment_context import project_queue_item

        payload["pending"] = [
            project_queue_item(item) if isinstance(item, dict) else item for item in payload["pending"]
        ]
    return jsonify(payload), resp.status_code


@bp.route("/api/sessions/<sid>/queue", methods=["POST"])
@require_auth
def api_session_queue_add(sid):
    """Enqueue a follow-up message while a turn is in flight."""
    gw, err = _resolve_gw_hash(sid)
    if err is not None:
        return err
    body = request.get_json(silent=True) or {}
    if isinstance(body, dict):
        from sharedagents_web import access

        texts = body.pop("texts", None)
        if texts is not None:
            from sharedagents_core.attachment_context import merge_queue_text

            if not isinstance(texts, list) or any(not isinstance(t, str) for t in texts):
                return jsonify({"error": "texts must be a list of strings"}), 400
            body["text"] = merge_queue_text([{"text": text} for text in texts])
        # The queue carries text only: saved files ride as the model-facing
        # attachment context (transcripts show them as chips again).
        files, files_error = _submitted_files(body)
        if files_error is not None:
            return files_error
        if files:
            from sharedagents_core.attachment_context import model_text

            body["text"] = model_text(str(body.get("text") or ""), files)
        # A queued item becomes a drain turn's prompt: same checks as /prompt.
        if not access.is_admin(g.user_email):
            if access.has_direct_dispatch(str(body.get("text") or "")):
                return _refused("this prompt form is not available to your account")
            refusal = _submit_refusal(store.get(gw), sid=gw)
            if refusal:
                return _refused(refusal)
        body = {
            **body,
            "actor": turn_actor(g.user_email),
            "access_policy": access.access_policy_for(g.user_email),
        }
    resp = gateway_request(
        "POST",
        f"/control/v1/orchestrator/sessions/{gw}/queue",
        json=body,
        headers=gateway_principal_headers(g.user_email),
    )
    if resp is None:
        return jsonify({"error": "Gateway unreachable"}), 502
    try:
        payload = resp.json()
    except ValueError:
        return jsonify({"error": "bad gateway response"}), 502
    if resp.status_code < 300 and isinstance(payload, dict) and isinstance(body, dict):
        from sharedagents_core.attachment_context import project_queue_item

        payload["item"] = project_queue_item({"queue_id": payload.get("queue_id", ""), "text": body.get("text", "")})
    return jsonify(payload), resp.status_code


def _queue_selection(gw, ids):
    """Read selected pending entries from the authoritative gateway."""
    if not isinstance(ids, list) or not ids or any(not isinstance(q, str) for q in ids):
        return [], (jsonify({"error": "queue_ids must be a nonempty list of strings"}), 400)
    resp = gateway_request(
        "GET",
        f"/control/v1/orchestrator/sessions/{gw}/queue",
        headers=gateway_principal_headers(g.user_email),
    )
    if resp is None:
        return [], (jsonify({"error": "Gateway unreachable"}), 502)
    try:
        payload = resp.json()
        if resp.status_code != 200:
            return [], (jsonify(payload), resp.status_code)
        pending = payload["pending"]
        selected = [item for item in pending if item.get("queue_id") in ids]
    except (ValueError, KeyError, TypeError, AttributeError):
        return [], (jsonify({"error": "bad gateway response"}), 502)
    if {item["queue_id"] for item in selected} != set(ids):
        return [], (jsonify({"error": "message already delivered", "error_code": "not_pending"}), 409)
    return selected, None


@bp.route("/api/sessions/<sid>/queue/prepare", methods=["POST"])
@require_auth
def api_session_queue_prepare(sid):
    """Prepare selected queue entries for display and Steer on the backend."""
    from sharedagents_core.attachment_context import merge_queue_text, project_queue_item

    gw, err = _resolve_gw_hash(sid)
    if err is not None:
        return err
    body = request.get_json(silent=True) or {}
    items, err = _queue_selection(gw, body.get("queue_ids") if isinstance(body, dict) else None)
    if err is not None:
        return err
    return jsonify(project_queue_item({"text": merge_queue_text(items)}))


@bp.route("/api/sessions/<sid>/queue/steer", methods=["POST"])
@require_auth
def api_session_queue_steer(sid):
    """Steer selected queue entries: the server decides what happens.

    ``{"steered": true}``: the running turn takes the queued text now.
    ``{"steered": false, "busy": bool}``: the page stops the turn when the
    server says it is busy, then sends the text as a new prompt.
    """
    gw, err = _resolve_gw_hash(sid)
    if err is not None:
        return err
    body = request.get_json(silent=True) or {}
    ids = body.get("queue_ids") if isinstance(body, dict) else None
    if not isinstance(ids, list) or not ids:
        return jsonify({"error": "queue_ids required"}), 400
    resp = gateway_request(
        "POST",
        f"/control/v1/orchestrator/sessions/{gw}/steer",
        json={"queue_ids": [str(q) for q in ids]},
        headers=gateway_principal_headers(g.user_email),
    )
    if resp is None:
        return jsonify({"error": "Gateway unreachable"}), 502
    try:
        return jsonify(resp.json()), resp.status_code
    except ValueError:
        return jsonify({"error": "bad gateway response"}), 502


def _queued_text_keeping_files(gw, queue_id, new_text):
    from sharedagents_core.attachment_context import keep_files_on_edit
    from sharedagents_session.session_reader import fold_queue_state, tail_jsonl

    try:
        pending = fold_queue_state(ev for ev, _ in tail_jsonl(GW_SESSIONS_DIR / f"{gw}.jsonl", 0))
    except OSError:
        pending = []
    for item in pending:
        if str(item.get("queue_id") or "") == str(queue_id):
            return keep_files_on_edit(item.get("text"), new_text)
    return str(new_text or "")


@bp.route("/api/sessions/<sid>/queue/<queue_id>", methods=["PATCH"])
@require_auth
def api_session_queue_update(sid, queue_id):
    """Edit a pending queued message."""
    gw, err = _resolve_gw_hash(sid)
    if err is not None:
        return err
    body = request.get_json(silent=True) or {}
    if isinstance(body, dict):
        from sharedagents_web import access

        if not access.is_admin(g.user_email):
            if access.has_direct_dispatch(str(body.get("text") or "")):
                return _refused("this prompt form is not available to your account")
            # The edited item runs on this conversation: its agent and model
            # must be granted to the editor, as for a new queued message.
            refusal = _submit_refusal(store.get(gw), sid=gw)
            if refusal:
                return _refused(refusal)
        # The edited item runs under no more than the editor's grants: the
        # gateway intersects them with the item's stamp.
        body = {**body, "access_policy": access.access_policy_for(g.user_email)}
        if "text" in body:
            # The edit box shows the typed text only; keep the item's files.
            ids = body.pop("queue_ids", None)
            if ids is not None:
                from sharedagents_core.attachment_context import keep_files_on_edit, merge_queue_text

                items, err = _queue_selection(gw, ids)
                if err is not None:
                    return err
                if queue_id not in ids:
                    return jsonify({"error": "edited item must be selected"}), 400
                body["text"] = keep_files_on_edit(merge_queue_text(items), body.get("text"))
            else:
                body["text"] = _queued_text_keeping_files(gw, queue_id, body.get("text"))
    resp = gateway_request(
        "PATCH",
        f"/control/v1/orchestrator/sessions/{gw}/queue/{queue_id}",
        json=body,
        headers=gateway_principal_headers(g.user_email),
    )
    if resp is None:
        return jsonify({"error": "Gateway unreachable"}), 502
    try:
        payload = resp.json()
    except ValueError:
        return jsonify({"error": "bad gateway response"}), 502
    if resp.status_code < 300 and isinstance(payload, dict) and isinstance(body, dict) and "text" in body:
        from sharedagents_core.attachment_context import project_queue_item

        payload["item"] = project_queue_item({"queue_id": queue_id, "text": body["text"]})
    return jsonify(payload), resp.status_code


@bp.route("/api/sessions/<sid>/queue/<queue_id>", methods=["DELETE"])
@require_auth
def api_session_queue_delete(sid, queue_id):
    """Delete a pending queued message."""
    gw, err = _resolve_gw_hash(sid)
    if err is not None:
        return err
    resp = gateway_request(
        "DELETE",
        f"/control/v1/orchestrator/sessions/{gw}/queue/{queue_id}",
        headers=gateway_principal_headers(g.user_email),
    )
    if resp is None:
        return jsonify({"error": "Gateway unreachable"}), 502
    try:
        return jsonify(resp.json()), resp.status_code
    except ValueError:
        return jsonify({"error": "bad gateway response"}), 502


@bp.route("/api/sessions/<sid>/cancel", methods=["POST"])
@require_auth
def api_session_cancel(sid):
    """Cancel a session's in-flight gateway turn (spec §9.3).

    Proxies to the orchestrator cancel route; the cancellation surfaces as a
    ``turn_ended(aborted)`` event in the session's log stream (the renderer
    finalizes from there — no special-casing here).
    """
    gw, err = _resolve_gw_hash(sid)
    if err is not None:
        return err
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        return jsonify({"error": "body must be a JSON object"}), 400
    return _proxy_session_command(gw, "cancel", body)


@bp.route("/api/approvals")
@require_auth
def api_approvals():
    """Aggregate every approval awaiting THIS operator across all their sessions
    . Read-only view: it lists pending decisions so one is
        reachable regardless of which conversation raised it, but actioning still
        goes through the same authenticated, session-bound answer route — the panel
        is a view, not a privilege widening by
        construction.

        The gateway owns the pending registry; we proxy it and filter to the
        sessions this user owns (mapping the gateway session hash back to the web
        session id so the client can deep-link), so no operator ever sees another
        user's pending decision (the auth boundary).
    """
    resp = gateway_request(
        "GET",
        "/control/v1/orchestrator/approvals",
        timeout=10,
        retry_unavailable=True,
    )
    if resp is None or resp.status_code != 200:
        return jsonify({"approvals": [], "error": "Gateway unreachable"}), 502
    try:
        raw = resp.json().get("approvals") or []
    except ValueError:
        raw = []
    return jsonify(
        {
            "approvals": _map_gateway_approvals(raw, g.user_email),
        }
    )
