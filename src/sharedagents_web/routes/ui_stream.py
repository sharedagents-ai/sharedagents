"""Chat-page stream: gateway transcript bytes + ONE ``event: state`` control frame.

The chat page opens exactly one EventSource — ``GET /api/sessions/<sid>/stream`` —
which forwards the gateway's ``event: log`` transcript frames and, on a single
tick, emits one consolidated ``event: state`` payload (active-session header,
health, approvals, live sessions, budget, goal, Agent-Monitor tree, widgets).
There is no second ``/api/ui/stream`` stream and no per-frame multiplexer; REST
is the lazy sidebar fallback only.

Two properties keep an IDLE tab cheap (it used to push 857 KB every 4s):

* model lists are requested only through the picker, never built on connect
  or rebuilt on a stream tick;
* a ``state`` snapshot byte-identical to the last one emitted carries no
  information, so the tick degrades to a 20-byte ``event: hb`` (see
  :func:`_state_or_heartbeat`).
"""

from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
from collections.abc import Iterator
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sharedagents_web.principal import gateway_principal_headers
from sharedagents_web.server import (
    _fetch_tree,
    _gw_session_log_exists,
    _merged_sessions,
    gateway_request,
    resolve_authoritative_session_title,
    store,
)

_STATE_TICK_SECONDS = 4.0
_STATE_MAX_SECONDS = 600.0
_STATE_HEARTBEAT_SECONDS = 15.0
# An external-auth login's credential is re-verified with the dashboard (an
# HTTP call) this often; the account itself is re-checked every tick.
_DASHBOARD_RECHECK_SECONDS = 30.0

# Frames whose arrival changes the activity snapshot the client reconciles its
# Stop button from. When one is forwarded, wake the ``state`` frame immediately
# (instead of waiting for the 4s tick) so the browser no longer needs the
# ``GET /api/sessions/<sid>/live`` poll to clear a sticky Stop. Matched on the
# JSON event-type value (quoted) so it cannot trip on assistant prose.
_TERMINAL_EVENT_MARKERS = (
    b'"turn_ended"',
    b'"subagent_spawned"',
    b'"agent_switched"',
    b'"subagent_linked"',
    b'"subagent_result"',
    b'"subagent_done"',
)
_APPLY_INFLIGHT_MAX_MS = 30_000.0


def _frame_wakes_state(frame: bytes) -> bool:
    """True when a forwarded gateway frame carries a terminal event whose
    arrival changes the activity snapshot (Stop reconciliation)."""
    return any(marker in frame for marker in _TERMINAL_EVENT_MARKERS)


# One in-process wake channel: any writer (budget / goal / agent switch) bumps
# the generation and the open stream re-emits ``state`` on the next tick.
_state_wakes: dict[str, int] = {}
_state_wake_lock = threading.Lock()
_active_streams: dict[str, int] = {}
_active_stream_lock = threading.Lock()


def register_session_stream(sid: str) -> None:
    """Increment active SSE stream count for a session."""
    key = str(sid or "").strip()
    if not key:
        return
    with _active_stream_lock:
        _active_streams[key] = int(_active_streams.get(key, 0) or 0) + 1
    try:
        store.register_stream(key)
    except Exception:
        pass


def unregister_session_stream(sid: str) -> None:
    """Decrement active SSE stream count for a session."""
    key = str(sid or "").strip()
    if not key:
        return
    with _active_stream_lock:
        cnt = int(_active_streams.get(key, 0) or 0) - 1
        if cnt <= 0:
            _active_streams.pop(key, None)
        else:
            _active_streams[key] = cnt
    try:
        store.unregister_stream(key)
    except Exception:
        pass


def is_session_stream_active(sid: str) -> bool:
    """True if there is at least one active SSE stream connected for this session."""
    key = str(sid or "").strip()
    if not key:
        return False
    try:
        if store.is_stream_active(key):
            return True
    except Exception:
        pass
    with _active_stream_lock:
        return int(_active_streams.get(key, 0) or 0) > 0


def broadcast_models_event(agent: str, models: list[dict[str, Any]]) -> None:
    """Compatibility hook: catalogues now arrive only in explicit HTTP replies."""
    return None


def notify_state_changed(sid: str) -> None:
    """Wake open chat streams for ``sid`` to emit a fresh ``event: state``."""
    key = str(sid or "").strip()
    if not key:
        return
    with _state_wake_lock:
        _state_wakes[key] = int(_state_wakes.get(key, 0) or 0) + 1


def take_state_wake(sid: str) -> int:
    """Current wake generation for ``sid`` (0 if never notified)."""
    key = str(sid or "").strip()
    with _state_wake_lock:
        return int(_state_wakes.get(key, 0) or 0)


# ── Snapshot builders (one per ``state`` field) ─────────────────────────


def _event(name: str, payload: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _emit_frame(name: str, payload: Any) -> bytes:
    data_str = json.dumps(payload, ensure_ascii=False) if isinstance(payload, (dict, list)) else str(payload)
    return f"event: {name}\ndata: {data_str}\n\n".encode()


# Model catalogues are loaded by the picker summary and explicit provider clicks.
# Keep the polling helper's empty answer for clients using the older contract.
def _models_catalogue(owner: str = "") -> dict[str, Any]:
    """Chat transports never read, build or send provider catalogues."""
    return {}


def _stamp_picker_tabs(agent: str, models: list[Any]) -> list[Any]:
    from sharedagents_agent.discovery.picker import engine_spec, stamp_tabs

    return stamp_tabs(agent, models, engine_spec(agent))


# ── Idle-tick suppression ───────────────────────────────────────────────


def _state_signature(snapshot: dict[str, Any]) -> str:
    """Digest of a ``state`` payload, for suppressing unchanged idle ticks."""
    try:
        blob = json.dumps(snapshot, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return ""
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _state_or_heartbeat(
    snapshot: dict[str, Any],
    last_sig: str,
) -> tuple[bytes, str]:
    """``(frame_bytes, signature)`` for one ``state`` tick.

    A snapshot byte-identical to the last one emitted carries no information
    the client does not already have, so it degrades to a 20-byte ``hb`` — the
    client treats ``hb`` as pure liveness (it re-arms the same watchdogs as
    ``state``), which is exactly what an unchanged snapshot is. Returns the
    signature to carry into the next tick.
    """
    sig = _state_signature(snapshot)
    # An unsignable payload must never suppress a real change (``sig`` is then
    # "" and every tick re-emits, matching the pre-suppression behaviour).
    if sig and sig == last_sig:
        return _emit_frame("hb", {}), last_sig
    return _emit_frame("state", snapshot), sig


def _session_hash_for(owner: str, sid: str) -> str:
    """Authorize and return the selected SID without translating it."""
    sid = str(sid or "").strip()
    if not sid:
        return ""
    # Re-authorized on every tick with fresh reads: a revoked share ends the
    # live view without waiting for the stream to reconnect.
    from sharedagents_web import access

    if access.session_access(owner, sid, fresh=True) == access.NONE:
        return ""
    return sid


def _still_reachable(owner: str, sid: str, token: str = "") -> bool:
    """Fresh (un-memoised) check that ``owner`` may still read ``sid``: the
    login is still valid (``token``, a native login: not revoked or expired;
    otherwise the account is not disabled) and the conversation, if any, is
    still theirs or shared with them."""
    from sharedagents_web import access, server as _web_server

    if not _web_server._loopback_debug_bypass_ok():
        if token:
            if _web_server.auth_store.token_active(token) is None:
                return False
        else:
            acct = _web_server.auth_store.account(owner)
            if acct is not None and acct["disabled"]:
                return False
    if not sid:
        return True
    return access.session_access(owner, sid, fresh=True) != access.NONE


def _stream_login() -> tuple[str, bool]:
    """How this stream's request was accepted: ``(native token, dashboard)``.

    A native login is re-checked by its token; an external-auth login's
    credential is not a native token — the account is re-checked every tick
    and the credential itself with the dashboard every
    ``_DASHBOARD_RECHECK_SECONDS``. The loopback debug identity has neither.
    """
    try:
        from flask import g

        from sharedagents_web.server import (
            AUTH_VIA_DASHBOARD,
            AUTH_VIA_NATIVE,
            _native_token_from_request,
        )

        via = str(getattr(g, "auth_via", "") or "")
        token = _native_token_from_request() if via == AUTH_VIA_NATIVE else ""
        return token, via == AUTH_VIA_DASHBOARD
    except Exception:  # noqa: BLE001 — outside a request (tests, direct calls)
        return "", False


def _dashboard_login_valid(owner: str) -> bool:
    """The dashboard still accepts the credential this stream was opened
    with, for the same person (fail closed: unreachable = no)."""
    try:
        from sharedagents_web import server as _web_server

        ok, info = _web_server.verify_dashboard_auth()
    except Exception:  # noqa: BLE001
        return False
    email = str((info or {}).get("email") or "") if ok else ""
    return bool(email) and email.strip().lower() == str(owner or "").strip().lower()


def _activity_hash_for(owner: str, sid: str) -> str:
    """Authorize and return the selected SID used for activity reads."""
    return _session_hash_for(owner, sid)


def _health_snapshot() -> dict[str, Any]:
    resp = gateway_request("GET", "/", timeout=10, retry_unavailable=False)
    return {"ok": True, "healthy": bool(resp is not None and resp.status_code == 200)}


def _approvals_snapshot(owner: str, sid: str = "") -> dict[str, Any]:
    resp = gateway_request(
        "GET",
        "/control/v1/orchestrator/approvals",
        timeout=10,
        retry_unavailable=False,
    )
    if resp is None or resp.status_code != 200:
        return {"ok": False, "approvals": []}
    try:
        raw = resp.json().get("approvals") or []
    except (AttributeError, ValueError):
        raw = []
    from sharedagents_web.approvals_map import map_gateway_approvals

    extra = (sid,) if sid else ()
    return {
        "ok": True,
        "approvals": map_gateway_approvals(
            raw,
            store,
            owner,
            extra_session_ids=extra,
        ),
    }


def _live_sessions_snapshot(owner: str) -> dict[str, Any]:
    """Sidebar pill data: only the sessions with live/waiting activity."""
    rows = _merged_sessions(owner, limit=200)
    live = [
        r
        for r in rows
        if r.get("live")
        or r.get("state")
        in (
            "streaming",
            "waiting",
            "prompting",
            "prefill",
            "reasoning",
            "tool_running",
            "ask",
            "paused",
            "compacting",
        )
    ]
    live.sort(key=lambda r: r.get("updated_at", ""), reverse=True)
    return {"ok": True, "sessions": live[:20]}


def _budget_snapshot(owner: str, sid: str) -> dict[str, Any]:
    from sharedagents_goal.goal_service import hierarchical_budget_view

    session_hash = _session_hash_for(owner, sid)
    if not session_hash:
        return {"ok": True, "session_id": sid, "budget": {}}
    try:
        budget = hierarchical_budget_view(session_hash)
    except Exception:  # noqa: BLE001
        return {"ok": False, "session_id": sid, "budget": {}}
    goal = None
    try:
        from sharedagents_goal.goal_registry import get_goal_registry
        from sharedagents_goal.goal_service import stamp_budget_from_goal
        from sharedagents_web.routes.goals import _web_goal_detail

        goal_id = get_goal_registry().chat_goal_id_for_root(session_hash)
        if goal_id:
            goal = _web_goal_detail(goal_id)
        budget = stamp_budget_from_goal(budget, goal)
    except Exception:  # noqa: BLE001
        pass
    return {"ok": True, "session_id": sid, "budget": budget or {}}


def _goal_snapshot(owner: str, sid: str) -> dict[str, Any]:
    session_hash = _session_hash_for(owner, sid)
    if not session_hash:
        return {"ok": True, "session_id": sid, "goal_id": "", "goal": None}
    try:
        from sharedagents_goal.goal_registry import get_goal_registry
        from sharedagents_web.routes.goals import _web_goal_detail

        goal_id = get_goal_registry().chat_goal_id_for_root(session_hash)
        if not goal_id:
            return {"ok": True, "session_id": sid, "goal_id": "", "goal": None}
        detail = _web_goal_detail(goal_id)
    except Exception:  # noqa: BLE001
        return {"ok": False, "session_id": sid, "goal_id": "", "goal": None}
    return {
        "ok": detail is not None,
        "session_id": sid,
        "goal_id": goal_id,
        "goal": detail,
    }


def _live_snapshot(owner: str, sid: str) -> dict[str, Any]:
    session_hash = _activity_hash_for(owner, sid)
    if not session_hash:
        return {
            "ok": True,
            "session_id": sid,
            "session_state": "idle",
            "tree_busy": False,
            "parent_session_id": "",
            "subagents": [],
        }
    resp = gateway_request(
        "GET",
        f"/control/v1/orchestrator/sessions/{session_hash}/activity",
        timeout=5,
        retry_unavailable=False,
    )
    data = {}
    if resp is not None and resp.status_code == 200:
        try:
            data = resp.json()
        except (AttributeError, ValueError):
            data = {}
    if not isinstance(data, dict):
        data = {}
    kids = data.get("subagents")
    payload = {
        "ok": True,
        "session_id": sid,
        "session_state": data.get("session_state") or "idle",
        "tree_busy": bool(data.get("tree_busy")),
        "parent_session_id": str(data.get("parent_session_id") or ""),
        "subagents": list(kids) if isinstance(kids, list) else [],
    }
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
        "snapshot_seq",
    ):
        if key in data:
            payload[key] = data[key]
    return payload


def _footer_chrome(
    owner: str,
    sid: str,
    footer: dict[str, Any] | None,
    live: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stamp footer ``current`` from the picker chrome and live session status.

    With a live activity snapshot the sub rows are reconciled against it here
    (:func:`reconcile_sub_liveness`), so the rows' dots and ``counts.active``
    are decided once, server side, from the same data.
    """
    from sharedagents_session.session_activity import TREE_BUSY_STATES
    from sharedagents_web.agent_monitor_footer import (
        reconcile_sub_liveness,
        stamp_footer_from_web_row,
    )

    row = store.get(sid, owner) if sid else None
    if not isinstance(row, dict):
        row = {}
    title = resolve_authoritative_session_title(
        str(row.get("id") or sid or ""),
        str(row.get("title") or ""),
    )
    live = live if isinstance(live, dict) else None
    session_state = ""
    live_flag = None
    if live:
        session_state = str(live.get("session_state") or live.get("state") or "")
        live_flag = bool(live.get("own_live") or session_state in TREE_BUSY_STATES)
    stamped = stamp_footer_from_web_row(
        footer,
        row,
        sid,
        title=title,
        state=session_state,
        live=live_flag,
    )
    if live:
        stamped = reconcile_sub_liveness(stamped, live)
    return stamped


def _unbound_tree_snapshot(owner: str, sid: str) -> dict[str, Any]:
    from sharedagents_web.agent_monitor_footer import unbound_footer_payload

    row = store.get(sid, owner) if sid else None
    if not isinstance(row, dict):
        row = {}
    agent = store.display_agent(row)
    return {
        "ok": True,
        "session_id": sid,
        "tree": {"agents": [], "summary": [], "totals": {}},
        "footer": _footer_chrome(
            owner,
            sid,
            unbound_footer_payload(
                agent,
                str(row.get("title") or ""),
                str(row.get("model") or row.get("engine_model") or ""),
            ),
        ),
    }


def _tree_node_for_hash(tree: dict[str, Any] | None, viewed: str) -> dict[str, Any] | None:
    """Walk the gateway tree for the viewed session node."""
    wanted = str(viewed or "").strip()
    if not wanted or not isinstance(tree, dict):
        return None
    found: dict[str, Any] | None = None

    def walk(node: Any) -> None:
        nonlocal found
        if found is not None or not isinstance(node, dict):
            return
        if str(node.get("hash") or "").strip() == wanted:
            found = node
            return
        for child in node.get("children") or []:
            walk(child)

    for bucket in tree.get("agents") or []:
        if not isinstance(bucket, dict):
            continue
        for root in bucket.get("roots") or []:
            walk(root)
            if found is not None:
                return found
    return found


def _apply_inflight(row: dict[str, Any]) -> bool:
    raw = row.get("apply_inflight_at")
    try:
        stamped = float(raw)
    except (TypeError, ValueError):
        return False
    return (time.time() * 1000.0) - stamped < _APPLY_INFLIGHT_MAX_MS


def mark_apply_inflight(row: dict[str, Any] | None, **_kwargs: Any) -> None:
    """Pin a web-initiated apply so history write-through cannot revert it."""
    if not isinstance(row, dict):
        return
    clear_apply_inflight(row)
    row["apply_inflight_at"] = int(time.time() * 1000)


def clear_apply_inflight(row: dict[str, Any] | None) -> None:
    if isinstance(row, dict):
        for key in ("apply_inflight_at", "apply_inflight_applied", "apply_inflight_seen"):
            row.pop(key, None)


def _reconcile_web_row_from_tree(
    row: dict[str, Any] | None,
    tree: dict[str, Any] | None,
    viewed: str,
) -> dict[str, Any] | None:
    """Fill empty bound-row chrome from the tree. Never overwrite a set value.

    The 1s tree cache is what snapped Antigravity 3.8 back to 3.7. Last
    successful apply (and history write-through) own non-empty agent/model.
    """
    if not isinstance(row, dict) or not isinstance(tree, dict):
        return row
    if store.is_unused_web_placeholder(row) or _apply_inflight(row):
        return row
    node = _tree_node_for_hash(tree, viewed)
    if not node:
        return row
    agent = str(node.get("agent") or "").strip()
    model = str(node.get("model") or node.get("engine_model") or "").strip()
    changed = False
    if agent and not str(row.get("agent") or "").strip():
        row["agent"] = agent
        changed = True
    if model and not str(row.get("model") or "").strip():
        row["model"] = model
        changed = True
    if changed:
        try:
            store._save(row)
        except Exception:
            pass
    return row


def _tree_snapshot(owner: str, sid: str) -> dict[str, Any]:
    session_hash = _session_hash_for(owner, sid)
    row = store.get(sid, owner) if sid else None
    unbound = isinstance(row, dict) and store.is_unused_web_placeholder(row)
    if unbound or not session_hash:
        return _unbound_tree_snapshot(owner, sid)
    tree = _fetch_tree(
        False,
        session_id=session_hash,
        headers=gateway_principal_headers(owner),
        retry_unavailable=False,
    )
    from sharedagents_web.agent_monitor_footer import enrich_footer_with_history, footer_payload
    from sharedagents_web.server import GW_SESSIONS_DIR

    viewed = _activity_hash_for(owner, sid) or session_hash
    if tree is not None:
        row = store.get(sid, owner) if sid else None
        _reconcile_web_row_from_tree(row, tree, viewed)
    if tree is None:
        footer = _footer_chrome(
            owner,
            sid,
            enrich_footer_with_history(
                {"session": viewed, "rows": [], "subs": 0},
                viewed,
                GW_SESSIONS_DIR,
            ),
        )
        return {
            "ok": bool(footer.get("rows") or footer.get("subs")),
            "session_id": sid,
            "tree": None,
            "footer": footer,
        }
    return {
        "ok": True,
        "session_id": sid,
        "tree": tree,
        "footer": _footer_chrome(
            owner,
            sid,
            enrich_footer_with_history(
                footer_payload(tree, viewed),
                viewed,
                GW_SESSIONS_DIR,
            ),
        ),
    }


def _widgets_snapshot(owner: str, sid: str) -> dict[str, Any]:
    from sharedagents_web.server import GW_SESSIONS_DIR

    session_hash = _activity_hash_for(owner, sid)
    todos: list[Any] = []
    tasks: list[Any] = []
    status: dict[str, Any] = {}
    if session_hash:
        try:
            from sharedagents_agent.tools import load_todos, todos_path_for_session

            raw = load_todos(todos_path_for_session(GW_SESSIONS_DIR, session_hash)) or []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                todos.append(
                    {
                        "content": item.get("content") or item.get("text") or item.get("body") or "",
                        "status": ("completed" if item.get("status") == "done" else item.get("status") or "pending"),
                        "activeForm": item.get("activeForm") or "",
                    }
                )
            if todos:
                todos = [todo for todo in todos if todo.get("status") not in {"completed", "done"}]
        except Exception:  # noqa: BLE001
            pass
        try:
            path = GW_SESSIONS_DIR / f"{session_hash}.tasks.json"
            if path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                task_map = data.get("tasks") if isinstance(data, dict) else {}
                if isinstance(task_map, dict):
                    for tid, task in sorted(
                        task_map.items(),
                        key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 0,
                    ):
                        if not isinstance(task, dict):
                            continue
                        tasks.append(
                            {
                                "content": f"#{tid} {task.get('subject') or ''}",
                                "status": task.get("status") or "pending",
                            }
                        )
        except Exception:  # noqa: BLE001
            pass
    row = store.get(sid, owner) if sid else None
    if isinstance(row, dict):
        agent = store.display_agent(row)
        if agent:
            status["agent"] = agent
        for key in ("model", "effort", "mode"):
            if row.get(key):
                status[key] = row[key]
        if row.get("teacher_active") is not None:
            status["teacher_active"] = bool(row.get("teacher_active"))
        if row.get("teacher_enabled") is not None:
            status["teacher_enabled"] = bool(row.get("teacher_enabled"))
        if row.get("teacher_model"):
            status["teacher_model"] = row["teacher_model"]
    queue_items: list[Any] = []
    if session_hash:
        try:
            from sharedagents_core.attachment_context import project_queue_item
            from sharedagents_session.session_reader import fold_queue_state, tail_jsonl

            queue_items = [
                project_queue_item(item)
                for item in fold_queue_state(
                    event
                    for event, _ in tail_jsonl(
                        GW_SESSIONS_DIR / f"{session_hash}.jsonl",
                        0,
                    )
                )
            ]
        except Exception:  # noqa: BLE001
            queue_items = []
    return {
        "ok": True,
        "session_id": sid,
        "todos": todos,
        "tasks": tasks,
        "status": status,
        "queue": queue_items,
    }


def _current_model_from_footer(footer: dict[str, Any] | None) -> str:
    """Model on the painted ``current`` Agents row, if any."""
    if not isinstance(footer, dict):
        return ""
    current = footer.get("current")
    if isinstance(current, dict):
        model = str(current.get("model") or "").strip()
        if model:
            return model
    for row in footer.get("rows") or []:
        if isinstance(row, dict) and row.get("role") == "current":
            model = str(row.get("model") or "").strip()
            if model:
                return model
    return ""


def _session_snapshot(
    owner: str,
    sid: str,
    live: dict[str, Any],
    tree_footer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Active-session header for the ``state`` frame (mirrors the REST header)."""
    row = store.get(sid, owner) if sid else None
    if not isinstance(row, dict):
        return {}
    sid_key = str(row.get("id") or sid or "").strip()
    try:
        from sharedagents_web.routes.sessions import _fast_session_header

        fast_hdr = _fast_session_header(sid_key) if sid_key else {}
        if fast_hdr and not _apply_inflight(row):
            fast_model = str(fast_hdr.get("model") or "").strip()
            if fast_model and str(row.get("model") or "").strip() != fast_model:
                row["model"] = fast_model
                try:
                    store._save(row)
                except Exception:
                    pass
            fast_engine = str(fast_hdr.get("engine_model") or "").strip()
            if fast_engine and str(row.get("engine_model") or "").strip() != fast_engine:
                row["engine_model"] = fast_engine
    except Exception:
        pass
    footer_model = _current_model_from_footer(tree_footer)
    model = str(row.get("model") or "").strip() or footer_model or None
    engine = str(row.get("engine_model") or "").strip() or footer_model or None
    return {
        "id": row.get("id"),
        "title": resolve_authoritative_session_title(
            sid_key,
            row.get("title") or "New Chat",
        ),
        "agent": store.display_agent(row),
        "model": model,
        "engine_model": engine,
        "engine_model_alias": row.get("engine_model_alias"),
        "updated_at": row.get("updated_at"),
        "session_summary": row.get("session_summary"),
        "revision": int(row.get("revision") or 0),
        "live": bool(live.get("tree_busy")),
        "state": live.get("session_state") or "idle",
        "context_window": int(row.get("context_window") or 0),
        "last_prompt_tokens": int(row.get("last_prompt_tokens") or 0),
        "compaction_last_stats": row.get("compaction_last_stats"),
        "compaction_summary": row.get("compaction_summary"),
        "compaction_count": int(row.get("compaction_count") or 0),
    }


def _delegation_snapshot(owner: str, sid: str) -> list[dict[str, Any]]:
    """Small, owner-scoped job records for delegation cards, including reloads.

    A parent's turn can finish before its children. The existing state stream
    reconciles those cards even when the terminal log event was missed.
    """
    session_hash = _session_hash_for(owner, sid)
    if not session_hash:
        return []
    from sharedagents_agent.agents import async_jobs

    return [
        {
            key: job.get(key)
            for key in (
                "call_id",
                "job_id",
                "child_sid",
                "agent",
                "status",
                "ok",
            )
        }
        for job in async_jobs.list_jobs()
        if isinstance(job, dict) and job.get("parent_sid") == session_hash and job.get("call_id")
    ]


def _state_snapshot(owner: str, sid: str) -> dict[str, Any]:
    """One consolidated control payload, emitted as a single ``state`` frame.

    Model data belongs to the picker endpoints. Embedding the full catalogue
    here made 851 KB of every 857 KB tick, re-sent unchanged every four seconds.
    """
    from sharedagents_web.agent_monitor_footer import descendants_running

    live = (
        _live_snapshot(owner, sid)
        if sid
        else {
            "ok": True,
            "session_id": sid,
            "session_state": "idle",
            "tree_busy": False,
            "subagents": [],
        }
    )
    tree = _tree_snapshot(owner, sid)
    if isinstance(tree, dict):
        tree = dict(tree)
        tree["footer"] = _footer_chrome(owner, sid, tree.get("footer"), live=live)
    footer = tree.get("footer") if isinstance(tree, dict) else None
    live["descendants_running"] = descendants_running(footer)
    return {
        "ok": True,
        "session_id": sid,
        "session": _session_snapshot(owner, sid, live, tree_footer=footer),
        "live": live,
        "delegations": _delegation_snapshot(owner, sid),
        "health": _health_snapshot(),
        "approvals": _approvals_snapshot(owner, sid),
        "live_sessions": _live_sessions_snapshot(owner),
        "budget": _budget_snapshot(owner, sid),
        "goal": _goal_snapshot(owner, sid),
        "tree": tree,
        "widgets": _widgets_snapshot(owner, sid),
    }


# ── The single chat stream ─────────────────────────────────────────────


def _resume_gateway_url(url: str, frames: bytes) -> str:
    """Resume an upstream reopen after the last complete log frame forwarded."""
    offset = None
    for frame in frames.split(b"\n\n"):
        lines = frame.splitlines()
        if b"event: log" not in lines:
            continue
        data = b"\n".join(line[5:].lstrip() for line in lines if line.startswith(b"data:"))
        try:
            payload = json.loads(data)
        except (ValueError, UnicodeError):
            continue
        value = payload.get("offset") if isinstance(payload, dict) else None
        if type(value) is int and value >= 0:
            offset = value
    if offset is None:
        return url
    parts = urlsplit(url)
    query = [(key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True) if key != "from"]
    query.append(("from", str(offset)))
    return urlunsplit(parts._replace(query=urlencode(query)))


def iter_session_stream(
    owner: str,
    sid: str,
    gw_url: str | None,
    gateway_headers: dict[str, str],
    read_timeout: float,
    pending_gw_url: str | None = None,
) -> Iterator[bytes]:
    """One SSE stream: gateway transcript bytes plus one ``event: state`` tick.

    Corporate proxies that buffer or drop ``text/event-stream`` fall back to
    ``GET /api/sessions/<sid>/events`` polling — this generator stays SSE-only.

    A brand-new web chat has no JSONL yet, so ``gw_url`` is None and this mux
    only emits control frames. The tab that *sends* the first prompt reopens
    EventSource; every other browser would stay blind unless we attach
    ``pending_gw_url`` the moment the log appears.
    """
    started = time.monotonic()
    # The login this stream was opened with, re-checked on its own deadline.
    login_token, dashboard_login = _stream_login()
    next_dashboard_check = started + _DASHBOARD_RECHECK_SECONDS
    last_emit = started
    if sid:
        register_session_stream(sid)
    next_state = started + _STATE_TICK_SECONDS
    next_auth_check = started + _STATE_TICK_SECONDS
    last_state_sig = ""
    wake_seen = take_state_wake(sid)
    # One-shot near-term tick armed by a terminal frame. The sessions_meta
    # activity row can trail the forwarded turn_ended by a beat, so after the
    # immediate wake we re-read once shortly after to land the settled snapshot.
    followup_state_at = 0.0

    yield b"event: ready\ndata: {}\n\n"

    if sid:
        frame, last_state_sig = _state_or_heartbeat(
            _state_snapshot(owner, sid),
            last_state_sig,
        )
        yield frame

    gw_q: queue.Queue = queue.Queue()
    stop = threading.Event()
    reconnect_after = 0.0
    reconnect_backoff = 0.25

    def _read_gateway() -> None:
        if not gw_url:
            gw_q.put(("end", None))
            return
        try:
            import requests

            resp = requests.get(
                gw_url,
                headers=gateway_headers,
                stream=True,
                timeout=(5, read_timeout),
            )
        except Exception:  # noqa: BLE001
            gw_q.put(("err", None))
            return
        try:
            if resp.status_code != 200:
                gw_q.put(("err", None))
                return
            # SSE is line-oriented. ``chunk_size=None`` can buffer the chunked
            # upstream body until a large socket read completes, turning live
            # token events into 45-second bursts behind the WebUI relay.
            for line in resp.iter_lines(chunk_size=64):
                if stop.is_set():
                    break
                gw_q.put(("gw", line + b"\n"))
            gw_q.put(("end", None))
        except Exception:  # noqa: BLE001
            gw_q.put(("err", None))
        finally:
            try:
                resp.close()
            except Exception:  # noqa: BLE001
                pass

    if gw_url:
        threading.Thread(target=_read_gateway, daemon=True).start()
    pending = b""
    try:
        while time.monotonic() - started < _STATE_MAX_SECONDS:
            try:
                kind, payload = gw_q.get(timeout=0.5)
            except queue.Empty:
                kind, payload = None, None
            if kind == "gw" and payload:
                reconnect_backoff = 0.25
                reconnect_after = 0.0
                chunk = payload if isinstance(payload, (bytes, bytearray)) else bytes(payload)
                pending += chunk
                split = pending.rfind(b"\n\n")
                if split >= 0:
                    complete, pending = pending[: split + 2], pending[split + 2 :]
                    yield complete
                    if gw_url:
                        gw_url = _resume_gateway_url(gw_url, complete)
                    last_emit = time.monotonic()
                    # A terminal frame changes tree_busy/activity — emit the
                    # consolidated ``state`` now (same iteration) so the client
                    # reconciles Stop without its own /live poll, and arm one
                    # follow-up read for a settled activity row.
                    if sid and _frame_wakes_state(complete):
                        notify_state_changed(sid)
                        followup_state_at = time.monotonic() + 0.6
            elif kind == "err":
                if pending:
                    yield pending
                    pending = b""
                if gw_url:
                    # A cancel / worker abort often resets the upstream HTTP
                    # tail. Killing this mux used to force every browser to
                    # close EventSource, wait 1s, and sometimes auto-downgrade
                    # to 4s polling — so the tab that did not click Stop missed
                    # turn_ended and the next turn_started. Re-attach the
                    # gateway reader the same way a clean ``end`` does; the
                    # browser connection stays up for state + the next turn.
                    reconnect_after = time.monotonic() + reconnect_backoff
                    reconnect_backoff = min(reconnect_backoff * 2.0, 4.0)
                    if sid:
                        notify_state_changed(sid)
                        frame, last_state_sig = _state_or_heartbeat(
                            _state_snapshot(owner, sid),
                            last_state_sig,
                        )
                        yield frame
                        last_emit = time.monotonic()
                else:
                    yield _emit_frame("gw_error", {})
                    break
            elif kind == "end":
                if pending:
                    yield pending
                    pending = b""
                if gw_url:
                    # Keep this stream open for ``state`` frames; reopen the
                    # gateway reader so a later turn still rides the same mux.
                    reconnect_after = time.monotonic() + reconnect_backoff
                    reconnect_backoff = min(reconnect_backoff * 2.0, 4.0)
            now = time.monotonic()
            # Authorization keeps its own deadline: wake-driven state frames
            # re-arm ``next_state``, so a busy conversation must not postpone
            # it. A revoked login or share ends the live view; the reconnect
            # then meets the gate's 401 / 404.
            if now >= next_auth_check:
                next_auth_check = now + _STATE_TICK_SECONDS
                if not _still_reachable(owner, sid, login_token):
                    break
                if dashboard_login and now >= next_dashboard_check:
                    next_dashboard_check = now + _DASHBOARD_RECHECK_SECONDS
                    if not _dashboard_login_valid(owner):
                        break
            wake = take_state_wake(sid)
            if sid and wake != wake_seen:
                # A writer bumped the generation — emit now, then resume the tick.
                frame, last_state_sig = _state_or_heartbeat(
                    _state_snapshot(owner, sid),
                    last_state_sig,
                )
                yield frame
                last_emit = time.monotonic()
                wake_seen = wake
                next_state = now + _STATE_TICK_SECONDS
            elif followup_state_at and now >= followup_state_at:
                # Settled re-read armed by a terminal frame (see above).
                followup_state_at = 0.0
                frame, last_state_sig = _state_or_heartbeat(
                    _state_snapshot(owner, sid),
                    last_state_sig,
                )
                yield frame
                last_emit = time.monotonic()
            elif now >= next_state:
                frame, last_state_sig = _state_or_heartbeat(
                    _state_snapshot(owner, sid),
                    last_state_sig,
                )
                yield frame
                last_emit = time.monotonic()
                next_state = now + _STATE_TICK_SECONDS
            if not gw_url and pending_gw_url and sid and not stop.is_set() and _gw_session_log_exists(sid):
                # First prompt just created the JSONL. Attach this already-open
                # browser mux so a second computer sees the turn without
                # waiting for EventSource to die and retry.
                gw_url = pending_gw_url
                threading.Thread(target=_read_gateway, daemon=True).start()
            if gw_url and reconnect_after and now >= reconnect_after and not stop.is_set():
                reconnect_after = 0.0
                threading.Thread(target=_read_gateway, daemon=True).start()
            if time.monotonic() - last_emit >= _STATE_HEARTBEAT_SECONDS:
                yield _emit_frame("hb", {})
                last_emit = time.monotonic()
        yield _emit_frame("reconnect", {})
    except GeneratorExit:
        return
    finally:
        if sid:
            unregister_session_stream(sid)
        stop.set()
