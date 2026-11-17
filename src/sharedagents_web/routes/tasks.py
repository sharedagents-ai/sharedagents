"""sharedagents_web.routes.tasks — the "Tasks" tab: a manual list of sessions the
operator explicitly chose to track, styled after the real Claude Code
`claude agents` view (code.claude.com/docs/en/agent-view). That view's exact
state machine is: Needs input (blocked on a decision only the operator can
make), Working (running), Completed (finished/failed/stopped). sharedagents chat
sessions have no bounded "process finished" signal the way a cloud task
does — a chat just idles forever, resumable — so the closest faithful
adaptation, decided with the operator, is client-side (see tasks.js):
Working = actively running OR ``waiting`` on a dispatched subagent/async job
(session_index.py only sets `waiting` when the parent's own turn ended but a
child is still live — never for "blocked on a human", which has no distinct
state); Completed = plain ``idle`` (turn ended, nothing pending) as the only
clean "at rest" signal; everything else (paused, error, closed, stale,
never-started) defaults to Needs input, same place `error` lands per the
operator's instruction. Two derived exceptions: a session whose own JSONL
ends on an unanswered ``question`` reads as ``"ask"`` — Needs input —
straight from the session log (the question tool parks the turn without
ever spawning a process, so no session state change exists to show; the log
signal works for any frontend and any deployment, incl. the split
north/south shape where the gateway approvals registry lives in another
process and requires a relay; it is NOT gated on the tree node being
``live`` — a parked question writes nothing, so the 300s freshness gate
would flip the node non-live 5 minutes into a 30-minute ask window, and the
ask-timeout age check is the staleness boundary instead, mirroring the
approvals overlay);
and a pending gateway approval (ask_user / tool_permission) on
the session SID is an ADDITIONAL ``"ask"`` signal.
Untracking (DELETE) ARCHIVES the row (``archived_at`` stamp; hidden by
default, revealed by ``?include_archived=1``) and never touches the
underlying session.

Title: the operator's typed label (if any) is the task's PERMANENT name on
this board — never superseded by the chat's own evolving title. A labelled
task also PINS its PARENT session's title to that label (``lock: true``, see
:mod:`sharedagents_web.task_recap`), so the sidebar shows the same name and the
title agent never spends a side call on that parent. Child sessions are never
pinned: they have no board row, so they keep their own automatic titles. Only
a task created with no label at all falls back to today's behaviour (live
tree title / web-store title / bare default — see api_tasks_list).

Completed rows also carry their recap (``summary*`` fields, written by
``task_recap``): the board renders it as a collapsed-by-default subcard the
operator expands. A follow-up verdict for the latest completed turn moves
the task to Needs input. Live work and archived tasks keep their own cards;
an older recap never overrides a new turn.
"""

from __future__ import annotations

import re
import sqlite3
import time

from flask import Blueprint, g, jsonify, render_template, request

from sharedagents_agent.orchestrator import LIVE_FRESHNESS_SECONDS
from sharedagents_session.ask_pending import session_pending_ask
from sharedagents_session.session_activity import (
    TREE_BUSY_STATES,
    count_fresh_live_descendants,
    derive_activity,
    legacy_phase,
    row_stale_for_display,
    tasks_card,
)
from sharedagents_web import task_recap
from sharedagents_web.routes.topics import _topics_payload
from sharedagents_web.server import (
    BASE_PATH,
    DEFAULT_AGENT,
    GW_SESSIONS_DIR,
    REALTIME_URL,
    _fetch_tree,
    _merged_sessions,
    gateway_request,
    require_auth,
    store,
)
from sharedagents_web.task_board import get_task_board_registry

bp = Blueprint("tasks", __name__)


@bp.route("/api/tasks/summary-model", methods=["GET", "PUT"])
@require_auth
def api_tasks_summary_model():
    from sharedagents_agent.agent import AgentConfig
    from sharedagents_session.session_recap import recap_model_rows
    from sharedagents_web import access, server

    rows = [
        row
        for row in recap_model_rows(AgentConfig.load("jarvis"))
        if access.model_refusal(g.user_email, row["id"], row["key"]) is None
    ]
    key = "tasks.summary_model:" + g.user_email
    if request.method == "PUT":
        body = request.get_json(silent=True)
        model = body.get("model") if isinstance(body, dict) else None
        if not isinstance(model, str) or (model and model not in {r["key"] for r in rows}):
            return jsonify({"error": "Choose an available recap model."}), 400
        server.web_state.kv_set(key, model)
    return jsonify(
        {
            "selected": server.web_state.kv_get(key) or "",
            "models": [{"id": r["key"], "label": r.get("alias") or r["id"]} for r in rows],
        }
    )


#: Approvals that mean "blocked on the operator" — same semantics as the
#: web approvals panel (routes/sessions.py api_approvals).
_ASK_APPROVAL_TYPES = ("ask_user", "tool_permission")

# The Tasks board has a strict first-paint budget.  Its shared sidebar only
# renders 50 sessions and SessionStore itself retains at most 200 list rows, so
# asking the gateway for a 500-row global DAG is pure latency.  These two
# bounded reads can consume at most 700 ms before the local metadata fallback
# completes the board.
_TASKS_BOOTSTRAP_ROOT_LIMIT = 200
_TASKS_ROOT_TIMEOUT_S = 0.55
_TASKS_APPROVAL_TIMEOUT_S = 0.15

#: Shape of a real session id/hash (session.py mints ``YYYYMMDDTHHMMSSffffff``
#: + ``-`` + hex; the web store uses dashed UUIDs). Anything else reaching the
#: path join below would fabricate a bogus JSONL path, so gate the join on
#: this pattern (no slashes, no dots, no spaces).
_GW_SID_RE = re.compile(r"[0-9A-Za-z_-]{6,128}")


def _session_log_pending_ask(sid: str) -> bool:
    """True when the session's OWN JSONL tail ends on an unanswered ask."""
    if not sid or not _GW_SID_RE.fullmatch(sid):
        return False
    return session_pending_ask(GW_SESSIONS_DIR, sid)


def _fetch_approval_records(
    *,
    timeout_s: float = 10.0,
    retry_unavailable: bool = True,
) -> list[dict]:
    """Fetch the gateway approval registry for pre-bundle compatibility."""
    resp = gateway_request(
        "GET",
        "/control/v1/orchestrator/approvals",
        timeout=timeout_s,
        retry_unavailable=retry_unavailable,
    )
    if resp is None or resp.status_code != 200:
        return []
    try:
        raw = resp.json().get("approvals") or []
    except ValueError:
        return []
    return [rec for rec in raw if isinstance(rec, dict)]


def _fetch_task_roots(
    *,
    limit: int = _TASKS_BOOTSTRAP_ROOT_LIMIT,
    timeout_s: float = _TASKS_ROOT_TIMEOUT_S,
) -> list[dict] | None:
    """Cheap, bounded root-session snapshot for the Tasks bootstrap.

    The full orchestrator tree refreshes/scans every JSONL before building a
    DAG.  Production has thousands of session rows, so that read can take
    multiple seconds even though the board tracks only a handful.  The roots
    endpoint is a direct indexed ``sessions_meta`` query.  ``None`` means the
    gateway missed the small latency budget; callers then use exact local
    metadata lookups for task state and the web store for sidebar rows.
    """
    resp = gateway_request(
        "GET",
        "/control/v1/orchestrator/sessions",
        params={"roots": "1", "limit": max(1, int(limit))},
        timeout=timeout_s,
        retry_unavailable=False,
    )
    if resp is None or resp.status_code != 200:
        return None
    try:
        payload = resp.json()
    except ValueError:
        return None
    rows = payload.get("sessions") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return None
    return [dict(row) for row in rows if isinstance(row, dict)]


def _ask_pending_hashes(records: list[dict] | None = None) -> set[str]:
    """Physical session hashes with a pending operator decision.

    The gateway owns the authoritative pending registry (control_endpoint
    ``_ASK_PENDING_META`` / ``_PERM_PENDING_META``). While a session is
    parked on an ask or a tool-permission approval its tree node stays in
    whatever live state the turn was in (e.g. ``tool_running`` — the
    ``question`` tool short-circuits before spawn, tool_execution.py), so
    the tasks board would otherwise show it as Working. Overlaying the
    pending hashes yields a derived ``"ask"`` state (client buckets it
    under Needs input) WITHOUT inventing a session state the lifecycle
    machine does not know — and without restarting the gateway (which
    would kill the parked ask).
    """
    raw = _fetch_approval_records() if records is None else records
    hashes: set[str] = set()
    for rec in raw:
        if not isinstance(rec, dict):
            continue
        if rec.get("type") not in _ASK_APPROVAL_TYPES:
            continue
        h = str(rec.get("session_hash") or "").strip()
        if h:
            hashes.add(h)
    return hashes


def _owned_approvals(records: list[dict], owner: str) -> list[dict]:
    """Map gateway hashes to this operator's Web session ids."""
    gw_to_web = {}
    for rec in store.list_sessions(owner, limit=200):
        sid = rec.get("id")
        if sid:
            gw_to_web[sid] = sid
    out = []
    for item in records:
        gw = item.get("session_hash")
        if gw not in gw_to_web:
            continue
        mapped = dict(item)
        mapped["session_id"] = gw_to_web[gw]
        out.append(mapped)
    return out


_NEW_CHAT_PLACEHOLDER = re.compile(r"^New Chat(?:\s+-\s+\d{4,})?$")


def _is_placeholder_title(t: str) -> bool:
    """Same placeholder set server.py's ``_merged_sessions`` uses for the
    sidebar, so the Tasks board and the sidebar agree on when a title is
    "real" vs. still a stand-in."""
    t = (t or "").strip()
    return (
        not t
        or t.startswith("untitled-")
        or t in ("New Chat", "Untitled", "Session")
        or t.startswith("New session")
        or bool(_NEW_CHAT_PLACEHOLDER.match(t))
    )


#: A Dev conductor with no gateway turn stays ``unstarted``. The run status
#: is the board authority for those rows.
_DEV_TASK_CARD = {
    "done": "completed",
    "running": "working",
    "pr_open": "working",
    "blocked": "needs_input",
    "intake": "needs_input",
    "ready": "needs_input",
    "aborted": "needs_input",
}


def _dev_tasks_card(
    card: str,
    sdlc_status: str,
    *,
    archived: bool = False,
    state: str = "",
) -> str:
    """Fold SDLC status onto a conductor that never got a gateway turn.

    A live tree state (ask, working, idle) stays authoritative. Archived
    stays archived. Only ``unstarted`` rows use the run status.
    """
    if archived or card == "archived":
        return "archived"
    if str(state or "") not in {"", "unstarted"}:
        return card
    mapped = _DEV_TASK_CARD.get(str(sdlc_status or "").strip().lower())
    return mapped or card


def _tree_state_index(tree: dict | None) -> dict:
    """Flatten the agent tree into {gateway_hash: node}."""
    idx: dict[str, dict] = {}

    def walk(n):
        h = n.get("hash")
        if h:
            idx[h] = n
        for c in n.get("children") or []:
            walk(c)

    for a in (tree or {}).get("agents") or []:
        for r in a.get("roots") or []:
            walk(r)
    return idx


def _tree_eviction_fallback(idx: dict, session_ids: list[str]) -> dict[str, dict]:
    """Resolve tracked sessions absent from the snapshot via ``sessions_meta``.

    The orchestrator tree payload is capped (``DEFAULT_TREE_MAX_ROWS=200``,
    newest activity first), so a tracked session that fell
    outside the window has NO tree node even though it ran fine — the board
    would otherwise read it as never-started ("unstarted" → Needs input
    client-side) instead of showing its real state. ``sessions_meta.
    session_state`` is the register the tree itself is built from, so it is
    the faithful fallback for missing nodes.

    Returns the small tree-node shape the Tasks renderer consumes: state,
    title, agent, timestamps, live flag, and the canonical activity snapshot.
    Lanes already present in ``idx`` are never touched.

    Read-only and best-effort: opens the shared sessions.db with ``mode=ro``
    and a 100 ms busy bound; ANY failure (file missing, unreadable, locked,
    corrupt) degrades to "unstarted", never a slow/failed page. Uses the
    package helper ``sessions_db_path()`` (the ONE sessions.db every agent
    shares), never a hardcoded path.
    """
    missing = [h for h in dict.fromkeys(session_ids) if h and h not in idx]
    if not missing:
        return {}
    from sharedagents_core.layout import sessions_db_path

    db_path = sessions_db_path()
    if not db_path.exists():
        return {}
    now_ms = int(time.time() * 1000)
    cutoff_ms = now_ms - LIVE_FRESHNESS_SECONDS * 1000
    out: dict[str, dict] = {}
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=0.1)
        try:
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA busy_timeout=100")
            placeholders = ",".join("?" for _ in missing)
            try:
                rows = con.execute(
                    "SELECT session_hash, session_state, title, agent_name, "
                    "topic, last_turn_ts, created_at, live_turn_idx, "
                    "session_phase, session_attention, session_lifecycle, "
                    "session_last_outcome, session_phase_changed_at, "
                    "session_heartbeat_at, session_outcome_at, "
                    "session_state_updated_at FROM sessions_meta "
                    f"WHERE session_hash IN ({placeholders})",
                    missing,
                ).fetchall()
            except sqlite3.OperationalError as exc:
                # Rolling migration / legacy fixture: state alone still keeps
                # an old completed task out of the "never started" bucket.
                if "no such column" not in str(exc).lower():
                    raise
                rows = con.execute(
                    f"SELECT session_hash, session_state FROM sessions_meta WHERE session_hash IN ({placeholders})",
                    missing,
                ).fetchall()
            for raw in rows:
                row = dict(raw)
                sid = str(row["session_hash"])
                state = str(row.get("session_state") or "idle")
                descendant_count = count_fresh_live_descendants(
                    con,
                    sid,
                    live_states=TREE_BUSY_STATES,
                    cutoff_ms=cutoff_ms,
                )
                own_live = (
                    row.get("live_turn_idx") is not None
                    and legacy_phase(
                        state,
                        str(row.get("session_phase") or ""),
                        live_turn_idx=row.get("live_turn_idx"),
                    )
                    != "idle"
                )
                heartbeat = int(row.get("session_heartbeat_at") or row.get("session_state_updated_at") or 0)
                raw_stale = bool(heartbeat > 0 and state in TREE_BUSY_STATES and heartbeat < cutoff_ms)
                activity = derive_activity(
                    row,
                    now_ms=now_ms,
                    stale=row_stale_for_display(
                        raw_stale=raw_stale,
                        own_live=own_live,
                        descendant_count=descendant_count,
                    ),
                    descendant_count=descendant_count,
                )
                out[sid] = {
                    "state": state,
                    "title": str(row.get("title") or ""),
                    "agent": str(row.get("agent_name") or ""),
                    "topic": str(row.get("topic") or ""),
                    "last_turn_ts": row.get("last_turn_ts") or 0,
                    "last_activity_at": (row.get("last_turn_ts") or row.get("created_at") or 0),
                    "live": bool(activity.get("tree_busy") and not activity.get("stale")),
                    "activity": activity,
                    **activity,
                }
        finally:
            con.close()
    except Exception:  # noqa: BLE001 — the board must degrade, not 500
        return {}
    return out


def _ts_key(value) -> float:
    """Normalise a last-activity stamp to a comparable epoch float.

    The agent tree reports numeric epochs (seconds, occasionally ms) while the
    web session store reports ISO strings; sorting the raw mix as strings put
    every tree-backed row on the wrong side of every store-backed one.
    """
    if value is None or value == "":
        return 0.0
    if isinstance(value, (int, float)):
        v = float(value)
        return v / 1000.0 if v > 1e12 else v
    from datetime import datetime, timezone

    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError:
        return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _session_for_task(sdata: dict) -> str:
    """Return the tracked session SID without identity translation."""
    return str(sdata.get("id") or "").strip()


def _task_topic(sdata: dict, node: dict) -> str:
    """Topic shown on a board row.

    Prefer the web store (what New Task wrote), then the tree / sessions_meta
    node. ``general`` is the absence of a topic — same rule as the sidebar.
    """
    for raw in (sdata.get("topic"), node.get("topic")):
        topic = str(raw or "").strip()
        if topic and topic.casefold() != "general":
            return topic
    return ""


@bp.route("/tasks")
@require_auth
def tasks_page():
    from sharedagents_web import access

    return render_template(
        "tasks.html", base=BASE_PATH, realtime_url=REALTIME_URL, can_manage_tasks=access.viewer_is_admin()
    )


@bp.route("/api/tasks", methods=["GET"])
@require_auth
def api_tasks_list():
    # Archived rows are hidden by default; the client asks
    # for them explicitly with ?include_archived=1.
    include_archived = request.args.get("include_archived") in ("1", "true", "yes")
    bootstrap = request.args.get("bootstrap") in ("1", "true", "yes")
    reg = get_task_board_registry()
    rows = reg.list(owner=g.user_email, include_archived=include_archived)
    from sharedagents_web import access

    shared_ids = access.visible_shared_ids(g.user_email)
    if shared_ids:
        rows += [
            row
            for row in reg.list()
            if row.get("owner") != g.user_email
            and row["session_id"] in shared_ids
            and access.session_access(g.user_email, row["session_id"]) >= access.VIEW
        ]
    if not rows and not bootstrap:
        # No tracked lane means there is no state to enrich. Avoid waiting on
        # the gateway for a response that can only remain an empty list.
        return jsonify({"items": [], "gw_error": False})
    bootstrap_roots: list[dict] | None = None
    if bootstrap:
        # The browser makes one bootstrap request, but that request must not
        # build the global agent DAG.  Pending approvals and the indexed roots
        # projection each have a hard, fail-soft latency bound; exact task
        # states are filled from sessions_meta below if either misses it.
        approval_records = _fetch_approval_records(
            timeout_s=_TASKS_APPROVAL_TIMEOUT_S,
            retry_unavailable=False,
        )
        bootstrap_roots = _fetch_task_roots()
        tree = {
            "agents": [{"agent": DEFAULT_AGENT, "roots": bootstrap_roots or []}],
        }
        ask_pending_hashes = _ask_pending_hashes(approval_records)
    else:
        tree = _fetch_tree(True, include_approvals=True)
        bundled_approvals = tree.get("approvals") if isinstance(tree, dict) and "approvals" in tree else None
        if isinstance(bundled_approvals, list):
            approval_records = [rec for rec in bundled_approvals if isinstance(rec, dict)]
            ask_pending_hashes = _ask_pending_hashes(approval_records)
        else:
            # A rolling deployment can briefly pair the new WebUI with an
            # older gateway that does not understand ?approvals=1.
            approval_records = []
            ask_pending_hashes = _ask_pending_hashes()

    idx = _tree_state_index(tree)

    # Resolve every tracked row once so the tree-eviction fallback can fill all
    # missing nodes with one batched query.
    task_sessions = [store.get(row["session_id"]) or {} for row in rows]
    resolved: list[tuple[dict, dict, str]] = []
    for row, sdata in zip(rows, task_sessions):
        resolved.append(
            (
                row,
                sdata,
                _session_for_task(sdata) or row["session_id"],
            )
        )

    # Tree-eviction fallback: the tree snapshot is capped at
    # DEFAULT_TREE_MAX_ROWS (200) newest-first rows, so a tracked lane can be
    # absent from it even though it ran fine. The bootstrap always performs
    # this exact, bounded read because it intentionally gives up waiting for a
    # slow roots response; the legacy non-bootstrap API preserves its existing
    # gateway-failure behavior.
    meta_nodes = {} if tree is None and not bootstrap else _tree_eviction_fallback(idx, [gw for _, _, gw in resolved])

    try:
        from sharedagents_web.dev_sdlc import get_sdlc_store

        conductor_runs = get_sdlc_store().conductor_run_meta([row["session_id"] for row in rows])
    except Exception:  # noqa: BLE001 — Tasks still lists chats if SDLC is down
        conductor_runs = {}

    # A task has no share flag of its own: it is shared exactly when its
    # conversation is. One bulk read gives every row the sharing payload the
    # chat header renders (lock / people / globe).
    sharing_by_sid = access.sharing_payloads(row["session_id"] for row in rows)

    items = []
    for row, sdata, session_id in resolved:
        node = idx.get(session_id) or {}
        if not node and session_id in meta_nodes:
            node = meta_nodes[session_id]
        # No tree node normally means no turn has ever run for this session
        # yet — BUT the tree payload is capped (DEFAULT_TREE_MAX_ROWS=200,
        # newest activity first), so an idle tracked lane can be EVICTED from
        # the snapshot even though it ran fine; such sessions are resolved from
        # the durable sessions_meta register above. Only a lane in NEITHER
        # source is truly never-started: "unstarted" is NOT the same as
        # "idle" (a turn ended cleanly) — a never-started task must not read
        # as Completed before it has done anything, so it gets its own value
        # and falls into the "needs input" catch-all client-side (nothing
        # done yet, waiting on the operator to start it).
        state = node.get("state") or "unstarted"
        pending_kind = None
        # Primary ask signal (log-derived): a session whose own JSONL ends on
        # an unanswered question is parked on the operator — read as "ask" no
        # matter what the tree reports. This is the mechanism that works in
        # EVERY deployment: it derives from the session log itself, not from
        # the gateway's volatile approvals registry (which requires the
        # north-to-south relay below in a split deployment). NOT gated on tree
        # `live`: a parked question writes nothing, so the 300s
        # freshness gate would flip the node non-live 5 minutes into a
        # 30-minute ask window; the ask-timeout age check inside
        # _session_log_pending_ask is the staleness boundary instead
        # (mirrors the approvals overlay below, which has no live gate).
        if _session_log_pending_ask(session_id):
            state = "ask"
            pending_kind = "question"
        # Additional ask signal: a pending operator
        # decision (ask_user / tool_permission) on THIS session's
        # hash reads as "ask" no matter what state the tree reports — the
        # session is parked waiting on the human, not working.
        if session_id in ask_pending_hashes:
            state = "ask"
            pending_kind = "permission"
        activity = node.get("activity")
        if not isinstance(activity, dict):
            activity = derive_activity(
                node,
                now_ms=int(time.time() * 1000),
                tree_busy=bool(node.get("live") or node.get("subagent_count")),
                stale=bool(node.get("stale")),
                pending_action=pending_kind,
                descendant_count=int(node.get("active_descendant_count") or 0),
            )
        elif pending_kind:
            # Re-derive only the attention/display precedence; the tree's
            # durable phase/outcome/heartbeat remain authoritative.
            activity = derive_activity(
                {**node, **activity},
                now_ms=int(time.time() * 1000),
                tree_busy=True,
                stale=bool(activity.get("stale")),
                pending_action=pending_kind,
                descendant_count=int(activity.get("active_descendant_count") or 0),
            )
        last_activity = (
            node.get("last_activity_at") or node.get("last_turn_ts") or sdata.get("updated_at") or row["created_at"]
        )
        # Title precedence: the operator's own typed label is the task's
        # PERMANENT identity on this board — it must not get silently
        # replaced by whatever the chat evolves into (the original design
        # treated it as a placeholder "until something better shows up",
        # which meant the exact text the operator typed at creation was
        # never actually shown once the session had any real activity —
        # not what a task board's name is for). Only fall back to the live
        # tree title / web-store title / bare default for a task that was
        # never given a label at all.
        tree_title = str(node.get("title") or "").strip()
        store_title = str(sdata.get("title") or "").strip()
        label = str(row["label"] or "").strip()
        if label:
            title = label
        elif tree_title and not _is_placeholder_title(tree_title):
            title = tree_title
        elif store_title and not _is_placeholder_title(store_title):
            title = store_title
        else:
            title = "New Chat"
        # The session's OWN title (what sharedagents's OpenCode title agent named
        # the conversation), surfaced alongside `title` so a labelled task can
        # show both: the operator's name for the task AND what the agent
        # decided the conversation is about. Empty when the session has no
        # real title yet (never-run, or still on a placeholder), or when it
        # would just repeat `title` — the client hides the line in that case.
        session_title = ""
        for candidate in (tree_title, store_title):
            if candidate and not _is_placeholder_title(candidate):
                session_title = candidate
                break
        if session_title == title:
            session_title = ""
        meta = conductor_runs.get(row["session_id"]) or {}
        sdlc_run_id = str(meta.get("run_id") or "")
        sdlc_status = str(meta.get("status") or "")
        card = tasks_card(
            state,
            last_outcome=str(activity.get("last_outcome") or ""),
            stale=bool(activity.get("stale")),
            tree_busy=bool(activity.get("tree_busy")),
            own_live=bool(activity.get("own_live")),
            descendant_count=int(activity.get("active_descendant_count") or 0),
            has_row=bool(node),
            archived=bool(row.get("archived_at")),
        )
        if card == "completed" and not task_recap.turn_stamp_ms(node.get("last_turn_ts")):
            # A materialized session can be idle before its first turn.
            card = "needs_input"
        card = _dev_tasks_card(
            card,
            sdlc_status,
            archived=bool(row.get("archived_at")),
            state=state,
        )
        completion_card = card
        card = task_recap.recap_card(
            row,
            card=card,
            parent_turn_stamp=task_recap.turn_stamp_ms(node.get("last_turn_ts")),
        )
        items.append(
            {
                "id": row["id"],
                "session_id": row["session_id"],
                # Live session at list time (tree / status). The click
                # target stays ``session_id`` (web UUID) so a later switch is
                # re-resolved on open instead of pinning a stale hash.
                "open_session_id": session_id,
                "title": title,
                "session_title": session_title,
                # The RAW tree title (never blanked), for the title-pin drift
                # check in task_recap.pin_due — ``session_title`` above is blanked
                # when it merely repeats the task name, which would hide drift.
                "session_node_title": tree_title,
                # Parent's own last turn stamp, raw (tree epoch seconds / ms or
                # the store's ISO). task_recap normalises it; the recap fires at
                # most once per NEW parent turn.
                "last_turn_ts": node.get("last_turn_ts") or "",
                "topic": _task_topic(sdata, node),
                "sdlc_run_id": sdlc_run_id,
                "sdlc_status": sdlc_status,
                "agent": sdata.get("agent") or node.get("agent") or "",
                "state": state,
                "card": card,
                "completion_card": completion_card,
                "shared_task": row.get("owner") != g.user_email,
                "sharing": sharing_by_sid.get(row["session_id"])
                or {
                    "visibility": "private",
                    "share_all": False,
                    "members": [],
                },
                "recap_archivable": row.get("owner") == g.user_email
                and task_recap.recap_archivable(
                    row,
                    card=card,
                    parent_turn_stamp=task_recap.turn_stamp_ms(node.get("last_turn_ts")),
                ),
                "ask_pending": activity.get("attention") in {"question", "permission"},
                "archived": bool(row.get("archived_at")),
                "archived_at": row.get("archived_at") or "",
                "live": bool(node.get("live")),
                "last_activity_at": last_activity,
                "created_at": row["created_at"],
                "activity": activity,
                # Completion recap and verdict. Read straight off the
                # registry row — never re-fetched from the gateway per poll.
                **task_recap.summary_fields(
                    row,
                    card=card,
                    parent_turn_stamp=task_recap.turn_stamp_ms(node.get("last_turn_ts")),
                ),
                **activity,
            }
        )
    items.sort(key=lambda x: _ts_key(x.get("last_activity_at")), reverse=True)
    gateway_failed = bootstrap_roots is None if bootstrap else tree is None
    payload = {"items": items, "gw_error": gateway_failed}
    if bootstrap:
        roots = bootstrap_roots or []
        sessions = _merged_sessions(
            g.user_email,
            limit=_TASKS_BOOTSTRAP_ROOT_LIMIT,
            gateway_roots=roots,
        )
        topics = _topics_payload(sessions, compact=True)["topics"]
        payload["snapshot"] = {
            "sessions": sessions[:50],
            "topics": topics,
            "approvals": _owned_approvals(approval_records, g.user_email),
        }
    # Completion recap + pinned task title, AFTER the payload is built: a slow
    # gateway side call must never add latency to this poll. request_recaps
    # only decides (pure due rule) and hands the work to a daemon thread.
    by_id = {row["id"]: row for row in rows}
    task_recap.request_recaps([(by_id[it["id"]], it) for it in items if it.get("id") in by_id])
    # Start the server-side sweeper once per process so recaps also happen
    # while nobody has this tab open (guarded; a no-op after the first call).
    task_recap.ensure_sweeper()
    return jsonify(payload)


@bp.route("/api/tasks", methods=["POST"])
@require_auth
def api_tasks_create():
    data = request.get_json() or {}
    label = str(data.get("title") or "").strip()[:120]
    agent = str(data.get("agent") or DEFAULT_AGENT)
    model = str(data.get("model") or "")
    topic = str(data.get("topic") or "").strip()[:120]
    if topic.casefold() == "general":
        topic = ""

    # Materialize the gateway session first. A web-only SID makes the first
    # bound chat fall back to a different ephemeral session; the tracked row
    # would then never see its turns, title pin, or completion recap.
    from urllib.parse import quote

    response = gateway_request(
        "POST",
        f"/control/v1/{quote(agent, safe='')}/sessions",
        json={},
        timeout=20,
        retry_unavailable=False,
    )
    try:
        created = response.json() if response is not None and response.status_code == 200 else {}
    except ValueError:
        created = {}
    sid = created.get("sessionId") if isinstance(created, dict) and created.get("ok") else None
    if not isinstance(sid, str) or not sid.strip():
        return jsonify({"error": "Could not create task session; retry when the gateway is available."}), 503

    sdata = store.create(g.user_email, agent, model, topic, session_id=sid)
    if label:
        # A labelled task's name IS the parent session's title (operator
        # intent). Write it now so board + sidebar agree immediately, then
        # task_recap pins it on the gateway with ``lock: true`` (which is what
        # stops the title agent from ever overwriting it). An UNLABELLED task
        # keeps its placeholder so the automatic title agent still names it —
        # see the task_board module docstring.
        updated = store.update_fields(sdata["id"], title=label)
        if updated:
            sdata = updated

    reg = get_task_board_registry()
    row_id = reg.add(sdata["id"], owner=g.user_email, label=label or None)
    return jsonify({"id": row_id, "session": sdata, "label": label}), 201


@bp.route("/api/tasks/<row_id>/describe", methods=["POST"])
@require_auth
def api_tasks_describe(row_id):
    """Name a fresh composer-created task; manual labels are never changed."""
    from urllib.parse import quote

    row = next((r for r in get_task_board_registry().list(owner=g.user_email) if r["id"] == row_id), None)
    if row is None:
        return jsonify({"error": "Task not found"}), 404
    sdata = store.get(row["session_id"], g.user_email)
    if not sdata or row.get("label") or sdata.get("submitted"):
        return jsonify({"error": "Only a new unlabelled task can be named automatically."}), 409
    body = request.get_json(silent=True)
    description = body.get("description") if isinstance(body, dict) else None
    if not isinstance(description, str) or not description.strip() or len(description) > 12_000:
        return jsonify({"error": "Describe the task in 1–12000 characters."}), 400
    topics = _topics_payload(_merged_sessions(g.user_email, limit=200), compact=True)["topics"]
    existing = {t["topic"].casefold(): t["topic"] for t in topics}
    existing.setdefault("general", "general")  # Existing unassigned bucket, even on an empty board.
    requested_topic = body.get("topic", "")
    if not isinstance(requested_topic, str) or (requested_topic and requested_topic.strip().casefold() not in existing):
        return jsonify({"error": "Select an existing topic or use Automatic topic."}), 400
    selected_topic = existing.get(requested_topic.strip().casefold()) if requested_topic else None
    candidates = (
        [selected_topic]
        if selected_topic
        else [name for key, name in existing.items() if key != "general"][:99] + [existing["general"]]
    )
    agent, sid = sdata.get("agent") or DEFAULT_AGENT, sdata["id"]
    response = gateway_request(
        "POST",
        f"/control/v1/{quote(agent, safe='')}/sessions/{quote(sid, safe='')}/task-metadata",
        json={"description": description, "topics": candidates},
        timeout=100,
        retry_unavailable=False,
    )
    try:
        metadata = response.json() if response is not None and response.status_code == 200 else {}
    except ValueError:
        metadata = {}
    title, topic = (metadata.get("title"), metadata.get("topic")) if isinstance(metadata, dict) else (None, None)
    if not (isinstance(title, str) and title.strip() and isinstance(topic, str) and topic.strip()):
        return jsonify({"error": "Could not create title and topic. Retry Send or open the created task."}), 503
    topic = selected_topic or existing.get(topic.strip().casefold())
    if topic is None:
        return jsonify({"error": "Automatic topic must match an existing topic. Retry Send or select one."}), 503
    updated = store.update_fields(sid, title=title.strip()[:120], topic="" if topic.casefold() == "general" else topic)
    if not updated:
        return jsonify({"error": "Could not save task title and topic."}), 503
    return jsonify({"title": updated["title"], "topic": updated["topic"]})


@bp.route("/api/tasks/<row_id>", methods=["DELETE"])
@require_auth
def api_tasks_remove(row_id):
    reg = get_task_board_registry()
    if not reg.remove(row_id, owner=g.user_email):
        return jsonify({"error": "not_found"}), 404
    return jsonify({"ok": True})


@bp.route("/api/tasks/archive", methods=["POST"])
@require_auth
def api_tasks_archive():
    data = request.get_json(silent=True)
    ids = data.get("ids") if isinstance(data, dict) else None
    if (
        not isinstance(ids, list)
        or not ids
        or len(ids) > 500
        or any(not isinstance(row_id, str) or not row_id.strip() for row_id in ids)
    ):
        return jsonify({"error": "Select between 1 and 500 tasks to archive."}), 400
    selected = list(dict.fromkeys(ids))
    reg = get_task_board_registry()
    owned = {row["id"] for row in reg.list(owner=g.user_email)}
    stamps = {}
    # Recheck current state: an old browser selection must not archive a
    # running task, a newer unfinished turn, or somebody else's row.
    for row, item in _sweep_pairs():
        row_id = row["id"]
        if row_id not in owned or row_id not in selected:
            continue
        stamp = task_recap.turn_stamp_ms(item.get("last_turn_ts"))
        if task_recap.recap_archivable(row, card=item.get("card"), parent_turn_stamp=stamp):
            stamps[row_id] = stamp
    archived = reg.archive_completed(stamps, owner=g.user_email)
    return jsonify(
        {"ok": True, "archived_ids": archived, "skipped_ids": [row_id for row_id in selected if row_id not in archived]}
    )


def _sweep_pairs() -> list[tuple[dict, dict]]:
    """Every active tracked row + its DERIVED card, for the recap sweeper.

    The sweeper runs with no HTTP request and no operator tab open, so it
    cannot use ``api_tasks_list``'s payload. It reuses the SAME derivation —
    the bounded roots projection, then the ``sessions_meta`` eviction
    fallback for rows outside the window, then ``tasks_card`` /
    ``_dev_tasks_card`` — so a card it acts on is the card the board shows.
    Best-effort by construction: any failure yields no pairs (nothing is
    recapped this pass, the next sweep retries).
    """
    try:
        reg = get_task_board_registry()
        rows = reg.list(owner=None, include_archived=False)
    except Exception:  # noqa: BLE001
        return []
    if not rows:
        return []
    try:
        roots = _fetch_task_roots()
        idx = _tree_state_index({"agents": [{"agent": DEFAULT_AGENT, "roots": roots or []}]})
        session_ids = [str(row["session_id"]) for row in rows]
        meta_nodes = _tree_eviction_fallback(idx, session_ids)
        try:
            from sharedagents_web.dev_sdlc import get_sdlc_store

            conductor_runs = get_sdlc_store().conductor_run_meta(session_ids)
        except Exception:  # noqa: BLE001 — SDLC is optional for a recap
            conductor_runs = {}
        now_ms = int(time.time() * 1000)
        ask_pending = _ask_pending_hashes(
            _fetch_approval_records(
                timeout_s=_TASKS_APPROVAL_TIMEOUT_S,
                retry_unavailable=False,
            )
        )
        pairs: list[tuple[dict, dict]] = []
        for row in rows:
            sid = str(row["session_id"])
            node = idx.get(sid) or meta_nodes.get(sid) or {}
            if not node:
                # Neither the tree nor sessions_meta knows this row: it never
                # ran a turn, so there is nothing to recap and no title drift
                # the pin could correct.
                continue
            state = node.get("state") or "unstarted"
            if sid in ask_pending or _session_log_pending_ask(sid):
                state = "ask"
            activity = node.get("activity")
            if not isinstance(activity, dict):
                activity = derive_activity(
                    node,
                    now_ms=now_ms,
                    tree_busy=bool(node.get("live") or node.get("subagent_count")),
                    stale=bool(node.get("stale")),
                    descendant_count=int(node.get("active_descendant_count") or 0),
                )
            card = tasks_card(
                state,
                last_outcome=str(activity.get("last_outcome") or ""),
                stale=bool(activity.get("stale")),
                tree_busy=bool(activity.get("tree_busy")),
                own_live=bool(activity.get("own_live")),
                descendant_count=int(activity.get("active_descendant_count") or 0),
                has_row=True,
                archived=bool(row.get("archived_at")),
            )
            if card == "completed" and not task_recap.turn_stamp_ms(node.get("last_turn_ts")):
                card = "needs_input"
            meta = conductor_runs.get(sid) or {}
            card = _dev_tasks_card(
                card,
                str(meta.get("status") or ""),
                archived=bool(row.get("archived_at")),
                state=state,
            )
            completion_card = card
            card = task_recap.recap_card(
                row,
                card=card,
                parent_turn_stamp=task_recap.turn_stamp_ms(node.get("last_turn_ts")),
            )
            pairs.append(
                (
                    row,
                    {
                        "id": row["id"],
                        "title": str(row.get("label") or ""),
                        "card": card,
                        "completion_card": completion_card,
                        "last_turn_ts": node.get("last_turn_ts") or "",
                        "session_node_title": str(node.get("title") or ""),
                    },
                )
            )
        return pairs
    except Exception:  # noqa: BLE001 — the sweeper must never raise
        return []


task_recap.register_pairs_provider(_sweep_pairs)
