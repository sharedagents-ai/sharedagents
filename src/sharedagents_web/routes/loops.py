"""sharedagents_web.routes.loops — Loops page + loop lifecycle API routes.

Mirrors ``routes/goals.py`` conventions: the list/detail endpoints enrich
each loop with its bound session's title (web record first, then the
gateway log header) and a heartbeat-driven ``live`` flag so the UI can
render a running-but-stalled loop honestly. Lifecycle actions are
gateway-first (the supervisor's drive task lives on the LIVE gateway
agent) with a local fallback when the gateway is unreachable.

(create + pause/resume/delete API — loops are DELETED, never archived;
the legacy stop/archive verbs land on the same hard delete). Registered
by server.create_app().
"""

import math
import time
from datetime import datetime, timezone

from flask import Blueprint, g, jsonify, render_template, request

from sharedagents_web.routes.sessions import _resolve_gw_hash
from sharedagents_web.server import (
    BASE_PATH,
    DEFAULT_AGENT,
    GW_SESSIONS_DIR,
    REALTIME_URL,
    _gw_session_log_exists,
    auth_store,
    ensure_gw_session,
    gateway_request,
    gw_session_history,
    require_auth,
    store,
)

bp = Blueprint("loops", __name__)

#: Supervisor lease window (seconds) — same heartbeat contract as goals
#: (loop_registry._SUPERVISOR_LEASE_S). A ``running`` loop whose heartbeat
#: is older than this is a *stalled* loop: no drive task is actually on it.
try:
    from sharedagents_loop.loop_registry import _SUPERVISOR_LEASE_S as _LOOP_LEASE_S
except Exception:  # noqa: BLE001 — web must boot even if the loop package is absent
    _LOOP_LEASE_S = 120.0


def _web_load_agent(agent_name: str):
    from sharedagents_agent.agents.factory import load_agent

    return load_agent(agent_name)


def _session_for_loop(agent, session_id: str, sdata: dict | None):
    """Return the gateway session for a loop, materializing a fresh web SID.

    ``POST /api/sessions`` only writes the web row. A New-session loop then
    posts that SID before any chat turn has created ``<sid>.jsonl``, so
    ``agent.get_session`` is None. Reuse the same durable id so the
    supervisor can fire immediately.
    """
    sess = agent.get_session(session_id)
    if sess is not None:
        return sess
    if _gw_session_log_exists(session_id):
        return None
    try:
        return agent.create_session(
            session_id=session_id,
            workspace_origin="web",
            model=str((sdata or {}).get("model") or ""),
        )
    except Exception:  # noqa: BLE001 — caller maps None to 404
        return None


def _loop_registry():
    from sharedagents_loop.loop_registry import get_loop_registry

    return get_loop_registry()


def _web_supervise_loop(agent_name: str, loop_id: str) -> bool:
    """Kick loop-supervision on the live gateway agent (idempotent)."""
    resp = gateway_request(
        "POST",
        f"/control/v1/{agent_name}/loops/{loop_id}/supervise",
        json={},
        timeout=10,
    )
    if resp is None:
        return False
    try:
        data = resp.json()
    except ValueError:
        return resp.status_code == 200
    return resp.status_code == 200 and bool(data.get("ok"))


def _web_loop_action(agent_name: str, loop_id: str, action: str) -> dict | None:
    """Run a loop action (``pause`` / ``resume`` / ``archive``) on the LIVE
    gateway agent.

    The action must run where the supervisor task lives (the gateway daemon),
    otherwise pausing only writes the DB status and never cancels the running
    drive task — same contract as the goals action. Returns the gateway JSON
    on success, or ``None`` if the gateway is unreachable (caller falls back
    to a local registry write)."""
    resp = gateway_request(
        "POST",
        f"/control/v1/{agent_name}/loops/{loop_id}/{action}",
        json={},
        timeout=10,
    )
    if resp is None:
        return None
    try:
        return resp.json()
    except ValueError:
        return {"ok": resp.status_code == 200}


def _is_placeholder_loop_title(title: str) -> bool:
    """True when a stored title is still a *placeholder* form.

    Sessions are born with ``New Chat - HHMMSS`` (or ``New Chat`` /
    ``untitled-<hash>``); the gateway stamps a real 2–6-word title on
    turn 1 and refreshes it on turn 10 — the web-store row keeps the
    placeholder it was created with. Comparisons that only skip the exact
    string ``"New Chat"`` therefore let a stale web row shadow the real
    gateway title forever (the "loop card title never updates" report),
    so placeholder detection must match the gateway's own
    ``session_title.is_default_title`` rule.
    """
    if not title:
        return True
    if title == "New Chat":
        return True
    try:
        from sharedagents_session.session_title import is_default_title

        return bool(is_default_title(title))
    except Exception:  # noqa: BLE001 — degrade to the exact-match check
        return False


def _loop_session_title(session_id: str, *, ran_once: bool = False) -> str:
    """Title for the loop's bound session: web record first (in-process,
    cheap), then the gateway session-log header; ``"Session"`` as the
    last-resort label.

    Placeholder titles (``New Chat - HHMMSS``) never win: the gateway owns
    the title lifecycle and may already have stamped a real turn-1 title
    the web row has not seen, so a stale web placeholder must not shadow
    it. The gateway header is consulted once the loop has a reason to have
    one (``ran_once`` — a fire, a run, or a running state); a never-run
    loop bound to a web placeholder keeps that cheap placeholder without a
    per-poll gateway round-trip.
    """
    web_title = ""
    try:
        sdata = store.get(session_id)
    except Exception:  # noqa: BLE001 — title enrichment must never break the list
        sdata = None
    web_title = str((sdata or {}).get("title") or "").strip()
    if web_title and not _is_placeholder_loop_title(web_title):
        return web_title
    if web_title and not ran_once:
        # A web row exists but the loop never fired — the gateway cannot
        # have stamped a turn-1 title yet. Keep the placeholder.
        return web_title
    try:
        hist = gw_session_history(session_id)
    except Exception:  # noqa: BLE001
        hist = None
    gw_title = str((hist or {}).get("title") or "").strip()
    if gw_title and not _is_placeholder_loop_title(gw_title):
        return gw_title
    # Still untitled: the gateway placeholder (stamped when the session log
    # materialized) is the freshest description; the web row's is second.
    return gw_title or web_title or "Session"


def _loop_session_model(session_id: str) -> str:
    """Model stamped on the loop's bound session (``""`` when unknown).

    Read from the shared gateway ``sessions.db`` header
    (``sessions_meta.model``) — the same row the gateway session object
    hydrates — so the detail card shows the model the next per-run mint
    inherits. The web shares the gateway filesystem (``GW_SESSIONS_DIR``,
    same as ``_gw_session_log_exists``).
    """
    if not session_id:
        return ""
    try:
        from sharedagents_session.session_index import SessionIndex, connect_sessions_db

        sdir = GW_SESSIONS_DIR
        SessionIndex(sdir).tail(session_id)
        con = connect_sessions_db(sdir / "sessions.db")
        try:
            row = con.execute(
                "SELECT model FROM sessions_meta WHERE session_hash = ?",
                (session_id,),
            ).fetchone()
        finally:
            con.close()
        if not row:
            return ""
        return str(row[0] or "").strip()
    except Exception:  # noqa: BLE001 — model display is best-effort
        return ""


def _loop_live(record) -> bool:
    """True iff a ``running`` loop's supervisor heartbeat is inside its
    lease window OR its bound session subtree still has live work — the
    honest-liveness flag behind the UI's Stalled badge.

    Same contract as the goals reconciler: the drive task can exit (a
    manual-turn skip, a clean ``finally`` release) while the work it
    delegated keeps running under the loop's bound session. When the
    heartbeat is stale/absent we probe the session index with the SAME
    shared helper the goals reconciler uses
    (``goal_registry.session_work_live``), so goals and loops agree on
    what "still working" means. The probe only runs on the stale path."""
    if getattr(record, "status", "") != "running":
        return False
    hb = getattr(record, "supervisor_heartbeat_at", None)
    if hb:
        try:
            ts = datetime.fromisoformat(str(hb))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except Exception:  # noqa: BLE001
            ts = None
        if ts is not None and (datetime.now(timezone.utc) - ts).total_seconds() <= _LOOP_LEASE_S:
            return True
    from sharedagents_goal.goal_registry import session_work_live

    live, _src = session_work_live(
        str(getattr(record, "session_id", "") or ""),
    )
    return live


def _loop_in_flight(record) -> bool:
    """True when the bound session subtree currently has live work.

    Supervisor heartbeat is not enough: an interval loop stays ``running``
    with a live lease while its session is idle between ticks. The board
    treats that idle wait as Completed and only Working when a turn (or
    child work) is actually in flight.
    """
    sid = str(getattr(record, "session_id", "") or "")
    if not sid:
        return False
    from sharedagents_goal.goal_registry import session_work_live_cheap

    live, _src, determinate, _term = session_work_live_cheap(sid)
    return bool(live) if determinate else False


def _to_web_dict(record) -> dict:
    """Projection for the API: to_dict() + session title + live flag."""
    out = record.to_dict()
    out["actor_name"] = auth_store.display_name(out.get("actor_email") or "") if out.get("actor_email") else ""
    out["session_title"] = _loop_session_title(
        str(record.session_id),
        # The gateway can only have stamped a real title once the loop
        # actually fired (turn-1 title lifecycle) — never-run loops skip
        # the gateway round-trip and keep their web placeholder.
        ran_once=bool(
            getattr(record, "run_count", 0)
            or getattr(record, "last_run_at", None)
            or getattr(record, "in_flight", False)
            or getattr(record, "status", "") == "running"
        ),
    )
    out["live"] = _loop_live(record)
    out["in_flight"] = _loop_in_flight(record)
    return out


@bp.route("/loops")
@bp.route("/loops/<loop_id>")
@require_auth
def loops_page(loop_id=None):
    return render_template(
        "loops.html",
        base=BASE_PATH,
        realtime_url=REALTIME_URL,
        initial_loop_id=loop_id or "",
    )


@bp.route("/api/loops", methods=["GET"])
@require_auth
def api_loops_list():
    from sharedagents_loop.loop_registry import LoopFilters

    status = request.args.get("status") or None
    agent = request.args.get("agent") or None
    loops = _loop_registry().list_loops(
        LoopFilters(status=status, agent_name=agent),
    )
    # Archived rows are deleted at boot (2026-09-07 owner contract:
    # loops are removed, never archived) — never surface a straggler here.
    loops = [r for r in loops if getattr(r, "status", "") != "archived"]
    return jsonify({"loops": [_to_web_dict(r) for r in loops]})


@bp.route("/api/loops/<loop_id>", methods=["GET"])
@require_auth
def api_loops_get(loop_id):
    import json as _json

    reg = _loop_registry()
    rec = reg.get(loop_id)
    if rec is None:
        return jsonify({"error": "not_found"}), 404
    out = _to_web_dict(rec)
    events = []
    for ev in reg.list_events(loop_id, limit=50):
        try:
            payload = _json.loads(ev.get("payload") or "{}")
        except (TypeError, ValueError):
            payload = {}
        events.append(
            {
                "event_id": ev.get("event_id"),
                "event_type": ev.get("event_type"),
                "payload": payload,
                "created_at": ev.get("created_at"),
            }
        )
    out["events"] = events
    out["bound_model"] = _loop_session_model(str(rec.session_id))
    return jsonify(out)


@bp.route("/api/loops", methods=["POST"])
@require_auth
def api_loops_create():
    """Create a loop bound to a session.

    Body: ``{session_id, prompt_text, interval_s | interval | time,
    fresh_session_each_run?, budget_usd?}``.
    ``interval_s`` is whole seconds (60 .. 28d); ``interval`` accepts the
    same text forms as ``/loop`` (e.g. ``30m``, ``every 2h``); ``time`` is a
    daily wall-clock time (24-hour ``HH:MM``, e.g. ``09:30``). Exactly one
    schedule form is required. Web-created loops fire their first run at the
    creation instant."""
    from sharedagents_loop.loop_service import (
        IntervalParseError,
        create_loop_for_session,
        parse_fresh_session_each_run,
        parse_interval,
        parse_loop_budget_usd,
        parse_time_of_day,
    )

    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "body must be a JSON object"}), 400
    prompt_text = str(data.get("prompt_text") or "").strip()
    if not prompt_text:
        return jsonify({"error": "prompt_text required"}), 400
    session_id = str(data.get("session_id") or "").strip()
    if not session_id:
        return jsonify({"error": "session_id required"}), 400
    try:
        budget_usd = parse_loop_budget_usd(data.get("budget_usd"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    fresh_session_each_run = parse_fresh_session_each_run(
        data.get("fresh_session_each_run"),
    )

    # Schedule: exactly one of interval / interval_s / time.
    time_raw = data.get("time")
    interval_raw = data.get("interval_s", data.get("interval"))
    has_time = time_raw is not None and str(time_raw).strip() != ""
    has_interval = interval_raw is not None and str(interval_raw).strip() != ""
    if has_time and has_interval:
        return jsonify(
            {
                "error": "specify either an interval or a specific time, not both",
            }
        ), 400
    cron = None
    if has_time:
        try:
            cron, _human = parse_time_of_day(str(time_raw))
        except IntervalParseError as exc:
            return jsonify({"error": str(exc)}), 400
        interval_s = None
    else:
        if not has_interval:
            return jsonify(
                {"error": "schedule required (interval seconds, text like 30m, or a specific time like 09:30)"}
            ), 400
        if isinstance(interval_raw, bool):
            return jsonify({"error": "interval must be a number or interval text"}), 400
        try:
            if isinstance(interval_raw, (int, float)):
                numeric = float(interval_raw)
                if not math.isfinite(numeric) or not numeric.is_integer():
                    return jsonify({"error": "interval_s must be whole seconds"}), 400
                interval_s = int(numeric)
                if interval_s < 60:
                    return jsonify({"error": "minimum interval is 1 minute"}), 400
                if interval_s > 28 * 86400:
                    return jsonify({"error": "maximum interval is 28 days"}), 400
            else:
                interval_s, _cron, _label, _note = parse_interval(str(interval_raw))
        except (IntervalParseError, OverflowError, TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

    # A loop can inject future prompts into its bound session, so use the
    # same ownership/adoption guard as the session prompt route. Bind the loop
    # to the gateway SID, and derive its agent from the
    # owned session record rather than trusting a client-supplied override.
    sdata = store.get(session_id, g.user_email)
    if sdata is not None and not _gw_session_log_exists(session_id):
        # New-session loops: the web row has no gateway binding yet. Use the
        # canonical lazy-bind (the same endpoint chat uses) so a fresh session
        # can host a loop immediately; failure = gateway unreachable.
        if not ensure_gw_session(sdata):
            return jsonify({"error": "Gateway unreachable"}), 502
    resolved_session_id, resolution_error = _resolve_gw_hash(session_id)
    if resolution_error is not None:
        return resolution_error
    sdata = store.get(session_id, g.user_email) or store.get(
        resolved_session_id,
        g.user_email,
    )
    agent_name = str((sdata or {}).get("agent") or DEFAULT_AGENT)
    session_id = str(resolved_session_id or "")

    try:
        agent = _web_load_agent(agent_name)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"agent unavailable: {exc}"}), 502
    sess = _session_for_loop(agent, session_id, sdata)
    if sess is None:
        return jsonify({"error": "session not found"}), 404

    try:
        result = create_loop_for_session(
            agent,
            sess,
            prompt_text=prompt_text,
            interval_s=interval_s,
            cron=cron,
            created_by=g.user_email,
            actor_email=g.user_email,
            fire_immediately=True,
            fresh_session_each_run=fresh_session_each_run,
            budget_usd=budget_usd,
        )
    except LookupError as exc:
        # The 50-scheduled-job ceiling is a conflict, not a server fault.
        return jsonify({"error": str(exc)}), 409
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500

    if not result.ok:
        return jsonify({"error": result.message}), 409
    if result.loop is not None:
        # Kick supervision on the LIVE gateway and CONFIRM it started, with
        # a bounded retry to close any supervisor-lease race (same as goals).
        for _attempt in range(6):
            if _web_supervise_loop(agent_name, result.loop.loop_id):
                break
            time.sleep(0.5)
    return jsonify(
        {
            "loop": _to_web_dict(result.loop) if result.loop else None,
        }
    ), 201


@bp.route("/api/loops/<loop_id>", methods=["PATCH"])
@require_auth
def api_loops_patch(loop_id):
    """Update per-run options on an existing loop.

    Body: ``{fresh_session_each_run}``. Minting a new session each tick
    is opt-in only — omitted/falsey/"false" turns the flag off.
    """
    from sharedagents_loop.loop_service import parse_fresh_session_each_run

    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "body must be a JSON object"}), 400
    if "fresh_session_each_run" not in data:
        return jsonify({"error": "fresh_session_each_run required"}), 400
    rec = _loop_registry().get(loop_id)
    if rec is None:
        return jsonify({"error": "not_found"}), 404
    updated = _loop_registry().set_fresh_session_each_run(
        loop_id,
        parse_fresh_session_each_run(data.get("fresh_session_each_run")),
    )
    if updated is None:
        return jsonify({"error": "not_found"}), 404
    return jsonify({"loop": _to_web_dict(updated)})


def _api_loop_action(loop_id: str, action: str):
    from sharedagents_loop.loop_service import run_loop_action

    rec = _loop_registry().get(loop_id)
    if rec is None:
        return jsonify({"error": "not_found"}), 404
    # The record is the authority: accepting a body override can route the
    # cancel to the wrong agent while falsely reporting success.
    agent_name = str(rec.agent_name or DEFAULT_AGENT)

    # Gateway-first: the action must run where the supervisor drive task
    # lives (the LIVE gateway agent), or pausing never cancels the task.
    gw = _web_loop_action(agent_name, loop_id, action)
    if gw is not None:
        if not gw.get("ok", False):
            return jsonify({"error": gw.get("message") or f"{action} failed"}), 409
        return jsonify({"loop": gw.get("loop")})
    # Gateway unreachable — fall back to a local registry write so the status
    # still flips; the supervisor folds out at its next checkpoint.
    try:
        agent = _web_load_agent(agent_name)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 502
    result = run_loop_action(agent, loop_id, action)
    if not result.get("ok", False):
        return jsonify(
            {
                "error": result.get("message") or f"{action} failed",
            }
        ), 409
    return jsonify({"loop": result.get("loop")})


@bp.route("/api/loops/<loop_id>/pause", methods=["POST"])
@require_auth
def api_loops_pause(loop_id):
    return _api_loop_action(loop_id, "pause")


@bp.route("/api/loops/<loop_id>/resume", methods=["POST"])
@require_auth
def api_loops_resume(loop_id):
    return _api_loop_action(loop_id, "resume")


def _api_loop_delete(loop_id: str):
    """Terminal loop delete: cancel the live supervisor, then remove the row.

    Best-effort gateway cancel first (the control endpoint's archive verb
    stops the drive task and clears the lease); the record is deleted
    locally regardless, so a gateway outage cannot strand a hidden archived
    row. 2026-09-07 owner contract: loops are deleted, never archived.
    """
    reg = _loop_registry()
    rec = reg.get(loop_id)
    if rec is None:
        return jsonify({"error": "not_found"}), 404
    agent_name = str(rec.agent_name or DEFAULT_AGENT)
    _web_loop_action(agent_name, loop_id, "archive")
    try:
        reg.delete_loop(loop_id)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500
    return jsonify({"ok": True, "deleted": loop_id}), 200


@bp.route("/api/loops/<loop_id>", methods=["DELETE"])
@require_auth
def api_loops_delete(loop_id):
    """Permanently delete a loop (the board's Delete button)."""
    return _api_loop_delete(loop_id)


@bp.route("/api/loops/<loop_id>/archive", methods=["POST"])
@require_auth
def api_loops_archive(loop_id):
    # Legacy verb: DELETE now — the board has no archived state, and old
    # cached clients must never be able to create an invisible archived row.
    return _api_loop_delete(loop_id)


@bp.route("/api/loops/<loop_id>/stop", methods=["POST"])
@require_auth
def api_loops_stop(loop_id):
    # ``stop`` is the terminal DELETE: record + run/event
    # history + wakeups are removed, never archived. The gateway drive task
    # is cancelled first (its archive verb stops the supervisor) so nothing
    # re-fires between the cancel and the row deletion.
    return _api_loop_delete(loop_id)
