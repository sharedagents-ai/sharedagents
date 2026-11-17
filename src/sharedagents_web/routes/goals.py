"""sharedagents_web.routes.goals — Durable goals portfolio page + goal lifecycle/budget/events routes.

Extracted from server.py (Task 2 de-monolith). Handler bodies are
byte-identical to the originals; only the decorator target changed
(``@app.route`` -> ``@bp.route``). Registered by server.create_app().
"""

import os
import threading
import time
from datetime import datetime, timezone

from flask import Blueprint, g, jsonify, render_template, request

from sharedagents_web.server import (
    BASE_PATH,
    DEFAULT_AGENT,
    GW_SESSIONS_DIR,
    REALTIME_URL,
    gateway_request,
    require_auth,
)

bp = Blueprint("goals", __name__)


def _web_load_agent(agent_name: str):
    from sharedagents_agent.agents.factory import load_agent

    return load_agent(agent_name)


def _web_supervise_goal(agent_name: str, goal_id: str) -> bool:
    """Kick outer-loop supervision on the live gateway agent (has asyncio loop)."""
    resp = gateway_request(
        "POST",
        f"/control/v1/{agent_name}/goals/{goal_id}/supervise",
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


def _web_goal_action(agent_name: str, goal_id: str, action: str) -> dict | None:
    """Run a stop-style goal action (``pause`` / ``archive``) on the LIVE
    gateway agent.

    The action must run where the supervisor task lives (the gateway daemon),
    otherwise pausing only writes the DB status and never cancels the running
    re-drive loop — the original "can't stop the goal" bug. Returns the gateway
    JSON on success, or ``None`` if the gateway is unreachable (caller falls
    back to a local registry write)."""
    resp = gateway_request(
        "POST",
        f"/control/v1/{agent_name}/goals/{goal_id}/{action}",
        json={},
        timeout=10,
    )
    if resp is None:
        return None
    try:
        return resp.json()
    except ValueError:
        return {"ok": resp.status_code == 200}


def _web_goal_detail(goal_id: str) -> dict | None:
    from sharedagents_goal.goal_registry import (
        _work_live_cheap_uncached,
        get_goal_registry,
    )

    reg = get_goal_registry()
    detail = reg.get_detail(goal_id)
    if detail is None:
        return None
    goal, events, root_history = detail
    out = goal.to_dict()
    # Detail carries status / terminal_reason / events from ONE read-only
    # registry snapshot.  Re-probe liveness without the display cache so a
    # cached terminal cannot contradict fresh live/unknown work, but do not
    # project that terminal onto this richer response: only the background
    # healer may persist a terminal and its matching reason/event history.
    try:
        _reconcile_goal_views(
            [out],
            str(out.get("agent_name") or "") or None,
            work_probe=_work_live_cheap_uncached,
            project_terminal=False,
        )
    except Exception:  # noqa: BLE001 — reconciliation must never break detail
        pass
    out["events"] = events
    out["root_history"] = root_history
    return out


try:
    from sharedagents_goal.goal_registry import _SUPERVISOR_LEASE_S as _LEASE_S
except Exception:  # noqa: BLE001
    _LEASE_S = 120.0


def _hb_is_fresh(hb: str | None, now) -> bool:
    """True iff a supervisor heartbeat is inside the lease window."""
    if not hb:
        return False
    try:
        ts = datetime.fromisoformat(str(hb))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
    except Exception:  # noqa: BLE001
        return False
    return (now - ts).total_seconds() <= _LEASE_S


@bp.route("/goals")
@bp.route("/goals/<goal_id>")
@require_auth
def goals_page(goal_id=None):
    return render_template(
        "goals.html",
        base=BASE_PATH,
        realtime_url=REALTIME_URL,
        initial_goal_id=goal_id or "",
    )


def _reconcile_goal_views(
    goal_dicts: list[dict],
    agent_name: str | None,
    *,
    work_probe=None,
    project_terminal: bool = True,
) -> None:
    """Make the goal list HONEST so the Goals page and chat widget agree
    on liveness (mutates the dicts in place).

    Adds ``supervised_live`` to every goal — True iff the goal is ``running``
    and either inside its bounded first-driver grace or carrying a supervisor
    heartbeat within the lease window, OR the goal's root session subtree
    still has live work. The subtree probe is the CHEAP sessions_meta read
    (``goal_registry.session_work_live_cheap``): it never constructs or
    refreshes the SessionIndex (``ensure_schema`` migrations/DDL plus two
    COUNT scans on a large sessions.db), never globs job files, and never
    loads an Agent — the /api/goals latency fix. When the probe cannot read
    the index at all the goal is surfaced with ``liveness_unknown``
    (explicitly degraded) rather than confidently Stalled.

    When ``project_terminal`` is enabled, a stale running/blocked goal whose
    root session's projected ``goal_terminal`` says it actually settled is
    shown at that terminal in this response. The WRITE is performed by the
    background heal pass (``_heal_stale_goals_once``, the safety
    net) so the read path never blocks on goals.db writes or Agent construction.
    """
    from datetime import datetime, timezone

    from sharedagents_goal.goal_registry import (
        STATUS_BLOCKED,
        STATUS_MET,
        STATUS_PAUSED,
        STATUS_RUNNING,
        _is_goal_starting,
        session_work_live_cheap,
    )

    if work_probe is None:
        work_probe = session_work_live_cheap
    now = datetime.now(timezone.utc)
    _ensure_heal_thread()

    for goal in goal_dicts:
        work_snapshot = None

        def _work_snapshot():
            nonlocal work_snapshot
            if work_snapshot is None:
                work_snapshot = work_probe(
                    str(goal.get("root_session_id") or ""),
                )
            return work_snapshot

        st = goal.get("status")
        heartbeat = goal.get("supervisor_heartbeat_at")
        starting = not heartbeat and _is_goal_starting(
            str(st or ""),
            goal.get("started_at") or goal.get("created_at"),
            int(goal.get("turn_count") or 0),
        )
        supervised = st == STATUS_RUNNING and (starting or _hb_is_fresh(heartbeat, now))
        if supervised:
            goal["supervised_live"] = True
            goal["liveness_source"] = "starting" if starting else "heartbeat"
        elif st == STATUS_RUNNING and goal.get("root_session_id"):
            # Supervisor heartbeat absent/stale AND start grace expired — the
            # drive task is gone. Before labelling the goal a zombie, ask the
            # session index whether its root subtree still has live work.
            # CHEAP probe only (sessions_meta, no SessionIndex construction/
            # refresh, no job glob, no Agent load). A determinate "not live"
            # is a real zombie; an indeterminate probe is marked unknown.
            work_live, src, determinate, _terminal = _work_snapshot()
            if work_live:
                supervised = True
                goal["supervised_live"] = True
                goal["liveness_source"] = src
            else:
                goal["supervised_live"] = False
                if not determinate:
                    goal["liveness_unknown"] = True
        else:
            goal["supervised_live"] = False
        if (
            project_terminal
            and st in (STATUS_RUNNING, STATUS_BLOCKED)
            and (not supervised or starting)
            and goal.get("root_session_id")
            and goal.get("status") in (STATUS_RUNNING, STATUS_BLOCKED)
        ):
            # Display-only terminal projection: the root session's projected
            # goal_terminal (folded in-line by the gateway SessionWriter) is
            # the cheap source for "this goal actually settled" — no Agent
            # load, no session JSONL read. The background heal pass persists
            # the write; this response shows the terminal immediately. A
            # fresh start ignores only ``paused`` (previous cycle's value).
            _wl, _src, _det, terminal = _work_snapshot()
            if terminal in (STATUS_MET, STATUS_BLOCKED, STATUS_PAUSED):
                if starting and terminal == STATUS_PAUSED:
                    continue
                goal["status"] = terminal
                goal["supervised_live"] = False


# --- background terminal healing (off the request path) -----------------------
# The zombie-goal safety net is retained, but the read path must
# not write goals.db: a heal can stall on the supervisor's write lock
# (busy_timeout=15000) and the Agent/session reads it used to need were the
# dominant /api/goals latency. A daemon thread performs the heal instead,
# reading the CHEAP projected ``goal_terminal`` from sessions_meta (the
# gateway folds the session header in-line — no Agent construction, no
# session JSONL read) and writing the terminal status in the background; the
# request path reads the healed status straight from goals.db on the next
# poll. Gated off under SHAREDAGENTS_WEB_DEBUG=1 (the web test suite) so a test
# process never mutates a real home.

_HEAL_INTERVAL_S = 5.0
_heal_lock = threading.Lock()
_heal_thread: threading.Thread | None = None


def _heal_stale_goals_once() -> None:
    """One background pass over running/blocked goals whose root session's
    projected ``goal_terminal`` says they actually settled. Mirrors the old
    request-path reconciler's heal (same ``*_reconciled`` reasons), but reads
    the CHEAP sessions_meta projection instead of loading an Agent, and never
    runs on a request.

    Race-free: for every candidate the pass takes a FRESH uncached subtree
    probe (``_work_live_cheap_uncached``) immediately before mutating — a
    live descendant or a now-indeterminate index skips the goal — and the
    registry transition is an atomic compare-and-set
    (``apply_terminal_if_unchanged``) on the goal's snapshot evidence
    (status + root_session_id + exact supervisor heartbeat), so a resume,
    rebind, or renewed lease between snapshot and apply is a no-op. Never
    raises."""
    from datetime import datetime, timezone

    from sharedagents_goal.goal_registry import (
        STATUS_BLOCKED,
        STATUS_MET,
        STATUS_PAUSED,
        STATUS_RUNNING,
        GoalFilters,
        _is_goal_starting,
        _work_live_cheap_uncached,
        get_goal_registry,
    )

    try:
        reg = get_goal_registry()
        goals = reg.list_goals(GoalFilters())
    except Exception:  # noqa: BLE001 — a background pass must never raise
        return
    candidates = [goal for goal in goals if goal.status in (STATUS_RUNNING, STATUS_BLOCKED) and goal.root_session_id]
    if not candidates:
        return
    now = datetime.now(timezone.utc)
    for goal in candidates:
        hb = getattr(goal, "supervisor_heartbeat_at", None)
        starting = (
            not hb
            and goal.status == STATUS_RUNNING
            and _is_goal_starting(
                STATUS_RUNNING,
                getattr(goal, "started_at", None) or getattr(goal, "created_at", None),
                int(getattr(goal, "turn_count", 0) or 0),
            )
        )
        supervised = goal.status == STATUS_RUNNING and (starting or _hb_is_fresh(hb, now))
        if supervised and not starting:
            continue  # a live supervisor owns the terminal transition
        # FRESH cheap probe immediately before the mutation: the subtree must
        # be determinately not-live (a live descendant means the goal is still
        # genuinely working) and the root's projected terminal must say the
        # cycle settled. An unreadable index or an absent root row is
        # indeterminate — never heal on evidence we cannot confirm.
        try:
            work_live, _src, determinate, terminal = _work_live_cheap_uncached(str(goal.root_session_id))
        except Exception:  # noqa: BLE001
            continue
        if work_live or not determinate:
            continue
        if terminal not in (STATUS_MET, STATUS_BLOCKED, STATUS_PAUSED):
            continue
        if starting and terminal == STATUS_PAUSED:
            # Registry creation/rebind is authoritative during start grace;
            # paused can be the previous goal cycle's derived header.
            continue
        try:
            if terminal == STATUS_MET:
                reason, summary = "goal_met_reconciled", None
            elif terminal == STATUS_BLOCKED:
                reason, summary = "goal_blocked_reconciled", None
            else:
                reason, summary = "goal_paused_reconciled", None
            reg.apply_terminal_if_unchanged(
                goal.goal_id,
                expected_status=goal.status,
                expected_root_session_id=str(goal.root_session_id),
                expected_supervisor_heartbeat_at=hb,
                terminal=terminal,
                reason=reason,
                summary=summary,
            )
        except Exception:  # noqa: BLE001
            continue


def _heal_thread_loop() -> None:
    while True:
        try:
            _heal_stale_goals_once()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(_HEAL_INTERVAL_S)


def _ensure_heal_thread() -> None:
    """Start the background goal healer once per process (lazily, on the
    first goals request). Daemon: dies with the worker. Never starts under
    SHAREDAGENTS_WEB_DEBUG=1 (the web test suite) so a test process cannot
    mutate a real home's goals.db."""
    if os.environ.get("SHAREDAGENTS_WEB_DEBUG") == "1":
        return
    global _heal_thread
    with _heal_lock:
        if _heal_thread is None or not _heal_thread.is_alive():
            _heal_thread = threading.Thread(target=_heal_thread_loop, name="goals-heal", daemon=True)
            _heal_thread.start()


@bp.route("/api/goals", methods=["GET"])
@require_auth
def api_goals_list():
    from sharedagents_goal.goal_registry import GoalFilters, get_goal_registry

    status = request.args.get("status") or None
    agent = request.args.get("agent") or None
    goals = get_goal_registry().list_goals(
        GoalFilters(status=status, agent_name=agent),
    )
    dicts = [g.to_dict() for g in goals]
    try:
        _reconcile_goal_views(dicts, agent)
    except Exception:  # noqa: BLE001 — reconciliation must never break the list
        pass
    return jsonify({"goals": dicts})


@bp.route("/api/goals/<goal_id>", methods=["GET"])
@require_auth
def api_goals_get(goal_id):
    detail = _web_goal_detail(goal_id)
    if detail is None:
        return jsonify({"error": "not_found"}), 404
    return jsonify(detail)


@bp.route("/api/budget/<session_hash>", methods=["GET"])
@require_auth
def api_budget(session_hash):
    """Hierarchy-aware cost/budget view for a chat: downstream (this sub-agent's
    own spend) + total (root-tree spend vs the root goal budget). Empty dict when
    the session is unknown, so the widget simply hides.
    """
    from sharedagents_goal.goal_service import hierarchical_budget_view

    # The cost ledger is keyed by the GATEWAY session hash. A web session id is a
    return jsonify(hierarchical_budget_view(session_hash))


@bp.route("/api/goals", methods=["POST"])
@require_auth
def api_goals_create():
    from sharedagents_goal.goal_service import create_goal_for_session

    data = request.get_json() or {}
    condition = str(data.get("condition_text") or "").strip()
    if not condition:
        return jsonify({"error": "condition_text required"}), 400
    agent_name = str(data.get("agent_name") or DEFAULT_AGENT)
    force = bool(data.get("force"))
    max_turns = data.get("max_turns")
    max_cost_usd = data.get("max_cost_usd")
    if max_cost_usd not in (None, ""):
        import math

        try:
            max_cost_usd = float(max_cost_usd)
        except (TypeError, ValueError):
            return jsonify({"error": "max_cost_usd must be a number"}), 400
        if not math.isfinite(max_cost_usd) or max_cost_usd < 0:
            return jsonify({"error": "max_cost_usd must be a finite number >= 0"}), 400
    else:
        max_cost_usd = None
    scope_key = data.get("scope_key")
    session_id = str(data.get("session_id") or "").strip()

    try:
        agent = _web_load_agent(agent_name)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"agent unavailable: {exc}"}), 502

    if session_id:
        sess = agent.get_session(session_id)
        if sess is None:
            return jsonify({"error": "session not found"}), 404
    else:
        resp = gateway_request("POST", f"/control/v1/{agent_name}/sessions", json={})
        if resp is None or resp.status_code not in (200, 201):
            return jsonify({"error": "failed to create session"}), 502
        body = resp.json() if resp is not None else {}
        session_id = body.get("sessionId") or body.get("session_id") or body.get("id") or ""
        session_id = str(session_id).strip()
        if not session_id:
            return jsonify({"error": "failed to create session"}), 502
        sess = agent.get_session(session_id)
        if sess is None:
            return jsonify({"error": "session create failed"}), 502

    try:
        result = create_goal_for_session(
            agent,
            sess,
            condition_text=condition,
            max_turns=int(max_turns) if max_turns is not None else None,
            max_cost_usd=max_cost_usd,
            scope_key=scope_key,
            force=force,
            created_by=g.user_email,
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500

    if not result.ok:
        code = 409 if result.conflicts else 400
        return jsonify(
            {
                "error": result.message,
                "conflicts": result.conflicts,
            }
        ), code
    if result.goal is not None:
        # Kick outer-loop supervision on the LIVE gateway and CONFIRM it started
        # (returns True only when the drive loop is active). A bounded retry
        # closes any lingering supervisor-lease race so the first turn begins
        # within a few seconds instead of stalling on the long claim-retry
        # backoff. The gateway supervise call is idempotent.
        for _attempt in range(6):
            if _web_supervise_goal(agent_name, result.goal.goal_id):
                break
            time.sleep(0.5)
    return jsonify({"goal": result.goal.to_dict() if result.goal else None}), 201


def _api_goal_action(goal_id: str, action: str):
    from sharedagents_goal.goal_service import (
        archive_goal,
        pause_goal,
        restart_goal,
        resume_goal,
    )

    data = request.get_json() or {}
    agent_name = str(data.get("agent_name") or DEFAULT_AGENT)
    force = bool(data.get("force"))
    session_id = str(data.get("session_id") or "").strip()
    try:
        agent = _web_load_agent(agent_name)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 502

    from sharedagents_goal.goal_registry import get_goal_registry

    goal = get_goal_registry().get(goal_id)
    if goal is None:
        return jsonify({"error": "not_found"}), 404

    if action in ("resume", "restart"):
        # Try gateway first (LIVE agent) so the supervisor's re-drive task
        # is actually started — a local resume/restart here mutates only an
        # ephemeral web-process agent (no running task). Falls back to local
        # if the gateway is unreachable or doesn't support the action yet.
        gw = _web_goal_action(agent_name, goal_id, action)
        if gw is not None:
            if not gw.get("ok", False):
                return jsonify({"error": gw.get("message") or f"{action} failed"}), 409
            return jsonify({"goal": gw.get("goal")})
        # Gateway unreachable — fall back to local agent
        if not session_id:
            session_id = goal.root_session_id or ""
        if not session_id:
            resp = gateway_request("POST", f"/control/v1/{agent_name}/sessions", json={})
            if resp is None or resp.status_code not in (200, 201):
                return jsonify({"error": "failed to create session"}), 502
            body = resp.json() if resp is not None else {}
            session_id = str(body.get("sessionId") or body.get("session_id") or body.get("id") or "").strip()
            if not session_id:
                return jsonify({"error": "failed to create session"}), 502
        sess = agent.get_session(session_id)
        if sess is None:
            return jsonify({"error": "session not found"}), 404
        fn = resume_goal if action == "resume" else restart_goal
        result = fn(agent, goal_id, sess, force=force)
    elif action in ("pause", "archive", "delete"):
        # Run on the LIVE gateway agent so the supervisor's re-drive task is
        # actually cancelled — a local pause_goal here mutates only an
        # ephemeral web-process agent (no running task), which is exactly the
        # "can't stop the goal" bug.
        gw = _web_goal_action(agent_name, goal_id, action)
        if gw is not None:
            if not gw.get("ok", False):
                return jsonify({"error": gw.get("message") or f"{action} failed"}), 409
            return jsonify({"goal": gw.get("goal")})
        # Gateway unreachable — fall back to a local registry write so the
        # status still flips (the supervisor folds out at its next checkpoint).
        result = pause_goal(agent, goal_id) if action == "pause" else archive_goal(agent, goal_id)
    else:
        return jsonify({"error": "unknown action"}), 400

    if not result.ok:
        return jsonify(
            {
                "error": result.message,
                "conflicts": result.conflicts,
            }
        ), 409
    if result.goal is not None and result.goal.status == "running":
        _web_supervise_goal(agent_name, goal_id)
    return jsonify({"goal": result.goal.to_dict() if result.goal else None})


@bp.route("/api/goals/<goal_id>/pause", methods=["POST"])
@require_auth
def api_goals_pause(goal_id):
    return _api_goal_action(goal_id, "pause")


@bp.route("/api/goals/<goal_id>/resume", methods=["POST"])
@require_auth
def api_goals_resume(goal_id):
    return _api_goal_action(goal_id, "resume")


@bp.route("/api/goals/<goal_id>/restart", methods=["POST"])
@require_auth
def api_goals_restart(goal_id):
    return _api_goal_action(goal_id, "restart")


@bp.route("/api/goals/<goal_id>/archive", methods=["POST"])
@require_auth
def api_goals_archive(goal_id):
    return _api_goal_action(goal_id, "archive")


@bp.route("/api/goals/<goal_id>/delete", methods=["POST"])
@require_auth
def api_goals_delete(goal_id):
    return _api_goal_action(goal_id, "delete")


@bp.route("/api/goals/<goal_id>/set-budget", methods=["POST"])
@require_auth
def api_goals_set_budget(goal_id):
    """Set a goal's USD cost budget. Body: {"max_cost_usd": float, "agent_name"?}.
    ``max_cost_usd <= 0`` resets to the default $25. Does not change the
    session-tree cap — that is ``/budget`` / the chat Budget editor."""
    import math

    from sharedagents_goal.goal_service import set_goal_budget

    data = request.get_json() or {}
    agent_name = str(data.get("agent_name") or DEFAULT_AGENT)
    try:
        usd = float(data.get("max_cost_usd"))
    except (TypeError, ValueError):
        return jsonify({"error": "max_cost_usd must be a number"}), 400
    if not math.isfinite(usd) or usd < 0:
        return jsonify({"error": "max_cost_usd must be a finite number >= 0"}), 400
    try:
        agent = _web_load_agent(agent_name)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 502
    result = set_goal_budget(agent, goal_id, usd)
    if not result.ok:
        return jsonify({"error": result.message}), 400
    return jsonify({"goal": result.goal.to_dict() if result.goal else None})


@bp.route("/api/goals/<goal_id>/events", methods=["GET"])
@require_auth
def api_goals_events(goal_id):
    from sharedagents_goal.goal_registry import get_goal_registry

    reg = get_goal_registry()
    if reg.get(goal_id) is None:
        return jsonify({"error": "not_found"}), 404
    limit = request.args.get("limit", type=int) or 100
    return jsonify({"events": reg.list_events(goal_id, limit=limit)})


@bp.route("/api/goals/<goal_id>/session-tail", methods=["GET"])
@require_auth
def api_goals_session_tail(goal_id):
    from sharedagents_goal.goal_registry import get_goal_registry

    goal = get_goal_registry().get(goal_id)
    if goal is None or not goal.root_session_id:
        return jsonify({"error": "not_found"}), 404
    limit = request.args.get("limit", type=int) or 200
    path = GW_SESSIONS_DIR / f"{goal.root_session_id}.jsonl"
    if not path.exists():
        return jsonify({"lines": []})
    lines: list[str] = []
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                lines.append(line.rstrip("\n"))
        lines = lines[-limit:]
    except OSError:
        lines = []
    return jsonify({"session_id": goal.root_session_id, "lines": lines})
