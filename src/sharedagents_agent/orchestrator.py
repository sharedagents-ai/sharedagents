"""Cross-agent live view backed by the unified sessions.db.

A single SQL predicate against the shared
``layout.sessions_db_path()`` answers "what is every agent doing right
now?" — the orchestrator's reason for existing. Built on top of the
``sessions_meta.session_state`` column (see
:class:`sharedagents_session.session_index.EventProjector`); no per-agent
walking required.

Public surface:

* :func:`list_live_sessions` — rows currently in a live state across
  every agent.
* :func:`all_sessions` — every session row regardless of state.
* :func:`render_table` — pretty-print helper used by the CLI
  (``sharedagents orchestrator status``).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any

from sharedagents_core.layout import sessions_root
from sharedagents_session.session_activity import (
    ABORT_LIVE_TURN_STATES,
    TREE_BUSY_STATES,
    derive_activity,
    legacy_phase,
    row_stale_for_display,
)


def _json_dict(value: Any) -> dict[str, Any]:
    try:
        data = json.loads(str(value or "[]"))
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


from sharedagents_session.session_index import SessionIndex  # noqa: E402

# A session in a "live" state but whose state register hasn't been touched
# within this many seconds is treated as STALE, not active. The projector
# re-stamps ``session_state_updated_at`` on essentially every streamed delta
# and tool event, so a genuinely running turn refreshes within seconds; a
# session left mid-``streaming``/``tool_running`` by a crash or a gateway
# restart (no terminal event ever folded) otherwise lingers as a false
# "active" forever. 5 min is generous enough not to flag a long-running
# single tool call while still catching abandoned sessions.
LIVE_FRESHNESS_SECONDS = 300

# Default row cap for the web agent monitor — one indexed sqlite scan, newest
# sessions first. Summary aggregates remain unbounded; only the tree payload is
# capped. Override via ``session_tree(max_rows=...)`` or ``?limit=`` on the API.
DEFAULT_TREE_MAX_ROWS = 200


# States that mean "the agent is busy and has not returned control to
# the operator" — what the orchestrator most often wants to know about.
LIVE_STATES: tuple[str, ...] = TREE_BUSY_STATES

# ``delete_promptless_sessions`` leaves a header-only log alone until it is
# this old: a just-created session may still be waiting for its first prompt.
PROMPTLESS_MIN_AGE_SECONDS = 600

# INCIDENT stubs are synthetic queue items, not conversations. Stubs already
# projected into ``sessions.db`` (and any incident-shaped file that lands in
# ``sessions/``) must NOT surface on the Agent Monitor. This clause drops them by their stable
# ``incident-`` hash prefix / ``*-incident`` agent name. The self-improver's
# real working sessions (``agent_name='self-improver'``, ``selfimprover-*`` hash)
# carry NEITHER marker, so they stay visible. Trailing space: it's concatenated
# directly before ``ORDER BY`` / ``GROUP BY``.
_INCIDENT_EXCLUSION = "WHERE session_hash NOT LIKE 'incident-%'   AND agent_name NOT LIKE '%-incident' "


def _index() -> SessionIndex:
    return SessionIndex(sessions_root())


def _refresh_quiet(idx: SessionIndex) -> None:
    """Bring ``sessions.db`` current with any JSONL writes the bus
    subscriber hasn't folded yet.

    Both ``list_live_sessions`` and ``all_sessions`` are read by the
    orchestrator before answering "what is every agent doing right
    now?" — without this refresh, a writer process whose subscriber
    is absent / lagged / not yet wired (subprocess tool runner, a
    fresh ad-hoc CLI invocation) would silently disappear from the
    cross-agent view. Cheap
    when every cursor is already at EOF; never raises into the
    caller.
    """
    try:
        idx.refresh()
    except Exception:  # noqa: BLE001
        pass


# --- monitor hot-path refresh throttle ---------------------------------
# The agent monitor (web SSE stream + TUI /monitor) re-fetches the tree on a
# tight cadence (~1.5s) from potentially several browsers/tabs at once. A
# brand-new session or — crucially — a SUBAGENT whose JSONL was written by a
# different process (subprocess tool runner, external CLI agent) is invisible
# until ``sessions.db`` is folded forward, because the in-process bus
# subscriber only ever sees ``session_saved`` for sessions born in THIS
# process. Reading the index without a refresh therefore lags such sessions by
# an unbounded amount (until some unrelated read happens to refresh). We fold
# on the monitor read path so they appear within one poll — but throttle the
# stat-scan to at most once per ``_TREE_REFRESH_MIN_INTERVAL`` GLOBALLY so N
# concurrent pollers don't each re-stat every session file every tick. A full
# refresh over ~200 sessions is ~45ms (stat-only when nothing grew), so the
# throttle is a courtesy, not a necessity; it keeps the cost flat as tabs/poll
# rate scale. Worst-case appearance latency ≈ throttle window + poll interval
# ≈ 1.0s + 1.5s, comfortably under the monitor's freshness budget.
_TREE_REFRESH_MIN_INTERVAL = 1.0
_tree_refresh_lock = threading.Lock()
_tree_refresh_at = 0.0


def _maybe_refresh_for_tree(idx: SessionIndex) -> None:
    """Throttled ``_refresh_quiet`` for the monitor hot path.

    Folds lagged/cross-process JSONL into ``sessions.db`` so new sessions and
    subagents surface promptly, but runs the stat-scan at most once per
    :data:`_TREE_REFRESH_MIN_INTERVAL` across all concurrent callers. Never
    raises into the caller.
    """
    global _tree_refresh_at
    now = time.monotonic()
    with _tree_refresh_lock:
        if now - _tree_refresh_at < _TREE_REFRESH_MIN_INTERVAL:
            return
        _tree_refresh_at = now
    _refresh_quiet(idx)


def reset_stale_sessions() -> int:
    """Sweep abandoned sessions out of a live state on boot.

    Used at gateway/orchestrator startup. Turns stamped by another live
    process sharing the sessions directory are preserved. Each remaining
    in-flight session (``streaming``, ``tool_running``, etc.) FIRST gets a
    canonical ``turn_ended(aborted)`` appended to its JSONL (so a stream
    consumer observes the turn closing — the §13 crash-isolation contract),
    which folds it to a terminal non-live state (``error``); sessions with no
    in-flight turn are flipped straight to ``idle``. Either way they leave the
    live set. Without this reset the
    freshness gate (``LIVE_FRESHNESS_SECONDS``) would keep them
    visible as "active" for up to 5 minutes, and the projector would
    re-stamp ``session_state_updated_at`` on re-tail, effectively
    resetting the clock and keeping them "active" forever.

    Returns the number of sessions reset.

    Streaming events (``text_delta``, ``tool_call``, …) are still NOT
    projected on the realtime write path (per-token SQL was the perf
    problem — see :meth:`SessionWriter._emit`); those fold lazily via the
    debounced bus subscriber / read-triggered ``tail``. Boundary events
    that change ``session_state`` (``turn_started``, ``turn_ended``, …)
    ARE projected inline so monitors see them immediately. At boot
    ``indexed_bytes`` can still lag for streaming events, leaving a
    just-crashed session's latest live state UNfolded — a blind
    ``UPDATE`` would miss it, and the first post-boot read would then
    fold it and re-stamp it freshly ``streaming``/``tool_running``. We
    therefore ``refresh`` first to materialise every session's true
    latest state, then sweep all live states to ``idle`` in one pass.


    Unstamped turns retain the legacy close-on-boot behavior. Waiting
    parents have no open turn and are reset to idle by the bulk UPDATE.
    """
    idx = _index()
    # Fold any JSONL the lazy projector hasn't caught up on, so a
    # crashed session's latest live event is materialised before the
    # sweep (otherwise the UPDATE misses it and a later read re-stamps
    # it fresh). Cheap when every cursor is already at EOF; never raises.
    _refresh_quiet(idx)
    con = idx._connect()
    try:
        placeholders = ",".join("?" for _ in LIVE_STATES)

        # Surface the abort in the SOURCE-OF-TRUTH log FIRST: append a canonical
        # ``turn_ended(aborted)`` to each in-flight session's JSONL so a stream
        # consumer (web / TUI / Telegram subscribed to ``mode=events``) observes
        # the turn CLOSING rather than the stream hanging forever. This is the
        # §13 crash-isolation contract: an orchestrator restart kills its
        # in-flight turns, and on recovery it closes them cleanly in the log.
        try:
            live_rows = con.execute(
                f"SELECT session_hash, live_turn_idx, last_turn_idx, engine, "
                f"       session_state "
                f"FROM sessions_meta "
                f"WHERE session_state IN ({placeholders})",
                tuple(LIVE_STATES),
            ).fetchall()
        except Exception:  # noqa: BLE001
            live_rows = []
        # Sessions whose open turn was written by ANOTHER process that is
        # still running (a second orchestrator sharing this sessions dir, a
        # detached delegate runner). That process owns the turn and closes
        # it itself; aborting it here showed a running turn as ended by an
        # ``orchestrator_restart`` that never happened. Unstamped turns keep
        # the old close-on-boot behavior.
        from sharedagents_session.turn_runner import turn_runner_alive_elsewhere

        owned_elsewhere: set[str] = set()
        for r in live_rows:
            d = dict(r)
            h = d.get("session_hash")
            ti = d.get("live_turn_idx")
            if ti is None:
                ti = d.get("last_turn_idx")
            if not h or ti is None or d.get("session_state") == "waiting":
                continue
            try:
                if turn_runner_alive_elsewhere(
                    sessions_root() / f"{h}.jsonl",
                    int(ti),
                ):
                    owned_elsewhere.add(str(h))
            except Exception:  # noqa: BLE001 — unknown owner: old behavior
                pass
        live_rows = [r for r in live_rows if str(dict(r).get("session_hash") or "") not in owned_elsewhere]
        n_live = len(live_rows)
        if live_rows:
            from sharedagents_session.session_writer import (
                close_session_writer,
                get_session_writer,
            )

            sd = sessions_root()
            for r in live_rows:
                d = dict(r)
                h = d.get("session_hash")
                # ``waiting`` means the parent's own turn already ended cleanly
                # and it is only holding for children/jobs. There is no live
                # turn to abort-close — the bare UPDATE to idle below is enough
                # (children are swept on the same boot path).
                if d.get("session_state") == "waiting":
                    continue
                # ``live_turn_idx`` is set on ``turn_started`` and cleared on
                # ``turn_ended``. But a session can be left in a LIVE state with
                # ``live_turn_idx IS NULL``: an aborted SUBAGENT whose parent
                # turn was killed mid-stream has a trailing ``text_delta`` (→
                # state ``streaming``) but no ``turn_ended`` to re-arm the live
                # idx, AND a prior turn may have already nulled it. Falling
                # through on NULL (the old ``continue``) left these orphans
                # dangling forever — the exact "subagent shows active while
                # nothing runs" monitor bug. Fall back to ``last_turn_idx`` (the
                # in-flight turn that never closed) so we can still emit the
                # canonical ``turn_ended(aborted)`` and fold them terminal.
                ti = d.get("live_turn_idx")
                if ti is None:
                    ti = d.get("last_turn_idx")
                if not h or ti is None:
                    continue  # genuinely no turn on record to close
                try:
                    w = get_session_writer(sd, str(h), engine=d.get("engine") or "")
                    # A clean abort (no ``error``) so the session folds to
                    # ``idle`` like a normal cancel — NOT an error state — while
                    # still emitting the canonical turn_ended(aborted) the
                    # stream consumers need. The metrics note the cause.
                    w.turn_ended(
                        turn_idx=int(ti),
                        aborted=True,
                        metrics={"abort_reason": "orchestrator_restart"},
                    )
                    close_session_writer(sd, str(h))
                except Exception:  # noqa: BLE001 — never block boot on one log
                    pass

        # Flip live rows to idle WITHOUT bumping ``session_state_updated_at`` —
        # a restart-time stamp would make old abandoned sessions look freshly
        # active in the web monitor (which sorts/displays by activity age).
        # In-flight sessions already received ``turn_ended(aborted)`` above,
        # which folds them to ``error`` with an accurate projector timestamp.
        keep = sorted(owned_elsewhere)
        keep_clause = f" AND session_hash NOT IN ({','.join('?' for _ in keep)})" if keep else ""
        cur = con.execute(
            f"UPDATE sessions_meta SET session_state='idle', "
            f"live_turn_idx=NULL, live_round_idx=NULL, "
            f"session_phase='idle', session_attention='none', "
            f"session_heartbeat_at=CASE WHEN session_heartbeat_at=0 "
            f"THEN session_state_updated_at ELSE session_heartbeat_at END, "
            f"session_state_version=session_state_version+1 "
            f"WHERE session_state IN ({placeholders}){keep_clause}",
            (*LIVE_STATES, *keep),
        )
        con.commit()
        # Re-fold so the just-written ``turn_ended`` events are reflected in
        # ``sessions_meta`` (and not re-stamped live on the next read).
        _refresh_quiet(idx)
        # Immediate projection of ``turn_ended`` may already have
        # moved abort-closed rows out of LIVE_STATES before this UPDATE,
        # so ``rowcount`` alone under-counts. Prefer the pre-sweep live set.
        return max(int(cur.rowcount or 0), n_live)
    except Exception:  # noqa: BLE001
        con.rollback()
        return 0
    finally:
        con.close()


def release_stale_waiting_sessions(
    freshness_seconds: int = LIVE_FRESHNESS_SECONDS,
) -> int:
    """Flip ``waiting`` parents whose whole child tree is dead/stale to ``idle``.

    Runtime counterpart of the boot sweep (:func:`reset_stale_sessions`).
    A parent whose own turn ended cleanly is left ``waiting`` while any
    child is busy (the projector's ``_maybe_clear_parent_waiting`` folds it
    back to ``idle`` on the last child's terminal event). But when an abort
    or crash leaves a child ghost-live — state stuck at a live value, no
    terminal event ever folded (the NULL-``live_turn_idx`` gap this fix
    closes) — the child never emits again, nothing re-folds the parent, and
    the parent stays ``waiting`` forever. The monitor's READ-side freshness
    gate (:meth:`session_tree`) hides the ghosts after
    ``LIVE_FRESHNESS_SECONDS`` but never REPAIRS them — the 2026-08-11
    incident (session ``<session-id>`` stuck
    ``waiting`` 20+ min on 2 ghost ``streaming``/``prompting`` explore
    children).

    This resolver applies the same window on the WRITE side: a ``waiting``
    parent is released once every descendant is either terminal or stale-live
    (state stamp older than ``freshness_seconds``), no async job is open, and
    no in-process broker lease is held — i.e. no live process/lease/heartbeat
    remains. A genuinely-live child re-stamps its own row (and the parent's,
    via the projector's ``_touch_parent_waiting``) on every event, so a real
    wait is never released.

    Never raises; returns the number of sessions released to ``idle``.
    """
    idx = _index()
    _refresh_quiet(idx)
    con = idx._connect()
    released = 0
    try:
        now_ms = int(time.time() * 1000)
        cutoff_ms = now_ms - max(0, int(freshness_seconds)) * 1000
        try:
            rows = con.execute("SELECT session_hash FROM sessions_meta WHERE session_state='waiting'").fetchall()
        except Exception:  # noqa: BLE001
            return 0
        if not rows:
            return 0
        from sharedagents_session.session_index import EventProjector

        ep = EventProjector()
        for (sid,) in rows:
            # Keep waiting on any lookup failure (fail-safe: a transient
            # lock/error must never release a genuinely waiting parent).
            try:
                if ep._parent_still_has_work(con, sid, freshness_ms=cutoff_ms):
                    continue
            except Exception:  # noqa: BLE001
                continue
            # Same in-process lease signal the memory-flush gate uses: a live
            # delegated turn books an AGENT lease under the parent's key.
            try:
                from sharedagents_agent import sandbox_broker

                if int(sandbox_broker.get_broker().session_live(str(sid)) or 0) > 0:
                    continue
            except Exception:  # noqa: BLE001 — unknown lease state → keep waiting
                continue
            con.execute(
                "UPDATE sessions_meta SET session_state='idle', "
                "session_phase='idle', session_attention='none', "
                "session_state_updated_at=? WHERE session_hash=? "
                "AND session_state='waiting'",
                (now_ms, sid),
            )
            released += 1
        con.commit()
        return released
    except Exception:  # noqa: BLE001
        con.rollback()
        return 0
    finally:
        con.close()


def finalize_stale_live_sessions(
    freshness_seconds: int = LIVE_FRESHNESS_SECONDS,
) -> int:
    """Fold stale live rows whose runner is provably gone (ghost aborts).

    Runtime counterpart of the boot sweep (:func:`reset_stale_sessions`)
    for NON-waiting live rows, and the write-side twin of the monitor's
    read-side freshness gate (:meth:`session_tree`). An aborted or crashed
    subagent whose abort path never folded its row (e.g. the CLI-engine
    session-cancel path) stays ``streaming``/``tool_running`` forever — the
    read gate only HIDES it after ``LIVE_FRESHNESS_SECONDS``, it never
    REPAIRS it. This resolver applies the same window on the write side and
    folds the row terminal, but ONLY when the runner is provably dead:

    * no OPEN async job references the session as its ``child_sid`` (a
      detached delegate runner may be legitimately mid-flight), and
    * no live engine subprocess is registered in the durable pidfile
      (identity-verified — a recycled pid is never treated as live).

    A genuinely running turn keeps re-stamping its row AND holds either an
    open job or a live engine pid, so it is never folded. Never raises;
    returns the number of sessions finalized.
    """
    idx = _index()
    _refresh_quiet(idx)
    con = idx._connect()
    finalized = 0
    try:
        now_ms = int(time.time() * 1000)
        cutoff_ms = now_ms - max(0, int(freshness_seconds)) * 1000
        live_turn_states = tuple(sorted(ABORT_LIVE_TURN_STATES))
        placeholders = ",".join("?" for _ in live_turn_states)
        try:
            rows = con.execute(
                f"SELECT session_hash, parent_session_id, session_state, "
                f"       live_turn_idx, last_turn_idx, engine "
                f"FROM sessions_meta "
                f"WHERE session_state IN ({placeholders}) "
                f"  AND session_state_updated_at < ?",
                (*live_turn_states, cutoff_ms),
            ).fetchall()
        except Exception:  # noqa: BLE001
            return 0
        if not rows:
            return 0
        sd = sessions_root()
        # Durable open jobs drive detached delegate runners; a session with
        # an open job may still be mid-flight even when its row is stale
        # (e.g. a runner blocked on a slow external CLI that streams no
        # deltas). Never fold those.
        try:
            from sharedagents_agent.agents import async_jobs

            open_child_sids = {str((j or {}).get("child_sid") or "") for j in async_jobs.open_jobs()}
        except Exception:  # noqa: BLE001
            open_child_sids = set()
        from sharedagents_agent import sandbox_broker
        from sharedagents_agent.agents import engine_pidfile
        from sharedagents_agent.tools._log import log_event
        from sharedagents_session.ask_pending import session_pending_interaction
        from sharedagents_session.session_index import (
            mark_session_terminal,
        )
        from sharedagents_session.session_writer import (
            close_session_writer,
            get_session_writer,
        )

        for r in rows:
            sid = str(r["session_hash"] or "")
            if not sid or sid in open_child_sids:
                continue
            # A local_llm turn (including its in-process compaction summary)
            # owns a BOOKKEEPING turn-root lease rather than an engine pidfile
            # or counted AGENT reservation.  Treat that lease as authoritative
            # liveness.  Visible compaction children inherit the parent's
            # lineage, so they must also be protected by the parent's lease;
            # otherwise a long summary is folded as stale after 300 seconds,
            # while the coroutine continues and later writes a second header
            # count without a parent compaction event.
            try:
                parent_sid = str(r["parent_session_id"] or "")
                if sandbox_broker.session_has_live_lease(sid) or (
                    parent_sid and sandbox_broker.session_has_live_lease(parent_sid)
                ):
                    continue
            except Exception:  # noqa: BLE001 — lease state is best-effort
                pass
            pending_interaction = session_pending_interaction(sd, sid)
            if pending_interaction:
                # In-process local_llm has no engine pidfile. A parked
                # operator interaction writes nothing, so the 300s freshness
                # stamp freezes while the turn is legitimately waiting. A
                # question has its own unanswered policy; a permission has no
                # wall-clock expiry. Never fold either parked turn.
                continue
            if engine_pidfile.is_alive(sd, sid):
                continue
            # The turn belongs to another live process (a second orchestrator
            # on a shared sessions dir, a detached runner): its leases and
            # pidfiles are not visible here, so its liveness is the stamp.
            try:
                from sharedagents_session.turn_runner import (
                    turn_runner_alive_elsewhere,
                )

                _ti = r["live_turn_idx"]
                if _ti is None:
                    _ti = r["last_turn_idx"]
                if _ti is not None and turn_runner_alive_elsewhere(
                    sd / f"{sid}.jsonl",
                    int(_ti),
                ):
                    continue
            except Exception:  # noqa: BLE001 — unknown owner: old behavior
                pass
            try:
                if int(sandbox_broker.get_broker().session_live(str(sid)) or 0) > 0:
                    continue
            except Exception:  # noqa: BLE001
                pass
            try:
                has_active_children = bool(
                    con.execute(
                        f"SELECT 1 FROM sessions_meta "
                        f"WHERE parent_session_id = ? "
                        f"  AND session_state IN ({placeholders}) LIMIT 1",
                        (sid, *live_turn_states),
                    ).fetchone()
                )
                if has_active_children:
                    continue
            except Exception:  # noqa: BLE001
                pass
            try:
                turn_idx = r["live_turn_idx"]
                if turn_idx is None:
                    turn_idx = r["last_turn_idx"]
                if turn_idx is not None:
                    try:
                        w = get_session_writer(
                            sd,
                            sid,
                            engine=str(r["engine"] or ""),
                        )
                        try:
                            w.turn_ended(
                                turn_idx=int(turn_idx),
                                aborted=True,
                                metrics={"abort_reason": "stale_live_reconcile"},
                            )
                        finally:
                            close_session_writer(sd, sid)
                    except Exception as exc:  # noqa: BLE001
                        # The runner was proven dead above. A damaged log must
                        # not bypass the terminal DB backstop and keep its
                        # open-turn freshness exemption alive forever.
                        log_event(
                            "STALE_LIVE_SESSION_ABORT_LOG_FAILED",
                            level="ERROR",
                            session_id=sid,
                            error=repr(exc),
                        )
                # Clear a leftover sidecar after the fold. A live
                # identity-valid pidfile already continued above, so this
                # is cleanup of a dead/mismatched record — not a second
                # kill of a proven-live engine.
                try:
                    engine_pidfile.kill_session_engine(sd, sid)
                except Exception:  # noqa: BLE001
                    pass
                # Unconditional terminal backstop (idempotent): the row must
                # leave the live set even when no turn event could be emitted.
                mark_session_terminal(sd, sid, state="error")
                finalized += 1
            except Exception as exc:  # noqa: BLE001 — never kill the sweep
                log_event(
                    "STALE_LIVE_SESSION_FINALIZE_FAILED",
                    level="ERROR",
                    session_id=sid,
                    error=repr(exc),
                )
                continue
        return finalized
    except Exception:  # noqa: BLE001
        con.rollback()
        return 0
    finally:
        con.close()


_STALE_WAITING_SWEEP_TASK: Any | None = None


def start_stale_waiting_sweep(
    *,
    interval_s: float = 60.0,
    freshness_seconds: int = LIVE_FRESHNESS_SECONDS,
) -> Any:
    """Start exactly one process-local stale-waiting resolver loop.

    Runs :func:`release_stale_waiting_sessions` on a slow cadence so a
    ``waiting`` session whose children turned ghost (abort/crash never
    folded them) self-heals to ``idle`` within ``interval_s + freshness``
    instead of wedging until the next gateway restart.

    ``interval_s`` / ``freshness_seconds`` are explicit test/config seams;
    production defaults release a wedged parent at most ~6 minutes after the
    last real child activity (60s sweep + the 300s freshness window).
    """
    import asyncio

    global _STALE_WAITING_SWEEP_TASK
    current = _STALE_WAITING_SWEEP_TASK
    if current is not None and not current.done():
        return current
    task = asyncio.create_task(
        _stale_waiting_sweep_loop(
            interval_s=interval_s,
            freshness_seconds=freshness_seconds,
        ),
        name="sharedagents-stale-waiting-sweep",
    )
    _STALE_WAITING_SWEEP_TASK = task

    def _clear(done: Any) -> None:
        global _STALE_WAITING_SWEEP_TASK
        if _STALE_WAITING_SWEEP_TASK is done:
            _STALE_WAITING_SWEEP_TASK = None

    task.add_done_callback(_clear)
    return task


async def _stale_waiting_sweep_loop(
    *,
    interval_s: float,
    freshness_seconds: int,
) -> None:
    import asyncio
    import logging

    _log = logging.getLogger("sharedagents.orchestrator")
    while True:
        try:
            released = release_stale_waiting_sessions(
                freshness_seconds=freshness_seconds,
            )
            if released:
                _log.info(
                    "stale-waiting sweep: released %d waiting session(s) to idle",
                    released,
                )
        except Exception:  # noqa: BLE001 — never kill the sweep loop
            pass
        try:
            finalized = finalize_stale_live_sessions(
                freshness_seconds=freshness_seconds,
            )
            if finalized:
                _log.info(
                    "stale-live sweep: finalized %d ghost-live session(s) (aborted, runner dead)",
                    finalized,
                )
        except Exception:  # noqa: BLE001 — never kill the sweep loop
            pass
        await asyncio.sleep(float(interval_s))


def backfill_parent_links(limit: int | None = None) -> dict:
    """Repair subagent ``parent_session_id`` links from the JSONL headers.

    Boot self-heal for the delegation tree: a legacy ``rebuild()``
    blanked every row's ``parent_session_id`` (the bare-row INSERT defeated the
    ``session_started`` upsert), flattening the orchestrator tree. The JSONL
    headers still carry the real parent, so we re-derive it. Idempotent — only
    fills rows whose stored parent is empty. ``limit`` bounds the scan to the
    most recent N sessions so startup stays cheap. Returns ``{scanned, filled}``.
    """
    return _index().backfill_parent_session_ids(limit=limit)


def delete_promptless_sessions() -> int:
    """Delete sessions that never received a prompt (boot cleanup).

    A "promptless" session has only ``session_started`` / ``header_update``
    events — no ``turn_started`` was ever written, so no user prompt was ever
    processed. These accumulate from aborted ``/new`` clicks, health probes,
    and scaffold churn; having no ``last_turn_ts`` they otherwise litter the
    monitor as phantom rows mis-sorted by activity age.

    SAFETY — never delete a session that did real work, especially an errored
    one (those are valuable for debugging):

    * A prompted session ALWAYS has a ``turn_started``.
    * A prompted session that errored/aborted additionally has a
      ``turn_ended`` → ``last_turn_idx >= 0`` (not even a candidate here).
    * A prompted session that crashed mid-turn has a ``turn_started`` but no
      ``turn_ended`` → caught by the authoritative JSONL guard below and KEPT.

    So we (a) shortlist rows with ``last_turn_idx < 0`` in a non-live state,
    then (b) confirm via the JSONL that NO ``turn_started`` exists before
    deleting the index rows AND the (header-only) log file. Must run AFTER
    :func:`reset_stale_sessions`, which aborts-closes live turns (moving any
    in-flight session to ``last_turn_idx >= 0``). Returns the count deleted.
    """
    from sharedagents_session.session_reader import decode_line

    idx = _index()
    con = idx._connect()
    try:
        placeholders = ",".join("?" for _ in LIVE_STATES)
        candidates = [
            r[0]
            for r in con.execute(
                "SELECT session_hash FROM sessions_meta " + _INCIDENT_EXCLUSION + "  AND last_turn_idx < 0 "
                f"  AND session_state NOT IN ({placeholders})",
                tuple(LIVE_STATES),
            ).fetchall()
        ]
    except Exception:  # noqa: BLE001
        return 0
    finally:
        con.close()

    if not candidates:
        return 0

    sd = sessions_root()
    deleted = 0
    for sid in candidates:
        jsonl = sd / f"{sid}.jsonl"
        # Authoritative guard: KEEP the session if its log shows ANY
        # turn_started (a prompt was processed) — even with no turn_ended
        # (crashed mid-turn), or ANY non-empty line that fails to decode
        # (corrupt / CRC mismatch: it may be the turn_started). Only purely
        # promptless logs are purged. If the file is unreadable or missing,
        # err on the side of keeping the row.
        had_prompt = True
        try:
            # A log touched in the last PROMPTLESS_MIN_AGE_SECONDS may be a
            # session another live process just created and is about to
            # prompt (web: POST /api/sessions, then /prompt ~2 s later). A
            # second orchestrator sharing this sessions dir booting in that
            # gap deleted it. Phantom rows are old by definition; keep fresh.
            if jsonl.exists() and (time.time() - jsonl.stat().st_mtime >= PROMPTLESS_MIN_AGE_SECONDS):
                had_prompt = False
                with jsonl.open("rb") as fp:
                    for raw in fp:
                        raw = raw.rstrip(b"\n")
                        if not raw:
                            continue
                        ev = decode_line(raw)
                        if ev is None or ev.get("type") == "turn_started":
                            had_prompt = True
                            break
        except OSError:
            had_prompt = True
        if had_prompt:
            continue
        try:
            idx.remove_session(sid)
            if jsonl.exists():
                jsonl.unlink()
            deleted += 1
        except OSError:  # noqa: PERF203 — best-effort cleanup
            pass
    return deleted


def list_live_sessions() -> list[dict[str, Any]]:
    """Rows for sessions currently in a live state, across all agents.

    Returned as plain dicts so callers don't need a ``sqlite3``
    dependency. Ordered by agent then most-recent activity.
    """
    idx = _index()
    _refresh_quiet(idx)
    placeholders = ",".join("?" for _ in LIVE_STATES)
    # Freshness gate: a row whose ``session_state`` claims a
    # live value, whose state register hasn't been touched within
    # ``LIVE_FRESHNESS_SECONDS``, AND whose turn is CLOSED is stale/abandoned
    # (a crash/gateway-restart left it mid-stream), NOT active. An OPEN turn
    # (``live_turn_idx`` set) is fresh regardless of the stamp age — one long
    # silent tool call must not age a running session out of the live view
    # (same open-turn rule as ``session_tree``'s Python-side gate; crash
    # rows are bounded by the boot/reset + stale-live resolvers). Exclude
    # everything else here so EVERY consumer of this live view sees only
    # genuinely-live rows. A NULL/absent timestamp with no open turn can't
    # prove freshness, so it reads as stale.
    cutoff_ms = int(time.time() * 1000) - LIVE_FRESHNESS_SECONDS * 1000
    con = idx._connect()
    try:
        rows = con.execute(
            "SELECT agent_name, session_hash, session_state, "
            "       live_turn_idx, live_round_idx, last_turn_ts, "
            "       session_state_updated_at "
            "FROM sessions_meta "
            f"{_INCIDENT_EXCLUSION}"
            f"  AND session_state IN ({placeholders}) "
            "  AND (session_state_updated_at >= ? "
            "       OR live_turn_idx IS NOT NULL) "
            "ORDER BY agent_name, last_turn_ts DESC",
            tuple(LIVE_STATES) + (cutoff_ms,),
        ).fetchall()
    finally:
        con.close()
    return [dict(r) for r in rows]


def _ts_ms(val: Any) -> int:
    """Normalise ``last_turn_ts`` / ``session_state_updated_at`` to epoch-ms."""
    if val is None:
        return 0
    try:
        v = int(val)
    except (TypeError, ValueError):
        return 0
    if v <= 0:
        return 0
    # Values > 1e12 are already ms; smaller values are legacy epoch-seconds.
    return v if v > 1_000_000_000_000 else v * 1000


def _iso_ms(val: Any) -> int:
    """Best-effort ISO-8601 (``created_at``) → epoch-ms; 0 on failure."""
    if not val:
        return 0
    if isinstance(val, (int, float)):
        return _ts_ms(val)
    try:
        from datetime import datetime, timezone

        s = str(val).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except (ValueError, TypeError):
        return 0


def _activity_ms(row: dict[str, Any], *, live: bool) -> int:
    """Best estimate of last meaningful activity for sort / age display.

    For idle/error/closed rows use ``last_turn_ts`` (the real timestamp of the
    last completed turn), falling back to ``created_at`` (session creation) —
    NEVER ``session_state_updated_at``, which a boot sweep / late projector
    re-tail re-stamps to ~now and would make every old session look freshly
    active in the monitor (the post-restart "all started together" bug). Live
    rows DO use ``session_state_updated_at`` since a running turn legitimately
    re-stamps it on every streamed delta.
    """
    lt = _ts_ms(row.get("last_turn_ts"))
    ct = _iso_ms(row.get("created_at"))
    if live:
        su = _ts_ms(row.get("session_state_updated_at"))
        return max(lt, su) if (lt and su) else (su or lt or ct)
    return lt or ct


#: Hidden utility children that stay on disk but never show in the monitor tree
#: or its counters: a title call is one short side call per session, pure noise
#: next to the real subagents. ``compaction`` is NOT hidden (operators read it);
#: ``summary`` is the hidden Tasks-board completion recap utility.
MONITOR_HIDDEN_AGENTS = ("title", "summary")


def monitor_visible_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop the hidden utility children from tree rows."""
    return [r for r in rows if str(r.get("agent_name") or "").strip().lower() not in MONITOR_HIDDEN_AGENTS]


_TREE_ROW_SELECT = (
    "SELECT agent_name, session_hash, parent_session_id, "
    "       session_state, title, engine, model, topic, "
    "       live_turn_idx, live_round_idx, last_turn_ts, "
    "       session_state_updated_at, created_at "
    "FROM sessions_meta "
)

_TREE_ROW_SELECT_V2 = (
    "SELECT agent_name, footer_state_json, session_hash, parent_session_id, "
    "       session_state, title, engine, model, engine_model, topic, "
    "       live_turn_idx, live_round_idx, last_turn_ts, "
    "       session_state_updated_at, created_at, session_phase, "
    "       session_attention, session_lifecycle, session_last_outcome, "
    "       session_phase_changed_at, session_heartbeat_at, "
    "       session_outcome_at, session_state_version "
    "FROM sessions_meta "
)


def _tree_rows(con, suffix: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    """Read V2 rows, falling back for legacy/minimal fixture databases."""
    try:
        rows = con.execute(_TREE_ROW_SELECT_V2 + suffix, params).fetchall()
    except sqlite3.OperationalError as exc:
        if "no such column" not in str(exc).lower():
            raise
        rows = con.execute(_TREE_ROW_SELECT + suffix, params).fetchall()
    return [dict(r) for r in rows]


def _normalize_anchor_hashes(anchor_hashes: Any) -> list[str]:
    """Deduped, bounded session hashes for a conversation-scoped tree."""
    if not anchor_hashes:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in anchor_hashes:
        h = str(raw or "").strip()
        if not h or h in seen or len(h) > 512:
            continue
        seen.add(h)
        out.append(h)
    return out


def _chunked(items: list[str], size: int = 400):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _fetch_newest_tree_rows(con, max_rows: int) -> list[dict[str, Any]]:
    live_ph = ",".join("?" for _ in LIVE_STATES)
    return _tree_rows(
        con,
        _INCIDENT_EXCLUSION + "ORDER BY CASE WHEN session_state IN (" + live_ph + ") "
        "              THEN MAX(COALESCE(NULLIF(last_turn_ts, 0), 0), "
        "                       COALESCE(session_state_updated_at, 0)) "
        "              ELSE COALESCE(NULLIF(last_turn_ts, 0), 0) "
        "         END DESC, "
        "         created_at DESC "
        "LIMIT ?",
        tuple(LIVE_STATES) + (int(max(1, max_rows)),),
    )


def _load_hashes(
    con,
    hashes: list[str],
    by_hash: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    missing = [h for h in hashes if h and h not in by_hash]
    loaded: list[dict[str, Any]] = []
    for chunk in _chunked(missing):
        ph = ",".join("?" for _ in chunk)
        rows = _tree_rows(
            con,
            "WHERE session_hash IN (" + ph + ") "
            "AND session_hash NOT LIKE 'incident-%' "
            "AND agent_name NOT LIKE '%-incident'",
            tuple(chunk),
        )
        for r in rows:
            by_hash[r["session_hash"]] = r
            loaded.append(r)
    return loaded


def _fetch_anchored_tree_rows(
    con,
    anchors: list[str],
    max_rows: int,
) -> list[dict[str, Any]]:
    """Load one conversation's lineage: anchors, ancestors, descendants, siblings."""
    cap = int(max(1, max_rows))
    by_hash: dict[str, dict[str, Any]] = {}
    _load_hashes(con, anchors, by_hash)
    pending = [str(r.get("parent_session_id") or "") for r in list(by_hash.values()) if r.get("parent_session_id")]
    while pending and len(by_hash) < cap:
        missing = [h for h in pending if h and h not in by_hash]
        if not missing:
            break
        loaded = _load_hashes(con, missing, by_hash)
        pending = [str(r.get("parent_session_id") or "") for r in loaded if r.get("parent_session_id")]
    grew = True
    while grew and len(by_hash) < cap:
        grew = False
        known = list(by_hash.keys())
        for chunk in _chunked(known):
            if len(by_hash) >= cap:
                break
            ph = ",".join("?" for _ in chunk)
            kids = _tree_rows(
                con,
                "WHERE parent_session_id IN (" + ph + ") "
                "AND session_hash NOT LIKE 'incident-%' "
                "AND agent_name NOT LIKE '%-incident'",
                tuple(chunk),
            )
            for r in kids:
                h = r["session_hash"]
                if h in by_hash:
                    continue
                if len(by_hash) >= cap:
                    break
                by_hash[h] = r
                grew = True
    return list(by_hash.values())


def session_tree(
    *,
    include_all: bool = False,
    include_unknown: bool | None = None,
    max_rows: int = DEFAULT_TREE_MAX_ROWS,
    freshness_seconds: int = LIVE_FRESHNESS_SECONDS,
    refresh: bool = False,
    anchor_hashes: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Agent → session → subagent hierarchy for the live-monitoring tree.

    Answers the two questions the web monitor asks in real time:

    * *how many sessions are active by each agent* — a session is
      "active" when its ``session_state`` is one of :data:`LIVE_STATES`
      **and** that state was touched within ``freshness_seconds`` (see
      :data:`LIVE_FRESHNESS_SECONDS`). The freshness gate is what stops a
      session abandoned mid-turn (crash / gateway restart, leaving no
      terminal event) from lingering as a false "active" forever — the
      projector re-stamps ``session_state_updated_at`` on every streamed
      delta and tool event, so a real running turn is always fresh.
    * *how many subagents is each agent calling* — a subagent is a
      session whose ``parent_session_id`` points at another session; we
      attribute it to the agent that owns the **parent** session.

    The tree is built from one bounded scan of ``sessions_meta`` (newest
    first) joined to itself on ``parent_session_id``. By default only the
    *live subtree* is returned:     every session currently in a live state,
    plus the ancestor chain that makes it reachable from a top-level
    root, plus any live descendants. Pass ``include_all=True`` to return
    every session (browse mode) capped at ``max_rows``. Pass
    ``anchor_hashes`` to return only the lineage of those sessions
    (ancestors, descendants, siblings) instead of the global newest-first
    scan — the chat Agent Monitor uses this so ``?conversation=`` is a
    prune, not a join on top of the whole DAG. Anchored trees always keep
    completed (idle/error/closed) descendants; live-only pruning is for
    the unscoped /agents page.

    **Agent-name gate (display-only):** rows whose ``agent_name`` is not
    a legitimate identity (delegation roster ∪ profile identities ∪
    built-in agent defaults ∪ on-disk configs — see
    :func:`agent_profiles.is_known_agent_name`) are hidden from the
    tree, its per-agent buckets and the summary aggregates, so a
    fixture-style phantom name (``rt`` / ``reload-test``) never renders
    as a real agent. Stored rows are NEVER deleted or altered. Escape
    hatches for debugging: ``include_all=True`` (browse mode, ``?all=1``
    on the API), ``include_unknown=True`` (``?show_unknown=1``), or the
    environment variable ``SHAREDAGENTS_TREE_SHOW_UNKNOWN_AGENTS=1``. Rows
    with an empty ``agent_name`` (the legacy unattributed edge) always
    stay visible as ``(unknown)``.

    Returns a JSON-serialisable dict::

        {
          "live_states": [...],
          "summary": [ {agent, active_sessions, subagents_calling,
                        total_sessions}, ... ],   # sorted, busiest first
          "totals": {agents, active_sessions, subagents, roots},
          "agents": [ {agent, active_sessions, subagents_calling,
                       roots: [ <node>, ... ]}, ... ],
        }

    where each ``<node>`` is::

        {hash, title, agent, engine, model, state, live, round, turn,
         last_turn_ts, state_updated_at, subagent_count, children: [...]}

    ``subagent_count`` is the number of **live** descendants at any depth.
    """
    import os as _os

    from sharedagents_agent.agent_profiles import is_known_agent_name

    anchors = _normalize_anchor_hashes(anchor_hashes)
    if anchors:
        # Conversation-scoped trees always keep completed delegations.
        include_all = True

    # Display-only phantom-agent gate. Unknown names are hidden by default;
    # every escape hatch (browse mode / explicit flag / env) restores them.
    show_unknown = bool(
        include_all or include_unknown or _os.environ.get("SHAREDAGENTS_TREE_SHOW_UNKNOWN_AGENTS") == "1"
    )
    _known_memo: dict[str, bool] = {}

    def _agent_known(name: str) -> bool:
        """True when *name* may render on the monitor.

        Empty names are the deliberate legacy/unattributed edge and stay
        visible (they bucket as ``(unknown)``). Memoised per distinct
        name — the monitor polls every ~1.5s and must stay cheap.
        """
        nm = (name or "").strip()
        if not nm:
            return True
        if show_unknown:
            return True
        ok = _known_memo.get(nm)
        if ok is None:
            ok = is_known_agent_name(nm)
            _known_memo[nm] = ok
        return ok

    idx = _index()
    # The monitor reads the index, not the JSONL — but a brand-new session or a
    # subagent written by another process isn't in the index until folded. Fold
    # on this read so they appear within one poll: ``refresh=True`` forces it
    # (recovery / explicit), otherwise a throttled fold keeps the hot path
    # cheap while still surfacing new sessions/subagents promptly (see
    # :func:`_maybe_refresh_for_tree`).
    if refresh:
        _refresh_quiet(idx)
    else:
        _maybe_refresh_for_tree(idx)
    con = idx._connect()
    try:
        rows = _fetch_anchored_tree_rows(con, anchors, max_rows) if anchors else _fetch_newest_tree_rows(con, max_rows)
    finally:
        con.close()

    # Display-only phantom-agent gate: drop rows whose agent identity is
    # unknown (fixture-style leaks) from the tree payload — the stored
    # sessions_meta row is untouched. Empty agent_name (legacy
    # unattributed edge) always stays.
    if not show_unknown:
        rows = [r for r in rows if _agent_known(r.get("agent_name") or "")]
    rows = monitor_visible_rows(rows)

    by_hash: dict[str, dict[str, Any]] = {r["session_hash"]: r for r in rows}
    children: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        parent = r.get("parent_session_id") or ""
        if parent:
            children.setdefault(parent, []).append(r)

    # Freshness gate: a session only counts as live if its state register was
    # touched recently. ``session_state_updated_at`` is epoch-ms; guard < 0 /
    # missing as stale. freshness_seconds <= 0 disables the gate.
    now_ms = time.time() * 1000.0
    cutoff_ms = (now_ms - freshness_seconds * 1000.0) if freshness_seconds > 0 else None
    for r in rows:
        in_live_state = r.get("session_state") in LIVE_STATES
        upd = r.get("session_state_updated_at") or 0
        fresh = cutoff_ms is None or upd >= cutoff_ms
        # Elapsed time alone is NOT proof of death. ``session_state_updated_at``
        # is re-stamped only when events are projected, so a single long silent
        # tool call (no deltas, no new tool_call) ages a genuinely-running turn
        # out of ``live`` after LIVE_FRESHNESS_SECONDS while its turn is still
        # OPEN. An open turn (``live_turn_idx`` set by ``turn_started``, cleared
        # by ``turn_ended``) is itself freshness evidence. Crash-abandoned rows
        # keep their open turn too — they are bounded by the boot sweep
        # (:func:`reset_stale_sessions`) and the 60 s write-side resolver
        # (:func:`finalize_stale_live_sessions`), which fold provably-dead rows
        # (no open job, no live lease, no live engine pidfile) terminal, so a
        # truly dead row cannot look live forever.
        open_turn = r.get("live_turn_idx") is not None
        r["_live"] = in_live_state and (fresh or open_turn)
        # Surface the distinction for browse mode (?all=1): a row in a live
        # state that failed the freshness gate AND has no open turn is
        # "stale" (abandoned), not idle.
        r["_stale"] = in_live_state and not (fresh or open_turn)

    # --- per-agent rollup (unbounded aggregate queries) ----------------
    # The tree itself is bounded by max_rows, but the summary counts must
    # reflect the FULL database — otherwise deployments with many sessions
    # silently undercount total_sessions / active_sessions once the scanned
    # set is truncated. Falls back to the truncated-row tally on any error.
    summary: dict[str, dict[str, Any]] = {}

    def _ensure(agent: str) -> dict[str, Any]:
        a = agent or "(unknown)"
        return summary.setdefault(
            a,
            {"agent": a, "active_sessions": 0, "subagents_calling": 0, "total_sessions": 0},
        )

    def _tally_from_rows() -> None:
        summary.clear()
        for r in rows:
            _ensure(r["agent_name"])["total_sessions"] += 1
        for r in rows:
            if not r["_live"]:
                continue
            _ensure(r["agent_name"])["active_sessions"] += 1
            parent = by_hash.get(r.get("parent_session_id") or "")
            if parent is not None:
                _ensure(parent["agent_name"])["subagents_calling"] += 1

    try:
        if anchors:
            # Chat footer trees must not pay whole-DB COUNT(*) rollups.
            _tally_from_rows()
        else:
            aggr_con = idx._connect()
            try:
                live_placeholders = ",".join("?" for _ in LIVE_STATES)
                # Total sessions per agent (unbounded).
                for r in aggr_con.execute(
                    "SELECT agent_name, COUNT(*) AS cnt FROM sessions_meta "
                    + _INCIDENT_EXCLUSION
                    + "GROUP BY agent_name ORDER BY agent_name"
                ).fetchall():
                    if not _agent_known(r["agent_name"] or ""):
                        continue
                    _ensure(r["agent_name"])["total_sessions"] = r["cnt"]
                # Live sessions per agent, applying the same freshness cutoff
                # the scanned set used (so counts agree with the tree).
                if cutoff_ms is not None:
                    # Match the row-level ``_live`` gate above: an OPEN turn
                    # (live_turn_idx set) is fresh regardless of the stamp age.
                    live_rows = aggr_con.execute(
                        f"SELECT agent_name, COUNT(*) AS cnt FROM sessions_meta "
                        f"{_INCIDENT_EXCLUSION}"
                        f"  AND session_state IN ({live_placeholders}) "
                        f"  AND (session_state_updated_at >= ? "
                        f"       OR live_turn_idx IS NOT NULL) "
                        f"GROUP BY agent_name ORDER BY agent_name",
                        (*LIVE_STATES, int(cutoff_ms)),
                    ).fetchall()
                else:
                    live_rows = aggr_con.execute(
                        f"SELECT agent_name, COUNT(*) AS cnt FROM sessions_meta "
                        f"{_INCIDENT_EXCLUSION}"
                        f"  AND session_state IN ({live_placeholders}) "
                        f"GROUP BY agent_name ORDER BY agent_name",
                        tuple(LIVE_STATES),
                    ).fetchall()
                for r in live_rows:
                    if not _agent_known(r["agent_name"] or ""):
                        continue
                    _ensure(r["agent_name"])["active_sessions"] = r["cnt"]
                # subagents_calling = live sessions whose parent is owned by
                # this agent. Apply the SAME freshness cutoff as active_sessions
                # (above): without it, a subagent whose session_state is STUCK at a
                # live value — e.g. a turn killed mid-flight by an orchestrator
                # restart or abort, leaving session_state="tool_running" that is
                # never finalized — is counted as "calling" forever, even though it
                # aged out of active_sessions. That inconsistency surfaced as the
                # Agent Monitor showing a subagent active (🔗1) while everything is
                # idle (active_sessions=0).
                _calling_where = (
                    f"c.session_state IN ({live_placeholders}) AND c.agent_name NOT IN ('title', 'summary') "
                )
                _calling_params: list[Any] = list(LIVE_STATES)
                if cutoff_ms is not None:
                    # Same open-turn-is-fresh rule as the row-level gate and
                    # the active_sessions rollup above.
                    _calling_where += "  AND (c.session_state_updated_at >= ?        OR c.live_turn_idx IS NOT NULL) "
                    _calling_params.append(int(cutoff_ms))
                for r in aggr_con.execute(
                    "SELECT p.agent_name AS agent_name, COUNT(*) AS cnt "
                    "FROM sessions_meta AS p "
                    "JOIN sessions_meta AS c ON c.parent_session_id = p.session_hash "
                    f"WHERE {_calling_where}"
                    "  AND p.session_hash NOT LIKE 'incident-%' "
                    "  AND p.agent_name NOT LIKE '%-incident' "
                    "GROUP BY p.agent_name ORDER BY p.agent_name",
                    tuple(_calling_params),
                ).fetchall():
                    if not _agent_known(r["agent_name"] or ""):
                        continue
                    _ensure(r["agent_name"])["subagents_calling"] = r["cnt"]
            finally:
                aggr_con.close()
    except Exception:  # noqa: BLE001 — fall back to the truncated-row tally
        _tally_from_rows()

    # --- which sessions to surface in the tree -------------------------
    # Keep a session if it is live, or has a kept (→ live) descendant, so
    # a live subagent always drags its parent chain into view.
    # include_all short-circuits the prune (browse mode). Conversation
    # anchors do too: completed subs in this lineage stay visible.
    _kept_memo: dict[str, bool] = {}

    def _kept(r: dict[str, Any]) -> bool:
        if include_all or anchors:
            return True
        h = r["session_hash"]
        cached = _kept_memo.get(h)
        if cached is not None:
            return cached
        # Seed False first to guard against cycles in malformed data.
        _kept_memo[h] = False
        keep = bool(r["_live"]) or any(_kept(c) for c in children.get(h, []))
        _kept_memo[h] = keep
        return keep

    _desc_memo: dict[str, int] = {}

    def _live_descendants(r: dict[str, Any]) -> int:
        h = r["session_hash"]
        cached = _desc_memo.get(h)
        if cached is not None:
            return cached
        _desc_memo[h] = 0  # cycle guard
        total = 0
        for c in children.get(h, []):
            if c["_live"]:
                total += 1
            total += _live_descendants(c)
        _desc_memo[h] = total
        return total

    def _node(r: dict[str, Any]) -> dict[str, Any]:
        kids = [_node(c) for c in children.get(r["session_hash"], []) if _kept(c)]
        # Stable, useful order: live first, then most-recent activity.
        kids.sort(key=lambda n: (not n["live"], -(n["last_activity_at"] or 0)))
        descendant_count = _live_descendants(r)
        own_live = (
            r.get("live_turn_idx") is not None
            and legacy_phase(
                r.get("session_state") or "idle",
                r.get("session_phase") or "",
                live_turn_idx=r.get("live_turn_idx"),
            )
            != "idle"
        )
        row_stale = row_stale_for_display(
            raw_stale=bool(r.get("_stale")),
            own_live=own_live,
            descendant_count=descendant_count,
        )
        activity = derive_activity(
            r,
            now_ms=int(now_ms),
            freshness_seconds=freshness_seconds,
            tree_busy=bool(r["_live"] or descendant_count),
            stale=row_stale,
            descendant_count=descendant_count,
        )
        return {
            "hash": r["session_hash"],
            "title": r.get("title") or f"untitled-{str(r['session_hash'])[-8:]}",
            "session_summary": r.get("session_summary") or "",
            "agent": r.get("agent_name") or "(unknown)",
            "footer_state": _json_dict(r.get("footer_state_json")),
            "engine": r.get("engine") or "",
            # Selected/picker model first. Using engine_model here made the
            # Agent Monitor footer show the resolved id (e.g. qwen-3.8) while
            # the chat selector still showed Auto — a split of the same agent.
            "model": r.get("model") or r.get("engine_model") or "",
            "engine_model": r.get("engine_model") or "",
            "topic": r.get("topic") or "general",
            "state": r.get("session_state") or "idle",
            "live": bool(r["_live"]),
            # True when the row claims a live state but failed the freshness
            # gate — i.e. abandoned mid-turn. Only ever surfaced via ?all=1
            # (stale rows are pruned from the default live view).
            "stale": row_stale,
            "round": r.get("live_round_idx"),
            "turn": r.get("live_turn_idx"),
            "last_turn_ts": r.get("last_turn_ts") or 0,
            "state_updated_at": r.get("session_state_updated_at") or 0,
            "created_at": r.get("created_at") or "",
            "last_activity_at": _activity_ms(r, live=bool(r["_live"])),
            "subagent_count": descendant_count,
            # V2 is additive. ``state``/``live`` retain their historical
            # meaning for TUI and older WebUI clients; new clients consume the
            # shared activity snapshot and never infer tree activity from the
            # own-row ``live`` bit.
            "activity": activity,
            **activity,
            "children": kids,
        }

    # Roots = top-level sessions (no parent) or sessions whose parent fell
    # outside the bounded window.
    roots = [r for r in rows if (r.get("parent_session_id") or "") not in by_hash and _kept(r)]

    agents: dict[str, dict[str, Any]] = {}
    for r in roots:
        a = r.get("agent_name") or "(unknown)"
        bucket = agents.setdefault(
            a,
            {"agent": a, "active_sessions": 0, "subagents_calling": 0, "roots": []},
        )
        bucket["roots"].append(_node(r))

    # Fold the rollup figures onto each agent bucket + sort roots.
    for a, bucket in agents.items():
        s = summary.get(a, {})
        bucket["active_sessions"] = s.get("active_sessions", 0)
        bucket["subagents_calling"] = s.get("subagents_calling", 0)
        bucket["roots"].sort(key=lambda n: (not n["live"], -(n["last_activity_at"] or 0)))

    agent_list = sorted(
        agents.values(),
        key=lambda b: (-b["active_sessions"], -b["subagents_calling"], b["agent"]),
    )
    summary_list = sorted(
        summary.values(),
        key=lambda s: (-s["active_sessions"], -s["subagents_calling"], s["agent"]),
    )

    return {
        "live_states": list(LIVE_STATES),
        "summary": summary_list,
        "totals": {
            "agents": len([s for s in summary_list if s["active_sessions"] or s["subagents_calling"]]),
            "active_sessions": sum(s["active_sessions"] for s in summary_list),
            "subagents": sum(s["subagents_calling"] for s in summary_list),
            "roots": len(roots),
        },
        "agents": agent_list,
    }


def all_sessions(limit: int = 100) -> list[dict[str, Any]]:
    """Every ``sessions_meta`` row across all agents, newest first."""
    idx = _index()
    _refresh_quiet(idx)
    con = idx._connect()
    try:
        rows = con.execute(
            "SELECT agent_name, session_hash, session_state, "
            "       last_turn_ts, created_at, engine, title "
            "FROM sessions_meta "
            + _INCIDENT_EXCLUSION
            + "ORDER BY COALESCE(NULLIF(last_turn_ts, 0), updated_at) DESC, "
            "         created_at DESC "
            "LIMIT ?",
            (int(max(1, limit)),),
        ).fetchall()
    finally:
        con.close()
    return [dict(r) for r in rows]


def render_table(rows: list[dict[str, Any]]) -> str:
    """Format rows as a fixed-width table for the CLI / TUI status."""
    if not rows:
        return "(no sessions)"
    cols = (
        ("agent_name", "agent", 14),
        ("session_state", "state", 14),
        ("session_hash", "session_hash", 30),
        ("live_round_idx", "round", 6),
        ("live_turn_idx", "turn", 6),
        ("last_turn_ts", "last_turn_ts", 14),
    )
    header_line = " | ".join(f"{name:<{w}}" for _, name, w in cols)
    out = [header_line, "-" * len(header_line)]
    for r in rows:
        out.append(
            " | ".join(f"{str(r.get(field) if r.get(field) is not None else '-'):<{w}}" for field, _name, w in cols)
        )
    return "\n".join(out)


def session_roots(
    *,
    limit: int = 50,
    freshness_seconds: int = LIVE_FRESHNESS_SECONDS,
) -> list[dict[str, Any]]:
    """Top-level sessions only (no subagent rows, no tree walk).

    Powers the web sidebar / topics list without building the full
    agent→session→child DAG. Same live/freshness rule as :func:`session_tree`.
    Reads ``sessions_meta`` only — do not ``refresh()`` here. A full JSONL
    stat-scan is ~1.5s on a large store and made the sidebar miss 500ms.
    """
    idx = _index()
    con = idx._connect()
    try:
        root_tail = (
            _INCIDENT_EXCLUSION + " AND (parent_session_id IS NULL OR parent_session_id = '') "
            "ORDER BY last_turn_ts DESC, created_at DESC LIMIT ?"
        )
        root_params = (int(max(1, limit)),)
        try:
            rows = [
                dict(r)
                for r in con.execute(
                    "SELECT session_hash, agent_name, title, model, topic, "
                    "       session_summary, session_state, last_turn_ts, "
                    "       session_state_updated_at, created_at, session_phase, "
                    "       session_attention, session_lifecycle, "
                    "       session_last_outcome, session_phase_changed_at, "
                    "       session_heartbeat_at, session_outcome_at, "
                    "       live_turn_idx "
                    "FROM sessions_meta " + root_tail,
                    root_params,
                ).fetchall()
            ]
        except sqlite3.OperationalError as exc:
            if "no such column" not in str(exc).lower():
                raise
            rows = [
                dict(r)
                for r in con.execute(
                    "SELECT session_hash, agent_name, title, model, topic, "
                    "       session_summary, session_state, last_turn_ts, "
                    "       session_state_updated_at, created_at "
                    "FROM sessions_meta " + root_tail,
                    root_params,
                ).fetchall()
            ]
    finally:
        con.close()
    now_ms = time.time() * 1000.0
    cutoff_ms = (now_ms - freshness_seconds * 1000.0) if freshness_seconds > 0 else None
    out: list[dict[str, Any]] = []
    for r in rows:
        in_live = r.get("session_state") in LIVE_STATES
        upd = r.get("session_state_updated_at") or 0
        fresh = cutoff_ms is None or upd >= cutoff_ms
        # Same open-turn-is-fresh rule as :func:`session_tree`: a long silent
        # tool call keeps its turn open (``live_turn_idx`` set) and must not
        # age out of ``live``. Rows with no open turn still age out.
        open_turn = r.get("live_turn_idx") is not None
        live_flag = bool(in_live and (fresh or open_turn))
        activity = derive_activity(
            r,
            now_ms=int(now_ms),
            freshness_seconds=freshness_seconds,
            tree_busy=live_flag,
            stale=bool(in_live and not (fresh or open_turn)),
        )
        out.append(
            {
                "hash": r.get("session_hash") or "",
                "title": r.get("title") or "",
                "session_summary": r.get("session_summary") or "",
                "agent": r.get("agent_name") or "",
                "model": r.get("model") or "",
                "topic": r.get("topic") or "general",
                "last_turn_ts": r.get("last_turn_ts") or 0,
                "live": live_flag,
                "state": r.get("session_state") or "idle",
                "activity": activity,
                **activity,
            }
        )
    return out


__all__ = [
    "LIVE_STATES",
    "DEFAULT_TREE_MAX_ROWS",
    "list_live_sessions",
    "session_tree",
    "session_roots",
    "all_sessions",
    "reset_stale_sessions",
    "release_stale_waiting_sessions",
    "finalize_stale_live_sessions",
    "start_stale_waiting_sweep",
    "delete_promptless_sessions",
    "render_table",
    "_activity_ms",
]
