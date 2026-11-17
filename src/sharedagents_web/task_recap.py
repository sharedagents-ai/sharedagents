"""sharedagents_web.task_recap — completion recap + pinned task title for the board.

The Tasks board (``routes/tasks.py``) derives ``card`` live on every poll. This
module owns the two things the board does *because* a card landed in
**Completed**:

* **the recap** — one small side call on the selected configured deployment,
  or the Jarvis default (the same resolution the session-title agent uses), that recaps the task and says
  whether it really finished or still needs a follow-up. It runs on the
  GATEWAY (``POST /control/v1/orchestrator/sessions/{sid}/summary``) because
  that is where the parent session's turns live; this side owns only the
  *trigger* and the *persistence* (``task_board.tracked_sessions.summary_*``).
  The recap is a hidden ``summary`` child session, so it can never itself make
  the parent look busy (``session_activity.count_fresh_live_descendants``
  excludes ``title``/``summary`` children).

* **the pinned title** — a task created with a label must show that exact
  label as the PARENT session's title, on the board and in the sidebar, with
  the automatic title agent disabled for that parent (``title_locked``).
  Children keep their own automatic titles: they have no ``tracked_sessions``
  row, so they are never pinned.

Design constraints that shape the code below:

* **Never add latency to a poll.** ``request_recaps`` only decides (pure
  ``recap_due``) and hands the work to a daemon thread.
* **Single flight across processes.** The claim is a CONDITIONAL UPDATE in
  SQLite (``TaskBoardRegistry.claim_summary``), so two web workers — or the
  poll racing the sweeper — cannot both spend a side call.
* **No re-fire without a new parent turn.** ``summary_stamp`` is the parent's
  OWN ``last_turn_ts``; the recap child's turns never move it.
* **No hardcoded credentials.** The gateway call rides
  ``server.gateway_request``, which attaches the existing bearer token.
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

from sharedagents_session.session_recap import VERDICT_UNKNOWN, parse_recap_reply

#: A Completed card is only recap-eligible once the parent has been quiet this
#: long — otherwise a task that just finished its last turn would be recapped
#: mid-settle (and again on the next turn).
MIN_QUIET_S = 30
#: Minimum spacing between attempts on one row (and after a ``failed`` run).
MIN_INTERVAL_S = 60
#: After this many consecutive failures, back off to ``FAILURE_BACKOFF_S``.
FAILURE_BACKOFF_THRESHOLD = 3
FAILURE_BACKOFF_S = 3600
#: A ``running`` claim older than this is treated as a dead worker and may be
#: re-claimed (rolling deploy / killed process).
STALE_RUNNING_S = 300
#: Drift re-pin throttle: at most one title push per row per this window.
PIN_RETRY_S = 120
#: Server-side sweeper period: recaps still run when nobody has the tab open.
SWEEP_PERIOD_S = 30
#: Gateway timeouts: the recap call itself is bounded to ~120 s server-side.
RECAP_TIMEOUT_S = 150.0
PIN_TIMEOUT_S = 20.0

#: Fields the API hands the client for the recap subcard.
RECAP_ITEM_FIELDS = (
    "summary",
    "summary_verdict",
    "summary_follow_up",
    "summary_status",
    "summary_at",
)

_sweeper_lock = threading.Lock()
_sweeper_started = False
_sweeper_thread: threading.Thread | None = None
_sweeper_stop: threading.Event | None = None
_batch_lock = threading.Lock()
_pairs_provider: Callable[[], list[tuple[dict, dict]]] | None = None


# --- time helpers --------------------------------------------------------


def _now_iso(now_ms: int | None = None) -> str:
    ms = int(now_ms if now_ms is not None else time.time() * 1000)
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()


def _iso_ms(value: Any) -> int | None:
    """ISO stamp (or epoch number) → epoch ms. ``None`` when unusable."""
    if value is None or value == "":
        return None
    try:
        v = float(value)
    except (TypeError, ValueError, OverflowError):
        pass
    else:
        if not math.isfinite(v):
            return None
        return int(v if v > 1e12 else v * 1000)
    try:
        dt = datetime.fromisoformat(str(value))
    except (ValueError, OverflowError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    try:
        return int(dt.timestamp() * 1000)
    except (ValueError, OverflowError, OSError):
        return None


def turn_stamp_ms(value: Any) -> int | None:
    """Normalise a tree/store ``last_turn_ts`` to epoch ms.

    The agent tree reports epoch seconds (occasionally ms) while the web store
    reports ISO strings — the same heterogeneity ``routes/tasks.py::_ts_key``
    handles. Returns ``None`` when there is no usable stamp (a task that never
    ran a turn has nothing to recap).
    """
    ms = _iso_ms(value)
    if not ms or ms <= 0:
        return None
    return ms


# --- due rule (pure) -----------------------------------------------------


def recap_due(
    row: dict,
    *,
    card: str,
    parent_turn_stamp: int | None,
    now_ms: int | None = None,
    min_quiet_s: int = MIN_QUIET_S,
    min_interval_s: int = MIN_INTERVAL_S,
) -> bool:
    """True when this row deserves a recap side call right now.

    Pure decision (no I/O) so the whole retry matrix is unit-testable:

    * only a **Completed**, non-archived row;
    * the parent must have run at least one turn (no stamp → nothing to say);
    * the parent must be **quiet** for ``min_quiet_s`` (a turn still settling
      is not a finished task);
    * a ``running`` claim blocks until it is stale (``STALE_RUNNING_S``);
    * the row is recapped once per parent turn: ``summary_stamp`` >= the
      parent's current ``last_turn_ts`` means there is nothing new;
    * attempts are spaced by ``min_interval_s``, and by
      ``FAILURE_BACKOFF_S`` once ``FAILURE_BACKOFF_THRESHOLD`` attempts failed
      in a row.
    """
    if not row:
        return False
    if str(row.get("archived_at") or "").strip():
        return False
    if str(card or "") != "completed":
        return False
    if parent_turn_stamp is None:
        return False
    if row.get("completion_stamp") != parent_turn_stamp:
        return False
    now = int(now_ms if now_ms is not None else time.time() * 1000)
    if now - int(parent_turn_stamp) < min_quiet_s * 1000:
        return False

    stamp = row.get("summary_stamp")
    try:
        stamp_ms = int(stamp) if stamp is not None and str(stamp) != "" else None
    except (TypeError, ValueError, OverflowError):
        stamp_ms = None
    if stamp_ms is not None and int(parent_turn_stamp) <= stamp_ms:
        return False

    status = str(row.get("summary_status") or "").strip().lower()
    attempt_ms = _iso_ms(row.get("summary_attempt_at"))
    if status == "running":
        return attempt_ms is None or now - attempt_ms > STALE_RUNNING_S * 1000

    failures = 0
    try:
        failures = int(row.get("summary_failures") or 0)
    except (TypeError, ValueError):
        failures = 0
    interval = FAILURE_BACKOFF_S if failures >= FAILURE_BACKOFF_THRESHOLD else min_interval_s
    return not (attempt_ms is not None and now - attempt_ms < interval * 1000)


def summary_fields(row: dict, *, card: str | None = None, parent_turn_stamp: int | None = None) -> dict:
    """Expose a recap only while it describes the finished current turn.

    A working task keeps its last readable recap, flagged
    ``summary_previous``, so the board can show what the previous turn did
    while the new one runs (never as a verdict on the current turn).
    """
    out = {key: row.get(key) or "" for key in RECAP_ITEM_FIELDS}
    verdict = str(out.get("summary_verdict") or "").strip().lower()
    out["summary_verdict"] = verdict if verdict and verdict != VERDICT_UNKNOWN else ""
    if not out["summary_verdict"] and not parse_recap_reply(out["summary"])["summary"]:
        out["summary"] = ""
    if card is not None:
        try:
            current = (
                row.get("summary_status") == "done"
                and bool(out["summary"])
                and parent_turn_stamp is not None
                and int(row.get("summary_stamp")) == parent_turn_stamp
            )
        except (TypeError, ValueError, OverflowError):
            current = False
        pending = card == "completed" and parent_turn_stamp is not None and not current
        previous = card == "working" and bool(out["summary"])
        if card != "archived" and (card == "working" or not current):
            status = str(row.get("summary_status") or "")
            kept = (
                {
                    key: out[key]
                    for key in (
                        "summary",
                        "summary_verdict",
                        "summary_follow_up",
                        "summary_at",
                    )
                }
                if previous
                else {}
            )
            out = dict.fromkeys(RECAP_ITEM_FIELDS, "")
            out.update(kept)
            if pending:
                out["summary_status"] = status if status in {"queued", "running", "failed"} else "queued"
        out["summary_pending"] = pending
        out["summary_previous"] = previous
    return out


def recap_card(row: dict, *, card: str, parent_turn_stamp: int | None) -> str:
    """Move an idle task with a current follow-up verdict to Needs input.

    Old recaps never override a new turn's live state or prevent the new
    completed turn from being recapped. Archive and working states win.
    """
    if card != "completed" or row.get("summary_status") != "done":
        return card
    if row.get("summary_verdict") != "follow_up" or parent_turn_stamp is None:
        return card
    try:
        stamp = int(row.get("summary_stamp"))
    except (TypeError, ValueError, OverflowError):
        return card
    return "needs_input" if stamp == parent_turn_stamp else card


def recap_archivable(row: dict, *, card: str, parent_turn_stamp: int | None) -> bool:
    """Only a complete recap covering the latest idle turn is archivable."""
    if card != "completed" or row.get("summary_status") != "done":
        return False
    if row.get("summary_verdict") != "complete" or parent_turn_stamp is None:
        return False
    try:
        return int(row.get("summary_stamp")) == parent_turn_stamp
    except (TypeError, ValueError, OverflowError):
        return False


# --- pin decision (pure) -------------------------------------------------


def pin_due(
    row: dict,
    *,
    gateway_title: str,
    now_ms: int | None = None,
    retry_s: int = PIN_RETRY_S,
) -> bool:
    """True when this labelled task's parent title must be (re)pinned.

    A labelled task is the operator's name for the task: the parent session's
    title must be exactly that label, locked against the title agent. The
    pin re-fires when a race (the turn-1 title call landing first) left a
    different title behind, throttled to one attempt per ``retry_s``.
    """
    label = str(row.get("label") or "").strip()
    if not label:
        return False
    if str(row.get("archived_at") or "").strip():
        return False
    if str(gateway_title or "").strip() == label and row.get("label_pin_confirmed"):
        return False
    now = int(now_ms if now_ms is not None else time.time() * 1000)
    last_attempt = _iso_ms(row.get("label_pinned_at"))
    return not (last_attempt is not None and now - last_attempt < retry_s * 1000)


# --- gateway calls -------------------------------------------------------


def _gateway():
    """Late-bound ``server`` module so tests patch one attribute."""
    from sharedagents_web import server

    return server


def run_recap(
    row: dict,
    item: dict,
    *,
    registry: Any,
    now_ms: int | None = None,
) -> dict | None:
    """Claim + execute one recap synchronously. Returns the parsed reply.

    ``None`` means "another worker owns it" (claim lost) — the caller simply
    does nothing. Any failure marks the row ``failed`` (bumping the backoff);
    the board keeps stale recap text hidden while the current turn is pending.
    """
    now = int(now_ms if now_ms is not None else time.time() * 1000)
    row_id = str(row.get("id") or "")
    sid = str(row.get("session_id") or "")
    if not row_id or not sid:
        return None
    stale_before = _now_iso(now - STALE_RUNNING_S * 1000)
    stamp = turn_stamp_ms(item.get("last_turn_ts"))
    if stamp is None:
        return None
    claim_at = _now_iso(now)
    if not registry.claim_summary(
        row_id,
        now_iso=claim_at,
        stale_before_iso=stale_before,
        parent_turn_stamp=stamp,
        retry_before_iso=_now_iso(now - MIN_INTERVAL_S * 1000),
        backoff_before_iso=_now_iso(now - FAILURE_BACKOFF_S * 1000),
        require_transition=True,
    ):
        return None
    label = str(row.get("label") or "").strip() or str(item.get("title") or "").strip()
    try:
        selected_model = _gateway().web_state.kv_get("tasks.summary_model:" + str(row.get("owner") or "")) or ""
        resp = _gateway().gateway_request(
            "POST",
            f"/control/v1/orchestrator/sessions/{sid}/summary",
            json={"label": label, "task_id": row_id, **({"model": selected_model} if selected_model else {})},
            timeout=RECAP_TIMEOUT_S,
            retry_unavailable=False,
        )
        if resp is None or resp.status_code != 200:
            registry.fail_summary(row_id, claim_at=claim_at)
            return None
        payload = resp.json()
        if not isinstance(payload, dict) or not payload.get("ok"):
            registry.fail_summary(row_id, claim_at=claim_at)
            return None
        if selected_model and payload.get("model") != selected_model:
            # A gateway from before model selection may silently ignore the
            # request. Never store its default-model reply as the selected one.
            registry.fail_summary(row_id, claim_at=claim_at)
            return None
        summary = payload.get("summary")
        if not isinstance(summary, str) or not parse_recap_reply(summary)["summary"]:
            registry.fail_summary(row_id, claim_at=claim_at)
            return None
        registry.finish_summary(
            row_id,
            summary=summary,
            verdict=str(payload.get("verdict") or ""),
            follow_up=str(payload.get("follow_up") or ""),
            stamp=stamp,
            now_iso=_now_iso(now),
            claim_at=claim_at,
        )
        return payload
    except Exception:  # noqa: BLE001 — a recap must never break the board
        try:
            registry.fail_summary(row_id, claim_at=claim_at)
        except Exception:  # noqa: BLE001
            pass
        return None


def push_pin(
    row: dict,
    *,
    gateway_title: str,
    registry: Any,
    now_ms: int | None = None,
) -> bool:
    """Push the task label as the parent session's PINNED title.

    The web store title is written too, so the sidebar (which prefers the
    gateway title) and the board agree the moment the pin lands. Only the
    parent session's id ever appears here — a child session has no board row.
    """
    label = str(row.get("label") or "").strip()
    sid = str(row.get("session_id") or "")
    row_id = str(row.get("id") or "")
    if not label or not sid or not row_id:
        return False
    if str(row.get("archived_at") or "").strip():
        return False
    if str(gateway_title or "").strip() == label and row.get("label_pin_confirmed"):
        return False
    now = int(now_ms if now_ms is not None else time.time() * 1000)
    ok = False
    try:
        resp = _gateway().gateway_request(
            "POST",
            f"/control/v1/orchestrator/sessions/{sid}/title",
            json={"title": label, "lock": True},
            timeout=PIN_TIMEOUT_S,
            retry_unavailable=False,
        )
        payload = resp.json() if resp is not None and resp.status_code == 200 else {}
        ok = isinstance(payload, dict) and payload.get("ok") is True
    except Exception:  # noqa: BLE001 — the next poll retries after the throttle
        pass
    try:
        registry.set_label_pinned(row_id, _now_iso(now), confirmed=ok)
    except Exception:  # noqa: BLE001
        pass
    if ok:
        try:
            from sharedagents_web.server import store

            store.update_fields(sid, title=label)
        except Exception:  # noqa: BLE001 — the gateway title is the source of truth
            pass
    return ok


# --- batching / scheduling ----------------------------------------------


def process_pairs(
    pairs: list[tuple[dict, dict]],
    *,
    registry: Any,
    now_ms: int | None = None,
) -> list[dict]:
    """Pin titles + run due recaps for ``(row, item)`` pairs, in order.

    Each step is independently guarded, so one failing row never starves the
    rest of the batch. Returns the parsed recap replies actually produced.
    """
    out: list[dict] = []
    for row, item in pairs or []:
        try:
            if pin_due(row, gateway_title=str(item.get("session_node_title") or ""), now_ms=now_ms):
                push_pin(
                    row,
                    gateway_title=str(item.get("session_node_title") or ""),
                    registry=registry,
                    now_ms=now_ms,
                )
        except Exception:  # noqa: BLE001
            pass
        try:
            if recap_due(
                row,
                card=str(item.get("card") or ""),
                parent_turn_stamp=turn_stamp_ms(item.get("last_turn_ts")),
                now_ms=now_ms,
            ):
                reply = run_recap(row, item, registry=registry, now_ms=now_ms)
                if reply:
                    out.append(reply)
        except Exception:  # noqa: BLE001
            pass
    return out


def _spawn_thread(pairs: list[tuple[dict, dict]], registry: Any) -> None:
    # A slow batch can outlast many polls. Keep at most one batch worker per
    # process; skipped snapshots are recomputed on the next poll/sweep.
    if os.environ.get("PYTEST_CURRENT_TEST") or not _batch_lock.acquire(blocking=False):
        return

    def _work() -> None:
        try:
            current = pairs
            if _pairs_provider is not None:
                wanted = {row["id"] for row, _ in pairs}
                current = [(row, item) for row, item in _pairs_provider() if row["id"] in wanted]
                observed = registry.observe_cards(
                    [
                        (
                            row["id"],
                            str(item.get("completion_card") or item.get("card") or ""),
                            turn_stamp_ms(item.get("last_turn_ts")),
                        )
                        for row, item in current
                    ]
                )
                current = [(observed.get(row["id"], row), item) for row, item in current]
            process_pairs(current, registry=registry)
        finally:
            _batch_lock.release()

    try:
        threading.Thread(target=_work, name="tasks-recap", daemon=True).start()
    except BaseException:
        _batch_lock.release()
        raise


def request_recaps(
    pairs: list[tuple[dict, dict]],
    *,
    registry: Any | None = None,
    spawn: Callable[[list[tuple[dict, dict]], Any], None] | None = None,
) -> int:
    """Fire-and-forget the work for ``pairs``. Returns how many looked due.

    Called from ``api_tasks_list`` AFTER the response payload is built, so a
    recap never adds latency to the 5 s poll. ``spawn`` is injectable so tests
    run the batch inline instead of on a thread.
    """
    if not pairs:
        return 0
    reg = registry
    if reg is None:
        from sharedagents_web.task_board import get_task_board_registry

        reg = get_task_board_registry()
    observed = reg.observe_cards(
        [
            (
                str(row["id"]),
                str(item.get("completion_card") or item.get("card") or ""),
                turn_stamp_ms(item.get("last_turn_ts")),
            )
            for row, item in pairs
        ]
    )
    pairs = [(observed.get(row["id"], row), item) for row, item in pairs]
    now_ms = int(time.time() * 1000)
    due = [
        (row, item)
        for row, item in pairs
        if recap_due(
            row,
            card=str(item.get("card") or ""),
            parent_turn_stamp=turn_stamp_ms(item.get("last_turn_ts")),
            now_ms=now_ms,
        )
        or pin_due(
            row,
            gateway_title=str(item.get("session_node_title") or ""),
            now_ms=now_ms,
        )
    ]
    if not due:
        return 0
    (spawn or _spawn_thread)(due, reg)
    return len(due)


# --- server-side sweeper -------------------------------------------------


def register_pairs_provider(provider: Callable[[], list[tuple[dict, dict]]]) -> None:
    """The tasks blueprint supplies "every tracked row + its derived card".

    Registered at blueprint import so the sweeper reuses the board's own
    enrichment (``routes/tasks.py::_enriched_pairs``) instead of deriving
    state a second way.
    """
    global _pairs_provider
    _pairs_provider = provider


def sweep_once() -> int:
    """One sweeper pass: recompute cards and fire whatever looks due."""
    provider = _pairs_provider
    if provider is None:
        return 0
    try:
        pairs = provider() or []
    except Exception:  # noqa: BLE001 — a sweep failure is never fatal
        return 0
    return request_recaps(pairs)


def ensure_sweeper(period_s: int = SWEEP_PERIOD_S) -> bool:
    """Start the daemon sweeper once per process. True if it is running.

    Without this, recaps would only happen while the operator has the Tasks
    tab open — but "every task that switched to Completed" is the operator's
    ask. Persisted completion edges survive restarts; previously idle rows
    without an observed transition are only a baseline, never a backfill.
    """
    global _sweeper_started, _sweeper_thread, _sweeper_stop
    if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get(
        "SHAREDAGENTS_TASKS_RECAP_SWEEPER", ""
    ).strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return False
    with _sweeper_lock:
        if _sweeper_started:
            return True
        stop = threading.Event()

        def _loop() -> None:
            while not stop.wait(max(5, int(period_s))):
                try:
                    sweep_once()
                except Exception:  # noqa: BLE001
                    pass

        thread = threading.Thread(target=_loop, name="tasks-recap-sweeper", daemon=True)
        thread.start()
        _sweeper_stop, _sweeper_thread = stop, thread
        _sweeper_started = True
    return True


def reset_sweeper_for_tests() -> None:
    """Test hook: drop the sweeper so a fresh SHAREDAGENTS_HOME re-arms it."""
    global _sweeper_started, _sweeper_thread, _pairs_provider, _sweeper_stop
    with _sweeper_lock:
        if _sweeper_stop is not None:
            _sweeper_stop.set()
        if _sweeper_thread is not None:
            _sweeper_thread.join(timeout=1)
        _sweeper_started = False
        _sweeper_thread = None
        _sweeper_stop = None
    _pairs_provider = None


def backoff_until(row: dict, now_ms: int | None = None) -> str:
    """Human-readable next-attempt stamp for tests/telemetry (never shown)."""
    now = int(now_ms if now_ms is not None else time.time() * 1000)
    try:
        failures = int(row.get("summary_failures") or 0)
    except (TypeError, ValueError):
        failures = 0
    wait = FAILURE_BACKOFF_S if failures >= FAILURE_BACKOFF_THRESHOLD else MIN_INTERVAL_S
    return (datetime.fromtimestamp(now / 1000.0, tz=timezone.utc) + timedelta(seconds=wait)).isoformat()


__all__ = [
    "FAILURE_BACKOFF_S",
    "FAILURE_BACKOFF_THRESHOLD",
    "MIN_INTERVAL_S",
    "MIN_QUIET_S",
    "PIN_RETRY_S",
    "RECAP_ITEM_FIELDS",
    "STALE_RUNNING_S",
    "SWEEP_PERIOD_S",
    "backoff_until",
    "ensure_sweeper",
    "pin_due",
    "process_pairs",
    "push_pin",
    "recap_archivable",
    "recap_card",
    "recap_due",
    "register_pairs_provider",
    "request_recaps",
    "reset_sweeper_for_tests",
    "run_recap",
    "sweep_once",
    "summary_fields",
    "turn_stamp_ms",
]
