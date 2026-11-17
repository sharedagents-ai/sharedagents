"""Loop supervisor — fires scheduled turns into bound sessions.

Hosted on :class:`Agent` (lazy ``agent.loop_supervisor`` accessor,
mirroring ``GoalSupervisor``). One asyncio drive task per loop (or cron
job, keyed ``job:<id>``): it claims the durable cross-process lease,
sleeps until ``next_run_at`` (renewing the lease in chunks), fires one
``run_turn(..., source="loop_supervisor")`` into the loop's bound
session, then records the run and re-arms.

Behavior mirrors (clean-room) the reference loop runtime semantics:

* idle-only firing — a fire whose session has a turn in flight is
  skipped: interval loops push ``next_run_at`` one interval (never
  queue, no catch-up); model-paced wakeups wait for idle (bounded) and
  then fall back to the keepalive;
* keepalive — a model-paced loop whose turn ended without re-arming
  gets a ``KEEPALIVE_FALLBACK_S`` (1200 s) fallback wakeup; the second
  consecutive missed reschedule ends the loop (``model_no_reschedule``);
* error backstop — 3 consecutive failed turns pause the loop;
* bounded crash re-drive — a drive task that RAISED is re-spawned in
  process up to ``loop_supervisor_max_redrive`` times (default 5) while
  the loop is still ``running``, mirroring the goals outer-loop rule.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
import uuid
from typing import Any

from sharedagents_loop.loop_registry import (
    _LEASE_RENEW_INTERVAL_S,
    STATUS_RUNNING,
    _iso_to_epoch,
    _now_iso,
    get_loop_registry,
)

# Imported at module level (not at fire time): a call-time import under a
# test's monkeypatched ``get_loop_registry`` would permanently poison the
# service module's captured binding for every later test in the process.
from sharedagents_loop.loop_service import (
    build_loop_prompt,
    cron_disabled,
    load_loop_md,
    prepare_loop_fire,
)

try:
    from sharedagents_loop.loop_registry import KEEPALIVE_FALLBACK_S
except ImportError:  # pragma: no cover
    KEEPALIVE_FALLBACK_S = 1200.0

#: How long the model-paced busy-wait polls the session for idle before
#: re-arming the keepalive fallback instead.
_IDLE_WAIT_CHUNK_S = 5.0

#: Bounded in-process re-drive of a loop whose drive task CRASHED
#: (raised) mid-run — mirror of the goals ``goal_supervisor_max_redrive``
#: rule. Bounded so a deterministically-crashing loop
#: cannot hot-loop; after the bound it is left to the operator
#: (pause/resume) rather than hammered forever.
_DEFAULT_MAX_REDRIVE = 5

#: Backstop + retry defaults (override via ``loop_supervisor_max_errors``
#: and ``loop_supervisor_error_delay_s`` config fields).
_DEFAULT_MAX_ERRORS = 3
_DEFAULT_ERROR_DELAY_S = 10.0

# Standby supervisors keep checking until the winner releases or its lease
# ages out. A module constant keeps the production cadence explicit and lets
# deterministic tests shorten it without sleeping for 25 seconds.
_CLAIM_RETRY_BACKOFF_S = _LEASE_RENEW_INTERVAL_S + 5.0

_LOG = logging.getLogger("sharedagents.loop_supervisor")

_OWNER_ID = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


class _LeaseLost(RuntimeError):
    """Internal control flow: this process no longer owns the drive lease."""


def schedule_loop_recovery(agent: Any) -> None:
    """Schedule durable-loop recovery once when an event loop is available."""
    try:
        if getattr(agent, "_loops_recovery_scheduled", False):
            return
        agent._loops_recovery_scheduled = True
    except Exception:  # noqa: BLE001 — tolerate adapter/test doubles
        return
    try:
        event_loop = asyncio.get_running_loop()
    except RuntimeError:
        # Agent construction commonly precedes the gateway event loop. Let the
        # first live prompt/backend access retry instead of losing recovery.
        agent._loops_recovery_scheduled = False
        return
    recovery = None
    try:
        recovery = agent.loop_supervisor.recover_running()
        task = event_loop.create_task(
            recovery,
            name=f"loop_recovery_{agent.name}",
        )

        def _recovery_done(done: asyncio.Task) -> None:
            if done.cancelled():
                failed = True
                error: BaseException | None = None
            else:
                try:
                    error = done.exception()
                except Exception as exc:  # noqa: BLE001
                    error = exc
                failed = error is not None
            if not failed:
                return
            try:
                agent._loops_recovery_scheduled = False
            except Exception:  # noqa: BLE001
                pass
            if error is not None:
                _LOG.warning("loop recovery failed; will retry: %s", error)

        task.add_done_callback(_recovery_done)
    except Exception:  # noqa: BLE001 — recovery must never block startup
        close = getattr(recovery, "close", None)
        if callable(close):
            close()
        try:
            agent._loops_recovery_scheduled = False
        except Exception:  # noqa: BLE001
            pass


class LoopSupervisor:
    """Per-agent supervisor for scheduled loops and cron jobs."""

    def __init__(self, agent: Any) -> None:
        self._agent = agent
        self._tasks: dict[str, asyncio.Task] = {}
        self._claim_retry_tasks: dict[str, asyncio.Task] = {}
        # The engine lock serialises turns but would queue a second loop. This
        # reservation makes the supervisor's idle-only decision atomic within
        # the process, so simultaneous due loops for one session cannot queue.
        self._driving_sessions: set[str] = set()
        # Per-key count of in-process re-drives after a drive-task crash,
        # reset on any clean (non-crash) exit so it bounds crash-loops only.
        self._redrive_counts: dict[str, int] = {}
        self._owner_id = _OWNER_ID

    @property
    def cfg(self) -> Any:
        return self._agent.cfg

    def _error_delay_s(self) -> float:
        return float(
            getattr(self.cfg, "loop_supervisor_error_delay_s", _DEFAULT_ERROR_DELAY_S) or _DEFAULT_ERROR_DELAY_S,
        )

    def _max_errors(self) -> int:
        return int(
            getattr(self.cfg, "loop_supervisor_max_errors", _DEFAULT_MAX_ERRORS) or _DEFAULT_MAX_ERRORS,
        )

    def _max_redrive(self) -> int:
        return int(
            getattr(self.cfg, "loop_supervisor_max_redrive", _DEFAULT_MAX_REDRIVE) or _DEFAULT_MAX_REDRIVE,
        )

    # -- key-aware lease helpers (loops vs cron jobs) ----------------------

    @staticmethod
    def _is_job_key(key: str) -> bool:
        return str(key).startswith("job:")

    def _claim(self, key: str) -> bool:
        try:
            reg = get_loop_registry()
            if self._is_job_key(key):
                return reg.try_claim_job(key[4:], self._owner_id)
            return reg.try_claim_supervisor(key, self._owner_id)
        except Exception:  # noqa: BLE001
            return False

    def _touch(self, key: str) -> bool:
        try:
            reg = get_loop_registry()
            if self._is_job_key(key):
                return reg.touch_job(key[4:], self._owner_id)
            return reg.touch_supervisor(key, self._owner_id)
        except Exception:  # noqa: BLE001
            return False

    def _release(self, key: str) -> None:
        try:
            reg = get_loop_registry()
            if self._is_job_key(key):
                reg.release_job(key[4:], self._owner_id)
            else:
                reg.release_supervisor(key, self._owner_id)
        except Exception:  # noqa: BLE001
            pass

    def _session_lease_owner(self, key: str) -> str:
        return f"{self._owner_id}:{key}"

    def _claim_session_turn(self, key: str, sid: str) -> bool:
        try:
            return get_loop_registry().try_claim_session_turn(
                sid,
                self._session_lease_owner(key),
            )
        except Exception:  # noqa: BLE001
            return False

    def _touch_session_turn(self, key: str, sid: str) -> bool:
        try:
            return get_loop_registry().touch_session_turn(
                sid,
                self._session_lease_owner(key),
            )
        except Exception:  # noqa: BLE001
            return False

    def _release_session_turn(self, key: str, sid: str) -> None:
        try:
            get_loop_registry().release_session_turn(
                sid,
                self._session_lease_owner(key),
            )
        except Exception:  # noqa: BLE001
            pass

    # -- task bookkeeping ---------------------------------------------------

    def is_active(self, key: str) -> bool:
        task = self._tasks.get(key)
        if task is None:
            return False
        if task.done():
            self._tasks.pop(key, None)
            return False
        cancelling = getattr(task, "cancelling", None)
        return not (callable(cancelling) and cancelling() > 0)

    def _schedule_claim_retry(self, key: str, *, source: str) -> None:
        if key in self._claim_retry_tasks:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        async def _retry() -> None:
            try:
                while True:
                    await asyncio.sleep(_CLAIM_RETRY_BACKOFF_S)
                    if cron_disabled() or not self._key_is_running(key):
                        return
                    if self.is_active(key):
                        return
                    result = await self.supervise_loop(
                        key,
                        source=f"{source}:lease_retry",
                    )
                    if bool(result.get("ok")):
                        return
            finally:
                self._claim_retry_tasks.pop(key, None)

        self._claim_retry_tasks[key] = loop.create_task(
            _retry(),
            name=f"loop_claim_retry_{str(key)[:16]}",
        )

    async def _sleep_with_lease(self, key: str, delay_s: float) -> bool:
        """Sleep in chunks, renewing the lease so long delays stay owned.

        The deadline is a monotonic instant for the requested duration.
        A slice that returns late because the event loop was busy is
        absorbed into the time still left. Subtracting the planned slice
        length instead stacks every overrun, which walks a daily cron
        minutes past its wall-clock time.
        """
        deadline = time.monotonic() + max(0.0, float(delay_s))
        while True:
            if deadline - time.monotonic() <= 0:
                break
            if cron_disabled() or not self._touch(key):
                return False
            # Renewal is synchronous and can itself run long. Measure
            # what is left after it returns, or the slice keeps a stale
            # length and the fire slips by that stall.
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            chunk = min(_LEASE_RENEW_INTERVAL_S, remaining)
            await asyncio.sleep(chunk)
        return not cron_disabled() and self._touch(key)

    async def _lease_renewal_loop(self, key: str, sid: str) -> bool:
        while True:
            await asyncio.sleep(_LEASE_RENEW_INTERVAL_S)
            if cron_disabled() or not self._touch(key) or not self._touch_session_turn(key, sid):
                return False

    def _key_is_running(self, key: str) -> bool:
        try:
            reg = get_loop_registry()
            if self._is_job_key(key):
                job = reg.get_job(key[4:])
                return job is not None and str(job.get("status", "")) == STATUS_RUNNING
            rec = reg.get(key)
            return rec is not None and rec.status == STATUS_RUNNING
        except Exception:  # noqa: BLE001
            return False

    # -- public entry points ------------------------------------------------

    async def supervise_loop(self, loop_id: str, *, source: str = "service") -> dict[str, bool]:
        """Idempotent kick: claim the durable lease for ``loop_id`` (or
        ``job:<id>``) and start its drive task if not already active. On a
        lost-claim race, schedule a bounded retry instead of double-driving.

        Returns ``{"ok": bool}`` — the control endpoint's contract.
        """
        if not loop_id:
            return {"ok": False}
        if cron_disabled():
            if self.is_active(loop_id):
                self.cancel(loop_id, pause=False)
            return {"ok": False}
        if self.is_active(loop_id):
            return {"ok": True}
        stale = self._tasks.get(loop_id)
        if stale is not None and not stale.done():
            stale.cancel()
        reg = get_loop_registry()
        agent_name = str(getattr(self._agent, "name", "") or "")
        if self._is_job_key(loop_id):
            job = reg.get_job(loop_id[4:])
            if (
                job is None
                or str(job.get("status", "")) != STATUS_RUNNING
                or (agent_name and str(job.get("agent_name", "")) != agent_name)
            ):
                return {"ok": False}
        else:
            rec = reg.get(loop_id)
            if rec is None or rec.status != STATUS_RUNNING or (agent_name and rec.agent_name != agent_name):
                return {"ok": False}
        if not self._claim(loop_id):
            self._schedule_claim_retry(loop_id, source=source)
            return {"ok": False}
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._release(loop_id)
            return {"ok": False}
        task = loop.create_task(
            self._drive_loop(loop_id, source=source),
            name=f"loop_supervisor_{str(loop_id)[:20]}",
        )
        self._tasks[loop_id] = task
        # Bounded crash resilience: if the drive task RAISES mid-run while
        # the loop is still ``running``, re-spawn in process (mirror of the
        # goals outer-loop rule). A clean exit never re-drives.
        task.add_done_callback(
            lambda t, key=loop_id: self._on_drive_done(key, t),
        )
        return {"ok": True}

    async def recover_running(self) -> dict[str, int]:
        """Re-arm this agent's durable loops/jobs after process restart."""
        reg = get_loop_registry()
        if cron_disabled():
            return {"started": 0, "expired": 0}

        agent_name = str(getattr(self._agent, "name", "") or "")
        started = 0

        def stopped(sid: str) -> bool:
            # A crash may persist the Stop before its job/loop pause. The
            # session log remains authoritative when recovering that window.
            from sharedagents_loop.loop_service import stop_loops_for_session
            from sharedagents_session.operator_stop import operator_stopped

            sdir = getattr(self._agent, "sessions_dir", None)
            if sdir is not None and operator_stopped(sdir, sid):
                stop_loops_for_session(self._agent, sid)
                return True
            return False

        for rec in reg.list_loops():
            if rec.status != STATUS_RUNNING or rec.agent_name != agent_name:
                continue
            if stopped(rec.session_id):
                continue
            result = await self.supervise_loop(rec.loop_id, source="recovery")
            started += int(bool(result.get("ok")))
        for job in reg.list_jobs():
            if str(job.get("agent_name") or "") != agent_name:
                continue
            if not bool(job.get("durable")):
                continue
            if stopped(str(job.get("session_id") or "")):
                continue
            key = f"job:{job['job_id']}"
            result = await self.supervise_loop(key, source="recovery")
            started += int(bool(result.get("ok")))
        return {"started": started, "expired": 0}

    def cancel(
        self,
        key: str,
        *,
        reason: str = "cancelled",
        pause: bool = True,
        release_lease: bool = True,
    ) -> None:
        """Shared cancel chokepoint: cancels the drive + retry tasks,
        releases the lease, and (for loops) pauses the running record so
        no supervisor re-drives it. Job status changes belong to the service;
        here only the task and lease are torn down.
        """
        task = self._tasks.get(key)
        if task is not None and not task.done():
            task.cancel()
        retry = self._claim_retry_tasks.pop(key, None)
        if retry is not None and not retry.done():
            retry.cancel()
        if release_lease:
            self._release(key)
        if pause and not self._is_job_key(key):
            try:
                reg = get_loop_registry()
                rec = reg.get(key)
                if rec is not None and rec.status == STATUS_RUNNING:
                    reg.pause_loop(key, reason=reason)
            except Exception:  # noqa: BLE001
                pass

    async def loop_action(self, loop_id: str, action: str) -> dict[str, Any]:
        """Lifecycle action entry (pause/resume/archive) — the control
        endpoint / web action path.
        """
        reg = get_loop_registry()
        if action == "pause":
            result = reg.pause_loop(loop_id, reason="user")
            self.cancel(loop_id, reason="paused")
        elif action == "resume":
            if cron_disabled():
                from sharedagents_loop.loop_service import cron_disabled_message

                rec = reg.get(loop_id)
                return {
                    "ok": False,
                    "message": cron_disabled_message(),
                    "loop": rec.to_dict() if rec else None,
                }
            result = reg.resume_loop(loop_id)
            if result.ok:
                await self.supervise_loop(loop_id, source="action")
        elif action == "archive":
            result = reg.archive_loop(loop_id, reason="user")
            self.cancel(loop_id, reason="archived", pause=False)
        elif action == "delete":
            self.cancel(loop_id, reason="deleted", pause=False)
            if not reg.delete_loop(loop_id):
                return {
                    "ok": False,
                    "message": f"loop not found: {loop_id}",
                    "loop": None,
                }
            return {"ok": True, "message": "deleted", "loop": None}
        else:
            return {"ok": False, "message": f"unknown loop action: {action}"}
        return {
            "ok": result.ok,
            "message": result.message,
            "loop": result.loop.to_dict() if result.loop else None,
        }

    # -- drive tasks --------------------------------------------------------

    async def _drive_loop(self, key: str, *, source: str) -> None:
        """Drive one scheduled key until it leaves ``running``."""
        if self._is_job_key(key):
            await self._drive_one_job(key[4:])
        else:
            await self._drive_one_loop(key)

    async def _drive_one_loop(self, loop_id: str) -> None:
        """Drive one loop: read record → exit if not ``running`` → sleep
        until ``next_run_at`` (chunked, cancels on pause) → skip if a turn
        is in flight → fire the loop's prompt via ``run_turn`` → record the
        run → keepalive/backstop bookkeeping.
        """
        reg = get_loop_registry()
        me = asyncio.current_task()
        consecutive_errors = 0
        try:
            while True:
                if cron_disabled():
                    return
                rec = reg.get(loop_id)
                if rec is None or rec.status != STATUS_RUNNING:
                    return
                sid = rec.session_id
                if not sid:
                    return
                now_epoch = time.time()
                next_at = _iso_to_epoch(rec.next_run_at) if rec.next_run_at else 0.0
                delay = next_at - now_epoch
                if delay > 0:
                    if not await self._sleep_with_lease(loop_id, delay):
                        return
                    continue  # re-evaluate status at the top

                # Due now.
                model_paced = rec.interval_s is None and not rec.cron
                if model_paced and int(rec.misses or 0) >= 2:
                    # 2nd missed reschedule (keepalive budget exhausted).
                    reg.end_model_no_reschedule(
                        loop_id,
                        reason="model_no_reschedule",
                    )
                    return

                # A schedule_wakeup row is bookkeeping for this due instant;
                # consume it even if the fire is subsequently skipped as busy.
                reg.fire_due_wakeups_for_loop(loop_id, _now_iso())

                fire_sid, prep_err = prepare_loop_fire(self._agent, rec)
                if prep_err:
                    started_at = _now_iso()
                    finished_at = started_at
                    if not self._touch(loop_id):
                        return
                    reg.record_run(
                        loop_id,
                        started_at=started_at,
                        finished_at=finished_at,
                        ok=False,
                        error=prep_err,
                        advance_schedule=False,
                    )
                    consecutive_errors = reg.consecutive_failures(
                        loop_id,
                        limit=self._max_errors(),
                    )
                    if consecutive_errors >= self._max_errors():
                        reg.pause_loop(loop_id, reason=prep_err)
                        return
                    if not await self._sleep_with_lease(
                        loop_id,
                        self._error_delay_s(),
                    ):
                        return
                    continue
                sid = fire_sid

                session_claimed = False
                if not self._session_busy(sid):
                    session_claimed = self._claim_session_turn(loop_id, sid)
                if not session_claimed:
                    if not model_paced:
                        # Overlap policy: skip this fire, push one interval,
                        # count the miss — never queue.
                        reg.mark_fire_skipped(loop_id)
                        continue
                    else:
                        # Model-paced wakeup: wait for both local idleness and
                        # the cross-process session reservation (bounded).
                        session_claimed = await self._wait_for_session_turn(
                            loop_id,
                            sid,
                            KEEPALIVE_FALLBACK_S,
                        )
                        if not session_claimed:
                            reg.arm_keepalive(loop_id, count_miss=False)
                            continue

                if not self._touch(loop_id):
                    self._release_session_turn(loop_id, sid)
                    return
                self._driving_sessions.add(sid)
                first_fire = int(rec.run_count or 0) == 0
                prompt = rec.prompt_text
                try:
                    prompt = build_loop_prompt(
                        rec,
                        loop_md=load_loop_md(),
                        first_fire=first_fire,
                    )
                except Exception:  # noqa: BLE001 — never block a fire
                    prompt = rec.prompt_text

                started_at = _now_iso()
                try:
                    ok, error = await self._run_supervised_turn(
                        loop_id,
                        sid,
                        prompt,
                    )
                except asyncio.CancelledError:
                    raise
                except _LeaseLost:
                    return
                except Exception as exc:  # noqa: BLE001
                    _LOG.warning(
                        "loop supervisor turn failed for %s: %s",
                        loop_id,
                        exc,
                    )
                    ok, error = False, str(exc)
                finally:
                    self._driving_sessions.discard(sid)
                    self._release_session_turn(loop_id, sid)
                finished_at = _now_iso()

                # A turn may finish between lease-renewal ticks. Re-check
                # ownership before mutating durable schedule anchors.
                if not self._touch(loop_id):
                    return

                if ok:
                    consecutive_errors = 0
                    reg.record_run(
                        loop_id,
                        started_at=started_at,
                        finished_at=finished_at,
                        ok=True,
                    )
                    rec = reg.get(loop_id)
                    if rec is not None and rec.status == STATUS_RUNNING and rec.interval_s is None and not rec.cron:
                        # Keepalive: the turn ended without the model
                        # re-arming a wakeup → 1200 s fallback; the second
                        # consecutive missed reschedule ends the loop.
                        stale = not rec.next_run_at or _iso_to_epoch(rec.next_run_at) <= _iso_to_epoch(finished_at)
                        if stale:
                            if int(rec.misses or 0) >= 1:
                                reg.end_model_no_reschedule(
                                    loop_id,
                                    reason="model_no_reschedule",
                                )
                                return
                            reg.arm_keepalive(loop_id)
                    continue

                # Failed turn: count toward the backstop.
                reg.record_run(
                    loop_id,
                    started_at=started_at,
                    finished_at=finished_at,
                    ok=False,
                    error=error or "turn_errors",
                    advance_schedule=False,
                )
                consecutive_errors = reg.consecutive_failures(
                    loop_id,
                    limit=self._max_errors(),
                )
                if consecutive_errors >= self._max_errors():
                    reg.pause_loop(loop_id, reason=error or "turn_errors")
                    return
                if not await self._sleep_with_lease(
                    loop_id,
                    self._error_delay_s(),
                ):
                    return
        except asyncio.CancelledError:
            pass
        finally:
            if self._tasks.get(loop_id) is me:
                self._tasks.pop(loop_id, None)
                try:
                    reg.release_supervisor(loop_id, self._owner_id)
                except Exception:  # noqa: BLE001
                    pass

    async def _drive_one_job(self, job_id: str) -> None:
        """Drive one cron job: sleep until ``next_fire_at``, fire the job's
        prompt into its session, then advance the fire anchors. A fire
        whose session is busy is skipped and rescheduled from now (no
        catch-up); a one-shot job is removed after firing."""
        reg = get_loop_registry()
        me = asyncio.current_task()
        key = f"job:{job_id}"
        try:
            while True:
                if cron_disabled():
                    return
                job = reg.get_job(job_id)
                if job is None or str(job.get("status", "")) != STATUS_RUNNING:
                    return
                now_epoch = time.time()
                next_at = _iso_to_epoch(job.get("next_fire_at")) if job.get("next_fire_at") else 0.0
                delay = next_at - now_epoch
                if delay > 0:
                    if not await self._sleep_with_lease(key, delay):
                        return
                    continue
                sid = str(job.get("session_id", ""))
                session_claimed = False
                if not self._session_busy(sid):
                    session_claimed = self._claim_session_turn(key, sid)
                if not session_claimed:
                    # Missed while busy: reschedule from now, never queue.
                    reg.record_job_fire(job_id, fired_at=_now_iso())
                    continue
                if not self._touch(key):
                    self._release_session_turn(key, sid)
                    return
                self._driving_sessions.add(sid)
                try:
                    ok, _err = await self._run_supervised_turn(
                        key,
                        sid,
                        str(job.get("prompt", "")),
                    )
                except asyncio.CancelledError:
                    raise
                except _LeaseLost:
                    return
                except Exception as exc:  # noqa: BLE001
                    _LOG.warning("loop supervisor job turn failed: %s", exc)
                    ok = False
                finally:
                    self._driving_sessions.discard(sid)
                    self._release_session_turn(key, sid)
                if not self._touch(key):
                    return
                reg.record_job_fire(job_id)
                if not ok:
                    continue  # no job backstop — the next cron match governs
        except asyncio.CancelledError:
            pass
        finally:
            if self._tasks.get(key) is me:
                self._tasks.pop(key, None)
                try:
                    reg.release_job(job_id, self._owner_id)
                except Exception:  # noqa: BLE001
                    pass

    async def _run_supervised_turn(
        self,
        key: str,
        sid: str,
        prompt: str,
    ) -> tuple[bool, str]:
        """Run one agent turn, renewing the lease while it is in flight."""

        async def _consume_turn() -> tuple[bool, str]:
            from sharedagents_agent.agents.engine import TurnDone

            state = ""
            error = ""
            async for ev in self._agent.run_turn(
                sid,
                prompt,
                source="loop_supervisor",
            ):
                if isinstance(ev, TurnDone):
                    state = ev.state or "done"
                    error = str(getattr(ev, "error", "") or "")
            if state == "ok":
                return True, ""
            if not state:
                return False, "turn_missing_terminal"
            return False, error or f"turn_{state}"

        turn = asyncio.create_task(
            _consume_turn(),
            name=f"loop_turn_{str(key)[:12]}",
        )
        renew = asyncio.create_task(
            self._lease_renewal_loop(key, sid),
            name=f"loop_lease_{str(key)[:12]}",
        )
        try:
            done, _pending = await asyncio.wait(
                (turn, renew),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if turn in done:
                return await turn
            if not await renew:
                turn.cancel()
                await asyncio.gather(turn, return_exceptions=True)
                raise _LeaseLost(key)
            return await turn
        finally:
            for task in (turn, renew):
                if not task.done():
                    task.cancel()
            await asyncio.gather(turn, renew, return_exceptions=True)

    # -- idle-only guard -----------------------------------------------------

    def _session_busy(self, sid: str) -> bool:
        """True when a turn is in flight on ``sid`` (idle-only firing).

        Sources, in order: an explicit agent-side ``active_turn_id(sid)`` /
        ``has_active_turn()`` reporter (gateway-shim family), then the
        engine's per-session lock — authoritative in-process truth while
        :meth:`run_turn` streams.
        """
        if sid in self._driving_sessions:
            return True
        for attr in ("active_turn_id", "has_active_turn"):
            fn = getattr(self._agent, attr, None)
            if not callable(fn):
                continue
            try:
                try:
                    busy = bool(fn(sid))
                except TypeError:
                    busy = bool(fn())
            except Exception:  # noqa: BLE001
                busy = False
            if busy:
                return True
        try:
            engine = getattr(self._agent, "engine", None)
            locks = getattr(engine, "_session_locks", None) if engine else None
            if locks:
                lock = locks.get(sid)
                if lock is not None and lock.locked():
                    return True
        except Exception:  # noqa: BLE001
            pass
        return False

    async def _wait_for_session_turn(
        self,
        key: str,
        sid: str,
        timeout_s: float,
    ) -> bool:
        """Bounded wait for local idleness plus the shared session lease."""
        waited = 0.0
        while waited < timeout_s:
            if not self._session_busy(sid) and self._claim_session_turn(key, sid):
                return True
            chunk = min(_IDLE_WAIT_CHUNK_S, timeout_s - waited)
            if not await self._sleep_with_lease(key, chunk):
                return False
            waited += chunk
        return False

    # -- crash re-drive -------------------------------------------------------

    def _on_drive_done(self, key: str, task: asyncio.Task) -> None:
        """Drive-task completion callback: re-drives ONLY on an abnormal
        crash (the task raised a non-``CancelledError`` exception) while
        the record is still ``running`` and the per-key re-drive bound is
        not exhausted. A clean exit is a deliberate stop and never
        re-drives; it resets the counter.
        """
        if task.cancelled():
            return
        try:
            exc = task.exception()
        except Exception:  # noqa: BLE001 — cancelled/loop-closing
            return
        if exc is None:
            self._redrive_counts.pop(key, None)
            return
        n = self._redrive_counts.get(key, 0)
        if n >= self._max_redrive():
            _LOG.error(
                "loop supervisor drive crashed for %s and exhausted %d re-drives; leaving to operator pause/resume: %r",
                key,
                n,
                exc,
            )
            return
        try:
            reg = get_loop_registry()
            if self._is_job_key(key):
                job = reg.get_job(key[4:])
                running = job is not None and str(job.get("status", "")) == STATUS_RUNNING
            else:
                rec = reg.get(key)
                running = rec is not None and rec.status == STATUS_RUNNING
        except Exception:  # noqa: BLE001
            return
        if not running:
            self._redrive_counts.pop(key, None)
            return
        self._redrive_counts[key] = n + 1
        _LOG.warning(
            "loop supervisor drive crashed for %s (re-drive %d/%d): %r",
            key,
            n + 1,
            self._max_redrive(),
            exc,
        )
        try:
            asyncio.ensure_future(self.supervise_loop(key, source="redrive"))
        except Exception:  # noqa: BLE001 — operator pause/resume remain
            _LOG.exception("loop supervisor re-drive failed for %s", key)
