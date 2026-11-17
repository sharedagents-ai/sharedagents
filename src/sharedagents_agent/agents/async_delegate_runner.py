"""Detached runner for ONE background delegation.

Spawned by the delegate shim with ``start_new_session=True`` so it outlives
the parent turn (and even a gateway restart). Runs the child turn via the SAME
in-process driver the synchronous path uses, then appends a ``subagent_result``
event to the PARENT session log (under the unified ``sessions_root()``) and
finalizes the job record, then submits one consolidated parent prompt only
when its whole delegation wave is terminal.

Invoked as ``python -m sharedagents_agent.agents.async_delegate_runner`` with a JSON
job spec on argv[1] or stdin:
    {"job_id","parent_sid","agent","task","call_id"}
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import sys
from typing import Any

from . import async_jobs
from .delegate_shims import _drive_child_agent_turn

_BACKGROUND_JOB_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "sharedagents_background_job_id",
    default="",
)


async def run_job(spec: dict[str, Any]) -> None:
    """Drive one background delegation's child turn in this detached process.

    Claims the job, drives the child turn in-process via
    :func:`_drive_child_agent_turn`, then durably emits the ``subagent_result``
    event to the parent log BEFORE finalizing the job record.

    The gateway pre-reserves this delegation's whole subtree budget before
    spawning us (spec §6 / addendum) and hands the grant down via env, so the
    child broker :func:`get_broker` here initialises from that grant rather than
    a fresh empty one — closing the cross-process cap escape. The top turn itself is pre-counted (``_sharedagents_precounted``); only its
    nested spawns draw from the grant.
    """
    job_id = spec["job_id"]
    parent_sid = spec["parent_sid"]
    agent = spec["agent"]
    # The detached process has no SHAREDAGENTS_SESSION_ID env, so the child links to
    # the parent via this arg; child_sid comes back on the same dict.
    args: dict[str, Any] = {
        "name": agent,
        "task": spec.get("task", ""),
        "_sharedagents_parent_session_id": parent_sid,
    }
    # This detached top turn was already counted by the gateway's pre-spawn
    # reservation (spec §6.3 / addendum); mark it so ``_drive_child_agent_turn``
    # does not re-reserve it. The child broker (initialised from the env grant)
    # still enforces the subtree cap on any nested spawns.
    if spec.get("_precounted"):
        args["_sharedagents_precounted"] = True
    if spec.get("call_id"):
        args["_sharedagents_call_id"] = spec["call_id"]
    # Execution options (cwd / model / mode) the submitter captured, so the
    # detached child honors the same behavior as the synchronous path.
    for k, v in (spec.get("args_extra") or {}).items():
        args[k] = v
    # Atomically claim the job: records the runner's real pid (this process is
    # its own session/group leader, so pid == pgid the canceller will kill) and
    # refuses if the job was already cancelled during startup — in which case we
    # exit WITHOUT driving the child, so a cancelled delegation never runs.
    if not async_jobs.claim(job_id, os.getpid()):
        return

    def _record_child(sid: str) -> None:
        # Record the child session id ASAP so cascade-cancel can find this
        # job's own subtree even mid-turn.
        async_jobs.update(job_id, child_sid=sid)

    error_text: Any = None
    child_sid = str((async_jobs.read(job_id) or {}).get("child_sid") or "")
    try:
        res = await _drive_child_agent_turn(agent, args, child_session_cb=_record_child)
        content = str(getattr(res, "content", "") or "")
        ok = bool(getattr(res, "ok", False))
        child_sid = str(args.get("_sharedagents_child_session_id_out") or "")
        # If a cancel landed while the child turn was running, do NOT emit a
        # result or finalize — the user killed this job; leave it cancelled.
        cur = async_jobs.read(job_id)
        if cur is None or cur.get("status") != "running":
            return
        # Emit the DURABLE result BEFORE terminalizing: if the process dies in
        # between, the job stays open and reconcile() repairs it idempotently.
        # try_finalize only writes done/failed if still running, so a cancel
        # racing the finalize can't be resurrected into a completed job.
        _emit_result(parent_sid, job_id, agent, child_sid, ok, content, None)
        async_jobs.update(job_id, metrics=args.get("_sharedagents_metrics_out") or {})
        async_jobs.try_finalize(job_id, "done" if ok else "failed", ok)
    except Exception as e:  # noqa: BLE001 — any worker failure is a job failure
        cur = async_jobs.read(job_id)
        if cur is None or cur.get("status") != "running":
            return  # cancelled mid-turn — the error is moot
        content = ""
        ok = False
        error_text = str(e)
        _emit_result(parent_sid, job_id, agent, "", False, "", error_text)
        async_jobs.try_finalize(job_id, "failed", False, error=error_text)

    await deliver_terminal_job(job_id, gateway_url=spec.get("gateway_url"))


def _emit_result(parent_sid: str, job_id: str, agent: str, child_sid: str, ok: bool, content: str, error: Any) -> None:
    from sharedagents_core.layout import sessions_root
    from sharedagents_session.session_writer import get_session_writer

    w = get_session_writer(sessions_root(), parent_sid)
    rec = async_jobs.read(job_id) or {}
    status = rec.get("status")
    if status not in async_jobs.TERMINAL:
        status = "done" if ok else "failed"
    w.lifecycle(
        kind="subagent_result",
        payload={
            "job_id": job_id,
            "parent_sid": parent_sid,
            "child_sid": child_sid,
            "call_id": rec.get("call_id") or "",
            "status": status,
            "agent": agent,
            "ok": bool(ok),
            "content": content,
            "error": error,
        },
    )


async def deliver_terminal_job(
    job_id: str,
    *,
    gateway_url: str | None = None,
) -> bool:
    """Durably inject and wake for one terminal job outside its runner.

    Cancellation kills the detached runner, so the cancelling path uses this
    seam to provide the same wave-barrier completion delivery. A full-wave
    cancellation delivers immediately after all members are terminal.
    """
    rec = async_jobs.read(job_id) or {}
    if rec.get("status") not in async_jobs.TERMINAL:
        return False
    parent_sid = str(rec.get("parent_sid") or "")
    if not parent_sid:
        return False
    from sharedagents_core.layout import sessions_root

    prior = async_jobs._find_result_event(sessions_root(), parent_sid, job_id)
    ok = bool((prior or {}).get("ok")) if prior is not None else bool(rec.get("ok"))
    content = str((prior or {}).get("content") or "")
    error = None if ok else ((prior or {}).get("error") if prior is not None else rec.get("error"))
    if prior is None:
        _emit_result(
            parent_sid,
            job_id,
            str(rec.get("agent") or ""),
            str(rec.get("child_sid") or ""),
            ok,
            content,
            error,
        )
    token = async_jobs.claim_idle_resume_ids(parent_sid, job_id)
    if token is None:
        return False
    try:
        posted = await _deliver_claimed(parent_sid, token, gateway_url=gateway_url)
    except Exception:
        async_jobs.release_resume_claim(parent_sid, token)
        raise
    if posted:
        async_jobs.mark_delivered(parent_sid, token)
    else:
        async_jobs.release_resume_claim(parent_sid, token)
    return posted


async def _deliver_claimed(parent_sid: str, job_ids: list[str], *, gateway_url: str | None = None) -> bool:
    """Deliver one claimed wave using a stable endpoint idempotency key.

    The receipt key is pinned in the wave manifest at first claim (the
    lexically first member job id) so it is stable across claimants, restarts
    and any membership that predates the pin, and it retains the gateway's
    existing job_<hex> receipt format. The wave id remains visible in the
    digest; no gateway/DB migration is necessary.
    """
    from sharedagents_core.injection_guard import wrap_untrusted
    from sharedagents_core.layout import sessions_root

    # A member record removed by external compaction between claim and delivery
    # must not crash the wave: its id stays claimed/delivered bookkeeping.
    members = [rec for rec in (async_jobs.read(jid) for jid in sorted(job_ids)) if rec]
    if not members:
        return True
    priors = {
        rec["job_id"]: async_jobs._find_result_event(sessions_root(), parent_sid, rec["job_id"]) for rec in members
    }
    # A parent stop suppresses the WHOLE wave, including members that finished
    # before the abort cascade. Never revive a stopped parent from recovery.
    # The Stop cascade records "cancelled (session subtree)" on the job itself;
    # the parent-abort reason lives in the silence stamp and the result event.
    if any(
        any(
            p in str(text or "").lower()
            for p in ("parent abort", "parent_cancelled")
            for text in (j.get("error"), j.get("delivery_silenced"), (priors[j["job_id"]] or {}).get("error"))
        )
        for j in members
    ):
        return True
    digest = []
    for rec in members:
        jid = rec["job_id"]
        prior = priors[jid]
        ok = bool(prior.get("ok")) if prior is not None else bool(rec.get("ok"))
        content = str((prior or {}).get("content") or "")
        error = None if ok else ((prior or {}).get("error") or rec.get("error"))
        if prior is None:
            _emit_result(
                parent_sid, jid, str(rec.get("agent") or ""), str(rec.get("child_sid") or ""), ok, content, error
            )
        if not rec.get("wave_id"):
            # Preserve the solo legacy prompt and adapter contract exactly.
            return await _resume_parent(
                parent_sid,
                str(rec.get("agent") or ""),
                ok,
                content,
                error,
                job_id=jid,
                task_id=str(rec.get("child_sid") or ""),
                gateway_url=gateway_url,
            )
        content = _strip_limit_notice(content)
        cost = _job_cost_usd(rec)
        digest.append(
            {
                "job_id": jid,
                "task_id": rec.get("child_sid") or "",
                "agent": rec.get("agent"),
                "status": rec.get("status"),
                "ok": ok,
                "result_ref": {
                    "parent_sid": parent_sid,
                    "event": "subagent_result",
                    "job_id": jid,
                    "child_sid": rec.get("child_sid") or "",
                },
                # Full metrics remain in job storage; the parent only needs cost.
                # Unknown cost is null, never a misleading zero.
                "cost_usd": cost,
                "content": content,
                "error": error,
            }
        )
    wave_id = members[0].get("wave_id")
    body = wrap_untrusted(json.dumps(digest, ensure_ascii=False), source=f"subagent-wave:{wave_id}")
    note = (
        f'<delegation_wave wave_id="{wave_id}" terminal="{len(members)}" '
        f'total="{len(members)}">\n{body}\n</delegation_wave>\n\n'
        "All children in this delegation wave are terminal. Incorporate the "
        "consolidated outcomes and continue. Cancelled or aborted children "
        "were explicitly stopped; do not retry or redispatch them."
    )
    manifest = async_jobs.read("." + str(wave_id or ""))
    receipt_key = (manifest or {}).get("receipt_key") or sorted(job_ids)[0]
    key_token = _BACKGROUND_JOB_ID.set(receipt_key)
    try:
        await _submit_parent_prompt(parent_sid, note, gateway_url=gateway_url)
        return True
    except Exception as exc:  # noqa: BLE001 — release claim for live recovery
        import logging

        logging.getLogger("sharedagents.async_delegate").warning(
            "wave resume POST failed for session %s, wave %s: %s",
            parent_sid,
            wave_id,
            exc,
        )
        return False
    finally:
        _BACKGROUND_JOB_ID.reset(key_token)


def _strip_limit_notice(text: str) -> str:
    """A child's relayed answer must not carry the runtime's budget notice."""
    try:
        from sharedagents_goal.goal_service import strip_budget_limit_notice
    except Exception:  # noqa: BLE001
        return text
    return strip_budget_limit_notice(text)


def _job_cost_usd(rec: dict) -> Any:
    """What this job really cost, from the usage ledger.

    The ledger is the figure every budget and the session footer use. The
    engine's own metrics are a fallback only: subscription CLIs reported 0
    there while the ledger held their real cost ($78 shown as $0).
    Unknown stays ``None``, never a misleading zero.
    """
    import datetime as _dt

    metrics = rec.get("metrics") or {}
    fallback = metrics.get("cost_usd_override", metrics.get("cost_usd"))
    child = str(rec.get("child_sid") or "")
    started = rec.get("created_at")
    if not child or not started:
        return fallback
    try:
        from sharedagents_model.usage_ledger import get_usage_ledger
        from sharedagents_model.usage_pricing import pico_to_usd

        def _iso(ts: float) -> str:
            return _dt.datetime.fromtimestamp(float(ts), _dt.timezone.utc).isoformat()

        ended = rec.get("ended_at")
        # Slack after the end: the runner may mark the job terminal a moment
        # before the child's last turn lands in the ledger. A continuation of
        # the same child bounds the window, so its spend is never counted
        # twice.
        end = float(ended) + 30.0 if ended else None
        later = [
            float(j.get("created_at") or 0)
            for j in async_jobs.list_jobs()
            if j.get("child_sid") == child
            and j.get("job_id") != rec.get("job_id")
            and float(j.get("created_at") or 0) > float(started)
        ]
        if later:
            end = min(later) if end is None else min(end, min(later))
        start_iso = _iso(float(started))
        end_iso = _iso(end) if end is not None else ""
        if end_iso and end_iso <= start_iso:
            return fallback
        pico = get_usage_ledger().session_window_cost_pico(
            child,
            start_iso,
            end_iso,
            end_exclusive=bool(later),
        )
    except Exception:  # noqa: BLE001 — cost display must not block delivery
        return fallback
    if pico is None:
        return fallback
    return round(float(pico_to_usd(pico)), 6)


def _resolve_gateway_url(explicit: str | None = None) -> str | None:
    """Base URL of the gateway the PARENT session lives on — NEVER a prod default.

    A detached async runner resumes the parent by POSTing to the gateway that
    OWNS the parent session. It must NOT hardcode ``127.0.0.1:8100``: the runner
    inherits ``os.environ`` from whatever spawned it, so on a dev/usersim
    instance (which does NOT set ``SHAREDAGENTS_GATEWAY_URL``) that default would
    misroute the resume to the PROD gateway — for a ``parent_sid`` that does not
    even exist there — and leak a dev resume to prod.

    Resolution order (first hit wins), authoritative → best-effort:
      1. ``explicit`` — the spawner's actual control bind, carried in the job
         spec (``spec['gateway_url']``). The orchestrator/monolith that spawned
         the runner knows exactly where its own turn endpoint listens; this is
         the ground truth and always preferred.
      2. ``SHAREDAGENTS_SELF_GATEWAY_URL`` — published by the gateway at boot to its
         OWN control bind (south → :8099, monolith → :8100/:8200); the detached
         runner and the in-process boot resume path both inherit it.
      3. ``SHAREDAGENTS_GATEWAY_URL`` — explicit operator override.
      4. Derive from THIS instance's config (via ``SHAREDAGENTS_HOME``): the
         orchestrator block (the turn endpoint) if present, else the gateway
         block. Loopback substituted for a 0.0.0.0 bind.

    Returns None when nothing resolves — the caller then REFUSES to POST (a
    stranded parent is logged loudly), rather than silently defaulting to prod.
    """
    for cand in (explicit, os.environ.get("SHAREDAGENTS_SELF_GATEWAY_URL"), os.environ.get("SHAREDAGENTS_GATEWAY_URL")):
        if cand and cand.strip():
            return cand.strip().rstrip("/")
    try:
        import json
        from pathlib import Path

        home = os.environ.get("SHAREDAGENTS_HOME") or str(Path.home() / ".sharedagents")
        cfg = json.loads((Path(home) / "config.json").read_text())
        # Prefer the orchestrator (turn-running) bind; monolith configs that
        # lack it fall back to the gateway block.
        blk = cfg.get("orchestrator") or cfg.get("gateway") or {}
        host = blk.get("host") or "127.0.0.1"
        if host in ("0.0.0.0", "::", ""):
            host = "127.0.0.1"
        port = blk.get("port")
        if port:
            return f"http://{host}:{port}".rstrip("/")
    except Exception:  # noqa: BLE001 — no config → refuse, never a prod default
        pass
    return None


async def _submit_parent_prompt(parent_sid: str, text: str, gateway_url: str | None = None, job_id: str = "") -> None:
    """Fire-and-forget a follow-up prompt to the parent's OWN gateway session.

    ``gateway_url`` (the spawner's actual bind, from the job spec) is preferred;
    resolution falls through env/config but NEVER to the prod :8100 default. If
    no gateway can be resolved we RAISE — the caller logs it loudly and the
    durable subagent_result stays on the log — rather than misrouting to prod.
    """
    import httpx

    base = _resolve_gateway_url(gateway_url)
    if not base:
        raise RuntimeError(
            f"cannot resume parent {parent_sid}: no gateway URL resolved "
            f"(spec gateway_url / SHAREDAGENTS_SELF_GATEWAY_URL / SHAREDAGENTS_GATEWAY_URL "
            f"/ config all empty); refusing to default to the prod :8100 gateway"
        )
    url = f"{base}/control/v1/orchestrator/sessions/{parent_sid}/prompt"
    async with httpx.AsyncClient(timeout=30.0) as c:
        payload = {"prompt": text}
        effective_job_id = job_id or _BACKGROUND_JOB_ID.get()
        if effective_job_id:
            payload["background_job_id"] = effective_job_id
        resp = await c.post(url, json=payload)
        # A 404/500 must NOT be treated as a delivered resume — raise
        # so the caller leaves the wave uncovered for reconcile / a retry.
        resp.raise_for_status()


async def _resume_parent(
    parent_sid: str,
    agent: str,
    ok: bool,
    content: str,
    error: Any = None,
    job_id: str = "",
    task_id: str = "",
    gateway_url: str | None = None,
) -> bool:
    """Start a NEW parent turn fed the delegation outcome so the (idle) agent
    reacts to it autonomously — success OR failure. On failure the parent is fed
    the error so jarvis can retry/adjust/report instead of waiting forever.

    Returns True only on a CONFIRMED (2xx) resume POST; False if the POST failed
    or no gateway resolved. NOT silent: the durable subagent_result is already on
    the log, so a failed resume degrades to the (passive) notify state rather than
    losing the result — but we LOG loudly so a stranded parent leaves a trace (the
    previous silent ``except: pass`` hid exactly this failure), and the caller
    uses the False return to roll the resume claim back.
    """
    # The envelope/trailer are trusted runtime control-plane metadata, but the
    # child body is external content. Fence only that body so a child cannot
    # turn its result into a parent-user instruction while the parent can still
    # act on the runtime's "incorporate and continue" directive.
    from sharedagents_core.injection_guard import wrap_untrusted

    source = f"subagent:{agent or 'unknown'}:background"
    content = _strip_limit_notice(content)
    if ok:
        body = wrap_untrusted(
            str(content or "[subagent returned no content]"),
            source=source,
        )
        note = (
            f'<task_result task_id="{task_id}" job_id="{job_id}" '
            f'agent="{agent}">\n{body}\n</task_result>\n\n'
            "Incorporate this completed task and continue."
        )
    else:
        reason = (str(error) if error else content) or "unknown error"
        if "parent abort" in reason.lower() or "parent_cancelled" in reason.lower():
            # Parent was explicitly stopped by the user; never revive an aborted parent.
            return True
        body = wrap_untrusted(
            f"FAILED: {reason.strip()}",
            source=source,
        )
        if "owner abort" in reason.lower() or "owner_aborted" in reason.lower():
            note = (
                f'<task_error task_id="{task_id}" job_id="{job_id}" '
                f'agent="{agent}">\n{body}\n</task_error>\n\n'
                "The user explicitly stopped this delegated task. Do NOT retry or "
                "re-dispatch this task. Acknowledge that the task was cancelled."
            )
        else:
            note = (
                f'<task_error task_id="{task_id}" job_id="{job_id}" '
                f'agent="{agent}">\n{body}\n</task_error>\n\n'
                "The delegated task did not complete. Decide whether to retry, "
                "adjust the approach, or report it."
            )
    try:
        # Keep the submit seam's historical three-argument call contract while
        # carrying the idempotency key in task-local state. Existing adapters
        # and tests commonly replace this seam; a new required keyword would
        # turn an otherwise successful delivery into a false failure.
        token = _BACKGROUND_JOB_ID.set(job_id)
        try:
            await _submit_parent_prompt(
                parent_sid,
                note,
                gateway_url=gateway_url,
            )
        finally:
            _BACKGROUND_JOB_ID.reset(token)
        return True
    except Exception as exc:  # noqa: BLE001
        import logging

        logging.getLogger("sharedagents.async_delegate").warning(
            "parent-resume POST failed for session %s (job result is durable "
            "on the log; parent may be stranded until next turn): %s",
            parent_sid,
            exc,
        )
        return False


async def fire_resume_intents(intents: list[dict[str, Any]], delay: float = 2.0) -> None:
    """Fire the parent-resume intents returned by :func:`async_jobs.reconcile`
    (boot recovery for parents stranded by a runner that died before resuming).

    Waits ``delay`` seconds first so the orchestrator's own control endpoint is
    bound before we POST to it (reconcile runs during startup, before the
    listener is up). Each intent was already RESERVED (not delivered) by
    ``claim_idle_resume_ids`` in reconcile; here we promote it to DELIVERED only on
    a confirmed POST (Residual 1), so this never double-wakes a parent and a
    reservation left by a crash between reconcile and this fire is re-attempted on
    the next boot. Best-effort and loud on failure — the durable subagent_result
    remains on the log regardless.
    """
    if not intents:
        return
    if delay > 0:
        await asyncio.sleep(delay)
    for it in intents:
        try:
            if (async_jobs.read(str(it.get("job_id") or "")) or {}).get("wave_id"):
                posted = await _deliver_claimed(
                    it["parent_sid"],
                    it["covered_ids"],
                    gateway_url=it.get("gateway_url"),
                )
            else:
                posted = await _resume_parent(
                    it["parent_sid"],
                    it.get("agent", ""),
                    bool(it.get("ok")),
                    str(it.get("content") or ""),
                    it.get("error"),
                    job_id=str(it.get("job_id") or ""),
                    task_id=str(it.get("task_id") or ""),
                    gateway_url=it.get("gateway_url"),
                )
            if posted:
                # Residual 1: only a CONFIRMED boot resume becomes durably
                # delivered (final) — so a later reconcile never re-fires it.
                async_jobs.mark_delivered(it["parent_sid"], it.get("covered_ids") or [])
            else:
                # An unconfirmed boot resume must NOT stay reserved —
                # release the claim token so the NEXT boot reconcile re-attempts.
                async_jobs.release_resume_claim(it["parent_sid"], it.get("covered_ids") or [])
        except Exception as exc:  # noqa: BLE001
            import logging

            logging.getLogger("sharedagents.async_delegate").warning(
                "boot resume-intent failed for session %s: %s",
                it.get("parent_sid"),
                exc,
            )
            async_jobs.release_resume_claim(it["parent_sid"], it.get("covered_ids") or [])


_LIVE_RECONCILER_TASK: asyncio.Task | None = None


async def _live_reconcile_loop(interval: float = 2.0) -> None:
    """Continuously recover dead/unconfirmed background completions."""
    while True:
        try:
            intents = await asyncio.to_thread(async_jobs.reconcile, max_age_seconds=600)
            await fire_resume_intents(intents, delay=0)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            import logging

            logging.getLogger("sharedagents.async_delegate").warning(
                "live background reconciliation failed: %s",
                exc,
            )
        await asyncio.sleep(interval)


def start_live_reconciler(interval: float = 2.0) -> asyncio.Task:
    """Start one process-local live reconciler, idempotently."""
    global _LIVE_RECONCILER_TASK
    if _LIVE_RECONCILER_TASK is None or _LIVE_RECONCILER_TASK.done():
        _LIVE_RECONCILER_TASK = asyncio.create_task(
            _live_reconcile_loop(interval),
            name="sharedagents-background-reconciler",
        )
    return _LIVE_RECONCILER_TASK


def main() -> int:
    raw = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
    spec = json.loads(raw)
    asyncio.run(run_job(spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
