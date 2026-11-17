"""Model-facing cron/loop tool surface.

Two halves:

1. **Schema builders** — the ``parameters`` JSON-schema dicts for the four
   loop tools (``cron_create`` / ``cron_delete`` / ``cron_list`` /
   ``schedule_wakeup``). Pure dicts so ``tools.registry`` can import them
   without an import cycle; the ``Tool`` entries themselves live in
   ``registry.BUILTIN_TOOLS`` next to ``goal``/``budget``.

2. **Runners** — thin coercers that turn the raw tool JSON args into the
   typed calls the shared ``loop_service`` backends expect, and compose
   the result text. They exist so the gateway short-circuit
   (``tool_execution``) and the ``CoreBackend`` ACP methods both reach
   exactly one implementation, mirroring the ``goal`` tool's path through
   ``execute_goal_command``.

The kill-switch (``SHAREDAGENTS_DISABLE_CRON=1``) is enforced in the shared
``loop_service`` backends — the single source of truth for "scheduled
work is off" — and every runner inherits it.

Deviations from the reference surface (documented in the clean-room
report): ``cron_list`` scopes to the calling session (sessions are
isolated; a model should not see another session's prompts); the
reference ``ScheduleWakeup`` ``prompt``/``reason``/``noop`` args are
dropped — a wakeup re-fires the session's running loop's OWN stored
prompt (``rearm_from_wakeup``), and there is no noop-fold UI; and
``scheduledFor`` is an ISO timestamp (registry stores ISO), not epoch ms.
"""

from __future__ import annotations

from typing import Any

from sharedagents_loop import loop_service

# ---------------------------------------------------------------------------
# Schema builders (imported by tools/registry.py — keep terse)
# ---------------------------------------------------------------------------


def _params(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": False,
    }


def cron_create_params() -> dict[str, Any]:
    return _params(
        {
            "cron": {
                "type": "string",
                "description": '5-field cron, local time: "M H DoM Mon DoW".',
            },
            "prompt": {"type": "string", "description": "Prompt to fire at each match."},
            "recurring": {"type": "boolean", "description": "Repeat until deleted; no age-based expiry."},
            "durable": {"type": "boolean", "description": "Persist across restarts."},
        },
        required=["cron", "prompt"],
    )


def cron_delete_params() -> dict[str, Any]:
    return _params(
        {"id": {"type": "string", "description": "Job ID from cron_create."}},
        required=["id"],
    )


def cron_list_params() -> dict[str, Any]:
    return _params({}, [])


def schedule_wakeup_params() -> dict[str, Any]:
    return _params(
        {
            "delaySeconds": {
                "type": "number",
                "description": "Seconds from now; clamped to [60, 3600], rounded up to a whole minute.",
            },
            "stop": {
                "type": "boolean",
                "description": "End the loop now; cancels pending wakeups.",
            },
        },
        required=[],
    )


# ---------------------------------------------------------------------------
# Runners — raw tool args -> loop_service backend calls
# ---------------------------------------------------------------------------


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes")


def run_cron_create(
    args: dict[str, Any],
    *,
    agent: Any,
    session_id: str,
) -> dict[str, Any]:
    """``cron_create`` tool call: schedule a cron job on the shared backend."""
    cron = str(args.get("cron") or "").strip()
    prompt = str(args.get("prompt") or "").strip()
    if not cron or not prompt:
        return {
            "ok": False,
            "message": "cron_create: `cron` and `prompt` are required",
        }
    res = loop_service.cron_create(
        agent,
        session_id,
        cron=cron,
        prompt=prompt,
        recurring=_as_bool(args.get("recurring"), True),
        durable=_as_bool(args.get("durable"), False),
    )
    if not res.get("ok"):
        return res
    job = res.get("job") or {}
    label = "recurring" if job.get("recurring") else "one-shot"
    out = dict(res)
    out["message"] = f"Scheduled {label} job {job.get('job_id')} ({job.get('human_schedule')})"
    return out


def run_cron_delete(
    args: dict[str, Any],
    *,
    agent: Any,
    session_id: str | None = None,
) -> dict[str, Any]:
    """``cron_delete`` tool call: cancel a scheduled job by ID."""
    job_id = str(args.get("id") or "").strip()
    if not job_id:
        return {"ok": False, "message": "cron_delete: `id` is required"}
    return loop_service.cron_delete(
        agent,
        job_id,
        session_id=session_id,
    )


def run_cron_list(
    args: dict[str, Any],
    *,
    agent: Any,
    session_id: str,
) -> dict[str, Any]:
    """``cron_list`` tool call: list this session's scheduled jobs."""
    res = loop_service.cron_list(agent, session_id)
    if not res.get("ok"):
        return res
    jobs = res.get("jobs") or []
    out = dict(res)
    out["message"] = f"{len(jobs)} scheduled job(s)"
    return out


def run_schedule_wakeup(
    args: dict[str, Any],
    *,
    agent: Any,
    session_id: str,
) -> dict[str, Any]:
    """``schedule_wakeup`` tool call: pace the session's running loop.

    ``stop:true`` ends the loop's wakeups; otherwise ``delaySeconds`` is
    required and the wakeup re-fires the session's most recent RUNNING
    loop (its own stored prompt — see module docstring).
    """
    stop = _as_bool(args.get("stop"), False)
    if loop_service.cron_disabled():
        return {"ok": False, "message": loop_service.cron_disabled_message()}
    delay = 0
    if not stop:
        raw = args.get("delaySeconds")
        if raw is None or str(raw).strip() == "":
            return {
                "ok": False,
                "message": ("schedule_wakeup: `delaySeconds` is required unless stop is true"),
            }
        if not _session_has_running_loop(session_id):
            return {
                "ok": False,
                "message": (
                    "schedule_wakeup: no running loop for this session — create one first (cron_create or /loop)"
                ),
            }
        try:
            delay = int(raw)
        except (TypeError, ValueError):
            delay = 60
    res = loop_service.schedule_wakeup(
        agent,
        session_id,
        delay_seconds=delay,
        stop=stop,
    )
    if not res.get("ok"):
        return res
    out = dict(res)
    if stop:
        out["message"] = f"Loop stopped — {res.get('cancelledWakeups', 0)} pending wakeup(s) cancelled."
    else:
        out["message"] = f"Wakeup scheduled in {res.get('clampedDelaySeconds')}s (for {res.get('scheduledFor')})."
    return out


def _session_has_running_loop(session_id: str) -> bool:
    from sharedagents_loop.loop_registry import (
        STATUS_RUNNING,
        LoopFilters,
        get_loop_registry,
    )

    loops = get_loop_registry().list_loops(
        LoopFilters(session_id=session_id, status=STATUS_RUNNING),
    )
    return bool(loops)
