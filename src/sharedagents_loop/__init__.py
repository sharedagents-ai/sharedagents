"""sharedagents_loop — durable scheduled loops (the /loop family).

Ports the behavior of Claude Code's ``/loop`` command (clean-room
reimplementation; see ``plan/2026-08-06-loops-tab.md``): interval- and
cron-backed loops that re-prompt a session on a schedule, with run
history, append-only events, a cross-process supervisor lease, no automatic
expiry, a 50-job ceiling, and the cron-tool family
(``CronCreate``/``CronDelete``/``CronList``/``ScheduleWakeup``).
"""

from __future__ import annotations

from sharedagents_loop.loop_registry import (
    STATUS_ARCHIVED,
    STATUS_EXPIRED,
    STATUS_PAUSED,
    STATUS_RUNNING,
    LoopFilters,
    LoopRecord,
    LoopRegistry,
    TransitionResult,
    get_loop_registry,
)
from sharedagents_loop.loop_supervisor import LoopSupervisor

__all__ = [
    "LoopRegistry",
    "LoopRecord",
    "LoopFilters",
    "TransitionResult",
    "LoopSupervisor",
    "STATUS_RUNNING",
    "STATUS_PAUSED",
    "STATUS_EXPIRED",
    "STATUS_ARCHIVED",
    "get_loop_registry",
]
