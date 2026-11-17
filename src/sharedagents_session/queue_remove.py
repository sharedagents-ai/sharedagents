"""Compatible ``queue_remove`` invocation and consume-reason vocabulary.

``queue_msg_removed.reason`` used to default to ``user_deleted`` on every
path, including gateway consumption of a child wrap-up. That made an
automatic drain read like a human deletion.

The positional arity stays ``(session_id, queue_id)``. ``reason`` is
keyword-only and optional so a rolling restart (new gateway + old
orchestrator, or the reverse) cannot TypeError on an extra positional
RPC argument.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

QUEUE_REMOVE_USER_DELETED = "user_deleted"
QUEUE_REMOVE_USER_DRAINED = "user_drained"
QUEUE_REMOVE_BACKGROUND_CONSUMED = "background_consumed"
QUEUE_REMOVE_BACKGROUND_PURGED = "background_purged"

QUEUE_REMOVE_REASONS = frozenset(
    {
        QUEUE_REMOVE_USER_DELETED,
        QUEUE_REMOVE_USER_DRAINED,
        QUEUE_REMOVE_BACKGROUND_CONSUMED,
        QUEUE_REMOVE_BACKGROUND_PURGED,
    }
)

_RemoveFn = Callable[..., Awaitable[Any]]


def normalize_queue_remove_reason(reason: str | None) -> str | None:
    """Return an allowlisted reason, or ``None`` to keep the callee default.

    Unknown tokens collapse to ``user_deleted`` so an ACP client cannot
    persist a spoofed provenance label. Empty/omitted stays ``None``.
    """
    value = str(reason or "").strip().lower()
    if not value:
        return None
    if value not in QUEUE_REMOVE_REASONS:
        return QUEUE_REMOVE_USER_DELETED
    return value


def _accepts_reason(fn: _RemoveFn) -> bool:
    try:
        return "reason" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


async def invoke_queue_remove(
    fn: _RemoveFn,
    session_id: str,
    queue_id: str,
    *,
    reason: str | None = None,
) -> Any:
    """Call ``queue_remove`` with ``reason`` only when the callee accepts it.

    Older backends / test fakes are ``(sid, queue_id)`` only. Passing a
    third positional argument — or an unexpected kwarg — during a mixed-
    version restart would drop the consume. Named JSON-RPC params plus
    signature probing keep both sides working; the old side just records
    the historical default until it is upgraded. A TypeError from the
    callee itself is not retried — that is a real backend failure.
    """
    normalized = normalize_queue_remove_reason(reason)
    if normalized and _accepts_reason(fn):
        return await fn(session_id, queue_id, reason=normalized)
    return await fn(session_id, queue_id)
