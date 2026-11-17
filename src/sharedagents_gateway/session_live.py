"""Per-session live-change notifier for the physical session log stream.

The physical append-only JSONL file is the single authoritative source of truth.
Watch mode is always enabled and permanently monitors the byte size of the
session JSONL on the shared filesystem with sub-100ms response time.
"""

from __future__ import annotations

import asyncio

_WATCH_POLL_S = 0.075


def _jsonl_size(session_hash: str) -> int:
    """Byte size of the session JSONL (0 if absent). Strictly monotonic."""
    from sharedagents_core.layout import sessions_root

    try:
        return (sessions_root() / f"{session_hash}.jsonl").stat().st_size
    except (OSError, ValueError):
        return 0


def current_version(session_hash: str) -> int:
    """Monotonic change marker: always returns the JSONL byte size."""
    return _jsonl_size(session_hash)


async def wait_for_change(
    session_hash: str,
    last_version: int,
    timeout: float,
) -> int:
    """Wait until the session JSONL file size advances past last_version or timeout expires."""
    import time as _time

    deadline = _time.monotonic() + timeout
    cur = _jsonl_size(session_hash)
    if cur != last_version:
        return cur
    while _time.monotonic() < deadline:
        await asyncio.sleep(_WATCH_POLL_S)
        cur = _jsonl_size(session_hash)
        if cur != last_version:
            return cur
    return last_version
