"""In-process async event bus with debounced subscribers.

The memory synthesizer wants to fire on "something happened, but
coalesce a burst into one run". This shared bus provides that debounce
pattern for in-process runtime events.

Design constraints:

- **In-process, single-loop.** The whole sharedagents runtime is one asyncio
  loop in one process. We don't need pub/sub across processes.
- **Per-key handlers, not broadcast.** Each event name routes to its
  registered handler(s). Multiple subscribers per event are allowed.
- **Debounce per subscriber.** Each subscriber declares its own quiet
  window. A burst of N publishes within the window collapses into one
  invocation, fired `quiet_seconds` after the LAST publish.
- **Late-publish merging.** If a publish lands while a handler is
  still running, the bus marks "another run is pending" and re-fires
  the handler once the current run finishes. We never run the same
  handler concurrently with itself.
- **Cancel-safe.** Calling `shutdown()` cancels pending timers cleanly;
  in-flight handlers run to completion (the bus waits for them with a
  short timeout before giving up).

Not implemented: priorities, persistent queues, retry policy. If we
need those, we'll switch to a real broker. For now this fits inside
~150 lines and covers the current memory case.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

_log = logging.getLogger(__name__)

EventHandler = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class _Subscription:
    handler: EventHandler
    quiet_seconds: float
    name: str
    # When set, payloads are kept in :attr:`_coalesced` keyed by
    # ``payload.get(coalesce_key)`` — the burst collapses PER KEY
    # instead of globally. Bursts of the SAME key reduce to one
    # run with the latest payload (the original debounce intent);
    # bursts across DISTINCT keys deliver one run PER key once the
    # debounce window expires. (Unified-layout writers across N agents share the bus,
    # so a global single-slot ``_latest_payload`` dropped every
    # session_saved event but the last.)
    coalesce_key: str | None = None

    # internal state — managed by the bus
    _timer: asyncio.TimerHandle | None = field(default=None, repr=False)
    _running: bool = field(default=False, repr=False)
    _pending: bool = field(default=False, repr=False)
    _last_publish_ts: float = field(default=0.0, repr=False)
    # Single-slot path: used when ``coalesce_key`` is None. The
    # handler receives one payload per burst.
    _latest_payload: dict[str, Any] = field(default_factory=dict, repr=False)
    # Coalesced path: maps unique ``payload[coalesce_key]`` -> latest
    # payload for that key. The handler is invoked once per entry.
    _coalesced: dict[Any, dict[str, Any]] = field(
        default_factory=dict,
        repr=False,
    )


class EventBus:
    """Single-loop async event bus.

    Usage:

        bus = EventBus()
        async def on_daily(payload):
            ...
        bus.subscribe("daily_appended", on_daily, quiet_seconds=60.0)
        bus.publish("daily_appended", {"home": "..."})
    """

    def __init__(self) -> None:
        self._subs: dict[str, list[_Subscription]] = {}
        self._closed: bool = False

    def subscribe(
        self,
        event: str,
        handler: EventHandler,
        *,
        quiet_seconds: float = 0.0,
        name: str = "",
        coalesce_key: str | None = None,
    ) -> None:
        """Register `handler` for `event`.

        ``quiet_seconds=0`` means fire immediately on every publish
        (no debounce). Anything > 0 debounces: the handler runs
        ``quiet_seconds`` after the LAST publish in a burst.

        ``coalesce_key`` controls how bursts collapse:

        * ``None`` (default) — single-slot mode: the burst reduces
          to ONE handler run with the most recent payload. Correct
          when every publish refers to the same logical thing.
        * ``"<field>"`` — payloads are partitioned by
          ``payload[<field>]``. Each unique key keeps its latest
          payload; when the debounce window fires, the handler is
          invoked ONCE PER KEY. Use this when one logical event
          (``session_saved``) is fired by N independent producers
          (every agent's writer) and the handler must process all
          of them — not just the last one. Mirrors the unified
          sessions.db: N agents, one bus, no dropped subjects.
        """
        if self._closed:
            raise RuntimeError("EventBus is closed")
        sub = _Subscription(
            handler=handler,
            quiet_seconds=max(0.0, float(quiet_seconds)),
            name=name or getattr(handler, "__name__", "anon"),
            coalesce_key=coalesce_key or None,
        )
        self._subs.setdefault(event, []).append(sub)

    def publish(self, event: str, payload: dict[str, Any] | None = None) -> None:
        """Synchronous publish. Schedules each subscriber per its
        debounce config. Cheap: no LLM work happens on this thread.
        """
        if self._closed:
            return
        payload = dict(payload or {})
        subs = self._subs.get(event)
        if not subs:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            _log.warning("EventBus.publish(%s) without a running loop — dropping", event)
            return
        now = time.monotonic()
        for sub in subs:
            sub._last_publish_ts = now
            if sub.coalesce_key:
                key = payload.get(sub.coalesce_key)
                # Drop publishes with no coalesce-key field — the
                # handler has no way to distinguish them from
                # other key-less publishes, so collapsing into a
                # single entry would re-introduce the dropped-
                # session bug.
                if key is None:
                    continue
                sub._coalesced[key] = payload
            else:
                sub._latest_payload = payload
            if sub.quiet_seconds <= 0.0:
                # Fire-immediately mode.
                self._schedule_run(sub, loop)
                continue
            # Debounced: cancel any pending timer + reschedule.
            if sub._timer is not None:
                try:
                    sub._timer.cancel()
                except Exception:  # noqa: BLE001
                    pass
            sub._timer = loop.call_later(
                sub.quiet_seconds,
                self._on_timer_fire,
                sub,
                loop,
            )

    def _on_timer_fire(self, sub: _Subscription, loop: asyncio.AbstractEventLoop) -> None:
        sub._timer = None
        self._schedule_run(sub, loop)

    def _schedule_run(self, sub: _Subscription, loop: asyncio.AbstractEventLoop) -> None:
        if sub._running:
            # A run is already in flight. Mark pending; the run-loop
            # will re-fire once it finishes.
            sub._pending = True
            return
        sub._running = True
        loop.create_task(self._run_subscription(sub))

    async def _run_subscription(self, sub: _Subscription) -> None:
        """Run a subscriber; a publish landing mid-run re-fires it once."""
        try:
            while True:
                sub._pending = False
                # Snapshot whatever the publish path coalesced for
                # this burst; clear so a publish landing mid-handler
                # accumulates a fresh batch for the next pass.
                if sub.coalesce_key:
                    payloads = list(sub._coalesced.values())
                    sub._coalesced = {}
                else:
                    payloads = [sub._latest_payload]
                    sub._latest_payload = {}
                for payload in payloads:
                    try:
                        await sub.handler(payload)
                    except Exception:  # noqa: BLE001
                        _log.exception(
                            "EventBus subscriber %r raised",
                            sub.name,
                        )
                # Did another publish land while we were running? If
                # so, run once more with the latest payload(s).
                if not sub._pending:
                    break
        finally:
            sub._running = False

    async def shutdown(self, *, drain_timeout: float = 5.0) -> None:
        """Cancel pending debounce timers, wait for in-flight handlers
        to finish (bounded by `drain_timeout`), then mark closed.
        """
        self._closed = True
        for subs in self._subs.values():
            for sub in subs:
                if sub._timer is not None:
                    try:
                        sub._timer.cancel()
                    except Exception:  # noqa: BLE001
                        pass
                    sub._timer = None
        # Wait for in-flight handlers (they self-clear `_running`).
        deadline = time.monotonic() + max(0.0, drain_timeout)
        while time.monotonic() < deadline:
            still_running = any(sub._running for subs in self._subs.values() for sub in subs)
            if not still_running:
                return
            await asyncio.sleep(0.05)


# ---- module-level default bus --------------------------------------

# The runtime uses ONE bus per process. Modules import this directly
# rather than threading a bus parameter through every call site.
_DEFAULT_BUS: EventBus | None = None


def get_bus() -> EventBus:
    """Return (lazy-init) the process-wide event bus."""
    global _DEFAULT_BUS
    if _DEFAULT_BUS is None:
        _DEFAULT_BUS = EventBus()
    return _DEFAULT_BUS


# Reset hooks — INVERTED dependency (C2). A subscriber that keeps its own
# module-level "already subscribed" guard (e.g. ``memory_synth._REGISTERED``)
# registers a callback here at import time; ``reset_bus_for_tests`` invokes them
# without the bus importing its consumers. This keeps the kernel a true leaf:
# the bus knows nothing about the memory package. (Was: a lazy import of
# ``memory_synth`` inside ``reset_bus_for_tests`` — a publisher reaching a
# subscriber, backwards; see spec §3.4 C2.)
_RESET_HOOKS: list[Callable[[], None]] = []


def register_reset_hook(hook: Callable[[], None]) -> None:
    """Register a callback run by :func:`reset_bus_for_tests` to clear a
    subscriber's own module-level registration state. Idempotent registration
    is the caller's responsibility (register once at import).
    """
    if hook not in _RESET_HOOKS:
        _RESET_HOOKS.append(hook)


def reset_bus_for_tests() -> None:
    """Drop the singleton AND clear the per-subscriber registration
    state in modules that subscribe at boot. Used by tests that want
    a fresh bus + a fresh ability to re-subscribe.
    """
    global _DEFAULT_BUS
    _DEFAULT_BUS = None
    # Clear module-level "already subscribed" guards via registered hooks so the
    # next subscribe call lands fresh — without the kernel importing its
    # consumers (inverted C2; keeps this module a leaf).
    for hook in list(_RESET_HOOKS):
        try:
            hook()
        except Exception:  # noqa: BLE001
            pass


__all__ = [
    "EventBus",
    "EventHandler",
    "get_bus",
    "register_reset_hook",
    "reset_bus_for_tests",
]
