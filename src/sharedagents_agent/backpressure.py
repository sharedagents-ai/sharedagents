"""Per-subscriber backpressure policy — spec §11 (Backpressure & delivery).

The gateway fans the canonical event log out to every subscriber over a
per-connection cursor SSE tail (see
``sharedagents_gateway.control_endpoint._stream_events_mode``, which emits
``{offset, seq, spawnedBy, event}`` ``log`` frames). Fan-out is per-connection
so one slow reader never blocks the writer or other subscribers (§11, §14.6).

This module is the pure state machine that decides, per subscriber, what to do
with each event waiting to be delivered. It holds NO sockets, NO asyncio, and
does NO I/O — the SSE generator owns the transport and calls this policy to
decide ``deliver`` / ``pause`` / ``drop`` / ``close``. The clock is INJECTED
(every time-dependent method takes ``now``) so the policy is fully
deterministic under test.

The two-tier contract (§11), implemented faithfully:

* **Per-subscriber bounded queue** — ``MAX_EVENTS`` events OR ``MAX_BYTES``
  bytes, whichever comes first (a soft cap).
* **On the soft cap → pause** the live tail push for that subscriber. The
  durable JSONL log is the backfill buffer; the event is still counted and is
  **never dropped**.
* **Hard ceiling** — if paused longer than ``PAUSE_CEILING_S`` OR the queued
  bytes exceed ``2 * MAX_BYTES`` → ``close`` the connection with a resumable
  ``reconnect`` carrying ``resume_offset()`` (the last delivered byte offset);
  the client reconnects and backfills from that offset → zero loss (§9.2).
* **``dropIfSlow`` is permitted ONLY for ephemeral side-channels** (typing /
  presence). Canonical log events (``ephemeral=False``) can NEVER return
  ``"drop"`` under any sequence (§11, §14.6).
* The writer NEVER blocks on frontend delivery — this policy only sheds the
  *live push* to a slow subscriber; truth (the log) is untouched.
"""

from __future__ import annotations

# Soft caps (§11): "≈256 events OR 4–8 MiB, whichever first".
MAX_EVENTS: int = 256
MAX_BYTES: int = 8 * 1024 * 1024
# Hard ceiling on time spent paused before we close-with-offset (§11: "~30 s").
PAUSE_CEILING_S: float = 30.0

# Verdict strings (the only values ``offer`` returns).
DELIVER: str = "deliver"
PAUSE: str = "pause"
DROP: str = "drop"
CLOSE: str = "close"

# Lifecycle states.
FLOWING: str = "flowing"
PAUSED: str = "paused"
CLOSING: str = "closing"


class BackpressurePolicy:
    """Per-subscriber delivery state machine (spec §11).

    Pure: no asyncio, no I/O, injected clock. One instance per subscribed
    connection. The owning SSE generator calls :meth:`offer` before pushing an
    event and :meth:`acked` after the socket drains, then consults
    :meth:`state` / :meth:`resume_offset` to decide whether to keep tailing,
    pause, or close-with-reconnect.

    The hard byte ceiling is ``2 * max_bytes`` (§11): once queued bytes pass it
    while paused, no amount of waiting will recover the reader, so we close.
    Once :meth:`state` is ``"closing"`` the policy is terminal — every further
    :meth:`offer` returns ``"close"``.
    """

    def __init__(
        self,
        *,
        max_events: int = MAX_EVENTS,
        max_bytes: int = MAX_BYTES,
        pause_ceiling_s: float = PAUSE_CEILING_S,
    ) -> None:
        self._max_events = int(max_events)
        self._max_bytes = int(max_bytes)
        self._pause_ceiling_s = float(pause_ceiling_s)
        # Hard byte ceiling — 2x the soft cap (§11).
        self._hard_bytes = 2 * self._max_bytes

        self._queued_events: int = 0
        self._queued_bytes: int = 0
        self._paused_since: float | None = None
        self._last_delivered_offset: int = 0
        self._closing: bool = False

    # --- read accessors ----------------------------------------------------

    @property
    def queued_events(self) -> int:
        """Events recorded as waiting to be delivered to this subscriber."""
        return self._queued_events

    @property
    def queued_bytes(self) -> int:
        """Bytes recorded as waiting to be delivered to this subscriber."""
        return self._queued_bytes

    @property
    def paused_since(self) -> float | None:
        """The injected ``now`` when the subscriber first crossed a soft cap,
        or ``None`` while flowing (cleared by :meth:`acked` on recovery)."""
        return self._paused_since

    def state(self) -> str:
        """``"flowing"`` | ``"paused"`` | ``"closing"`` (terminal)."""
        if self._closing:
            return CLOSING
        if self._paused_since is not None:
            return PAUSED
        return FLOWING

    def resume_offset(self) -> int:
        """Last delivered byte offset — the cursor put in the ``reconnect``
        frame so the client backfills from here with zero loss (§9.2)."""
        return self._last_delivered_offset

    # --- state transitions -------------------------------------------------

    def offer(
        self,
        *,
        nbytes: int,
        offset: int,
        now: float,
        ephemeral: bool = False,
    ) -> str:
        """Record an event waiting for this subscriber; return a verdict.

        Verdicts:

        * ``"deliver"`` — room under the soft caps; send it.
        * ``"pause"`` — a canonical event that crosses a soft cap. It is STILL
          enqueued (never dropped); the live push pauses.
        * ``"drop"`` — ONLY when ``ephemeral=True`` and over a soft cap; the
          event is NOT enqueued. Canonical events never return this.
        * ``"close"`` — the hard ceiling is breached (paused longer than
          ``pause_ceiling_s`` OR queued bytes over ``2 * max_bytes``); close
          with a reconnect-from-offset. Terminal.

        ``offset`` is the byte position after this event (its frame's
        ``offset``); we hold it so a later :meth:`acked` advances the resume
        cursor to exactly what was delivered.
        """
        # Terminal: once closing, always closing (§11).
        if self._closing:
            return CLOSE

        # Ephemeral side-channels (typing / presence) MAY drop when over cap;
        # canonical log events never reach this branch.
        if ephemeral:
            if self._would_cross_soft_cap(nbytes):
                return DROP
            self._enqueue(nbytes)
            return DELIVER

        # Canonical event — always enqueued (never dropped).
        self._enqueue(nbytes)

        over_soft_cap = self._queued_events > self._max_events or self._queued_bytes > self._max_bytes
        if over_soft_cap and self._paused_since is None:
            # First crossing — record when we entered the paused state.
            self._paused_since = now

        if self._paused_since is not None:
            # While paused, check the hard ceiling: too long OR too many bytes.
            paused_for = now - self._paused_since
            if paused_for > self._pause_ceiling_s or self._queued_bytes > self._hard_bytes:
                self._closing = True
                return CLOSE
            return PAUSE

        return DELIVER

    def acked(
        self,
        *,
        count: int,
        nbytes: int,
        offset: int,
        now: float,
    ) -> None:
        """The subscriber drained ``count`` events / ``nbytes`` bytes up to
        byte ``offset``. Reduce the queue, advance the last-delivered offset,
        and clear ``paused_since`` once back under the soft cap.

        No-op once closing — a draining ack cannot revive a connection already
        committed to a reconnect-from-offset close.
        """
        if self._closing:
            return

        self._queued_events = max(0, self._queued_events - int(count))
        self._queued_bytes = max(0, self._queued_bytes - int(nbytes))
        if offset > self._last_delivered_offset:
            self._last_delivered_offset = int(offset)

        # Back under both soft caps → flowing again, clear the pause clock.
        if self._queued_events <= self._max_events and self._queued_bytes <= self._max_bytes:
            self._paused_since = None

    # --- internals ---------------------------------------------------------

    def _enqueue(self, nbytes: int) -> None:
        self._queued_events += 1
        self._queued_bytes += int(nbytes)

    def _would_cross_soft_cap(self, nbytes: int) -> bool:
        """Whether adding ``nbytes`` would push past either soft cap — used to
        decide an ephemeral drop *without* enqueuing."""
        return self._queued_events + 1 > self._max_events or self._queued_bytes + int(nbytes) > self._max_bytes


__all__ = [
    "BackpressurePolicy",
    "MAX_EVENTS",
    "MAX_BYTES",
    "PAUSE_CEILING_S",
    "DELIVER",
    "PAUSE",
    "DROP",
    "CLOSE",
    "FLOWING",
    "PAUSED",
    "CLOSING",
]
