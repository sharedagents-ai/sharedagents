"""Tail one gateway session's canonical event log → WhatsApp deliveries.

One :class:`SessionObserver` per bound session renders EVERY turn of it: the
ones this channel submitted, turns started on web/TUI, and the background
wake-up turns the gateway starts on its own when a delegated job finishes.
A submit-scoped reader would silently drop those delegated results.

Per turn, ``text_delta`` accumulates; in ``blocks`` mode the text is flushed as
its own message whenever a tool call starts (so a long turn shows progress), in
``final`` mode once at ``turn_ended``. Ask-family tool calls and
``permission_request`` events park an interaction the operator answers by reply.
Files the turn recorded (``artifact_reference``) are sent right after the text
that links to them.

Replay after a restart is bounded by event age (``max_age_s``, the same knob as
inbound staleness): stale text and outcomes are skipped and summarised once the
observer is caught up — signalled by the first heartbeat, which the gateway only
emits after the backlog is drained — while prompts still parked at that point
are surfaced, because they are still actionable. The persisted cursor only
advances after a delivery and never while a prompt is parked, so a restart
neither repeats a delivered block nor loses a pending question.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from sharedagents_agent.delivery import is_subagent_event, strip_runtime_notices
from sharedagents_session.ask_pending import is_ask_tool_name
from sharedagents_whatsapp.formatting import is_memory_flush_event, strip_internal_envelopes
from sharedagents_whatsapp.gateway import GatewayClient, SessionNotFound
from sharedagents_whatsapp.transport import TransportUnavailable

RETRY_DELAYS_S = (1.0, 2.0, 5.0, 10.0, 30.0)
# How long one failing delivery (e.g. WhatsApp offline) is retried before the
# observer gives up on it and moves on — transient outages recover, while a
# message WhatsApp permanently rejects cannot stall the session forever.
DELIVERY_RETRY_BUDGET_S = 600.0
_BLANKS_RE = re.compile(r"\n{3,}")


class ObserverSink(Protocol):
    """What the observer tells its owner (the channel)."""

    async def on_turn_started(self, sid: str, turn_idx: int, offset: int, event: dict) -> None: ...
    async def on_block(self, sid: str, turn_idx: int, text: str) -> None: ...
    async def on_turn_ended(self, sid: str, turn_idx: int, event: dict, delivered: bool) -> None: ...
    async def on_artifact(self, sid: str, turn_idx: int, event: dict) -> None: ...
    async def on_question(self, sid: str, call_id: str, arguments: Any) -> None: ...
    async def on_permission(self, sid: str, event: dict) -> None: ...
    async def on_interaction_resolved(self, sid: str, key: str) -> None: ...
    async def on_cursor(self, sid: str, offset: int) -> None: ...
    async def on_skipped(self, sid: str, turns: int) -> None: ...
    async def on_session_gone(self, sid: str) -> None: ...
    async def on_queue_event(self, sid: str, event: dict) -> None: ...


def event_epoch(event: dict) -> float | None:
    """The event's ``ts`` (ISO-8601) as epoch seconds, or ``None``."""
    ts = event.get("ts")
    if not isinstance(ts, str) or not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


@dataclass
class _Turn:
    idx: int
    fresh: bool = True
    buf: list[str] = field(default_factory=list)
    delivered: bool = False
    silent: bool = False


class SessionObserver:
    """Long-lived subscriber for one session's canonical event stream."""

    def __init__(
        self,
        gateway: GatewayClient,
        sid: str,
        sink: ObserverSink,
        *,
        start_offset: int = 0,
        delivery_mode: str = "blocks",
        max_age_s: float = 900.0,
        clock: Callable[[], float] = time.time,
        log: logging.Logger | None = None,
        retry_delays: tuple[float, ...] = RETRY_DELAYS_S,
        retry_budget_s: float = DELIVERY_RETRY_BUDGET_S,
    ) -> None:
        self._gateway = gateway
        self.sid = sid
        self._sink = sink
        self._cursor = max(0, int(start_offset))
        self._resuming_turn = self._cursor > 0
        self._mode = delivery_mode
        self._max_age = float(max_age_s)
        self._clock = clock
        self._log = log or logging.getLogger("sharedagents.whatsapp.observer")
        self._retry = tuple(retry_delays) or (1.0,)
        self._retry_budget = float(retry_budget_s)
        self._turns: dict[int, _Turn] = {}
        self._pending: dict[str, list] = {}  # key → [kind, payload, sent]
        self._caught_up = False
        self._skipped = 0
        self._task: asyncio.Task | None = None

    @property
    def cursor(self) -> int:
        return self._cursor

    def start(self) -> None:
        if self._task is None or self._task.done():
            # Empty context: an observer outlives the message that started it and
            # must not inherit that message's reply target.
            self._task = contextvars.Context().run(
                asyncio.create_task,
                self.run(),
                name=f"wa-observer-{self.sid[:8]}",
            )

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def run(self) -> None:
        """Stream forever from the cursor; back off and resume on transport errors."""
        attempt = 0
        while True:
            try:
                self._caught_up = False
                async for item in self._gateway.stream(self.sid, self._cursor):
                    attempt = 0
                    if item.kind == "reconnect":
                        self._caught_up = False
                        continue
                    if item.kind == "heartbeat":
                        await self._deliver(self._catch_up, "catch-up")
                        continue
                    event = item.event
                    await self._deliver(
                        lambda: self._handle(item.offset, event),
                        str(event.get("type")),
                    )
                    self._cursor = max(self._cursor, item.offset)
            except asyncio.CancelledError:
                raise
            except SessionNotFound:
                self._log.warning("session %s is gone; stopping its observer", self.sid)
                await self._sink.on_session_gone(self.sid)
                return
            except Exception as exc:  # noqa: BLE001 — gateway restarts are routine
                self._log.warning(
                    "observer %s: stream dropped (%s); resuming from %d",
                    self.sid[:8],
                    exc,
                    self._cursor,
                )
            delay = self._retry[min(attempt, len(self._retry) - 1)]
            attempt += 1
            await asyncio.sleep(delay)

    async def _deliver(self, step: Callable[[], Awaitable[None]], what: str) -> None:
        """Run one delivery step; retry failures with backoff within the budget.

        A step that still fails when the budget runs out is logged and skipped
        so one undeliverable event cannot stall the session's stream forever.
        """
        deadline: float | None = None
        failures = 0
        while True:
            attempt_started = time.monotonic()
            try:
                await step()
                return
            except asyncio.CancelledError:
                raise
            except TransportUnavailable:
                # WhatsApp is reconnecting: wait it out; this is not the
                # delivery's fault, so the give-up budget is not spent.
                failures += 1
                await asyncio.sleep(self._retry[min(failures - 1, len(self._retry) - 1)])
                if deadline is not None:
                    deadline += time.monotonic() - attempt_started
                continue
            except Exception:  # noqa: BLE001 — retry, then give up on this one delivery
                failures += 1
                now = time.monotonic()
                if deadline is None:
                    deadline = now + self._retry_budget
                if now >= deadline:
                    self._log.exception(
                        "observer %s: giving up on %s after %d attempts",
                        self.sid[:8],
                        what,
                        failures,
                    )
                    return
                self._log.warning(
                    "observer %s: delivering %s failed (attempt %d); retrying",
                    self.sid[:8],
                    what,
                    failures,
                    exc_info=True,
                )
                await asyncio.sleep(self._retry[min(failures - 1, len(self._retry) - 1)])

    def _fresh(self, event: dict) -> bool:
        epoch = event_epoch(event)
        return epoch is None or epoch >= self._clock() - self._max_age

    def _turn(self, idx: Any) -> _Turn | None:
        if not isinstance(idx, int) or isinstance(idx, bool):
            return None
        return self._turns.setdefault(idx, _Turn(idx, delivered=self._resuming_turn))

    async def _handle(self, offset: int, event: dict) -> None:
        if is_subagent_event(event):
            return
        etype = event.get("type")
        before = self._cursor
        fresh = self._fresh(event)
        idx = event.get("turn_idx")
        if etype == "turn_started":
            self._resuming_turn = False
            if isinstance(idx, int) and not isinstance(idx, bool):
                silent = is_memory_flush_event(event)
                self._turns[idx] = _Turn(idx, fresh=fresh and not silent, silent=silent)
                if fresh and not silent:
                    await self._sink.on_turn_started(self.sid, idx, offset, event)
            return
        if etype == "text_delta":
            turn = self._turn(idx)
            piece = str(event.get("delta") or event.get("chunk") or "")
            if turn is not None and piece:
                # A block's freshness is its newest delta: a block that spans
                # the cutoff is delivered whole rather than cut mid-sentence.
                turn.buf.append(piece)
                turn.fresh = fresh
            return
        if etype == "tool_call":
            turn = self._turn(idx)
            is_ask = is_ask_tool_name(event.get("name"))
            if turn is not None:
                if self._mode == "blocks" or is_ask:
                    await self._flush(turn, cursor_after=before)
                elif turn.buf:
                    turn.buf.append("\n\n")  # final mode: keep segments apart
            if is_ask:
                call_id = str(event.get("call_id") or "")
                if call_id:
                    await self._park(call_id, "question", event.get("arguments"), fresh)
            return
        if etype == "tool_result_meta":
            await self._resolve(str(event.get("call_id") or ""))
            return
        if etype == "permission_request":
            # The agent's explanation must land before the 🔐 prompt, in both modes.
            for open_turn in list(self._turns.values()):
                if open_turn.buf:
                    await self._flush(open_turn, cursor_after=before)
            request_id = str(event.get("request_id") or "")
            if request_id:
                await self._park(request_id, "permission", event, fresh)
            return
        if etype == "artifact_reference":
            await self._artifact(offset, event, before, fresh)
            return
        if etype in ("queue_msg_injected", "queue_msg_removed"):
            await self._sink.on_queue_event(self.sid, event)
            return
        if etype == "permission_resolved":
            await self._resolve(str(event.get("request_id") or ""))
            return
        if etype == "turn_ended":
            turn = None
            if isinstance(idx, int) and not isinstance(idx, bool):
                turn = self._turn(idx)
            if turn is None:
                turn = _Turn(idx if isinstance(idx, int) else -1, fresh=fresh)
            await self._flush(turn, cursor_after=None)
            # The turn is over: whatever it had parked is moot — cleared
            # silently (the sink drops its own prompt state on turn end).
            self._pending.clear()
            silent = turn.silent or is_memory_flush_event(event)
            if silent:
                await self._sink.on_cursor(self.sid, offset)
                self._turns.pop(turn.idx, None)
                return
            if fresh:
                await self._sink.on_turn_ended(self.sid, turn.idx, event, turn.delivered)
            else:
                self._skipped += 1
            await self._sink.on_cursor(self.sid, offset)
            self._turns.pop(turn.idx, None)

    async def _artifact(self, offset: int, event: dict, before: int, fresh: bool) -> None:
        """Send one file the turn recorded, after the text that links to it.

        The session writer records a turn's files just before ``turn_ended``,
        so the reply prose is complete here: flushing it first keeps the
        message that names the file above the file. The cursor then moves past
        the file, so a retry or restart re-sends the file, never the text.
        """
        idx = event.get("turn_idx")
        if isinstance(idx, int) and not isinstance(idx, bool):
            turn = self._turn(idx)
        else:  # a gateway that predates turn-scoped artifact frames
            turn = max(self._turns.values(), key=lambda t: t.idx, default=None)
        if turn is None or turn.silent or not fresh or not event.get("sha256"):
            return
        await self._flush(turn, cursor_after=before)
        await self._sink.on_artifact(self.sid, turn.idx, event)
        turn.delivered = True
        if not self._pending:
            await self._sink.on_cursor(self.sid, offset)

    async def _flush(self, turn: _Turn, *, cursor_after: int | None) -> None:
        text = _BLANKS_RE.sub(
            "\n\n",
            strip_internal_envelopes(strip_runtime_notices("".join(turn.buf))),
        ).strip()
        if turn.silent or not text or not turn.fresh:
            turn.buf.clear()
            return
        await self._sink.on_block(self.sid, turn.idx, text)
        turn.buf.clear()
        turn.delivered = True
        if cursor_after is not None and not self._pending:
            await self._sink.on_cursor(self.sid, cursor_after)

    async def _park(self, key: str, kind: str, payload: Any, fresh: bool) -> None:
        entry = [kind, payload, False]
        self._pending[key] = entry
        if fresh or self._caught_up:
            await self._send_prompt(key, entry)

    async def _send_prompt(self, key: str, entry: list) -> None:
        kind, payload, _ = entry
        if kind == "question":
            await self._sink.on_question(self.sid, key, payload)
        else:
            await self._sink.on_permission(self.sid, payload)
        entry[2] = True

    async def _resolve(self, key: str) -> None:
        entry = self._pending.pop(key, None) if key else None
        if entry is not None and entry[2]:
            await self._sink.on_interaction_resolved(self.sid, key)

    async def _catch_up(self) -> None:
        if self._caught_up:
            return
        for key, entry in list(self._pending.items()):
            if not entry[2]:
                await self._send_prompt(key, entry)
        if self._skipped:
            skipped, self._skipped = self._skipped, 0
            await self._sink.on_skipped(self.sid, skipped)
        self._caught_up = True
