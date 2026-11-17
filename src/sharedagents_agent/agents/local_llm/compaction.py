"""CompactionMixin — extracted from the ``local_llm`` monolith (behaviour-preserving)."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Awaitable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .state import (
    _MIN_COMPACTION_CHARS_FREED,
    _CompactionAborted,
    _NoSummaryCompaction,
)

# Summary text carried on the durable ``compaction`` event (the WebUI card's
# expandable body). The full memo lives in the compaction child session.
_COMPACTION_EVENT_SUMMARY_MAX_CHARS = 12_000


# OpenCode (sst/opencode, session/overflow.ts + session/compaction.ts) —
# ported as-is. Auto-compaction fires when the last round's billed total
# reaches ``context - max_output`` and keeps a recent tail of
# ``clamp(usable * 0.25, MIN, MAX)`` tokens; everything older is summarized.
_OC_MIN_PRESERVE_RECENT_TOKENS = 2_000
_OC_MAX_PRESERVE_RECENT_TOKENS = 15_000


def _oc_usable(ctx_window: int, max_output: int) -> int:
    """OpenCode ``usable()`` for a model with no separate input limit."""
    return max(0, int(ctx_window) - int(max_output))


def _oc_preserve_recent_tokens(usable: int) -> int:
    """OpenCode ``preserveRecentBudget()``: tail kept verbatim after a compaction."""
    return min(
        _OC_MAX_PRESERVE_RECENT_TOKENS,
        max(_OC_MIN_PRESERVE_RECENT_TOKENS, int(usable * 0.25)),
    )


class CompactionMixin:
    def _maybe_compact(
        self,
        session: Any,
        messages: list[dict[str, Any]],
        *,
        force: bool = False,
        shrink: bool = False,
        trigger: str = "",
    ) -> list[dict[str, Any]]:
        """Proactive compaction trigger.

        Reference: acp_bridge.py L1949+ and turn_runtime.compact_messages
        (L1071). When estimated ``prompt_tokens + max_tokens >=
        compact_at_frac × upstream_max_context``, squeeze old tool
        results in place and persist the rolling summary onto the
        session. ``force=True`` bypasses the threshold with a window-sized
        target; ``shrink=True`` (model ``compact_context``)
        bypasses the threshold with a target sized off CURRENT usage and
        still honours the no-progress guard. A squeeze that actually shrinks
        the prompt is announced as a visible compaction subagent
        (:meth:`_announce_compaction`); a no-op is silent.

        Also handles the memory-flush hook: when the upcoming
        compaction would cross the reserve threshold
        (``cfg.memory_flush_reserve_tokens_floor`` +
        ``cfg.memory_flush_soft_threshold_tokens``), pause and run a
        silent agentic turn that lets the model save context to
        MEMORY.md before squeezing. See cfg.memory_flush_* defaults
        and the existing flow inside acp_bridge.
        """
        try:
            from sharedagents_agent.turn_runtime import (
                _msg_chars,
                compact_messages,
            )
        except ImportError:
            return messages

        target = self._compaction_target(
            session,
            messages,
            force=force,
            shrink=shrink,
        )
        if target is None:
            return messages
        # ``compact_messages`` deliberately rewrites old tool-message dicts in
        # place before rebuilding the list. Snapshot both measures first;
        # re-measuring ``messages`` afterwards would make every mechanical
        # reduction look like a no-op and suppress its visible child/event.
        before_tokens = self._wire_tokens(session, messages)
        before_chars = _msg_chars(messages)
        ctx_window = int(self._effective_context_window(session) or 0)
        try:
            from sharedagents_agent.compaction import prune_session_tool_outputs

            pruned_chars, _ = prune_session_tool_outputs(session, context_window=ctx_window)
            if pruned_chars > 0 and hasattr(session, "save"):
                session.save()
        except Exception:
            pass
        try:
            squeezed, removed_est, _audit = compact_messages(
                messages,
                target,
                context_window=ctx_window,
            )
        except Exception:  # noqa: BLE001
            return messages
        if not force:
            self._mark_proactive_compact(session)
        self._log_event(
            "COMPACTION_DONE",
            session_id=getattr(session, "hash", ""),
            forced=force,
            shrink=shrink,
            kept_msgs=len(squeezed),
            removed=max(0, len(messages) - len(squeezed)),
            removed_tokens_est=removed_est,
            target_tokens=target,
        )
        after_tokens = self._wire_tokens(session, squeezed)
        if not force:
            self._record_squeeze_progress(
                session,
                messages,
                squeezed,
                before_chars=before_chars,
            )
        self._announce_compaction(
            session,
            trigger=trigger or ("forced" if force else "threshold"),
            before_tokens=before_tokens,
            after_tokens=after_tokens,
            audit=_audit,
        )
        return squeezed

    def _compaction_target(
        self,
        session: Any,
        messages: list[dict[str, Any]],
        *,
        force: bool = False,
        shrink: bool = False,
    ) -> int | None:
        """Return the compaction target-token budget if a squeeze is due, else
        ``None``. Single source of truth for the threshold gate, shared by the
        mechanical :meth:`_maybe_compact` and the LLM :meth:`_maybe_compact_llm`.

        ``shrink=True`` — the model explicitly called ``compact_context``:
        the caller has decided a squeeze is due, so the threshold gate is
        skipped, but the target is sized off CURRENT usage
        (half, like manual /compact) instead of the window. A window-sized
        target on a 1M model is ~963K, which removes nothing from a 450K prompt
        — the no-op compaction storm of session d422f7ca7358. The no-progress
        guard still applies, so a squeeze that cannot shrink the working set
        does not re-fire until it grows.
        """
        try:
            from sharedagents_agent.turn_runtime import (
                _COMPACT_SAFETY_BUFFER,
                _approx_tokens,
                _manual_compact_target,
                _msg_chars,
            )
        except ImportError:
            return None

        # Bound model row / session / cfg — see ``_effective_context_window``.
        ctx_window = int(self._effective_context_window(session) or 0)
        if ctx_window <= 0:
            return None

        # Loop-proofing (structural, estimate-independent): if a previous
        # PROACTIVE squeeze for this session freed almost nothing, refuse to
        # squeeze again until the working set has actually GROWN past the size
        # it failed to shrink. A compaction can therefore never re-trigger on
        # a working set it already couldn't reduce — no matter how wrong any
        # token estimate is. ``force=True`` (a real upstream context-overflow
        # retry) always bypasses this.
        sid = getattr(session, "hash", "") or ""
        if not force and sid:
            stall = getattr(self, "_compaction_stall_chars", {}).get(sid)
            if stall is not None:
                try:
                    if _msg_chars(messages) <= stall:
                        return None
                except Exception:  # noqa: BLE001
                    pass

        if shrink and not force:
            target = _manual_compact_target(
                _approx_tokens(messages),
                ctx_window,
                self._effective_output_budget(session),
            )
            return target if target > 0 else None

        max_tokens = self._effective_output_budget(session)
        window_target = max(1024, ctx_window - max_tokens - _COMPACT_SAFETY_BUFFER)
        if force:
            return window_target

        # Pre-send budget: count the request about to be sent and compact BEFORE
        # it when the minimum useful output no longer fits. The count is exact for
        # a self-served model (its own tokenizer); the margin (``safety``) and the
        # reactive overflow handler cover everything the count cannot see.
        budget = self._presend_budget(session, messages)
        if budget is None or not budget["due"]:
            return None
        # Target: a tail of ``preserveRecentBudget`` tokens survives verbatim.
        # ``_split_for_summary`` keeps ~half the target as tail, so ask for
        # twice the budget; the mechanical squeeze uses the same bound.
        return min(
            window_target,
            max(1024, 2 * _oc_preserve_recent_tokens(ctx_window - budget["out_min"])),
        )

    def _wire_tokens(self, session: Any, messages: list[dict[str, Any]]) -> int:
        """Real token size of ``messages`` as the next request would carry them.

        The same count the pre-send budget uses (own tokenizer when the row has
        one), so compaction cards and the meter agree. Falls back to the
        character estimate.
        """
        try:
            budget = self._presend_budget(session, messages)
            if budget is not None:
                return int(budget["prompt"])
        except Exception:  # noqa: BLE001
            pass
        from sharedagents_agent.turn_runtime import _approx_tokens

        return int(_approx_tokens(messages))

    def _presend_budget(
        self,
        session: Any,
        messages: list[dict[str, Any]],
        *,
        sampling: dict[str, Any] | None = None,
        endpoint: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any] | None:
        """Token budget of the NEXT request: how big the prompt is and how much
        output still fits.

        ``ceiling`` is the ``max_tokens`` to send: the room left in the window
        after this prompt (minus a small safety margin), capped at the model's
        output limit ``out_cap``. An empty context therefore allows the full
        limit and a filling one shrinks it. ``due`` is true when even
        ``out_min`` (the least output worth sending) no longer fits, i.e. the
        session must compact first. Returns ``None`` when the window is unknown.
        """
        from sharedagents_model import token_count

        ctx = int(self._effective_context_window(session) or 0)
        if ctx <= 0:
            return None
        sid = getattr(session, "hash", "") or ""
        endpoint = endpoint or getattr(session, "endpoint", "") or getattr(self.cfg, "upstream_endpoint", "") or ""
        model_id = model or getattr(session, "model", "") or ""
        if not model_id or str(model_id).lower() in ("auto", "ask", "plan"):
            model_id = getattr(self.cfg, "upstream_model", "") or model_id
        wire_messages: list[Any] = list(messages)
        tools: Any = None
        try:
            from sharedagents_gateway.worker_request import _build_worker_request

            payload = _build_worker_request(
                SimpleNamespace(cfg=self.cfg),
                session,
                list(messages),
                allowed_tool_names=self._round_allowed_tools(session),
                sampling=sampling,
                endpoint_override=endpoint or None,
                model_override=model or None,
            )
            wire_messages = list(payload.get("messages") or wire_messages)
            tools = payload.get("tools") or None
        except Exception:  # noqa: BLE001 — count the raw messages instead
            pass
        try:
            row = self.cfg.model_row_for_endpoint(endpoint, model_id) or {}
        except Exception:  # noqa: BLE001
            row = {}
        tokenizer = token_count.tokenizer_for_row(row)
        raw = token_count.count_payload_tokens(wire_messages, tools, tokenizer)
        prompt = raw
        if tokenizer is None:
            # No tokenizer (hosted row): scale the character estimate by how far
            # it missed on the last billed round.
            prompt = int(raw * self._presend_ratio.get(sid, 1.0))
        if tokenizer is not None:
            # The template adds a roughly constant amount we cannot see (tool
            # boilerplate): learn it from the last billed round.
            prompt = raw + self._presend_offset.get(sid, 0)
        out_cap = int(self._effective_output_budget(session))
        safety = max(1024, ctx // 64)
        out_min = min(out_cap, max(4096, ctx // 16))
        room = ctx - prompt - safety
        return {
            "prompt": prompt,
            "raw_prompt": raw,
            "window": ctx,
            "out_cap": out_cap,
            "out_min": out_min,
            "safety": safety,
            "room": room,
            "ceiling": max(256, min(out_cap, room)),
            "exact": tokenizer is not None,
            "due": room < out_min,
        }

    def _mark_proactive_compact(self, session: Any) -> None:
        """Record turn count after a proactive compact (min-turns thrash gate)."""
        sid = getattr(session, "hash", "") or ""
        if not sid:
            return
        self._last_proactive_compact_turn[sid] = len(getattr(session, "turns", None) or [])

    def _record_squeeze_progress(
        self,
        session: Any,
        before: list[dict[str, Any]],
        after: list[dict[str, Any]],
        *,
        before_chars: int | None = None,
    ) -> None:
        """Arm/disarm the proactive-compaction loop guard after a squeeze.

        If a squeeze freed fewer than :data:`_MIN_COMPACTION_CHARS_FREED`
        characters it made no useful progress — re-running it on the same
        working set would just loop. Record the pre-squeeze size so
        :meth:`_compaction_target` refuses further PROACTIVE squeezes until
        the working set grows past it. A squeeze that DID free enough clears
        the guard. This is the estimate-independent backstop: even if a token
        estimate is wildly wrong, compaction cannot fire twice on a working
        set it cannot shrink.
        """
        sid = getattr(session, "hash", "") or ""
        if not sid:
            return
        try:
            from sharedagents_agent.turn_runtime import _msg_chars

            before_n = int(before_chars) if before_chars is not None else _msg_chars(before)
            freed = before_n - _msg_chars(after)
        except Exception:  # noqa: BLE001
            return
        if freed < _MIN_COMPACTION_CHARS_FREED:
            self._compaction_stall_chars[sid] = before_n
        else:
            self._compaction_stall_chars.pop(sid, None)

    async def _maybe_compact_llm(
        self,
        session: Any,
        messages: list[dict[str, Any]],
        *,
        force: bool = False,
        shrink: bool = False,
        trigger: str = "",
        sharedagents_sid: str | None = None,
    ) -> list[dict[str, Any]]:
        """Hermes-style LLM context compaction (with mechanical fallback).

        ``sharedagents_sid`` is the cancel-flag key (``_cancelled``). The round loop
        threads it down explicitly — DON'T re-derive it from ``session.hash``
        deep in the summariser, where a stale/empty hash would silently make a
        mid-summary ``/abort`` un-interruptible. Falls back to ``session.hash``
        only for direct callers (tests) that don't pass it.

        When a squeeze is due, summarize the OLD span of the conversation with
        the model into a compact running memo and keep ``[system, memo, *tail]``
        — so the small model never loses the thread (the dominant cause of
        mid-task false-positive loops / abrupt failures on qwen3.6-27b). Any
        failure (disabled, no clean split, summarizer HTTP/timeout/empty) falls
        back to the existing mechanical :meth:`_maybe_compact` — never worse
        than today. A failed summarizer arms a short cooldown so a broken
        endpoint doesn't burn an LLM round every turn.
        """
        trigger = trigger or ("tool" if shrink else ("forced" if force else "threshold"))

        def _mechanical() -> list[dict[str, Any]]:
            return self._maybe_compact(
                session,
                messages,
                force=force,
                shrink=shrink,
                trigger=trigger,
            )

        if not self._llm_compaction_enabled() or self._compact_llm_on_cooldown():
            return _mechanical()

        target = self._compaction_target(
            session,
            messages,
            force=force,
            shrink=shrink,
        )
        if target is None:
            return messages  # not due

        # Single guarded attempt: ANYTHING that goes wrong (no clean split,
        # transcript-render edge case, summarizer HTTP/timeout/empty, final
        # squeeze failure) raises out and lands in the fallback below — the
        # turn is NEVER allowed to break on compaction. A genuine summarizer
        # failure (vs. a benign "no clean split") arms a cooldown so a broken
        # endpoint doesn't burn an LLM round every subsequent squeeze.
        try:
            new_messages = await self._compact_via_summary(
                session,
                messages,
                target,
                force=force,
                sharedagents_sid=sharedagents_sid,
                trigger=trigger,
            )
        except _NoSummaryCompaction:
            # Benign: nothing useful to summarize. Mechanical, no cooldown.
            return _mechanical()
        except _CompactionAborted:
            # Hard abort landed mid-summary — bail to the fast mechanical
            # squeeze (no cooldown, not an endpoint failure). The round loop's
            # post-compaction cancel checkpoint ends the turn on the next line.
            return _mechanical()
        except Exception as exc:  # noqa: BLE001
            self._arm_compact_llm_cooldown()
            self._log_event(
                "COMPACTION_LLM_FALLBACK",
                session_id=getattr(session, "hash", ""),
                error=str(exc)[:200],
            )
            return _mechanical()
        return new_messages

    async def _await_unless_hard_cancel(
        self,
        coro: Awaitable[Any],
        sharedagents_sid: str,
        *,
        poll_s: float = 0.2,
    ) -> Any:
        """Await ``coro`` but abandon it the moment a hard abort flips
        ``_cancelled[sharedagents_sid]``.

        Engine-side awaits (the compaction summariser HTTP call) run IN THIS
        PROCESS — a force ``/abort`` SIGKILLs the worker subprocess, not this
        coroutine — so the round loop's ``_cancelled`` checkpoint can't land
        until the await returns on its own. For a multi-hundred-K-token
        summary that is minutes, during which the operator's repeated aborts
        appear ignored. This races the ``coro`` against
        the cancel flag: if the flag flips first we cancel the task and raise
        :class:`_CompactionAborted`; otherwise we return the result.

        Graceful ``/stop`` is intentionally NOT honoured here — by contract it
        lets the in-flight op finish and ends the turn at the next round
        boundary. Only the hard kill interrupts mid-await.
        """
        task = asyncio.ensure_future(coro)
        try:
            while True:
                if self._cancelled.get(sharedagents_sid):
                    task.cancel()
                    raise _CompactionAborted()
                done, _ = await asyncio.wait({task}, timeout=poll_s)
                if task in done:
                    return task.result()
        finally:
            # Drain the child task on every exit path (abort, outer
            # cancellation, exception) so it can't leak as "Task was destroyed
            # but it is pending". The drain must NOT mask a cancellation of THIS
            # task that arrives mid-drain: ``shield`` keeps the child unwinding
            # while letting our own ``CancelledError`` surface. We then suppress
            # only when it was the CHILD that finished cancelled
            # (``task.cancelled()``); a CancelledError raised while the child is
            # still pending is OURS → re-raise so ``run_turn`` teardown is never
            # swallowed. The originating exception (``_CompactionAborted`` or an
            # outer cancel from the try) otherwise propagates unchanged.
            if not task.done():
                task.cancel()
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    if not task.cancelled():
                        raise
                except Exception:  # noqa: BLE001 — child's own teardown error
                    pass

    async def _compact_via_summary(
        self,
        session: Any,
        messages: list[dict[str, Any]],
        target: int,
        *,
        force: bool,
        sharedagents_sid: str | None = None,
        trigger: str = "threshold",
    ) -> list[dict[str, Any]]:
        """Summarize the old span and rebuild ``[system, *tail]`` with the memo
        merged into the first kept user turn. Raises :class:`_NoSummaryCompaction`
        when there is nothing to summarize (caller → mechanical, no cooldown);
        raises other exceptions on a real failure (caller → mechanical + cooldown).

        The summarizer runs as a visible ``compaction`` child session (the
        same helper the manual /compact uses): spawned before the LLM call so
        the Agent Monitor shows it live, finished with the stats + summary.
        A rebuild that does not shrink the prompt is discarded — no count, no
        event, original messages returned.
        """
        from sharedagents_agent.turn_runtime import (
            _approx_tokens,
            _msg_chars,
            _render_messages_as_transcript,
            _split_for_summary_oc,
            _summarize_context_span,
            compact_messages,
        )

        # cfg must expose the upstream wiring. Some engine instances (tests,
        # lightweight shims) carry a SimpleNamespace cfg without it — that's a
        # STRUCTURAL gap, not a transient endpoint failure, so treat it as a
        # benign "no summary" (mechanical fallback, NO cooldown) rather than a
        # generic exception that would arm the cooldown pointlessly.
        # Session endpoint wins (per-session upstream override), matching
        # _distill_recovery — don't silently send summary calls elsewhere.
        endpoint = getattr(session, "endpoint", "") or getattr(
            self.cfg,
            "upstream_endpoint",
            "",
        )
        model = getattr(session, "model", "") or getattr(
            self.cfg,
            "upstream_model",
            "",
        )
        home = getattr(self.cfg, "home", None)
        headers_fn = getattr(self.cfg, "upstream_headers", None)
        if not endpoint or not model or home is None or not callable(headers_fn):
            raise _NoSummaryCompaction()

        # OpenCode ``select``: the newest whole rounds within the preserve
        # budget stay verbatim (possibly none); everything older is summarized.
        split = _split_for_summary_oc(messages, max(1, target // 2))
        if split is None:
            raise _NoSummaryCompaction()
        sys_msg, head, tail = split
        # A head smaller than the no-progress floor is not worth an LLM round
        # (or a visible subagent): such compactions were observed to summarize
        # ONE old message each.
        if _msg_chars(head) < _MIN_COMPACTION_CHARS_FREED:
            raise _NoSummaryCompaction()

        # The "task" anchor = the most recent user message (the live request).
        live_content: Any = None
        task = ""
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                live_content = m.get("content")
                task = self._task_text_from_user_content(live_content)
                break

        transcript = _render_messages_as_transcript(head)
        headers = dict(headers_fn(endpoint, model))
        sid = getattr(session, "hash", "") or ""
        if sid:
            headers.setdefault("X-Session-Id", sid)
        previous_summary = str(getattr(session, "compaction_summary", "") or "")
        # Cancel-flag key: prefer the sid the round loop threaded down
        # explicitly; fall back to ``session.hash`` only for direct callers
        # (tests). They are equal by construction in ``run_turn``
        # (``sharedagents_sid = session.hash``) — the explicit arg just removes the
        # silent "never abort mid-summary" failure mode if the hash is ever
        # empty or normalised differently.
        cancel_sid = sharedagents_sid or sid
        child = self._spawn_compaction_child(session, trigger)
        # Abort-aware await: a hard ``/abort`` mid-summary tears the HTTP call
        # down NOW rather than after the (possibly minutes-long) summary lands.
        # ``_CompactionAborted`` bubbles to ``_maybe_compact_llm`` as a benign
        # no-summary fallback; the round loop's post-compaction cancel checkpoint
        # then ends the turn.
        try:
            summary = await self._await_unless_hard_cancel(
                _summarize_context_span(
                    endpoint=endpoint,
                    model=model,
                    headers=headers,
                    transcript=transcript,
                    task=task,
                    home=home,
                    previous_summary=previous_summary,
                    cfg=self.cfg,
                ),
                cancel_sid,
            )
            if not summary.strip():
                raise RuntimeError("empty compaction summary")
        except BaseException as exc:
            if child is not None:
                child.finish(
                    content="",
                    aborted=isinstance(
                        exc,
                        (_CompactionAborted, asyncio.CancelledError),
                    ),
                    error=str(exc) or type(exc).__name__,
                )
            raise

        # Place the memo at the start of the kept conversation. The tail begins
        # at a round boundary (user OR assistant — agentic turns have a single
        # user message that lives in the summarized head):
        #   * tail starts with a USER turn  → merge the memo INTO it when that
        #     user turn IS the live task; otherwise restore ``task`` verbatim.
        #   * tail starts with an ASSISTANT turn → insert a USER message that
        #     carries the summary AND preserves the live ``task`` verbatim.
        merged_tail = list(tail)
        prefix = [sys_msg] if sys_msg is not None else []
        task_in_tail = live_content is not None and self._user_content_in_messages(
            merged_tail,
            live_content,
            task,
        )
        if merged_tail and merged_tail[0].get("role") == "user":
            first = dict(merged_tail[0])
            if live_content is not None and not task_in_tail:
                first["content"] = self._merge_summary_with_user_content(
                    summary,
                    live_content,
                )
            else:
                first["content"] = self._merge_summary_with_user_content(
                    summary,
                    first.get("content"),
                )
            merged_tail[0] = first
            new_messages = prefix + merged_tail
        else:
            if live_content is not None:
                user_content = self._merge_summary_with_user_content(
                    summary,
                    live_content,
                )
            else:
                user_content = summary
            new_messages = prefix + [{"role": "user", "content": user_content}] + merged_tail

        # Final size guard: a still-oversized result (rare — tail is sized to
        # ~half target and the memo is small) gets a mechanical squeeze. If
        # that itself fails it raises → caller falls back to mechanical on the
        # original messages (so we never return an oversized prompt).
        if _approx_tokens(new_messages) > target:
            ctx_window = int(getattr(session, "context_window", 0) or getattr(self.cfg, "upstream_max_context", 0) or 0)
            new_messages, _r, _a = compact_messages(
                new_messages,
                target,
                context_window=ctx_window,
            )

        # Real token sizes (the same count the budget uses), not the character
        # estimate: the card must agree with the meter the operator was watching.
        before_tokens = self._wire_tokens(session, messages)
        after_tokens = self._wire_tokens(session, new_messages)
        if after_tokens >= before_tokens:
            # The memo is no smaller than what it replaced: keep the original
            # context, count nothing, and arm the no-progress guard.
            if child is not None:
                child.finish(
                    content=("compaction made no reduction — kept the original context"),
                )
            if not force:
                self._record_squeeze_progress(session, messages, messages)
            self._log_event(
                "COMPACTION_LLM_NO_REDUCTION",
                session_id=getattr(session, "hash", ""),
                head_msgs=len(head),
                summary_chars=len(summary),
            )
            return messages

        from sharedagents_agent.turn_runtime import _merge_compaction_summary

        merged_summary = _merge_compaction_summary(previous_summary, summary)
        next_compaction_count = int(getattr(session, "compaction_count", 0) or 0) + 1
        try:
            from sharedagents_agent.compaction import prune_session_tool_outputs

            ctx_window = int(getattr(session, "context_window", 0) or getattr(self.cfg, "upstream_max_context", 0) or 0)
            prune_session_tool_outputs(session, context_window=ctx_window)
        except Exception:  # noqa: BLE001
            pass
        self._mark_proactive_compact(session)
        announced = self._announce_compaction(
            session,
            trigger=trigger,
            before_tokens=before_tokens,
            after_tokens=after_tokens,
            summary=summary,
            child=child,
            compaction_count=next_compaction_count,
        )
        # Commit the rolling summary/count only after the parent lifecycle
        # event was durably written. A stale-session reconciler (or an
        # operator abort) can close the parent turn while the in-process
        # summarizer is still returning; persisting first used to leave
        # ``compaction_count=2`` with no second ``compaction`` event/card.
        if announced:
            try:
                session.compaction_summary = merged_summary
                session.compaction_count = next_compaction_count
                session.compaction_turn = len(getattr(session, "turns", None) or [])
                session.save()
            except Exception:  # noqa: BLE001
                pass

        self._log_event(
            "COMPACTION_LLM_DONE",
            session_id=getattr(session, "hash", ""),
            forced=force,
            head_msgs=len(head),
            kept_msgs=len(new_messages),
            summary_chars=len(summary),
            target_tokens=target,
        )
        if not force:
            self._record_squeeze_progress(session, messages, new_messages)
        return new_messages

    async def _manual_compact_via_summary(
        self,
        session: Any,
        messages: list[dict[str, Any]],
        *,
        sharedagents_sid: str | None = None,
    ) -> tuple[list[dict[str, Any]] | None, str]:
        """LLM-first MANUAL /compact (opencode V2 parity): summarize the OLD span.

        The manual ``compact_context`` direct-dispatch path (/compact, WebUI 🗜)
        replaces the old-chunk mechanical squeeze with the hidden compaction
        agent: the older messages are rendered as a transcript, summarized in
        isolation (temp 0.1, thinking budget 1024), and the rebuilt wire list
        becomes ``[system, memo, *tail]`` with the tail verbatim — opencode's
        compaction process. The AUTO path (``_maybe_compact_llm`` /
        ``_compact_via_summary``) is untouched.

        Manual semantics vs the AUTO path:

        * the split budget is ``_manual_compact_target`` — HALF the current
          usage, capped by the safe window — NOT ``_compaction_target``'s
          ``compact_at_frac`` threshold gate; a manual /compact ALWAYS bites,
          even at low usage;
        * the AUTO cooldown / min-turns thrash gates do not apply;
        * ``_llm_compaction_enabled`` (``cfg.llm_compaction``) is still
          respected — disabled → mechanical;
        * the transcript is capped window-aware (``min(240_000, 0.6 ×
          ctx_window × 3)`` chars ≈ 0.6 × window tokens at ~3 chars/token) so
          the summarizer prompt fits a small window; if even the STRIPPED head
          cannot fit alongside the template overhead / thinking budget /
          bounded summary output, degrade to mechanical (opencode raises
          ContextOverflowError; we degrade — same outcome class, safer).

        Returns ``(new_messages, memo)``:

        * ``(None, "")`` → the caller runs the existing mechanical squeeze
          (fallback on ANY failure: disabled, no clean split, window too
          small, summarizer HTTP/timeout/empty, hard abort mid-summary);
        * ``(list, memo)`` → success; ``memo`` is the WRAPPED fresh summary
          (``_wrap_compact_summary`` output — what ``_summarize_context_span``
          returns) the caller surfaces in the turn body and persists via the
          stats contract.

        Never raises; never persists (the direct-dispatch caller owns the
        stats contract); never touches the AUTO loop-guards
        (``_mark_proactive_compact`` / ``_record_squeeze_progress``).
        """
        if not self._llm_compaction_enabled():
            self._log_event(
                "COMPACTION_LLM_FALLBACK",
                session_id=getattr(session, "hash", ""),
                reason="manual llm compaction disabled (cfg.llm_compaction)",
            )
            return None, ""

        ctx_window = int(getattr(session, "context_window", 0) or getattr(self.cfg, "upstream_max_context", 0) or 0)
        if ctx_window <= 0:
            return None, ""
        from sharedagents_agent.turn_runtime import (
            _COMPACT_MAX_TRANSCRIPT_CHARS,
            _COMPACT_THINKING_BUDGET,
            _approx_tokens,
            _manual_compact_target,
            _render_messages_as_transcript,
            _split_for_summary,
            _summarize_context_span,
            compact_messages,
        )

        before_tokens = _approx_tokens(messages)
        out_budget = self._effective_output_budget(session)
        target = _manual_compact_target(before_tokens, ctx_window, out_budget)
        if target <= 0:
            return None, ""

        # cfg must expose the upstream wiring (mirror ``_compact_via_summary``):
        # a structural gap → benign no-summary (mechanical, no cooldown).
        endpoint = getattr(session, "endpoint", "") or getattr(
            self.cfg,
            "upstream_endpoint",
            "",
        )
        model = getattr(session, "model", "") or getattr(
            self.cfg,
            "upstream_model",
            "",
        )
        home = getattr(self.cfg, "home", None)
        headers_fn = getattr(self.cfg, "upstream_headers", None)
        if not endpoint or not model or home is None or not callable(headers_fn):
            return None, ""

        split = _split_for_summary(messages, target)
        if split is None:
            return None, ""  # nothing worth summarizing → mechanical
        sys_msg, head, tail = split

        # The "task" anchor = the most recent user message (mirror
        # ``_compact_via_summary``).
        live_content: Any = None
        task = ""
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                live_content = m.get("content")
                task = self._task_text_from_user_content(live_content)
                break

        # Window-aware transcript cap: min(240_000, 0.6 × ctx_window × 3
        # chars) ≈ 0.6 × window tokens at ~3 chars/token, so the summary
        # prompt fits the summarizer's OWN window (opencode's prompt-fit
        # check); the global 240K cap still bounds huge windows.
        max_chars = min(
            _COMPACT_MAX_TRANSCRIPT_CHARS,
            int(0.6 * ctx_window * 3),
        )
        transcript = _render_messages_as_transcript(head, max_chars=max_chars)
        # Fit guard: even the STRIPPED transcript must fit alongside the
        # prompt template (~1K tokens), the thinking budget, and the bounded
        # summary output (~4K — opencode's ≤4096 summarizer cap). A window
        # that cannot carry the head at all degrades to mechanical instead of
        # 400-ing the summarizer round.
        transcript_tokens = _approx_tokens(
            [{"role": "user", "content": transcript}],
        )
        if transcript_tokens + 1024 + _COMPACT_THINKING_BUDGET + 4096 >= ctx_window:
            return None, ""

        headers = dict(headers_fn(endpoint, model))
        sid = getattr(session, "hash", "") or ""
        if sid:
            headers.setdefault("X-Session-Id", sid)
        previous_summary = str(getattr(session, "compaction_summary", "") or "")
        cancel_sid = sharedagents_sid or sid
        try:
            # Abort-aware await (same machinery as the AUTO path): a hard
            # ``/abort`` mid-summary tears the HTTP call down NOW rather than
            # after the (possibly minutes-long) summary lands; the caller then
            # completes the fast mechanical squeeze and ends the turn.
            summary = await self._await_unless_hard_cancel(
                _summarize_context_span(
                    endpoint=endpoint,
                    model=model,
                    headers=headers,
                    transcript=transcript,
                    task=task,
                    home=home,
                    previous_summary=previous_summary,
                    cfg=self.cfg,
                ),
                cancel_sid,
            )
        except _CompactionAborted:
            return None, ""
        if not summary.strip():
            return None, ""  # empty memo → mechanical

        # Rebuild ``[system] + memo + tail`` — mirror ``_compact_via_summary``'s
        # rebuild so the wire shape stays vLLM-legal: the memo is merged into
        # the first kept user turn when the tail starts with one, else inserted
        # as its own user message; the tail is kept verbatim.
        merged_tail = list(tail)
        prefix = [sys_msg] if sys_msg is not None else []
        task_in_tail = live_content is not None and self._user_content_in_messages(
            merged_tail,
            live_content,
            task,
        )
        if merged_tail and merged_tail[0].get("role") == "user":
            first = dict(merged_tail[0])
            if live_content is not None and not task_in_tail:
                first["content"] = self._merge_summary_with_user_content(
                    summary,
                    live_content,
                )
            else:
                first["content"] = self._merge_summary_with_user_content(
                    summary,
                    first.get("content"),
                )
            merged_tail[0] = first
            new_messages = prefix + merged_tail
        else:
            if live_content is not None:
                user_content = self._merge_summary_with_user_content(
                    summary,
                    live_content,
                )
            else:
                user_content = summary
            new_messages = prefix + [{"role": "user", "content": user_content}] + merged_tail

        # Final size guard: a still-oversized rebuild (rare — the tail is
        # sized to ~half the manual target and the memo is small) gets a
        # mechanical squeeze to the manual target; never return an oversized
        # prompt.
        if _approx_tokens(new_messages) > target:
            new_messages, _r, _a = compact_messages(
                new_messages,
                target,
                context_window=ctx_window,
            )

        self._log_event(
            "COMPACTION_LLM_DONE",
            session_id=sid,
            manual=True,
            head_msgs=len(head),
            kept_msgs=len(new_messages),
            summary_chars=len(summary),
            target_tokens=target,
        )
        return new_messages, summary

    def _llm_compaction_enabled(self) -> bool:
        """Whether Hermes-style LLM compaction is on (default ON; env/cfg off)."""
        env = os.environ.get("SHAREDAGENTS_LLM_COMPACTION")
        if env is not None:
            return env.strip().lower() not in ("0", "false", "no", "off")
        return bool(getattr(self.cfg, "llm_compaction", True))

    def _compact_llm_on_cooldown(self) -> bool:
        until = getattr(self, "_compact_llm_cooldown_until", 0.0)
        return time.monotonic() < until

    def _arm_compact_llm_cooldown(self, seconds: float = 120.0) -> None:
        """After a summarizer failure, skip LLM compaction for ``seconds`` so a
        broken endpoint doesn't burn an LLM round on every subsequent squeeze."""
        self._compact_llm_cooldown_until = time.monotonic() + seconds

    # -----------------------------------------------------------------
    # Visible compaction — automatic compactions surface exactly like the
    # manual /compact: a ``compaction`` child session (Agent Monitor + the
    # parent's ``subagent_spawned``) finished with its stats + summary, plus
    # a durable turn-scoped ``compaction`` event on the parent log (forwarded
    # on SSE, replayed as a history ``notice``/``compaction`` segment, and
    # rendered by the TUI as ``CompactionNotice``).
    # Only a compaction that actually shrank the prompt is announced.
    # -----------------------------------------------------------------

    _COMPACTION_TRIGGER_LABELS = {
        "budget": "turn token budget",
        "threshold": "context threshold",
        "overflow": "context overflow",
        "tool": "requested by the agent",
        "forced": "forced",
    }

    def _compaction_engine_label(self) -> str:
        """The engine that RUNS in-process compaction (this one) — never the
        session header's engine, which can be stale after an agent switch."""
        kind = getattr(self, "kind", None)
        return str(getattr(kind, "value", kind) or "local_llm")

    def _spawn_compaction_child(self, session: Any, trigger: str) -> Any:
        """Open the visible ``compaction`` child — the manual /compact helper."""
        try:
            from sharedagents_agent.compaction import spawn_visible_compaction_subagent
        except ImportError:
            return None
        label = self._COMPACTION_TRIGGER_LABELS.get(trigger, trigger or "auto")
        return spawn_visible_compaction_subagent(
            session,
            user_prompt=f"Auto-compact the parent session context ({label}).",
            engine=self._compaction_engine_label(),
        )

    def _announce_compaction(
        self,
        session: Any,
        *,
        trigger: str,
        before_tokens: int,
        after_tokens: int,
        audit: dict[str, Any] | None = None,
        summary: str = "",
        child: Any = None,
        reason: str = "compacted",
        compaction_count: int | None = None,
    ) -> bool:
        """Surface a compaction that ACTUALLY shrank the prompt.

        Finishes ``child`` (spawning one first for the instant mechanical
        paths) with the manual-/compact body — stats line + summary — and
        writes the durable ``compaction`` event. When the prompt did not
        shrink, returns False without emitting the parent event or spawning a
        new child; an already-running summarizer child is closed with a
        no-reduction result. The return value is True only when the parent
        lifecycle event was written; callers must not commit a count for an
        announcement that lost its open turn. Never raises.
        """
        try:
            from sharedagents_agent.turn_runtime import _format_compaction_summary

            before_t = int(before_tokens)
            after_t = int(after_tokens)
        except Exception:  # noqa: BLE001
            return False
        removed = before_t - after_t
        if removed <= 0:
            if child is not None:
                child.finish(
                    content=("compaction made no reduction — kept the original context"),
                )
            return False
        audit = audit if isinstance(audit, dict) else {}
        memo = str(summary or "").strip()
        stats_line = _format_compaction_summary(removed, audit, after_t)
        if child is None:
            child = self._spawn_compaction_child(session, trigger)
        payload = {
            "reason": reason,
            "trigger": trigger,
            "phase": "done",
            "before_tokens": before_t,
            "after_tokens": after_t,
            "removed_tokens": removed,
            "new_prompt_tokens": after_t,
            "truncated_tool_call_ids": list(
                audit.get("truncated_tool_call_ids") or [],
            ),
            "dropped_tool_call_ids": list(
                audit.get("dropped_tool_call_ids") or [],
            ),
            "dropped_turns": int(audit.get("dropped_turns") or 0),
            "summary_chars": len(memo),
            "summary": memo[:_COMPACTION_EVENT_SUMMARY_MAX_CHARS],
            "child_session_id": str(getattr(child, "child_sid", "") or ""),
            "compaction_count": int(
                getattr(session, "compaction_count", 0) if compaction_count is None else compaction_count
            ),
            "context_window": int(self._effective_context_window(session) or 0),
        }
        emitted = self._emit_compaction_event(session, payload)
        if child is not None:
            child.finish(
                content=stats_line + ("\n\n" + memo if memo else ""),
                aborted=not emitted,
                error=("parent turn is no longer open" if not emitted else None),
            )
        self._log_event(
            "COMPACTION_ANNOUNCED",
            session_id=getattr(session, "hash", ""),
            trigger=trigger,
            before_tokens=before_t,
            after_tokens=after_t,
            removed_tokens=removed,
            child_session_id=payload["child_session_id"],
            event_emitted=emitted,
        )
        return bool(emitted)

    def _emit_compaction_event(
        self,
        session: Any,
        payload: dict[str, Any],
    ) -> bool:
        """Write the ``compaction`` lifecycle event into the parent's OPEN
        turn. Turn-scoped via the writer's own open-turn state (the durable
        ``queue_msg_injected`` pattern): the history fold drops events with no
        ``turn_idx`` and a wrong one would split the turn, so with no open
        turn this is a no-op. Never raises."""
        sid = str(getattr(session, "hash", "") or "")
        root = getattr(session, "_storage_root", None)
        if not sid or root is None:
            return False
        try:
            from sharedagents_session.session_writer import (
                get_session_writer,
                release_session_writer,
            )

            writer = get_session_writer(Path(root), sid, pin=True)
        except Exception:  # noqa: BLE001
            return False
        try:
            turn_idx = writer.current_turn_idx()
            if turn_idx is None:
                return False
            writer.lifecycle(
                kind="compaction",
                turn_idx=int(turn_idx),
                payload=payload,
            )
            return True
        except Exception:  # noqa: BLE001
            return False
        finally:
            release_session_writer(writer)

    def _model_requested_compaction(
        self,
        session: Any,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str]:
        """The model's own ``compact_context`` tool call: a shrink-sized
        mechanical squeeze, announced like every automatic compaction, with a
        tool result that reports what happened (it used to answer "context
        compacted" even when nothing was removed)."""
        from sharedagents_agent.turn_runtime import (
            _format_compaction_summary,
        )

        before = self._wire_tokens(session, messages)
        squeezed = self._maybe_compact(
            session,
            messages,
            shrink=True,
            trigger="tool",
        )
        after = self._wire_tokens(session, squeezed)
        if after >= before:
            return messages, "compaction skipped: nothing to trim"
        return squeezed, _format_compaction_summary(before - after, {}, after)

    def _thinking_capped_by_context(self, session: Any) -> bool:
        """True when the last round's output cap was shrunk by a nearly full window.

        A thinking-starved round under the full output limit means the model
        really reasoned that long; under a shrunken ceiling it ran out of room,
        and another round on the same prompt would hit the same wall.
        """
        last = self._presend_last.get(getattr(session, "hash", "") or "") or {}
        ceiling, out_cap = int(last.get("ceiling") or 0), int(last.get("out_cap") or 0)
        return bool(ceiling and out_cap and ceiling < out_cap)

    async def _compact_for_starved_thinking(
        self,
        session: Any,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        """LLM summary compaction instead of the reasoning-only nudge.

        Returns the compacted messages, or ``None`` when nothing could be freed
        (the caller then nudges as before).
        """
        if not (self._llm_compaction_enabled() and not self._compact_llm_on_cooldown()):
            return None
        from sharedagents_agent.turn_runtime import _approx_tokens as _tok

        ctx_window = int(self._effective_context_window(session) or 0)
        if ctx_window <= 0:
            return None
        out_budget = int(self._effective_output_budget(session))
        usable = _oc_usable(ctx_window, min(out_budget, max(1024, ctx_window // 4)))
        tail_target = max(1024, 2 * _oc_preserve_recent_tokens(usable))
        try:
            compacted = await self._compact_via_summary(
                session,
                messages,
                tail_target,
                force=True,
                sharedagents_sid=getattr(session, "hash", "") or None,
                trigger="thinking-starved",
            )
        except (_NoSummaryCompaction, _CompactionAborted):
            return None
        except Exception as exc:  # noqa: BLE001
            self._arm_compact_llm_cooldown()
            self._log_event(
                "COMPACTION_LLM_FALLBACK",
                session_id=getattr(session, "hash", ""),
                error=str(exc)[:200],
            )
            return None
        if _tok(compacted) >= _tok(messages):
            return None
        self._log_event(
            "COMPACTION_THINKING_STARVED",
            session_id=getattr(session, "hash", ""),
            kept_msgs=len(compacted),
            target_tokens=tail_target,
        )
        return compacted

    async def _reactive_compact_on_overflow(
        self,
        session: Any,
        messages: list[dict[str, Any]],
        upstream_error: str,
        *,
        classification: str = "",
    ) -> tuple[list[dict[str, Any]], bool]:
        """Reactive squeeze after a context-overflow 400.

        Triggered when the error classifies as a context-length
        overflow. The caller passes the already-resolved
        ``classification`` string (which may have been rewritten by
        an ``UpstreamError`` hook — see :meth:`run_turn`), so a
        plugin can:

        * Promote a normally-unrecognised model-specific overflow
          string to ``"upstream context-length overflow"`` (or the
          equivalent hook-claimed marker) so the reactive squeeze
          DOES fire.
        * Suppress reactive compaction by rewriting the
          classification to anything OTHER than the overflow marker
          (e.g. ``"hook_overridden"``) — the turn then surfaces the
          original error to the user instead of silently retrying.

        Recognises both the legacy spelling (``"context_length"``,
        kept for back-compat with callers that pass the short tag)
        and the modern ``_classify_upstream_error`` return string
        (``"upstream context-length overflow"``). Falls back to
        re-running the classifier on the raw message only when
        ``classification`` is empty, so a caller that wasn't updated
        for round 4 still gets the routing decision.

        Returns ``(messages, retried)`` — ``retried=True`` means we
        squeezed and the caller should re-spawn the worker.
        """
        if not classification:
            classification = self._classify_upstream_error(upstream_error)
        overflow_markers = (
            "context_length",
            "upstream context-length overflow",
        )
        if classification not in overflow_markers:
            return messages, False
        try:
            from sharedagents_agent.turn_runtime import (
                compact_messages,
                parse_overflow_numbers,
            )
        except ImportError:
            return messages, False

        # Prefer the numbers the upstream actually reported. When the
        # user has cfg.upstream_max_context mis-configured (e.g. 96000
        # local default against a 262144 hosted model) the upstream's
        # own "maximum context length is N tokens / requested M output
        # tokens" line is authoritative — using it lets us trim to the
        # right budget on the first try instead of squeezing to the
        # wrong target and 400-ing again. Fall back to cfg only when the
        # error body doesn't carry the numbers.
        parsed = parse_overflow_numbers(upstream_error)
        cfg_ctx = int(self._effective_context_window(session) or 0)
        # What the round actually asked for (the wire's own resolution),
        # not the agent knob, which is now only the cap.
        cfg_out = int(self._effective_output_budget(session))
        ctx_window = int(parsed.get("model_max") or cfg_ctx)
        out_budget = int(parsed.get("requested_max_tokens") or cfg_out)
        if ctx_window <= 0:
            return messages, False
        # Heal a stale larger session.context_window so the NEXT proactive
        # compact sizes against the upstream that just 400'd us.
        try:
            sess_cw = int(getattr(session, "context_window", 0) or 0)
            if sess_cw <= 0 or sess_cw > ctx_window:
                session.context_window = ctx_window
        except Exception:  # noqa: BLE001
            pass

        # LLM summary FIRST, with the same tail target as the proactive path
        # (OpenCode: a recent tail of ``clamp(usable/4, 2K, 15K)`` survives, the
        # rest becomes a summary). The mechanical squeeze below only trims old
        # tool results down to just under the limit and keeps no memo, so the
        # next round overflows again: three reactive squeezes in 13 minutes,
        # then the turn died on the retry cap. Any failure falls through to it.
        if self._llm_compaction_enabled() and not self._compact_llm_on_cooldown():
            try:
                from sharedagents_agent.turn_runtime import _approx_tokens as _tok

                _usable = _oc_usable(ctx_window, min(out_budget, max(1024, ctx_window // 4)))
                _tail_target = max(1024, 2 * _oc_preserve_recent_tokens(_usable))
                _llm_msgs = await self._compact_via_summary(
                    session,
                    messages,
                    _tail_target,
                    force=True,
                    sharedagents_sid=getattr(session, "hash", "") or None,
                    trigger="overflow",
                )
                if _tok(_llm_msgs) < _tok(messages):
                    self._log_event(
                        "COMPACTION_REACTIVE_LLM",
                        session_id=getattr(session, "hash", ""),
                        kept_msgs=len(_llm_msgs),
                        target_tokens=_tail_target,
                        input_tokens_reported=parsed.get("input_tokens"),
                    )
                    self._reactive_via_llm[getattr(session, "hash", "") or ""] = True
                    return _llm_msgs, True
            except (_NoSummaryCompaction, _CompactionAborted):
                pass  # nothing to summarize / aborted: mechanical squeeze below
            except Exception as exc:  # noqa: BLE001
                self._arm_compact_llm_cooldown()
                self._log_event(
                    "COMPACTION_LLM_FALLBACK",
                    session_id=getattr(session, "hash", ""),
                    error=str(exc)[:200],
                )

        # Reactive squeeze runs AFTER an overflow 400, so the previous
        # input-size estimate was demonstrably too generous. Leave a
        # bigger safety margin than the proactive path (256) so the
        # retry doesn't immediately 400 again on chat-template overhead
        # the char-count estimator can't see (tool schemas, BOS, role
        # markers, prefix-cache padding).
        safety = 1024
        # Cap the reserved output for the retry target — a 16k–32k
        # max_tokens against a 65k window leaves almost no room to
        # compact into when the prompt is already ~50k.
        out_budget = min(out_budget, max(1024, ctx_window // 4))
        target = max(1024, ctx_window - out_budget - safety)
        try:
            from sharedagents_agent.compaction import prune_session_tool_outputs

            pruned_chars, _ = prune_session_tool_outputs(session, context_window=ctx_window)
            if pruned_chars > 0 and hasattr(session, "save"):
                session.save()
        except Exception:
            pass
        try:
            from sharedagents_agent.turn_runtime import _approx_tokens
        except ImportError:
            return messages, False
        before_est = _approx_tokens(messages)
        before_shown = self._wire_tokens(session, messages)
        reported_in = int(parsed.get("input_tokens") or 0)
        adjusted_target = target
        if reported_in > 0 and before_est > 0:
            if reported_in > before_est:
                # Token estimator significantly undercounted relative to authoritative upstream count.
                # Scale target down proportionally so compact_messages actually squeezes to fit.
                ratio = reported_in / before_est
                adjusted_target = max(1024, int(target / ratio))
            elif before_est <= target and reported_in + out_budget + safety >= ctx_window:
                # Upstream overflowed despite before_est <= target
                overflow_margin = (reported_in + out_budget + safety) - ctx_window
                adjusted_target = max(1024, before_est - max(2048, overflow_margin))
        try:
            squeezed, removed_est, _audit = compact_messages(
                messages,
                adjusted_target,
                context_window=ctx_window,
            )
        except Exception:  # noqa: BLE001
            return messages, False
        after_est = _approx_tokens(squeezed)
        after_shown = self._wire_tokens(session, squeezed)
        self._log_event(
            "COMPACTION_REACTIVE",
            session_id=getattr(session, "hash", ""),
            kept_msgs=len(squeezed),
            removed_tokens_est=removed_est,
            target_tokens=target,
            adjusted_target=adjusted_target,
            model_max=parsed.get("model_max"),
            requested_max_tokens=parsed.get("requested_max_tokens"),
            input_tokens_reported=parsed.get("input_tokens"),
        )

        # If compact_messages made real progress AND we're now under
        # the budget, retry as usual.
        made_progress = after_est < before_est
        if made_progress and (after_est <= adjusted_target or after_est <= target):
            self._announce_compaction(
                session,
                trigger="overflow",
                before_tokens=before_shown,
                after_tokens=after_shown,
                audit=_audit,
                reason="compacted_after_overflow",
            )
            return squeezed, True

        # No progress (or still over budget) — `compact_messages` couldn't
        # squeeze further. This is the production failure mode the user
        # hit: a single oversized live user prompt that protected-tail
        # logic refuses to drop. Pre-fix the engine relooped on the
        # unchanged payload for 76+ rounds against the same prefix hash
        # before the operator cancelled. Fall back to shrinking
        # `max_tokens` so input + new_max_tokens fits, set a per-session
        # override the next worker request honours, and signal one
        # retry. If even that can't carve out ≥256 output tokens, surface
        # the error so the operator sees it instead of an infinite loop.
        reported_input = int(
            parsed.get("input_tokens") or after_est,
        )
        new_max_tokens = ctx_window - reported_input - safety
        if new_max_tokens >= 256:
            try:
                session.max_tokens_override = int(new_max_tokens)
            except Exception:  # noqa: BLE001
                return messages, False
            self._log_event(
                "COMPACTION_REACTIVE_MAX_TOKENS_FALLBACK",
                session_id=getattr(session, "hash", ""),
                ctx_window=ctx_window,
                input_tokens=reported_input,
                old_max_tokens=out_budget,
                new_max_tokens=int(new_max_tokens),
            )
            self._announce_compaction(
                session,
                trigger="overflow",
                before_tokens=before_shown,
                after_tokens=after_shown,
                audit=_audit,
                reason="compacted_after_overflow",
            )
            return squeezed, True
        self._log_event(
            "COMPACTION_REACTIVE_NO_PROGRESS",
            session_id=getattr(session, "hash", ""),
            ctx_window=ctx_window,
            input_tokens=reported_input,
            target_tokens=target,
            attempted_new_max_tokens=int(new_max_tokens),
        )
        return messages, False

    @staticmethod
    def _merge_summary_with_user_content(summary: str, content: Any) -> Any:
        if not summary:
            return content
        if isinstance(content, str):
            return (summary + "\n\n" + content) if content else summary
        if isinstance(content, list):
            parts = [dict(p) if isinstance(p, dict) else p for p in content]
            if parts and isinstance(parts[0], dict) and parts[0].get("type") == "text":
                parts[0]["text"] = summary + "\n\n" + str(parts[0].get("text") or "")
            else:
                parts.insert(0, {"type": "text", "text": summary + "\n\n"})
            return parts
        return content
