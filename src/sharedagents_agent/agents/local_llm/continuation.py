"""ContinuationMixin — extracted from the ``local_llm`` monolith (behaviour-preserving)."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from .state import (
    BACKGROUND_DELEGATE_EMPTY_NUDGE_PROMPT_BODY,
    COLON_CLIFFHANGER_NUDGE_PROMPT_BODY,
    CUTOFF_NUDGE_PROMPT_BODY,
    EMPTY_RESPONSE_NUDGE_PROMPT_BODY,
    NO_TERMINAL_NUDGE_PROMPT_BODY,
    NUDGE_PROMPT_BODY,
    REASONING_ONLY_NUDGE_PROMPT_BODY,
    STREAM_TRUNCATED_NUDGE_PROMPT_BODY,
    THINKING_STARVED_NUDGE_PROMPT_BODY,
    TOOL_CALL_TRUNCATED_NUDGE_PROMPT_BODY,
    TRANSPORT_ERROR_NUDGE_PROMPT_BODY,
    _now_iso,
)


class ContinuationMixin:
    async def _continuation_nudge(
        self,
        messages: list[dict[str, Any]],
        nudge_log: list[dict[str, Any]],
        *,
        kind: str = "silent",
        tail: str | None = None,
        body_override: str | None = None,
        reason_code: str = "",
    ) -> tuple[list[dict[str, Any]], bool]:
        """Inject one of the nudge bodies for a premature-finish round.

        Three variants, chosen by ``kind``:

        * ``"silent"`` — the round produced no content and no tool calls
          (truly empty / whitespace-only). Injects ``NUDGE_PROMPT_BODY``.
          Terse-but-complete answers ("pong", "ok") are NOT silent.
        * ``"cutoff"`` — content present but ends mid-syntax (open
          backticks / fences); ``tail`` is the offending last ~80
          chars. Injects ``CUTOFF_NUDGE_PROMPT_BODY`` formatted with
          the tail.
        * ``"colon"`` — content ends on a colon with no tool calls
          following; ``tail`` is the last ~80 chars. Injects
          ``COLON_CLIFFHANGER_NUDGE_PROMPT_BODY`` formatted with the
          tail.

        ``body_override`` lets a ``PrematureTurnEnd`` hook substitute
        the entire body — the canonical injection point for ECC
        plugins that want per-model / per-context nudge wording. The
        canned bodies are the FALLBACK when no hook overrides.

        Mechanical truncation/cutoff/colon nudges may retry within the
        active turn; the caller's ``classify_turn_end`` budget bounds them.
        A marker is suppressed only when a restored transcript has no local
        nudge audit to prove that the model has already been given a fresh
        continuation request. Idle-empty ``silent`` continues
        (reasoning-only or empty-after-tools) may retry as well.

        ``nudge_log`` is mutated in place with the audit entry.
        """
        if kind == "cutoff":
            body = CUTOFF_NUDGE_PROMPT_BODY.format(tail=tail or "")
            marker = "[sharedagents continuation] Your previous response was cut mid-sentence"
        elif kind == "colon":
            body = COLON_CLIFFHANGER_NUDGE_PROMPT_BODY.format(tail=tail or "")
            marker = "[sharedagents continuation] Your previous response ended on a colon"
        elif kind == "stream_truncated" and reason_code == "transport_error":
            body = TRANSPORT_ERROR_NUDGE_PROMPT_BODY.format(tail=tail or "")
            marker = "[sharedagents continuation] The upstream stream dropped"
        elif kind == "stream_truncated" and reason_code == "no_terminal":
            body = NO_TERMINAL_NUDGE_PROMPT_BODY.format(tail=tail or "")
            marker = "[sharedagents continuation] The stream ended without a terminal"
        elif kind == "stream_truncated" and reason_code == "tool_call_truncated":
            # The cap hit mid tool-call arguments: the call never ran, so
            # there is no sentence to resume (``tail`` names the call).
            body = TOOL_CALL_TRUNCATED_NUDGE_PROMPT_BODY.format(
                tail=tail or "a tool call",
            )
            marker = "[sharedagents continuation] The output-token limit"
        elif kind == "stream_truncated":
            # length (default when reason_code is omitted)
            body = STREAM_TRUNCATED_NUDGE_PROMPT_BODY.format(tail=tail or "")
            marker = "[sharedagents continuation] Token budget ran out"
        elif kind == "silent" and reason_code == "reasoning_only_no_output":
            # Reasoning-only idle stop (RESUME_REASONING_ONLY): same "silent"
            # machinery as the silent-after-tools case, but a distinct body.
            # Silent continues may retry up to the turn cap — do not skip
            # just because the previous user message was already a continue.
            body = REASONING_ONLY_NUDGE_PROMPT_BODY
            marker = None
        elif kind == "silent" and reason_code == "empty_no_output":
            # Completely blank terminal round (no content, no reasoning, no
            # tool call — including a dangling ``tool_calls`` label whose
            # call JSON never parsed). Zero-output guarantee 2026-08-14.
            body = EMPTY_RESPONSE_NUDGE_PROMPT_BODY
            marker = None
        elif kind == "silent" and reason_code == "thinking_starved_length":
            # The final round hit the output-token cap while still reasoning
            # (finish_reason=length, zero content, heavy thinking) — there is
            # no partial answer to resume from, so re-prompt for the answer.
            # Observed: 38,610 reasoning chars, content_chars=0; the
            # turn ended "complete" silently and the operator had to re-prompt.
            body = THINKING_STARVED_NUDGE_PROMPT_BODY
            marker = None
        elif kind == "silent" and reason_code == "empty_after_background_dispatch":
            body = BACKGROUND_DELEGATE_EMPTY_NUDGE_PROMPT_BODY
            marker = None
        else:
            body = NUDGE_PROMPT_BODY
            marker = None
        if body_override:
            # Hook-supplied body wins. Marker stays the canned one so
            # restored-transcript guards below still detect truncation/
            # cutoff/colon markers even when the hook rewrites the body.
            body = body_override

        # A mechanical marker is present in the conversation after the first
        # continuation. Do not mistake that marker for an in-turn duplicate:
        # the next model round can hit the same ``length`` stop (including an
        # empty tail), and the classifier's nudge budget is the loop guard.
        # Keep the legacy marker-only suppression for restored transcripts
        # that have no matching local audit entry.
        if marker is not None:
            for m in reversed(messages):
                if isinstance(m, dict) and m.get("role") == "user":
                    content = str(m.get("content") or "")
                    if marker in content:
                        prior_kind = [
                            entry for entry in nudge_log if isinstance(entry, dict) and entry.get("kind") == kind
                        ]
                        if not prior_kind:
                            # A caller may restore a transcript containing a
                            # mechanical nudge without restoring its audit
                            # log. Preserve the historical marker guard for
                            # that case.
                            return messages, False
                    break
        nudge_log.append({"at": _now_iso(), "kind": kind})
        out = list(messages)
        out.append({"role": "user", "content": body})
        self._log_event("NUDGE_FIRED", kind=kind, reason_code=reason_code)
        return out, True

    async def _force_terminate(
        self,
        messages: list[dict[str, Any]],
        *,
        hook_engine: Any = None,
        hook_ctx: dict[str, Any] | None = None,
        tier: str = "",
        round_idx: int = -1,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Last-ditch "write a closing paragraph and END" round.

        Reference: the ``LOOP_FORCE_TERMINATE`` block referenced in
        acp_bridge.py L2933. Fires at most once per turn
        (``force_terminate_attempted`` guard) after distill recoveries
        are exhausted.

        The ``ForceTerminate`` hook fires BEFORE the canned body so an
        ECC plugin can substitute a model-tailored closing prompt
        (e.g. a Qwen-specific brevity nudge, or a tighter
        "summarize then exit" body). A ``continue: false`` skips
        force-terminate entirely (operator wants the turn to die now).
        """
        DEFAULT_FORCE_TERMINATE_BODY = (
            "[FORCE TERMINATE] Loop budget exhausted. Write a closing "
            "paragraph that summarizes the user-facing answer with what "
            "you have so far, then STOP. Do not call any more tools."
        )
        body = DEFAULT_FORCE_TERMINATE_BODY
        if hook_engine is not None:
            ft_hook = await self._fire_hook(
                hook_engine,
                hook_ctx or {},
                "ForceTerminate",
                {
                    "tier": tier,
                    "round_idx": round_idx,
                },
            )
            if not ft_hook.should_continue:
                return messages, False
            if ft_hook.blocked and ft_hook.block_reason:
                body = ft_hook.block_reason
        out = list(messages)
        out.append({"role": "user", "content": body})
        self._log_event("LOOP_FORCE_TERMINATE")
        return out, True

    # ---- autonomous completion gate (teacher-independent) ----------------

    def _incomplete_todos(self, sharedagents_sid: str) -> list[str]:
        """Contents of the session's TodoWrite items that are NOT done.

        Empty when there are no todos or no resolvable sessions_dir. This is
        the teacher-independent progress signal: the model's OWN plan items
        still pending are strong evidence it stopped short of the task.
        """
        try:
            from sharedagents_agent.tools.runners import (
                _canonical_todo_status,
                load_todos,
                todos_path_for_session,
            )
        except ImportError:
            return []
        sessions_dir = getattr(self.cfg, "sessions_dir", None)
        if sessions_dir is None:
            return []
        try:
            todos = load_todos(todos_path_for_session(sessions_dir, sharedagents_sid))
        except Exception:  # noqa: BLE001
            return []
        out: list[str] = []
        for item in todos or []:
            if _canonical_todo_status(item.get("status")) not in {"done", "cancelled"}:
                c = str(item.get("content") or "").strip()
                if c:
                    out.append(c)
        return out

    def _auto_complete_continuation_enabled(self, sharedagents_sid: str) -> bool:
        """Whether to auto-re-drive on a clean stop with incomplete todos.

        AUTO mode only (ask/plan stops are user-interactive). Default ON;
        disable via env ``SHAREDAGENTS_AUTO_COMPLETE_CONTINUE`` or
        ``cfg.auto_complete_continue``. This is what lets the agent finish a
        multi-step task with the TEACHER OFF — the dominant small-model failure
        is declaring done with plan items still pending.
        """
        mode = self._mode_overrides.get(sharedagents_sid) or str(getattr(self.cfg, "permission_mode", "auto") or "auto")
        if mode != "auto":
            return False
        env = os.environ.get("SHAREDAGENTS_AUTO_COMPLETE_CONTINUE")
        if env is not None:
            return env.strip().lower() not in ("0", "false", "no", "off")
        return bool(getattr(self.cfg, "auto_complete_continue", True))

    # How long an idle-looking turn waits for a just-finished child's result
    # to land in the queue before falling back to the pending-todo nudge.
    ARRIVING_EVENT_WAIT_S = 15.0
    ARRIVING_EVENT_POLL_S = 1.0

    async def _take_arriving_events(self, sharedagents_sid: str, request: Any) -> list:
        """Queued follow-ups to feed into this turn instead of a nudge.

        Drains the durable queue (a child's result or an owner message typed
        meanwhile). If nothing is queued but a child finished moments ago and
        its result is still being delivered, waits up to
        ``ARRIVING_EVENT_WAIT_S`` for it. Returns ``[]`` when nothing arrives;
        the caller then nudges as before, so a session is never left waiting
        on an event that does not come (lookup errors included).
        """
        drain = (getattr(request, "extra", None) or {}).get("queue_drain")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + float(self.ARRIVING_EVENT_WAIT_S)
        while True:
            try:
                from sharedagents_agent.agents.async_jobs import undelivered_results_for

                arriving = bool(undelivered_results_for(sharedagents_sid)) if sharedagents_sid else False
            except Exception:  # noqa: BLE001 — retry until the deadline
                arriving = True
            # Probe delivery before draining: a delivery acknowledged during
            # the probe has already queued its result. Draining first could
            # miss that result and then mistake its delivered marker for idle.
            # Never take a follow-up out of the queue for a turn that is being
            # stopped: it would be marked injected and never reach a model.
            # Checked before every drain, so a Stop during the wait counts.
            if self._cancelled.get(sharedagents_sid) or self._graceful_stop.get(sharedagents_sid):
                return []
            items: list = []
            if callable(drain):
                try:
                    items = list(drain() or [])
                except Exception:  # noqa: BLE001 — a queue error must never break the turn
                    items = []
            if items:
                return items
            if not arriving or loop.time() >= deadline:
                return []
            await asyncio.sleep(float(self.ARRIVING_EVENT_POLL_S))

    def _open_background_jobs_for(self, sharedagents_sid: str) -> list[dict[str, Any]]:
        """OPEN (non-terminal) background jobs spawned by ``sharedagents_sid``.

        Used to suppress the teacher-independent AUTO_COMPLETE Stop re-drive
        when the parent correctly ended its turn while waiting on async
        children — durable follow-up / turn-end auto-drain wake the parent
        when results land; todo pressure resumes once jobs are terminal.
        Observed: jarvis stopped after dispatching explore/
        build Agent jobs; AUTO_COMPLETE forced continuation on an incomplete
        TodoWrite item and re-drove without the still-running children.
        """
        if not sharedagents_sid:
            return []
        try:
            from sharedagents_agent.agents.async_jobs import open_jobs_for
        except ImportError:
            return []
        try:
            return list(open_jobs_for(sharedagents_sid) or [])
        except Exception:  # noqa: BLE001
            return []
