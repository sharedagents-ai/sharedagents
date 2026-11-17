"""LoopDetectionMixin — extracted from the ``local_llm`` monolith (behaviour-preserving)."""

from __future__ import annotations

import time
from typing import Any

from .state import (
    LOOP_CHARS_PER_TOKEN_MIN,
    LOOP_CHECK_STRIDE,
    LOOP_DETECT_ENABLED,
    LOOP_RATIO_MIN_TOKENS,
    _RoundState,
)


class LoopDetectionMixin:
    # -----------------------------------------------------------------
    # Permission + ASK protocol
    # -----------------------------------------------------------------

    # Orphan ``_request_permission`` / ``_ask_user`` methods removed
    # 2026-05-15. The live permission flow runs at the Agent layer:
    # :class:`LocalLLMEngine` yields a :class:`PermissionRequested`
    # event mid-dispatch, :meth:`Agent.resolve_permission` runs the
    # policy + ``permissions_integration.ask_user`` (which talks to
    # the bound :class:`AcpServer` via
    # ``request_outbound("session/request_permission", payload)``),
    # and sets the event's response future. The engine never makes
    # the policy decision itself.

    # -----------------------------------------------------------------
    # Loop detection
    # -----------------------------------------------------------------

    def _check_streamed_loop(
        self,
        buf: list[str],
        last_check_chars: int,
        stream_label: str,  # "thinking" | "content"
        sustain_state: dict[str, float] | None = None,
        *,
        loop_detect: bool = True,
    ) -> tuple[tuple[str, str] | None, int]:
        """Sample the layered detector on the buffer.

        Reference: acp_bridge.py L3083–3159 and turn_runtime detectors
        (L214 / L271 / L300 / L320). Throttled to once per
        ``LOOP_CHECK_STRIDE`` chars so cost stays bounded.

        Returns ``(hit_or_none, new_last_check_chars)`` where ``hit``
        is ``(tier, evidence_ngram)`` when a loop is detected.

        ``sustain_state`` is the per-channel ambiguous-chrome first-seen
        map (see :class:`_RoundState.loop_sustain`). A fresh
        ambiguous-chrome hit (``--------`` / ``========`` / ``****`` /
        etc.) is RECORDED but SUPPRESSED — the round only aborts after
        the same evidence sustains for
        :data:`turn_runtime.LOOP_AMBIGUOUS_SUSTAIN_SECONDS` (default 30 s).
        Unambiguous heads (``!`` / ``?`` / alphanumeric / paraphrase
        rotation) skip the gate inside ``_should_fire_with_sustain``
        and fire immediately, preserving catch on every documented
        small-model collapse mode (Qwen 3.6 FP8-KV, Gemma 4
        word-doubling, GDN+FP8 corruption, "Let's Let's…").
        """
        if not LOOP_DETECT_ENABLED or not loop_detect:
            return None, last_check_chars
        text = "".join(buf)
        # The tight tier's own min_context_chars is 80; gating here at
        # LOOP_WINDOW_CHARS (1500/4096) silently disabled the tier-0
        # fast filter for the first 1.5k–4k chars of every stream.
        # Average Qwen3.6 turn is 1–3k content chars — tight collapses
        # inside that range were invisible. Let each detector enforce
        # its own minimum inside _detect_loops_layered.
        from sharedagents_agent.turn_runtime import LOOP_TIGHT_MIN_CONTEXT_CHARS

        if len(text) < LOOP_TIGHT_MIN_CONTEXT_CHARS:
            return None, last_check_chars
        if len(text) - last_check_chars < LOOP_CHECK_STRIDE:
            return None, last_check_chars
        try:
            from sharedagents_agent.turn_runtime import (
                _detect_loops_layered,
                _should_fire_with_sustain,
            )
        except ImportError:
            return None, len(text)
        try:
            # Thread the channel through so the tool_args carve-out
            # (turn_runtime.py:381) and the structured-content carve-out
            # (T10) actually fire. Pre-fix the engine was channel-blind.
            hit = _detect_loops_layered(text, channel=stream_label)
        except Exception:  # noqa: BLE001
            return None, len(text)
        if not hit:
            return None, len(text)
        if isinstance(hit, tuple):
            tier = str(hit[0])
            evidence = str(hit[1] if len(hit) > 1 else "")
        else:
            tier = str(hit)
            evidence = ""
        # Ambiguous-chrome sustain gate. Suppresses a brief markdown
        # horizontal rule / table separator / divider while the model
        # is still in legitimate chrome territory; converts to a real
        # fire once the SAME evidence persists past the sustain window.
        # ``sustain_state`` is None when the caller opts out (legacy
        # path, tests) — preserves immediate-fire behaviour there.
        try:
            should_fire = _should_fire_with_sustain(
                evidence,
                sustain_state,
                time.monotonic(),
            )
        except Exception:  # noqa: BLE001
            should_fire = True
        if not should_fire:
            return None, len(text)
        return (tier, evidence), len(text)

    def _check_ratio_loop(
        self,
        metrics: dict[str, Any],
        *,
        model_name: str = "",
        tail_text: str = "",
    ) -> tuple[str, str] | None:
        """Post-stream token-loop ratio guard.

        Reference: worker.py L572–609 (computes
        ``token_loop_suspected``) and acp_bridge.py L3292–3327
        (acts on it). Fires when completion_tokens >=
        :data:`LOOP_RATIO_MIN_TOKENS` AND
        ``chars_per_token < models.for_name(model).loop_chars_per_token_min``
        (per-model override, e.g. 1.5 for Qwen3.6, 2.0 for Gemma4)
        or :data:`LOOP_CHARS_PER_TOKEN_MIN` (env default).

        The previous implementation read
        ``getattr(self.cfg, "loop_chars_per_token_min", 0)`` but
        ``AgentConfig`` has no such field — the engine fallback was
        silently 3× looser than the worker-side per-model floor.
        Resolve the per-model row instead (same pattern as
        ``_maybe_compact`` uses for ``compact_at_max_prompt_tokens``).
        """
        if not metrics:
            return None
        # If the worker already pre-classified the loop, surface it.
        if metrics.get("token_loop_suspected"):
            return ("ratio", str(metrics.get("evidence") or "token_loop_suspected"))
        # Skip the backstop when the upstream gave a clean termination
        # signal — mirrors the worker pre-classifier. A clean finish
        # means the model chose to end this round on purpose, so a
        # high reasoning/content ratio is dense thinking, not a loop.
        if metrics.get("finish_reason") in ("stop", "tool_calls"):
            return None
        # Only a max-token exhaustion is even loop-PLAUSIBLE for the
        # engine-side backstop: error/abort/filter/None finishes are
        # partial output whose low chars/token says nothing about
        # looping, and they were the bulk of the 101 production ratio
        # fires. The worker's own
        # pre-classifier (token_loop_suspected, handled above) keeps
        # full coverage of degenerate-token streams.
        if metrics.get("finish_reason") not in ("length", "max_tokens"):
            return None
        # Corroboration: a length-capped round whose tail is varied
        # structured text (code, JSON, tables) is dense output that got
        # truncated, not a token loop.
        if tail_text:
            try:
                from sharedagents_agent.turn_runtime import (
                    _suffix_is_structured_content,
                )

                if _suffix_is_structured_content(tail_text):
                    return None
            except Exception:  # noqa: BLE001
                pass
        completion_tokens = int(metrics.get("completion_tokens") or 0)
        if completion_tokens < LOOP_RATIO_MIN_TOKENS:
            return None
        chars = int(metrics.get("completion_chars") or 0)
        if chars <= 0:
            return None
        ratio = chars / max(1, completion_tokens)
        # Resolve the per-model floor; fall back to the env default
        # only when the model isn't known or its row doesn't specify.
        floor: float = LOOP_CHARS_PER_TOKEN_MIN
        if model_name:
            try:
                from sharedagents_model.models import for_name

                model_row = for_name(model_name)
                row_floor = getattr(model_row, "loop_chars_per_token_min", None)
                if row_floor:
                    floor = float(row_floor)
            except Exception:  # noqa: BLE001
                pass
        if ratio < floor:
            return ("ratio", f"chars_per_token={ratio:.2f} < floor={floor:.2f}")
        return None

    # -----------------------------------------------------------------
    # Distill recovery + continuation nudge + force-terminate
    # -----------------------------------------------------------------

    async def _distill_recovery(
        self,
        session: Any,
        messages: list[dict[str, Any]],
        loop_hit: tuple[str, str],
        round_state: _RoundState,
        recovery_count: int,
        *,
        hook_engine: Any = None,
        hook_ctx: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Distill the partial reasoning into a fresh frame, then continue.

        Reference: turn_runtime._distill_partial_reasoning (L759) plus
        the surrounding loop in ``CoreBackend.prompt`` (the
        distill_recoveries / distill_max_recoveries counters,
        acp_bridge L2921–2931).

        Steps:

        1. Pause the worker (if still running), kill it.
        2. Fire the ``DistillRecovery`` hook. The hook may
           ``continue: false`` (skip recovery; fall through to force-
           terminate) or ``decision: block`` + ``reason: <preamble>``
           (substitute the operator's preamble for the canned
           distiller call — useful when the operator already has a
           cheap on-disk summariser and doesn't want to spend an LLM
           round).
        3. Otherwise: call ``_distill_partial_reasoning(thinking_buf,
           model=…)`` — a cheap LLM round that produces a structured
           JSON ``{"summary", "key_findings", "next_step"}``.
        4. Parse via ``_parse_distill_json``; format via
           ``_format_distill_for_recovery``.
        5. Replace the assistant-round-in-progress message with the
           formatted recovery preamble, signal the loop to start a
           fresh worker round.

        ``cfg.distill_max_recoveries`` (default 3) caps how many times
        this fires per turn. After that, fall through to
        :meth:`_force_terminate`.

        Returns ``(messages, recovered)``.
        """
        try:
            from sharedagents_agent.turn_runtime import _distill_partial_reasoning
        except ImportError:
            return messages, False

        max_recoveries = int(getattr(self.cfg, "distill_max_recoveries", 3) or 3)
        if recovery_count >= max_recoveries:
            return messages, False

        thinking = "".join(round_state.thinking_buf)
        if not thinking.strip():
            return messages, False

        # ``DistillRecovery`` hook: gives operators a substitution point
        # for the (expensive) LLM-based distiller round. A
        # ``decision: block`` with a ``reason`` body short-circuits the
        # distiller entirely — the hook's text becomes the recovery
        # preamble. A ``continue: false`` skips distill so the cascade
        # advances to force-terminate immediately.
        preamble: str | None = None
        if hook_engine is not None:
            distill_hook = await self._fire_hook(
                hook_engine,
                hook_ctx or {},
                "DistillRecovery",
                {
                    "tier": loop_hit[0] if loop_hit else "",
                    "evidence": loop_hit[1] if loop_hit else "",
                    "attempt": recovery_count + 1,
                    "max_recoveries": max_recoveries,
                    "thinking_chars": len(thinking),
                },
            )
            if not distill_hook.should_continue:
                return messages, False
            if distill_hook.blocked and distill_hook.block_reason:
                preamble = distill_hook.block_reason

        if preamble is None:
            # Resolve the last user prompt from the message history.
            user_prompt = ""
            for m in reversed(messages):
                if isinstance(m, dict) and m.get("role") == "user":
                    user_prompt = str(m.get("content") or "").strip()
                    break
            distill_headers = dict(
                self.cfg.upstream_headers(
                    getattr(session, "endpoint", "") or self.cfg.upstream_endpoint,
                    getattr(session, "model", "") or self.cfg.upstream_model,
                )
            )
            # Include X-Session-Id so the vllm-mux router keeps
            # distill recovery on the same replica as the session
            # (warm prefix cache).  NOT X-Session-New — distill
            # is always a mid-session recovery, never a fresh start.
            if hasattr(session, "hash") and session.hash:
                distill_headers["x-session-id"] = session.hash
            # A distill/recovery failure surfaces to the
            # caller — recovery must not swallow it or launch a second layer of
            # automated recovery. _distill_partial_reasoning
            # already returns the formatted handoff (and its built-in default
            # template makes this recover with no on-disk DISTILL.md), so use
            # that string directly instead of re-parsing/re-formatting it.
            preamble, _distill_debug = await _distill_partial_reasoning(
                endpoint=getattr(session, "endpoint", "") or self.cfg.upstream_endpoint,
                model=getattr(session, "model", "") or self.cfg.upstream_model,
                headers=distill_headers,
                user_prompt=user_prompt,
                thinking=thinking,
                home=self.cfg.home,
                cfg=self.cfg,
            )
        # Drop the partial assistant message at the tail and insert
        # the recovery preamble in its place.
        out = list(messages)
        while out and isinstance(out[-1], dict) and out[-1].get("role") == "assistant":
            out.pop()
        out.append({"role": "assistant", "content": preamble})
        self._log_event(
            "DISTILL_RECOVERY",
            session_id=getattr(session, "hash", ""),
            attempt=recovery_count + 1,
            tier=str(loop_hit[0]) if loop_hit else "",
        )
        return out, True
