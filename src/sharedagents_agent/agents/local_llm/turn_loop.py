"""TurnLoopMixin — extracted from the ``local_llm`` monolith (behaviour-preserving)."""

from __future__ import annotations

import asyncio
import hashlib
import os
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .state import (
    _GOAL_BLOCKED_RE,
    _HOOK_OUTCOMES_LOG,
    _MAX_REACTIVE_COMPACT_RETRIES,
    _MAX_REACTIVE_MAXTOK_FALLBACKS,
    _MAX_READONLY_HARD_LIMIT,
    _MAX_READONLY_SOFT_LIMIT,
    _MAX_STOP_CONTINUATIONS,
    _MAX_TRANSIENT_RETRIES,
    FINAL_ANSWER_NUDGE_BODY,
    MAX_NUDGES_PER_TURN,
    MAX_STEPS_HANDOFF_USER,
    NUDGE_MIN_TEXT_CHARS,
    AgentTurnEvent,
    EngineSessionState,
    QueueMessageInjected,
    RoundComplete,
    TextDelta,
    TurnDone,
    TurnError,
    TurnRequest,
    UserMessageInjected,
    _detect_stream_truncation,
    _drain_stderr,
    _format_runtime_warning,
    _is_endpoint_down,
    _is_phantom_round_metrics,
    _is_transient_upstream,
    _local_worker_idle_timeout_s,
    _next_worker_event,
    _now_iso,
    _readonly_budget_class,
    _round_is_background_delegate_wait,
    _RoundState,
    wrap_untrusted,
)


def _llm_fallback_allowed(*, already_used: bool, round_error: str) -> bool:
    """Whether this turn may silently leave its selected model route."""
    strict = os.environ.get(
        "SHAREDAGENTS_STRICT_MODEL_ROUTE",
        "0",
    ).strip().lower() in {"1", "true", "yes", "on"}
    return not strict and not already_used and _is_endpoint_down(round_error)


def _stop_cap_short_circuits(
    *,
    cap_spent: bool,
    reviewer_enabled: bool,
) -> bool:
    """Whether the Stop-continuation cap should end the turn with NO review.

    The cap bounds the RE-DRIVE, never the REVIEW (operator directive
    2026-09-10). When turn-end is owned by a reviewer — explicit ``/teacher
    on``, or a ``/goal`` Stop-hook evaluator — the review must still run with
    the continuation budget spent, because the verdict IS the product; the
    re-drive is suppressed separately at the continuation point. Only a
    session with no reviewer is short-circuited by the cap.

    Regression guarded by ``tests/test_teacher_stop_cap_review.py``:
    ``max_stop_continuations`` defaults to 0 and nothing sets it, so an
    unconditional ``cap_spent`` break made the deep ``pre_idle`` review
    unreachable in every default deployment — every supervised turn end
    rendered the shallow, no-fire-back ``stop-cap`` terminal review instead.
    """
    return bool(cap_spent) and not bool(reviewer_enabled)


class TurnLoopMixin:
    # -----------------------------------------------------------------
    # Main turn driver
    # -----------------------------------------------------------------

    async def run_turn(  # type: ignore[override]
        self,
        request: TurnRequest,
        state: EngineSessionState,
        *,
        sessions_dir: Path,
        turn_id: str,
        env: dict[str, str] | None = None,
    ) -> AsyncIterator[AgentTurnEvent]:
        """Drive a full multi-round turn against the local LLM.

        This method ABSORBS the body of ``CoreBackend.prompt`` (acp_bridge.py
        L2838–~L3500). The translation from ACP sessionUpdate dicts to
        :data:`AgentTurnEvent` instances happens at every ``yield``.

        Sequence (high level):

        1. **Load session** — fetch ``Session`` by ``request.extra["sid"]``
           via :meth:`_session_for`.
        2. **Direct-tool-call bypass** — if ``request.extra`` carries a
           ``__SHAREDAGENTS_DELEGATE__:`` sentinel, route through
           :meth:`_run_direct_tool_call`. This is what ``/claude``,
           ``/codex``, ``/gemini``, ``/compact`` slash commands hit.
           (acp_bridge.py L2860–2873)
        3. **Build message list** — :meth:`_build_messages_simple`. The
           composed system prompt arrives via ``request.system_prompt``;
           the below-cache-boundary preamble blocks (skills, memory
           recall, pinned facts, rolling summary, todos) arrive via
           ``request.preamble``. The engine NEVER reaches into
           SOUL/AGENTS/TOOLS.md or the memory provider itself —
           composition is exclusively the Agent layer's responsibility.
        4. **Round loop** — up to ``self._max_rounds_for_mode()`` (auto
           = effectively uncapped). Per round:

           a. :meth:`_apply_round_preamble` — splice
              ``request.preamble`` back into the working message list.
              Idempotent — same content replaces prior placement.
           b. :meth:`_maybe_compact` — proactive context squeeze.
           c. :meth:`_log_round` — KV-cache hash instrumentation.
           d. Spawn worker via :meth:`_spawn_worker`.
           e. Consume worker events via :meth:`_consume_worker_events`,
              translating each to :data:`AgentTurnEvent`.
           f. Streaming loop-detection on ``thinking`` + ``content``
              channels (3 detectors layered).
           g. On ``tool_calls`` event: dispatch via
              :meth:`_dispatch_tool_calls`. May yield
              :class:`PermissionRequested` events back through the
              :class:`Agent` for ASK-mode calls. After each call,
              append the tool-result message and continue.
           h. On natural ``done`` with no tool_calls: end-of-turn.
           i. On ``aborted`` / streamed-loop / ratio-loop / context-
              overflow: route through the appropriate recovery
              (:meth:`_distill_recovery` / :meth:`_reactive_compact` /
              :meth:`_continuation_nudge` / :meth:`_force_terminate`).

        8. **Post-turn**:

           * :func:`session_log.record_turn` — uniform turn record
             (engine="local_llm", model, tool_calls, tool_results,
             metrics).
           * ``Session.save()``.
           * ``HookEngine.post_turn`` if hooks loaded
             (``cfg.hooks_path``).
           * Memory provider ``post_turn_observe`` hook
             (``memory_manager.observe_turn(session)``).
           * Pre-compaction memory flush check
             (``cfg.memory_flush_enabled`` + threshold math).

        9. **Yield** ``TurnDone(state=…)`` with final metrics.
        """
        from ..engine import (
            HandoffAccepted,
            TextDelta,
            ToolCallResult,
            ToolCallStarted,
            TurnDone,
            TurnError,
            TurnStarted,
        )

        # Pull caller-supplied session out of request.extra. The Agent
        # layer puts it there so the engine doesn't need to re-load it.
        session = request.extra.get("session")
        if session is None:
            # Even an early validation error must be
            # bracketed by exactly one TurnStarted ... TurnDone, with the
            # TurnError BETWEEN them (a TurnError never itself terminates a
            # turn). Emit the mandatory TurnStarted first.
            yield TurnStarted(
                turn_id=turn_id,
                sharedagents_session_id=state.sharedagents_session_id,
                engine_session_id="",
                resumed=False,
            )
            yield TurnError(
                message="LocalLLMEngine.run_turn requires request.extra['session']",
                classification="bad_request",
            )
            yield TurnDone(state="error")
            return

        # Apply a per-turn model override (``Agent`` / ``BatchAgent`` /
        # ``/delegate`` ``model`` arg → ``TurnRequest.model``) onto the
        # session BEFORE anything reads ``session.model`` (engine audit
        # stamp, worker spawn). The worker request is built from
        # ``session.model`` / ``session.endpoint`` — without this the
        # override was silently dropped and every subagent ran on the
        # parent's model. Isolated to this session (see
        # :meth:`_apply_model_override`); the global cfg is untouched.
        self._apply_model_override(session, getattr(request, "model", None))

        sharedagents_sid = session.hash
        manual_budget_exhausted = bool(request.extra.get("_budget_exhausted_manual"))
        # Reset stale cancel flag at the START of a turn so an abort
        # of a PRIOR turn doesn't carry over.
        self._cancelled[sharedagents_sid] = False
        # Same for the graceful-stop flag — a ``/stop`` of a prior turn
        # must not silently halt this fresh one.
        self._graceful_stop[sharedagents_sid] = False
        # Set sid → turn_id BEFORE spawn so ESC during the spawn
        # window still routes correctly.
        self._session_to_turn[sharedagents_sid] = turn_id

        # Hook engine + per-turn hook context — supplied by the Agent
        # layer. The Agent owns the :class:`HookEngine`; the *engine*
        # owns the turn loop, so this is where Claude Code's hook
        # lifecycle (UserPromptSubmit / PreToolUse / PostToolUse /
        # Stop / PreCompact) actually fires.
        hook_engine = request.extra.get("hook_engine")
        hook_ctx = dict(request.extra.get("hook_ctx") or {})

        # Per-turn aggregates — populated as rounds run so the Agent
        # layer can persist a uniform record regardless of how we exit.
        thinking_buf: list[str] = []
        content_buf: list[str] = []
        engine_tool_calls: list[dict[str, Any]] = []
        engine_tool_results: list[dict[str, Any]] = []
        last_metrics: dict[str, Any] = {}
        terminal_state = "error"
        # F1 (plan v2): structured finish reason — ``"stop"`` for natural
        # completion, ``"round_cap"`` when the per-turn round budget was
        # exhausted. Rides TurnDone + the engine audit + the turn record.
        completion_reason = "stop"
        # The FINAL round's OWN wrap-up text (its
        # content after the max-steps handoff injection) — the only
        # content that qualifies a CAPPED turn as ``ok``; cumulative
        # earlier-round content is NOT a handoff. Updated at every
        # RoundComplete and at the natural-stop-capped exit; stays ""
        # when no round ever completed (e.g. max_rounds=0).
        final_round_text = ""
        # Whether ``final_round_text`` was refreshed by
        # the round that JUST ran (``_round_advanced``) — reset at the top
        # of every round, set on every booking. Lets the terminal
        # chokepoint's F4 wrap-up gate distinguish "the final round
        # delivered its own wrap-up" from "no final round completed
        # (worker failure / retry-exhausted / abort)" — the gate must NEVER
        # fall back to a PREVIOUS round's text for the final round (the
        # stale-wrap-up bug: capped incomplete work reported ok).
        final_round_booked = False
        # ACTUAL completed rounds this turn (a round
        # that yielded RoundComplete, plus the final round that ended via
        # a natural stop on the capped boundary). ``round_idx`` cannot be
        # used directly: on the cap-hit exit it already includes every
        # completed round, while on the natural-stop-capped exit the final
        # round has not yet incremented it — ``round_idx + 1``
        # double-counted the cap-hit path (a 2-round cap reported 3).
        rounds_completed = 0
        research_publication_nudges = 0

        # EVERY round-advancing exit must refresh the
        # final-round bookkeeping — not only the RoundComplete path. Loop
        # recovery (nudge/redirect) and the inject pivots advance
        # ``round_idx`` without a RoundComplete; the cap-time terminal
        # decision then read STALE values: an early round's content
        # qualified a content-less final round as ``ok``, and the rounds
        # metric under-counted. ``rs`` is the round state of the round
        # that JUST ended; its ``content_buf`` is that round's own content
        # (F4: cumulative earlier-round content never qualifies).
        # Deliberately NOT called on the INFRA retry advances (transient
        # upstream / reactive max-tokens / endpoint failover): those
        # re-send the SAME round and mark it ``round_had_result=None`` —
        # not a completed student round.
        def _round_advanced() -> None:
            nonlocal final_round_text, rounds_completed, final_round_booked
            final_round_text = "".join(rs.content_buf).strip()
            rounds_completed += 1
            final_round_booked = True

        started_at_iso = _now_iso()
        # Anchor for per-turn ``round_metrics`` slicing. The
        # ``session.round_metrics`` list is session-scope and
        # append-only across every turn; without capturing the
        # length at turn start we'd persist all PRIOR turns' round
        # metrics onto every subsequent turn record (gemini round-4
        # critical finding — defeats the per-turn debug use case).
        # Snapshot here so the engine audit at TurnDone can slice
        # out exactly this turn's rounds.
        _round_metrics_start_idx = len(
            getattr(session, "round_metrics", None) or [],
        )
        # Per-turn hook-outcome collector. Set as the active
        # context-var so every ``_fire_hook`` call inside this
        # frame appends a stripped outcome record. The contextvar
        # reset happens in the existing ``finally`` block at
        # turn end; until then the same list is shared with the
        # audit stash below so updates land on disk even if the
        # engine takes an early-error exit (codex round-4
        # critical finding: hook_outcomes was dead-fielded).
        hook_outcomes_log: list[dict[str, Any]] = []
        _hook_log_token = _HOOK_OUTCOMES_LOG.set(hook_outcomes_log)

        # Stamp an INITIAL engine-audit dict up-front. Early-error
        # exits (UserPromptSubmit / UserPromptExpansion hook block,
        # malformed request) yield TurnDone(state="error") BEFORE
        # the normal terminal stash runs, so without this initial
        # stamp the agent layer would receive an empty dict and
        # the saved turn would have no effective_system_prompt /
        # preamble / endpoint — defeating the "debug a regression
        # from the JSON alone" goal. The dict is mutated in-place
        # by the normal terminal path (which preserves the same
        # reference) so concurrent agent-layer reads always see
        # the latest state. Codex round-4 critical finding.
        request.extra["_engine_audit"] = {
            "thinking": "",
            "content": "",
            "tool_calls": [],
            "tool_results": [],
            "metrics": {},
            "started_at": started_at_iso,
            "ended_at": "",
            "state": "incomplete",
            "model_input_messages": [],
            "round_metrics": [],
            "effective_system_prompt": (
                request.system_prompt
                if request.system_prompt is not None
                else (getattr(session, "system_prompt", "") or "")
            ),
            "effective_system_prompt_sha256": getattr(
                request,
                "system_prompt_sha256",
                "",
            )
            or "",
            "effective_preamble": list(
                getattr(request, "preamble", None) or [],
            ),
            # Record the PER-SESSION endpoint + model the round actually
            # hit (``_build_worker_request`` builds the upstream call from
            # ``sess.endpoint``/``sess.model``), not the global
            # ``cfg.upstream_*`` default — which a model switch in another
            # session can have mutated out from under this one, making the
            # audit trail point at the wrong upstream.
            "upstream_endpoint": str(
                getattr(session, "endpoint", "") or getattr(self.cfg, "upstream_endpoint", "") or "",
            ),
            "upstream_model": str(
                getattr(session, "model", "") or getattr(self.cfg, "upstream_model", "") or "",
            ),
            # NB: same list reference as ``hook_outcomes_log`` —
            # _fire_hook appends here for the duration of the
            # context-var set, so the on-disk record always
            # reflects every hook that fired up to the terminal
            # yield.
            "hook_outcomes": hook_outcomes_log,
        }

        yield TurnStarted(
            turn_id=turn_id,
            sharedagents_session_id=sharedagents_sid,
            engine_session_id="",
            resumed=False,
        )

        # --- SessionStart hook (first turn per session) -----------------
        if hook_engine is not None and sharedagents_sid not in self._sessions_started:
            self._sessions_started.add(sharedagents_sid)
            # Resume vs startup: any prior turns on the session means
            # this is a resume; otherwise startup. Operator-facing
            # `clear` / `compact` sources land here only if the caller
            # set them on the engine state — we default to inferring.
            _source = "resume" if (getattr(session, "turns", None) and len(session.turns) > 0) else "startup"
            # ``Setup`` (Claude Code spec) — fires on first-init flows
            # before SessionStart. We treat any startup with no prior
            # turns AND no prior session metadata as the first-init
            # case. An ECC plugin can stage one-time scaffolding
            # (writing CLAUDE.md, generating a starter SOUL, etc.)
            # here; SessionStart fires next for the regular per-session
            # init work.
            if _source == "startup":
                _setup = await self._fire_hook(
                    hook_engine,
                    hook_ctx,
                    "Setup",
                    {"source": "startup"},
                )
                for _sm in _setup.system_messages:
                    yield TextDelta(text=f"[hook] {_sm}\n", channel="content")
                if _setup.additional_context:
                    yield TextDelta(
                        text=_setup.additional_context + "\n",
                        channel="content",
                    )
            _ss = await self._fire_hook(
                hook_engine,
                hook_ctx,
                "SessionStart",
                {"source": _source},
            )
            # Surface system messages; additionalContext from
            # SessionStart is folded into the user turn below.
            for _sm in _ss.system_messages:
                yield TextDelta(text=f"[hook] {_sm}\n", channel="content")

        # --- Direct-dispatch bypass (/claude /codex /gemini /compact) ----
        # The Agent passes the raw prompt; if it leads with the
        # ``__SHAREDAGENTS_DELEGATE__`` sentinel the operator drove the
        # dispatch decision (no LLM round) — route straight to the
        # delegate tool. ``_run_direct_tool_call`` yields its own
        # terminal ``TurnDone``.
        direct = None
        try:
            from sharedagents_gateway.acp_bridge import _parse_direct_delegate_marker

            direct = _parse_direct_delegate_marker([{"type": "text", "text": request.prompt or ""}])
        except Exception:  # noqa: BLE001
            direct = None
        if direct is not None:
            # A restricted (Web user) session never skips the model and the
            # tool gate — except for /compact.
            from sharedagents_session.access_binding import direct_dispatch_allowed

            if not direct_dispatch_allowed(sessions_dir, session.hash, direct):
                direct = None
        if direct is not None and not manual_budget_exhausted:
            # Record the delegate's reply as the parent turn's ``content``.
            # The reply streams on the ``tool_delta`` channel, which the
            # Agent does NOT fold into ``streamed_content`` — so without
            # this the parent turn's content is empty, and a later
            # CROSS-kind delegate (e.g. /codex after /claude) can't see the
            # prior reply (the preamble forwards turn ``content``, not the
            # tool_result). Stash it into the engine audit BEFORE the
            # ``TurnDone`` passes through (the Agent records the turn on
            # ``TurnDone`` reading ``_engine_audit['content']``).
            from ..engine import ToolCallResult as _TCR, TurnDone as _TD

            _direct_body = ""
            async for ev in self._run_direct_tool_call(session, turn_id, direct):
                if isinstance(ev, _TCR) and getattr(ev, "content", None):
                    _direct_body = str(ev.content)
                elif isinstance(ev, _TD):
                    audit = request.extra.get("_engine_audit")
                    if not isinstance(audit, dict):
                        audit = {}
                        request.extra["_engine_audit"] = audit
                    if _direct_body:
                        # Assignment, NOT setdefault: run_turn pre-stamps
                        # ``request.extra["_engine_audit"]`` with ``content:
                        # ""`` (see the audit seeding above), so setdefault
                        # would keep the empty string and the committed
                        # assistant turn would read back EMPTY — the prod
                        # empty-bubble bug (session
                        # <session-id>: direct-dispatch
                        # /compact turn committed content="" while its metrics
                        # stash one line below landed fine — proof it is the
                        # same dict).
                        audit["content"] = _direct_body
                    # Persist the sharedagents↔delegate link on the PARENT turn:
                    # the TurnDone metrics carry delegate_child_session_id /
                    # delegate_engine_session_id / delegate_engine, so the
                    # parent session log records exactly which delegate
                    # session handled this turn.
                    if getattr(ev, "metrics", None):
                        audit.setdefault("metrics", ev.metrics)
                yield ev
            _HOOK_OUTCOMES_LOG.reset(_hook_log_token)
            return

        # --- UserPromptSubmit hooks (Claude Code) -----------------------
        # Fires for EVERY prompt, including multimodal (image / file
        # parts). Skipping it on multipart would let a user bypass a
        # blocking inspection hook simply by attaching an image —
        # contradicts Claude Code's security contract.
        user_text = request.prompt or ""
        if hook_engine is not None:
            ups = await self._fire_hook(
                hook_engine,
                hook_ctx,
                "UserPromptSubmit",
                {"prompt": user_text},
            )
            for sm in ups.system_messages:
                yield TextDelta(text=f"[hook] {sm}\n", channel="content")
            if ups.blocked or not ups.should_continue:
                yield TurnError(
                    message=(ups.block_reason or ups.stop_reason or "blocked by a UserPromptSubmit hook"),
                    classification="hook_blocked",
                )
                yield TurnDone(state="error")
                _HOOK_OUTCOMES_LOG.reset(_hook_log_token)
                return
            if ups.additional_context:
                user_text = f"{ups.additional_context}\n\n{user_text}" if user_text else ups.additional_context

            # ``UserPromptExpansion`` (Claude Code spec) — fires AFTER
            # UserPromptSubmit so a plugin can rewrite slash commands
            # / argument-expanded skill invocations before the prompt
            # is built into the message list. Most useful for
            # plugins that ship custom slash commands (e.g. ``/research``
            # → "run the multi-loop research skill with topic X").
            # Hook ``decision: block`` + ``reason: <expanded prompt>``
            # rewrites ``user_text`` wholesale; ``additionalContext``
            # is prepended.
            command_name = ""
            if user_text.lstrip().startswith("/"):
                first_token = user_text.lstrip().split(maxsplit=1)[0]
                command_name = first_token  # includes the leading "/"
            upe = await self._fire_hook(
                hook_engine,
                hook_ctx,
                "UserPromptExpansion",
                {"prompt": user_text, "command": command_name},
            )
            for sm in upe.system_messages:
                yield TextDelta(text=f"[hook] {sm}\n", channel="content")
            if upe.blocked and upe.block_reason:
                user_text = upe.block_reason
            elif upe.additional_context:
                user_text = f"{upe.additional_context}\n\n{user_text}" if user_text else upe.additional_context
            if not upe.should_continue:
                yield TurnError(
                    message=(upe.stop_reason or "blocked by a UserPromptExpansion hook"),
                    classification="hook_blocked",
                )
                yield TurnDone(state="error")
                _HOOK_OUTCOMES_LOG.reset(_hook_log_token)
                return

        # Build the initial message list (system + preamble + transcript
        # + the live user turn).
        user_content: str | list[dict[str, Any]]
        user_content = list(request.prompt_parts) if request.prompt_parts else user_text
        handoff_preamble = list(request.preamble)
        if request.handoff is not None and getattr(request.handoff, "rendered", ""):
            handoff_preamble.append(
                {
                    "role": "user",
                    "content": (
                        "<sharedagents-canonical-handoff>\n"
                        f"{request.handoff.rendered}\n"
                        "</sharedagents-canonical-handoff>"
                    ),
                }
            )
        messages = self._build_messages_simple(
            session,
            user_content,
            request.system_prompt,
            handoff_preamble,
            # Pass this turn's persisted ``started_at`` so the LIVE-turn
            # marker is byte-identical to its future frozen replay —
            # keeps the upstream prefix cache warm across turns (see
            # ``_build_messages_simple``). ``started_at_iso`` is the same
            # value recorded onto the turn at ``record_turn`` time.
            live_started_at=started_at_iso,
        )

        # ``InstructionsLoaded`` (Claude Code spec) — fires once per
        # turn after the system prompt + preamble (CLAUDE.md /
        # AGENTS.md / SOUL.md / pinned facts) is composed, so an ECC
        # plugin can:
        #   * Observe what instructions are active (audit trail).
        #   * Inject ``additionalContext`` that is prepended to the
        #     system prompt for THIS round only (per-turn dynamic
        #     context, e.g. "user is in plan mode — read-only tools
        #     allowed", or freshly-fetched docs from a remote source).
        # The hook receives the joined preamble text as ``source``
        # match-target (the file paths are not exposed by
        # ``request.preamble`` directly — it's already-rendered
        # text — but the plugin can scope by inspecting the
        # ``preamble`` payload field for substrings).
        if hook_engine is not None and hook_engine.has_event(
            "InstructionsLoaded",
        ):
            _preamble_text = (
                "\n\n".join(request.preamble or [])
                if isinstance(request.preamble, (list, tuple))
                else str(request.preamble or "")
            )
            instr = await self._fire_hook(
                hook_engine,
                hook_ctx,
                "InstructionsLoaded",
                {
                    "system_prompt": request.system_prompt or "",
                    "preamble": _preamble_text,
                    "source": "round_preamble",
                },
            )
            for _sm in instr.system_messages:
                yield TextDelta(text=f"[hook] {_sm}\n", channel="content")
            if instr.additional_context and messages:
                # Splice the hook's text BEFORE the system prompt
                # (so it acts as a prefix wrapper). If no system
                # prompt is present, fall through to inserting at
                # position 0 of the message list.
                if isinstance(messages[0], dict) and messages[0].get("role") == "system":
                    sys_body = str(messages[0].get("content") or "")
                    messages[0] = dict(messages[0])
                    messages[0]["content"] = (
                        f"{instr.additional_context}\n\n{sys_body}" if sys_body else instr.additional_context
                    )
                else:
                    messages.insert(
                        0,
                        {
                            "role": "system",
                            "content": instr.additional_context,
                        },
                    )

        # A post-cap operator prompt is intentionally still answerable, but
        # it is one short, tool-less round. Work that would START something
        # never reaches this path — Agent admission rejects it before
        # dispatch. What DOES reach it besides an operator prompt is the
        # delivery of already-authored text (a finished child's wrap-up via
        # ``background_task``, a turn-end ``queue_drain``): admitted, and
        # bounded by exactly this one round with an empty tool surface.
        max_rounds = self._max_rounds_for_request(
            sharedagents_sid,
            manual_budget_exhausted=manual_budget_exhausted,
        )
        promoted: set[str] = set()  # Promotion removed; vestigial
        nudge_log: list[dict[str, Any]] = []
        # FP2: cumulative user-visible content streamed across ALL rounds of
        # this turn. The silent continuation-nudge premise ("you stopped
        # without writing anything") is per-TURN; without this the engine
        # nudged turns that answered in an earlier round then ended on a quiet
        # bookkeeping tool round. Threaded into ``_should_nudge``.
        turn_visible_chars = 0
        # Cumulative content ``TextDelta`` payloads emitted on the wire this
        # turn. Compared against ``content_buf`` at turn end to detect stream
        # gaps (final audit text longer than streamed deltas).
        turn_emitted_content_parts: list[str] = []
        # Cross-round SEMANTIC reasoning-loop detector (the false-NEGATIVE
        # fix). Turn-scoped so it survives the per-round ``rs = _RoundState()``
        # reset — that reset is exactly why the per-round char/ngram/entropy
        # tiers never saw the 30-round/87k-char paraphrased spin in session
        # <session-id>. ``observe`` is called once per completed round
        # below at the buffer-roll chokepoint.
        from sharedagents_agent.turn_runtime import (
            PROGRESS_HARDSTOP_STREAK,
            PROGRESS_KEPT_TEXT_TIERS as _PROGRESS_KEPT_TEXT_TIERS,
            PROGRESS_NUDGE_STREAK,
            XROUND_LOOP_ACTION as _xround_action,
            XROUND_TELEMETRY as _xround_telemetry,
            ProgressTracker as _ProgressTracker,
            TurnLoopTracker as _TurnLoopTracker,
            progress_finalize_body as _progress_finalize_body,
            progress_nudge_body as _progress_nudge_body,
        )

        # Gate the cross-round SEMANTIC loop detector per-model, exactly like
        # the per-round streamed detector below (``_loop_detect_on``): only the
        # local repetition-prone families (qwen3.6 fp8/int4, gemma) that carry a
        # documented token/paragraph repeat failure mode get it. deepseek-v4-*
        # (and other GENERIC / external models) do NOT loop that way and were
        # FALSE-POSITIVING on legitimate repeated structure in long reasoning —
        # keep the monitor OFF for them. Without this, ``TurnLoopTracker()``
        # defaulted to the global ``XROUND_LOOP_DETECT_ENABLED`` env flag and
        # terminated deepseek turns regardless of model.
        from sharedagents_model.models import needs_loop_detection as _needs_ld

        _xr_sess = self._sessions.get(sharedagents_sid)
        _xround_enabled = _needs_ld(
            (getattr(_xr_sess, "model", "") or "") or getattr(self.cfg, "upstream_model", "") or ""
        ) and not self._protections_off(sharedagents_sid)
        xround_tracker = _TurnLoopTracker(enabled=_xround_enabled)
        # Per-TURN sustained-zero-progress recovery (replaces the surface
        # distill→force-terminate loop response). Counts rounds that make no
        # progress and drives a re-grounding nudge (streak>=nudge) then a
        # graceful finalize (streak>=hardstop; terminal_state "ok", never an
        # error). Thresholds are cfg-overridable (env defaults 3 / 8). Skipped
        # entirely when protections are off for this agent/model.
        progress_tracker = _ProgressTracker(
            nudge_streak=int(getattr(self.cfg, "progress_nudge_streak", 0) or 0) or PROGRESS_NUDGE_STREAK,
            hardstop_streak=int(getattr(self.cfg, "progress_hardstop_streak", 0) or 0) or PROGRESS_HARDSTOP_STREAK,
        )
        # Per-TURN ring of recent rounds' tool-call signatures —
        # _detect_tool_call_loop (turn_runtime.py:404) needs at least
        # _TOOL_CALL_LOOP_WINDOW (10) prior turns of history. Lives at
        # turn scope so it survives the `rs = _RoundState()` reset
        # inside the per-round loop. Bounded by the larger of the strict
        # and broad windows + 1 so memory never grows unboundedly mid-turn.
        tool_call_history: list[list[dict[str, Any]]] = []
        distill_recoveries = 0
        stop_continuations = 0
        # Per-TURN count of reactive max-tokens-fallback retries (the
        # ``COMPACTION_REACTIVE_MAX_TOKENS_FALLBACK`` recovery). Bounds the
        # overflow-retry storm — see ``_MAX_REACTIVE_MAXTOK_FALLBACKS``.
        reactive_maxtok_retries = 0
        starved_compactions = 0
        reactive_compact_retries = 0
        # Summarizing can cost tokens without producing an accepted prompt.
        # Bound consecutive overflow recovery, even when LLM compaction
        # reports success. Only a successful worker round resets this count.
        consecutive_overflow_retries = 0
        # Per-TURN count of TRANSIENT upstream-error retries (thinking-
        # inactivity ReadTimeout, mid-stream connection drops). Bounded by
        # ``_MAX_TRANSIENT_RETRIES`` so a persistently-wedged endpoint
        # surfaces the error instead of retrying forever.
        transient_retries = 0
        # Per-TURN guard: only fail over to the LLM fallback model
        # (``role == 'fallback'``) ONCE, so a dead primary doesn't ping-pong.
        llm_fallback_used = False
        # Per-turn REQUEST-LEVEL failover override (set when the primary
        # endpoint is down). Deliberately NOT written onto the durable
        # ``session``: the cost tracker / compaction / tool dispatch call
        # ``session.save()`` mid-turn, and a mutated ``session.model`` /
        # ``session.endpoint`` gets persisted into the session header —
        # silently re-stamping the operator's chosen model (observed: a
        # qwen3.8-27b-fp8 session header flipped to
        # deepseek-v4-pro after the first turn's failover).
        llm_fallback_ep: str | None = None
        llm_fallback_model: str | None = None
        # Tracks rounds whose worker yielded one or more ``tool_calls`` —
        # the continuation-nudge policy fires ONLY after at least one
        # tool-using round so a normal first-round end_turn is never
        # nudged. Mirrors ``_should_nudge.tool_rounds_so_far``.
        tool_rounds_so_far = 0
        from sharedagents_agent.continuation import TurnEventTracker

        nudge_tracker = TurnEventTracker()
        coordination = None
        if getattr(self.cfg, "name", "") == "jarvis" and request.extra.get("_turn_source") != "memory_flush":
            from sharedagents_agent.agents.async_jobs import wave_origin

            from .coordination import (
                live_children_note,
                start_turn,
                workflow_disposition,
            )

            coordination = start_turn(
                request.prompt_parts or request.prompt,
                str(request.extra.get("_turn_source") or "user"),
                wave_origin_of=wave_origin,
            )
            self._coordination_turns[sharedagents_sid] = coordination
            if coordination.owner_prompt:
                # A new owner message while earlier children still run: show
                # them so a same-subject follow-up steers instead of
                # dispatching a duplicate wave.
                _note = live_children_note(self._open_background_jobs_for(sharedagents_sid))
                if _note:
                    messages.append({"role": "user", "content": _note})
        # How many nudges had been fired when this turn last made real
        # progress. The streak is derived from the slice after that anchor, so
        # progress restores capacity without a second counter to keep in sync.
        nudges_at_last_progress = 0

        def _idle_streak() -> int:
            """Consecutive SILENT nudges since the last real progress.

            Mechanical `stream_truncated` retries share `nudge_log` but are a
            different event -- they resume text that exists. Counting them here
            let three truncation retries exhaust the silent bound.
            """
            return sum(1 for _e in nudge_log[nudges_at_last_progress:] if str((_e or {}).get("kind") or "") == "silent")

        # All tool names invoked this turn (for ToolSearch-stall detection).
        turn_tools_called: set[str] = set()
        # Proactive teacher firing state. The
        # teacher fires mid-turn to HELP the student the moment either trips:
        #   * tool_error_streak  — CONSECUTIVE errored tool calls in this turn;
        #     reset on any successful tool call. Fires at
        #     ``_teacher_tool_fail_threshold``.
        #   * no_result_rounds   — CONSECUTIVE rounds in this turn-segment that
        #     delivered NO result (no new visible content AND no successful tool
        #     call). ``round_had_result`` is set during a round when it makes
        #     progress; the counter is settled at the TOP of the next iteration
        #     (a single chokepoint every non-terminal path loops back through),
        #     so a productive tool round does NOT count and the count is of
        #     COMPLETED no-result rounds (no off-by-one). Fires at
        #     ``_teacher_no_result_limit``.
        # Both reset to 0 the moment the teacher fires (it just helped) and on
        # a Stop-hook continuation (the segment restarts with fresh guidance).
        tool_error_streak = 0
        no_result_rounds = 0
        # Read-only *research* round streak: consecutive Grep/Glob/Read/Web
        # rounds with no visible answer. Agent(list) / BashOutput polls are
        # waits (neutral). Resets on any mutating call or user-visible content.
        readonly_round_streak = 0
        # Whether the soft-limit synthesize nudge already fired this turn —
        # prevents repeated injections.
        _readonly_soft_fired = False
        # Once the hard read-only limit injects a finalization request, permit
        # one more round. A further read-only-only round then ends gracefully.
        readonly_finalize_fired = False
        # Zero-output guarantee: shared
        # one-shot guard across ALL graceful-finalize exits (read-only budget
        # + sustained-zero-progress, both sites). When a finalize_stop is
        # about to end a turn whose visible output is ZERO, inject ONE
        # tool-less "write the answer now" round instead of breaking
        # silently. Bounded to one extra round per turn no matter which
        # guard trips; a model that ignores it ends at the next verdict.
        final_answer_round_fired = False
        # Seeded True so the FIRST iteration settles to 0 (there is no prior
        # round to judge); set True by any round that produces a result.
        round_had_result = True
        (_teacher_tool_fail_threshold, _teacher_no_result_limit) = self._teacher_round_thresholds()

        # Fresh turn → clear any per-turn budget state left over from a
        # previous turn on this long-lived session. The max-steps handoff
        # flag disables tools for a turn's wrap-up round; without this reset
        # it would stay set forever once any turn hit ``MAX_ROUNDS_AUTO``,
        # denying ALL tools on every subsequent turn (the bug surfaced as a
        # bogus "not in this agent's allowlist" denial).
        self._reset_turn_budget_state(sharedagents_sid, session)
        if manual_budget_exhausted:
            # Reuse the existing handoff/allowlist chokepoints: the wire and
            # dispatch gate both expose an empty tool surface, and the outer
            # loop cannot re-drive the turn after its single reply.
            self._max_steps_handoff[sharedagents_sid] = True
            self._budget_handoff[sharedagents_sid] = True

        # Round budget is shared ACROSS Stop-hook continuations — a
        # blocking Stop hook gets whatever's left of ``max_rounds``, not
        # a full fresh budget every time. (Without this, 3 continuations
        # could spin the agent for 4× the configured cap.)
        round_idx = 0
        try:
            # Outer loop: a blocking ``Stop`` hook re-enters the round
            # loop with the hook's reason appended as a fresh user
            # message (bounded by ``_MAX_STOP_CONTINUATIONS``).
            while True:
                while max_rounds is None or round_idx < max_rounds:
                    if self._cancelled.get(sharedagents_sid):
                        terminal_state = "aborted"
                        break
                    # Graceful stop (``/stop`` / first ESC): the prior
                    # round (and its tool batch) has fully wound down by
                    # the time we loop back here, so this is the safe
                    # point to end the turn WITHOUT killing anything. A
                    # later hard abort still wins via the ``_cancelled``
                    # check above.
                    if self._graceful_stop.get(sharedagents_sid):
                        terminal_state = "aborted"
                        break

                    # --- proactive teacher: consecutive no-result rounds -----
                    # Single chokepoint every non-terminal path loops back
                    # through. Settle the counter for the round that JUST
                    # completed: a round that delivered a result resets it, a
                    # round with no result advances it. When the run hits
                    # ``_teacher_no_result_limit`` CONSECUTIVE completed
                    # no-result rounds the teacher fires to HELP the student,
                    # then the counter resets (it just helped) and the
                    # continuation is appended as a fresh user message so the
                    # upcoming round acts on the guidance.
                    #
                    # Firing ONLY here (mid-grind, with rounds still left) is
                    # deliberate: the continuation is help the student can act on
                    # next round. The final round before a budget-cap exit is NOT
                    # settled here, and that is correct — firing at the cap would
                    # append guidance the terminating turn can never use; the
                    # turn-end review (pre_idle / _teacher_terminal_review) owns
                    # the boundary instead. In AUTO mode (max_rounds=35) the
                    # default limit (20) is reached and fired with rounds to
                    # spare.
                    # Tri-state ``round_had_result``:
                    #   True  → the prior round delivered a result → reset count.
                    #   False → the prior round completed with no result → +1.
                    #   None  → the prior "round" was an INFRA retry (transient
                    #           upstream / reactive max-tokens fallback), not a
                    #           completed student round → TRANSPARENT: neither
                    #           increment nor reset (a 502 on round 20 must not
                    #           wipe a real grind streak, nor inflate it).
                    if round_had_result is None:
                        pass
                    elif round_had_result:
                        no_result_rounds = 0
                    else:
                        no_result_rounds += 1
                    # Reset the flag for the round about to run; a productive
                    # round sets it back to True below.
                    round_had_result = False
                    # The round about to run has NOT yet
                    # delivered a wrap-up — reset the booked flag so the
                    # terminal chokepoint's F4 gate never trusts a PREVIOUS
                    # round's text for THIS round (an infra failure or a
                    # hook-stop on the final round leaves it unbooked → the
                    # cap reports error + round_cap_no_wrap_up, never a stale
                    # ok).
                    final_round_booked = False
                    if (
                        max_rounds is not None
                        and round_idx >= max_rounds - 1
                        and round_idx < max_rounds
                        and not self._max_steps_handoff.get(sharedagents_sid)
                    ):
                        self._max_steps_handoff[sharedagents_sid] = True
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"[sharedagents max-steps] You reached this turn's "
                                    f"per-turn step limit ({max_rounds} rounds). This "
                                    f"is a STEP limit, not a tool error — tools are "
                                    f"disabled for this final round only. " + MAX_STEPS_HANDOFF_USER
                                ),
                            }
                        )
                    # Cost budget gate — cost is the only limit on goal/prompt
                    # work. When this turn's accrued spend (in-turn accumulator +
                    # prior ledger spend for a goal) reaches the budget, hand off
                    # like max-steps: disable tools for a final summary round.
                    # For a goal the turn finalizer then pauses it with reason
                    # cost_budget_exhausted (resumable via topup); for a non-goal
                    # prompt the prose response ends the turn. The retained
                    # max_rounds cap is the within-turn backstop either way.
                    if not self._budget_handoff.get(sharedagents_sid):
                        _budget_msg = self._budget_handoff_message(session, sharedagents_sid)
                        if _budget_msg:
                            self._budget_handoff[sharedagents_sid] = True
                            messages.append(
                                {
                                    "role": "user",
                                    "content": _budget_msg,
                                }
                            )
                    if (
                        not manual_budget_exhausted
                        and self._teacher_enabled(sharedagents_sid)
                        and _teacher_no_result_limit > 0
                        and no_result_rounds >= _teacher_no_result_limit
                    ):
                        async for _ev in self._fire_no_result(sharedagents_sid, messages, no_result_rounds):
                            yield _ev
                        # COOLDOWN: reset after EVERY fire (helped, vetoed, or
                        # fail-open) so the next fire needs another full
                        # ``_teacher_no_result_limit`` no-result rounds. This
                        # grants a fresh budget after guidance, and on a teacher
                        # OUTAGE re-attempts once per budget (the fire is still
                        # attempted + reported ok=False, so it is never swallowed)
                        # instead of firing every single round. Reset BOTH proactive
                        # counters — a no-result fire is a fresh intervention, so
                        # the student also gets a fresh tool-error budget (symmetry
                        # with the reasoning-loop / Stop-hook resets).
                        no_result_rounds = 0
                        tool_error_streak = 0

                    # Mid-turn user-message inject drain. Any prompts
                    # the frontend pushed via ``inject_user_message``
                    # while the previous round was streaming land here
                    # as fresh user-role messages — so a slow plan can
                    # pivot on the steering input on its very next
                    # worker spawn instead of waiting for the turn to
                    # end. Drained AFTER the cancel check (so an abort
                    # racing the inject still wins) and BEFORE the
                    # preamble splice (so the new user turn appears
                    # below the standard pinned/summary/todos blocks).
                    #
                    # Each inject is wrapped in an explicit override
                    # directive — without it, qwen3.6-27b treats the
                    # new prompt as a follow-up question (observed:
                    # the model finished a 36K-char
                    # categorized table AND addressed the steer
                    # instead of dropping the prior plan). The marker
                    # makes "user is changing direction" explicit so
                    # the small model knows to drop expansion of the
                    # prior plan and act on the new directive alone.
                    pending = self._drain_pending_user_messages(
                        sharedagents_sid,
                    )
                    for text in pending:
                        wrapped = (
                            "⚠ MID-TURN USER STEER — the user has "
                            "interrupted with a new instruction. Stop "
                            "expanding on the prior plan. Treat this "
                            "as the operative request; any prior tool "
                            "work counts only if it directly serves "
                            "this new directive.\n\n"
                            f"{text}"
                        )
                        messages.append({"role": "user", "content": wrapped})
                        yield UserMessageInjected(
                            text=text,
                            round_idx=round_idx,
                        )

                    # Durable follow-up queue (session JSONL queue_msg_*).
                    # Parent Agent(action=steer) writes source="steer" into the
                    # same log — resync+drain picks it up cross-process.
                    # Gentle for ordinary type-ahead;
                    # steer-sourced items get the mid-turn wrapper.
                    # SELF-GATES on open-turn state.
                    _qdrain = (
                        None
                        if (manual_budget_exhausted or (coordination is not None and coordination.delegated))
                        else request.extra.get("queue_drain")
                    )
                    # A fast wave can finish during the post-dispatch reply.
                    # Leave its durable result queued for a FRESH turn: consuming
                    # it here would strand the next phase behind this turn's
                    # one-wave latch. Owner cancellation still acts immediately.
                    if _qdrain is not None:
                        try:
                            _q_items = _qdrain() or []
                        except Exception:  # noqa: BLE001 — a queue error must never break the turn
                            _q_items = []
                        for _qitem in _q_items:
                            if len(_qitem) >= 3:
                                _qid, _qtext, _qsrc = (
                                    _qitem[0],
                                    _qitem[1],
                                    _qitem[2],
                                )
                            else:
                                _qid, _qtext, _qsrc = (
                                    _qitem[0],
                                    _qitem[1],
                                    "",
                                )
                            if str(_qsrc) == "steer":
                                _qtext = (
                                    "⚠ MID-TURN USER STEER — the user has "
                                    "interrupted with a new instruction. Stop "
                                    "expanding on the prior plan. Treat this "
                                    "as the operative request; any prior tool "
                                    "work counts only if it directly serves "
                                    "this new directive.\n\n"
                                    f"{_qtext}"
                                )
                                yield UserMessageInjected(
                                    text=_qitem[1],
                                    round_idx=round_idx,
                                )
                            messages.append({"role": "user", "content": _qtext})
                            if str(_qsrc) != "steer":
                                yield QueueMessageInjected(
                                    text=_qtext,
                                    queue_id=_qid,
                                    round_idx=round_idx,
                                )

                    # The Agent-composed preamble is folded into the LIVE
                    # user turn by `_build_messages_simple` (the rolling
                    # user-role tail) and rides there for the whole turn —
                    # compaction protects the most-recent user message, so
                    # the preamble survives every round without a per-round
                    # re-splice. (It used to be re-inserted as leading
                    # `system` blocks here, which collapsed into message[0]
                    # and made the cache anchor dynamic every turn.)
                    # PreCompact only fires when compaction will
                    # actually fire (cheap threshold check), not every
                    # round_idx > 0 — an expensive PreCompact hook
                    # firing 20 times per turn was the original
                    # complaint.
                    #
                    # A PreCompact hook can SUBSTITUTE the compaction
                    # strategy via two output shapes:
                    #   * ``hookSpecificOutput.replacementMessages``
                    #     (a list of {role, content} dicts) — the
                    #     engine uses these wholesale instead of
                    #     running ``_maybe_compact``.
                    #   * ``decision: block`` with no replacement →
                    #     compaction is SKIPPED entirely (operator
                    #     asks the engine to push the unmodified
                    #     message list at the model and risk a
                    #     context-overflow error, which the
                    #     reactive-compact path catches).
                    compaction_handled_by_hook = False
                    if hook_engine is not None and round_idx > 0:
                        _ctx = int(self._effective_context_window(session) or 0)
                        if _ctx > 0:
                            try:
                                from sharedagents_agent.turn_runtime import (
                                    _approx_tokens_image_aware,
                                )

                                # Hooks and the built-in compactor share the
                                # model-specific gate, including output
                                # headroom and the no-progress/min-turn guards.
                                if (
                                    self._compaction_target(
                                        session,
                                        messages,
                                    )
                                    is not None
                                ):
                                    _est = self._last_authoritative_prompt_tokens(
                                        session,
                                    ) or _approx_tokens_image_aware(messages)
                                    pc = await self._fire_hook(
                                        hook_engine,
                                        hook_ctx,
                                        "PreCompact",
                                        {
                                            "trigger": "auto",
                                            "custom_instructions": "",
                                            "current_tokens": _est,
                                            "context_window": _ctx,
                                        },
                                    )
                                    # Honor replacementMessages (sub-
                                    # stitute strategy) or block:skip.
                                    for r in pc.results or []:
                                        j = getattr(r, "json_output", None)
                                        if not isinstance(j, dict):
                                            continue
                                        hso = j.get("hookSpecificOutput")
                                        if isinstance(hso, dict):
                                            rm = hso.get(
                                                "replacementMessages",
                                            )
                                            if isinstance(rm, list) and all(isinstance(m, dict) for m in rm):
                                                messages = list(rm)
                                                compaction_handled_by_hook = True
                                                break
                                    if not compaction_handled_by_hook and pc.blocked:
                                        # Hook asked to skip compaction
                                        # without a substitute. Leave
                                        # ``messages`` unchanged.
                                        compaction_handled_by_hook = True
                            except Exception:  # noqa: BLE001
                                pass
                    if not compaction_handled_by_hook:
                        # Hermes-style LLM compaction: summarize the dropped
                        # span instead of deleting it, so the small model
                        # keeps the thread. Falls back to the mechanical
                        # squeeze on any failure (never worse than before).
                        # Automatic compaction depends on live context
                        # occupancy, never cumulative tokens processed across
                        # rounds. Replaying a small prompt can spend many tokens
                        # while remaining far below the selected model's window.
                        messages = await self._maybe_compact_llm(
                            session,
                            messages,
                            trigger="threshold",
                            sharedagents_sid=sharedagents_sid,
                        )
                    # Compaction can block this round for a long time (a large
                    # LLM summary). A hard ``/abort`` that landed WHILE it ran
                    # set ``_cancelled`` but the top-of-loop checkpoint already
                    # passed, so re-check here BEFORE spawning another worker
                    # round — otherwise the operator's abort doesn't land until
                    # a full extra round runs. Graceful
                    # ``/stop`` is honoured too: this is a safe round boundary.
                    if self._cancelled.get(sharedagents_sid) or self._graceful_stop.get(sharedagents_sid):
                        terminal_state = "aborted"
                        break
                    self._log_round(
                        sharedagents_sid,
                        turn_id,
                        round_idx,
                        messages,
                        [],
                    )

                    # Spawn the worker round. Pass len(nudge_log) so
                    # _rotate_seed fires on continuation-nudge retries —
                    # see _default_spawn_worker (acp_bridge.py:581).
                    try:
                        turn_sampling = request.extra.get("sampling")
                        if not isinstance(turn_sampling, dict):
                            turn_sampling = None
                        turn_state, event_stream = await self._spawn_worker(
                            session=session,
                            messages=messages,
                            turn_id=turn_id,
                            env=env,
                            nudge_retry_count=len(nudge_log),
                            sampling=turn_sampling,
                            endpoint=llm_fallback_ep,
                            model=llm_fallback_model,
                        )
                    except Exception as exc:  # noqa: BLE001
                        # Always surface the exception TYPE — several spawn
                        # failure modes (asyncio.TimeoutError, a bare OSError,
                        # ConnectionResetError) stringify to "" and produced
                        # the useless "worker spawn failed: " with no clue
                        # what went wrong. Mirror the codebase idiom
                        # (`{type(exc).__name__}: {exc}`) so the next failure
                        # is diagnosable.
                        detail = f"{type(exc).__name__}: {exc}".rstrip(": ")
                        # Route this in-process turn-runtime crash boundary
                        # through structured capture so the SessionError-hook
                        # machinery fires uniformly across surfaces, not only
                        # in the TUI. Best-effort — capture
                        # must never mask the TurnError the caller sees.
                        try:
                            from sharedagents_core import errors as _errors

                            _errors.log_error(
                                "agent:local_llm",
                                exc,
                                summary=self._classify_upstream_error(str(exc)),
                                context={"turn_id": turn_id, "phase": "worker_spawn"},
                                session_id=sharedagents_sid,
                                agent_name=getattr(self.cfg, "name", None),
                            )
                        except Exception:  # noqa: BLE001
                            pass
                        yield TurnError(
                            message=f"worker spawn failed: {detail}",
                            classification="spawn_failed",
                        )
                        terminal_state = "error"
                        break
                    self._active_turns[turn_id] = turn_state
                    # Thread the ORIGINAL user task onto the turn_state so the
                    # dispatcher's single-phase specialist reroute
                    # (_dispatch_tool_calls → runners._shim_agent_canonical) can
                    # read it via ``getattr(turn, "user", "")``. TurnState itself
                    # never sets ``.user`` (acp_bridge builds it), so without this
                    # the reroute silently no-ops LIVE — the model's generic
                    # explore/general delegation is never corrected to the right
                    # specialist. (Root-caused 2026-07-16: the reroute passed its
                    # unit tests but never fired in the gateway/CLI path.)
                    try:
                        turn_state.user = user_content
                    except Exception:
                        pass
                    proc = getattr(turn_state, "proc", None)
                    stderr_task: asyncio.Task[None] | None = None
                    if proc is not None and proc.stderr is not None:
                        stderr_task = asyncio.create_task(
                            _drain_stderr(proc.stderr, sharedagents_sid, turn_id),
                        )

                    rs = _RoundState()
                    nudge_tracker.begin_round()
                    # Gate the runtime streaming repetition/loop detector per
                    # model: only qwen3.6 (fp8 + int4) and gemma have the
                    # documented token/paragraph repeat failure mode the detector
                    # exists for. deepseek-v4-flash (and other GENERIC models)
                    # do NOT loop that way and were FALSE-POSITIVING on
                    # legitimate repeated structure in long reasoning — keep the
                    # monitor OFF for them.
                    from sharedagents_model.models import needs_loop_detection

                    _ld_sess = self._sessions.get(sharedagents_sid)
                    # Master kill-switch (per-agent OR per-model-row), resolved
                    # once per round and AND-ed into every protection below.
                    _prot_off = self._protections_off(sharedagents_sid)
                    _loop_detect_on = (
                        needs_loop_detection(
                            (getattr(_ld_sess, "model", "") or "") or getattr(self.cfg, "upstream_model", "") or ""
                        )
                        and not _prot_off
                    )
                    round_error: str | None = None
                    round_error_class = ""
                    had_done = False
                    try:
                        worker_idle_timeout_s = _local_worker_idle_timeout_s()
                        event_iter = event_stream.__aiter__()
                        while True:
                            try:
                                ev = await _next_worker_event(
                                    event_iter,
                                    worker_idle_timeout_s,
                                )
                            except StopAsyncIteration:
                                break
                            except asyncio.TimeoutError:
                                secs = f"{worker_idle_timeout_s:g}"
                                round_error = f"local worker produced no events for {secs}s (engine idle timeout)"
                                round_error_class = "worker_idle_timeout"
                                try:
                                    turn_state.stop()
                                except Exception:  # noqa: BLE001
                                    pass
                                break
                            if self._cancelled.get(sharedagents_sid):
                                try:
                                    turn_state.stop()
                                except Exception:  # noqa: BLE001
                                    pass
                                break
                            if not isinstance(ev, dict):
                                continue
                            kind = ev.get("kind") or ""
                            if (
                                kind
                                in {
                                    "thinking",
                                    "content",
                                    "tool_calls",
                                    "tool_call",
                                    "tool_result",
                                }
                                and request.handoff is not None
                                and not request.extra.get("_handoff_consumed")
                            ):
                                request.extra["_handoff_consumed"] = True
                                yield HandoffAccepted(
                                    handoff_id=request.handoff.handoff_id,
                                    through_seq=request.handoff.through_seq,
                                )
                            if kind == "thinking":
                                txt = str(ev.get("text") or "")
                                if txt:
                                    rs.thinking_buf.append(txt)
                                    nudge_tracker.on_thinking(txt)
                                    # Privacy gate: stream thinking
                                    # deltas to the frontend only when
                                    # the operator opted in. Buffering
                                    # for the session record stays on
                                    # so the turn audit still has the
                                    # trace (subject to
                                    # session.persist_reasoning_traces).
                                    # Read off the bound Session — the
                                    # field is per-session by design and
                                    # `set_config_option("reasoning",…)`
                                    # mutates it on the same object —
                                    # AgentConfig deliberately has no
                                    # such field (frontend-driven only).
                                    if (
                                        str(
                                            getattr(
                                                session,
                                                "reasoning_visibility",
                                                "off",
                                            )
                                            or "off"
                                        ).lower()
                                        != "off"
                                    ):
                                        yield TextDelta(
                                            text=txt,
                                            channel="thinking",
                                        )
                                    # Mid-stream loop check on the
                                    # reasoning buffer. Mirrors the
                                    # content-channel block below: tier-0
                                    # token collapse and tier-1
                                    # paraphrase rotation in thinking-
                                    # mode CoT (Qwen3.6's most common
                                    # failure) must surface BEFORE the
                                    # model burns its thinking_budget.
                                    # Privacy gate on the yield above
                                    # keeps the trace off the wire; the
                                    # detector still runs on the local
                                    # buffer.
                                    hit, rs.thinking_loop_check_chars = self._check_streamed_loop(
                                        rs.thinking_buf,
                                        rs.thinking_loop_check_chars,
                                        "thinking",
                                        rs.loop_sustain.setdefault(
                                            "thinking",
                                            {},
                                        ),
                                        loop_detect=_loop_detect_on,
                                    )
                                    # Retired surface tiers (tight / ngram)
                                    # no longer abort — the FP audit measured
                                    # them at ~0-15% precision. Only the
                                    # high-precision degenerate-token tiers
                                    # (entropy / entropy_bigram / ratio) still
                                    # abort the round; their turn-level action
                                    # is now the progress nudge, not a hard
                                    # kill. Soft-log surface hits so path/
                                    # bullet-list FPs remain auditable.
                                    if hit is not None and hit[0] not in (_PROGRESS_KEPT_TEXT_TIERS):
                                        self._log_event(
                                            "TEXT_LOOP_SOFT",
                                            tier=hit[0],
                                            evidence=str(hit[1] or "")[:80],
                                            channel="thinking",
                                            round_idx=round_idx,
                                            session_id=getattr(
                                                session,
                                                "hash",
                                                "",
                                            ),
                                        )
                                    if hit is not None and hit[0] in _PROGRESS_KEPT_TEXT_TIERS:
                                        rs.loop_detected = True
                                        rs.loop_tier, rs.loop_evidence = hit
                                        # Persist so the metrics-merge at
                                        # :1577 records it in the session
                                        # JSON. Without this, engine-side
                                        # fires (thinking/content) were
                                        # logged-and-lost: only the worker
                                        # ``tool_args_loop`` path
                                        # populated loop_event.
                                        rs.loop_event = {
                                            "tier": rs.loop_tier,
                                            "evidence": rs.loop_evidence,
                                            "channel": "thinking",
                                        }
                                        try:
                                            turn_state.stop()
                                        except Exception:  # noqa: BLE001
                                            pass
                                        break
                            elif kind == "content":
                                txt = str(ev.get("text") or "")
                                if txt:
                                    rs.content_buf.append(txt)
                                    nudge_tracker.on_content(txt)
                                    yield TextDelta(
                                        text=txt,
                                        channel="content",
                                    )
                                    turn_emitted_content_parts.append(txt)
                                    hit, rs.content_loop_check_chars = self._check_streamed_loop(
                                        rs.content_buf,
                                        rs.content_loop_check_chars,
                                        "content",
                                        rs.loop_sustain.setdefault(
                                            "content",
                                            {},
                                        ),
                                        loop_detect=_loop_detect_on,
                                    )
                                    # Retired tiers (tight / ngram) no longer
                                    # abort — only degenerate-token tiers do
                                    # (see the thinking branch above). Soft-log
                                    # surface hits (path/bullet-list FPs).
                                    if hit is not None and hit[0] not in (_PROGRESS_KEPT_TEXT_TIERS):
                                        self._log_event(
                                            "TEXT_LOOP_SOFT",
                                            tier=hit[0],
                                            evidence=str(hit[1] or "")[:80],
                                            channel="content",
                                            round_idx=round_idx,
                                            session_id=getattr(
                                                session,
                                                "hash",
                                                "",
                                            ),
                                        )
                                    if hit is not None and hit[0] in _PROGRESS_KEPT_TEXT_TIERS:
                                        rs.loop_detected = True
                                        rs.loop_tier, rs.loop_evidence = hit
                                        # Symmetric persist for the
                                        # content-channel fire — see
                                        # the thinking branch above.
                                        rs.loop_event = {
                                            "tier": rs.loop_tier,
                                            "evidence": rs.loop_evidence,
                                            "channel": "content",
                                        }
                                        try:
                                            turn_state.stop()
                                        except Exception:  # noqa: BLE001
                                            pass
                                        break
                            elif kind == "tool_calls":
                                for c in ev.get("calls") or ev.get("tool_calls") or []:
                                    if isinstance(c, dict):
                                        rs.pending_tool_calls.append(dict(c))
                                if isinstance(ev.get("reasoning_replay"), dict):
                                    rs.reasoning_replay = ev["reasoning_replay"]
                                from sharedagents_agent.tools.impl._utils import (
                                    worker_idle_timeout_s_for_pending,
                                )

                                worker_idle_timeout_s = worker_idle_timeout_s_for_pending(
                                    [str(c.get("name") or "") for c in rs.pending_tool_calls if isinstance(c, dict)],
                                    worker_idle_timeout_s,
                                )
                            elif kind == "leaked_tool_call_recovered":
                                # The worker reconstructed a tool call the
                                # upstream parser missed; its raw markup was
                                # already streamed as ``content`` and sits in
                                # ``rs.content_buf``. Replace the round's
                                # buffered content with the cleaned prose so
                                # the markup is not persisted into history (or
                                # re-fed to the model next round). The live
                                # stream already showed it, but the recovered
                                # turn continues with the tool round, so the
                                # transient markup is superseded.
                                cleaned = ev.get("cleaned_content")
                                if cleaned is not None:
                                    rs.content_buf = [str(cleaned)]
                            elif kind == "tool_args_loop" and not _prot_off:
                                # Master kill-switch: an unprotected model's
                                # runaway-arg self-abort must not turn into a
                                # turn-level loop force-terminate.
                                rs.loop_detected = True
                                # Honour the worker's own tier label
                                # (entropy / ngram / tight / ratio) so
                                # the recovery hook and the persisted
                                # round metrics carry the actual
                                # detector that fired, not a stub.
                                rs.loop_tier = str(
                                    ev.get("tier") or "ratio",
                                )
                                rs.loop_evidence = str(
                                    ev.get("evidence") or "tool_args_loop",
                                )
                                # Stash diagnostic detail so it
                                # survives the abort into round
                                # metrics — without this the abort
                                # branch breaks before ``rs.metrics``
                                # is set, so the loop event was
                                # logged-and-lost.
                                rs.loop_event = {
                                    "tier": rs.loop_tier,
                                    "evidence": rs.loop_evidence,
                                    "args_chars": ev.get("args_chars"),
                                    "checked_at": ev.get("checked_at"),
                                    "tool_name": str(ev.get("name") or ""),
                                    "tool_index": ev.get("index"),
                                    "channel": "tool_args",
                                }
                            elif kind == "tool_args_loop_soft":
                                # Surface ngram/tight hit on tool_args —
                                # telemetry only (worker kept streaming).
                                # Do NOT set loop_detected / abort recovery.
                                self._log_event(
                                    "TOOL_ARGS_LOOP_SOFT",
                                    tier=str(ev.get("tier") or ""),
                                    evidence=str(ev.get("evidence") or "")[:80],
                                    tool_name=str(ev.get("name") or ""),
                                    args_chars=ev.get("args_chars"),
                                    session_id=getattr(session, "hash", ""),
                                )
                            elif kind == "metrics":
                                rs.metrics = {k: v for k, v in ev.items() if k != "kind"}
                            elif kind == "aborted":
                                # Three abort shapes:
                                #
                                # 1. Soft-abort flag set by
                                #    ``inject_user_message`` — user
                                #    steered mid-turn; the round-loop
                                #    drains the inject and continues.
                                # 2. Bare ``aborted`` (no reason) =
                                #    operator hit ESC / ACP session/
                                #    cancel sent SIGTERM. Hard cancel.
                                # 3. ``aborted`` with a worker-supplied
                                #    ``reason`` (today only
                                #    ``"tool_args_loop"``) = worker
                                #    self-protective abort because its
                                #    in-flight detector fired. Leave
                                #    ``_cancelled`` alone so the post-
                                #    stream ``rs.loop_detected`` branch
                                #    routes through the existing
                                #    distill / force-terminate /
                                #    continuation-nudge cascade.
                                if self._soft_aborts.pop(
                                    sharedagents_sid,
                                    False,
                                ):
                                    break
                                if str(ev.get("reason") or ""):
                                    break
                                self._cancelled[sharedagents_sid] = True
                                break
                            elif kind == "error":
                                round_error = str(
                                    ev.get("message") or "worker error",
                                )
                                round_error_class = self._classify_upstream_error(round_error)
                                # ``UpstreamError`` hook — lets an ECC
                                # plugin override the classification
                                # (e.g. recognise a model-specific
                                # error string as ``upstream context-
                                # length overflow`` so the engine
                                # fires reactive-compact instead of
                                # giving up). Two control paths:
                                #   * ``hookSpecificOutput.classification``
                                #     rewrites the tag fed to
                                #     ``_reactive_compact_on_overflow``
                                #     AND to the surfaced TurnError
                                #     (so a plugin's classification
                                #     actually STEERS the routing,
                                #     not just the user-facing label).
                                #   * ``decision: block`` with no
                                #     replacement → suppress the
                                #     reactive-compact retry; surface
                                #     the error as-is (operator
                                #     wants a hard stop).
                                if hook_engine is not None:
                                    ue = await self._fire_hook(
                                        hook_engine,
                                        hook_ctx,
                                        "UpstreamError",
                                        {
                                            "message": round_error,
                                            "classification": round_error_class,
                                            "round_idx": round_idx,
                                        },
                                    )
                                    for r in ue.results or []:
                                        j = getattr(r, "json_output", None)
                                        if not isinstance(j, dict):
                                            continue
                                        hso = j.get("hookSpecificOutput")
                                        if not isinstance(hso, dict):
                                            continue
                                        cls = hso.get("classification")
                                        if isinstance(cls, str) and cls.strip():
                                            round_error_class = cls.strip()
                                            break
                                    # If the hook BLOCKED (decision:
                                    # block) without supplying a
                                    # classification override, rewrite
                                    # to a tag that won't trigger
                                    # reactive-compact, so the engine
                                    # surfaces the error verbatim.
                                    if ue.blocked and round_error_class in (
                                        "context_length",
                                        "upstream context-length overflow",
                                    ):
                                        round_error_class = "upstream error (hook-suppressed)"
                                break
                            elif kind == "done":
                                had_done = True
                                break
                    except asyncio.CancelledError:
                        try:
                            turn_state.stop()
                        except Exception:  # noqa: BLE001
                            pass
                        self._cancelled[sharedagents_sid] = True
                    finally:
                        # Reap this round's worker before the next one.
                        # On the ABORT path we must return fast — SIGTERM
                        # now and hand the reap to a background task
                        # (the inline 2 s wait is happy-path only).
                        if self._cancelled.get(sharedagents_sid):
                            if proc is not None and proc.returncode is None:
                                try:
                                    proc.terminate()
                                except Exception:  # noqa: BLE001
                                    pass
                            if proc is not None or stderr_task is not None:
                                self._schedule_background_reap(
                                    proc,
                                    stderr_task,
                                )
                        else:
                            if stderr_task is not None and not stderr_task.done():
                                stderr_task.cancel()
                                try:
                                    await stderr_task
                                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                                    pass
                            if proc is not None:
                                try:
                                    if proc.returncode is None:
                                        proc.terminate()
                                    await asyncio.wait_for(
                                        proc.wait(),
                                        timeout=2.0,
                                    )
                                except asyncio.TimeoutError:
                                    # SIGTERM didn't take — escalate to
                                    # SIGKILL and hand off to the
                                    # background reaper so the round
                                    # doesn't leak a zombie worker.
                                    try:
                                        proc.kill()
                                    except (ProcessLookupError, OSError):
                                        pass
                                    self._schedule_background_reap(proc, None)
                                except ProcessLookupError:
                                    pass
                                except Exception:  # noqa: BLE001
                                    pass
                        self._active_turns.pop(turn_id, None)
                        # Clear soft-abort regardless of how the round
                        # ended. Without this, a SIGTERM that exits
                        # the worker via EOF (no terminal ``aborted``
                        # event) leaves the flag set, and a later
                        # genuine operator abort gets silently
                        # downgraded to a soft pivot.
                        self._soft_aborts.pop(sharedagents_sid, None)

                    # Roll round buffers into the turn aggregates.
                    thinking_buf.extend(rs.thinking_buf)
                    content_buf.extend(rs.content_buf)
                    # FP2: accumulate this round's user-visible content into the
                    # turn-cumulative total consulted by the silent-nudge gate.
                    turn_visible_chars += len("".join(rs.content_buf).strip())
                    # Cross-round semantic-loop detection (false-NEGATIVE fix).
                    # Observe EVERY round's reasoning + the tool calls it issued
                    # so the turn-scoped novelty / progress state stays accurate
                    # even on rounds where another tier fired. On a confirmed
                    # cross-round loop, log telemetry always; only in
                    # "terminate" mode (opt-in; default is "log"/dark after a
                    # prod false positive — see turn_runtime.XROUND_LOOP_ACTION)
                    # route into the existing loop-recovery cascade via
                    # rs.loop_detected unless a more specific detector already
                    # fired this round.
                    try:
                        _xround_hit = xround_tracker.observe(
                            "".join(rs.thinking_buf),
                            list(rs.pending_tool_calls),
                        )
                    except Exception:  # noqa: BLE001
                        _xround_hit = None
                    # Forward per-round telemetry (correct round attribution) on
                    # armed rounds, so the operator can tune novelty/K on real
                    # data before promoting the action to "terminate".
                    _xs = getattr(xround_tracker, "last_stats", None)
                    if _xround_telemetry and isinstance(_xs, dict) and _xs.get("armed"):
                        self._log_event(
                            "XROUND_ROUND",
                            round_idx=round_idx,
                            novelty=_xs.get("novelty"),
                            words=_xs.get("words"),
                            progressed=_xs.get("progressed"),
                            consecutive_stalls=_xs.get("consecutive_stalls"),
                            cum_reasoning_chars=_xs.get("cum_reasoning_chars"),
                            session_id=getattr(session, "hash", ""),
                        )
                    if _xround_hit is not None and not rs.loop_detected:
                        _xr_tier, _xr_ev = _xround_hit
                        self._log_event(
                            "XROUND_LOOP_DETECTED",
                            tier=_xr_tier,
                            evidence=_xr_ev[:200],
                            action=_xround_action,
                            round_idx=round_idx,
                            session_id=getattr(session, "hash", ""),
                        )
                        if _xround_action == "terminate":
                            rs.loop_detected = True
                            rs.loop_tier = _xr_tier
                            rs.loop_evidence = _xr_ev
                            rs.loop_event = {
                                "tier": _xr_tier,
                                "evidence": _xr_ev,
                                "channel": "thinking",
                            }
                    # Fold the loop-detector event (if any) into the
                    # metrics dict so it lands in ``session.round_metrics``
                    # alongside the worker's own numbers. When the worker
                    # self-aborted on a ``tool_args_loop`` it exits 0
                    # before emitting its final ``metrics`` event, so
                    # ``rs.metrics`` is ``None``. Synthesize a minimal
                    # metrics dict in that case so the diagnostic isn't
                    # silently dropped.
                    if rs.loop_event and rs.metrics is None:
                        rs.metrics = {"finish_reason": "tool_args_loop"}

                    # Post-stream loop checks BEFORE the metrics-merge
                    # below — both branches may populate rs.loop_event
                    # which the merge then reads. Moving these checks
                    # AFTER the merge meant ratio/tool_call fires were
                    # silently dropped from round_metrics. The thinking
                    # and content branches fire mid-stream (before this
                    # point) so they were not affected.
                    # Skip the ratio guard on tool-call rounds: a tool-call
                    # round emits only the tool-call JSON as content, so
                    # chars_per_token is legitimately tiny (e.g. 2 content
                    # chars / 168 tokens = 0.012 on session <session-id>)
                    # and must NOT read as a degenerate-token loop. This
                    # mirrors the internal guard in ``_check_ratio_loop``
                    # (finish_reason in {stop, tool_calls} -> None); keeping it
                    # at the call site documents intent and future-proofs the
                    # call if that internal guard is ever refactored.
                    _ratio_finish = (rs.metrics or {}).get("finish_reason", "")
                    # Server-native repetition guard: vLLM's own
                    # ``repetition_detection`` sampler (configured per-model
                    # in models.py, same qwen3.6/gemma families gated by
                    # ``_loop_detect_on``) returns finish_reason="repetition"
                    # directly from the API when it aborts a degenerate
                    # token/pattern loop server-side. That string was
                    # previously unrecognized anywhere in the engine — it
                    # matched neither ``rs.loop_detected`` (so the recovery
                    # cascade below never ran) nor ``classify_turn_end``'s
                    # ``nfr == "stop"`` checks (so no continuation nudge fired
                    # either), so the round — and often the whole turn — died
                    # silently with zero visible content and no recovery
                    # attempt. Confirmed across 25 sessions since 2026-05-27
                    # (e.g. <session-id>: reasoning_chars=5147,
                    # content_chars=0, turn simply ended with nothing shown).
                    # Route it through the SAME cascade as the client-side
                    # ratio/xround/tool_call detectors below — the server
                    # already made the degenerate-loop call authoritatively,
                    # so there's nothing left to re-derive.
                    #
                    # Soft carve-out: if the round ALREADY
                    # delivered user-visible content, treat server
                    # repetition as telemetry only — the answer exists;
                    # forcing progress-recovery would risk truncating or
                    # duplicating a complete reply (same FP class as the
                    # ping→nudge incident). Empty/reasoning-only
                    # repetition still enters the hard recovery path.
                    if not rs.loop_detected and _loop_detect_on and _ratio_finish == "repetition":
                        _rep_content = len("".join(rs.content_buf).strip())
                        _rep_visible = int(turn_visible_chars or 0)
                        if max(_rep_content, _rep_visible) > 0:
                            self._log_event(
                                "SERVER_REPETITION_SOFT",
                                content_chars=_rep_content,
                                turn_visible_chars=_rep_visible,
                                round_idx=round_idx,
                                session_id=getattr(session, "hash", ""),
                            )
                            if rs.metrics is not None:
                                rs.metrics["server_repetition"] = True
                                rs.metrics["server_repetition_soft"] = True
                                # Stick observability without claiming a
                                # hard loop_detected kill.
                                rs.metrics["loop_tier"] = "server_repetition_soft"
                        else:
                            rs.loop_detected = True
                            rs.loop_tier = "server_repetition"
                            rs.loop_evidence = (
                                "vLLM server-side repetition_detection aborted generation (finish_reason=repetition)"
                            )
                            rs.loop_event = {
                                "tier": rs.loop_tier,
                                "evidence": rs.loop_evidence,
                                "channel": "server",
                            }
                    if not rs.loop_detected and rs.metrics and _loop_detect_on and _ratio_finish != "tool_calls":
                        # Same per-model gate as the streamed / xround detectors:
                        # the chars-per-token ratio guard is a repetition
                        # (degenerate-token) check, so it runs ONLY for the
                        # local repetition-prone families. deepseek-v4-* (and
                        # other GENERIC models) skip it — keeps loop monitoring
                        # off for them entirely, per operator policy.
                        ratio_hit = self._check_ratio_loop(
                            rs.metrics,
                            model_name=str(getattr(session, "model", "") or ""),
                            tail_text=("".join(rs.thinking_buf) + "".join(rs.content_buf))[-4096:],
                        )
                        if ratio_hit is not None:
                            rs.loop_detected = True
                            rs.loop_tier, rs.loop_evidence = ratio_hit
                            rs.loop_event = {
                                "tier": rs.loop_tier,
                                "evidence": rs.loop_evidence,
                                "channel": "ratio",
                            }

                    if rs.pending_tool_calls and not rs.loop_detected:
                        # Persistent TOOL_CALL log — one INFO line per call,
                        # emitted BEFORE the loop detector potentially
                        # clears rs.pending_tool_calls. Lets operators
                        # replay any later loop diagnosis from gateway.log
                        # alone. Args truncated to 1200 chars per
                        # call to keep log lines readable.
                        import json as _json_tc

                        for _tc in rs.pending_tool_calls:
                            if not isinstance(_tc, dict):
                                continue
                            # pending_tool_calls holds the flat worker
                            # shape {"id","name","arguments","raw_arguments"};
                            # also accept the nested OpenAI shape. Reading
                            # function.parameters logged blank name/args.
                            _fn = _tc.get("function") if isinstance(_tc.get("function"), dict) else {}
                            _name = str(_fn.get("name") or _tc.get("name") or "")
                            _params = _fn.get("arguments")
                            if _params is None:
                                _params = _tc.get("arguments")
                            if isinstance(_params, dict):
                                _args_repr = _json_tc.dumps(
                                    _params,
                                    ensure_ascii=False,
                                    sort_keys=True,
                                )
                            else:
                                _args_repr = str(_params or "")
                            self._log_event(
                                "TOOL_CALL",
                                session_id=getattr(session, "hash", ""),
                                round_idx=round_idx,
                                name=_name,
                                args=_args_repr[:1200],
                            )

                        # Identical-tool-call repetition detector — fires
                        # when the model retries the same call set across
                        # _TOOL_CALL_LOOP_WINDOW (10) consecutive rounds.
                        # Routes to the existing loop-recovery cascade
                        # via rs.loop_detected. Hooks at per-turn scope
                        # (tool_call_history declared above at :1080).
                        # NOTE: the BROAD (near-duplicate) tool-call detector
                        # (``tool_call_broad``) was RETIRED — the FP audit
                        # measured it at ~0-15% precision (it fired on
                        # legitimate varied search over one tree). The strict
                        # VERBATIM detector below is kept (window 10) but now
                        # routes through the sustained-zero-progress recovery
                        # (nudge → graceful finalize), not distill→terminate.
                        from sharedagents_agent.turn_runtime import (
                            _TOOL_CALL_LOOP_BROAD_WINDOW,
                            _TOOL_CALL_LOOP_WINDOW,
                            _detect_tool_call_loop,
                        )

                        if not _prot_off and _detect_tool_call_loop(
                            rs.pending_tool_calls,
                            tool_call_history,
                        ):
                            rs.loop_detected = True
                            rs.loop_tier = "tool_call"
                            rs.loop_evidence = f"identical tool-call set repeated >= {_TOOL_CALL_LOOP_WINDOW} times"
                            rs.loop_event = {
                                "tier": rs.loop_tier,
                                "evidence": rs.loop_evidence,
                                "channel": "tool_call",
                            }
                            # Clear pending so the next-round recovery
                            # cascade owns the request — don't re-execute
                            # or re-detect on the same signature.
                            rs.pending_tool_calls = []
                        else:
                            tool_call_history.append(list(rs.pending_tool_calls))
                            # Retain enough rounds for WHICHEVER detector has
                            # the wider window. The strict window is now 10 and
                            # exceeds the broad window (6), so size the buffer to
                            # the larger of the two — otherwise the strict
                            # detector never accumulates enough history to fire.
                            _loop_hist_cap = (
                                max(
                                    _TOOL_CALL_LOOP_WINDOW,
                                    _TOOL_CALL_LOOP_BROAD_WINDOW,
                                )
                                + 1
                            )
                            if len(tool_call_history) > _loop_hist_cap:
                                tool_call_history.pop(0)

                    # Emit the fire-moment line FIRST (operators tailing
                    # gateway.log see the event immediately) — TURN_METRICS
                    # follows with the merged loop_* fields. Codex round-3
                    # Ordering matched the commit-message narrative.
                    if rs.loop_detected and rs.loop_event:
                        self._log_event(
                            "LOOP_DETECTED",
                            tier=str(rs.loop_event.get("tier") or rs.loop_tier or ""),
                            evidence=str(rs.loop_event.get("evidence") or rs.loop_evidence or "")[:200],
                            channel=str(rs.loop_event.get("channel") or ""),
                            round_idx=round_idx,
                            session_id=getattr(session, "hash", ""),
                        )

                    if rs.metrics:
                        # In-turn COST accumulator (pico-USD) — run for EVERY round
                        # that produced metrics, INCLUDING a phantom (empty/1-token
                        # stop) round: it was still a real LLM API call, so the
                        # provider billed its full prompt. Accumulating BEFORE the
                        # phantom display-suppression keeps billed phantom rounds in
                        # the budget (otherwise they vanish from cost
                        # while still costing money). The usage ledger is only
                        # written at turn end, so the mid-turn budget gate reads
                        # this accumulator for the in-flight turn; at turn end it is
                        # recorded as the turn's cost override.
                        try:
                            from sharedagents_model.usage_ledger import get_pricing_table
                            from sharedagents_model.usage_pricing import price_metrics

                            _tbl = get_pricing_table(getattr(self.cfg, "pricing_path", "") or None)
                            self._turn_cost_pico[sharedagents_sid] = self._turn_cost_pico.get(
                                sharedagents_sid, 0
                            ) + int(
                                price_metrics(
                                    rs.metrics,
                                    _tbl,
                                    when=datetime.now(timezone.utc),
                                ).total_pico
                            )
                        except Exception:  # noqa: BLE001
                            pass
                        if _is_phantom_round_metrics(rs.metrics, content_buf):
                            self._log_event(
                                "PHANTOM_ROUND_SUPPRESSED",
                                session_id=getattr(session, "hash", ""),
                                round_idx=round_idx,
                                completion_tokens=rs.metrics.get("completion_tokens"),
                            )
                        else:
                            if rs.loop_event:
                                # Assignment (not setdefault): worker metrics
                                # often arrive with loop_detected=false /
                                # missing tier. setdefault left finish
                                # payloads stuck at loop_detected=false even
                                # after server_repetition / tool_args_loop
                                # set rs.loop_event (2026-07 session audit).
                                rs.metrics["loop_detected"] = True
                                rs.metrics["loop_tier"] = rs.loop_event.get("tier") or ""
                                rs.metrics["loop_evidence"] = rs.loop_event.get("evidence") or ""
                                if rs.loop_event.get("args_chars") is not None:
                                    rs.metrics["loop_args_chars"] = rs.loop_event.get("args_chars")
                                if rs.loop_event.get("checked_at") is not None:
                                    rs.metrics["loop_checked_at"] = rs.loop_event.get("checked_at")
                                if rs.loop_event.get("tool_name"):
                                    rs.metrics["loop_tool_name"] = rs.loop_event.get("tool_name") or ""
                                if rs.loop_event.get("channel"):
                                    rs.metrics["loop_channel"] = rs.loop_event.get("channel") or ""
                            last_metrics = rs.metrics
                            self._record_round_metrics(
                                session,
                                round_idx,
                                rs.metrics,
                                str(rs.metrics.get("endpoint") or ""),
                            )

                    if self._cancelled.get(sharedagents_sid):
                        terminal_state = "aborted"
                        break

                    if round_error is not None:
                        if round_error_class in (
                            "context_length",
                            "upstream context-length overflow",
                        ):
                            if consecutive_overflow_retries >= _MAX_REACTIVE_COMPACT_RETRIES:
                                self._log_event(
                                    "COMPACTION_REACTIVE_EXHAUSTED",
                                    session_id=sharedagents_sid,
                                    retries=consecutive_overflow_retries,
                                )
                                yield TurnError(
                                    message=round_error,
                                    classification=round_error_class,
                                )
                                terminal_state = "error"
                                break
                            consecutive_overflow_retries += 1
                        # Reactive squeeze + retry on a context-overflow.
                        # Threads the hook-overridable
                        # ``round_error_class`` (set by the
                        # UpstreamError hook above) so a plugin's
                        # classification rewrite actually steers the
                        # routing decision — not just the user-facing
                        # error label.
                        prev_override = getattr(
                            session,
                            "max_tokens_override",
                            None,
                        )
                        new_messages, retried = await self._reactive_compact_on_overflow(
                            session,
                            messages,
                            round_error,
                            classification=round_error_class,
                        )
                        if retried:
                            # The pre-send count missed (or the model's window is
                            # smaller than configured): tell the client what the
                            # upstream said and what we are doing about it.
                            yield TextDelta(
                                text=self._overflow_notice(session, round_error),
                                channel="content",
                            )
                        if retried and self._reactive_via_llm.pop(
                            getattr(session, "hash", "") or "",
                            False,
                        ):
                            # Successful rounds between overflows permit further
                            # compaction during long tasks. Repeated rejection
                            # without a successful round is bounded above.
                            messages = new_messages
                            round_had_result = None
                            continue
                        if retried:
                            # Distinguish the two retry paths. A real
                            # ``compact_messages`` squeeze leaves
                            # ``max_tokens_override`` untouched and is
                            # self-limiting (it actually drops messages).
                            # The max-tokens fallback shrinks the override
                            # WITHOUT shrinking the prompt, so if the retry
                            # 400s again it re-fires every round — that's
                            # the storm. Bound only the fallback path; the
                            # override changing this round is its signature.
                            cur_override = getattr(
                                session,
                                "max_tokens_override",
                                None,
                            )
                            used_maxtok_fallback = isinstance(cur_override, int) and cur_override != prev_override
                            if used_maxtok_fallback:
                                reactive_maxtok_retries += 1
                                if reactive_maxtok_retries > _MAX_REACTIVE_MAXTOK_FALLBACKS:
                                    # The max-tokens fallback already fired
                                    # this turn and the request STILL
                                    # overflowed — shrinking further only
                                    # storms the upstream. Surface the error.
                                    self._log_event(
                                        "COMPACTION_REACTIVE_FALLBACK_EXHAUSTED",
                                        session_id=getattr(
                                            session,
                                            "hash",
                                            "",
                                        ),
                                        retries=reactive_maxtok_retries,
                                    )
                                    yield TurnError(
                                        message=round_error,
                                        classification=round_error_class,
                                    )
                                    terminal_state = "error"
                                    break
                            else:
                                # Real compact_messages progress path —
                                # OpenCode retries the overflow step once.
                                reactive_compact_retries += 1
                                if reactive_compact_retries > _MAX_REACTIVE_COMPACT_RETRIES:
                                    self._log_event(
                                        "COMPACTION_REACTIVE_EXHAUSTED",
                                        session_id=getattr(
                                            session,
                                            "hash",
                                            "",
                                        ),
                                        retries=reactive_compact_retries,
                                    )
                                    yield TurnError(
                                        message=round_error,
                                        classification=round_error_class,
                                    )
                                    terminal_state = "error"
                                    break
                            messages = new_messages
                            # An infra retry (reactive max-tokens fallback) is
                            # NOT a completed student round — mark it TRANSPARENT
                            # (None) so it neither advances NOR resets the
                            # no-result grind counter.
                            # The retry re-sends the SAME
                            # round — it must NOT increment ``round_idx``
                            # either (no round budget is consumed by an infra
                            # retry; otherwise a retry in the LAST slot
                            # exhausted the loop and the cap terminal trusted
                            # the PREVIOUS round's stale text as the wrap-up).
                            round_had_result = None
                            continue
                        if round_error_class == "worker_idle_timeout":
                            yield TurnError(
                                message=round_error,
                                classification=round_error_class,
                            )
                            terminal_state = "error"
                            break
                        # TRANSIENT upstream error (thinking-inactivity
                        # ReadTimeout / mid-stream connection drop): re-send
                        # the SAME round (no compaction — not a context
                        # problem) with a short backoff, bounded by
                        # ``_MAX_TRANSIENT_RETRIES``. Keeps a complex local-
                        # model task alive across the hiccups a small model /
                        # muxed endpoint inevitably produces, instead of
                        # dying on the first stall. A persistently-wedged
                        # endpoint still surfaces after the cap.
                        _transient_cap = int(
                            getattr(self.cfg, "max_transient_retries", _MAX_TRANSIENT_RETRIES) or _MAX_TRANSIENT_RETRIES
                        )
                        if _is_transient_upstream(round_error) and transient_retries < _transient_cap:
                            transient_retries += 1
                            self._log_event(
                                "TRANSIENT_UPSTREAM_RETRY",
                                session_id=getattr(session, "hash", ""),
                                attempt=transient_retries,
                                cap=_transient_cap,
                                error=round_error[:200],
                            )
                            yield TextDelta(
                                text=_format_runtime_warning(
                                    "upstream-retry",
                                    reason=(
                                        f"transient upstream error "
                                        f"(retry {transient_retries}/{_transient_cap}): "
                                        f"{round_error[:120]}"
                                    ),
                                ),
                                channel="content",
                            )
                            await asyncio.sleep(min(1.5 * transient_retries, 5.0))
                            # An infra retry (transient upstream hiccup) is NOT a
                            # completed student round — mark it TRANSPARENT (None)
                            # so it neither advances NOR resets the no-result
                            # grind counter. P1b: re-send the SAME
                            # round — do NOT increment ``round_idx`` (an infra
                            # retry consumes no round budget; see the reactive
                            # max-tokens branch).
                            round_had_result = None
                            continue
                        # PRIMARY ENDPOINT DOWN → fail over to the LLM fallback
                        # model (``role == 'fallback'``) and re-send the SAME
                        # round against it. Once per turn (``llm_fallback_used``)
                        # so a dead primary can't ping-pong. Only on a genuine
                        # unreachable-endpoint error — never on 4xx / context /
                        # auth, which the fallback would fail identically.
                        fb_row = None
                        if _llm_fallback_allowed(
                            already_used=llm_fallback_used,
                            round_error=round_error,
                        ):
                            try:
                                fb_row = self.cfg.fallback_model()
                            except Exception:  # noqa: BLE001
                                fb_row = None
                        if fb_row and (fb_row.get("baseUrl") or fb_row.get("endpoint")):
                            llm_fallback_used = True
                            fb_ep = str(fb_row.get("baseUrl") or fb_row.get("endpoint") or "")
                            fb_model = str(fb_row.get("id") or fb_row.get("model") or "")
                            self._log_event(
                                "LLM_FALLBACK_FAILOVER",
                                session_id=getattr(session, "hash", ""),
                                from_endpoint=str(getattr(session, "endpoint", "")),
                                to_endpoint=fb_ep,
                                to_model=fb_model,
                                error=round_error[:200],
                            )
                            yield TextDelta(
                                text=_format_runtime_warning(
                                    "llm-fallback",
                                    reason=(
                                        f"primary LLM endpoint unreachable — "
                                        f"failing over to fallback {fb_model}@{fb_ep}: "
                                        f"{round_error[:120]}"
                                    ),
                                ),
                                channel="content",
                            )
                            # Route the REST of this turn through the
                            # fallback via per-turn request-level overrides
                            # (consumed at the _spawn_worker call site).
                            # Do NOT mutate ``session.endpoint`` /
                            # ``session.model``: every mid-turn
                            # ``session.save()`` (cost tracker, compaction,
                            # tool dispatch) persists the session header,
                            # so a mutated session silently re-stamps the
                            # operator's configured model as the fallback —
                            # the web UI then correctly shows the wrong
                            # model after a refresh.
                            llm_fallback_ep = fb_ep
                            if fb_model:
                                llm_fallback_model = fb_model
                            await asyncio.sleep(0.5)
                            round_had_result = None
                            # The failover re-sends the SAME
                            # round against the fallback — do NOT increment
                            # ``round_idx`` (an infra retry consumes no round
                            # budget; see the reactive max-tokens branch).
                            continue
                        yield TurnError(
                            message=round_error,
                            classification=round_error_class,
                        )
                        terminal_state = "error"
                        break
                    consecutive_overflow_retries = 0
                    # A "result" for the no-result counter is a DELIVERABLE:
                    # new visible content this round, or a successful tool call
                    # (recorded in the dispatch block below). Thinking is
                    # deliberately EXCLUDED — it is internal activity, not a
                    # result (matching the verdict-engine digest's definition),
                    # so a model that only ruminates round after round without
                    # producing content or successful tool output still trips
                    # ``no-result``. Reasoning models (qwen-style) emit content
                    # and/or tool calls ALONGSIDE their thinking in the same
                    # round, so they reset via that output and are not falsely
                    # flagged.
                    if "".join(rs.content_buf).strip():
                        round_had_result = True
                    if rs.pending_tool_calls and not rs.loop_detected:
                        # OpenAI-compatible chat APIs require an
                        # ``assistant`` message carrying THIS round's
                        # ``tool_calls`` immediately before the matching
                        # ``role=tool`` result messages — without it the
                        # next worker round 400s on the upstream. Append
                        # the assistant message first, normalised, then
                        # dispatch (which appends the tool results).
                        from sharedagents_gateway.acp_bridge import (
                            _build_assistant_round_msg,
                        )

                        norm_calls = [self._normalize_call(c) for c in rs.pending_tool_calls]
                        messages.append(
                            _build_assistant_round_msg(
                                content="".join(rs.content_buf),
                                thinking="".join(rs.thinking_buf),
                                tool_calls=norm_calls,
                                reasoning_replay=rs.reasoning_replay,
                            )
                        )
                        round_record: dict[str, Any] = {
                            "round_idx": round_idx,
                            "calls": [],
                        }
                        # Re-register the turn for the tool-dispatch
                        # window. The per-round worker ``finally`` popped
                        # it from ``_active_turns`` after streaming, but
                        # tool calls run AFTER that — and a runner can
                        # block for minutes (a long Bash / WebSearch /
                        # subagent). Without this, an ESC / ``/abort``
                        # during tool execution can't find the TurnState
                        # to SIGKILL + killpg the runner, so the tool runs
                        # to completion — the "hard kill does nothing
                        # while a tool runs" bug. The runner
                        # registered itself on ``turn_state`` via
                        # ``register_runner`` / ``turn.proc``.
                        self._active_turns[turn_id] = turn_state
                        # Mark where THIS round's tool results begin so the
                        # proactive teacher can extend the consecutive-error
                        # streak over just this round's results below.
                        _te_pre = len(engine_tool_results)
                        try:
                            async for ev in self._dispatch_tool_calls(
                                norm_calls,
                                session,
                                messages,
                                turn_state,
                                round_record,
                                promoted,
                                hook_engine=hook_engine,
                                hook_ctx=hook_ctx,
                            ):
                                if isinstance(ev, ToolCallStarted):
                                    if not ev.is_outer:
                                        engine_tool_calls.append(
                                            {
                                                "id": ev.call_id,
                                                "name": ev.name,
                                                "arguments": ev.arguments,
                                            }
                                        )
                                elif isinstance(ev, ToolCallResult):
                                    # Content-hash + background-poll flag feed
                                    # the sustained-zero-progress tracker: a
                                    # NEW result hash is progress, and a
                                    # still-running BashOutput poll
                                    # (``running: True`` in the body) is the
                                    # neutral "wait for the long job" carve-out.
                                    # Hash the content, never store it (the row
                                    # is turn-cumulative).
                                    _ev_c = ev.content or ""
                                    _matching_calls = [c for c in norm_calls if c.get("id") == ev.call_id]
                                    _executed = _matching_calls[0] if len(_matching_calls) == 1 else {}
                                    engine_tool_results.append(
                                        {
                                            "id": ev.call_id,
                                            "ok": ev.ok,
                                            "name": _executed.get("name"),
                                            "action": (_executed.get("arguments") or {}).get("action"),
                                            "content_size": len(_ev_c),
                                            "hash": hashlib.sha1(_ev_c.encode("utf-8", "replace")).hexdigest(),
                                            "bg_running": ("running: true" in _ev_c.lower()),
                                        }
                                    )
                                yield ev
                        finally:
                            self._active_turns.pop(turn_id, None)
                        # --- proactive teacher: consecutive tool-call errors -
                        # The Agent consumer flushes an in-progress snapshot on
                        # EVERY ToolCallResult (agents/agent.py), so by here all
                        # of this round's results (incl. ok:false + bodies) are
                        # already on the JSONL the teacher reads — the review
                        # sees the failures it is firing on.
                        #
                        # Extend the turn-wide consecutive-error streak over THIS
                        # round's results IN COMPLETION ORDER, resetting on any
                        # success. Fire the MOMENT the streak reaches the
                        # threshold — even if a later result in the same (e.g.
                        # parallel) batch succeeds — so a trailing success can
                        # never mask a run of N failures from the teacher. Fire AT
                        # MOST ONCE per round (``_te_fired`` guard) so one big
                        # failed batch can't burn several interventions before the
                        # student acts on the first continuation, but KEEP walking
                        # the rest of the batch so a trailing OK still resets the
                        # streak (a later lone error must not re-fire as if the
                        # run never broke). The "tool-errors" event forces a
                        # Tier-2 review (error bodies); its continuation is
                        # appended for the next round.
                        _round_any_ok = False
                        _te_fired = False
                        for _tr in engine_tool_results[_te_pre:]:
                            _ok = _tr.get("ok")
                            if _ok is True:
                                tool_error_streak = 0
                                _round_any_ok = True
                                continue
                            if _ok is not False:
                                # Defensive: count ONLY an explicit ``ok is False``
                                # as a failed tool call. Every engine_tool_results
                                # row carries a real bool today, but a statusless /
                                # non-result record must stay TRANSPARENT — never
                                # advance the streak (no false tool-errors fire).
                                continue
                            if _te_fired:
                                # Already fired this round — a trailing OK (above)
                                # still resets, but do NOT keep incrementing on
                                # failures the student committed BEFORE receiving
                                # the continuation (else a 10-error batch leaves
                                # the streak at threshold and the next round's
                                # first error instantly re-fires, robbing the
                                # student of a fresh budget to act on the advice).
                                continue
                            tool_error_streak += 1
                            if not (
                                self._teacher_enabled(sharedagents_sid)
                                and _teacher_tool_fail_threshold > 0
                                and tool_error_streak >= _teacher_tool_fail_threshold
                            ):
                                continue
                            _te_fired = True
                            _te_holder: list = []
                            async for _ev in self._teacher_proactive_delegate(
                                "tool-errors",
                                "tool_errors",
                                sharedagents_sid,
                                summary=f"{tool_error_streak} consecutive tool errors",
                                holder=_te_holder,
                            ):
                                yield _ev
                            _te_out = _te_holder[0] if _te_holder else None
                            if _te_out is not None and getattr(_te_out, "block_reason", ""):
                                messages.append(
                                    {
                                        "role": "user",
                                        "content": (
                                            "[teacher — your tool calls keep failing; "
                                            "follow this guidance] "
                                            + wrap_untrusted(
                                                _te_out.block_reason,
                                                source="teacher:tool-errors",
                                            )
                                        ),
                                    }
                                )
                            # COOLDOWN: reset the streak after EVERY fire (helped,
                            # vetoed, or fail-open) so the next fire needs another
                            # full ``_teacher_tool_fail_threshold`` consecutive
                            # errors — a fresh budget to act on the guidance. On a
                            # teacher OUTAGE the fire is still attempted + reported
                            # (ok=False), so the trigger is never swallowed; it
                            # just re-attempts after another N errors rather than
                            # on every erroring round.
                            tool_error_streak = 0
                            # Treat the failing round as handled so the no-result
                            # counter does not ALSO count it.
                            round_had_result = True
                        # A successful tool call is a result for the no-result
                        # counter (settled at the top of the next iteration).
                        if _round_any_ok:
                            round_had_result = True
                            # Real progress: a tool call SUCCEEDED. Reset the
                            # consecutive-idle streak -- a turn that is getting
                            # somewhere is not the loop the streak guards.
                            nudges_at_last_progress = len(nudge_log)
                        # Tool-using round → bump the budget the
                        # continuation-nudge policy keys on (see
                        # ``_should_nudge``).
                        tool_rounds_so_far += 1
                        for _rc in round_record.get("calls") or []:
                            _rn = _rc.get("name")
                            if _rn:
                                turn_tools_called.add(str(_rn))
                                nudge_tracker.on_tool(
                                    str(_rn),
                                    ok=bool(_rc.get("ok", True)),
                                    arguments=_rc.get("arguments") if isinstance(_rc.get("arguments"), dict) else {},
                                    result=str(_rc.get("result") or ""),
                                )

                        # Background Agent dispatch is the parent's last
                        # in-turn act when status text already streamed
                        # before the tool: children wake the parent via
                        # durable follow-up. Another model round here is
                        # a 100k–300k+ prefill that leaves WebUI
                        # streaming/Stop instead of waiting (observed in
                        # a jarvis session).
                        #
                        # Carve-out: ending with ZERO
                        # user-visible text skipped classify_turn_end
                        # entirely, so the silent empty-after-tools
                        # injection never ran (nudges=0) and the
                        # operator only saw ``no-output-end``. Inject
                        # one silent continuation so the model can
                        # write a status line; if that is suppressed
                        # or the budget is spent, fall through to the
                        # wait-end below.
                        if _round_is_background_delegate_wait(
                            norm_calls,
                            list(round_record.get("calls") or []) + list(engine_tool_results[_te_pre:]),
                        ):
                            if coordination is not None and not coordination.summary_requested:
                                # Always give Jarvis a post-dispatch reply round,
                                # even when it narrated before the tool. This is
                                # the wave's expected-output explanation, not a
                                # pending-todo nudge and not another dispatch.
                                from .coordination import WAVE_SUMMARY_PROMPT

                                coordination.summary_requested = True
                                messages.append({"role": "user", "content": WAVE_SUMMARY_PROMPT})
                                yield RoundComplete(
                                    round_idx=round_idx,
                                    metrics=self._round_complete_metrics(sharedagents_sid, rs.metrics),
                                )
                                _round_advanced()
                                round_idx += 1
                                continue
                            _has_visible = int(turn_visible_chars or 0) > 0 or bool("".join(content_buf).strip())
                            from sharedagents_agent.continuation import (
                                MAX_IDLE_EMPTY_CONTINUES,
                            )

                            _nudge_cap = int(
                                getattr(
                                    self.cfg,
                                    "max_nudges_per_turn",
                                    MAX_NUDGES_PER_TURN,
                                )
                                or MAX_NUDGES_PER_TURN
                            )
                            if (
                                not _has_visible
                                and not _prot_off
                                and not manual_budget_exhausted
                                and len(nudge_log) < _nudge_cap
                                # The background-delegate carve-out bypasses
                                # `classify_turn_end`, so it must apply the
                                # consecutive-idle bound itself or a statusless
                                # handle nudges past the loop guard.
                                and _idle_streak() < MAX_IDLE_EMPTY_CONTINUES
                                and not self._cancelled.get(sharedagents_sid, False)
                            ):
                                messages, fired = await self._continuation_nudge(
                                    messages,
                                    nudge_log,
                                    kind="silent",
                                    reason_code="empty_after_background_dispatch",
                                )
                                if fired:
                                    self._log_event(
                                        "BACKGROUND_DELEGATE_EMPTY_NUDGE",
                                        session_id=sharedagents_sid,
                                        round_idx=round_idx,
                                        calls=len(norm_calls),
                                        nudges=len(nudge_log),
                                    )
                                    _rm = self._round_complete_metrics(
                                        sharedagents_sid,
                                        rs.metrics,
                                    )
                                    yield RoundComplete(
                                        round_idx=round_idx,
                                        metrics=_rm,
                                    )
                                    _round_advanced()
                                    yield TextDelta(
                                        text=_format_runtime_warning(
                                            "nudge",
                                            kind="silent",
                                            reason="empty_after_background_dispatch",
                                        ),
                                        channel="content",
                                    )
                                    await self._notify(
                                        hook_engine,
                                        hook_ctx,
                                        "nudge",
                                        "continuation nudge (silent)",
                                        nudge_kind="silent",
                                        overridden=False,
                                    )
                                    round_idx += 1
                                    continue
                            self._log_event(
                                "BACKGROUND_DELEGATE_TURN_END",
                                session_id=sharedagents_sid,
                                round_idx=round_idx,
                                calls=len(norm_calls),
                            )
                            _round_advanced()
                            terminal_state = "ok"
                            break

                        # --- sustained-zero-progress recovery ------------
                        # Replaces the FP-heavy surface loop detectors. The
                        # separate read-only budget above bounds evidence sweeps
                        # without calling them loops. A round makes
                        # PROGRESS (resets the streak) on a NEW tool target, a
                        # mutation, a NEW tool-result content-hash, or new
                        # user-visible content. A still-running background-job
                        # poll is a neutral carve-out (never advances the
                        # streak) — the biggest strict-tool_call FP class. At
                        # ``nudge_streak`` we inject a plain re-grounding
                        # message; at ``hardstop_streak`` we inject "finalize
                        # now", allow ONE more round, then end the turn
                        # gracefully (terminal_state "ok", never an error).
                        _round_has_content = bool("".join(rs.content_buf).strip())
                        # Bound Explore-style research even when every new file /
                        # result is technically progress. The general
                        # zero-progress tracker intentionally permits new-target
                        # reads forever; this orthogonal budget nudges at the soft
                        # threshold and, at the hard threshold, grants exactly one
                        # synthesis round before a graceful stop. Any visible
                        # answer or mutating call resets the streak. Agent(list)
                        # / BashOutput polls are waits — they must not consume
                        # the research budget (native parents poll while children
                        # run).
                        _round_readonly_only = bool(norm_calls) and all(
                            self._effective_criticality_for(
                                str(call.get("name") or ""),
                                dict(call.get("arguments") or {}),
                            )
                            == "read_only"
                            for call in norm_calls
                        )
                        _readonly_kind = _readonly_budget_class(
                            norm_calls,
                            all_read_only=_round_readonly_only,
                        )
                        _readonly_verdict = ""
                        if not _prot_off:
                            _readonly_advanced = False
                            if _round_has_content:
                                readonly_round_streak = 0
                                readonly_finalize_fired = False
                            elif _readonly_kind == "research":
                                readonly_round_streak += 1
                                _readonly_advanced = True
                            elif _readonly_kind == "wait":
                                pass
                            else:
                                readonly_round_streak = 0
                                readonly_finalize_fired = False
                            if (
                                _readonly_advanced
                                and _MAX_READONLY_HARD_LIMIT > 0
                                and readonly_round_streak >= _MAX_READONLY_HARD_LIMIT
                            ):
                                if readonly_finalize_fired:
                                    _readonly_verdict = "finalize_stop"
                                else:
                                    readonly_finalize_fired = True
                                    _readonly_verdict = "finalize_inject"
                            elif (
                                _readonly_advanced
                                and _MAX_READONLY_SOFT_LIMIT > 0
                                and readonly_round_streak >= _MAX_READONLY_SOFT_LIMIT
                                and not _readonly_soft_fired
                            ):
                                _readonly_soft_fired = True
                                _readonly_verdict = "synthesize"
                        if _readonly_verdict:
                            self._log_event(
                                "READONLY_ROUND_BUDGET",
                                verdict=_readonly_verdict,
                                streak=readonly_round_streak,
                                round_idx=round_idx,
                                session_id=getattr(session, "hash", ""),
                            )
                        # Zero-output guarantee: a graceful finalize may only
                        # END the turn when the operator has actually seen
                        # something. A read-only grind's rounds are content-
                        # free by definition, so without this guard the break
                        # below ended the whole turn with NOTHING rendered.
                        # One tool-less answer round is granted (shared
                        # one-shot ``final_answer_round_fired``); a cancelled
                        # turn or a spent guard falls through to the break.
                        _readonly_skip_progress = False
                        if _readonly_verdict == "finalize_stop" and (
                            turn_visible_chars > 0
                            or final_answer_round_fired
                            or self._cancelled.get(sharedagents_sid, False)
                        ):
                            _round_advanced()
                            yield TextDelta(
                                text=_format_runtime_warning(
                                    "read-only-finalize",
                                    streak=readonly_round_streak,
                                ),
                                channel="content",
                            )
                            await self._notify(
                                hook_engine,
                                hook_ctx,
                                "read-only-finalize",
                                "read-only research budget exhausted — graceful finalize",
                                streak=readonly_round_streak,
                            )
                            terminal_state = "ok"
                            break
                        if _readonly_verdict == "finalize_stop":
                            final_answer_round_fired = True
                            # This round's interference is decided — skip the
                            # progress ladder below so a same-round progress
                            # finalize_stop cannot cancel the granted round.
                            _readonly_skip_progress = True
                            messages.append(
                                {
                                    "role": "user",
                                    "content": FINAL_ANSWER_NUDGE_BODY,
                                }
                            )
                            yield TextDelta(
                                text=_format_runtime_warning(
                                    "final-answer-inject",
                                    trigger="read-only-finalize",
                                    streak=readonly_round_streak,
                                ),
                                channel="content",
                            )
                            await self._notify(
                                hook_engine,
                                hook_ctx,
                                "final-answer-inject",
                                "zero visible output at read-only finalize — granting one final answer round",
                                streak=readonly_round_streak,
                            )
                            round_had_result = True
                        elif _readonly_verdict == "synthesize":
                            messages.append(
                                {
                                    "role": "user",
                                    "content": (
                                        "[engine] You have spent several consecutive "
                                        "rounds only gathering information. Synthesize "
                                        "what you already know, narrow the remaining "
                                        "search to the minimum, and prepare the answer."
                                    ),
                                }
                            )
                            yield TextDelta(
                                text=_format_runtime_warning(
                                    "read-only-synthesize",
                                    streak=readonly_round_streak,
                                ),
                                channel="content",
                            )
                            round_had_result = True
                        elif _readonly_verdict == "finalize_inject":
                            messages.append(
                                {
                                    "role": "user",
                                    "content": (
                                        "[engine] Read-only research limit reached. "
                                        "Finalize now from the evidence already gathered; "
                                        "do not call another tool."
                                    ),
                                }
                            )
                            yield TextDelta(
                                text=_format_runtime_warning(
                                    "read-only-finalize-inject",
                                    streak=readonly_round_streak,
                                ),
                                channel="content",
                            )
                            round_had_result = True
                        _prog_verdict = None
                        if not _prot_off and not _readonly_skip_progress:
                            _prog_verdict = progress_tracker.observe_round(
                                norm_calls,
                                engine_tool_results[_te_pre:],
                                _round_has_content,
                            )
                        if _prog_verdict:
                            self._log_event(
                                "PROGRESS_RECOVERY",
                                verdict=_prog_verdict,
                                streak=progress_tracker.zero_progress_streak,
                                nudge_count=progress_tracker.nudge_count,
                                round_idx=round_idx,
                                session_id=getattr(session, "hash", ""),
                            )
                        if _prog_verdict == "finalize_stop" and (
                            turn_visible_chars > 0
                            or final_answer_round_fired
                            or self._cancelled.get(sharedagents_sid, False)
                        ):
                            # The injected finalize round ALSO made zero
                            # progress → end the turn gracefully with whatever
                            # answer exists. GRACEFUL: terminal_state "ok", no
                            # distill, no mid-thought kill — the real-loop
                            # termination backstop the operator asked for.
                            # The finalize round still
                            # RAN — book it (content + rounds) before breaking
                            # so the terminal chokepoint's cap decision (when
                            # the max-steps handoff is active) reflects THIS
                            # round's content and round count, never a stale
                            # earlier round.
                            _round_advanced()
                            yield TextDelta(
                                text=_format_runtime_warning(
                                    "progress-finalize",
                                    streak=progress_tracker.zero_progress_streak,
                                ),
                                channel="content",
                            )
                            await self._notify(
                                hook_engine,
                                hook_ctx,
                                "progress-finalize",
                                "sustained zero progress — graceful finalize",
                                streak=progress_tracker.zero_progress_streak,
                            )
                            terminal_state = "ok"
                            break
                        if _prog_verdict == "finalize_stop":
                            # Zero-output guarantee: same one-shot answer
                            # round as the read-only guard — never end a
                            # turn the operator has seen nothing from.
                            final_answer_round_fired = True
                            messages.append(
                                {
                                    "role": "user",
                                    "content": FINAL_ANSWER_NUDGE_BODY,
                                }
                            )
                            yield TextDelta(
                                text=_format_runtime_warning(
                                    "final-answer-inject",
                                    trigger="progress-finalize",
                                    streak=progress_tracker.zero_progress_streak,
                                ),
                                channel="content",
                            )
                            await self._notify(
                                hook_engine,
                                hook_ctx,
                                "final-answer-inject",
                                "zero visible output at progress finalize — granting one final answer round",
                                streak=progress_tracker.zero_progress_streak,
                            )
                            round_had_result = True
                        elif _prog_verdict == "nudge":
                            messages.append(
                                {
                                    "role": "user",
                                    "content": _progress_nudge_body(progress_tracker.nudge_count),
                                }
                            )
                            yield TextDelta(
                                text=_format_runtime_warning(
                                    "progress-nudge",
                                    streak=progress_tracker.zero_progress_streak,
                                    attempt=progress_tracker.nudge_count,
                                ),
                                channel="content",
                            )
                            round_had_result = True
                        elif _prog_verdict == "finalize_inject":
                            messages.append(
                                {
                                    "role": "user",
                                    "content": _progress_finalize_body(),
                                }
                            )
                            yield TextDelta(
                                text=_format_runtime_warning(
                                    "progress-finalize-inject",
                                    streak=progress_tracker.zero_progress_streak,
                                ),
                                channel="content",
                            )
                            round_had_result = True
                        # Signal the round boundary so the Agent can
                        # checkpoint the in-progress turn to disk and
                        # any mid-turn user-message injects queued
                        # during this round get drained at the top of
                        # the next iteration. Carry THIS round's worker
                        # metrics (the same dict logged as TURN_METRICS)
                        # so the ACP translator can emit a per-round
                        # usage_update — that's what makes the frontend's
                        # `ctx X/Y` indicator climb after each tool round
                        # instead of only at TurnDone.
                        # Piggyback the live goal state onto the per-round
                        # metrics so the TUI footer stays in sync even when the
                        # teacher clears/overrides it (the metrics ride through to
                        # the frontend verbatim via _meta.sharedagents.metrics).
                        # ALWAYS stamp the goal tracker onto the per-round
                        # metrics — even when ``rs.metrics`` is empty/None this
                        # round. The goal widget's met/blocked badge rides on
                        # these fields, so dropping them when worker metrics are
                        # missing would strand a stale badge on the frontend
                        # until the next /status. Built by ``_round_complete_metrics``
                        # so the always-emit contract is unit-testable.
                        _rm = self._round_complete_metrics(sharedagents_sid, rs.metrics)
                        if not rs.loop_detected:
                            yield RoundComplete(
                                round_idx=round_idx,
                                metrics=_rm,
                            )
                            _round_advanced()
                            round_idx += 1
                            continue
                        # Hard read-only limit OR other post-dispatch loop
                        # detection fired — fall through to the loop recovery
                        # cascade below without yielding RoundComplete.

                    # --- loop detected → distill / force-terminate ------
                    if rs.loop_detected and manual_budget_exhausted:
                        # A capped manual prompt gets its one bounded reply;
                        # never spend another model call on loop recovery.
                        _round_advanced()
                        terminal_state = "ok" if content_buf else "error"
                        break
                    if rs.loop_detected:
                        loop_hit = (
                            rs.loop_tier or "loop",
                            rs.loop_evidence or "",
                        )
                        # LOOP_DETECTED already emitted to gateway.log
                        # above (before the metrics-merge) so the
                        # fire-moment line appears in tail order BEFORE
                        # the TURN_METRICS dump.
                        _ch = ""
                        if isinstance(rs.loop_event, dict):
                            _ch = str(rs.loop_event.get("channel") or "")
                        # ``ReasoningLoopDetected`` hook fires BEFORE
                        # the hardcoded distill→force-terminate cascade
                        # so an ECC plugin can:
                        #   * ``decision: block`` (exit 2) → treat the
                        #     detection as a false positive; suppress
                        #     the recovery and keep the round going.
                        #   * ``continue: false`` → end the turn now
                        #     (operator wants hard stop, not recovery).
                        #   * ``additionalContext`` → inject a custom
                        #     preamble into the message tail BEFORE the
                        #     normal cascade fires (e.g. "you were
                        #     looping; here are the constraints…").
                        # No-op / no hook → fall through to the canned
                        # distill→force-terminate flow exactly as before.
                        # NOTE: explicit /teacher on reviews the loop; otherwise
                        # the ReasoningLoopDetected hook decides.
                        if not manual_budget_exhausted and self._teacher_enabled(sharedagents_sid):
                            import uuid as _uuid

                            _lc_id = f"teacher-{_uuid.uuid4().hex[:8]}"
                            yield ToolCallStarted(
                                call_id=_lc_id,
                                name="delegate_teacher",
                                arguments={
                                    "model": self._teacher_model_label(sharedagents_sid),
                                    "trigger": "reasoning_loop",
                                },
                                is_outer=False,
                            )
                            loop_hook = await self._teacher_review_native("reasoning-loop", sharedagents_sid)
                            _lr_body = self._teacher_review_body(loop_hook)
                            if _lr_body:
                                yield TextDelta(
                                    text=_lr_body[:2000],
                                    channel="tool_delta",
                                    call_id=_lc_id,
                                )
                            _lr_failed = self._teacher_review_failed_open(loop_hook)
                            if loop_hook.blocked:
                                _lr_decision = "decision: continue — firing continuation back to break the loop"
                            elif _lr_failed:
                                _lr_decision = "decision: supervision unavailable"
                            else:
                                _lr_decision = "decision: not a loop / no action"
                            yield ToolCallResult(
                                call_id=_lc_id,
                                ok=not _lr_failed,
                                content=(f"{_lr_decision}\n\n{_lr_body[:2000]}" if _lr_body else _lr_decision),
                            )
                        else:
                            # Teacher OFF → fire the ReasoningLoopDetected hook
                            # for non-teacher (self-improver / operator) plugins.
                            # (The teacher path above never reaches a hook.)
                            loop_hook = await self._fire_hook(
                                hook_engine,
                                hook_ctx,
                                "ReasoningLoopDetected",
                                {
                                    "tier": loop_hit[0],
                                    "evidence": loop_hit[1],
                                    # channel: which detection source fired
                                    # (thinking / content / ratio / tool_call
                                    # / tool_args). Lets a plugin treat e.g.
                                    # ratio-channel fires as soft signals
                                    # and entropy-channel fires as hard.
                                    "channel": _ch,
                                    "round_idx": round_idx,
                                    "thinking_chars": sum(len(p) for p in rs.thinking_buf),
                                    "content_chars": sum(len(p) for p in rs.content_buf),
                                    "distill_recoveries": distill_recoveries,
                                },
                            )
                        for sm in loop_hook.system_messages:
                            yield TextDelta(
                                text=f"[hook] {sm}\n",
                                channel="content",
                            )
                        if not loop_hook.should_continue:
                            # Hook asked to stop the turn entirely.
                            # The looped round still RAN —
                            # book it (content + rounds) before breaking so the
                            # terminal chokepoint's cap decision (when the
                            # max-steps handoff is active) reflects THIS round's
                            # content, never a stale earlier round.
                            _round_advanced()
                            terminal_state = "ok" if content_buf else "error"
                            if loop_hook.stop_reason:
                                yield TextDelta(
                                    text=(f"\n[ReasoningLoopDetected hook] {loop_hook.stop_reason}\n"),
                                    channel="content",
                                )
                            break
                        if loop_hook.blocked:
                            # ``blocked`` means "keep going past this loop". For a
                            # shell hook that is a FALSE-POSITIVE override ("that
                            # wasn't really a loop"); for the TEACHER it is a
                            # REDIRECT ("you ARE stuck — change approach and
                            # continue"). codex #7: label them differently so the
                            # injected guidance steers the small model correctly
                            # instead of telling it to dismiss a real loop. Either
                            # way, clear the round's loop flags and resume.
                            _teacher_redirect = self._teacher_enabled(sharedagents_sid)
                            rs.loop_detected = False
                            rs.loop_tier = ""
                            rs.loop_evidence = ""
                            # Clear loop_event too — otherwise the
                            # next round's metrics-merge re-stamps the
                            # overridden event onto the new metrics.
                            rs.loop_event = None
                            if loop_hook.block_reason:
                                # Reason can echo external text → fence it as
                                # data; the framework label stays outside as the
                                # trusted signal.
                                _loop_label = (
                                    "[teacher — you are stuck in a reasoning loop; change your approach and continue]"
                                    if _teacher_redirect
                                    else "[ReasoningLoopDetected hook — false-positive override]"
                                )
                                messages.append(
                                    {
                                        "role": "user",
                                        "content": (
                                            _loop_label
                                            + " reason (untrusted data): "
                                            + wrap_untrusted(
                                                loop_hook.block_reason,
                                                source="hook:ReasoningLoopDetected",
                                            )
                                        ),
                                    }
                                )
                                yield TextDelta(
                                    text=_format_runtime_warning(
                                        "loop-suppressed",
                                        reason=loop_hook.block_reason,
                                    ),
                                    channel="content",
                                )
                                await self._notify(
                                    hook_engine,
                                    hook_ctx,
                                    "loop-suppressed",
                                    loop_hook.block_reason,
                                    tier=loop_hit[0],
                                )
                                # A reasoning-loop redirect is a fresh teacher
                                # intervention with new guidance — reset the
                                # proactive counters (like a Stop-hook
                                # continuation) so this just-redirected loop
                                # round doesn't ALSO immediately trip no-result /
                                # tool-errors back-to-back.
                                no_result_rounds = 0
                                tool_error_streak = 0
                                round_had_result = True
                                # This loop-redirect round is a completed
                                # model round — refresh the final-round
                                # bookkeeping before advancing.
                                _round_advanced()
                                round_idx += 1
                                continue
                            # No reason supplied — just resume the
                            # round (no extra user message); fall
                            # through past the cascade.
                        else:
                            if loop_hook.additional_context:
                                # Hook additionalContext is informational and
                                # can echo external text → fence as data.
                                messages.append(
                                    {
                                        "role": "user",
                                        "content": wrap_untrusted(
                                            loop_hook.additional_context,
                                            source="hook:ReasoningLoopDetected",
                                        ),
                                    }
                                )
                            # Surface the detection so the TUI /
                            # Telegram / one-shot frontends render an
                            # inline warning instead of looking like
                            # the turn died.
                            yield TextDelta(
                                text=_format_runtime_warning(
                                    "loop-detected",
                                    tier=loop_hit[0],
                                    evidence=loop_hit[1],
                                ),
                                channel="content",
                            )
                            await self._notify(
                                hook_engine,
                                hook_ctx,
                                "loop-detected",
                                f"reasoning loop ({loop_hit[0]})",
                                tier=loop_hit[0],
                                evidence=loop_hit[1],
                                round_idx=round_idx,
                            )
                            # Sustained-zero-progress recovery REPLACES the old
                            # distill→force-terminate cascade. The FP audit
                            # measured the surface loop triggers at ~0-15%
                            # precision — distilling / force-terminating them
                            # destroyed legitimate work. A kept detector fire
                            # (ratio / entropy / verbatim tool-call) is treated
                            # as one zero-progress round: observe it, then take
                            # the SAFE graduated action — a plain re-grounding
                            # NUDGE, escalating to a GRACEFUL finalize (write
                            # your best answer; terminal_state "ok", never an
                            # error). No distill, no mid-thought kill. A genuine
                            # loop still terminates: repeated zero-progress
                            # rounds drive the streak to the hardstop backstop.
                            # ``_prot_off`` models skip BOTH the nudge and the
                            # hard stop (per the disable_protections contract).
                            if not _prot_off:
                                _prog_verdict = progress_tracker.observe_round(
                                    [],
                                    [],
                                    False,
                                )
                                self._log_event(
                                    "PROGRESS_RECOVERY",
                                    verdict=str(_prog_verdict or "below-threshold"),
                                    streak=progress_tracker.zero_progress_streak,
                                    nudge_count=progress_tracker.nudge_count,
                                    tier=loop_hit[0],
                                    round_idx=round_idx,
                                    session_id=getattr(session, "hash", ""),
                                )
                                if _prog_verdict == "finalize_stop" and (
                                    turn_visible_chars > 0
                                    or final_answer_round_fired
                                    or self._cancelled.get(sharedagents_sid, False)
                                ):
                                    # Streak past the hardstop even after the
                                    # finalize round → end the turn gracefully
                                    # with whatever answer exists. This is the
                                    # real-loop termination backstop.
                                    # The finalize round
                                    # still RAN — book it (content + rounds)
                                    # before breaking so the terminal
                                    # chokepoint's cap decision (when the
                                    # max-steps handoff is active) reflects
                                    # THIS round, never a stale earlier round.
                                    _round_advanced()
                                    yield TextDelta(
                                        text=_format_runtime_warning(
                                            "progress-finalize",
                                            streak=progress_tracker.zero_progress_streak,
                                        ),
                                        channel="content",
                                    )
                                    await self._notify(
                                        hook_engine,
                                        hook_ctx,
                                        "progress-finalize",
                                        "sustained zero progress — graceful finalize",
                                        streak=progress_tracker.zero_progress_streak,
                                    )
                                    terminal_state = "ok"
                                    break
                                # A degenerate-token detector fire always gets a
                                # nudge (its turn-level action is the nudge, not
                                # a kill); at the hardstop-inject step swap in
                                # the finalize message and allow one more round.
                                # A zero-output finalize_stop swaps in the
                                # one-shot final-answer round instead.
                                if _prog_verdict == "finalize_stop":
                                    final_answer_round_fired = True
                                    _prog_body = FINAL_ANSWER_NUDGE_BODY
                                    _prog_kind = "final-answer-inject"
                                elif _prog_verdict == "finalize_inject":
                                    _prog_body = _progress_finalize_body()
                                    _prog_kind = "progress-finalize-inject"
                                else:
                                    _prog_body = _progress_nudge_body(max(1, progress_tracker.nudge_count))
                                    _prog_kind = "progress-nudge"
                                messages.append(
                                    {
                                        "role": "user",
                                        "content": _prog_body,
                                    }
                                )
                                yield TextDelta(
                                    text=_format_runtime_warning(
                                        _prog_kind,
                                        streak=progress_tracker.zero_progress_streak,
                                    ),
                                    channel="content",
                                )
                            # Clear the round's loop flags and resume — for both
                            # the nudged path and the protections-off path.
                            rs.loop_detected = False
                            rs.loop_tier = ""
                            rs.loop_evidence = ""
                            rs.loop_event = None
                            no_result_rounds = 0
                            tool_error_streak = 0
                            round_had_result = True
                            # The loop-recovery round is a completed
                            # model round — refresh the final-round
                            # bookkeeping before advancing.
                            _round_advanced()
                            round_idx += 1
                            continue
                        # Loop suppressed by a blocking hook with no
                        # reason supplied — fall through to the
                        # premature-end / natural-completion branches
                        # exactly as if no loop had fired.

                    # --- premature turn end → continuation nudge --------
                    # Three variants — silent EOS after a tool chain,
                    # mid-syntax cutoff (open backticks / fence), and
                    # colon cliffhanger (content ends on ``:`` with no
                    # tool calls). All gated by ``_should_nudge``: must
                    # have had at least one prior tool round, not be
                    # cancelled, and stay under ``MAX_NUDGES_PER_TURN``.
                    if had_done:
                        # Workflow recovery precedes the wire-only classifier
                        # and the zero-default Stop-hook continuation cap. A
                        # prose promise is not a delegation. Use the existing
                        # nudge and round budgets without resetting either.
                        if coordination is not None:
                            _pending = self._incomplete_todos(sharedagents_sid)
                            _disposition = workflow_disposition(
                                _pending,
                                "".join(rs.content_buf),
                            )
                            _coord_cap = getattr(self.cfg, "max_nudges_per_turn", MAX_NUDGES_PER_TURN)
                            _coord_cap = MAX_NUDGES_PER_TURN if _coord_cap is None else int(_coord_cap)
                            _coord_stopped = bool(
                                self._cancelled.get(sharedagents_sid)
                                or self._graceful_stop.get(sharedagents_sid)
                                or _GOAL_BLOCKED_RE.search("".join(rs.content_buf))
                            )
                            _coord_limited = bool(
                                _prot_off
                                or manual_budget_exhausted
                                or self._budget_handoff.get(sharedagents_sid)
                                or self._max_steps_handoff.get(sharedagents_sid)
                                or not self._auto_complete_continuation_enabled(sharedagents_sid)
                                or (max_rounds is not None and round_idx + 1 >= max_rounds)
                            )
                            _coord_open = bool(self._open_background_jobs_for(sharedagents_sid)) if _pending else False
                            # Nudge only an idle session. A queued event (a
                            # child's result, an owner message typed meanwhile)
                            # or a result still being delivered is processed
                            # as the next round instead; if nothing arrives
                            # within the short wait, the nudge below applies.
                            if (
                                _disposition == "working"
                                and _pending
                                and not coordination.delegated
                                and not coordination.owner_boundary
                                and not _coord_open
                                and not _coord_stopped
                                and not _coord_limited
                            ):
                                _arrived = await self._take_arriving_events(sharedagents_sid, request)
                                # Stop can arrive while the helper sleeps. Its
                                # empty return must not become a recovery nudge.
                                if self._cancelled.get(sharedagents_sid) or self._graceful_stop.get(sharedagents_sid):
                                    terminal_state = "aborted"
                                    _round_advanced()
                                    break
                                if _arrived:
                                    for _qitem in _arrived:
                                        _qid, _qtext = _qitem[0], _qitem[1]
                                        _qsrc = _qitem[2] if len(_qitem) >= 3 else ""
                                        if str(_qsrc) == "steer":
                                            messages.append(
                                                {
                                                    "role": "user",
                                                    "content": (
                                                        "⚠ MID-TURN USER STEER — the user has "
                                                        "interrupted with a new instruction. Stop "
                                                        "expanding on the prior plan. Treat this "
                                                        "as the operative request; any prior tool "
                                                        "work counts only if it directly serves "
                                                        "this new directive.\n\n" + str(_qtext)
                                                    ),
                                                }
                                            )
                                            yield UserMessageInjected(text=_qtext, round_idx=round_idx)
                                            continue
                                        messages.append({"role": "user", "content": _qtext})
                                        yield QueueMessageInjected(
                                            text=_qtext,
                                            queue_id=_qid,
                                            round_idx=round_idx,
                                        )
                                    yield RoundComplete(
                                        round_idx=round_idx,
                                        metrics=self._round_complete_metrics(sharedagents_sid, rs.metrics),
                                    )
                                    _round_advanced()
                                    round_idx += 1
                                    continue
                            if coordination.should_nudge(
                                pending=_pending,
                                open_jobs=_coord_open,
                                stopped=_coord_stopped,
                                limited=_coord_limited,
                                nudge_count=len(nudge_log),
                                max_nudges=_coord_cap,
                                disposition=_disposition,
                            ):
                                _todo_body = wrap_untrusted(
                                    "\n".join(_pending[:8]),
                                    source="pending-todos",
                                )
                                messages, fired = await self._continuation_nudge(
                                    messages,
                                    nudge_log,
                                    kind="silent",
                                    reason_code="pending_todos_without_delegation",
                                    body_override=(
                                        "There are still things to do: you stopped without "
                                        "dispatching any delegation this turn, and your todo "
                                        "list still has pending work. Continue the authorized "
                                        "work now. Use one AgentAsync(runs=[...]) wave for "
                                        "independent subagents, then explain the expected "
                                        "consolidated output AFTER the tool call. A promise "
                                        "to dispatch does not start work. If genuinely blocked "
                                        "on approval or missing information, explain 'GOAL "
                                        "BLOCKED: <reason>' instead. Pending items:\n" + _todo_body
                                    ),
                                )
                                if fired:
                                    yield RoundComplete(
                                        round_idx=round_idx,
                                        metrics=self._round_complete_metrics(sharedagents_sid, rs.metrics),
                                    )
                                    _round_advanced()
                                    yield TextDelta(
                                        text=_format_runtime_warning(
                                            "nudge",
                                            kind="silent",
                                            reason="pending_todos_without_delegation",
                                        ),
                                        channel="content",
                                    )
                                    round_idx += 1
                                    continue
                        _stream_gap = _detect_stream_truncation(
                            "".join(turn_emitted_content_parts),
                            "".join(content_buf),
                        )
                        if _stream_gap:
                            yield TextDelta(
                                text=_stream_gap,
                                channel="content",
                            )
                            turn_emitted_content_parts.append(_stream_gap)
                            turn_visible_chars += len(_stream_gap.strip())
                            self._log_event(
                                "STREAM_TAIL_EMITTED",
                                session_id=getattr(session, "hash", ""),
                                chars=len(_stream_gap),
                                round_idx=round_idx,
                            )
                        round_content = "".join(rs.content_buf)
                        final_text = round_content.strip()
                        # Signal-first continuation decision (replaces the old
                        # content-shape heuristics). Truncation is decided by
                        # ``finish_reason`` + stream-termination facts, never by
                        # punctuation/syntax shape. ``classify_turn_end`` is a pure
                        # function; the engine only gathers signals and maps the
                        # verdict to a nudge body. The PrematureTurnEndGate hook
                        # below can still override the verdict in BOTH directions
                        # (force-fire / suppress) — see spec section 4.5.
                        from sharedagents_agent.continuation import (
                            Action,
                            apply_nudge_hooks,
                        )

                        _m = rs.metrics or {}
                        _floor = int(
                            getattr(
                                self.cfg,
                                "nudge_min_text_chars",
                                NUDGE_MIN_TEXT_CHARS,
                            )
                            or NUDGE_MIN_TEXT_CHARS
                        )
                        _cap = int(
                            getattr(
                                self.cfg,
                                "max_nudges_per_turn",
                                MAX_NUDGES_PER_TURN,
                            )
                            or MAX_NUDGES_PER_TURN
                        )
                        _nudge_kwargs = {
                            "finish_reason": _m.get("finish_reason"),
                            "terminal_finish_reason_seen": bool(_m.get("terminal_finish_reason_seen", True)),
                            "cancelled": self._cancelled.get(sharedagents_sid, False),
                            "nudge_count": len(nudge_log),
                            "max_nudges": _cap,
                            "pending_tool_calls": bool(rs.pending_tool_calls),
                            "idle_streak": _idle_streak(),
                            "tool_call_truncated": bool(_m.get("truncated_tool_calls")),
                        }
                        decision = nudge_tracker.decide_stop(**_nudge_kwargs)
                        nudge_reason_code = decision.reason_code
                        nudge_kind = decision.nudge_kind
                        if nudge_reason_code == "tool_call_truncated":
                            # Name the calls that never ran and how far each
                            # got, so the model can size smaller ones.
                            nudge_tail = (
                                ", ".join(
                                    f"`{c.get('name') or 'tool'}` "
                                    f"({int(c.get('args_chars') or 0):,} "
                                    "characters of arguments)"
                                    for c in _m.get("truncated_tool_calls") or []
                                    if isinstance(c, dict)
                                )
                                or None
                            )
                        elif decision.action is Action.RESUME_TRUNCATED:
                            nudge_tail = round_content[-80:]
                        else:
                            nudge_tail = None

                        # ``PrematureTurnEndGate`` hook (sharedagents
                        # extension) — runs alongside the wire-signal
                        # continuation decision (``classify_turn_end``).
                        # The DECISION TO CONTINUE is a
                        # pure function of wire-level signals only; the hook may
                        # therefore reshape or SUPPRESS a nudge the wire
                        # decision already authorized, but it may NOT force a
                        # continuation on an ``Action.NONE`` verdict from the
                        # reply's content shape. Hook return semantics:
                        #   * ``hookSpecificOutput.kind`` /
                        #     ``hookSpecificOutput.tail`` → override
                        #     the variant + tail of an authorized nudge.
                        #   * ``continue: false`` → suppress nudge
                        #     entirely (end the turn).
                        #   * ``decision: block`` + ``reason: <body>``
                        #     → substitute the hook's body for an
                        #     already-authorized nudge.
                        gate_override_body: str | None = None
                        gate_force_suppress = False
                        _use_shared_nudge = not self._teacher_enabled(sharedagents_sid)
                        premature_hook = None
                        if _use_shared_nudge:
                            from sharedagents_agent.hooks import HookOutcome

                            plan = await apply_nudge_hooks(
                                nudge_tracker,
                                hook_engine=hook_engine,
                                hook_ctx=hook_ctx,
                                fire_hook=self._fire_hook,
                                **_nudge_kwargs,
                            )
                            nudge_tracker.commit_round()
                            for sm in plan.system_messages:
                                yield TextDelta(
                                    text=f"[hook] {sm}\n",
                                    channel="content",
                                )
                            if plan.kind:
                                nudge_kind = plan.kind
                            if plan.tail is not None:
                                nudge_tail = plan.tail
                            if plan.suppressed:
                                gate_force_suppress = True
                            elif plan.body:
                                gate_override_body = plan.body
                            premature_hook = HookOutcome(event="PrematureTurnEnd")
                            if plan.suppressed:
                                premature_hook.should_continue = False
                            elif plan.body and decision.action is not Action.NONE:
                                premature_hook.blocked = True
                                premature_hook.block_reason = plan.body
                        elif hook_engine is not None:
                            gate_hook = await self._fire_hook(
                                hook_engine,
                                hook_ctx,
                                "PrematureTurnEndGate",
                                {
                                    "kind": nudge_kind or "",
                                    "tail": nudge_tail or "",
                                    "final_text_chars": len(final_text),
                                    "tool_rounds_so_far": tool_rounds_so_far,
                                    "nudge_count": len(nudge_log),
                                    "round_idx": round_idx,
                                },
                            )
                            for sm in gate_hook.system_messages:
                                yield TextDelta(
                                    text=f"[hook] {sm}\n",
                                    channel="content",
                                )
                            # Honor hookSpecificOutput.kind / tail — but ONLY
                            # when the wire decision already authorized a
                            # resume. A Gate that invents a
                            # ``kind`` on an Action.NONE verdict must not open
                            # the nudge path; otherwise PrematureTurnEnd below
                            # can force-fire on a complete terse reply (e.g.
                            # "PONG") just because a hook rewrote the kind.
                            for r in reversed(gate_hook.results):
                                j = getattr(r, "json_output", None)
                                if not isinstance(j, dict):
                                    continue
                                hso = j.get("hookSpecificOutput")
                                if not isinstance(hso, dict):
                                    continue
                                if decision.action is not Action.NONE:
                                    k = hso.get("kind")
                                    if isinstance(k, str) and k.strip():
                                        nudge_kind = k.strip()
                                    t = hso.get("tail")
                                    if isinstance(t, str):
                                        nudge_tail = t
                                break
                            if not gate_hook.should_continue:
                                gate_force_suppress = True
                            elif gate_hook.blocked and gate_hook.block_reason and decision.action is not Action.NONE:
                                # Body substitution only — the wire decision,
                                # not the hook, governs whether a nudge fires
                                gate_override_body = gate_hook.block_reason

                        if (
                            nudge_kind is not None
                            and not gate_force_suppress
                            and not manual_budget_exhausted
                            and not self._budget_handoff.get(sharedagents_sid)
                            and not self._max_steps_handoff.get(sharedagents_sid)
                        ) and not _prot_off:
                            # ``classify_turn_end`` already applied the budget
                            # (nudge_count >= max_nudges -> NONE) and the
                            # cancelled short-circuit, so a non-NONE verdict means
                            # "fire by default". The continuation decision is
                            # wire-signal-only: the
                            # PrematureTurnEndGate hook can suppress
                            # (``gate_force_suppress``) but never force a nudge on
                            # a NONE verdict from content shape. In-turn
                            # mechanical retries are allowed by
                            # ``_continuation_nudge``; only a restored marker
                            # without a local audit is suppressed. The
                            # ``MAX_NUDGES_PER_TURN`` budget binds both paths.
                            should_fire = (
                                decision.action is not Action.NONE
                                and len(nudge_log) < _cap
                                and not self._cancelled.get(sharedagents_sid, False)
                                and not manual_budget_exhausted
                                and not self._budget_handoff.get(sharedagents_sid)
                                and not self._max_steps_handoff.get(sharedagents_sid)
                            )
                            # ``PrematureTurnEnd`` hook fires BEFORE
                            # the canned nudge body so an ECC plugin
                            # can:
                            #   * ``continue: false`` → skip nudging
                            #     and end the turn (operator considers
                            #     the round complete enough).
                            #   * ``decision: block`` + ``reason: …`` →
                            #     substitute the hook's body for the
                            #     canned ``NUDGE_PROMPT_BODY`` /
                            #     ``CUTOFF_NUDGE_PROMPT_BODY`` /
                            #     ``COLON_CLIFFHANGER_NUDGE_PROMPT_BODY``.
                            # No-op / no hook → default canned body.
                            if self._teacher_enabled(sharedagents_sid):
                                import uuid as _uuid

                                _pc_id = f"teacher-{_uuid.uuid4().hex[:8]}"
                                yield ToolCallStarted(
                                    call_id=_pc_id,
                                    name="delegate_teacher",
                                    arguments={
                                        "model": self._teacher_model_label(sharedagents_sid),
                                        "trigger": "premature_end",
                                    },
                                    is_outer=False,
                                )
                                premature_hook = await self._teacher_review_native("premature-end", sharedagents_sid)
                                _pe_body = self._teacher_review_body(premature_hook)
                                if _pe_body:
                                    yield TextDelta(
                                        text=_pe_body[:2000],
                                        channel="tool_delta",
                                        call_id=_pc_id,
                                    )
                                _pe_failed = self._teacher_review_failed_open(
                                    premature_hook,
                                )
                                if premature_hook.blocked:
                                    _pe_decision = "decision: continue — firing continuation back to the agent"
                                elif _pe_failed:
                                    _pe_decision = "decision: supervision unavailable"
                                else:
                                    _pe_decision = "decision: complete"
                                yield ToolCallResult(
                                    call_id=_pc_id,
                                    ok=not _pe_failed,
                                    content=(f"{_pe_decision}\n\n{_pe_body[:2000]}" if _pe_body else _pe_decision),
                                )
                            elif not _use_shared_nudge:
                                premature_hook = await self._fire_hook(
                                    hook_engine,
                                    hook_ctx,
                                    "PrematureTurnEnd",
                                    {
                                        "kind": nudge_kind,
                                        "tail": nudge_tail or "",
                                        "tool_rounds_so_far": tool_rounds_so_far,
                                        "nudge_count": len(nudge_log),
                                        "final_text_chars": len(final_text),
                                        "should_fire_default": should_fire,
                                        "round_idx": round_idx,
                                    },
                                )
                            if premature_hook is None:
                                from sharedagents_agent.hooks import HookOutcome

                                premature_hook = HookOutcome(
                                    event="PrematureTurnEnd",
                                )
                            if not _use_shared_nudge:
                                for sm in premature_hook.system_messages:
                                    yield TextDelta(
                                        text=f"[hook] {sm}\n",
                                        channel="content",
                                    )
                            # The Gate hook's override body (if any)
                            # is the default; PrematureTurnEnd may
                            # rewrite it again below.
                            override_body: str | None = gate_override_body
                            if not premature_hook.should_continue:
                                # Hook asked to stop. Treat as a clean
                                # turn end without nudging.
                                should_fire = False
                            elif (
                                premature_hook.blocked
                                and premature_hook.block_reason
                                and decision.action is not Action.NONE
                            ):
                                # Hook supplied a substitute nudge body —
                                # only when the wire classifier already
                                # authorized a resume.
                                # Never force-fire on a complete terse
                                # reply like "PONG".
                                override_body = premature_hook.block_reason
                                should_fire = True
                            if should_fire:
                                if rs.metrics and not rs.loop_detected:
                                    _rm_nudge = self._round_complete_metrics(
                                        sharedagents_sid,
                                        rs.metrics,
                                    )
                                    yield RoundComplete(
                                        round_idx=round_idx,
                                        metrics=_rm_nudge,
                                    )
                                    _round_advanced()
                                if (
                                    nudge_reason_code == "thinking_starved_length"
                                    and starved_compactions < 1
                                    and self._thinking_capped_by_context(session)
                                ):
                                    # The reply was cut while still reasoning and
                                    # the window is nearly full: nudging resumes
                                    # on the same prompt and starves again.
                                    # Compact (once per turn), then retry.
                                    starved_compactions += 1
                                    _freed = await self._compact_for_starved_thinking(
                                        session,
                                        messages,
                                    )
                                    if _freed is not None:
                                        messages = _freed
                                        yield TextDelta(
                                            text=_format_runtime_warning(
                                                "context-nearly-full",
                                                reason="the model used its whole thinking budget without answering",
                                                action="compacted the context and retrying",
                                            ),
                                            channel="content",
                                        )
                                        round_idx += 1
                                        continue
                                messages, fired = await self._continuation_nudge(
                                    messages,
                                    nudge_log,
                                    kind=nudge_kind,
                                    tail=nudge_tail,
                                    body_override=override_body,
                                    reason_code=nudge_reason_code,
                                )
                                if fired:
                                    yield TextDelta(
                                        text=_format_runtime_warning(
                                            "nudge",
                                            kind=nudge_kind,
                                            reason=nudge_reason_code,
                                        ),
                                        channel="content",
                                    )
                                    await self._notify(
                                        hook_engine,
                                        hook_ctx,
                                        "nudge",
                                        f"continuation nudge ({nudge_kind})",
                                        nudge_kind=nudge_kind,
                                        overridden=override_body is not None,
                                    )
                                    round_idx += 1
                                    continue
                        if self._pending_user_messages.get(sharedagents_sid):
                            # The nudge-path round completed (it ended
                            # with the model's own stop) — book the round
                            # before pivoting on the pending inject.
                            _round_advanced()
                            round_idx += 1
                            continue
                        # Fall through if neither premature-end variant
                        # nor inject pivot applied — handled below by
                        # the natural-completion path.

                    # --- mid-turn inject pivot --------------------------
                    # The model wants to terminate (no more tool calls)
                    # but a steering user prompt has arrived in the
                    # inject buffer since the round started. Spin up
                    # another round so the drain at the top of the
                    # next iteration picks it up and the assistant
                    # responds to the new instruction — instead of
                    # ending the turn and forcing the operator to
                    # start a fresh one. Without this guard a late
                    # inject (between the last tool result and
                    # finish_reason=stop) silently sits in the buffer
                    # until the next ``run_turn`` invocation.
                    if had_done and self._pending_user_messages.get(sharedagents_sid):
                        # The round the inject landed on completed with
                        # its own content — book it (wrap-up / rounds) before
                        # pivoting, so a later cap trusts the ACTUAL final
                        # round, not the stale pre-inject value.
                        _round_advanced()
                        round_idx += 1
                        continue

                    # --- natural completion -----------------------------
                    if had_done:
                        from sharedagents_research.native_tools import publication_required

                        _research_profile = (
                            getattr(self.cfg, "_agent_profile_name", "")
                            or getattr(self.cfg, "agent_profile", "")
                            or getattr(self.cfg, "name", "")
                        )
                        if publication_required(_research_profile, engine_tool_calls, engine_tool_results):
                            if research_publication_nudges < 2 and not manual_budget_exhausted:
                                research_publication_nudges += 1
                                messages.append(
                                    {
                                        "role": "user",
                                        "content": (
                                            "Research has no successful ontology publication for this turn. "
                                            "Use ontology_update to begin, extract each captured source, and finish "
                                            "with your cited answer and limitations. If storage is unavailable, "
                                            "report the exact blocker; do not claim the ontology was updated."
                                        ),
                                    }
                                )
                                _round_advanced()
                                round_idx += 1
                                continue
                            terminal_state = "error"
                            yield TurnError(
                                message="Research ontology publication incomplete; evidence was not reported as published.",
                                classification="ontology_publication",
                            )
                            _round_advanced()
                            break
                        # Deterministic delegation guard: the model ended the
                        # turn with a final answer. If this was a clear single-
                        # phase specialist task AND the model made NO delegation
                        # at all (pure DIRECT — it answered inline, but it has no
                        # Write tool so the artifact is lost), force the
                        # delegation now so the task reaches its specialist. Only
                        # fires on ZERO prior Agent calls this turn, so a model
                        # that already delegated (incl. a generic delegation the
                        # dispatcher reroute corrected) is never double-delegated.
                        _made_delegation = any(
                            c.get("name")
                            in (
                                "AgentSync",
                                "AgentAsync",
                            )
                            for c in engine_tool_calls
                        )
                        if (
                            not _made_delegation
                            and not manual_budget_exhausted
                            and not self._budget_handoff.get(sharedagents_sid)
                            and not self._max_steps_handoff.get(sharedagents_sid)
                        ):
                            _guard_target = await self._route_guard_target(user_text)
                            if _guard_target:
                                import json as _json

                                from sharedagents_gateway.acp_bridge import (
                                    _build_assistant_round_msg,
                                )

                                _syn = [
                                    self._normalize_call(
                                        {
                                            "id": f"routeguard-{turn_id}-{round_idx}",
                                            "function": {
                                                "name": "AgentAsync",
                                                "arguments": _json.dumps(
                                                    {
                                                        "subagent_type": _guard_target,
                                                        "description": (f"Deliver: {user_text[:80]}"),
                                                        "prompt": user_text,
                                                    }
                                                ),
                                            },
                                        }
                                    )
                                ]
                                messages.append(
                                    _build_assistant_round_msg(
                                        content="",
                                        thinking="",
                                        tool_calls=_syn,
                                    )
                                )
                                self._log_event(
                                    "ROUTE_GUARD_FORCED_DELEGATION",
                                    session_id=sharedagents_sid,
                                    target=_guard_target,
                                    round_idx=round_idx,
                                )
                                _guard_record: dict[str, Any] = {
                                    "round_idx": round_idx,
                                    "calls": [],
                                }
                                async for ev in self._dispatch_tool_calls(
                                    _syn,
                                    session,
                                    messages,
                                    turn_state,
                                    _guard_record,
                                    promoted,
                                    hook_engine=hook_engine,
                                    hook_ctx=hook_ctx,
                                ):
                                    if isinstance(ev, ToolCallStarted) and not getattr(ev, "is_outer", False):
                                        engine_tool_calls.append(
                                            {
                                                "id": ev.call_id,
                                                "name": ev.name,
                                                "arguments": ev.arguments,
                                            }
                                        )
                                    yield ev
                        # F1 (plan v2): a "natural" stop on the FINAL round is
                        # still a cap event — the max-steps / cost-budget
                        # handoff was injected ("tools disabled; summarize
                        # your progress") and the model's stop is its wrap-up
                        # answer to that prompt, not a free completion.
                        # Report round_cap / budget so callers resume with a
                        # compact brief instead of believing the task
                        # completed (the incident). Book the round here:
                        # without ``_round_advanced``, the later wrap-up gate
                        # sees ``final_round_booked=False`` and reports
                        # error even when this round streamed wrap-up text.
                        if self._max_steps_handoff.get(sharedagents_sid):
                            completion_reason = "round_cap"
                            # A capped turn is only
                            # ``ok`` when the FINAL round actually
                            # delivered its wrap-up — the content it
                            # produced AFTER the handoff injection.
                            # Earlier-round content does not qualify:
                            # the wrap-up is the model's answer to the
                            # handoff prompt, and without it there is
                            # no clean handoff to resume from.
                            _round_advanced()
                            terminal_state = "ok" if final_round_text else "error"
                        elif self._budget_handoff.get(sharedagents_sid):
                            completion_reason = "budget"
                            _round_advanced()
                            terminal_state = "ok" if final_round_text else "error"
                        else:
                            terminal_state = "ok"
                        break

                    # Soft-aborted by ``inject_user_message`` — the
                    # worker was SIGTERM'd mid-stream so the model
                    # would stop wasting compute on the now-stale
                    # plan. ``had_done`` is False because no proper
                    # ``done`` event reached us before EOF. Treat as
                    # "round ended, drain inject, pivot" rather than
                    # surfacing a worker_eof error.
                    if self._pending_user_messages.get(sharedagents_sid):
                        # The soft-aborted round still RAN (partial
                        # content is the only final-round text we have) —
                        # book it before the inject pivot so the cap
                        # decision reflects this round, not a stale one.
                        _round_advanced()
                        round_idx += 1
                        continue

                    yield TurnError(
                        message="worker exited without a `done` event",
                        classification="worker_eof",
                    )
                    terminal_state = "error"
                    break
                else:
                    # Round cap hit — treat as a (degraded) completion.
                    # Don't pivot on round-cap: the cap exists to bound
                    # runaway loops; honouring an inject here would
                    # let a chained inject keep extending the turn
                    # past its safety budget. Operator must start a
                    # fresh turn instead.
                    #
                    # F1 (plan v2): the cap is TERMINAL and carries a
                    # structured finish reason. ``state`` keeps the engine
                    # contract (``ok`` when the wrap-up emitted content,
                    # ``error`` when it produced nothing), but
                    # ``completion_reason="round_cap"`` rides the TurnDone
                    # event + engine audit so callers can distinguish a
                    # CAPPED turn from a COMPLETED one (the incident: the
                    # cap reported ``ok`` and every caller — TUI, delegate
                    # driver, goal supervisor — believed the task was
                    # done). The Stop-hook boundary below refuses any
                    # continuation once ``_max_steps_handoff`` is set, so
                    # no re-entry can reset ``round_idx`` / clear the
                    # handoff and re-enable tools (the 4x ceiling bug).
                    # The cap only counts as an ``ok``
                    # finish when the FINAL round itself produced a real
                    # wrap-up (its own content after the handoff
                    # injection) — cumulative earlier-round content must
                    # NOT qualify (it is progress, not a handoff).
                    terminal_state = "ok" if final_round_text else "error"
                    completion_reason = "round_cap"
                    # NOTE: the no-result trigger deliberately does NOT fire here
                    # at budget exhaustion. It only fires from the top-of-loop
                    # chokepoint DURING an ongoing grind, where rounds remain for
                    # the student to act on the continuation — firing at the cap
                    # would append guidance the terminating turn can never use.
                    # The turn-end review below (pre_idle on a clean/degraded
                    # ``ok`` terminal, ``_teacher_terminal_review`` on an error
                    # terminal) supervises the boundary instead. In AUTO mode
                    # (max_rounds == MAX_ROUNDS_AUTO == 35) the limit (default
                    # 20) is reached and fired with rounds to spare; only a
                    # bounded mode whose budget is < the limit never fires
                    # mid-grind, and there a grind is a failing-tool grind that
                    # the tool-errors trigger has already caught repeatedly.

                # --- Stop hooks (Claude Code) — with continuation -------
                if terminal_state != "ok" or hook_engine is None or self._cancelled.get(sharedagents_sid):
                    # A non-`ok` ERROR terminal (worker_eof / "worker exited
                    # without done" / round-cap with no content) fires no
                    # `Stop` hook, so an observer like the teacher-model
                    # supervisor would never see it and the session could
                    # end unsupervised. Fire `StopFailure` here so the
                    # teacher (and any ECC plugin) can dispatch a recovery.
                    # A user-cancelled turn (``self._cancelled``) is an
                    # INTENTIONAL stop — never fire on it (respect the
                    # operator; the teacher must not fight a `/stop`).
                    if terminal_state != "ok" and not self._cancelled.get(sharedagents_sid):
                        if not manual_budget_exhausted and self._teacher_enabled(sharedagents_sid):
                            # The turn died on a failure terminal (worker EOF /
                            # empty round-cap). Let the teacher SEE it as a
                            # delegate node + log a verdict — NO inline re-drive
                            # (codex: re-driving a broken runtime risks an
                            # infinite loop). Autonomous recovery, if any, is the
                            # bounded after-idle path's job.
                            async for _ev in self._teacher_terminal_review(
                                "error-terminal",
                                sharedagents_sid,
                            ):
                                yield _ev
                        elif hook_engine is not None:
                            await self._fire_hook(
                                hook_engine,
                                hook_ctx,
                                "StopFailure",
                                {
                                    "reason": "non_ok_terminal",
                                    "terminal_state": terminal_state,
                                },
                            )
                    break
                # COST is the only hard limit. If THIS turn ended on the cost
                # budget (the per-round gate set ``_budget_handoff``), NEVER let a
                # Stop hook force a continuation — the agent literally cannot
                # afford more work, and re-driving it tool-less is the deadlock
                # that traps the agent into "I'm blocked, can't call tools". End
                # the turn cleanly with goal_terminal UNSET so the turn finalizer
                # pauses the goal with reason ``cost_budget_exhausted`` (resumable
                # via the ``goal_budget`` tool / ``/goal topup``).
                if self._budget_handoff.get(sharedagents_sid):
                    self._log_event(
                        "BUDGET_STOP_RELEASE",
                        session_id=sharedagents_sid,
                        reason="cost_budget_exhausted",
                    )
                    break
                # COST is only one hard limit. The STEP budget is the other
                # (F1, plan v2): if THIS turn ended on the round cap (the
                # max-steps injection set ``_max_steps_handoff``), NEVER let
                # a Stop hook — nor the AUTO_COMPLETE re-drive below — force
                # a continuation. This is the exact cap-defeating path from
                # the incident: a blocking Stop hook / AUTO_COMPLETE used to
                # reset ``round_idx = 0`` and pop ``_max_steps_handoff``,
                # re-driving a tool-less final round up to 3 more times
                # (4x ceiling, 400 rounds) and reporting ``ok``. The cap is
                # TERMINAL: end the turn here; any further work happens in a
                # FRESH turn with a compact resume brief — never by a
                # within-turn continuation. The engine never auto-resumes.
                if self._max_steps_handoff.get(sharedagents_sid):
                    self._log_event(
                        "ROUND_CAP_STOP_RELEASE",
                        session_id=sharedagents_sid,
                        reason="round_cap",
                        terminal_state=terminal_state,
                    )
                    break
                # Cap is overridable per-session — ``cfg.max_stop_continuations``
                # lets the operator extend or tighten the loop without
                # editing the module-level constant. F4 (plan v2): the old
                # ``value or _MAX_STOP_CONTINUATIONS`` made 0 impossible
                # (``0 or 3`` → 3), so ``max_stop_continuations=0`` could
                # never disable the continuation loop. Explicit None
                # resolution: None → module default; 0 → ZERO continuations.
                _stop_cfg = getattr(
                    self.cfg,
                    "max_stop_continuations",
                    _MAX_STOP_CONTINUATIONS,
                )
                _stop_cap = _MAX_STOP_CONTINUATIONS if _stop_cfg is None else int(_stop_cfg)
                _cap_spent = stop_continuations >= _stop_cap
                # Who owns turn-end for this session? A reviewer's verdict is
                # the product, so the cap may never suppress it — only the
                # re-drive (guarded at the continuation point below).
                _reviewer_enabled = not manual_budget_exhausted and self._teacher_enabled(sharedagents_sid)
                if _stop_cap_short_circuits(
                    cap_spent=_cap_spent,
                    reviewer_enabled=_reviewer_enabled,
                ):
                    # Continuation budget exhausted AND no reviewer to render:
                    # the teacher already pushed ``max_stop_continuations``
                    # times and the goal is still unmet. Review-and-stop ONLY —
                    # no further fire-back, or the cap would be meaningless
                    # (codex). The teacher still SEES the cap-exhaustion as a
                    # delegate node + logs a verdict.
                    if not manual_budget_exhausted and self._teacher_enabled(sharedagents_sid):
                        async for _ev in self._teacher_terminal_review(
                            "stop-cap",
                            sharedagents_sid,
                        ):
                            yield _ev
                    elif hook_engine is not None:
                        await self._fire_hook(
                            hook_engine,
                            hook_ctx,
                            "StopFailure",
                            {
                                "reason": "stop_continuation_cap",
                                "continuations": stop_continuations,
                                "max_continuations": _stop_cap,
                            },
                        )
                    break
                if self._goal_active(sharedagents_sid):
                    # ``GOAL BLOCKED:`` ends the turn; the goal evaluator
                    # judges it (an assistant claim is evidence, not proof).
                    if _GOAL_BLOCKED_RE.search("".join(rs.content_buf)):
                        break
                if not manual_budget_exhausted and self._teacher_enabled(sharedagents_sid):
                    # REVIEW-BEFORE-IDLE — explicit /teacher on (optionally
                    # before /goal) supervises toward the active condition.
                    # before the agent settles into the idle state, it passes
                    # through a visible teacher REVIEW STATE — immediately, no
                    # delay. The review runs IN-TURN as a ``delegate_teacher``
                    # node (rendered in the TUI; the footer shows "reviewing…"),
                    # so both the review AND any re-drive are visible (this is
                    # what replaced the old headless 30s after-idle watch). If the
                    # teacher says continue, its continuation fires BACK into THIS
                    # turn via the block→inject path below (re-drive, in-turn,
                    # bounded by max_stop_continuations); if complete, the turn
                    # ends and the session goes idle.
                    import uuid as _uuid

                    _tc_id = f"teacher-{_uuid.uuid4().hex[:8]}"
                    yield ToolCallStarted(
                        call_id=_tc_id,
                        name="delegate_teacher",
                        arguments={
                            "model": self._teacher_model_label(sharedagents_sid),
                            "trigger": "pre_idle",
                            "summary": "review before idle",
                        },
                        is_outer=False,
                    )
                    stop = await self._teacher_review_native("pre_idle", sharedagents_sid)
                    # Surface the teacher's actual review (verdict/progress/
                    # gaps/rationale) into the session under the node — on
                    # EVERY decision, not just continue. The continuation
                    # instruction (block_reason) is appended when blocking.
                    _review_body = self._teacher_review_body(stop)
                    if _review_body:
                        yield TextDelta(
                            text=_review_body[:2000],
                            channel="tool_delta",
                            call_id=_tc_id,
                        )
                    _review_failed = self._teacher_review_failed_open(stop)
                    if stop.blocked and _cap_spent:
                        _decision = (
                            "decision: continue — continuation budget exhausted; "
                            "ending the turn (reviewed, no re-drive)"
                        )
                    elif stop.blocked:
                        _decision = "decision: continue — re-driving the agent in-turn"
                    elif _review_failed:
                        _decision = "decision: supervision unavailable — ending the turn without teacher approval"
                    else:
                        _decision = "decision: complete — approving turn end (going idle)"
                    # Put the review in the RESULT content (not only the
                    # tool_delta) so it PERSISTS to the JSONL and replicates
                    # to other frontends — tool_delta is live-only.
                    yield ToolCallResult(
                        call_id=_tc_id,
                        ok=not _review_failed,
                        content=(f"{_decision}\n\n{_review_body[:2000]}" if _review_body else _decision),
                    )
                else:
                    stop = await self._fire_hook(
                        hook_engine,
                        hook_ctx,
                        "Stop",
                        {"stop_hook_active": stop_continuations > 0},
                    )
                    # Teacher-INDEPENDENT autonomous completion gate: in AUTO
                    # mode, if the model is stopping cleanly but its OWN
                    # TodoWrite plan still has incomplete items, it almost
                    # certainly stopped short of the task (the dominant
                    # small-model failure on qwen3.6 — declaring done with work
                    # pending; observed live). Re-drive with a
                    # reminder instead of ending the turn. Bounded by the
                    # stop-continuation cap checked above; a GOAL BLOCKED
                    # sentinel (checked below) still wins.
                    #
                    # Carve-out: OPEN background jobs mean the parent is
                    # correctly waiting for durable follow-up (jarvis
                    # "report status, don't stall"). Forcing AUTO_COMPLETE
                    # re-drive then fights that wait — it burns a Stop
                    # continuation and pushes the model to keep working
                    # without child results. Result delivery is already
                    # covered by mid-turn queue drain / turn-end
                    # auto-drain / idle resume; todo pressure returns
                    # once open_jobs is empty (live 2026-07-31 jarvis
                    # worker restart).
                    if not stop.blocked and self._auto_complete_continuation_enabled(sharedagents_sid):
                        _pending = self._incomplete_todos(sharedagents_sid)
                        _open_jobs = self._open_background_jobs_for(sharedagents_sid) if _pending else []
                        if _pending and _open_jobs:
                            self._log_event(
                                "AUTO_COMPLETE_SKIP_WAITING_JOBS",
                                session_id=sharedagents_sid,
                                incomplete=len(_pending),
                                open_jobs=len(_open_jobs),
                                job_ids=[str(j.get("job_id") or "") for j in _open_jobs[:8]],
                            )
                        elif _pending:
                            from sharedagents_agent.hooks import HookOutcome

                            _shown = "; ".join(_pending[:6])
                            _more = f" (+{len(_pending) - 6} more)" if len(_pending) > 6 else ""
                            stop = HookOutcome(
                                event="Stop",
                                blocked=True,
                                block_reason=(
                                    f"You stopped, but {len(_pending)} to-do "
                                    f"item(s) are still incomplete: {_shown}{_more}. "
                                    "Keep working until every item is marked done "
                                    "(update them with TodoWrite as you finish). If "
                                    "you are GENUINELY blocked on one, reply 'GOAL "
                                    "BLOCKED: <why>' and stop; otherwise continue now."
                                ),
                                # Preserve any non-blocking messages the real
                                # Stop hook emitted — replacing
                                # ``stop`` wholesale would otherwise drop them.
                                system_messages=list(stop.system_messages),
                                additional_context=stop.additional_context,
                            )
                            self._log_event(
                                "AUTO_COMPLETE_CONTINUE",
                                session_id=sharedagents_sid,
                                incomplete=len(_pending),
                                continuation=stop_continuations + 1,
                            )
                for sm in stop.system_messages:
                    yield TextDelta(
                        text=f"[hook] {sm}\n",
                        channel="content",
                    )
                if stop.additional_context:
                    yield TextDelta(
                        text=stop.additional_context + "\n",
                        channel="content",
                    )
                if not stop.blocked:
                    break
                # ``GOAL BLOCKED:`` on the last assistant message ends the turn
                # instead of being re-driven by a Stop hook (no goal state
                # change — the goal evaluator judges it after TurnDone).
                if _GOAL_BLOCKED_RE.search("".join(rs.content_buf)):
                    break
                if _cap_spent:
                    # A reviewer approved more work but the continuation budget
                    # is spent: its verdict has rendered (and any terminal goal
                    # state is stamped above), so end the turn WITHOUT
                    # re-driving. Review-and-stop ONLY — the cap must stay
                    # meaningful (codex).
                    break
                # A Stop hook asked the agent to keep going — append a fresh
                # user turn and re-enter the round loop. The continuation
                # INSTRUCTION ("you are not done; continue working") is
                # framework-authored and stays UNFENCED so the model acts on
                # it; the hook-supplied reason can echo external text, so it
                # is fenced as untrusted data.
                stop_continuations += 1
                # Give the teacher's redirect a FRESH round budget. A blocking
                # Stop means the teacher told the model to keep going (e.g.
                # "stop documenting — write server.py now"), so it needs rounds
                # to act on that. Resetting per continuation (vs. sharing the
                # budget) is bounded by ``max_stop_continuations``, so the
                # turn-wide ceiling is max_rounds × max_stop_continuations —
                # the teacher reviews every ``max_rounds`` rounds and the run
                # can't exceed that product. This is what lets a teacher-
                # supervised auto turn make sustained progress under periodic
                # review instead of dying at the first cap.
                round_idx = 0
                # CRITICAL: a fresh round budget is useless without tools. The
                # max-steps handoff emptied the tool surface for the wrap-up
                # round; a Stop hook then said "keep going" and gave fresh
                # rounds — so we MUST re-enable tools too, or the agent is driven
                # tool-less and can only emit "I'm blocked, can't call tools"
                # while the hook keeps re-firing (the reported deadlock). Clear
                # both handoff flags so the continuation can actually act. (The
                # cost-budget case already broke out above — cost is the hard
                # limit and is never continued past.)
                self._max_steps_handoff.pop(sharedagents_sid, None)
                self._budget_handoff.pop(sharedagents_sid, None)
                # A Stop-hook continuation restarts the turn-segment with fresh
                # guidance, so the proactive-firing streaks restart too.
                no_result_rounds = 0
                tool_error_streak = 0
                round_had_result = True
                _stop_reason = (stop.block_reason or "").strip()
                _stop_content = "[Stop hook] You are not done; continue working."
                if _stop_reason:
                    _stop_content += " Reason (untrusted data): " + wrap_untrusted(_stop_reason, source="hook:Stop")
                messages.append(
                    {
                        "role": "user",
                        "content": _stop_content,
                    }
                )
                yield TextDelta(
                    text=(f"\n[Stop hook requested continuation] {stop.block_reason}\n"),
                    channel="content",
                )
        finally:
            self._coordination_turns.pop(sharedagents_sid, None)
            self._active_turns.pop(turn_id, None)
            self._session_to_turn.pop(sharedagents_sid, None)

        ended_at_iso = _now_iso()

        # Update the engine-audit dict that was stamped at
        # run_turn entry. Using ``update`` (NOT replace) preserves
        # the hook_outcomes_log reference shared with
        # ``_HOOK_OUTCOMES_LOG`` so every hook that fired during
        # the turn lands in the saved record — including ones
        # that fired AFTER an early non-terminal yield. Tool
        # result content gets stripped on the way to disk by
        # session_log; the engine still uses the full messages
        # list in-memory.
        last_metrics_dict = last_metrics if isinstance(last_metrics, dict) else {}
        request.extra["_engine_audit"].update(
            {
                "thinking": "".join(thinking_buf),
                "content": "".join(content_buf),
                "tool_calls": engine_tool_calls,
                "tool_results": engine_tool_results,
                "metrics": last_metrics,
                "ended_at": ended_at_iso,
                "state": terminal_state,
                # F1 (plan v2): structured finish reason — "round_cap" when the
                # per-turn round budget was exhausted (sticky across the turn
                # record; callers must NOT treat a capped turn as completed).
                "completion_reason": completion_reason,
                # Replay-fidelity (fix_20260519 round-4):
                "model_input_messages": list(messages),
                # Per-turn slice, NOT the full session-level append-only
                # list — without the slice, every turn's persisted record
                # would carry all PRIOR turns' rounds (gemini round-4
                # critical finding).
                "round_metrics": list(
                    (getattr(session, "round_metrics", None) or [])[_round_metrics_start_idx:],
                ),
                "upstream_endpoint": str(
                    last_metrics_dict.get("endpoint") or request.extra["_engine_audit"].get("upstream_endpoint") or "",
                ),
            }
        )

        # Reset the hook-outcomes contextvar so a same-task delegate
        # parent that called us during its own tool dispatch
        # doesn't keep writing PostToolUse / Stop hook outcomes into
        # OUR list after we return. Codex round-5 minor #1.
        _HOOK_OUTCOMES_LOG.reset(_hook_log_token)

        # A turn that ends while the max-steps handoff is
        # active is a CAPPED turn no matter which break exited it — loop-
        # detected ``continue:false``, the zero-progress finalize_stop, a
        # worker failure / retry-exhaustion on the final round, or the
        # ROUND_CAP_STOP_RELEASE guard — not just the two normal cap exits
        # (natural-stop-capped / cap-hit, which already set the reason). This
        # single chokepoint normalizes the reason so callers never see capped
        # incomplete work as completed (was: TurnDone(state='ok',
        # completion_reason='stop') with no TurnError — no auto-resume, the
        # incident class). The F4 wrap-up gate applies here too: ``ok`` only
        # when the FINAL round actually delivered its own wrap-up text
        # (``final_round_booked`` — a worker failure on the final round must
        # never fall back to a PREVIOUS round's text as the handoff); an
        # operator abort is NEVER a cap. Must run BEFORE the metrics
        # stamping below (rounds / wrap_up_text key on completion_reason).
        if (
            self._max_steps_handoff.get(sharedagents_sid)
            and completion_reason != "round_cap"
            and terminal_state != "aborted"
        ):
            completion_reason = "round_cap"
            if not final_round_booked:
                # The final round never completed (worker failure / retry
                # exhaustion / spawn failure) — there is no wrap-up to
                # qualify the cap as ``ok``; clear any stale earlier-round
                # text so the gate below reports error + TurnError.
                final_round_text = ""
            terminal_state = "ok" if final_round_text else "error"
            # Keep the engine audit (written above, before this chokepoint)
            # consistent with the normalized terminal — callers audit the
            # reason/state from the persisted record, not only the event.
            request.extra["_engine_audit"].update(
                {
                    "completion_reason": completion_reason,
                    "state": terminal_state,
                }
            )

        # Cost-budget wrap-up is the same incomplete class as round-cap:
        # tools were disabled for a final summary, so a natural ``stop``
        # is the model's answer to the handoff, not a completed task.
        # Without this stamp, delegate callers reported ``ok`` /
        # ``state=completed`` (incident: plan $25 wrap-up looked done).
        # Budget wins when both flags are set (manual-over-budget reuses
        # the max-steps empty tool surface AND the budget flag).
        if self._budget_handoff.get(sharedagents_sid) and terminal_state != "aborted":
            completion_reason = "budget"
            if not final_round_booked:
                final_round_text = ""
            terminal_state = "ok" if final_round_text else "error"
            request.extra["_engine_audit"].update(
                {
                    "completion_reason": completion_reason,
                    "state": terminal_state,
                }
            )

        # Every successful exit, including capped/budget wrap-ups, must satisfy
        # the same research publication contract. Cancellation stays unchanged.
        from sharedagents_research.native_tools import publication_required

        _research_profile = (
            getattr(self.cfg, "_agent_profile_name", "")
            or getattr(self.cfg, "agent_profile", "")
            or getattr(self.cfg, "name", "")
        )
        if terminal_state == "ok" and publication_required(_research_profile, engine_tool_calls, engine_tool_results):
            terminal_state = "error"
            request.extra["_engine_audit"]["state"] = "error"
            yield TurnError(
                message="Research ended without a successful ontology publication.",
                classification="ontology_publication",
            )

        # Compute cumulative turn-level metrics from all rounds — the
        # ``last_metrics`` dict only carries the LAST round's numbers,
        # but multi-round turns need aggregated totals (completion,
        # reasoning, content tokens summed across all rounds) so the
        # TUI stats line shows the full turn, not just the last model
        # call. ``prompt_tokens`` from the last round is already
        # cumulative (the context grows each round), and per-call
        # timing (ttft / prefill / tps) stays round-local since callers
        # want the final round's generation speed.
        this_turn_rounds = list(
            (getattr(session, "round_metrics", None) or [])[_round_metrics_start_idx:],
        )
        if this_turn_rounds and terminal_state == "ok":
            _turn_metrics = dict(last_metrics) if isinstance(last_metrics, dict) else {}
            total_comp = 0
            total_reas = 0
            total_cont = 0
            for rm in this_turn_rounds:
                total_comp += rm.get("completion_tokens") or 0
                total_reas += rm.get("reasoning_tokens") or 0
                total_cont += rm.get("content_tokens") or 0
            if total_comp:
                _turn_metrics["completion_tokens"] = total_comp
            if total_reas:
                _turn_metrics["reasoning_tokens"] = total_reas
            if total_cont:
                _turn_metrics["content_tokens"] = total_cont
            # Recalculate total_tokens: cumulative prompt + cumulative
            # completion across ALL rounds, not just the last round.
            # Do NOT stamp prompt_tokens_billed* here — that would change
            # the JSONL turn_ended metrics shape. Ledger billing sums
            # rounds at record_turn_cost time from round_metrics.
            pt = _turn_metrics.get("prompt_tokens") or 0
            tt = pt + total_comp
            if tt:
                _turn_metrics["total_tokens"] = tt
        else:
            _turn_metrics = dict(last_metrics) if isinstance(last_metrics, dict) else {}

        # F1 (plan v2): stamp the structured finish reason + the final round
        # count so the persisted turn record and the TurnDone event let
        # callers distinguish a CAPPED turn from a COMPLETED one and resume
        # from exactly where the budget ran out.
        _turn_metrics["completion_reason"] = completion_reason
        if completion_reason in ("round_cap", "budget"):
            # ACTUAL completed rounds. The cap-hit path
            # already counted every round at its RoundComplete; the
            # natural-stop-capped path just counted the final round above.
            # (The old ``round_idx + 1`` double-counted the cap-hit path: a
            # 2-round cap reported 3 rounds.)
            _turn_metrics["rounds"] = rounds_completed
            # Carry the final round's OWN wrap-up text
            # (content produced after the max-steps handoff injection) so
            # the delegate driver's resume brief quotes the actual handoff,
            # not the cumulative turn body.
            if final_round_text:
                _turn_metrics["wrap_up_text"] = final_round_text

        # COST: bill the WHOLE turn, not just the final round. record_turn_cost
        # prices ``_turn_metrics`` ONCE per turn, but ``prompt_tokens`` here is
        # deliberately the LAST round's value (it doubles as the displayed
        # "context size" — summing it would show a bogus 250k for a 50k context).
        # So a multi-round turn was priced at one round's input → ~80% undercount
        # (the $0.0057-across-2569-turns bug). ``_turn_cost_pico`` already summed
        # every round's priced cost using the SAME pricing table the mid-turn
        # budget gate uses; record it as an explicit override so the ledger row,
        # the gate, and the displayed spend all agree on the true per-turn cost
        # while the token columns stay meaningful for display. 0 == no-op
        # (fall back to last-round token pricing).
        if isinstance(_turn_metrics, dict):
            _turn_pico = int(self._turn_cost_pico.get(sharedagents_sid, 0) or 0)
            if _turn_pico > 0:
                from sharedagents_model.usage_pricing import pico_to_usd

                _turn_metrics["cost_usd_override"] = pico_to_usd(_turn_pico)

        # Also update the engine-audit so the persisted turn record
        # carries CUMULATIVE turn metrics, not just the last round's
        # numbers. The audit was stamped with ``last_metrics`` above;
        # overwrite with the cumulative dict for replay fidelity — the
        # TUI's live view already receives ``_turn_metrics`` via the
        # ``TurnDone`` event, but the on-disk session log reads from
        # ``_engine_audit["metrics"]``, and without this update the
        # saved record would show per-round (non-cumulative) totals.
        request.extra["_engine_audit"]["metrics"] = dict(_turn_metrics)

        # Stamp the live goal state onto the FINAL turn metrics too so the TUI's
        # goal-bar syncs even on a short single-round turn (which emits no
        # intermediate RoundComplete), and reflects a goal cleared/advanced by
        # the teacher at the just-passed Stop boundary.
        if isinstance(_turn_metrics, dict):
            _turn_metrics["active_goal"] = self._active_goals.get(sharedagents_sid, "") or ""
            # Stamp the outer-loop tracker (terminal badge + turns/budget) onto
            # the FINAL metrics too. The inner loop may have just cleared
            # active_goal on a met/blocked settle — without the terminal field
            # the widget would collapse and the badge would never show. This rides the same live channel the web/TUI already read.
            _turn_metrics.update(self._goal_progress_metrics(sharedagents_sid))
            # A turn that ends with the remaining todo blocked on the owner or
            # an outside event is NOT a completed task. Stamp the durable
            # disposition so the session projector records ``interrupted``
            # instead of ``succeeded`` (the parent-succeeded-after-abort bug).
            try:
                from .coordination import workflow_disposition

                _turn_metrics["workflow_disposition"] = workflow_disposition(
                    self._incomplete_todos(sharedagents_sid),
                )
            except Exception:  # noqa: BLE001 — projection metadata only
                pass

        _final_stream_gap = _detect_stream_truncation(
            "".join(turn_emitted_content_parts),
            "".join(content_buf),
        )
        if _final_stream_gap and terminal_state == "ok":
            yield TextDelta(text=_final_stream_gap, channel="content")
            turn_emitted_content_parts.append(_final_stream_gap)
            self._log_event(
                "STREAM_TAIL_EMITTED",
                session_id=getattr(session, "hash", ""),
                chars=len(_final_stream_gap),
                round_idx=round_idx,
                terminal=True,
            )

        # Zero-output guarantee backstop:
        # if — despite the continuation nudges and the final-answer round —
        # an ``ok`` turn is STILL ending with no model output ever delivered
        # (nudge cap exhausted, protections disabled for the model, or a
        # restored-marker suppression), mark it visibly instead of leaving
        # the operator staring at a blank turn. Warning-only: fires purely
        # on "nothing was delivered", so it has no false-positive surface
        # (a delegate-style answer that lives in the reconciled final
        # content keeps this quiet via ``content_buf``). Cancelled turns
        # end quiet on purpose.
        if (
            terminal_state == "ok"
            and not self._cancelled.get(sharedagents_sid, False)
            and turn_visible_chars == 0
            and not "".join(content_buf).strip()
        ):
            self._log_event(
                "NO_OUTPUT_TURN_END",
                session_id=getattr(session, "hash", ""),
                rounds=rounds_completed,
                nudges=len(nudge_log),
            )
            yield TextDelta(
                text=_format_runtime_warning(
                    "no-output-end",
                    nudges=len(nudge_log),
                ),
                channel="content",
            )

        if terminal_state == "error" and completion_reason in (
            "round_cap",
            "budget",
        ):
            # A CAPPED turn whose final round delivered
            # no wrap-up is an error terminal with NO user-facing message —
            # persistence keys error/error_classification off the last
            # TurnError, the session index projects "error" (vs idle) off a
            # truthy error, and the memory hook skips clean turn_completed
            # ingestion on any truthy error. Without this event the turn was
            # recorded as a clean stop, projected idle, and its partial
            # content seeded durable memory. TurnError is never terminal —
            # the error TurnDone below still terminates the turn (engine
            # contract).
            if completion_reason == "budget":
                yield TurnError(
                    message="turn hit the cost budget without producing a final-round wrap-up",
                    classification="budget_no_wrap_up",
                )
            else:
                yield TurnError(
                    message="turn hit the round cap without producing a final-round wrap-up",
                    classification="round_cap_no_wrap_up",
                )

        yield TurnDone(
            state=terminal_state,
            completion_reason=completion_reason,
            # Metrics ride TurnDone on BOTH terminals.
            # The stamped fields (completion_reason, rounds, wrap_up_text)
            # are what let callers distinguish a capped-error turn from a
            # worker failure — dropping them to {} on state=error made the
            # round bookkeeping invisible exactly when it matters most.
            metrics=_turn_metrics,
        )

    async def _notify(
        self,
        hook_engine: Any,
        hook_ctx: dict[str, Any] | None,
        kind: str,
        message: str,
        **extra: Any,
    ) -> None:
        """Fire Claude Code's ``Notification`` hook for a runtime event.

        Used at every operator-visible runtime warning surface — loop
        detected, distill recovery, force-terminate, continuation
        nudge — so an ECC plugin can route the event to a webhook /
        desktop notifier / telemetry sink. The hook's effects on
        return value are intentionally IGNORED here: Notification is
        observational, not gating. Never raises.
        """
        if hook_engine is None or not hook_engine.has_event("Notification"):
            return
        payload: dict[str, Any] = {
            "kind": kind,
            "message": message,
        }
        for k, v in (extra or {}).items():
            if v is not None and v != "":
                payload[k] = v
        try:
            await hook_engine.run(
                "Notification",
                payload,
                **(hook_ctx or {}),
            )
        except Exception:  # noqa: BLE001
            # Notification firing must never affect the turn — swallow
            # everything, including a misbehaving plugin's exception.
            pass

    def _overflow_notice(self, session: Any, upstream_error: str) -> str:
        """Visible runtime notice for a context overflow the pre-send count missed."""
        try:
            from sharedagents_agent.turn_runtime import parse_overflow_numbers

            parsed = parse_overflow_numbers(upstream_error) or {}
        except Exception:  # noqa: BLE001
            parsed = {}
        last = self._presend_last.get(getattr(session, "hash", "") or "") or {}
        counted = int(last.get("prompt") or 0)
        window = parsed.get("model_max") or last.get("window")
        out = parsed.get("requested_max_tokens")
        seen = parsed.get("input_tokens")
        self._log_event(
            "CONTEXT_OVERFLOW_NOTICE",
            session_id=getattr(session, "hash", ""),
            counted=counted,
            upstream_input=seen,
            window=window,
            output=out,
        )
        fields: dict[str, Any] = {
            "reason": "upstream rejected the request: prompt + output over the window",
        }
        if seen:
            fields["upstream_prompt"] = f">= {int(seen):,}"
        if out:
            fields["output_requested"] = f"{int(out):,}"
        if window:
            fields["window"] = f"{int(window):,}"
        if counted:
            fields["our_count"] = f"{counted:,}"
        fields["action"] = "compacting and retrying"
        return _format_runtime_warning("context-overflow", **fields)

    def _round_allowed_tools(self, session: Any) -> set[str] | None:
        """The tool-name ceiling for this round's wire request (``None`` = the
        agent's full permission set). Shared by the spawn and the pre-send
        token count so both describe the same request."""
        sid = session.hash
        # Normal rounds -> ``None`` so the wire builder derives
        # tools[] from the agent's FULL permission set (compose_tool_allowlist,
        # incl. session-scoped MCP). An explicit ceiling ONLY on a handoff round,
        # mirroring the dispatch gate (_runtime_allowed_tools): max-steps -> {}
        # (empty tools[] forces a text wrap-up); budget -> {"budget"}.
        if self._max_steps_handoff.get(sid):
            return set()
        if self._budget_handoff.get(sid):
            return {"budget"}
        # Delegation-depth cap: a LEAF agent sends the gate-filtered tool set so
        # the model never sees a delegation tool it cannot use.
        try:
            from sharedagents_agent.sandbox_broker import MAX_DELEGATION_DEPTH

            if int(os.environ.get("SHAREDAGENTS_SANDBOX_AGENT_DEPTH", "0")) >= MAX_DELEGATION_DEPTH:
                return self._runtime_allowed_tools(session)
        except Exception:  # noqa: BLE001
            pass
        return None

    async def _spawn_worker(
        self,
        *,
        session: Any,
        messages: list[dict[str, Any]],
        turn_id: str,
        env: dict[str, str] | None = None,
        nudge_retry_count: int = 0,
        sampling: dict[str, Any] | None = None,
        endpoint: str | None = None,
        model: str | None = None,
    ) -> tuple[Any, AsyncIterator[dict[str, Any]]]:
        """Spawn a worker subprocess and return ``(TurnState,
        event_stream)``.

        Reuses the legacy ``_default_spawn_worker`` helper from
        ``sharedagents_gateway.acp_bridge`` by providing a minimal
        backend-shim that exposes ``cfg`` and ``_session_for``. The
        legacy code path is unchanged; we just call into it.

        ``nudge_retry_count`` is forwarded to the spawner so the sampler
        seed is rotated on continuation-nudge retries — see
        ``_default_spawn_worker``'s ``_rotate_seed`` block. Default 0
        means "no rotation".

        ``endpoint`` / ``model`` are per-round REQUEST-LEVEL overrides
        (the LLM-fallback failover route). They change only the wire
        payload for this round — never the durable session header.
        """
        # Test-seam: caller-injected spawner wins.
        if self._worker_spawner is not None:
            return await self._worker_spawner(
                self,
                session,
                messages,
                turn_id,
                env,
            )

        from sharedagents_gateway.acp_bridge import _default_spawn_worker

        # The legacy spawner expects a CoreBackend-shaped object with ``cfg``
        # and ``_session_for``. Build a one-off shim.
        class _Shim:
            def __init__(self, cfg, sess):
                self.cfg = cfg
                self._sess = sess

            def _session_for(self, sid: str):
                return self._sess if sid == self._sess.hash else None

        sid = session.hash
        allowed = self._round_allowed_tools(session)

        # Per-request output budget: the room left in the window after THIS
        # prompt, capped at the model's output limit. A big context shrinks the
        # reserve instead of overflowing; an empty one allows the full limit.
        budget = self._presend_budget(
            session,
            messages,
            sampling=sampling,
            endpoint=endpoint,
            model=model,
        )
        if budget is not None:
            self._presend_last[sid] = budget
            if budget["ceiling"] < budget["out_cap"]:
                _override = dict(sampling or {})
                _current = int(_override.get("max_completion_tokens") or 0)
                _override["max_completion_tokens"] = min(_current, budget["ceiling"]) if _current else budget["ceiling"]
                sampling = _override

        shim = _Shim(self.cfg, session)
        turn_state, event_stream = await _default_spawn_worker(
            shim,  # backend
            sid,  # session_id
            [],  # prompt_blocks (unused; messages passed)
            turn_id,  # turn_id
            messages,  # explicit message list
            allowed_tool_names=allowed,
            nudge_retry_count=nudge_retry_count,
            sampling=sampling,
            endpoint_override=endpoint,
            model_override=model,
        )
        return turn_state, event_stream

    async def _consume_worker_events(
        self,
        event_stream: AsyncIterator[dict[str, Any]],
        turn: Any,
        round_state: _RoundState,
    ) -> AsyncIterator[AgentTurnEvent]:
        """Translate worker SSE-JSONL events to :data:`AgentTurnEvent`.

        Worker event kinds (worker.py L34 + downstream paths):

        * ``thinking``       → buffered into ``round_state.thinking_buf``;
                              forwarded as :class:`TextDelta` ONLY when
                              ``cfg.reasoning_visibility != "off"``.
                              Persistence into the session record is
                              GATED by ``session.persist_reasoning_traces``
                              (default True; operator-flippable for
                              privacy).
        * ``content``        → :class:`TextDelta`. Buffered into
                              ``round_state.content_buf``.
        * ``tool_start``     → :class:`ToolCallStarted` (legacy; no
                              current worker emits these — reserved for
                              future workers).
        * ``tool_delta``     → :class:`TextDelta` routed through a
                              ``call_id`` (today's ``tool_call_update``
                              with in-progress + content).
        * ``tool_result``    → :class:`ToolCallResult` (currently used
                              only by direct-dispatch path; LLM path
                              builds these from runner replies).
        * ``tool_calls``     → queue into ``round_state.pending_tool_calls``
                              for the post-stream dispatch step. Do NOT
                              yield — :meth:`_dispatch_tool_calls` will
                              fire the structured events when each call
                              actually runs.
        * ``metrics``        → record on the :class:`TurnState`,
                              ``session_log`` will pick it up at turn
                              end. Also yields :class:`TextDelta` with
                              empty text + a usage_update side-channel
                              for the gateway's ACP usage_update
                              translation (TUI status bar).
        * ``tool_args_loop`` → set ``round_state.loop_detected`` with
                              tier=``ratio`` + evidence (worker pre-
                              empted a tool-args token-collapse loop —
                              workers.py L362–380).
        * ``decode_drop``    → log + count; not a fatal event.
        * ``done``           → end of round; check
                              ``round_state.pending_tool_calls``.
        * ``aborted``        → :class:`TurnDone(state="aborted")` and
                              break out of the round loop.
        * ``error``          → :class:`TurnError` with the message, then
                              re-classify via
                              :meth:`_classify_upstream_error` to decide
                              between abort / reactive-compact-retry.
        """
        reasoning_visibility = str(getattr(self.cfg, "reasoning_visibility", "") or "off").lower()

        async for ev in event_stream:
            if not isinstance(ev, dict):
                continue
            kind = ev.get("kind") or ""
            if kind == "thinking":
                text = str(ev.get("text") or "")
                if not text:
                    continue
                round_state.thinking_buf.append(text)
                if reasoning_visibility != "off":
                    yield TextDelta(text=text, channel="thinking")
            elif kind == "content":
                text = str(ev.get("text") or "")
                if not text:
                    continue
                round_state.content_buf.append(text)
                yield TextDelta(text=text, channel="content")
            elif kind == "tool_calls":
                # Accumulate for post-stream dispatch.
                calls = ev.get("calls") or ev.get("tool_calls") or []
                for c in calls:
                    if isinstance(c, dict):
                        round_state.pending_tool_calls.append(dict(c))
                if isinstance(ev.get("reasoning_replay"), dict):
                    round_state.reasoning_replay = ev["reasoning_replay"]
            elif kind == "tool_args_loop":
                round_state.loop_detected = True
                round_state.loop_tier = str(ev.get("tier") or "ratio")
                round_state.loop_evidence = str(
                    ev.get("evidence") or "tool_args_loop",
                )
                round_state.loop_event = {
                    "tier": round_state.loop_tier,
                    "evidence": round_state.loop_evidence,
                    "args_chars": ev.get("args_chars"),
                    "checked_at": ev.get("checked_at"),
                    "tool_name": str(ev.get("name") or ""),
                    "tool_index": ev.get("index"),
                    "channel": "tool_args",
                }
            elif kind == "tool_args_loop_soft":
                # Telemetry only — surface detector hit, stream continues.
                self._log_event(
                    "TOOL_ARGS_LOOP_SOFT",
                    tier=str(ev.get("tier") or ""),
                    evidence=str(ev.get("evidence") or "")[:80],
                    tool_name=str(ev.get("name") or ""),
                    args_chars=ev.get("args_chars"),
                )
            elif kind == "decode_drop":
                round_state.json_decode_failures += 1
                self._log_event(
                    "DECODE_DROP",
                    session_id=getattr(turn, "session_hash", ""),
                    reason=str(ev.get("reason") or ""),
                )
            elif kind == "metrics":
                round_state.metrics = {k: v for k, v in ev.items() if k != "kind"}
            elif kind == "done":
                round_state.terminated_naturally = True
                return
            elif kind == "aborted":
                # An ``aborted`` event carrying ``reason`` is a worker
                # self-protective abort (e.g. ``tool_args_loop``); the
                # caller is expected to route it through the loop-
                # recovery cascade rather than ending the turn. A bare
                # ``aborted`` is an operator cancel / SIGTERM and ends
                # the turn. Mirrors the active ``run_turn`` consume
                # loop at L1316.
                if str(ev.get("reason") or ""):
                    return
                yield TurnDone(state="aborted", metrics=round_state.metrics or {})
                return
            elif kind == "error":
                msg = str(ev.get("message") or "worker error")
                yield TurnError(
                    message=msg,
                    classification=self._classify_upstream_error(msg),
                )
                return

    # -----------------------------------------------------------------
    # Worker subprocess
    # -----------------------------------------------------------------

    def _build_worker_request(
        self,
        session: Any,
        messages: list[dict[str, Any]],
        tools_payload: list[dict[str, Any]],
        *,
        endpoint: str,
        model: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Build the JSON payload passed to ``worker.main`` on stdin.

        Reference: acp_bridge.py L324–381. Carries:

        * ``endpoint`` / ``model`` / ``backend`` (provider hint).
        * Sampling knobs from the model row + per-turn override.
        * ``tools`` schemas (OpenAI function-tool format).
        * ``tool_choice`` (today: always ``"auto"``).
        * ``stream_options`` to request token-usage in the final chunk.
        * ``chat_template_kwargs.enable_thinking`` so Gemma 4 / Qwen3
          take the right prefill path.
        * ``headers`` — upstream-specific (Bearer, X-API-Key, …)
          resolved via ``cfg.upstream_headers(endpoint)``.
        """
        try:
            from sharedagents_gateway.acp_bridge import _build_worker_request
        except ImportError as exc:
            raise RuntimeError("sharedagents_gateway.acp_bridge._build_worker_request is unavailable") from exc

        # The legacy helper reads cfg from a backend-shaped object.
        # Build a thin shim so we don't have to instantiate the
        # full legacy backend just to format a request.
        class _Shim:
            def __init__(self, cfg: Any) -> None:
                self.cfg = cfg

        # Caller-provided endpoint/model/max_tokens override the
        # session-derived values the legacy helper picks up.
        original_endpoint = session.endpoint
        original_model = session.model
        try:
            if endpoint:
                session.endpoint = endpoint
            if model:
                session.model = model
            payload = _build_worker_request(
                _Shim(self.cfg),
                session,
                messages,
                include_tools=bool(tools_payload),
            )
        finally:
            session.endpoint = original_endpoint
            session.model = original_model

        if tools_payload:
            # Override the auto-discovered tools with the caller's list.
            payload["tools"] = list(tools_payload)
            payload.setdefault("tool_choice", "auto")
            # Explicitly enable parallel tool calls. OpenAI's default
            # is ``true`` but several OpenAI-compatible servers (incl.
            # some deepseek deployments) treat the missing field as
            # ``false`` and emit a single tool_use per response —
            # which serialises an 8-subagent fan-out into 8 round
            # trips. Setting it explicitly forces deepseek-v4-pro /
            # qwen / kimi to emit ALL eligible tool calls in one
            # round when the model decides multiple are appropriate.
            payload.setdefault("parallel_tool_calls", True)
        if max_tokens:
            payload["max_tokens"] = int(max_tokens)
        return payload

    async def _route_guard_target(self, user_task: str) -> str | None:
        """Deterministic single-phase delegation guard for the orchestrator.

        Returns the specialist a clear single-deliverable task must go to, or
        ``None``. Used ONLY when the model ends a turn with a final answer and
        made NO delegation at all (pure DIRECT): the small local model routinely
        answers a "write a spec / run the tests / deploy X" task inline, but it
        has no Write tool, so the artifact is lost. The guard then forces the
        delegation so every clear specialist task reaches its specialist (which
        owns the right tools + SDLC/research ontology process).

        Regex intent (high precision, ``subagent_policy.specialist_intent``)
        first, embedding semantic router (recall, margin-protected) as fallback.
        Compound / multi-phase requests are left to the orchestrator to sequence.
        Enabled for ``main``/``jarvis`` by default; a per-agent
        ``route_guard_enabled`` config flag overrides either way.
        """
        rg = getattr(self.cfg, "route_guard_enabled", None)
        if rg is False:
            return None
        if rg is None and getattr(self.cfg, "name", "") not in ("jarvis",):
            return None
        t = (user_task or "").strip()
        if not t or len(t) > 600:
            return None
        low = t.lower()
        # Multi-phase / compound request → let the orchestrator sequence it
        # rather than collapsing it onto a single specialist.
        if any(
            m in low
            for m in (
                " then ",
                " and then ",
                "; then",
                " after that",
                " afterwards",
                " followed by ",
                "step 1",
                "step 2",
                "firstly",
            )
        ):
            return None
        from sharedagents_agent.subagent_policy import specialist_intent

        want = specialist_intent(t)
        if not want:
            try:
                from sharedagents_agent.routing_semantic import predict_specialist

                want = await predict_specialist(t)
            except Exception:
                want = None
        if want and want not in ("explore", "general"):
            return want
        return None

    @staticmethod
    def _user_content_in_messages(
        messages: list[dict[str, Any]],
        live_content: Any,
        task_text: str,
    ) -> bool:
        for m in messages:
            if m.get("role") != "user":
                continue
            c = m.get("content")
            if c is live_content:
                return True
            if task_text and isinstance(c, str) and task_text in c:
                return True
            if task_text and isinstance(c, list):
                if TurnLoopMixin._task_text_from_user_content(c) == task_text:
                    return True
        return False

    @staticmethod
    def _task_text_from_user_content(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = str(part.get("text") or "").strip()
                    if text:
                        return text
        return ""
