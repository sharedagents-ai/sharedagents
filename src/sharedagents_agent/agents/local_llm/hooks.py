"""HooksMixin — extracted from the ``local_llm`` monolith (behaviour-preserving)."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from sharedagents_model.sampling import configured_sampling

from .state import (
    _HOOK_OUTCOMES_LOG,
    _now_iso,
    wrap_untrusted,
)


class HooksMixin:
    async def _fire_hook(
        self,
        hook_engine: Any,
        hook_ctx: dict[str, Any],
        event: str,
        payload: dict[str, Any],
    ) -> Any:
        """Run one Claude Code hook event; never raises.

        Returns an empty-effect :class:`~sharedagents_agent.hooks.HookOutcome`
        when the engine has no matching hooks or errors — callers treat
        "no hooks" and "hooks ran, no effect" identically.

        Appends a stripped outcome record to the per-turn
        ``_HOOK_OUTCOMES_LOG`` context-var so the session JSON can
        later answer "did a hook deny this tool call" without
        scanning gateway.log. The contextvar is set by
        ``run_turn`` and unset on exit; outside that frame
        (heartbeat, tests) this method runs without logging.
        """
        from sharedagents_agent.hooks import HookOutcome
        from sharedagents_core.secrets_scrubber import scrub_text

        if hook_engine is None or not hook_engine.has_event(event):
            return HookOutcome(event=event)
        try:
            outcome = await hook_engine.run(event, payload, **hook_ctx)
        except Exception as exc:  # noqa: BLE001
            self._log_event("HOOK_ERROR", event=event, error=str(exc))
            return HookOutcome(event=event)

        log = _HOOK_OUTCOMES_LOG.get()
        if log is not None:
            ac = getattr(outcome, "additional_context", "") or ""
            log.append(
                {
                    "event": event,
                    # ``payload`` for PreToolUse/PostToolUse hooks carries
                    # ``tool_name``; for UserPromptSubmit it's empty. The
                    # matcher is what the hook config keyed on; we
                    # surface ``tool_name`` as a proxy because the engine
                    # doesn't see the matcher pattern directly.
                    "matcher": str(payload.get("tool_name") or ""),
                    "decision": str(
                        getattr(outcome, "permission_decision", "") or "",
                    ),
                    "blocked": bool(getattr(outcome, "blocked", False)),
                    "block_reason": scrub_text(
                        str(getattr(outcome, "block_reason", "") or "")[:500],
                    ),
                    "additional_context_chars": len(ac),
                    "ran": int(getattr(outcome, "ran", 0) or 0),
                    "should_continue": bool(
                        getattr(outcome, "should_continue", True),
                    ),
                    "fired_at": _now_iso(),
                }
            )
        return outcome

    async def _fire_no_result(self, sharedagents_sid: str, messages: list, count: int, *, note: str = ""):
        """Render the ``no-result`` proactive delegate node and append its
        continuation to ``messages``. Async generator (yields the node events).
        The caller owns the counter COOLDOWN reset — this only fires. Called
        from the mid-grind top-of-loop chokepoint (the only no-result fire site;
        the budget-cap boundary is intentionally left to the turn-end review)."""
        _summary = f"{count} rounds with no result" + (f" ({note})" if note else "")
        _holder: list = []
        async for _ev in self._teacher_proactive_delegate(
            "no-result",
            "no_result_rounds",
            sharedagents_sid,
            summary=_summary,
            holder=_holder,
        ):
            yield _ev
        _out = _holder[0] if _holder else None
        if _out is not None and getattr(_out, "block_reason", ""):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "[teacher — you have run many rounds with no result; "
                        "follow this guidance] " + wrap_untrusted(_out.block_reason, source="teacher:no-result")
                    ),
                }
            )

    # -----------------------------------------------------------------
    # Mode / config
    # -----------------------------------------------------------------

    def _teacher_explicitly_on(self, sharedagents_sid: str | None) -> bool:
        """True when the operator ran ``/teacher on`` for this session."""
        return sharedagents_sid is not None and sharedagents_sid in self._teacher_on

    def _protections_off(self, sharedagents_sid: str | None = None) -> bool:
        """True when ALL loop/nudge/supervision protections are disabled for
        the active turn — the master kill-switch.

        Resolved from two SUBTRACTIVE sources (either true → protections off):
          * per-AGENT ``self.cfg.disable_protections`` (agents/<n>/config.json)
          * per-MODEL-ROW ``disableProtections`` on the resolved ``models[]``
            entry — keyed per inference-endpoint (row ``id``), so the SAME
            base model can be guarded on one quant/endpoint (e.g. nvfp4) and
            unguarded on another (fp8). Resolution mirrors ``_loop_detect_on``
            (local_llm.py:2154): session's endpoint+model, falling back to the
            agent's upstream_* defaults.

        Never force-ENABLES: a False result leaves each protection's own gate
        (family-based loop detection, teacher policy, …) untouched.
        """
        if getattr(self.cfg, "disable_protections", False):
            return True
        try:
            sess = self._sessions.get(sharedagents_sid) if sharedagents_sid else None
            endpoint = (
                ((getattr(sess, "endpoint", "") or "") if sess else "")
                or getattr(self.cfg, "upstream_endpoint", "")
                or ""
            )
            model = (
                ((getattr(sess, "model", "") or "") if sess else "") or getattr(self.cfg, "upstream_model", "") or ""
            )
            row = self.cfg.model_row_for_endpoint(endpoint, model)
            return bool(row.get("disableProtections"))
        except Exception:  # noqa: BLE001
            return False

    def _teacher_enabled(self, sharedagents_sid: str | None = None) -> bool:
        """True when a teacher model supervises this session.

        DEFAULT IS OFF — for ``/goal`` sessions AND for ordinary chat. Nothing
        turns the teacher on by itself: supervision starts when the operator
        runs ``/teacher on`` (or when the policy EXPLICITLY opts into
        policy-wide supervision with ``require_goal: false``, which is not the
        shipped default). A ``/goal`` never activates the teacher: the goal
        supervisor re-drives the session and the goal tool's completion claim,
        checked by the runtime, ends it. The teacher only advises.

        Resolution order (the per-session toggle is the authority):
          1. ``/teacher off`` → OFF, always.
          2. No teacher model configured → OFF.
          3. ``/teacher on`` → ON (including while a ``/goal`` is active —
             teacher helps toward a goal but is never *required* for it).
          4. Active ``/goal`` without explicit ``/teacher on`` → OFF; the goal
             supervisor owns continuation.
          5. Otherwise: ON only when the policy explicitly opts in
             (``require_goal == false``). The default is OFF.
        """
        # (0) master protections kill-switch wins over everything — a
        # big/hosted API model (Opus) gets no teacher babysitting.
        if self._protections_off(sharedagents_sid):
            return False
        # (1) explicit off wins over everything.
        if sharedagents_sid is not None and sharedagents_sid in self._teacher_off:
            return False
        try:
            pol = self.cfg.teacher_policy() if hasattr(self.cfg, "teacher_policy") else {}
        except Exception:  # noqa: BLE001
            pol = {}
        # (2) no configured teacher → can't supervise.
        if not pol.get("enabled"):
            return False
        # (3) explicit on → supervise (helps toward an active /goal if set).
        if self._teacher_explicitly_on(sharedagents_sid):
            return True
        # (4) a goal never opts into Teacher by itself; the goal supervisor
        # owns continuation unless the operator explicitly selected Teacher.
        if sharedagents_sid is not None and self._goal_active(sharedagents_sid):
            return False
        # (5) outside a goal, a goal-gated policy remains off until the operator
        # explicitly enables it. Rare engine-internal callers without a session
        # id preserve the historical policy-only probe.
        return not (pol.get("require_goal", True) and sharedagents_sid is not None)

    def _set_goal_state(
        self,
        sharedagents_sid: str,
        active: str,
        *,
        terminal: str | None = None,
        reset_progress: bool = False,
    ) -> bool:
        """Set the live goal (engine dict = source of truth), mirror it onto the
        Session, and PERSIST so the sessions DB reflects the current goal — for
        restart durability and orchestrator observation. The engine dict is
        authoritative because a Session instance can be reloaded mid-turn; the
        persisted field is the durable shadow. ONE goal per session at a time —
        a new goal OVERRIDES the previous one (no queue).

        ``terminal`` records the outer-loop terminal state ("met" | "blocked" |
        "paused") on the persisted header. A MET settle DETACHES the projection:
        the widget hides while the
        durable row stays met until separately archived — so ``goal_terminal``
        is cleared, not stamped. Blocked/paused keep their badge (the inner
        loop clears the goal the instant it settles, so the badge can't ride on
        the goal text alone). ``reset_progress`` wipes ``goal_terminal`` +
        ``goal_turn_count`` — used when a FRESH goal is set/cleared so a new
        goal never inherits the previous cycle's stale badge or turn count.

        A terminal settle writes the durable registry FIRST and then projects
        the record the registry returned, so the header never shows an
        outcome that was not recorded: a refused transition leaves the badge
        of the status the goal actually has, and is logged as
        ``GOAL_TRANSITION_REJECTED``. The header-only stamp remains for a
        session with no durable goal and for a registry that cannot be read.

        Returns True when this call performed a NEW terminal transition
        (running → met/blocked/paused, or paused → met) so callers emit the
        lightweight "Goal completed" notice EXACTLY once; False when the goal
        was already terminal (idempotent no-op), the transition was refused,
        or no durable goal was touched.
        """
        self._active_goals[sharedagents_sid] = active or ""
        sess = self._sessions.get(sharedagents_sid)
        # Durable terminal write at the evaluator DECISION. Persist
        # met/blocked/paused into goals.db HERE so the supervisor's re-drive
        # loop stops at its next top-of-loop status check even if the deferred
        # finalizers (agent._maybe_goal_re_drive / supervisor._sync_terminal_to
        # _registry) are skipped (aborted turn, lost lease, crash). Gated on a
        # terminal state → no extra sqlite write on the /goal set/clear path.
        # Fail-soft: any DB error falls back to the header-only stamp below
        # (finalizers still act as an idempotent backstop).
        if terminal in ("met", "blocked", "paused"):
            try:
                from sharedagents_goal.goal_registry import (
                    STATUS_BLOCKED,
                    STATUS_MET,
                    STATUS_PAUSED,
                    get_goal_registry,
                    sync_session_goal_projection,
                )

                reg = get_goal_registry()
                gid = ""
                if sess is not None:
                    gid = str(getattr(sess, "goal_id", "") or "").strip()
                if not gid:
                    gid = reg.goal_id_for_root_session(sharedagents_sid) or ""
                if gid:
                    # ATOMIC transition outcome: the
                    # registry CAS returns (record, transitioned) — exactly
                    # one racing callback (the evaluator vs stop.goal_met)
                    # sees transitioned=True, so the "Goal completed" notice
                    # fires exactly once. A read-then-infer
                    # (before = get(); mark(); before.status == RUNNING) can
                    # BOTH report a transition.
                    if terminal == "met":
                        rec, transitioned = reg._terminal_transition(
                            gid,
                            STATUS_MET,
                            reason="goal_met",
                        )
                    elif terminal == "blocked":
                        rec, transitioned = reg._terminal_transition(
                            gid,
                            STATUS_BLOCKED,
                            reason="goal_blocked",
                        )
                    else:
                        rec, transitioned = reg._terminal_transition(
                            gid,
                            STATUS_PAUSED,
                            reason=terminal,
                            paused=True,
                        )
                    if sess is not None:
                        sync_session_goal_projection(sess, rec)
                    if rec.status != terminal:
                        try:
                            self._log_event(
                                "GOAL_TRANSITION_REJECTED",
                                session_id=sharedagents_sid,
                                goal_id=gid,
                                requested=terminal,
                                status=rec.status,
                            )
                        except Exception:  # noqa: BLE001 — the record stands
                            pass
                    return transitioned
            except Exception:  # noqa: BLE001
                pass
        if sess is not None:
            try:
                sess.active_goal = active or ""
                if reset_progress:
                    sess.goal_terminal = ""
                    sess.goal_turn_count = 0
                if terminal is not None:
                    # MET detaches the projection (no badge); blocked/paused
                    # stamp their badge on the header.
                    sess.goal_terminal = "" if terminal == "met" else terminal or ""
                sess.save()
            except Exception:  # noqa: BLE001
                pass
        return False

    def _resolve_teacher_model_alias(self, alias: str) -> dict | None:
        """Find the ``models[]`` row whose ``alias`` or ``id`` matches — used
        by ``/teacher model <alias>`` to pick a supervising model. Returns the
        row dict, or None if no model matches (caller surfaces the error)."""
        a = (alias or "").strip()
        if not a:
            return None
        for m in getattr(self.cfg, "models", None) or []:
            if isinstance(m, dict) and (m.get("alias") == a or m.get("id") == a):
                return dict(m)
        return None

    def _teacher_model_row(self, sharedagents_sid: str | None = None) -> dict | None:
        """The EFFECTIVE teacher model row for this session: the per-session
        ``/teacher model <alias>`` override if set and resolvable, else the
        config's ``role=="teacher"`` row."""
        if sharedagents_sid is not None:
            ov = (self._teacher_model_override.get(sharedagents_sid) or "").strip()
            if ov:
                row = self._resolve_teacher_model_alias(ov)
                if row is not None:
                    return row
        try:
            return self.cfg.teacher_model() if hasattr(self.cfg, "teacher_model") else None
        except Exception:  # noqa: BLE001
            return None

    def _teacher_model_label(self, sharedagents_sid: str | None = None) -> str:
        """Display label for the ``delegate_teacher`` nodes — the EFFECTIVE
        teacher model's id (per-session override, else the ``role=="teacher"``
        row). Read live so the TUI always shows the ACTUAL supervising model
        and never drifts when the operator swaps it via ``/teacher model``.
        Falls back to ``"teacher"`` if unresolved."""
        try:
            row = self._teacher_model_row(sharedagents_sid) or {}
            return row.get("id") or row.get("alias") or "teacher"
        except Exception:  # noqa: BLE001
            return "teacher"

    def _load_verdict_engine(self):
        """Import the stdlib-only teacher verdict engine
        (``sharedagents_agent/agents/data/teacher_review.py``) as a module
        and cache it.

        The engine is loaded by path (not a package import) because it is
        deliberately dependency-free — it must run even when the sharedagents
        venv is broken (the robustness doctrine), so it lives under
        ``sharedagents_agent/agents/data/`` rather than a regular package import.
        Loading + calling it in-process is what makes the teacher run
        COMPLETELY inside sharedagents — no shell, no subprocess, no hooks."""
        eng = getattr(self, "_verdict_engine", None)
        if eng is not None:
            return eng
        import importlib.util

        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        path = os.path.join(data_dir, "teacher_review.py")
        if not os.path.exists(path):
            self._verdict_engine = None
            return None
        spec = importlib.util.spec_from_file_location("sharedagents_teacher_review", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._verdict_engine = mod
        return mod

    def _teacher_settings(self, sharedagents_sid: str | None = None) -> dict:
        """Resolve the review settings from the agent config (the EFFECTIVE
        teacher ``models[]`` row — per-session ``/teacher model`` override
        else the ``role=="teacher"`` row — + the ``teacher`` policy block).
        Resolved directly from the live agent config — no env, no config-file
        re-reads. This is the config plumbing the native path owns directly."""
        row = self._teacher_model_row(sharedagents_sid) or {}
        try:
            pol = (self.cfg.teacher_policy() if hasattr(self.cfg, "teacher_policy") else {}) or {}
        except Exception:  # noqa: BLE001
            pol = {}
        api_key = os.environ.get(row.get("apiKeyEnv") or "", "") if row.get("apiKeyEnv") else ""
        # The shipped profile prompt rides in as ``prompt_text``; the verdict
        # engine falls back to its embedded _DEFAULT_PROMPT when it is empty.
        # Never load from ~/.sharedagents.
        prompt_path = ""
        try:
            from sharedagents_agent.teacher_supervisor import teacher_agent_system_prompt

            prompt_text = teacher_agent_system_prompt()
        except Exception:  # noqa: BLE001
            prompt_text = ""
        # Honour an explicit 0 (the round-loop trigger treats 0 as disabled; keep
        # the verdict-engine view consistent rather than resurrecting the default
        # via ``or``), but fall back to the default on missing/None/garbage.
        _tft_raw = pol.get("tool_failure_threshold")
        try:
            tool_fail_threshold = 5 if _tft_raw is None else int(_tft_raw)
        except (TypeError, ValueError):
            tool_fail_threshold = 5
        # Carried for completeness/inspection even though run_review does not use
        # it — the no-result trigger is an ENGINE round-loop concept resolved by
        # _teacher_round_thresholds() directly from policy. Including it keeps the
        # settings view consistent with the policy.
        _nrl_raw = pol.get("no_result_round_limit")
        try:
            no_result_round_limit = 20 if _nrl_raw is None else int(_nrl_raw)
        except (TypeError, ValueError):
            no_result_round_limit = 20
        return {
            "sampling": configured_sampling(self.cfg, row, row.get("id", ""), purpose="teacher"),
            "endpoint": row.get("baseUrl") or "",
            "model": row.get("id") or "",
            "api_key": api_key,
            "timeout": float(pol.get("boundary_timeout_seconds", pol.get("timeout_seconds", 100)) or 100),
            "prompt_text": prompt_text,
            "prompt_path": prompt_path,
            "deep_tier": pol.get("deep_tier", "auto"),
            "min_confidence": float(pol.get("min_confidence", 0.6) or 0.6),
            "max_interventions": int(pol.get("max_interventions", 6) or 6),
            "stall_limit": int(pol.get("stall_limit", 2) or 2),
            "tool_failure_threshold": tool_fail_threshold,
            "no_result_round_limit": no_result_round_limit,
        }

    # -----------------------------------------------------------------
    # /goal Stop-hook evaluator (Claude Code semantics — no teacher agent)
    # -----------------------------------------------------------------

    def _active_goal_from_registry(self, sharedagents_sid: str | None) -> str:
        """Condition text of the RUNNING durable goal bound to this session
        (goals.db), or "". Lets the engine recognise a supervisor-driven goal
        whose session-header ``active_goal`` projection is stale/empty — the
        root cause of the infinite false-positive re-drive (the Stop-hook
        evaluator gate ``_uses_goal_stop_hook`` was always False)."""
        if not sharedagents_sid:
            return ""
        try:
            from sharedagents_goal.goal_registry import get_goal_registry

            reg = get_goal_registry()
            gid = reg.goal_id_for_root_session(sharedagents_sid)
            if not gid:
                return ""
            g = reg.get(gid)
            return str(getattr(g, "condition_text", "") or "").strip() if g else ""
        except Exception:  # noqa: BLE001
            return ""

    def _goal_active(self, sharedagents_sid: str | None) -> bool:
        if not sharedagents_sid:
            return False
        if (self._active_goals.get(sharedagents_sid, "") or "").strip():
            return True
        # Fall back to the durable registry: a supervisor-driven goal may be
        # live in goals.db while the in-memory projection is empty. Seed it so
        # the evaluator can run AND has the condition text to judge against.
        ag = self._active_goal_from_registry(sharedagents_sid)
        if ag:
            self._active_goals[sharedagents_sid] = ag
            return True
        return False

    @staticmethod
    def _teacher_ctrl_to_hook(ctrl: dict) -> Any:
        from types import SimpleNamespace

        blocked = ctrl.get("decision") == "block"
        goal_met = bool(ctrl.get("goal_met"))
        goal_blocked = bool(ctrl.get("goal_blocked"))
        unavailable = bool(ctrl.get("goal_unavailable")) or (
            not blocked
            and not goal_met
            and not goal_blocked
            and bool(ctrl.get("systemMessage"))
            and "unavailable" in str(ctrl.get("systemMessage", "")).lower()
        )
        return SimpleNamespace(
            blocked=blocked,
            block_reason=(ctrl.get("reason") or "") if blocked else "",
            review_text=(ctrl.get("review") or ""),
            system_messages=([ctrl["systemMessage"]] if ctrl.get("systemMessage") else []),
            additional_context="",
            should_continue=True,
            stop_reason="",
            goal_met=goal_met,
            goal_blocked=goal_blocked,
            goal_unavailable=unavailable,
        )

    def _teacher_round_thresholds(self) -> tuple[int, int]:
        """Resolve the two PROACTIVE mid-turn firing thresholds the round loop
        watches, from the teacher policy:

          * ``tool_failure_threshold`` — fire after this many CONSECUTIVE
            errored tool calls in a turn (default 5).
          * ``no_result_round_limit`` — fire after this many consecutive rounds
            in a turn-segment that deliver no result (default 20).

        A non-positive value disables that trigger — so a configured ``0`` MUST
        survive (the ``int(x or default)`` idiom would wrongly coerce 0 back to
        the default, making the trigger un-disableable). Resolved from the live
        agent config — no env, no config-file re-reads."""
        try:
            pol = (self.cfg.teacher_policy() if hasattr(self.cfg, "teacher_policy") else {}) or {}
        except Exception:  # noqa: BLE001
            pol = {}

        def _int(key: str, default: int) -> int:
            # Honour an explicit 0 (disable); only fall back to default on
            # missing/None/garbage — never on a real 0.
            v = pol.get(key, default)
            if v is None:
                return default
            try:
                return int(v)
            except (TypeError, ValueError):
                return default

        return _int("tool_failure_threshold", 5), _int("no_result_round_limit", 20)

    @staticmethod
    def _teacher_review_failed_open(outcome) -> bool:
        """True when a proactive review did NOT actually run.

        Used to REPORT the outcome honestly in the ``delegate_teacher`` node
        (ok=False + 'supervision unavailable' instead of a misleading clean
        'on track'). It does NOT gate the counter reset — the round loop applies
        a uniform COOLDOWN: it resets the streak/counter after EVERY fire
        (helped / vetoed / fail-open) so the next fire needs another full budget.
        A fail-open is therefore re-ATTEMPTED after another budget (never
        swallowed) rather than every round (no spam).

        A review that genuinely ran always yields evidence: either a fire-back
        (``blocked`` with a continuation) or a readable verdict digest
        (``review_text`` — emitted on EVERY real verdict incl. complete /
        needs_human). A fail-open yields NEITHER:
          * ``run_review`` internal fail-open → ``{"continue": True,
            "systemMessage": "…supervision unavailable…"}`` (no review),
          * the native wrapper's outer timeout / missing-engine / generic
            exception → ``{"continue": True}`` (no review, no systemMessage).
        Keying on the ABSENCE of evidence (not a single message string) catches
        all of these uniformly."""
        if outcome is None:
            return True
        if getattr(outcome, "blocked", False):
            return False
        return not (getattr(outcome, "review_text", "") or "").strip()

    @staticmethod
    def _teacher_review_body(outcome) -> str:
        """Visible teacher result, including its durable delegation session."""
        body = getattr(outcome, "review_text", "") or ""
        child_sid = getattr(outcome, "delegate_session_id", "") or ""
        if child_sid:
            session_line = f"delegation session_id: {child_sid}"
            body = f"{session_line}\n\n{body}" if body else session_line
        block_reason = getattr(outcome, "block_reason", "") or ""
        if block_reason:
            body = f"{body}\n\n→ {block_reason}" if body else block_reason
        return body

    async def _teacher_proactive_delegate(
        self, event: str, trigger: str, sharedagents_sid: str, *, summary: str, holder: list
    ):
        """Render a PROACTIVE mid-turn ``delegate_teacher`` node and run the
        review natively, IN-TURN (no boundary needed).

        Async generator: yields the standard tool-node events
        (ToolCallStarted → optional review TextDelta → ToolCallResult), exactly
        like the reasoning-loop / pre_idle delegations. The HookOutcome-shaped
        result is appended to ``holder`` so the caller can read
        ``block_reason`` and fire the continuation BACK into the round loop.
        Fail-open: a failed review just renders a node and the turn proceeds."""
        import uuid as _uuid

        from ..engine import TextDelta, ToolCallResult, ToolCallStarted

        _id = f"teacher-{_uuid.uuid4().hex[:8]}"
        yield ToolCallStarted(
            call_id=_id,
            name="delegate_teacher",
            arguments={"model": self._teacher_model_label(sharedagents_sid), "trigger": trigger, "summary": summary},
            is_outer=False,
        )
        out = await self._teacher_review_native(event, sharedagents_sid)
        holder.append(out)
        # Defensive attribute access: the review path fail-opens to a full
        # SimpleNamespace today, but never assume — a stripped/None outcome must
        # NOT raise inside this generator (that would abort the student's turn
        # instead of bypassing the review).
        _body = self._teacher_review_body(out)
        if _body:
            yield TextDelta(text=_body[:2000], channel="tool_delta", call_id=_id)
        # Report the OUTCOME honestly. A fail-open (timeout / missing engine /
        # 'supervision unavailable') did NOT review anything — never render it as
        # a clean "on track" approval (that would mislead the operator and make
        # repeated proactive fires look like teacher sign-offs). Mark it ok=False.
        _failed_open = self._teacher_review_failed_open(out)
        if getattr(out, "blocked", False):
            _decision = "decision: continue — firing a continuation back to help the student"
        elif _failed_open:
            _decision = (
                "decision: supervision unavailable — the teacher review did "
                "not run; the streak/counter resets (cooldown) and the "
                "trigger re-attempts after another full budget"
            )
        else:
            _decision = "decision: no action — the teacher reviewed and the student is on track"
        yield ToolCallResult(
            call_id=_id,
            ok=not _failed_open,
            content=(f"{_decision}\n\n{_body[:2000]}" if _body else _decision),
        )

    async def _teacher_review_native(self, event: str, sharedagents_sid: str):
        """Run the teacher review NATIVELY, fully in-process.

        The verdict engine (``teacher_review.run_review``) is imported and
        CALLED directly here — no shell, no subprocess, no hooks. The engine
        owns the invocation, the result is surfaced as a ``delegate_teacher``
        node (see the call sites), and the continuation prompt is fired BACK
        into the calling agent. Returns a HookOutcome-shaped object (``blocked`` /
        ``block_reason`` / ``system_messages`` / ``additional_context``) so
        the existing continue/stop handling downstream is unchanged.
        Fail-open: any error → not blocked (turn ends), never wedges."""
        sa_home = os.environ.get("SHAREDAGENTS_HOME") or os.path.expanduser("~/.sharedagents")
        transcript = os.path.join(sa_home, "sessions", f"{sharedagents_sid}.jsonl")
        # Pass the CURRENT active goal explicitly so the review judges against
        # it — not the transcript's first user message, which goes stale after
        # the engine auto-promotes a queued follow-up goal mid-session.
        _agoal = self._active_goals.get(sharedagents_sid, "") or ""
        settings = self._teacher_settings(sharedagents_sid)
        agent_name = getattr(self.cfg, "name", "jarvis")
        visible_child = None
        try:
            from sharedagents_agent.teacher_supervisor import (
                spawn_visible_teacher_subagent,
            )

            visible_child = spawn_visible_teacher_subagent(
                self._sessions.get(sharedagents_sid),
                teacher_row=self._teacher_model_row(sharedagents_sid),
                system_prompt=settings.get("prompt_text", ""),
                event=event,
                goal=_agoal,
            )
        except Exception:  # noqa: BLE001 — review still runs without a child row
            visible_child = None
        ctrl = {"continue": True}
        out = None
        try:
            mod = self._load_verdict_engine()
            if mod is not None:
                # run_review does blocking urllib + its own per-attempt timeout +
                # bounded retries, and NEVER raises (it fail-opens internally), so
                # run it in a worker thread to keep the event loop free. The outer
                # wait_for is a generous backstop around the engine's own timeout;
                # on TimeoutError we fail open (the orphan thread finishes on its
                # own — bounded by the socket timeout — and its result is dropped).
                loop = asyncio.get_event_loop()
                outer = settings["timeout"] * 2 + 20
                try:
                    ctrl = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: mod.run_review(
                                event,
                                session_id=sharedagents_sid,
                                transcript_path=transcript,
                                cwd=sa_home,
                                goal=_agoal,
                                agent=agent_name,
                                settings=settings,
                            ),
                        ),
                        timeout=outer,
                    )
                except Exception:  # noqa: BLE001 (incl. asyncio.TimeoutError)
                    ctrl = {"continue": True}
                # The verdict engine must return a JSON OBJECT; coerce anything
                # else (defensive — run_review always returns a dict).
                if not isinstance(ctrl, dict):
                    ctrl = {"continue": True}
            out = self._teacher_ctrl_to_hook(ctrl)
        finally:
            # The child row must be closed on EVERY exit path. Cancellation is a
            # ``BaseException``, so it slips past every ``except Exception``
            # above: cancelling the turn mid-review (Stop, gateway shutdown,
            # worker teardown) used to leave the child with no ``turn_ended``
            # and ``session_state='prompting'`` — a permanently live row in the
            # Agent Monitor for a review that could never report. ``finish()``
            # is internally guarded and never raises.
            child_sid = getattr(visible_child, "child_sid", "") or ""
            if child_sid:
                if out is not None:
                    out.delegate_session_id = child_sid
                    failed_open = self._teacher_review_failed_open(out)
                else:
                    # Torn down before a verdict existed: the review did not
                    # run, so say so rather than implying a clean finish.
                    failed_open = True
                child_body = (ctrl.get("review") or ctrl.get("reason") or "").strip() if isinstance(ctrl, dict) else ""
                visible_child.finish(
                    content=child_body,
                    error=("teacher review unavailable" if failed_open else None),
                )
        return out

    async def _teacher_terminal_review(self, event: str, sharedagents_sid: str):
        """Emit a teacher review at a FAILURE terminal (error / cap) as a
        ``delegate_teacher`` node, WITHOUT firing a continuation back.

        Per the teacher design: infrastructure-failure terminals
        (worker EOF, upstream/transient error, stop-continuation cap) must NOT
        auto-re-drive — that risks an infinite loop against a broken runtime.
        Here the teacher only SEES the failure and logs a verdict (visible in
        the TUI as a delegate node), then the turn ends. Async-generator: yields
        the standard tool-node events. Fail-open: a failed review just renders a
        'reviewed' node and ends the turn."""
        import uuid as _uuid

        from ..engine import TextDelta, ToolCallResult, ToolCallStarted

        _tc_id = f"teacher-{_uuid.uuid4().hex[:8]}"
        yield ToolCallStarted(
            call_id=_tc_id,
            name="delegate_teacher",
            arguments={
                "model": self._teacher_model_label(sharedagents_sid),
                "trigger": event,
                "summary": "terminal review (no fire-back)",
            },
            is_outer=False,
        )
        out = await self._teacher_review_native(event, sharedagents_sid)
        _t_body = self._teacher_review_body(out)
        if _t_body:
            yield TextDelta(text=_t_body[:2000], channel="tool_delta", call_id=_tc_id)
        _t_failed = self._teacher_review_failed_open(out)
        if out.blocked:
            _t_decision = "decision: not-done — turn ended on a failure terminal (reviewed, no continuation)"
        elif _t_failed:
            _t_decision = "decision: supervision unavailable — turn ended"
        else:
            _t_decision = "decision: reviewed — turn ended (no continuation)"
        yield ToolCallResult(
            call_id=_tc_id,
            ok=not _t_failed,
            content=(f"{_t_decision}\n\n{_t_body[:2000]}" if _t_body else _t_decision),
        )
