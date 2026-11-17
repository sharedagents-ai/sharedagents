"""CostMixin — extracted from the ``local_llm`` monolith (behaviour-preserving)."""

from __future__ import annotations

import sys
from typing import Any

from .state import (
    _LOG,
    BUDGET_HANDOFF_USER,
    _classify,
)


def billed_prompt_from_rounds(rounds: list[dict[str, Any]]) -> dict[str, int]:
    """Sum per-round prompt buckets for ledger billing.

    Display ``prompt_tokens`` stays last-round occupancy (context meter).
    ``prompt_tokens_billed*`` is the SUM across every LLM round so the
    ledger token columns match actual prefill, the way ``rebuild_usage.py``
    wrote them.
    """
    billed = 0
    cached = 0
    cache_w = 0
    for rm in rounds:
        if not isinstance(rm, dict):
            continue
        # Some providers/engines already expose one-call billed totals under
        # the explicit ``*_billed`` keys. Preserve those rather than treating
        # a display-only context occupancy as the billing basis.
        if rm.get("prompt_tokens_billed") is not None:
            billed += int(rm.get("prompt_tokens_billed") or 0)
            cached += int(rm.get("prompt_tokens_billed_cached") or 0)
            cache_w += int(rm.get("prompt_tokens_billed_cache_write") or 0)
            continue
        ca = int(rm.get("prompt_tokens_cached") or 0)
        cw = int(rm.get("prompt_tokens_cache_write") or 0)
        pt = int(rm.get("prompt_tokens") or 0)
        fresh = rm.get("prompt_tokens_fresh")
        if pt <= 0 and fresh is not None:
            pt = int(fresh or 0) + ca
        billed += pt
        cached += ca
        cache_w += cw
    return {
        "prompt_tokens_billed": billed,
        "prompt_tokens_billed_cached": cached,
        "prompt_tokens_billed_cache_write": cache_w,
    }


class CostMixin:
    # -----------------------------------------------------------------
    # Compaction
    # -----------------------------------------------------------------

    def _effective_context_window(self, session: Any) -> int:
        """Authoritative context window for compaction / overflow guards.

        Prefer the bound model row's ``contextWindow`` when present — a
        session can still carry a stale larger ``context_window`` after
        ``/model int4`` (65k) while ``session.context_window`` remains
        256k from the previous local model. When both disagree, take the
        smaller so proactive compaction never under-fires against a
        tighter upstream (Jul 30: plan/build HTTP 400 at 65536).
        """
        sess_cw = int(getattr(session, "context_window", 0) or 0)
        cfg_cw = int(getattr(self.cfg, "upstream_max_context", 0) or 0)
        row_cw = 0
        try:
            row = self._resolve_model_row(
                getattr(session, "model", "") or "",
                endpoint=getattr(session, "endpoint", "") or "",
            )
            if isinstance(row, dict):
                row_cw = int(row.get("contextWindow") or row.get("context_window") or 0)
        except Exception:  # noqa: BLE001
            row_cw = 0
        if row_cw > 0 and sess_cw > 0 and row_cw != sess_cw:
            return min(row_cw, sess_cw)
        return row_cw or sess_cw or cfg_cw

    def _effective_output_budget(self, session: Any) -> int:
        """The ``max_tokens`` the NEXT worker request will actually
        reserve for output.

        Mirrors ``_build_worker_request`` (acp_bridge.py) precedence:

          1. a live ``session.max_tokens_override`` (set by the reactive
             overflow squeeze) — wins, because the wire request honours
             it over the profile;
          2. else OpenCode's rule, ``resolve_max_tokens``: min(the
             model's output limit, the global cap), for the session's own
             route. A per-turn ``sampling`` override or a failover route
             is resolved on the wire only.

        The proactive compaction guard (:meth:`_maybe_compact`) MUST
        size both its trigger gate and its compaction target off THIS
        number. Reading ``cfg.upstream_max_completion_tokens`` instead
        under-reserved output headroom by ~12k (4096 vs the profile's
        16384) and let rounds overflow the context window despite
        nominally "compacting at 85%".
        """
        override = getattr(session, "max_tokens_override", None)
        if isinstance(override, int) and override > 0:
            return override
        from sharedagents_model.models import (
            for_name,
            output_token_cap,
            resolve_max_tokens,
        )

        try:
            # Same model/endpoint fallback as ``_build_worker_request``: a
            # mode-named or empty session model is sent as the agent's
            # upstream model, so reserve THAT model's budget.
            model = str(getattr(session, "model", "") or "")
            if not model or model.lower() in ("auto", "ask", "plan"):
                model = str(getattr(self.cfg, "upstream_model", "") or model)
            row: Any = {}
            row_for = getattr(self.cfg, "model_row_for_endpoint", None)
            if callable(row_for):
                endpoint = str(getattr(session, "endpoint", "") or getattr(self.cfg, "upstream_endpoint", "") or "")
                row = row_for(endpoint, model) or {}
            mode = str(getattr(session, "system_mode", "") or getattr(session, "mode", "") or "auto")
            mdl = for_name(model)
            profile = getattr(mdl, mode, None) or getattr(mdl, "auto", None)
            return resolve_max_tokens(
                row=row,
                model_id=model,
                default_limit=getattr(profile, "max_completion_tokens", None),
                cfg=self.cfg,
            )
        except Exception:  # noqa: BLE001
            return output_token_cap(self.cfg)

    def _last_authoritative_prompt_tokens(self, session: Any) -> int | None:
        """Real ``prompt_tokens`` vLLM reported for the most recent round.

        Returns the upstream-authoritative count from the latest
        ``session.round_metrics`` entry that carries one, falling back to the
        PERSISTED ``session.last_prompt_tokens`` (so the count survives a
        reload — ``round_metrics`` is in-memory only and comes back empty on a
        fresh load, which would otherwise drop the gate to the inflated char
        estimate and spuriously compact). ``None`` only when no round has ever
        run (a genuinely fresh session). The compaction gate
        (:meth:`_compaction_target`) prefers this over the char-length estimate
        so neither an inline base64 image (counted as ~200k text "tokens" by
        ``_approx_tokens`` but billed as a few hundred visual tokens) NOR a
        char-heavy replayed-but-wire-squeezed history can spuriously trip the
        compaction / memory-flush machinery.
        """
        try:
            rms = getattr(session, "round_metrics", None) or []
            for rm in reversed(rms):
                pt = int((rm or {}).get("prompt_tokens") or 0)
                if pt > 0:
                    return pt
        except Exception:  # noqa: BLE001
            pass
        # Persisted fallback — survives reload / process restart.
        try:
            pt = int(getattr(session, "last_prompt_tokens", 0) or 0)
            if pt > 0:
                return pt
        except Exception:  # noqa: BLE001
            pass
        return None

    def _record_presend_accuracy(self, session: Any, billed_prompt: int) -> None:
        """Compare the pre-send count with the upstream's billed prompt, log the
        miss, and learn the character-estimate ratio for rows with no tokenizer."""
        sid = getattr(session, "hash", "") or ""
        last = self._presend_last.get(sid)
        if not last or billed_prompt <= 0:
            return
        counted = int(last.get("prompt") or 0)
        raw = int(last.get("raw_prompt") or counted)
        if not last.get("exact") and raw > 0:
            self._presend_ratio[sid] = min(3.0, max(1.0, billed_prompt / raw))
        elif last.get("exact") and raw > 0:
            self._presend_offset[sid] = max(0, min(8_000, billed_prompt - raw))
        self._log_event(
            "PRESEND_ACCURACY",
            session_id=sid,
            counted=counted,
            billed=billed_prompt,
            error_pct=round(100.0 * (counted - billed_prompt) / billed_prompt, 1),
            exact=bool(last.get("exact")),
            ceiling=int(last.get("ceiling") or 0),
            window=int(last.get("window") or 0),
        )

    def _reset_turn_budget_state(self, sid: str, session: Any | None = None) -> None:
        """Clear per-turn budget state at the start of a new turn.

        Clear turn-scoped handoff state before a new turn. The cost handoff
        empties the normal tool surface once the session-tree budget is spent;
        retaining that flag on a resumed turn would make a later authorized
        budget top-up appear to be a permission failure. Keyed by
        ``session.hash`` to match every read site.
        """
        self._max_steps_handoff.pop(sid, None)
        self._budget_handoff.pop(sid, None)
        self._turn_cost_pico[sid] = 0
        baseline = None
        if session is not None:
            try:
                from sharedagents_goal.goal_service import session_root_id
                from sharedagents_model.usage_ledger import get_usage_ledger

                root = session_root_id(session)
                baseline = int(get_usage_ledger().session_cost_pico(root) if root else 0)
            except Exception:  # noqa: BLE001
                baseline = 0
        self._prompt_cost_baseline_pico[sid] = baseline

    @staticmethod
    def _budget_exhausted(*, spent_pico: int, budget_pico: int) -> bool:
        """True when the session-tree cap has been reached. 0 floors to default."""
        from sharedagents_goal.goal_service import DEFAULT_BUDGET_USD
        from sharedagents_model.usage_pricing import usd_to_pico

        if budget_pico <= 0:
            budget_pico = int(usd_to_pico(DEFAULT_BUDGET_USD))
        return spent_pico >= budget_pico

    def _budget_handoff_message(self, session: Any, sid: str) -> str | None:
        """Return the budget-handoff message when THIS turn's accrued cost has
        reached the session-tree cap, else ``None``.

        One cap, goal or not: ledger SUM on the cost root + this turn's local
        accumulator vs the ROOT tree cap (0/unset → $25). Best-effort:
        never raises into the round loop.
        """
        try:
            from sharedagents_goal.goal_service import session_root_id, tree_cap_usd
            from sharedagents_model.usage_ledger import get_usage_ledger
            from sharedagents_model.usage_pricing import pico_to_usd, usd_to_pico

            in_turn = int(self._turn_cost_pico.get(sid, 0))
            budget_pico = usd_to_pico(tree_cap_usd(session))
            root = session_root_id(session)
            ledger_spent = int(get_usage_ledger().session_cost_pico(root) if root else 0)
            # RoundComplete checkpoints persist the same local accumulator
            # before this handoff runs. Count only the not-yet-recorded delta;
            # otherwise a multi-round native turn appears to spend twice its
            # actual cost and can hit a $25 cap at $12.50.
            baseline_map = getattr(self, "_prompt_cost_baseline_pico", None)
            baseline = baseline_map.get(sid) if isinstance(baseline_map, dict) else None
            if baseline is None:
                # A test/fake engine may not initialize the per-turn map. In
                # that case, treating the current ledger as the baseline is
                # the least surprising fallback.
                baseline = ledger_spent
            recorded_delta = max(0, ledger_spent - int(baseline))
            unrecorded_in_turn = max(0, in_turn - recorded_delta)
            spent_pico = ledger_spent + unrecorded_in_turn
            lift = (
                "Ask the user to authorize more, then call the `budget` tool with `action='topup'` (or `/budget add`)."
            )
            if not self._budget_exhausted(spent_pico=spent_pico, budget_pico=budget_pico):
                return None
            return (
                f"[sharedagents budget] session cost budget reached "
                f"(${pico_to_usd(spent_pico):.4f} spent / "
                f"${pico_to_usd(budget_pico):.2f} budget). This is a COST limit, "
                f"not a tool error — tools are disabled for this final round so "
                f"you can wrap up. " + BUDGET_HANDOFF_USER + " " + lift
            )
        except Exception:  # noqa: BLE001
            return None

    # -----------------------------------------------------------------
    # Per-round metrics, logging, kv-cache instrumentation
    # -----------------------------------------------------------------

    def _log_round(
        self,
        session_id: str,
        turn_id: str,
        round_idx: int,
        messages: list[dict[str, Any]],
        tools_payload: list[dict[str, Any]],
    ) -> None:
        """KV-cache instrumentation: hash the prefix + tools.

        Reference: acp_bridge.py L2028–2062. Logs a structured
        ``ROUND_PREFIX`` event with the first-1024-char hash of the
        message list and a separate hash of the tools schema list, so
        an operator can tell at a glance whether two consecutive rounds
        shared a cached prefix (vLLM APC / SGLang RadixAttention KV
        reuse signal).
        """
        try:
            from sharedagents_agent.turn_runtime import (
                _hash_prefix,
                _hash_tools_payload,
            )

            prefix_hash = _hash_prefix(messages)
            tools_hash = _hash_tools_payload(tools_payload)
        except Exception:  # noqa: BLE001
            import hashlib
            import json

            try:
                blob = json.dumps(messages[:8], sort_keys=True)[:4096]
            except (TypeError, ValueError):
                blob = str(messages[:8])[:4096]
            prefix_hash = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
            try:
                tblob = json.dumps(tools_payload, sort_keys=True)
            except (TypeError, ValueError):
                tblob = str(tools_payload)
            tools_hash = hashlib.sha256(tblob.encode("utf-8")).hexdigest()[:16]

        self._log_event(
            "ROUND_PREFIX",
            session_id=session_id[:12],
            turn_id=turn_id[:12],
            round=round_idx,
            prefix=prefix_hash,
            tools=tools_hash,
        )

    def _record_round_metrics(
        self,
        session: Any,
        round_idx: int,
        metrics: dict[str, Any],
        endpoint: str,
    ) -> None:
        """Capture worker ``metrics`` payload onto the round record.

        Reference: acp_bridge.py L3200–3291 (the ``elif kind == "metrics"``
        block). Augments the worker payload with:

        * ``context_window`` from the PER-SESSION ``session.context_window``
          (the model this session is actually bound to), falling back to
          the global ``cfg.upstream_max_context`` only when the session
          carries none.
        * ``model_alias`` resolved via ``cfg.model_row_for_endpoint``.
        * Persistent ``TURN_METRICS …`` line written to
          ``cfg.logfile`` in append mode (this is gateway.log).
        """
        if not isinstance(metrics, dict):
            return
        enriched = dict(metrics)
        # Per-session context window first: ``cfg.upstream_max_context``
        # is GLOBAL and a model switch in any other session (or the
        # heartbeat) mutates it, so reading it here would stamp the
        # WRONG context window onto this session's round metrics — which
        # is exactly what the TUI footer renders as ``ctx used/max``.
        # ``session.context_window`` is set from the bound model's
        # ``contextWindow`` on every /model switch, so it always tracks
        # the model this round actually ran on.
        ctx = int(getattr(session, "context_window", 0) or getattr(self.cfg, "upstream_max_context", 0) or 0)
        if ctx:
            enriched.setdefault("context_window", ctx)
        if endpoint and not enriched.get("endpoint"):
            enriched["endpoint"] = endpoint
        try:
            row = self._resolve_model_row(
                enriched.get("model") or "",
                endpoint=enriched.get("endpoint") or endpoint or "",
            )
            if row is not None:
                _alias = row.get("alias") or row.get("name") or ""
                # Force (not setdefault): a blank/placeholder model_alias must
                # be replaced by the resolved bucket so the stats line renders
                # ``local/qwen3.6-27b-fp8`` rather than the bare wire id.
                if _alias:
                    enriched["model_alias"] = _alias
                else:
                    enriched.setdefault("model_alias", "")
                # Same row ``_build_worker_request`` selected. If the worker
                # omitted ``provider`` (older child), copy it here so the
                # ledger records the catalog choice, not a model-name guess.
                _row_provider = str(row.get("provider") or "").strip()
                if _row_provider and not str(enriched.get("provider") or "").strip():
                    enriched["provider"] = _row_provider
        except Exception:  # noqa: BLE001
            pass

        # Attach to the session's per-round metrics array so the
        # session log picks it up at turn end.
        try:
            rounds = getattr(session, "round_metrics", None)
            if not isinstance(rounds, list):
                rounds = []
                session.round_metrics = rounds
            rounds.append({"round_index": round_idx, **enriched})
        except Exception:  # noqa: BLE001
            pass

        # Stamp the latest AUTHORITATIVE prompt_tokens onto a PERSISTED session
        # field so the compaction gate can still size off the real count after a
        # reload (``round_metrics`` is in-memory only and comes back empty on a
        # fresh load — see ``_last_authoritative_prompt_tokens`` /
        # ``Session.last_prompt_tokens``). ``record_turn`` no longer calls
        # ``session.save()``, and ``_session_header`` is only emitted at
        # ``session_started`` — so without an explicit save here the column
        # would stay 0 on disk and reload back as 0. Persist via
        # ``session.save()`` (a small header_update upsert; ``to_dict`` now
        # carries the field) only when the value actually changed, so a
        # mid-turn LRU eviction / restart still reloads a real count.
        try:
            _pt = int(enriched.get("prompt_tokens") or 0)
            self._record_presend_accuracy(session, _pt)
            if _pt > 0 and _pt != int(getattr(session, "last_prompt_tokens", 0) or 0):
                session.last_prompt_tokens = _pt
                # Persist to the CONFIGURED sessions dir (honors
                # SHAREDAGENTS_SESSIONS_DIR / SHAREDAGENTS_HOME), not the hardcoded
                # default — otherwise a redirected/test runtime would write the
                # count to one DB and reload it from another.
                try:
                    session.save(root=getattr(self.cfg, "sessions_dir", None))
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass

        # Dump the full enriched payload (wt_ms, ttft_ms, tps,
        # prefill_ms, …) so gateway.log keeps the observability
        # contract the pre-5e6591e bridge emitter had. The earlier
        # five-field whitelist silently truncated everything operators
        # actually grep for; test_06 regressed exactly here.
        #
        # Filter out any keys that collide with our explicit kwargs
        # (``round`` / ``endpoint``) before splatting — a worker that
        # ever emits a ``round`` field would otherwise crash with
        # ``TypeError: got multiple values for keyword argument 'round'``.
        _RESERVED = {"round", "endpoint"}
        self._log_event(
            "TURN_METRICS",
            round=round_idx,
            endpoint=endpoint,
            **{k: v for k, v in enriched.items() if k not in _RESERVED},
        )

    def _log_event(self, event: str, **fields: Any) -> None:
        """Structured event log via the unified gateway JSONL writer.

        Reference: acp_bridge.py L2063 + tools.runners._log_delegate_error
        pattern. Used for: ``LOOP_DETECTED``, ``TOOL_RUNNER_*``,
        ``TOOL_ARGS_MALFORMED``, ``ROUND_PREFIX``, ``COMPACTION_*``,
        ``DISTILL_*``, ``NUDGE_*``, ``DELEGATE_*``.

        Routed through :func:`sharedagents_agent.tools._log.log_event` so the
        event lands in the SAME JSONL stream as every other gateway
        line — with origin tags (agent / service / session_id) plus
        the cross-process flock. (The previous plaintext ``EVENT {json}\\n`` writer
        corrupted the unified JSONL contract.) Never raises: the
        logfile is best-effort observability.
        """
        from sharedagents_agent.tools._log import log_event as _gw_log_event

        try:
            _gw_log_event(
                event,
                agent=str(getattr(getattr(self, "cfg", None), "name", "") or ""),
                **fields,
            )
        except Exception:  # noqa: BLE001
            try:
                print(f"{event} {fields!r}", file=sys.stderr, flush=True)
            except Exception:  # noqa: BLE001
                pass
        # Also surface via the python logger so anything that wires a
        # handler (tests, dev runs, the gateway-level JsonlLogHandler)
        # sees the event without having to tail gateway.log directly.
        parts: list[str] = [event]
        for k, v in fields.items():
            if v is None:
                continue
            sv = str(v)
            if len(sv) > 200:
                sv = sv[:197] + "..."
            parts.append(f"{k}={sv}")
        try:
            _LOG.info(" ".join(parts))
        except Exception:  # noqa: BLE001
            pass

    # -----------------------------------------------------------------
    # Upstream-error triage
    # -----------------------------------------------------------------

    def _classify_upstream_error(self, msg: str) -> str:
        """Forward to ``turn_runtime._classify_upstream_error`` (L1462).

        Returns the human-readable tag the underlying function emits:
        ``"upstream context-length overflow"`` /
        ``"upstream rate-limited"`` / ``"upstream timeout"`` /
        ``"upstream HTTP <code>"`` / ``"upstream error"``. Drives the
        reactive-compact retry decision (matched against the legacy
        short-tag spelling AND the long-form here in
        :meth:`_reactive_compact_on_overflow`), the backoff/retry
        helper, and the user-facing error message. An ``UpstreamError``
        hook can override the tag mid-stream — see :meth:`run_turn`'s
        error branch.
        """
        return _classify(msg or "")

    def _round_complete_metrics(self, sharedagents_sid: str, round_metrics: dict[str, Any] | None) -> dict[str, Any]:
        """Build the metrics dict carried on every ``RoundComplete`` event.

        ALWAYS stamps the goal tracker (active_goal + terminal + turns/budget)
        even when ``round_metrics`` is empty/None this round. Previously the
        engine emitted ``metrics=None`` when worker metrics were missing, which
        dropped the goal fields and left a stale met/blocked badge on the
        frontend until the next /status. Starting from an
        empty dict and stamping the goal fields unconditionally guarantees the
        widget stays live on the round channel."""
        rm = dict(round_metrics) if round_metrics else {}
        rm["active_goal"] = self._active_goals.get(sharedagents_sid, "") or ""
        rm.update(self._goal_progress_metrics(sharedagents_sid))
        return rm

    def _goal_progress_metrics(self, sharedagents_sid: str) -> dict[str, Any]:
        """The goal-progress snapshot stamped onto live turn/round metrics so
        the goal widget (web tracker, TUI bar) gets terminal state + turns/budget
        on the SAME live channel that already carries ``active_goal``.

        This is what makes the requirement "fed by … state_patch" real on the
        live path: every RoundComplete/TurnDone now carries the full tracker, so
        a met/blocked badge reaches the frontend the instant the goal settles —
        before the next /status. ``active_goal`` itself is stamped by the caller
        (it predates this helper); here we add the three tracker fields.

        ``goal_terminal`` / ``goal_turn_count`` are read from the persisted
        Session (the outer loop in agent.py writes them there); ``max_goal_turns``
        comes from config. Best-effort — never raises into the stream."""
        sess = self._sessions.get(sharedagents_sid)
        terminal = ""
        turn_count = 0
        goal_id = ""
        if sess is not None:
            terminal = str(getattr(sess, "goal_terminal", "") or "")
            goal_id = str(getattr(sess, "goal_id", "") or "")
            try:
                turn_count = int(getattr(sess, "goal_turn_count", 0) or 0)
            except Exception:  # noqa: BLE001
                turn_count = 0
        try:
            max_turns = int(getattr(self.cfg, "max_goal_turns", 0) or 0)
        except Exception:  # noqa: BLE001
            max_turns = 0
        return {
            "goal_id": goal_id,
            "goal_terminal": terminal,
            "goal_turn_count": turn_count,
            # 0 == NO turn cap (autonomous system — the goal COST budget is the
            # only limiter, not a turn count). Emit None so the UI renders ∞,
            # matching command_runtime._goal_state. (Was coerced to 100, a
            # human-in-the-loop default that misrepresented the unbounded loop.)
            "max_goal_turns": max_turns if max_turns > 0 else None,
        }
