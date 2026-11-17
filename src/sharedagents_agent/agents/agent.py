"""``Agent`` — the composition root.

One concrete class. Owns the agent's identity (name, home dir, context
files, sessions dir, permission mode, hooks) and a single
:class:`Engine` instance. Replaces the implicit "agent = ``AgentConfig``
+ the runtime around it" pattern.

The :class:`Agent` is what:

* The gateway mounts at ``/acp/<name>`` (one Agent per discovered
  ``~/.sharedagents/agents/<name>/`` dir).
* ``/agent <name>`` in TUI / Telegram swaps to (today this is a
  sharedagents-agent-only switch; in the new model it accepts CLI-engine
  agents like ``dashboard_dev`` too).
* The ``delegate_agent`` tool targets when the model wants a sibling
  agent to handle one task — same Agent class, no special "delegate"
  type.

The Agent's responsibilities (orthogonal to engine choice):

1. **System-prompt composition** from on-disk files (SOUL/AGENTS/TOOLS/
   USER/MEMORY/MODE-*.md) — exactly the way
   ``turn_runtime._compose_system_prompt`` does today.
2. **Session lifecycle** under ``<home>/sessions/``.
3. **First-call vs follow-up routing** to the engine: send the
   composed system prompt on the first turn, ``None`` after.
4. **Hook engine** dispatch — pre/post tool, post turn. Sourced from
   ``cfg.hooks_path``; same shape as today's ``HookEngine``.
5. **Memory provider** lifecycle — fact extraction post-turn, recall
   prefetch pre-turn (recall_mode=context/hybrid).
6. **Heartbeat scheduler** — periodic background turns through
   ``engine.run_turn`` against a dedicated heartbeat session.
7. **Pre-compaction memory flush** — when the local-LLM engine is
   about to compact, the Agent runs a silent turn that lets the model
   save context to MEMORY.md first. (CLI engines have their own
   resume so this is a no-op for them.)
8. **Skills auto-invoke** preamble selection (the actual injection
   happens engine-side).
9. **Uniform session-log writing** via :mod:`session_log` so every
   engine produces identically-shaped session JSON for fine-tuning.
10. **Translate Engine events to ACP** (the gateway's wire format).
    A thin shim lives at the Agent level so subclasses don't need to
    know about ACP.

The Agent does NOT know how a turn is executed. That's the engine's
job.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sharedagents_agent.hooks import HookDispatchMixin
from sharedagents_model.sampling import apply_sampling, configured_sampling, sampling_row

from .engine import AgentTurnEvent, Engine, EngineKind

# Engine-event / metric translation helpers live in event_translation.py;
# session→legacy-dict rendering helpers live in session_log.py. Re-exported
# here so ``from sharedagents_agent.agents.agent import _fold_engine_metrics``
# (and _cost_failure_should_surface / agent_session_to_dict / _loads_list)
# still resolve for legacy importers after the split.
from .event_translation import (  # noqa: F401
    _cost_failure_should_surface,
    _fold_engine_metrics,
)
from .followup_queue import FollowupQueueMixin
from .heartbeat import HeartbeatMixin
from .memory_integration import MemoryIntegrationMixin
from .session_lifecycle import SessionLifecycleMixin
from .session_log import (  # noqa: F401
    _loads_list,
    agent_session_to_dict,
)
from .system_prompt import SystemPromptMixin
from .turn_execution import TurnExecutionMixin

if TYPE_CHECKING:
    from sharedagents_agent.agents.engine import PermissionRequested

_LOG = logging.getLogger("sharedagents.agent")

# Cost admission is intentionally different for operator prompts and runtime
# work.  The session-tree cap is the safety stop for autonomous continuations;
# a human must still be able to ask a short status/follow-up question after
# that stop.  Callers that launch runtime work use one of these explicit
# sources so a child, heartbeat, or queued drain cannot bypass the gate by
# looking like a normal user turn.
_AUTOMATIC_BUDGET_SOURCES = frozenset(
    {
        "background_task",
        "delegate",
        "heartbeat",
        "loop_supervisor",
        "memory_flush",
        "queue_drain",
        "teacher",
        "goal_supervisor",
    }
)


def _is_automatic_budget_source(source: str) -> bool:
    """Whether ``source`` is runtime work that must stop at the cap."""
    value = str(source or "user").strip().lower()
    if value in _AUTOMATIC_BUDGET_SOURCES:
        return True
    return value.startswith(
        (
            "background:",
            "delegate:",
            "goal:",
            "loop:",
            "memory:",
            "queue:",
            "teacher:",
            "subagent:",
        )
    )


# --- Bounded autonomous turn retry ----------------------
# A turn that ends in a RETRYABLE error (any upstream/transient failure
# that is not a deterministic local refusal) is re-run automatically
# after a fixed interval, up to a hard retry cap. Default policy is a
# denylist: unknown and upstream classifications retry so the operator
# does not have to send continue on transient errors. The cap is the
# hard-iteration bound: at most ``1 + max_retries``
# attempts per logical turn, never unbounded. Defaults are 5 retries
# at 5 s — short enough to recover, short enough to Stop.
_DEFAULT_TURN_RETRY_MAX = 5
_DEFAULT_TURN_RETRY_INTERVAL_S = 5.0
# Deterministic / operator-intent failures — retrying the same turn
# cannot recover these. ``cli_timeout`` is handled inside the CLI
# engine as a continue-nudge; retrying the original owner prompt
# after that wall is exhausted just reprints the same red card.
# Everything else (auth, cli_error, rate-limit, spawn_failed,
# unclassified upstream, …) is retried by default.
_DEFAULT_TURN_RETRY_BLOCKED = frozenset(
    {
        "context_overflow",
        "upstream context-length overflow",
        "cost_budget_exhausted",
        "cost_budget_unavailable",
        "budget_no_wrap_up",
        "round_cap_no_wrap_up",
        "secret_egress_blocked",
        "hook_blocked",
        "bad_request",
        "invalid_params",
        "model_not_allowed",
        "needs_memory_flush",
        "turn_interrupted",
        "user_abort",
        "aborted",
        "cancelled",
        "cli_not_available",
        "cli_timeout",
        "quota_exhausted",
        "config",
        "not_yet_implemented",
        "agent_exception",
    }
)
# HTTP 4xx client errors that `_classify_upstream_error` may still emit
# as ``upstream HTTP NNN`` (legacy / unclassified bodies). 408 is a
# timeout and 429 is a rate-limit — those stay retryable.
_CLIENT_HTTP_RE = re.compile(
    r"(?:upstream\s+)?HTTP\s+(400|401|402|403|404|422)\b",
    re.IGNORECASE,
)
_CLIENT_ERROR_MARKERS = (
    "invalid reasoning effort",
    "invalid-argument",
    "invalid argument",
)


def _turn_retry_max() -> int:
    """Max automatic retries after the first failed attempt (env-tunable)."""
    try:
        return max(
            0,
            int(os.environ.get("SHAREDAGENTS_TURN_RETRY_MAX", str(_DEFAULT_TURN_RETRY_MAX))),
        )
    except (TypeError, ValueError):
        return _DEFAULT_TURN_RETRY_MAX


def _turn_retry_interval_s() -> float:
    """Seconds to wait between a failed attempt and its retry (env-tunable)."""
    try:
        return max(
            0.0,
            float(os.environ.get("SHAREDAGENTS_TURN_RETRY_INTERVAL_S", str(_DEFAULT_TURN_RETRY_INTERVAL_S))),
        )
    except (TypeError, ValueError):
        return _DEFAULT_TURN_RETRY_INTERVAL_S


def _retry_budget_allows(session: Any, admitted_over_budget: bool) -> bool:
    """Whether an automatic retry may spend on this session tree now.

    A retry is automatic work, so it needs budget left when it would start;
    a turn admitted only as a bounded over-budget manual answer never retries.
    Admission runs once per logical turn, which let a 2026-09-23 burst keep
    retrying after the $25 cap was spent. Fails closed like admission.
    """
    if admitted_over_budget:
        return False
    try:
        from sharedagents_goal.goal_service import remaining_session_budget_usd

        return float(remaining_session_budget_usd(session)) > 0
    except Exception:  # noqa: BLE001
        return False


def _retry_budget_stop_notice(session: Any, classification: str) -> str:
    """Operator-visible reason a retryable failure is not retried."""
    spent = ""
    try:
        from sharedagents_goal.goal_service import session_budget_snapshot

        snap = session_budget_snapshot(session)
        spent = f" (${snap['spent_usd']:.2f} of ${snap['budget_usd']:.2f} spent)"
    except Exception:  # noqa: BLE001
        pass
    return (
        f"[sharedagents] turn failed ({classification or 'error'}); not retrying: "
        f"the session cost budget is exhausted{spent}. Top up the session "
        "budget to continue."
    )


def _parse_class_csv(raw: str) -> frozenset[str]:
    return frozenset(c.strip() for c in (raw or "").split(",") if c.strip())


def _is_non_retryable_client_error(classification: str) -> bool:
    """True for deterministic HTTP 4xx / invalid-argument refusals."""
    text = classification or ""
    if _CLIENT_HTTP_RE.search(text):
        return True
    low = text.lower()
    return any(marker in low for marker in _CLIENT_ERROR_MARKERS)


def _user_stop_recorded(sessions_dir: Any, sid: str) -> bool:
    """Whether a user Stop was recorded on ``sid`` since its last turn start.

    An engine killed by a signal (``<cli> exited -9``) is either a user Stop
    reaching it from another process or an external kill such as the OOM
    killer. Only the Stop leaves an ``operator_stop`` record on the session
    log (written before the kill), so it — not the error text — decides:
    a stopped engine is not re-spawned, an OOM-killed one is retried.
    """
    try:
        from sharedagents_session.operator_stop import operator_stopped

        return operator_stopped(sessions_dir, sid)
    except Exception:  # noqa: BLE001 — unreadable log: retry as before
        return False


def _turn_error_is_retryable(classification: str) -> bool:
    """Whether a TurnError classification should start another attempt.

    Default: denylist — retry everything except deterministic local
    refusals (overflow, budget, secret egress, hook block, bad request,
    model-not-allowed, abort-adjacent, config, unimplemented, agent
    crash) and permanent client HTTP errors (400/401/403/404/422,
    invalid-argument / invalid reasoning effort). Empty / unknown
    classifications retry: those are typically unclassified upstream
    failures, the exact case that used to force the operator to send
    continue.

    Operator overrides:
    * ``SHAREDAGENTS_TURN_RETRY_CLASSES`` set (even empty) → allowlist mode.
    * ``SHAREDAGENTS_TURN_RETRY_BLOCK_CLASSES`` → extra denylist entries
      (ignored in allowlist mode).
    """
    from .retry_errors import permanent_error_classification

    if permanent_error_classification(classification):
        return False
    if "SHAREDAGENTS_TURN_RETRY_CLASSES" in os.environ:
        return classification in _parse_class_csv(
            os.environ.get("SHAREDAGENTS_TURN_RETRY_CLASSES", ""),
        )
    blocked = set(_DEFAULT_TURN_RETRY_BLOCKED)
    blocked.update(
        _parse_class_csv(os.environ.get("SHAREDAGENTS_TURN_RETRY_BLOCK_CLASSES", "")),
    )
    if classification in blocked:
        return False
    return not _is_non_retryable_client_error(classification)


#: Sources whose prompt is text somebody ALREADY authored — a finished
#: child's wrap-up, or follow-ups queued while the parent was busy — being
#: delivered to a now-idle parent. Delivery is not new autonomous work, so
#: the cap admits it through the bounded manual path (one round, empty tool
#: surface) instead of refusing it. Everything that would START work
#: (heartbeat, goal_supervisor, loop_supervisor, memory_flush, teacher, a
#: new delegate spawn) stays refused.
_RELAY_BUDGET_SOURCES = frozenset(
    {
        # control_endpoint drives the parent directly with a completed job's
        # payload once the parent goes idle.
        "background_task",
        # turn_execution's turn-end auto-drain: the SAME payload when the
        # parent was mid-turn as the child finished (the gateway re-queues it
        # with its background provenance), joined with any operator follow-up
        # typed while the agent was busy. ``_auto_drain_text`` has already
        # removed those items from the queue, so a refusal here drops them
        # with no open turn to record them.
        "queue_drain",
    }
)


def _is_relay_budget_source(source: str) -> bool:
    """Delivery of already-authored text to an idle parent.

    The incident: plan wrapped up at $25 and jarvis never presented it,
    because admission classified the delivery as autonomous work and
    refused it BEFORE ``turn_started`` — so the error had no open turn to
    land in and the summary vanished. Delivering paid-for text is not
    "starting another automatic turn".
    """
    return str(source or "").strip().lower() in _RELAY_BUDGET_SOURCES


# Token budget for the hidden title/summary side calls.
#
# These MUST leave room for a reasoning model to think first. The side caller
# asks for no thinking via ``chat_template_kwargs={"enable_thinking": False}``,
# but that is a **vLLM** parameter — a hosted API (DeepSeek, OpenRouter, …)
# ignores it and reasons anyway. At 64 tokens `deepseek-v4-flash` spent the
# whole budget on ~300 reasoning tokens and returned `finish_reason="length"`
# with EMPTY content, which `clean_title` turns into "" → a silent no-op that
# left every session on its placeholder title. Summaries were unaffected only
# because they already asked for 256.
#
# Measured on deepseek-v4-flash with the REAL phase-2 title prompt (summary +
# both user turns), not a toy one: 64 → empty, 512 → still empty
# (finish="length", ~1400 reasoning chars), 1024 → finish="stop" with a title.
# The caller also sends thinking={"type":"disabled"}, which alone is enough on
# DeepSeek; the budget is the backstop for any provider that honours neither
# switch. Harmless for non-reasoning backends — they stop at EOS long before.
SIDE_LLM_TITLE_MAX_TOKENS = 1024
# Task-completion recap side call (Tasks board). Larger than the title call:
# it emits a JSON object (recap + verdict + follow-up) rather than one short
# line, and the input digest is larger. Same default-deployment resolution as
# the title agent; the budget bounds the reply, the digest builder bounds the
# input.
SIDE_LLM_SUMMARY_MAX_TOKENS = 2048


async def _aclose_quietly(agen: Any) -> None:
    """Close a turn attempt's generator in the caller's context (no-op when
    absent or finished); never raises."""
    aclose = getattr(agen, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except (Exception, asyncio.CancelledError):  # noqa: BLE001
        pass


@dataclass(slots=True)
class Agent(
    SystemPromptMixin,
    SessionLifecycleMixin,
    TurnExecutionMixin,
    MemoryIntegrationMixin,
    HeartbeatMixin,
    FollowupQueueMixin,
    HookDispatchMixin,
):
    """Identity + engine for one sharedagents agent.

    Built by :func:`factory.load_agent`. The ``cfg`` is the existing
    ``AgentConfig`` (``sharedagents_agent.agent.AgentConfig``) extended with
    one new field, ``engine: str``; ``factory.load_agent`` reads that
    field to pick which :class:`Engine` subclass to instantiate.
    """

    cfg: Any  # sharedagents_agent.agent.AgentConfig
    engine: Engine
    # Optional pre-resolved sibling registry (other agents we know
    # about). Today the bridge derives this lazily from agents_root();
    # making it explicit on the Agent dataclass lets tests inject it.
    siblings: list[str] = field(default_factory=list)
    # Cached, lazily-loaded ``HookEngine`` instance. Populated on
    # first hook fire; reused across the agent's lifetime so a
    # hot path doesn't pay file-IO every turn. ``None`` means
    # uninitialized; an empty :class:`HookEngine` means we've
    # loaded and there are no hooks.
    _hook_engine: Any = None
    # ``CLAUDE_PLUGIN_ROOT`` exported to hook commands — derived from
    # the hooks file location by :meth:`_load_hook_engine`.
    _hook_plugin_root: str = ""
    # Cached embedding-backed skill auto-invoke index — lazily built on
    # the first turn with auto-invoke enabled and reused so we don't
    # re-embed the skill corpus every round. ``None`` = uninitialized.
    # MUST be a declared field: ``Agent`` is ``slots=True``, so an
    # undeclared ``self._skill_index = …`` would raise ``AttributeError``
    # (which the auto-invoke's broad ``except`` would silently swallow,
    # leaving the feature dead).
    _skill_index: Any = None
    # Active heartbeat scheduler (when :meth:`start_heartbeat` is
    # running). Single instance per Agent — re-entrant start is a
    # no-op.
    _heartbeat_scheduler: Any = None
    # Optional ``sid → outbound channel`` resolver installed by the
    # backend layer (CoreBackend / AgentACPBackend). When set,
    # :meth:`resolve_permission` calls it to find the AcpServer that
    # owns the session so ``mode=ask`` permission prompts reach the
    # connection that actually opened the session instead of silently
    # auto-denying. ``None`` for stand-alone Agent callers (tests,
    # delegate-shim child turns) — those paths can't surface a
    # permission prompt anyway.
    _outbound_resolver: Any = None
    # Goal outer-loop supervisor (lazy-init via :meth:`goal_supervisor`).
    _goal_supervisor: Any = None
    _goals_recovery_scheduled: bool = False
    # Loop supervisor (lazy-init via :meth:`loop_supervisor`) — drives
    # scheduled /loop turns on this agent's sessions.
    _loop_supervisor: Any = None
    _loops_recovery_scheduled: bool = False
    # Availability is probed lazily at the first attempted turn. A successful
    # probe is stable for this Agent instance; a failed probe is retried on the
    # next attempt so installing the missing CLI does not require a restart.
    _engine_availability_confirmed: bool = False
    # Session ids currently running the silent MEMORY.md flush turn.
    # Prevents the flush turn's own completion hook from scheduling a
    # recursive flush before the last-flush boundary is persisted.
    _memory_flush_inflight: set[str] = field(default_factory=set)
    # In-memory mirror of the durable follow-up queue per session.
    # Keys are session ids; values are OrderedDicts of queue_id → item dict.
    # Populated lazily by ``_queue_load``; mutated in-place by queue methods.
    # MUST be a declared field: Agent is slots=True, so undeclared attrs fail.
    _queue_mirror: dict[str, Any] = field(default_factory=dict)
    # Per-session "allow, don't ask again" allowlists.
    # sid hash → shared set of ``tool:<name>`` / ``bash-prefix:<cmd>`` grants,
    # also persisted to ``{sessions_dir}/{hash}.allowlist.json``. Owned by the
    # long-lived Agent (not the Session snapshot, which a reload would drop) so
    # a remembered ``allow_always`` survives across the per-call PermissionPolicy
    # instances and a slotted Agent can actually retain it.
    # MUST be a declared field: Agent is slots=True, so an undeclared
    # ``self._perm_allowlists = …`` would raise AttributeError (swallowed by the
    # broad except in resolve_permission), leaving "don't ask again" dead.
    _perm_allowlists: dict[str, set[str]] = field(default_factory=dict)
    # Normalized-command hash → local validator verdict. The validator is a
    # short model call, so identical plan-mode probes should not repeatedly
    # consume inference capacity during one Agent lifetime.
    _command_validation_cache: dict[str, tuple[Any, ...]] = field(
        default_factory=dict,
    )
    # Per-session DEBOUNCED memory-flush timers (sid → pending Task). Each
    # completed turn resets this session's timer to ``memory_flush_idle_seconds``
    # from now; the flush only fires once the session has been idle that long
    # (i.e. the session has effectively ENDED). Combined with the persisted
    # monotonic ``last_memory_flush_at`` boundary (stamp-before-run; only turns
    # added after it are ever flushed) this makes the historical compaction-
    # armed flush LOOP impossible by construction, while still allowing a later
    # flush of NEW turns if the client keeps talking in the same session.
    _memory_flush_timers: dict[str, Any] = field(default_factory=dict)
    # Per-session composed-system-prompt cache
    # (sid -> (compose_key, prompt)). Built once per session and
    # reused every turn; recomposed only when the compose key changes.
    # MUST be declared: Agent is slots=True, so undeclared attrs fail.
    _system_prompt_cache: dict[str, Any] = field(default_factory=dict)
    # Detached title calls are serialized per session so a slow turn-1 call
    # cannot race the turn-10 refresh or persist an older title afterwards.
    _title_tasks: dict[str, Any] = field(default_factory=dict)
    # Retry-burst cancel + liveness. The outer
    # ``run_turn`` loop sleeps between attempts AFTER the inner turn
    # (and its session lock) have already finished. Without these, Stop
    # aborts a stale turn_id and no-ops, /model stays conversation_busy,
    # and a loop supervisor can see the session as idle and fire again.
    # MUST be declared: Agent is slots=True.
    _retry_cancel: dict[str, Any] = field(default_factory=dict)
    _retry_burst_sids: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Give engines that run live-only tools their owning Agent."""
        bind = getattr(self.engine, "bind_agent", None)
        if callable(bind):
            bind(self)

    @property
    def goal_supervisor(self) -> Any:
        if self._goal_supervisor is None:
            from sharedagents_goal.goal_supervisor import GoalSupervisor

            self._goal_supervisor = GoalSupervisor(self)
        return self._goal_supervisor

    @property
    def loop_supervisor(self) -> Any:
        if self._loop_supervisor is None:
            from sharedagents_loop.loop_supervisor import LoopSupervisor

            self._loop_supervisor = LoopSupervisor(self)
        return self._loop_supervisor

    # ---- identity passthrough --------------------------------------

    @property
    def name(self) -> str:
        """Agent name. Same as today's ``cfg.name``."""
        return self.cfg.name

    @property
    def home(self) -> Path:
        """``~/.sharedagents/agents/<name>/`` — same as today's ``cfg.home``."""
        return self.cfg.home

    @property
    def sessions_dir(self) -> Path:
        """``<home>/sessions/`` — same as today's ``cfg.sessions_dir``."""
        return self.cfg.sessions_dir

    @property
    def engine_kind(self) -> EngineKind:
        """The engine's ``kind``. For status displays and routing."""
        return self.engine.kind

    @property
    def permission_mode(self) -> str:
        """Static cfg permission mode — fallback when no session is in
        scope. Per-session ``/mode`` overrides land on
        ``Session.system_mode`` and are picked up by
        :meth:`_effective_mode` below.
        """
        return getattr(self.cfg, "permission_mode", "auto") or "auto"

    def _effective_mode(self, session: Any) -> str:
        """Resolve the permission mode for THIS session.

        Regression guard for the 2026-05-07 bug ("``/mode plan`` was
        purely cosmetic"): :meth:`resolve_permission` previously read
        ``self.permission_mode`` only, ignoring
        ``Session.system_mode`` that the frontend's ``set_mode`` call
        wrote. The bridge stayed in AUTO regardless of what the TUI
        footer said.

        Resolution order:
          1. ``session.system_mode`` if it parses as ``{plan,ask,auto}``
          2. ``cfg.permission_mode`` if set
          3. ``"auto"`` (a corrupted on-disk system_mode never elevates
             the bridge to a weaker mode by accident — falls back to
             auto, never to ``cfg.permission_mode`` here, because a
             bogus value should be loud-ish but not catastrophic).
        """
        # Single source of truth for the valid mode set — the
        # PermissionPolicy constructor raises on anything else.
        # ``VALID_PERMISSION_MODES`` is a public re-export pinned to
        # ``PermissionPolicy._VALID_MODES`` at module scope, so we
        # don't reach into a sibling class's private attribute.
        from .permissions_integration import VALID_PERMISSION_MODES

        sm = str(getattr(session, "system_mode", "") or "").strip().lower()
        if sm in VALID_PERMISSION_MODES:
            return sm
        if sm:
            # Non-empty but bogus → drop straight to auto. See test_35
            # test_effective_mode_falls_back_to_auto_when_session_mode_unrecognised.
            return "auto"
        return self.permission_mode

    # ---- system-prompt composition ---------------------------------
    #
    # ``compose_system_prompt`` / ``compose_round_preamble`` /
    # ``compose_round_preamble_async`` / ``_append_skill_auto_invoke`` live
    # in :class:`SystemPromptMixin` (``agents/system_prompt.py``), mixed
    # into ``Agent`` below.

    # ---- session CRUD ----------------------------------------------
    #
    # ``create_session`` / ``list_sessions`` / ``get_session`` /
    # ``close_session`` / ``inject_user_message`` live in
    # :class:`SessionLifecycleMixin` (``agents/session_lifecycle.py``), mixed
    # into ``Agent`` below.

    # ---- turn dispatch ---------------------------------------------

    async def run_turn(
        self,
        sid: str,
        prompt: str = "",
        *,
        prompt_parts: list[dict[str, Any]] | None = None,
        mode: str | None = None,
        cwd: str | None = None,
        model: str | None = None,
        env: dict[str, str] | None = None,
        source: str = "user",
        sampling: dict[str, Any] | None = None,
        model_source: str | None = None,
        goal_id: str | None = None,
        actors: list[dict[str, Any]] | None = None,
        access: Any = None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[AgentTurnEvent]:
        """Dispatch one turn against ``sid``.

        ``actors`` — WebUI author(s) of this prompt;
        provenance only, never model-visible.

        ``attachments`` — saved-file references sent beside ``prompt``
        (see :mod:`sharedagents_core.attachment_context`). The model receives
        their context; the log keeps the typed text and the references apart.

        ``access`` — the submitter's grants (an
        :class:`sharedagents_core.access_grants.AccessStamp`), bound to the
        session once the session lock is held; ``None`` keeps whatever is
        bound.

        Sequence:

        1. **Per-session lock** — ``async with await
           self.engine.session_lock(sid):`` so two concurrent prompts
           on the same sid don't race on engine-session creation.
        2. Load the :class:`Session` and the engine's
           :class:`EngineSessionState` for ``sid``.
        3. Compose the system prompt + preamble:

           * ``system_prompt = self.compose_system_prompt(mode=mode)``
             and ``sha = sha256(system_prompt)``.
           * ``preamble = self.compose_round_preamble(session,
             user_text=prompt_or_parts_text)``.

        4. Consult :meth:`Engine.system_prompt_policy(state)`:

           * ``"every_turn"`` (local LLM): include both system_prompt
             and preamble in :class:`TurnRequest`. Local LLM splices
             every round.
           * ``"first_turn"`` (CLI, ``priming_status in
             {never, in_progress, failed}``): include both. Engine
             primes via ``--append-system-prompt`` then sets
             ``priming_status="completed"`` + stamps the sha.
           * ``"sha_changed"`` (CLI, prompt edits detected): include
             both. Engine starts a fresh CLI session (don't reuse the
             stale one — old SOUL.md is baked into Claude's session
             state).
           * ``"never"`` (CLI resume, sha matches): pass
             ``system_prompt=None``, ``preamble=[]``. CLI resumes;
             no re-injection.

        5. Run :meth:`apply_pre_turn_hooks` — operator-defined hooks.
        6. Build :class:`TurnRequest` (carrying ``prompt`` /
           ``prompt_parts`` for multimodal,
           plus the system_prompt / preamble / mode / cwd / model
           chosen above).
        7. ``async for event in self.engine.run_turn(request, state,
           ...)``:

           * :class:`PermissionRequested` —
             INTERCEPT before yielding upward. Call
             :meth:`resolve_permission` which consults
             :class:`PermissionPolicy.evaluate` and (for ``"ask"``)
             surfaces an outbound ACP ``request_permission``. Set
             ``ev.response.set_result(decision)``. DO NOT yield this
             event to the gateway translator.
           * :class:`TurnStarted` — capture ``engine_session_id``;
             record into session metadata.
           * :class:`ToolCallStarted` / :class:`ToolCallResult` —
             accumulate into structured lists for
             :func:`session_log.record_turn`. Yield through unchanged.
           * :class:`TextDelta` / :class:`TurnError` — yield through.
           * :class:`TurnDone` — terminal. Move to step 8.

        8. On :class:`TurnDone`:

           * :meth:`apply_post_turn_hooks` — operator hooks + memory
             provider's ``observe_turn`` + fact extraction.
           * :func:`session_log.record_turn` with FULL attribution:
             engine, model, engine_session_id, tool_calls[],
             tool_results[], permission_mode, approval_decisions[],
             round_index, finish_reason, error_classification, cwd,
             cli_argv (CLI engines only), thinking_chars (if
             persistence enabled), compaction_events,
             distill_events, nudge_events, pause_resume_log.
           * ``session.save()``.
           * The engine has ALREADY written its state mid-stream
             (priming_status transitions) — no separate write here.

        9. **Pre-compaction memory flush** intercept (local-LLM only):
           if the engine yields
           :class:`TurnError(classification="needs_memory_flush")`,
           the Agent pauses, runs a silent flush turn (via
           :meth:`maybe_memory_flush`), then re-enters step 7 from
           the same engine state. CLI engines never produce this.

        Engine-kind matrix:

        * Local-LLM agents (``engine == "local_llm"``) reach this
          method through :class:`AgentACPBackend` →
          :class:`CoreBackend.prompt` → ``_prompt_via_new_backend`` —
          a 4-line proxy. The multi-round loop, compaction, loop
          detection, distill recovery, hooks, premature-turn-end
          nudges, and memory recall all live inside
          :class:`LocalLLMEngine.run_turn`; CoreBackend now only
          owns session lifecycle + cancel routing + outbound
          channel binding.
        * CLI agents (``claude_cli`` / ``codex_cli`` /
          ``gemini_cli``) flow through Agent.run_turn → Engine.run_turn
          → ``run_delegate_<cli>_async``. The CLI subprocess owns its
          own hooks / compaction / heartbeat / loop semantics; sharedagents
          contributes the composed system prompt via
          ``--append-system-prompt`` (claude) or ``<system>...</system>``
          (codex / gemini) and intercepts permission via
          :class:`PermissionRequested`.

        Permission interception consults
        :class:`permissions_integration.PermissionPolicy.evaluate`
        (auto-mode auto-allows safe tools, prompts on destructive
        ones; ASK mode round-trips through the bound outbound
        channel). Engines that yield :class:`PermissionRequested`
        set their response future and the turn resumes; the event is
        not yielded upward.
        """
        from sharedagents_session.operator_stop import TurnStopGuard

        from .engine import (
            TextDelta,
            ToolCallStarted,
            TurnDone,
            TurnError,
        )

        stop_guard = TurnStopGuard.capture(self.sessions_dir, sid, source)

        # Resolve / pre-load the session BEFORE acquiring the lock so
        # a 404 fails fast.
        session = self.get_session(sid)
        if session is None:
            yield TurnError(
                message=f"session not found: {sid!r}",
                classification="bad_request",
            )
            yield TurnDone(state="error")
            return

        # Autonomous work is the hard stop for every engine, not only
        # delegated children. Manual operator prompts remain admissible after
        # the cap so they can receive a concise answer, but they carry a
        # framework-authored budget notice and a bounded runtime restriction.
        manual_budget_notice = ""
        manual_budget_exhausted = False
        try:
            from sharedagents_goal.goal_service import remaining_session_budget_usd

            remaining = remaining_session_budget_usd(session)
        except Exception as exc:  # noqa: BLE001
            yield TurnError(
                message=f"session budget check unavailable: {exc}",
                classification="cost_budget_unavailable",
            )
            yield TurnDone(state="error")
            return
        if remaining <= 0:
            if _is_automatic_budget_source(source) and not _is_relay_budget_source(source):
                yield TurnError(
                    message=(
                        "automatic run paused: session cost budget exhausted; "
                        "top up the session with the budget tool before "
                        "starting another automatic turn"
                    ),
                    classification="cost_budget_exhausted",
                )
                yield TurnDone(state="error")
                return
            try:
                from sharedagents_goal.goal_service import manual_budget_system_prompt

                manual_budget_notice = manual_budget_system_prompt(session)
            except Exception:  # noqa: BLE001
                # A budget display failure must not turn a manual status
                # question into a silent refusal. Keep the safety instruction
                # even when the optional spend lookup is unavailable.
                manual_budget_notice = (
                    "[sharedagents session budget notice] The session cost limit "
                    "has been reached. Answer this manual prompt briefly; do "
                    "not use tools, delegate, or start a long-running task."
                )
            manual_budget_exhausted = True
        else:
            # The previous answer may end with the limit notice while the cap
            # has since been raised; say so, or the model keeps asking for
            # budget it already has.
            try:
                from sharedagents_goal.goal_service import budget_raised_system_prompt

                manual_budget_notice = budget_raised_system_prompt(session)
            except Exception:  # noqa: BLE001
                manual_budget_notice = ""

        # Runtime work is not a manual interrupt: it must NOT suppress the
        # goal's next auto-turn or consume a manual-yield token. All automatic
        # sources share the same classification used by budget admission.
        if not _is_automatic_budget_source(source):
            try:
                self.goal_supervisor.notify_manual_turn_start(sid)
            except Exception:  # noqa: BLE001
                pass
            try:
                from sharedagents_goal.goal_service import dismiss_met_goal_on_follow_up

                dismiss_met_goal_on_follow_up(session)
            except Exception:  # noqa: BLE001
                pass
            try:
                # Claude Code: a goal the runtime paused (no progress,
                # evaluator / turn errors) continues after the user's next
                # message; the supervisor picks it up at this turn's end.
                from sharedagents_goal.goal_service import resume_goal_on_user_message

                resume_goal_on_user_message(self, session)
            except Exception:  # noqa: BLE001
                pass

        # Top-level terminal-emission safety net. Without this an exception in
        # ``session_lock`` / ``compose_system_prompt`` /
        # ``compose_round_preamble`` would bubble up before any
        # ``TurnDone`` ever yields — frontend hangs in "thinking"
        # state forever. We delegate the actual work to an inner
        # async generator and wrap it here so we can emit a
        # ``TurnError`` + ``TurnDone(error)`` fallback when no
        # ``TurnDone`` was produced before an exception fired.
        #
        # ``GeneratorExit`` (consumer ``aclose``'d us) is the one
        # case we do NOT emit on — yielding from within
        # ``GeneratorExit`` raises RuntimeError. Let it propagate.
        # Bounded autonomous retry burst. A turn that
        # ends in a RETRYABLE error (any upstream/transient failure not
        # on the denylist) is re-run after a fixed interval, up to a
        # hard cap (default 5 retries => at most 6 attempts, per the
        # hard-iteration bound). Each retry is a
        # fresh ``_run_turn_inner`` invocation: the ACP engine re-spawns
        # its child process, so a dead/hung transport is replaced
        # wholesale. Denylisted classifications and user aborts break
        # out immediately. The between-attempt sleep is ``retrying``
        # and Stop / ``cancel_retry_burst`` ends it immediately.
        retry_max = _turn_retry_max()
        retry_interval_s = _turn_retry_interval_s()
        retries_used = 0
        # Whether a terminal TurnDone was actually DELIVERED to the consumer.
        # A suppressed intermediate terminal does not count: the safety net
        # below must still emit a fallback terminal if the burst dies before
        # any TurnDone reached the wire (one stopReason per logical turn).
        terminal_delivered = False
        cancel_ev = self._retry_cancel_event(sid)
        cancel_ev.clear()
        self._retry_burst_sids.add(sid)
        # Budget verdict for the current attempt, read once when it ends so the
        # inner bookkeeping and the burst decision below cannot disagree.
        attempt_budget_ok: bool | None = None

        def _attempt_retryable(classification: str) -> bool:
            nonlocal attempt_budget_ok
            if not _turn_error_is_retryable(classification):
                return False
            if attempt_budget_ok is None:
                attempt_budget_ok = _retry_budget_allows(
                    session,
                    manual_budget_exhausted,
                )
            return attempt_budget_ok

        try:
            while True:
                if cancel_ev.is_set():
                    if not terminal_delivered:
                        # Operator Stop / steer / abort is control, not a
                        # retryable failure. Do not emit TurnError.
                        yield TurnDone(state="aborted")
                        terminal_delivered = True
                    break
                attempt_budget_ok = None
                # Whether this attempt ran a tool or streamed answer text.
                # After that, replaying the operator's text makes the model
                # redo the request, so a retry continues instead.
                attempt_progress = False
                attempt_content: list[str] = []
                attempt_tool_count = 0
                attempt_terminal: TurnDone | None = None
                attempt_done = False
                done_state = ""
                attempt_error_classification = ""
                attempt_error_message = ""
                attempt_user_stopped = False
                last_error_classification = ""
                last_error_message = ""
                _attempt = None
                try:
                    inner_kwargs: dict[str, Any] = {
                        "source": source,
                        "sampling": sampling,
                        "model_source": model_source,
                        "goal_id": goal_id,
                    }
                    # Keep compatibility with lightweight test/adaptor
                    # implementations that still expose the pre-notice
                    # _run_turn_inner signature.
                    if manual_budget_exhausted or manual_budget_notice:
                        inner_kwargs.update(
                            {
                                "manual_budget_notice": manual_budget_notice,
                                "manual_budget_exhausted": manual_budget_exhausted,
                            }
                        )
                    # An intermediate retry attempt is not the end of the
                    # logical turn: it must not advance the goal no-progress
                    # backstop, check goal sentinels on its partial content, or
                    # re-drive the goal supervisor mid-burst. Only the final
                    # attempt (no retries left) carries the goal bookkeeping.
                    # The predicate is the same one the outer loop uses so
                    # bookkeeping is skipped exactly when this TurnDone will
                    # be suppressed and the attempt re-run.
                    try:
                        _inner_params = inspect.signature(
                            self._run_turn_inner,
                        ).parameters
                    except (TypeError, ValueError):
                        _inner_params = {}
                    if "error_is_retryable" in _inner_params:
                        inner_kwargs["error_is_retryable"] = _attempt_retryable
                    if "retry_intermediate_attempt" in _inner_params and retries_used < retry_max:
                        inner_kwargs["retry_intermediate_attempt"] = True
                    if "retry_attempt" in _inner_params:
                        inner_kwargs["retry_attempt"] = retries_used
                    if "stop_guard" in _inner_params:
                        inner_kwargs["stop_guard"] = TurnStopGuard(
                            stop_guard.token,
                            stop_guard.source,
                            retry=retries_used > 0,
                        )
                    if actors and "actors" in _inner_params:
                        inner_kwargs["actors"] = [dict(a) for a in actors]
                    if access is not None and "access" in _inner_params:
                        inner_kwargs["access"] = access
                    if attachments and "attachments" in _inner_params:
                        inner_kwargs["attachments"] = [dict(a) for a in attachments]
                    _attempt = self._run_turn_inner(
                        session,
                        sid,
                        prompt,
                        prompt_parts,
                        mode,
                        cwd,
                        model,
                        env,
                        **inner_kwargs,
                    )
                    async for ev in _attempt:
                        if isinstance(ev, TurnError):
                            # Only errors BEFORE the attempt's own terminal
                            # killed the attempt — a nested queue-drain turn
                            # may stream its own errors after it.
                            if not attempt_done:
                                from .retry_errors import permanent_error_classification

                                last_error_message = ev.message or ""
                                last_error_classification = (
                                    permanent_error_classification(ev.classification, last_error_message)
                                    or ev.classification
                                    or ""
                                )
                        if not attempt_done:
                            if isinstance(ev, ToolCallStarted):
                                attempt_progress = True
                                attempt_tool_count += 1
                            elif isinstance(ev, TextDelta) and ev.channel == "content":
                                attempt_content.append(ev.text)
                                if ev.text.strip():
                                    attempt_progress = True
                        if isinstance(ev, TurnDone) and not attempt_done:
                            # The attempt's own terminal — the retry
                            # decision keys off this one, never off a
                            # nested drain turn's terminal.
                            attempt_done = True
                            attempt_terminal = ev
                            done_state = ev.state or ""
                            attempt_error_classification = last_error_classification
                            attempt_error_message = last_error_message
                            attempt_user_stopped = (
                                done_state == "error"
                                and not cancel_ev.is_set()
                                and _user_stop_recorded(
                                    self.sessions_dir,
                                    sid,
                                )
                            )
                            retry_candidate = (
                                done_state == "error"
                                and not cancel_ev.is_set()
                                and not attempt_user_stopped
                                and retries_used < retry_max
                                and _turn_error_is_retryable(
                                    attempt_error_classification,
                                )
                            )
                            if retry_candidate and _attempt_retryable(
                                attempt_error_classification,
                            ):
                                # Suppress the intermediate terminal: the
                                # logical turn is not over. The consumer
                                # (ACP wire / gateway) must see exactly ONE
                                # stopReason per logical turn, and it must
                                # come from the FINAL attempt — the wire
                                # dedup would otherwise drop the final one.
                                continue
                            if retry_candidate:
                                yield TextDelta(
                                    text=_retry_budget_stop_notice(
                                        session,
                                        attempt_error_classification,
                                    ),
                                    channel="content",
                                )
                        yield ev
                        if isinstance(ev, TurnDone):
                            terminal_delivered = True
                except GeneratorExit:
                    # Abandoned (the consumer closed us): close the attempt now,
                    # in this context, so its session lock and pinned grants
                    # are released here rather than later by the GC from
                    # another task.
                    await _aclose_quietly(_attempt)
                    raise
                except Exception as exc:  # noqa: BLE001
                    await _aclose_quietly(_attempt)
                    if not terminal_delivered:
                        yield TurnError(
                            message=f"agent.run_turn aborted: {exc}",
                            classification="agent_exception",
                        )
                        yield TurnDone(state="error")
                        terminal_delivered = True
                    # An agent-level exception is not retried: the safety net
                    # above already converted it to a clean terminal.
                    break
                if not attempt_done:
                    # The inner generator returned cleanly without a terminal
                    # (engine contract violation). Emit a clean terminal rather
                    # than letting the consumer hang in "thinking".
                    yield TurnError(
                        message="turn ended without a terminal event",
                        classification="agent_exception",
                    )
                    yield TurnDone(state="error")
                    break
                if (
                    cancel_ev.is_set()
                    or done_state != "error"
                    or attempt_user_stopped
                    or not _attempt_retryable(attempt_error_classification)
                    or retries_used >= retry_max
                ):
                    if cancel_ev.is_set() and not terminal_delivered:
                        # Operator Stop / steer / abort is control, not a
                        # retryable failure. Do not emit TurnError.
                        yield TurnDone(state="aborted")
                        terminal_delivered = True
                    break
                retries_used += 1
                yield TextDelta(
                    text=(
                        f"[sharedagents] turn failed ({attempt_error_classification}): "
                        f"{attempt_error_message[:200]} — retrying in "
                        f"{retry_interval_s:g}s (retry {retries_used}/{retry_max})"
                        + (", continuing from where it stopped." if attempt_progress else ".")
                    ),
                    channel="content",
                )
                self._emit_turn_retry(
                    sid,
                    retry=retries_used,
                    retry_max=retry_max,
                    interval_s=retry_interval_s,
                    classification=attempt_error_classification,
                    message=attempt_error_message,
                )
                # Operator Stop / /abort sets ``cancel_ev`` even after the
                # inner attempt has already finished (the gateway's mapped
                # turn_id is stale during this sleep). Wait on that event
                # instead of a bare sleep so Stop ends the burst immediately
                # instead of leaving conversation_busy / turn_active locked.
                try:
                    await asyncio.wait_for(
                        cancel_ev.wait(),
                        timeout=retry_interval_s,
                    )
                except asyncio.TimeoutError:
                    pass
                else:
                    if not terminal_delivered:
                        # Operator Stop / steer / abort is control, not a
                        # retryable failure. Do not emit TurnError.
                        yield TurnDone(state="aborted")
                        terminal_delivered = True
                    break
                if cancel_ev.is_set():
                    if not terminal_delivered:
                        # Operator Stop / steer / abort is control, not a
                        # retryable failure. Do not emit TurnError.
                        yield TurnDone(state="aborted")
                        terminal_delivered = True
                    break
                # Child sessions can spend during the retry delay. The
                # memoized end-of-attempt verdict is not admission for now.
                if not _retry_budget_allows(session, manual_budget_exhausted):
                    self._clear_retrying_state(sid)
                    yield TextDelta(
                        text=_retry_budget_stop_notice(
                            session,
                            attempt_error_classification,
                        ),
                        channel="content",
                    )
                    terminal = attempt_terminal or TurnDone(state="error")
                    yield terminal
                    terminal_delivered = True
                    # The inner attempt skipped final goal bookkeeping when
                    # a retry was still affordable. This is now its terminal.
                    if not manual_budget_exhausted and source != "memory_flush":
                        final_content = "".join(attempt_content)
                        try:
                            self._maybe_goal_re_drive(
                                sid,
                                session,
                                terminal,
                                final_content,
                                source=source,
                                tool_call_count=attempt_tool_count,
                            )
                        except Exception:  # noqa: BLE001
                            pass
                    break
                # A replayed prompt must not keep the previous attempt's
                # background children alive — that is how a retry burst
                # stacked a delegation tree past the 3-level cap. Do not
                # cancel the job that DRIVES this session: that is this
                # turn's own runner.
                try:
                    from sharedagents_agent.agents import async_jobs

                    cancelled = async_jobs.cancel_subtree(
                        sid,
                        include_driver=False,
                    )
                    if cancelled:
                        async_jobs.silence_deliveries(
                            cancelled,
                            reason="cancelled (turn retry)",
                        )
                except Exception:  # noqa: BLE001
                    pass
                if attempt_progress:
                    # The failed attempt already acted; its tool calls and
                    # text are in the conversation. Send what the operator
                    # would: continue — never the original request again.
                    from sharedagents_agent.continuation import turn_retry_continue_body

                    prompt = turn_retry_continue_body(
                        attempt_error_classification,
                        attempt_error_message,
                    )
                    prompt_parts = None
        finally:
            if cancel_ev.is_set():
                self._clear_retrying_state(sid)
            self._retry_burst_sids.discard(sid)
            self._retry_cancel.pop(sid, None)

    # The inner streaming turn loop ``_run_turn_inner`` (driven by
    # ``run_turn`` above) lives in :class:`TurnExecutionMixin`
    # (``agents/turn_execution.py``), mixed into ``Agent`` below.

    async def resolve_permission(
        self,
        ev: PermissionRequested,
        session: Any,
        *,
        outbound: Any | None = None,  # ACP outbound channel
    ) -> None:
        """Decide a permission request and set ``ev.response``.

        The engine yields
        :class:`PermissionRequested` events and the Agent resolves
        them. Steps:

        1. Run :meth:`apply_pre_tool_hooks` — operator can veto.
        2. Run :class:`PermissionPolicy.evaluate(tool_name, args,
           criticality)` → ``"allow"`` / ``"deny"`` / ``"ask"``.
        3. For ``"ask"``: run :func:`permissions_integration.ask_user`
           (auto-mode auto-answers; non-auto surfaces ACP
           ``request_permission``).
        4. Populate ``ev.reason`` when denying.
        5. ``ev.response.set_result(decision)`` to unblock the engine.

        Never yields. Never raises (failures map to ``"deny"`` with a
        reason).
        """
        from .permissions_integration import (
            AskQuestion,
            PermissionPolicy,
            ask_user,
            load_persisted_allowlist,
            persist_allowlist,
        )

        decision = "deny"
        reason = ""
        decision_source = "policy"

        try:
            if ev.response is None or ev.response.done():
                return

            # Step 1: PreToolUse hooks already ran inside the engine's
            # dispatch loop — it handled `deny` (skipped the call) and
            # `allow` (never yielded this event) itself. Only "" (no
            # opinion) or "ask" (force the confirmation path) reach
            # here, carried on ev.hook_decision.
            hook_perm = str(getattr(ev, "hook_decision", "") or "")

            # Step 2: policy evaluation. Use the per-session resolved
            # mode (so `/mode plan` actually flips the gate); falls
            # back to cfg.permission_mode when the session hasn't
            # been steered. The 2026-05-07 regression was caused by
            # reading self.permission_mode here directly.
            mode = self._effective_mode(session)
            try:
                # Plan-mode Python probes are opaque to the deterministic shell
                # classifier. For the narrow ``python -c`` + UNKNOWN case, run
                # one tiny history-free verdict turn on a LOCAL model using the
                # teacher call/parser machinery. Known destructive/forbidden
                # classifications never reach this downgrade path.
                if (
                    mode == "plan"
                    and (ev.criticality or "").lower() == "mutating"
                    and ev.tool_name in ("Bash", "bash", "Shell", "shell")
                ):
                    _cmd = (ev.arguments or {}).get("command")
                    if isinstance(_cmd, str) and _cmd:
                        from sharedagents_agent.tools.safety import (
                            CommandSafety,
                            classify_command,
                        )

                        from .command_safety_validator import (
                            contains_python_dash_c,
                            resolve_local_validator_model,
                            validate_python_command_read_only,
                        )

                        if classify_command(_cmd) == CommandSafety.UNKNOWN and contains_python_dash_c(_cmd):
                            import asyncio
                            import hashlib

                            _cache = getattr(
                                self,
                                "_command_validation_cache",
                                None,
                            )
                            if not isinstance(_cache, dict):
                                _cache = {}
                                self._command_validation_cache = _cache
                            _validator_row = resolve_local_validator_model(
                                self.cfg,
                            )
                            _fingerprint = "|".join(
                                (
                                    str(_validator_row.get("baseUrl") or ""),
                                    str(_validator_row.get("id") or ""),
                                )
                            )
                            _key = hashlib.sha256(
                                f"v1|{_fingerprint}|{_cmd.strip()}".encode(),
                            ).hexdigest()
                            _verdict = _cache.get(_key)
                            _cache_hit = isinstance(_verdict, tuple)
                            if not isinstance(_verdict, tuple):
                                _verdict = await asyncio.to_thread(
                                    validate_python_command_read_only,
                                    self.cfg,
                                    _cmd,
                                )
                                _cache[_key] = _verdict
                            _is_read_only, _validation_reason, _row = _verdict
                            _LOG.info(
                                "plan python validator command_hash=%s model=%s read_only=%s cache_hit=%s reason=%s",
                                _key[:12],
                                str((_row or {}).get("id") or ""),
                                bool(_is_read_only),
                                _cache_hit,
                                str(_validation_reason or "")[:300],
                            )
                            if _is_read_only:
                                ev.criticality = "read_only"

                # The "allow, don't ask again" allowlist must
                # outlive this per-call policy. Store it on the long-lived Agent
                # keyed by session hash (a reloaded ``Session`` snapshot would
                # lose an attr set on it, so the Agent — not the Session object —
                # owns it) and share the same set into every per-call policy.
                _store = getattr(self, "_perm_allowlists", None)
                if not isinstance(_store, dict):
                    _store = {}
                    try:
                        self._perm_allowlists = _store
                    except Exception:  # noqa: BLE001
                        pass
                _sess_hash = getattr(session, "hash", "") or ""
                _allow = _store.get(_sess_hash)
                if not isinstance(_allow, set):
                    _sessions_dir = None
                    try:
                        _sessions_dir = self.sessions_dir
                    except Exception:  # noqa: BLE001
                        _sessions_dir = getattr(self.cfg, "sessions_dir", None)
                    _allow = load_persisted_allowlist(_sessions_dir, _sess_hash)
                    _store[_sess_hash] = _allow
                policy = PermissionPolicy(mode=mode, allowlist=_allow)
                decision, policy_reason = policy.evaluate(
                    ev.tool_name,
                    ev.arguments or {},
                    ev.criticality or "mutating",
                )
                reason = policy_reason or reason
            except Exception:  # noqa: BLE001
                decision, reason = "deny", "policy_unavailable"
                return

            # A PreToolUse hook asking for confirmation upgrades an
            # otherwise-allowed call to the ASK path (Claude Code's
            # `permissionDecision: ask`).
            if hook_perm == "ask" and decision == "allow" and mode != "auto":
                decision = "ask"
                reason = reason or "PreToolUse hook requested confirmation"
                decision_source = "hook"

            # Step 3: surface ASK if needed.
            if decision == "ask":
                # Do NOT stamp source=user until an answer exists — a
                # missing outbound channel is ``no_channel``, not a human
                # decision (audit of 172 destructive denials all claimed
                # source=user while nobody was prompted).
                decision_source = "ask"
                # Only offer "allow always" when the call
                # is actually rememberable — a curated structured tool or a
                # Bash command with a derivable prefix, AND a non-destructive
                # criticality. Never for network/destructive or an unknown
                # tool, so the affordance never implies a suppression we won't
                # honor; destructive/irreversible calls stay on their
                # independent per-action gate.
                _crit = (ev.criticality or "mutating").lower()
                # Shell calls join the rememberable set
                # via a COMMAND-PREFIX grant ("Allow always" on
                # `git worktree list` remembers the `git worktree` family).
                # Never for destructive calls, and never for a command whose
                # prefix cannot be derived.
                _bash_prefix = ""
                if ev.tool_name in PermissionPolicy._BASH_TOOLS and _crit in ("read_only", "mutating"):
                    try:
                        _bash_prefix = PermissionPolicy.bash_prefix_for(
                            str((ev.arguments or {}).get("command", "")),
                        )
                    except Exception:  # noqa: BLE001
                        _bash_prefix = ""
                _rememberable = policy.is_rememberable(
                    ev.tool_name,
                    ev.arguments or {},
                    ev.criticality or "mutating",
                )
                _opts = ("Allow", "Allow always", "Deny") if _rememberable else ("Allow", "Deny")
                _detail = str(ev.arguments)[:400]
                if _bash_prefix:
                    _detail += f'\nAllow always covers the command prefix: "{_bash_prefix} …"'
                try:
                    answers = await ask_user(
                        [
                            AskQuestion(
                                id=f"perm:{ev.call_id}",
                                title=f"Allow {ev.tool_name}?",
                                detail=_detail,
                                options=_opts,
                                criticality=_crit,
                            )
                        ],
                        session_id=getattr(session, "hash", "") or "",
                        mode=mode,
                        outbound=outbound,
                    )
                except Exception:  # noqa: BLE001
                    answers = []
                if answers and isinstance(answers[0], dict):
                    # `ask_user` normalizes ACP option ids like
                    # `allow_once` into `decision=allow`. Prefer that
                    # normalized value so an explicit TUI approval does
                    # not get re-read as an unknown option id here.
                    chosen = answers[0].get("decision") or answers[0].get("optionId")
                    decision = (
                        "allow"
                        if str(chosen).lower()
                        in (
                            "allow",
                            "allow_once",
                            "allow_always",
                            "allow always",
                            "allow, don't ask again",
                            "yes",
                            "y",
                            "0",
                        )
                        else "deny"
                    )
                    ans_reason = str(answers[0].get("reason") or "")
                    if decision == "deny":
                        if "no outbound channel" in ans_reason.lower():
                            decision_source = "no_channel"
                            reason = ans_reason or reason or "no outbound channel"
                        else:
                            decision_source = "user"
                            reason = ans_reason or reason or "user denied"
                    else:
                        decision_source = "user"
                    # Persist an "allow always" selection so an
                    # identical later call in this session auto-allows. remember()
                    # itself re-checks the safe tool set + criticality (defense in
                    # depth); irreversible / Tier-A calls stay gated by the
                    # independent per-action gate, never this allowlist.
                    _opt = str(answers[0].get("optionId") or "").lower()
                    if (
                        decision == "allow"
                        and _rememberable
                        and _opt
                        in (
                            "allow_always",
                            "allow always",
                            "allow, don't ask again",
                        )
                    ):
                        try:
                            if policy.remember(
                                ev.tool_name,
                                ev.arguments or {},
                                ev.criticality or "mutating",
                            ):
                                _sessions_dir = None
                                try:
                                    _sessions_dir = self.sessions_dir
                                except Exception:  # noqa: BLE001
                                    _sessions_dir = getattr(
                                        self.cfg,
                                        "sessions_dir",
                                        None,
                                    )
                                persist_allowlist(
                                    _sessions_dir,
                                    _sess_hash,
                                    policy._allowlist,
                                )
                        except Exception:  # noqa: BLE001
                            pass
                else:
                    decision = "deny"
                    if outbound is None:
                        decision_source = "no_channel"
                        reason = reason or "no outbound channel attached"
                    else:
                        decision_source = "user"
                        reason = reason or "no user answer"

            if decision != "allow":
                ev.reason = reason
        finally:
            ev.decision = decision if decision in ("allow", "deny") else "deny"
            ev.decision_source = decision_source
            try:
                if ev.response is not None and not ev.response.done():
                    ev.response.set_result(
                        ev.decision,
                    )
            except Exception:  # noqa: BLE001
                pass

    def _retry_cancel_event(self, sid: str) -> asyncio.Event:
        """Per-session event that Stop / /abort sets to end a retry burst."""
        ev = self._retry_cancel.get(sid)
        if ev is None:
            ev = asyncio.Event()
            self._retry_cancel[sid] = ev
        return ev

    def cancel_retry_burst(self, sid: str) -> None:
        """End the retry-burst sleep for ``sid``.

        Called from the gateway cancel path with the session id (the
        mapped turn_id is stale between attempts). Safe when no burst
        is running.
        """
        if not sid:
            return
        ev = self._retry_cancel.get(sid)
        if ev is not None:
            ev.set()

    def _emit_turn_retry(
        self,
        sid: str,
        *,
        retry: int,
        retry_max: int,
        interval_s: float,
        classification: str = "",
        message: str = "",
    ) -> None:
        """Persist ``turn_retry`` so session_state becomes ``retrying``."""
        try:
            from sharedagents_session.session_writer import get_session_writer

            writer = get_session_writer(
                self.sessions_dir,
                sid,
                engine=self.engine.kind.value,
            )
            writer.turn_retry(
                retry=retry,
                retry_max=retry_max,
                interval_s=interval_s,
                classification=classification,
                message=message,
            )
        except Exception:  # noqa: BLE001
            pass

    def _clear_retrying_state(self, sid: str) -> None:
        """Fold ``retrying`` to a terminal so Stop cannot stick after abort."""
        try:
            from sharedagents_session.session_index import mark_session_terminal

            mark_session_terminal(self.sessions_dir, sid, state="idle")
        except Exception:  # noqa: BLE001
            pass

    def has_active_turn(self, sid: str = "") -> bool:
        """True while a logical turn — including retry sleep — is in flight.

        Loop idle-only firing and conversation_busy consult this so a
        session sleeping between retry attempts is not treated as idle.
        """
        if sid:
            return sid in self._retry_burst_sids
        return bool(self._retry_burst_sids)

    async def abort(
        self,
        turn_id: str,
        *,
        force: bool = False,
        session_id: str = "",
    ) -> None:
        """Stop the in-flight turn and any between-attempt retry sleep.

        Forwards ``force`` to ``engine.abort``. The mapped turn_id is
        stale during retry sleep, so this also cancels the session's
        retry burst when ``session_id`` is known or this agent has
        exactly one live burst.
        """
        sid = session_id or ""
        if not sid:
            sessions = getattr(self.engine, "_session_turns", None)
            if isinstance(sessions, dict):
                for mapped_sid, tids in sessions.items():
                    if turn_id in (tids or ()):
                        sid = str(mapped_sid)
                        break
        if not sid and len(self._retry_burst_sids) == 1:
            sid = next(iter(self._retry_burst_sids))
        if sid:
            self.cancel_retry_burst(sid)
        abort = getattr(self.engine, "abort", None)
        if not callable(abort):
            return
        try:
            await abort(turn_id, force=force)
        except TypeError:
            await abort(turn_id)

    # ---- hooks — Claude Code lifecycle events ----------------------
    #
    # sharedagents's hooks ARE Claude Code's hooks: the same ``hooks.json``
    # config, the same event names, the same stdin-payload / exit-code
    # / JSON-stdout protocol (see :mod:`sharedagents_agent.hooks`). The
    # methods below are the runtime's call sites — each builds the
    # per-event payload, runs the engine, and hands a
    # :class:`~sharedagents_agent.hooks.HookOutcome` back to the caller to
    # act on (deny a tool, inject context, halt the agent, warn the
    # operator).
    #
    # sharedagents's tool registry already uses Claude Code's tool names
    # verbatim (``Bash`` / ``Read`` / ``Edit`` / ``Write`` / ``Grep``
    # / ``Glob`` / ``TodoWrite`` …), so a ``tool_name`` from a tool
    # call is fed to the hook payload as-is — a matcher written for
    # Claude Code (``"matcher": "Bash"``) fires here unchanged with no
    # translation layer.

    # Hook dispatch (``_hook_kwargs`` / ``_run_hook_event`` /
    # ``_load_hook_engine``) lives in :class:`HookDispatchMixin`
    # (``sharedagents_agent.hooks``), mixed into ``Agent`` below.

    # Per-event hook wrappers — UserPromptSubmit / PreToolUse /
    # PostToolUse / Stop / SubagentStop / SessionStart / SessionEnd /
    # PreCompact / Notification — used to live here as thin
    # ``_run_hook_event(...)`` wrappers. None had callers — every
    # live hook firing goes through :meth:`LocalLLMEngine._fire_hook`
    # (which calls ``hook_engine.run(event, payload, **hook_ctx)``
    # directly). Removed.

    # ---- goal auto-re-drive (outer loop) -----------------------------------

    def _maybe_goal_re_drive(
        self,
        sid: str,
        session: Any,
        turn_done: Any,
        final_content: str,
        *,
        source: str = "user",
        tool_call_count: int = 0,
    ) -> None:
        """After TurnDone, advance goal bookkeeping and start outer loop.

        ``tool_call_count`` / ``final_content`` describe the COMPLETED turn
        (this is called once, on TurnDone). A turn that produced neither an
        assistant message nor any tool call made no real progress; such turns
        are not counted toward ``goal_turn_count``, but the goal keeps being
        driven — the supervisor slows down and alerts instead of pausing.

        The goal-own budget and the root-session cost cap are checked as
        two independent gates, each spend against its own cap; either one
        pauses the goal with ``cost_budget_exhausted``. While work this goal
        already started is still running, the pause is left to the supervisor,
        which writes it once that work stops.
        """
        active_goal = (getattr(session, "active_goal", "") or "").strip()
        goal_id = str(getattr(session, "goal_id", "") or "").strip()
        if not active_goal and not goal_id:
            return

        if goal_id:
            # Claims and owner commands already committed their transition.
            # A late finalizer must never copy an old blocked/met header back
            # over the owner's reject/resume in the authoritative registry.
            from sharedagents_goal.goal_registry import (
                get_goal_registry,
                sync_session_goal_projection,
            )

            goal_rec = get_goal_registry().get(goal_id)
            sync_session_goal_projection(session, goal_rec)
            if goal_rec is None:
                return

        goal_terminal = (getattr(session, "goal_terminal", "") or "").strip()
        if goal_terminal in ("met", "blocked", "paused"):
            if goal_id:
                try:
                    self.goal_supervisor.cancel(
                        goal_id,
                        reason=goal_terminal,
                        pause=False,
                    )
                except Exception:  # noqa: BLE001
                    pass
            return

        # --- empty turns ----------------------------------------------
        # A turn with neither an assistant message nor a tool call did no
        # work: it is not counted toward ``goal_turn_count``. The goal is NOT
        # paused for it — the supervisor keeps driving with a growing delay and
        # tells the owner once (see GoalSupervisor). Ending a turn never ends
        # a goal.
        made_progress = bool((final_content or "").strip()) or tool_call_count > 0
        if not made_progress:
            session.goal_no_progress_count = int(getattr(session, "goal_no_progress_count", 0) or 0) + 1
            try:
                session.save()
            except Exception:  # noqa: BLE001
                pass
            if goal_id:
                try:
                    self.goal_supervisor.ensure_started(goal_id, source=source)
                except Exception:  # noqa: BLE001
                    pass
            return
        # Real work happened — reset the consecutive no-progress counter.
        session.goal_no_progress_count = 0

        goal_turn_count = int(getattr(session, "goal_turn_count", 0) or 0) + 1
        session.goal_turn_count = goal_turn_count

        # COST budget governs the loop (replaces the legacy turn cap). The
        # turn's spend was already recorded into the usage ledger at turn
        # finalization; here we read the authoritative ledger SUMs and stop
        # once a cap is reached. ``turn_count`` is still tracked for display
        # only.
        goal_rec = None
        if goal_id:
            try:
                from sharedagents_goal.goal_registry import get_goal_registry

                reg = get_goal_registry()
                reg.update_turn_count(goal_id, goal_turn_count)
                goal_rec = reg.get(goal_id)
                # Two scopes, each against its own cap: the spend attributed
                # to this goal against the goal's budget, and the session
                # tree's spend against the session budget. The supervisor
                # checkpoints read the same pair.
                from sharedagents_goal.goal_service import goal_budget_state

                budget = goal_budget_state(goal_rec, session)
                reg.update_cost_cache(goal_id, budget.goal_spent_pico)
                if goal_rec is not None and budget.exhausted_scope() and not self._goal_budget_settles_after_work(sid):
                    session.goal_terminal = "paused"
                    try:
                        from sharedagents_goal.goal_registry import (
                            sync_session_goal_projection,
                        )

                        goal_rec = reg.pause_goal(
                            goal_id,
                            reason="cost_budget_exhausted",
                        )
                        sync_session_goal_projection(session, goal_rec)
                    except Exception:  # noqa: BLE001
                        pass
            except Exception:  # noqa: BLE001
                pass

        try:
            session.save()
        except Exception:  # noqa: BLE001
            pass

        if session.goal_terminal in ("met", "blocked", "paused"):
            if goal_id:
                try:
                    self.goal_supervisor.cancel(
                        goal_id,
                        reason=session.goal_terminal,
                        pause=False,
                    )
                except Exception:  # noqa: BLE001
                    pass
            return

        if goal_id:
            try:
                self.goal_supervisor.ensure_started(goal_id, source=source)
            except Exception:  # noqa: BLE001
                pass

    def _goal_budget_settles_after_work(self, sid: str) -> bool:
        """True when a reached budget limit must not pause the goal yet.

        A turn can end with children it dispatched still running. They were
        admitted before the limit and keep working whatever the goal's status
        says, so pausing now would show a stopped goal that is still spending.
        The supervisor holds every further turn while that work is live and
        pauses the goal when it stops; without a supervisor nothing would
        settle it later, so the pause is written here as before.
        """
        try:
            if not self.goal_supervisor._enabled():
                return False
            from sharedagents_goal.goal_registry import session_work_live

            return bool(session_work_live(sid)[0])
        except Exception:  # noqa: BLE001
            return False

    def _side_llm_endpoint_model(self, session: Any) -> tuple[str, str]:
        """Resolve an OpenAI-compatible endpoint+model for title/summary calls.

        Prefers OpenCode-style ``cfg.small_model_row()`` (dedicated cheap
        utility model). Falls back to the session binding / primary upstream
        only when the small-model row has no usable endpoint or model id —
        so CLI/ACP sessions still get titles when only one model is configured.
        """
        row: dict = {}
        try:
            row = dict(self.cfg.small_model_row() or {})
        except Exception:  # noqa: BLE001
            row = {}
        endpoint = (
            str(row.get("baseUrl") or "").rstrip("/")
            or getattr(session, "endpoint", "")
            or getattr(self.cfg, "upstream_endpoint", "")
            or ""
        )
        model = (
            str(row.get("id") or "").strip()
            or getattr(session, "model", "")
            or getattr(self.cfg, "upstream_model", "")
            or ""
        )
        return str(endpoint or ""), str(model or "")

    def _make_compatible_llm_caller(
        self,
        endpoint: str,
        model: str,
        headers_cfg: Any,
        *,
        temperature: float | None = None,
        purpose: str = "title",
        max_tokens: int,
        read_timeout: float = 30.0,
    ):
        """Build the shared non-streaming OpenAI-compatible side caller."""
        endpoint = str(endpoint or "").rstrip("/")
        model = str(model or "").strip()
        if not endpoint or not model:
            return None
        try:
            from sharedagents_gateway.codex_responses import is_codex_responses_endpoint

            if is_codex_responses_endpoint(endpoint):
                return None
        except Exception:  # noqa: BLE001 — never skip a usable OpenAI-compat URL
            pass
        try:
            headers = headers_cfg.upstream_headers(endpoint, model)
        except Exception:  # noqa: BLE001
            headers = {}

        async def _caller(messages: list) -> str:
            import httpx

            base = {
                "model": model,
                "stream": False,
                "messages": messages,
                "max_tokens": max_tokens,
            }
            apply_sampling(
                base,
                configured_sampling(
                    headers_cfg,
                    sampling_row(headers_cfg, endpoint, model),
                    model,
                    purpose=purpose,
                    override={"temperature": temperature} if temperature is not None else None,
                ),
            )
            # Side calls must not spend their short output budget reasoning.
            payload = dict(
                base,
                chat_template_kwargs={"enable_thinking": False},
                thinking={"type": "disabled"},
            )
            url = endpoint + "/chat/completions"
            timeout = httpx.Timeout(
                connect=10.0,
                read=float(read_timeout),
                write=10.0,
                pool=10.0,
            )
            async with httpx.AsyncClient(
                timeout=timeout,
                headers=headers,
            ) as c:
                resp = await c.post(url, json=payload)
                if resp.status_code == 400:
                    resp = await c.post(url, json=base)
                resp.raise_for_status()
                data = resp.json()
            choices = data.get("choices") or []
            if not choices:
                return ""
            choice = choices[0] or {}
            msg = choice.get("message") or {}
            content = str(msg.get("content") or "")
            if not content:
                _LOG.warning(
                    "side LLM (%s @ %s) returned EMPTY content: finish_reason=%r max_tokens=%d reasoning_chars=%d",
                    model,
                    endpoint,
                    choice.get("finish_reason"),
                    max_tokens,
                    len(str(msg.get("reasoning_content") or "")),
                )
            return content

        return _caller

    def _make_hidden_agent_caller(
        self,
        session: Any,
        profile_name: str,
        *,
        model_override: str | None = None,
    ):
        """Build the shared messages→text caller for a hidden utility agent.

        ``title`` (session-title lifecycle) and ``summary`` (Tasks-board
        completion recap) use the Jarvis default deployment unless summary
        has an explicit configured selection. The session engine (CLI, ACP, Codex-via
        local LLM) may not expose an OpenAI-compatible endpoint. The reply is
        wrapped in a view-only child session by ``_caller_visible``.
        """
        profile_name = str(profile_name or "").strip().lower()
        if profile_name in ("title", "summary"):
            try:
                from sharedagents_agent.agent import AgentConfig

                cfg = AgentConfig.load("jarvis")
            except Exception:  # noqa: BLE001
                cfg = self.cfg
            rows: list[dict] = []
            for row in list(getattr(cfg, "models", None) or []):
                if isinstance(row, dict) and str(row.get("role") or "").lower() == "primary":
                    rows.append(dict(row))
                    break
            if not rows:
                try:
                    rows.append(dict(cfg._primary_model_row() or {}))
                except Exception:  # noqa: BLE001
                    pass
            try:
                rows.append(dict(cfg.small_model_row() or {}))
            except Exception:  # noqa: BLE001
                pass
            rows.append(
                {
                    "baseUrl": str(getattr(cfg, "upstream_endpoint", "") or ""),
                    "id": str(getattr(cfg, "upstream_model", "") or ""),
                }
            )
            if profile_name == "summary" and model_override:
                from sharedagents_session.session_recap import recap_model_rows

                rows = [row for row in recap_model_rows(cfg) if row["key"] == model_override]
                if not rows:
                    raise ValueError(f"unknown or unsupported recap model: {model_override!r}")
            max_tokens, read_timeout = (
                (SIDE_LLM_TITLE_MAX_TOKENS, 90.0) if profile_name == "title" else (SIDE_LLM_SUMMARY_MAX_TOKENS, 120.0)
            )
        else:
            raise ValueError(f"unknown hidden agent profile: {profile_name!r}")

        candidates: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for row in rows:
            endpoint = str(row.get("baseUrl") or "").rstrip("/")
            model = str(row.get("id") or "").strip()
            key = (endpoint, model)
            if not endpoint or not model or key in seen:
                continue
            seen.add(key)
            candidates.append(key)
        if not candidates:
            return None

        async def _caller(messages: list) -> str:
            last_exc: Exception | None = None
            for endpoint, model in candidates:
                fn = self._make_compatible_llm_caller(
                    endpoint,
                    model,
                    cfg,
                    purpose=profile_name,
                    max_tokens=max_tokens,
                    read_timeout=read_timeout,
                )
                if fn is None:
                    continue
                try:
                    text = await fn(messages)
                except Exception as exc:  # noqa: BLE001
                    _LOG.warning(
                        "%s LLM (%s @ %s) failed: %s",
                        profile_name,
                        model,
                        endpoint,
                        exc,
                    )
                    last_exc = exc
                    continue
                if str(text or "").strip():
                    return text
                _LOG.warning(
                    "%s LLM (%s @ %s) returned EMPTY content",
                    profile_name,
                    model,
                    endpoint,
                )
            if last_exc is not None:
                raise last_exc
            return ""

        async def _caller_visible(messages: list) -> str:
            """Run the call inside a view-only child session so the operator
            can open and read the turn (same as a compaction)."""
            child = None
            try:
                from sharedagents_agent.compaction import spawn_visible_internal_subagent

                def _text(m: Any) -> str:
                    c = m.get("content") if isinstance(m, dict) else m
                    return c if isinstance(c, str) else str(c or "")

                system = "\n\n".join(_text(m) for m in messages if isinstance(m, dict) and m.get("role") == "system")
                user = "\n\n".join(
                    _text(m) for m in messages if not (isinstance(m, dict) and m.get("role") == "system")
                )
                child = await asyncio.to_thread(
                    spawn_visible_internal_subagent,
                    session,
                    agent_name=profile_name,
                    system_prompt=system or f"Hidden {profile_name} agent.",
                    user_prompt=user[:12_000],
                    engine="local_llm",
                )
            except Exception:  # noqa: BLE001 — the side call must still run
                child = None
            try:
                text = await _caller(messages)
            except BaseException as exc:
                if child is not None:
                    await asyncio.to_thread(
                        child.finish,
                        content="",
                        aborted=isinstance(exc, asyncio.CancelledError),
                        error=str(exc) or type(exc).__name__,
                    )
                raise
            if child is not None:
                await asyncio.to_thread(child.finish, content=str(text or ""))
            return text

        return _caller_visible

    def _persist_title_header(self, session: Any) -> None:
        """Persist only the title fields of the session header."""
        from sharedagents_session.session_writer import get_session_writer

        get_session_writer(
            self.sessions_dir,
            session.hash,
        ).header_update(
            patch={
                "title": getattr(session, "title", "") or "",
                "title_refinements": int(getattr(session, "title_refinements", 0) or 0),
            },
        )

    async def maybe_create_title(
        self,
        session: Any,
        *,
        turn_number: int,
        prompt: Any,
        latest_user_text: Any = "",
        model: str | None = None,
    ) -> bool:
        """Run one lifecycle title call without affecting the user turn."""
        try:
            from sharedagents_session.session_title import (
                ensure_title_for_turn,
            )
        except Exception:  # noqa: BLE001
            return False
        caller = self._make_hidden_agent_caller(
            session,
            "title",
            model_override=model,
        )
        if caller is None:
            _LOG.warning(
                "title: no OpenAI-compatible Jarvis endpoint for session %s",
                str(getattr(session, "hash", "") or "")[:12],
            )
            return False
        try:
            return await ensure_title_for_turn(
                session,
                caller,
                turn_number=turn_number,
                user_text=prompt,
                latest_user_text=latest_user_text,
                persist=lambda: self._persist_title_header(session),
            )
        except Exception as exc:  # noqa: BLE001 — title is best-effort
            _LOG.warning(
                "title: generation failed for session %s: %s",
                str(getattr(session, "hash", "") or "")[:12],
                exc,
            )
            return False

    def schedule_title_creation(
        self,
        session: Any,
        *,
        turn_number: int,
        prompt: Any,
        prompt_parts: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> Any:
        """Detach a title call from the active turn and serialize refreshes."""
        try:
            from sharedagents_session.session_title import should_create_title

            if not should_create_title(session, turn_number):
                return None
        except Exception:  # noqa: BLE001
            return None

        sid = str(getattr(session, "hash", "") or "")
        previous = self._title_tasks.get(sid)
        title_prompt: Any = list(prompt_parts) if prompt_parts else (prompt or "")
        turns = list(getattr(session, "turns", None) or [])
        opening_prompt: Any = (
            turns[0].get("user") if turn_number != 1 and turns and isinstance(turns[0], dict) else title_prompt
        )
        # Snapshot mutable session context before the engine records this turn.

        async def _run() -> None:
            if previous is not None and not previous.done():
                try:
                    await previous
                except Exception:  # noqa: BLE001 — a failed refresh is retryable
                    pass
            await self.maybe_create_title(
                session,
                turn_number=turn_number,
                prompt=opening_prompt,
                latest_user_text=title_prompt,
                model=model,
            )

        try:
            task = asyncio.create_task(
                _run(),
                name=f"session_title_{sid[:12]}_t{turn_number}",
            )
        except Exception:  # noqa: BLE001 — no running loop / shutdown
            return None
        self._title_tasks[sid] = task

        def _forget(done: Any) -> None:
            if self._title_tasks.get(sid) is done:
                self._title_tasks.pop(sid, None)
            try:
                done.exception()
            except (asyncio.CancelledError, Exception):
                pass

        task.add_done_callback(_forget)
        return task

    async def maybe_ensure_title(self, session: Any) -> bool:
        """Backward-compatible entry point for a scheduled title turn."""
        try:
            from sharedagents_session.session_title import (
                ensure_title_for_turn,
            )
        except Exception:  # noqa: BLE001
            return False
        turns = list(getattr(session, "turns", None) or [])
        turn_number = len(turns)
        try:
            from sharedagents_session.session_title import should_create_title

            if not should_create_title(session, turn_number):
                return False
        except Exception:  # noqa: BLE001
            return False
        caller = self._make_hidden_agent_caller(session, "title")
        if caller is None:
            return False
        first = turns[0].get("user") if turns else ""
        latest = turns[-1].get("user") if turns else first
        try:
            return await ensure_title_for_turn(
                session,
                caller,
                turn_number=turn_number,
                user_text=first,
                latest_user_text=latest,
                persist=lambda: self._persist_title_header(session),
            )
        except Exception:  # noqa: BLE001 — never surface into the turn
            return False

    # ---- delegate-as-tool compatibility ----------------------------

    def delegate_sidecar_path(self, parent_sid: str, kind: EngineKind) -> Path:
        """Legacy sidecar path for ``delegate_<kind>`` tool result resume.

        Returns ``<parent_agent.sessions_dir>/<parent_sid>.delegate_<kind>.json``.
        Reads/writes this file when the Agent is operating as a child
        of a ``delegate_claude`` / ``delegate_codex`` / ``delegate_antigravity``
        tool call, so an in-flight chain from before the refactor
        keeps resuming after upgrade (T11 in the regression list).

        Under the new architecture, the sharedagents model that calls
        ``delegate_claude(prompt=…)`` actually triggers:

            child_agent = factory.load_agent("claude")  # the built-in
            child_session = child_agent.create_session()  # or resume by sidecar
            return child_agent.run_turn(child_session, prompt=…)

        — i.e. "delegate as a tool" and "agent IS Claude" collapse to
        one code path.
        """
        # Map EngineKind → the legacy filename suffix today's
        # delegate runners write (``.delegate_<cli>.json``).
        suffix_map = {
            EngineKind.CLAUDE_CLI: "delegate_claude",
            EngineKind.CODEX_CLI: "delegate_codex",
            EngineKind.ANTIGRAVITY_CLI: "delegate_antigravity",
        }
        suffix = suffix_map.get(kind)
        if suffix is None:
            # Local LLM / unknown → fall back to engine-kind tag
            # so callers always get a deterministic path.
            suffix = f"delegate_{getattr(kind, 'value', str(kind))}"
        return self.sessions_dir / f"{parent_sid}.{suffix}.json"

    # ---- ACP / openai-compat translation ---------------------------
    #
    # The gateway speaks ACP today and a thin OpenAI-compatible REST
    # endpoint (``sharedagents_gateway.openai_endpoint``). Both consume the
    # ``AgentTurnEvent`` stream the Agent emits. The Agent provides
    # two small adapters so the gateway code stays simple.

    # ---- durable follow-up queue -----------------------------------
    #
    # ``_queue_writer`` / ``_queue_load`` / ``queue_add`` / ``queue_list`` /
    # ``queue_update`` / ``queue_remove`` / ``queue_inject_record`` /
    # ``_auto_drain_text`` / ``_make_queue_drain`` / ``queue_has_pending`` live
    # in :class:`FollowupQueueMixin` (``agents/followup_queue.py``), mixed into
    # ``Agent`` below.
