"""``LocalLLMEngine`` package facade.

The god-class was split into cohesive mixins (turn loop, tool dispatch,
compaction, loop detection, continuation, cost, injection, hooks) plus a
shared :mod:`.state` module.  This ``__init__`` re-declares the class from
those mixins and re-exports the public surface, so every existing
``from sharedagents_agent.agents.local_llm import X`` keeps resolving.
"""

from __future__ import annotations

import asyncio
import os
import sys as _sys
import types as _types
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from ..engine import Engine, EngineKind
from .state import *  # noqa: F403  (re-export the module-level surface)
from .state import (  # noqa: F401
    _LOG,
    EngineSessionState,
    _CompactionAborted,
    _NoSummaryCompaction,
    _RoundState,
    _terminate_and_reap,
)

# Env-driven constants are recomputed here (identical logic to state.py) so
# that ``importlib.reload(<this package>)`` re-reads the environment — the
# monolith recomputed them on reload, and state.py is not itself reloaded.
_MAX_READONLY_SOFT_LIMIT = int(os.environ.get("SHAREDAGENTS_READONLY_SOFT_LIMIT", "32") or "32")
_MAX_READONLY_HARD_LIMIT = int(os.environ.get("SHAREDAGENTS_READONLY_HARD_LIMIT", "48") or "48")
try:
    LOOP_CHECK_STRIDE = int(os.environ.get("SHAREDAGENTS_LOOP_CHECK_STRIDE", "300") or "300")
except ValueError:
    LOOP_CHECK_STRIDE = 300
LOOP_DETECT_ENABLED = os.environ.get("SHAREDAGENTS_LOOP_DETECT", "1") not in ("0", "false", "False")
from .compaction import CompactionMixin  # noqa: E402
from .continuation import ContinuationMixin  # noqa: E402
from .cost import CostMixin  # noqa: E402
from .hooks import HooksMixin  # noqa: E402
from .injection import InjectionMixin  # noqa: E402
from .loop_detection import LoopDetectionMixin  # noqa: E402
from .tool_dispatch import ToolDispatchMixin  # noqa: E402
from .turn_loop import TurnLoopMixin  # noqa: E402


class LocalLLMEngine(
    TurnLoopMixin,
    ToolDispatchMixin,
    CompactionMixin,
    LoopDetectionMixin,
    ContinuationMixin,
    CostMixin,
    InjectionMixin,
    HooksMixin,
    Engine,
):
    """Drive a local-LLM-backed turn end-to-end.

    Composition site: one instance per :class:`Agent`, configured from
    that Agent's :class:`AgentConfig`. The engine reads the live config
    on every turn (no caching) so a runtime ``/model`` switch or
    ``/mode`` change takes effect on the next call.
    """

    kind = EngineKind.LOCAL_LLM

    def __init__(
        self,
        agent_config: Any,
        *,
        # Test seams — production passes None and the defaults below kick in.
        worker_spawner: Callable[..., Awaitable[Any]] | None = None,
        tool_runner: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        """Build with an :class:`AgentConfig`.

        Test seams (``worker_spawner`` / ``tool_runner``) let tests
        inject scripted streams without spawning real worker / tool
        subprocesses. Production passes ``None`` and the engine uses
        :func:`sharedagents_gateway.acp_bridge._default_spawn_worker` /
        :func:`_default_run_tool` from the helper module.
        """
        self.cfg = agent_config
        # Bound by Agent.__post_init__. Standalone engine tests may leave this
        # unset; live-only tools then return a clear "agent unavailable" result
        # instead of crashing through the fallback runner shim.
        self._agent: Any = None
        self._worker_spawner = worker_spawner
        self._tool_runner = tool_runner

        # Per-session live state. Persisted across turns inside the
        # same session, dropped on close_session. All local-LLM-
        # specific — CLI engines don't have rounds, tool dispatch,
        # or context promotion.
        #
        # Persisted across turns inside the same session, dropped on
        # close_session.
        self._cancelled: dict[str, bool] = {}  # sharedagents_sid → flag
        # Graceful-stop flag (``/stop`` / first ESC). Distinct from the
        # hard ``_cancelled`` kill: it lets the in-flight tool/round
        # finish, then breaks the round loop BEFORE the next round / next
        # tool group instead of SIGKILLing anything. The operator's
        # second ESC (or ``/abort``) escalates to ``_cancelled`` +
        # ``abort(force=True)``.
        self._graceful_stop: dict[str, bool] = {}  # sharedagents_sid → flag
        self._max_steps_handoff: dict[str, bool] = {}  # sharedagents_sid → flag
        self._budget_handoff: dict[str, bool] = {}  # sharedagents_sid → cost-budget flag
        self._turn_cost_pico: dict[str, int] = {}  # sharedagents_sid → in-turn LOCAL spend (pico)
        # sharedagents_sid → root-session ledger spend snapshotted at turn start, so the
        # prompt-budget gate can add THIS turn's SUBAGENT spend (delegates inherit
        # the parent root, so their ledger rows roll up to it) on top of the local
        # accumulator. None == not yet snapshotted this turn.
        self._prompt_cost_baseline_pico: dict[str, int | None] = {}
        self._doom_loop_sigs: dict[str, list[str]] = {}  # sharedagents_sid → recent tool sigs
        self._mode_overrides: dict[str, str] = {}  # sharedagents_sid → mode
        # Loop-proofing for proactive compaction: ``sharedagents_sid →`` the
        # ``_msg_chars`` size of a working set a PROACTIVE squeeze just failed
        # to meaningfully shrink. While the working set stays at/below that
        # size, ``_compaction_target`` refuses to squeeze again (see
        # :meth:`_record_squeeze_progress`) — so compaction can never loop on
        # content it can't reduce, regardless of token-estimate accuracy.
        self._compaction_stall_chars: dict[str, int] = {}
        # ``sharedagents_sid →`` turn count at last successful proactive compact
        # (OpenCode min_messages thrash guard).
        self._last_proactive_compact_turn: dict[str, int] = {}
        # ``sharedagents_sid →`` the pre-send budget of the last request (prompt
        # tokens counted, ceiling sent) and the character-estimate miss ratio
        # learned from billed rounds (only used for rows with no tokenizer).
        self._presend_last: dict[str, dict[str, Any]] = {}
        self._presend_ratio: dict[str, float] = {}
        # Constant tokens the chat template adds beyond our count (tool
        # boilerplate), learned per session from the last billed round.
        self._presend_offset: dict[str, int] = {}
        # ``sharedagents_sid →`` True when the last reactive overflow compaction was
        # an LLM summary (OpenCode's flow): the turn loop retries without
        # charging it to the mechanical-squeeze retry cap.
        self._reactive_via_llm: dict[str, bool] = {}
        # Last Session object passed to ``attach_session`` — Agent.get_session
        # falls back to this when the on-disk file hasn't been written
        # yet (fresh zero-turn sessions, race between create and first
        # prompt). The bridge force-saves on create_session so this is
        # rarely needed, but it's the defense-in-depth path for any
        # caller that skips the bridge entirely.
        self._sessions: dict[str, Any] = {}  # sharedagents_sid → Session
        # Mid-turn user message inject buffer. Frontends (TUI / Telegram)
        # push fresh user prompts here while a turn is already streaming
        # — ``run_turn`` drains the list at every round boundary and
        # appends each entry to the live message list as a ``user``-role
        # message before the next worker spawn. Lets a slow multi-round
        # plan pivot on a new user instruction without waiting for the
        # current turn to end. Per-sid because two sessions on the same
        # process must not cross-contaminate.
        self._pending_user_messages: dict[str, list[str]] = {}
        # Soft-abort flag set by ``inject_user_message`` when it SIGTERMs
        # the current round's worker so the model stops generating its
        # now-stale response. Distinguishes "user is pivoting, drain the
        # inject and continue the turn" from "operator hit ESC / sent
        # session/cancel, terminate the turn outright" (which sets
        # ``_cancelled[sid]`` instead). Cleared at the next round.
        self._soft_aborts: dict[str, bool] = {}
        # Sessions that have already received a ``SessionStart`` hook
        # event this process lifetime. Used to fire SessionStart at most
        # once per session — at the FIRST run_turn — without needing
        # tighter coupling to session create/load.
        self._sessions_started: set[str] = set()

        # Unified abort routing.
        # The Engine ABC contract says ``abort(turn_id)`` — the engine
        # MUST be able to look up the live subprocess by turn_id, not
        # by sid. Two maps:
        #
        # * ``_active_turns: dict[turn_id, TurnState]`` — what abort()
        #   reads. One entry per in-flight turn; populated by run_turn,
        #   cleared in its finally block.
        # * ``_session_to_turn: dict[sharedagents_sid, turn_id]`` — what the
        #   gateway's ACP ``cancel(session_id)`` resolves through. The
        #   AgentACPBackend reads this to translate sid → turn_id, then
        #   calls ``engine.abort(turn_id)``. Only one turn at a time
        #   per sid (session_lock guarantees), so this is a 1:1 map.
        self._active_turns: dict[str, Any] = {}  # turn_id → TurnState
        self._session_to_turn: dict[str, str] = {}  # sharedagents_sid → turn_id
        # Jarvis workflow state spans every model round of ONE physical turn.
        self._coordination_turns: dict[str, Any] = {}
        # Per-session teacher toggle (codex #8: per-session, not engine-global
        # — one engine instance serves many sids). THREE states:
        #   * sid in ``_teacher_off`` → forced OFF (no review fires);
        #   * sid in ``_teacher_on``  → forced ON, even with NO active goal
        #     (overrides the ``require_goal`` policy gate — "activate for every
        #     session, be it a goal or not");
        #   * sid in neither → default: follow policy + the goal gate.
        # Both are seeded on attach from the persisted ``Session.teacher_override``
        # ("on"/"off"/"") so the choice survives a restart, symmetric with
        # ``active_goal``.
        self._teacher_off: set[str] = set()  # sharedagents_sids with /teacher off
        self._teacher_on: set[str] = set()  # sharedagents_sids with /teacher on (goal-independent)
        # Per-session SUPERVISING-MODEL override set by ``/teacher model <alias>``.
        # Maps sharedagents_sid → model alias/id; when present, the teacher review
        # uses that ``models[]`` row instead of the config's ``role==teacher``
        # row. Persisted on ``Session.teacher_model_override`` (seeded on attach).
        self._teacher_model_override: dict[str, str] = {}
        # Active goal lives in an ENGINE-LEVEL dict (keyed by sid), NOT only on
        # the Session object — a Session instance can be reloaded/replaced
        # mid-turn (attach on first prompt, disk reload), which silently dropped
        # a goal set via /goal so the teacher never fired. This dict is the live
        # source of truth (like _teacher_off / _mode_overrides); the Session
        # field is the persistence shadow (seeded on attach, saved back for
        # restart durability). ONE goal per session — no queue.
        self._active_goals: dict[str, str] = {}  # sharedagents_sid → active goal
        # Per-session async lock — serialize concurrent prompts per sid.
        # Lazy-allocated on first
        # session_lock() call.
        self._session_locks: dict[str, asyncio.Lock] = {}
        # Background reap tasks for workers that were terminated from a
        # sync ``finally`` block (e.g. consumer aclose mid-yield). Each
        # task awaits ``proc.wait()`` so the subprocess is reaped, and
        # cleans up the stderr drainer. Self-removes on completion via
        # ``add_done_callback``.
        self._background_reapers: set[asyncio.Task[None]] = set()

    def bind_agent(self, agent: Any) -> None:
        """Attach the live composition-root Agent to this engine."""
        self._agent = agent

    # -----------------------------------------------------------------
    # Engine ABC surface
    # -----------------------------------------------------------------

    @property
    def name(self) -> str:
        return "local_llm"

    def state_path(self, sessions_dir: Path, sharedagents_sid: str) -> Path:
        """Returns ``<sessions_dir>/<sid>.local_llm.json`` symbolically.

        Never written — every round re-sends the full context, so there
        is no server-side resume id to persist. The path exists for
        symmetry with the CLI engines so callers can probe ``exists()``
        without branching on engine kind.
        """
        return Path(sessions_dir) / f"{sharedagents_sid}.local_llm.json"

    def read_state(self, sessions_dir: Path, sharedagents_sid: str) -> EngineSessionState:
        """Return a blank state. Driver doesn't depend on it.

        Local LLM has no engine-side session id. The Agent's
        first-call-vs-resume decision is driven instead by
        :meth:`system_prompt_policy` which always returns
        ``"every_turn"`` for local LLM — so this returns a fresh blank
        :class:`EngineSessionState` and that's it.
        """
        return EngineSessionState(
            sharedagents_session_id=sharedagents_sid,
            engine=EngineKind.LOCAL_LLM,
        )

    def write_state(self, sessions_dir: Path, state: EngineSessionState) -> None:
        """No-op. Local LLM has no engine-side session id."""
        return None

    def system_prompt_policy(
        self,
        state: EngineSessionState,
        *,
        current_system_prompt_sha256: str,
    ) -> str:
        """Always ``"every_turn"``.

        The local LLM is stateless across rounds — every prompt
        request to the upstream HTTP API carries the full message
        list including a fresh system prompt. The Agent layer composes
        the prompt once per turn and the engine prepends it on every
        round within that turn. ``current_system_prompt_sha256`` is
        ignored (no persisted session to invalidate).
        """
        return "every_turn"

    async def session_lock(self, sharedagents_sid: str) -> Any:  # asyncio.Lock
        """Lazy-allocate a per-session asyncio.Lock.

        Cached in ``self._session_locks: dict[str, asyncio.Lock]``.
        Local LLM mostly doesn't NEED this since rounds are sequential
        within a turn — but the Engine ABC contract requires it, and
        we use it to serialize the "two simultaneous prompts on the
        same sid" case the same way CLI engines do.
        """
        lock = self._session_locks.get(sharedagents_sid)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[sharedagents_sid] = lock
        return lock

    def attach_session(self, sharedagents_sid: str, session: Any) -> None:
        """Initialize per-session live state.

        Sets ``self._cancelled[sid] = False``. Pulls any prior mode override from
        ``session.system_mode`` (today's :class:`Session` carries
        this field). Also caches ``session`` on ``self._sessions`` so
        ``Agent.get_session`` can recover it when the on-disk file
        hasn't been written yet.
        """
        self._cancelled.setdefault(sharedagents_sid, False)
        if session is not None:
            self._sessions[sharedagents_sid] = session
        mode = str(getattr(session, "system_mode", "") or "") or None
        if mode:
            self._mode_overrides[sharedagents_sid] = mode
        # Seed the engine goal dicts from the persisted Session on FIRST attach
        # (e.g. resuming after a restart) so an in-flight goal survives. Never
        # clobber a live goal already in the dict — a mid-turn re-attach with a
        # reloaded (empty) Session must NOT erase the active goal (that was the
        # bug: the goal vanished and the teacher stopped firing).
        if sharedagents_sid not in self._active_goals:
            self._active_goals[sharedagents_sid] = str(getattr(session, "active_goal", "") or "")
        # Seed the per-session teacher toggle from the persisted Session on
        # first attach (restart durability), without clobbering a live toggle
        # already in the sets. "on"/"off" → explicit override; "" → default
        # (follow policy + goal gate).
        if sharedagents_sid not in self._teacher_on and sharedagents_sid not in self._teacher_off:
            _tov = str(getattr(session, "teacher_override", "") or "").lower()
            if _tov == "on":
                self._teacher_on.add(sharedagents_sid)
            elif _tov == "off":
                self._teacher_off.add(sharedagents_sid)
        # Seed the per-session teacher-MODEL override (which model supervises)
        # from the persisted Session, without clobbering a live override.
        if sharedagents_sid not in self._teacher_model_override:
            _tmo = str(getattr(session, "teacher_model_override", "") or "").strip()
            if _tmo:
                self._teacher_model_override[sharedagents_sid] = _tmo

    def detach_session(self, sharedagents_sid: str) -> None:
        """Cleanup on ``close_session``.

        * If a turn is in flight (``_session_to_turn[sid]`` present)
          preserve EVERYTHING — including ``_cancelled[sid]`` —
          so an abort flag set during the spawn window doesn't get
          silently dropped while the still-active run_turn is
          waiting to observe it. Caller should abort + await
          TurnDone, then re-call ``close_session``.
        * Once idle, drop the per-session bookkeeping atomically.
        """
        if sharedagents_sid in self._session_to_turn:
            _LOG.warning(
                "detach_session called while turn is active for sid=%s; "
                "all per-session state preserved until run_turn exits",
                sharedagents_sid[:12],
            )
            return
        # Clear SessionStart memory so a later attach of the same sid
        # re-fires the hook with source="resume" (and prevents a
        # slow set leak across many sessions).
        self._sessions_started.discard(sharedagents_sid)
        self._cancelled.pop(sharedagents_sid, None)
        self._graceful_stop.pop(sharedagents_sid, None)
        self._mode_overrides.pop(sharedagents_sid, None)
        self._session_locks.pop(sharedagents_sid, None)
        self._sessions.pop(sharedagents_sid, None)
        self._pending_user_messages.pop(sharedagents_sid, None)
        self._soft_aborts.pop(sharedagents_sid, None)
        self._active_goals.pop(sharedagents_sid, None)
        self._teacher_off.discard(sharedagents_sid)
        self._teacher_on.discard(sharedagents_sid)
        self._teacher_model_override.pop(sharedagents_sid, None)

    def is_available(self) -> tuple[bool, str]:
        """Whether this engine can actually complete a turn.

        This must validate more than binary/endpoint
        existence — at minimum, whether the credentials the ACTIVE model row
        requires are present. A model row that declares an ``apiKeyEnv`` whose
        value is resolvable from neither the per-agent ``api_keys`` map nor the
        process env cannot authenticate, so the engine reports UNAVAILABLE
        (one half of the "upstream 401 -> reported ok" symptom). Rows without a
        credential requirement (e.g. a local qwen endpoint) stay available;
        endpoint reachability itself is still checked lazily on first call.
        """
        try:
            cfg = self.cfg
            models = getattr(cfg, "models", None) or []
            upstream_model = str(getattr(cfg, "upstream_model", "") or "")
            api_keys = getattr(cfg, "api_keys", None) or {}
            row = None
            for m in models:
                if isinstance(m, dict) and str(m.get("id") or "") == upstream_model:
                    row = m
                    break
            if isinstance(row, dict):
                key_env = str(row.get("apiKeyEnv") or "")
                if key_env:
                    resolved = api_keys.get(key_env) or "" if key_env in api_keys else os.environ.get(key_env, "")
                    if not resolved:
                        return (
                            False,
                            f"required credential {key_env} for model "
                            f"{upstream_model!r} is absent — run `sharedagents "
                            f"configure` (request would 401).",
                        )
        except Exception:  # noqa: BLE001 - availability must never raise
            return True, ""
        return True, ""

    def _schedule_background_reap(
        self,
        proc: asyncio.subprocess.Process | None,
        stderr_task: asyncio.Task[None] | None,
    ) -> None:
        """Schedule a fire-and-forget cleanup task for a worker proc.

        Called from a sync ``finally`` block (where we can't ``await``)
        when the generator has been ``aclose()``d mid-yield, OR from
        the inner cancellation handler so the abort path returns
        quickly without inline-awaiting ``proc.wait(timeout=2)``.

        The task awaits ``proc.wait()`` so the subprocess is reaped
        cleanly, cancels and awaits the stderr drainer, then removes
        itself from ``_background_reapers``.
        """
        if proc is None and stderr_task is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no loop — nothing to do
        task = loop.create_task(_terminate_and_reap(proc, stderr_task))
        self._background_reapers.add(task)
        task.add_done_callback(self._background_reapers.discard)

    async def abort(self, turn_id: str, *, force: bool = False) -> None:
        """Best-effort kill of the named in-flight turn.

        Keyed by ``turn_id``. Looks up
        the :class:`TurnState`, calls its ``stop(force=…)`` to SIGTERM
        (or SIGKILL+killpg when ``force=True``) the worker and every
        registered runner subprocess, then reverse-resolves the sid
        to mark cancellation.

        ``force=True`` is the two-press hard-kill escalation
        ``CoreBackend.cancel`` triggers when a second ESC lands inside
        ``_HARD_KILL_WINDOW_S`` — propagates the SIGKILL +
        ``killpg`` semantics that ``TurnState.stop`` already
        implements.

        Flags cancellation EVEN IF the turn
        hasn't reached the ``_active_turns`` map yet (we're in the
        spawn window). The flag is checked by run_turn at the top of
        the read loop, so the turn either dies at spawn-time-cancel
        or right after the first event.

        Idempotent — unknown turn_ids are still recorded as cancelled
        via the sid reverse-resolve below.
        """
        turn = self._active_turns.get(turn_id)
        if turn is not None:
            try:
                turn.stop(force=force)
            except TypeError:
                # Older ``TurnState.stop`` had no ``force`` kwarg.
                try:
                    turn.stop()
                except Exception:  # noqa: BLE001
                    pass
            except Exception:  # noqa: BLE001
                pass
        # Reverse-resolve sid and flag cancellation. This works even
        # when the spawn hasn't yet populated _active_turns — run_turn
        # checks self._cancelled[sid] before yielding any event.
        for sid, tid in list(self._session_to_turn.items()):
            if tid == turn_id:
                self._cancelled[sid] = True
                break

    async def request_graceful_stop(self, turn_id: str) -> None:
        """Ask the named turn to stop at its next safe point.

        The graceful counterpart to :meth:`abort`: it kills NOTHING. It
        flips the per-session ``_graceful_stop`` flag, which the round
        loop honours at two cooperative checkpoints — the top of the
        round loop (so no new round spawns) and between tool-call groups
        (so no further tools dispatch). The in-flight tool / streaming
        round is allowed to finish cleanly first ("let current op
        finish, then stop"). A subsequent :meth:`abort` (the operator's
        2nd ESC / ``/abort``) escalates to a hard SIGKILL.

        Idempotent; unknown turn_ids are recorded against any session
        they happen to map to, mirroring :meth:`abort`.
        """
        for sid, tid in list(self._session_to_turn.items()):
            if tid == turn_id:
                self._graceful_stop[sid] = True
                break

    def _max_rounds_for_mode(self, session_id: str) -> None:
        """Return no numeric work cap; the session-tree budget is authoritative."""
        return None

    def _max_rounds_for_request(
        self,
        session_id: str,
        requested: Any = None,
        *,
        manual_budget_exhausted: bool = False,
    ) -> int | None:
        """Return no round cap, except for an over-budget manual reply."""
        if manual_budget_exhausted:
            return 1
        return None

    async def set_mode(self, session_id: str, mode_id: str) -> None:
        """Per-session permission-mode override.

        Reference: acp_bridge.py L1654–1687. Validates ``mode_id ∈
        {plan, ask, auto}``, updates ``self._mode_overrides[sid]``,
        emits a ``mode_changed`` ACP event upstream.
        """
        # Use the canonical tuple, not a hardcoded literal — keeps
        # this defense-in-depth check from drifting if a 4th mode is
        # ever added to PermissionPolicy. Today the bridge + shim
        # already validate at entry, so this branch is unreachable
        # via normal paths; the literal is kept for the direct
        # ``engine.set_mode`` test/debug seam.
        from ..permissions_integration import coerce_permission_mode

        mode_id = coerce_permission_mode(mode_id)
        self._mode_overrides[session_id] = mode_id
        self._log_event(
            "MODE_CHANGED",
            session_id=session_id[:12],
            mode=mode_id,
        )

    async def set_config_option(self, session_id: str, option_id: str, value: Any) -> None:
        """Per-session config-option override.

        Reference: acp_bridge.py L1688–1759. Handles ``/reasoning``,
        ``/tool``, ``/thinking``, ``/model``, ``/temperature``. Stores
        on the :class:`Session` (so persisted across restart) and
        echoes a sessionUpdate so all frontends sync.
        """
        if option_id == "reasoning":
            # Matches `normalize_reasoning_level()` in
            # sharedagents_command.commands — the single user-facing vocabulary
            # ({off, on, stream}). The gate at `_run_round` reads any
            # non-"off" as "show thinking deltas".
            allowed = {"off", "on", "stream"}
            v = str(value or "").lower()
            if v not in allowed:
                raise ValueError(f"reasoning: invalid value {value!r}")
            sess = self._sessions.get(session_id)
            if sess is None:
                raise RuntimeError(
                    f"reasoning: session not attached ({session_id!r})",
                )
            sess.reasoning_visibility = v
        elif option_id == "teacher":
            # ``/teacher on|off`` — live PER-SESSION toggle for teacher
            # supervision (codex #8). ``off`` forces it off; ``on`` forces it on
            # EVEN WITH NO ACTIVE GOAL (overrides the require_goal gate) — the
            # operator can supervise any session, goal or not. The choice is
            # persisted on the Session (``teacher_override``) so it survives a
            # restart, symmetric with ``active_goal``. Other sessions this
            # engine serves are unaffected.
            v = str(value or "").lower()
            override = "off" if v in ("off", "0", "false", "no", "disable", "stop") else "on"
            if override == "off":
                self._teacher_off.add(session_id)
                self._teacher_on.discard(session_id)
            else:
                self._teacher_on.add(session_id)
                self._teacher_off.discard(session_id)
            sess = self._sessions.get(session_id)
            if sess is not None:
                try:
                    sess.teacher_override = override
                    sess.save()
                except Exception:  # noqa: BLE001
                    pass
        elif option_id == "teacher_model":
            # ``/teacher model <alias>`` — pick which models[] row supervises
            # THIS session (per-session override of the config role==teacher
            # row). Empty / "default" / "auto" clears the override (back to the
            # configured teacher). An unknown alias raises so the frontend can
            # tell the operator. Persisted on Session.teacher_model_override.
            alias = str(value or "").strip()
            if alias.lower() in ("", "default", "auto", "reset", "clear", "none"):
                self._teacher_model_override.pop(session_id, None)
                alias = ""
            else:
                if self._resolve_teacher_model_alias(alias) is None:
                    avail = ", ".join(
                        str(m.get("alias") or m.get("id"))
                        for m in (getattr(self.cfg, "models", None) or [])
                        if isinstance(m, dict) and (m.get("alias") or m.get("id"))
                    )
                    raise ValueError(
                        f"teacher model: no models[] row matches {alias!r}"
                        + (f" (available: {avail})" if avail else "")
                    )
                self._teacher_model_override[session_id] = alias
            sess = self._sessions.get(session_id)
            if sess is not None:
                try:
                    sess.teacher_model_override = alias
                    sess.save()
                except Exception:  # noqa: BLE001
                    pass
        elif option_id == "goal":
            goal = str(value or "").strip()
            if goal.lower() in ("", "clear", "off", "none", "stop", "reset"):
                self._set_goal_state(session_id, "", reset_progress=True)
            else:
                self._set_goal_state(session_id, goal, reset_progress=True)
        elif option_id == "topic":
            # Conversation topic — the Telegram bot pushes the forum-topic
            # name here when it binds a session to a thread so the WebUI
            # Topics page can group this session. Empty / placeholder values
            # fall back to "general" (the universal default). Persisted on the
            # Session so the projector folds it into ``sessions_meta.topic``.
            topic = str(value or "").strip() or "general"
            sess = self._sessions.get(session_id)
            if sess is None:
                # ``topic`` is a DURABLE data path (it lands in
                # sessions_meta), so a missing attachment is worth a signal —
                # the caller (CoreBackend.set_config_option) attaches before
                # forwarding, so reaching here means a timing/wiring bug, not
                # an expected no-op. Surface it instead of dropping silently.
                self._log_event(
                    "CONFIG_OPTION_TOPIC_NO_SESSION",
                    session_id=session_id[:12],
                    topic=topic,
                )
            else:
                try:
                    sess.topic = topic
                    sess.save()
                except Exception as exc:  # noqa: BLE001
                    self._log_event(
                        "CONFIG_OPTION_TOPIC_SAVE_FAILED",
                        session_id=session_id[:12],
                        topic=topic,
                        error=str(exc),
                    )
        elif option_id == "thinking":
            v = (
                bool(value)
                if not isinstance(value, str)
                else value.lower()
                in (
                    "1",
                    "true",
                    "yes",
                    "on",
                )
            )
            self.cfg.enable_thinking = v
        elif option_id == "effort":
            from sharedagents_command.commands import normalize_effort

            v = normalize_effort(str(value or ""))
            v = "" if v == "default" else v
            sess = self._sessions.get(session_id)
            if sess is None:
                raise RuntimeError(
                    f"effort: session not attached ({session_id!r})",
                )
            sess.effort = v
            try:
                sess.save()
            except Exception:  # noqa: BLE001
                pass
        elif option_id == "speed":
            v = str(value or "").strip().lower()
            if v in ("", "default", "auto", "reset", "clear", "none"):
                v = ""
            elif v not in {"fast", "standard"}:
                raise ValueError(
                    f"speed: invalid value {value!r} — use default|fast|standard|auto",
                )
            sess = self._sessions.get(session_id)
            if sess is None:
                raise RuntimeError(
                    f"speed: session not attached ({session_id!r})",
                )
            sess.speed = v
            try:
                sess.save()
            except Exception:  # noqa: BLE001
                pass
        elif option_id == "model":
            alias = str(value or "").strip()
            if not alias:
                raise ValueError("model: empty alias")
            sess = self._sessions.get(session_id)
            if sess is None:
                raise RuntimeError(
                    f"model: session not attached ({session_id!r})",
                )
            # Per-session only. ``cfg.upstream_*`` is the agent DEFAULT
            # (from ``AgentConfig.load`` / disk config) and must stay
            # untouched so ``/model`` in one session does not leak into
            # sibling sessions or the next ``/new`` / heartbeat session.
            row = self._find_deployment_row(alias)
            if row is None:
                raise ValueError(f"model: unknown alias {alias!r}")
            row_id = str(row.get("id") or "").strip()
            base_url = str(row.get("baseUrl") or "").strip()
            if not row_id:
                raise ValueError(
                    f"model alias {alias!r} has no `id` configured "
                    f"({base_url or '<no baseUrl>'}). Set the model "
                    f"id explicitly in ~/.sharedagents/config.json."
                )
            sess.model = row_id
            if base_url:
                sess.endpoint = base_url
            sess.backend = str(row.get("backend") or "")
            cw = int(row.get("contextWindow") or 0)
            if cw > 0:
                sess.context_window = cw
            try:
                save_root = getattr(self.cfg, "sessions_dir", None)
                if save_root:
                    sess.save(root=save_root, force=False)
                else:
                    sess.save()
            except Exception:  # noqa: BLE001
                pass
        elif option_id == "temperature":
            # 'null'/'none' (or empty) clears it → omit the param on the
            # wire so the upstream applies its own default.
            if value is None or (isinstance(value, str) and value.strip().lower() in ("null", "none", "")):
                self.cfg.temperature = None
            else:
                try:
                    self.cfg.temperature = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"temperature: {exc}") from exc
        else:
            raise ValueError(f"unknown config option: {option_id!r}")
        self._log_event(
            "CONFIG_OPTION",
            session_id=session_id[:12],
            option=option_id,
            value=str(value),
        )

    def _find_deployment_row(self, target: str) -> dict[str, Any] | None:
        """Find a deployment row by ``alias`` OR wire ``id``.

        Reloads the model rows from disk once on a miss (mirrors
        :meth:`set_config_option`) so a deployment added AFTER the
        gateway booted is still found. Returns ``None`` when nothing
        matches — callers decide whether a miss is fatal.
        """
        target = str(target or "").strip()
        if not target:
            return None

        def _scan(t: str) -> dict[str, Any] | None:
            rows = [
                (dict(m) if isinstance(m, dict) else getattr(m, "__dict__", {}))
                for m in (getattr(self.cfg, "models", None) or [])
            ]
            # A provider refresh (web process) persists newly advertised rows in
            # the shared native cache. The long-lived gateway must merge them so
            # `/model <discovered-alias>` resolves without a restart — mirrors
            # ``acp_bridge.CoreBackend._resolve_model_row``.
            try:
                from sharedagents_agent.discovery.cache_lock import native_models_with_cache

                rows = native_models_with_cache(
                    str(getattr(self.cfg, "name", "") or "jarvis"),
                    rows,
                )
            except Exception:  # noqa: BLE001
                pass
            for m_dict in rows:
                if not isinstance(m_dict, dict):
                    continue
                if t in (str(m_dict.get("alias") or ""), str(m_dict.get("id") or "")):
                    return dict(m_dict)
            return None

        row = _scan(target)
        if row is None:
            reload_fn = getattr(self.cfg, "reload_models", None)
            if callable(reload_fn):
                try:
                    reload_fn()
                except Exception:  # noqa: BLE001
                    pass
                row = _scan(target)
        return row

    def _apply_model_override(
        self,
        session: Any,
        model: str | None,
    ) -> bool:
        """Apply a per-turn ``model`` override onto ``session``.

        The ``Agent`` / ``BatchAgent`` tools and the ``/delegate`` shims
        accept a ``model`` argument that flows ``Agent.run_turn`` →
        ``TurnRequest.model`` → :meth:`run_turn`. The worker request is
        built from ``session.model`` / ``session.endpoint`` (see
        :meth:`_build_worker_request` / ``_default_spawn_worker``), so an
        override is meaningless unless it lands on the SESSION before the
        spawn. Without this the arg was silently dropped — every subagent
        ran on the parent's model (e.g. 3 "INT4" subagents actually hit
        the FP8 pool).

        Resolution is ISOLATED to ``session`` — we never touch the global
        ``cfg.upstream_*`` triple (that would leak the override into
        sibling sessions and fresh ``/new`` sessions):

          * empty / falsy           → no-op, returns ``False``
          * a ``cfg.models`` row matched by ``alias`` OR wire ``id``
                                    → apply the row's ``id`` + ``baseUrl``
          * a raw ``http...`` URL   → resolve via ``cfg.resolve_model``
          * otherwise (unknown)     → stamp the raw string as
            ``session.model`` and leave ``session.endpoint`` alone, so a
            wire id served by the SAME mux endpoint as the parent still
            routes (the INT4-pool case) instead of failing the turn.

        No path refuses a model solely because it is not the reference
        model: any configured (or best-effort raw) model is applied.

        Returns ``True`` when ``session`` was changed.
        """
        target = str(model or "").strip()
        if not target:
            return False

        row: dict[str, Any] | None = None
        if target.startswith("http"):
            resolve = getattr(self.cfg, "resolve_model", None)
            if callable(resolve):
                try:
                    row = resolve(target)
                except Exception:  # noqa: BLE001
                    row = None
        else:
            row = self._find_deployment_row(target)

        if row:
            row_id = str(row.get("id") or "").strip()
            base_url = str(
                row.get("baseUrl") or row.get("endpoint") or "",
            ).strip()
            if row_id:
                session.model = row_id
            elif not target.startswith("http"):
                # A non-URL row with no wire id — fall back to the raw
                # string rather than blanking the session's model.
                session.model = target
            if base_url:
                session.endpoint = base_url
        else:
            # Unknown id and not a URL — best-effort raw passthrough so a
            # model served by the parent's mux endpoint still routes.
            session.model = target
        return True

    def _resolve_model_row(
        self,
        alias: str,
        endpoint: str = "",
    ) -> dict[str, Any] | None:
        """Resolve a model row by alias OR endpoint.

        Production ``cfg.models`` rows carry the operator-chosen
        ``alias`` (e.g. ``"testbox"`` or ``"gpu-a"``) — that's what the
        TUI metrics footer wants to render. ``models.for_name`` only
        knows about the CANONICAL Model dataclasses in
        ``sharedagents_model.models``; it doesn't see the operator's per-
        endpoint cfg.models[]. So when the upstream returns the raw
        model id (``qwen3.6-27b-fp8``), ``for_name`` matches the
        canonical Model and the metrics row reads bare instead of
        ``<alias>/<model>``.

        Resolution order:
          1. ``cfg.model_row_for_endpoint(endpoint)`` if endpoint is
             provided AND the row carries an ``alias`` / ``id`` /
             ``name`` (i.e. it's a real cfg.models[] entry, not the
             fallback ``{"baseUrl": …}`` stub for unknown endpoints).
          2. ``models.for_name(alias, strict=True)`` as the canonical
             fallback for built-in models.
          3. ``None`` on miss so the caller surfaces an error instead
             of flipping the active model to a default.
        """
        if not alias and not endpoint:
            return None

        # 1. cfg.models[] lookup by endpoint — owns the operator's
        #    alias mapping.
        if endpoint:
            try:
                cfg_row = self.cfg.model_row_for_endpoint(endpoint, alias)
            except TypeError:
                try:
                    cfg_row = self.cfg.model_row_for_endpoint(endpoint)
                except Exception:  # noqa: BLE001
                    cfg_row = None
            except Exception:  # noqa: BLE001
                cfg_row = None
            if cfg_row and (cfg_row.get("alias") or cfg_row.get("id") or cfg_row.get("name")):
                return cfg_row

        # 2. Canonical fallback via the in-tree Model registry.
        if not alias:
            return None
        try:
            from sharedagents_model.models import for_name
        except ImportError:
            return None
        try:
            row = for_name(alias, strict=True)
        except TypeError:
            # Older ``for_name`` signatures don't support ``strict``.
            try:
                row = for_name(alias)
            except Exception:  # noqa: BLE001
                return None
        except Exception:  # noqa: BLE001
            return None
        if row is None:
            return None
        # ``for_name`` may return a Model dataclass; normalize to dict.
        if isinstance(row, dict):
            return row
        out: dict[str, Any] = {}
        for attr in ("name", "alias", "endpoint", "backend", "context_window", "extra_sampling"):
            v = getattr(row, attr, None)
            if v is not None:
                out[attr] = v
        return out or {"alias": alias}


from . import state as _state  # noqa: E402

_PROPAGATE_CONSTS = frozenset(
    {
        "_MAX_READONLY_SOFT_LIMIT",
        "_MAX_READONLY_HARD_LIMIT",
        "LOOP_CHECK_STRIDE",
        "LOOP_DETECT_ENABLED",
    }
)
_PATCH_TARGET_NAMES = (
    "sharedagents_agent.agents.local_llm.turn_loop",
    "sharedagents_agent.agents.local_llm.tool_dispatch",
    "sharedagents_agent.agents.local_llm.compaction",
    "sharedagents_agent.agents.local_llm.loop_detection",
    "sharedagents_agent.agents.local_llm.continuation",
    "sharedagents_agent.agents.local_llm.cost",
    "sharedagents_agent.agents.local_llm.injection",
    "sharedagents_agent.agents.local_llm.hooks",
)


def _patch_targets():
    mods = [_state]
    for _n in _PATCH_TARGET_NAMES:
        _m = _sys.modules.get(_n)
        if _m is not None:
            mods.append(_m)
    return mods


class _LocalLLMPackageModule(_types.ModuleType):
    def __setattr__(self, name, value):
        _types.ModuleType.__setattr__(self, name, value)
        if name in _PROPAGATE_CONSTS:
            for _m in _patch_targets():
                try:
                    _types.ModuleType.__setattr__(_m, name, value)
                except Exception:
                    pass


_sys.modules[__name__].__class__ = _LocalLLMPackageModule
# Force-sync env-derived constants into the already-imported submodules on every
# package execution — import AND importlib.reload(). Reload recomputes the facade
# values in the module body above via a plain module-dict write that runs BEFORE
# the class swap on the line above, so __setattr__ never fires for them; without
# this force-sync the submodules (and the turn-loop budget guard in turn_loop.py,
# which reads its own ``from .state import *`` copy) would keep the pre-reload
# value. __setattr__ stays for later external monkeypatch assignments.
for _sync_mod in _patch_targets():
    for _sync_name in _PROPAGATE_CONSTS:
        if _sync_name in globals():
            try:
                _types.ModuleType.__setattr__(_sync_mod, _sync_name, globals()[_sync_name])
            except Exception:
                pass
