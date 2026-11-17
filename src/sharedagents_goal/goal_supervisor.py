"""Goal outer-loop supervisor — drives turn N+1 after TurnDone.

Hosted on :class:`Agent`; keyed by durable ``goal_id`` but executes on
``goal.root_session_id``.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import logging
import os
import socket
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sharedagents_goal.goal_registry import (
    REASON_EVALUATOR_UNAVAILABLE,
    REASON_IMPOSSIBLE,
    REASON_NO_PROGRESS,
    REASON_TURN_ERRORS,
    REASON_WAITING_OWNER,
    SESSION_ENDED_STATES,
    STATUS_RUNNING,
    GoalContractChangedError,
    GoalRecord,
    _utcnow_iso,
    get_goal_registry,
    root_session_state,
    sync_session_goal_projection,
)

if TYPE_CHECKING:
    from sharedagents_goal.goal_service import GoalBudgetState

try:
    from sharedagents_goal.goal_registry import _SUPERVISOR_LEASE_S
except ImportError:  # pragma: no cover
    _SUPERVISOR_LEASE_S = 120.0

_LEASE_RENEW_INTERVAL_S = 20.0

# Bounded in-process re-drive of a goal whose drive task CRASHED
# (raised) mid-goal — the root-cause complement to the once-per-process
# ``recover_running_goals`` recovery and the zombie-reconciler safety net. Bounded
# so a deterministically-crashing goal cannot hot-loop; after the bound it is left
# to the reconciler / next-process recovery rather than hammered forever.
_DEFAULT_MAX_REDRIVE = 5

# Claude Code /goal no-progress guard: this many consecutive goal turns
# without a single tool call stop the loop. The goal stays set; the user's
# next message resumes it.
_DEFAULT_NO_PROGRESS_TURNS = 3

_LOG = logging.getLogger("sharedagents.goal_supervisor")

_OWNER_ID = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"

# ``(goal_id, turn_seq)`` of the supervised auto-turn the current code runs in.
# Tasks copy the context they are created in, so the tool task that
# ``tool_dispatch`` starts for a ``goal`` call inherits it (see ``cancel``).
_SUPERVISED_TURN: contextvars.ContextVar[tuple[str, int] | None] = contextvars.ContextVar(
    "sharedagents_goal_supervised_turn", default=None
)


@dataclass
class SupervisedTurnReceipt:
    """What one supervised auto-turn actually DID — the receipt the drive
    loop uses for effective-progress classification (TIER 1a/1b).

    Shape (public contract, mirrored in tests):
    - ``turn_id``: the engine turn id (``TurnStarted.turn_id``).
    - ``turn_state``: ``"ok"`` / ``"error"`` / ``"aborted"`` / ``"done"``.
    - ``saw_turn_done``: exactly one TurnDone must terminate every launch
      (TIER 3c — a missing terminal counts as an error, never as progress).
    - ``tool_calls``: ``[{name, action, ok}]`` — action is the tool's own
      ``action`` arg when present (goal tool), else ``""``; call ids,
      timestamps and prose are deliberately NOT carried (fingerprint input).
    - ``artifacts``: artifact uris/paths emitted this turn.
    - ``child_job_transitions``: completed delegate/subagent/async-job calls.
    - ``terminal_transition``: ``"met"`` / ``"blocked"`` / ``"paused"`` when
      the turn ended in a goal-state transition, else ``""``.
    - ``cost_pico``: this turn's usage-ledger delta (authoritative SUM
      before/after), used for the TIER 3a admission reservation.
    - ``fingerprint``: normalized sha256 over sorted ``(name, action, ok)``
      tuples — equal fingerprints across turns mean the model repeated the
      same (ineffective) pattern modulo call ids/timestamps/prose.
    - ``output_digest``: sha256 of the streamed final content ("" when the
      turn produced none).
    """

    turn_id: str = ""
    turn_state: str = "error"
    saw_turn_done: bool = False
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    child_job_transitions: int = 0
    terminal_transition: str = ""
    cost_pico: int = 0
    fingerprint: str = ""
    output_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "turn_state": self.turn_state,
            "saw_turn_done": self.saw_turn_done,
            "tool_calls": self.tool_calls,
            "artifacts": self.artifacts,
            "child_job_transitions": self.child_job_transitions,
            "terminal_transition": self.terminal_transition,
            "cost_pico": self.cost_pico,
            "fingerprint": self.fingerprint,
            "output_digest": self.output_digest,
        }


#: Subagent/child/job dispatch tool names — a SUCCESSFUL one means a
#: completed child/job (progress), regardless of the parent's prose.
_SUBAGENT_TOOL_NAMES = frozenset({"AgentSync", "AgentAsync", "delegate"})

#: Calls that manage the workflow but do not advance the goal. A turn made
#: only of these is the "productive-looking but stalled" pattern the
#: no-progress guard exists to catch: listing agents, rewriting the todo,
#: reading goal status, loading a skill, asking a question or compacting.
_ADMIN_TOOL_NAMES = frozenset(
    {
        "todowrite",
        "goal",
        "budget",
        "Skill",
        "question",
        "compact_context",
    }
)
_ADMIN_ACTIONS = frozenset({"list", "status", "abort", "steer"})


def _evaluator_outcome(verdict: Any) -> tuple[str, str, str]:
    """Map an evaluator verdict to (state kind, registry outcome, reason).

    ``waiting_owner`` is nonterminal: the registry records a pause with its own
    reason, the goal stays set, and no automatic turn is admitted. The owner's
    next message resumes it.
    """
    if getattr(verdict, "ok", False):
        return "met", "met", "condition_met"
    if getattr(verdict, "waiting_owner", False):
        return "waiting_owner", "paused", REASON_WAITING_OWNER
    if getattr(verdict, "impossible", False):
        return "impossible", "impossible", REASON_IMPOSSIBLE
    return "not_met", "not_met", ""


def _has_effective_progress(receipt: SupervisedTurnReceipt) -> bool:
    """True when the turn produced evidence of work, not just bookkeeping.

    Acceptance evidence (an artifact, a completed child/job) always counts.
    A tool call counts unless it is administrative: todo/goal/budget/status
    edits, skill loads, questions, compaction, memory reads, or a
    list/abort/steer control call. A failed call never counts.
    """
    if receipt.artifacts or receipt.child_job_transitions:
        return True
    for call in receipt.tool_calls:
        if not call.get("ok"):
            continue
        name = str(call.get("name") or "")
        action = str(call.get("action") or "").strip().lower()
        if name in _ADMIN_TOOL_NAMES or name.startswith("mcp__memory__"):
            continue
        if name in _SUBAGENT_TOOL_NAMES and action in _ADMIN_ACTIONS:
            continue
        return True
    return False


def _receipt_fingerprint(tool_calls: list[dict[str, Any]]) -> str:
    """Normalized hash of (tool name, action, success class) — the TIER 3b
    repeated-pattern key. Call ids, timestamps, arguments' prose and output
    content are deliberately excluded; a ``goal`` list vs a failed ``goal``
    archive vs a successful ``bash`` all hash differently."""
    parts = sorted(f"{c.get('name', '')}|{c.get('action', '')}|{1 if c.get('ok') else 0}" for c in tool_calls)
    if not parts:
        parts = ["<no_tools>"]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _output_digest(content: str) -> str:
    return hashlib.sha256((content or "").encode("utf-8")).hexdigest()


@dataclass
class GoalSupervisorSnapshot:
    goal_id: str
    active: bool
    turn_count: int
    # Display/back-compat only — the outer loop is cost-terminated, never
    # turn-count-terminated (see _drive_loop).
    max_turns: int


def build_goal_continuation_prompt(
    goal: GoalRecord,
    *,
    reason: str = "",
    root_session_state: dict[str, Any] | None = None,
) -> str:
    """The auto-turn prompt: the goal evaluator's "not met" reason as the
    guidance for the next turn (Claude Code ``/goal`` semantics)."""
    n = goal.turn_count + 1
    # The prompt reports the goal's own spend against the goal's own cap.
    from sharedagents_goal.goal_service import (
        _effective_goal_cap_usd,
        _goal_cost_pico,
    )

    spent = _goal_cost_pico(goal.goal_id) / 1_000_000_000_000
    budget = _effective_goal_cap_usd(goal)
    parts = [f"[goal auto-turn {n} · ${spent:.4f}/${budget:.2f}]"]
    if root_session_state:
        st = str(root_session_state.get("session_state") or "")
        if st:
            parts.append(f"Root session state: {st}.")
    parts.append("The goal is not met yet." + (f" Goal evaluator: {reason.strip()}" if reason.strip() else ""))
    parts.append(f"Keep working until this condition is met:\n{goal.condition_text}")
    parts.append(
        "After each turn the condition is checked against this conversation, "
        "so show the evidence (command output, file contents, results) in "
        "the conversation itself."
    )
    return "\n\n".join(parts)


class GoalSupervisor:
    def __init__(self, agent: Any) -> None:
        self._agent = agent
        self._tasks: dict[str, asyncio.Task] = {}
        self._claim_retry_tasks: dict[str, asyncio.Task] = {}
        self._skip_next_auto: set[str] = set()
        self._manual_turn_versions: dict[str, int] = {}
        # Per-goal count of in-process re-drives after a drive-task
        # crash, reset on any clean (non-crash) exit so it bounds crash-loops only.
        self._redrive_counts: dict[str, int] = {}
        # TIER 3a — per-goal turn-cost reservation (previous turn's actual
        # spend), reconciled to the receipt's cost delta after every turn.
        self._last_turn_cost: dict[str, int] = {}
        # goal_id -> seq of its supervised turn in flight (see ``cancel``).
        self._inflight_turns: dict[str, int] = {}
        self._turn_seq = 0
        self._owner_id = _OWNER_ID

    @property
    def cfg(self) -> Any:
        return self._agent.cfg

    def _enabled(self) -> bool:
        return bool(getattr(self.cfg, "goal_outer_loop_enabled", True))

    def _delay_s(self) -> float:
        return float(getattr(self.cfg, "goal_outer_loop_delay_s", 2.0) or 2.0)

    def _error_delay_s(self) -> float:
        return float(
            getattr(self.cfg, "goal_outer_loop_error_delay_s", 10.0) or 10.0,
        )

    def _max_errors(self) -> int:
        return int(getattr(self.cfg, "goal_outer_loop_max_errors", 3) or 3)

    def _max_redrive(self) -> int:
        """Bound on in-process crash re-drives per goal."""
        return int(
            getattr(self.cfg, "goal_supervisor_max_redrive", _DEFAULT_MAX_REDRIVE) or _DEFAULT_MAX_REDRIVE,
        )

    def _no_progress_turns(self) -> int:
        """Stop after this many consecutive goal turns without a tool call."""
        return int(
            getattr(self.cfg, "goal_no_progress_turns", _DEFAULT_NO_PROGRESS_TURNS) or _DEFAULT_NO_PROGRESS_TURNS,
        )

    def _reserve_last_turn(self) -> bool:
        """TIER 3a — admit a new turn only while committed spend + the previous
        turn's actual cost (the reservation) stays under the budget."""
        return bool(getattr(self.cfg, "goal_outer_loop_reserve_last_turn", True))

    def _compact_directive_enabled(self) -> bool:
        """TIER 3g — append a compact-first directive when prompt size is
        growing and the progress streak is nonzero."""
        return bool(getattr(self.cfg, "goal_outer_loop_compact_directive", True))

    @staticmethod
    def _budget(goal: GoalRecord, session: Any) -> GoalBudgetState:
        """Both budget scopes for ``goal``: its own spend against its own cap
        and the session tree's spend against the session cap. The same reading
        the turn finalizer and the top-up commands use."""
        from sharedagents_goal.goal_service import goal_budget_state

        return goal_budget_state(goal, session)

    @staticmethod
    def _goal_spent_pico(goal_id: str) -> int:
        """Spend attributed to the goal (its own ledger rows)."""
        try:
            from sharedagents_goal.goal_service import _goal_cost_pico

            return int(_goal_cost_pico(goal_id))
        except Exception:  # noqa: BLE001
            return 0

    def notify_manual_turn_start(self, session_id: str) -> None:
        """Record that a manual turn started, so the outer loop can skip its
        very next auto-turn — only when
        ``goal_outer_loop_manual_interrupt_policy`` is the (default, and only
        implemented) ``"yield"`` value.
        """
        self._manual_turn_versions[session_id] = self._manual_turn_versions.get(session_id, 0) + 1
        policy = str(
            getattr(self.cfg, "goal_outer_loop_manual_interrupt_policy", "yield") or "yield",
        )
        if policy == "yield":
            self._skip_next_auto.add(session_id)

    def is_active(self, goal_id: str) -> bool:
        task = self._tasks.get(goal_id)
        if task is None:
            return False
        if task.done():
            self._tasks.pop(goal_id, None)
            return False
        cancelling = getattr(task, "cancelling", None)
        return not (callable(cancelling) and cancelling() > 0)

    def snapshot(self, goal_id: str) -> GoalSupervisorSnapshot:
        reg = get_goal_registry()
        goal = reg.get(goal_id)
        if goal is None:
            return GoalSupervisorSnapshot(goal_id, False, 0, 0)
        return GoalSupervisorSnapshot(
            goal_id=goal_id,
            active=self.is_active(goal_id),
            turn_count=goal.turn_count,
            max_turns=goal.max_turns,
        )

    def _renew_lease(self, goal_id: str) -> bool:
        try:
            return get_goal_registry().touch_supervisor(
                goal_id,
                owner_id=self._owner_id,
            )
        except Exception:  # noqa: BLE001
            return False

    async def _sleep_with_lease(self, goal_id: str, delay_s: float) -> bool:
        """Sleep in chunks, renewing the supervisor lease so long delays stay owned."""
        remaining = float(delay_s)
        while remaining > 0:
            if not self._renew_lease(goal_id):
                return False
            chunk = min(_LEASE_RENEW_INTERVAL_S, remaining)
            await asyncio.sleep(chunk)
            remaining -= chunk
        return self._renew_lease(goal_id)

    def _work_wait_poll_s(self) -> float:
        """Poll interval while holding auto-turns for live delegated work."""
        return float(
            getattr(self.cfg, "goal_outer_loop_work_wait_s", 5.0) or 5.0,
        )

    async def _wait_while_subtree_live(
        self,
        goal_id: str,
        sid: str,
        *,
        yield_to_manual: bool = True,
    ) -> bool:
        """Hold the outer loop (no auto-turn) while delegated subtree work is live.

        This is the doom-loop guard for jarvis-style orchestration: the root
        turn ends after dispatching background Agent children / async jobs,
        and without this wait the supervisor would fire another
        ``[goal auto-turn N]`` every ``goal_outer_loop_delay_s`` even though
        the model is correctly waiting for child results. Uses the same
        :func:`session_work_live` probe as the web/CLI reconcilers.

        The budget is NOT read here. Work that is running was admitted before
        any limit was reached; it is left to finish and the goal stays running
        for as long as it does. The drive loop reads the budget once the
        subtree is idle and settles a pause then.

        Returns ``False`` when the driver should exit: the lease was lost, or
        the goal left ``RUNNING`` — an operator pause/abort/clear takes effect
        on the next poll, whatever is still live. Returns ``True`` when the
        subtree is idle — or, with ``yield_to_manual``, a manual-yield skip
        is pending — so the caller can proceed with its normal checks.
        """
        from sharedagents_goal.goal_registry import session_work_live

        poll_s = self._work_wait_poll_s()
        while True:
            if yield_to_manual and sid in self._skip_next_auto:
                return True
            try:
                live, src = session_work_live(sid)
            except Exception:  # noqa: BLE001 — probe must never break the loop
                live, src = False, ""
            if not live:
                return True

            reg = get_goal_registry()
            goal = reg.get(goal_id)
            if goal is None or goal.status != STATUS_RUNNING:
                return False

            _LOG.info(
                "goal %s holding auto-turn: subtree work still live (source=%s) under %s",
                goal_id[:16],
                src or "?",
                sid[:16],
            )
            if not await self._sleep_with_lease(goal_id, poll_s):
                return False

    async def _settle_budget_limit(self, goal_id: str, sid: str) -> bool:
        """A budget limit is reached: admit nothing more, let the work already
        admitted stop, then pause the goal.

        Reaching a limit closes admission, it does not end the work in
        flight. Pausing while children run would report the goal stopped while
        it is still spending and still producing its result, so the pause is
        written only once the subtree is idle.

        Returns ``True`` when the limit no longer holds by then (the cap was
        raised while waiting) — the caller re-runs its checkpoint. Returns
        ``False`` when the driver should exit: the goal was paused here, or it
        left ``RUNNING`` / the lease was lost during the wait.
        """
        if not await self._wait_while_subtree_live(
            goal_id,
            sid,
            yield_to_manual=False,
        ):
            return False
        goal = get_goal_registry().get(goal_id)
        if goal is None or goal.status != STATUS_RUNNING:
            return False
        session = await self._load_session(sid)
        budget = self._budget(goal, session)
        scope = budget.exhausted_scope()
        if not scope:
            return True
        _LOG.info(
            "goal %s exit: %s cost budget exhausted (goal %d/%d pico, session %d/%d pico)",
            goal_id[:16],
            scope,
            budget.goal_spent_pico,
            budget.goal_cap_pico,
            budget.tree_spent_pico,
            budget.tree_cap_pico,
        )
        self._pause_for_budget(goal_id, session)
        return False

    def _pause_for_budget(self, goal_id: str, session: Any) -> None:
        """Pause ``goal_id`` for cost and project the record the registry holds."""
        goal = get_goal_registry().pause_goal(
            goal_id,
            reason="cost_budget_exhausted",
        )
        if session is not None:
            sync_session_goal_projection(session, goal)

    async def _run_supervised_turn(
        self,
        sid: str,
        *,
        goal_id: str,
        prompt: str,
    ) -> SupervisedTurnReceipt:
        """Run one agent turn, renewing the lease while it is in flight, and
        return a full :class:`SupervisedTurnReceipt` — the TIER 1a evidence
        the drive loop classifies into effective progress vs. a repeated
        no-op pattern.

        TIER 2e — the goal id is threaded into ``Agent.run_turn`` as an
        explicit turn context (not read back from mutable session fields), so
        the cost finalizer attributes this turn to the goal even if a
        projection write clears ``session.goal_id`` mid-flight.

        TIER 3c — a launch that never yields a TurnDone is reported as an
        error (``saw_turn_done=False``); it is never progress.
        """
        renew = asyncio.create_task(
            self._lease_renewal_loop(goal_id),
            name=f"goal_lease_{goal_id[:12]}",
        )
        receipt = SupervisedTurnReceipt()
        spent_before = self._goal_spent_pico(goal_id)
        pending: dict[str, tuple[str, str]] = {}  # call_id -> (name, action)
        content_chunks: list[str] = []
        # Mark this turn so a ``goal`` tool call made inside it is recognised
        # by ``cancel`` as the turn itself. Set after ``renew`` is created so
        # the lease task does not carry the mark.
        self._turn_seq += 1
        turn_seq = self._turn_seq
        self._inflight_turns[goal_id] = turn_seq
        turn_mark = _SUPERVISED_TURN.set((goal_id, turn_seq))
        try:
            from sharedagents_agent.agents.engine import (
                ArtifactReference,
                TextDelta,
                ToolCallResult,
                ToolCallStarted,
                TurnDone,
                TurnError,
                TurnStarted,
            )

            async for ev in self._agent.run_turn(
                sid,
                prompt,
                source="goal_supervisor",
                goal_id=goal_id,
            ):
                if isinstance(ev, TurnStarted):
                    receipt.turn_id = ev.turn_id
                elif isinstance(ev, TurnDone):
                    receipt.saw_turn_done = True
                    receipt.turn_state = ev.state or "done"
                elif isinstance(ev, TurnError):
                    # Non-fatal mid-turn error — recorded; state still follows
                    # TurnDone (or the missing-TurnDone error path below).
                    if receipt.turn_state == "error":
                        receipt.turn_state = "error"
                elif isinstance(ev, TextDelta):
                    if ev.channel in ("", "content"):
                        content_chunks.append(ev.text)
                elif isinstance(ev, ToolCallStarted):
                    action = ""
                    if isinstance(ev.arguments, dict):
                        action = str(ev.arguments.get("action", "") or "")
                    pending[ev.call_id] = (ev.name, action)
                elif isinstance(ev, ToolCallResult):
                    name, action = pending.pop(ev.call_id, (ev.call_id, ""))
                    receipt.tool_calls.append(
                        {
                            "name": name,
                            "action": action,
                            "ok": bool(ev.ok),
                        }
                    )
                    if bool(ev.ok) and name in _SUBAGENT_TOOL_NAMES:
                        receipt.child_job_transitions += 1
                elif isinstance(ev, ArtifactReference):
                    uri = ev.uri or ev.path or ev.locator or ""
                    if uri:
                        receipt.artifacts.append(uri)
        finally:
            _SUPERVISED_TURN.reset(turn_mark)
            if self._inflight_turns.get(goal_id) == turn_seq:
                self._inflight_turns.pop(goal_id, None)
            renew.cancel()
            try:
                await renew
            except asyncio.CancelledError:
                pass
        receipt.output_digest = _output_digest("".join(content_chunks))
        receipt.fingerprint = _receipt_fingerprint(receipt.tool_calls)
        receipt.cost_pico = max(0, self._goal_spent_pico(goal_id) - spent_before)
        return receipt

    async def _lease_renewal_loop(self, goal_id: str) -> None:
        try:
            while True:
                await asyncio.sleep(_LEASE_RENEW_INTERVAL_S)
                if not self._renew_lease(goal_id):
                    return
        except asyncio.CancelledError:
            pass

    def _schedule_claim_retry(self, goal_id: str, *, source: str) -> None:
        if goal_id in self._claim_retry_tasks:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        async def _retry() -> None:
            try:
                await asyncio.sleep(_SUPERVISOR_LEASE_S + 5.0)
                if not self.is_active(goal_id):
                    self.ensure_started(goal_id, source=f"{source}:lease_retry")
            finally:
                self._claim_retry_tasks.pop(goal_id, None)

        self._claim_retry_tasks[goal_id] = loop.create_task(
            _retry(),
            name=f"goal_claim_retry_{goal_id[:16]}",
        )

    def ensure_started(
        self,
        goal_id: str,
        *,
        source: str = "turn_done",
    ) -> None:
        """Claim the supervisor lease for ``goal_id`` and start its drive
        task, if not already active; on a lost claim race, schedule a
        bounded retry instead of double-driving the goal.
        """
        if not self._enabled() or not goal_id:
            return
        if self.is_active(goal_id):
            return
        stale = self._tasks.get(goal_id)
        if stale is not None and not stale.done():
            stale.cancel()
        reg = get_goal_registry()
        if not reg.try_claim_supervisor(goal_id, self._owner_id):
            self._schedule_claim_retry(goal_id, source=source)
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            reg.release_supervisor(goal_id, self._owner_id)
            return
        task = loop.create_task(
            self._drive_loop(goal_id, source=source),
            name=f"goal_supervisor_{goal_id[:20]}",
        )
        self._tasks[goal_id] = task
        # Root-cause crash resilience. If the drive task CRASHES
        # (raises) mid-goal, the goal is left ``running`` with no supervisor until
        # the next process restart re-runs recovery — the defect the zombie
        # reconciler exists to paper over. Re-drive in-process (bounded) so a
        # crashed supervisor's goal resumes without waiting for a restart.
        task.add_done_callback(
            lambda t, gid=goal_id: self._on_drive_done(gid, t),
        )

    def _on_drive_done(self, goal_id: str, task: asyncio.Task) -> None:
        """Drive-task completion callback.

        Re-drives ONLY on an abnormal crash — the task finished by raising a
        non-``CancelledError`` exception — and only while the goal is still
        ``running`` and the per-goal re-drive bound is not exhausted. A clean
        return (goal reached a terminal state, was cancelled, or hit the cost /
        error backstop) is a deliberate stop and never re-drives; it resets the
        counter so a later independent crash gets the full budget. Bounded so a
        deterministically-crashing goal falls through to the reconciler / next-
        process recovery instead of hot-looping.

        The crashed task's own ``finally`` already released the supervisor lease
        and popped ``_tasks``; ``ensure_started`` re-claims cleanly from here (we
        run on the event loop, inside the done-callback).
        """
        if task.cancelled():
            return
        try:
            exc = task.exception()
        except Exception:  # noqa: BLE001 — cancelled/loop-closing
            return
        if exc is None:
            # Clean exit — the supervisor decided to stop. Clear any crash count.
            self._redrive_counts.pop(goal_id, None)
            return
        # Abnormal crash. Bound the re-drives.
        n = self._redrive_counts.get(goal_id, 0)
        if n >= self._max_redrive():
            # Exhausted: leave to the reconciler / next-process recovery. Keep the
            # count (do NOT reset) so subsequent crashes of the same goal stay
            # blocked here rather than restarting the burst — a clean exit is the
            # only thing that resets the counter.
            _LOG.error(
                "goal supervisor drive crashed for %s and exhausted %d re-drives; "
                "leaving to reconciler/next-process recovery: %r",
                goal_id,
                n,
                exc,
            )
            return
        try:
            reg = get_goal_registry()
            goal = reg.get(goal_id)
        except Exception:  # noqa: BLE001
            return
        if goal is None or goal.status != STATUS_RUNNING:
            # Already terminal/paused — nothing to resume.
            self._redrive_counts.pop(goal_id, None)
            return
        self._redrive_counts[goal_id] = n + 1
        _LOG.warning(
            "goal supervisor drive crashed for %s (re-drive %d/%d): %r",
            goal_id,
            n + 1,
            self._max_redrive(),
            exc,
        )
        try:
            self.ensure_started(goal_id, source="redrive")
        except Exception:  # noqa: BLE001 — recovery/reconciler remain as the net
            _LOG.exception("goal supervisor re-drive failed for %s", goal_id)

    def cancel(
        self,
        goal_id: str,
        *,
        reason: str = "cancelled",
        pause: bool = True,
        release_lease: bool = True,
    ) -> None:
        task = self._tasks.get(goal_id)
        # The in-flight auto-turn ends its own goal with the ``goal`` tool
        # (met/pause/clear, as its prompt instructs). Cancelling ourselves
        # aborts that turn with ``turn interrupted before completion`` and
        # paints Chat as failed even though the transition succeeded. The
        # drive loop already exits on the next checkpoint when
        # status != RUNNING. The tool runner executes each tool in its own
        # task, so ``current_task()`` is not the drive task there; the
        # per-turn mark it inherits identifies it instead.
        in_own_drive = task is not None and (task is asyncio.current_task() or self._called_from_own_turn(goal_id))
        if task is not None and not task.done() and not in_own_drive:
            task.cancel()
        retry = self._claim_retry_tasks.pop(goal_id, None)
        if retry is not None and not retry.done():
            retry.cancel()
        if release_lease and not in_own_drive:
            try:
                get_goal_registry().release_supervisor(goal_id, self._owner_id)
            except Exception:  # noqa: BLE001
                pass
        if goal_id and pause and not in_own_drive:
            try:
                reg = get_goal_registry()
                goal = reg.get(goal_id)
                if goal and goal.status == STATUS_RUNNING:
                    reg.pause_goal(goal_id, reason=reason)
            except Exception:  # noqa: BLE001
                pass

    def _called_from_own_turn(self, goal_id: str) -> bool:
        """True when the caller runs inside ``goal_id``'s in-flight supervised turn.

        The mark only counts while that same turn is still in flight: a task
        the turn left running (an AgentAsync child, say) inherits the mark but
        is not the turn once it has ended.
        """
        mark = _SUPERVISED_TURN.get()
        return mark is not None and mark[0] == goal_id and self._inflight_turns.get(goal_id) == mark[1]

    def cancel_for_session(self, session_id: str, *, reason: str = "cancelled") -> None:
        reg = get_goal_registry()
        for goal in reg.list_goals():
            if goal.root_session_id == session_id and goal.status == STATUS_RUNNING:
                self.cancel(goal.goal_id, reason=reason)

    async def recover_running_goals(self) -> int:
        """Re-start supervision for running goals owned by this agent.

        Runs once per constructed agent (idempotent, non-blocking) so a
        supervisor that crashed with the process — leaving a goal ``running``
        under a now-stale lease — is re-driven by the next process.

        TIER 1d / TIER 3d / E1 — recovery is ELIGIBLE only for roots that
        exist and are not explicitly ended: a goal whose root session state
        is ``closed`` / ``error`` (the projector folds an operator-aborted
        ``turn_ended`` into ``error``) is auto-paused instead of revived —
        explicit session termination outranks a stale running row. Ordinary
        ``idle`` is NOT ended: an idle-running root is exactly what recovery
        should reclaim. An explicit ``resume`` / ``restart`` (``source=``
        ``resume``/``restart``) is NOT routed through recovery and always
        overrides.
        """
        reg = get_goal_registry()
        count = 0
        for goal in reg.list_goals():
            if goal.agent_name != self._agent.name:
                continue
            if goal.status != STATUS_RUNNING:
                continue
            if not goal.root_session_id:
                continue
            state = root_session_state(goal.root_session_id)
            if state is None:
                # Root row does not exist — the durable binding is broken.
                _LOG.info(
                    "goal %s recovery skip: root session %s missing — pausing",
                    goal.goal_id[:16],
                    goal.root_session_id[:16],
                )
                reg.pause_goal(goal.goal_id, reason="root_session_missing")
                count += 1
                continue
            if bool(state.get("probe_error")):
                _LOG.warning(
                    "goal %s recovery deferred: root session %s state probe failed",
                    goal.goal_id[:16],
                    goal.root_session_id[:16],
                )
                continue
            if str(state.get("session_state") or "") in SESSION_ENDED_STATES:
                _LOG.info(
                    "goal %s recovery skip: root session %s ended (state=%s) — pausing instead of reviving",
                    goal.goal_id[:16],
                    goal.root_session_id[:16],
                    state.get("session_state"),
                )
                reg.pause_goal(goal.goal_id, reason="root_session_ended")
                count += 1
                continue
            _LOG.info(
                "goal %s recovery candidate: root %s state=%s — starting",
                goal.goal_id[:16],
                goal.root_session_id[:16],
                state.get("session_state"),
            )
            self.ensure_started(goal.goal_id, source="recovery")
            count += 1
        return count

    async def _drive_loop(self, goal_id: str, *, source: str) -> None:
        """Drive one goal's outer loop: on each iteration check status and
        terminal state, enforce the cost budget, honor a pending
        manual-interrupt skip, then sleep (renewing the supervisor lease) and
        run the next auto-turn — until the goal reaches a terminal state, is
        cancelled, or the error/lease backstops fire. Releases the
        supervisor lease in ``finally`` on any clean exit.

        COST TERMINATES, NEVER A TURN COUNT. The only spend/lifecycle
        terminators here are the cost budget (the goal's own spend against
        ``max_cost_pico`` and the session tree's spend against the session
        cap — two scopes, never cross-compared; see ``_budget``),
        terminal state, cancellation, and the consecutive-error backstop.
        ``goal.max_turns`` is deliberately NOT consulted as a stop condition:
        there is no ``turn_count >= max_turns`` branch in this loop. The stored
        ``max_turns`` is advisory/display-only (schema + "auto-turn N/M" label)
        and never terminates the loop.
        override 2026-07-11 "only cost limit no turn limit"); the governing
        cost-not-turns rule is owned by requirements/17-goals.md §3.

        goal may overshoot its budget by one full turn's cost; see
        spec/as-built/goals.md Defect D2)
        MID-FLIGHT CRASH — see spec/as-built/goals.md Defects D1, D6)

        Auto-turns are also deferred while ``session_work_live(root)`` is
        true — see :meth:`_wait_while_subtree_live`. Without that hold, a
        root that ends a turn after dispatching background children would
        burn a ``[goal auto-turn N]`` every ``goal_outer_loop_delay_s``
        ("doom loop") while correctly waiting for child results. A budget
        limit reached while that work is live closes admission but does not
        pause the goal until the work stops — see
        :meth:`_settle_budget_limit`.
        """
        reg = get_goal_registry()
        consecutive_errors = 0
        failure_reason = "turns ended without a result"
        me = asyncio.current_task()
        last_prompt_tokens = 0
        try:
            while True:
                try:
                    reg.expire_due()
                except Exception:  # noqa: BLE001
                    pass
                goal = reg.get(goal_id)
                if goal is None or goal.status != STATUS_RUNNING:
                    _LOG.info("goal %s exit: not running anymore", goal_id[:16])
                    return
                if not goal.root_session_id:
                    _LOG.info("goal %s exit: no root session binding", goal_id[:16])
                    return

                sid = goal.root_session_id
                session = await self._load_session(sid)
                if session is None:
                    _LOG.info("goal %s exit: root session %s missing", goal_id[:16], sid[:16])
                    reg.pause_goal(goal_id, reason="root_session_missing")
                    return

                # E1 — supervisor session awareness: refuse to auto-turn a
                # root session that has been explicitly CLOSED. The durable
                # state register outranks the stale running goal row.
                state = root_session_state(sid)
                if state is not None and str(state.get("session_state") or "") == "closed":
                    _LOG.info(
                        "goal %s exit: root session %s closed — pausing root_session_ended (E1)",
                        goal_id[:16],
                        sid[:16],
                    )
                    session.goal_terminal = "paused"
                    reg.pause_goal(goal_id, reason="root_session_ended")
                    sync_session_goal_projection(session, reg.get(goal_id))
                    return
                if state is not None:
                    last_prompt_tokens = int(
                        state.get("last_prompt_tokens") or last_prompt_tokens or 0,
                    )

                goal_id_on_sess = str(getattr(session, "goal_id", "") or "")
                if goal_id_on_sess and goal_id_on_sess != goal_id:
                    _LOG.info(
                        "goal %s exit: session bound to different goal %s",
                        goal_id[:16],
                        goal_id_on_sess[:16],
                    )
                    return

                goal = self._sync_terminal_to_registry(goal_id, "", session)
                if goal is None or goal.status != STATUS_RUNNING:
                    return

                if self._budget(goal, session).exhausted_scope():
                    # No further turn is admitted. Children this goal already
                    # started may still be running: the pause waits for them.
                    if await self._settle_budget_limit(goal_id, sid):
                        continue
                    return

                if sid in self._skip_next_auto:
                    # Operator input (manual turn) — the owner is steering; the
                    # goal's very next auto-turn is suppressed. Operator input
                    # is effective progress (TIER 1b): reset the streak.
                    self._skip_next_auto.discard(sid)
                    consecutive_errors = 0
                    _LOG.info(
                        "goal %s skip next auto-turn (manual turn) — resetting no-progress streak",
                        goal_id[:16],
                    )
                    reg.update_effective_progress(
                        goal_id,
                        streak=0,
                        last_effective_progress_at=_utcnow_iso(),
                    )
                    continue

                reg.touch_supervisor(goal_id, owner_id=self._owner_id)
                if not await self._sleep_with_lease(goal_id, self._delay_s()):
                    _LOG.info("goal %s exit: lease lost while sleeping", goal_id[:16])
                    return

                goal = reg.get(goal_id)
                if goal is None or goal.status != STATUS_RUNNING:
                    _LOG.info("goal %s exit: not running after sleep", goal_id[:16])
                    return

                # Doom-loop guard: if the previous turn (or a concurrent
                # dispatch) left background Agent children / async jobs
                # running under this root, hold here — renewing the lease —
                # until that work finishes. Do NOT burn another auto-turn
                # just to restate "still waiting".
                if not await self._wait_while_subtree_live(goal_id, sid):
                    _LOG.info("goal %s exit: subtree hold released (stop)", goal_id[:16])
                    return
                if sid in self._skip_next_auto:
                    self._skip_next_auto.discard(sid)
                    consecutive_errors = 0
                    _LOG.info(
                        "goal %s skip auto-turn after subtree hold (manual turn) — resetting no-progress streak",
                        goal_id[:16],
                    )
                    reg.update_effective_progress(
                        goal_id,
                        streak=0,
                        last_effective_progress_at=_utcnow_iso(),
                    )
                    continue
                goal = reg.get(goal_id)
                if goal is None or goal.status != STATUS_RUNNING:
                    _LOG.info("goal %s exit: not running after subtree hold", goal_id[:16])
                    return

                # Claude Code /goal: judge the condition against the transcript
                # after every turn. Met / impossible / an evaluator failure end
                # the drive here; "not met" hands its reason to the next turn.
                outcome, reason = await self._evaluate(goal_id, sid)
                if outcome == "retry":
                    continue
                if outcome != "not_met":
                    return
                goal = reg.get(goal_id)
                if goal is None or goal.status != STATUS_RUNNING:
                    return

                # Judge the last turn before enforcing retry/no-tool guards:
                # that turn may itself contain the evidence of success.
                if consecutive_errors >= self._max_errors():
                    self._auto_pause(
                        goal_id,
                        sid,
                        REASON_TURN_ERRORS,
                        f"Goal paused after {consecutive_errors} automatic "
                        f"retries ({failure_reason}). Send a message to retry.",
                    )
                    return
                streak = int(goal.no_effective_progress_streak or 0)
                if streak >= self._no_progress_turns():
                    self._auto_pause(
                        goal_id,
                        sid,
                        REASON_NO_PROGRESS,
                        f"Goal paused: {streak} turns in a row without using a tool. Send a message to continue.",
                    )
                    return

                # TIER 3a — budget admission with reservation: committed spend
                # PLUS the previous turn's actual cost (the reservation) must
                # fit the budget before another turn is admitted. The
                # reservation is reconciled to actual spend by the receipt's
                # cost delta at the end of the turn. Read fresh: children may
                # have spent during the subtree hold, and the subtree is idle
                # now, so a refusal pauses with nothing left running.
                budget = self._budget(goal, session)
                reserved_pico = self._last_turn_cost.get(goal_id, 0) if self._reserve_last_turn() else 0
                scope = budget.exhausted_scope(reserved_pico)
                if scope:
                    _LOG.info(
                        "goal %s exit: %s budget admission refused (goal %d/%d pico, session %d/%d pico, reserved=%d)",
                        goal_id[:16],
                        scope,
                        budget.goal_spent_pico,
                        budget.goal_cap_pico,
                        budget.tree_spent_pico,
                        budget.tree_cap_pico,
                        reserved_pico,
                    )
                    self._pause_for_budget(goal_id, session)
                    return

                prompt = build_goal_continuation_prompt(
                    goal,
                    reason=reason,
                    root_session_state=state,
                )
                # TIER 3g — context damage control: when the prompt size is
                # growing and the streak is nonzero, direct the model to
                # squeeze first. Reduces burn; never terminates the loop.
                if (
                    self._compact_directive_enabled()
                    and last_prompt_tokens > 0
                    and goal.no_effective_progress_streak > 0
                ):
                    prompt += (
                        "\n\nContext is growing and no effective progress was "
                        "made recently: call compact_context as your FIRST "
                        "action this turn, then continue working."
                    )
                self._emit_goal_auto_turn_started(
                    sid,
                    goal_id=goal_id,
                    goal=goal,
                    root_session_state=state,
                )
                _LOG.info(
                    "goal %s turn admit: goal %d/%d pico session %d/%d pico streak=%d",
                    goal_id[:16],
                    budget.goal_spent_pico,
                    budget.goal_cap_pico,
                    budget.tree_spent_pico,
                    budget.tree_cap_pico,
                    goal.no_effective_progress_streak,
                )
                receipt = SupervisedTurnReceipt()
                failure_reason = "turns ended without a result"
                try:
                    receipt = await self._run_supervised_turn(
                        sid,
                        goal_id=goal_id,
                        prompt=prompt,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    _LOG.warning("goal supervisor turn failed: %s", exc)
                    receipt = SupervisedTurnReceipt(turn_state="error")
                    failure_reason = f"turn errors: {exc}"

                # TIER 3c — a supervised launch must yield exactly one
                # TurnDone; a missing/zero-event launch is an error, never
                # progress.
                if not receipt.saw_turn_done or receipt.turn_state == "error":
                    _LOG.warning(
                        "goal %s turn %s: no terminal TurnDone (state=%s) — counting as error",
                        goal_id[:16],
                        receipt.turn_id or "?",
                        receipt.turn_state,
                    )
                    consecutive_errors += 1
                    self._last_turn_cost[goal_id] = 0
                    if not await self._sleep_with_lease(goal_id, self._error_delay_s()):
                        _LOG.info("goal %s exit: lease lost after bad turn", goal_id[:16])
                        return
                    continue

                consecutive_errors = 0
                _LOG.info(
                    "goal %s turn done: state=%s cost_delta=%d pico tools=%d fingerprint=%s",
                    goal_id[:16],
                    receipt.turn_state,
                    receipt.cost_pico,
                    len(receipt.tool_calls),
                    (receipt.fingerprint or "")[:16],
                )
                self._last_turn_cost[goal_id] = receipt.cost_pico

                session = await self._load_session(sid)
                if session is None:
                    _LOG.info("goal %s exit: root session vanished after turn", goal_id[:16])
                    return
                goal = self._sync_terminal_to_registry(goal_id, "", session)
                if goal is None or goal.status != STATUS_RUNNING:
                    _LOG.info(
                        "goal %s exit: terminal %s after turn %s",
                        goal_id[:16],
                        goal.status if goal else "deleted",
                        receipt.turn_id or "?",
                    )
                    receipt.terminal_transition = goal.status if goal else "deleted"
                    return

                goal = reg.get(goal_id)
                if goal:
                    tc = int(getattr(session, "goal_turn_count", 0) or 0)
                    # TIER 2c — never regress the durable counter: a
                    # projection split-brain (lazy loader restoring an empty
                    # turn count) must not reset 12 → 1 in goals.db.
                    if tc < int(goal.turn_count or 0):
                        _LOG.warning(
                            "goal %s copy-back guard: session turn_count %d < registry %d — keeping registry value",
                            goal_id[:16],
                            tc,
                            int(goal.turn_count or 0),
                        )
                    else:
                        reg.update_turn_count(goal_id, tc)
                    # TIER 2e — refresh the display cost cache from the
                    # authoritative ledger SUM after finalization.
                    try:
                        reg.update_cost_cache(goal_id, self._goal_spent_pico(goal_id))
                    except Exception:  # noqa: BLE001
                        pass
                    # Claude Code no-progress guard: consecutive goal turns
                    # without EFFECTIVE progress. Administrative calls
                    # (todo/goal/budget/skill/question/compaction) and
                    # list/steer controls do not reset it; an artifact or a
                    # completed child/job does. A manual turn resets it.
                    used_tools = _has_effective_progress(receipt)
                    streak = 0 if used_tools else int(goal.no_effective_progress_streak or 0) + 1
                    reg.update_effective_progress(
                        goal_id,
                        streak=streak,
                        last_effective_progress_at=_utcnow_iso() if used_tools else None,
                        last_turn_fingerprint=receipt.fingerprint,
                    )
        except asyncio.CancelledError:
            pass
        finally:
            if self._tasks.get(goal_id) is me:
                self._tasks.pop(goal_id, None)
                try:
                    get_goal_registry().release_supervisor(goal_id, self._owner_id)
                except Exception:  # noqa: BLE001
                    pass

    def _sync_terminal_to_registry(
        self,
        goal_id: str,
        terminal: str,
        session: Any,
        receipt: SupervisedTurnReceipt | None = None,
    ) -> GoalRecord | None:
        """Refresh the projection from the committed claim/owner decision.

        ``terminal`` and ``receipt`` are observations of an earlier turn. They
        cannot override an owner's later reject, resume or confirmation.
        """
        goal = get_goal_registry().get(goal_id)
        sync_session_goal_projection(session, goal)
        return goal

    def _alert_owner(
        self,
        goal_id: str,
        sid: str,
        kind: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Tell the owner something about a running goal without stopping it:
        one ``goal_events`` row plus one lifecycle marker on the root session."""
        body = {"goal_id": goal_id, "message": message, "source": "goal_supervisor", **(payload or {})}
        try:
            get_goal_registry().append_event(goal_id, kind, body)
        except Exception:  # noqa: BLE001 — an alert must never break the loop
            _LOG.exception("goal %s alert %s failed", goal_id[:16], kind)
        try:
            from sharedagents_session.session_writer import get_session_writer

            get_session_writer(self._agent.sessions_dir, sid).lifecycle(
                kind=kind,
                payload=body,
            )
        except Exception:  # noqa: BLE001 — best-effort lifecycle marker
            pass
        _LOG.info("goal %s alert %s: %s", goal_id[:16], kind, message)

    def _auto_pause(
        self,
        goal_id: str,
        sid: str,
        reason: str,
        notice: str,
        *,
        expected: GoalRecord | None = None,
    ) -> bool:
        """Pause only the running goal snapshot this supervisor still owns."""
        reg = get_goal_registry()
        expected = expected or reg.get(goal_id)
        if expected is None:
            return False
        try:
            goal = reg.commit_evaluator_update(
                expected,
                owner_id=self._owner_id,
                state_updates={"notice": notice},
                outcome="paused",
                reason=reason,
            )
        except GoalContractChangedError:
            return False
        self._alert_owner(goal_id, sid, f"goal_paused_{reason}", notice)
        try:
            session = self._agent.get_session(sid)
            if session is not None:
                sync_session_goal_projection(session, goal)
        except Exception:  # noqa: BLE001
            pass
        return True

    async def _evaluate(self, goal_id: str, sid: str) -> tuple[str, str]:
        renew = asyncio.create_task(self._lease_renewal_loop(goal_id))
        try:
            return await self._evaluate_owned(goal_id, sid)
        finally:
            renew.cancel()
            await asyncio.gather(renew, return_exceptions=True)

    async def _evaluate_owned(self, goal_id: str, sid: str) -> tuple[str, str]:
        """Run the goal evaluator on the session transcript.

        Returns ``(outcome, reason)``; ``outcome`` is ``not_met`` (keep going),
        ``met``, ``impossible``, ``waiting_owner`` (nonterminal pause: the
        remaining step needs the owner), ``paused`` (transient evaluator
        failures exhausted their retries) or ``cleared`` (unrecoverable
        evaluator error — the goal is removed, as Claude Code does).
        """
        from sharedagents_goal import goal_evaluator as ge

        reg = get_goal_registry()
        goal = reg.get(goal_id)
        if goal is None or goal.status != STATUS_RUNNING:
            return "gone", ""
        manual_version = self._manual_turn_versions.get(sid, 0)

        def manual_changed() -> bool:
            return self._manual_turn_versions.get(sid, 0) != manual_version or sid in self._skip_next_auto

        loop = asyncio.get_running_loop()
        try:
            transcript, turns = await loop.run_in_executor(
                None,
                ge.session_transcript,
                getattr(self._agent, "sessions_dir", ""),
                sid,
            )
        except Exception as exc:  # noqa: BLE001 — unreadable log: nothing to judge
            _LOG.warning("goal %s transcript read failed: %s", goal_id[:16], exc)
            transcript, turns = "", 0
        if manual_changed():
            return "retry", ""
        state = goal.contract or {}
        transcript_digest = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
        same_turn = int(state.get("evaluated_turns", -1)) == turns
        same_transcript = state.get("evaluated_transcript_sha256") == transcript_digest
        cached_kind = state.get("verdict")
        cached_terminal = bool(
            transcript.strip()
            and same_turn
            and cached_kind in ("met", "impossible")
            and (same_transcript or not state.get("evaluated_transcript_sha256"))
        )
        if transcript.strip() and same_turn and same_transcript and cached_kind == "not_met":
            return "not_met", str(state.get("last_check") or "")

        # Older supervisors wrote a verdict before its terminal transition.
        # Recovery must finish that durable decision instead of re-judging it.
        verdict = (
            ge.Verdict(
                ok=cached_kind == "met",
                impossible=cached_kind == "impossible",
                reason=str(state.get("last_check") or ""),
            )
            if cached_terminal
            else None
        )
        last_err: ge.EvaluatorError | None = None
        for attempt in range(0 if cached_terminal else ge.TRANSIENT_RETRIES):
            try:
                if not transcript.strip():
                    # Never drive a goal nobody can judge: an unreadable
                    # session log is an evaluator failure, not "not met".
                    raise ge.EvaluatorError("session transcript is empty or unreadable")
                verdict = await loop.run_in_executor(
                    None,
                    lambda: ge.evaluate(goal.condition_text, transcript, cfg=self.cfg),
                )
                break
            except ge.EvaluatorError as exc:
                if manual_changed():
                    return "retry", ""
                last_err = exc
                _LOG.warning("goal %s evaluator error (attempt %d): %s", goal_id[:16], attempt + 1, exc)
                if exc.unrecoverable:
                    break
                if attempt + 1 < ge.TRANSIENT_RETRIES:
                    if not await self._sleep_with_lease(goal_id, 2.0 * (2**attempt)):
                        return "gone", ""

        if manual_changed():
            return "retry", ""
        if verdict is None:
            err = str(last_err or "unknown error")
            if last_err is not None and last_err.unrecoverable:
                notice = f"Goal cleared after an unrecoverable error ({err}). Run /goal again to continue."
                try:
                    reg.commit_evaluator_update(
                        goal,
                        owner_id=self._owner_id,
                        state_updates={},
                        outcome="cleared",
                    )
                except GoalContractChangedError:
                    return "retry", ""
                self._alert_owner(goal_id, sid, "goal_cleared_error", notice)
                try:
                    session = self._agent.get_session(sid)
                    if session is not None:
                        sync_session_goal_projection(session, None)
                except Exception:  # noqa: BLE001
                    pass
                return "cleared", err
            paused = self._auto_pause(
                goal_id,
                sid,
                REASON_EVALUATOR_UNAVAILABLE,
                f"Goal paused after {ge.TRANSIENT_RETRIES} automatic retries "
                f"(goal evaluator: {err}). Send a message to retry.",
                expected=goal,
            )
            return ("paused" if paused else "retry"), err

        kind, outcome, outcome_reason = _evaluator_outcome(verdict)
        evaluations = int(state.get("evaluations", 0) or 0) + 1

        try:
            goal = reg.commit_evaluator_update(
                goal,
                owner_id=self._owner_id,
                state_updates={}
                if cached_terminal
                else {
                    "verdict": kind,
                    "last_check": verdict.reason,
                    "evaluated_turns": turns,
                    "evaluated_transcript_sha256": transcript_digest,
                    "evaluations": evaluations,
                    "checked_at": _utcnow_iso(),
                },
                outcome=outcome,
                reason=outcome_reason,
                summary=verdict.reason or None,
            )
        except GoalContractChangedError:
            return "retry", ""
        _LOG.info("goal %s evaluated: %s — %s", goal_id[:16], kind, verdict.reason[:200])
        if kind == "not_met":
            return "not_met", verdict.reason
        try:
            session = self._agent.get_session(sid)
            if session is not None:
                sync_session_goal_projection(session, goal)
        except Exception:  # noqa: BLE001
            pass
        return kind, verdict.reason

    async def _load_session(self, sid: str) -> Any | None:
        try:
            return self._agent.get_session(sid)
        except Exception:  # noqa: BLE001
            return None

    def _emit_goal_auto_turn_started(
        self,
        sid: str,
        *,
        goal_id: str,
        goal: GoalRecord,
        root_session_state: dict[str, Any] | None = None,
    ) -> None:
        """Persist a lifecycle marker so UIs can show auto-turn N/M.

        E1 — surfaces the root session's durable state register and the
        subtree-liveness answer in the payload, so a UI can see WHERE the
        goal's session is, not just that the row says running."""
        try:
            from sharedagents_goal.goal_registry import session_work_live
            from sharedagents_session.session_writer import get_session_writer

            turn_n = int(goal.turn_count or 0) + 1
            # ``max_turns`` here is a PURELY COSMETIC "N/M" progress label — it
            # never terminates the loop (the loop is cost-terminated; see
            # _drive_loop). 0 == no turn cap
            # (autonomous) → None (∞), consistent with
            # local_llm._goal_progress_metrics + command_runtime._goal_state.
            max_label = goal.max_turns if goal.max_turns > 0 else None
            payload: dict[str, Any] = {
                "goal_id": goal_id,
                "goal_turn_count": turn_n,
                "max_goal_turns": max_label,
                "source": "goal_supervisor",
            }
            try:
                live, live_src = session_work_live(sid)
                payload["subtree_live"] = live
                payload["subtree_live_source"] = live_src
            except Exception:  # noqa: BLE001
                pass
            if root_session_state:
                payload["root_session_state"] = str(
                    root_session_state.get("session_state") or "",
                )
                payload["root_last_prompt_tokens"] = int(
                    root_session_state.get("last_prompt_tokens") or 0,
                )
            writer = get_session_writer(self._agent.sessions_dir, sid)
            writer.lifecycle(
                kind="goal_auto_turn_started",
                payload=payload,
            )
        except Exception:  # noqa: BLE001
            pass
