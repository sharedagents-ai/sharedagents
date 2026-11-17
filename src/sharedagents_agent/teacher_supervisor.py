"""Teacher supervision — shared gate helpers + the off-clock live-action layer.

The module serves two distinct roles:

1. The SHARED effective-state helpers — :func:`is_teacher_active`,
   :func:`effective_teacher_model`, :func:`teacher_agent_system_prompt` and
   :func:`spawn_visible_teacher_subagent`. These are LIVE: the engine gate, the
   TUI footer widget, Telegram ``/status``, the web badge and the in-turn
   review path all read them, so every surface agrees without re-importing the
   engine.

2. :class:`TeacherSupervisor` — the OFF-CLOCK live-action layer for an
   *already-running* agent: mid-turn live injection (drift / tool-failure /
   periodic), recovery of an ended turn (``recover`` = start a fresh turn), and
   the in-place escalation model-swap ladder. It calls the engine APIs directly
   as typed Python (no ACP JSON-RPC re-entry):

     * ``agent.inject_user_message(sid, text)`` — steer a live turn,
     * ``agent.run_turn(sid, prompt=…)``       — recover / start a fresh turn,
     * ``agent.set_config_option(sid, "model", alias)`` — escalation swap.

The verdict itself comes from the shared stdlib verdict engine
(``sharedagents_agent/agents/data/teacher_review.py``), called IN-PROCESS via
``run_review`` (no subprocess, no shell) so there is exactly one
implementation of the review logic. The resulting sidecar
(``teacher_reviews/<sid>.json``) carries the structured outcome this
module acts on.

NOTE: nothing instantiates :class:`TeacherSupervisor` today — the in-turn
engine round-loop (``agents/local_llm/``) is what actually supervises, calling
``run_review`` and rendering a ``delegate_teacher`` node. If the ladder is ever
wired, its gate must stay in step with ``HooksMixin._teacher_enabled``. It runs
entirely in-process — no shell, no subprocess, no hooks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from sharedagents_model.sampling import configured_sampling

_log = logging.getLogger("sharedagents.teacher")

TEACHER_SUBAGENT_NAME = "teacher"
TEACHER_SUBAGENT_TITLE = "Review session goal"

# Async hook events that route here (observe/dispatch, never block).
SIGNAL_EVENTS = {"stop-failure", "session-error", "tool-failure", "periodic"}


# ---------------------------------------------------------------------------
# Shared effective-state helpers — the SINGLE place that decides "is the
# teacher supervising THIS session right now, and with which model". The
# engine's authoritative gate (``local_llm._teacher_enabled``) follows the
# same resolution order; these mirror it for the frontends (TUI footer
# widget, Telegram /status, the web indicator) so every surface agrees
# without re-importing the engine. Pure functions of (policy + persisted
# session header fields) — no engine handle required.
# ---------------------------------------------------------------------------


def is_teacher_active(
    policy: dict | None,
    teacher_override: str = "",
    active_goal: str = "",
) -> bool:
    """True when a teacher model supervises a session with this state.

    Resolution order (matches ``local_llm._teacher_enabled``):
      1. ``policy.enabled`` false / no policy → OFF (nothing to call).
      2. ``/teacher off`` (override == "off") → OFF, always.
      3. ``/teacher on`` (override == "on") → ON, even with NO goal.
      4. Active ``/goal`` without explicit ``/teacher on`` → OFF; the goal
         evaluator owns continuation.
      5. Otherwise, ``require_goal == true`` → OFF until explicitly enabled.
      6. Otherwise → ON from policy alone for ordinary non-goal sessions.

    This is the frontend/shared equivalent of
    ``LocalLLMEngine._teacher_enabled``. A goal never activates Teacher by
    itself; the goal supervisor drives it. ``/teacher on`` adds the teacher
    as an advisor.
    """
    pol = policy or {}
    if not pol.get("enabled"):
        return False
    ov = (teacher_override or "").strip().lower()
    if ov == "off":
        return False
    if ov == "on":
        return True
    if (active_goal or "").strip():
        return False
    return not pol.get("require_goal", True)


def effective_teacher_model(
    teacher_row: dict | None,
    teacher_model_override: str = "",
) -> str:
    """The supervising model label — the per-session ``/teacher model``
    override if set, else the configured ``role == teacher`` row's
    id/alias. Empty string when neither is known."""
    ov = (teacher_model_override or "").strip()
    if ov:
        return ov
    row = teacher_row or {}
    return str(row.get("id") or row.get("alias") or "")


def teacher_agent_system_prompt() -> str:
    """Load the shipped hidden teacher profile prompt."""
    try:
        from sharedagents_session.hidden_agent import load_hidden_agent_system

        return load_hidden_agent_system(TEACHER_SUBAGENT_NAME).strip()
    except Exception:  # noqa: BLE001 — review has its own embedded fallback
        return ""


class VisibleTeacherSubagent:
    """Foreground teacher-review child visible in Agent Monitor."""

    __slots__ = ("child_sid", "writer", "sessions_dir")

    def __init__(self, child_sid: str, writer: Any, sessions_dir: Path) -> None:
        self.child_sid = child_sid
        self.writer = writer
        self.sessions_dir = sessions_dir

    def finish(self, *, content: str = "", error: str | None = None) -> None:
        try:
            self.writer.turn_ended(
                turn_idx=1,
                final_content=content or "",
                aborted=False,
                error=error,
            )
        except Exception:  # noqa: BLE001 — visibility must not break a review
            pass
        try:
            from sharedagents_session.session_writer import close_session_writer

            close_session_writer(self.sessions_dir, self.child_sid)
        except Exception:  # noqa: BLE001
            pass


def spawn_visible_teacher_subagent(
    parent: Any,
    *,
    teacher_row: dict | None = None,
    system_prompt: str = "",
    event: str = "pre_idle",
    goal: str = "",
) -> VisibleTeacherSubagent | None:
    """Create one durable foreground child for a teacher review.

    This mirrors ``spawn_visible_compaction_subagent``: the child carries the
    parent link, emits ``subagent_spawned`` on the parent stream, and owns one
    started/ended turn. The caller still performs and awaits the in-process
    review; this function only establishes its canonical session identity.
    """
    parent_sid = str(getattr(parent, "hash", "") or "").strip()
    sessions_dir = getattr(parent, "_storage_root", None)
    if not parent_sid or sessions_dir is None:
        return None
    sessions_dir = Path(sessions_dir)
    row = dict(teacher_row or {})
    prompt = (system_prompt or teacher_agent_system_prompt()).strip()
    try:
        from sharedagents_agent.agents.session_log import record_session_creation
        from sharedagents_session.session import Session
        from sharedagents_session.session_writer import get_session_writer

        child = Session.new(
            system_prompt=prompt,
            model=str(row.get("id") or row.get("alias") or ""),
            endpoint=str(row.get("baseUrl") or ""),
            sessions_dir=sessions_dir,
            context_window=int(row.get("contextWindow") or 0),
            parent_session_id=parent_sid,
            workspace_origin=f"delegate:{parent_sid}",
        )
        child.title = TEACHER_SUBAGENT_TITLE
        record_session_creation(
            child,
            agent_name=TEACHER_SUBAGENT_NAME,
            engine="local_llm",
            system_prompt=prompt,
            sessions_dir=sessions_dir,
            parent_session_id=parent_sid,
        )
        writer = get_session_writer(sessions_dir, child.hash, engine="local_llm")
        writer.header_update(patch={"title": TEACHER_SUBAGENT_TITLE})
        task = f"Review parent session after {event}."
        if goal.strip():
            task += f" Goal: {goal.strip()}"
        writer.turn_started(turn_idx=1, user=task, permission_mode="plan")
        return VisibleTeacherSubagent(child.hash, writer, sessions_dir)
    except Exception:  # noqa: BLE001 — supervision remains best-effort
        return None


def _sa_home() -> Path:
    return Path(os.environ.get("SHAREDAGENTS_HOME") or (Path.home() / ".sharedagents"))


def _review_script() -> Path:
    # Ships inside sharedagents_agent/agents/data/ alongside the teacher supervisor.
    return Path(__file__).resolve().parent / "agents" / "data" / "teacher_review.py"


_VERDICT_ENGINE = None


def _load_verdict_engine():
    """Import the stdlib-only verdict engine and cache it (module-level so
    one import is shared). In-process — no subprocess, no shell."""
    global _VERDICT_ENGINE
    if _VERDICT_ENGINE is not None:
        return _VERDICT_ENGINE
    import importlib.util

    path = _review_script()
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("sharedagents_teacher_review_sup", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _VERDICT_ENGINE = mod
    return mod


class TeacherSupervisor:
    """One per agent. ``agent`` is the live engine handle (duck-typed:
    needs ``inject_user_message``, ``run_turn``, and optionally
    ``set_config_option``). ``policy`` is ``AgentConfig.teacher_policy()``;
    ``teacher_row`` is ``AgentConfig.teacher_model()``."""

    def __init__(self, agent: Any, *, agent_name: str, policy: dict, teacher_row: dict | None):
        self.agent = agent
        self.agent_name = agent_name
        self.policy = policy or {}
        self.teacher_row = teacher_row or {}
        self._periodic_task: asyncio.Task | None = None

    # ---- sidecar -------------------------------------------------------
    def _sidecar(self, sid: str) -> dict:
        p = _sa_home() / "agents" / self.agent_name / "teacher_reviews" / f"{sid}.json"
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    # ---- review (IN-PROCESS call to the shared verdict engine) ---------
    def _settings_for_review(self) -> dict:
        pol = self.policy
        api_key = ""
        keyenv = self.teacher_row.get("apiKeyEnv")
        if keyenv and os.environ.get(keyenv):
            api_key = os.environ[keyenv]

        def _int(key: str, default: int) -> int:
            # Honour an explicit 0 (disable) but fall back to the default on
            # missing/None/garbage — never crash the off-clock supervisor path.
            v = pol.get(key, default)
            if v is None:
                return default
            try:
                return int(v)
            except (TypeError, ValueError):
                return default

        return {
            "sampling": configured_sampling(
                getattr(self.agent, "cfg", None),
                self.teacher_row,
                self.teacher_row.get("id", ""),
                purpose="teacher",
            ),
            "endpoint": (self.teacher_row.get("baseUrl") or "").rstrip("/"),
            "model": self.teacher_row.get("id") or "",
            "api_key": api_key,
            "timeout": float(pol.get("timeout_seconds", 45) or 45),
            "prompt_text": "",
            "prompt_path": "",
            "deep_tier": str(pol.get("deep_tier", "auto")),
            "min_confidence": float(pol.get("min_confidence", 0.6) or 0.6),
            "max_interventions": _int("max_interventions", 6),
            "stall_limit": _int("stall_limit", 2),
            "tool_failure_threshold": _int("tool_failure_threshold", 5),
            # Carried for completeness/consistency with the policy; the no-result
            # trigger itself is resolved in the engine round loop, not here.
            "no_result_round_limit": _int("no_result_round_limit", 20),
        }

    def _run_review(self, sid: str, transcript_path: str, event: str) -> dict:
        """Run the verdict engine for an out-of-band event, IN-PROCESS (no
        subprocess / no shell). The caller awaits this pass without a review-
        lifecycle deadline; ``settings.timeout`` is confined to the individual
        endpoint request. Returns the control dict (with optional
        ``teacher_action``/``teacher_text``). Never raises.
        """
        mod = _load_verdict_engine()
        if mod is None:
            return {}
        try:
            return (
                mod.run_review(
                    event,
                    session_id=sid,
                    transcript_path=transcript_path,
                    cwd=str(_sa_home()),
                    agent=self.agent_name,
                    settings=self._settings_for_review(),
                )
                or {}
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("teacher review failed for %s: %s", sid, exc)
            return {}

    # ---- live actions (direct typed calls) -----------------------------
    def _inject(self, sid: str, text: str) -> bool:
        fn = getattr(self.agent, "inject_user_message", None)
        if fn is None:
            return False
        try:
            return bool(fn(sid, text))
        except Exception:  # noqa: BLE001
            return False

    async def _recover(self, sid: str, prompt: str) -> bool:
        """Start a fresh turn carrying the [teacher] continuation. This is
        the 'recover an ended turn' contract — not a queued message."""
        run = getattr(self.agent, "run_turn", None)
        if run is None:
            return False
        try:
            # Teacher recovery is runtime work, not an operator prompt. It
            # must remain paused once the session-tree cost cap is spent.
            import inspect

            run_params = inspect.signature(run).parameters
            supports_source = "source" in run_params or any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in run_params.values()
            )
            run_kwargs = {"prompt": prompt}
            if supports_source:
                run_kwargs["source"] = "teacher"
            res = run(sid, **run_kwargs)
            if asyncio.iscoroutine(res):
                await res
            elif hasattr(res, "__aiter__"):
                async for _ in res:  # drive the turn to completion
                    pass
            return True
        except Exception as exc:  # noqa: BLE001
            _log.warning("teacher recover failed for %s: %s", sid, exc)
            return False

    def _swap_model(self, sid: str, alias: str) -> bool:
        fn = getattr(self.agent, "set_config_option", None)
        if fn is None:
            return False
        try:
            fn(sid, "model", alias)
            return True
        except Exception:  # noqa: BLE001
            return False

    # ---- escalation ladder (spec §3.3–3.4) -----------------------------
    async def _escalate(self, sid: str, goal_prompt: str) -> bool:
        targets = self.policy.get("escalation_targets") or ["teacher"]
        if (self.policy.get("escalation") or "auto") == "off":
            return False
        for tgt in targets:
            alias = self.teacher_row.get("alias", "teacher") if tgt == "teacher" else tgt
            if not self._swap_model(sid, alias):
                continue
            handoff = (
                "[teacher:escalation] The smaller model could not complete this. "
                "You are a stronger model; finish the ORIGINAL task now.\n\n" + goal_prompt
            )
            if await self._recover(sid, handoff):
                # one verification review after the escalated turn
                ctrl = self._run_review(sid, self._transcript_for(sid), "stop")
                if ctrl.get("continue") and self._sidecar(sid).get("terminal_state") == "completed_by_small_model":
                    return True
        return False

    def _transcript_for(self, sid: str) -> str:
        return str(_sa_home() / "agents" / self.agent_name / "sessions" / f"{sid}.jsonl")

    # ---- entry points --------------------------------------------------
    async def signal(self, event: str, sid: str, *, transcript_path: str = "") -> None:
        """Async-hook / scanner entry. Runs a review off-clock and acts.
        A progressing review has no wall-clock deadline. A verdict-engine
        escalation flag or exhausted stall budget instead halts normal
        injection and routes through the model-swap escalation ladder.
        Never raises into the caller.
        """
        if not self.policy.get("enabled"):
            return
        tp = transcript_path or self._transcript_for(sid)
        ctrl = await asyncio.get_event_loop().run_in_executor(
            None, self._run_review, sid, tp, "periodic" if event == "periodic" else event
        )
        sc = self._sidecar(sid)
        text = ctrl.get("teacher_text") or ""
        if sc.get("escalate") or sc.get("stalls", 0) >= int(self.policy.get("stall_limit", 2)):
            goal = (sc.get("subgoals") and json.dumps(sc["subgoals"])) or text or "Complete the task."
            await self._escalate(sid, goal)
            return
        if ctrl.get("teacher_action") == "inject" and text:
            # steer a live turn; if it already ended, recover instead.
            if not self._inject(sid, text):
                await self._recover(sid, text)

    def start_periodic(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        minutes = int(self.policy.get("periodic_minutes", 0) or 0)
        if minutes <= 0 or not self.policy.get("enabled"):
            return

        async def _loop():
            while True:
                await asyncio.sleep(minutes * 60)
                try:
                    await self._scan_running()
                except Exception:  # noqa: BLE001
                    _log.debug("teacher periodic scan error", exc_info=True)

        self._periodic_task = (loop or asyncio.get_event_loop()).create_task(_loop())

    async def _scan_running(self) -> None:
        sess = getattr(self.agent, "active_session_ids", None)
        sids = list(sess() if callable(sess) else (sess or []))
        for sid in sids:
            await self.signal("periodic", sid)
