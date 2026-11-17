"""Heartbeat scheduler wiring for :class:`Agent`.

The periodic background-turn scheduler half of the Agent: start/stop of
the per-agent ``HeartbeatScheduler`` and the thin ``AgentApp``-shaped shim
that lets the legacy scheduler drive turns through this Agent uniformly
for both local-LLM and CLI engines. Split out of ``agents/agent.py`` as a
mixin so the composition root keeps only turn orchestration.

Methods resolve against the ``Agent`` instance via normal MRO
(``self.cfg``, ``self.engine``, ``self.get_session``, ``self.run_turn``,
and the per-Agent ``self._heartbeat_scheduler`` field).

``__slots__ = ()`` is REQUIRED: ``Agent`` is ``@dataclass(slots=True)`` and
relies on undeclared-attribute assignment raising ``AttributeError``. A
mixin base without ``__slots__`` would re-introduce ``__dict__`` and break
that guarantee.
"""

from __future__ import annotations

import asyncio  # noqa: F401  # referenced in the ``Optional["asyncio.Task"]`` annotation string
from typing import Any

from .session_log import agent_session_to_dict


class HeartbeatMixin:
    """Start/stop the periodic heartbeat scheduler + its app shim."""

    __slots__ = ()

    def start_heartbeat(self) -> asyncio.Task[None] | None:
        """Start the periodic background heartbeat task.

        Reference: ``sharedagents_gateway.heartbeat.HeartbeatScheduler``
        (heartbeat.py L152). When ``cfg.heartbeat_enabled`` is True,
        spawns a task that:

        * Maintains a DEDICATED heartbeat session (one per agent, not
          per chat) so the prefix cache stays warm.
        * Every ``cfg.heartbeat_interval_s`` (default 1800 = 30 min),
          fires one turn into that session via ``engine.run_turn``.
        * The prompt is the standard ``HEARTBEAT_PROMPT`` plus the
          contents of ``HEARTBEAT.md``; the model decides whether to
          do background work (memory synth, idle cleanup) or reply
          ``HEARTBEAT_OK`` and stay quiet.
        * State (current heartbeat sid, per-session last-analyzed
          turn idx) lives in ``<home>/heartbeat-state.json``.

        Returns the task handle, or ``None`` if heartbeat is disabled.
        """
        if not getattr(self.cfg, "heartbeat_enabled", False):
            return None
        if getattr(self, "_heartbeat_scheduler", None) is not None:
            # Already running.
            return getattr(self._heartbeat_scheduler, "_task", None)
        try:
            from sharedagents_gateway.heartbeat import HeartbeatScheduler

            # HeartbeatScheduler expects an AgentApp-shaped object
            # exposing ``_read_session_file``, ``_resolve_model``,
            # ``turns``, and a way to fire turns. Build a thin shim
            # over this Agent so the scheduler keeps working
            # uniformly for both local-LLM and CLI agents.
            scheduler = HeartbeatScheduler(self.cfg, self._build_heartbeat_app_shim())
            self._heartbeat_scheduler = scheduler
            import asyncio

            task = asyncio.create_task(scheduler.start())
            return task
        except Exception:  # noqa: BLE001
            self._heartbeat_scheduler = None
            return None

    async def stop_heartbeat(self) -> None:
        """Cancel + drain the heartbeat task."""
        scheduler = getattr(self, "_heartbeat_scheduler", None)
        if scheduler is None:
            return
        try:
            await scheduler.stop()
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._heartbeat_scheduler = None

    def _build_heartbeat_app_shim(self) -> Any:
        """Build a HeartbeatScheduler-compatible app shim around this Agent.

        The legacy scheduler expects an ``AgentApp``-shaped object
        with ``_read_session_file``, ``_resolve_model``, ``turns``
        (an empty dict-of-TurnState is fine for new-architecture
        agents that don't share the legacy in-flight map), and a
        ``session_turn`` method. The shim forwards each to the
        Agent / engine.
        """
        agent = self

        class _HeartbeatAppShim:
            def __init__(self) -> None:
                # Empty in-flight turns map — the legacy scheduler
                # only reads this to skip ticks when an existing
                # turn on the heartbeat session is still running.
                # New-architecture in-flight tracking lives on the
                # engine; we surface zero turns so the scheduler
                # always fires (engine-level locking serializes).
                self.turns: dict[str, Any] = {}

            def _read_session_file(self, sid: str) -> dict[str, Any] | None:
                # header from sessions.db + turns folded from JSONL.
                sess = agent.get_session(sid)
                if sess is None:
                    return None
                d = agent_session_to_dict(sess)
                return d

            async def _resolve_model(self) -> str:
                return getattr(agent.cfg, "upstream_model", "") or agent.engine.kind.value

            async def session_turn(self, sid: str, prompt: str) -> None:
                import inspect

                from .engine import TurnDone

                run = agent.run_turn
                try:
                    params = inspect.signature(run).parameters
                    supports_source = "source" in params or any(
                        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
                    )
                except (TypeError, ValueError):
                    supports_source = False
                run_kwargs = {"source": "heartbeat"} if supports_source else {}
                async for ev in run(sid, prompt, **run_kwargs):
                    if isinstance(ev, TurnDone):
                        break

        return _HeartbeatAppShim()
