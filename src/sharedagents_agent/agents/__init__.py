"""Agent + Engine layer for sharedagents.

The architecture composes two orthogonal axes:

* :class:`Agent` (identity)  — name, home dir, ``SOUL.md`` /
  ``AGENTS.md`` / ``TOOLS.md`` / ``USER.md`` / ``MEMORY.md``, sessions
  dir, permission mode, hooks. See :mod:`.agent`.

* :class:`Engine` (turn dispatch) — abstract; implementations live
  alongside this file:

  - :class:`LocalLLMEngine` — drives the local LLM via worker + tool
    dispatch + compaction + loop detection + distill / nudge /
    force-terminate recovery. See :mod:`.local_llm`.
  - :class:`ClaudeCliEngine` / :class:`CodexCliEngine` /
    :class:`AntigravityCliEngine` — subclasses of
    :class:`ExternalCliEngine`; spawn the matching CLI, ``--resume``
    after the first call, translate the CLI's stream-JSON into the
    normalized :data:`AgentTurnEvent` stream.

================================================================
File map
================================================================

========================= ============================================
File                      Role
========================= ============================================
__init__.py               this docstring + ``__all__`` re-exports
engine.py                 Engine ABC + AgentTurnEvent dataclasses + types
local_llm.py              LocalLLMEngine — drives the local-LLM turn loop
external_cli.py           ExternalCliEngine — shared CLI scaffolding
claude_engine.py          ClaudeCliEngine subclass
codex_engine.py           CodexCliEngine subclass
antigravity_engine.py          AntigravityCliEngine subclass
agent.py                  Agent runtime — composes Engine + identity
factory.py                load_agent + auto-scaffold helpers
session_log.py            uniform record_turn writer (fine-tune ready)
gateway_shim.py           AgentACPBackend — ACP backend over Agent
delegate_shims.py         cross-agent delegate-tool shims
default_scaffolds.py      SOUL/AGENTS/TOOLS templates for auto-scaffolded
                           external-engine agents
acp_translation.py        AgentTurnEvent ↔ ACP sessionUpdate mapping
permissions_integration.  PermissionPolicy + AskQuestion + ASK protocol
   py
========================= ============================================
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .agent import Agent
from .engine import (
    AgentTurnEvent,
    Engine,
    EngineKind,
    EngineSessionState,
    TextDelta,
    ToolCallResult,
    ToolCallStarted,
    TurnDone,
    TurnError,
    TurnRequest,
    TurnStarted,
)
from .session_log import record_turn

# Keep the public re-export surface without importing every concrete engine
# when callers merely import ``sharedagents_agent.agents`` or its factory.  The
# factory is the sole point that selects a concrete implementation, and its
# own per-kind import map remains lazy.
_LAZY_EXPORTS = {
    "AgentACPBackend": (".gateway_shim", "AgentACPBackend"),
    "ExternalCliEngine": (".external_cli", "ExternalCliEngine"),
    "LocalLLMEngine": (".local_llm", "LocalLLMEngine"),
    "ClaudeCliEngine": (".claude_engine", "ClaudeCliEngine"),
    "CodexCliEngine": (".codex_engine", "CodexCliEngine"),
    "AntigravityCliEngine": (".antigravity_engine", "AntigravityCliEngine"),
    "AcpEngine": (".acp_engine", "AcpEngine"),
    "load_agent": (".factory", "load_agent"),
}


def __getattr__(name: str) -> Any:
    """Lazily resolve compatibility re-exports of concrete engines."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "Agent",
    "AgentACPBackend",
    "AgentTurnEvent",
    "Engine",
    "EngineKind",
    "EngineSessionState",
    "ExternalCliEngine",
    "LocalLLMEngine",
    "ClaudeCliEngine",
    "CodexCliEngine",
    "AntigravityCliEngine",
    "AcpEngine",
    "TextDelta",
    "ToolCallStarted",
    "ToolCallResult",
    "TurnDone",
    "TurnError",
    "TurnRequest",
    "TurnStarted",
    "load_agent",
    "record_turn",
]
