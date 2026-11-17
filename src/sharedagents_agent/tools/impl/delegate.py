"""Thin re-export shim — delegate runners live in ``sharedagents_agent.delegates``.

Phase 1 of ``plan/agents/external_agents_consolidation.md`` moved the
3.2k-line monolith into ``delegates/{base,claude,codex,antigravity}.py``.
Lazy attribute loading avoids a circular import with ``tools.runners``
during package init (``base`` → ``tools._log`` → ``runners`` → this shim).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "_CLAUDE_MODE_MAP",
    "_ANTIGRAVITY_MODE_MAP",
    "_looks_like_context_overflow",
    "_resolve_delegate_cwd",
    "_log_delegate_error",
    "_claude_error_is_abort",
    "_claude_state_path",
    "_read_claude_session_id",
    "_write_claude_session_id",
    "_format_claude_tool_use",
    "_format_claude_tool_result",
    "_antigravity_state_path",
    "_read_antigravity_session_id",
    "_write_antigravity_session_id",
    "_format_antigravity_tool_use",
    "_format_antigravity_tool_result",
    "run_delegate_antigravity_async",
    "run_delegate_claude_async",
    "_CODEX_MODE_FLAGS",
    "_codex_state_path",
    "_read_codex_state",
    "_update_codex_state",
    "_read_codex_thread_id",
    "_write_codex_thread_id",
    "_codex_usage_from_event",
    "_codex_raw_cumulative",
    "_codex_usage_delta",
    "_format_codex_command_started",
    "_format_codex_command_completed",
    "_run_codex_compact_resume",
    "run_delegate_codex_async",
    "_zcode_state_path",
    "_read_zcode_session_id",
    "_write_zcode_session_id",
    "run_delegate_zcode_async",
]

_BASE_NAMES = frozenset(
    {
        "_ANTIGRAVITY_MODE_MAP",
        "_CLAUDE_MODE_MAP",
        "_CLI_EFFORT_LEVELS",
        "_REASONING_BLOCK_TYPES",
        "_REASONING_DELTA_TYPES",
        "_claude_effort_flags",
        "_claude_error_is_abort",
        "_claude_result_is_task_notification",
        "_codex_effort_flags",
        "_codex_local_artifact_references",
        "_codex_raw_cumulative",
        "_codex_usage_delta",
        "_codex_usage_from_event",
        "_effective_cli_effort",
        "_is_numeric_zero",
        "_log_delegate_error",
        "_looks_like_context_overflow",
        "_reasoning_signal_chars",
        "_reasoning_text_chars",
        "_resolve_delegate_cwd",
    }
)

_CLAUDE_NAMES = frozenset(
    {
        "_claude_state_path",
        "_format_claude_tool_result",
        "_format_claude_tool_use",
        "_read_claude_session_id",
        "_write_claude_session_id",
        "run_delegate_claude_async",
    }
)

_ANTIGRAVITY_NAMES = frozenset(
    {
        "_ANTIGRAVITY_STREAM_TYPES",
        "_antigravity_event_type",
        "_antigravity_result_payload",
        "_antigravity_session_id",
        "_antigravity_state_path",
        "_antigravity_step_payload",
        "_antigravity_tool_output",
        "_format_antigravity_tool_result",
        "_format_antigravity_tool_use",
        "_read_antigravity_session_id",
        "_write_antigravity_session_id",
        "run_delegate_antigravity_async",
    }
)

_CODEX_NAMES = frozenset(
    {
        "_CODEX_MODE_FLAGS",
        "_codex_state_path",
        "_format_codex_command_completed",
        "_format_codex_command_started",
        "_read_codex_state",
        "_read_codex_thread_id",
        "_run_codex_compact_resume",
        "_update_codex_state",
        "_write_codex_thread_id",
        "run_delegate_codex_async",
    }
)

_ZCODE_NAMES = frozenset(
    {
        "_ZCODE_MODE_MAP",
        "_read_zcode_session_id",
        "_write_zcode_session_id",
        "_zcode_extract_result",
        "_zcode_spawn_args",
        "_zcode_state_path",
        "_zcode_usage_to_snake",
        "run_delegate_zcode_async",
    }
)


def __getattr__(name: str) -> Any:
    if name in _BASE_NAMES:
        import sharedagents_agent.delegates.base as mod

        return getattr(mod, name)
    if name in _CLAUDE_NAMES:
        import sharedagents_agent.delegates.claude as mod

        return getattr(mod, name)
    if name in _ANTIGRAVITY_NAMES:
        import sharedagents_agent.delegates.antigravity as mod

        return getattr(mod, name)
    if name in _CODEX_NAMES:
        import sharedagents_agent.delegates.codex as mod

        return getattr(mod, name)
    if name in _ZCODE_NAMES:
        import sharedagents_agent.delegates.zcode as mod

        return getattr(mod, name)
    # The old implementation also exposed the shared ``_utils`` names as
    # incidental module attributes via ``from ._utils import *``.  Keep those
    # direct imports working without adding them to ``__all__`` (and therefore
    # without widening the historical ``tools.runners`` star-export surface).
    from sharedagents_agent.tools.impl import _utils

    if name in _utils.__all__:
        return getattr(_utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
