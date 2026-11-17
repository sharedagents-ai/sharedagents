"""Tools subsystem — host-derived criticality + permission evaluator + registry.

The user picks one of three permission modes per session:

- **plan**: read-only operations only. Anything mutating or destructive is
  refused before the tool runs.
- **ask**: prompt before mutating / destructive operations. Three responses
  on each prompt: `yes` / `yes-and-don't-ask-again-for-this` / `no`.
  The session allowlist remembers `yes-and-don't-ask-again` choices for
  the rest of the session, but **never** for destructive operations —
  those always re-prompt as a safety net.
- **auto**: run without prompting, except destructive operations always
  pause for confirmation. That's the auto-mode safety net.

Criticality classification is host-led: registered tool kinds provide the
floor and `tools.safety` classifies shell commands. Legacy/internal callers may
still provide a claim, but it can only raise severity.

There is **no LLM-protocol wiring here yet**. The worker still streams
plain text; once we add OpenAI tool-calling to the worker, the agent
runtime will instantiate a `PermissionEvaluator` per session and gate
each tool call through it. This module is the permission/safety brain.

Public API:
"""

from .permissions import (
    Criticality,
    Decision,
    PermissionEvaluator,
    PermissionMode,
    Response,
    exec_targets_memory,
    is_memory_path,
    most_cautious,
    path_under_memory,
)
from .prompt import CRITICALITY_SYSTEM_PROMPT, build_tools_system_prompt
from .registry import (
    BUILTIN_TOOLS,
    Tool,
    ToolKind,
    ToolRegistry,
    acp_session_mcp_tools_sync,
    discovered_mcp_tools,
    discovered_mcp_tools_sync,
)
from .runners import ToolResult, dispatch, format_todo_md, load_todos, todos_path_for_session
from .safety import (
    CommandSafety,
    SafetyClassifier,
    classify_command,
)

__all__ = [
    "BUILTIN_TOOLS",
    "acp_session_mcp_tools_sync",
    "discovered_mcp_tools",
    "discovered_mcp_tools_sync",
    "CRITICALITY_SYSTEM_PROMPT",
    "CommandSafety",
    "Criticality",
    "Decision",
    "PermissionEvaluator",
    "PermissionMode",
    "Response",
    "SafetyClassifier",
    "Tool",
    "ToolKind",
    "ToolRegistry",
    "ToolResult",
    "build_tools_system_prompt",
    "classify_command",
    "dispatch",
    "exec_targets_memory",
    "format_todo_md",
    "is_memory_path",
    "load_todos",
    "most_cautious",
    "path_under_memory",
    "todos_path_for_session",
]
