"""Deprecated stub.

The runtime no longer injects prose at request time; behavior steering
lives in editable `.md` files under the agent's home (`WORKSPACE.md`,
`TOOLCALL.md`, `MODE-PLAN.md`, `MODE-AUTO.md`). See
`tools.context.load_system_prompt` and `agent.scaffold_context_files`.

This module is kept for backward-compatible imports. New code should
not call `build_tools_system_prompt` — it returns an empty string.
"""

from __future__ import annotations

from .permissions import PermissionMode  # noqa: F401  (re-export)

# Empty placeholder kept so the criticality contract block isn't
# dropped at import time by anyone who happened to inspect it. Real
# content lives in `TOOLCALL.md`.
CRITICALITY_SYSTEM_PROMPT = ""


def build_tools_system_prompt(mode: PermissionMode) -> str:  # noqa: ARG001
    """No-op shim. Behavior moved to `MODE-{PLAN,AUTO}.md`.

    Kept as an importable symbol so we don't break old callers, but
    returns "" so callers that still concatenate it into their prompts
    are a no-op rather than a crash.
    """
    return ""
