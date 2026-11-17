"""sharedagents delegate-control MCP server.

Re-exposes the three delegation-control tools that used to be builtins —
``check_delegations`` / ``cancel_delegation`` / ``capture_delegation_pattern``
— as ``mcp__delegate__check`` / ``mcp__delegate__cancel`` /
``mcp__delegate__capture_pattern``. It is a thin wrapper: each handler calls the
SAME runner the builtin used (``run_check_delegations_async`` /
``run_cancel_delegation_async`` / ``run_capture_delegation_pattern_async`` in
``sharedagents_agent.tools.runners``), so behavior — the async-jobs registry, the
killpg+subtree cancel, the deduped pattern upsert — is unchanged.

Identity / parent context (``SHAREDAGENTS_AGENT_HOME``, ``SHAREDAGENTS_SESSION_ID``, …)
is read by the runners from the env the server inherits — it is NEVER passed as
a tool argument.

``check`` declares ``readOnlyHint`` — it only lists jobs. That maps to READ
criticality in discovery. ``cancel`` and ``capture_pattern`` MUTATE, so they
carry NO annotation and default to the NETWORK/mutating criticality: ``cancel``
stays low-friction (deliberately not ``destructiveHint``) so stopping a runaway
delegation is cheap.

Run: ``python -m sharedagents_mcp.delegate_mcp_server`` (stdio; configured under
``mcpServers.delegate``).

Delegation *dispatch* (``Agent``/``BatchAgent``) stays a first-party builtin;
this delegate-control server re-exposes only the lifecycle tools, never the
dispatch itself.
"""

from __future__ import annotations

import asyncio
from typing import Any

from sharedagents_mcp._stdio_server import result_text, run_stdio_server


def _check(arguments: dict[str, Any]) -> str:
    from sharedagents_agent.tools.runners import run_check_delegations_async

    return result_text(asyncio.run(run_check_delegations_async(arguments)))


def _cancel(arguments: dict[str, Any]) -> str:
    from sharedagents_agent.tools.runners import run_cancel_delegation_async

    return result_text(asyncio.run(run_cancel_delegation_async(arguments)))


def _capture_pattern(arguments: dict[str, Any]) -> str:
    from sharedagents_agent.tools.runners import run_capture_delegation_pattern_async

    return result_text(asyncio.run(run_capture_delegation_pattern_async(arguments)))


TOOL_DEFS: list[dict[str, Any]] = [
    {
        "name": "check",
        "description": (
            "List your background (async) delegations and their status "
            "(started / running / done / failed / cancelled). Use to see "
            "what's still in flight before deciding to wait or cancel."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "cancel",
        "description": (
            "Cancel a background delegation NOW. Pass `job_id` to kill one "
            "job AND everything it spawned, or `session_id` to kill every "
            "background job in that session's subtree. Cancellation walks "
            "DOWN the session tree only — cancelling a sub-agent never stops "
            "its parent."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "Job to cancel (from the handle / check).",
                },
                "session_id": {
                    "type": "string",
                    "description": "Cancel the whole background subtree under this session.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "capture_pattern",
        "description": (
            "Save a successful subagent delegation prompt as a reusable skill "
            "under ``~/.sharedagents/agents/jarvis/skills/delegation-*/SKILL.md``. "
            "Orchestrator (jarvis) only — call AFTER a sync delegation succeeds when "
            "the prompt shape is worth reusing. The runtime merges near-duplicates "
            "(same subagent_type + similar title) instead of bloating the corpus. "
            "Load captured patterns via ``SkillSearch`` before similar delegations."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "subagent_type": {
                    "type": "string",
                    "description": "Profile used (build / explore / plan / claude / …).",
                },
                "title": {
                    "type": "string",
                    "description": "Short label for this objective class.",
                },
                "prompt_template": {
                    "type": "string",
                    "description": "The delegation prompt that worked (paths/hostnames as placeholders).",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional extra tags for SkillSearch.",
                },
            },
            "required": ["subagent_type", "title", "prompt_template"],
        },
    },
]

TOOL_DISPATCH = {
    "check": _check,
    "cancel": _cancel,
    "capture_pattern": _capture_pattern,
}


if __name__ == "__main__":
    run_stdio_server("delegate", "1.0.0", TOOL_DEFS, TOOL_DISPATCH)
