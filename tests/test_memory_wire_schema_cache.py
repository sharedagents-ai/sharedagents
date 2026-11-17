"""The memory wire must not trigger an SDK ``list_tools`` per internal op."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from sharedagents_mcp.memory.client import _mark_tool_schema_known


def test_marks_unlisted_tool_without_clobbering_listed_schema() -> None:
    sess = SimpleNamespace(_tool_output_schemas={"search": {"type": "object"}})
    _mark_tool_schema_known(sess, "observe_turn")
    _mark_tool_schema_known(sess, "search")
    assert sess._tool_output_schemas == {"search": {"type": "object"}, "observe_turn": None}


def test_tolerates_sessions_without_the_cache() -> None:
    _mark_tool_schema_known(SimpleNamespace(), "observe_turn")  # no AttributeError


def test_real_sdk_skips_list_tools_for_marked_tool() -> None:
    """Pin the SDK behaviour the helper relies on (fails loudly if it changes)."""
    from mcp import ClientSession, types

    sess = ClientSession.__new__(ClientSession)
    if not hasattr(ClientSession, "validate_tool_result"):
        return  # older SDKs never re-list on call
    sess._tool_output_schemas = {}
    calls = []

    async def _list_tools(*_a, **_k):
        calls.append(1)

    sess.list_tools = _list_tools  # type: ignore[method-assign]
    _mark_tool_schema_known(sess, "observe_turn")
    result = types.CallToolResult(content=[])
    asyncio.run(sess.validate_tool_result("observe_turn", result))
    assert calls == []
