"""``_mcp_attr`` reads MCP SDK fields under 1.x (camelCase) and 2.x (snake_case)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from sharedagents_mcp.mcp_client import _mcp_attr
from sharedagents_research.workflow import KnowledgeClient, ResearchError


def test_mcp_attr_camel_and_snake() -> None:
    assert _mcp_attr(SimpleNamespace(isError=True), "isError") is True
    assert _mcp_attr(SimpleNamespace(is_error=True), "isError") is True
    assert _mcp_attr(SimpleNamespace(input_schema={"a": 1}), "inputSchema") == {"a": 1}
    assert _mcp_attr(SimpleNamespace(read_only_hint=False), "readOnlyHint") is False
    assert _mcp_attr(SimpleNamespace(), "isError", False) is False


def test_research_client_honours_snake_case_result() -> None:
    class _Sess:
        async def call_tool(self, name, arguments):
            return SimpleNamespace(is_error=False, structured_content={"ok": 1}, content=[])

    assert asyncio.run(KnowledgeClient(_Sess()).call("x")) == {"ok": 1}

    class _ErrSess:
        async def call_tool(self, name, arguments):
            return SimpleNamespace(
                is_error=True,
                structured_content=None,
                content=[SimpleNamespace(type="text", text="nope")],
            )

    try:
        asyncio.run(KnowledgeClient(_ErrSess()).call("x"))
    except ResearchError as exc:
        assert "nope" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ResearchError")
