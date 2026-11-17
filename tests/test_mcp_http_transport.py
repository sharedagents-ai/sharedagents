"""Regression: ``mcp_session`` must open Streamable-HTTP servers on the
installed MCP SDK (``streamablehttp_client`` was renamed in mcp 2.x)."""

from __future__ import annotations

import asyncio
import socket
import threading
import time

import pytest

from sharedagents_mcp.mcp_client import MCPServerConfig, call_tool, list_tools, mcp_session

uvicorn = pytest.importorskip("uvicorn")
_server_mod = pytest.importorskip("mcp.server")
MCPServer = getattr(_server_mod, "MCPServer", None) or getattr(
    pytest.importorskip("mcp.server.fastmcp"), "FastMCP", None
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def http_mcp_url():
    srv = MCPServer("echo")

    from mcp.types import ToolAnnotations

    @srv.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def echo(text: str) -> str:
        return f"echo:{text}"

    @srv.tool()
    def boom() -> str:
        raise ValueError("kaboom")

    port = _free_port()
    config = uvicorn.Config(srv.streamable_http_app(), host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started, "MCP test server failed to start"
    yield f"http://127.0.0.1:{port}/mcp"
    server.should_exit = True
    t.join(timeout=10)


def test_http_transport_lists_and_calls_tools(http_mcp_url: str) -> None:
    cfg = MCPServerConfig(name="echo", transport="http", url=http_mcp_url, headers={"X-Test": "1"})

    async def _run() -> tuple[list[str], str]:
        async with mcp_session(cfg, timeout_s=10) as sess:
            tools = await sess.list_tools()
            res = await sess.call_tool("echo", {"text": "hi"})
            text = "".join(getattr(c, "text", "") for c in res.content)
            return [t.name for t in tools.tools], text

    names, text = asyncio.run(_run())
    assert "echo" in names
    assert text == "echo:hi"


def test_helpers_read_fields_across_sdk_versions(http_mcp_url: str) -> None:
    """``isError`` / ``inputSchema`` / annotation hints are snake_case in mcp 2.x."""
    cfg = MCPServerConfig(name="echo", transport="http", url=http_mcp_url)

    ok, tools, err = asyncio.run(list_tools(cfg, timeout_s=10))
    assert ok, err
    echo = next(t for t in tools if t["name"] == "echo")
    assert "text" in echo["inputSchema"].get("properties", {})
    assert echo["annotations"].get("readOnlyHint") is True

    ok, body = asyncio.run(call_tool(cfg, "echo", {"text": "x"}, timeout_s=10))
    assert ok and body == "echo:x"
    ok, body = asyncio.run(call_tool(cfg, "boom", {}, timeout_s=10))
    assert ok is False
    assert "boom" in body
