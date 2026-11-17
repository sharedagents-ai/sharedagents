"""Stdio entry point for :class:`AcpServer`.

Wires any backend conforming to :class:`AgentBackend` to a stdio ACP
transport — a child process launched by an ACP client over the
newline-delimited-JSON JSON-RPC stdio protocol. This module owns the
stdin/stdout plumbing so backends stay focused on the
:class:`AgentBackend` surface and don't import asyncio stream APIs.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from ..protocol import StdioTransport
from ..protocol.jsonrpc import ACP_STREAM_LIMIT
from .backend import AgentBackend
from .server import AcpServer


async def serve_stdio(
    backend: AgentBackend,
    *,
    agent_capabilities: Any,
    agent_info: Any,
    auth_methods: list[Any] | None = None,
    on_server_ready: Callable[[AcpServer], Awaitable[None]] | None = None,
) -> None:
    """Run an :class:`AcpServer` over stdin/stdout until EOF.

    Wraps ``sys.stdin`` / ``sys.stdout`` into an asyncio
    StreamReader/StreamWriter pair (via the SDK's ``stdio_streams``
    helper, which handles platform-specific stdio attach), constructs a
    :class:`StdioTransport`, builds an :class:`AcpServer`, and runs its
    read loop.

    Returns when the read loop exits cleanly (EOF on stdin) or raises
    on an unrecoverable transport error.

    Args:
        on_server_ready: optional async callback invoked with the
            constructed :class:`AcpServer` BEFORE the read loop starts.
            Used by backends that need the server handle to make
            outbound requests (e.g. ``session/request_permission``).
            If the callback raises, the server is torn down and the
            error propagates.
    """
    reader, writer = await _stdio_streams()
    transport = StdioTransport(reader, writer)
    try:
        server = AcpServer(
            transport,
            backend,
            agent_capabilities=agent_capabilities,
            agent_info=agent_info,
            auth_methods=auth_methods,
        )
        if on_server_ready is not None:
            await on_server_ready(server)
        await server.serve_until_close()
    finally:
        await transport.close()


async def _stdio_streams() -> tuple[Any, Any]:
    """Wrap ``sys.stdin`` and ``sys.stdout`` as asyncio streams.

    Defers to the official SDK's helper (``acp.stdio_streams``) when
    available so platform-specific quirks (Windows pipe attach, line
    buffering disable on POSIX) match the SDK exactly. Falls back to
    a plain asyncio implementation for environments without the SDK
    installed (e.g. test isolation).

    Both paths size the stdin reader's buffer to ``ACP_STREAM_LIMIT``:
    newline-delimited framing buffers a whole line before the ``\\n``
    delimiter, and ACP prompts carry base64 image blocks that overflow
    asyncio's 64 KiB default.
    """
    try:
        import acp  # type: ignore[import-not-found]

        if hasattr(acp, "stdio_streams"):
            return await acp.stdio_streams(limit=ACP_STREAM_LIMIT)
    except ImportError:
        pass

    # Fallback: connect_read_pipe / connect_write_pipe over the real
    # stdio fds.
    import asyncio
    import sys

    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=ACP_STREAM_LIMIT)
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)
    transport, _ = await loop.connect_write_pipe(asyncio.streams.FlowControlMixin, sys.stdout)
    writer = asyncio.StreamWriter(transport, protocol, reader, loop)
    return reader, writer


__all__ = ["serve_stdio"]
