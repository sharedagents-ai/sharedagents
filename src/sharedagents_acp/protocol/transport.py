"""Transport primitives shared across stdio / WS / HTTP ACP wires.

The ABC is deliberately minimal: every transport eventually reduces
to a framed byte stream, so we expose just ``read_message`` /
``write_message`` / ``close``. Higher-level dispatcher logic lives in
``sharedagents_acp.client`` / ``sharedagents_acp.server``; this module only
owns the raw-bytes plumbing.

``StdioTransport`` wraps the newline-delimited-JSON stdio
reader/writer pair from ``jsonrpc.read_message`` /
``jsonrpc.write_message``; WS and HTTP transport adapters subclass
the same ABC.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from .jsonrpc import read_message as _read_message, write_message as _write_message


class Transport(ABC):
    """Abstract bidirectional framed-byte transport.

    All methods are async. Implementations MUST treat
    ``read_message`` / ``write_message`` as one-message-at-a-time
    operations — concurrent reads or writes on the same transport are
    undefined.

    ``close`` is idempotent. Calling ``read_message`` after ``close``
    raises ``ConnectionError`` or returns end-of-stream depending on
    the transport.
    """

    @abstractmethod
    async def read_message(self) -> bytes:
        """Block until one full message frame is available, then return
        its raw bytes (no parsing). The caller decides how to interpret
        the bytes (typically by handing them to
        ``sharedagents_acp.protocol.jsonrpc.parse_message``)."""

    @abstractmethod
    async def write_message(self, payload: bytes) -> None:
        """Frame and write one message. ``payload`` is the raw body
        bytes — the framer is responsible for adding any framing
        envelope (newline terminator for stdio, WS text framing, SSE
        event wrapping, etc.)."""

    @abstractmethod
    async def close(self) -> None:
        """Tear down the transport. Idempotent."""


class StdioTransport(Transport):
    """Newline-delimited-JSON stdio JSON-RPC transport (ACP wire format).

    Wraps a paired ``asyncio.StreamReader`` / ``asyncio.StreamWriter``
    so any bidirectional stream pair works — real stdin/stdout via
    ``acp.stdio.stdio_streams()``, an in-memory pipe pair from the test
    harness, or a subprocess's stdio.
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._closed = False

    async def read_message(self) -> bytes:
        return await _read_message(self._reader)

    async def write_message(self, payload: bytes) -> None:
        await _write_message(self._writer, payload)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._writer.close()
            await self._writer.wait_closed()
        except (ConnectionError, OSError):
            # Already gone — close is idempotent.
            pass


__all__ = ["Transport", "StdioTransport"]
