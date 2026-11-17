"""WebSocket :class:`Transport` adapter for ACP.

Wraps a WebSocket-like object so the existing
:class:`~sharedagents_acp.client.AcpClient` and
:class:`~sharedagents_acp.server.AcpServer` can run unchanged: they only
know about :class:`Transport`, not the wire shape.

Conformance bullets:

* Text frames carry JSON-RPC payloads. Binary frames are rejected — we
  raise :class:`WebSocketBinaryFrameError` so the higher-level adapter
  can close the connection with an unsupported-data status (1003) per
  RFC 6455 §7.4.1. The dispatcher layer never sees binary data.
* WS-level ping/pong is handled by the underlying library
  (``aiohttp``'s autoping is on by default; ``websockets`` likewise) —
  there is no vendor heartbeat notification at the JSON-RPC layer.
* Close frame turns into a :class:`JsonRpcFramingError` whose message
  starts with ``"EOF before any header bytes"`` — the dispatcher's
  ``serve_until_close`` loop already matches that exact prefix to
  return cleanly on EOF (this is the same idiom the stdio transport
  uses when the peer closes its write side).

The wrapped object is duck-typed on a tiny surface
(``receive / send_str / close``) so we can run the same adapter
against an aiohttp ``WebSocketResponse`` (server side), an
``aiohttp.ClientWebSocketResponse`` (client side), or any other
library that exposes the same shape (``websockets``-derived shims, the
test harness's in-memory pair, etc.).
"""

from __future__ import annotations

import logging
from typing import Any

from .jsonrpc import JsonRpcFramingError
from .transport import Transport

logger = logging.getLogger(__name__)


class WebSocketBinaryFrameError(Exception):
    """Raised when a binary frame arrives on a WS ACP transport.

    ACP-over-WS carries JSON-RPC text frames only. Binary frames are a
    protocol violation; the WS adapter surfaces this error so a
    higher-level server can close the connection with an unsupported-data
    status code.
    """


class WebSocketTransport(Transport):
    """:class:`Transport` over a duck-typed WebSocket.

    Accepts any object with the following async surface:

    * ``receive()`` — returns an aiohttp-style ``WSMessage``
      (``.type`` is one of ``WSMsgType.{TEXT, BINARY, CLOSE,
      CLOSING, CLOSED, ERROR}`` and ``.data`` carries the payload).
      Used in :meth:`read_message`.
    * ``send_str(s: str)`` — writes a text frame.
    * ``close(*, code=..., message=...)`` — closes the WS.

    Concurrent :meth:`read_message` calls are not allowed (Transport
    contract); the dispatcher / client read loop is single-reader so
    this is fine in practice.

    WS-level keepalive (ping/pong) is the underlying library's
    responsibility:

    * aiohttp ``WebSocketResponse`` / ``ClientWebSocketResponse`` —
      autopings are on by default; ``heartbeat=`` configures cadence.
    * ``websockets`` package — ping_interval/ping_timeout configures it.

    We intentionally do NOT inject a vendor heartbeat at the JSON-RPC
    layer — keepalive belongs to the WS library.
    """

    # We import the aiohttp WSMsgType lazily. The transport works
    # without aiohttp installed if the wrapped object happens to use a
    # different (compatible) enum — in that case the integer values
    # match anyway for the cases we care about, but we resolve the type
    # symbolically rather than by integer.
    _WS_MSG_TYPE: Any = None

    def __init__(self, ws: Any) -> None:
        self._ws = ws
        self._closed = False

    @classmethod
    def _ws_msg_type(cls) -> Any:
        if cls._WS_MSG_TYPE is None:
            from aiohttp import WSMsgType  # local import keeps the module light

            cls._WS_MSG_TYPE = WSMsgType
        return cls._WS_MSG_TYPE

    async def read_message(self) -> bytes:
        """Read one text frame and return its UTF-8 bytes.

        Raises:
            JsonRpcFramingError: the peer sent a CLOSE frame, or the
                WS is already closing/closed. The message starts with
                ``"EOF before any header bytes"`` so the dispatcher's
                serve loop returns cleanly via its existing EOF branch
                (the same idiom the stdio transport uses).
            WebSocketBinaryFrameError: the peer sent a binary frame.
                Caller should close with status 1003 (unsupported data).
            ConnectionError: an underlying transport error.
        """
        if self._closed:
            raise JsonRpcFramingError("EOF before any header bytes (WS transport already closed)")
        ws_types = self._ws_msg_type()
        msg = await self._ws.receive()
        mtype = msg.type
        if mtype == ws_types.TEXT:
            data = msg.data
            if isinstance(data, bytes):
                return data
            return data.encode("utf-8")
        if mtype == ws_types.BINARY:
            raise WebSocketBinaryFrameError("binary WS frame received; ACP-over-WS carries JSON-RPC text frames only")
        if mtype in (ws_types.CLOSE, ws_types.CLOSING, ws_types.CLOSED):
            raise JsonRpcFramingError("EOF before any header bytes (WS peer closed)")
        if mtype == ws_types.ERROR:
            exc = self._ws.exception() if hasattr(self._ws, "exception") else None
            if exc is not None:
                raise ConnectionError(f"WebSocket error: {exc}") from exc
            raise ConnectionError("WebSocket error frame received")
        # PING/PONG/CONTINUATION are handled by the underlying library;
        # if one leaks through we ignore it and read again.
        if mtype in (ws_types.PING, ws_types.PONG, ws_types.CONTINUATION):
            return await self.read_message()
        raise ConnectionError(f"unexpected WS message type: {mtype!r}")

    async def write_message(self, payload: bytes) -> None:
        """Write a single JSON-RPC text frame.

        ``payload`` must be valid UTF-8 (the JSON-RPC framing layer
        always serializes to UTF-8 bytes). We decode and use
        ``send_str`` so the frame opcode is unambiguously TEXT (0x1)
        regardless of the underlying WS library's heuristic for
        bytes-like inputs.
        """
        # One validated NDJSON line per frame — reject a
        # corrupt multi-line payload and append the shared ``\n`` delimiter.
        from .jsonrpc import frame_ndjson_line

        await self._ws.send_str(frame_ndjson_line(payload).decode("utf-8"))

    async def close(self) -> None:
        """Close the underlying WebSocket. Idempotent."""
        if self._closed:
            return
        self._closed = True
        try:
            await self._ws.close()
        except (ConnectionError, OSError):
            # Already gone — close is idempotent.
            pass


__all__ = ["WebSocketTransport", "WebSocketBinaryFrameError"]
