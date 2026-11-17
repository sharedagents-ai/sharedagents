"""Streamable HTTP transport adapters (server + client).

This module is the wire-bytes-only layer for the ACP Streamable HTTP
transport. The *application*-layer wiring — Starlette routes, header
validation, SSE keepalive — lives in
:mod:`sharedagents_acp.server.http_server`. This module keeps the same
shape every other transport in the package has: two concrete
:class:`Transport` subclasses, one for the server side and one for
the client side, both reduced to ``read_message`` /
``write_message`` / ``close`` so the existing :class:`AcpServer` /
:class:`AcpClient` dispatch loops drop in unchanged.

Stream model:

* One long-lived **per-connection** GET SSE stream per
  ``Acp-Connection-Id``. Carries responses, notifications, and
  server→client requests for **all** sessions on that connection.
* Server demultiplexes by including ``sessionId`` in each JSON-RPC
  message body. The transport itself is session-agnostic — it just
  pumps framed bytes.
* SSE keepalive comments (``: keepalive\\n\\n``) every 15s. Emitted by
  the Starlette layer (``sse-starlette`` does this for us); not the
  transport's concern.
* JSON-RPC batch arrays are forbidden over HTTP. The dispatcher's
  ``parse_message`` already rejects them with ``-32600``; the HTTP
  layer additionally returns 501 at the wire boundary so the client
  sees the framing-level rejection consistently.

Both transports use raw bytes on the wire. The body of each SSE
``data:`` event is one full JSON-RPC message — no framing wrapper, just
the JSON object on a single ``data:`` line per the SSE spec. POSTs
carry the body verbatim; the response is either inline (200 with the
JSON-RPC response body) or 202+SSE (the response lands on the GET
stream).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from .transport import Transport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Server-side
# ---------------------------------------------------------------------------


class HttpServerTransport(Transport):
    """Server-side Streamable HTTP transport.

    Wraps a per-connection state object: a queue of pending POST bodies
    (set by the Starlette layer when a POST arrives) plus an outbound
    queue that the GET SSE stream drains. The dispatcher reads from the
    in-queue and writes to the out-queue; the Starlette layer is the
    bridge to actual sockets.

    Lifetime is exactly one ACP connection (one
    ``Acp-Connection-Id``). When the connection closes (DELETE or
    client disconnect), :meth:`close` is called — that drains the
    out-queue and signals EOF to the dispatcher, which then unwinds
    cleanly.

    All sessions on the connection multiplex onto this one out-queue /
    SSE stream, demultiplexed solely by the ``sessionId`` in each
    message body; the transport itself holds no per-session state.
    """

    # Sentinel pushed onto the in-queue to signal end-of-stream.
    _CLOSED = object()

    def __init__(
        self,
        *,
        connection_id: str,
        out_queue: asyncio.Queue[bytes | None],
    ) -> None:
        self._connection_id = connection_id
        # Inbound POST bodies. Each entry is one raw JSON-RPC message
        # body (bytes). ``None`` is the close sentinel.
        self._in_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        # Outbound SSE events. Each entry is one raw JSON-RPC message
        # body that the SSE stream emits as a ``data:`` line. ``None``
        # signals stream close.
        self._out_queue = out_queue
        self._closed = False
        # Per-pending-POST awaitable replies (request id -> Future).
        # The ASGI POST handler pre-registers a future BEFORE pushing
        # the body, so a synchronous (200-with-body) response can be
        # returned to the caller. If the dispatcher writes the response
        # before the SSE stream picks it up, we resolve the future
        # inline; otherwise the message goes out via SSE.
        self._sync_replies: dict[Any, asyncio.Future[bytes]] = {}

    @property
    def connection_id(self) -> str:
        return self._connection_id

    # ------------------------------------------------------------------
    # ASGI-side helpers — called by ``http_server.py``.
    # ------------------------------------------------------------------

    def push_inbound(self, body: bytes) -> None:
        """Enqueue a POST body for the dispatcher to read next.

        Called by the Starlette POST handler after header validation
        and (optional) sync-reply future registration.
        """
        if self._closed:
            return
        self._in_queue.put_nowait(body)

    def register_sync_reply(self, request_id: Any) -> asyncio.Future[bytes]:
        """Register a future that will resolve to the response bytes
        for ``request_id`` if it lands during the POST handler's
        request lifetime, allowing the handler to return 200 with the
        body inline. The handler awaits the future with a short timeout
        and falls back to 202 if the future hasn't resolved.
        """
        loop = asyncio.get_event_loop()
        fut: asyncio.Future[bytes] = loop.create_future()
        self._sync_replies[request_id] = fut
        return fut

    def discard_sync_reply(self, request_id: Any) -> None:
        """Remove a pending sync-reply future (timeout, error, etc.)."""
        self._sync_replies.pop(request_id, None)

    # ------------------------------------------------------------------
    # Transport ABC.
    # ------------------------------------------------------------------

    async def read_message(self) -> bytes:
        item = await self._in_queue.get()
        if item is None:
            # Mimic the ``EOF before any header bytes`` convention used
            # by the stdio transport so :meth:`AcpServer.serve_until_close`
            # exits cleanly.
            from .jsonrpc import JsonRpcFramingError

            raise JsonRpcFramingError("EOF before any header bytes")
        return item

    async def write_message(self, payload: bytes) -> None:
        if self._closed:
            return
        # Emit exactly one validated NDJSON line. Reject a
        # corrupt multi-line payload BEFORE it reaches the wire (raises), and
        # append the single ``\n`` delimiter every transport shares.
        from .jsonrpc import frame_ndjson_line

        framed = frame_ndjson_line(payload)
        # Try to satisfy any registered sync-reply future first.
        sync_id = _peek_response_id(payload)
        if sync_id is not None:
            fut = self._sync_replies.pop(sync_id, None)
            if fut is not None and not fut.done():
                fut.set_result(framed)
                return
        # Otherwise route via the SSE stream.
        await self._out_queue.put(framed)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Wake up any pending read.
        self._in_queue.put_nowait(None)
        # Signal SSE stream to terminate.
        try:
            self._out_queue.put_nowait(None)
        except Exception:  # noqa: BLE001 — close is best-effort
            pass
        # Fail any outstanding sync-reply futures.
        for fut in self._sync_replies.values():
            if not fut.done():
                fut.cancel()
        self._sync_replies.clear()


def _peek_response_id(payload: bytes) -> Any:
    """Best-effort extract the JSON-RPC ``id`` from a serialized response.

    Returns the id (str/int) if the message is a response/error, else
    ``None``. Used to route synchronous-200 replies. Wrong inputs (a
    notification, a server-initiated request) reduce to ``None``, which
    is fine — the message just goes out via SSE.
    """
    try:
        obj = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(obj, dict):
        return None
    if "method" in obj:
        # Server-initiated request or notification — not a response.
        return None
    return obj.get("id")


# ---------------------------------------------------------------------------
# Client-side
# ---------------------------------------------------------------------------


class HttpClientTransport(Transport):
    """Client-side Streamable HTTP transport.

    The reader half streams events off the GET SSE connection. The
    writer half POSTs each outgoing message; if the POST returns 200
    with a body, the body is one synchronous JSON-RPC response which
    we feed back into the read queue so the upstream :class:`AcpClient`
    sees it the same way it would see any other inbound message. If
    the POST returns 202 with no body, the response will arrive on the
    SSE stream; we just don't synthesize anything locally.
    """

    def __init__(
        self,
        *,
        post: Callable[[bytes], Awaitable[_PostResult]],
        sse_iterator: AsyncIterator[bytes],
        on_close: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._post = post
        self._sse_iterator = sse_iterator
        self._on_close = on_close
        self._closed = False
        # Inbound queue: SSE stream + synthesized 200-with-body replies
        # land here; ``read_message`` drains it.
        self._in_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._sse_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Launch the background SSE pump."""
        if self._sse_task is not None:
            return
        self._sse_task = asyncio.create_task(self._pump_sse(), name="acp-http-client-sse")

    async def _pump_sse(self) -> None:
        try:
            async for body in self._sse_iterator:
                if self._closed:
                    return
                if not body:
                    continue
                self._in_queue.put_nowait(body)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            # A long-lived SSE stream ending with the peer closing the
            # connection mid-body ("incomplete chunked read") is a normal
            # EOF, not a transport fault — log it quietly and let the
            # ``finally`` signal EOF so the upstream reopens cleanly.
            # Anything else is a real crash.
            if _is_peer_closed(exc):
                logger.debug("acp-http-client SSE stream closed by peer (EOF)")
            else:
                logger.exception("acp-http-client SSE pump crashed")
        finally:
            # Signal EOF to any pending reader.
            self._in_queue.put_nowait(None)

    async def read_message(self) -> bytes:
        item = await self._in_queue.get()
        if item is None:
            from .jsonrpc import JsonRpcFramingError

            raise JsonRpcFramingError("EOF before any header bytes")
        return item

    async def write_message(self, payload: bytes) -> None:
        if self._closed:
            raise ConnectionError("transport closed")
        result = await self._post(payload)
        if result.status_code == 200 and result.body:
            # Synchronous response — feed back into the read queue so
            # the dispatcher's normal inbound path picks it up.
            self._in_queue.put_nowait(result.body)
        elif result.status_code == 202:
            # Async response — the GET SSE pump will deliver it.
            return
        else:
            # Anything else is an error. Synthesize a JSON-RPC error
            # body so the read loop can surface it via the pending
            # future on the originating request id.
            err_id = _peek_request_id(payload)
            err_body = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": err_id,
                    "error": {
                        "code": -32000,
                        "message": (f"HTTP {result.status_code}: {result.body.decode('utf-8', 'replace')[:200]}"),
                    },
                },
                separators=(",", ":"),
            ).encode("utf-8")
            self._in_queue.put_nowait(err_body)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Cancel SSE pump.
        if self._sse_task is not None:
            self._sse_task.cancel()
            try:
                await self._sse_task
            except (asyncio.CancelledError, Exception):
                pass
        # Wake any pending reader.
        self._in_queue.put_nowait(None)
        if self._on_close is not None:
            try:
                await self._on_close()
            except Exception:  # noqa: BLE001 — close is best-effort
                logger.debug("on_close raised", exc_info=True)


def _is_peer_closed(exc: BaseException) -> bool:
    """True when ``exc`` represents the peer closing the SSE stream
    without completing the chunked body — a normal end-of-stream for a
    long-lived SSE connection, not a transport fault.

    httpx (and its httpcore backing) both raise a ``RemoteProtocolError``
    when the server tears the stream down mid-body. We match by exact
    type name + a closed set of known peer-close message substrings so a
    genuine ``RemoteProtocolError`` that is NOT a peer-close is still
    logged as a crash (not silently swallowed). The transport layer
    stays wire-only: no hard httpx import (a client-side dep).

    Known peer-close messages (httpx/httpcore, observed in production):
      * "peer closed connection without sending complete message body
         (incomplete chunked read)"  — httpx wrapper
      * "incomplete chunked read"     — httpcore inner
      * "Server disconnected" / "connection closed" — HTTP/2 / variant
    """
    if type(exc).__name__ != "RemoteProtocolError":
        return False
    msg = str(exc).lower()
    return (
        "peer closed connection" in msg
        or "incomplete chunked read" in msg
        or "server disconnected" in msg
        or "connection closed" in msg
    )


def _peek_request_id(payload: bytes) -> Any:
    """Best-effort extract a request ``id`` from an outgoing payload."""
    try:
        obj = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(obj, dict):
        return None
    return obj.get("id")


class _PostResult:
    """Tiny value object for ``HttpClientTransport``'s ``post`` callback."""

    __slots__ = ("status_code", "body")

    def __init__(self, status_code: int, body: bytes) -> None:
        self.status_code = status_code
        self.body = body


__all__ = [
    "HttpServerTransport",
    "HttpClientTransport",
    "_PostResult",
]
