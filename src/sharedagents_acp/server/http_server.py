"""Starlette ASGI app for the Streamable HTTP transport.

This module is the wire-frontend for the per-connection ACP HTTP
transport. It exposes three routes under ``/acp``:

- ``POST /acp`` — header-validate, route the body into the dispatcher's
  inbound queue, then either return 200 with the JSON-RPC response
  inline (for ``initialize`` and any other method that resolves before
  the handler's short timeout) or 202 with no body (in which case the
  response lands on the GET SSE stream).
- ``GET /acp`` — opens the long-lived per-connection SSE stream. One
  stream per ``Acp-Connection-Id`` carries responses, notifications,
  and server→client requests for **all** sessions on that connection.
- ``DELETE /acp`` — tears down the connection: closes the SSE stream,
  ends the dispatcher, and calls ``backend.close_session`` on every
  session associated with the connection.

The connection registry is a single in-memory dict keyed by
``Acp-Connection-Id``. Each entry owns one :class:`HttpServerTransport`
+ one :class:`AcpServer` task + the set of session ids the client
opened on that connection (tracked by sniffing ``session/new`` /
``session/load`` responses).

The middleware:

- **Bearer auth** (optional). A constant-time comparison against the
  configured token; 401 on mismatch. Disabled when ``bearer_token``
  is ``None``.
- **Header validation** is per-route (not global) because the
  required-header set differs per verb.

Everything else — JSON-RPC parsing, ContentBlock policing, MCP
gating, dispatcher state — is unchanged. Only the wire framing
differs from stdio.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import secrets
from collections.abc import Mapping
from typing import Any

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

try:
    from sse_starlette.sse import EventSourceResponse
except ImportError:  # pragma: no cover - dev-deps required
    EventSourceResponse = None  # type: ignore[assignment]

from ..protocol.http import HttpServerTransport
from .backend import AgentBackend
from .server import AcpServer

logger = logging.getLogger(__name__)


# Time the POST handler waits for a synchronous response before
# falling back to 202. Keep small — the inline-response optimization
# is only worthwhile when the dispatcher already has the answer ready
# (e.g. ``initialize``).
_SYNC_REPLY_TIMEOUT_S: float = 1.0

# How long the SSE stream waits between keepalive comments.
_SSE_KEEPALIVE_S: int = 15


# ---------------------------------------------------------------------------
# Per-connection state
# ---------------------------------------------------------------------------


class _ConnectionState:
    """One ACP HTTP connection's worth of mutable state."""

    def __init__(self, connection_id: str) -> None:
        self.connection_id = connection_id
        self.out_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.transport = HttpServerTransport(connection_id=connection_id, out_queue=self.out_queue)
        self.session_ids: set[str] = set()
        self.dispatcher_task: asyncio.Task[None] | None = None
        self.cookies_to_set: dict[str, str] = {}

    def teardown(self) -> None:
        if self.dispatcher_task is not None and not self.dispatcher_task.done():
            self.dispatcher_task.cancel()


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class _BearerAuthMiddleware(BaseHTTPMiddleware):
    """Constant-time bearer-token check.

    Disabled when ``token`` is ``None``. Compares the
    ``Authorization: Bearer <token>`` header against the configured
    token using ``hmac.compare_digest`` (constant-time).
    """

    def __init__(self, app: Any, token: str | None) -> None:
        super().__init__(app)
        self._token = token

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        if self._token is None:
            return await call_next(request)
        # Don't gate non-/acp paths (the daemon may serve a health
        # endpoint without auth in future); we keep the middleware
        # narrowly scoped to the protected surface.
        if not request.url.path.startswith("/acp"):
            return await call_next(request)
        header = request.headers.get("authorization", "")
        prefix = "Bearer "
        if not header.startswith(prefix):
            return Response(status_code=401, content=b"missing Bearer token")
        offered = header[len(prefix) :].strip()
        if not hmac.compare_digest(offered, self._token):
            return Response(status_code=401, content=b"invalid Bearer token")
        return await call_next(request)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def make_acp_app(
    *,
    backend: AgentBackend,
    agent_capabilities: Any,
    agent_info: Any,
    auth_methods: list[Any] | None = None,
    bearer_token: str | None = None,
) -> Starlette:
    """Build the Starlette ASGI app for the Streamable HTTP transport.

    Args:
        backend: the dispatcher backend (same Protocol used by stdio
            and WS transports).
        agent_capabilities / agent_info / auth_methods: forwarded
            verbatim into every per-connection :class:`AcpServer`.
        bearer_token: optional Bearer token for auth middleware. When
            ``None``, no auth is enforced. When set, requests must
            send ``Authorization: Bearer <token>`` exactly.

    Returns:
        A Starlette app with three routes on ``/acp``. Mount under
        Hypercorn (or any ASGI server) bound to the configured port.
    """

    if EventSourceResponse is None:
        raise RuntimeError(
            "sse-starlette not installed — Streamable HTTP transport "
            "requires sse-starlette>=2.1 (see pyproject [dev] deps)"
        )

    connections: dict[str, _ConnectionState] = {}
    auth_methods_l: list[Any] = list(auth_methods or [])

    # ------------------------------------------------------------------
    # Helpers shared by routes.
    # ------------------------------------------------------------------

    def _spawn_connection(connection_id: str) -> _ConnectionState:
        state = _ConnectionState(connection_id)
        connections[connection_id] = state
        wrapped_backend = _BackendSessionTracker(backend, state.session_ids)
        server = AcpServer(
            state.transport,
            wrapped_backend,
            agent_capabilities=agent_capabilities,
            agent_info=agent_info,
            auth_methods=auth_methods_l,
        )
        state.dispatcher_task = asyncio.create_task(
            server.serve_until_close(),
            name=f"acp-http-dispatcher-{connection_id[:8]}",
        )
        return state

    async def _drop_connection(state: _ConnectionState) -> None:
        # Close every session the client opened on this connection.
        for sid in list(state.session_ids):
            try:
                await backend.close_session(sid)
            except Exception:
                logger.exception("close_session(%s) failed during DELETE teardown", sid)
        state.session_ids.clear()
        # Close the transport — this drives serve_until_close to exit.
        await state.transport.close()
        if state.dispatcher_task is not None:
            try:
                await asyncio.wait_for(state.dispatcher_task, timeout=2.0)
            except asyncio.TimeoutError:
                state.dispatcher_task.cancel()
                try:
                    await state.dispatcher_task
                except (asyncio.CancelledError, Exception):
                    pass
        connections.pop(state.connection_id, None)

    # ------------------------------------------------------------------
    # Routes.
    # ------------------------------------------------------------------

    async def post_acp(request: Request) -> Response:
        # 1. Content-Type gate.
        ctype = request.headers.get("content-type", "")
        if not _is_json_ctype(ctype):
            return Response(
                status_code=415,
                content=b"Content-Type must be application/json",
            )

        body = await request.body()

        # 2. Reject batch arrays (ACP HTTP transport: 501).
        try:
            decoded = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return Response(status_code=400, content=b"malformed JSON-RPC body")
        if isinstance(decoded, list):
            # Reject a top-level batch array with HTTP 501
            # (the HTTP-transport wire-boundary signal) AND the JSON-RPC
            # ``-32600`` (INVALID_REQUEST) envelope in the body, so a caller
            # sees the SAME framing-level rejection code every other transport
            # raises centrally in ``parse_message``.
            from ..protocol.jsonrpc import JsonRpcErrorCode

            return JSONResponse(
                status_code=501,
                content={
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": int(JsonRpcErrorCode.INVALID_REQUEST),
                        "message": "JSON-RPC batch requests are not supported",
                    },
                },
            )
        if not isinstance(decoded, dict):
            return Response(
                status_code=400,
                content=b"JSON-RPC body must be an object",
            )

        method = decoded.get("method")
        request_id = decoded.get("id")  # may be None for notifications

        # 3. Header rules.
        # `initialize` is the only method allowed to *not* carry an
        # ``Acp-Connection-Id`` — we mint one. Everything else
        # requires it.
        connection_id = request.headers.get("acp-connection-id")
        if method == "initialize":
            if connection_id is not None and connection_id not in connections:
                # Client gave us a connection id that's not active —
                # treat as 404 like any other unknown id.
                return Response(
                    status_code=404,
                    content=b"unknown Acp-Connection-Id",
                )
            if connection_id is None:
                connection_id = secrets.token_urlsafe(16)
                _spawn_connection(connection_id)
            state = connections[connection_id]
        else:
            if connection_id is None:
                return Response(
                    status_code=400,
                    content=b"Acp-Connection-Id header required",
                )
            state = connections.get(connection_id)
            if state is None:
                return Response(
                    status_code=404,
                    content=b"unknown Acp-Connection-Id",
                )
            # Session-scoped methods require Acp-Session-Id when the
            # JSON-RPC body doesn't already carry it. The ACP spec
            # makes the header authoritative; we accept either source
            # but require *one* of them.
            if isinstance(method, str) and method.startswith("session/"):
                # `session/new` and `session/list` don't carry a
                # session id (they create / enumerate). Everything
                # else does.
                needs_sid = method not in {"session/new", "session/list"}
                if needs_sid:
                    sid_hdr = request.headers.get("acp-session-id")
                    body_sid = (
                        decoded.get("params", {}).get("sessionId") if isinstance(decoded.get("params"), dict) else None
                    )
                    if sid_hdr is None and body_sid is None:
                        return Response(
                            status_code=400,
                            content=b"Acp-Session-Id header required for session-scoped methods",
                        )

        # 4. Register sync-reply waiter ONLY for ``initialize``.
        # Other methods route their response over the SSE GET stream
        # so the relative ordering of ``session/update`` notifications
        # and the final ``PromptResponse`` is preserved on the wire.
        # ``initialize`` is special because the GET stream isn't open
        # yet (the client needs the ``Acp-Connection-Id`` first), so
        # the response must come back inline.
        sync_fut: asyncio.Future[bytes] | None = None
        wants_sync = method == "initialize" and request_id is not None
        if wants_sync:
            sync_fut = state.transport.register_sync_reply(request_id)

        # 5. Forward into dispatcher.
        state.transport.push_inbound(body)

        # 6. Build response.
        headers: dict[str, str] = {}
        if method == "initialize":
            headers["Acp-Connection-Id"] = state.connection_id

        # Cookie passthrough: any cookies the request sent are echoed
        # back. sharedagents doesn't issue any, but the spec requires the
        # transport not to drop them silently.
        for name, value in request.cookies.items():
            headers.setdefault("Set-Cookie", f"{name}={value}")

        if sync_fut is None:
            # Notification or method that prefers async response —
            # 202 with no body. Response (if any) lands on SSE.
            if request_id is not None:
                # Sniff session/new + session/load so DELETE can
                # cascade close_session even when responses go via
                # SSE rather than inline. The wrapped backend records
                # this when the dispatcher invokes new_session /
                # load_session, so we don't need to peek at the
                # response body here — the BackendSessionTracker
                # has already done it.
                pass
            return Response(status_code=202, headers=headers)

        try:
            payload = await asyncio.wait_for(sync_fut, _SYNC_REPLY_TIMEOUT_S)
        except asyncio.TimeoutError:
            state.transport.discard_sync_reply(request_id)
            # Async path — response will land on SSE stream.
            return Response(status_code=202, headers=headers)
        # Track sessions opened/loaded so DELETE can cascade.
        _maybe_track_session(state, decoded, payload)
        return Response(
            status_code=200,
            content=payload,
            media_type="application/json",
            headers=headers,
        )

    async def get_acp(request: Request) -> Response:
        # 1. Accept gate.
        accept = request.headers.get("accept", "")
        if "text/event-stream" not in accept:
            return Response(
                status_code=406,
                content=b"Accept must include text/event-stream",
            )
        # 2. Connection id required + must be known.
        connection_id = request.headers.get("acp-connection-id")
        if not connection_id:
            return Response(
                status_code=400,
                content=b"Acp-Connection-Id header required",
            )
        state = connections.get(connection_id)
        if state is None:
            return Response(
                status_code=404,
                content=b"unknown Acp-Connection-Id",
            )

        async def event_publisher():
            # Emit one initial keepalive comment so HTTP clients (and
            # in-process ASGI transports) that buffer the response
            # until the first chunk lands see the stream open
            # immediately. The SSE spec treats lines starting with
            # ``:`` as comments; clients ignore them.
            yield {"comment": "stream-ready"}
            # sse-starlette emits ": ping\n\n" as keepalive every
            # ``ping`` seconds when the queue is idle — same wire
            # shape the ACP HTTP transport asks for (an SSE comment
            # line every 15s).
            while True:
                try:
                    item = await asyncio.wait_for(state.out_queue.get(), timeout=_SSE_KEEPALIVE_S)
                except asyncio.TimeoutError:
                    # Yield a comment to keep the connection alive.
                    yield {"comment": "keepalive"}
                    continue
                if item is None:
                    # DELETE or transport close.
                    return
                # The transport frames each message as one NDJSON line;
                # strip that trailing delimiter here so the
                # SSE ``data:`` field carries the bare JSON object (SSE supplies
                # its own ``\n\n`` event framing).
                yield {"data": item.decode("utf-8").rstrip("\n")}

        return EventSourceResponse(
            event_publisher(),
            ping=_SSE_KEEPALIVE_S,
            headers={"X-Accel-Buffering": "no"},
        )

    async def delete_acp(request: Request) -> Response:
        connection_id = request.headers.get("acp-connection-id")
        if not connection_id:
            return Response(
                status_code=400,
                content=b"Acp-Connection-Id header required",
            )
        state = connections.get(connection_id)
        if state is None:
            return Response(
                status_code=404,
                content=b"unknown Acp-Connection-Id",
            )
        await _drop_connection(state)
        return Response(status_code=204)

    routes = [
        Route("/acp", post_acp, methods=["POST"]),
        Route("/acp", get_acp, methods=["GET"]),
        Route("/acp", delete_acp, methods=["DELETE"]),
    ]

    middleware = []
    if bearer_token is not None:
        middleware.append(Middleware(_BearerAuthMiddleware, token=bearer_token))

    app = Starlette(routes=routes, middleware=middleware)
    # Stash connection registry on app.state so tests can inspect.
    app.state.acp_connections = connections  # type: ignore[attr-defined]
    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_json_ctype(value: str) -> bool:
    """Accept ``application/json`` with optional ``; charset=utf-8``."""
    if not value:
        return False
    primary = value.split(";", 1)[0].strip().lower()
    return primary == "application/json"


def _maybe_track_session(
    state: _ConnectionState,
    request_obj: Mapping[str, Any],
    response_bytes: bytes,
) -> None:
    """If the request was ``session/new`` or ``session/load`` and the
    response carries a ``sessionId``, remember it so DELETE can
    cascade ``backend.close_session`` later."""
    method = request_obj.get("method")
    if method not in {"session/new", "session/load"}:
        return
    try:
        resp = json.loads(response_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return
    result = resp.get("result") if isinstance(resp, dict) else None
    if not isinstance(result, dict):
        return
    sid = result.get("sessionId")
    if isinstance(sid, str):
        state.session_ids.add(sid)
    elif method == "session/load":
        # session/load echoes the session id back in the request body.
        params = request_obj.get("params")
        if isinstance(params, dict) and isinstance(params.get("sessionId"), str):
            state.session_ids.add(params["sessionId"])


class _BackendSessionTracker:
    """Wraps an :class:`AgentBackend` so the HTTP layer learns about
    sessions even when the dispatcher returns a synchronous response
    via SSE (and the POST handler doesn't see the body).

    Forwards every method 1:1; intercepts ``new_session`` /
    ``load_session`` to record the session id, and ``close_session``
    to forget it (so we don't double-close on DELETE).
    """

    def __init__(self, inner: AgentBackend, session_ids: set[str]) -> None:
        self._inner = inner
        self._session_ids = session_ids

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def initialize(self, *args: Any, **kwargs: Any) -> Any:
        return await self._inner.initialize(*args, **kwargs)

    async def authenticate(self, *args: Any, **kwargs: Any) -> Any:
        return await self._inner.authenticate(*args, **kwargs)

    async def new_session(self, cwd: str, mcp_servers: list[Mapping[str, Any]]) -> Mapping[str, Any]:
        result = await self._inner.new_session(cwd, mcp_servers)
        sid = result.get("sessionId") if isinstance(result, Mapping) else None
        if isinstance(sid, str):
            self._session_ids.add(sid)
        return result

    async def load_session(
        self,
        session_id: str,
        cwd: str,
        mcp_servers: list[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        result = await self._inner.load_session(session_id, cwd, mcp_servers)
        self._session_ids.add(session_id)
        return result

    async def list_sessions(self, cursor: str | None, cwd: str | None) -> Mapping[str, Any]:
        return await self._inner.list_sessions(cursor, cwd)

    async def close_session(self, session_id: str) -> None:
        try:
            await self._inner.close_session(session_id)
        finally:
            self._session_ids.discard(session_id)

    async def set_mode(self, session_id: str, mode_id: str) -> None:
        return await self._inner.set_mode(session_id, mode_id)

    async def set_config_option(self, session_id: str, config_id: str, value: Any) -> None:
        return await self._inner.set_config_option(session_id, config_id, value)

    async def inject_user_message(
        self,
        session_id: str,
        text: str,
    ) -> Mapping[str, Any]:
        inner = getattr(self._inner, "inject_user_message", None)
        if inner is None:
            raise NotImplementedError("wrapped backend does not implement inject_user_message")
        return await inner(session_id, text)

    def prompt(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.prompt(*args, **kwargs)

    async def cancel(self, session_id: str, *, force: bool = False) -> None:
        return await self._inner.cancel(session_id, force=force)


__all__ = ["make_acp_app"]
