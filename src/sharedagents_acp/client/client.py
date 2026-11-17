"""``AcpClient`` — outbound ACP client.

Drives a remote ACP-compliant agent across any ``Transport``. The
JSON-RPC framing, parsing, and transport abstraction are all from
``sharedagents_acp.protocol``; we layer request/response correlation,
notification streaming, and inbound-method dispatch on top.

Concurrency model:

- One **read loop** runs as a background task for the lifetime of the
  client. It reads framed messages off the transport, decides whether
  each is a response (correlate by ``id`` to a pending future), an
  inbound request from the agent (dispatch in a tracked task via the
  handler registry, send a response), or an inbound notification (route
  to per-session prompt-stream queues for ``session/update``, drop
  everything else with a debug warning).
- Every outbound **request** allocates an integer id, registers a
  ``Future`` in ``_pending``, and writes the framed request to the
  transport. The read loop completes the future when the matching
  response arrives.
- Every outbound **notification** (``session/cancel``) is fire-and-forget;
  no future, no waiting.
- ``session_prompt`` is special: it allocates an ``asyncio.Queue`` for
  ``session/update`` notifications scoped to the prompt's session, fires
  the ``session/prompt`` request, and returns an async iterator that
  drains the queue until the matching ``PromptResponse`` lands.

Ids are positive integers chosen by ``itertools.count(1)``. The spec
allows null/string/int; positive integers are simplest and match what
the SDK uses too. We never reuse an id within a single client lifetime.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any, Union

# The SDK's schema dataclasses live in ``acp.schema``; the top-level
# ``acp`` re-exports a curated subset (``InitializeRequest`` etc.) but
# NOT capability containers (``ClientCapabilities``,
# ``FileSystemCapabilities``, etc.). We grab the full module so every
# model is reachable from one symbol — keeps the rest of this file
# uniform regardless of which curated subset the SDK chooses to expose.
import acp.schema as _acp  # noqa: E402

from ..protocol.jsonrpc import (
    JsonRpcError,
    JsonRpcErrorCode,
    JsonRpcFramingError,
    JsonRpcMessage,
    JsonRpcNotification,
    JsonRpcParseError,
    JsonRpcRequest,
    JsonRpcResponse,
    parse_message,
)
from ..protocol.schema import PROTOCOL_VERSION as _PROTOCOL_VERSION
from ..protocol.transport import Transport
from .handlers import (
    ClientHandlerRegistry,
    Handler,
    TerminalHandlers,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types we want the caller to see (re-exported via ``__init__``)
# ---------------------------------------------------------------------------

# A prompt stream yields any number of ``SessionNotification`` updates,
# then exactly one final ``PromptResponse``. We type both as ``Any``
# at the public boundary because both are pydantic models from the
# vendored SDK; the user can ``isinstance``-narrow against
# ``schema_models.SessionNotification`` / ``.PromptResponse``.
PromptStreamItem = Union["_acp.SessionNotification", "_acp.PromptResponse"]


class _Sentinels:
    """Module-private sentinels for the prompt stream queue.

    A queue with mixed types is fine, but using ``None`` for end-of-stream
    is fragile if the agent's notification model ever exposes a
    falsy-empty variant. A dedicated singleton sentinel is clearer.
    """

    END = object()


# ---------------------------------------------------------------------------
# Errors raised by this module
# ---------------------------------------------------------------------------


class AcpClientError(Exception):
    """Base for client-side ACP errors raised on the request path."""


class AcpRemoteError(AcpClientError):
    """The agent returned a JSON-RPC error response.

    ``code`` / ``message`` / ``data`` mirror the JSON-RPC error object
    so callers can branch on ``code`` (e.g. -32601 methodNotFound).
    """

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"ACP remote error {code}: {message}")
        self.code = code
        self.message = message
        self.data = data


class AcpTransportClosed(AcpClientError):
    """Transport closed before the awaited response arrived."""


class AcpCapabilityError(AcpClientError):
    """Method requires a capability the agent did not advertise."""


# ---------------------------------------------------------------------------
# AcpClient
# ---------------------------------------------------------------------------


class AcpClient:
    """Outbound ACP client over a ``Transport``.

    Construct directly with an existing transport, or via
    ``spawn_stdio_agent`` for the common subprocess case. The client
    is **not started** at construction — call ``start()`` (or use it
    as an async context manager) to launch the read loop, then
    ``initialize()`` to negotiate.

    Multiple in-flight requests are fine; each gets its own ``Future``.
    Only one read loop runs, so messages are processed in arrival
    order.
    """

    def __init__(
        self,
        transport: Transport,
        *,
        client_capabilities: _acp.ClientCapabilities | None = None,
        client_info: _acp.Implementation | None = None,
        protocol_version: int = _PROTOCOL_VERSION,
        frontend: str | None = None,
    ) -> None:
        self._transport = transport
        self._client_capabilities = client_capabilities or _acp.ClientCapabilities()
        self._client_info = client_info
        self._protocol_version = protocol_version
        # The frontend this client speaks for (tui/telegram/…).
        # Declared to the agent at ``initialize`` so the server can pin it as the
        # non-spoofable ceiling for mode-setting on this connection.
        self._frontend = frontend

        self._next_id = itertools.count(1)
        self._pending: dict[int, asyncio.Future[JsonRpcResponse]] = {}
        # Per-session queues for streaming session/update notifications
        # while a session/prompt is in flight. Keyed by sessionId.
        self._prompt_streams: dict[str, asyncio.Queue[Any]] = {}

        self._handlers = ClientHandlerRegistry()
        self._read_task: asyncio.Task[None] | None = None
        # Terminal waits and permission decisions may outlive other requests.
        # Keep them off the sole reader so their own completion/control
        # messages, prompt updates, and unrelated responses remain readable.
        self._inbound_tasks: set[asyncio.Task[None]] = set()
        self._closed = False
        self._write_lock = asyncio.Lock()

        # Captured from initialize() response.
        self.agent_capabilities: _acp.AgentCapabilities | None = None
        self.agent_info: _acp.Implementation | None = None
        self.auth_methods: list = []
        self.negotiated_protocol_version: int | None = None
        # Raw session configuration advertised by the driven ACP server.
        # Keep this outside the generated models because vendors can add
        # config options before the vendored schema learns about them.
        self.last_session_config_options: list[Mapping[str, Any]] = []
        self.last_session_modes: Mapping[str, Any] | None = None
        self.last_session_new: dict[str, Any] = {}

    # -- lifecycle ----------------------------------------------------

    async def start(self) -> None:
        """Launch the background read loop."""

        if self._read_task is not None:
            return
        self._read_task = asyncio.create_task(self._read_loop(), name="acp-client-read")

    async def __aenter__(self) -> AcpClient:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        """Stop the read loop, close the transport, fail pending futures."""

        # Cancel dispatch once, before transport.close() can yield to EOF
        # cleanup. A second cancellation could interrupt callback finalizers.
        if not self._closed and self._read_task is not None:
            self._read_task.cancel()
        self._closed = True
        try:
            await self._transport.close()
        except Exception:
            logger.debug("transport close raised", exc_info=True)
        if self._read_task is not None:
            try:
                await self._read_task
            except (asyncio.CancelledError, Exception):
                pass
        await self._cancel_inbound_tasks()
        # Fail any pending futures.
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(AcpTransportClosed("transport closed"))
        self._pending.clear()
        # Drain any open prompt streams.
        for q in self._prompt_streams.values():
            q.put_nowait(_Sentinels.END)
        self._prompt_streams.clear()

    async def _cancel_inbound_tasks(self) -> None:
        tasks = list(self._inbound_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _inbound_request_done(self, task: asyncio.Task[None]) -> None:
        self._inbound_tasks.discard(task)
        if not task.cancelled():
            exc = task.exception()
            if exc is not None:
                logger.error(
                    "acp-client inbound callback failed",
                    exc_info=(type(exc), exc, exc.__traceback__),
                )

    # -- handler registration ----------------------------------------

    def register_fs_read_text_file(self, handler: Handler) -> None:
        """Wire a handler for ``fs/read_text_file``.

        Raises ``ValueError`` if ``client_capabilities.fs.readTextFile``
        wasn't advertised — registering a handler the agent isn't
        allowed to call is silent footgun territory; better fail loud.
        """

        fs = self._client_capabilities.fs
        if fs is None or not fs.read_text_file:
            raise ValueError(
                "cannot register fs/read_text_file handler when client_capabilities.fs.readTextFile is not advertised"
            )
        self._handlers.register("fs/read_text_file", handler)

    def register_fs_write_text_file(self, handler: Handler) -> None:
        fs = self._client_capabilities.fs
        if fs is None or not fs.write_text_file:
            raise ValueError(
                "cannot register fs/write_text_file handler when client_capabilities.fs.writeTextFile is not advertised"
            )
        self._handlers.register("fs/write_text_file", handler)

    def register_terminal(self, handlers: TerminalHandlers) -> None:
        if not self._client_capabilities.terminal:
            raise ValueError("cannot register terminal handlers when client_capabilities.terminal is not advertised")
        self._handlers.register("terminal/create", handlers.create)
        self._handlers.register("terminal/output", handlers.output)
        self._handlers.register("terminal/wait_for_exit", handlers.wait_for_exit)
        self._handlers.register("terminal/kill", handlers.kill)
        self._handlers.register("terminal/release", handlers.release)

    def register_request_permission(self, handler: Handler) -> None:
        """``session/request_permission`` has no capability gate per spec
        — it's always available. We still gate registration so callers
        can choose whether to expose a UI."""

        self._handlers.register("session/request_permission", handler)

    def register_ask_user(self, handler: Handler) -> None:
        """``session/ask_user`` is an sharedagents extension to ACP, used to
        round-trip the model's ``AskUserQuestion`` tool through the
        connected client. No capability gate — the agent treats an
        unregistered handler as an error from the receiving end and
        the bridge falls back to a synthesized auto-answer."""

        self._handlers.register("session/ask_user", handler)

    # -- ACP request methods -----------------------------------------

    async def initialize(self) -> _acp.InitializeResponse:
        params = self._dump(
            _acp.InitializeRequest(
                protocol_version=self._protocol_version,
                client_capabilities=self._client_capabilities,
                client_info=self._client_info,
            )
        )
        # Declare the calling frontend at the handshake so the
        # agent pins it as this connection's authoritative identity.
        if self._frontend:
            params["frontend"] = self._frontend
        result = await self._call("initialize", params)
        resp = _acp.InitializeResponse.model_validate(result)
        self.agent_capabilities = resp.agent_capabilities
        self.agent_info = resp.agent_info
        self.auth_methods = list(resp.auth_methods or [])
        self.negotiated_protocol_version = resp.protocol_version
        return resp

    async def authenticate(self, method_id: str) -> Mapping[str, Any]:
        """Authenticate with the agent using one of its advertised methods.

        ``method_id`` must be one of the ids in ``self.auth_methods``
        (populated by :meth:`initialize`) — e.g. Cursor's ACP server
        advertises ``"cursor_login"``. Returns the (usually empty)
        result mapping; raises :class:`AcpRemoteError` on failure.
        """
        result = await self._call("authenticate", {"methodId": method_id})
        return result if isinstance(result, Mapping) else {}

    async def session_new(
        self,
        cwd: str | Path,
        mcp_servers: list | None = None,
    ) -> str:
        params = self._dump(
            _acp.NewSessionRequest(
                cwd=str(cwd),
                mcp_servers=list(mcp_servers or []),
            )
        )
        result = await self._call("session/new", params)
        # Capture the raw ``models`` block (a Cursor/agent extension some
        # ACP servers add to NewSessionResponse) before SDK validation,
        # which may drop unmodeled fields. Engines read this to surface
        # the real upstream model. ``None`` when the agent omits it.
        self.last_session_models = result.get("models") if isinstance(result, dict) else None
        self.last_session_config_options = list(result.get("configOptions") or []) if isinstance(result, dict) else []
        self.last_session_modes = result.get("modes") if isinstance(result, dict) else None
        self.last_session_new = result if isinstance(result, dict) else {}
        resp = _acp.NewSessionResponse.model_validate(result)
        return resp.session_id

    async def session_load(
        self,
        session_id: str,
        *,
        cwd: str | Path,
        mcp_servers: list | None = None,
    ) -> _acp.LoadSessionResponse:
        if not self._has_capability("loadSession"):
            raise AcpCapabilityError("session/load requires agent_capabilities.loadSession")
        params = self._dump(
            _acp.LoadSessionRequest(
                session_id=session_id,
                cwd=str(cwd),
                mcp_servers=list(mcp_servers or []),
            )
        )
        result = await self._call("session/load", params)
        if isinstance(result, dict) and result.get("models") is not None:
            self.last_session_models = result.get("models")
        if isinstance(result, dict) and result.get("configOptions") is not None:
            self.last_session_config_options = list(result.get("configOptions") or [])
        if isinstance(result, dict):
            if result.get("modes") is not None:
                self.last_session_modes = result.get("modes")
            self.last_session_new = result
        return _acp.LoadSessionResponse.model_validate(result)

    async def session_list(
        self,
        *,
        cursor: str | None = None,
        cwd: str | Path | None = None,
    ) -> _acp.ListSessionsResponse:
        # Capability gate: agent must advertise sessionCapabilities.list.
        caps = self.agent_capabilities
        if caps is None or caps.session_capabilities is None or caps.session_capabilities.list is None:
            raise AcpCapabilityError("session/list requires agent_capabilities.sessionCapabilities.list")
        params = self._dump(
            _acp.ListSessionsRequest(
                cursor=cursor,
                cwd=str(cwd) if cwd is not None else None,
            )
        )
        result = await self._call("session/list", params)
        return _acp.ListSessionsResponse.model_validate(result)

    async def session_close(self, session_id: str) -> None:
        params = self._dump(_acp.CloseSessionRequest(session_id=session_id))
        await self._call("session/close", params)

    async def session_set_mode(
        self,
        session_id: str,
        mode_id: str,
        frontend: str | None = None,
    ) -> None:
        params = self._dump(_acp.SetSessionModeRequest(session_id=session_id, mode_id=mode_id))
        # Declare the calling frontend so the server's capability choke-point
        # can enforce the ask-mode boundary on this raw wire
        # surface. Absent → server defaults to a non-ask-capable "acp".
        if frontend:
            params["frontend"] = frontend
        await self._call("session/set_mode", params)

    async def session_set_config_option(self, session_id: str, option_id: str, value: Any) -> Mapping[str, Any]:
        """Issue ``session/set_config_option`` against the agent.

        The canonical ACP schema has two discriminated request shapes
        (boolean / select), but sharedagents's dispatcher and bridge accept
        any JSON-shaped ``value`` for forward-compat with future option
        types. We send the wire payload directly rather than going
        through one of the SDK's validated dataclasses so the client
        stays uniform across option kinds (today: ``model`` alias as a
        string; tomorrow: any other sharedagents-defined option).
        """

        params: dict[str, Any] = {
            "sessionId": session_id,
            "configId": option_id,
            "value": value,
        }
        result = await self._call("session/set_config_option", params)
        if isinstance(result, Mapping) and result.get("configOptions") is not None:
            self.last_session_config_options = list(result.get("configOptions") or [])
        return result if isinstance(result, Mapping) else {}

    async def session_set_model(self, session_id: str, model_id: str) -> Mapping[str, Any]:
        """Issue ``session/set_model`` (Hermes and other model-block ACP servers).

        Kimi-style servers advertise ``configOptions`` and take
        ``session/set_config_option``; Hermes advertises ``models`` and
        switches on this method. An empty success object still counts as
        JSON-RPC acknowledgement.
        """
        result = await self._call(
            "session/set_model",
            {"sessionId": session_id, "modelId": model_id},
        )
        if isinstance(result, Mapping) and result.get("models") is not None:
            self.last_session_models = result.get("models")
        return result if isinstance(result, Mapping) else {}

    async def session_inject_user_message(
        self,
        session_id: str,
        text: str,
    ) -> Mapping[str, Any]:
        """Push a mid-turn user message to the agent's inject buffer.

        Returns the backend's ack dict (``{"ok": True, "queued": True}``
        on success). The agent drains injects at every round boundary
        so the operator can steer an in-flight plan without aborting.
        Does NOT require an active turn — a queued message lands at
        the start of the next prompt turn for this sid if none is
        running.
        """
        params: dict[str, Any] = {
            "sessionId": session_id,
            "text": text,
        }
        result = await self._call("session/inject_user_message", params)
        if isinstance(result, Mapping):
            return result
        return {"ok": False, "reason": "non-dict response"}

    async def session_queue_list(self, session_id: str) -> Mapping[str, Any]:
        """Return pending follow-up items for *session_id*.

        Returns the backend's ``{"ok": True, "pending": [...]}`` dict.
        """
        result = await self._call("session/queue/list", {"sessionId": session_id})
        if isinstance(result, Mapping):
            return result
        return {"ok": False, "reason": "non-dict response"}

    async def session_queue_add(
        self, session_id: str, text: str, *, source: str = "", author: str = ""
    ) -> Mapping[str, Any]:
        """Append *text* to *session_id*'s follow-up queue.

        Returns ``{"ok": True, "queue_id": "<id>"}`` on success.
        """
        params: dict[str, Any] = {
            "sessionId": session_id,
            "text": text,
            "source": source,
            "author": author,
        }
        result = await self._call("session/queue/add", params)
        if isinstance(result, Mapping):
            return result
        return {"ok": False, "reason": "non-dict response"}

    async def session_queue_update(self, session_id: str, queue_id: str, text: str) -> Mapping[str, Any]:
        """Update the text of an existing queued item *queue_id*.

        Returns ``{"ok": True}`` or ``{"ok": False, "error": "..."}`` if the
        item was not found.
        """
        params: dict[str, Any] = {
            "sessionId": session_id,
            "queueId": queue_id,
            "text": text,
        }
        result = await self._call("session/queue/update", params)
        if isinstance(result, Mapping):
            return result
        return {"ok": False, "reason": "non-dict response"}

    async def session_queue_remove(
        self,
        session_id: str,
        queue_id: str,
        *,
        reason: str | None = None,
    ) -> Mapping[str, Any]:
        """Remove queued item *queue_id* from *session_id*'s follow-up queue.

        Returns ``{"ok": True}`` or ``{"ok": False, "error": "..."}`` if not
        found. Optional ``reason`` is a named JSON-RPC field so an older
        orchestrator (which only reads ``sessionId`` / ``queueId``) ignores
        it instead of rejecting a third positional argument.
        """
        params: dict[str, Any] = {
            "sessionId": session_id,
            "queueId": queue_id,
        }
        if reason:
            params["reason"] = reason
        result = await self._call("session/queue/remove", params)
        if isinstance(result, Mapping):
            return result
        return {"ok": False, "reason": "non-dict response"}

    async def session_command(
        self,
        session_id: str,
        text: str,
        *,
        frontend: str = "tui",
        confirmed: bool = False,
    ) -> Mapping[str, Any]:
        """Run a slash command via the agent's shared server-side runtime.

        Returns the ``CommandResponse`` envelope dict
        (``{schema_version, ok, command, result:{kind,text,data,actions},
        effects, refresh, warnings, error?}``). All command logic lives in
        the agent (``sharedagents_command.command_runtime``); this is just the wire
        call. Raises if the agent is too old to support ``session/command``
        so the caller can fall back.
        """
        # ``confirmed`` rides the wire so the orchestrator's runtime
        # can execute (not re-challenge) an irreversible command.
        params: dict[str, Any] = {
            "sessionId": session_id,
            "text": text,
            "frontend": frontend,
            "confirmed": confirmed,
        }
        result = await self._call("session/command", params)
        if isinstance(result, Mapping):
            return result
        return {"ok": False, "error": "non-dict response", "error_code": "bad_response"}

    def session_prompt(
        self,
        session_id: str,
        content_blocks: list,
    ) -> AsyncIterator[PromptStreamItem]:
        """Drive one prompt turn on ``session_id``.

        Returns an async iterator that yields each ``SessionNotification``
        (``session/update``) as it arrives during the turn, followed by
        exactly one ``PromptResponse`` when the turn ends. After yielding
        the ``PromptResponse`` the iterator stops cleanly.

        Cancel mid-turn by sending ``session_cancel`` from another task —
        the agent will then return a ``PromptResponse`` with
        ``stopReason='cancelled'`` and the iterator terminates as usual.
        """

        return self._iter_prompt(session_id, content_blocks)

    async def _iter_prompt(self, session_id: str, content_blocks: list) -> AsyncIterator[PromptStreamItem]:
        # Allocate the per-session update queue BEFORE firing the
        # request — otherwise an early notification could land before
        # we've registered the queue.
        if session_id in self._prompt_streams:
            raise AcpClientError(f"a session/prompt is already in flight for session {session_id!r}")
        queue: asyncio.Queue[Any] = asyncio.Queue()
        self._prompt_streams[session_id] = queue

        # Cleanup contract: any path that raises after the queue is
        # registered MUST pop it, otherwise the session looks "in
        # flight" forever and every subsequent prompt is rejected.
        # Building the PromptRequest can raise pydantic
        # ValidationError on a malformed ContentBlock; sending the
        # request can raise on transport failure. Both must release
        # the slot. The successful path falls through to the
        # iteration loop's own ``finally`` (below) for cleanup.
        try:
            params = self._dump(
                _acp.PromptRequest(
                    session_id=session_id,
                    prompt=list(content_blocks),
                )
            )
            # Reserve the id and write the request, but await the response
            # in the iteration loop so we can interleave updates.
            req_id, fut = self._reserve_call()
            await self._send_request(req_id, "session/prompt", params)
        except BaseException:
            self._prompt_streams.pop(session_id, None)
            raise

        try:
            while True:
                # If the response landed, drain any queued updates
                # first so the iterator surfaces them in arrival order,
                # THEN yield the final PromptResponse.
                if fut.done():
                    if not queue.empty():
                        item = await queue.get()
                        if item is _Sentinels.END:
                            break
                        yield item
                        continue
                    response = fut.result()
                    if response.error is not None:
                        raise AcpRemoteError(
                            code=response.error.code,
                            message=response.error.message,
                            data=response.error.data,
                        )
                    yield _acp.PromptResponse.model_validate(response.result)
                    break

                # Wait for either the next update or the response.
                # ``asyncio.shield`` on the future is essential — we
                # cancel ``response_task`` below to release the wait;
                # without the shield that would propagate to ``fut``
                # itself and the caller's send_request would never see
                # the result the read loop set.
                queue_task = asyncio.create_task(queue.get())
                response_task = asyncio.ensure_future(asyncio.shield(fut))
                done, _pending = await asyncio.wait(
                    {queue_task, response_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if queue_task in done:
                    item = queue_task.result()
                    response_task.cancel()
                    try:
                        await response_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    if item is _Sentinels.END:
                        break
                    yield item
                else:
                    # The response arrived first. Cancel the queue.get
                    # and loop back to the ``fut.done()`` branch — that
                    # branch drains any updates that landed in the same
                    # tick before yielding the final PromptResponse.
                    queue_task.cancel()
                    try:
                        await queue_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    # fall through to top of loop where fut.done() is True
        finally:
            self._prompt_streams.pop(session_id, None)

    async def session_cancel(
        self,
        session_id: str,
        *,
        force: bool = False,
    ) -> None:
        """Fire-and-forget ``session/cancel`` notification.

        Returns as soon as the bytes are written to the transport. The
        agent answers by completing the in-flight ``session/prompt``
        with ``stopReason='cancelled'`` (per spec it MUST do so).

        ``force`` is an sharedagents extension carried alongside the standard
        ``sessionId``: ``False`` (default) requests a graceful stop (the
        in-flight tool / round finishes, then the turn ends); ``True``
        requests a hard kill (SIGKILL + killpg the worker and every
        runner / subagent immediately). The agent reads it in
        ``_handle_session_cancel``; agents that don't understand it just
        ignore the extra key and fall back to a graceful cancel.
        """

        params = self._dump(_acp.CancelNotification(session_id=session_id))
        if isinstance(params, dict):
            params["force"] = bool(force)
        await self._send_notification("session/cancel", params)

    # -- core dispatch helpers ---------------------------------------

    def _reserve_call(self) -> tuple[int, asyncio.Future[JsonRpcResponse]]:
        # Reject reservations on a dead client so callers don't await a
        # future the (already-exited) read loop will never satisfy.
        # ``close()`` and the read-loop ``finally`` both flip
        # ``_closed`` and wake pre-existing futures; this guards
        # late-comers that try to issue a new call AFTER the loop has
        # already drained ``_pending``.
        if self._closed:
            raise AcpTransportClosed("client is closed")
        req_id = next(self._next_id)
        fut: asyncio.Future[JsonRpcResponse] = asyncio.get_event_loop().create_future()
        self._pending[req_id] = fut
        return req_id, fut

    async def _call(self, method: str, params: Any) -> Any:
        """Send a request, await response, return ``result`` or raise."""

        req_id, fut = self._reserve_call()
        await self._send_request(req_id, method, params)
        try:
            response = await fut
        finally:
            self._pending.pop(req_id, None)
        if response.error is not None:
            raise AcpRemoteError(
                code=response.error.code,
                message=response.error.message,
                data=response.error.data,
            )
        return response.result

    async def _send_request(self, req_id: int, method: str, params: Any) -> None:
        body: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
        }
        if params is not None:
            body["params"] = params
        await self._write(body)

    async def _send_notification(self, method: str, params: Any) -> None:
        body: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            body["params"] = params
        await self._write(body)

    async def _send_response(
        self,
        req_id: Any,
        *,
        result: Any = None,
        error: JsonRpcError | None = None,
    ) -> None:
        body: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id}
        if error is not None:
            body["error"] = error.to_dict()
        else:
            body["result"] = result if result is not None else {}
        await self._write(body)

    async def _write(self, body: dict[str, Any]) -> None:
        payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
        # Lock so concurrent writes don't interleave on the same transport.
        async with self._write_lock:
            await self._transport.write_message(payload)

    # -- read loop ----------------------------------------------------

    async def _read_loop(self) -> None:
        try:
            while True:
                try:
                    raw = await self._transport.read_message()
                except (asyncio.IncompleteReadError, ConnectionError, OSError):
                    break
                except JsonRpcFramingError as exc:
                    # A clean end-of-stream — the SSE/stdio peer closed
                    # after delivering its last message — raises the
                    # "EOF before any header bytes" sentinel. That's a
                    # normal shutdown (every one-shot turn ends this way),
                    # NOT an error: the prompt result already arrived via
                    # session/update, and the finally-block below sends the
                    # stream-END sentinel. Don't dump a traceback for it.
                    # Genuine mid-frame corruption also lands here; the
                    # stream is unusable either way, so break — but log it.
                    if "EOF before any header bytes" not in str(exc):
                        logger.warning("acp-client framing error; closing: %s", exc)
                    break
                except Exception:
                    logger.exception("acp-client read error; closing")
                    break
                if not raw:
                    break
                try:
                    msg = parse_message(raw)
                except JsonRpcParseError:
                    logger.warning("acp-client received unparseable message; dropping")
                    continue
                await self._handle_message(msg)
        finally:
            # Mark dead BEFORE waking the futures so callers polling
            # ``_closed`` don't observe a still-alive client while the
            # finally block is mid-flight. close() will await this cleanup
            # rather than cancelling the reader a second time.
            self._closed = True
            # Wake up any pending futures so awaiters don't hang.
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(AcpTransportClosed("transport closed mid-flight"))
            self._pending.clear()
            for q in list(self._prompt_streams.values()):
                q.put_nowait(_Sentinels.END)
            self._prompt_streams.clear()
            await self._cancel_inbound_tasks()

    async def _handle_message(self, msg: JsonRpcMessage) -> None:
        if isinstance(msg, JsonRpcResponse):
            fut = self._pending.get(msg.id) if isinstance(msg.id, int) else None
            if fut is None or fut.done():
                logger.debug("acp-client got response for unknown id %r", msg.id)
                return
            fut.set_result(msg)
            return

        if isinstance(msg, JsonRpcNotification):
            await self._handle_notification(msg)
            return

        if isinstance(msg, JsonRpcRequest):
            task = asyncio.create_task(
                self._handle_inbound_request(msg),
                name=f"acp-client-callback-{msg.method}",
            )
            self._inbound_tasks.add(task)
            task.add_done_callback(self._inbound_request_done)
            return

    async def _handle_notification(self, msg: JsonRpcNotification) -> None:
        if msg.method == "session/update":
            params = msg.params or {}
            session_id = params.get("sessionId") if isinstance(params, dict) else None
            queue = self._prompt_streams.get(session_id) if session_id else None
            if queue is None:
                logger.debug(
                    "acp-client got session/update for session %r with no in-flight prompt",
                    session_id,
                )
                return
            try:
                notif = _acp.SessionNotification.model_validate(params)
            except Exception:
                logger.exception("acp-client could not validate session/update params")
                return
            queue.put_nowait(notif)
            return

        # Anything else is an unexpected notification — drop with a debug log.
        logger.debug("acp-client got unknown notification %r", msg.method)

    async def _handle_inbound_request(self, msg: JsonRpcRequest) -> None:
        handler = self._handlers.dispatch(msg.method)
        if handler is None:
            await self._send_response(
                msg.id,
                error=JsonRpcError(
                    code=int(JsonRpcErrorCode.METHOD_NOT_FOUND),
                    message=f"method {msg.method!r} not found or not advertised",
                ),
            )
            return
        try:
            params = msg.params if isinstance(msg.params, dict) else {}
            result = await handler(params)
        except AcpRemoteError as exc:
            await self._send_response(
                msg.id,
                error=JsonRpcError(code=exc.code, message=exc.message, data=exc.data),
            )
        except Exception as exc:
            logger.exception("inbound handler for %s raised", msg.method)
            await self._send_response(
                msg.id,
                error=JsonRpcError(
                    code=int(JsonRpcErrorCode.INTERNAL_ERROR),
                    message=f"handler error: {exc}",
                ),
            )
        else:
            await self._send_response(msg.id, result=result)

    # -- helpers ------------------------------------------------------

    def _has_capability(self, name: str) -> bool:
        caps = self.agent_capabilities
        if caps is None:
            return False
        # Normalize the few capability names that map directly.
        if name == "loadSession":
            return bool(caps.load_session)
        return False

    @staticmethod
    def _dump(model: Any) -> dict[str, Any]:
        """Pydantic ``model_dump(by_alias=True, exclude_none=True)``.

        ACP wires use camelCase per the canonical schema; the SDK
        models store snake_case fields with ``alias`` set to the
        camelCase form. ``by_alias=True`` is the canonical wire
        serialization; ``exclude_none`` keeps the message body small
        and matches what the SDK does internally.
        """

        return model.model_dump(by_alias=True, exclude_none=True)


__all__ = [
    "AcpCapabilityError",
    "AcpClient",
    "AcpClientError",
    "AcpRemoteError",
    "AcpTransportClosed",
    "PromptStreamItem",
]
