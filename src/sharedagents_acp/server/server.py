"""Inbound JSON-RPC dispatcher for agent-side ACP methods.

:class:`AcpServer` reads framed JSON-RPC messages from a
:class:`~sharedagents_acp.protocol.Transport`, routes each request to the
backend handler keyed off its ``method`` string, and writes the response
back through the same transport. Notifications (no ``id``) are
dispatched without writing a response.

Transport-agnostic by construction: stdio, WebSocket, and HTTP-SSE
adapters all reduce to ``read_message() -> bytes`` /
``write_message(bytes) -> None``, so the dispatcher does not need to
know which transport it has — every transport reuses this dispatcher
unchanged.

Validation policy enforced at the wire boundary (NOT in the backend):

* JSON-RPC batch arrays are rejected with ``-32600`` at parse time
  (already enforced inside :func:`parse_message`); the dispatcher just
  surfaces that error to the wire so the client sees the framing-level
  rejection consistently across stdio / WS / HTTP.
* ``session/prompt``: any ContentBlock whose ``type`` is not in the
  sharedagents-supported set (``text``, ``resource_link``, ``image``) is
  rejected with ``-32602``. sharedagents advertises
  ``promptCapabilities.image = true`` (image blocks fan out to the
  OpenAI multi-part ``image_url`` payload via the bridge),
  ``audio = false`` and ``embeddedContext = false``.
* ``session/new`` / ``session/load``: any ``mcpServers[i].type`` of
  ``http`` while ``agentCapabilities.mcpCapabilities.http`` is False is
  rejected with ``-32602`` (same for ``sse``). The dispatcher refuses
  what it advertises it can't accept.

The dispatcher does NOT validate every field of every request body —
that's the SDK's job. We just enforce the capability surface this
agent advertises.

Killability:

The dispatcher itself runs entirely in the agent's event loop, but the
hot operation (``session/prompt``) hands off to
:meth:`AgentBackend.prompt`, which produces values from a backend-owned
``AsyncIterator``. The backend is responsible for running the actual
LLM/tool work in a subprocess so an ESC →
:meth:`AgentBackend.cancel` → SIGTERM kill path is intact. The
dispatcher itself is non-blocking: it just relays.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
from collections.abc import AsyncIterator, Mapping
from typing import Any

from ..protocol import (
    JsonRpcErrorCode,
    JsonRpcFramingError,
    JsonRpcNotification,
    JsonRpcParseError,
    JsonRpcRequest,
    Transport,
    parse_message,
)
from ..protocol.jsonrpc import JsonRpcResponse
from .backend import AgentBackend

logger = logging.getLogger(__name__)


# ContentBlock types this agent accepts on session/prompt. Anything
# else gets rejected with JSON-RPC -32602, mirroring what
# promptCapabilities advertises in the initialize response.
#
# ``image`` is here because ``_blocks_to_user_content`` in the bridge
# translates image blocks into the OpenAI multi-part ``image_url``
# payload that vision-capable upstreams (vLLM / openrouter / kimi)
# consume natively — the wire/advertisement/handler triple stays in
# sync. Audio + embedded resources are still rejected (their handler
# paths aren't wired).
_SUPPORTED_CONTENT_BLOCK_TYPES: frozenset[str] = frozenset({"text", "resource_link", "image"})


# Sentinel used by ``_require_list`` to distinguish "no default given,
# raise on miss" from "default given (possibly an empty list)".
_MISSING: Any = object()


def _serialize_body(body: Mapping[str, Any]) -> bytes:
    """Serialize a JSON-RPC body to compact UTF-8 bytes."""
    return json.dumps(body, separators=(",", ":")).encode("utf-8")


class AcpServer:
    """Agent-side ACP dispatcher.

    Construct with a :class:`Transport`, an :class:`AgentBackend`, and
    static advertisement metadata (capabilities, agent info, auth
    methods). Run :meth:`serve_until_close` to take over the read loop.

    Multiple ``AcpServer`` instances can run side by side over distinct
    transports (e.g. one stdio child + one WS connection in the same
    process), each with its own backend or a shared one.
    """

    def __init__(
        self,
        transport: Transport,
        backend: AgentBackend,
        *,
        agent_capabilities: Any,
        agent_info: Any,
        auth_methods: list[Any] | None = None,
    ) -> None:
        self._transport = transport
        self._backend = backend
        self._agent_capabilities = agent_capabilities
        self._agent_info = agent_info
        self._auth_methods: list[Any] = list(auth_methods) if auth_methods else []
        self._closed = False
        # The frontend identity for this ACP connection,
        # pinned once at ``initialize`` from the handshake (``params["frontend"]``
        # / ``clientInfo``). It is the AUTHORITATIVE, non-spoofable ceiling for
        # every mode-setting request on this connection: a per-request
        # ``frontend`` may only NARROW it, never widen it. Empty == the client
        # did not declare one at init (legacy client) → per-request frontend is
        # used as-is for backward compatibility.
        self._connection_frontend: str = ""
        # Outbound request bookkeeping. A per-id futures map (NOT a
        # single queue) is the right model: multiple outbound requests
        # may be in flight on the same connection and responses can
        # land out of order — keying by id correlates each future to
        # its own response. Mirrors what AcpClient does on the other
        # side of the wire.
        self._next_outbound_id = itertools.count(1)
        self._pending_outbound: dict[int, asyncio.Future[JsonRpcResponse]] = {}
        # Serialize concurrent writes so a notification emitted by
        # ``_handle_session_prompt`` can't interleave the JSON of a
        # request_outbound() body with a session/update body. Reads
        # remain single-threaded (the read loop is the only reader).
        self._write_lock = asyncio.Lock()
        # In-flight ``session/prompt`` handlers run in their own tasks
        # so the read loop can keep pumping inbound responses to
        # outbound ``session/request_permission`` calls the prompt
        # made. Without this split the dispatcher deadlocks: the
        # prompt awaits a JSON-RPC response that only the read loop
        # can deliver, and the read loop is parked on the prompt.
        self._prompt_tasks: set[asyncio.Task[None]] = set()
        # ``session/cancel`` handlers ALSO run in their own tasks. A
        # graceful first-ESC cancel can park up to ``CLI_GRACEFUL_ABORT_S``
        # (~15s) inside ``ExternalCliEngine.abort()``'s SIGINT grace
        # window; inline-awaiting it in the read loop wedged the loop so
        # the SECOND ESC's hard kill (``force=True``) could not be read
        # off the transport until the grace elapsed — the operator pressed
        # ESC twice and nothing died (``sharedagents -a claude``).
        # Tracked here (mirrors ``_prompt_tasks``) so a spawned cancel
        # isn't GC'd mid-flight and connection teardown can drain it.
        self._cancel_tasks: set[asyncio.Task[None]] = set()
        # Session ids this server bound an outbound channel for. The
        # HTTP/WS transports share one backend across many connections
        # (each its own AcpServer); a single ``backend.server_handle``
        # would race between e.g. a TUI and a Telegram client on the
        # same agent. Instead we register THIS server as the outbound
        # channel for every session opened on its connection (see
        # ``_bind_session``), so ``session/request_permission`` /
        # ``session/ask_user`` reach the connection that owns the
        # session. Tracked here so connection teardown unbinds exactly
        # the sessions it bound — no more, no less.
        self._bound_sessions: set[str] = set()

    # ------------------------------------------------------------------
    # Per-session outbound-channel binding.
    # ------------------------------------------------------------------

    def _bind_session(self, session_id: str) -> None:
        """Register this server as the outbound channel for ``session_id``.

        Called after a successful ``session/new`` / ``session/load`` so
        the backend can route ``session/request_permission`` and
        ``session/ask_user`` for that session back to *this*
        connection's wire. Backends that don't support per-session
        binding (CLI-engine wrappers with no inner CoreBackend, stub
        test backends) simply don't expose the hook — the ``getattr``
        guard makes binding a no-op there and the backend falls back to
        its single ``server_handle``.
        """
        if not session_id:
            return
        binder = getattr(self._backend, "bind_session_server", None)
        if callable(binder):
            try:
                binder(session_id, self)
            except Exception:  # noqa: BLE001 - binding is best-effort
                logger.debug("bind_session_server(%s) raised", session_id, exc_info=True)
        self._bound_sessions.add(session_id)

    def _unbind_session(self, session_id: str) -> None:
        """Drop this server's outbound binding for ``session_id``.

        Called on ``session/close`` and (for every still-bound session)
        when the connection tears down — see :meth:`serve_until_close`.
        """
        if not session_id:
            return
        unbinder = getattr(self._backend, "unbind_session_server", None)
        if callable(unbinder):
            try:
                unbinder(session_id)
            except Exception:  # noqa: BLE001 - unbinding is best-effort
                logger.debug("unbind_session_server(%s) raised", session_id, exc_info=True)
        self._bound_sessions.discard(session_id)

    # ------------------------------------------------------------------
    # Main read loop.
    # ------------------------------------------------------------------

    async def serve_until_close(self) -> None:
        """Read messages from the transport until EOF, dispatching each.

        Returns cleanly on EOF (``JsonRpcFramingError("EOF before any
        header bytes")``). Any other framing error sends a parse-error
        response with ``id: null`` per the JSON-RPC spec when an ``id``
        cannot be recovered.
        """
        try:
            while not self._closed:
                try:
                    buf = await self._transport.read_message()
                except JsonRpcFramingError as exc:
                    msg = str(exc)
                    if msg.startswith("EOF before any header bytes"):
                        return
                    # Mid-stream framing error — send a parse-error response
                    # with id=null and return; we have no useful id to bind
                    # to and the wire is now unsynced.
                    await self._send_error(None, JsonRpcErrorCode.PARSE_ERROR, msg)
                    return
                except (ConnectionError, *asyncio_eof_classes()):
                    # Transport-level EOF / connection drop. The
                    # tuple is unpacked because ``except`` requires
                    # a flat tuple of exception classes — nesting a
                    # sub-tuple raises ``TypeError: catching classes
                    # that do not inherit from BaseException``.
                    return

                try:
                    msg = parse_message(buf)
                except JsonRpcParseError as exc:
                    # parse_message rejects batch arrays here with
                    # INVALID_REQUEST (-32600). The id is unrecoverable so
                    # respond with id=null.
                    await self._send_error(None, exc.code, str(exc))
                    continue

                await self._dispatch(msg)
        finally:
            # Connection is going away — fail any in-flight outbound
            # requests so awaiters don't hang. ``request_outbound``
            # awaits a future that this loop is the sole completer of;
            # without this fail-out a Decision.ASK that fired right
            # before EOF would block the prompt iterator forever.
            self._closed = True
            # Drop every per-session outbound binding this server made
            # so the (shared) backend doesn't keep routing permission /
            # ask_user requests to a dead connection. Snapshot first —
            # ``_unbind_session`` mutates ``_bound_sessions``.
            for sid in list(self._bound_sessions):
                self._unbind_session(sid)
            for fut in list(self._pending_outbound.values()):
                if not fut.done():
                    fut.set_exception(ConnectionError("acp server transport closed mid-flight"))
            self._pending_outbound.clear()
            # Wait briefly for prompt tasks to wind down. They each
            # await an outbound response that we just failed, so the
            # bridge's permission-request safety-net path engages and
            # the iterator yields its terminal stopReason. Bound the
            # wait so a buggy backend can't keep the read loop alive
            # forever; on timeout cancel and move on.
            # Drain in-flight cancel handlers too (a graceful cancel can
            # still be parked in its grace window when the connection
            # drops) so they don't outlive the server as orphan tasks.
            if self._prompt_tasks or self._cancel_tasks:
                pending = list(self._prompt_tasks) + list(self._cancel_tasks)
                done, undone = await asyncio.wait(pending, timeout=2.0)
                for t in undone:
                    t.cancel()
                if undone:
                    try:
                        await asyncio.gather(*undone, return_exceptions=True)
                    except Exception:  # noqa: BLE001
                        pass

    # ------------------------------------------------------------------
    # Dispatch.
    # ------------------------------------------------------------------

    async def _dispatch(self, msg: Any) -> None:
        if isinstance(msg, JsonRpcNotification):
            # Only session/cancel is a defined agent-side notification.
            if msg.method == "session/cancel":
                # Run as its own task — NEVER inline-await it here. A
                # graceful cancel parks inside the CLI engine's SIGINT
                # grace window for up to ~15s; awaiting it would wedge
                # this read loop so a FOLLOW-UP cancel (the second ESC's
                # ``force=True`` hard kill) couldn't be read off the
                # transport until the grace elapsed — the user mashes ESC
                # and nothing dies. Spawning keeps the loop pumping so the
                # hard kill lands the instant it arrives (mirrors the
                # ``session/prompt`` split below).
                task = asyncio.create_task(
                    self._handle_session_cancel(msg.params),
                    name="acp-server-cancel",
                )
                self._cancel_tasks.add(task)
                task.add_done_callback(self._cancel_tasks.discard)
            # Ignore unknown notifications (per JSON-RPC spec — no
            # response on notifications, ever).
            return

        if isinstance(msg, JsonRpcRequest):
            method = msg.method
            req_id = msg.id
            params = msg.params or {}

            # ``session/prompt`` is the only long-running request method:
            # it streams session/update notifications and may issue
            # outbound ``session/request_permission`` requests that
            # need responses to land on the SAME read loop. Run the
            # prompt handler as its own task so the read loop can keep
            # pumping inbound messages while the prompt awaits its
            # outbound responses. All other methods stay synchronous
            # — they're short and serializing them keeps ordering
            # predictable.
            if method == "session/prompt":
                task = asyncio.create_task(
                    self._run_prompt_request(msg.params or {}, req_id),
                    name=f"acp-server-prompt-{req_id}",
                )
                self._prompt_tasks.add(task)
                task.add_done_callback(self._prompt_tasks.discard)
                return

            try:
                if method == "initialize":
                    result = await self._handle_initialize(params)
                elif method == "authenticate":
                    result = await self._handle_authenticate(params)
                elif method == "session/new":
                    result = await self._handle_session_new(params)
                elif method == "session/load":
                    result = await self._handle_session_load(params)
                elif method == "session/list":
                    result = await self._handle_session_list(params)
                elif method == "session/close":
                    result = await self._handle_session_close(params)
                elif method == "session/set_mode":
                    result = await self._handle_session_set_mode(params)
                elif method == "session/set_config_option":
                    # Standard ACP method — no vendor alias.
                    result = await self._handle_session_set_config_option(params)
                elif method in (
                    "_sharedagents/inject_user_message",
                    "session/inject_user_message",
                ):
                    # sharedagents extension (no ACP equivalent). Canonical
                    # name is the vendor-prefixed ``_sharedagents/*``; the
                    # ``session/*`` form is a deprecated back-compat alias
                    # kept for sharedagents's own TUI/telegram clients until
                    # they migrate.
                    result = await self._handle_session_inject_user_message(params)
                elif method in ("_sharedagents/command", "session/command"):
                    # sharedagents extension; same vendor-prefix policy as
                    # inject_user_message above.
                    result = await self._handle_session_command(params)
                elif method in ("_sharedagents/session/queue/list", "session/queue/list"):
                    result = await self._handle_session_queue_list(params)
                elif method in ("_sharedagents/session/queue/add", "session/queue/add"):
                    result = await self._handle_session_queue_add(params)
                elif method in ("_sharedagents/session/queue/update", "session/queue/update"):
                    result = await self._handle_session_queue_update(params)
                elif method in ("_sharedagents/session/queue/remove", "session/queue/remove"):
                    result = await self._handle_session_queue_remove(params)
                else:
                    await self._send_error(
                        req_id,
                        JsonRpcErrorCode.METHOD_NOT_FOUND,
                        f"method not found: {method!r}",
                    )
                    return
            except _DispatchError as exc:
                await self._send_error(req_id, exc.code, exc.message, exc.data)
                return
            except Exception as exc:  # noqa: BLE001 - wire boundary
                await self._send_error(
                    req_id,
                    JsonRpcErrorCode.INTERNAL_ERROR,
                    f"backend error: {exc.__class__.__name__}: {exc}",
                )
                return

            await self._send_result(req_id, result)
            return

        # Response to an outbound request we issued (e.g. an outbound
        # ``session/request_permission``). Correlate by id and complete
        # the awaiting future. Responses we don't recognize are
        # silently dropped — they'd be out-of-band garbage.
        if isinstance(msg, JsonRpcResponse):
            rid = msg.id
            fut = self._pending_outbound.get(rid) if isinstance(rid, int) else None
            if fut is None or fut.done():
                return
            fut.set_result(msg)
            return

        return

    # ------------------------------------------------------------------
    # Per-method handlers. Each returns a JSON-shaped dict that the
    # dispatcher serializes as the JSON-RPC ``result``.
    # ------------------------------------------------------------------

    async def _handle_initialize(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        # Notify the backend so it can record peer version / caps for
        # its own use; the response we send is built from our static
        # config (locked-decision capabilities, agent_info, auth).
        protocol_version = params.get("protocolVersion") if isinstance(params, Mapping) else None
        try:
            pv_int = int(protocol_version) if protocol_version is not None else 1
        except (TypeError, ValueError):
            pv_int = 1
        client_caps = params.get("clientCapabilities") if isinstance(params, Mapping) else None
        client_info = params.get("clientInfo") if isinstance(params, Mapping) else None
        # Pin this connection's frontend identity from the
        # handshake. Prefer an explicit top-level ``frontend`` param; fall back
        # to ``clientInfo.frontend``. This becomes the non-spoofable ceiling for
        # mode-setting — a later per-request ``frontend`` may only narrow it.
        _fe = params.get("frontend") if isinstance(params, Mapping) else None
        if not (isinstance(_fe, str) and _fe) and isinstance(client_info, Mapping):
            _fe = client_info.get("frontend")
        if isinstance(_fe, str) and _fe:
            self._connection_frontend = _fe
        try:
            await self._backend.initialize(pv_int, client_caps, client_info)
        except Exception:
            # Backend's hook is informational; failure does not fail
            # initialize. The wire response is determined by us.
            pass
        return {
            "agentCapabilities": _to_jsonable(self._agent_capabilities),
            "agentInfo": _to_jsonable(self._agent_info),
            "authMethods": [_to_jsonable(m) for m in self._auth_methods],
            "protocolVersion": pv_int,
        }

    async def _handle_authenticate(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        # Default stance: no auth methods advertised. Refuse with
        # METHOD_NOT_FOUND — subclasses that DO advertise auth methods
        # override this method.
        if not self._auth_methods:
            raise _DispatchError(
                JsonRpcErrorCode.METHOD_NOT_FOUND,
                "authenticate not supported (no authMethods advertised)",
            )
        method_id = (params or {}).get("methodId") if isinstance(params, Mapping) else None
        if not isinstance(method_id, str):
            raise _DispatchError(
                JsonRpcErrorCode.INVALID_PARAMS,
                "authenticate requires string methodId",
            )
        await self._backend.authenticate(method_id)
        return {}

    async def _handle_session_new(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        cwd = self._require_str(params, "cwd")
        mcp_servers = self._require_list(params, "mcpServers", default=[])
        self._validate_mcp_servers(mcp_servers)
        result = await self._backend.new_session(cwd, list(mcp_servers))
        # Route this session's outbound permission / ask_user requests
        # back to this connection's wire (see ``_bind_session``).
        sid = result.get("sessionId") if isinstance(result, Mapping) else None
        if isinstance(sid, str):
            self._bind_session(sid)
        return _to_jsonable(result)

    async def _handle_session_load(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        session_id = self._require_str(params, "sessionId")
        cwd = self._require_str(params, "cwd")
        mcp_servers = self._require_list(params, "mcpServers", default=[])
        self._validate_mcp_servers(mcp_servers)
        result = await self._backend.load_session(session_id, cwd, list(mcp_servers))
        # A loaded session is owned by this connection just like a new
        # one — bind its outbound channel here too.
        self._bind_session(session_id)
        return _to_jsonable(result)

    async def _handle_session_list(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        cursor = params.get("cursor") if isinstance(params, Mapping) else None
        cwd = params.get("cwd") if isinstance(params, Mapping) else None
        if cursor is not None and not isinstance(cursor, str):
            raise _DispatchError(JsonRpcErrorCode.INVALID_PARAMS, "cursor must be string or null")
        if cwd is not None and not isinstance(cwd, str):
            raise _DispatchError(JsonRpcErrorCode.INVALID_PARAMS, "cwd must be string or null")
        result = await self._backend.list_sessions(cursor, cwd)
        return _to_jsonable(result)

    async def _handle_session_close(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        session_id = self._require_str(params, "sessionId")
        await self._backend.close_session(session_id)
        self._unbind_session(session_id)
        return {}

    def _effective_frontend(self, requested: Any) -> str:
        """Narrow a per-request ``frontend`` against this connection's pinned
        identity. The connection frontend is the ceiling; a
        request may only select a frontend no more capable than it. When the
        client declared no frontend at ``initialize`` (legacy), fall back to the
        per-request value (default ``"acp"``, a non-ask-capable, fail-closed
        origin).
        """
        req = requested if isinstance(requested, str) and requested else ""
        if self._connection_frontend:
            from sharedagents_command.command_runtime import resolve_effective_frontend

            return resolve_effective_frontend(
                self._connection_frontend,
                req or None,
            )
        return req or "acp"

    async def _handle_session_set_mode(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        session_id = self._require_str(params, "sessionId")
        mode_id = self._require_str(params, "modeId")
        # Route the raw ACP mode-set through the SAME frontend-aware capability
        # choke-point as ``session/command`` and the REST ``permission_mode``
        # field, so this wire surface can no longer bypass the ask-mode boundary.
        # The caller declares its
        # frontend is the identity pinned at ``initialize`` (the non-spoofable
        # ceiling); a per-request ``params["frontend"]`` may only NARROW it,
        # never widen it (``_effective_frontend``). A connection that declared no
        # frontend at init falls back to the per-request value, defaulting to a
        # non-ask-capable ``"acp"`` that fails closed on ``ask``.
        _req_fe = params.get("frontend") if isinstance(params, Mapping) else None
        frontend = self._effective_frontend(_req_fe)
        from sharedagents_command.command_runtime import enforce_mode_capability

        refusal = enforce_mode_capability(frontend, mode_id)
        if refusal is not None:
            raise ValueError(refusal)
        await self._backend.set_mode(session_id, mode_id)
        return {}

    async def _handle_session_set_config_option(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        session_id = self._require_str(params, "sessionId")
        config_id = self._require_str(params, "configId")
        if "value" not in (params or {}):
            raise _DispatchError(
                JsonRpcErrorCode.INVALID_PARAMS,
                "session/set_config_option requires 'value' param",
            )
        value = params["value"]
        await self._backend.set_config_option(session_id, config_id, value)
        return {}

    async def _handle_session_inject_user_message(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        """``session/inject_user_message`` — mid-turn steering prompt.

        Forwards to the backend's ``inject_user_message`` which pushes
        the text into the engine's per-session inject buffer. The
        engine drains the buffer at the next round boundary so the
        in-flight plan can pivot on the new instruction instead of
        waiting for the turn to complete.
        """
        session_id = self._require_str(params, "sessionId")
        text = self._require_str(params, "text")
        inj = getattr(self._backend, "inject_user_message", None)
        if inj is None:
            raise _DispatchError(
                JsonRpcErrorCode.METHOD_NOT_FOUND,
                "backend does not support session/inject_user_message",
            )
        return await inj(session_id, text)

    async def _handle_session_queue_list(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        """``session/queue/list`` — return pending follow-up items for a session."""
        sid = self._require_str(params, "sessionId")
        fn = getattr(self._backend, "queue_list", None)
        if fn is None:
            raise _DispatchError(JsonRpcErrorCode.METHOD_NOT_FOUND, "backend has no queue_list")
        return await fn(sid)

    async def _handle_session_queue_add(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        """``session/queue/add`` — append a follow-up item to a session's queue."""
        sid = self._require_str(params, "sessionId")
        text = self._require_str(params, "text")
        source = params.get("source", "") if isinstance(params, Mapping) else ""
        author = params.get("author", "") if isinstance(params, Mapping) else ""
        fn = getattr(self._backend, "queue_add", None)
        if fn is None:
            raise _DispatchError(JsonRpcErrorCode.METHOD_NOT_FOUND, "backend has no queue_add")
        return await fn(sid, text, source=source, author=author)

    async def _handle_session_queue_update(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        """``session/queue/update`` — update the text of a queued follow-up item."""
        sid = self._require_str(params, "sessionId")
        qid = self._require_str(params, "queueId")
        text = self._require_str(params, "text")
        fn = getattr(self._backend, "queue_update", None)
        if fn is None:
            raise _DispatchError(JsonRpcErrorCode.METHOD_NOT_FOUND, "backend has no queue_update")
        return await fn(sid, qid, text)

    async def _handle_session_queue_remove(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        """``session/queue/remove`` — remove a queued follow-up item."""
        sid = self._require_str(params, "sessionId")
        qid = self._require_str(params, "queueId")
        fn = getattr(self._backend, "queue_remove", None)
        if fn is None:
            raise _DispatchError(JsonRpcErrorCode.METHOD_NOT_FOUND, "backend has no queue_remove")
        raw_reason = params.get("reason")
        reason = str(raw_reason).strip() if raw_reason else ""
        from sharedagents_session.queue_remove import invoke_queue_remove

        return await invoke_queue_remove(fn, sid, qid, reason=reason or None)

    async def _handle_session_command(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        """``session/command`` — run a slash command via the shared runtime.

        The single command implementation lives server-side
        (:mod:`sharedagents_command.command_runtime`); this is one of its transport
        shells (the other is the gateway REST control endpoint). Returns the
        ``CommandResponse`` dict verbatim so every frontend renders the same
        output and carries zero command logic.

        ``sessionId`` is OPTIONAL: session-less commands (``/skills``,
        ``/models``, ``/mcp`` …) run without one; the runtime
        falls back to disk/read-only when no live session is bound.
        """
        text = self._require_str(params, "text")
        session_id = ""
        confirmed = False
        if isinstance(params, Mapping):
            sid = params.get("sessionId")
            session_id = sid if isinstance(sid, str) else ""
            frontend = self._effective_frontend(params.get("frontend"))
            # Same coercion the gateway control endpoint applies —
            # truthy confirmation lets an irreversible command execute;
            # absent/false re-challenges. (See control_endpoint.py.)
            _cf = params.get("confirmed")
            confirmed = (_cf is True) or (str(_cf).strip().lower() in ("1", "true", "yes"))
        else:
            frontend = self._effective_frontend(None)
        run = getattr(self._backend, "run_command", None)
        if run is None:
            raise _DispatchError(
                JsonRpcErrorCode.METHOD_NOT_FOUND,
                "backend does not support session/command",
            )
        return await run(session_id, text, frontend=frontend, confirmed=confirmed)

    async def _run_prompt_request(self, params: Mapping[str, Any], req_id: str | int) -> None:
        """Task body for an in-flight ``session/prompt`` request.

        Mirrors the synchronous-method error handling in :meth:`_dispatch`
        but runs concurrently with the read loop. Critical for the
        outbound-request path: the read loop has to keep pumping
        inbound messages so responses to ``session/request_permission``
        can land while the prompt is still iterating.
        """
        try:
            result = await self._handle_session_prompt(params, req_id)
        except _DispatchError as exc:
            await self._send_error(req_id, exc.code, exc.message, exc.data)
            return
        except asyncio.CancelledError:
            # Read loop tore us down on close. Don't try to write a
            # response — the transport is gone.
            raise
        except Exception as exc:  # noqa: BLE001 - wire boundary
            await self._send_error(
                req_id,
                JsonRpcErrorCode.INTERNAL_ERROR,
                f"backend error: {exc.__class__.__name__}: {exc}",
            )
            return
        await self._send_result(req_id, result)

    async def _handle_session_prompt(self, params: Mapping[str, Any], req_id: str | int) -> Mapping[str, Any]:
        session_id = self._require_str(params, "sessionId")
        message_id = params.get("messageId") if isinstance(params, Mapping) else None
        if message_id is not None and not isinstance(message_id, str):
            raise _DispatchError(JsonRpcErrorCode.INVALID_PARAMS, "messageId must be string or null")
        prompt_blocks = self._require_list(params, "prompt")

        # Validate each ContentBlock's type at the wire boundary —
        # see _SUPPORTED_CONTENT_BLOCK_TYPES (text + resource_link +
        # image today; audio + embedded_resource still rejected).
        for i, blk in enumerate(prompt_blocks):
            if not isinstance(blk, Mapping):
                raise _DispatchError(
                    JsonRpcErrorCode.INVALID_PARAMS,
                    f"prompt[{i}] must be an object",
                )
            blk_type = blk.get("type")
            if blk_type not in _SUPPORTED_CONTENT_BLOCK_TYPES:
                raise _DispatchError(
                    JsonRpcErrorCode.INVALID_PARAMS,
                    (
                        f"prompt[{i}].type={blk_type!r} not supported by this agent. "
                        f"Accepted: {sorted(_SUPPORTED_CONTENT_BLOCK_TYPES)}. "
                        "(Per advertised promptCapabilities; audio/resource "
                        "are rejected at the dispatcher.)"
                    ),
                    data={"index": i, "type": blk_type},
                )

        # Run the prompt iterator. Intermediate values are session/update
        # notifications; terminal value is the PromptResponse.
        #
        # We hold the iterator and ``aclose()`` it in ``finally`` rather
        # than just ``break``-ing: the terminal ``stopReason`` is the
        # generator's last yielded value, so a bare break abandons the
        # generator suspended at its ``yield`` and its ``finally`` (which
        # releases the backend's per-session prompt-queue slot and the
        # turn-root execution lease) would only run at GC time. Explicit
        # ``aclose()`` makes that teardown deterministic at request end.
        prompt_result: Mapping[str, Any] | None = None
        _it = _ensure_async_iter(self._backend.prompt(session_id, message_id, list(prompt_blocks)))
        try:
            async for upd in _it:
                if not isinstance(upd, Mapping):
                    raise _DispatchError(
                        JsonRpcErrorCode.INTERNAL_ERROR,
                        "backend.prompt yielded non-mapping value",
                    )
                if "stopReason" in upd:
                    # Terminal sentinel.
                    prompt_result = upd
                    break
                # Session update notification.
                await self._send_session_update(session_id, upd)
        finally:
            _aclose = getattr(_it, "aclose", None)
            if _aclose is not None:
                try:
                    await _aclose()
                except Exception:  # noqa: BLE001
                    pass

        if prompt_result is None:
            raise _DispatchError(
                JsonRpcErrorCode.INTERNAL_ERROR,
                "backend.prompt iterator finished without yielding a PromptResponse",
            )

        return _to_jsonable(prompt_result)

    async def _handle_session_cancel(self, params: Any) -> None:
        if not isinstance(params, Mapping):
            return  # silently ignore malformed cancels
        session_id = params.get("sessionId")
        if not isinstance(session_id, str):
            return
        # ``force`` is an sharedagents extension to the ACP cancel
        # notification: False (default) = graceful stop (let the
        # in-flight op finish, then end the turn); True = hard kill
        # (SIGKILL + killpg the worker + every runner / subagent now).
        # The TUI's first ESC / ``/stop`` sends graceful; the second
        # ESC / ``/abort`` sends force=True.
        force = bool(params.get("force"))
        try:
            await self._backend.cancel(session_id, force=force)
        except TypeError:
            # Backend predates the ``force`` kwarg — degrade to a plain
            # cancel so ESC still works against older agents.
            try:
                await self._backend.cancel(session_id)
            except Exception:
                return
        except Exception:
            # session/cancel is a notification — no error path back.
            return

    # ------------------------------------------------------------------
    # Helpers.
    # ------------------------------------------------------------------

    def _validate_mcp_servers(self, servers: list[Any]) -> None:
        """Reject http/sse MCP entries while the corresponding capability
        is False. Stdio entries (no ``type`` field) are always accepted —
        sharedagents supports stdio MCP as a baseline.
        """
        caps = _to_jsonable(self._agent_capabilities) or {}
        mcp_caps = caps.get("mcpCapabilities") if isinstance(caps, Mapping) else None
        http_ok = bool(mcp_caps.get("http")) if isinstance(mcp_caps, Mapping) else False
        sse_ok = bool(mcp_caps.get("sse")) if isinstance(mcp_caps, Mapping) else False
        for i, srv in enumerate(servers):
            if not isinstance(srv, Mapping):
                raise _DispatchError(
                    JsonRpcErrorCode.INVALID_PARAMS,
                    f"mcpServers[{i}] must be an object",
                )
            # Normalize case BEFORE the gate. ``MCPServerConfig.from_acp``
            # lowercases ``type``, so a case-sensitive check here (e.g.
            # ``t == "http"``) would let ``"Http"``/``"SSE"`` slip past the
            # capability gate and then get connected — failing OPEN on a
            # transport we advertised as unsupported. Match on the
            # normalized form so the gate fails closed.
            t = str(srv.get("type") or "").strip().lower()
            if t == "http" and not http_ok:
                raise _DispatchError(
                    JsonRpcErrorCode.INVALID_PARAMS,
                    (f"mcpServers[{i}].type=http rejected: agent advertises mcpCapabilities.http=false"),
                    data={"index": i, "type": "http"},
                )
            if t == "sse" and not sse_ok:
                raise _DispatchError(
                    JsonRpcErrorCode.INVALID_PARAMS,
                    (f"mcpServers[{i}].type=sse rejected: agent advertises mcpCapabilities.sse=false"),
                    data={"index": i, "type": "sse"},
                )

    @staticmethod
    def _require_str(params: Any, key: str) -> str:
        if not isinstance(params, Mapping) or key not in params:
            raise _DispatchError(JsonRpcErrorCode.INVALID_PARAMS, f"missing required field: {key!r}")
        v = params[key]
        if not isinstance(v, str):
            raise _DispatchError(JsonRpcErrorCode.INVALID_PARAMS, f"{key} must be a string")
        return v

    @staticmethod
    def _require_list(params: Any, key: str, *, default: Any = _MISSING) -> list[Any]:
        if not isinstance(params, Mapping) or key not in params:
            if default is _MISSING:
                raise _DispatchError(
                    JsonRpcErrorCode.INVALID_PARAMS,
                    f"missing required field: {key!r}",
                )
            return list(default)
        v = params[key]
        if not isinstance(v, list):
            raise _DispatchError(JsonRpcErrorCode.INVALID_PARAMS, f"{key} must be an array")
        return v

    # ------------------------------------------------------------------
    # Wire writers.
    # ------------------------------------------------------------------

    async def _send_result(self, req_id: str | int, result: Mapping[str, Any]) -> None:
        body = {"jsonrpc": "2.0", "id": req_id, "result": result}
        await self._write(_serialize_body(body))

    async def _send_error(
        self,
        req_id: str | int | None,
        code: int,
        message: str,
        data: Any = None,
    ) -> None:
        err: dict[str, Any] = {"code": int(code), "message": message}
        if data is not None:
            err["data"] = data
        body = {"jsonrpc": "2.0", "id": req_id, "error": err}
        await self._write(_serialize_body(body))

    async def _send_session_update(self, session_id: str, update: Mapping[str, Any]) -> None:
        body = {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {"sessionId": session_id, "update": _to_jsonable(update)},
        }
        await self._write(_serialize_body(body))

    async def notify_session_update(
        self,
        *,
        session_id: str,
        update: Mapping[str, Any],
    ) -> None:
        """Public wrapper around :meth:`_send_session_update`.

        Backends (e.g. the bridge's ``set_mode``) call this to push a
        ``current_mode_update`` / ``available_commands_update`` /
        ``config_option_update`` notification outside the normal prompt
        stream. Keyword-only so the call site reads as a typed
        intent rather than positional plumbing.
        """
        await self._send_session_update(session_id, update)

    async def _write(self, payload: bytes) -> None:
        """Serialize transport writes.

        Holding ``_write_lock`` across each write keeps a session-update
        notification from interleaving with the JSON of an outbound
        ``session/request_permission`` request frame on the wire.
        """
        async with self._write_lock:
            await self._transport.write_message(payload)

    # ------------------------------------------------------------------
    # Outbound requests (agent → client).
    # ------------------------------------------------------------------

    async def request_outbound(self, method: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        """Send a JSON-RPC request from agent → client and await result.

        Mints a fresh integer request id, parks an :class:`asyncio.Future`
        on ``self._pending_outbound`` keyed by that id, writes the
        framed request through the transport, and awaits the future.
        The read loop completes the future when the matching response
        lands (correlate by ``id``).

        Per-id futures (NOT a single queue) is the right model: a
        prompt may issue multiple permission requests during the same
        turn, and tool-runner orchestration may interleave them with
        other outbound calls in the future. Keying by id keeps the
        correlation precise without ordering assumptions.

        Errors:

        * The connection closes before a response arrives → raises
          ``ConnectionError`` (the read loop's finally block fails all
          pending futures).
        * The peer returns a JSON-RPC error envelope → returns the
          result anyway via raise of ``RuntimeError`` carrying the
          error code/message; callers that need fine-grained control
          can inspect via :attr:`JsonRpcResponse.error`.
        """
        if self._closed:
            raise ConnectionError("acp server transport already closed")
        loop = asyncio.get_running_loop()
        req_id = next(self._next_outbound_id)
        fut: asyncio.Future[JsonRpcResponse] = loop.create_future()
        self._pending_outbound[req_id] = fut
        body: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": _to_jsonable(params) if params is not None else None,
        }
        if body["params"] is None:
            body.pop("params")
        try:
            await self._write(_serialize_body(body))
            response = await fut
        finally:
            self._pending_outbound.pop(req_id, None)
        if response.error is not None:
            raise RuntimeError(f"outbound {method!r} returned error {response.error.code}: {response.error.message}")
        return response.result if isinstance(response.result, Mapping) else {}


# ---------------------------------------------------------------------------
# Internal helpers.
# ---------------------------------------------------------------------------


class _DispatchError(Exception):
    """Internal control-flow exception that carries a JSON-RPC error
    code/message back to the dispatcher's response writer."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = int(code)
        self.message = message
        self.data = data


def _to_jsonable(obj: Any) -> Any:
    """Convert a Pydantic model or dict to a JSON-friendly shape.

    Uses ``model_dump(by_alias=True, exclude_none=True)`` to keep the
    wire form aligned with the schema's camelCase field names. Plain
    dicts and lists are passed through.
    """
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(by_alias=True, exclude_none=True)
        except Exception:
            pass
    if isinstance(obj, Mapping):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


async def _ensure_async_iter(
    val: AsyncIterator[Mapping[str, Any]] | Any,
) -> AsyncIterator[Mapping[str, Any]]:
    """Yield from ``val`` whether it's an AsyncIterator or an awaitable
    that returns one. Backends that mark ``prompt`` ``async def`` and
    ``yield`` produce native async generators; either pattern works."""
    if hasattr(val, "__aiter__"):
        async for v in val:
            yield v
        return
    # Maybe it's a coroutine returning an async iterator.
    awaited = await val  # type: ignore[misc]
    async for v in awaited:
        yield v


def asyncio_eof_classes() -> tuple[type[BaseException], ...]:
    """Return the exception classes that mean ``the underlying transport
    closed mid-read``. Defined as a function so the import stays lazy."""
    import asyncio

    return (asyncio.IncompleteReadError, EOFError)


__all__ = ["AcpServer"]
