"""Remote :class:`AgentBackend` — the gateway's south-bound ACP client.

Implements the :class:`sharedagents_acp.server.backend.AgentBackend` Protocol
by talking to a SEPARATE orchestrator process over the existing
ACP-over-HTTP transport (``connect_http`` / :class:`AcpClient`), rather
than driving an in-process :class:`AgentACPBackend`. This is what lets
the orchestrator run as its own crash-isolated service: the gateway
(north) keeps only the read surface + a thin command membrane, and every
session method it would have called on the in-process backend is now a
JSON-RPC call to the orchestrator (south) instead.

Design
------
The orchestrator service split:

      INTERFACES → sharedagents-gateway :8100 (NORTH, restartable)
          • serves READS locally, FS-direct
          • PROXIES commands south over ACP-HTTP
              │ ACP-over-HTTP 127.0.0.1:8099 (commands only)
              ▼
      sharedagents-orchestrator :8099 (SOUTH, write-master, singleton)
          • AgentRegistry + AgentACPBackend per agent
          • the turn RUNS HERE

  Project 2 names this module: *"``RemoteBackend`` (new) — implements the
  ``AgentBackend`` Protocol by calling the orchestrator over ACP-HTTP
  (``connect_http`` / ``AcpClient``); the gateway's ``AgentRegistry.get()``
  returns these."* The orchestrator owns the authoritative
  ``active_turn_id`` (the ``_SUBMIT_INFLIGHT`` single-flight guard moves
  south); the gateway reads it back across the wire.
* ``specs/sharedagents.md`` §13 — the ACP membrane / backend Protocol contract
  this mirrors.

Transport
---------
Each :class:`RemoteBackend` lazily opens (and reuses) one
:class:`AcpClient` connected to ``{orchestrator_url}/acp/{agent}`` with a
bearer token — exactly the connection shape ``sharedagents_tui.agent_client``
uses to reach the gateway's per-agent ACP surface. The Protocol methods
map one-for-one onto the client's ``session/*`` calls; ``prompt`` is the
single streaming method and passes the orchestrator's ``session/update``
dicts (terminated by a ``PromptResponse``) through unchanged.

Three methods the in-process backend exposes have NO dedicated ACP wire
method (``active_turn_id``, ``get_session``, ``set_session_title``); see
their docstrings for how they use ``session/list`` and the control API.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
from collections.abc import AsyncIterator, Mapping
from typing import Any
from urllib.parse import urlparse

from sharedagents_acp.client.client import (
    AcpClient,
    AcpClientError,
    AcpTransportClosed,
)

logger = logging.getLogger(__name__)


def _is_offhost_ip(host: str) -> bool:
    """True iff ``host`` is a routable, non-loopback IP LITERAL.

        The gateway->orchestrator channel assumes same-host loopback transport
    . A bare hostname (``localhost``, a container service
        name) resolves at connect time and is treated as same-trust-domain by
        that assumption; an explicit non-loopback IP literal (e.g. ``192.0.2.10``)
        is the unambiguous off-host case the requirement targets. We do not do DNS
        here — construction stays side-effect-free.
    """
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False  # not an IP literal → hostname → same-host assumption
    return not ip.is_loopback


def _require_bearer_for_offhost(url: str, token: str | None) -> None:
    """Enforce the fail-closed token rule.

    The channel assumes same-host loopback transport. If the orchestrator
    URL points off-host (a routable non-loopback IP literal), the two
    processes are not the same trust domain, so a non-empty bearer token is
    mandatory — an unauthenticated off-host command channel would fail OPEN.
    Loopback / hostname keeps the token optional, matching the same-host
    assumption stated throughout.
    """
    host = (urlparse(url or "").hostname or "").lower()
    if _is_offhost_ip(host) and not (token or "").strip():
        raise ValueError(
            f"gateway->orchestrator channel to off-host {host!r} requires a "
            "bearer token: an unauthenticated off-host command channel would "
            "fail open. Configure orchestrator.auth_token."
        )


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class OrchestratorUnreachable(RuntimeError):
    """The orchestrator could not be reached after bounded retries.

    The gateway turns this into a ``503`` for the caller — a clean
    "south is down, retry" signal rather than a leaked transport
    traceback.
    """


# ---------------------------------------------------------------------------
# RemoteBackend
# ---------------------------------------------------------------------------

# Bounded reconnect schedule (seconds). Short and capped — the
# orchestrator is same-host (127.0.0.1) so a real outage surfaces fast;
# we only paper over a restart window, never an extended downtime.
_RECONNECT_BACKOFF_S: tuple[float, ...] = (0.05, 0.2, 0.5)


class RemoteBackend:
    """An :class:`AgentBackend` that proxies to a remote orchestrator.

    Constructed per agent. Holds (lazily) one :class:`AcpClient` to
    ``{orchestrator_url}/acp/{agent}``; every Protocol method issues the
    corresponding ``session/*`` (or ``initialize`` / ``authenticate``)
    JSON-RPC call on that client and returns the same JSON-shaped dict
    the in-process :class:`AgentACPBackend` returns.

    Safe to call concurrently across distinct ``session_id`` values. A
    dropped connection is transparently re-established with a small
    bounded backoff; an orchestrator that stays unreachable raises
    :class:`OrchestratorUnreachable`.
    """

    def __init__(
        self,
        orchestrator_url: str,
        agent: str,
        *,
        token: str | None = None,
    ) -> None:
        """``token`` authenticates to the orchestrator's ACP-HTTP mount.
        It stays ``Optional`` for the same-host loopback case (single trust
        domain), but is MANDATORY whenever ``orchestrator_url`` points
        off-host: an off-host channel with no token fails open, so the
        constructor raises ``ValueError`` in that case.
        """
        _require_bearer_for_offhost(orchestrator_url, token)
        self._base_url = (orchestrator_url or "").rstrip("/")
        self._agent = agent
        self._token = token
        # The per-agent ACP route on the orchestrator. ``connect_http``
        # appends ``/acp`` internally, matching the gateway mount shape
        # (sub-app under ``/acp/<agent>`` whose JSON-RPC route is
        # ``/acp``) — see sharedagents_tui.agent_client._ensure_connected.
        self._agent_url = f"{self._base_url}/acp/{agent}"
        self._acp: AcpClient | None = None
        self._connect_lock = asyncio.Lock()

    # -- connection management --------------------------------------------

    async def _connect_once(self) -> AcpClient:
        """Open a single fresh :class:`AcpClient` (no retry).

        Passes ``delete_on_close=False`` to :func:`connect_http` — the
        single one-line mechanism the entire north/south crash-isolation
        contract depends on. It means closing this connection (e.g. on a
        gateway restart) never cascades a ``close_session`` (hard-cancel)
        onto the orchestrator: the connection just EOFs, and the
        orchestrator keeps the session and its in-flight turn alive. Do
        not drop or "simplify away" that flag.
        """
        from sharedagents_acp.client.http_spawn import connect_http

        # delete_on_close=False: this is the gateway→orchestrator membrane.
        # A gateway restart must NOT cascade close_session (hard-cancel) onto
        # turns running in the orchestrator — the connection just EOFs, the
        # orchestrator keeps the session and its in-flight turn alive.
        return await connect_http(self._agent_url, bearer_token=self._token, delete_on_close=False)

    async def _ensure_connected(self) -> AcpClient:
        """Return a live :class:`AcpClient`, (re)connecting if needed.

        Reuses the existing client unless it has closed; on a closed or
        absent client, reconnects under the lock with a bounded backoff.
        Raises :class:`OrchestratorUnreachable` if every attempt fails.
        """
        acp = self._acp
        if acp is not None and not getattr(acp, "_closed", False):
            return acp
        async with self._connect_lock:
            # Re-check under the lock — a concurrent caller may have
            # already rebuilt the client while we waited.
            acp = self._acp
            if acp is not None and not getattr(acp, "_closed", False):
                return acp
            last_exc: BaseException | None = None
            for i in range(len(_RECONNECT_BACKOFF_S) + 1):
                try:
                    acp = await self._connect_once()
                    self._acp = acp
                    return acp
                except (ConnectionError, OSError, AcpClientError) as exc:
                    last_exc = exc
                    if i < len(_RECONNECT_BACKOFF_S):
                        await asyncio.sleep(_RECONNECT_BACKOFF_S[i])
                except Exception as exc:  # noqa: BLE001 - httpx & friends
                    last_exc = exc
                    if i < len(_RECONNECT_BACKOFF_S):
                        await asyncio.sleep(_RECONNECT_BACKOFF_S[i])
            raise OrchestratorUnreachable(f"orchestrator unreachable at {self._agent_url!r}: {last_exc}") from last_exc

    async def _reset_client(self) -> None:
        """Tear down the current client so the next call reconnects."""
        acp = self._acp
        self._acp = None
        if acp is not None:
            try:
                await acp.close()
            except Exception:  # noqa: BLE001
                logger.debug("remote backend client close raised", exc_info=True)

    async def _call_with_reconnect(self, method_name: str, *args: Any, **kw: Any) -> Any:
        """Invoke ``AcpClient.<method_name>`` once, retrying on a dropped
        connection.

        A first attempt runs on the live client; if it fails because the
        transport had closed (``AcpTransportClosed``), we drop the dead
        client, reconnect, and try exactly once more. Any remote error
        (``AcpRemoteError``) is a real agent-side failure and propagates
        unchanged — we do NOT mask it as a transport problem.
        """
        acp = await self._ensure_connected()
        try:
            return await getattr(acp, method_name)(*args, **kw)
        except AcpTransportClosed:
            await self._reset_client()
            acp = await self._ensure_connected()
            return await getattr(acp, method_name)(*args, **kw)

    async def _raw_call_with_reconnect(self, method: str, params: Any) -> Any:
        """Issue a raw JSON-RPC ``_call`` with the same reconnect policy.

        Used where we need the byte-faithful response dict the
        orchestrator returned (``session/new`` / ``session/load`` /
        ``session/list``), rather than the SDK-validated model the
        high-level ``AcpClient`` helper would hand back. The in-process
        backend returns these raw dicts, so passing them through keeps
        the two backends wire-identical.
        """
        acp = await self._ensure_connected()
        try:
            return await acp._call(method, params)  # noqa: SLF001
        except AcpTransportClosed:
            await self._reset_client()
            acp = await self._ensure_connected()
            return await acp._call(method, params)  # noqa: SLF001

    async def close(self) -> None:
        """Close the underlying client (idempotent)."""
        await self._reset_client()

    # -- handshake --------------------------------------------------------

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: Mapping[str, Any] | None,
        client_info: Mapping[str, Any] | None,
    ) -> Mapping[str, Any] | None:
        """``connect_http`` already negotiated ``initialize`` when the
        client opened, so this only ensures the connection exists. The
        dispatcher ignores the return value (it builds the
        InitializeResponse from its own static config), so ``None`` is
        the right answer — same as the in-process CLI path."""
        await self._ensure_connected()
        return None

    async def authenticate(self, method_id: str) -> None:
        await self._call_with_reconnect("authenticate", method_id)

    # -- session lifecycle ------------------------------------------------

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        """``session/new`` → the raw ``NewSessionResponse`` dict
        (``{"sessionId", "modes", "configOptions", ...}``), byte-faithful
        to what the in-process backend returns."""
        params = {"cwd": str(cwd), "mcpServers": list(mcp_servers or [])}
        result = await self._raw_call_with_reconnect("session/new", params)
        return result if isinstance(result, Mapping) else {}

    async def load_session(
        self,
        session_id: str,
        cwd: str,
        mcp_servers: list[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        """``session/load`` → the raw ``LoadSessionResponse`` dict."""
        params = {
            "sessionId": session_id,
            "cwd": str(cwd),
            "mcpServers": list(mcp_servers or []),
        }
        result = await self._raw_call_with_reconnect("session/load", params)
        return result if isinstance(result, Mapping) else {}

    async def list_sessions(
        self,
        cursor: str | None,
        cwd: str | None,
    ) -> Mapping[str, Any]:
        """``session/list`` → the raw ``ListSessionsResponse`` dict
        (``{"sessions": [...], "nextCursor": ...}``)."""
        params: dict[str, Any] = {}
        if cursor is not None:
            params["cursor"] = cursor
        if cwd is not None:
            params["cwd"] = cwd
        result = await self._raw_call_with_reconnect("session/list", params)
        return result if isinstance(result, Mapping) else {"sessions": []}

    async def close_session(self, session_id: str) -> None:
        await self._call_with_reconnect("session_close", session_id)

    def is_session_loaded(self, session_id: str) -> bool:
        """A proxy cannot prove the remote process still has an attachment."""
        return False

    async def set_mode(self, session_id: str, mode_id: str) -> None:
        await self._call_with_reconnect("session_set_mode", session_id, mode_id)

    async def set_config_option(
        self,
        session_id: str,
        config_id: str,
        value: Any,
    ) -> None:
        await self._call_with_reconnect(
            "session_set_config_option",
            session_id,
            config_id,
            value,
        )

    async def inject_user_message(
        self,
        session_id: str,
        text: str,
    ) -> Mapping[str, Any]:
        result = await self._call_with_reconnect(
            "session_inject_user_message",
            session_id,
            text,
        )
        return result if isinstance(result, Mapping) else {"ok": False}

    async def run_command(
        self,
        session_id: str,
        text: str,
        *,
        frontend: str = "api",
        confirmed: bool = False,
    ) -> Mapping[str, Any]:
        # Forward the frontend's explicit confirmation so the
        # orchestrator runtime executes (not re-challenges) an irreversible
        # command — this is the north→south proxy hop that used to drop it.
        result = await self._call_with_reconnect(
            "session_command",
            session_id,
            text,
            frontend=frontend,
            confirmed=confirmed,
        )
        return result if isinstance(result, Mapping) else {"ok": False}

    # -- durable follow-up queue -----------------------------------------
    #
    # The gateway's ACP server resolves ``session/queue/*`` against this
    # backend's ``queue_*`` methods. Without these the gateway returned
    # ``-32601 backend has no queue_add`` and every frontend silently fell
    # back to its non-durable local queue ("durable enqueue failed"). Each
    # forwards 1:1 to the orchestrator's matching ``AcpClient`` helper —
    # same proxy shape as :meth:`inject_user_message`.

    async def queue_list(self, session_id: str) -> Mapping[str, Any]:
        result = await self._call_with_reconnect(
            "session_queue_list",
            session_id,
        )
        return result if isinstance(result, Mapping) else {"ok": False}

    async def queue_add(
        self,
        session_id: str,
        text: str,
        *,
        source: str = "",
        author: str = "",
    ) -> Mapping[str, Any]:
        result = await self._call_with_reconnect(
            "session_queue_add",
            session_id,
            text,
            source=source,
            author=author,
        )
        return result if isinstance(result, Mapping) else {"ok": False}

    async def queue_update(
        self,
        session_id: str,
        queue_id: str,
        text: str,
    ) -> Mapping[str, Any]:
        result = await self._call_with_reconnect(
            "session_queue_update",
            session_id,
            queue_id,
            text,
        )
        return result if isinstance(result, Mapping) else {"ok": False}

    async def queue_remove(
        self,
        session_id: str,
        queue_id: str,
        *,
        reason: str | None = None,
    ) -> Mapping[str, Any]:
        # ``reason`` is keyword-only so a mixed-version orchestrator that
        # still has the two-arg helper cannot see a third positional.
        kwargs: dict[str, str] = {}
        if reason:
            kwargs["reason"] = reason
        result = await self._call_with_reconnect(
            "session_queue_remove",
            session_id,
            queue_id,
            **kwargs,
        )
        return result if isinstance(result, Mapping) else {"ok": False}

    # -- prompt (streaming) ----------------------------------------------

    def prompt(
        self,
        session_id: str,
        message_id: str | None,
        prompt: list[Mapping[str, Any]],
    ) -> AsyncIterator[Mapping[str, Any]]:
        """``session/prompt`` → an async iterator of the orchestrator's
        ``session/update`` dicts, terminated by the ``PromptResponse``.

        Single-consumer, same contract as the in-process backend: each
        intermediate value is a ``sessionUpdate``-shaped dict ready to
        drop into a ``SessionNotification``; the terminal value is the
        ``PromptResponse`` (carries ``stopReason``). We pass every item
        through unchanged so the dispatcher relays the remote turn
        byte-faithfully — including a ``stopReason="cancelled"`` when a
        concurrent :meth:`cancel` lands on the orchestrator.

        ``message_id`` has no ACP slot (the wire keys a prompt by
        ``sessionId``), so it is accepted for Protocol parity and not
        forwarded — matching the in-process backend.
        """
        return self._iter_prompt(session_id, list(prompt))

    async def _iter_prompt(
        self,
        session_id: str,
        blocks: list[Mapping[str, Any]],
    ) -> AsyncIterator[Mapping[str, Any]]:
        acp = await self._ensure_connected()
        # The orchestrator emits SDK-validated stream items
        # (``SessionNotification`` then ``PromptResponse``). Normalise
        # each to the raw JSON dict the dispatcher expects, so the
        # remote and in-process backends yield identical shapes.
        async for item in acp.session_prompt(session_id, blocks):
            yield _stream_item_to_dict(item)

    # -- cancel -----------------------------------------------------------

    async def cancel(self, session_id: str, *, force: bool = False) -> None:
        """``session/cancel`` (fire-and-forget). The confirmation flows
        back via the :meth:`prompt` iterator yielding
        ``stopReason="cancelled"`` — never a return value here.
        """
        await self._call_with_reconnect(
            "session_cancel",
            session_id,
            force=force,
        )

    # -- backend extras with no dedicated ACP wire method -----------------

    def active_turn_id(self, session_id: str) -> str | None:
        """Best-effort live ``turn_id`` for ``session_id``, or ``None``.

        The in-process backend reads its in-memory
        ``_new_backend_active_turns`` map; over the wire there is no
        ``session/active_turn`` method. Per the design, the orchestrator
        owns the authoritative single-flight guard — so the gateway only
        needs a best-effort read here, and the gateway's submit route
        also keeps its OWN ``_SUBMIT_INFLIGHT`` guard. We surface
        whatever the orchestrator publishes on the session's
        ``_meta.sharedagents.active_turn_id`` field (populated by the
        orchestrator's ``list_sessions``); ``None`` when absent.

        SYNC, like the in-process method — ``control_endpoint`` calls it
        without ``await`` (single-flight check). It therefore reads a
        locally-cached snapshot rather than issuing a blocking
        round-trip: the snapshot is refreshed as a side effect of
        :meth:`get_session` / :meth:`list_sessions`. With no snapshot
        yet, it returns ``None`` and the gateway's local
        ``_SUBMIT_INFLIGHT`` guard is the backstop.
        """
        return self._active_turns.get(session_id)

    async def get_session(self, session_id: str) -> Mapping[str, Any] | None:
        """Return the orchestrator's session entry, or ``None``.

        No ``session/get`` exists in ACP, so this reads it off
        ``session/list`` (the orchestrator's authoritative session
        index) and returns the matching entry dict. Also refreshes the
        local ``active_turn_id`` snapshot as a side effect, so a
        subsequent sync :meth:`active_turn_id` read is current.
        """
        result = await self.list_sessions(cursor=None, cwd=None)
        sessions = result.get("sessions") if isinstance(result, Mapping) else None
        if not isinstance(sessions, list):
            return None
        for entry in sessions:
            if not isinstance(entry, Mapping):
                continue
            sid = entry.get("sessionId") or entry.get("session_id")
            if sid != session_id:
                continue
            self._cache_active_turn(session_id, entry)
            return dict(entry)
        # Session not listed (e.g. zero-turn, filtered) — clear any stale
        # active-turn snapshot so single-flight doesn't wedge.
        self._active_turns.pop(session_id, None)
        return None

    async def set_session_title(
        self,
        session_id: str,
        title: str,
        *,
        lock: bool = False,
    ) -> Mapping[str, Any]:
        """Persist ``title`` for ``session_id`` on the orchestrator.

        The control endpoint preserves ``lock`` across the split. Relaying
        through the operator's ``/title`` command would always pin, even for
        ordinary automatic title synchronization with ``lock=False``.
        """
        clean = (title or "").strip()
        if not clean or clean.startswith("untitled-"):
            return {"ok": False, "reason": "empty or placeholder title"}
        import httpx

        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=3.0)) as client:
                response = await client.post(
                    f"{self._base_url}/control/v1/orchestrator/sessions/{session_id}/title",
                    json={"title": clean, "lock": lock},
                    headers=headers,
                )
        except httpx.RequestError as exc:
            raise OrchestratorUnreachable("orchestrator title endpoint unreachable") from exc
        response.raise_for_status()
        envelope = response.json()
        return {"ok": isinstance(envelope, Mapping) and envelope.get("ok") is True, "title": clean, "result": envelope}

    # -- active-turn snapshot cache ---------------------------------------

    @property
    def _active_turns(self) -> dict[str, str]:
        # Lazily created so __init__ stays trivial / picklable-ish.
        cache = self.__dict__.get("_active_turns_cache")
        if cache is None:
            cache = {}
            self.__dict__["_active_turns_cache"] = cache
        return cache

    def _cache_active_turn(
        self,
        session_id: str,
        entry: Mapping[str, Any],
    ) -> None:
        """Refresh the local active-turn snapshot from a list entry."""
        turn_id = _extract_active_turn_id(entry)
        if turn_id:
            self._active_turns[session_id] = turn_id
        else:
            self._active_turns.pop(session_id, None)


# ---------------------------------------------------------------------------
# RemoteAgentRegistry
# ---------------------------------------------------------------------------


class RemoteAgentRegistry:
    """Gateway-side registry of :class:`RemoteBackend` clients.

    Same surface as :class:`sharedagents_gateway.gateway.AgentRegistry`
    (``get`` / ``keys`` / async ``shutdown``) so the gateway can use it
    as its ``manager`` unchanged. ``get(name)`` returns a cached
    :class:`RemoteBackend` pointed at the orchestrator's
    ``/acp/<name>`` surface; the clients connect lazily on first use.
    """

    def __init__(
        self,
        agent_names: list[str],
        orchestrator_url: str,
        *,
        token: str | None = None,
    ) -> None:
        _require_bearer_for_offhost(orchestrator_url, token)
        self.agent_names = list(agent_names)
        self._orchestrator_url = (orchestrator_url or "").rstrip("/")
        self._token = token
        self._backends: dict[str, RemoteBackend] = {}

    def get(self, name: str) -> RemoteBackend:
        """Return the cached :class:`RemoteBackend` for ``name``."""
        backend = self._backends.get(name)
        if backend is not None:
            return backend
        backend = RemoteBackend(
            self._orchestrator_url,
            name,
            token=self._token,
        )
        self._backends[name] = backend
        return backend

    def keys(self) -> list[str]:
        return list(self.agent_names)

    async def shutdown(self) -> None:
        """Close every constructed remote client."""
        backends = list(self._backends.values())
        self._backends.clear()
        for backend in backends:
            try:
                await backend.close()
            except Exception:  # noqa: BLE001
                logger.debug("remote backend shutdown close raised", exc_info=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stream_item_to_dict(item: Any) -> Mapping[str, Any]:
    """Normalise an :class:`AcpClient` prompt-stream item to a raw dict.

    The high-level ``session_prompt`` yields SDK models
    (``SessionNotification`` for updates, ``PromptResponse`` for the
    terminal). The dispatcher / in-process backend speak raw JSON dicts,
    so we unwrap:

    * ``SessionNotification`` → its ``update`` payload (the
      ``sessionUpdate``-shaped dict the dispatcher relays).
    * ``PromptResponse`` → ``{"stopReason": ...}`` (the terminal the
      dispatcher writes as the JSON-RPC response).
    * a plain dict (fake transport / future SDK) → passed through.
    """
    if isinstance(item, Mapping):
        return dict(item)
    # SessionNotification carries the relayable update under ``.update``.
    update = getattr(item, "update", None)
    if update is not None:
        return _model_to_dict(update)
    # Otherwise it is the terminal PromptResponse.
    return _model_to_dict(item)


def _model_to_dict(obj: Any) -> Mapping[str, Any]:
    """Dump a pydantic model (or pass a dict) to a wire-shaped dict."""
    if isinstance(obj, Mapping):
        return dict(obj)
    dump = getattr(obj, "model_dump", None)
    if dump is not None:
        return dump(by_alias=True, exclude_none=True)
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {"value": obj}


def _extract_active_turn_id(entry: Mapping[str, Any]) -> str | None:
    """Pull a live ``active_turn_id`` out of a ``session/list`` entry.

    The orchestrator publishes it on the spec ``_meta.sharedagents`` extras
    block (mirrored from its in-memory map). Tolerant of both the
    aliased (``_meta``) and snake (``field_meta`` / ``meta``) shapes the
    SDK may surface, and of a top-level ``activeTurnId`` for forward
    compat.
    """
    for key in ("activeTurnId", "active_turn_id"):
        val = entry.get(key)
        if isinstance(val, str) and val:
            return val
    meta = entry.get("_meta") or entry.get("field_meta") or entry.get("meta") or {}
    if isinstance(meta, Mapping):
        rt = meta.get("sharedagents")
        if isinstance(rt, Mapping):
            val = rt.get("active_turn_id") or rt.get("activeTurnId")
            if isinstance(val, str) and val:
                return val
    return None


__all__ = [
    "RemoteBackend",
    "RemoteAgentRegistry",
    "OrchestratorUnreachable",
]
