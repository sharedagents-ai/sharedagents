"""Connect to a Streamable HTTP ACP agent and wrap it in an :class:`AcpClient`.

Mirrors :func:`spawn_stdio_agent` for the HTTP transport. The user
passes a base URL (typically ``http://host:port`` — the ``/acp`` path
is appended automatically), gets back a started + initialized
:class:`AcpClient`. The transport's GET SSE pump is running by the
time we return.

Lifecycle anchor: the caller closes the client via
``client.close()`` when done — that cancels the SSE pump and (if
configured) issues a DELETE to tear the connection down on the
server side.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..protocol.http import HttpClientTransport, _PostResult
from .client import AcpClient


async def connect_http(
    base_url: str,
    *,
    client_capabilities: Any | None = None,
    client_info: Any | None = None,
    bearer_token: str | None = None,
    httpx_client: httpx.AsyncClient | None = None,
    delete_on_close: bool = True,
    frontend: str | None = None,
) -> AcpClient:
    """Connect to a Streamable HTTP ACP agent at ``base_url``.

    The function:

    1. Sends ``initialize`` POST (which mints the connection id on the
       server side).
    2. Captures ``Acp-Connection-Id`` from the response headers.
    3. Opens the GET SSE stream with the same connection id.
    4. Wraps both into an :class:`HttpClientTransport`.
    5. Returns a started + initialized :class:`AcpClient`.

    Args:
        base_url: ``http://host:port`` (no trailing slash). The
            ``/acp`` route is appended internally.
        client_capabilities / client_info: forwarded to ``AcpClient``.
        bearer_token: optional Bearer token; the same value the server
            advertised. ``None`` disables auth.
        httpx_client: optional pre-built :class:`httpx.AsyncClient`.
            Required for the ASGI in-memory transport used by tests
            (where ``base_url`` is something like ``http://test``).
        delete_on_close: when ``True`` (default), closing the client
            issues an ACP ``DELETE`` that tears the connection down on
            the server — which cascades ``close_session`` (hard-cancels
            in-flight turns). Set ``False`` for the gateway→orchestrator
            *membrane* connection so a gateway restart does NOT cancel
            turns running in the orchestrator: the connection just hits
            EOF on the server with no session teardown (crash-isolation).

    Returns:
        A started, initialized :class:`AcpClient`. Caller closes it
        when done.
    """

    acp_url = base_url.rstrip("/") + "/acp"
    headers: dict[str, str] = {}
    if bearer_token is not None:
        headers["Authorization"] = f"Bearer {bearer_token}"

    owns_client = httpx_client is None
    client = httpx_client or httpx.AsyncClient(timeout=30.0)

    # 1. POST initialize.
    init_body = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": _initialize_params(
                client_capabilities,
                client_info,
                frontend,
            ),
        },
        separators=(",", ":"),
    ).encode("utf-8")
    init_resp = await client.post(
        acp_url,
        content=init_body,
        headers={**headers, "Content-Type": "application/json"},
    )
    if init_resp.status_code != 200:
        if owns_client:
            await client.aclose()
        raise ConnectionError(f"initialize failed: HTTP {init_resp.status_code}: {init_resp.text[:200]}")
    connection_id = init_resp.headers.get("acp-connection-id")
    if not connection_id:
        if owns_client:
            await client.aclose()
        raise ConnectionError("server did not return Acp-Connection-Id on initialize")

    # 2. Build the SSE iterator (long-lived GET).
    sse_iterator = _sse_iterator(
        client,
        acp_url,
        headers={
            **headers,
            "Accept": "text/event-stream",
            "Acp-Connection-Id": connection_id,
        },
    )

    # 3. Build the POST callback for outgoing messages.
    async def _post(payload: bytes) -> _PostResult:
        # The session id (if any) for this message is in the body —
        # we surface it as a header for spec compliance with the RFD's
        # "session-scoped POSTs MUST include Acp-Session-Id".
        post_headers: dict[str, str] = {
            **headers,
            "Content-Type": "application/json",
            "Acp-Connection-Id": connection_id,
        }
        sid = _peek_session_id(payload)
        if sid is not None:
            post_headers["Acp-Session-Id"] = sid
        resp = await client.post(acp_url, content=payload, headers=post_headers)
        return _PostResult(status_code=resp.status_code, body=resp.content)

    async def _on_close() -> None:
        if delete_on_close:
            try:
                await client.delete(
                    acp_url,
                    headers={**headers, "Acp-Connection-Id": connection_id},
                )
            except Exception:
                pass
        if owns_client:
            await client.aclose()

    transport = HttpClientTransport(post=_post, sse_iterator=sse_iterator, on_close=_on_close)
    await transport.start()

    acp_client = AcpClient(
        transport,
        client_capabilities=client_capabilities,
        client_info=client_info,
        frontend=frontend,
    )
    await acp_client.start()
    # We already negotiated ``initialize`` over HTTP before the
    # transport was even built (the connection id arrives on that
    # POST's response headers, so we can't defer it to the read
    # loop). Hydrate the AcpClient's post-initialize state directly
    # rather than running a second initialize round.
    _hydrate_initialize_state(acp_client, init_resp.content)
    return acp_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _sse_iterator(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str],
) -> AsyncIterator[bytes]:
    """Yield each SSE ``data:`` event body (raw JSON-RPC bytes).

    Discards comment lines (``: keepalive``) and ignores ``id:`` /
    ``event:`` fields — we only need the JSON-RPC body in the
    ``data:`` line(s).
    """
    async with client.stream("GET", url, headers=headers) as resp:
        if resp.status_code != 200:
            # Propagate the failure as a single error event the read
            # loop will surface.
            return
        # httpx exposes raw lines via ``aiter_lines``; SSE framing is
        # newline-separated.
        data_lines: list[str] = []
        async for raw_line in resp.aiter_lines():
            if not raw_line:
                if data_lines:
                    body = "\n".join(data_lines).encode("utf-8")
                    data_lines = []
                    yield body
                continue
            if raw_line.startswith(":"):
                # Comment (keepalive). Discard.
                continue
            if raw_line.startswith("data:"):
                data_lines.append(raw_line[len("data:") :].lstrip())
                continue
            # Other SSE fields (id:, event:) — ignore, we don't use them.


def _hydrate_initialize_state(client: AcpClient, body: bytes) -> None:
    """Populate ``AcpClient`` post-initialize state from an HTTP-level
    initialize response. The wire round-trip already happened in
    :func:`connect_http`; this just parses the JSON body and stuffs
    the result fields where ``AcpClient.initialize`` would have."""
    import acp.schema as _acp

    try:
        obj = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return
    if not isinstance(obj, dict):
        return
    result = obj.get("result")
    if not isinstance(result, dict):
        return
    try:
        resp = _acp.InitializeResponse.model_validate(result)
    except Exception:
        return
    client.agent_capabilities = resp.agent_capabilities
    client.agent_info = resp.agent_info
    client.auth_methods = list(resp.auth_methods or [])
    client.negotiated_protocol_version = resp.protocol_version


def _peek_session_id(payload: bytes) -> str | None:
    try:
        obj = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(obj, dict):
        return None
    params = obj.get("params")
    if isinstance(params, dict):
        sid = params.get("sessionId")
        if isinstance(sid, str):
            return sid
    return None


def _initialize_params(
    client_capabilities: Any,
    client_info: Any,
    frontend: Any = None,
) -> dict[str, Any]:
    """Best-effort serialize the initialize params.

    ``AcpClient.initialize`` will be the authoritative call once the
    transport is wired — but we need to send an initialize *before*
    we can build the transport (the connection id comes from the
    initialize response). So this is a minimal early initialize that
    primes the connection; ``AcpClient.initialize`` then runs a
    second initialize through the transport for SDK-validated state.
    """
    params: dict[str, Any] = {"protocolVersion": 1}
    if client_capabilities is not None and hasattr(client_capabilities, "model_dump"):
        params["clientCapabilities"] = client_capabilities.model_dump(by_alias=True, exclude_none=True)
    if client_info is not None and hasattr(client_info, "model_dump"):
        params["clientInfo"] = client_info.model_dump(by_alias=True, exclude_none=True)
    # Declare the calling frontend so the server pins it as the
    # connection's authoritative identity for mode-setting.
    if isinstance(frontend, str) and frontend:
        params["frontend"] = frontend
    return params


__all__ = ["connect_http"]
