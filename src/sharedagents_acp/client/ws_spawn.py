"""Client-side WebSocket connect helper for :class:`AcpClient`.

Wraps ``aiohttp.ClientSession.ws_connect`` so a caller can drop-in a
WebSocket transport in place of the stdio spawn path:

.. code-block:: python

    client, session = await connect_ws("ws://server/acp")
    await client.initialize()
    ...
    await client.close()
    await session.close()

Lifecycle:

* The function returns ``(AcpClient, aiohttp.ClientSession)`` so the
  caller owns both teardown points. Closing the client tears down the
  WS (which sends a close frame); closing the aiohttp session frees
  the underlying connector.
* The negotiated ``Acp-Connection-Id`` (returned by the server on the
  upgrade response) is stashed on the client as
  ``client.acp_connection_id`` for callers that want to log it /
  echo it on a future reconnect. We also re-read it onto the
  ``aiohttp.ClientWebSocketResponse`` for parity with the server side.

The helper does NOT call ``initialize()`` for the caller — leaving
the negotiation step explicit matches the stdio spawn helper's
contract and keeps room for callers to wire client capabilities /
inbound handlers between connect and initialize.
"""

from __future__ import annotations

from typing import Any

import aiohttp

from ..protocol.ws import WebSocketTransport
from .client import AcpClient

ACP_CONNECTION_ID_HEADER = "Acp-Connection-Id"


async def connect_ws(
    url: str,
    *,
    client_capabilities: Any = None,
    client_info: Any = None,
    headers: dict[str, str] | None = None,
    session: aiohttp.ClientSession | None = None,
    frontend: str | None = None,
) -> tuple[AcpClient, aiohttp.ClientSession]:
    """Open a WS connection and wrap it in :class:`AcpClient`.

    Args:
        url: full ``ws://`` or ``wss://`` URL (e.g. ``ws://host/acp``).
        client_capabilities: passed through to :class:`AcpClient`.
        client_info: passed through to :class:`AcpClient`.
        headers: extra HTTP request headers to send on the upgrade
            (e.g. an ``Authorization: Bearer <token>``). The current
            server doesn't enforce anything at the WS layer.
        session: optional pre-existing :class:`aiohttp.ClientSession`
            to reuse. If ``None``, this helper creates one and
            ownership returns to the caller via the second return
            value — the caller MUST close it eventually.

    Returns:
        ``(AcpClient, aiohttp.ClientSession)``. The client's read loop
        is started; ``await client.initialize()`` is the next step.

        The negotiated ``Acp-Connection-Id`` (if the server returned
        one on the upgrade response) lands on the returned
        :class:`AcpClient` as ``client.acp_connection_id``.
    """

    own_session = session is None
    if session is None:
        session = aiohttp.ClientSession()

    try:
        ws = await session.ws_connect(url, headers=headers or {})
    except Exception:
        if own_session:
            await session.close()
        raise

    transport = WebSocketTransport(ws)
    client = AcpClient(
        transport,
        client_capabilities=client_capabilities,
        client_info=client_info,
        frontend=frontend,
    )

    # Pull the connection-id off the upgrade response if the server
    # provided one. aiohttp exposes the upgrade response headers via
    # ``ws.response.headers`` (when the response was kept) or
    # ``ws._response`` (private but stable across versions). We try
    # the public path first.
    connection_id: str | None = None
    response = getattr(ws, "_response", None)
    if response is not None and hasattr(response, "headers"):
        connection_id = response.headers.get(ACP_CONNECTION_ID_HEADER)
    # Stash on the client (attribute, not constructor arg, to keep
    # the AcpClient signature stable across transport adapters).
    client.acp_connection_id = connection_id

    await client.start()
    return client, session


__all__ = ["connect_ws", "ACP_CONNECTION_ID_HEADER"]
