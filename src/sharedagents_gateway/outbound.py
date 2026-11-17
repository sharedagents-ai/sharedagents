"""Outbound ACP channel resolution for :class:`CoreBackend`.

Extracted from ``acp_bridge.py`` (behavior-preserving de-monolith). The
single helper here resolves the outbound ACP channel (an ``AcpServer``)
that owns a given session so ``session/request_permission`` and
``session/ask_user`` reach the connection that actually holds it — the
gateway shares one ``CoreBackend`` across many HTTP/WS connections.

Re-exported from ``sharedagents_gateway.acp_bridge`` for import stability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sharedagents_gateway.acp_bridge import CoreBackend


# --- outbound permission request ----------------------------------------


def _resolve_outbound_channel(backend: CoreBackend, session_id: str) -> Any | None:
    """Resolve the outbound ACP channel (an :class:`AcpServer`) for a
    session.

    Prefers the backend's per-session resolver (``_outbound_for`` —
    routes to the connection that owns the session even when one
    CoreBackend is shared by many HTTP/WS connections); falls back to
    the bare ``server_handle`` attribute for stub backends in tests.
    Returns ``None`` when no usable channel exists.
    """
    resolver = getattr(backend, "_outbound_for", None)
    handle = resolver(session_id) if callable(resolver) else getattr(backend, "server_handle", None)
    if handle is None or not hasattr(handle, "request_outbound"):
        return None
    return handle
