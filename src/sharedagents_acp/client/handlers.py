"""Inbound-method handler registry for ``AcpClient``.

When the client advertises ``ClientCapabilities`` flags (``fs.readTextFile``,
``fs.writeTextFile``, ``terminal``), the agent is allowed to call back
into the client with the corresponding methods. Independently of those
gates, the agent can ALWAYS issue ``session/request_permission`` — the
spec doesn't put it behind a capability flag.

This module owns the small registry the client uses to route those
inbound JSON-RPC requests. Default behavior for any method that isn't
registered (or whose capability isn't advertised) is to reject with
JSON-RPC error code ``-32601`` (``methodNotFound``). That keeps a
mis-configured client honest: an agent that calls ``fs/read_text_file``
on a client that didn't advertise the capability gets a clean error,
not a hang.

Handlers run in tracked tasks so terminal waits and permission decisions
do not block the connection reader. Each request receives its actual
result or RPC error under its original id; pending handlers are cancelled
when the connection closes. Handlers must still yield during long waits
and respect cancellation.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

# A handler is just an async callable that takes the JSON-RPC ``params``
# (already parsed to a dict) and returns the JSON-RPC ``result`` (a dict
# the caller will hand to the framing layer for serialization). Raising
# any exception bubbles up as ``-32603 Internal error`` unless the
# exception is a ``RequestError`` from the SDK, which propagates its
# code/message verbatim.
Handler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


FsReadTextFileHandler = Handler
FsWriteTextFileHandler = Handler
RequestPermissionHandler = Handler
AskUserHandler = Handler


@dataclass
class TerminalHandlers:
    """Bundle of the five ``terminal/*`` handlers.

    All five are required when ``clientCapabilities.terminal`` is
    advertised; the spec doesn't let a client opt into a partial
    terminal surface. We keep them as one bundle so the client's
    handler registry mirrors that all-or-nothing semantics.
    """

    create: Handler
    output: Handler
    wait_for_exit: Handler
    kill: Handler
    release: Handler


class ClientHandlerRegistry:
    """Routes inbound client-side JSON-RPC method calls.

    The registry is populated from ``AcpClient`` based on the
    capabilities the client wants to advertise plus the handlers the
    caller registers explicitly. ``dispatch`` is the lookup the client
    read loop uses; missing routes raise ``MethodNotFound``.
    """

    def __init__(self) -> None:
        self._routes: dict[str, Handler] = {}
        # Methods the client *could* advertise but didn't; the agent
        # shouldn't be calling these and we want the
        # ``-32601 methodNotFound`` error to be unambiguous when it does.
        self._known_methods: set[str] = {
            "fs/read_text_file",
            "fs/write_text_file",
            "session/request_permission",
            "session/ask_user",
            "terminal/create",
            "terminal/output",
            "terminal/wait_for_exit",
            "terminal/kill",
            "terminal/release",
        }

    def register(self, method: str, handler: Handler) -> None:
        """Register ``handler`` for ``method``.

        Raises ``ValueError`` if ``method`` isn't one of the spec's
        client-side methods — there's no useful path for handing the
        agent a route it can't reach by spec.
        """

        if method not in self._known_methods:
            raise ValueError(f"unknown client-side method {method!r}; expected one of {sorted(self._known_methods)}")
        self._routes[method] = handler

    def has(self, method: str) -> bool:
        return method in self._routes

    def dispatch(self, method: str) -> Handler | None:
        """Return the handler for ``method`` or ``None`` if unregistered.

        Distinct from ``KeyError`` so the client read loop can convert
        the absence into a JSON-RPC ``-32601`` error envelope rather
        than crashing.
        """

        return self._routes.get(method)


__all__ = [
    "AskUserHandler",
    "ClientHandlerRegistry",
    "FsReadTextFileHandler",
    "FsWriteTextFileHandler",
    "Handler",
    "RequestPermissionHandler",
    "TerminalHandlers",
]
