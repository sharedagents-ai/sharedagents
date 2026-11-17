"""Public surface of the ``sharedagents_acp.server`` submodule.

Inbound ACP dispatcher hosting agent-side methods. Transport-agnostic
by design: stdio, WebSocket, and Streamable HTTP all share the same
dispatcher; only the underlying
:class:`~sharedagents_acp.protocol.Transport` differs.

Wire it up by:

* providing an :class:`AgentBackend` implementation (see
  :mod:`sharedagents_acp.server.backend`),
* constructing static advertisement metadata (capabilities / agent_info
  / auth_methods) — these are session-fixed and live in the prefix so
  prefix-cache stability is preserved,
* picking a transport (:class:`~sharedagents_acp.protocol.StdioTransport`
  for now) and handing the pair to :class:`AcpServer`,
* awaiting :meth:`AcpServer.serve_until_close`.

For the standard stdio entry point, :func:`serve_stdio` packages the
transport setup so callers only provide the backend + advertisement
metadata.
"""

from __future__ import annotations

from .backend import AgentBackend
from .server import AcpServer
from .spawn import serve_stdio

__all__ = [
    "AcpServer",
    "AgentBackend",
    "serve_stdio",
]
