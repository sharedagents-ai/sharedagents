"""``sharedagents_acp.client`` — outbound ACP client.

Lets any caller in the process drive a remote ACP-compliant agent —
sharedagents_gateway over its own stdio loop, Claude Code's ``claude --acp``
mode, or any other compliant agent. Stdio, WebSocket, and Streamable
HTTP are all wired; the seam is the ``Transport`` ABC from
``sharedagents_acp.protocol`` — swap that and the rest of this module is
transport-agnostic.

Design choices that matter:

- **Schema models are the SDK's**, re-exported via
  ``sharedagents_acp.protocol.schema``. We don't redefine
  ``InitializeResponse`` / ``StopReason`` / ``ContentBlock`` etc. — the
  pinned ``agent-client-protocol`` package already does, and it
  regenerates from the canonical schema on version bumps.
- **Killability is the caller's problem.** ``spawn_stdio_agent``
  returns an ``AcpClient`` whose ``close()`` tears down the transport
  (which in turn closes the subprocess's stdin → SIGPIPE / clean
  shutdown). For a hard kill, the caller holds the
  ``asyncio.subprocess.Process`` reference (returned alongside the
  client when ``return_process=True``) and SIGTERMs it directly. The
  sharedagents agent's ``TurnState.stop`` path uses that handle.
- **Async-iterator prompt.** ``session_prompt`` returns an
  ``AsyncIterator`` that yields each ``session/update`` notification as
  it arrives, then yields the final ``PromptResponse`` and stops. The
  client correlates updates and the response by ``sessionId`` and the
  request id of the original ``session/prompt`` call.
"""

from __future__ import annotations

from .client import AcpClient, PromptStreamItem
from .handlers import (
    FsReadTextFileHandler,
    FsWriteTextFileHandler,
    RequestPermissionHandler,
    TerminalHandlers,
)
from .http_spawn import connect_http
from .spawn import spawn_stdio_agent
from .ws_spawn import connect_ws

__all__ = [
    "AcpClient",
    "FsReadTextFileHandler",
    "FsWriteTextFileHandler",
    "PromptStreamItem",
    "RequestPermissionHandler",
    "TerminalHandlers",
    "connect_http",
    "connect_ws",
    "spawn_stdio_agent",
]
