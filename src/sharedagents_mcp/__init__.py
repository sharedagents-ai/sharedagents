"""sharedagents-mcp — MCP (Model Context Protocol) client + bundled servers.

Holds the in-process MCP client used by the agent's tool runners
(``mcp_client``) and any first-party MCP servers shipped with sharedagents
(currently ``youtube_mcp_server``). Lives outside ``sharedagents_core`` and
``sharedagents_gateway`` so MCP plumbing can evolve without churning either.
"""

from __future__ import annotations

__all__: list[str] = []
