"""sharedagents-agent — the gateway distribution.

This package owns:

- the north gateway and south orchestrator (``sharedagents_gateway.gateway``) —
  two Streamable HTTP ACP listeners: ``:8100`` (edge) and ``:8099`` (turn owner),
- ACP-stdio entry point used by remote clients,
- gateway-side concerns that are not ACP-expressible (doctor, etc.).

Modules of interest:

- ``gateway`` — north/south serve loops and agent registry wiring.
- ``gateway_cli`` — ``sharedagents gateway {start,stop,status,...}`` CLI
  (also the entrypoint systemd's ``sharedagents-gateway.service`` calls).
- ``acp_bridge.CoreBackend`` — adapts the sharedagents-core session/turn
  machinery to :class:`sharedagents_acp.server.AgentBackend`.
- ``acp_stdio_main`` — stdio entry point for ACP clients.
- ``agent_cli`` — config-only ``sharedagents agent {init,list,status,...}``
  surface (gateway/orchestrator lifecycle lives in ``gateway_cli``).
- ``runtime`` — shared ``_autodetect_display_env`` /
  ``_load_agent_env_file`` helpers used by ``acp_stdio_main``.
- ``heartbeat``, ``doctor``, ``worker``.

Sibling package: ``sharedagents_mcp`` (MCP client + bundled servers — the
YouTube transcript server and the builtin memory capability, whose
fact / vector / synth engine lives under ``sharedagents_mcp.memory.engine``).
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
