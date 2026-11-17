"""sharedagents memory — the builtin memory MCP capability.

Memory lives here, under ``sharedagents_mcp``, because it *is* a builtin MCP
server (``mcpServers.memory``) — the same way every other bundled MCP
server gets a folder. The public seam is the wire: the MCP surface in
``sharedagents_mcp.memory.server`` (run ``python -m sharedagents_mcp.memory.server``),
re-exposing ``mcp__memory__search`` / ``get`` / ``session_search`` /
``remember``.

Layout:

- ``server`` — the MCP surface (the wire seam). Was ``memory_mcp_server``.
- ``engine`` — the memory pipeline (index, embed, manager, provider,
  primitives, rerank, synth, tools, write). Relocated wholesale from the
  former top-level ``sharedagents_memory`` package (Phase B of
  ``spec/memory-as-builtin-mcp.md``).

No in-process engine API is exported from this package's top level. The
wire-only internal-access boundary (§5, owner-decided 2026-07-08) — every
caller reaching memory over the MCP protocol rather than importing the
engine — is a later phase (Phase C); until then, first-party callers still
import ``sharedagents_mcp.memory.engine.*`` directly.
"""

from __future__ import annotations

__all__: list[str] = []
