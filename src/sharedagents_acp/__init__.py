"""sharedagents-acp — Agent Client Protocol wire library.

Three submodules:

- ``sharedagents_acp.protocol`` — schema models, JSON-RPC framing, transport
  primitives. Independent of any direction (client / server).
- ``sharedagents_acp.client`` — outbound ACP client.
- ``sharedagents_acp.server`` — inbound ACP dispatcher.

Nothing is imported eagerly at package load. Submodules are lazy so
``import sharedagents_acp`` stays cheap and the import-linter contracts
(``acp-purity``, ``tui-isolation``, ``telegram-isolation``) can be
enforced without accidental transitive imports.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
