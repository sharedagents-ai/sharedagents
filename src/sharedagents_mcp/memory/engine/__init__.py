"""sharedagents-memory — fact / vector / synth memory subsystem.

Split out of ``sharedagents_core`` so the memory pipeline (index, embed,
manager, provider, primitives, synth, tools, write) is its own
distribution-shaped package. ``sharedagents_core`` no longer needs to grow
every time a new memory module lands.

Public surface is the modules themselves; import them directly:

    from sharedagents_mcp.memory.engine.memory_index import MemoryIndex
    from sharedagents_mcp.memory.engine.memory_embed import resolve_embedder
    from sharedagents_mcp.memory.engine.memory_write import remember
"""

from __future__ import annotations

__all__: list[str] = []
