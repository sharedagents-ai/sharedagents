"""sharedagents_core.kernel — the shared-primitive leaf layer.

This package is a **true leaf**: its modules import only the standard
library and third-party packages, and **never** import from the rest of
``sharedagents_core`` or from the memory engine (``sharedagents_mcp.memory``) (not at
module top level, not lazily inside a function). It is the layer *below both* — the small set of
primitives that the memory engine and core proper both need — so the
dependency arrow points one way (``core → kernel``, ``memory → kernel``)
with no back-edge and no import-time cycle.

Why it exists (Phase A of the memory-as-builtin-MCP relocation): memory and
core were coupled bidirectionally at import time. Memory borrowed
``wrap_untrusted`` (the untrusted-content scrub), ``get_bus``, the prompt-pair
parsers, ``resolve_service``, path helpers and light config accessors from
core; core's ``session_index`` reached back into memory for the recall
primitives. Pulling those shared primitives down into this leaf breaks the
cycle before the engine is relocated (Phase B), so the memory engine
(now ``src/sharedagents_mcp/memory/engine/*``, relocated in Phase B) imports
nothing from ``sharedagents_core`` at module top level.

Leaf discipline is a rule, not a mechanism: there is no CI on this repo, so the
review step is load-bearing. The guard is a grep — no import of a non-kernel
``sharedagents_core`` submodule and no import of the memory engine
(``sharedagents_mcp.memory``) anywhere under this directory. See
``spec/memory-as-builtin-mcp.md`` §3.1, §7 R5.

Modules
-------
- ``injection_guard`` — the structural prompt-injection scrub (``wrap_untrusted``
  et al.). SECURITY primitive; canonical home is here, re-exported from
  ``sharedagents_core.injection_guard`` for existing importers.
- ``event_bus`` — the in-process debounced event bus (``get_bus``, ``EventBus``).
- ``prompt_pair`` — the ``<!-- system -->`` / ``<!-- user -->`` markdown
  prompt-pair parsers (``load_prompt_pair``, ``parse_prompt_pair_text``).
- ``services`` — the unified primary/backup service resolver (``resolve_service``).
- ``layout`` — on-disk path math (``sessions_root`` and siblings).
- ``config`` — light global-config accessors + the shipped memory-synth template.
- ``recall`` — leaf recall math shared by memory and session indexing
  (``RecallResult``, ``build_fts_query``, ``bm25_rank_to_score``,
  ``format_recall_hits``, ``DEFAULT_SNIPPET_CHARS``).
"""
