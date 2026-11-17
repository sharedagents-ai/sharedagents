"""Tool implementations for `memory_search` and `memory_get`.

Both tools open a fresh `MemoryIndex` per call (the runner subprocess
exits after one call, so there's no connection to reuse). Lazy
re-index on read: if any tracked file is newer than `files.mtime`/
`files.hash`, we run `index_all()` first so the search sees the latest
content.

When `_resolve_embedder()` returns ``None`` (no embedder configured
or available), search runs in BM25-only mode. Wiring up an embedder
through the resolver is the single switchpoint that enables hybrid
search across every memory tool call.
"""

from __future__ import annotations

import os
from pathlib import Path

from sharedagents_mcp.memory.engine.memory_index import (
    DEFAULT_HYBRID_CANDIDATE_MULTIPLIER,
    DEFAULT_HYBRID_TEXT_WEIGHT,
    DEFAULT_HYBRID_VECTOR_WEIGHT,
    DEFAULT_MAX_RESULTS,
    DEFAULT_MIN_SCORE,
    DEFAULT_SNIPPET_CHARS,
    Embedder,
    MemoryHit,
    MemoryIndex,
)

# ---- agent-home discovery ----


def _resolve_agent_home() -> Path | None:
    """Locate the agent's home directory.

    The runner is a subprocess; the parent agent runtime sets two env
    vars we can rely on:

    - `SHAREDAGENTS_AGENT_HOME`: explicit override (preferred — set by
      `_spawn_tool_runner`).
    - `SHAREDAGENTS_SESSIONS_DIR`: the session JSON dir; agent home is its
      parent. Falls back to this when the home env var is missing for
      back-compat with older runtime builds.
    """
    explicit = os.environ.get("SHAREDAGENTS_AGENT_HOME")
    if explicit:
        return Path(explicit).expanduser()
    sdir = os.environ.get("SHAREDAGENTS_SESSIONS_DIR")
    if sdir:
        p = Path(sdir).expanduser()
        if p.name == "sessions":
            return p.parent
    return None


def _resolve_embedder() -> Embedder | None:
    """Build the configured embedder, or None when disabled / unavailable.

    Defers to ``memory_embed.resolve_embedder()`` if the optional
    embedder module is importable; returns ``None`` otherwise so
    callers degrade to BM25-only search.
    """
    try:
        from sharedagents_mcp.memory.engine.memory_embed import resolve_embedder as _resolve  # type: ignore
    except ImportError:
        return None
    try:
        return _resolve()
    except Exception:  # noqa: BLE001
        return None


def _format_hit(hit: MemoryHit, *, idx: int) -> str:
    """Render one search hit as a markdown block the model can cite directly.

    Format chosen so the model gets `path:start-end` line ranges (which
    the `read_file` tool can then read in full) plus a 700-char snippet
    cap. Score and source labels mirror what `manager.ts::formatHit`
    emits in OpenClaw.
    """
    snippet = hit.snippet[:DEFAULT_SNIPPET_CHARS]
    if not snippet:
        snippet = "(empty)"
    bits = [
        f"### {idx}. {hit.path}:{hit.start_line}-{hit.end_line}  (score {hit.score:.3f})",
        f"_vector {hit.vector_score:.3f} · text {hit.text_score:.3f}_",
        snippet,
    ]
    return "\n".join(bits)


# ---- public callables (returned strings == tool result content) ----


def memory_search(
    query: str,
    *,
    top_k: int = DEFAULT_MAX_RESULTS,
    min_score: float = DEFAULT_MIN_SCORE,
    relevance_floor: float = 0.0,
    sources: list[str] | None = None,
    workspace_origin: str = "",
    agent_home: Path | None = None,
    chat_id: str = "",
) -> str:
    """Hybrid search across MEMORY.md + memory/*.md.

    Returns a markdown blob the model sees in its `tool` message. Empty
    matches return a "no hits" string rather than an error so the model
    can decide what to do next.
    """
    home = agent_home or _resolve_agent_home()
    if home is None:
        return "memory_search failed: agent home not resolvable from env."
    if not isinstance(query, str) or not query.strip():
        return "memory_search failed: query required."
    try:
        top_k = max(1, min(20, int(top_k)))
    except (TypeError, ValueError):
        top_k = DEFAULT_MAX_RESULTS
    try:
        min_score = max(0.0, min(1.0, float(min_score)))
    except (TypeError, ValueError):
        min_score = DEFAULT_MIN_SCORE
    try:
        relevance_floor = max(0.0, min(1.0, float(relevance_floor)))
    except (TypeError, ValueError):
        relevance_floor = 0.0

    embedder = _resolve_embedder()
    idx = MemoryIndex(home, embedder=embedder)
    if idx.needs_reindex():
        idx.index_all()

    hits = idx.search(
        query,
        top_k=top_k,
        min_score=min_score,
        relevance_floor=relevance_floor,
        vector_weight=DEFAULT_HYBRID_VECTOR_WEIGHT,
        text_weight=DEFAULT_HYBRID_TEXT_WEIGHT,
        candidate_multiplier=DEFAULT_HYBRID_CANDIDATE_MULTIPLIER,
        sources=sources,
    )
    if not hits:
        return (
            f"memory_search: no matches for {query!r} "
            f"(min_score={min_score}, mode={'hybrid' if idx.has_vec0 and embedder else 'bm25-only'})."
        )
    bits = [
        f"# memory_search: {query!r}",
        f"_{len(hits)} hit(s) · {'hybrid' if idx.has_vec0 and embedder else 'bm25-only'}_",
        "",
    ]
    for i, h in enumerate(hits, 1):
        bits.append(_format_hit(h, idx=i))
        bits.append("")
    return "\n".join(bits).rstrip() + "\n"


def memory_get(
    file_path: str,
    *,
    start_line: int | None = None,
    end_line: int | None = None,
    workspace_origin: str = "",
    agent_home: Path | None = None,
    chat_id: str = "",
) -> str:
    """Read a memory file (or line range) under the agent's home directory.

    Path is relative to `agent_home`. Refuses absolute paths and `..`
    traversal so a misbehaving model can't pivot from a privacy
    decision into reading arbitrary files.
    """
    home = agent_home or _resolve_agent_home()
    if home is None:
        return "memory_get failed: agent home not resolvable from env."
    if not isinstance(file_path, str) or not file_path:
        return "memory_get failed: file_path required."
    idx = MemoryIndex(home)  # no embedder needed for direct file read
    try:
        body = idx.get_file(
            file_path,
            start_line=start_line,
            end_line=end_line,
        )
    except (ValueError, FileNotFoundError) as e:
        return f"memory_get failed: {e}"
    header = f"# {file_path}"
    if start_line is not None or end_line is not None:
        header += f" (lines {start_line or 1}–{end_line or 'end'})"
    return f"{header}\n\n{body}"


__all__ = ["memory_search", "memory_get"]
