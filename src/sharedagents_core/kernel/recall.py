"""Leaf recall primitives shared by memory and session indexing.

Session indexing used to reach *into* the memory package for these helpers,
a ``core → memory`` edge that would close a startup import cycle once the
engine relocates. They live here, below both packages, so:

  * ``sharedagents_session.session_index`` imports them downward from the
    kernel, and
  * the memory engine (``memory_index``) imports them downward from the kernel
    too.

Everything here is pure math / string work with no sharedagents dependency:

  * ``RecallResult`` — the source-agnostic recall-hit dataclass every source
    (memory hits, session hits, provider facts) serializes through.
  * ``build_fts_query`` — natural-language → SQLite FTS5 MATCH expression.
  * ``bm25_rank_to_score`` — normalize SQLite ``bm25()`` output to [0, 1].
  * ``format_recall_hits`` — markdown renderer for a hit list.
  * ``DEFAULT_SNIPPET_CHARS`` — snippet cap (matches OpenClaw's manager.ts).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

# Snippet cap (matches OpenClaw's manager.ts).
DEFAULT_SNIPPET_CHARS: int = 700


def fts_terms(raw: str) -> list[str]:
    """Distinct, lowercased lexical tokens the FTS query is built from.

    Shared with the recall scorer so term-*coverage* (how many of the
    query's terms a candidate chunk actually contains) is measured against
    the exact same tokenization the MATCH expression uses. Order-preserving,
    de-duplicated.
    """
    if not raw:
        return []
    seen: dict[str, None] = {}
    for t in re.findall(r"[\w]+", raw, flags=re.UNICODE):
        if t:
            seen.setdefault(t.lower(), None)
    return list(seen)


def build_fts_query(raw: str, *, mode: str = "and") -> str | None:
    """Build an FTS5 MATCH expression from a natural-language query.

    Strategy: extract `[\\w]+` tokens (Unicode-aware), wrap each in
    double quotes (FTS5 phrase syntax), and join them. Returns None on an
    empty query — the caller skips the FTS lookup entirely in that case.

    ``mode`` selects the boolean joiner:

    - ``"and"`` (default) — every term must be present. High precision; the
      right choice when the query is already a tight set of content terms.
    - ``"or"`` — any term may match. A *recall* fallback the caller uses when
      the AND expression matched nothing because the query carried a term the
      corpus simply doesn't contain (e.g. a stop-word like "what" the model
      tacked onto an otherwise-exact question). Ranking (BM25 + term coverage)
      then floats the chunk that contains the most of the query terms to the
      top, so OR does not mean "anything goes".
    """
    tokens = [t for t in re.findall(r"[\w]+", raw, flags=re.UNICODE) if t] if raw else []
    if not tokens:
        return None
    joiner = " OR " if mode == "or" else " AND "
    return joiner.join(f'"{t.replace(chr(34), "")}"' for t in tokens)


def bm25_rank_to_score(rank: float) -> float:
    """Normalize SQLite FTS5 `bm25()` output to [0, 1].

    SQLite returns BM25 ranks; the convention is that lower is better,
    and FTS5 emits negative values when the relevance is high. We
    fold both branches onto a [0, 1] scale that makes the weighted-sum
    fusion well-defined.
    """
    if not math.isfinite(rank):
        return 1 / 1000
    if rank < 0:
        relevance = -rank
        return relevance / (1 + relevance)
    return 1 / (1 + rank)


@dataclass(frozen=True)
class RecallResult:
    """One recall hit, source-agnostic.

    Used by:
      - `session_index.SessionIndex.search` (source="session")
      - `MemoryProvider.prefetch` adapters that want to pre-format
        (external providers may set source="facts")

    `metadata` is intentionally untyped — different sources carry
    different context (date for sessions, category+trust for facts).
    Renderers should treat unknown keys as opaque.
    """

    source: str
    path_or_id: str
    title: str
    snippet: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


def format_recall_hits(hits: Iterable[RecallResult], *, header: str = "") -> str:
    """Render hits as markdown. Empty input renders the empty string.

    Layout matches `memory_tools._format_hit` so the model sees a
    consistent shape across `memory_search` / `session_search` /
    `fact_store search`.
    """
    hits_list = list(hits)
    if not hits_list:
        return ""
    bits: list[str] = []
    if header:
        bits.append(header)
        bits.append("")
    for i, h in enumerate(hits_list, 1):
        bits.append(f"### {i}. {h.title}  ({h.source}, score {h.score:.3f})")
        if h.metadata:
            meta_str = " · ".join(f"{k}={v}" for k, v in h.metadata.items() if v not in (None, "", []))
            if meta_str:
                bits.append(f"_{meta_str}_")
        snippet = (h.snippet or "(empty)")[:DEFAULT_SNIPPET_CHARS]
        bits.append(snippet)
        bits.append("")
    return "\n".join(bits).rstrip() + "\n"


__all__ = [
    "DEFAULT_SNIPPET_CHARS",
    "RecallResult",
    "bm25_rank_to_score",
    "build_fts_query",
    "format_recall_hits",
    "fts_terms",
]
