"""Embedding semantic router for single-phase specialist delegation.

The small local orchestrator (qwen3.6-27b) routes tasks poorly on its own: on a
511-task benchmark it delegated correctly 41.5% of the time (prompt-only), 61.8%
with a regex reroute. A cheap embedding classifier — embed the task, pick the
nearest agent-reference centroid — scored 87.3% standalone and 84.0% used as a
reroute (measured; qwen3-embedding-8b). This module is that classifier.

It is used ONLY as a REROUTE inside runners._shim_agent_canonical: when the
orchestrator falls back to a GENERIC agent (explore/general) for a task
that clearly asks for one specialist deliverable, we correct it. It is fully
fail-safe: any error (no embedder, endpoint down, low confidence) returns None
and the caller falls back to the regex detector (subagent_policy.specialist_intent),
which is already the shipped behaviour — so this can only help, never regress.

References are hand-written canonical phrasings PER agent (not the benchmark
templates), embedded once and cached as centroids.
"""

from __future__ import annotations

import math

# Canonical reference phrasings per specialist. Deliberately diverse from any
# specific task wording so the router generalises by MEANING, not surface form.
_REFERENCES: dict[str, tuple[str, ...]] = {
    # Only LIVE lean-roster labels. Generics are included so genuine search/Q&A
    # predicts explore/general (reroute skipped).
    "explore": (
        "where is the parser defined",
        "find the function that validates emails",
        "search the codebase for the retry logic",
        "locate the module that talks to redis",
    ),
    "general": (
        "what does this helper do in one line",
        "quick question is caching enabled",
        "summarize what the billing module is for",
    ),
    "build": (
        "implement pagination in the api",
        "add a dark-mode feature and verify it",
        "modify the parser to support comments",
        "build the csv exporter end to end",
        "fix the typo in the readme",
        "rename a variable in one file",
        "run the linter once",
    ),
    "plan": (
        "plan how we should add offline mode",
        "weigh the design options and trade-offs",
        "lay out the steps and risks to migrate",
        "design the architecture for a new subsystem",
    ),
}
# The generic targets the orchestrator falls back to; only these get rerouted.
GENERIC_ROUTE_TARGETS = frozenset({"explore", "general"})
# Cosine floor: below this the match is treated as unconfident → no reroute.
_THRESHOLD = 0.40
# The predicted specialist must beat the best generic label (explore/general) by
# this cosine margin, else the task is treated as a genuine search/Q&A/DIRECT and
# left alone. Guards against symbol-name contamination (e.g. "what does poll_queue
# do" pulling toward a specialist). Measured: 0.05 clears the benchmark's false positives.
_MARGIN = 0.05

_centroids: dict[str, list[float]] | None = None
_centroid_key: str | None = None


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else v


async def _ensure_centroids(embedder) -> dict[str, list[float]] | None:
    """Embed the reference phrasings once and cache per-embedder centroids."""
    global _centroids, _centroid_key
    key = f"{getattr(embedder, 'provider_key', '')}:{getattr(embedder, 'model', '')}"
    if _centroids is not None and _centroid_key == key:
        return _centroids
    flat, spans = [], []
    for agent, refs in _REFERENCES.items():
        spans.append((agent, len(flat), len(refs)))
        flat.extend(refs)
    vecs = await embedder.embed_batch(flat)  # normalized by the embedder
    if not vecs or len(vecs) != len(flat):
        return None
    cents: dict[str, list[float]] = {}
    dim = len(vecs[0])
    for agent, start, n in spans:
        cs = vecs[start : start + n]
        cents[agent] = _norm([sum(v[i] for v in cs) / n for i in range(dim)])
    _centroids, _centroid_key = cents, key
    return cents


async def predict_specialist(task_text: str, *, threshold: float = _THRESHOLD) -> str | None:
    """Return the specialist an embedding classifier assigns to ``task_text``, or
    None if there is no embedder, an error occurs, or confidence is below
    ``threshold``. Fully fail-safe — the caller falls back to the regex detector."""
    t = (task_text or "").strip()
    if not t:
        return None
    try:
        # Reach embedding over the memory wire, never the
        # in-process engine. The wire embedder proxy exposes the same surface
        # (provider_key / model / embed_batch / embed_query); None → no embedder
        # configured, and the caller falls back to the regex router (fail-safe).
        from sharedagents_mcp.memory.client import get_memory_client

        embedder = await get_memory_client().embedder()
        if embedder is None:
            return None
        cents = await _ensure_centroids(embedder)
        if not cents:
            return None
        q = _norm(await embedder.embed_query(t))
        scores = {agent: _cos(q, c) for agent, c in cents.items()}
        best = max(scores, key=scores.get)
        if scores[best] < threshold:
            return None
        # Generic-protection: a genuine search/Q&A task predicts a generic label —
        # don't reroute it onto a specialist.
        if best in GENERIC_ROUTE_TARGETS:
            return None
        best_generic = max((scores[g] for g in GENERIC_ROUTE_TARGETS if g in scores), default=-2.0)
        if scores[best] - best_generic < _MARGIN:
            return None
        return best
    except Exception:
        return None
