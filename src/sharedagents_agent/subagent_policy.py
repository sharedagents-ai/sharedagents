"""Shared subagent routing and memory policy (single source of truth)."""

from __future__ import annotations

import re as _re

# Curated-memory tools plus their MCP twins. Saved-session recall is
# intentionally excluded: transcripts are shared evidence in this single-owner
# runtime, even when a specialist opts out of MEMORY.md recall.
MEMORY_TOOL_NAMES = frozenset(
    {
        "remember",
        "memory_search",
        "memory_get",
        "meeting_search",
        "mcp__memory__search",
        "mcp__memory__get",
        "mcp__memory__remember",
    }
)


# Single-phase specialist-intent detection. A task that unambiguously asks for
# ONE specialist deliverable ("plan the approach", "implement feature X") should
# route straight to that specialist — the small local model (qwen3.6-27b)
# otherwise defaults to `explore` first or answers DIRECT. This is the PRIMARY
# (high-precision) routing signal; the embedding semantic router is the recall
# fallback for phrasings these miss.
#
# Each pattern encodes the DEFINITIONAL verb + object of a live lean role.
# ORDER MATTERS — first match wins — so plan precedes the general build implementer.
_SPECIALIST_INTENT_PATTERNS: tuple[tuple[str, _re.Pattern[str]], ...] = (
    # Only LIVE lean-roster specialists. (web-research has no specialist row:
    # `general` handles it natively.)
    (
        "plan",
        _re.compile(
            r"\bplan the approach\b|\bplanning artifact\b|\bdesign the architecture\b|\b(write|produce|lay out|draft|weigh)\b[^.]{0,45}\b(plan|implementation plan|design options|steps and risks|approach)\b",
            _re.I,
        ),
    ),
    # `build` LAST: implement/modify a feature + trivial one-file edits.
    (
        "build",
        _re.compile(
            r"\b(implement|modify)\b|\bwrite the implementation\b|\badd\b[^.]{0,40}\bfeature\b|\bfix\b[^.]{0,30}\btypo\b|\brename\b[^.]{0,30}\bvariable\b|\badd\b[^.]{0,25}\bprint\b|\brun the linter\b|\bbump\b[^.]{0,25}\bversion\b",
            _re.I,
        ),
    ),
)

# The generic targets a small model falls back to instead of the right specialist.
# Only these get rerouted; a specialist the model already picked is never overridden.
GENERIC_ROUTE_TARGETS = frozenset({"explore", "general"})


def specialist_intent(task_text: str) -> str | None:
    """Return the specialist agent a single-phase task unambiguously asks for, or
    None. Pass the ORIGINAL user task, not the orchestrator's paraphrase to a
    subagent (the paraphrase strips the intent). See _SPECIALIST_INTENT_PATTERNS."""
    t = task_text or ""
    for name, rx in _SPECIALIST_INTENT_PATTERNS:
        if rx.search(t):
            return name
    return None


def profile_skips_memory(cfg: object) -> bool:
    """True when per-agent ``config.json`` disables memory recall / MEMORY.md.

    Delegation (``parent_session_id``) does NOT suppress memory — some agents
    are invoked specifically for their memory corpus. Only the config flag
    controls skip behavior.
    """
    return not getattr(cfg, "memory_search_enabled", True)
