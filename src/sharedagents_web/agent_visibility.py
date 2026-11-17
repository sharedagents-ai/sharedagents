"""Which agents a human may start from the Web UI.

Delegation-only utilities are not conversational: they exist so jarvis /
plan / build can dispatch them, not so the picker can open a New Chat.
"""

from __future__ import annotations

# Explore / general are search-and-research specialists. Compaction / title /
# summary / docs are runtime utilities. None of these belong in the Web
# picker; a monitor deep-link can still open their existing session.
#
# ⚠ This is the ONE list that decides WebUI picker membership (a name here is
# skipped by ``routes/sessions._agent_roster`` and coerced to the default at
# session create). Two other look-alike frozensets — ``engine_modes.discover``'s
# ``non_conversational`` and ``command_runtime._NATIVE_NON_CONVERSATIONAL`` —
# carry the same names but only decide the /mode matrix. They are NOT the same
# list and must not be "kept in sync" with this one. ``review`` is deliberately
# absent here so it stays visible under "Native" in the picker, while being
# present in those two (auto-only). Adding it here would delete the reviewer
# from the WebUI while every unit test still passed.
NON_CONVERSATIONAL_AGENTS = frozenset(
    {
        "explore",
        "general",
        "docs",
        "compaction",
        "title",
        "summary",
    }
)


def is_non_conversational_agent(name: str | None) -> bool:
    return str(name or "").strip().lower() in NON_CONVERSATIONAL_AGENTS


def conversational_agent_or_default(name: str | None, default: str) -> str:
    agent = str(name or "").strip() or default
    if is_non_conversational_agent(agent):
        return default
    return agent
