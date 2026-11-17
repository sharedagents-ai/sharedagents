"""Fuzzy wake-word gating for the realtime voice agent.

In a noisy room the ASR routinely mistranscribes the wake word — in the real
2026-06-13 Jarvis session "Hey Jarvis" came through as "Hey drivers" / "Hi
Jers" almost every time. So we can't literal-match "jarvis": we match a curated
set of phonetic neighbours / observed mishearings PLUS an edit-distance fallback
to the canonical word, only near the START of the utterance, and return the
command with the wake token (and any leading greeting filler) stripped.

Pure string logic — no model, reuses the ASR transcript we already produce.
"""

from __future__ import annotations

import re

# Observed/likely mishearings of "jarvis". Edit distance alone does NOT catch
# "drivers"/"jers" (too far), so the alias set is the primary signal; the
# distance check just catches close typos like "jarvus" near any alias.
DEFAULT_ALIASES = [
    "jarvis",
    "jervis",
    "jarvus",
    "jarviss",
    "jarvi",
    "jarvees",
    "jervais",
    "drivers",
    "driver",
    "jers",
    "jarv",
    "harvis",
    "travis",
    "service",
    "jarvaa",
]

# Leading greeting fillers to skip before the wake token (EN + PT).
_FILLERS = {"hey", "hi", "hello", "ok", "okay", "yo", "hej", "ei", "oi", "ola", "olá", "ô", "o"}

# How many leading tokens to scan for the wake word (filler + wake). Keeps a
# stray "jarvis" deep in a sentence from counting as an address.
_MAX_LEAD_TOKENS = 3

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def is_wake_token(token: str, aliases: list[str], max_distance: int = 1) -> bool:
    """True if a single lowercased word is one of ``aliases`` (or within
    ``max_distance`` edits of one — catches close typos like "jarvus"). The
    length guard keeps short common words from matching."""
    t = token.lower()
    if t in aliases:
        return True
    return any(abs(len(t) - len(a)) <= max_distance and _levenshtein(t, a) <= max_distance for a in aliases)


def match_wake_word(transcript: str, aliases: list[str] | None = None, max_distance: int = 1) -> tuple[bool, str]:
    """Return ``(matched, command)``.

    ``matched`` is True when the utterance is addressed to the wake word near
    its start. ``command`` is the rest of the utterance (original case) with the
    wake token and any leading greeting filler / punctuation stripped — possibly
    empty (a bare "Jarvis?").
    """
    if not transcript or not transcript.strip():
        return False, ""
    al = [a.lower() for a in (aliases if aliases is not None else DEFAULT_ALIASES)]

    # Token spans so we can slice the ORIGINAL string (preserve case/content)
    # after the matched wake token.
    spans = [(m.group(0), m.start(), m.end()) for m in _WORD_RE.finditer(transcript)]
    for idx, (tok, _start, end) in enumerate(spans):
        if idx >= _MAX_LEAD_TOKENS:
            break
        low = tok.lower()
        if low in _FILLERS:
            continue  # skip a leading greeting and keep scanning
        if is_wake_token(low, al, max_distance):
            command = transcript[end:].lstrip(" ,.!?;:—-\t\n").strip()
            return True, command
        # First non-filler token is NOT the wake word → not addressed to us.
        return False, ""
    return False, ""
