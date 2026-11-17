"""Explicit end-of-turn ("over") detection for the realtime voice agent.

Aviation / walkie-talkie style: the user signals they have finished speaking by
ending the utterance with a closing word ("over"). Until that word lands as the
FINAL token, the session keeps buffering utterance segments instead of replying
— so a mid-thought pause (which the energy VAD would otherwise treat as
end-of-turn) no longer cuts the user off.

Mid-phrase safety (an explicit requirement): the closing word only triggers
when it is the LAST word of the utterance — "do the thing, over" commits, but
"move it over there" does not, and only the trailing "over" is stripped (an
earlier, embedded "over" is preserved as content).

Pure string logic — no model — mirroring :mod:`sharedagents_realtime.wake`.
"""

from __future__ import annotations

import re

from .wake import is_wake_token

# Default closing word. Kept to an EXACT, short set on purpose: "over" is a
# common English word, so it is NOT fuzzy-expanded by default (that would let
# "ever"/"oven"/"hover" end a turn). Extra spellings come from config aliases.
DEFAULT_END_ALIASES = ["over"]

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def match_end_word(transcript: str, aliases: list[str] | None = None, max_distance: int = 0) -> tuple[bool, str]:
    """Return ``(is_terminal, remainder)``.

    ``is_terminal`` is True when the utterance ENDS with the closing word.
    ``remainder`` is the utterance with that one trailing closing word (and any
    trailing punctuation/whitespace) removed — original case/content preserved;
    it may be empty (a bare "over"). When not terminal, ``remainder`` is the
    transcript unchanged (ready to be buffered verbatim).

    ``max_distance`` defaults to 0 (exact match) because the closing word is
    short and common; raise it only with a rarer configured word.
    """
    if not transcript or not transcript.strip():
        return False, ""
    al = [a.lower() for a in (aliases if aliases is not None else DEFAULT_END_ALIASES)]
    spans = [(m.group(0), m.start(), m.end()) for m in _WORD_RE.finditer(transcript)]
    if not spans:
        return False, ""
    tok, start, _end = spans[-1]
    if is_wake_token(tok.lower(), al, max_distance):
        # Strip the connective separator before the closing word (space, comma,
        # dash, …) but keep sentence-ending . ! ? as content.
        remainder = transcript[:start].rstrip(" ,;:—-\t\n").strip()
        return True, remainder
    return False, transcript
