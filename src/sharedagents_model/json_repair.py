"""Best-effort recovery for JSON values truncated mid-stream.

Some upstreams (notably vLLM with ``--tool-call-parser gemma4`` driving
gemma-4-31b-nvfp4-mtp on long structured tool calls like ``TodoWrite``)
cut the streamed ``arguments`` JSON off partway through — typically
inside a string value, leaving an unterminated quote and unclosed
braces. ``json.loads`` then raises and the caller is left with the raw
fragment. The model that emitted the call has no way to detect this
and tends to retry the same call verbatim, looping.

``repair_truncated_json`` walks the buffer once, tracks the structural
stack, closes any unterminated string + open scopes, and drops dangling
key tails (``"status":`` with no value yet) so the result parses.
Returns the parsed value on success or raises ``json.JSONDecodeError``
when the buffer is too damaged to recover.

Repair is REFUSED — even when structurally repairable — when the
trailing fragment looks like LLM-repetition garbage (3+ adjacent
backticks, fullwidth-CJK punctuation interleaved with ASCII, or a long
run of identical non-whitespace chars). Those signatures show up when
gemma-4-31b-nvfp4-mtp slips into a token loop inside the streamed
arguments; "repairing" the structure would just dump the garbage into
the recovered value and let the call dispatch with corrupted content.
The caller's ``_arguments_unparsed`` fallback then surfaces a clean
error to the model.

Intentionally narrow: this is for streaming-truncation patterns we've
actually observed, not a general JSON repair library.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Patterns that flag the trailing buffer as LLM-repetition garbage.
# Triggered AFTER the structural-balance check passes but BEFORE we
# return a "successfully" repaired value, so a clean truncation
# repairs but a loopy one fails clean.
_GARBAGE_TAIL_PATTERNS = (
    # 2+ adjacent backticks. Code-fence syntax is rare in tool-call
    # `arguments` JSON (args are paths, queries, ints, short strings)
    # and on gemma-4-31b-nvfp4-mtp specifically `\`\`` shows up at the
    # start of a token loop ("…references``,status:…").
    re.compile(r"`{2,}"),
    # Single backtick immediately followed by what looks like a JSON
    # key boundary — "`,status:" or "`,\"content\":". Standalone
    # backticks appear in legitimate args (regex patterns, code
    # fragments), so we only flag when the model produced a backtick
    # outside a string boundary AND continued with a `,key:` tail.
    # Pattern witnessed on a local vLLM route: '...features\n`,status:"pending"...'.
    re.compile(r"`\s*,\s*\"?\w+\"?\s*:"),
    # Fullwidth-CJK punctuation (`，`, `」`, …) interleaved with ASCII
    # is another fingerprint of the same token loop on gemma-4 NVFP4.
    re.compile(r"[＀-￯]"),
    # 7+ run of the same non-whitespace char (model collapsed onto a
    # single token).
    re.compile(r"([^\s])\1{6,}"),
)
# How many trailing chars of the buffer to scan. Loops typically
# manifest in the last few hundred chars (the rest is the model's
# legitimate output that we want to keep).
_GARBAGE_TAIL_SCAN_CHARS = 400


def _looks_like_garbage_tail(s: str) -> bool:
    """True iff the trailing ``_GARBAGE_TAIL_SCAN_CHARS`` of ``s`` match
    any of the LLM-repetition garbage signatures.
    """
    tail = s[-_GARBAGE_TAIL_SCAN_CHARS:] if len(s) > _GARBAGE_TAIL_SCAN_CHARS else s
    return any(pat.search(tail) for pat in _GARBAGE_TAIL_PATTERNS)


def repair_truncated_json(s: str) -> Any:
    """Parse ``s`` as JSON, repairing trailing-truncation damage if needed."""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Refuse to repair garbage — see module docstring.
    if _looks_like_garbage_tail(s):
        raise json.JSONDecodeError(
            "trailing buffer matches LLM-repetition garbage signature; "
            "refusing to repair (see sharedagents_model.json_repair)",
            s,
            max(0, len(s) - 1),
        )
    in_string = False
    escape = False
    stack: list[str] = []
    for ch in s:
        if escape:
            escape = False
            continue
        if in_string:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack and ((ch == "}" and stack[-1] == "{") or (ch == "]" and stack[-1] == "[")):
            stack.pop()
    repaired = s
    if in_string:
        repaired += '"'
    while stack:
        top = stack.pop()
        repaired += "}" if top == "{" else "]"
    # Drop dangling key tails like `, "status":` or `, "status":}` —
    # the model started a key/value pair but didn't get to the value.
    repaired = re.sub(r',?\s*"[^"\\]*"\s*:\s*([}\]])', r"\1", repaired)
    # Drop trailing commas before closers.
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return json.loads(repaired)
