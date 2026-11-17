"""Streaming + cross-round loop detection extracted from ``turn_runtime``.

Holds the env-tunable loop-detector constants, the surface/tool-call/
cross-round detectors, and the ``TurnLoopTracker`` / ``ProgressTracker``
state machines. Also hosts the small ``_env_int`` / ``_env_float`` env
parsers (defined here alongside the first constants that consume them).
Every symbol is re-exported from ``sharedagents_agent.turn_runtime``.
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any

# --- streaming loop detector ---------------------------------------------
#
# Catches degenerative repetition in the reasoning stream BEFORE the
# model runs out of thinking budget, because repetition self-reinforces
# (arXiv 2512.04419, "Let's Let's Let's…" OpenReview lPwsHTEPQ9): once
# the model loops once, it gets monotonically more likely to loop
# again. The longer we wait, the
# harder recovery is. The detector runs O(n) over a rolling window of
# the reasoning buffer; the streaming loop only invokes it every
# LOOP_CHECK_STRIDE chars to keep amortised cost bounded.
#
# Algorithm: count occurrences of fixed-length substrings (n-grams) in
# the recent suffix of the buffer. If any n-gram appears at least
# LOOP_REPEAT_THRESHOLD times in the last LOOP_WINDOW_CHARS, the model
# is in a loop and the upstream stream is SIGTERM'd.
#
# Tunables are env-overridable so an operator can A/B without a code
# change. Defaults are conservative (long n-gram + many repeats) to
# avoid false positives on legitimate self-quoting like "as noted
# above…" or repeating tool-result snippets.


def _env_int(name: str, default: int) -> int:
    """Parse an int from `os.environ[name]`, falling back to `default`
    on missing OR garbage values. The agent imports this module at
    boot, and a bad env var would otherwise crash startup with a
    `ValueError` that doesn't name the offending knob clearly.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        # Fall back silently — operator typo'd a tunable, the agent
        # still starts. The whole reason the var is configurable is
        # to be opt-in; defaults must work without it.
        import logging as _logging

        _logging.getLogger("sharedagents.config").warning(
            "ignoring bad env %s=%r; using default %s",
            name,
            raw,
            default,
        )
        return default


def _env_float(name: str, default: float) -> float:
    """Float variant of `_env_int`: same missing/garbage-value fallback contract."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        import logging as _logging

        _logging.getLogger("sharedagents.config").warning(
            "ignoring bad env %s=%r; using default %s",
            name,
            raw,
            default,
        )
        return default


LOOP_NGRAM_LEN: int = _env_int("SHAREDAGENTS_LOOP_NGRAM_LEN", 60)
LOOP_REPEAT_THRESHOLD: int = _env_int("SHAREDAGENTS_LOOP_REPEAT_THRESHOLD", 4)
LOOP_WINDOW_CHARS: int = _env_int("SHAREDAGENTS_LOOP_WINDOW_CHARS", 1500)
LOOP_MIN_CONTEXT_CHARS: int = _env_int("SHAREDAGENTS_LOOP_MIN_CONTEXT_CHARS", 1500)
LOOP_CHECK_STRIDE: int = _env_int("SHAREDAGENTS_LOOP_CHECK_STRIDE", 300)
LOOP_TIGHT_NGRAM_LEN: int = _env_int("SHAREDAGENTS_LOOP_TIGHT_NGRAM_LEN", 8)
LOOP_TIGHT_REPEAT_THRESHOLD: int = _env_int("SHAREDAGENTS_LOOP_TIGHT_REPEAT_THRESHOLD", 8)
LOOP_TIGHT_MIN_CONTEXT_CHARS: int = _env_int("SHAREDAGENTS_LOOP_TIGHT_MIN_CONTEXT_CHARS", 80)
LOOP_TIGHT_WINDOW_CHARS: int = _env_int("SHAREDAGENTS_LOOP_TIGHT_WINDOW_CHARS", 240)
# On the ``tool_args`` channel a short n-gram (a path / keyword / JSON
# key) legitimately recurs in otherwise-varied structured input — e.g.
# a Bash command that references ``self-improver`` several times. The
# tight/ngram detectors would flag that as a loop even though the unit
# covers only a small fraction of the payload. Require the repeated unit
# to COVER at least this fraction of the recent window before treating a
# tool_args repeat as a real collapse (a degenerate run covers ~1.0; a
# domain term scattered through a varied command covers ~0.2-0.3).
LOOP_TOOL_ARGS_MIN_COVERAGE: float = _env_float("SHAREDAGENTS_LOOP_TOOL_ARGS_MIN_COVERAGE", 0.5)
# On the reasoning (``thinking``) / ``content`` channels the SAME class of
# false positive appears on the tight tier — a subsystem name or file path
# legitimately re-appears many times in CoT prose or a markdown file
# listing (ground truth: session <session-id> round 6, tier=tight
# evidence='/self-im' channel=thinking, the model enumerating
# ``assets/self-improver/*``). Coverage CANNOT gate this: a genuine
# medium-period reasoning loop has coverage just as low as the scattered
# keyword (a period-28 collapse covers ~0.30, the listing ~0.27), so a
# coverage gate would drop real loops too. Periodicity DOES separate them:
# a real collapse tiles the window so the repeated unit has a single
# continuation (share ~1.0) regardless of period, while a domain term in
# varied text has many distinct continuations (share well below this). A
# tight hit on these channels must clear this dominant-continuation share
# to count as a collapse.
LOOP_REASONING_MIN_PERIODICITY: float = _env_float("SHAREDAGENTS_LOOP_REASONING_MIN_PERIODICITY", 0.5)
LOOP_ENTROPY_WINDOW: int = _env_int("SHAREDAGENTS_LOOP_ENTROPY_WINDOW", 200)
LOOP_ENTROPY_BITS: float = _env_float("SHAREDAGENTS_LOOP_ENTROPY_BITS", 2.5)
# Set SHAREDAGENTS_LOOP_DETECT=0 to disable the loop detector entirely
# (e.g. during distill itself, since enable_thinking=False distillers
# never emit reasoning anyway, but a stray hook could mis-fire).
LOOP_DETECT_ENABLED: bool = os.environ.get(
    "SHAREDAGENTS_LOOP_DETECT",
    "1",
) not in ("0", "false", "False")
LOOP_STRUCTURED_WINDOW: int = _env_int("SHAREDAGENTS_LOOP_STRUCTURED_WINDOW", 500)
LOOP_STRUCTURED_DISTINCT_LINES: int = _env_int(
    "SHAREDAGENTS_LOOP_STRUCTURED_DISTINCT_LINES",
    6,
)
LOOP_STRUCTURED_DISTINCT_SHAPES: int = _env_int(
    "SHAREDAGENTS_LOOP_STRUCTURED_DISTINCT_SHAPES",
    4,
)

# Ambiguous-chrome carve-out (added 2026-05-28 after the ``--------`` Edit
# tool_args false-positive on session <session-id>). Each of these
# ASCII chars is BOTH a documented Qwen/Gemma FP8-KV collapse mode AND a
# legitimate markdown / ASCII-art primitive:
#   ``-`` / ``=`` — horizontal rules, table separators, setext headers
#   ``*`` / ``_`` — emphasis, thematic breaks, divider art
#   ``#``         — markdown headings
#   ``~``         — strikethrough, dividers
#   ``|``         — table column separators (QwenLM/Qwen3-VL#1611:
#                   table transcription false-positives)
# A single window snapshot CANNOT distinguish a legit table separator
# from a model collapse — both look identical to ngram counting and
# entropy. The fix is temporal: a real collapse runs to ``max_tokens``
# (every documented Qwen/Gemma failure mode does — see survey 2026-05-28),
# a markdown table separator finishes in well under 30 s. Gate the fire
# on sustained observation of the same evidence for at least
# ``LOOP_AMBIGUOUS_SUSTAIN_SECONDS`` before letting the detector abort
# the round. Unambiguous collapse heads (``!`` / ``?`` / alphanumeric
# repetition / paraphrase rotation) are NOT on this list — they fire
# immediately as before, because they have no legitimate-chrome
# explanation. Override via ``SHAREDAGENTS_LOOP_AMBIGUOUS_SUSTAIN_SECONDS``;
# set to ``0`` to bypass the sustain gate entirely (restores the
# pre-fix behaviour for A/B regression hunts).
LOOP_AMBIGUOUS_CHARS: frozenset[str] = frozenset("-=*#_~|")
LOOP_AMBIGUOUS_SUSTAIN_SECONDS: float = _env_float(
    "SHAREDAGENTS_LOOP_AMBIGUOUS_SUSTAIN_SECONDS",
    30.0,
)


# Box-drawing glyphs commonly used in ASCII trees. Kept module-level
# so the per-line classifier doesn't re-build the set on every call.
_TREE_GLYPHS: frozenset[str] = frozenset("│├└┌┐┘┴┬┤├─")


# Characters commonly used as horizontal rules in markdown / ASCII art.
# A run made entirely of these is presentation chrome, not a model
# degeneracy. This is INTENTIONALLY a narrow set — only box-drawing
# horizontals and middle-dot variants. ASCII chars `-=*#_~` were
# considered but DELIBERATELY excluded: each is a documented
# Qwen/Gemma FP8-KV collapse mode (same family as the `!`*N pattern
# the rest of the detector ladder catches). A model emitting `#` ×
# 2000 or `_` × 2000 is degenerate, not chrome, and must continue
# to fire through all four detector tiers.
_RULE_CHARS: frozenset[str] = frozenset(
    "─━═╌╍╴╶╾╼╘╛╞╡╤╧╪╫╬"  # box-drawing horizontals
    "·•‧"  # middle-dot variants used as separators
)


def _is_rule_run(s: str) -> bool:
    """True iff ``s`` is non-empty, contains AT LEAST one rule char,
    and is otherwise rule chars or whitespace only.

    Used by every detector tier as a pre-fire skip — a 60-char run of
    `─` is markdown chrome. The "at least one rule char" guard
    prevents a pure-whitespace string from sneaking past.
    """
    if not s:
        return False
    saw_rule = False
    for ch in s:
        if ch in _RULE_CHARS:
            saw_rule = True
        elif not ch.isspace():
            return False
    return saw_rule


_EVIDENCE_METADATA_CHAR = re.compile(r"(?:\bchar=|\bbigram=)'([^']*)'")


def _is_ambiguous_chrome_run(s: str) -> bool:
    """True iff ``s`` is non-empty, contains AT LEAST one ambiguous-chrome
    char (``-=*#_~|``), and is otherwise ambiguous-chrome chars or
    whitespace only.

    Used by :func:`_should_fire_with_sustain` as the gate selector:
    ambiguous-chrome evidence must sustain for
    :data:`LOOP_AMBIGUOUS_SUSTAIN_SECONDS` before a hit fires;
    everything else fires immediately. The "at least one chrome char"
    guard prevents a pure-whitespace string from sneaking past — same
    shape as :func:`_is_rule_run`.

    The entropy tiers don't hand raw text here — they emit description
    strings (``H=0.00bits unique=1 char='-'`` /
    ``H2=0.00bits unique_bigrams=1 bigram='--'``). The alphanumeric
    description chars made every such hit read as unambiguous, so a
    markdown rule caught by the entropy tier bypassed the sustain gate
    entirely (production: 100 single-char-run fires, only 12 gated). Extract the quoted char/bigram
    payload and judge THAT.
    """
    if not s:
        return False
    m = _EVIDENCE_METADATA_CHAR.search(s)
    if m:
        payload = m.group(1)
        return (
            bool(payload)
            and all(ch in LOOP_AMBIGUOUS_CHARS or ch.isspace() for ch in payload)
            and any(ch in LOOP_AMBIGUOUS_CHARS for ch in payload)
        )
    saw_chrome = False
    for ch in s:
        if ch in LOOP_AMBIGUOUS_CHARS:
            saw_chrome = True
        elif not ch.isspace():
            return False
    return saw_chrome


def _should_fire_with_sustain(
    evidence: str,
    sustain_state: dict[str, float] | None,
    now: float,
    *,
    sustain_seconds: float = -1.0,
) -> bool:
    """Decide whether an in-progress loop hit has sustained long enough
    to abort the round.

    Returns True iff the hit should fire NOW:
      - ``evidence`` is NOT ambiguous chrome (every documented
        small-model collapse head — ``!`` / ``?`` / alphanumeric /
        paraphrase opener — falls here, fires immediately);
      - OR ``sustain_state`` is ``None`` (caller opted out: tests,
        legacy paths, or a sustain-disabled config);
      - OR ``sustain_seconds == 0`` (operator A/B'd the gate off);
      - OR at least ``sustain_seconds`` wall-clock has elapsed since
        the first observation of this exact evidence ngram in the
        current round/slot.

    Mutates ``sustain_state`` in place to record the first-seen
    timestamp. On a fresh ambiguous-chrome hit, records the time and
    returns False so the caller suppresses the abort, letting the
    model finish a legitimate horizontal rule / table separator /
    divider block. If the same evidence comes back later the elapsed
    delta decides.

    ``sustain_seconds < 0`` (the default) resolves to the module-level
    :data:`LOOP_AMBIGUOUS_SUSTAIN_SECONDS`. ``sustain_seconds == 0``
    disables the gate entirely (immediate fire) so an operator can
    A/B test the pre-fix behaviour via env.

    The dict key is ``evidence.strip()`` — sliding-window variation
    can produce ``--------`` on one stride check and ``------- `` (a
    trailing whitespace edge) on the next, which the raw-key version
    would treat as distinct evidence and reset the sustain timer for
    each. Stripping collapses these to the same key so the timer
    accumulates correctly across stride checks on the same root cause.

    """
    secs = LOOP_AMBIGUOUS_SUSTAIN_SECONDS if sustain_seconds < 0 else sustain_seconds
    if not _is_ambiguous_chrome_run(evidence):
        return True
    if sustain_state is None or secs <= 0:
        return True
    key = evidence.strip()
    first = sustain_state.get(key)
    if first is None:
        sustain_state[key] = now
        return False
    return (now - first) >= secs


def _suffix_is_structured_content(
    buf_text: str,
    *,
    window: int = LOOP_STRUCTURED_WINDOW,
) -> bool:
    """Recognise structured text that legitimately repeats.

    Two-axis line analysis on the suffix:
      - distinct stripped lines  → varied content (tables, bullets,
        numbered lists, varied code)
      - distinct line shapes     → varied structure even when content
        cycles (Mermaid edges, ASCII trees, indented config dumps)

    Either axis clearing its threshold whitelists the suffix.

    Hard rejects (cannot be legitimate structure regardless of axis):
      - fewer than 2 newlines (pure prose or unary collapse)
      - fewer than 4 non-empty lines (too short to be structure)
      - only 1 distinct stripped line (identical-line degeneracy)

    Multi-line rotation ``(A, B, C, D) × N`` is a known fundamental
    limit — byte-isomorphic with a legit 4-edge cyclic Mermaid
    diagram. No O(window) lexical primitive can distinguish them.
    """
    if len(buf_text) < 40:
        return False
    suffix = buf_text[-window:] if len(buf_text) > window else buf_text
    if suffix.count("\n") < 2:
        return False
    raw = [ln for ln in suffix.split("\n") if ln.strip()]
    if len(raw) < 4:
        return False

    distinct_lines = len({ln.strip() for ln in raw})
    if distinct_lines < 2:
        return False  # single-line collapse

    shapes: set[tuple[int, str, int, str]] = set()
    for ln in raw:
        indent = len(ln) - len(ln.lstrip())
        head = ln.lstrip()[:1]
        if head.isalpha():
            cls = "A"
        elif head.isdigit():
            cls = "9"
        elif head in "-*+":
            cls = "-"
        elif head == "|":
            cls = "|"
        elif head == "#":
            cls = "#"
        elif head == "`":
            cls = "`"
        elif head in _TREE_GLYPHS:
            cls = "T"
        else:
            cls = "?"
        length_bucket = min(len(ln) // 8, 20)
        payload = ln.lstrip()[:2]  # disambiguates Mermaid edges
        shapes.add((min(indent, 16), cls, length_bucket, payload))

    distinct_shapes = len(shapes)

    if distinct_lines >= LOOP_STRUCTURED_DISTINCT_LINES:
        return True
    return bool(distinct_shapes >= LOOP_STRUCTURED_DISTINCT_SHAPES and distinct_lines >= 3)


_MD_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")


def _suffix_has_markdown_table(
    buf_text: str,
    *,
    window: int = LOOP_WINDOW_CHARS,
) -> bool:
    """Recognise markdown tables even when they are small.

    The generic structured-content helper intentionally needs several
    distinct lines/shapes. Small markdown tables can be fewer lines while
    still carrying separator fragments like ``--------`` that should not be
    mistaken for tight token collapse.
    """
    if len(buf_text) < 20:
        return False
    suffix = buf_text[-window:] if len(buf_text) > window else buf_text
    raw = [ln.strip() for ln in suffix.split("\n") if ln.strip()]
    pipe_rows = [ln for ln in raw if ln.startswith("|") and ln.endswith("|")]
    if len(pipe_rows) < 3:
        return False
    if len(set(pipe_rows)) < 2:
        return False  # identical-row degeneracy, not legitimate structure
    if any(_MD_TABLE_SEP_RE.match(ln) for ln in pipe_rows):
        return True
    # Some runtime status output is emitted as pipe-delimited rows without
    # a header/separator line. Require several rows and several separators
    # per row so ordinary prose containing a lone | is not whitelisted.
    return len(pipe_rows) >= 4 and all(ln.count("|") >= 4 for ln in pipe_rows)


def _suffix_is_structured_text(buf_text: str) -> bool:
    """Text-channel structured guard used after a concrete loop hit.

    Markdown tables get a wider lookback because long status rows can exceed
    the default window. Concrete tight/ngram evidence still gets a dominance
    check before this guard suppresses it, so a later unary collapse is not
    hidden by an earlier table.
    """
    return _suffix_is_structured_content(buf_text) or _suffix_has_markdown_table(buf_text)


def _is_unary_collapse_evidence(s: str) -> bool:
    stripped = s.strip()
    return bool(stripped) and len(set(stripped)) == 1


def _suffix_is_identical_line_degeneracy(buf_text: str) -> bool:
    suffix = buf_text[-LOOP_STRUCTURED_WINDOW:] if len(buf_text) > LOOP_STRUCTURED_WINDOW else buf_text
    raw = [ln.strip() for ln in suffix.split("\n") if ln.strip()]
    return len(raw) >= 4 and len(set(raw)) == 1


def _detect_reasoning_loop(
    buf_text: str,
    *,
    ngram_len: int = LOOP_NGRAM_LEN,
    repeat_threshold: int = LOOP_REPEAT_THRESHOLD,
    window_chars: int = LOOP_WINDOW_CHARS,
    min_context_chars: int = LOOP_MIN_CONTEXT_CHARS,
) -> str | None:
    """Return the offending n-gram if the suffix of `buf_text` shows
    degenerative repetition, else None.

    Pure function: no I/O, no side effects, fully testable. The streaming
    loop calls it on the joined `turn.thinking_buf`; we only inspect the
    last `window_chars` characters so the cost is bounded regardless of
    how long the round runs.

    Heuristic: slide a window of length `ngram_len` over the suffix and
    count occurrences. If the most-frequent n-gram appears at least
    `repeat_threshold` times, treat it as a loop. We require the buffer
    to have at least `min_context_chars` before firing so a short
    legitimate repetition (e.g. early in the CoT, "Let me re-check —
    let me re-check") doesn't trip the detector.

    The detector is intentionally simple — Tier 1 of the literature's
    repetition-detection ladder (n-gram counting; SpecRA-style FFT
    autocorrelation is Tier 2 and would only be added if Tier 1 has
    too many false negatives in production).
    """
    if not buf_text or len(buf_text) < min_context_chars:
        return None
    suffix = buf_text[-window_chars:] if len(buf_text) > window_chars else buf_text
    if len(suffix) < ngram_len * repeat_threshold:
        # Not enough material for `repeat_threshold` non-overlapping
        # occurrences even in the worst case.
        return None
    counts: dict[str, int] = {}
    n = ngram_len
    # Step by 1 so overlapping repetitions still count. Most loops are
    # exact repeats so the dominant n-gram emerges quickly.
    for i in range(len(suffix) - n + 1):
        ng = suffix[i : i + n]
        counts[ng] = counts.get(ng, 0) + 1
        if counts[ng] >= repeat_threshold:
            # Skip n-grams that are mostly whitespace / formatting (e.g.
            # 60 spaces in a markdown table). They're usually false
            # positives — formatting chrome, not reasoning.
            stripped = ng.strip()
            if len(stripped) < n // 2:
                # Mostly whitespace; not a real loop signature.
                continue
            # NOTE: the prior `len(set(stripped)) <= 2: continue` skip
            # was intended to suppress markdown rules ("=====", "----").
            # It silently exempted the dominant Qwen3.6+FP8-KV failure
            # pattern (`!`×N — opencode#25129, QwenLM/Qwen3.6#145).
            # NARROW carve-out via _is_rule_run (chars in _RULE_CHARS
            # only) restores tolerance for `─` / `═` / `===` /  etc.
            # markdown chrome without re-exempting the FP8-KV `!`*N
            # collapse — `!` / `?` / letters are not rule chars.
            if _is_rule_run(stripped):
                continue
            return ng
    return None


def _detect_tight_loop(
    buf_text: str,
    *,
    ngram_len: int = LOOP_TIGHT_NGRAM_LEN,
    repeat_threshold: int = LOOP_TIGHT_REPEAT_THRESHOLD,
    window_chars: int = LOOP_TIGHT_WINDOW_CHARS,
    min_context_chars: int = LOOP_TIGHT_MIN_CONTEXT_CHARS,
) -> str | None:
    """Tier-0 detector for short repeated units such as token collapse."""
    if not buf_text or len(buf_text) < min_context_chars:
        return None
    suffix = buf_text[-window_chars:] if len(buf_text) > window_chars else buf_text
    if len(suffix) < ngram_len * repeat_threshold:
        return None
    counts: dict[str, int] = {}
    n = ngram_len
    for i in range(len(suffix) - n + 1):
        ng = suffix[i : i + n]
        counts[ng] = counts.get(ng, 0) + 1
        if counts[ng] >= repeat_threshold:
            stripped = ng.strip()
            if len(stripped) < n // 2:
                # Mostly whitespace; not a real loop signature.
                continue
            # NOTE: the prior `len(set(stripped)) <= 2: continue` skip
            # was intended to suppress markdown rules ("=====", "----").
            # It silently exempted the dominant Qwen3.6+FP8-KV failure
            # pattern (`!`×N — opencode#25129, QwenLM/Qwen3.6#145).
            # NARROW carve-out via _is_rule_run (chars in _RULE_CHARS
            # only) restores tolerance for `─` / `═` / `===` /  etc.
            # markdown chrome without re-exempting the FP8-KV `!`*N
            # collapse — `!` / `?` / letters are not rule chars.
            if _is_rule_run(stripped):
                continue
            return ng
    return None


def _detect_entropy_collapse(buf_text: str) -> str | None:
    """Tier-2 detector for low-entropy collapse in the recent suffix.

    `len(counts) == 1` is the strongest loop signal we can see, not the
    weakest — the prior `if len(counts) <= 2: return None` early-out
    silently exempted the canonical Qwen3.6+FP8-KV `!` × N collapse.
    Markdown-rule false positives are now filtered by length: a short
    `====` run won't reach LOOP_ENTROPY_WINDOW chars; a 200-char unary
    run is degenerate output and must fire.
    """
    if len(buf_text) < LOOP_ENTROPY_WINDOW:
        return None
    suffix = buf_text[-LOOP_ENTROPY_WINDOW:]
    counts: dict[str, int] = {}
    for ch in suffix:
        counts[ch] = counts.get(ch, 0) + 1
    total = len(suffix)
    if len(counts) == 1:
        ch = next(iter(counts))
        # Long runs of horizontal-rule characters (markdown ===, ASCII
        # box ───, etc.) are chrome — don't fire even though H=0.
        if _is_rule_run(suffix):
            return None
        # Pure unary collapse: H = 0 bits.
        return f"H=0.00bits unique=1 char={ch!r}"
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    if entropy >= LOOP_ENTROPY_BITS:
        return None
    return f"H={entropy:.2f}bits unique={len(counts)}"


def _detect_bigram_entropy_collapse(buf_text: str) -> str | None:
    """Tier-2b detector for low-bigram-entropy collapse.

    Catches 'AAAA BBBB AAAA BBBB' which has high unigram entropy
    but low bigram entropy. Uses the same window/threshold as
    the unigram detector for consistency.
    """
    if len(buf_text) < LOOP_ENTROPY_WINDOW:
        return None
    suffix = buf_text[-LOOP_ENTROPY_WINDOW:]
    counts: dict[str, int] = {}
    for i in range(len(suffix) - 1):
        bg = suffix[i : i + 2]
        counts[bg] = counts.get(bg, 0) + 1
    if len(counts) == 1:
        bg = next(iter(counts))
        # Long runs of rule chars produce a single bigram (e.g. `──`
        # for the all-`─` markdown rule). Skip same as unigram tier.
        if _is_rule_run(suffix):
            return None
        # Single bigram (e.g. `!!` from all-`!` suffix) — H2 = 0.
        return f"H2=0.00bits unique_bigrams=1 bigram={bg!r}"
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    # Bigram entropy threshold slightly higher since bigrams have
    # more possible values. If H < 3.0 bits, flag as suspicious.
    BIGRAM_THRESHOLD = max(LOOP_ENTROPY_BITS + 0.5, 3.0)
    if entropy >= BIGRAM_THRESHOLD:
        return None
    return f"H2={entropy:.2f}bits unique_bigrams={len(counts)}"


def _loop_coverage(buf_text: str, ng: str) -> float:
    """Fraction of the recent window covered by NON-overlapping
    occurrences of ``ng``. ~1.0 for a degenerate run; small when ``ng``
    is a domain term recurring through otherwise-varied text."""
    if not ng or not buf_text:
        return 0.0
    suffix = buf_text[-LOOP_TIGHT_WINDOW_CHARS:] if len(buf_text) > LOOP_TIGHT_WINDOW_CHARS else buf_text
    if not suffix:
        return 0.0
    return min(1.0, suffix.count(ng) * len(ng) / len(suffix))


def _ngram_periodicity(buf_text: str, ng: str) -> float:
    """Dominant-continuation share of ``ng`` in the recent window.

    For each occurrence of ``ng`` look at the next ``len(ng)`` chars; the
    share is ``(most-common continuation count / total occurrences)``. A
    periodic collapse — the unit tiling the window — yields a SINGLE
    continuation (share ~1.0) regardless of the period length, because a
    P-periodic buffer always presents the same next-chars after the unit.
    A domain term recurring through otherwise-varied text yields MANY
    distinct continuations (share small). Unlike :func:`_loop_coverage`
    this does NOT conflate a medium-period contiguous loop (share ~1.0,
    coverage ~0.3) with a scattered keyword (share ~0.4) — see the
    empirical table in test_67.

    Occurrences are found with a step of 1 so a self-overlapping unit
    (e.g. ``!!!!!!!!``) still resolves to a single continuation.
    """
    if not ng or not buf_text:
        return 1.0
    suffix = buf_text[-LOOP_TIGHT_WINDOW_CHARS:] if len(buf_text) > LOOP_TIGHT_WINDOW_CHARS else buf_text
    n = len(ng)
    if len(suffix) < 2 * n:
        return 1.0  # too little context to judge — don't suppress
    conts: dict[str, int] = {}
    start = 0
    while True:
        i = suffix.find(ng, start)
        if i < 0:
            break
        cont = suffix[i + n : i + 2 * n]
        if len(cont) == n:
            conts[cont] = conts.get(cont, 0) + 1
        start = i + 1
    if not conts:
        return 1.0
    return max(conts.values()) / sum(conts.values())


def _loop_hit_survives_channel(
    buf_text: str,
    ng: str,
    channel: str,
    *,
    tier: str = "tight",
) -> bool:
    """Channel-specific guard on a tight/ngram hit.

    On ``tool_args`` a real collapse is a degenerate run that DOMINATES
    the payload; a short n-gram that merely recurs (a path / keyword in
    an otherwise-varied command — e.g. ``self-improver`` mentioned a few
    times) is NOT a loop, even if it clears the raw repeat count. Require
    high coverage there (both tiers).

    On the reasoning (``thinking``) / ``content`` channels the same false
    positive bites the TIGHT tier — a subsystem name or file path
    re-appears many times in CoT prose or a markdown listing (ground
    truth: session <session-id> round 6, tier=tight
    evidence='/self-im' channel=thinking). Coverage cannot gate that
    without also dropping genuine medium-period loops (which have equally
    low coverage), so require periodicity instead: a real collapse tiles
    the window (one continuation, share ~1.0) while a scattered domain
    term does not (share < threshold). The longer NGRAM tier stays
    count-only on these channels — a 60-char unit repeating is a genuine
    paragraph loop — and the entropy tier remains the low-entropy
    collapse backstop. All other channels keep count-only behavior.
    """
    if channel == "tool_args":
        return _loop_coverage(buf_text, ng) >= LOOP_TOOL_ARGS_MIN_COVERAGE
    if channel in ("thinking", "content") and _is_unary_collapse_evidence(ng) and _loop_coverage(buf_text, ng) >= 0.5:
        # If the candidate covers most of the tight window, it is the current
        # tail behavior, not a scattered table/domain fragment. Preserve early
        # detection for unary collapse after structured text.
        return True
    if channel in ("thinking", "content") and _suffix_is_identical_line_degeneracy(
        buf_text,
    ):
        return True
    if channel in ("thinking", "content") and _suffix_is_structured_text(
        buf_text,
    ):
        # Production false positives on 2026-05-21 came from the model
        # inspecting structured material in its private reasoning: lspci
        # blocks, markdown status tables, and table separator lines. Treat
        # varied line-structured suffixes as data/prose, not collapse. The
        # helpers hard-reject identical-line/row degeneracy, so a true
        # one-line loop keeps firing.
        return False
    if tier in ("tight", "ngram") and channel in ("thinking", "content"):
        # FP1 fix: the ngram tier was COUNT-ONLY on these
        # channels, so enumerative CoT — a 60-char domain phrase recurring
        # >=4x through a list of near-identical items (e.g.
        # "n1: KINGSTON SNV3S2000G ... PCIe 3.", "n2: KINGSTON ...") — tripped
        # it (gateway.log: LOOP_DETECTED tier=ngram channel=thinking on real
        # hardware-inventory / name-list reasoning). The tight tier already
        # gated on periodicity for exactly this class (step 67, the
        # ``/self-im`` listing); extend the SAME discriminator to ngram. A
        # genuine paragraph loop tiles the window (one continuation,
        # periodicity ~1.0) and still survives; a scattered domain term
        # followed by varied continuations (periodicity < threshold) does not.
        return _ngram_periodicity(buf_text, ng) >= LOOP_REASONING_MIN_PERIODICITY
    return True


def _detect_loops_layered(
    buf_text: str,
    *,
    channel: str = "",
    **_kwargs: Any,
) -> tuple[str, str] | None:
    """Run loop detectors in tight-to-loose priority order.

    ``channel`` selects which detectors apply. The two entropy
    detectors were tuned for the reasoning/content channel (long CoT
    text from local Qwen/Deepseek thinking-mode); on the ``"tool_args"``
    channel they trip on legitimate JSON-encoded markdown, base64
    payloads, repeated-prefix diffs, and similar structured input. A
    hit there used to poison the round (worker emits ``aborted`` and
    the engine hard-cancels). So on ``"tool_args"`` we keep only
    ``tight`` + ``ngram`` for *detection* / telemetry. Abort policy is
    separate: the worker consults ``tool_args_loop_should_abort`` and
    only hard-stops on ``TOOL_ARGS_ABORT_TIERS`` (entropy/ratio) —
    surface ngram/tight hits emit ``tool_args_loop_soft`` and continue
    (2026-07-28 FP demotion; SSH/Write/path-list false aborts).

    Unknown kwargs are accepted for forward/backward compat:
    ``_check_streamed_loop`` historically passes ``ngram_len`` /
    ``repeat_threshold`` overrides, and the runtime wraps the call in
    a ``TypeError`` fallback. Accepting them via ``**_kwargs`` makes
    the fallback path a no-op without coordinating signature bumps.
    """
    if not LOOP_DETECT_ENABLED:
        return None
    evidence = _detect_tight_loop(buf_text)
    if evidence is not None and _loop_hit_survives_channel(buf_text, evidence, channel, tier="tight"):
        return ("tight", evidence)
    evidence = _detect_reasoning_loop(buf_text)
    if evidence is not None and _loop_hit_survives_channel(buf_text, evidence, channel, tier="ngram"):
        return ("ngram", evidence)
    if channel == "tool_args":
        # Entropy detectors are calibrated for reasoning-channel CoT.
        # On tool-args (JSON-encoded prose, markdown bodies, base64,
        # repeated-prefix diffs) they false-positive.
        return None
    if channel in ("content", "thinking") and _suffix_is_structured_text(
        buf_text,
    ):
        # Once tight/ngram hits have been rejected as structured text, don't
        # let low-entropy tiers reclassify a legitimate table/list/code block
        # as a collapse. True unary collapse after the structure is still
        # caught above by the tight tier because this guard no longer runs
        # before concrete evidence is evaluated.
        return None
    evidence = _detect_entropy_collapse(buf_text)
    if evidence is not None:
        return ("entropy", evidence)
    evidence = _detect_bigram_entropy_collapse(buf_text)
    if evidence is not None:
        return ("entropy_bigram", evidence)
    return None


def _rotate_seed(seed: int, retry_idx: int) -> int:
    """Rotate sampler seed on nudge retries to avoid reproducing the same loop."""
    return (seed + retry_idx * 9973) % (2**31)


_TOOL_CALL_LOOP_WINDOW = _env_int("SHAREDAGENTS_TOOL_CALL_LOOP_WINDOW", 10)
# The STRICT (identical-call) detector only fires when the EXACT same
# tool-call set (same names + same JSON-sorted args) repeats for this
# many consecutive rounds. Calling the same tool two or three times in a
# row is normal, legitimate work (retries, paging, re-checking after an
# edit), so a small window false-fires on healthy behaviour. Ten rounds
# of a byte-identical call set is unambiguously a stuck loop.
# NOTE: the caller's tool_call_history buffer must retain >= this many
# rounds or the detector never has enough history to fire (see the trim
# in local_llm.py, which sizes the buffer to the larger of the two
# windows).
# The BROAD (near-duplicate) detector keys on coarse tool+target signatures,
# so it false-fires on normal multi-faceted investigation (e.g. several
# distinct greps over the same tree). A wider window than the strict
# identical-call detector spares legitimate investigation sweeps while still
# trivially catching the ~30-round paraphrased-search loop it exists for.
# NOTE: the caller's tool_call_history buffer must retain >= this many rounds
# or the detector never has enough history to fire.
_TOOL_CALL_LOOP_BROAD_WINDOW = _env_int("SHAREDAGENTS_TOOL_CALL_LOOP_BROAD_WINDOW", 6)


def _detect_tool_call_loop(
    current_calls: list[dict],
    history: list[list[dict]],
) -> bool:
    """Return True if the model is repeating the same tool calls.

    Compares the sorted (name, arguments) signature of the current
    tool calls against each of the last ``_TOOL_CALL_LOOP_WINDOW``
    turns in ``history``.  Arguments are JSON-serialised so dict
    order doesn't cause false negatives.

    Accepts both the flat worker shape (``{"id", "name", "arguments",
    "raw_arguments"}`` — what ``pending_tool_calls`` actually holds)
    and the nested OpenAI shape (``{"function": {"name", "arguments"}}``).
    Reading the wrong key collapsed every signature to ``("", "")``,
    blinding the detector to both the tool name and its args.
    """
    if len(history) < _TOOL_CALL_LOOP_WINDOW:
        return False
    if not current_calls:
        return False
    # Waiting on a background job via identical BashOutput/Monitor polls
    # is NOT a doom loop — ProgressTracker already treats these as
    # streak-neutral; the verbatim detector must match (2026-07-28 FP).
    if _calls_are_bg_poll_only(current_calls):
        return False

    def _sig(calls: list[dict]) -> tuple[tuple[str, str], ...]:
        items: list[tuple[str, str]] = []
        for c in calls:
            fn = c.get("function") if isinstance(c.get("function"), dict) else {}
            name = str(fn.get("name") or c.get("name") or "")
            params = fn.get("arguments")
            if params is None:
                params = fn.get("parameters")
            if params is None:
                params = c.get("arguments")
            if isinstance(params, dict):
                params = json.dumps(params, ensure_ascii=False, sort_keys=True)
            else:
                params = str(params or "")
            items.append((name, params))
        return tuple(sorted(items))

    current_sig = _sig(current_calls)
    return all(_sig(prev_turn) == current_sig for prev_turn in history[-_TOOL_CALL_LOOP_WINDOW:])


def _broad_sig(calls: list[dict]) -> frozenset[str]:
    """Broader tool-call signature: ``tool_name:target_key`` where
    ``target_key`` is the first path/command/pattern argument (not the
    full JSON arguments dict). Two calls to ``Grep`` on the same directory
    with different regex patterns produce the SAME broad signature, catching
    the session-<session-id> pattern where the model varied grep
    patterns while re-searching the same tree."""
    items: set[str] = set()
    for c in calls:
        fn = c.get("function") if isinstance(c.get("function"), dict) else {}
        name = str(fn.get("name") or c.get("name") or "").lower()
        if not name:
            continue
        args = fn.get("arguments")
        if args is None:
            args = fn.get("parameters")
        if args is None:
            args = c.get("arguments")
        if not isinstance(args, dict):
            items.add(f"{name}:*")
            continue
        # Extract the "target" identity from ALL targeting fields, not the
        # first one found. The first-key version keyed Grep on ``path`` alone
        # (``pattern`` shadowed) and truncated at [:60], so six rounds of
        # DIFFERENT patterns / long sibling paths collapsed to one signature
        # and false-fired on legitimate search sweeps (136 tool_call_broad
        # production fires). Paraphrase
        # re-derivation loops are covered by the cross-round SEMANTIC
        # detector below; broad now requires the targeting fields themselves
        # to repeat.
        # Shell commands are NOT broad-collapsed: distinct commands that share
        # a long prefix (``cd <long path> && source venv && python A/B/C``)
        # would otherwise truncate to one signature and false-fire the
        # broad loop detector. Use the FULL command so distinct commands stay
        # distinct; only the strict identical-call detector should match them.
        primary = ""
        if name in ("bash", "shell") and isinstance(args.get("command"), str):
            primary = args["command"].strip()
        else:
            parts = [
                f"{k}={str(args[k]).strip()[:120]}"
                for k in ("file_path", "path", "command", "pattern", "query", "glob_pattern")
                if args.get(k)
            ]
            primary = "|".join(parts)
        if not primary:
            primary = "*"
        items.add(f"{name}:{primary}")
    return frozenset(items)


def _detect_tool_call_loop_broad(
    current_calls: list[dict],
    history: list[list[dict]],
    *,
    window: int = _TOOL_CALL_LOOP_BROAD_WINDOW,
) -> bool:
    """Return True when the last ``window`` rounds share the same BROAD
    tool-call signature — same tool names targeting the same directories /
    command families, even if arguments vary (e.g. different grep patterns
    on the same directory).

    Catches paraphrased search loops that evade ``_detect_tool_call_loop``
    by shuffling regex patterns every round.

    A mutating/progress tool (Edit/Write/TodoWrite/...) exempts the round
    from broad detection — real progress is never a search loop.
    """
    if len(history) < window:
        return False
    if not current_calls:
        return False
    # PROGRESS VETO: a round with a mutating tool is real work, not a loop.
    for c in current_calls:
        fn = c.get("function") if isinstance(c.get("function"), dict) else {}
        name = str(fn.get("name") or c.get("name") or "").lower()
        if name in _XROUND_PROGRESS_TOOLS:
            return False
    current = _broad_sig(current_calls)
    if not current:
        return False
    if not all(_broad_sig(prev) == current for prev in history[-window:]):
        return False

    # ARG-DIVERSITY VETO: if every round in the window used fully distinct
    # full-argument calls (no exact call ever repeated), the model is
    # sweeping — paging a big file with moving offsets, walking a target
    # list — not looping. A real broad loop repeats at least one exact
    # call somewhere in the window.
    def _strict_sigs(calls: list[dict]) -> set[tuple[str, str]]:
        sigs: set[tuple[str, str]] = set()
        for c in calls:
            fn = c.get("function") if isinstance(c.get("function"), dict) else {}
            name = str(fn.get("name") or c.get("name") or "")
            params = fn.get("arguments")
            if params is None:
                params = fn.get("parameters")
            if params is None:
                params = c.get("arguments")
            if isinstance(params, dict):
                params = json.dumps(params, ensure_ascii=False, sort_keys=True)
            else:
                params = str(params or "")
            sigs.add((name, params))
        return sigs

    seen: set[tuple[str, str]] = _strict_sigs(current_calls)
    n_calls = len(seen)
    for prev in history[-window:]:
        prev_sigs = _strict_sigs(prev)
        if seen & prev_sigs:
            return True
        seen |= prev_sigs
        n_calls += len(prev_sigs)
    # No exact call repeated anywhere in the window: arg diversity.
    return False


# ---------------------------------------------------------------------------
# Cross-round SEMANTIC reasoning-loop detector  (the false-NEGATIVE fix)
# ---------------------------------------------------------------------------
#
# Ground truth: session <session-id> turn 1 ran 30
# rounds / 87,101 reasoning chars re-deriving the SAME conclusion ~50× and
# NOTHING fired. Every other detector is per-round (buffers reset each round
# via ``_RoundState()``) and matches verbatim char patterns, so a
# PARAPHRASED loop spread ACROSS rounds is invisible. ``_detect_tool_call_loop``
# is the only cross-round guard and needs IDENTICAL tool signatures — but the
# spinning agent varied its tool calls.
#
# Design:
#   * Signal = turn-scoped word-shingle NOVELTY rate, NOT raw Jaccard. Raw
#     Jaccard is dominated by the growing accumulated set; novelty answers the
#     direct question "how much of THIS round's reasoning is new?".
#   * Armed only after a warmup (round floor + cumulative-chars floor) so a
#     normal short turn never trips.
#   * Fire after K CONSECUTIVE low-novelty rounds; ONE novel round resets.
#   * Progress veto: a round that edits/writes/completes-a-todo or introduces
#     a NEW tool-target key is never a stall. Tool VARIETY alone is NOT
#     progress (the incident varied tools while spinning) — only a new TARGET
#     (path / command / pattern / query) or a mutating tool counts.
# Ship-dark friendly: the tracker only DETECTS; the engine decides the
# action (log → steer → distill → terminate) and gates it on env.

XROUND_LOOP_DETECT_ENABLED: bool = os.environ.get(
    "SHAREDAGENTS_XROUND_LOOP_DETECT",
    "1",
) not in ("0", "false", "False")
# Defaults TUNED ON REPLAY of the real incident (session
# <session-id> turn 1, faithful rounds 1-20): unigram content-word
# novelty stays 0.08-0.40 for the looping rounds while the turn issues ZERO
# mutating tools; it arms by round ~6 (cum ~3k) and fires ~round 19. Larger
# shingles (n>=2) do NOT catch this loop — it is genuinely PARAPHRASED, so
# only word-level novelty separates it. That sensitivity is why the default
# ACTION is "log" (see below): word-level novelty also dips on legitimate work
# that stays on one codebase's vocabulary, so the kill must be enabled only
# after forward per-round telemetry confirms the threshold on clean data.
XROUND_MIN_ROUND_IDX: int = _env_int("SHAREDAGENTS_XROUND_MIN_ROUNDS", 8)
XROUND_MIN_CUM_REASONING_CHARS: int = _env_int(
    "SHAREDAGENTS_XROUND_MIN_CUM_CHARS",
    3_000,
)
# Min PROSE WORDS in a round for it to count as a stall — a short round (a
# one-line thought, a path, a tiny ack) is not enough signal. Char-based
# gating misclassified the incident's short-but-looping rounds.
XROUND_MIN_ROUND_WORDS: int = _env_int("SHAREDAGENTS_XROUND_MIN_ROUND_WORDS", 20)
XROUND_NOVELTY_THRESHOLD: float = _env_float(
    "SHAREDAGENTS_XROUND_NOVELTY_THRESHOLD",
    0.40,
)
XROUND_CONSECUTIVE_STALL: int = _env_int("SHAREDAGENTS_XROUND_STALL_K", 5)
XROUND_SHINGLE_N: int = _env_int("SHAREDAGENTS_XROUND_SHINGLE_N", 1)
# What to do on a confirmed cross-round loop:
#   "log"       — DEFAULT (dark mode): emit XROUND_LOOP_DETECTED telemetry
#                 only, take NO action. This is the "ship dark first"
#                 doctrine for these detectors — collect forward telemetry
#                 and only promote to "terminate" after the threshold is
#                 proven on clean prod data.
#   "terminate" — route into the existing loop-recovery cascade
#                 (distill → force-terminate → continuation nudge) via
#                 rs.loop_detected. Opt IN via SHAREDAGENTS_XROUND_LOOP_ACTION.
#
# DEMOTED back to "log" on 2026-07-03 after a PROD FALSE POSITIVE
# force-terminated a legitimate 21-round deep-analysis turn
# (evidence: novelty=0.21 consecutive_stalls=5 rounds=21). The signal is
# UNIGRAM shingle-novelty (XROUND_SHINGLE_N=1), which unavoidably decays
# on any focused read-only turn that stays on one codebase's vocabulary —
# exactly the failure mode the comment above _shingle_novelty already
# flagged. Terminating on that signal kills real work, so it must not be
# the default. Re-promote only after bigram/trigram novelty or a proven
# threshold makes the signal safe. Detection itself is gated separately by
# SHAREDAGENTS_XROUND_LOOP_DETECT.
XROUND_LOOP_ACTION: str = os.environ.get("SHAREDAGENTS_XROUND_LOOP_ACTION", "log").strip().lower()
# Emit one XROUND_ROUND telemetry line per ARMED round (novelty / words /
# progressed / consecutive_stalls). Only armed rounds (round>=floor &
# cum>=floor) emit, so normal/short turns stay silent and the log captures
# exactly the long-turn population to tune the threshold on. Default on.
XROUND_TELEMETRY: bool = os.environ.get(
    "SHAREDAGENTS_XROUND_TELEMETRY",
    "1",
) not in ("0", "false", "False")

# Mutating / state-advancing tools — their mere presence in a round is
# "progress" regardless of arguments (a real edit/write/todo move means the
# model is acting on the world, not re-deriving).
_XROUND_PROGRESS_TOOLS = frozenset(
    {
        "edit",
        "write",
        "multiedit",
        "notebookedit",
        "applypatch",
        "apply_patch",
        "todowrite",
        "str_replace_editor",
        "create_file",
        "outputsection",
    }
)

# A light stopword set — dropping these stops generic agent glue ("the fix
# is", "I will now") from inflating overlap with real domain repetition.
_XROUND_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "to",
        "of",
        "and",
        "or",
        "is",
        "are",
        "be",
        "it",
        "this",
        "that",
        "we",
        "i",
        "in",
        "on",
        "for",
        "so",
        "now",
        "then",
        "will",
        "would",
        "can",
        "could",
        "should",
        "but",
        "if",
        "as",
        "at",
        "by",
        "with",
        "do",
        "does",
        "did",
        "let",
        "lets",
        "me",
        "my",
        "you",
        "your",
        "they",
        "them",
        "need",
        "needs",
        "want",
        "going",
        "im",
        "ive",
        "actually",
        "wait",
        "hmm",
    }
)

_XROUND_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_XROUND_INLINE_CODE_RE = re.compile(r"`[^`]*`")
# Lines that are enumeration / dumps / structured chrome — not prose. A loop
# detector for REASONING must not ingest a directory listing or a markdown
# table the model is reading.
_XROUND_NONPROSE_LINE_RE = re.compile(r"^\s*(?:[-*+]\s|\d+[.)]\s|\||#{1,6}\s|>{1,}\s|/|\$|>>>|\.\.\.)")
_XROUND_WORD_RE = re.compile(r"[a-z0-9_]+")


def _reasoning_prose_text(text: str) -> str:
    """Strip code fences / inline code / enumeration & dump lines, keeping
    only the natural-language prose of a reasoning buffer. Lowercased."""
    if not text:
        return ""
    t = _XROUND_FENCE_RE.sub(" ", text)
    t = _XROUND_INLINE_CODE_RE.sub(" ", t)
    kept: list[str] = []
    for line in t.splitlines():
        if _XROUND_NONPROSE_LINE_RE.match(line):
            continue
        kept.append(line)
    return "\n".join(kept).lower()


def _reasoning_prose_shingles(
    text: str,
    n: int = XROUND_SHINGLE_N,
) -> set[tuple[str, ...]]:
    """Word n-gram shingles of a reasoning buffer's PROSE (code/enumeration
    stripped, light stopwords dropped). Paraphrase-tolerant: reworded text
    still shares most content shingles."""
    prose = _reasoning_prose_text(text)
    words = [w for w in _XROUND_WORD_RE.findall(prose) if w not in _XROUND_STOPWORDS]
    if len(words) < n:
        # Too little prose to shingle — return the whole (sorted) bag as a
        # single shingle so identical short prose still overlaps.
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def _shingle_novelty(
    round_shingles: set[tuple[str, ...]],
    seen: set[tuple[str, ...]],
) -> float:
    """Fraction of THIS round's shingles not seen earlier this turn. 1.0
    (fully novel) when the round has no shingles or nothing has been seen —
    a no-signal round must never count as a stall."""
    if not round_shingles:
        return 1.0
    unseen = round_shingles - seen
    return len(unseen) / len(round_shingles)


# File-path argument keys — the ONLY arg class used as a "progress target".
# Deliberately excludes command / query / pattern / url / prompt: those are
# far too volatile (a spinning agent issues a different ``sed -n '80,160p'`` or
# a reworded query every round, which would masquerade as progress and
# suppress the loop forever). A genuinely
# progressing run still passes the gate via a mutation OR a NEW file path, and
# even a varied-tool run that is truly advancing shows it through NOVEL
# reasoning (the primary signal), so the progress veto can afford to be strict.
_XROUND_PATH_KEYS = ("file_path", "path", "filePath", "notebook_path")


def _call_args(call: dict) -> dict:
    """Best-effort extraction of a call's argument dict across the flat-worker
    and nested-OpenAI shapes; args may be a dict or a JSON string."""
    if not isinstance(call, dict):
        return {}
    fn = call.get("function") if isinstance(call.get("function"), dict) else {}
    args: Any = fn.get("arguments")
    if args is None:
        args = fn.get("parameters")
    if args is None:
        args = call.get("arguments")
    if args is None:
        args = call.get("parameters")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:  # noqa: BLE001
            return {}
    return args if isinstance(args, dict) else {}


def _call_name(call: dict) -> str:
    if not isinstance(call, dict):
        return ""
    fn = call.get("function") if isinstance(call.get("function"), dict) else {}
    return str(fn.get("name") or call.get("name") or "")


def _tool_target_key(call: dict) -> str:
    """General ``name:primary-arg`` identity for a call (logging / dedup).
    NOT used as the progress signal — see ``_progress_target_key`` for that.
    Accepts flat-worker and nested-OpenAI shapes."""
    name = _call_name(call)
    args = _call_args(call)
    primary = ""
    for k in (
        "file_path",
        "path",
        "filePath",
        "command",
        "pattern",
        "query",
        "url",
        "prompt",
        "notebook_path",
        "cmd",
    ):
        v = args.get(k)
        if v:
            primary = str(v)
            break
    else:
        if args:
            primary = json.dumps(args, sort_keys=True, ensure_ascii=False)
    return f"{name}:{primary.strip()[:120]}"


def _progress_target_key(call: dict) -> str:
    """The FILE-PATH target a call acts on, as ``name:path``, or ``""`` if the
    call carries no file path. Retained for logging/telemetry; NOT used by the
    progress veto (see ``_round_progressed``)."""
    name = _call_name(call)
    args = _call_args(call)
    for k in _XROUND_PATH_KEYS:
        v = args.get(k)
        if v:
            return f"{name}:{str(v).strip()[:120]}"
    return ""


def _round_progressed(tool_calls: list, _unused: object = None) -> bool:
    """True when a round shows real forward progress (suppresses a stall):
    a MUTATING / state-advancing tool (Edit / Write / TodoWrite / …).

    Read-only tools (Read / Grep / Glob / Bash inspection / WebSearch) — even on
    NEW files — are NOT progress: replaying the real incident proved it ran 30
    read-only rounds (Grep×26, Glob×11, Read×9, Bash×18, ZERO mutations) while
    re-deriving the same conclusion, whereas the legitimate coding turn it must
    NOT kill made 12 Edits + 3 TodoWrites. Mutation is the clean separator;
    "touched a new file" is far too lenient (the spin explored files too) and
    "varied command/query" is the disguise the spin actually used. Genuinely
    progressing read-only work is instead protected by NOVEL reasoning (the
    primary signal). The ``_unused`` arg keeps the call site stable."""
    return any(isinstance(c, dict) and _call_name(c).lower() in _XROUND_PROGRESS_TOOLS for c in tool_calls or [])


class TurnLoopTracker:
    """Per-TURN cross-round semantic-loop detector. Constructed once per turn
    (survives the per-round ``_RoundState()`` reset); ``observe`` is called
    once after each completed round with that round's reasoning text and the
    tool calls it issued. Returns ``(tier, evidence)`` the moment a loop is
    confirmed, else ``None``. Pure / no I/O — fully unit-testable."""

    def __init__(
        self,
        *,
        enabled: bool = None,  # type: ignore[assignment]
        min_round_idx: int = XROUND_MIN_ROUND_IDX,
        min_cum_chars: int = XROUND_MIN_CUM_REASONING_CHARS,
        min_round_words: int = XROUND_MIN_ROUND_WORDS,
        novelty_threshold: float = XROUND_NOVELTY_THRESHOLD,
        consecutive_stall: int = XROUND_CONSECUTIVE_STALL,
        shingle_n: int = XROUND_SHINGLE_N,
    ) -> None:
        self.enabled = XROUND_LOOP_DETECT_ENABLED if enabled is None else bool(enabled)
        self.min_round_idx = min_round_idx
        self.min_cum_chars = min_cum_chars
        self.min_round_words = min_round_words
        self.novelty_threshold = novelty_threshold
        self.consecutive_stall = consecutive_stall
        self.shingle_n = shingle_n
        self.seen_shingles: set[tuple[str, ...]] = set()
        self.rounds = 0
        self.cum_chars = 0
        self.stalls = 0
        self.last_novelty: float = 1.0
        self.last_stats: dict = {}

    def observe(
        self,
        reasoning_text: str,
        tool_calls: list,
    ) -> tuple[str, str] | None:
        self.rounds += 1
        self.cum_chars += len(reasoning_text or "")
        prose = _reasoning_prose_text(reasoning_text)
        words = [w for w in _XROUND_WORD_RE.findall(prose) if w not in _XROUND_STOPWORDS]
        n = self.shingle_n
        if len(words) < n:
            shingles = {tuple(words)} if words else set()
        else:
            shingles = {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}
        novelty = _shingle_novelty(shingles, self.seen_shingles)
        self.last_novelty = novelty
        self.seen_shingles |= shingles
        progressed = _round_progressed(tool_calls)

        armed = self.enabled and self.rounds >= self.min_round_idx and self.cum_chars >= self.min_cum_chars
        stall = armed and len(words) >= self.min_round_words and novelty < self.novelty_threshold and not progressed
        if stall:
            self.stalls += 1
        else:
            # Any novel OR progressing round resets the consecutive counter.
            self.stalls = 0
        # Per-round stats for forward telemetry — the engine logs these on
        # armed rounds (correct round attribution, unlike the lossy session
        # logs) so the operator can tune ``novelty_threshold`` / ``K`` on real
        # data before promoting the action from "log" to "terminate".
        self.last_stats = {
            "round": self.rounds,
            "novelty": round(novelty, 3),
            "words": len(words),
            "cum_reasoning_chars": self.cum_chars,
            "progressed": progressed,
            "armed": armed,
            "stall": stall,
            "consecutive_stalls": self.stalls,
        }
        # Return a hit on EVERY stall round once the threshold is reached (not
        # one-shot): in "terminate" mode the caller acts once (the turn ends);
        # in "log"/dark mode each continued-stall round is logged so threshold
        # tuning on prod replay sees the full stall run.
        if self.stalls >= self.consecutive_stall:
            return (
                "xround_semantic",
                f"novelty={novelty:.2f} consecutive_stalls={self.stalls} "
                f"rounds={self.rounds} cum_reasoning_chars={self.cum_chars}",
            )
        return None


# ==========================================================================
# Sustained-zero-progress recovery (replaces the surface loop-detection
# *response*). An empirical FP audit (~/.sharedagents/tmp/loopaudit) measured the
# surface detectors (ngram / tight / xround unigram-novelty / broad tool-call)
# at ~0-15% precision: ~93% of fires KILLED legitimately-
# progressing work. "Progress per round" instead separated real loops (zero
# progress) from focused work (always progresses) at ~100% precision. So we
# stop TERMINATING on surface repetition and instead count a per-turn
# ``zero_progress_streak``, driving two graduated, SAFE responses: a plain
# re-grounding NUDGE at streak>=nudge_streak, and a GRACEFUL finalize (write
# your best answer, terminal_state "ok" — never an error) at
# streak>=hardstop_streak. Priorities, in order: (1) never destroy legit work
# with a false positive; (2) tolerate ~30s of a genuine thinking loop
# (detection can be late); (3) recover toward an answer autonomously; (4) a
# real loop MUST still terminate (the graceful finalize is that backstop).
# A separate read-only research budget in LocalLLMEngine does not label work a
# loop or kill a worker; it requests synthesis after a bounded evidence sweep.

# Streak thresholds — env-overridable, mirrored by AgentConfig fields
# ``progress_nudge_streak`` / ``progress_hardstop_streak`` so an operator can
# tune per-agent. Defaults 3 (nudge) and 8 (~30s of stalled rounds → finalize).
PROGRESS_NUDGE_STREAK: int = _env_int("SHAREDAGENTS_PROGRESS_NUDGE_STREAK", 3)
PROGRESS_HARDSTOP_STREAK: int = _env_int("SHAREDAGENTS_PROGRESS_HARDSTOP_STREAK", 8)

# Mid-stream text-loop tiers that STILL abort the round (high-precision
# degenerate-token spew). The FP-heavy surface tiers (``tight`` / ``ngram``)
# are retired — they must not set a terminating/aborting state. ``ratio`` and
# the entropy tiers catch genuine within-round token collapse and are kept;
# their turn-level action is now the nudge, not a hard kill.
PROGRESS_KEPT_TEXT_TIERS: frozenset[str] = frozenset(
    {
        "entropy",
        "entropy_bigram",
        "ratio",
    }
)

# Mid-stream ``tool_args`` abort allowlist. Same high-precision set as
# ``PROGRESS_KEPT_TEXT_TIERS``. Surface tiers (``tight`` / ``ngram``) still
# *detect* on tool_args (coverage/periodicity gates apply) but the worker
# must NOT hard-abort on them — 2026-07-28 audit: July fires were SSH
# boilerplate, Write bodies, path lists, Bash fragments (same ~0-15%
# precision class as the thinking/content FP audit). Real tool-arg
# collapse is caught post-round by the ratio guard, server
# ``repetition_detection``, and ProgressTracker. Entropy tiers are
# still skipped inside ``_detect_loops_layered`` for tool_args (JSON/
# markdown FP), so this allowlist is empty in practice today and exists
# so a future high-precision tool_args tier can opt into abort without
# rewiring the worker.
TOOL_ARGS_ABORT_TIERS: frozenset[str] = PROGRESS_KEPT_TEXT_TIERS


def tool_args_loop_should_abort(tier: str) -> bool:
    """Return True only when a tool_args detector tier may hard-abort the worker."""
    return str(tier or "") in TOOL_ARGS_ABORT_TIERS


# Background-job status-poll tools. A round whose ONLY calls are these AND
# whose result reports the job is still running is a "wait for the long job"
# turn — a carve-out that must NEVER advance the zero-progress streak. This was
# the single biggest strict-tool_call false-positive class in the audit.
# ``monitor`` is the legacy alias for the same poll family.
_BG_POLL_TOOLS: frozenset[str] = frozenset({"bashoutput", "monitor"})


def _calls_are_bg_poll_only(calls: list[dict]) -> bool:
    """True when every call is a BashOutput/Monitor-family status poll.

    Used to exempt identical-poll loops from the verbatim tool-call
    detector — waiting on a background job is legitimate even across
    many identical poll rounds (ProgressTracker already carves these
    out of the zero-progress streak; the tool-call detector must too).
    """
    if not calls:
        return False
    return all(_call_name(c).lower() in _BG_POLL_TOOLS for c in calls)


def progress_nudge_body(count: int) -> str:
    """Re-grounding nudge injected as a plain user-role message when the turn
    stalls (``zero_progress_streak >= nudge_streak``). Harmless on a false
    positive — the model just reads it and continues. Wording escalates on
    repeat so a model that ignored the first nudge gets a firmer push."""
    if count <= 1:
        return (
            "[engine] You seem to be repeating an approach without "
            "converging. State what you already know and take the single most "
            "useful next action, or give your best answer now — don't repeat "
            "the previous step."
        )
    if count == 2:
        return (
            "[engine] You are still repeating the same step without new "
            "information. Do NOT run it again. Either take a genuinely "
            "different action or write your best answer from what you already "
            "have."
        )
    return (
        "[engine] Repeating this step is not producing progress. Stop "
        "retrying it and write your best answer from current information now."
    )


def progress_finalize_body() -> str:
    """Final message injected once the stall reaches ``hardstop_streak``. The
    model gets ONE more round to write its answer; whatever content exists then
    ends the turn with terminal_state "ok" (a graceful stop, never an error).
    """
    return (
        "[engine] finalize now: you are not making progress. Write your best "
        "answer from the information you already have — do not call any more "
        "tools or repeat previous steps."
    )


def _round_is_bg_poll_wait(tool_calls: list, tool_results: list) -> bool:
    """True when the round is purely a background-job status poll of a
    STILL-RUNNING job (every call a ``BashOutput``-family poll AND a result
    reporting ``running: True``). Such rounds are the carve-out — neutral,
    never advancing the streak — so legit "wait for the long job" turns don't
    look stuck.
    """
    if not tool_calls:
        return False
    for c in tool_calls:
        if _call_name(c).lower() not in _BG_POLL_TOOLS:
            return False
    return any(isinstance(r, dict) and r.get("bg_running") for r in tool_results or [])


class ProgressTracker:
    """Per-TURN sustained-zero-progress counter driving the nudge-ladder +
    graceful-finalize recovery. Constructed once per turn; ``observe_round`` is
    called once per completed round with that round's tool calls, its tool
    results (content-hash + background-poll flag rows — never raw content), and
    whether new user-visible content was emitted. Pure / no I/O — fully
    unit-testable. Returns a verdict string the engine acts on:

      * ``None``            — progress this round (or below threshold): continue.
      * ``"nudge"``         — streak >= nudge_streak: inject a re-grounding
                              message and continue.
      * ``"finalize_inject"`` — streak first reached hardstop_streak: inject the
                              "finalize now" message and allow ONE more round.
      * ``"finalize_stop"`` — the extra finalize round ALSO made zero progress:
                              end the turn gracefully with existing content.

    A round MAKES PROGRESS (resets the streak) if ANY of:
      * a tool call on a target key not seen earlier this turn;
      * a mutation / output tool fired (``_XROUND_PROGRESS_TOOLS``);
      * a tool result whose content-hash differs from all prior results;
      * new user-visible final content was emitted.
    """

    def __init__(
        self,
        *,
        nudge_streak: int = PROGRESS_NUDGE_STREAK,
        hardstop_streak: int = PROGRESS_HARDSTOP_STREAK,
    ) -> None:
        # Guarantee the nudge threshold sits strictly BELOW
        # the finalize (hard-stop) threshold no matter the order or values
        # passed in, so a stalled turn always receives at least one nudge
        # before the graceful finalize. The smaller of the two thresholds is
        # the nudge point; the larger is the hard stop; if they collapse onto
        # the same value the hard stop is pushed one round later so a nudge
        # still precedes every finalize.
        lo = max(1, int(nudge_streak))
        hi = max(1, int(hardstop_streak))
        self.nudge_streak = min(lo, hi)
        self.hardstop_streak = max(lo, hi)
        if self.hardstop_streak <= self.nudge_streak:
            self.hardstop_streak = self.nudge_streak + 1
        self.zero_progress_streak = 0
        self.seen_targets: set[str] = set()
        self.seen_hashes: set[str] = set()
        # Failure-content hashes seen this turn. A round whose failures all
        # repeat one of these is zero progress (see ``_round_progressed``).
        self.seen_error_hashes: set[str] = set()
        self.nudge_count = 0
        self.finalize_fired = False
        self.last_stats: dict = {}

    def _round_progressed(
        self,
        tool_calls: list,
        tool_results: list,
        new_content: bool,
    ) -> bool:
        """Four-disjunct per-round progress signal gating hard-termination.

        Vetoed when the round's ONLY results repeat a failure already seen this
        turn. A model that re-tries a call that cannot succeed re-narrates it
        every round, which used to count as both new content and a new
        target — so a four-round identical-error loop (observed: the review
        seat's ``decision`` calls, all answered "nonempty state required")
        never reached the nudge ladder. A repeated failure is not progress,
        however it is dressed.
        """
        results = [r for r in (tool_results or []) if isinstance(r, dict)]
        failures = [r for r in results if r.get("ok") is False]
        new_failure = any(r.get("hash") and r["hash"] not in self.seen_error_hashes for r in failures)
        repeats_only = (
            bool(failures)
            and len(failures) == len(results)
            # A failure without a content hash cannot be proven a repeat; the
            # first occurrence must count as progress.
            and all(r.get("hash") for r in failures)
            and not new_failure
        )
        progressed = bool(new_content) and not repeats_only
        for c in tool_calls or []:
            if not isinstance(c, dict):
                continue
            if _call_name(c).lower() in _XROUND_PROGRESS_TOOLS:
                progressed = True
            # Broad target key (name:primary-arg) — a NEW target is progress.
            # Deliberately broad (command / url / pattern / query count, per the
            # audit): being lenient here suppresses false positives, and a real
            # loop still reaches zero progress because it REPEATS its target.
            # A repeated failure vetoes it: an argument that changes only
            # because the model re-phrased the same doomed call is not a new
            # target.
            key = _tool_target_key(c)
            if key:
                if key not in self.seen_targets and not repeats_only:
                    progressed = True
                self.seen_targets.add(key)
        for r in results:
            h = r.get("hash")
            if not h:
                continue
            if r.get("ok") is False:
                self.seen_error_hashes.add(h)
            if h not in self.seen_hashes:
                self.seen_hashes.add(h)
                if not repeats_only:
                    progressed = True
        return progressed

    def _verdict(self) -> str | None:
        s = self.zero_progress_streak
        if s >= self.hardstop_streak:
            if not self.finalize_fired:
                self.finalize_fired = True
                return "finalize_inject"
            return "finalize_stop"
        if s >= self.nudge_streak:
            self.nudge_count += 1
            return "nudge"
        return None

    def observe_round(
        self,
        tool_calls: list,
        tool_results: list,
        new_content: bool,
    ) -> str | None:
        # Carve-out: a still-running background-job poll is neutral. Return
        # BEFORE touching the streak or seen-sets so a long polling wait can
        # never nudge or finalize.
        if _round_is_bg_poll_wait(tool_calls, tool_results):
            self.last_stats = {
                "streak": self.zero_progress_streak,
                "progressed": True,
                "bg_poll_wait": True,
            }
            return None
        progressed = self._round_progressed(tool_calls, tool_results, new_content)
        if progressed:
            self.zero_progress_streak = 0
        else:
            self.zero_progress_streak += 1
        self.last_stats = {
            "streak": self.zero_progress_streak,
            "progressed": progressed,
            "bg_poll_wait": False,
            "nudge_count": self.nudge_count,
            "finalize_fired": self.finalize_fired,
        }
        return self._verdict()
