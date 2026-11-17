"""Session-title strategy (pure logic, no I/O).

**The sequence** (operator-specified):

1. A new session starts with ``New Chat - HHMMSS``. This is the visible
   placeholder while the title call is running; the opening prompt is never
   copied directly into the title.
2. On turn 1, the title prompt is scheduled at the same time as the user
   turn, always using Jarvis's default deployment (not the turn's model).
   It is deliberately detached from the response stream, so a slow title
   call cannot add latency to the first reply.
3. On turn 10, the same title procedure runs again to refresh the title.
   Root and child/subagent sessions follow the same lifecycle. The Agent
   layer always selects the Jarvis default deployment, because the session
   engine (CLI, ACP, or Codex-via-local-LLM) may not expose an
   OpenAI-compatible ``/chat/completions`` endpoint.

This module holds the *decisions* (when to fire, how to build prompts,
how to clean the model's reply, how to detect churn) so the strategy is
unit-testable without a running agent. The async LLM call + persistence
trigger live on :class:`sharedagents_agent.agents.agent.Agent`.

OpenCode parity: packages/opencode/src/session/prompt.ts ``ensureTitle``
(prompt cue + ``<think>`` strip + first-non-empty-line + cap), adapted here
to the explicit timestamp placeholder and turn-1/turn-10 lifecycle.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

_log = logging.getLogger(__name__)

from sharedagents_session.session import (  # noqa: E402
    SESSION_TITLE_MAX_CHARS,
    Session,
    build_new_chat_title as _build_new_chat_title,
)

# An injected async callable that takes the chat ``messages`` and returns
# the model's raw reply text. The Agent supplies the real httpx call; tests
# supply a fake. Keeping it injected is what makes the trigger logic below
# unit-testable with no network and no Agent.
TitleCaller = Callable[[list[dict[str, str]]], Awaitable[str]]

# Generated titles are 2–6 words. The title agent prompt still asks for
# 3–6, but the examples (and a loaded 27B under GPU contention) often
# emit two-word titles such as "Ontology explanation" / "Config review".
# Rejecting those left the sidebar on ``New Chat - HHMMSS`` with no log.
TITLE_MIN_WORDS = 2
TITLE_MAX_WORDS = 6
HIDDEN_AGENT_TITLE_MAX_CHARS = 80
# Short openers the title model often collapses to one word ("Greeting")
# even though validation requires at least two.
_GREETING_PROMPTS = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "yo",
        "sup",
        "lol",
        "hiya",
        "howdy",
    }
)
# Repair stale title-agent outputs and one-word model slips before reject.
_SINGLE_WORD_TITLE_REPAIRS = {
    "greeting": "Quick greeting",
    "hello": "Quick hello",
    "chat": "Light chat",
    "intro": "Intro message",
}
# Runaway cap for the title-call *input*. The model must see the whole
# opening prompt (not the first line), but a pasted dump must not fill
# the side-call context.
TITLE_PROMPT_MAX_CHARS = 8000

# Title calls are launched in parallel with these user turns.
TITLE_GENERATION_TURNS: tuple[int, ...] = (1, 10)
HIDDEN_AGENT_TITLE_TURN = 1

_TEMP_CHAT_TITLE_RE = re.compile(r"^New Chat - \d{6}$")

_STALE_TITLE_CHAR_LIMIT = re.compile(
    r"≤\s*50\s*characters|<=\s*50\s*characters|≤50 chars",
    re.IGNORECASE,
)

_QUOTE_CHARS = "\"'“”‘’`"
_PREFIXES = ("session title:", "title:", "topic:", "new title:")


def is_default_title(title: Any) -> bool:
    """True for OpenCode/sharedagents placeholder titles that should be replaced.

    Matches OpenCode ``Session.isDefaultTitle`` plus sharedagents's legacy
    ``untitled-<hash8>`` and active ``New Chat - HHMMSS`` placeholders.
    """
    t = str(title or "").strip()
    if not t:
        return True
    if t.startswith("untitled-"):
        return True
    if _TEMP_CHAT_TITLE_RE.fullmatch(t):
        return True
    if t.startswith("New session"):
        return True
    return t in ("New Chat", "Untitled")


def build_new_chat_title(timestamp: Any = None) -> str:
    """Public title-layer alias for the shared session placeholder helper."""
    return _build_new_chat_title(timestamp)


def build_temp_title(user_text: Any) -> str:
    """Legacy prompt-derived title helper retained for compatibility.

    New sessions use :func:`build_new_chat_title` instead. Existing callers
    still use this clipped derivation when inspecting legacy records.
    """
    return Session._derive_temp_title(user_text)


def build_web_temp_title(user_text: Any) -> str:
    """Legacy WebUI prompt-derived title helper retained for old records."""
    if isinstance(user_text, list):  # vision multipart — same extraction
        user_text = first_line(user_text)
    content = str(user_text or "")
    title = content[:60].strip()
    if len(content) > 60:
        title += "..."
    return title or "New Chat"


def temp_title_variants(session: Any) -> set[str]:
    """Legacy prompt-derived stand-ins that remain replaceable on turn 1."""
    turns = list(getattr(session, "turns", None) or [])
    if not turns or not isinstance(turns[0], dict):
        return set()
    user = turns[0].get("user")
    return {
        build_temp_title(user),
        first_line(user),
        build_web_temp_title(user),
    }


def temp_title_for(session: Any) -> str:
    """The temp title this session's opening prompt would produce today."""
    turns = list(getattr(session, "turns", None) or [])
    if not turns or not isinstance(turns[0], dict):
        return ""
    return build_temp_title(turns[0].get("user"))


def is_replaceable_title(session: Any) -> bool:
    """True when the current title is a placeholder or a turn-1 temp title.

    Mirrors OpenCode ``Session.isDefaultTitle``, widened by the temp phase:
    the first-line stand-in is an interim UI value, not an operator rename,
    so the title agent may overwrite it. A real operator ``/title`` (or any
    title that is neither default nor a temp variant) is never touched.
    """
    title = str(getattr(session, "title", "") or "")
    if is_default_title(title):
        return True
    return title in temp_title_variants(session)


def should_create_title(session: Any, turn_number: int) -> bool:
    """Whether the title task should be launched for ``turn_number``.

    Turn 1 replaces a placeholder, preserving an operator title set before
    the first prompt. Turn 10 is an intentional refresh. If the session is
    still on ``New Chat - HHMMSS`` after turn 1 (title LLM timed out under
    load, or the model returned a rejected short title), later completed
    turns retry until a real title sticks — jarvis's second turn is often a
    ``<task_result>`` or ``<delegation_wave>``. Both follow the same title
    lifecycle as any other completed turn.

    A ``title_locked`` session (operator ``/title`` rename, or a Tasks-board
    task whose title is the pinned task name) never fires the title agent —
    this single gate covers turn 1, the unconditional turn-10 refresh, the
    gateway boot reconciler and ``maybe_ensure_title``.
    """
    if getattr(session, "title_locked", False):
        return False
    try:
        number = int(turn_number)
    except (TypeError, ValueError):
        return False
    if number < 1:
        return False
    if number == 10:
        return True
    return is_replaceable_title(session)


def should_ensure_hidden_agent_title(session: Any) -> bool:
    turns = list(getattr(session, "turns", None) or [])
    return should_create_title(session, len(turns))


def load_title_agent_system() -> str:
    """Load the shipped title agent's AGENTS.md as the title-call system prompt.

    Hidden title homes are not a second source of truth. A stale
    ``~/.sharedagents/agents/title/AGENTS.md`` (the old ≤50-character prompt)
    must not override ``profile_agents/title/AGENTS.md``.
    """
    from sharedagents_session.hidden_agent import load_hidden_agent_system

    return load_hidden_agent_system("title")


def build_hidden_agent_title_messages(
    user_text: Any,
    summary: str = "",
) -> list[dict[str, str]]:
    """OpenCode title-agent messages: system=PROMPT_TITLE, user=cue+content.

    Keeps OpenCode's exact cue (``Generate a title for this conversation:``)
    and message shape. ``summary`` is the addition that lets the title
    describe the exchange rather than echo the opening sentence.

    **The title is always in the language of the user's PROMPT** — never a
    fixed language, and never the summary's. Two things enforce that:

    * the summary is written by a different agent and is the only other
      language in this prompt, so it is framed as background and placed
      FIRST; the user's own words come LAST, beside the cue;
    * the rule is then stated explicitly.

    Only the OPENING prompt is passed, deliberately. Feeding the latest turn
    as well made the title follow whatever language the user had drifted
    into, so an English prompt could come back with a Portuguese title. The
    prompt is the anchor. The whole opening prompt is passed — not the first
    line — so a dispatch preamble cannot become the title.
    """
    from sharedagents_session.hidden_agent import build_hidden_agent_title_messages as build

    return build(user_text, summary)


async def ensure_hidden_agent_title(
    session: Any,
    title_caller: TitleCaller,
    *,
    persist: Callable[[], None] | None = None,
) -> bool:
    """Backward-compatible wrapper for the active turn-1 title call."""
    turns = list(getattr(session, "turns", None) or [])
    if len(turns) != HIDDEN_AGENT_TITLE_TURN:
        return False
    user_text = turns[0].get("user") if turns else ""
    return await ensure_title_for_turn(
        session,
        title_caller,
        turn_number=HIDDEN_AGENT_TITLE_TURN,
        user_text=user_text,
        persist=persist,
    )


def first_line(text: Any) -> str:
    """First non-empty line of a string or vision-multipart payload.

    Delegates to :meth:`Session._derive_title` so the extraction (incl.
    the ``[image]`` fallback for caption-less photos and the 160-char
    cap) matches the heuristic title exactly.
    """
    return Session._derive_title(text)


def opening_prompt_text(
    user_text: Any,
    limit: int = TITLE_PROMPT_MAX_CHARS,
) -> str:
    """Full opening-prompt text for the title agent (not first-line).

    Vision multipart payloads contribute every text part, joined with
    newlines. A runaway paste is clipped at ``limit`` so the side call
    stays bounded.
    """
    if isinstance(user_text, list):
        chunks: list[str] = []
        for part in user_text:
            if not isinstance(part, dict):
                continue
            if str(part.get("type") or "").lower() != "text":
                continue
            chunk = str(part.get("text") or "").strip()
            if chunk:
                chunks.append(chunk)
        text = "\n".join(chunks) if chunks else "[image]"
    else:
        text = str(user_text or "").strip()
    if not text:
        return "(empty)"
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def clean_title(raw: Any, limit: int = SESSION_TITLE_MAX_CHARS) -> str:
    """Normalise a model-produced title to a single bare line.

    Strips a leading ``Title:`` style prefix, surrounding quotes,
    markdown emphasis and backticks; keeps only the first non-empty
    line; caps at ``limit``. Returns ``""`` when nothing usable remains
    (the caller treats this as a no-op that does not spend the budget).
    """
    if not isinstance(raw, str):
        raw = str(raw or "")
    # Defence-in-depth: drop any inline reasoning block. The title call asks the
    # model NOT to think (chat_template_kwargs.enable_thinking=False), but a
    # backend that ignores that and emits ``<think>…</think>title`` must still
    # resolve to the title, not the opening ``<think>`` line.
    raw = re.sub(r"<think>.*?</think>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
    # …and an UNCLOSED block (model hit max_tokens before </think>) — strip from
    # the opening tag to end so the title never resolves to a bare "<think>".
    raw = re.sub(r"<think>.*$", " ", raw, flags=re.DOTALL | re.IGNORECASE)
    first = next((ln for ln in raw.strip().splitlines() if ln.strip()), "")
    s = first.strip()
    low = s.lower()
    for prefix in _PREFIXES:
        if low.startswith(prefix):
            s = s[len(prefix) :].strip()
            break
    # Markdown emphasis / code ticks, then any (possibly repeated)
    # surrounding quote pair. Order matters: ``**"x"**`` → ``"x"`` → ``x``.
    s = s.strip("*_`").strip()
    while len(s) >= 2 and s[0] in _QUOTE_CHARS and s[-1] in _QUOTE_CHARS:
        s = s[1:-1].strip()
    words = s.split()
    if len(words) > TITLE_MAX_WORDS:
        s = " ".join(words[:TITLE_MAX_WORDS])
    return s[:limit]


# Unicode script ranges that a Latin-script prompt must never produce a title
# in. Deliberately coarse: this is a wrong-LANGUAGE detector, not a language
# identifier. Distinguishing English from Portuguese by codepoint is hopeless
# (and unnecessary — the model does not confuse them); what it actually does,
# measured at ~1 run in 5 on deepseek-v4-flash, is answer an English prompt in
# Korean. A script check catches exactly that class and nothing else.
_NON_LATIN_RANGES: tuple[tuple[int, int], ...] = (
    (0x0400, 0x04FF),  # Cyrillic
    (0x0590, 0x05FF),  # Hebrew
    (0x0600, 0x06FF),  # Arabic
    (0x0900, 0x097F),  # Devanagari
    (0x0E00, 0x0E7F),  # Thai
    (0x3040, 0x30FF),  # Hiragana + Katakana
    (0x3400, 0x4DBF),  # CJK ext A
    (0x4E00, 0x9FFF),  # CJK unified
    (0xAC00, 0xD7AF),  # Hangul syllables
    (0x1100, 0x11FF),  # Hangul jamo
)


def _has_non_latin_script(text: Any) -> bool:
    """True when the text contains a non-Latin script block."""
    for ch in str(text or ""):
        cp = ord(ch)
        for lo, hi in _NON_LATIN_RANGES:
            if lo <= cp <= hi:
                return True
    return False


def is_language_mismatch(prompt: Any, title: Any) -> bool:
    """True when ``title`` is obviously not in the language of ``prompt``.

    Only fires on a SCRIPT change — a Latin-script prompt answered in Hangul,
    CJK, Cyrillic, … A prompt that itself uses a non-Latin script is never
    flagged, because then that script is the expected answer.

    This exists because the instruction alone is not reliable: with the real
    production prompt, ``deepseek-v4-flash`` returned a Korean title for an
    English prompt in 1 of 5 identical calls.
    """
    if _has_non_latin_script(prompt):
        return False
    return _has_non_latin_script(title)


def _norm(x: Any) -> str:
    s = re.sub(r"\s+", " ", str(x or "").strip().casefold())
    return s.strip(" .!?,;:")


def is_near_identical(a: Any, b: Any) -> bool:
    """True when two titles are the same modulo case/space/punctuation.

    Containment counts as churn too — refining ``"Fix scrollbar"`` to
    ``"Fix scrollbar bug"`` isn't worth spending a refinement on.
    """
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return na == nb
    if na == nb:
        return True
    return na in nb or nb in na


def build_title_messages(
    first_user_line: str,
    latest_user_line: str,
    compaction_summary: str,
    current_title: str,
) -> list[dict[str, str]]:
    """Build the lightest chat prompt that still captures topic drift.

    System prompt is the OpenCode title agent (``title/AGENTS.md``). Context
    (Codex decision #2): the opening request, the latest request, and —
    only when present — the rolling compaction summary. No transcript, no
    tool output.
    """
    system = load_title_agent_system()
    parts = [
        f"Opening request: {first_user_line}".strip(),
        f"Latest request: {latest_user_line}".strip(),
    ]
    if compaction_summary and compaction_summary.strip():
        parts.append("Conversation summary so far:\n" + compaction_summary.strip())
    parts.append(f"Current title: {current_title}")
    parts.append(
        f"Generate a title for this conversation (single line, 3-{TITLE_MAX_WORDS} words, max {TITLE_MAX_WORDS} words):"
    )
    # Same reasoning as the first-shot path: NO explicit language rule here.
    # The system prompt already carries one, and a second louder copy made
    # the model pick a language instead of mirroring the user's. The opening
    # request leads the prompt, and is_language_mismatch backstops the rest.
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def build_session_title_messages(
    user_text: Any,
    *,
    turn_number: int,
    latest_user_text: Any = "",
    summary: str = "",
    current_title: str = "",
) -> list[dict[str, str]]:
    """Build the title prompt for one of the active lifecycle turns.

    The first call is anchored to the opening prompt and can include the
    existing summary as background. The tenth-turn refresh also includes the
    latest request and current title so the model can update the topic without
    receiving the full transcript or tool output.
    """
    if int(turn_number) == 1:
        return build_hidden_agent_title_messages(
            user_text,
            summary=str(summary or ""),
        )
    return build_title_messages(
        opening_prompt_text(user_text),
        opening_prompt_text(latest_user_text or user_text),
        str(summary or ""),
        str(current_title or ""),
    )


def _title_word_count(title: Any) -> int:
    return len(str(title or "").split())


def is_valid_generated_title(title: Any) -> bool:
    """Return whether a cleaned title satisfies the 2–6 word contract."""
    count = _title_word_count(title)
    return TITLE_MIN_WORDS <= count <= TITLE_MAX_WORDS


def repair_generated_title(title: Any) -> str:
    """Expand known one-word model slips into a valid 2–6 word title."""
    candidate = str(title or "").strip()
    if not candidate or is_valid_generated_title(candidate):
        return candidate
    repaired = _SINGLE_WORD_TITLE_REPAIRS.get(candidate.lower())
    if repaired and is_valid_generated_title(repaired):
        return repaired
    return candidate


def fallback_title_from_prompt(user_text: Any) -> str:
    """Last-resort title when the side LLM never produced a valid title."""
    opening = opening_prompt_text(user_text).strip().lower()
    if opening in _GREETING_PROMPTS:
        return "Quick check-in"
    derived = clean_title(
        build_temp_title(user_text),
        limit=HIDDEN_AGENT_TITLE_MAX_CHARS,
    )
    if not derived or is_default_title(derived):
        return ""
    if derived.lower() in ("session", "(empty)", "[image]"):
        return ""
    return derived


async def ensure_title_for_turn(
    session: Any,
    title_caller: TitleCaller,
    *,
    turn_number: int,
    user_text: Any,
    latest_user_text: Any = "",
    summary: str = "",
    persist: Callable[[], None] | None = None,
) -> bool:
    """Generate and persist the title for an active lifecycle turn.

    The caller is injected so scheduling/model selection remain in ``Agent``
    while this function stays deterministic and straightforward to test.
    Invalid model output is ignored; a later lifecycle trigger can try again.
    """
    if not should_create_title(session, turn_number):
        return False
    current = str(getattr(session, "title", "") or "")
    messages = build_session_title_messages(
        user_text,
        turn_number=turn_number,
        latest_user_text=latest_user_text,
        summary=summary,
        current_title=current,
    )

    new_title = ""
    sid = str(getattr(session, "hash", "") or "")[:12]
    for _attempt in range(3):
        if getattr(session, "title_locked", False):
            return False
        try:
            raw = await title_caller(messages)
        except Exception as exc:  # noqa: BLE001 — title is best-effort
            _log.warning(
                "title: caller failed for session %s turn %s: %s",
                sid,
                turn_number,
                exc,
            )
            continue
        # An explicit pin may have arrived while the side call was in flight.
        if getattr(session, "title_locked", False):
            return False
        candidate = repair_generated_title(
            clean_title(raw, limit=HIDDEN_AGENT_TITLE_MAX_CHARS),
        )
        if not candidate or not is_valid_generated_title(candidate):
            _log.warning(
                "title: rejected %r for session %s (words=%s)",
                candidate,
                sid,
                _title_word_count(candidate),
            )
            continue
        if is_language_mismatch(user_text, candidate):
            _log.warning(
                "title: language mismatch %r for session %s",
                candidate,
                sid,
            )
            continue
        new_title = candidate
        break
    if not new_title and is_replaceable_title(session):
        fb = fallback_title_from_prompt(user_text)
        if fb and fb != current and not is_default_title(fb):
            _log.info(
                "title: prompt fallback %r for session %s turn %s",
                fb,
                sid,
                turn_number,
            )
            new_title = fb
    if not new_title or new_title == current:
        return False

    session.title = new_title
    saver = persist if persist is not None else getattr(session, "save", None)
    if callable(saver):
        try:
            saver()
        except Exception:  # noqa: BLE001 — persistence is best-effort
            pass
    return True
