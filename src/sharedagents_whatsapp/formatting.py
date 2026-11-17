"""Agent markdown → WhatsApp markup, split into fence-safe message chunks.

WhatsApp's markup is ``*bold*``, ``_italic_``, ``~strike~`` and backtick code; it
has no headings or link syntax, and very long messages fold behind "Read more".
Code spans and fenced blocks pass through untouched (fences lose their language
tag, which WhatsApp would print literally).
"""

from __future__ import annotations

import re

MAX_CHUNK = 3800
_HEADER_ROOM = 16  # room for "🤖 [12/12] " so prefixed parts stay under the limit

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$")
_BULLET_RE = re.compile(r"^(\s*)[*+]\s+")
_INLINE_CODE_RE = re.compile(r"(`+)(.+?)\1")
_LINK_RE = re.compile(r"!?\[([^\]\n]*)\]\(<?([^()\s<>]+)>?\)")
_BOLD_RE = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")
# ``__bold__`` only as a standalone token, never inside an identifier (my_func__name__).
_UBOLD_RE = re.compile(r"(?<!\w)__(?=[^\s_])(.+?)(?<=[^\s_])__(?!\w)")
_ITALIC_RE = re.compile(r"(?<![*\w])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![*\w])")
_STRIKE_RE = re.compile(r"~~(?=\S)(.+?)(?<=\S)~~")
_FENCE_LINE_RE = re.compile(r"^\s*```", re.M)
_BOLD_MARK = "\x00"
_TASK_ENVELOPE_RE = re.compile(
    r"<(task_result|task_error|delegation_wave)\b[^>]*>(.*?)</\1>",
    re.DOTALL | re.IGNORECASE,
)
_UNTRUSTED_RE = re.compile(
    r"(?:⟦|〔|\[)UNTRUSTED-DATA\b[^\]⟧〕\n]*?(?:⟧|〕|\])"
    r"(.*?)"
    r"(?:⟦|〔|\[)/UNTRUSTED-DATA\b[^\]⟧〕\n]*?(?:⟧|〕|\])",
    re.DOTALL | re.IGNORECASE,
)
_RESUME_NOTE_RE = re.compile(
    r"(?:Incorporate this completed task and continue\.|"
    r"The delegated task did not complete\.[^\n]*|"
    r"All children in this delegation wave are terminal\.[^\n]*)",
    re.IGNORECASE,
)
_QUOTED_INTERNAL_RE = re.compile(
    r"^>\s*(?:<(?:task_result|task_error|delegation_wave)\b|\[UNTRUSTED-DATA|⟦UNTRUSTED-DATA|"
    r"\[MEMORY[ _]FLUSH).*\n\n",
    re.IGNORECASE | re.DOTALL,
)
_INTERNAL_HEAD = (
    "<task_result",
    "<task_error",
    "<delegation_wave",
    "[untrusted-data",
    "⟦untrusted-data",
    "[memory flush]",
    "[memory_flush",
    "(memory_flush)",
)


def is_memory_flush_text(text: str) -> bool:
    """True for the host's end-of-session memory-flush prompt or its ack."""
    head = (text or "").lstrip()
    if not head:
        return False
    upper = head[:24].upper()
    return upper.startswith("[MEMORY FLUSH]") or upper.startswith("[MEMORY_FLUSH") or upper.startswith("(MEMORY_FLUSH)")


def is_memory_flush_event(event: dict) -> bool:
    """True when a session event is the silent memory-flush maintenance turn."""
    if str(event.get("source") or "") == "memory_flush":
        return True
    user = event.get("user")
    if isinstance(user, str):
        return is_memory_flush_text(user)
    if isinstance(user, list):
        joined = "\n".join(str(block.get("text") or "") for block in user if isinstance(block, dict))
        return is_memory_flush_text(joined)
    return False


def is_internal_prompt(text: str) -> bool:
    """True for a parent-resume envelope or memory flush, not a person's question."""
    head = (text or "").lstrip().lower()
    if not head:
        return False
    return head.startswith(_INTERNAL_HEAD) or head.startswith("incorporate this completed task")


def strip_internal_envelopes(text: str) -> str:
    """Drop task/wave / UNTRUSTED-DATA / memory-flush wrappers."""
    if not text:
        return text
    if is_memory_flush_text(text):
        return ""
    cleaned = _QUOTED_INTERNAL_RE.sub("", text)
    if is_memory_flush_text(cleaned):
        return ""
    cleaned = _TASK_ENVELOPE_RE.sub(lambda m: (m.group(2) or "").strip(), cleaned)
    cleaned = _UNTRUSTED_RE.sub(lambda m: (m.group(1) or "").strip(), cleaned)
    cleaned = _RESUME_NOTE_RE.sub("", cleaned)
    return cleaned.strip()


def to_whatsapp(markdown: str) -> str:
    """Convert agent markdown to WhatsApp markup, leaving code untouched."""
    out: list[str] = []
    in_fence = False
    for line in (markdown or "").split("\n"):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            out.append("```")
            continue
        out.append(line if in_fence else _convert_line(line))
    return "\n".join(out)


def _convert_line(line: str) -> str:
    heading = _HEADING_RE.match(line)
    if heading:
        title = heading.group(1).replace("**", "").replace("__", "").strip()
        return f"*{title}*" if title else ""
    line = _BULLET_RE.sub(r"\1- ", line)
    pieces: list[str] = []
    pos = 0
    for code in _INLINE_CODE_RE.finditer(line):
        pieces.append(_convert_inline(line[pos : code.start()]))
        pieces.append(code.group(0))
        pos = code.end()
    pieces.append(_convert_inline(line[pos:]))
    return "".join(pieces)


def _convert_inline(text: str) -> str:
    if not text:
        return text
    text = _LINK_RE.sub(_link, text)
    text = _BOLD_RE.sub(lambda m: f"{_BOLD_MARK}{m.group(1)}{_BOLD_MARK}", text)
    text = _UBOLD_RE.sub(lambda m: f"{_BOLD_MARK}{m.group(1)}{_BOLD_MARK}", text)
    text = _ITALIC_RE.sub(r"_\1_", text)
    text = _STRIKE_RE.sub(r"~\1~", text)
    return text.replace(_BOLD_MARK, "*")


def _link(match: re.Match) -> str:
    label, url = match.group(1).strip(), match.group(2)
    if not label or label == url:
        return url
    return f"{label} ({url})"


def chunk_text(text: str, limit: int = MAX_CHUNK) -> list[str]:
    """Split ``text`` into chunks of at most ``limit`` characters.

    Prefers paragraph, then line, then sentence, then word boundaries. A chunk
    that ends inside a fenced block is closed with a fence and the next chunk
    reopens it, so every chunk renders correctly on its own.
    """
    if limit < 64:
        raise ValueError("chunk limit too small")
    rest = (text or "").strip()
    chunks: list[str] = []
    in_fence = False
    while rest:
        head = "```\n" if in_fence else ""
        budget = limit - len(head) - len("\n```")
        if len(rest) <= budget:
            piece, rest = rest, ""
        else:
            cut = _cut_point(rest, budget)
            piece, rest = rest[:cut], rest[cut:]
        toggles = len(_FENCE_LINE_RE.findall(piece))
        ends_inside = in_fence ^ (toggles % 2 == 1)
        body = (head + piece).rstrip()
        if ends_inside and rest:
            body += "\n```"
        if body.strip() and body.strip() != "```":
            chunks.append(body)
        in_fence = ends_inside
        if not in_fence:
            rest = rest.lstrip("\n")
    return chunks


def _cut_point(text: str, budget: int) -> int:
    window = text[:budget]
    floor = budget // 2
    for sep in ("\n\n", "\n", ". ", " "):
        idx = window.rfind(sep)
        if idx >= floor:
            return idx + len(sep)
    return budget


def render_messages(markdown: str, prefix: str = "", limit: int = MAX_CHUNK) -> list[str]:
    """Convert, chunk and prefix one reply into ready-to-send WhatsApp texts."""
    chunks = chunk_text(to_whatsapp(markdown), max(64, limit - _HEADER_ROOM))
    total = len(chunks)
    out: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        header = " ".join(p for p in (prefix, f"[{i}/{total}]" if total > 1 else "") if p)
        if not header:
            out.append(chunk)
        elif chunk.startswith("```"):
            out.append(f"{header}\n{chunk}")
        else:
            out.append(f"{header} {chunk}")
    return out
