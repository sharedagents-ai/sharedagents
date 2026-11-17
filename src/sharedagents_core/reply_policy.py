"""Per-chat reply triggers for Telegram and WhatsApp allowlists.

Modes:
- ``any`` — reply to every message in that chat
- ``to_me`` — only DMs, or messages that reply to / address the bot
- ``mention`` — the message must start with one of the chat's ``@x`` tokens
  (space-separated); ``@bot`` only when none is set
"""

from __future__ import annotations

import unicodedata
from typing import Any

REPLY_ANY = "any"
REPLY_TO_ME = "to_me"
REPLY_MENTION = "mention"
MODES = (REPLY_ANY, REPLY_TO_ME, REPLY_MENTION)


def _as_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("0", "false", "no", "off"):
        return False
    if text in ("1", "true", "yes", "on"):
        return True
    return default


def normalize_access(read: Any = None, send: Any = None) -> dict[str, bool]:
    """Per-row Read / Send flags. Missing values default to both on."""
    return {"read": _as_bool(read, True), "send": _as_bool(send, True)}


def normalize_rule(
    reply: Any = None,
    mention: Any = None,
    *,
    kind: str = "person",
    read: Any = None,
    send: Any = None,
) -> dict[str, Any]:
    mode = str(reply or "").strip().lower()
    if mode not in MODES:
        mode = REPLY_ANY if kind == "person" else REPLY_MENTION
    # A 1:1 person chat is always "to this account"; that option is groups-only.
    if kind == "person" and mode == REPLY_TO_ME:
        mode = REPLY_ANY
    parts: list[str] = []
    for part in str(mention or "").split():
        tok = part if part.startswith("@") else "@" + part
        if tok.lower() not in {p.lower() for p in parts}:
            parts.append(tok)
    token = " ".join(parts)
    if mode != REPLY_MENTION:
        token = ""
    return {"reply": mode, "mention": token, **normalize_access(read, send)}


def normalize_route(agent: Any = None, model: Any = None) -> dict[str, str]:
    """Per-row agent + model that should reply. Empty means channel default."""
    return {
        "agent": str(agent or "").strip(),
        "model": str(model or "").strip(),
    }


def mention_tokens(rule: dict[str, str], *, bot_username: str = "") -> list[str]:
    """Tokens that trigger a mention-mode reply.

    The mention field may list several tokens separated by spaces; when it
    does, only those count (a chat set to ``@helper`` ignores ``@bot``).
    Without one, ``@bot`` and the Telegram bot username are the tokens.
    """
    tokens: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        tok = str(raw or "").strip()
        if not tok:
            return
        if not tok.startswith("@"):
            tok = "@" + tok
        key = tok.lower()
        if key in seen:
            return
        seen.add(key)
        tokens.append(tok)

    for part in str(rule.get("mention") or "").split():
        _add(part)
    if tokens:
        return tokens
    if bot_username:
        _add(bot_username)
    _add("@bot")
    return tokens


def text_starts_with_mention(text: str, tokens: list[str]) -> bool:
    """True when the message opens with one of ``tokens``.

    A token quoted mid-sentence ("start with @x and it replies") is talk
    about the bot, not a call to it; ``@xy`` does not count as ``@x``.
    """
    lower = (text or "").lstrip().lower()
    for token in tokens:
        tok = token.lower()
        if tok and lower.startswith(tok):
            rest = lower[len(tok) :]
            if not rest:
                return True
            category = unicodedata.category(rest[0])
            # Combining marks and connector punctuation continue a Unicode
            # identifier just as letters, digits and ASCII underscore do.
            if not (rest[0].isalnum() or category.startswith("M") or category == "Pc"):
                return True
    return False


def should_reply(
    rule: dict[str, str],
    *,
    is_private: bool,
    addressed_to_me: bool,
    text: str,
    bot_username: str = "",
) -> bool:
    mode = str(rule.get("reply") or "").strip().lower()
    if mode == REPLY_ANY:
        return True
    if mode == REPLY_TO_ME:
        return bool(is_private or addressed_to_me)
    tokens = mention_tokens(rule, bot_username=bot_username)
    return text_starts_with_mention(text, tokens)
