"""Telegram forum-topic name registry.

The Bot API cannot list forum topics; Telethon (user MTProto) can. We persist
the fetched ``{chat_id: {thread_id: title}}`` map to
``<telegram_home>/forum_topics.json`` and share it with:

* the Telegram bot (seeds ``_topic_names`` on cold start), and
* the WebUI Topics page (resolves ``topic-<id>`` fallbacks + lists every
  known forum topic, even when no session is bound yet).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sharedagents_telegram.config import telegram_home

FORUM_TOPICS_FILE = "forum_topics.json"
SESSIONS_FILE = "sessions.json"
_TOPIC_ID_RE = re.compile(r"^topic-(\d+)$", re.IGNORECASE)

TOPIC_ALIASES_FILE = "topic_aliases.json"


def legacy_topic_aliases(*, path: Path | None = None) -> dict[str, str]:
    """Operator-provided slug → forum-title aliases.

    Old sessions can carry a ``topic`` slug that no longer matches any live
    forum title (e.g. an AI-backfilled name from before forum sync). Operators
    who need such slugs remapped place ``topic_aliases.json`` in the telegram
    home — a flat ``{"slug": "title"}`` map. Missing/invalid file means no
    aliases.
    """
    p = path or (telegram_home() / TOPIC_ALIASES_FILE)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if str(k).strip() and str(v).strip()}


def forum_topics_path() -> Path:
    return telegram_home() / FORUM_TOPICS_FILE


def _empty_registry() -> dict[str, Any]:
    return {"updated_at": "", "chats": {}}


def load_forum_topics(*, path: Path | None = None) -> dict[str, Any]:
    """Return the on-disk registry, or an empty shell when missing/invalid."""
    p = path or forum_topics_path()
    if not p.is_file():
        return _empty_registry()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_registry()
    if not isinstance(data, dict):
        return _empty_registry()
    chats = data.get("chats")
    if not isinstance(chats, dict):
        data["chats"] = {}
    return data


def save_forum_topics(
    chats: dict[str, dict[str, str]],
    *,
    path: Path | None = None,
) -> Path:
    """Atomically write ``{chat_id: {thread_id: title}}``."""
    p = path or forum_topics_path()
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "chats": {
            str(chat_id): {str(tid): str(title) for tid, title in topics.items()} for chat_id, topics in chats.items()
        },
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, p)
    return p


def chat_topic_map(
    registry: dict[str, Any],
    chat_id: int | str,
) -> dict[int, str]:
    """``{thread_id: title}`` for one forum supergroup."""
    raw = (registry.get("chats") or {}).get(str(chat_id)) or {}
    if not isinstance(raw, dict):
        return {}
    out: dict[int, str] = {}
    for tid, title in raw.items():
        try:
            out[int(tid)] = str(title)
        except (TypeError, ValueError):
            continue
    return out


def all_canonical_names(registry: dict[str, Any]) -> list[str]:
    """Every distinct forum topic title across all chats."""
    seen: set[str] = set()
    names: list[str] = []
    for chat_topics in (registry.get("chats") or {}).values():
        if not isinstance(chat_topics, dict):
            continue
        for title in chat_topics.values():
            t = str(title).strip()
            if not t:
                continue
            key = t.casefold()
            if key in seen:
                continue
            seen.add(key)
            names.append(t)
    return sorted(names, key=lambda s: s.casefold())


def resolve_topic_name(
    raw: str,
    registry: dict[str, Any] | None = None,
) -> str:
    """Map a session ``topic`` string to the best-known forum title.

    * ``topic-1842`` → ``test`` when the registry knows thread 1842
    * ``general`` → ``General`` when the registry lists that title
    * otherwise returns the trimmed input (or ``general``)
    """
    topic = (raw or "").strip() or "general"
    reg = registry if registry is not None else load_forum_topics()
    m = _TOPIC_ID_RE.match(topic)
    if m:
        tid = int(m.group(1))
        for chat_topics in (reg.get("chats") or {}).values():
            if not isinstance(chat_topics, dict):
                continue
            hit = chat_topics.get(str(tid))
            if hit:
                return str(hit)
    fold = topic.casefold()
    for name in all_canonical_names(reg):
        if name.casefold() == fold:
            return name
    aliases = legacy_topic_aliases()
    alias = aliases.get(topic)
    if alias is None:
        folded = {k.casefold(): v for k, v in aliases.items()}
        alias = folded.get(fold)
    if alias:
        return alias
    return topic


def load_session_topic_bindings(
    registry: dict[str, Any] | None = None,
    *,
    path: Path | None = None,
) -> dict[str, str]:
    """Map gateway session hash → forum topic title via ``sessions.json``."""
    reg = registry if registry is not None else load_forum_topics()
    p = path or (telegram_home() / SESSIONS_FILE)
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, str] = {}
    for key, agents in data.items():
        if not isinstance(agents, dict) or ":" not in str(key):
            continue
        chat_id, tid_s = str(key).split(":", 1)
        try:
            tid = int(tid_s)
        except ValueError:
            continue
        title = chat_topic_map(reg, chat_id).get(tid)
        if not title:
            continue
        for sid in agents.values():
            if isinstance(sid, str) and sid.strip():
                out[sid.strip()] = title
    return out


def resolve_session_topic(
    session_id: str,
    raw: str,
    registry: dict[str, Any] | None = None,
    bindings: dict[str, str] | None = None,
) -> str:
    """Resolve a session row to a forum topic title.

    Priority: Telegram ``sessions.json`` binding → ``resolve_topic_name`` →
    fold unknown slugs to ``General`` when a registry is loaded.
    """
    reg = registry if registry is not None else load_forum_topics()
    sid = (session_id or "").strip()
    bind = bindings if bindings is not None else load_session_topic_bindings(reg)
    if sid and sid in bind:
        return bind[sid]
    resolved = resolve_topic_name(raw, reg)
    if reg.get("chats"):
        canon = {n.casefold(): n for n in all_canonical_names(reg)}
        if resolved.casefold() not in canon:
            stored = (raw or "").strip()
            # Keep an explicit web-local topic. Only fold unknown Telegram
            # slugs (topic-<id>) and empty/general onto the forum General.
            if stored and stored.casefold() not in {"", "general"} and not _TOPIC_ID_RE.match(stored):
                return stored
            return canon.get("general", "General")
    return resolved


def hydrate_topic_names(
    registry: dict[str, Any],
) -> dict[str, dict[int, str]]:
    """Shape consumed by ``TelegramBot._topic_names``."""
    out: dict[str, dict[int, str]] = {}
    for chat_id, topics in (registry.get("chats") or {}).items():
        if not isinstance(topics, dict):
            continue
        bucket: dict[int, str] = {}
        for tid, title in topics.items():
            try:
                bucket[int(tid)] = str(title)
            except (TypeError, ValueError):
                continue
        if bucket:
            out[str(chat_id)] = bucket
    return out


def _load_telethon_creds() -> tuple[int, str]:
    env_path = telegram_home() / "botapi.env"
    if env_path.is_file():
        for raw in env_path.read_text().splitlines():
            s = raw.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            if s.startswith("export "):
                s = s[7:].lstrip()
            k, v = s.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    api_id = os.environ.get("TELEGRAM_API_ID") or os.environ.get("TELETHON_API_ID")
    api_hash = os.environ.get("TELEGRAM_API_HASH") or os.environ.get("TELETHON_API_HASH")
    if not api_id or not api_hash:
        raise RuntimeError(
            "TELEGRAM_API_ID / TELEGRAM_API_HASH not set (check botapi.env)",
        )
    return int(api_id), api_hash


async def fetch_forum_topics_async(
    chat_id: int,
    *,
    session_file: Path | None = None,
) -> dict[int, str]:
    """Pull every forum topic title for ``chat_id`` via Telethon."""
    from telethon import TelegramClient
    from telethon.sessions import StringSession
    from telethon.tl.functions.messages import GetForumTopicsRequest

    api_id, api_hash = _load_telethon_creds()
    sess_path = session_file or (telegram_home() / "telethon.session")
    if not sess_path.is_file():
        raise RuntimeError(f"Telethon session missing: {sess_path}")
    session = StringSession(sess_path.read_text().strip())

    topics: dict[int, str] = {}
    offset_date, offset_id, offset_topic = 0, 0, 0
    total = 0
    async with TelegramClient(session, api_id, api_hash) as client:
        while True:
            r = await client(
                GetForumTopicsRequest(
                    peer=chat_id,
                    offset_date=offset_date,
                    offset_id=offset_id,
                    offset_topic=offset_topic,
                    limit=100,
                )
            )
            if not total:
                total = r.count
            if not r.topics:
                break
            for t in r.topics:
                title = getattr(t, "title", None)
                if title:
                    topics[int(t.id)] = str(title)
            last = r.topics[-1]
            offset_topic, offset_id = last.id, last.top_message
            offset_date = next((m.date for m in r.messages if m.id == offset_id), 0)
            if len(topics) >= total:
                break
    return topics


async def refresh_forum_topics_async(
    chat_ids: list[int],
    *,
    session_file: Path | None = None,
) -> dict[str, Any]:
    """Fetch + persist forum topics for each chat id."""
    existing = load_forum_topics()
    chats: dict[str, dict[str, str]] = dict(existing.get("chats") or {})
    for cid in chat_ids:
        fetched = await fetch_forum_topics_async(cid, session_file=session_file)
        chats[str(cid)] = {str(tid): title for tid, title in fetched.items()}
    save_forum_topics(chats)
    return load_forum_topics()


def refresh_forum_topics(chat_ids: list[int]) -> dict[str, Any]:
    return asyncio.run(refresh_forum_topics_async(chat_ids))


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Refresh Telegram forum topic names")
    p.add_argument(
        "chat_ids",
        nargs="+",
        type=int,
        help="Forum supergroup id(s)",
    )
    args = p.parse_args()
    reg = refresh_forum_topics(args.chat_ids)
    n = sum(len(v) for v in (reg.get("chats") or {}).values())
    print(f"Wrote {n} topics to {forum_topics_path()}")


if __name__ == "__main__":
    _cli()
