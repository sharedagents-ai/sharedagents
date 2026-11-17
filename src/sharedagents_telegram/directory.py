"""Known Telegram people and groups for the Channels tab.

Telegram bots have no address book. We list chats the bot has already
seen (sessions + forum cache) plus any allowlisted ids.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sharedagents_telegram.config import TelegramConfig
from sharedagents_telegram.forum_topics import load_forum_topics


def list_directory() -> dict[str, Any]:
    cfg = TelegramConfig.load()
    labels = dict(cfg.allowlist_labels or {})
    people: dict[str, dict[str, str]] = {}
    groups: dict[str, dict[str, str]] = {}

    for uid in cfg.allowed_user_ids:
        ident = str(uid)
        people[ident] = {
            "id": ident,
            "label": labels.get(f"person:{ident}", ""),
            "phone": "",
        }
    for cid in cfg.allowed_chat_ids:
        ident = str(cid)
        bucket = groups if _looks_group(cid) else people
        key = "group" if bucket is groups else "person"
        bucket[ident] = {
            "id": ident,
            "label": labels.get(f"{key}:{ident}", ""),
            "phone": "",
        }

    for chat_id, title in _session_chats(cfg).items():
        ident = str(chat_id)
        if _looks_group(chat_id):
            groups.setdefault(ident, {"id": ident, "label": title, "phone": ""})
            if title and not groups[ident]["label"]:
                groups[ident]["label"] = title
        else:
            people.setdefault(ident, {"id": ident, "label": title, "phone": ""})
            if title and not people[ident]["label"]:
                people[ident]["label"] = title

    registry = load_forum_topics()
    for raw_id, topics in (registry.get("chats") or {}).items():
        try:
            chat_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        ident = str(chat_id)
        title = ""
        if isinstance(topics, dict):
            for value in topics.values():
                text = str(value or "").strip()
                if text:
                    title = text
                    break
        groups.setdefault(ident, {"id": ident, "label": title, "phone": ""})
        if title and not groups[ident]["label"]:
            groups[ident]["label"] = title

    def _sort(rows: dict[str, dict[str, str]]) -> list[dict[str, str]]:
        return sorted(
            rows.values(),
            key=lambda row: (row["label"].casefold() or "zzz", row["id"]),
        )

    return {
        "people": _sort(people),
        "groups": _sort(groups),
        "linked": bool(cfg.bot_token),
    }


def _looks_group(chat_id: int) -> bool:
    return int(chat_id) < 0


def _session_chats(cfg: TelegramConfig) -> dict[int, str]:
    path: Path = cfg.sessions_path
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8") or "{}")
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[int, str] = {}
    for key in data:
        raw = str(key).split(":", 1)[0].strip()
        try:
            out[int(raw)] = ""
        except (TypeError, ValueError):
            continue
    return out
