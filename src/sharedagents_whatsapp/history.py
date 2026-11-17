"""Bounded per-chat inbound history for allowlisted WhatsApp people/groups.

The last ``HISTORY_LIMIT`` messages of each selected chat are kept on disk so a
restart can replay the latest unreplied prompt without asking the operator to
resend it.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from sharedagents_whatsapp.config import identity_keys

HISTORY_LIMIT = 100
HISTORY_NAME = "history.json"


def history_path(home: Path) -> Path:
    return Path(home) / HISTORY_NAME


def canonical_chat_key(
    ident: str,
    aliases: dict[str, str] | None = None,
    *,
    group: bool = False,
) -> str:
    text = (ident or "").strip()
    if not text:
        return ""
    if group or text.endswith("@g.us"):
        user = text.split("@", 1)[0]
        return f"{user}@g.us" if user else text
    keys = identity_keys(text, aliases)
    digits = [key for key in keys if key.isdigit()]
    if not digits:
        return text
    return min(digits, key=len)


def load_history(home: Path) -> dict[str, list[dict[str, Any]]]:
    path = history_path(home)
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8") or "{}")
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for key, items in raw.items():
        if not isinstance(items, list):
            continue
        kept = [item for item in items if isinstance(item, dict) and item.get("id")]
        if kept:
            out[str(key)] = kept[-HISTORY_LIMIT:]
    return out


def _write_history(home: Path, data: dict[str, list[dict[str, Any]]]) -> None:
    path = history_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)


def record(
    home: Path,
    ident: str,
    rec: dict[str, Any],
    aliases: dict[str, str] | None = None,
    *,
    group: bool = False,
) -> None:
    """Append ``rec`` (or replace the same ``id``) and keep the last 100."""
    key = canonical_chat_key(ident, aliases, group=group)
    msg_id = str(rec.get("id") or "").strip()
    if not key or not msg_id:
        return
    data = load_history(home)
    items = [item for item in data.get(key, []) if str(item.get("id") or "") != msg_id]
    row = {
        "id": msg_id,
        "chat": str(rec.get("chat") or ident),
        "sender": str(rec.get("sender") or ""),
        "text": str(rec.get("text") or ""),
        "kind": str(rec.get("kind") or "text"),
        "ts": float(rec.get("ts") or 0),
        "from_me": bool(rec.get("from_me")),
        "from_own_device": bool(rec.get("from_own_device")),
        "is_self_chat": bool(rec.get("is_self_chat")),
        "is_group": bool(rec.get("is_group", group)),
        "replied": bool(rec.get("replied")),
    }
    items.append(row)
    data[key] = items[-HISTORY_LIMIT:]
    _write_history(home, data)


def mark_replied(
    home: Path,
    ident: str,
    msg_id: str,
    aliases: dict[str, str] | None = None,
    *,
    group: bool = False,
) -> None:
    key = canonical_chat_key(ident, aliases, group=group)
    want = str(msg_id or "").strip()
    if not key or not want:
        return
    data = load_history(home)
    items = data.get(key) or []
    changed = False
    for item in items:
        if str(item.get("id") or "") == want and not item.get("replied"):
            item["replied"] = True
            changed = True
    if changed:
        data[key] = items
        _write_history(home, data)


def last_unreplied(home: Path) -> list[dict[str, Any]]:
    """The newest unreplied message of each stored chat (if any)."""
    out: list[dict[str, Any]] = []
    for items in load_history(home).values():
        if not items:
            continue
        last = items[-1]
        if not last.get("replied"):
            out.append(last)
    return out
