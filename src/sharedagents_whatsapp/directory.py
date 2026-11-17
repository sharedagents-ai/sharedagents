"""Read-only WhatsApp people/groups directory from the neonize store.

Used by the operator Channels tab. The agent skill must not call this —
v1 still refuses contact dumps on the skill path.

The store may be open by ``sharedagents whatsapp run``; we only connect
read-only so the live channel keeps the write lock.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from sharedagents_whatsapp.config import WhatsAppConfig
from sharedagents_whatsapp.neonize_transport import linked_account


def list_directory(*, home: Path | None = None) -> dict[str, Any]:
    """``{people, groups, linked}`` from ``neonize.sqlite3``, or empty lists."""
    cfg = WhatsAppConfig.load(home)
    account = linked_account(cfg.store_path)
    if account is None:
        return {"people": [], "groups": [], "linked": False}
    own = str(account.get("user") or "")
    try:
        con = sqlite3.connect(f"file:{cfg.store_path}?mode=ro", uri=True, timeout=5)
        try:
            people = _people(con, own)
            groups = _groups(con)
        finally:
            con.close()
    except sqlite3.Error:
        return {"people": [], "groups": [], "linked": True}
    return {"people": people, "groups": groups, "linked": True}


def _label(*parts: Any) -> str:
    for part in parts:
        text = str(part or "").strip()
        if text:
            return text
    return ""


def _people(con: sqlite3.Connection, own: str) -> list[dict[str, str]]:
    try:
        rows = con.execute(
            "SELECT their_jid, first_name, full_name, push_name, business_name FROM whatsmeow_contacts"
        ).fetchall()
    except sqlite3.Error:
        return []
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for their_jid, first_name, full_name, push_name, business_name in rows:
        jid = str(their_jid or "").strip()
        if not jid or "@s.whatsapp.net" not in jid:
            continue
        user = jid.split("@", 1)[0].split(":", 1)[0]
        if not user or user == own or user in seen:
            continue
        seen.add(user)
        name = _label(full_name, first_name, business_name, push_name)
        out.append(
            {
                "id": jid,
                "label": name,
                "phone": f"+{user}" if user.isdigit() else user,
            }
        )
    out.sort(key=lambda row: (row["label"].casefold() or "zzz", row["id"]))
    return out


def _groups(con: sqlite3.Connection) -> list[dict[str, str]]:
    names: dict[str, str] = {}
    try:
        for their_jid, first_name, full_name, push_name, business_name in con.execute(
            "SELECT their_jid, first_name, full_name, push_name, business_name "
            "FROM whatsmeow_contacts WHERE their_jid LIKE '%@g.us'"
        ):
            jid = str(their_jid or "").strip()
            if jid:
                names[jid] = _label(full_name, first_name, business_name, push_name)
    except sqlite3.Error:
        pass
    jids: list[str] = []
    try:
        for (chat_jid,) in con.execute(
            "SELECT DISTINCT chat_jid FROM whatsmeow_chat_settings WHERE chat_jid LIKE '%@g.us'"
        ):
            jid = str(chat_jid or "").strip()
            if jid:
                jids.append(jid)
    except sqlite3.Error:
        pass
    for jid in names:
        if jid not in jids:
            jids.append(jid)
    out = []
    for jid in jids:
        out.append({"id": jid, "label": names.get(jid, ""), "phone": ""})
    out.sort(key=lambda row: (row["label"].casefold() or "zzz", row["id"]))
    return out
