"""WhatsApp skill: mutate Channels allowlist settings (admin only).

The linked WhatsApp self-chat, a web/TUI session, or a Telegram trusted
chat may change reply/access/agent/model rows. Allowlisted contacts cannot.
"""

from __future__ import annotations

import json
import os
import shlex
from typing import Any

from sharedagents_whatsapp.state import SELF_CHAT


def caller_is_admin(session_id: str | None = None) -> bool:
    """True when this turn is the operator, not an allowlisted contact."""
    sid = (session_id if session_id is not None else os.environ.get("SHAREDAGENTS_SESSION_ID") or "").strip()
    if not sid:
        return False
    wa_chat = _whatsapp_chat_for_session(sid)
    if wa_chat is not None:
        return wa_chat == SELF_CHAT
    tg_chat = _telegram_chat_for_session(sid)
    if tg_chat is not None:
        return _telegram_trusted(tg_chat)
    return True


def run_allowlist_args(raw: str, *, session_id: str | None = None) -> str:
    """Handle ``allowlist`` / ``allow`` / ``update`` / ``revoke`` skill args."""
    tokens = shlex.split(raw or "")
    if not tokens:
        return ""
    verb = tokens[0].lower()
    if verb in {"allowlist", "list-allow", "list-allowlist"}:
        if not caller_is_admin(session_id):
            return _refuse()
        return _format_snapshot()
    if verb in {"allow", "add-allow", "allowlist-add"}:
        if not caller_is_admin(session_id):
            return _refuse()
        return _mutate("add", tokens[1:])
    if verb in {"update", "patch-allow", "allowlist-update"}:
        if not caller_is_admin(session_id):
            return _refuse()
        return _mutate("update", tokens[1:])
    if verb in {"revoke", "remove-allow", "allowlist-revoke"}:
        if not caller_is_admin(session_id):
            return _refuse()
        return _revoke(tokens[1:])
    return ""


def _refuse() -> str:
    return (
        "## Channels allowlist\n"
        "Refused. Only the admin (WhatsApp self-chat, web/TUI operator, "
        "or a Telegram trusted chat) can change these settings.\n"
    )


def _whatsapp_chat_for_session(sid: str) -> str | None:
    try:
        from sharedagents_whatsapp.config import WhatsAppConfig, whatsapp_home
        from sharedagents_whatsapp.state import ChannelState
    except Exception:
        return None
    cfg = WhatsAppConfig.load()
    state = ChannelState.load(whatsapp_home() / "state.json", cfg.default_agent)
    for chat, _agent, bound in state.all_bindings():
        if bound == sid:
            return chat
    return None


def _telegram_chat_for_session(sid: str) -> str | None:
    try:
        from sharedagents_telegram.bot_helpers import _load_acp_session_map
        from sharedagents_telegram.config import TelegramConfig
    except Exception:
        return None
    cfg = TelegramConfig.load()
    for chat_key, agents in _load_acp_session_map(cfg).items():
        if sid in (agents or {}).values():
            return chat_key
    return None


def _telegram_trusted(chat_key: str) -> bool:
    try:
        from sharedagents_telegram.config import TelegramConfig
    except Exception:
        return False
    cfg = TelegramConfig.load()
    try:
        chat_id = int(str(chat_key).split(":")[0])
    except ValueError:
        return False
    return chat_id in (cfg.trusted_chat_ids or [])


def _parse_kv(tokens: list[str]) -> tuple[list[str], dict[str, str]]:
    positional: list[str] = []
    fields: dict[str, str] = {}
    for tok in tokens:
        if "=" in tok:
            key, value = tok.split("=", 1)
            fields[key.strip().lower()] = value.strip()
        else:
            positional.append(tok)
    return positional, fields


def _flags(fields: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "label" in fields:
        out["label"] = fields["label"]
    if "reply" in fields:
        out["reply"] = fields["reply"]
    if "mention" in fields:
        out["mention"] = fields["mention"]
    if "agent" in fields:
        out["agent"] = fields["agent"]
    if "model" in fields:
        out["model"] = fields["model"]
    access = fields.get("access") or fields.get("mode")
    if access:
        mode = access.strip().lower().replace(" ", "")
        out["read"] = True
        out["send"] = mode not in {"readonly", "read-only", "ro"}
    return out


def _mutate(action: str, tokens: list[str]) -> str:
    from sharedagents_web import channel_allowlist

    positional, fields = _parse_kv(tokens)
    if len(positional) < 3:
        return (
            "## Channels allowlist\n"
            "Need `channel kind id`. Example: `allow whatsapp person +5511… "
            "label=Ana reply=mention access=readwrite agent=jarvis`.\n"
        )
    channel, kind, ident = positional[0], positional[1], positional[2]
    kwargs = _flags(fields)
    try:
        if action == "add":
            channel_allowlist.add(channel, kind, ident, **kwargs)
        else:
            channel_allowlist.update(channel, kind, ident, **kwargs)
    except ValueError as exc:
        return f"## Channels allowlist\n{exc}\n"
    return f"## Channels allowlist\nUpdated **{channel}** {kind} `{ident}`.\n\n" + _format_snapshot()


def _revoke(tokens: list[str]) -> str:
    from sharedagents_web import channel_allowlist

    positional, _fields = _parse_kv(tokens)
    if len(positional) < 3:
        return "## Channels allowlist\nNeed `revoke channel kind id`.\n"
    channel, kind, ident = positional[0], positional[1], positional[2]
    try:
        channel_allowlist.remove(channel, kind, ident)
    except ValueError as exc:
        return f"## Channels allowlist\n{exc}\n"
    return f"## Channels allowlist\nRemoved **{channel}** {kind} `{ident}`.\n\n" + _format_snapshot()


def _format_snapshot() -> str:
    from sharedagents_web import channel_allowlist

    snap = channel_allowlist.snapshot()
    rows: list[dict[str, Any]] = []
    for ch in snap.get("channels") or []:
        for kind in ("people", "groups"):
            for row in ch.get(kind) or []:
                rows.append(
                    {
                        "channel": ch.get("id"),
                        "kind": "person" if kind == "people" else "group",
                        "id": row.get("id"),
                        "label": row.get("label") or "",
                        "reply": row.get("reply"),
                        "mention": row.get("mention") or "",
                        "access": "read/write" if row.get("send") else "readonly",
                        "agent": row.get("agent") or "",
                        "model": row.get("model") or "",
                    }
                )
    return "## Channels allowlist\n```json\n" + json.dumps(rows, indent=2, ensure_ascii=False) + "\n```\n"
