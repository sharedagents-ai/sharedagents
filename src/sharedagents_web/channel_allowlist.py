"""Telegram + WhatsApp send-allowlist CRUD used by the Channels tab.

Telegram persists ``allowed_user_ids`` (people) and ``allowed_chat_ids``
(groups/chats). WhatsApp persists ``allowed_people`` / ``allowed_groups``
on ``whatsapp/config.json``. Self-chat stays always allowed on WhatsApp.
"""

from __future__ import annotations

import re
from typing import Any, Literal

ChannelId = Literal["telegram", "whatsapp"]
Kind = Literal["person", "group"]

CHANNELS: tuple[ChannelId, ...] = ("telegram", "whatsapp")
KINDS: tuple[Kind, ...] = ("person", "group")

_TELEGRAM_NOTES = (
    "This is a bot, not you. Search a contact or group to add it. "
    "Access is readonly or read/write. Restart the Telegram bot after saving."
)
_WHATSAPP_NOTES = (
    "Self-chat is always allowed. Search a contact or group to add it. "
    "Access is readonly or read/write. Generate a QR here to link this host."
)


def snapshot() -> dict[str, Any]:
    agents = _agent_choices()
    return {
        "channels": [_telegram_payload(), _whatsapp_payload()],
        "agents": agents,
        "models": _model_ids(agents),
    }


def add(
    channel: str,
    kind: str,
    identifier: str,
    label: str = "",
    *,
    reply: str | None = None,
    mention: str | None = None,
    read: bool | None = None,
    send: bool | None = None,
    agent: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    channel_id, kind_id = _require_ids(channel, kind)
    ident = _normalize_id(channel_id, identifier, kind=kind_id)
    name = (label or "").strip()
    if channel_id == "telegram":
        _telegram_upsert(
            kind_id,
            ident,
            name,
            replace_id=None,
            reply=reply,
            mention=mention,
            read=read,
            send=send,
            agent=agent,
            model=model,
        )
    else:
        _whatsapp_upsert(
            kind_id,
            ident,
            name,
            replace_id=None,
            reply=reply,
            mention=mention,
            read=read,
            send=send,
            agent=agent,
            model=model,
        )
    return snapshot()


def update(
    channel: str,
    kind: str,
    identifier: str,
    *,
    label: str | None = None,
    new_id: str | None = None,
    reply: str | None = None,
    mention: str | None = None,
    read: bool | None = None,
    send: bool | None = None,
    agent: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    channel_id, kind_id = _require_ids(channel, kind)
    ident = _normalize_id(channel_id, identifier)
    next_id = _normalize_id(
        channel_id,
        new_id if new_id is not None else ident,
        kind=kind_id,
    )
    if channel_id == "telegram":
        _telegram_upsert(
            kind_id,
            next_id,
            label,
            replace_id=ident,
            reply=reply,
            mention=mention,
            read=read,
            send=send,
            agent=agent,
            model=model,
        )
    else:
        _whatsapp_upsert(
            kind_id,
            next_id,
            label,
            replace_id=ident,
            reply=reply,
            mention=mention,
            read=read,
            send=send,
            agent=agent,
            model=model,
        )
    return snapshot()


def remove(channel: str, kind: str, identifier: str) -> dict[str, Any]:
    channel_id, kind_id = _require_ids(channel, kind)
    ident = _normalize_id(channel_id, identifier)
    if channel_id == "telegram":
        _telegram_remove(kind_id, ident)
    else:
        _whatsapp_remove(kind_id, ident)
    return snapshot()


def _require_ids(channel: str, kind: str) -> tuple[ChannelId, Kind]:
    if channel not in CHANNELS:
        raise ValueError(f"unknown channel: {channel}")
    if kind not in KINDS:
        raise ValueError(f"unknown kind: {kind}")
    return channel, kind  # type: ignore[return-value]


def _normalize_id(channel: ChannelId, identifier: str, *, kind: Kind | None = None) -> str:
    ident = (identifier or "").strip()
    if not ident:
        raise ValueError("id is required")
    if channel == "telegram":
        try:
            return str(int(ident))
        except (TypeError, ValueError) as exc:
            raise ValueError("Telegram ids must be integers (user id or chat id)") from exc
    if kind == "group":
        if re.fullmatch(r"\d+(?:-\d+)?@g\.us", ident):
            return ident
        from sharedagents_whatsapp.directory import list_directory

        matches = [
            row["id"] for row in list_directory()["groups"] if row.get("label", "").casefold() == ident.casefold()
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError("Multiple WhatsApp groups have that name. Select a group from the search results.")
        raise ValueError("Select a WhatsApp group from the search results or enter its numeric @g.us ID.")
    if any(ch.isspace() for ch in ident):
        raise ValueError("WhatsApp ids cannot contain whitespace")
    return ident


def _label_key(kind: Kind, ident: str) -> str:
    return f"{kind}:{ident}"


def _telegram_payload() -> dict[str, Any]:
    from sharedagents_core.reply_policy import normalize_rule
    from sharedagents_telegram.config import TelegramConfig
    from sharedagents_telegram.directory import list_directory

    cfg = TelegramConfig.load()
    directory = list_directory()
    allowed_people = []
    seen_people: set[str] = set()
    for uid in cfg.allowed_user_ids:
        ident = str(uid)
        seen_people.add(ident)
        rule = cfg.allowlist_rules.get(_label_key("person", ident)) or {}
        allowed_people.append(
            {
                "id": ident,
                "label": cfg.allowlist_labels.get(_label_key("person", ident), ""),
                **normalize_rule(
                    rule.get("reply"),
                    rule.get("mention"),
                    kind="person",
                    read=rule.get("read"),
                    send=rule.get("send"),
                ),
                **_row_route(rule, ident, cfg.agent_routes, cfg.default_agent),
            }
        )
    allowed_groups = []
    for cid in cfg.allowed_chat_ids:
        ident = str(cid)
        if int(cid) >= 0:
            if ident in seen_people:
                continue
            rule = cfg.allowlist_rules.get(_label_key("person", ident)) or {}
            allowed_people.append(
                {
                    "id": ident,
                    "label": cfg.allowlist_labels.get(_label_key("person", ident), ""),
                    **normalize_rule(
                        rule.get("reply"),
                        rule.get("mention"),
                        kind="person",
                        read=rule.get("read"),
                        send=rule.get("send"),
                    ),
                    **_row_route(rule, ident, cfg.agent_routes, cfg.default_agent),
                }
            )
            continue
        rule = cfg.allowlist_rules.get(_label_key("group", ident)) or {}
        allowed_groups.append(
            {
                "id": ident,
                "label": cfg.allowlist_labels.get(_label_key("group", ident), ""),
                **normalize_rule(
                    rule.get("reply"),
                    rule.get("mention"),
                    kind="group",
                    read=rule.get("read"),
                    send=rule.get("send"),
                ),
                **_row_route(rule, ident, cfg.agent_routes, cfg.default_agent),
            }
        )
    people = _whitelist_rows(allowed_people, directory.get("people") or [], "person")
    groups = _whitelist_rows(allowed_groups, directory.get("groups") or [], "group")
    return {
        "id": "telegram",
        "name": "Telegram",
        "people": people,
        "groups": groups,
        "suggest_people": _suggestions(directory.get("people") or [], people),
        "suggest_groups": _suggestions(directory.get("groups") or [], groups),
        "linked": bool(directory.get("linked")),
        "default_agent": cfg.default_agent or "jarvis",
        "notes": _TELEGRAM_NOTES,
    }


def _whatsapp_payload() -> dict[str, Any]:
    from sharedagents_whatsapp.config import WhatsAppConfig
    from sharedagents_whatsapp.directory import list_directory
    from sharedagents_whatsapp.neonize_transport import linked_account, mask_user

    cfg = WhatsAppConfig.load()
    directory = list_directory(home=cfg.home)
    people = _whitelist_rows(list(cfg.allowed_people), directory.get("people") or [], "person")
    groups = _whitelist_rows(list(cfg.allowed_groups), directory.get("groups") or [], "group")
    acct = linked_account(cfg.store_path)
    return {
        "id": "whatsapp",
        "name": "WhatsApp",
        "people": people,
        "groups": groups,
        "suggest_people": _suggestions(directory.get("people") or [], people),
        "suggest_groups": _suggestions(directory.get("groups") or [], groups),
        "linked": bool(directory.get("linked")),
        "account": mask_user(acct["user"]) if acct else "",
        "default_agent": cfg.default_agent or "jarvis",
        "notes": _WHATSAPP_NOTES,
    }


def _id_keys(ident: str) -> set[str]:
    text = str(ident or "").strip()
    keys = {text}
    if not text:
        return keys
    bare = text.split("@", 1)[0]
    keys.add(bare)
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        keys.add(digits)
        keys.add("+" + digits)
    return {k for k in keys if k}


def _same_identity(left: str, right: str) -> bool:
    return bool(_id_keys(left) & _id_keys(right))


def _whitelist_rows(
    allow_rows: list[dict[str, str]],
    directory_rows: list[dict[str, str]],
    kind: str,
) -> list[dict[str, Any]]:
    from sharedagents_core.reply_policy import normalize_rule

    out: list[dict[str, Any]] = []
    for row in allow_rows:
        ident = str(row.get("id") or "").strip()
        if not ident:
            continue
        label = str(row.get("label") or "").strip()
        phone = ""
        for item in directory_rows:
            if not _same_identity(ident, str(item.get("id") or "")):
                continue
            label = label or str(item.get("label") or "").strip()
            phone = str(item.get("phone") or "")
            break
        rule = normalize_rule(
            row.get("reply"),
            row.get("mention"),
            kind=kind,
            read=row.get("read"),
            send=row.get("send"),
        )
        from sharedagents_core.reply_policy import normalize_route

        out.append(
            {
                "id": ident,
                "label": label,
                "phone": phone,
                "allowed": True,
                "source": "allowlist",
                **rule,
                **normalize_route(row.get("agent"), row.get("model")),
            }
        )
    out.sort(key=lambda row: ((row["label"] or "").casefold() or "zzz", row["id"]))
    return out


def _suggestions(
    directory_rows: list[dict[str, str]],
    allow_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in directory_rows:
        ident = str(row.get("id") or "").strip()
        if not ident or any(_same_identity(ident, str(item.get("id") or "")) for item in allow_rows):
            continue
        out.append(
            {
                "id": ident,
                "label": str(row.get("label") or "").strip(),
                "phone": str(row.get("phone") or ""),
            }
        )
    return out


def _telegram_upsert(
    kind: Kind,
    ident: str,
    label: str | None,
    replace_id: str | None,
    reply: str | None = None,
    mention: str | None = None,
    read: bool | None = None,
    send: bool | None = None,
    agent: str | None = None,
    model: str | None = None,
) -> None:
    from sharedagents_core.reply_policy import normalize_route, normalize_rule
    from sharedagents_telegram.config import TelegramConfig

    cfg = TelegramConfig.load()
    value = int(ident)
    old = int(replace_id) if replace_id is not None else None
    lists = [cfg.allowed_chat_ids]
    if kind == "person":
        lists.append(cfg.allowed_user_ids)
    for ids in lists:
        if old is not None and old in ids:
            ids[:] = [value if item == old else item for item in ids]
        elif value not in ids:
            ids.append(value)
    if old is not None:
        old_key = _label_key(kind, str(old))
        cfg.allowlist_labels.pop(old_key, None)
        moved = cfg.allowlist_rules.pop(old_key, None)
        if moved:
            if reply is None:
                reply = moved.get("reply")
            if mention is None:
                mention = moved.get("mention")
            if read is None:
                read = moved.get("read")
            if send is None:
                send = moved.get("send")
            if agent is None:
                agent = moved.get("agent")
            if model is None:
                model = moved.get("model")
        old_route = cfg.agent_routes.pop(str(old), None)
        if agent is None and old_route:
            agent = old_route
    key = _label_key(kind, ident)
    if label is not None:
        if label.strip():
            cfg.allowlist_labels[key] = label.strip()
        else:
            cfg.allowlist_labels.pop(key, None)
    current = cfg.allowlist_rules.get(key) or {}
    cfg.allowlist_rules[key] = {
        **normalize_rule(
            reply if reply is not None else current.get("reply"),
            mention if mention is not None else current.get("mention"),
            kind=kind,
            read=read if read is not None else current.get("read"),
            send=send if send is not None else current.get("send"),
        ),
        **normalize_route(
            agent if agent is not None else current.get("agent"),
            model if model is not None else current.get("model"),
        ),
    }
    _sync_telegram_route(cfg, ident, cfg.allowlist_rules[key].get("agent"))
    cfg.save()


def _telegram_remove(kind: Kind, ident: str) -> None:
    from sharedagents_telegram.config import TelegramConfig

    cfg = TelegramConfig.load()
    value = int(ident)
    for ids in (cfg.allowed_user_ids, cfg.allowed_chat_ids):
        if value in ids:
            ids.remove(value)
    cfg.allowlist_labels.pop(_label_key(kind, ident), None)
    cfg.allowlist_rules.pop(_label_key(kind, ident), None)
    cfg.agent_routes.pop(ident, None)
    cfg.save()


def _whatsapp_rows(cfg: Any, kind: Kind) -> list[dict[str, str]]:
    return cfg.allowed_people if kind == "person" else cfg.allowed_groups


def _whatsapp_upsert(
    kind: Kind,
    ident: str,
    label: str | None,
    replace_id: str | None,
    reply: str | None = None,
    mention: str | None = None,
    read: bool | None = None,
    send: bool | None = None,
    agent: str | None = None,
    model: str | None = None,
) -> None:
    from sharedagents_core.reply_policy import normalize_route, normalize_rule
    from sharedagents_whatsapp.config import WhatsAppConfig

    cfg = WhatsAppConfig.load()
    rows = _whatsapp_rows(cfg, kind)
    found = None
    for row in rows:
        if row.get("id") == (replace_id or ident) or _same_identity(str(row.get("id") or ""), ident):
            found = row
            break
    if found is None:
        found = {"id": ident, "label": ""}
        rows.append(found)
    found["id"] = ident
    if label is not None:
        found["label"] = label.strip()
    found.update(
        normalize_rule(
            reply if reply is not None else found.get("reply"),
            mention if mention is not None else found.get("mention"),
            kind=kind,
            read=read if read is not None else found.get("read"),
            send=send if send is not None else found.get("send"),
        )
    )
    found.update(
        normalize_route(
            agent if agent is not None else found.get("agent"),
            model if model is not None else found.get("model"),
        )
    )
    cfg.save()
    _sync_whatsapp_agent(cfg, ident, found.get("agent"))


def _whatsapp_remove(kind: Kind, ident: str) -> None:
    from sharedagents_whatsapp.config import WhatsAppConfig

    cfg = WhatsAppConfig.load()
    rows = _whatsapp_rows(cfg, kind)
    rows[:] = [row for row in rows if not _same_identity(str(row.get("id") or ""), ident)]
    cfg.save()


def _row_route(
    rule: dict[str, Any],
    ident: str,
    routes: dict[str, str],
    default_agent: str,
) -> dict[str, str]:
    from sharedagents_core.reply_policy import normalize_route

    route = normalize_route(rule.get("agent"), rule.get("model"))
    if not route["agent"]:
        route["agent"] = str(routes.get(ident) or default_agent or "").strip()
    return route


def _sync_telegram_route(cfg: Any, ident: str, agent: str | None) -> None:
    name = str(agent or "").strip()
    if name:
        cfg.agent_routes[str(ident)] = name
    else:
        cfg.agent_routes.pop(str(ident), None)


def _sync_whatsapp_agent(cfg: Any, ident: str, agent: str | None) -> None:
    name = str(agent or "").strip()
    if not name:
        return
    try:
        from sharedagents_whatsapp.state import ChannelState

        state = ChannelState.load(cfg.state_path, cfg.default_agent)
        state.set_agent(ident, name)
        state.save()
    except Exception:
        return


def _agent_choices() -> list[dict[str, str]]:
    names: list[str] = []
    try:
        from sharedagents_agent.agent import list_agents

        names = list(list_agents())
    except Exception:
        names = []
    if not names:
        names = ["jarvis"]
    out: list[dict[str, str]] = []
    for name in names:
        model = ""
        models: list[str] = []
        try:
            from sharedagents_agent.agent import AgentConfig

            cfg = AgentConfig.load(name)
            row = cfg._primary_model_row()
            model = str(row.get("id") or row.get("alias") or cfg.upstream_model or "").strip()
            for item in cfg.models or []:
                mid = str(item.get("id") or item.get("alias") or "").strip()
                if mid and mid not in models:
                    models.append(mid)
        except Exception:
            pass
        out.append({"name": name, "model": model, "models": models})
    return out


def _model_ids(agents: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for row in agents:
        for mid in [row.get("model"), *(row.get("models") or [])]:
            name = str(mid or "").strip()
            if name and name not in seen:
                seen.append(name)
    return seen
