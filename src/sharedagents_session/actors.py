"""Per-turn attribution for multi-user WebUI sessions.

An actor is the authenticated WebUI user who typed a prompt:
``{"email": <login, lowercased>, "name": <display name>, "via": "web"}``,
or the agent that delegated it to a child session:
``{"email": "", "name": <agent name>, "via": "agent"}``.
It is provenance only — never an access decision. Access is decided by
``sharedagents_web.access`` (roles, grants and sharing);
this module only records and renders WHO authored a turn. Stdlib-only leaf so the
web, gateway, agent and session layers can all import it.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from typing import Any

ACTOR_VIA_WEB = "web"
ACTOR_VIA_AGENT = "agent"
TURN_ACTOR_KIND = "turn_actor"
ACTOR_META_KEY = "sharedagents_actor"
MAX_EMAIL_LEN = 254
MAX_NAME_LEN = 64


def _strip_controls(text: str) -> str:
    """Drop control/format characters (incl. bidi overrides) — spoofing guard."""
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")


def clean_display_name(name: Any) -> str:
    """Printable, single-line, at most ``MAX_NAME_LEN`` characters ('' if nothing is left)."""
    text = " ".join(_strip_controls(str(name or "")).split())
    return text[:MAX_NAME_LEN].strip()


def display_name_fallback(email: str) -> str:
    """Email local part (``user@example.com`` → ``user``); the address if it has none."""
    email = str(email or "").strip()
    return clean_display_name(email.split("@", 1)[0]) or clean_display_name(email)


def agent_actor(name: Any) -> dict[str, str] | None:
    """The delegating agent as the author of a child turn, or ``None``."""
    clean = clean_display_name(name)
    if not clean:
        return None
    return {"email": "", "name": clean, "via": ACTOR_VIA_AGENT}


def actor_key(actor: Mapping[str, str]) -> str:
    """Identity of a normalized actor: the login, or the agent name."""
    return actor["email"] or f"{ACTOR_VIA_AGENT}:{actor['name']}"


def normalize_actor(raw: Any) -> dict[str, str] | None:
    """Validated ``{email, name, via}`` (fixed key order) or ``None``.

    An agent actor has no login: it needs ``via == "agent"`` and a name.
    """
    if not isinstance(raw, Mapping):
        return None
    if not raw.get("email") and clean_display_name(raw.get("via")) == ACTOR_VIA_AGENT:
        return agent_actor(raw.get("name"))
    email = _strip_controls(str(raw.get("email") or "")).strip().lower()
    if not email or len(email) > MAX_EMAIL_LEN or any(ch.isspace() for ch in email):
        return None
    name = clean_display_name(raw.get("name")) or display_name_fallback(email)
    via = clean_display_name(raw.get("via")) or ACTOR_VIA_WEB
    return {"email": email, "name": name, "via": via}


def normalize_actors(raw: Any) -> list[dict[str, str]]:
    """List (or one actor) → validated, deduplicated, order-preserving list."""
    if isinstance(raw, Mapping):
        items: list[Any] = [raw]
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        items = []
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in items:
        actor = normalize_actor(item)
        if actor is None or actor_key(actor) in seen:
            continue
        seen.add(actor_key(actor))
        out.append(actor)
    return out


def with_actor_meta(block: Mapping[str, Any], actor: Mapping[str, str]) -> dict[str, Any]:
    """Copy of an ACP content block carrying ``actor`` in its ``_meta`` slot.

    ``_meta`` survives ``acp.schema.PromptRequest`` serialization (a top-level
    ``_sharedagents_*`` key does not) and every prompt flattener ignores it, so the
    actor crosses gateway→agent hops without ever becoming model text.
    """
    out = dict(block)
    meta = dict(out["_meta"]) if isinstance(out.get("_meta"), Mapping) else {}
    meta[ACTOR_META_KEY] = dict(actor)
    out["_meta"] = meta
    return out


def actor_from_blocks(blocks: Any) -> dict[str, str] | None:
    """First valid actor in any block's ``_meta`` slot, else ``None``."""
    for block in blocks or []:
        if not isinstance(block, Mapping):
            continue
        meta = block.get("_meta")
        if isinstance(meta, Mapping):
            actor = normalize_actor(meta.get(ACTOR_META_KEY))
            if actor is not None:
                return actor
    return None


def actor_index_payload(actor: Mapping[str, str]) -> dict[str, str]:
    """Exact ``turn_events.payload_json`` body of a ``turn_actor`` row.

    One construction site with a fixed key order, so the copies projected from
    ``turn_started`` and ``turn_ended`` hash identically and the UNIQUE
    (session_hash, turn_idx, kind, payload_hash) index keeps one row.
    """
    return {"email": actor["email"], "name": actor["name"], "via": actor["via"]}
