"""WhatsApp channel configuration and on-disk layout.

Everything the channel persists lives under :func:`whatsapp_home`
(``~/.sharedagents/whatsapp`` by default)::

    config.json       operator settings (never secrets — see ``gateway_token``)
    state.json        chat → agent/session bindings, stream cursors, dedupe ids
    neonize.sqlite3   the linked-device credentials (a live WhatsApp Web login)
    whatsapp.pid      single-instance lock
    link-qr.png       the current pairing QR while ``sharedagents whatsapp link`` runs
    outbound.jsonl    self-chat send queue written by the WhatsApp skill
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger("sharedagents.whatsapp")

ENV_HOME = "SHAREDAGENTS_WHATSAPP_HOME"
ENV_GATEWAY_URL = "SHAREDAGENTS_WHATSAPP_GATEWAY_URL"
ENV_GATEWAY_TOKEN = "SHAREDAGENTS_WHATSAPP_GATEWAY_TOKEN"
DELIVERY_MODES = ("blocks", "final")
DEFAULT_GATEWAY_PORT = 8100

_STRING_KEYS = ("default_agent", "reply_prefix", "ack_reaction", "delivery_mode", "gateway_url")
_NUMBER_KEYS = ("max_message_age_s", "max_media_mb")
_SECRET_KEYS = ("gateway_token", "token", "auth_token", "bearer_token")


def _plain_identity_keys(identifier: str) -> set[str]:
    text = (identifier or "").strip()
    if not text:
        return set()
    keys = {text, text.split("@", 1)[0]}
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        keys.update({digits, "+" + digits, f"{digits}@s.whatsapp.net", f"{digits}@lid"})
    return {key for key in keys if key}


def identity_keys(identifier: str, aliases: dict[str, str] | None = None) -> set[str]:
    """Match a WhatsApp person across JID, +phone, and LID↔PN aliases."""
    keys = _plain_identity_keys(identifier)
    if not aliases:
        return keys
    extra: set[str] = set()
    for key in list(keys):
        bare = key.split("@", 1)[0].lstrip("+")
        other = aliases.get(bare) or aliases.get(key)
        if other:
            extra |= _plain_identity_keys(other)
    return keys | extra


def load_lid_aliases(store_path: Path) -> dict[str, str]:
    """``lid ↔ phone`` map from the neonize store, or empty when unreadable."""
    store = Path(store_path)
    if not store.is_file():
        return {}
    try:
        con = sqlite3.connect(f"file:{store}?mode=ro", uri=True, timeout=5)
        try:
            rows = con.execute("SELECT lid, pn FROM whatsmeow_lid_map").fetchall()
        finally:
            con.close()
    except sqlite3.Error:
        return {}
    aliases: dict[str, str] = {}
    for lid, pn in rows:
        left = "".join(ch for ch in str(lid or "") if ch.isdigit())
        right = "".join(ch for ch in str(pn or "") if ch.isdigit())
        if not left or not right:
            continue
        aliases[left] = right
        aliases[right] = left
    return aliases


def whatsapp_home() -> Path:
    """Root of the WhatsApp channel tree.

    Resolution order: ``$SHAREDAGENTS_WHATSAPP_HOME`` → ``$SHAREDAGENTS_HOME/whatsapp`` →
    ``~/.sharedagents/whatsapp`` (the same precedence as ``telegram_home``).
    """
    direct = os.environ.get(ENV_HOME, "").strip()
    if direct:
        return Path(direct).expanduser()
    root = os.environ.get("SHAREDAGENTS_HOME", "").strip()
    if root:
        return Path(root).expanduser() / "whatsapp"
    return Path.home() / ".sharedagents" / "whatsapp"


def _default_gateway_port() -> int:
    """The local gateway's port from the global sharedagents config (8100 if unset)."""
    try:
        from sharedagents_gateway.gateway import _gateway_config

        return int(_gateway_config().get("port") or DEFAULT_GATEWAY_PORT)
    except Exception:  # noqa: BLE001 — a missing/odd global config keeps the default
        return DEFAULT_GATEWAY_PORT


@dataclass
class WhatsAppConfig:
    """Operator settings for the WhatsApp channel (every key optional)."""

    home: Path = field(default_factory=whatsapp_home)
    default_agent: str = "jarvis"
    reply_prefix: str = "🤖"
    ack_reaction: str = "👀"
    delivery_mode: str = "blocks"
    max_message_age_s: float = 900.0
    max_media_mb: float = 16.0
    gateway_url: str = ""
    # People and groups the operator may allow the agent to read or send
    # besides the linked-device self-chat. Each entry is
    # ``{"id", "label", "reply", "mention", "read", "send", "agent", "model"}``.
    allowed_people: list[dict[str, Any]] = field(default_factory=list)
    allowed_groups: list[dict[str, Any]] = field(default_factory=list)
    lid_aliases: dict[str, str] = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, home: Path | None = None) -> WhatsAppConfig:
        """Overlay ``<home>/config.json`` on the defaults; ``ValueError`` on bad values.

        A secret-looking key is refused outright so a gateway token can never be
        parked on disk next to the settings.
        """
        cfg = cls(home=Path(home) if home is not None else whatsapp_home())
        path = cfg.config_path
        if not path.is_file():
            return cfg
        try:
            text = path.read_text(encoding="utf-8") or "{}"
            try:
                data: Any = json.loads(text)
            except json.JSONDecodeError:
                data, _ = json.JSONDecoder().raw_decode(text)
        except ValueError as exc:
            raise ValueError(f"{path}: invalid JSON ({exc})") from exc
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a JSON object")
        leaked = [k for k in _SECRET_KEYS if k in data]
        if leaked:
            raise ValueError(
                f"{path}: {', '.join(leaked)} must not be stored in config.json — "
                f"set ${ENV_GATEWAY_TOKEN} in the environment instead"
            )
        for key in _STRING_KEYS:
            if key in data:
                value = data[key]
                if not isinstance(value, str):
                    raise ValueError(f"{path}: {key} must be a string")
                setattr(cfg, key, value if key == "reply_prefix" else value.strip())
        for key in _NUMBER_KEYS:
            if key in data:
                value = data[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                    raise ValueError(f"{path}: {key} must be a positive number")
                setattr(cfg, key, float(value))
        if not cfg.default_agent:
            raise ValueError(f"{path}: default_agent must not be empty")
        if cfg.delivery_mode not in DELIVERY_MODES:
            raise ValueError(f"{path}: delivery_mode must be one of {', '.join(DELIVERY_MODES)}")
        cfg.allowed_people = _parse_allowlist(data.get("allowed_people"), path, "allowed_people", kind="person")
        cfg.allowed_groups = _parse_allowlist(data.get("allowed_groups"), path, "allowed_groups", kind="group")
        cfg.refresh_lid_aliases()
        return cfg

    @property
    def config_path(self) -> Path:
        return self.home / "config.json"

    @property
    def state_path(self) -> Path:
        return self.home / "state.json"

    @property
    def store_path(self) -> Path:
        return self.home / "neonize.sqlite3"

    @property
    def pid_path(self) -> Path:
        return self.home / "whatsapp.pid"

    @property
    def qr_png_path(self) -> Path:
        return self.home / "link-qr.png"

    @property
    def max_media_bytes(self) -> int:
        return int(self.max_media_mb * 1024 * 1024)

    def resolved_gateway_url(self) -> str:
        """``gateway_url`` → ``$SHAREDAGENTS_WHATSAPP_GATEWAY_URL`` → the local gateway."""
        for candidate in (self.gateway_url, os.environ.get(ENV_GATEWAY_URL, "")):
            if candidate and candidate.strip():
                return candidate.strip().rstrip("/")
        return f"http://127.0.0.1:{_default_gateway_port()}"

    @staticmethod
    def gateway_token() -> str | None:
        """The gateway bearer token — environment only, never config.json."""
        token = os.environ.get(ENV_GATEWAY_TOKEN, "").strip()
        return token or None

    def ensure_home(self) -> Path:
        """Create the channel home as an owner-only (0700) directory."""
        self.home.mkdir(parents=True, exist_ok=True)
        os.chmod(self.home, 0o700)
        return self.home

    def save(self) -> None:
        """Write operator settings atomically. Never persists secret keys."""
        self.ensure_home()
        path = self.config_path
        existing: dict[str, Any] = {}
        if path.is_file():
            try:
                raw: Any = json.loads(path.read_text(encoding="utf-8") or "{}")
            except ValueError:
                raw = {}
            if isinstance(raw, dict):
                existing = {k: v for k, v in raw.items() if k not in _SECRET_KEYS}
        for key in _STRING_KEYS:
            existing[key] = getattr(self, key)
        for key in _NUMBER_KEYS:
            existing[key] = getattr(self, key)
        existing["allowed_people"] = list(self.allowed_people)
        existing["allowed_groups"] = list(self.allowed_groups)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(existing, indent=2, ensure_ascii=False) + "\n")
            os.replace(tmp_name, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise
        os.chmod(path, 0o600)

    def allows(self, identifier: str, *, group: bool = False, access: str = "any") -> bool:
        """True when ``identifier`` is on the people or groups allowlist.

        ``access`` is ``read``, ``send``, or ``any`` (either flag).
        """
        row = self.entry(identifier, group=group)
        if row is None:
            return False
        if access == "read":
            return bool(row.get("read", True))
        if access == "send":
            return bool(row.get("send", True))
        return True

    def refresh_lid_aliases(self) -> None:
        """Reload ``lid ↔ phone`` from the neonize store when the file changes."""
        store = self.store_path
        try:
            mtime = store.stat().st_mtime if store.is_file() else None
        except OSError:
            return
        if mtime == getattr(self, "_lid_mtime", None):
            return
        self.lid_aliases = load_lid_aliases(store) if mtime else {}
        self._lid_mtime = mtime

    def refresh_allowlist(self) -> bool:
        """Re-read the allowlists when config.json changed (Channels tab edits).

        One stat per call; the file is parsed only when its stamp moves. A
        missing or invalid file keeps the lists already in memory.
        """
        try:
            st = self.config_path.stat()
        except OSError:
            return False
        stamp = (st.st_ino, st.st_mtime_ns, st.st_size)
        if stamp in (getattr(self, "_allowlist_stamp", None), getattr(self, "_allowlist_bad", None)):
            return False
        try:
            fresh = type(self).load(self.home)
        except Exception as exc:  # noqa: BLE001 — keep the in-memory list on a bad file
            self._allowlist_bad = stamp
            _log.warning("%s is invalid (%s); keeping the previous allowlist", self.config_path, exc)
            return False
        self.allowed_people, self.allowed_groups = fresh.allowed_people, fresh.allowed_groups
        self._allowlist_stamp = stamp
        _log.info(
            "WhatsApp allowlist reloaded: %d people, %d groups", len(self.allowed_people), len(self.allowed_groups)
        )
        return True

    def entry(self, identifier: str, *, group: bool = False) -> dict[str, Any] | None:
        ident = (identifier or "").strip()
        if not ident:
            return None
        rows = self.allowed_groups if group else self.allowed_people
        wanted = identity_keys(ident, self.lid_aliases)
        for row in rows:
            if identity_keys(str(row.get("id") or ""), self.lid_aliases) & wanted:
                return row
        return None

    def routed_agent(self, identifier: str) -> str:
        """Allowlist agent for this chat, or empty to use ``state`` / default."""
        for group in (False, True):
            row = self.entry(identifier, group=group)
            if not row:
                continue
            name = str(row.get("agent") or "").strip()
            if name:
                return name
        return ""

    def routed_model(self, identifier: str) -> str:
        """Allowlist model for this chat, or empty to use the agent default.

        The model belongs to the chat route just like ``agent`` does.  It is
        deliberately resolved here rather than read from a caller's cached
        allowlist row, so a Channels-tab edit takes effect on the next inbound
        message without restarting the WhatsApp service.
        """
        for group in (False, True):
            row = self.entry(identifier, group=group)
            if not row:
                continue
            model = str(row.get("model") or "").strip()
            if model:
                return model
        return ""


def _parse_allowlist(raw: Any, path: Path, key: str, *, kind: str) -> list[dict[str, str]]:
    """Normalize ``[{id, label, reply, mention}]`` or bare-string lists."""
    from sharedagents_core.reply_policy import normalize_route, normalize_rule

    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{path}: {key} must be a list")
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw:
        ident = ""
        label = ""
        reply = None
        mention = None
        read = None
        send = None
        agent = None
        model = None
        if isinstance(item, str):
            ident = item.strip()
        elif isinstance(item, dict):
            ident = str(item.get("id") or item.get("identifier") or "").strip()
            label = str(item.get("label") or "").strip()
            reply = item.get("reply")
            mention = item.get("mention")
            read = item.get("read")
            send = item.get("send")
            agent = item.get("agent")
            model = item.get("model")
        else:
            raise ValueError(f"{path}: {key} entries must be strings or objects")
        if not ident or ident in seen:
            continue
        seen.add(ident)
        rule = normalize_rule(reply, mention, kind=kind, read=read, send=send)
        out.append({"id": ident, "label": label, **rule, **normalize_route(agent, model)})
    return out
