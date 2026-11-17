"""Telegram-gateway on-disk config.

Independent of any single agent. The bot is now its own service with
its own home directory, mirroring how each agent has one. Layout:

    ~/.sharedagents/telegram/
        config.json       — bot_token, allowed_chat_ids, default_agent
        sessions.json     — chat→{agent,session_id} bindings (Phase B)
        offset.json       — getUpdates ack cursor
        logs/
            telegram.log
        telegram.pid      — written while the service is running
        telegram.spawn.lock

Resolution honors ``$SHAREDAGENTS_HOME`` (workspace root) so the test
sandbox at ``~/.sharedagents-test/telegram/`` works without per-test
patching. ``$SHAREDAGENTS_TELEGRAM_HOME`` overrides that for one-off runs.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def telegram_home() -> Path:
    """Top of the Telegram-gateway tree.

    Resolution order:
      1. ``$SHAREDAGENTS_TELEGRAM_HOME`` — exact directory.
      2. ``$SHAREDAGENTS_HOME`` (workspace root) → ``<root>/telegram``.
      3. ``~/.sharedagents/telegram``.
    """
    direct = os.environ.get("SHAREDAGENTS_TELEGRAM_HOME")
    if direct:
        return Path(direct).expanduser()
    root = os.environ.get("SHAREDAGENTS_HOME")
    if root:
        return Path(root).expanduser() / "telegram"
    return Path.home() / ".sharedagents" / "telegram"


def _config_path() -> Path:
    return telegram_home() / "config.json"


def _retired_main_should_heal() -> bool:
    """True when a Telegram default/route named ``main`` should heal to ``jarvis``.

    ``main`` heals ONLY when there is no LIVE CUSTOM operator agent by that name —
    mirroring the gateway-boot quarantine (``_retire_managed_main_agent``), which
    removes the retired *managed* orchestrator but preserves a genuinely custom
    agent named ``main``:

    - ``agents/main`` absent  → the retired built-in is gone → heal.
    - ``agents/main`` config carries the retired managed marker
      (``agent_profile == "main"``, pending/at quarantine) → heal.
    - ``agents/main`` is a custom operator agent (any other / no ``agent_profile``)
      → PRESERVE, so Telegram keeps routing to it.
    - unreadable / malformed ``main`` home → conservatively PRESERVE (don't rewrite
      routing toward an agent whose managed status we can't confirm).
    """
    try:
        from sharedagents_agent.agent import agents_root

        raw = (agents_root() / "main" / "config.json").read_text(encoding="utf-8")
    except FileNotFoundError:
        return True  # no main home → retired built-in removed → heal
    except Exception:  # noqa: BLE001
        return False  # unreadable → don't touch its routing
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return False  # malformed custom home → preserve
    if not isinstance(data, dict):
        return False
    return str(data.get("agent_profile") or "").strip() == "main"


@dataclass
class TelegramConfig:
    """Bot-wide configuration. Lives in
    ``<telegram_home>/config.json``; per-(chat,agent) state lives in
    sibling files (``sessions.json`` / ``offset.json``) so a config
    edit doesn't risk corrupting in-flight runtime state.
    """

    bot_token: str = ""
    # Bot-wide allowlist. An empty list denies every chat (refuses to
    # boot loud rather than open to the internet).
    allowed_chat_ids: list[int] = field(default_factory=list)
    # Trusted-chats get fs.readTextFile; everyone else stays text-only.
    # Mirrors the agent-side env var, hoisted into config so a single
    # bot deployment doesn't need a per-process env var.
    trusted_chat_ids: list[int] = field(default_factory=list)
    # User-id allowlist. When non-empty, the bot only replies to
    # messages whose ``message.from.id`` is in this set, even if the
    # surrounding chat is in ``allowed_chat_ids``. Useful for shared
    # groups where the bot should ignore everyone except the operator.
    # Empty (default) ⇒ disabled — every member of an allowed chat
    # can address the bot, matching the legacy behavior.
    allowed_user_ids: list[int] = field(default_factory=list)
    # Group-mention gate. When True, in non-private chats (groups,
    # supergroups, channels) the bot only replies to messages that
    # explicitly ``@<botname>`` it (slash commands' implicit
    # ``/cmd@<botname>`` suffix counts) or that are direct replies
    # to one of the bot's own messages. DMs (chat.type == "private")
    # are unaffected — there's no other recipient there to confuse,
    # so requiring a mention would just be noise. Default False to
    # preserve drop-in upgrade behavior.
    mention_required_in_groups: bool = False
    # Which agent gets a brand-new chat's first message.
    default_agent: str = "jarvis"
    # Static (chat_id|chat_id:thread_id) → agent_name map. Overrides
    # ``default_agent`` for chats / forum topics that should be pinned
    # to a specific agent (e.g. one Telegram group per agent, or one
    # forum topic per agent). Empty in fresh installs; the wizard
    # writes entries when the operator sets up a multi-agent layout.
    # The dynamic ``/agent <name>`` command (Phase B) writes here too,
    # so a chat's last selection survives bot restarts.
    agent_routes: dict[str, str] = field(default_factory=dict)
    # Optional display names for Channels-tab allowlist rows.
    # Keys are ``person:<user_id>`` or ``group:<chat_id>``.
    allowlist_labels: dict[str, str] = field(default_factory=dict)
    # Per-row reply + access: ``person:<id>`` / ``group:<id>`` →
    # ``{"reply", "mention", "read", "send", "agent", "model"}``.
    allowlist_rules: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls) -> TelegramConfig:
        """Read the on-disk config. Missing file → defaults (empty
        token, no allowlist, agent=jarvis). Callers are expected to
        check ``bot_token`` before booting the long-poll loop.
        """
        p = _config_path()
        if not p.is_file():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        if not isinstance(data, dict):
            return cls()
        token = str(data.get("bot_token") or "")
        allowed: list[int] = []
        for x in data.get("allowed_chat_ids") or []:
            try:
                allowed.append(int(x))
            except (TypeError, ValueError):
                continue
        trusted: list[int] = []
        for x in data.get("trusted_chat_ids") or []:
            try:
                trusted.append(int(x))
            except (TypeError, ValueError):
                continue
        allowed_users: list[int] = []
        for x in data.get("allowed_user_ids") or []:
            try:
                allowed_users.append(int(x))
            except (TypeError, ValueError):
                continue
        mention_required = bool(data.get("mention_required_in_groups") or False)
        # ``main`` was the generated default before jarvis became the sole
        # orchestrator.  Heal that retired built-in name while loading an old
        # install — but ONLY when there is no LIVE CUSTOM operator agent named
        # ``main`` (the gateway-boot quarantine deliberately preserves such an
        # agent, so Telegram must keep routing to it too).  Arbitrary operator
        # agent names are always left intact.
        heal_main = _retired_main_should_heal()
        default_agent = str(data.get("default_agent") or "jarvis") or "jarvis"
        if default_agent == "main" and heal_main:
            default_agent = "jarvis"
        routes_raw = data.get("agent_routes") or {}
        routes: dict[str, str] = {}
        if isinstance(routes_raw, dict):
            for k, v in routes_raw.items():
                if not isinstance(k, str) or not isinstance(v, str):
                    continue
                if not k.strip() or not v.strip():
                    continue
                route_agent = v.strip()
                routes[k.strip()] = "jarvis" if (route_agent == "main" and heal_main) else route_agent
        labels_raw = data.get("allowlist_labels") or {}
        labels: dict[str, str] = {}
        if isinstance(labels_raw, dict):
            for k, v in labels_raw.items():
                if not isinstance(k, str) or not isinstance(v, str):
                    continue
                key = k.strip()
                if key:
                    labels[key] = v.strip()
        from sharedagents_core.reply_policy import normalize_route, normalize_rule

        rules_raw = data.get("allowlist_rules") or {}
        rules: dict[str, dict[str, str]] = {}
        if isinstance(rules_raw, dict):
            for k, v in rules_raw.items():
                if not isinstance(k, str) or not isinstance(v, dict):
                    continue
                key = k.strip()
                if not key:
                    continue
                kind = "group" if key.startswith("group:") else "person"
                rules[key] = {
                    **normalize_rule(
                        v.get("reply"),
                        v.get("mention"),
                        kind=kind,
                        read=v.get("read"),
                        send=v.get("send"),
                    ),
                    **normalize_route(v.get("agent"), v.get("model")),
                }
        return cls(
            bot_token=token,
            allowed_chat_ids=allowed,
            trusted_chat_ids=trusted,
            allowed_user_ids=allowed_users,
            mention_required_in_groups=mention_required,
            default_agent=default_agent,
            agent_routes=routes,
            allowlist_labels=labels,
            allowlist_rules=rules,
        )

    # ---- routing -------------------------------------------------------

    @staticmethod
    def chat_key(chat_id: int, thread_id: int | None = None) -> str:
        """Compose the persistence key for a (chat, topic) pair.

        Forum topics are keyed ``"<chat_id>:<thread_id>"``; non-forum
        chats and the General topic (no ``message_thread_id``) keep
        the bare chat-id key, matching the Telegram bot's existing
        on-disk session-map convention.
        """
        if thread_id is None:
            return str(chat_id)
        return f"{chat_id}:{thread_id}"

    def agent_for(
        self,
        chat_id: int,
        thread_id: int | None = None,
    ) -> str:
        """Resolve which agent name handles ``(chat_id, thread_id)``.

        Order: exact ``chat_id:thread_id`` match → bare ``chat_id``
        match → ``default_agent``. Returns a name only — the caller
        loads the matching :class:`AgentConfig`.
        """
        if thread_id is not None:
            specific = f"{chat_id}:{thread_id}"
            if specific in self.agent_routes:
                return self.agent_routes[specific]
        bare = str(chat_id)
        if bare in self.agent_routes:
            return self.agent_routes[bare]
        return self.default_agent or "jarvis"

    def reply_rule(self, chat_id: int, *, is_private: bool) -> dict[str, Any]:
        """Reply trigger for this chat. Missing row keeps the legacy default."""
        from sharedagents_core.reply_policy import REPLY_ANY, REPLY_MENTION, normalize_rule

        kind = "person" if is_private else "group"
        stored = self.allowlist_rules.get(f"{kind}:{chat_id}")
        if stored:
            return normalize_rule(
                stored.get("reply"),
                stored.get("mention"),
                kind=kind,
                read=stored.get("read"),
                send=stored.get("send"),
            )
        if not is_private and self.mention_required_in_groups:
            return normalize_rule(REPLY_MENTION, "", kind="group")
        return normalize_rule(REPLY_ANY, "", kind=kind)

    def _access_for(self, chat_id: int) -> dict[str, bool]:
        from sharedagents_core.reply_policy import normalize_access

        for kind in ("person", "group"):
            stored = self.allowlist_rules.get(f"{kind}:{chat_id}")
            if stored:
                return normalize_access(stored.get("read"), stored.get("send"))
        return normalize_access(True, True)

    def can_read(self, chat_id: int) -> bool:
        return self._access_for(chat_id)["read"]

    def can_send(self, chat_id: int) -> bool:
        return self._access_for(chat_id)["send"]

    def save(self) -> None:
        """Write the config atomically. Caller is responsible for
        choosing when to save (the wizard does, the runtime doesn't —
        a bot restart re-reads it fresh).
        """
        p = _config_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.name}.tmp")
        tmp.write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, p)

    # ---- on-disk paths -------------------------------------------------

    @property
    def home(self) -> Path:
        return telegram_home()

    @property
    def config_path(self) -> Path:
        return _config_path()

    @property
    def logs_dir(self) -> Path:
        from sharedagents_core.layout import logs_root

        return logs_root()

    @property
    def logfile(self) -> Path:
        # Telegram lines land in the ONE shared
        # ``layout.telegram_logfile()`` so per-line attribution (agent
        # field on every JSONL row) is the only thing distinguishing
        # bots.
        from sharedagents_core.layout import telegram_logfile as _tg_log

        return _tg_log()

    @property
    def pidfile(self) -> Path:
        return telegram_home() / "telegram.pid"

    @property
    def spawn_lock(self) -> Path:
        return telegram_home() / "telegram.spawn.lock"

    @property
    def sessions_path(self) -> Path:
        """Per-chat session bindings. Phase A stores
        ``{chat_key: session_id}`` (one agent per bot); Phase B will
        change the value shape to ``{agent_name: session_id}`` so a
        chat can talk to multiple agents.
        """
        return telegram_home() / "sessions.json"

    @property
    def offset_path(self) -> Path:
        return telegram_home() / "offset.json"

    @property
    def visibility_path(self) -> Path:
        """Per-(chat, thread) display-visibility map (`/tools` and
        `/reasoning` levels). Persisted so a bot restart doesn't reset
        every chat to the off/off default — the operator's prior
        ``/tools on`` should outlive a redeploy."""
        return telegram_home() / "visibility.json"


__all__ = [
    "TelegramConfig",
    "telegram_home",
]
