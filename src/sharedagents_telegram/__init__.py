"""sharedagents-telegram — the standalone Telegram gateway service.

Independent of any one sharedagents agent: the bot lives at
``~/.sharedagents/telegram/`` (config + sessions + offset + logs) and
routes each chat to whichever agent ``TelegramConfig.agent_for(...)``
resolves — per-(chat,thread) override → bare-chat override →
``default_agent``. ACP client over the sharedagents_gateway stdio surface
(or WebSocket if ``SHAREDAGENTS_TELEGRAM_ACP_URL`` is set). One AcpClient
per ``(chat,thread)`` so the ``ClientCapabilities`` advertisement
reflects per-chat trust — public chats stay text-only, trusted chats
(in ``TelegramConfig.trusted_chat_ids`` or
``SHAREDAGENTS_TELEGRAM_TRUSTED_CHATS``) get ``fs.readTextFile`` (never
``writeTextFile``, never terminal). Maps each Telegram chat to one
ACP session, persists the binding at
``~/.sharedagents/telegram/sessions.json`` and reattaches saved chats
through ``session/load`` on startup.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
