"""Telegram bot frontend for sharedagents — ACP-only.

The bot is a STANDALONE service (no longer spawned by an agent
gateway). It owns its own home at ``~/.sharedagents/telegram/`` and talks
to whichever sharedagents agent the chat picks. Phase A binds every chat
to one default agent (``TelegramConfig.default_agent``); Phase B will
add per-(chat,agent) sessions and a ``/agent <name>`` switch.

Long-polls Telegram ``getUpdates``, maps each chat to one ACP session
running on a child ``sharedagents_gateway acp-stdio`` process, and streams
the agent's ``session/update`` notifications back to Telegram as
in-place message edits.

On-disk state (under ``~/.sharedagents/telegram/``):

    config.json
        ``bot_token``, ``allowed_chat_ids``, ``trusted_chat_ids``,
        ``default_agent``. Edited by ``sharedagents telegram setup`` /
        directly with an editor; bot reads it once at boot.

    sessions.json
        ``{"<chat_key>": "<acp_session_id>"}`` — the chat → session
        binding that survives bot restarts. The bot calls
        ``session/load`` on each entry at startup and falls through
        to ``session/new`` for any the agent has forgotten.

    offset.json
        ``{"update_offset": <int>}`` — the ``getUpdates`` ack cursor.
        Persisted before dispatching each update so a crash
        mid-handler doesn't replay the same message on restart.

    logs/telegram.log
        Structured stdout/stderr of the poller. Separate from each
        agent's ``logs/gateway.log`` so the bot's own noise doesn't
        drown a gateway log.

Auth: :attr:`TelegramConfig.allowed_chat_ids` is the allowlist; an
empty list denies everyone. The first message from a non-whitelisted
chat is logged once and gets one ``⛔ this bot is private`` reply
(with the offending ``chat_id`` so the operator can copy-paste it
into ``~/.sharedagents/telegram/config.json``). Subsequent messages from
the same chat are silently dropped.

Streaming UX: each user message gets a ``💭 thinking…`` placeholder reply
that is edited as ``session/update`` events arrive from the agent. Edits
are throttled to ~1/sec; the rolling-message helper rolls into a fresh
follow-up message when the buffer crosses Telegram's 4000-char cap rather
than truncating the tail.

Slash commands (text-only, no inline keyboards):

    /start   — greet, force-create a fresh session for this chat
    /new     — drop this chat's session binding; next message opens a
               fresh ACP session via ``session/new``
    /status  — agent name, capabilities advertisement, bound session id
    /stop    — fire ``session/cancel`` on the chat's in-flight prompt
    /cancel  — alias for /stop
    /help    — list commands
"""

# isort: skip_file
# Import order in this facade is load-bearing: the split-out modules import
# names back from it, so each re-export block must bind before the next runs.
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any


from sharedagents_acp.client.client import AcpRemoteError as AcpRemoteError
from sharedagents_agent import delivery as _delivery  # noqa: F401  (re-exported)
from sharedagents_agent.agent import AgentConfig
from sharedagents_agent.metrics_format import metrics_line as _metrics_line  # noqa: F401  (re-exported)
from sharedagents_telegram.config import TelegramConfig, _retired_main_should_heal as _retired_main_should_heal
from sharedagents_telegram.forum_topics import hydrate_topic_names, load_forum_topics


# ---------------------------------------------------------------------------
# Foundational helpers + constants — extracted to ``bot_helpers.py`` (de-
# monolith). ``bot_helpers`` imports nothing from this module (one-way), so this
# re-export sits early and cycle-free; every name stays importable from
# ``sharedagents_telegram.bot``.
# ---------------------------------------------------------------------------
from sharedagents_telegram.bot_helpers import (  # noqa: E402  re-export
    TELEGRAM_MAX_TEXT as TELEGRAM_MAX_TEXT,
    _apply_tool_watchdog as _apply_tool_watchdog,
    _assemble_error_reply as _assemble_error_reply,
    _await_with_turn_supervision as _await_with_turn_supervision,
    _cap_to_single_message as _cap_to_single_message,
    _capabilities_for_chat as _capabilities_for_chat,
    _chunk_for_telegram as _chunk_for_telegram,
    _clear_tool_watchdog as _clear_tool_watchdog,
    _effective_idle_timeout as _effective_idle_timeout,
    _format_relative_age as _format_relative_age,
    _is_recoverable_session_error as _is_recoverable_session_error,
    _load_acp_session_map,
    _load_inflight_turns,
    _load_update_offset as _load_update_offset,
    _load_visibility_map,
    _maybe_record_agent_error as _maybe_record_agent_error,
    _maybe_record_throttle_incident as _maybe_record_throttle_incident,
    _parse_chat_key as _parse_chat_key,
    _parse_tool_args as _parse_tool_args,
    _queue_timeout_s as _queue_timeout_s,
    _record_bot_incident as _record_bot_incident,
    _save_acp_session_map as _save_acp_session_map,
    _save_inflight_turns as _save_inflight_turns,
    _save_update_offset as _save_update_offset,
    _save_visibility_map as _save_visibility_map,
    _split_at as _split_at,
    _truncate_for_telegram as _truncate_for_telegram,
    _trusted_chats_from_env_or_cfg,
    _turn_idle_timeout_s as _turn_idle_timeout_s,
)


def _cold_external_engine_model(engine: str) -> str:
    """Model the next turn of a CLI engine will run, for the /new banner.

    Codex reports its model from the CLI stream once a turn runs, but /new
    is rendered before the next turn exists. Show the SAME effective default
    the spawn will pass (``codex_effective_default_model``) instead of a
    pinned id (hardcoded fallbacks removed). ``""`` when unknown
    or for other engines, so the caller's discovery fallback applies.
    """
    if engine != "codex_cli":
        return ""
    try:
        from sharedagents_agent.discovery._common import codex_effective_default_model

        return codex_effective_default_model()
    except Exception:  # noqa: BLE001
        return ""


# Initial throttle for in-place message edits. Telegram's *practical* edit
# limit is around 1 edit/sec/message; bursts above that get 429-throttled
# fast. With four active Telegram chats, 4s/chat keeps the aggregate edit
# cadence near 1 edit/sec before any adaptive backoff kicks in.
EDIT_THROTTLE_S = 4.0
# Bot-wide edit cadence guard. Telegram documents per-chat limits poorly, but
# production 429s show that independent per-chat throttles can still burst into
# a shared limit. Rolling updates claim this slot opportunistically; if it is
# busy, the newest text stays buffered for the next chunk/final paint.
EDIT_GLOBAL_MIN_GAP_S = 1.10
# Coalesce tiny stream deltas. A chat may paint before ``EDIT_THROTTLE_S`` only
# when a meaningful amount of new text arrived and the local/global min gaps are
# both clear; otherwise it waits for the regular time threshold.
EDIT_COALESCE_MIN_CHARS = 160
EDIT_COALESCE_MIN_GAP_S = 1.50
# Even "force" paints must respect Telegram's hard per-chat edit cadence.
# Tool-boundary force flushes are latency hints, not permission to provoke a
# 429 that freezes the bubble for retry_after seconds.
EDIT_FORCE_MIN_GAP_S = 1.25
# Cap how far the adaptive backoff can stretch. If we hit this we just
# stop editing the current placeholder mid-turn and let the final paint
# or follow-up message carry the result.
EDIT_THROTTLE_MAX_S = 30.0
# Frozen-bubble escape. Sustained per-chat edit 429s push ``cur_throttle``
# toward ``EDIT_THROTTLE_MAX_S`` and leave a ``retry_after`` embargo sitting on
# the live message, so streaming ``update()`` calls just return early and the
# bubble stays stuck on its last paint for retry_after stretches (up to ~34s) —
# the turn still completes, but the user reads the frozen bubble as "stuck"
# (133 telegram_edit_throttled incidents). Once a turn has been
# edit-throttled this many times, stop fighting the per-message EDIT limit:
# roll the stream over to a FRESH follow-up message. A new-message SEND is
# governed by the per-chat SEND limit, which is INDEPENDENT of the per-message
# EDIT 429 — so streamed progress becomes visible again. Bounded by
# ``EDIT_429_MAX_ROLLOVERS`` per turn so a chat that 429s every edit can't spam
# an unbounded number of bubbles.
EDIT_429_ROLLOVER_THRESHOLD = 3
EDIT_429_MAX_ROLLOVERS = 3

# A *new message* send (the turn's placeholder, each 🔧 tool-header bubble, the
# 💬 answer bubble, error/follow-up messages) is a separate sendMessage /
# sendRichMessage call and counts against Telegram's per-chat SEND rate limit
# (≈20 msgs/min for groups) — which is INDEPENDENT of the per-message EDIT limit.
# A turn with many tool calls used to fire these back-to-back and trip a per-chat
# 429; the embargo then sat on the chat and dropped the NEXT incoming message's
# placeholder send entirely → "seeing the messages but not replying" (operator
# 2026-06-16). Pace new-message sends per chat so the tool-output cadence stays
# under the limit. This is intentionally close to ``EDIT_GLOBAL_MIN_GAP_S``.
NEW_MSG_MIN_GAP_S = 1.10
# Bot-wide (cross-chat) new-message SEND floor — the aggregate companion to the
# per-chat ``NEW_MSG_MIN_GAP_S`` and the send-side mirror of the edit path's
# ``EDIT_GLOBAL_MIN_GAP_S``. Independent per-chat pacing can still burst into a
# shared per-BOT send limit when several chats stream at once: each chat's FIRST
# send is never delayed, so N active chats fire N placeholder sends in the same
# instant and trip a global 429 (generalized to the many-
# chat case). This paces sends across ALL chats at an aggregate floor kept BELOW
# the per-chat gap, so a single chat's own cadence is unchanged and only cross-
# chat bursts are throttled.
NEW_MSG_GLOBAL_MIN_GAP_S = 0.5
# Even with pacing, a send can still 429 (another client/chat sharing the limit,
# a pre-existing embargo). Honor an explicit ``retry_after`` and retry the SAME
# send instead of dropping it (which aborted the whole turn). Bounded so a large
# ``retry_after`` can never hang the bot: at most this many retries, each waiting
# at most ``NEW_MSG_RETRY_MAX_WAIT_S``.
NEW_MSG_RETRY_ATTEMPTS = 2
NEW_MSG_RETRY_MAX_WAIT_S = 30.0

# How often to re-send the "typing…" chat action while a turn runs. Telegram
# clears a chat action after ~5s, so we refresh just under that to keep the
# "sharedagents is typing…" status visible for the whole turn.
TYPING_REFRESH_S = 4.0


INFLIGHT_RECOVERY_WAIT_S = 20.0
INFLIGHT_RECOVERY_POLL_S = 1.0

# Cap on image upload size forwarded to upstream vision models. Matches
# the legacy bot's threshold; keeps a single base64 payload below
# upstream HTTP body limits even after the ~33% inflation.
_MAX_IMAGE_BYTES = 8 * 1024 * 1024  # 8 MiB

# Long-poll timeout sent to Telegram. Inside this window getUpdates blocks
# server-side until a new update arrives, so the poller is effectively
# idle (no spinning). 25s is well under Telegram's 50s ceiling.
LONG_POLL_TIMEOUT_S = 25


# Telegram's CLOUD Bot API caps ``getFile`` (and thus any bot-side file
# download) at 20 MB. A larger voice/audio attachment cannot be fetched
# at all — getFile returns ``{"ok": false, "description": "file is too
# big"}``. Detect it from the message's ``file_size`` up front so we can
# tell the user clearly instead of emitting a generic download failure.
# (A self-hosted ``telegram-bot-api`` server raises this to ~2 GB.)
TG_GETFILE_MAX_BYTES = 20 * 1024 * 1024
# Self-hosted telegram-bot-api (--local) lifts the download cap; uploads
# (and thus practical file size) top out at ~2 GB.
TG_LOCAL_MAX_BYTES = 2000 * 1024 * 1024

# Two-press hard-kill window for ``/stop``/``/cancel``/``/abort``.
# Mirrors the bridge's ``_HARD_KILL_WINDOW_S`` (acp_bridge.py) so the
# Telegram ack reflects whether the server will escalate from SIGTERM
# to SIGKILL on this press. Kept slightly *wider* than the bridge value
# because the Telegram round-trip adds ~100–300 ms vs the TUI's local
# transport — a strict 1.0 s window would under-report force-kills.
_ABORT_DOUBLE_PRESS_S = 1.5

# Commands registered with Telegram's ``setMyCommands`` at startup so the
# "/" autocomplete popup in every chat shows what the bot actually handles.
# These are the persistent built-in commands — file-based commands from
# ``commands/*.md`` are NOT pre-registered (they vary per operator and
# ``setMyCommands`` has a 100-command ceiling). The ``/help`` card and
# ``/commands`` listing remain the canonical discovery surface for those.
_PERSISTENT_COMMANDS: list[dict[str, str]] = [
    {"command": "help", "description": "Show this help card"},
    {"command": "new", "description": "Start a fresh session"},
    {"command": "status", "description": "Show agent and session info"},
    {"command": "stop", "description": "Abort the in-flight prompt"},
    {"command": "cancel", "description": "Alias for /stop"},
    {"command": "abort", "description": "Force-kill the running prompt"},
    {"command": "steer", "description": "Redirect the running turn: /steer <text>"},
    {"command": "sessions", "description": "List saved sessions"},
    {"command": "memory", "description": "Show memory availability"},
    {"command": "history", "description": "Show recent turns"},
    {"command": "todo", "description": "Show the model's todo list"},
    {"command": "compact", "description": "Squeeze the running prompt"},
    {"command": "skills", "description": "List installed skills"},
    {"command": "skill", "description": "Show a skill's SKILL.md body"},
    {"command": "commands", "description": "List file-based commands"},
    {"command": "refresh", "description": "Reload session state from disk"},
    {"command": "mode", "description": "Set permission mode: plan|ask|auto"},
    {"command": "models", "description": "List configured models"},
    {"command": "model", "description": "Switch the live model"},
    {"command": "reasoning", "description": "Toggle thinking-token visibility"},
    {"command": "effort", "description": "Set model-specific reasoning depth"},
    {"command": "reason", "description": "Alias for /reasoning"},
    {"command": "on", "description": "Shortcut: /reasoning on"},
    {"command": "off", "description": "Shortcut: /reasoning off"},
    {"command": "stream", "description": "Shortcut: /reasoning stream"},
    {"command": "thinking", "description": "Toggle model thinking"},
    {"command": "tools", "description": "Toggle tool call visibility"},
    {"command": "tool", "description": "Alias for /tools"},
    {"command": "teacher", "description": "Teacher supervision: on|off|model <alias>"},
    {"command": "goal", "description": "Durable goal: /goal <condition> | list | pause|resume <id>"},
    {"command": "loop", "description": "Recurring loop: /loop <prompt> | list | pause|resume|archive|stop <id>"},
    {"command": "budget", "description": "Root cost budget: /budget set <usd> | topup <usd>"},
    {"command": "claude", "description": "Bridge to Claude Code"},
    {"command": "codex", "description": "Bridge to Codex CLI"},
    {"command": "antigravity", "description": "Bridge to Antigravity CLI"},
    {"command": "agents", "description": "List configured agents"},
    {"command": "agent", "description": "Switch to a different agent"},
    {"command": "plugin", "description": "Manage plugin bundles"},
    {"command": "plugins", "description": "Alias for /plugin"},
    {"command": "mcp", "description": "List MCP servers"},
    {"command": "mcps", "description": "Alias for /mcp"},
    {"command": "start", "description": "Show help + new session"},
    {"command": "restart", "description": "Restart the sharedagents gateway"},
    {"command": "context", "description": "Show context detail report"},
    {"command": "ctx", "description": "Alias for /context"},
    {"command": "todos", "description": "Alias for /todo"},
    {"command": "teacher", "description": "Toggle teacher-model supervision: on|off|model <alias>"},
    {"command": "subagents", "description": "List / enter delegated subagent sessions"},
    {"command": "monitors", "description": "Background shells the model spawned"},
    {"command": "shells", "description": "Alias for /monitors"},
    {"command": "mouse", "description": "Toggle TUI mouse capture"},
    {"command": "task", "description": "Operator harness-task store"},
    {"command": "tasks", "description": "Alias for /task"},
    {"command": "title", "description": "Set the session title: /title <text>"},
]

# Hint the user sees on first contact when they're not whitelisted.
DENY_LOG_TEMPLATE = (
    "DENIED chat_id={chat_id} username=@{username} ({first_name!r}) — "
    "not in telegram allowed_chat_ids. To whitelist, edit "
    "{config_path} and add {chat_id} to allowed_chat_ids, "
    "then `sharedagents telegram restart`."
)

# Env var overrides. ``ACP_DEFAULT_WS_URL_ENV`` is consulted only when the
# constructor's ``acp_url`` is None. The default is the north gateway over
# Streamable HTTP — set this to
# ``http://127.0.0.1:<gateway-port>`` (the gateway routes by URL path,
# so the bot appends ``/acp/<agent>`` per chat). ``ws://`` is still
# accepted for legacy / fake-server tests; empty disables and falls
# back to a stdio child spawn (kept solely as a unit-test fallback).
ACP_DEFAULT_WS_URL_ENV = "SHAREDAGENTS_TELEGRAM_ACP_URL"
ACP_BEARER_TOKEN_ENV = "SHAREDAGENTS_TELEGRAM_ACP_TOKEN"

# Rich Messages (Bot API 10.1). ``SHAREDAGENTS_TELEGRAM_RICH`` controls whether
# the bot uses ``sendRichMessage`` / ``editMessageText(rich_message=…)`` for
# final answer delivery. Values:
#   ``off``  — exact current behaviour (plain text).
#   ``auto`` — probe once at startup; if the API supports rich messages, use
#              them for final paints; cache the result per-process.
#   ``on``   — always use rich; log loudly if the API rejects (still falls
#              back to plain text for delivery).
# Default: ``auto``.
_RICH_ENV = os.environ.get("SHAREDAGENTS_TELEGRAM_RICH", "auto")

# Rich draft streaming (Bot API 10.1 ``sendRichMessageDraft``). Disabled by
# default because the draft path is more fragile (rollover desync, reply-link
# loss). Enable only after visual verification. Values: ``off`` (default),
# ``on`` (use drafts when rich is also enabled).
_RICH_DRAFTS_ENV = os.environ.get("SHAREDAGENTS_TELEGRAM_RICH_DRAFTS", "off")

# Replies for inputs ACP-v1 doesn't yet route to the agent. Surfaced as
# constants so tests can assert on them by symbol rather than string.
ACP_IMAGE_UNSUPPORTED_REPLY = "📎 image input not yet supported on this bot — send the question as text instead."
ACP_DOCUMENT_UNSUPPORTED_REPLY = "📎 file input not yet supported on this bot — paste the contents as text instead."
ACP_STICKER_UNSUPPORTED_REPLY = "🖼️ stickers aren't routed to the agent — send the question as text."
ACP_IMAGE_FORMAT_UNSUPPORTED_REPLY = "📎 that image format can't be read — send it as PNG, JPEG, WebP or GIF."


# ---------------------------------------------------------------------------
# Persistence helpers (chatId → ACP session id; getUpdates offset cursor)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Voice/audio subsystem (STT + TTS) — extracted to ``stt.py`` (de-monolith).
# Re-exported here so callers/tests importing these from ``sharedagents_telegram.bot``
# keep working. ``stt.py`` imports nothing from this module (one-way).
# ---------------------------------------------------------------------------
from sharedagents_telegram.stt import (  # noqa: E402  re-export
    OPENROUTER_TTS_DEFAULT_VOICE as OPENROUTER_TTS_DEFAULT_VOICE,
    STT_DECODE_UNAVAILABLE_MSG as STT_DECODE_UNAVAILABLE_MSG,
    STT_EMPTY_TRANSCRIPT_MSG as STT_EMPTY_TRANSCRIPT_MSG,
    TranscriptionUnavailableError as TranscriptionUnavailableError,
    _make_default_stt as _make_default_stt,
    _make_default_tts as _make_default_tts,
    _transcode_to_mp3 as _transcode_to_mp3,
)


# --- Queued / timed-out turn indicators ----------------------------------
# A second message for the same (chat, topic) cannot start until the
# in-flight turn releases the per-chat lock (see ``_dispatch_blocks``). While
# it waits the placeholder must SAY so — not lie with "thinking" — and a
# message that waits longer than ``queue_timeout_s`` is dropped rather than
# answered stale (same reasoning as the startup backlog drain).
# Voice/audio: the FIRST bubble confirms the audio was received and is being
# transcribed (sequence: 🎙️ hearing… → 🎙️ heard: "<text>" → 💭 thinking… → answer).
_HEARING_PLACEHOLDER = "🎙️ hearing…"
_QUEUED_PLACEHOLDER = "⏳ queued — waiting for the current turn…"
_QUEUE_TIMEOUT_REPLY = "⌛ queued too long — send it again if you still need it."

# Lifecycle reaction on the user's INCOMING message. Every glyph is from
# Telegram's fixed allowed-reaction set (⏳/⌛ are NOT — they work only as
# message text), so these intentionally differ from the placeholder glyphs.
_REACT_SEEN = "👀"  # received, not yet started
_REACT_QUEUED = "🤔"  # waiting behind a running turn
_REACT_ACTIVE = "✍️"  # turn is running
_REACT_REPLIED = "👍"  # answered
_REACT_EXPIRED = "🥱"  # dropped — queued too long


# ---------------------------------------------------------------------------
# Media download + doc conversion — extracted to ``media.py`` (de-monolith).
# Re-exported here so callers/tests importing these from ``sharedagents_telegram.bot``
# keep working. ``media.py`` imports nothing from this module (one-way).
# ---------------------------------------------------------------------------
from sharedagents_telegram.media import (  # noqa: E402  re-export
    _OFFICE_EXTENSIONS as _OFFICE_EXTENSIONS,
    _convert_office_bytes_to_markdown as _convert_office_bytes_to_markdown,
    _make_default_audio_downloader as _make_default_audio_downloader,
)


# ---------------------------------------------------------------------------
# Telegram HTTP client — extracted to ``api.py`` (de-monolith). Re-exported here
# so callers/tests importing it from ``sharedagents_telegram.bot`` keep working.
# ---------------------------------------------------------------------------
from sharedagents_telegram.api import (  # noqa: E402  re-export
    TELEGRAM_API as TELEGRAM_API,
    _TelegramAPI,
)


# ---------------------------------------------------------------------------
# Streaming renderers — extracted to ``rendering.py`` (de-monolith). Re-exported
# here so callers/tests importing them from ``sharedagents_telegram.bot`` keep working.
# ---------------------------------------------------------------------------
from sharedagents_telegram.rendering import (  # noqa: E402  re-export
    _RollingMessage as _RollingMessage,
    _BlockStream as _BlockStream,
    _TelegramRenderCtx as _TelegramRenderCtx,
    _THINKING_PLACEHOLDER as _THINKING_PLACEHOLDER,
    _direct_answer_ok as _direct_answer_ok,
    _DIRECT_DISPATCH_TITLES as _DIRECT_DISPATCH_TITLES,
    _DELEGATE_BADGES as _DELEGATE_BADGES,
    _INTERNAL_TOOL_TITLES as _INTERNAL_TOOL_TITLES,
)


# ---------------------------------------------------------------------------
# AcpTelegramBot
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# AcpTelegramBot method groups — split into mixins (de-monolith). The concrete
# class below keeps __init__ + all instance-attribute setup; each mixin holds a
# cohesive group of methods, moved verbatim. Method names are unique across the
# bases (no override); MRO resolves every method to its single definition.
# ---------------------------------------------------------------------------
from sharedagents_telegram.bot_sessions import SessionsMixin  # noqa: E402
from sharedagents_telegram.bot_streaming import StreamingMixin  # noqa: E402
from sharedagents_telegram.bot_handlers import HandlersMixin  # noqa: E402
from sharedagents_telegram.bot_commands import CommandsMixin  # noqa: E402
from sharedagents_telegram.bot_lifecycle import LifecycleMixin  # noqa: E402


class AcpTelegramBot(
    SessionsMixin,
    StreamingMixin,
    HandlersMixin,
    CommandsMixin,
    LifecycleMixin,
):
    """ACP-driven Telegram bot.

    Connects to one sharedagents agent per chat over the ACP transport
    (WebSocket if ``acp_url`` is set, otherwise stdio-spawned
    ``sharedagents_gateway acp-stdio``). Maps each Telegram chat to one ACP
    session; per-chat :class:`acp.schema.ClientCapabilities` so trusted
    chats can opt into ``fs.readTextFile`` while the public default stays
    text-only.

    Concurrency model:

    * One :class:`AcpClient` per chat, cached in ``self._clients``.
      Reusing one client across chats would force a single capability
      profile on the whole process; per-chat clients let the trusted
      uplift survive next to the locked-down default.
    * One ``asyncio.Lock`` per chat in ``self._chat_locks`` so two
      messages from the same chat serialize on the same session. The
      agent does not enforce per-session serialization, so we have to.
    * Telegram I/O is plain JSON-over-HTTP through :class:`_TelegramAPI`.

    Test seam: every external dependency is injectable.
    ``client_factory`` is an async callable
    ``(chat_id, capabilities) -> AcpClient`` that returns an
    already-initialized client (no ``initialize()`` call from the bot).
    ``api`` is the Telegram-side bridge. ``stt``, ``download_audio`` are
    the audio path stubs. Tests pass stubs and never open a network
    socket.
    """

    def __init__(
        self,
        tg_cfg: TelegramConfig,
        *,
        acp_url: str | None = None,
        acp_bearer_token: str | None = None,
        trusted_chats: set[int] | None = None,
        log: logging.Logger | None = None,
        client_factory: Any | None = None,
        api: _TelegramAPI | None = None,
        stt: Any | None = None,
        tts: Any | None = None,
        download_audio: Any | None = None,
        download_image: Any | None = None,
    ) -> None:
        self.tg_cfg = tg_cfg
        self._log = log or logging.getLogger("sharedagents.telegram")
        self.acp_url = acp_url if acp_url is not None else os.environ.get(ACP_DEFAULT_WS_URL_ENV, "").strip()
        self.acp_bearer_token = (
            acp_bearer_token
            if acp_bearer_token is not None
            else os.environ.get(ACP_BEARER_TOKEN_ENV, "").strip() or None
        )
        self.trusted_chats = set(trusted_chats) if trusted_chats is not None else _trusted_chats_from_env_or_cfg(tg_cfg)
        self._client_factory = client_factory or self._default_client_factory
        # ``api`` defaults to ``None`` so unit tests that never run the
        # long-poll loop (and thus never need outbound Telegram I/O)
        # don't have to construct one. ``run()`` instantiates a real
        # ``_TelegramAPI`` if ``api`` is None.
        self._api = api
        self._stt = stt
        self._tts = tts
        self._download_audio = download_audio
        # Photo download factory — mirrors the audio path. Defaults
        # to the bot-file CDN downloader if the AcpTelegramBot is
        # constructed via :meth:`from_config`; tests inject a stub.
        self._download_image = download_image or download_audio

        # ``_sessions`` is nested chat_key → agent_name → session_id
        # so a chat that switches agents back and forth resumes the
        # right session for each. The on-disk map auto-migrates from
        # the Phase-A flat shape on first read.
        self._sessions: dict[str, dict[str, str]] = _load_acp_session_map(tg_cfg)
        self._stale: set[str] = set()
        self._inflight_turns: dict[str, dict[str, Any]] = _load_inflight_turns(tg_cfg)
        # ACP clients keyed by chat_key — each chat gets its own
        # connection to whichever agent the route resolves to. When a
        # chat switches agents (Phase B), the cached entry is dropped
        # and the next message rebuilds against the new target.
        self._clients: dict[str, Any] = {}
        # Subprocess handles for stdio-spawned children, keyed on the
        # same chat_id string. Held so ``close()`` can ``terminate()`` /
        # ``wait()`` them — orderly stdin EOF first, then SIGTERM.
        self._child_procs: dict[str, Any] = {}
        # Monotonic timestamp of the most recent ``/stop``/``/cancel``/
        # ``/abort`` per (chat, thread). Used purely for UX visibility:
        # if the user double-taps within ``_ABORT_DOUBLE_PRESS_S``, the
        # ack reply notes that the bridge will have escalated to
        # SIGKILL on its side (see acp_bridge.py ``_HARD_KILL_WINDOW_S``,
        # fix D). The escalation happens server-side regardless of
        # whether we surface it here — this dict just powers the
        # "(force-kill escalation)" suffix on the second ack.
        self._last_abort_at: dict[tuple[int, int | None], float] = {}
        # Live turn handles for instant ``/abort`` UI teardown (chat_key →
        # roller / dispatch / typing tasks). Populated for the duration of
        # ``_stream_turn_for_chat`` only.
        self._live_turns: dict[str, dict[str, Any]] = {}
        # AgentConfig cache so we don't re-read disk per-message.
        # Keyed by agent name; bot restart picks up edits.
        self._agent_cfg_cache: dict[str, AgentConfig] = {}
        self._chat_locks: dict[str, asyncio.Lock] = {}
        # Per-(chat,thread) display visibility — frontend-side state,
        # never on the ACP wire. The agent always emits
        # ``agent_thought_chunk`` and ``tool_call_update`` regardless;
        # we decide whether to render. Mirrors the TUI's
        # ``reasoning_visibility`` / ``tool_output_visible`` toggles.
        # Defaults: thoughts hidden, tool headers hidden — match the
        # legacy per-chat defaults so a fresh chat behaves the same.
        # Hydrated from disk so ``/tools on`` and ``/reasoning stream``
        # outlive a bot restart — without this, redeploying the bot
        # silently flips every chat back to off/off and the operator
        # has to retype every toggle.
        self._visibility: dict[str, dict[str, Any]] = _load_visibility_map(tg_cfg)

        # Forum-topic names learned from incoming updates, keyed
        # ``str(chat_id) -> {thread_id: name}``. Telegram only ships the
        # human topic NAME on the ``forum_topic_created`` / ``forum_topic_edited``
        # service messages (and on ``reply_to_message`` of a topic's first
        # reply), never on every message — so we cache it here and stamp the
        # name onto a session's ``topic`` when we bind one. The bound session's
        # topic is persisted durably (sessions_meta), so this in-memory cache
        # only has to survive long enough to name the NEXT new session in a
        # thread; a fallback of ``topic-<thread_id>`` covers a cold cache.
        self._topic_names: dict[str, dict[int, str]] = hydrate_topic_names(
            load_forum_topics(),
        )
        # Topic last pushed onto each bound session (``sid -> topic``). Lets
        # ``_sync_session_topic`` skip a redundant ACP round-trip when the
        # topic is already in sync, while still re-pushing when a forum-topic
        # RENAME changes the resolved name for an already-bound session.
        self._applied_topic: dict[str, str] = {}

        # Long-poll loop state.
        self._stop = asyncio.Event()
        self._dispatched: set[asyncio.Task] = set()
        # Chats we've already bounced once with a deny reply, so a
        # script-kiddie can't get us to flood Telegram.
        self._denied_chats: set[int] = set()
        # Bot's @username, populated by ``getMe`` at startup. Used to
        # strip ``@<botname>`` from inbound text — Telegram delivers
        # ``/cmd@<botname>`` for slash commands in groups and a bare
        # leading ``@<botname>`` for natural-language mentions, neither
        # of which the agent should see verbatim. ``""`` until ``run()``
        # has connected; ``_strip_bot_mention`` is a no-op until then.
        self._bot_username: str = ""
        # Bot's numeric user id, also from ``getMe``. Used by
        # ``mention_required_in_groups`` to recognise direct replies
        # to the bot's own messages as an implicit mention — common
        # Telegram convention is that "swipe to reply" addresses the
        # author without typing the @handle.
        self._bot_user_id: int = 0
        # In-flight ``session/request_permission`` prompts awaiting a
        # button press. token → asyncio.Future[str]; the per-client
        # permission handler (see ``_make_permission_handler``) parks a
        # future here and ``_handle_callback_query`` resolves it with
        # the chosen ACP optionId when the user taps an inline button.
        self._pending_permissions: dict[str, asyncio.Future] = {}

        # ---- live cross-interface renderer (FIX 1, spec §8) ---------------
        # Per (chat,thread)-bound-session background observer that tails the
        # session's canonical event stream and renders turns STARTED ELSEWHERE
        # (web / TUI / another chat) into THIS Telegram chat, so the operator's
        # #1 goal holds: the SAME session shows the SAME behaviour on TUI, web
        # AND Telegram. Keyed by ``"<chat_key>::<sid>"`` so a chat that
        # switches session (``/new``) gets a fresh observer and the old one is
        # cancelled. One task per (chat,thread,session); idempotent.
        self._render_log_tasks: dict[str, asyncio.Task] = {}
        # Per chat_key: the highest turn_idx already rendered into the chat
        # (by either the local submit path or the renderer). The renderer skips
        # any turn below this so a turn renders EXACTLY once across both paths.
        self._rendered_turn_count: dict[str, int] = {}
        # Per chat_key: armed by the local submit the instant a turn is POSTed
        # (before its turn_idx is known), so the renderer claims that next turn
        # as locally-owned and does NOT double-render it. Cleared once claimed.
        self._local_turn_pending: dict[str, bool] = {}
        # Per chat_key: the live byte offset the renderer starts tailing from
        # (the first local submit's ``start_offset``), so it never replays the
        # session's pre-existing history into the chat — only NEW turns.
        self._rendered_offset: dict[str, int] = {}


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def _setup_logging(tg_cfg: TelegramConfig) -> logging.Logger:
    """Wire ``sharedagents.telegram`` to the unified JSONL telegram log.

    (A plaintext FileHandler
    would violate the "each line JSONL with origin tags" contract for the
    shared ``telegram.log`` file.) Stderr stays plaintext for
    journalctl readability.
    """
    from sharedagents_core.jsonlog import JsonlLogHandler

    log = logging.getLogger("sharedagents.telegram")
    log.setLevel(logging.INFO)
    log.propagate = False
    log_path = tg_cfg.logfile
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = JsonlLogHandler(log_path, service="telegram", default_agent="telegram")
    fh.setLevel(logging.INFO)
    log.addHandler(fh)
    # Stderr handler is kept plaintext for journalctl readability —
    # it never touches the shared JSONL file.
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    log.addHandler(sh)
    return log


def _load_env_file(path: Path) -> int:
    """Merge ``KEY=value`` lines from ``path`` into ``os.environ``.

    Mirrors ``sharedagents_gateway.runtime._load_agent_env_file`` so the
    Telegram bot picks up secrets the same way an agent does. Vars
    already set non-empty in ``os.environ`` win over the file (so an
    explicit ``Environment=`` on a systemd unit / shell-exported var
    can still override). Returns the number of vars added.
    """
    if not path.is_file():
        return 0
    added = 0
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if not key:
                continue
            if os.environ.get(key):
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            os.environ[key] = value
            added += 1
    except OSError:
        pass
    return added


def _bootstrap_env(tg_cfg: TelegramConfig) -> list[tuple[Path, int]]:
    """Load ``.env`` files into ``os.environ`` before the bot starts.

    Order (lowest priority first; later loads don't overwrite already
    populated vars):

      1. ``<telegram_home>/.env`` — bot-specific secrets.
      2. ``<agents_root>/<default_agent>/.env`` — falls back to the
         agent's own credential set so a Telegram operator who put
         ``OPENROUTER_API_KEY=...`` in ``agents/jarvis/.env`` (the most
         common layout) gets STT for free without having to duplicate
         the key into the telegram home.

    Returns ``[(path, vars_added), …]`` so the caller can log which
    files contributed which secrets.
    """
    loaded: list[tuple[Path, int]] = []
    tg_env = tg_cfg.home / ".env"
    n = _load_env_file(tg_env)
    if n or tg_env.is_file():
        loaded.append((tg_env, n))
    # Agent .env: best-effort — the AgentConfig.load can fail (no
    # config.json yet, malformed file, etc.); a missing default-agent
    # config shouldn't block the bot from booting on its own .env.
    try:
        agent_cfg = AgentConfig.load(tg_cfg.default_agent or "jarvis")
        agent_env = agent_cfg.home / ".env"
        n = _load_env_file(agent_env)
        if n or agent_env.is_file():
            loaded.append((agent_env, n))
    except Exception:  # noqa: BLE001
        pass
    return loaded


def _resolve_deploy_acp_url(explicit: str | None) -> str:
    """Resolve the acp_url the DEPLOYED bot (systemd / CLI) should use.

    Precedence: an explicit ``--acp-url`` → the ``SHAREDAGENTS_TELEGRAM_ACP_URL`` env
    var → the sharedagents gateway over Streamable HTTP (``http://127.0.0.1:<port>``)
    as the PRODUCTION DEFAULT, so the bot rides the canonical submit +
    ``mode=events`` stream path (Project-1 parity) rather than the legacy stdio
    child. Only collapses to "" (stdio) if the gateway port can't be resolved.
    """
    if explicit:
        return explicit
    env = os.environ.get(ACP_DEFAULT_WS_URL_ENV, "").strip()
    if env:
        return env
    try:
        from sharedagents_gateway.gateway import _gateway_config

        port = _gateway_config().get("port") or 8100
        return f"http://127.0.0.1:{port}"
    except Exception:  # noqa: BLE001 — degrade to the stdio fallback
        return ""


def acp_main(argv: list[str] | None = None) -> int:
    """Module entry point. Run as ``python -m sharedagents_telegram.bot``
    (no agent arg — routing comes from
    ``~/.sharedagents/telegram/config.json``).
    """
    parser = argparse.ArgumentParser(
        prog="sharedagents-telegram",
        description=(
            "Standalone Telegram bot service for sharedagents. Routes each "
            "chat to the agent named in TelegramConfig.agent_routes "
            "(or default_agent for new chats)."
        ),
    )
    parser.add_argument(
        "--acp-url",
        default=None,
        help=(
            "ACP gateway URL, e.g. http://127.0.0.1:8700 (Streamable "
            "HTTP). Per-chat routing appends /acp/<agent>. ws:// "
            f"accepted for legacy/test servers. Falls back to "
            f"${ACP_DEFAULT_WS_URL_ENV} or, if unset, a stdio child "
            "spawn (test-only fallback)."
        ),
    )
    parser.add_argument(
        "--acp-token",
        default=None,
        help=(
            f"Bearer token for the gateway, if it was started with auth_token. Falls back to ${ACP_BEARER_TOKEN_ENV}."
        ),
    )
    args = parser.parse_args(argv)

    tg_cfg = TelegramConfig.load()
    if not tg_cfg.bot_token:
        sys.stderr.write(
            "no Telegram bot_token configured in "
            f"{tg_cfg.config_path}\n"
            "Run `sharedagents telegram setup` to create one, or edit the "
            "file directly.\n",
        )
        return 1

    # Populate os.environ from .env files BEFORE _setup_logging /
    # AcpTelegramBot.run reads them. Without this, a fresh shell with
    # no exported OPENROUTER_API_KEY would boot the bot with STT
    # disabled even when the operator already configured the key in
    # their agent's .env.
    env_loaded = _bootstrap_env(tg_cfg)

    log = _setup_logging(tg_cfg)
    for path, n in env_loaded:
        if n:
            log.info(".env: loaded %d var(s) from %s", n, path)
        else:
            log.debug(".env: %s present but no new vars added", path)

    # Deployment default: the canonical HTTP gateway (submit + mode=events
    # stream) — the Project-1 parity requirement. A bot started by systemd with
    # neither --acp-url nor SHAREDAGENTS_TELEGRAM_ACP_URL set MUST still ride the
    # canonical path, NOT the legacy stdio child. (Scoped to the CLI entry so
    # direct-construction unit tests keep their stdio fallback.)
    deploy_acp_url = _resolve_deploy_acp_url(args.acp_url)

    async def _go() -> int:
        bot = AcpTelegramBot(
            tg_cfg,
            acp_url=deploy_acp_url,
            acp_bearer_token=args.acp_token,
            log=log,
        )
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, bot.request_stop)
            except (NotImplementedError, RuntimeError):
                pass
        return await bot.run()

    try:
        return asyncio.run(_go())
    except KeyboardInterrupt:
        return 0


# Backwards-compatible alias. The gateway-spawn path is
# ``python -m sharedagents_telegram.bot``; that lands on ``__main__`` below.
# ``main`` is exported so existing callers (operators with shell aliases,
# external tooling) that imported it continue to work.
main = acp_main


# --- de-monolith facade monkeypatch liveness -------------------------------
# bot.py was split into extracted submodules (bot_helpers/bot_handlers/
# bot_sessions/bot_streaming/bot_commands/bot_lifecycle/stt/media/api/rendering).
# Tests (and callers) monkeypatch module-level names on THIS facade
# (``sharedagents_telegram.bot``); pre-split those reads were intra-module so the patch
# took effect. Post-split a relocated reader holds its own copy, so a facade patch
# would silently miss it. Propagate any attribute ASSIGNED on this module to every
# ``sharedagents_telegram.*`` submodule that already defines that name — restoring
# pre-split monkeypatch semantics (and covering internal reader chains, e.g.
# _transcode_to_mp3 -> _resolve_ffmpeg). Fires ONLY on external setattr (module-body
# assignments ran before this class swap), so production is unaffected.
import types as _types  # noqa: E402


class _BotFacadeModule(_types.ModuleType):
    def __setattr__(self, name, value):
        _types.ModuleType.__setattr__(self, name, value)
        # NEVER propagate module dunders: importlib.reload() re-assigns
        # __name__/__spec__/__file__/__loader__/__class__ etc. on the facade;
        # propagating those would rewrite each sibling's identity (and __class__
        # would convert siblings into _BotFacadeModule), corrupting them.
        if name.startswith("__"):
            return
        for _mn, _m in list(sys.modules.items()):
            if _m is not None and _mn.startswith("sharedagents_telegram.") and _mn != __name__ and hasattr(_m, name):
                try:
                    _types.ModuleType.__setattr__(_m, name, value)
                except Exception:
                    pass


sys.modules[__name__].__class__ = _BotFacadeModule


if __name__ == "__main__":
    sys.exit(acp_main())
