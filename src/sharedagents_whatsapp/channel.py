"""WhatsApp channel core: inbound routing, commands, prompts and deliveries.

Inbound: every message from the transport is filtered (self-chat only, not our
own device, not a redelivery, not an echoed agent reply, not stale), then routed
to a local command, to the answer of a parked question/permission, or to a new
prompt submitted through the gateway's canonical route.

Outbound: the channel is the sink of one :class:`SessionObserver` per bound
session; the observer turns the session's canonical events into deliveries and
the channel formats them (reply prefix, WhatsApp markup, chunking), reacts to
the prompting message with the outcome, and tracks parked prompts. Files a
turn records are fetched from the gateway and sent as a photo or a document.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import contextvars
import hashlib
import inspect
import logging
import mimetypes
import os
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

from sharedagents_agent.delivery import strip_runtime_notices
from sharedagents_core.attachments import (
    INLINE_IMAGE_MIMES,
    MAX_DOCUMENT_CHARS,
    clip_text,
    document_to_markdown,
    is_readable_document,
    sniff_image_mime,
)
from sharedagents_whatsapp.config import WhatsAppConfig
from sharedagents_whatsapp.formatting import (
    is_internal_prompt,
    render_messages,
    strip_internal_envelopes,
)
from sharedagents_whatsapp.gateway import (
    ArtifactUnavailable,
    GatewayClient,
    GatewayError,
    SessionNotFound,
)
from sharedagents_whatsapp.interactions import (
    CANCEL,
    PendingPermission,
    PendingQuestion,
    parse_answer,
    parse_permission_reply,
    parse_questions,
    render_permission,
    render_question,
)
from sharedagents_whatsapp.observer import SessionObserver
from sharedagents_whatsapp.state import SELF_CHAT, ChannelState
from sharedagents_whatsapp.transport import InboundMessage, Transport, TransportUnavailable

SttFn = Callable[..., Awaitable[str]]
Pending = PendingQuestion | PendingPermission

STALE_NOTICE_DELAY_S = 5.0
FORCE_STOP_WINDOW_S = 10.0
READY_TIMEOUT_S = 60.0
DONE, FAILED, STOPPED, ANSWERED = "✅", "❌", "⏹️", "👍"
_ALWAYS_COMMANDS = frozenset({"stop", "cancel", "new", "help"})
# The inbound message whose handler is running. ``_send`` replies to that
# message's chat unless the caller names a destination, so a handler's own
# replies (/new, /stop, errors, notices) cannot land in the self-chat while the
# user is talking from another chat. Tasks spawned from a handler run in an
# empty context and keep addressing their own destinations explicitly.
_REPLY_TO: contextvars.ContextVar[InboundMessage | None] = contextvars.ContextVar(
    "sharedagents_whatsapp_reply_to",
    default=None,
)
_DEST_UNSET: Any = object()
_PREVIEW_CHARS = 100
_ERROR_CHARS = 500
DOCUMENT_CHAR_LIMIT = MAX_DOCUMENT_CHARS
# Messages are handled one at a time (arrival order), so a document being read
# holds up the next message, /stop included: bound the wait.
DOCUMENT_CONVERT_TIMEOUT_S = 60.0
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".gif")
_READABLE_KINDS = "an image, a PDF, a Markdown/text file or an Office document"


def _msg_from_history(rec: dict) -> InboundMessage:
    from types import SimpleNamespace

    ident = str(rec.get("chat") or "")
    user, _, server = ident.partition("@")
    chat = SimpleNamespace(User=user, Server=server)
    sender_ident = str(rec.get("sender") or ("" if server == "g.us" else ident))
    sender_user, _, sender_server = sender_ident.partition("@")
    sender = SimpleNamespace(User=sender_user, Server=sender_server)
    return InboundMessage(
        msg_id=str(rec.get("id") or ""),
        chat=chat,
        sender=sender,
        timestamp=float(rec.get("ts") or 0),
        is_self_chat=bool(rec.get("is_self_chat")),
        from_me=bool(rec.get("from_me")),
        from_own_device=bool(rec.get("from_own_device")),
        kind=str(rec.get("kind") or "text"),
        text=str(rec.get("text") or ""),
        quoted=str(rec.get("quoted") or ""),
    )


HELP_TEXT = (
    "**sharedagents on WhatsApp** — write here and your agent answers.\n"
    "/new — start a fresh session\n"
    "/stop — stop the running turn (twice within 10 s to force)\n"
    "/agent [name] — show or switch the agent\n"
    "/status — link, agent and session status\n"
    "Any other /command (e.g. /model, /mode, /compact) runs in the agent runtime.\n"
    "Voice notes are transcribed; images, PDFs, Markdown/text and Office files go "
    "to the agent, and files the agent links in its answer come back as attachments."
)


def render_command_response(resp: Any) -> str:
    """Text for a gateway CommandResponse envelope (``""`` when it has none)."""
    if not isinstance(resp, dict):
        return ""
    raw = resp.get("result")
    result: dict = raw if isinstance(raw, dict) else {}
    if resp.get("ok") is False:
        return f"⚠️ {resp.get('error') or result.get('text') or 'command failed'}"
    text = str(result.get("text") or "").strip()
    if text:
        return text
    data = result.get("data")
    if isinstance(data, dict):
        return "\n".join(f"{key}: {value}" for key, value in data.items() if not isinstance(value, (dict, list)))
    return ""


def _prompt_text(user: Any) -> str:
    """The text part of a prompt (a string, or a list of content blocks)."""
    if isinstance(user, str):
        return user.strip()
    if isinstance(user, list):
        return "\n".join(
            str(block.get("text") or "")
            for block in user
            if isinstance(block, dict) and block.get("type", "text") == "text"
        ).strip()
    return ""


def _error_text(error: Any) -> str:
    if not error:
        return ""
    if isinstance(error, dict):
        error = error.get("message") or error.get("error") or error
    text = str(error).strip()
    return text if len(text) <= _ERROR_CHARS else text[: _ERROR_CHARS - 1] + "…"


def _safe_filename(name: str, fallback: str) -> str:
    """A display-only base name: no directories or control characters."""
    base = os.path.basename(str(name or "").replace("\\", "/"))
    base = "".join(ch for ch in base if ch.isprintable()).strip()[:200]
    return base if base and base not in {".", ".."} else fallback


def _document_name(msg: InboundMessage) -> str:
    """The sender's file name, or ``document<ext>`` guessed from the MIME type."""
    guessed = mimetypes.guess_extension((msg.mime or "").split(";", 1)[0].strip()) or ""
    return _safe_filename(msg.filename, f"document{guessed}")


def _fmt_duration(seconds: float) -> str:
    minutes = max(0, int(seconds)) // 60
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


@dataclass(eq=False)
class _Awaiting:
    """A submitted WhatsApp message waiting for the turn that answers it."""

    msg: InboundMessage
    prompt_text: str
    start_offset: int | None = None
    queued: bool = False
    submitted: bool = False
    queue_id: str = ""
    injected_turn: int | None = None
    drained: bool = False


class WhatsAppChannel:
    """Routes WhatsApp self-chat messages to sharedagents and delivers every turn back."""

    def __init__(
        self,
        cfg: WhatsAppConfig,
        transport: Transport,
        gateway: GatewayClient,
        state: ChannelState,
        *,
        stt: SttFn | None = None,
        log: logging.Logger | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.cfg = cfg
        self.transport = transport
        self.gateway = gateway
        self.state = state
        self._stt = stt
        self._log = log or logging.getLogger("sharedagents.whatsapp")
        self._clock = clock
        self._lock = asyncio.Lock()
        self._observers: dict[str, SessionObserver] = {}
        self._sid_chat: dict[str, str] = {}
        self._awaiting: dict[str, deque[_Awaiting]] = {}
        self._by_queue: dict[str, _Awaiting] = {}
        self._held: list[tuple[str, InboundMessage, list]] = []
        self._turn_msg: dict[tuple[str, int], InboundMessage] = {}
        self._turn_meta: dict[tuple[str, int], dict] = {}
        self._headered: set[tuple[str, int]] = set()
        self._pending: dict[str, Pending] = {}
        self._pending_sid: dict[str, str] = {}
        self._last_stop = float("-inf")
        self._stale_skipped = 0
        self._stale_task: asyncio.Task | None = None
        self._outbound_task: asyncio.Task | None = None
        self._sent_parts: dict[str, tuple[list[str], int]] = {}  # delivery → frozen parts, progress
        self._last_question: dict[str, str] = {}
        self._last_inbound: dict[str, InboundMessage] = {}
        self._last_inbound_by_chat: dict[str, InboundMessage] = {}
        self._started_at = clock()

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        """Connect the transport, then resume observing every bound session."""
        await self.transport.start(self.handle_inbound)
        if not await self.transport.wait_ready(READY_TIMEOUT_S):
            if getattr(self.transport, "failed", None) is not None and self.transport.failed.is_set():
                return
            if not self.transport.fatal.is_set():
                raise TimeoutError("WhatsApp did not become ready")
            return
        async with self._lock:
            for chat, agent, sid in self.state.all_bindings():
                try:
                    exists = await self.gateway.session_exists(sid)
                except Exception as exc:  # noqa: BLE001 — unsure ≠ gone: keep it
                    self._log.warning("could not verify session %s (%s); keeping it", sid[:8], exc)
                    exists = True
                if not exists:
                    self._log.info("session %s is gone; the next message starts a new one", sid[:8])
                    self.state.unbind_session(chat, agent)
                    continue
                self._start_observer(chat, sid)
            self.state.save()
        if self._outbound_task is None or self._outbound_task.done():
            self._outbound_task = asyncio.create_task(self._outbound_loop())
        if getattr(self.transport, "on_history", None) is None:
            self.transport.on_history = self.replay_unreplied
        await self.replay_unreplied()

    async def drain_outbound(self) -> int:
        """Send the lines and files queued by the WhatsApp skill.

        One bad job (a refused dest, a vanished file) is logged and skipped
        instead of dropping the rest of the batch. While WhatsApp reconnects,
        the unsent remainder — minus any text already sent — is queued again.
        """
        from sharedagents_whatsapp.outbound import requeue_outbound, take_outbound

        jobs = take_outbound(self.cfg.home)
        for index, job in enumerate(jobs):
            dest = job.get("dest") or None
            text = str(job.get("text") or "")
            files = list(job.get("files") or [])
            try:
                if text:
                    digest = hashlib.sha1(f"{dest}\0{text}".encode()).hexdigest()[:16]
                    await self._send(text, key=f"queued:{digest}", dest=dest)
                    text = ""
                while files:
                    try:
                        await self._send_queued_file(files[0], dest)
                    except (TransportUnavailable, asyncio.CancelledError):
                        raise
                    except Exception:  # noqa: BLE001 — one bad file, not its siblings
                        self._log.exception(
                            "queued WhatsApp file %s failed; skipped",
                            files[0].get("name") or files[0].get("path"),
                        )
                    files.pop(0)
            except asyncio.CancelledError:
                # Stopping mid-send: keep the unsent rest for the next run.
                rest = [{**job, "text": text, "files": files}] if text or files else []
                requeue_outbound(self.cfg.home, rest + jobs[index + 1 :])
                raise
            except TransportUnavailable:
                # Reconnecting: keep it (and everything after it) for the next
                # drain, ahead of anything queued meanwhile.
                rest = [{**job, "text": text, "files": files}] if text or files else []
                requeue_outbound(self.cfg.home, rest + jobs[index + 1 :])
                return index
            except Exception:  # noqa: BLE001
                self._log.exception("queued WhatsApp send %d/%d failed; skipped", index + 1, len(jobs))
        return len(jobs)

    async def _send_queued_file(self, spec: dict, dest: Any) -> None:
        from sharedagents_whatsapp.outbound import validate_send_file

        path, reason = validate_send_file(str(spec.get("path") or ""), self.cfg.max_media_bytes)
        if path is None:
            self._log.warning("queued WhatsApp file not sent: %s", reason)
            return
        data = await asyncio.to_thread(path.read_bytes)
        name = _safe_filename(str(spec.get("name") or ""), path.name)
        mimetype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        msg_id = await self._send_file(data, name, mimetype, dest)
        if msg_id:
            self.state.mark_processed(msg_id)
            self.state.save()

    async def _outbound_loop(self) -> None:
        while True:
            try:
                await self.drain_outbound()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                self._log.exception("outbound self-chat send failed")
            await asyncio.sleep(0.4)

    async def stop(self) -> None:
        if self._outbound_task is not None:
            self._outbound_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._outbound_task
            self._outbound_task = None
        if self._stale_task is not None:
            self._stale_task.cancel()
        for sid in list(self._observers):
            await self._stop_observer(sid)
        self.state.save()
        with contextlib.suppress(Exception):
            await self.transport.stop()
        with contextlib.suppress(Exception):
            await self.gateway.aclose()

    def _start_observer(self, chat: str, sid: str) -> None:
        self._sid_chat[sid] = chat
        if sid in self._observers:
            return
        observer = SessionObserver(
            self.gateway,
            sid,
            self,
            start_offset=self.state.cursor_for(sid),
            delivery_mode=self.cfg.delivery_mode,
            max_age_s=self.cfg.max_message_age_s,
            clock=self._clock,
            log=self._log,
        )
        self._observers[sid] = observer
        observer.start()

    async def _stop_observer(self, sid: str) -> None:
        observer = self._observers.pop(sid, None)
        if observer is not None:
            await observer.stop()
        self._forget_session(sid)

    def _forget_session(self, sid: str) -> None:
        for entry in list(self._awaiting.get(sid, ())):
            if entry.queue_id:
                self._by_queue.pop(entry.queue_id, None)
        self._awaiting.pop(sid, None)
        self._held = [h for h in self._held if h[0] != sid]
        for table in (self._turn_msg, self._turn_meta):
            for key in [k for k in table if k[0] == sid]:
                table.pop(key, None)
        self._headered = {k for k in self._headered if k[0] != sid}
        chat = self._sid_chat.pop(sid, None)
        self._last_inbound.pop(sid, None)
        if chat is not None and self._pending_sid.get(chat) == sid:
            self._pending.pop(chat, None)
            self._pending_sid.pop(chat, None)

    def _agent_for(self, chat: str) -> str:
        return self.state.explicit_agent_for(chat) or self.cfg.routed_agent(chat) or self.state.agent_for(chat)

    def _model_for(self, chat: str) -> str:
        return self.cfg.routed_model(chat)

    @staticmethod
    def _ident_of(jid: Any) -> str:
        user = str(getattr(jid, "User", "") or "")
        server = str(getattr(jid, "Server", "") or "")
        return f"{user}@{server}" if server else user

    def _actor_for(self, msg: InboundMessage) -> dict[str, str]:
        """A stable, display-safe author for one WhatsApp-originated turn.

        WhatsApp has no email identity to hand to the Web UI.  Its raw JID can
        be a phone number, so retain only a short one-way digest as the actor
        id and render the approved allowlist label instead.  ``from_me`` is
        the direction signal: the linked account is the sender when true;
        otherwise ``msg.sender`` identifies the external sender (especially
        important in groups).
        """
        if msg.from_me:
            identity = "self"
            account_name = getattr(self.transport, "account_name", lambda: "")()
            label = str(account_name or "").strip() or "Linked WhatsApp account"
        else:
            identity = self._ident_of(msg.sender)
            sender_group = identity.endswith("@g.us")
            row = self.cfg.entry(identity, group=sender_group) or {}
            # A private chat's route is keyed by the same JID as its sender.
            # Group senders are not necessarily allowlisted themselves, so a
            # group label is intentionally not misrepresented as a person.
            label = str(row.get("label") or "").strip() or "Unknown contact"
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        return {
            "email": f"whatsapp:{digest}",
            "name": f"Whatsapp: {label}",
            "via": "whatsapp",
        }

    def _command_text(self, msg: InboundMessage, text: str) -> str:
        """Return a slash command, including one directly after an @mention.

        Mention-mode conversations commonly send ``@bot /new``.  Reply policy
        correctly accepts that message, but treating the whole string as a
        prompt means the local command is never reached.  Keep mentions on
        ordinary prompts (they provide useful context), and strip one or more
        configured leading mention tokens only when the remaining text starts
        with ``/``.
        """
        if text.startswith("/"):
            return text
        from sharedagents_core.reply_policy import mention_tokens

        ident, is_group = self._chat_ident(msg)
        row = self.cfg.entry(ident, group=is_group) or {}
        tokens = {token.lower() for token in mention_tokens(row)}
        rest = text
        while rest:
            head, sep, tail = rest.partition(" ")
            if head.lower() not in tokens or not sep:
                break
            rest = tail.lstrip()
        return rest if rest.startswith("/") else ""

    # -- inbound -----------------------------------------------------------

    async def handle_inbound(self, msg: InboundMessage) -> None:
        """Transport callback: one message at a time, in arrival order."""
        self._record_if_allowlisted(msg)
        async with self._lock:
            token = _REPLY_TO.set(msg)
            try:
                await self._handle(msg)
            except Exception:  # noqa: BLE001
                self._log.exception("handling WhatsApp message %s failed", msg.msg_id)
                with contextlib.suppress(Exception):
                    await self._send(
                        "⚠️ Internal error while handling that message — see the sharedagents-whatsapp log."
                    )
            finally:
                _REPLY_TO.reset(token)

    async def replay_unreplied(self) -> int:
        """On restart, handle the last unreplied allowlisted message per chat."""
        from sharedagents_whatsapp.history import last_unreplied

        self._refresh_allowlist()
        replayed = 0
        for rec in last_unreplied(self.cfg.home):
            if rec.get("is_self_chat"):
                continue
            msg = _msg_from_history(rec)
            async with self._lock:
                before = self.state.is_processed(msg.msg_id)
                token = _REPLY_TO.set(msg)
                try:
                    await self._handle(msg, replay=True)
                finally:
                    _REPLY_TO.reset(token)
                if self.state.is_processed(msg.msg_id) and not before:
                    replayed += 1
        if replayed:
            self._log.info("replayed %s unreplied WhatsApp message(s) from history", replayed)
        return replayed

    def _record_if_allowlisted(self, msg: InboundMessage) -> None:
        if msg.is_self_chat or msg.kind == "other" or not msg.msg_id:
            return
        self._refresh_allowlist()
        if not self._readable(msg):
            return
        from sharedagents_whatsapp.history import record

        ident, is_group = self._chat_ident(msg)
        record(
            self.cfg.home,
            ident,
            {
                "id": msg.msg_id,
                "chat": ident,
                "sender": self._ident_of(msg.sender),
                "text": msg.text,
                "quoted": msg.quoted,
                "kind": msg.kind,
                "ts": msg.timestamp,
                "from_me": msg.from_me,
                "from_own_device": msg.from_own_device,
                "is_self_chat": False,
                "is_group": is_group,
                "replied": False,
            },
            self.cfg.lid_aliases,
            group=is_group,
        )

    def _mark_replied(self, msg: InboundMessage) -> None:
        from sharedagents_whatsapp.history import mark_replied

        ident, is_group = self._chat_ident(msg)
        if not ident or msg.is_self_chat:
            return
        mark_replied(self.cfg.home, ident, msg.msg_id, self.cfg.lid_aliases, group=is_group)

    def _session_chat(self, msg: InboundMessage) -> str:
        ident, _ = self._chat_ident(msg)
        return ident if ident and not msg.is_self_chat else SELF_CHAT

    async def _handle(self, msg: InboundMessage, *, replay: bool = False) -> None:
        if not self._accept(msg, replay=replay):
            return
        self.state.mark_processed(msg.msg_id)
        self.state.save()
        chat = self._session_chat(msg)
        text = msg.text.strip()
        pending = self._pending.get(chat)
        command = self._command_text(msg, text) if msg.kind == "text" else ""
        if command:
            name, _, arg = command[1:].partition(" ")
            name = name.strip().lower()
            if name and (pending is None or name in _ALWAYS_COMMANDS):
                await self._command(chat, name, arg.strip(), command)
                return
        if pending is not None and msg.kind in ("text", "voice"):
            reply = text if msg.kind == "text" else await self._transcribe(msg)
            if reply is not None:
                await self._answer_pending(msg, chat, pending, reply)
            return
        await self._prompt(msg, chat)

    def _refresh_allowlist(self) -> None:
        """Pick up Channels tab edits without restarting ``whatsapp run``.

        The stamp check, the reload, and the invalid-file WARNING all live in
        :meth:`WhatsAppConfig.refresh_allowlist` so the transport gate reuses
        the same code path.
        """
        self.cfg.refresh_lid_aliases()
        self.cfg.refresh_allowlist()

    def _accept(self, msg: InboundMessage, *, replay: bool = False) -> bool:
        """Which messages may start work at all."""
        self._refresh_allowlist()
        if msg.from_own_device:
            return False
        if msg.is_self_chat and msg.from_me:
            pass
        elif not self._readable(msg):
            return False
        if not msg.msg_id or (not replay and self.state.is_processed(msg.msg_id)):
            return False
        prefix = self.cfg.reply_prefix
        if prefix and msg.text.startswith(prefix):
            return False
        if msg.kind == "other":
            return False
        if not replay and self._clock() - msg.timestamp > self.cfg.max_message_age_s:
            self.state.mark_processed(msg.msg_id)
            self._note_stale()
            return False
        if not msg.is_self_chat:
            from sharedagents_core.reply_policy import should_reply

            ident, is_group = self._chat_ident(msg)
            row = self.cfg.entry(ident, group=is_group) or {}
            if not should_reply(
                row,
                is_private=not is_group,
                addressed_to_me=False,
                text=msg.text,
            ):
                return False
        return True

    def _chat_ident(self, msg: InboundMessage) -> tuple[str, bool]:
        chat = msg.chat
        user = str(getattr(chat, "User", "") or "")
        server = str(getattr(chat, "Server", "") or "")
        ident = f"{user}@{server}" if server else user
        return ident, server == "g.us" or ident.endswith("@g.us")

    def _readable(self, msg: InboundMessage) -> bool:
        ident, is_group = self._chat_ident(msg)
        return self.cfg.allows(ident, group=is_group, access="read")

    def _note_stale(self) -> None:
        self._stale_skipped += 1
        if self._stale_task is None or self._stale_task.done():
            self._stale_task = contextvars.Context().run(
                asyncio.create_task,
                self._stale_notice(),
            )

    async def _stale_notice(self) -> None:
        await asyncio.sleep(STALE_NOTICE_DELAY_S)
        count, self._stale_skipped = self._stale_skipped, 0
        if count:
            minutes = max(1, round(self.cfg.max_message_age_s / 60))
            await self._send(
                f"⏭️ Skipped {count} message(s) older than {minutes} min (sent while "
                "sharedagents was offline) — resend if still needed."
            )

    # -- prompts -----------------------------------------------------------

    def _has_open_turn(self, sid: str) -> bool:
        return any(key[0] == sid for key in self._turn_msg) or any(key[0] == sid for key in self._turn_meta)

    @staticmethod
    def _is_image_blocks(blocks: list) -> bool:
        return any(isinstance(b, dict) and b.get("type") == "image" for b in blocks)

    async def _prompt(self, msg: InboundMessage, chat: str) -> None:
        # A document can take a while to download and read: acknowledge first.
        early_ack = msg.kind == "document"
        if early_ack:
            await self._react(msg, self.cfg.ack_reaction)
        blocks = await self._blocks_for(msg)
        if not blocks:
            if early_ack:
                await self._react(msg, FAILED)
            return
        if not early_ack:
            await self._react(msg, self.cfg.ack_reaction)
        agent = self._agent_for(chat)
        sid = self.state.session_for(chat, agent)
        if sid and self._is_image_blocks(blocks) and self._has_open_turn(sid):
            self._held.append((sid, msg, blocks))
            await self._send("⏳ Still busy — I'll send that image when the current turn finishes.")
            return
        await self._submit_blocks(msg, chat, blocks, ack=False)

    async def _submit_blocks(
        self,
        msg: InboundMessage,
        chat: str,
        blocks: list,
        *,
        ack: bool = True,
    ) -> None:
        if ack:
            await self._react(msg, self.cfg.ack_reaction)
        agent = self._agent_for(chat)
        entry = _Awaiting(msg, _prompt_text(blocks))
        try:
            sid = await self._ensure_session(chat, agent)
            asked = " ".join(str(msg.text or "").split())
            if asked and not is_internal_prompt(asked):
                self._last_question[sid] = asked
                self._last_inbound[sid] = msg
                self._last_inbound_by_chat[chat] = msg
            self._awaiting.setdefault(sid, deque()).append(entry)
            try:
                result = await self._submit_with_actor(sid, blocks, msg)
            except SessionNotFound:
                self._forget(entry)
                await self._stop_observer(sid)
                self.state.unbind_session(chat, agent)
                self.state.save()
                sid = await self._ensure_session(chat, agent)
                if asked and not is_internal_prompt(asked):
                    self._last_question[sid] = asked
                    self._last_inbound[sid] = msg
                    self._last_inbound_by_chat[chat] = msg
                await self._send("🆕 The previous session was gone — started a new session.")
                self._awaiting.setdefault(sid, deque()).append(entry)
                result = await self._submit_with_actor(sid, blocks, msg)
        except (GatewayError, httpx.HTTPError) as exc:
            self._forget(entry)
            await self._react(msg, FAILED)
            detail = (
                exc.message
                if isinstance(exc, GatewayError)
                else "sharedagents gateway unreachable — try again shortly."
            )
            await self._send(f"⚠️ {detail}")
            return
        entry.start_offset = result.start_offset
        entry.queued = result.queued
        entry.queue_id = result.queue_id
        entry.submitted = True
        self._mark_replied(msg)
        if result.queue_id:
            self._by_queue[result.queue_id] = entry
        if result.queued and result.queue_id and self._is_image_blocks(blocks):
            await self.gateway.remove_queued(sid, result.queue_id)
            self._forget(entry)
            self._by_queue.pop(result.queue_id, None)
            self._held.append((sid, msg, blocks))
            return

    async def _flush_held(self, sid: str) -> None:
        held = [item for item in self._held if item[0] == sid]
        self._held = [item for item in self._held if item[0] != sid]
        for _sid, msg, blocks in held:
            await self._submit_blocks(msg, SELF_CHAT, blocks, ack=False)

    def _forget(self, entry: _Awaiting) -> None:
        if entry.queue_id:
            self._by_queue.pop(entry.queue_id, None)
        for queue in self._awaiting.values():
            with contextlib.suppress(ValueError):
                queue.remove(entry)

    async def _ensure_session(self, chat: str, agent: str) -> str:
        sid = self.state.session_for(chat, agent)
        if sid is None:
            sid = await self.gateway.create_session(agent)
            self.state.bind_session(chat, agent, sid)
            self.state.save()
        model = self._model_for(chat)
        if model and self.state.model_for_session(sid) != model:
            response = await self.gateway.command(agent, f"/model {model}", sid)
            if not isinstance(response, dict) or response.get("ok") is not True:
                detail = render_command_response(response)
                self._log.warning(
                    "Configured model %r not found for agent %s on session %s (%s); using default model",
                    model,
                    agent,
                    sid,
                    detail,
                )
                response = await self.gateway.command(agent, "/model default", sid)
                if not isinstance(response, dict) or response.get("ok") is not True:
                    detail = render_command_response(response)
                    raise GatewayError(
                        409,
                        "default_model_failed",
                        detail.removeprefix("⚠️ ") or "could not apply default model",
                    )
            self.state.set_session_model(sid, model)
            self.state.save()
        self._start_observer(chat, sid)
        return sid

    async def _submit_with_actor(
        self,
        sid: str,
        blocks: list[dict],
        msg: InboundMessage,
    ) -> Any:
        """Submit with actor provenance when the gateway exposes that option.

        The signature guard keeps lightweight gateway doubles and a rolling
        upgrade's older client seam working; a real gateway failure remains a
        ``GatewayError`` and is never mistaken for a compatibility fallback.
        """
        submit = self.gateway.submit
        try:
            params = inspect.signature(submit).parameters.values()
            supports_actor = any(
                param.name == "actor" or param.kind is inspect.Parameter.VAR_KEYWORD for param in params
            )
        except (TypeError, ValueError):
            supports_actor = True
        if supports_actor:
            return await submit(sid, blocks, actor=self._actor_for(msg))
        return await submit(sid, blocks)

    def _speaker_context(self, msg: InboundMessage) -> str:
        """Which chat this is and who wrote the message, as a prompt preamble.

        Every chat routed to an agent shares that agent's memory, so without
        it the agent guesses the speaker from memory notes about another
        contact. The self-chat is the owner talking to the agent: no header.
        """
        if msg.is_self_chat:
            return ""
        from sharedagents_session.actors import clean_display_name

        ident, is_group = self._chat_ident(msg)
        row = self.cfg.entry(ident, group=is_group) or {}
        chat = clean_display_name(row.get("label")) or ("this group" if is_group else "this contact")
        if msg.from_me:
            owner = "the owner of this WhatsApp account"
            name = clean_display_name(getattr(self.transport, "account_name", lambda: "")())
            sender = f"{name}, {owner}" if name else owner
        elif is_group:
            sender_ident = self._ident_of(msg.sender)
            member = self.cfg.entry(sender_ident) or {}
            sender = clean_display_name(member.get("label"))
            if not sender:
                sender = "a group member"
                if sender_ident and not member:
                    sender += " who is not on the allowlist"
        else:
            sender = chat
        where = f"WhatsApp group {chat}" if is_group else f"WhatsApp chat with {chat}"
        return f"[{where} — message from {sender}]\n"

    def _quote_context(self, msg: InboundMessage) -> str:
        """The earlier message this one replies to, as a prompt preamble.

        Without it the agent sees only the reply ("why is this important?")
        and none of what it refers to. Our own reply prefix is not part of
        the quote.
        """
        quoted = str(msg.quoted or "").strip()
        prefix = self.cfg.reply_prefix
        if prefix and quoted.startswith(prefix):
            quoted = quoted[len(prefix) :].strip()
        if not quoted:
            return ""
        body = "\n".join(f"> {line}" for line in quoted.splitlines())
        return f"[Replying to this earlier WhatsApp message]\n{body}\n\n"

    async def _blocks_for(self, msg: InboundMessage) -> list[dict]:
        """Prompt content blocks for one message."""
        context = self._speaker_context(msg) + self._quote_context(msg)
        if msg.kind == "text":
            return [{"type": "text", "text": context + msg.text}] if msg.text.strip() else []
        if msg.kind in ("voice", "audio"):
            transcript = await self._transcribe(msg)
            return [{"type": "text", "text": f"{context}[voice note] {transcript}"}] if transcript else []
        if msg.kind == "image":
            data = await self._download(msg)
            if data is None:
                return []
            blocks: list[dict] = []
            if msg.text.strip() or context:
                blocks.append({"type": "text", "text": context + msg.text})
            blocks.append(
                {
                    "type": "image",
                    "data": base64.b64encode(data).decode("ascii"),
                    "mimeType": msg.mime or "image/jpeg",
                }
            )
            return blocks
        if msg.kind == "document":
            blocks = await self._document_blocks(msg)
            if blocks and context and not msg.is_self_chat:
                blocks.insert(0, {"type": "text", "text": context.rstrip("\n")})
            return blocks
        await self._send(
            f"⚠️ {msg.kind.capitalize()} messages aren't supported yet — send text, a voice note, {_READABLE_KINDS}."
        )
        return []

    async def _document_blocks(self, msg: InboundMessage) -> list[dict]:
        """A document as prompt content: an image block, or its text.

        The file type is checked before downloading; after it, the bytes
        decide — an image signature makes an image block whatever the name
        says, and anything else must be a PDF/Office/Markdown/text file.
        """
        name = _document_name(msg)
        maybe_image = msg.mime.lower().startswith("image/") or name.lower().endswith(_IMAGE_SUFFIXES)
        if not maybe_image and not is_readable_document(name):
            await self._send(f"⚠️ {name}: I can't read that file type — send {_READABLE_KINDS}.")
            return []
        data = await self._download(msg)
        if data is None:
            return []
        caption = msg.text.strip()
        image_mime = sniff_image_mime(data)
        if image_mime:
            blocks: list[dict] = [{"type": "text", "text": caption}] if caption else []
            blocks.append(
                {
                    "type": "image",
                    "data": base64.b64encode(data).decode("ascii"),
                    "mimeType": image_mime,
                }
            )
            return blocks
        if not is_readable_document(name):  # image-like, but not a format models read
            await self._send(f"⚠️ {name}: that image format can't be read — send it as PNG, JPEG, WebP or GIF.")
            return []
        try:
            text = await asyncio.wait_for(
                asyncio.to_thread(document_to_markdown, data, name),
                DOCUMENT_CONVERT_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            self._log.warning("reading %s took over %ss; skipped", name, DOCUMENT_CONVERT_TIMEOUT_S)
            await self._send(
                f"⚠️ {name} took too long to read (over {DOCUMENT_CONVERT_TIMEOUT_S:g} s) — "
                "send a smaller file or just the pages you need."
            )
            return []
        if text is None:
            await self._send(f"⚠️ Couldn't read {name} — is it a valid, unencrypted file?")
            return []
        text = text.strip()
        clipped, truncated = clip_text(text, DOCUMENT_CHAR_LIMIT)
        body = f"[document: {name}]\n\n{clipped}"
        if truncated:
            body += f"\n\n[… truncated: showing the first {DOCUMENT_CHAR_LIMIT:,} of {len(text):,} characters]"
        return [{"type": "text", "text": f"{caption}\n\n{body}" if caption else body}]

    async def _download(self, msg: InboundMessage) -> bytes | None:
        limit = self.cfg.max_media_bytes
        too_large = f"⚠️ That {msg.kind} is too large (limit {self.cfg.max_media_mb:g} MB)."
        if msg.size and msg.size > limit:
            await self._send(too_large)
            return None
        try:
            data = await self.transport.download_media(msg)
        except Exception as exc:  # noqa: BLE001
            self._log.warning("media download failed for %s: %s", msg.msg_id, exc)
            await self._send(f"⚠️ Couldn't download that {msg.kind}.")
            return None
        if len(data) > limit:
            await self._send(too_large)
            return None
        return data

    async def _transcribe(self, msg: InboundMessage) -> str | None:
        if self._stt is None:
            await self._send(
                '⚠️ Voice notes need speech-to-text — add an "stt" block '
                "(primary/backup) to ~/.sharedagents/config.json."
            )
            return None
        data = await self._download(msg)
        if data is None:
            return None
        try:
            transcript = await self._stt(data, mime=msg.mime, filename="voice.ogg")
        except Exception as exc:  # noqa: BLE001
            if type(exc).__name__ == "TranscriptionUnavailableError":
                await self._send("⚠️ Couldn't decode that voice note (ffmpeg unavailable or unsupported audio).")
            else:
                self._log.warning("STT failed for %s: %s", msg.msg_id, exc)
                await self._send("⚠️ Couldn't transcribe that voice note — try again or type it.")
            return None
        transcript = (transcript or "").strip()
        if not transcript:
            await self._send("⚠️ No speech detected in that voice note.")
            return None
        return transcript

    # -- commands ----------------------------------------------------------

    async def _command(self, chat: str, name: str, arg: str, raw: str) -> None:
        if name == "help":
            await self._send(HELP_TEXT)
        elif name == "new":
            await self._cmd_new(chat)
        elif name in ("stop", "cancel"):
            await self._cmd_stop(chat)
        elif name == "agent":
            await self._cmd_agent(chat, arg)
        elif name == "status":
            await self._cmd_status(chat)
        else:
            await self._forward_command(chat, raw)

    async def _cmd_new(self, chat: str) -> None:
        agent = self._agent_for(chat)
        sid = self.state.unbind_session(chat, agent)
        if sid:
            await self._stop_observer(sid)
        self.state.save()
        self._pending.pop(chat, None)
        self._pending_sid.pop(chat, None)
        await self._send(f"🆕 **New session** — your next message starts fresh with **{agent}**.")

    async def _cmd_stop(self, chat: str) -> None:
        sid = self.state.session_for(chat)
        if not sid:
            await self._send("Nothing to stop.")
            return
        now = self._clock()
        force = now - self._last_stop <= FORCE_STOP_WINDOW_S
        # A forced stop re-arms the cycle: the next /stop is a soft one again.
        self._last_stop = float("-inf") if force else now
        try:
            await self.gateway.cancel(sid, force=force)
        except (GatewayError, httpx.HTTPError) as exc:
            await self._send(f"⚠️ Stop failed: {exc}")
            return
        # /stop cancels a parked gate; don't later claim it was answered elsewhere.
        if self._pending_sid.get(chat) == sid:
            self._pending.pop(chat, None)
            self._pending_sid.pop(chat, None)
        await self._send(
            "⏹️ Force-stopping the turn." if force else "⏹️ Stopping… (send /stop again within 10 s to force)"
        )

    async def _cmd_agent(self, chat: str, arg: str) -> None:
        current = self._agent_for(chat)
        target = arg.split()[0] if arg else ""
        if not target:
            await self._send(f"Current agent: **{current}**. Switch with /agent <name>.")
            return
        if target == current:
            await self._send(f"Already talking to **{current}**.")
            return
        sid = self.state.session_for(chat, target)
        resumed = sid is not None
        if sid is None:
            try:
                sid = await self.gateway.create_session(target)
            except GatewayError as exc:
                await self._send(f"⚠️ Couldn't switch to {target}: {exc.message}")
                return
            except httpx.HTTPError:
                await self._send("⚠️ sharedagents gateway unreachable — try again shortly.")
                return
            self.state.bind_session(chat, target, sid)
        # The previous agent's session stays observed: work it delegated can
        # still finish there and must reach WhatsApp.
        self.state.set_agent(chat, target)
        self.state.save()
        self._start_observer(chat, sid)
        how = "resumed its session" if resumed else "new session"
        await self._send(f"🔀 Now talking to **{target}** ({how}).")

    async def _cmd_status(self, chat: str) -> None:
        agent = self._agent_for(chat)
        sid = self.state.session_for(chat)
        pending = self._pending.get(chat)
        waiting = (
            "question" if isinstance(pending, PendingQuestion) else "permission" if pending is not None else "none"
        )
        lines = [
            "**sharedagents WhatsApp**",
            f"account: {self.transport.account_label()}",
            f"agent: {agent}",
            f"session: {sid[:8] if sid else '(none yet)'}",
            f"waiting for you: {waiting}",
            f"uptime: {_fmt_duration(self._clock() - self._started_at)}",
        ]
        if sid:
            try:
                server = render_command_response(
                    await self.gateway.command(agent, "/status", sid),
                )
            except Exception as exc:  # noqa: BLE001
                server = f"(agent status unavailable: {exc})"
            if server:
                lines += ["", server]
        await self._send("\n".join(lines))

    async def _forward_command(self, chat: str, raw: str) -> None:
        agent = self._agent_for(chat)
        sid = self.state.session_for(chat, agent)
        command = raw.split()[0]
        try:
            if command.lower() == "/model" and not sid:
                sid = await self.gateway.create_session(agent)
                self.state.bind_session(chat, agent, sid)
                self.state.save()
            response = await self.gateway.command(agent, raw, sid)
        except (GatewayError, httpx.HTTPError) as exc:
            await self._send(f"⚠️ {command} failed: {exc}")
            return
        if command.lower() == "/model" and isinstance(response, dict) and response.get("ok"):
            parts = raw.split(maxsplit=1)
            if len(parts) > 1 and sid:
                # Consume the current route, so the next prompt preserves
                # this explicit switch. A later route edit still applies.
                self.state.set_session_model(sid, self._model_for(chat))
                self.state.save()
        await self._send(render_command_response(response) or f"{command}: done")

    # -- parked questions / permissions ---------------------------------------

    async def _answer_pending(
        self,
        msg: InboundMessage,
        chat: str,
        pending: Pending,
        reply: str,
    ) -> None:
        sid = self._pending_sid.get(chat, "")
        try:
            if isinstance(pending, PendingQuestion):
                answer = parse_answer(pending.current, reply)
                if answer is None:
                    count = len(pending.current.options)
                    await self._send(
                        f"Reply with a number 1–{count} or type your answer." if count else "Type your answer."
                    )
                    return
                pending.answers.append(answer)
                if not pending.done:
                    await self._send(
                        render_question(
                            pending.current,
                            len(pending.answers) + 1,
                            len(pending.questions),
                        )
                    )
                    return
                try:
                    await self.gateway.answer(sid, pending.call_id, pending.answers)
                except (GatewayError, httpx.HTTPError):
                    pending.answers.pop()
                    raise
            else:
                option = parse_permission_reply(pending, reply)
                if option is None:
                    await self._send(f"Reply 1–{len(pending.options)} (or yes / no).")
                    return
                await self.gateway.permission(
                    sid,
                    pending.request_id,
                    None if option == CANCEL else option,
                )
        except SessionNotFound:
            self._pending.pop(chat, None)
            self._pending_sid.pop(chat, None)
            await self._send("That prompt is no longer waiting for an answer.")
            return
        except (GatewayError, httpx.HTTPError) as exc:
            await self._send(f"⚠️ Couldn't deliver your answer: {exc}")
            return
        self._pending.pop(chat, None)
        self._pending_sid.pop(chat, None)
        await self._react(msg, ANSWERED)

    # -- observer sink -------------------------------------------------------

    async def on_turn_started(self, sid: str, turn_idx: int, offset: int, event: dict) -> None:
        # Submit and its SSE events can race. Read the authoritative response
        # before pairing, including whether the prompt was queued.
        async with self._lock:
            self._pair_turn(sid, turn_idx, offset, event)

    def _pair_turn(self, sid: str, turn_idx: int, offset: int, event: dict) -> None:
        """Pair a starting turn with the WhatsApp message that caused it, if any.

        A non-queued submit's turn is the next ``turn_started`` after its
        ``start_offset`` (the gateway serializes a session's turns), and while
        the submit is still in flight only that one message can be pending.
        Anything else is an external turn (web/TUI, background wake-up).
        """
        prompt = _prompt_text(event.get("user"))
        source = str(event.get("source") or "")
        queue = self._awaiting.get(sid)
        match: _Awaiting | None = None
        if queue and source == "queue_drain":
            match = next(
                (e for e in queue if e.queued and e.drained and e.prompt_text == prompt),
                None,
            )
            if match is None:
                match = next((e for e in queue if e.queued and e.drained), None)
            if match is not None:
                queue.remove(match)
                if match.queue_id:
                    self._by_queue.pop(match.queue_id, None)
                self._turn_msg[(sid, turn_idx)] = match.msg
                return
        if queue:
            candidates = [entry for entry in queue if not entry.queued]
            match = next(
                (e for e in candidates if e.start_offset is not None and offset > e.start_offset),
                None,
            )
            if match is not None:
                queue.remove(match)
        if match is not None:
            self._turn_msg[(sid, turn_idx)] = match.msg
        else:
            self._turn_meta[(sid, turn_idx)] = {
                "source": source,
                "prompt": prompt,
            }

    def _with_header(
        self,
        sid: str,
        turn_idx: int,
        text: str,
        event: dict | None = None,
    ) -> str:
        """Prefix a turn's first delivery with where it came from.

        A WhatsApp message's own turn on the current agent needs no header. A
        turn in another bound agent's session is tagged with that agent; an
        external turn (web/TUI, background wake-up) with its source and prompt —
        taken from ``turn_started``, or from ``turn_ended`` when the start was
        never seen (the turn ran while WhatsApp was offline).
        """
        key = (sid, turn_idx)
        if key in self._headered:
            return text
        chat = self._sid_chat.get(sid, SELF_CHAT)
        owner = self.state.agent_of(chat, sid)
        tags: list[str] = []
        if owner and owner != self._agent_for(chat):
            tags.append(owner)
        preview = ""
        if key not in self._turn_msg:
            meta = self._turn_meta.get(key)
            if meta is None and event is not None:
                meta = {
                    "source": str(event.get("source") or ""),
                    "prompt": _prompt_text(event.get("user")),
                }
            if meta is not None:
                preview = " ".join(str(meta.get("prompt") or "").split())
                if is_internal_prompt(preview):
                    preview = ""
                else:
                    tags.append(str(meta.get("source") or "other interface"))
        dest = self._dest_for_sid(sid, turn_idx)
        if dest is not None or not tags:
            return text
        if len(preview) > _PREVIEW_CHARS:
            preview = preview[: _PREVIEW_CHARS - 1] + "…"
        header = f"↪️ ({' · '.join(tags)})" + (f" {preview}" if preview else "")
        return f"{header}\n\n{text}"

    def _reply_dest(self, msg: InboundMessage | None) -> Any:
        if msg is None or msg.is_self_chat:
            return None
        ident, is_group = self._chat_ident(msg)
        if ident and self.cfg.allows(ident, group=is_group, access="send"):
            return msg.chat
        return None

    def _dest_for_sid(self, sid: str, turn_idx: int | None = None) -> Any:
        if turn_idx is not None:
            dest = self._reply_dest(self._turn_msg.get((sid, turn_idx)))
            if dest is not None:
                return dest
        chat = self._sid_chat.get(sid, SELF_CHAT)
        if not chat or chat == SELF_CHAT:
            return None
        user, _, server = str(chat).partition("@")
        if not user:
            return None
        from types import SimpleNamespace

        fake = SimpleNamespace(User=user, Server=server)
        if self.cfg.allows(chat, group=server == "g.us", access="send"):
            return fake
        return None

    def _quote_for_sid(self, sid: str, turn_idx: int | None = None) -> InboundMessage | None:
        inbound = self._last_inbound.get(sid)
        if inbound is None:
            chat = self._sid_chat.get(sid)
            if chat:
                inbound = self._last_inbound_by_chat.get(chat)
        if turn_idx is not None:
            msg = self._turn_msg.get((sid, turn_idx))
            if msg is not None and getattr(msg, "raw", None) is not None:
                return msg
            if inbound is not None and getattr(inbound, "raw", None) is not None:
                return inbound
            if msg is not None:
                return msg
        return inbound

    async def on_block(self, sid: str, turn_idx: int, text: str) -> None:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
        await self._send(
            self._with_header(sid, turn_idx, text),
            key=f"block:{sid}:{turn_idx}:{digest}",
            dest=self._dest_for_sid(sid, turn_idx),
            quote=self._quote_for_sid(sid, turn_idx),
        )
        self._headered.add((sid, turn_idx))

    async def on_artifact(self, sid: str, turn_idx: int, event: dict) -> None:
        """Send a file the turn recorded, where that turn's text goes.

        PNG/JPEG/WebP bytes go as a photo; everything else (PDF, Markdown, a
        GIF, a spreadsheet…) as a document carrying its name and type. A file
        that is gone or over ``max_media_mb`` gets one notice instead — those
        are permanent, so they are not retried. Transport errors propagate:
        the observer retries them within its delivery budget.
        """
        digest = str(event.get("sha256") or "").lower()
        name = _safe_filename(str(event.get("display_name") or ""), f"attachment-{digest[:8]}")
        dest = self._dest_for_sid(sid, turn_idx)
        quote = self._quote_for_sid(sid, turn_idx)
        limit = self.cfg.max_media_bytes
        key = f"artifact:{sid}:{digest}"
        try:
            declared = int(event.get("size_bytes") or 0)
        except (TypeError, ValueError):
            declared = 0
        too_large = (
            f"📎 {name} is too large to send here (limit {self.cfg.max_media_mb:g} MB) — "
            "open it in the sharedagents web UI."
        )
        if declared > limit:
            await self._send(too_large, key=key, dest=dest, quote=quote)
            return
        try:
            artifact = await self.gateway.fetch_artifact(sid, digest, max_bytes=limit)
        except ArtifactUnavailable as exc:
            if exc.status == 413:
                await self._send(too_large, key=key, dest=dest, quote=quote)
            else:
                self._log.warning("artifact %s of %s is unavailable: %s", digest[:12], sid[:8], exc)
                await self._send(
                    f"📎 Couldn't attach {name} — it is no longer available.",
                    key=key,
                    dest=dest,
                    quote=quote,
                )
            return
        if name.startswith("attachment-") and artifact.filename:
            name = _safe_filename(artifact.filename, name)
        mimetype = (
            str(event.get("media_type") or "").strip()
            or sniff_image_mime(artifact.data)
            or artifact.content_type
            or "application/octet-stream"
        )
        msg_id = await self._send_file(artifact.data, name, mimetype, dest, quote)
        if msg_id:
            self.state.mark_processed(msg_id)
            self.state.save()

    async def on_turn_ended(self, sid: str, turn_idx: int, event: dict, delivered: bool) -> None:
        async with self._lock:
            await self._finish_turn(sid, turn_idx, event, delivered)

    async def _finish_turn(self, sid: str, turn_idx: int, event: dict, delivered: bool) -> None:
        key = (sid, turn_idx)
        msg = self._turn_msg.get(key)
        error = _error_text(event.get("error"))
        aborted = bool(event.get("aborted"))
        parts: list[str] = []
        if not delivered:
            final = strip_runtime_notices(str(event.get("final_content") or "")).strip()
            if final:
                parts.append(final)
            elif msg is not None and not error and not aborted:
                parts.append("_(no text reply)_")
        if error:
            parts.append(f"⚠️ {error}")
        elif aborted:
            parts.append("⏹️ stopped")
        if parts:
            await self._send(
                self._with_header(sid, turn_idx, "\n\n".join(parts), event),
                key=f"end:{sid}:{turn_idx}",
                dest=self._dest_for_sid(sid, turn_idx),
                quote=self._quote_for_sid(sid, turn_idx),
            )
        self._turn_msg.pop(key, None)
        prefix = f"block:{sid}:{turn_idx}:"
        for stale in [k for k in self._sent_parts if k.startswith(prefix)]:
            self._sent_parts.pop(stale, None)
        outcome = FAILED if error else STOPPED if aborted else DONE
        if msg is not None:
            await self._react(msg, outcome)
        queue = self._awaiting.get(sid)
        if queue:
            for entry in [e for e in queue if e.queued]:
                if entry.queue_id and entry.injected_turn != turn_idx:
                    continue
                queue.remove(entry)
                if entry.queue_id:
                    self._by_queue.pop(entry.queue_id, None)
                await self._react(entry.msg, outcome)
        self._turn_meta.pop(key, None)
        self._headered.discard(key)
        chat = self._sid_chat.get(sid, SELF_CHAT)
        if self._pending_sid.get(chat) == sid:
            self._pending.pop(chat, None)
            self._pending_sid.pop(chat, None)
        await self._flush_held(sid)

    async def on_question(self, sid: str, call_id: str, arguments: Any) -> None:
        questions = parse_questions(arguments)
        if not questions:
            await self._send("❓ The agent asked something WhatsApp can't render — answer it in the web UI.")
            return
        chat = self._sid_chat.get(sid, SELF_CHAT)
        async with self._lock:
            note = self._displaced_note(chat, sid)
            await self._send(
                note + render_question(questions[0], 1, len(questions)),
                key=f"question:{sid}:{call_id}",
                dest=self._dest_for_sid(sid),
                quote=self._quote_for_sid(sid),
            )
            self._pending[chat] = PendingQuestion(call_id, questions)
            self._pending_sid[chat] = sid

    async def on_permission(self, sid: str, event: dict) -> None:
        request_id = str(event.get("request_id") or "")
        if not request_id:
            return
        pending = PendingPermission(
            request_id,
            [str(option) for option in (event.get("options") or []) if option],
            str(event.get("summary") or event.get("tool_name") or ""),
            str(event.get("detail") or ""),
        )
        chat = self._sid_chat.get(sid, SELF_CHAT)
        async with self._lock:
            note = self._displaced_note(chat, sid)
            await self._send(
                note + render_permission(pending),
                key=f"permission:{sid}:{request_id}",
                dest=self._dest_for_sid(sid),
                quote=self._quote_for_sid(sid),
            )
            self._pending[chat] = pending
            self._pending_sid[chat] = sid

    def _displaced_note(self, chat: str, sid: str) -> str:
        """Say so when a new prompt takes the reply slot from another session's."""
        previous_sid = self._pending_sid.get(chat)
        if self._pending.get(chat) is None or not previous_sid or previous_sid == sid:
            return ""
        owner = self.state.agent_of(chat, previous_sid) or "another session"
        return f"_(An earlier prompt from {owner} is still waiting — answer that one in the web UI.)_\n\n"

    async def on_interaction_resolved(self, sid: str, key: str) -> None:
        async with self._lock:
            chat = self._sid_chat.get(sid, SELF_CHAT)
            pending = self._pending.get(chat)
            if pending is not None and self._pending_sid.get(chat) == sid and pending.key == key:
                self._pending.pop(chat, None)
                self._pending_sid.pop(chat, None)
                await self._send("✓ Answered from another interface.")

    async def on_cursor(self, sid: str, offset: int) -> None:
        self.state.set_cursor(sid, offset)
        self.state.save()

    async def on_skipped(self, sid: str, turns: int) -> None:
        minutes = max(1, round(self.cfg.max_message_age_s / 60))
        await self._send(
            f"⏭️ {turns} turn(s) finished more than {minutes} min ago while WhatsApp "
            "was offline — not replayed; see the web UI."
        )

    async def on_queue_event(self, sid: str, event: dict) -> None:
        async with self._lock:
            qid = str(event.get("queue_id") or "")
            entry = self._by_queue.get(qid)
            etype = event.get("type")
            if etype == "queue_msg_injected" and entry is not None:
                idx = event.get("turn_idx")
                if isinstance(idx, int) and not isinstance(idx, bool):
                    entry.injected_turn = idx
            elif etype == "queue_msg_removed" and entry is not None:
                reason = str(event.get("reason") or "")
                if reason == "user_deleted":
                    self._forget(entry)
                    await self._react(entry.msg, STOPPED)
                elif reason == "user_drained":
                    entry.drained = True

    async def on_session_gone(self, sid: str) -> None:
        chat = self._sid_chat.get(sid, SELF_CHAT)
        self._observers.pop(sid, None)  # called from the observer's own task
        owner = self.state.agent_of(chat, sid)
        if owner is not None:
            self.state.unbind_session(chat, owner)
            self.state.save()
        self._forget_session(sid)
        await self._send("🆕 The previous session is gone — your next message starts a new session.")

    # -- outbound ------------------------------------------------------------

    async def _send(
        self,
        markdown: str,
        *,
        key: str | None = None,
        dest: Any = _DEST_UNSET,
        quote: InboundMessage | None = None,
    ) -> None:
        """Send one reply to self-chat, or to the asker's allowlisted chat.

        Without ``dest``, a handler replies to the chat of the message it is
        handling (self-chat when that chat is not send-allowlisted); an explicit
        ``dest=None`` always means self-chat.

        Raises when WhatsApp refuses a part, so the observer can retry the
        delivery; ``key`` names a retried delivery, whose parts that already
        went out are skipped instead of repeated. Outbound message ids are
        recorded as processed so an echo can never start a turn.
        """
        if dest is _DEST_UNSET:
            dest = self._reply_dest(_REPLY_TO.get())
        cached = self._sent_parts.get(key) if key else None
        markdown = strip_internal_envelopes(markdown)
        if not markdown.strip() or is_internal_prompt(markdown):
            return
        parts, start = cached if cached is not None else (render_messages(markdown, self.cfg.reply_prefix), 0)
        if key:
            self._sent_parts[key] = (parts, start)
        for index in range(start, len(parts)):
            msg_id = await self.transport.send_text(parts[index], dest, quote)
            if key:
                self._sent_parts[key] = (parts, index + 1)
            if msg_id:
                self.state.mark_processed(msg_id)
                self.state.save()
        if key:
            self._sent_parts.pop(key, None)

    async def _send_file(
        self,
        data: bytes,
        name: str,
        mimetype: str,
        dest: Any,
        quote: Any = None,
    ) -> str | None:
        """A photo when WhatsApp shows it inline (PNG/JPEG/WebP), else a document.

        A photo the client cannot process (e.g. it fails to thumbnail it)
        still arrives, as a document; link and allowlist errors propagate.
        """
        if sniff_image_mime(data) in INLINE_IMAGE_MIMES:
            try:
                return await self.transport.send_image(data, dest, quote)
            except (TransportUnavailable, PermissionError):
                raise
            except Exception as exc:  # noqa: BLE001
                self._log.warning("sending %s as a photo failed (%s); sending a document", name, exc)
        return await self.transport.send_document(
            data,
            filename=name,
            mimetype=mimetype,
            dest=dest,
            quote=quote,
        )

    async def _react(self, msg: InboundMessage, emoji: str) -> None:
        if not self.cfg.ack_reaction or not emoji:
            return
        try:
            await self.transport.react(msg, emoji)
        except Exception as exc:  # noqa: BLE001
            self._log.debug("reaction failed on %s: %s", msg.msg_id, exc)
