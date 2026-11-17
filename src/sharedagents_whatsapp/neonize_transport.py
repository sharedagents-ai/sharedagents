"""neonize (Go whatsmeow) implementation of the WhatsApp transport.

The service links to the operator's account as a companion device (like
WhatsApp Web) and shows up as "sharedagents" under *Linked devices*. It sees every
chat of that account, so it normalizes each message, keeps the account's
own self-chat (phone-number or LID addressing) plus read-allowlisted chats,
and drops everything else without logging content. Replies go to the
self-chat or to a send-allowlisted person/group that asked.

neonize is imported lazily so this module (and ``sharedagents whatsapp status``)
works without the optional ``[whatsapp]`` extra. neonize's Go worker thread does
not return after ``stop()``; the CLI entry points therefore finish with
``os._exit`` (see ``cli.py``).
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import os
import re
import sqlite3
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from sharedagents_whatsapp.transport import InboundMessage, OnMessage, TransportUnavailable

EX_CONFIG = 78  # sysexits EX_CONFIG — the unit's RestartPreventExitStatus
PN_SERVER = "s.whatsapp.net"
LID_SERVER = "lid"
_PAIR_SUCCESS = 2  # neonize PairStatus.Status: ERROR=1, SUCCESS=2
_STORE_SUFFIXES = ("", "-wal", "-shm", "-journal")


def mask_user(user: str) -> str:
    """``5511987654321`` → ``+55…21`` (never log or print a full number)."""
    digits = re.sub(r"\D", "", user or "")
    if len(digits) < 4:
        return "(unknown)"
    return f"+{digits[:2]}…{digits[-2:]}"


def linked_account(store_path: Path) -> dict | None:
    """The linked device row from the whatsmeow store, read without starting Go.

    Returns ``{"jid", "lid", "push_name", "user"}`` or ``None`` when the store
    is missing, unreadable or holds no device (never linked / logged out).
    """
    store = Path(store_path)
    if not store.is_file():
        return None
    try:
        con = sqlite3.connect(f"file:{store}?mode=ro", uri=True, timeout=5)
        try:
            row = con.execute(
                "SELECT jid, lid, push_name FROM whatsmeow_device LIMIT 1",
            ).fetchone()
        finally:
            con.close()
    except sqlite3.Error:
        return None
    if not row or not row[0]:
        return None
    jid = str(row[0])
    return {
        "jid": jid,
        "lid": str(row[1] or ""),
        "push_name": str(row[2] or ""),
        "user": jid.split("@", 1)[0].split(":", 1)[0],
    }


def harden_store(home: Path, store_path: Path) -> None:
    """Owner-only permissions: 0700 home, 0600 store (+ its WAL/SHM files).

    The store is a live WhatsApp login; whatsmeow creates it 0644.
    """
    with contextlib.suppress(OSError):
        os.chmod(home, 0o700)
    for suffix in _STORE_SUFFIXES:
        path = Path(f"{store_path}{suffix}")
        if path.exists():
            with contextlib.suppress(OSError):
                os.chmod(path, 0o600)


def remove_store(store_path: Path) -> None:
    for suffix in _STORE_SUFFIXES:
        with contextlib.suppress(OSError):
            Path(f"{store_path}{suffix}").unlink()


def _device_props() -> Any:
    from neonize.proto.waCompanionReg.WAWebProtobufsCompanionReg_pb2 import DeviceProps

    return DeviceProps(os="sharedagents", platformType=DeviceProps.DESKTOP)


def _classify(msg: Any) -> tuple[str, str, str, int]:
    """(kind, text, mime, size) for a waE2E ``Message``."""
    has = msg.HasField
    if has("reactionMessage") or has("protocolMessage") or has("editedMessage"):
        return "other", "", "", 0
    if has("extendedTextMessage"):
        return "text", msg.extendedTextMessage.text or "", "", 0
    if msg.conversation:
        return "text", msg.conversation, "", 0
    if has("imageMessage"):
        m = msg.imageMessage
        return "image", m.caption or "", m.mimetype or "image/jpeg", int(m.fileLength or 0)
    if has("audioMessage"):
        m = msg.audioMessage
        kind = "voice" if m.PTT else "audio"
        return kind, "", m.mimetype or "audio/ogg", int(m.fileLength or 0)
    if has("videoMessage") or has("ptvMessage"):
        m = msg.videoMessage if has("videoMessage") else msg.ptvMessage
        return "video", m.caption or "", m.mimetype or "video/mp4", int(m.fileLength or 0)
    if has("documentMessage"):
        m = msg.documentMessage
        return "document", m.caption or "", m.mimetype or "", int(m.fileLength or 0)
    if has("documentWithCaptionMessage"):
        inner = msg.documentWithCaptionMessage.message
        if inner.HasField("documentMessage"):
            m = inner.documentMessage
            return "document", m.caption or "", m.mimetype or "", int(m.fileLength or 0)
        return "document", "", "", 0
    if has("stickerMessage"):
        return "sticker", "", "", 0
    return "other", "", "", 0


_QUOTE_CHARS = 1500
_QUOTED_MEDIA_LABEL = {
    "image": "[image]",
    "voice": "[voice note]",
    "audio": "[audio]",
    "video": "[video]",
    "document": "[document]",
    "sticker": "[sticker]",
}
_CONTEXT_CARRIERS = (
    "extendedTextMessage",
    "imageMessage",
    "videoMessage",
    "ptvMessage",
    "audioMessage",
    "documentMessage",
    "stickerMessage",
)


def _context_info_of(msg: Any) -> Any:
    """The ``ContextInfo`` WhatsApp attaches to a message (replies carry it)."""
    if msg is None:
        return None
    if msg.HasField("documentWithCaptionMessage"):
        return _context_info_of(msg.documentWithCaptionMessage.message)
    for field in _CONTEXT_CARRIERS:
        if msg.HasField(field):
            carrier = getattr(msg, field)
            return carrier.contextInfo if carrier.HasField("contextInfo") else None
    return None


def _quoted_text_of(msg: Any) -> str:
    """Text of the message ``msg`` replies to, or ``""`` when it is not a reply.

    A quoted message without text (an image, a voice note) becomes a short
    label so the agent still knows what was replied to.
    """
    info = _context_info_of(msg)
    if info is None or not info.HasField("quotedMessage"):
        return ""
    kind, text, _mime, _size = _classify(info.quotedMessage)
    text = text.strip()
    if not text:
        return _QUOTED_MEDIA_LABEL.get(kind, "")
    return text if len(text) <= _QUOTE_CHARS else text[: _QUOTE_CHARS - 1] + "…"


def _document_of(msg: Any) -> Any:
    """The ``DocumentMessage`` of a document (captioned or not), else ``None``."""
    if msg is None:
        return None
    if msg.HasField("documentMessage"):
        return msg.documentMessage
    if msg.HasField("documentWithCaptionMessage"):
        inner = msg.documentWithCaptionMessage.message
        if inner.HasField("documentMessage"):
            return inner.documentMessage
    return None


def _history_record(info: Any, ident: str, is_group: bool) -> dict | None:
    """``WebMessageInfo`` → a history.json row, or None when it has no body."""
    key = getattr(info, "key", None)
    msg_id = str(getattr(key, "ID", "") or "")
    if not msg_id:
        return None
    body = getattr(info, "message", None)
    if body is None:
        return None
    kind, text, _mime, _size = _classify(body)
    if kind == "other":
        return None
    timestamp = float(getattr(info, "messageTimestamp", 0) or 0)
    if timestamp > 1e12:
        timestamp /= 1000.0
    sender = str(getattr(key, "participant", "") or getattr(info, "participant", "") or "")
    if not sender and not is_group:
        sender = ident
    return {
        "id": msg_id,
        "chat": ident,
        "sender": sender,
        "text": text,
        "quoted": _quoted_text_of(body),
        "kind": kind,
        "ts": timestamp,
        "from_me": bool(getattr(key, "fromMe", False)),
        "from_own_device": False,
        "is_self_chat": False,
        "is_group": is_group,
        "replied": False,
    }


class NeonizeTransport:
    """Linked-device WhatsApp client restricted to the account's self-chat."""

    def __init__(self, store_path: Path, *, home: Path, log: logging.Logger | None = None) -> None:
        self._store = Path(store_path)
        self._home = Path(home)
        self._log = log or logging.getLogger("sharedagents.whatsapp.transport")
        self.fatal = asyncio.Event()
        self.fatal_reason = ""
        self.failed = asyncio.Event()
        self.failure_reason = ""
        self._ready = asyncio.Event()
        self._client: Any = None
        self._on_message: OnMessage | None = None
        self.may_read: Callable[[str, bool], bool] | None = None
        self.may_send: Callable[[str, bool], bool] | None = None
        self.on_history: Callable[[], Awaitable[None]] | None = None
        self._me_users: set[str] = set()
        self._me_jids: set[tuple[str, str]] = set()
        self._me_device = 0
        self._pn_user = ""
        self._account_name = ""
        self._self_chat: Any = None

    # -- identity + normalization (pure; unit-tested) ------------------------

    def _set_identity(self, pn_user: str, lid_user: str, device: int) -> None:
        from neonize.proto import Neonize_pb2 as N

        self._pn_user = pn_user
        self._me_users = {u for u in (pn_user, lid_user) if u}
        self._me_jids = {(PN_SERVER, pn_user)} if pn_user else set()
        if lid_user:
            self._me_jids.add((LID_SERVER, lid_user))
        self._me_device = int(device or 0)
        self._self_chat = N.JID(
            User=pn_user,
            Server=PN_SERVER,
            Device=0,
            RawAgent=0,
            Integrator=0,
            IsEmpty=False,
        )

    def _remember_self_address(self, chat: Any) -> None:
        """Keep dest=None sends on the same PN/LID chat the inbound used."""
        from neonize.proto import Neonize_pb2 as N

        user = str(getattr(chat, "User", "") or "")
        server = str(getattr(chat, "Server", "") or "")
        if not user or not self.is_self_chat(chat):
            return
        self._self_chat = N.JID(
            User=user,
            Server=server or PN_SERVER,
            Device=0,
            RawAgent=0,
            Integrator=0,
            IsEmpty=False,
        )

    def is_self_chat(self, jid: Any) -> bool:
        """True for the account's own chat under either addressing mode."""
        return (getattr(jid, "Server", ""), getattr(jid, "User", "")) in self._me_jids

    def normalize(self, ev: Any) -> InboundMessage:
        """neonize ``MessageEv`` → :class:`InboundMessage`."""
        info = ev.Info
        src = info.MessageSource
        sender = src.Sender
        own_device = (
            bool(self._me_device) and sender.User in self._me_users and int(sender.Device or 0) == self._me_device
        )
        kind, text, mime, size = _classify(ev.Message)
        timestamp = float(info.Timestamp or 0)
        if timestamp > 1e12:  # milliseconds
            timestamp /= 1000.0
        is_self = (not src.IsGroup) and self.is_self_chat(src.Chat)
        if is_self:
            self._remember_self_address(src.Chat)
        document = _document_of(ev.Message) if kind == "document" else None
        filename = str(document.fileName or document.title or "") if document is not None else ""
        return InboundMessage(
            msg_id=str(info.ID),
            chat=src.Chat,
            sender=sender,
            timestamp=timestamp,
            is_self_chat=is_self,
            from_me=bool(src.IsFromMe),
            from_own_device=own_device,
            kind=kind,
            text=text,
            mime=mime,
            size=size,
            raw=ev,
            filename=filename,
            quoted=_quoted_text_of(ev.Message),
        )

    def account_label(self) -> str:
        return mask_user(self._pn_user) if self._pn_user else "(not connected)"

    def account_name(self) -> str:
        """The linked account's WhatsApp push name, if available."""
        return self._account_name

    def seed_identity_from_store(self) -> bool:
        """Load the linked JID/LID/device from the store before connect.

        Offline self-chat messages can arrive before ``ConnectedEv``. Without
        this seed they would be dropped as foreign.
        """
        account = linked_account(self._store)
        if not account:
            return False
        self._account_name = str(account.get("push_name") or "").strip()
        jid = account["jid"]
        user = account["user"]
        device = 0
        local = jid.split("@", 1)[0]
        if ":" in local:
            with contextlib.suppress(ValueError):
                device = int(local.split(":", 1)[1])
        lid = account.get("lid") or ""
        lid_user = lid.split("@", 1)[0].split(":", 1)[0] if lid else ""
        self._set_identity(user, lid_user, device)
        return True

    def _on_connect_done(self, task: asyncio.Future) -> None:
        """Unblock ``wait_ready`` when the connect task fails."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return
        self.failure_reason = str(exc)
        self._log.error("WhatsApp connect failed: %s", self.failure_reason)
        self.failed.set()

    # -- lifecycle ------------------------------------------------------------

    def _set_fatal(self, reason: str) -> None:
        if not self.fatal.is_set():
            self.fatal_reason = reason
            self._log.error("WhatsApp: %s", reason)
            self.fatal.set()

    async def start(self, on_message: OnMessage) -> None:
        from neonize.aioze.client import NewAClient
        from neonize.aioze.events import (
            ClientOutdatedEv,
            ConnectedEv,
            DisconnectedEv,
            HistorySyncEv,
            LoggedOutEv,
            MessageEv,
            OfflineSyncCompletedEv,
            StreamReplacedEv,
            TemporaryBanEv,
        )

        self._on_message = on_message
        harden_store(self._home, self._store)
        self.seed_identity_from_store()
        client = NewAClient(str(self._store), props=_device_props())
        self._client = client

        async def _on_qr(_client: Any, _data: bytes) -> None:
            # A QR request means the store holds no linked device: never pair
            # from the unattended service.
            self._set_fatal("not linked — run `sharedagents whatsapp link` first")

        client.event.qr(_on_qr)

        @client.event(ConnectedEv)
        async def _connected(_client: Any, _ev: Any) -> None:
            try:
                await self._on_connected()
            except Exception:  # noqa: BLE001
                self._log.exception("could not read the linked account identity")
                self._set_fatal("could not read the linked account identity")

        @client.event(MessageEv)
        async def _message(_client: Any, ev: Any) -> None:
            await self._dispatch(ev)

        @client.event(HistorySyncEv)
        async def _history(_client: Any, ev: Any) -> None:
            await self._ingest_history(ev)

        @client.event(OfflineSyncCompletedEv)
        async def _offline_done(_client: Any, _ev: Any) -> None:
            if self.on_history is not None:
                try:
                    await self.on_history()
                except Exception:  # noqa: BLE001
                    self._log.exception("WhatsApp history replay failed")

        @client.event(LoggedOutEv)
        async def _logged_out(_client: Any, ev: Any) -> None:
            self._set_fatal(
                f"logged out from the phone (reason {ev.Reason}) — relink with `sharedagents whatsapp link`"
            )

        @client.event(StreamReplacedEv)
        async def _replaced(_client: Any, _ev: Any) -> None:
            self._set_fatal("another client connected with this device's credentials")

        @client.event(TemporaryBanEv)
        async def _banned(_client: Any, ev: Any) -> None:
            self._set_fatal(f"temporarily banned by WhatsApp (code {ev.Code}, expires in {ev.Expire}s)")

        @client.event(ClientOutdatedEv)
        async def _outdated(_client: Any, _ev: Any) -> None:
            self._set_fatal("WhatsApp rejected this client version — upgrade neonize")

        @client.event(DisconnectedEv)
        async def _disconnected(_client: Any, _ev: Any) -> None:
            self._log.info("WhatsApp disconnected; whatsmeow reconnects automatically")

        connect = asyncio.ensure_future(client.connect())
        connect.add_done_callback(self._on_connect_done)
        try:
            await connect
        except Exception:  # noqa: BLE001 — ``_on_connect_done`` records the failure
            return

    async def _on_connected(self) -> None:
        client = self._client
        me = client.me
        if me is None or not me.JID.User:
            me = await client.get_me()
        self._set_identity(me.JID.User, me.LID.User, int(me.JID.Device or 0))
        harden_store(self._home, self._store)
        self._ready.set()
        self._log.info("WhatsApp connected as %s", self.account_label())
        try:
            await asyncio.wait_for(self._sync_group_names(), timeout=20)
        except Exception:  # noqa: BLE001 — directory refresh must not block chat
            self._log.warning("could not refresh WhatsApp group names; keeping the stored directory")

    async def _sync_group_names(self) -> None:
        """Populate the local directory used by the Channels group search."""
        from neonize.proto import Neonize_pb2 as N

        groups = await self._client.get_joined_groups()
        entries = [
            N.ContactEntry(JID=group.JID, FullName=group.GroupName.Name.strip(), FirstName="")
            for group in groups
            if group.JID.Server == "g.us" and group.JID.User and group.GroupName.Name.strip()
        ]
        if entries:
            # ContactStore updates are local; they do not rename remote groups.
            await self._client.contact.put_all_contact_name(entries)

    def _may_read_ident(self, ident: str, is_group: bool) -> bool:
        if not ident:
            return False
        if not is_group:
            user, _, server = ident.partition("@")
            if (server or PN_SERVER, user) in self._me_jids:
                return True
        return self.may_read is not None and self.may_read(ident, is_group)

    def _ingest_history_sync(self, ev: Any) -> int:
        """Persist allowlisted HistorySync conversations. Returns rows stored."""
        from sharedagents_whatsapp.history import record

        stored = 0
        for convo in getattr(ev, "conversations", None) or []:
            ident = str(getattr(convo, "ID", "") or "").strip()
            is_group = ident.endswith("@g.us")
            if not ident or not self._may_read_ident(ident, is_group):
                continue
            for item in getattr(convo, "messages", None) or []:
                info = getattr(item, "message", None)
                if info is None:
                    continue
                rec = _history_record(info, ident, is_group)
                if rec is None:
                    continue
                record(self._home, ident, rec, group=is_group)
                stored += 1
        return stored

    async def _ingest_history(self, ev: Any) -> None:
        try:
            stored = self._ingest_history_sync(ev)
        except Exception:  # noqa: BLE001
            self._log.warning("could not ingest a WhatsApp history sync")
            return
        if stored:
            self._log.info("stored %s WhatsApp history message(s)", stored)
            if self.on_history is not None:
                try:
                    await self.on_history()
                except Exception:  # noqa: BLE001
                    self._log.exception("WhatsApp history replay failed")

    async def _dispatch(self, ev: Any) -> None:
        try:
            src = ev.Info.MessageSource
            chat = src.Chat
            ident = f"{getattr(chat, 'User', '')}@{getattr(chat, 'Server', '')}".strip("@")
            is_group = bool(src.IsGroup)
            readable = (not is_group and self.is_self_chat(chat)) or (
                self.may_read is not None and self.may_read(ident, is_group)
            )
            if not readable:
                # Identity fields only — never the message body — so a dropped
                # chat is diagnosable without leaking its content.
                self._log.debug(
                    "dropped a message from non-allowlisted %s %s@%s",
                    "group" if is_group else "chat",
                    mask_user(getattr(chat, "User", "") or ""),
                    getattr(chat, "Server", "") or "?",
                )
                return
            msg = self.normalize(ev)
        except Exception:  # noqa: BLE001
            # Parse errors may contain payload text from an untrusted chat.
            self._log.warning("could not parse an incoming WhatsApp message")
            return
        if not msg.is_self_chat:
            ident = f"{getattr(msg.chat, 'User', '')}@{getattr(msg.chat, 'Server', '')}".strip("@")
            is_group = getattr(msg.chat, "Server", "") == "g.us"
            if self.may_read is None or not self.may_read(ident, is_group):
                self._log.debug("ignored a message outside the read allowlist")
                return
        if self._on_message is None:
            return
        try:
            await self._on_message(msg)
        except Exception:  # noqa: BLE001
            self._log.exception("channel failed on WhatsApp message %s", msg.msg_id)

    async def wait_ready(self, timeout: float) -> bool:
        ready = asyncio.ensure_future(self._ready.wait())
        fatal = asyncio.ensure_future(self.fatal.wait())
        failed = asyncio.ensure_future(self.failed.wait())
        try:
            await asyncio.wait(
                {ready, fatal, failed},
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for task in (ready, fatal, failed):
                task.cancel()
        return self._ready.is_set() and not self.fatal.is_set() and not self.failed.is_set()

    async def _require_client(self) -> Any:
        if self._client is None or self._self_chat is None:
            raise TransportUnavailable("WhatsApp transport is not connected")
        connected = getattr(self._client, "is_connected", True)
        # neonize 0.5.2's aio client exposes a coroutine through this property.
        if inspect.isawaitable(connected):
            connected = await connected
        if not connected:
            raise TransportUnavailable("WhatsApp is reconnecting")
        return self._client

    def _jid_for_dest(self, dest: Any) -> Any:
        from neonize.proto import Neonize_pb2 as N

        if dest is None:
            return self._self_chat
        user = str(getattr(dest, "User", "") or "")
        server = str(getattr(dest, "Server", "") or "")
        if not user and isinstance(dest, str):
            user, _, server = dest.partition("@")
        if not user:
            return self._self_chat
        return N.JID(
            User=user,
            Server=server or PN_SERVER,
            Device=0,
            RawAgent=0,
            Integrator=0,
            IsEmpty=False,
        )

    def _may_send_ident(self, ident: str, is_group: bool) -> bool:
        if not ident:
            return False
        return self.may_send is not None and self.may_send(ident, is_group)

    @staticmethod
    def _is_quote_event(raw: Any) -> bool:
        return raw is not None and hasattr(raw, "Info") and hasattr(raw, "Message")

    def _quoted_text(self, text: str, quote: Any) -> Any:
        """Build an ExtendedTextMessage with native reply-to ContextInfo.

        WhatsApp only shows the quoted bubble for ``extendedTextMessage`` +
        ``contextInfo``. A plain ``conversation`` string is dropped as a
        normal chat line — that is what the operator saw at 18:50 / 18:51.
        """
        from neonize.proto.waE2E.WAWebProtobufsE2E_pb2 import (
            ContextInfo,
            ExtendedTextMessage,
            Message,
        )

        raw = getattr(quote, "raw", None)
        stanza = ""
        quoted_body = str(getattr(quote, "text", "") or "")
        participant = self._participant_of(getattr(quote, "sender", None))
        if self._is_quote_event(raw):
            stanza = str(getattr(raw.Info, "ID", "") or "")
            src = raw.Info.MessageSource
            participant = self._participant_of(src.Sender, getattr(src, "SenderAlt", None))
            inner = raw.Message
            quoted_body = (
                (getattr(inner, "conversation", None) or "")
                or (getattr(getattr(inner, "extendedTextMessage", None), "text", None) or "")
                or quoted_body
            )
        stanza = stanza or str(getattr(quote, "msg_id", "") or "")
        return Message(
            extendedTextMessage=ExtendedTextMessage(
                text=text,
                contextInfo=ContextInfo(
                    stanzaID=stanza,
                    participant=participant,
                    quotedMessage=Message(conversation=quoted_body),
                ),
            )
        )

    def _pn_participant(self) -> str:
        return f"{self._pn_user}@{PN_SERVER}" if self._pn_user else ""

    def _participant_of(self, sender: Any, alt: Any = None) -> str:
        chosen = sender
        try:
            from neonize.utils.jid import Jid2String, JIDToNonAD, jid_is_lid

            if sender is not None and hasattr(sender, "SerializeToString") and jid_is_lid(sender):
                if alt is not None and getattr(alt, "ListFields", lambda: [])():
                    chosen = alt
            if chosen is not None and hasattr(chosen, "SerializeToString"):
                user = str(getattr(chosen, "User", "") or "")
                if user in self._me_users and self._pn_user:
                    return self._pn_participant()
                return Jid2String(JIDToNonAD(chosen))
        except Exception:  # noqa: BLE001 — history rows are SimpleNamespace
            chosen = sender
        user = str(getattr(chosen, "User", "") or "")
        server = str(getattr(chosen, "Server", "") or PN_SERVER)
        if user in getattr(self, "_me_users", set()) and self._pn_user:
            return self._pn_participant()
        return f"{user}@{server}" if user else self._pn_participant()

    def _guard(self, target: Any) -> None:
        """Allow the self-chat or a send-allowlisted chat; refuse anything else."""
        if self.is_self_chat(target):
            return
        ident = f"{target.User}@{target.Server}".strip("@")
        is_group = getattr(target, "Server", "") == "g.us"
        if not self._may_send_ident(ident, is_group):
            raise PermissionError("refusing to send outside the send allowlist")
        self._log.info("WhatsApp reply sent to %s", mask_user(target.User))

    def _target_for(self, dest: Any, quote: Any) -> Any:
        raw = getattr(quote, "raw", None) if quote is not None else None
        if dest is not None:
            if self._is_quote_event(raw):
                return raw.Info.MessageSource.Chat
            chat = getattr(quote, "chat", None)
            if chat is not None:
                return self._jid_for_dest(chat)
            return self._jid_for_dest(dest)
        chat = getattr(quote, "chat", None)
        if chat is not None and self.is_self_chat(chat):
            return self._jid_for_dest(chat)
        return self._self_chat

    async def send_text(
        self,
        text: str,
        dest: Any = None,
        quote: Any = None,
    ) -> str | None:
        """Send ``text`` to self-chat, or to a send-allowlisted dest.

        When ``quote`` carries the inbound neonize event, use neonize's
        ``reply_message`` (stanzaID + full quoted Message + participant).
        Keep self-chat replies in the inbound address namespace so the
        destination and the quoted message identify the same conversation.
        """
        client = await self._require_client()
        target = self._target_for(dest, quote)
        raw = getattr(quote, "raw", None) if quote is not None else None
        self._guard(target)
        if self._is_quote_event(raw) and hasattr(client, "reply_message"):
            self._log.info(
                "WhatsApp native reply-to id=%s via=reply_message to=%s@%s",
                getattr(quote, "msg_id", ""),
                mask_user(getattr(target, "User", "") or ""),
                getattr(target, "Server", "") or "?",
            )
            response = await client.reply_message(text, raw, to=target)
        elif quote is not None and str(getattr(quote, "msg_id", "") or ""):
            self._log.info(
                "WhatsApp native reply-to id=%s via=contextinfo to=%s@%s",
                quote.msg_id,
                mask_user(getattr(target, "User", "") or ""),
                getattr(target, "Server", "") or "?",
            )
            response = await client.send_message(target, self._quoted_text(text, quote))
        else:
            response = await client.send_message(target, text)
        return str(getattr(response, "ID", "") or "") or None

    async def react(self, msg: InboundMessage, emoji: str) -> None:
        """React on the prompting message: self-chat or a send-allowlisted chat."""
        ident = f"{getattr(msg.chat, 'User', '')}@{getattr(msg.chat, 'Server', '')}".strip("@")
        is_group = getattr(msg.chat, "Server", "") == "g.us"
        allowed = (msg.is_self_chat and self.is_self_chat(msg.chat)) or (self._may_send_ident(ident, is_group))
        if not allowed:
            raise PermissionError("refusing to react outside the send allowlist")
        client = await self._require_client()
        reaction = await client.build_reaction(msg.chat, msg.sender, msg.msg_id, emoji)
        await client.send_message(msg.chat, reaction)

    async def send_image(
        self,
        data: bytes,
        dest: Any = None,
        quote: Any = None,
        caption: str = "",
    ) -> str | None:
        """Send a photo to self-chat or a send-allowlisted dest (same guard as text)."""
        client = await self._require_client()
        target = self._target_for(dest, quote)
        self._guard(target)
        response = await client.send_image(
            target,
            data,
            caption=caption or None,
            quoted=self._media_quote(quote),
        )
        return str(getattr(response, "ID", "") or "") or None

    async def send_document(
        self,
        data: bytes,
        *,
        filename: str,
        mimetype: str,
        dest: Any = None,
        quote: Any = None,
        caption: str = "",
    ) -> str | None:
        """Send a file as a WhatsApp document (name + type shown to the reader)."""
        client = await self._require_client()
        target = self._target_for(dest, quote)
        self._guard(target)
        response = await client.send_document(
            target,
            data,
            caption=caption or None,
            filename=filename,
            mimetype=mimetype or None,
            quoted=self._media_quote(quote),
        )
        return str(getattr(response, "ID", "") or "") or None

    def _media_quote(self, quote: Any) -> Any:
        """The live inbound event to reply to; media never hand-builds a quote."""
        raw = getattr(quote, "raw", None) if quote is not None else None
        return raw if self._is_quote_event(raw) else None

    async def download_media(self, msg: InboundMessage) -> bytes:
        client = await self._require_client()
        message = msg.raw.Message
        if message.HasField("documentWithCaptionMessage"):
            inner = message.documentWithCaptionMessage.message
            if inner.HasField("documentMessage"):
                message = inner
        return await client.download_any(message)

    async def stop(self) -> None:
        client, self._client = self._client, None
        if client is None:
            return
        with contextlib.suppress(Exception):
            await asyncio.wait_for(client.disconnect(), 5)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(client.stop(), 5)


async def link_device(
    store_path: Path,
    *,
    home: Path,
    phone: str | None,
    qr_png: Path,
    timeout: float = 180.0,
    out: Callable[[str], None] = print,
) -> tuple[int, str]:
    """Pair this host as a linked device. Returns ``(exit_code, detail)``.

    QR mode renders the code in the terminal and rewrites ``qr_png`` on every
    rotation; phone mode requests an 8-character pairing code for ``phone``.
    """
    import segno
    from neonize.aioze.client import NewAClient
    from neonize.aioze.events import ConnectedEv, PairStatusEv

    digits = re.sub(r"\D", "", phone or "")
    if phone is not None and len(digits) < 8:
        return 1, "use the international format, e.g. --phone +5511987654321"
    client = NewAClient(str(store_path), props=_device_props())
    done = asyncio.Event()
    outcome: dict[str, Any] = {"code": 1, "detail": "timed out waiting for the phone"}
    requested = False

    async def _on_qr(c: Any, data: bytes) -> None:
        nonlocal requested
        if digits:
            if requested:
                return
            requested = True
            try:
                code = await c.PairPhone(digits, True)
            except Exception as exc:  # noqa: BLE001
                outcome.update(code=1, detail=f"pairing-code request failed: {exc}")
                done.set()
                return
            pretty = f"{code[:4]}-{code[4:]}" if len(code) == 8 else code
            out(
                f"Pairing code: {pretty}\n"
                "On your phone: WhatsApp → Settings → Linked devices → Link a device → "
                "Link with phone number instead, then enter the code."
            )
            return
        qr = segno.make_qr(data)
        qr.save(str(qr_png), scale=8, border=4)
        with contextlib.suppress(OSError):
            os.chmod(qr_png, 0o600)
        qr.terminal(compact=True)
        out(
            "Scan it: WhatsApp → Settings → Linked devices → Link a device.\n"
            f"(Also saved to {qr_png}; the code refreshes about every 20 s.)"
        )

    client.event.qr(_on_qr)

    @client.event(PairStatusEv)
    async def _paired(_c: Any, ev: Any) -> None:
        if ev.Status == _PAIR_SUCCESS:
            outcome.update(code=0, detail=mask_user(ev.ID.User))
            # Let the post-pair reconnect finish its first login before we stop.
            asyncio.get_running_loop().call_later(20.0, done.set)
        else:
            outcome.update(code=1, detail=f"pairing failed: {ev.Error or 'unknown error'}")
            done.set()

    @client.event(ConnectedEv)
    async def _connected(_c: Any, _ev: Any) -> None:
        account = linked_account(store_path)
        if account:
            outcome.update(code=0, detail=mask_user(account["user"]))
            done.set()

    try:
        await client.connect()
        await asyncio.wait_for(done.wait(), timeout)
    except asyncio.TimeoutError:
        pass
    finally:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(client.disconnect(), 5)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(client.stop(), 5)
        with contextlib.suppress(OSError):
            Path(qr_png).unlink()
        harden_store(home, store_path)
    return int(outcome["code"]), str(outcome["detail"])


async def logout_device(store_path: Path, *, home: Path, timeout: float = 30.0) -> bool:
    """Unlink this device on WhatsApp's side, then delete the local store.

    Returns True when the remote logout went through; the local credentials are
    removed either way.
    """
    if linked_account(store_path) is None:
        remove_store(store_path)
        return False
    from neonize.aioze.client import NewAClient
    from neonize.aioze.events import ConnectedEv

    client = NewAClient(str(store_path), props=_device_props())
    connected = asyncio.Event()
    unlinked = False

    async def _on_qr(_c: Any, _data: bytes) -> None:
        connected.set()  # the server no longer knows this device

    client.event.qr(_on_qr)

    @client.event(ConnectedEv)
    async def _on_connected(_c: Any, _ev: Any) -> None:
        connected.set()

    try:
        await client.connect()
        await asyncio.wait_for(connected.wait(), timeout)
        logged_in = client.is_logged_in
        if callable(logged_in):
            logged_in = logged_in()
        if inspect.isawaitable(logged_in):
            logged_in = await logged_in
        if logged_in:
            await client.logout()
            unlinked = True
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("sharedagents.whatsapp.transport").warning("remote logout failed: %s", exc)
    finally:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(client.disconnect(), 5)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(client.stop(), 5)
        remove_store(store_path)
        harden_store(home, store_path)
    return unlinked
