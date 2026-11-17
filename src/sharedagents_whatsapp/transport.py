"""The seam between a WhatsApp protocol client and the channel core.

The channel only ever sees :class:`InboundMessage` values and talks back through
:class:`Transport`. ``send_text``/``send_image``/``send_document`` default to the
linked self-chat. A ``dest`` is allowed only for a send-allowlisted person or
group (the chat that asked).
It is also the test seam: the channel is exercised with an in-memory fake.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol


class TransportUnavailable(RuntimeError):
    """WhatsApp is not connected right now (still starting or reconnecting).

    Deliveries that hit this are retried until the link is back, without
    counting against the observer's give-up budget.
    """


KINDS = frozenset({"text", "voice", "audio", "image", "video", "document", "sticker", "other"})


@dataclass(frozen=True)
class InboundMessage:
    """One normalized incoming WhatsApp message.

    ``chat``/``sender`` are opaque transport handles; ``raw`` is the transport's
    own message object (needed to download media). ``filename`` is a document's
    name as the sender attached it (``""`` for every other kind). ``quoted`` is the
    text of the earlier message this one replies to (``""`` when it is not a reply).
    """

    msg_id: str
    chat: Any
    sender: Any
    timestamp: float
    is_self_chat: bool
    from_me: bool
    from_own_device: bool
    kind: str
    text: str = ""
    mime: str = ""
    size: int = 0
    raw: Any = None
    filename: str = ""
    quoted: str = ""


OnMessage = Callable[[InboundMessage], Awaitable[None]]


class Transport(Protocol):
    """What the channel needs from a WhatsApp client."""

    fatal: asyncio.Event
    fatal_reason: str

    async def start(self, on_message: OnMessage) -> None: ...
    async def wait_ready(self, timeout: float) -> bool: ...
    async def send_text(
        self,
        text: str,
        dest: Any = None,
        quote: Any = None,
    ) -> str | None: ...
    async def send_image(
        self,
        data: bytes,
        dest: Any = None,
        quote: Any = None,
        caption: str = "",
    ) -> str | None: ...
    async def send_document(
        self,
        data: bytes,
        *,
        filename: str,
        mimetype: str,
        dest: Any = None,
        quote: Any = None,
        caption: str = "",
    ) -> str | None: ...
    async def react(self, msg: InboundMessage, emoji: str) -> None: ...
    async def download_media(self, msg: InboundMessage) -> bytes: ...
    async def stop(self) -> None: ...
    def account_label(self) -> str: ...
    def account_name(self) -> str: ...
