"""Telegram Bot API HTTP client — extracted from ``bot.py`` (de-monolith).

:class:`_TelegramAPI` owns the ``httpx.AsyncClient`` and builds the bot-token URL
lazily so the token never enters a stack trace or repr. It is used by both
:class:`~sharedagents_telegram.bot.AcpTelegramBot` (the long-poll loop) and the
streaming renderers in :mod:`sharedagents_telegram.rendering`.

``bot.py`` re-exports :class:`_TelegramAPI` (and :func:`_retry_after_s`) so
callers/tests importing them from ``sharedagents_telegram.bot`` keep working. The
shared new-message pacing constants + text helpers stay owned by ``bot.py`` and
are imported at the BOTTOM of this module (see the note there) so the
``bot`` <-> ``api`` import cycle is deadlock-free in either import order.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import httpx

# API-owned: default value of ``_TelegramAPI.__init__(api_base=...)`` — evaluated
# at class-definition time, so it must exist before the class body executes and
# therefore can't be bottom-imported. ``bot.py`` re-imports this name from here.
TELEGRAM_API = "https://api.telegram.org"


def _retry_after_s(res: dict) -> float:
    """Extract Telegram's explicit ``retry_after`` (seconds) from a failed Bot
    API response, or ``0.0`` if it is not a rate-limit reply. A 429 carries it
    in ``parameters.retry_after``; mirrors the edit-path 429 handling."""
    if not isinstance(res, dict) or res.get("ok"):
        return 0.0
    params = res.get("parameters")
    if isinstance(params, dict):
        ra = params.get("retry_after")
        if isinstance(ra, (int, float)) and ra > 0:
            return float(ra)
    return 0.0


class _TelegramAPI:
    """Minimal Telegram Bot API client.

    Owns the ``httpx.AsyncClient`` and builds the bot-token URL lazily so
    the token never enters a stack trace or repr. Used by both
    :class:`AcpTelegramBot` (for the long-poll loop) and
    :class:`_RollingMessage` (for streaming edits).
    """

    def __init__(
        self,
        token: str,
        *,
        api_base: str = TELEGRAM_API,
        log: logging.Logger | None = None,
    ) -> None:
        self._token = token
        self._api_base = api_base.rstrip("/")
        # "local mode" = pointed at a self-hosted telegram-bot-api server
        # (not the public cloud). In local mode getFile returns an
        # absolute on-disk path and the 20 MB download cap is lifted.
        self.local_mode = not self._api_base.startswith("https://api.telegram.org")
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
        self._log = log or logging.getLogger("sharedagents.telegram")
        # Per-chat next-allowed new-message send time (monotonic). Paces the
        # turn's burst of placeholder/🔧/💬 sends under Telegram's per-chat send
        # rate limit. See ``_await_send_slot`` / ``NEW_MSG_MIN_GAP_S``.
        self._chat_send_next: dict[int, float] = {}
        # Bot-wide (cross-chat) next-allowed send time (monotonic). The ONE
        # ``_TelegramAPI`` per bot is the shared point across every chat, so this
        # instance slot is the aggregate floor. See ``NEW_MSG_GLOBAL_MIN_GAP_S``.
        self._global_send_next: float = 0.0

    async def close(self) -> None:
        await self._http.aclose()

    async def call(self, method: str, **params: Any) -> dict:
        """Call a Telegram Bot API method. Telegram-side errors come
        back as ``{"ok": false, ...}``; we surface them to the caller
        without raising so most callers can log-and-continue.
        """
        url = f"{self._api_base}/bot{self._token}/{method}"
        try:
            r = await self._http.post(url, json=params)
        except httpx.HTTPError as e:
            self._log.warning("telegram %s: transport error: %s", method, e)
            return {"ok": False, "description": str(e)}
        try:
            return r.json()
        except json.JSONDecodeError:
            self._log.warning(
                "telegram %s: non-json response (status=%s): %s",
                method,
                r.status_code,
                r.text[:300],
            )
            return {"ok": False, "description": f"http {r.status_code}"}

    async def _await_send_slot(self, chat_id: int) -> None:
        """Pace new-message sends to stay under Telegram's send rate limits.

        TWO floors apply, and a send may fire only once BOTH are clear:

        * the per-chat floor (``NEW_MSG_MIN_GAP_S``) — a chat's own back-to-back
          sends (a turn's burst of 🔧 tool bubbles) stay spaced so the cadence
          does not trip a per-chat 429 that would strand the next placeholder; and
        * the bot-wide aggregate floor (``NEW_MSG_GLOBAL_MIN_GAP_S``) — sends
          across ALL chats are spaced so several chats streaming at once (each
          chat's FIRST send is otherwise never delayed) cannot burst into a shared
          per-bot 429. This is the send-side mirror of the edit path's
          ``EDIT_GLOBAL_MIN_GAP_S`` and is the binding aggregate floor of

        Both slots are reserved SYNCHRONOUSLY (before any await) so overlapping
        callers each claim a distinct, stacked slot instead of all reading the
        same time, sleeping together, and stampeding the limit. A pre-existing 429
        embargo (written by ``_send_with_retry``) is folded in via the per-chat
        slot. State is lazy so test instances built via ``__new__`` work unchanged.
        """
        nxt_map = getattr(self, "_chat_send_next", None)
        if nxt_map is None:
            nxt_map = {}
            self._chat_send_next = nxt_map
        now = time.monotonic()
        # ``start`` is when this send may fire: the latest of now, the per-chat
        # slot, and the shared aggregate slot. Advancing BOTH slots off ``start``
        # keeps each monotonic and stacks concurrent reservations.
        gnext = getattr(self, "_global_send_next", 0.0)
        start = max(now, nxt_map.get(chat_id, 0.0), gnext)
        nxt_map[chat_id] = start + _bot.NEW_MSG_MIN_GAP_S
        self._global_send_next = start + _bot.NEW_MSG_GLOBAL_MIN_GAP_S
        if start > now:
            await asyncio.sleep(start - now)

    async def _send_with_retry(self, method: str, **params: Any) -> dict:
        """POST ``method`` and, on an explicit ``retry_after`` 429, honor it and
        retry the SAME send (bounded by ``NEW_MSG_RETRY_ATTEMPTS``, each wait
        capped at ``NEW_MSG_RETRY_MAX_WAIT_S``). Returns the raw response — the
        success dict, or the final 429/failure so the caller logs and aborts as
        before. Unlike the old path, a 429 does NOT fan out into a second send."""
        attempt = 0
        while True:
            res = await self.call(method, **params)
            if res.get("ok"):
                return res
            retry_after = _retry_after_s(res)
            if retry_after <= 0 or attempt >= _bot.NEW_MSG_RETRY_ATTEMPTS:
                return res
            attempt += 1
            wait = min(retry_after, _bot.NEW_MSG_RETRY_MAX_WAIT_S)
            # Push the per-chat send slot past this embargo so a CONCURRENT send
            # to the same chat waits the full retry_after too — not just the
            # ordinary gap — instead of piling onto the active 429 window.
            chat_id = params.get("chat_id")
            if chat_id is not None:
                nxt_map = getattr(self, "_chat_send_next", None)
                if nxt_map is None:
                    nxt_map = {}
                    self._chat_send_next = nxt_map
                embargo_until = time.monotonic() + wait
                nxt_map[chat_id] = max(nxt_map.get(chat_id, 0.0), embargo_until)
            self._log.warning(
                "SEND 429 chat=%s method=%s retry_after=%.1fs — backing off (attempt %d/%d)",
                chat_id,
                method,
                retry_after,
                attempt,
                _bot.NEW_MSG_RETRY_ATTEMPTS,
            )
            await asyncio.sleep(wait)

    async def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        message_thread_id: int | None = None,
        parse_mode: str | None = None,
        reply_to_message_id: int | None = None,
    ) -> int | None:
        """Send ``text`` to ``chat_id``; return the new message_id or None.

        ``parse_mode`` is optional (default: plain text). Pass
        ``"MarkdownV2"`` or ``"HTML"`` when the caller has explicitly
        prepared the body to satisfy Telegram's strict escaping rules —
        an unescaped special character will 400 the whole reply. Plain
        text remains the right default for anything carrying user /
        model output (tool args, error blobs, etc.) where escaping is
        impractical.

        A reply that exceeds Telegram's per-message cap is NOT truncated
        with a ``[…]`` marker — :func:`_chunk_for_telegram` splits it
        into as many messages as it takes and we send each one. The id
        of the LAST message is returned so any later edit lands on the
        tail (short replies are a single chunk, preserving the old
        one-message contract).
        """
        chunks = _chunk_for_telegram(text)
        last_msg_id: int | None = None
        await self._await_send_slot(chat_id)
        for chunk in chunks:
            t0 = time.monotonic()
            params: dict[str, Any] = {
                "chat_id": chat_id,
                "text": chunk,
                "disable_web_page_preview": True,
            }
            if message_thread_id is not None:
                params["message_thread_id"] = message_thread_id
            if parse_mode is not None:
                params["parse_mode"] = parse_mode
            # Link the reply to the user's prompt for clarity (the whole 💭/🔧/💬
            # sequence threads off the question). Only the FIRST chunk carries
            # the reply anchor so a rolled-over long reply doesn't fan out.
            if reply_to_message_id is not None and last_msg_id is None:
                params["reply_to_message_id"] = reply_to_message_id
                params["allow_sending_without_reply"] = True
            res = await self._send_with_retry("sendMessage", **params)
            elapsed_ms = (time.monotonic() - t0) * 1000
            if not res.get("ok"):
                self._log.warning(
                    "TX FAIL chat=%s len=%d t=%.0fms err=%s",
                    chat_id,
                    len(chunk),
                    elapsed_ms,
                    res.get("description"),
                )
                # Stop the roll on the first failure — the remaining
                # chunks would just stack more failures in the chat.
                return last_msg_id
            last_msg_id = (res.get("result") or {}).get("message_id")
            preview = chunk[:120].replace("\n", " ")
            self._log.info(
                "TX chat=%s msg=%s len=%d t=%.0fms text=%r%s%s",
                chat_id,
                last_msg_id,
                len(chunk),
                elapsed_ms,
                preview,
                "…" if len(chunk) > 120 else "",
                f" [{len(chunks)} parts]" if len(chunks) > 1 else "",
            )
        return last_msg_id

    async def send_chat_action(
        self,
        chat_id: int,
        action: str = "typing",
        *,
        message_thread_id: int | None = None,
    ) -> None:
        """Send a chat action — the ephemeral "… is typing" status Telegram
        shows at the top of the chat. Telegram auto-clears it after ~5s or
        when the next message lands, so a caller keeps it visible for a
        long turn by re-sending on a timer. Best-effort: any failure is
        swallowed by :meth:`call` so the typing hint never affects the reply.
        """
        params: dict[str, Any] = {"chat_id": chat_id, "action": action}
        if message_thread_id is not None:
            params["message_thread_id"] = message_thread_id
        await self.call("sendChatAction", **params)

    async def send_voice(
        self,
        chat_id: int,
        audio_bytes: bytes,
        *,
        message_thread_id: int | None = None,
        caption: str | None = None,
        mime: str = "audio/mpeg",
        filename: str = "voice.mp3",
    ) -> int | None:
        """Upload ``audio_bytes`` as a Telegram voice/audio message.

        Sends via the ``sendAudio`` Bot API method (multipart upload) —
        ``sendVoice`` requires OGG/Opus framing which would mean an
        extra ffmpeg pass. ``sendAudio`` accepts MP3 directly and the
        Telegram client renders it with a play-button bubble next to
        the text reply, which is the same UX users expect from a
        voice reply.

        Returns the new message_id or None on failure.
        """
        url = f"{self._api_base}/bot{self._token}/sendAudio"
        data: dict[str, str] = {"chat_id": str(chat_id)}
        if message_thread_id is not None:
            data["message_thread_id"] = str(message_thread_id)
        if caption:
            data["caption"] = _truncate_for_telegram(caption)
        files = {"audio": (filename, audio_bytes, mime)}
        t0 = time.monotonic()
        try:
            r = await self._http.post(url, data=data, files=files)
        except httpx.HTTPError as e:
            self._log.warning("telegram sendAudio: transport error: %s", e)
            return None
        elapsed_ms = (time.monotonic() - t0) * 1000
        try:
            res = r.json()
        except json.JSONDecodeError:
            self._log.warning(
                "telegram sendAudio: non-json (status=%s): %s",
                r.status_code,
                r.text[:300],
            )
            return None
        if not res.get("ok"):
            self._log.warning(
                "TX-AUDIO FAIL chat=%s bytes=%d t=%.0fms err=%s",
                chat_id,
                len(audio_bytes),
                elapsed_ms,
                res.get("description"),
            )
            return None
        msg_id = (res.get("result") or {}).get("message_id")
        self._log.info(
            "TX-AUDIO chat=%s msg=%s bytes=%d t=%.0fms",
            chat_id,
            msg_id,
            len(audio_bytes),
            elapsed_ms,
        )
        return msg_id

    async def send_document(
        self,
        chat_id,
        document_bytes,
        *,
        filename="file.txt",
        caption=None,
        mime="text/plain; charset=utf-8",
        message_thread_id=None,
    ):
        """Upload bytes as a Telegram document. Returns message_id or None."""
        data = {"chat_id": str(chat_id)}
        if message_thread_id is not None:
            data["message_thread_id"] = str(message_thread_id)
        if caption:
            data["caption"] = _truncate_for_telegram(caption)
        return await self._upload(
            "sendDocument",
            chat_id,
            data,
            {"document": (filename, document_bytes, mime)},
        )

    async def _upload(
        self,
        method: str,
        chat_id,
        data: dict,
        files: dict,
    ) -> int | None:
        """Multipart upload on the same send pacing + 429 backoff as text.

        Waits for the chat's send slot, then POSTs; an explicit
        ``retry_after`` 429 is honored and the SAME upload retried (bounded
        like :meth:`_send_with_retry`). Returns the message_id or None.
        """
        await self._await_send_slot(chat_id)
        url = f"{self._api_base}/bot{self._token}/{method}"
        attempt = 0
        while True:
            try:
                r = await self._http.post(url, data=data, files=files)
            except httpx.HTTPError as exc:
                self._log.warning("%s transport error: %s", method, exc)
                return None
            try:
                res = r.json()
            except (json.JSONDecodeError, ValueError):
                res = {}
            if r.status_code == 200 and isinstance(res, dict) and res.get("ok", True):
                return (res.get("result") or {}).get("message_id")
            retry_after = _retry_after_s(res)
            if r.status_code != 429 or retry_after <= 0 or attempt >= _bot.NEW_MSG_RETRY_ATTEMPTS:
                self._log.warning("%s HTTP %s: %s", method, r.status_code, r.text[:200])
                return None
            attempt += 1
            wait = min(retry_after, _bot.NEW_MSG_RETRY_MAX_WAIT_S)
            nxt_map = getattr(self, "_chat_send_next", None)
            if nxt_map is None:
                nxt_map = {}
                self._chat_send_next = nxt_map
            nxt_map[chat_id] = max(nxt_map.get(chat_id, 0.0), time.monotonic() + wait)
            self._log.warning(
                "SEND 429 chat=%s method=%s retry_after=%.1fs — backing off (attempt %d/%d)",
                chat_id,
                method,
                retry_after,
                attempt,
                _bot.NEW_MSG_RETRY_ATTEMPTS,
            )
            await asyncio.sleep(wait)

    async def send_photo(
        self,
        chat_id,
        photo_bytes,
        *,
        filename="photo.jpg",
        caption=None,
        message_thread_id=None,
    ):
        """Upload bytes as a Telegram photo. Returns message_id or None."""
        data = {"chat_id": str(chat_id)}
        if message_thread_id is not None:
            data["message_thread_id"] = str(message_thread_id)
        if caption:
            data["caption"] = _truncate_for_telegram(caption)
        return await self._upload(
            "sendPhoto",
            chat_id,
            data,
            {"photo": (filename, photo_bytes, "application/octet-stream")},
        )

    async def edit_message(
        self,
        chat_id: int,
        message_id: int,
        text: str,
    ) -> tuple[bool, float]:
        """In-place edit. Returns ``(ok, retry_after_s)``.

        ``retry_after_s > 0`` only when Telegram explicitly returned 429
        with ``parameters.retry_after``. The caller uses it to bump the
        adaptive throttle. ``ok=True`` covers both real success and the
        benign "message is not modified" reject (same text).
        """
        truncated = _truncate_for_telegram(text)
        t0 = time.monotonic()
        res = await self.call(
            "editMessageText",
            chat_id=chat_id,
            message_id=message_id,
            text=truncated,
            disable_web_page_preview=True,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000
        if not res.get("ok"):
            desc = res.get("description") or ""
            desc_lc = desc.lower()
            if "not modified" in desc_lc:
                return (True, 0.0)
            retry_after = 0.0
            params = res.get("parameters") or {}
            if isinstance(params, dict):
                ra = params.get("retry_after")
                if isinstance(ra, (int, float)) and ra > 0:
                    retry_after = float(ra)
            if retry_after > 0:
                self._log.warning(
                    "EDIT 429 chat=%s msg=%s retry_after=%.1fs",
                    chat_id,
                    message_id,
                    retry_after,
                )
            else:
                self._log.warning(
                    "EDIT FAIL chat=%s msg=%s len=%d t=%.0fms err=%s",
                    chat_id,
                    message_id,
                    len(truncated),
                    elapsed_ms,
                    desc,
                )
            return (False, retry_after)
        self._log.debug(
            "EDIT chat=%s msg=%s len=%d t=%.0fms",
            chat_id,
            message_id,
            len(truncated),
            elapsed_ms,
        )
        return (True, 0.0)

    async def delete_message(self, chat_id: int, message_id: int) -> bool:
        """Delete a message (used to drop a 💭 block that never received any
        real reasoning, so the chat isn't left with a stray "💭 thinking…")."""
        try:
            res = await self.call(
                "deleteMessage",
                chat_id=chat_id,
                message_id=message_id,
            )
        except Exception:  # noqa: BLE001
            return False
        return bool(res.get("ok"))

    # ------------------------------------------------------------------
    # Rich Messages (Bot API 10.1)
    # ------------------------------------------------------------------

    # Cached capability flag — None = not yet probed.
    _rich_capable: bool | None = None

    async def _probe_rich(self) -> bool:
        """Probe whether the Bot API server supports Rich Messages.

        Sends a trivial ``sendRichMessage`` to a non-existent chat so it
        fails with a known error. Checks the HTTP status code:
          400 → method EXISTS (chat-level error) → rich OK
          404 → method does NOT exist → server too old

        The result is cached on the class so we only probe once per
        process lifetime.
        """
        if self._rich_capable is not None:
            return self._rich_capable

        url = f"{self._api_base}/bot{self._token}/sendRichMessage"
        try:
            r = await self._http.post(
                url,
                json={
                    "chat_id": 0,
                    "rich_message": {"markdown": "*probe*"},
                },
            )
            status = r.status_code
        except httpx.HTTPError as e:
            self._log.warning("telegram rich-probe: transport error: %s", e)
            self._rich_capable = False
            return False

        if status == 404:
            # Method not found — server is too old
            self._rich_capable = False
            self._log.info("telegram rich-messages: NOT supported (server too old, HTTP 404)")
        else:
            # Check the response body for a definitive "method not found"
            # rather than relying on raw HTTP status (a proxy could 404
            # even when the server supports the method).
            try:
                body = r.json()
                is_method_missing = (
                    body.get("error_code") == 404 or "not found" in (body.get("description") or "").lower()
                )
            except Exception:
                is_method_missing = False

            if is_method_missing:
                self._rich_capable = False
                self._log.info(
                    "telegram rich-messages: NOT supported (response: %s)",
                    body.get("description", "unknown"),
                )
            else:
                # 400 (bad chat_id), 429 (rate limit), or anything else
                # → the method was recognized → rich OK
                self._rich_capable = True
                self._log.info("telegram rich-messages: supported (HTTP %d)", status)
        return self._rich_capable

    async def is_rich_enabled(self) -> bool:
        """Return True if Rich Messages should be used for this client.

        Respects the ``_RICH_ENV`` gate (off/auto/on). In ``auto`` mode
        probes the API once and caches the result.
        """
        if _bot._RICH_ENV == "off":
            return False
        if _bot._RICH_ENV == "on":
            return True
        # auto: probe once
        return await self._probe_rich()

    async def is_rich_drafts_enabled(self) -> bool:
        """Return True if rich draft streaming should be used.

        Requires both rich messages AND the draft flag to be enabled.
        Drafts are more fragile (rollover desync, reply-link loss) so
        they are gated separately.
        """
        if not await self.is_rich_enabled():
            return False
        return _bot._RICH_DRAFTS_ENV == "on"

    async def send_rich_message(
        self,
        chat_id: int,
        markdown: str,
        *,
        message_thread_id: int | None = None,
        reply_to_message_id: int | None = None,
        skip_entity_detection: bool = False,
    ) -> int | None:
        """Send a rich-formatted message using ``sendRichMessage``.

        Returns the new message_id or None on failure. Falls back to
        plain ``send_message`` if rich is not enabled or fails.
        """
        if not await self.is_rich_enabled():
            return await self.send_message(
                chat_id,
                markdown,
                message_thread_id=message_thread_id,
                reply_to_message_id=reply_to_message_id,
            )

        chunks = _chunk_for_telegram(markdown)
        last_msg_id: int | None = None
        await self._await_send_slot(chat_id)
        for chunk in chunks:
            params: dict[str, Any] = {
                "chat_id": chat_id,
                "rich_message": {
                    "markdown": chunk,
                    "skip_entity_detection": skip_entity_detection,
                },
            }
            if message_thread_id is not None:
                params["message_thread_id"] = message_thread_id
            # Link the reply to the user's prompt for clarity. Only the FIRST
            # chunk carries the reply anchor so a rolled-over long reply doesn't
            # fan out (mirrors ``send_message``).
            if reply_to_message_id is not None and last_msg_id is None:
                params["reply_to_message_id"] = reply_to_message_id
                params["allow_sending_without_reply"] = True

            t0 = time.monotonic()
            res = await self._send_with_retry("sendRichMessage", **params)
            elapsed_ms = (time.monotonic() - t0) * 1000
            if not res.get("ok"):
                desc = res.get("description", "")
                self._log.warning(
                    "RICH-SEND FAIL chat=%s len=%d t=%.0fms err=%s",
                    chat_id,
                    len(chunk),
                    elapsed_ms,
                    desc,
                )
                # Rich delivery must ALWAYS degrade to plain
                # text on any rich-API rejection — the message must never be
                # dropped. This holds for a 429 too: the rich body already
                # exhausted its bounded retries above WITHOUT delivering, so the
                # plain-text send is the sole delivery, not a duplicate fan-out.
                # (Delivery-never-lost outranks avoiding the historical
                # RICH-SEND FAIL → TX FAIL doubling, which only avoided a dupe by
                # dropping the answer entirely.)
                return await self.send_message(
                    chat_id,
                    markdown,
                    message_thread_id=message_thread_id,
                    reply_to_message_id=reply_to_message_id,
                )
            last_msg_id = (res.get("result") or {}).get("message_id")
            preview = chunk[:120].replace("\n", " ")
            self._log.info(
                "RICH-SEND chat=%s msg=%s len=%d t=%.0fms text=%r%s%s",
                chat_id,
                last_msg_id,
                len(chunk),
                elapsed_ms,
                preview,
                "…" if len(chunk) > 120 else "",
                f" [{len(chunks)} parts]" if len(chunks) > 1 else "",
            )
        return last_msg_id

    async def edit_rich_message(
        self,
        chat_id: int,
        message_id: int,
        markdown: str,
        *,
        skip_entity_detection: bool = False,
    ) -> tuple[bool, float]:
        """In-place edit using rich markdown. Returns ``(ok, retry_after_s)``.

        Falls back to plain ``edit_message`` if rich is not enabled or fails.
        """
        if not await self.is_rich_enabled():
            return await self.edit_message(chat_id, message_id, markdown)

        # Do NOT truncate here — :class:`_RollingMessage` drains overflow
        # across messages before calling us. Appending ``[…]`` would crop the
        # reply; plain ``edit_message`` keeps a last-resort truncate for
        # one-shot callers only.
        t0 = time.monotonic()
        res = await self.call(
            "editMessageText",
            chat_id=chat_id,
            message_id=message_id,
            rich_message={
                "markdown": markdown,
                "skip_entity_detection": skip_entity_detection,
            },
            disable_web_page_preview=True,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000
        if not res.get("ok"):
            desc = res.get("description") or ""
            desc_lc = desc.lower()
            if "not modified" in desc_lc:
                return (True, 0.0)
            retry_after = 0.0
            params = res.get("parameters") or {}
            if isinstance(params, dict):
                ra = params.get("retry_after")
                if isinstance(ra, (int, float)) and ra > 0:
                    retry_after = float(ra)
            if retry_after > 0:
                self._log.warning(
                    "RICH-EDIT 429 chat=%s msg=%s retry_after=%.1fs",
                    chat_id,
                    message_id,
                    retry_after,
                )
                # Don't fall back to plain edit on rate-limit — that would
                # fire a second write in the same embargo window and also 429.
                # Return the retry_after so the caller can back off properly.
                return (False, retry_after)
            else:
                self._log.warning(
                    "RICH-EDIT FAIL chat=%s msg=%s len=%d t=%.0fms err=%s",
                    chat_id,
                    message_id,
                    len(markdown),
                    elapsed_ms,
                    desc,
                )
                # Only fall back to plain edit on capability/format failures
                return await self.edit_message(chat_id, message_id, markdown)
        self._log.debug(
            "RICH-EDIT chat=%s msg=%s len=%d t=%.0fms",
            chat_id,
            message_id,
            len(markdown),
            elapsed_ms,
        )
        return (True, 0.0)

    async def send_rich_message_draft(
        self,
        chat_id: int,
        draft_id: int,
        markdown: str,
        *,
        message_thread_id: int | None = None,
        skip_entity_detection: bool = False,
    ) -> bool:
        """Stream a partial rich message via ``sendRichMessageDraft``.

        Returns True on success. Falls back to False (caller should use
        ``edit_rich_message`` / ``edit_message``) if rich is disabled or
        the draft API rejects the call.
        """
        if not await self.is_rich_enabled():
            return False

        params: dict[str, Any] = {
            "chat_id": chat_id,
            "draft_id": draft_id,
            "rich_message": {
                "markdown": markdown,
                "skip_entity_detection": skip_entity_detection,
            },
        }
        if message_thread_id is not None:
            params["message_thread_id"] = message_thread_id

        t0 = time.monotonic()
        res = await self.call("sendRichMessageDraft", **params)
        elapsed_ms = (time.monotonic() - t0) * 1000
        if not res.get("ok"):
            desc = res.get("description", "")
            self._log.warning(
                "RICH-DRAFT FAIL chat=%s draft=%s len=%d t=%.0fms err=%s",
                chat_id,
                draft_id,
                len(markdown),
                elapsed_ms,
                desc,
            )
            return False
        self._log.debug(
            "RICH-DRAFT chat=%s draft=%s len=%d t=%.0fms",
            chat_id,
            draft_id,
            len(markdown),
            elapsed_ms,
        )
        return True

    async def finalize_rich_message_draft(
        self,
        chat_id: int,
        draft_id: int,
        markdown: str,
        *,
        message_thread_id: int | None = None,
        reply_to_message_id: int | None = None,
        skip_entity_detection: bool = False,
    ) -> int | None:
        """Persist a streamed rich draft via ``sendRichMessage``.

        Bot API 10.1 drafts are ephemeral; final delivery uses
        ``sendRichMessage``. Returns the new message_id or None on failure.
        Falls back to plain ``send_message`` if rich is disabled or fails.
        """
        return await self.send_rich_message(
            chat_id,
            markdown,
            message_thread_id=message_thread_id,
            reply_to_message_id=reply_to_message_id,
            skip_entity_detection=skip_entity_detection,
        )


# ---------------------------------------------------------------------------
# ``bot.py`` still owns the shared new-message pacing CONSTANTS + text helpers
# this client uses. The pacing constants (NEW_MSG_*) and the rich-mode env gates
# (_RICH_ENV / _RICH_DRAFTS_ENV) are read through the ``_bot`` module alias — NOT
# copied via ``from bot import`` — so tests that monkeypatch
# ``sharedagents_telegram.bot.<NAME>`` (e.g. NEW_MSG_MIN_GAP_S, _RICH_ENV) still take
# effect here, matching the original single-module semantics. The pure text
# helpers are never reassigned, so a plain re-import is equivalent. Both live at
# the BOTTOM (after the class) so the bot<->api cycle can't deadlock in either
# import order. Do NOT hoist to the top.
# ---------------------------------------------------------------------------
from sharedagents_telegram import bot as _bot  # noqa: E402,F401
from sharedagents_telegram.bot import (  # noqa: E402
    _chunk_for_telegram,
    _truncate_for_telegram,
)
