"""Inbound-message / auth / permission / abort methods for :class:`AcpTelegramBot`.

Extracted from the :class:`AcpTelegramBot` god-class (de-monolith). This is a
mixin: every method still binds ``self`` and dispatches identically through the
concrete class's MRO. Behaviour is unchanged — the method bodies moved verbatim.

The module-level helpers/constants these methods read stay owned by ``bot.py``
and are imported at the BOTTOM (after the class) so the ``bot`` <-> mixin import
cycle is deadlock-free in either import order.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_IMAGE_FILE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".heic", ".heif"})


class HandlersMixin:
    # ---- chat → session persistence ------------------------------------

    async def _reply(
        self,
        chat_id: int,
        text: str,
        thread_id: int | None = None,
    ) -> None:
        """One-line wrapper around ``send_message`` that propagates
        ``thread_id`` so replies in a Telegram forum topic stay in
        that topic. No-op when ``self._api`` is None (unit-test path).
        """
        if not self.tg_cfg.can_send(chat_id):
            self._log.info("suppress reply chat=%s — send not allowed", chat_id)
            return
        if self._api is None:
            return
        await self._api.send_message(
            chat_id,
            text,
            message_thread_id=thread_id,
        )

    # ---- permission prompt (session/request_permission) ----------------

    def _make_permission_handler(self, chat_id: int, thread_id: int | None):
        """Build the per-client ``session/request_permission`` handler.

        The returned async callable is registered on the chat's
        :class:`AcpClient`; it closes over ``(chat_id, thread_id)`` so
        the prompt always lands in the chat that owns the session — no
        sid→chat reverse map needed.
        """

        async def _handler(params: dict) -> dict:
            return await self._prompt_permission(chat_id, thread_id, params)

        return _handler

    async def _prompt_permission(
        self,
        chat_id: int,
        thread_id: int | None,
        params: dict,
    ) -> dict:
        """Surface an ACP permission request as an inline-keyboard prompt.

        Sends a 3-button message (Allow once / Allow always / Reject),
        parks a future keyed by an opaque token, and awaits the user's
        tap. It remains parked until that explicit answer, turn cancellation,
        or transport failure. Failures default to ``reject_once``; elapsed
        wall-clock time alone does not answer the prompt. Returns the ACP
        ``RequestPermissionResponse`` mapping verbatim.

        Runs inline in the chat's AcpClient read loop (same as
        ``_inbound_ask_user``); that's fine — the agent's turn is
        parked on this very decision, so nothing else is inbound on
        that client until we answer.
        """

        def _outcome(option_id: str) -> dict:
            return {"outcome": {"outcome": "selected", "optionId": option_id}}

        tool_call = params.get("toolCall") if isinstance(params, dict) else {}
        if not isinstance(tool_call, dict):
            tool_call = {}
        title = str(tool_call.get("title") or "tool")
        raw_input = tool_call.get("rawInput")
        args_preview = ""
        if isinstance(raw_input, dict) and raw_input:
            try:
                args_preview = json.dumps(raw_input, ensure_ascii=False)
            except (TypeError, ValueError):
                args_preview = str(raw_input)
            if len(args_preview) > 600:
                args_preview = args_preview[:600] + " …"

        if self._api is None:
            # No transport (shouldn't happen once run() has started) —
            # default to deny rather than hang the agent's turn.
            return _outcome("reject_once")

        # Plain text deliberately — no parse_mode. Tool args routinely
        # contain ``_`` / ``*`` / backticks that would break Telegram's
        # fragile legacy Markdown and could fail the send outright.
        text = f"⛔ Destructive tool call needs approval\n\ntool: {title}"
        if args_preview:
            text += f"\nargs: {args_preview}"
        text += "\n\nWaiting for an explicit decision."
        token = secrets.token_urlsafe(8)
        keyboard = {
            "inline_keyboard": [
                [
                    {"text": "✅ Allow once", "callback_data": f"perm:{token}:allow_once"},
                    {"text": "🔓 Allow always", "callback_data": f"perm:{token}:allow_always"},
                    {"text": "⛔ Reject", "callback_data": f"perm:{token}:reject_once"},
                ]
            ]
        }
        send_params: dict[str, Any] = {
            "chat_id": chat_id,
            "text": _truncate_for_telegram(text),
            "reply_markup": keyboard,
            "disable_web_page_preview": True,
        }
        if thread_id is not None:
            send_params["message_thread_id"] = thread_id

        res = await self._api.call("sendMessage", **send_params)
        if not res.get("ok"):
            self._log.warning(
                "permission prompt sendMessage failed chat=%s err=%s — defaulting to deny",
                chat_id,
                res.get("description"),
            )
            return _outcome("reject_once")
        message_id = (res.get("result") or {}).get("message_id")

        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending_permissions[token] = fut
        try:
            option_id = await fut
        except Exception:  # noqa: BLE001
            option_id = "reject_once"
        finally:
            self._pending_permissions.pop(token, None)

        # Reflect the verdict back into the chat and drop the keyboard
        # so a stale prompt can't be tapped a second time.
        if isinstance(message_id, int):
            verdict = {
                "allow_once": "✅ Allowed once",
                "allow_always": "✅ Allowed (always)",
                "reject_once": "⛔ Rejected",
                "reject_always": "⛔ Rejected (always)",
            }.get(option_id, "⛔ Rejected")
            try:
                await self._api.call(
                    "editMessageText",
                    chat_id=chat_id,
                    message_id=message_id,
                    text=_truncate_for_telegram(f"{verdict} — {title}"),
                    disable_web_page_preview=True,
                )
            except Exception:  # noqa: BLE001
                self._log.debug(
                    "permission prompt edit failed",
                    exc_info=True,
                )

        return _outcome(option_id)

    async def _handle_callback_query(self, update: dict) -> None:
        """Resolve an inline-keyboard tap on a permission prompt.

        Parses ``perm:<token>:<optionId>`` callback data, acks the
        callback query (so Telegram clears the button spinner), and
        completes the parked future so ``_prompt_permission`` returns
        the user's choice. Unknown / stale tokens are acked and dropped.
        """
        cq = update.get("callback_query") or {}
        cq_id = cq.get("id")
        data = cq.get("data") or ""
        msg = cq.get("message") or {}
        chat = msg.get("chat") or {}
        chat_id = chat.get("id")

        async def _ack(note: str = "") -> None:
            if not cq_id or self._api is None:
                return
            try:
                ack_params: dict[str, Any] = {"callback_query_id": cq_id}
                if note:
                    ack_params["text"] = note
                await self._api.call("answerCallbackQuery", **ack_params)
            except Exception:  # noqa: BLE001
                self._log.debug("answerCallbackQuery failed", exc_info=True)

        # Authorize — only allowed chats can answer prompts.
        if not isinstance(chat_id, int) or not self._is_authorized(chat_id):
            await _ack()
            return
        if not data.startswith("perm:"):
            # Not one of ours — ack so the client stops spinning, ignore.
            await _ack()
            return
        parts = data.split(":", 2)
        if len(parts) != 3:
            await _ack()
            return
        _, token, option_id = parts
        if option_id not in ("allow_once", "allow_always", "reject_once", "reject_always"):
            option_id = "reject_once"
        fut = self._pending_permissions.get(token)
        if fut is None or fut.done():
            # Stale prompt — bot restarted, or already resolved / timed out.
            await _ack("This prompt has expired.")
            return
        fut.set_result(option_id)
        await _ack()

    def chat_lock(
        self,
        chat_id: int,
        thread_id: int | None = None,
    ) -> asyncio.Lock:
        # Per-(chat,topic) lock so two topics in the same forum can run
        # turns concurrently without serializing on each other.
        key = self._session_key(chat_id, thread_id)
        lock = self._chat_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._chat_locks[key] = lock
        return lock

    async def _acquire_turn_slot(
        self,
        lock: asyncio.Lock,
        *,
        queue_timeout_s: float | None = None,
        on_queued: Any | None = None,
    ) -> bool:
        """Take ``lock`` for a turn, honouring the queue + its timeout.

        Free slot → acquire immediately. Slot already busy → call
        ``on_queued`` once (so the caller can flip the placeholder/reaction
        to a "queued" state), then wait. ``queue_timeout_s`` bounds that
        wait: ``None``/``0`` waits forever (legacy behaviour), a positive
        value gives up after that many seconds.

        Returns ``True`` if acquired — the CALLER OWNS the lock and MUST
        release it — or ``False`` if it timed out while queued (the lock is
        NOT held; the original turn still owns it).
        """
        # Snapshot before we touch it: ``locked()`` is the only honest read
        # of "is a turn already running". A benign TOCTOU (it frees between
        # here and ``acquire``) just means we waited 0s — still correct.
        queued = lock.locked()
        if queued and on_queued is not None:
            try:
                await on_queued()
            except Exception:  # noqa: BLE001
                self._log.debug("on_queued callback raised", exc_info=True)
        if queued and queue_timeout_s and queue_timeout_s > 0:
            try:
                async with asyncio.timeout(queue_timeout_s):
                    await lock.acquire()
            except (asyncio.TimeoutError, TimeoutError):
                # ``asyncio.timeout`` cancels the pending ``acquire`` and
                # (3.11+) leaves the lock untouched — we never hold it here.
                return False
            return True
        await lock.acquire()
        return True

    async def _set_status_reaction(
        self,
        chat_id: int,
        message_id: int | None,
        emoji: str | None,
    ) -> None:
        """Best-effort: set (or clear, when ``emoji is None``) a single
        lifecycle reaction on the user's incoming message. Never raises
        and never blocks the turn on a failure — a reaction is cosmetic.
        No-ops without a message_id."""
        if not isinstance(message_id, int) or self._api is None:
            return
        reaction = [{"type": "emoji", "emoji": emoji}] if emoji is not None else []
        try:
            await self._api.call(
                "setMessageReaction",
                chat_id=chat_id,
                message_id=message_id,
                reaction=reaction,
                is_big=False,
            )
        except Exception:  # noqa: BLE001
            self._log.debug(
                "status reaction failed chat=%s emoji=%s",
                chat_id,
                emoji,
            )

    # ---- inbound dispatch (testable seams) -----------------------------

    async def handle_text_message(
        self,
        chat_id: int,
        text: str,
        *,
        thread_id: int | None = None,
        artifacts: list | None = None,
    ) -> str:
        """Process an inbound text message; return the assembled reply.

        Returning the reply (rather than calling ``send_text`` directly)
        keeps this method trivially testable: the test asserts on the
        return value plus the recorded calls on the stub
        :class:`AcpClient`. The long-poll loop in :meth:`run` is what
        wires the real Telegram POST.
        """
        text = (text or "").strip()
        if not text:
            return ""
        import acp.schema as _acp

        return await self._dispatch_blocks(
            chat_id,
            [_acp.TextContentBlock(type="text", text=text)],
            thread_id=thread_id,
            artifacts=artifacts,
        )

    async def handle_voice_message(
        self,
        chat_id: int,
        file_id: str,
        *,
        mime: str = "",
        filename: str = "",
    ) -> str:
        """Resolve the voice file → STT → forward as text."""
        if self._stt is None or self._download_audio is None:
            self._log.warning(
                "voice received chat=%s but STT not configured",
                chat_id,
            )
            return ""
        result = await self._download_audio(file_id)
        if not result:
            return ""
        audio_bytes, fname = result
        if not filename:
            filename = fname
        try:
            transcript = await self._stt(
                audio_bytes,
                mime=mime,
                filename=filename,
            )
        except TranscriptionUnavailableError:
            # Transcription could not run (undecodable /
            # no ffmpeg). This legacy forward-path has no rolling UI bubble to
            # render a distinct message, so it preserves its empty-return
            # contract; the live path (_stream_turn_for_chat) renders the
            # distinct operator message.
            self._log.warning("voice transcription unavailable (decode)")
            return ""
        if not transcript:
            return ""
        return await self.handle_text_message(chat_id, transcript)

    async def handle_image_message(
        self,
        chat_id: int,
        msg: Mapping[str, Any] | None = None,
        *,
        thread_id: int | None = None,
    ) -> str | None:
        """Forward a Telegram photo to the agent as an ACP ImageContent block.

        Picks the largest photo size (Telegram delivers an array sorted
        ascending), downloads it via the bot-file CDN, and rides a
        ContentBlock list ``[TextContentBlock(caption), ImageContentBlock(
        data, mimeType)]`` through ``_stream_turn_for_chat`` — the same
        streaming pipeline text/voice/documents use, so the turn renders
        live (💭/🔧/💬 block sequence, placeholder, reply-linking) instead
        of one buffered end-of-turn blob. The agent's bridge translates
        the blocks into the OpenAI multipart ``image_url`` format
        upstreams (vLLM / openrouter) accept natively, so vision-capable
        models like ``kimi`` can read the image without extra plumbing.

        ``msg`` is the raw Telegram message dict; required to pull the
        ``photo[]`` array + optional ``caption``. Returns ``None`` once
        the turn is dispatched (the caller must not send a separate
        reply — same contract as ``handle_document_message``); returns
        an error string when anything fails before dispatch (download
        error, oversize, unauth) so a single broken upload doesn't kill
        the chat.
        """
        if msg is None or not isinstance(msg.get("photo"), list) or not msg["photo"]:
            return ACP_IMAGE_UNSUPPORTED_REPLY
        # Telegram's `photo[]` is sorted ascending by size; pick the
        # largest so the model sees the highest-quality version it has.
        largest = msg["photo"][-1]
        file_id = largest.get("file_id") if isinstance(largest, dict) else None
        if not isinstance(file_id, str) or not file_id:
            return ACP_IMAGE_UNSUPPORTED_REPLY
        return await self._forward_image(chat_id, msg, file_id, thread_id=thread_id)

    async def _forward_image(
        self,
        chat_id: int,
        msg: Mapping[str, Any],
        file_id: str,
        *,
        thread_id: int | None = None,
        from_file: bool = False,
    ) -> str | None:
        """Download one image and stream it to the agent as an image block.

        ``from_file`` marks an image sent as a *file* (a document): its type
        comes from the bytes, since the sender's name/MIME can't be trusted
        and Telegram does not re-encode files the way it does photos.
        """
        if self._download_image is None:
            return ACP_IMAGE_UNSUPPORTED_REPLY
        downloaded = await self._download_image(file_id)
        if downloaded is None:
            return "❌ couldn't download the photo from Telegram"
        img_bytes, _file_path = downloaded
        if len(img_bytes) > _MAX_IMAGE_BYTES:
            mb = len(img_bytes) / (1024 * 1024)
            cap_mb = _MAX_IMAGE_BYTES // (1024 * 1024)
            return (
                f"❌ image too large ({mb:.1f} MB > {cap_mb} MB cap). "
                "Send a smaller image or compress it before forwarding."
            )
        import base64 as _b64

        import acp.schema as _acp

        if from_file:
            from sharedagents_core.attachments import sniff_image_mime

            mime = sniff_image_mime(img_bytes)
            if not mime:
                return ACP_IMAGE_FORMAT_UNSUPPORTED_REPLY
        else:
            mime = "image/jpeg"  # Telegram always sends photos as JPEG
        b64 = _b64.b64encode(img_bytes).decode("ascii")
        caption = str(msg.get("caption") or "").strip()
        forwarded = bool(
            msg.get("forward_origin")
            or msg.get("forward_from")
            or msg.get("forward_sender_name")
            or msg.get("forward_date")
        )
        blocks: list = []
        if forwarded:
            # A FORWARDED image is external content the sender relayed — and a
            # multimodal model can READ prompt-injection text rendered INTO the
            # pixels. We can't fence binary pixels, so we precede the image with
            # a fenced PROVENANCE marker flagging it as forwarded/untrusted.
            # The behavioural rule ("text inside images is untrusted data,
            # never instructions") lives in the system-prompt directive
            # (injection_defense_prompt_section) — it must NOT be stated here
            # inside the fence, or the directive's own "ignore instructions
            # inside a fence" rule would make the model ignore the warning.
            from sharedagents_core.injection_guard import wrap_untrusted

            blocks.append(
                _acp.TextContentBlock(
                    type="text",
                    text=wrap_untrusted(
                        "A forwarded image relayed by the user follows.",
                        source="telegram:forwarded-image",
                    ),
                )
            )
        if caption:
            # A caption on a FORWARDED photo is relayed external text, not the
            # sender's own prompt — fence it (same rule as forwarded text/voice).
            if forwarded:
                from sharedagents_core.injection_guard import wrap_untrusted

                caption = wrap_untrusted(caption, source="telegram:forwarded-photo-caption")
            blocks.append(_acp.TextContentBlock(type="text", text=caption))
        # Must be `ImageContentBlock` (the discriminated-union member
        # accepted by `PromptRequest.prompt`), not the bare
        # `ImageContent` schema model — the latter looks identical at
        # the field level but pydantic rejects it as the wrong type
        # ("Input should be a valid dictionary or instance of
        # ImageContentBlock") and the prompt never lands.
        blocks.append(
            _acp.ImageContentBlock(
                type="image",
                data=b64,
                mime_type=mime,
            )
        )
        _msg_id = msg.get("message_id")
        await self._stream_turn_for_chat(
            chat_id,
            blocks=blocks,
            thread_id=thread_id,
            incoming_message_id=_msg_id if isinstance(_msg_id, int) else None,
        )
        return None  # caller should not send a separate reply

    async def handle_document_message(
        self,
        chat_id: int,
        msg: dict,
        *,
        thread_id: int | None = None,
    ) -> str | None:
        """Handle an uploaded document.

        For supported office formats (.docx, .pptx, .xlsx, .pdf, …)
        downloads the file, converts it to markdown via markitdown,
        and forwards the markdown content to the agent as the user's
        prompt (plus any caption). Returns ``None`` to indicate the
        turn was dispatched.

        For unsupported formats returns the unsupported-file reply
        string so the caller can reply with it.
        """
        document = msg.get("document")
        if not isinstance(document, dict):
            return ACP_DOCUMENT_UNSUPPORTED_REPLY

        file_id = document.get("file_id")
        file_name = document.get("file_name", "document.bin")
        caption = msg.get("caption") or ""

        if not file_id:
            return ACP_DOCUMENT_UNSUPPORTED_REPLY

        # An image sent as a file (uncompressed) goes to the agent as an
        # image, exactly like a photo.
        ext = Path(file_name).suffix.lower()
        mime_type = str(document.get("mime_type") or "").lower()
        if mime_type.startswith("image/") or ext in _IMAGE_FILE_EXTENSIONS:
            file_size = document.get("file_size")
            if isinstance(file_size, int) and file_size > _MAX_IMAGE_BYTES:
                cap_mb = _MAX_IMAGE_BYTES // (1024 * 1024)
                return f"❌ image too large ({file_size / 1_048_576:.1f} MB > {cap_mb} MB cap). Send a smaller image."
            return await self._forward_image(
                chat_id,
                msg,
                file_id,
                thread_id=thread_id,
                from_file=True,
            )

        # Check extension — only handle supported office formats.
        if ext not in _OFFICE_EXTENSIONS:
            return ACP_DOCUMENT_UNSUPPORTED_REPLY

        # File size guard — don't attempt downloads over 200 MB.
        file_size = document.get("file_size")
        if isinstance(file_size, int) and file_size > 200 * 1024 * 1024:
            return f"📎 that file is too large for conversion ({file_size / 1_048_576:.0f} MB). Max 200 MB."

        # Download via the same infrastructure used for audio.
        if self._download_audio is None:
            return "⚠️ file download not available"
        try:
            dl = await self._download_audio(file_id)
        except Exception:  # noqa: BLE001
            self._log.exception("document download failed chat=%s", chat_id)
            return "⚠️ couldn't download file"
        if not dl:
            return "⚠️ couldn't download file"
        data, _filename = dl

        # Real byte-length cap — the metadata guard above trusts
        # document.file_size, which Telegram may omit (or the download
        # may exceed). Enforce the same 200 MB ceiling on actual bytes.
        if len(data) > 200 * 1024 * 1024:
            return f"📎 that file is too large for conversion ({len(data) / 1_048_576:.0f} MB). Max 200 MB."

        # Convert to markdown OFF the event loop — markitdown parsing a
        # large PDF/Office file is CPU-bound and would otherwise block
        # the single-threaded bot from servicing other chats.
        md = await asyncio.to_thread(
            _convert_office_bytes_to_markdown,
            data,
            file_name,
        )
        if md is None:
            return "⚠️ couldn't convert file to markdown"

        self._log.info(
            "DOC CONVERT chat=%s file=%s markdown_len=%d",
            chat_id,
            file_name,
            len(md),
        )

        # Persist the conversion so the user can re-read it.
        try:
            agent_cfg = self._agent_cfg(chat_id, thread_id)
            xdir = agent_cfg.home / "doc_conversions"
            xdir.mkdir(parents=True, exist_ok=True)
            now = datetime.now(timezone.utc)
            ts = now.strftime("%Y-%m-%d_%H-%M-%S")
            stem = Path(file_name).stem
            xpath = xdir / f"{ts}_{stem}.md"
            xpath.write_text(
                f"# {stem}\n\n*Converted {now.isoformat()} | source: `{file_name}`*\n\n{md}\n",
                encoding="utf-8",
            )
            self._log.info("conversion saved to %s", xpath)
        except Exception:  # noqa: BLE001
            self._log.exception("conversion save failed chat=%s", chat_id)

        # Build the prompt text. If the file came with a caption, use
        # "caption\n\n{converted_content}"; otherwise just the content.
        text_content = md
        if caption:
            text_content = f"{caption}\n\n{text_content}"

        # Forward to the agent as a text turn.
        forwarded = bool(
            msg.get("forward_origin")
            or msg.get("forward_from")
            or msg.get("forward_sender_name")
            or msg.get("forward_date")
        )
        await self._stream_turn_for_chat(
            chat_id,
            text=text_content,
            thread_id=thread_id,
            untrusted=forwarded,
        )
        return None  # caller should not send a separate reply

    async def handle_sticker_message(self, chat_id: int) -> str:
        return ACP_STICKER_UNSUPPORTED_REPLY

    async def cancel_chat(
        self,
        chat_id: int,
        thread_id: int | None = None,
        *,
        via: str = "stop",
    ) -> tuple[bool, bool]:
        """Imperative cancel — backs ``/stop``, ``/cancel``, ``/abort``.

        Looks up the (chat, topic)'s session id, fires
        ``session/cancel`` on the cached client (a fire-and-forget
        JSON-RPC notification, so the wire write returns the moment
        bytes are flushed — no waiting for the agent to do anything).

        Returns ``(stopped, force_escalated)``:

        * ``stopped`` — ``True`` when a cancel was dispatched; ``False``
          when there is no session or no client to dispatch through.
        * ``force_escalated`` — ``True`` when this call landed within
          ``_ABORT_DOUBLE_PRESS_S`` of the previous abort on the same
          (chat, thread). The bridge's own double-press window
          (acp_bridge.py ``_HARD_KILL_WINDOW_S``) is what *actually*
          escalates SIGTERM→SIGKILL on the runner subprocesses; this
          flag is purely so the Telegram ack reply can surface that
          the server-side escalation has fired. We always send the
          ``session/cancel`` regardless — even on the first press —
          so a single ``/abort`` still gets the same imperative
          behaviour it had before.
        """
        sid = self.get_session_id(chat_id, thread_id)
        if sid is None:
            return False, False
        # Clients are cached per (chat, topic) — ``_get_or_open_client``
        # keys them on ``chat_key(chat_id, thread_id)`` ("<chat>:<thread>"
        # for forum topics). Look up with the SAME key: a bare
        # ``str(chat_id)`` misses the topic-keyed client, so a turn
        # running in a topic would get the bogus "no in-flight prompt to
        # cancel" while it kept running.
        client_key = TelegramConfig.chat_key(chat_id, thread_id)
        client = self._clients.get(client_key)
        # A cached client whose read-loop died (gateway crash / SSE
        # teardown flips ``_closed``) can't deliver the cancel — drop it
        # so the reopen path below replaces it.
        if client is not None and getattr(client, "_closed", False):
            self._clients.pop(client_key, None)
            client = None
        # When a turn is actually in flight (the per-(chat,topic) lock is
        # held) but we have no usable cached client, reopen one so
        # ``/abort`` ALWAYS reaches the gateway. The gateway routes
        # ``session/cancel`` by sid regardless of which connection
        # carries it, so a fresh connection still kills the running turn.
        # We only reopen when a turn is live — otherwise a stray
        # ``/abort`` with nothing running would needlessly spin up a
        # connection instead of replying "no in-flight prompt to cancel".
        if client is None and self.chat_lock(chat_id, thread_id).locked():
            try:
                client = await self._get_or_open_client(chat_id, thread_id)
            except Exception:  # noqa: BLE001
                self._log.exception(
                    "cancel_chat reopen failed chat=%s thread=%s",
                    chat_id,
                    thread_id,
                )
                client = None
        if client is None:
            return False, False
        now_mono = time.monotonic()
        key = (chat_id, thread_id)
        is_abort = via == "abort"
        # ``/abort`` is the imperative HARD kill — SIGKILL + killpg the
        # worker and every runner / subagent. ``/stop`` / ``/cancel`` are
        # graceful by contract (let the in-flight op finish, then end the
        # turn), so mashing them must NOT silently upgrade to a kill. The
        # double-press escalation is therefore tracked for ``/abort`` only;
        # ``force_escalated`` is purely the flag the ack reply uses to
        # surface that a second abort landed inside the window.
        if is_abort:
            last_mono = self._last_abort_at.get(key)
            force_escalated = bool(last_mono is not None and (now_mono - last_mono) <= _ABORT_DOUBLE_PRESS_S)
            self._last_abort_at[key] = now_mono
        else:
            force_escalated = False
        force = is_abort
        try:
            await client.session_cancel(sid, force=force)
        except Exception:  # noqa: BLE001
            self._log.exception(
                "session_cancel failed chat=%s thread=%s via=%s",
                chat_id,
                thread_id,
                via,
            )
            return False, force_escalated
        # Log the ACTUAL force flag sent (a first ``/abort`` is forceful);
        # ``escalated`` separately records the double-press window hit.
        self._log.info(
            "abort_dispatched chat=%s thread=%s via=%s force=%s escalated=%s sid=%s",
            chat_id,
            thread_id,
            via,
            force,
            force_escalated,
            sid[:12],
        )
        return True, force_escalated

    async def _dispatch_abort(
        self,
        chat_id: int,
        thread_id: int | None,
        *,
        via: str,
        reply: Any,
    ) -> None:
        """Shared imperative cancel path for ``/stop``, ``/cancel``,
        ``/abort``.

        Fires ``session/cancel`` immediately (no chat-lock wait — the
        caller has already routed slash commands outside the lock so
        the cancel is dispatched even while a turn is mid-stream),
        then echoes a wall-clock-stamped ack. A second press within
        ``_ABORT_DOUBLE_PRESS_S`` flags the bridge-side force-kill
        escalation in the reply so the operator can tell which press
        upgraded SIGTERM→SIGKILL.

        ``reply`` is the per-command reply closure built by the
        outer handler (it carries the thread routing for forum
        topics so we don't have to re-derive it).
        """
        sent_at = time.strftime("%H:%M:%S")
        stopped, escalated = await self.cancel_chat(
            chat_id,
            thread_id,
            via=via,
        )
        if not stopped:
            await reply("no in-flight prompt to cancel")
            return
        if via == "abort":
            await self._abort_live_turn(chat_id, thread_id)
        suffix = " · force-kill escalation" if escalated else ""
        await reply(f"⏹ session/cancel sent at {sent_at} (/{via}){suffix}")

    async def _abort_live_turn(
        self,
        chat_id: int,
        thread_id: int | None,
    ) -> None:
        """Instant UI teardown for a hard abort — engine cancel is async."""
        ck = TelegramConfig.chat_key(chat_id, thread_id)
        live = self._live_turns.get(ck)
        if not live:
            return
        roller = live.get("roller")
        dispatch = live.get("dispatch")
        typing = live.get("typing")
        if roller is not None:
            try:
                await roller.abort_fast("⏹ aborted")
            except Exception:  # noqa: BLE001
                self._log.debug("abort_fast failed chat=%s", ck, exc_info=True)
        if dispatch is not None and not dispatch.done():
            dispatch.cancel()
        if typing is not None and not typing.done():
            typing.cancel()

    async def _dispatch_steer(
        self,
        chat_id: int,
        thread_id: int | None,
        text: str,
        reply: Any,
    ) -> None:
        """Redirect the running turn with a steering message (``/steer``).

        Routed OUTSIDE the per-chat lock (like ``/stop``) so it lands
        even while a turn is mid-stream. Two backends, chosen by the
        chat's engine kind:

        * **local-LLM turn in flight** → push the text into the engine's
          mid-turn inject buffer via ``session/inject_user_message``; it
          drains at the next round boundary so the model pivots without
          aborting.
        * **CLI-bridge turn in flight** (claude/codex/antigravity) → the
          CLI can't accept a live inject, so gracefully abort the
          current turn (SIGINT → 15s → kill, preserving its ``--resume``
          state) and re-dispatch the text as the next turn, which the
          per-chat lock naturally sequences after the wind-down.

        With **no turn in flight** (either engine) ``/steer <text>`` is
        just a normal prompt. Empty ``/steer`` prints usage.
        """
        text = (text or "").strip()
        if not text:
            await reply("usage: /steer <text> — redirect the running turn")
            return

        sid = self.get_session_id(chat_id, thread_id)
        agent_cfg = self._agent_cfg(chat_id, thread_id)
        engine = (getattr(agent_cfg, "engine", "") or "").strip().lower()
        is_cli = engine in ("claude_cli", "codex_cli", "antigravity_cli", "zcode_cli")
        live = self.chat_lock(chat_id, thread_id).locked()

        if live and not is_cli and sid:
            # Local-LLM turn → true mid-turn inject. Same per-topic
            # client keying as cancel_chat — bare str(chat_id) would
            # miss a forum-topic client.
            client = self._clients.get(TelegramConfig.chat_key(chat_id, thread_id))
            if client is not None:
                inject = getattr(client, "session_inject_user_message", None)
                if callable(inject):
                    try:
                        ack = await inject(sid, text)
                    except Exception as exc:  # noqa: BLE001
                        ack = {"ok": False, "reason": str(exc)}
                    if hasattr(ack, "get") and ack.get("ok"):
                        await reply("↳ steering current turn — lands at the next round boundary")
                        return
                    # inject unavailable/failed → fall through to a normal
                    # dispatch so the message is never lost.
        elif live and is_cli:
            # CLI bridge → graceful abort, then pivot to the new prompt.
            stopped, _ = await self.cancel_chat(
                chat_id,
                thread_id,
                via="steer",
            )
            if stopped:
                await reply("↳ steering — winding down the current turn, then pivoting to your message")

        # No live turn, CLI pivot, or a failed inject: run the steer text
        # as a normal turn. When a CLI turn is still winding down this
        # blocks on the per-chat lock until it releases — i.e. the new
        # turn lands right after, with full --resume context.
        artifacts: list = []
        reply_text = await self.handle_text_message(
            chat_id,
            text,
            thread_id=thread_id,
            artifacts=artifacts,
        )
        if reply_text:
            await self._reply(chat_id, reply_text, thread_id)
        # Files the turn linked go after the reply text.
        await self._deliver_artifacts(chat_id, thread_id, artifacts)

    def _is_authorized(self, chat_id: int) -> bool:
        return chat_id in set(self.tg_cfg.allowed_chat_ids or [])

    def _user_allowed(self, user_id: int | None) -> bool:
        """Per-user allowlist gate.

        Empty ``allowed_user_ids`` ⇒ feature disabled, every member of
        an allowed chat may interact (legacy behavior). Non-empty list
        ⇒ strict — only those user ids' messages are processed; every
        other sender is dropped silently (no reply, no log spam) so
        the bot doesn't advertise its presence to non-permitted
        members of a shared group.
        """
        allowed = self.tg_cfg.allowed_user_ids or []
        if not allowed:
            return True
        if user_id is None:
            return False
        if int(user_id) not in set(allowed):
            return False
        return self.tg_cfg.can_read(int(user_id))

    def _addressed_to_bot(self, raw_text: str, msg: dict) -> bool:
        """True when an inbound group message is targeted at this bot.

        Recognises three signals:
          1. ``@<botname>`` anywhere in the raw (un-stripped) text —
             covers both ``/cmd@<botname>`` and conversational
             mentions.
          2. Direct reply (``msg.reply_to_message``) whose author is
             this bot. Telegram's "swipe to reply" UX is the
             standard way to address a user without typing the
             @handle; treat it as an implicit mention.
          3. ``message.entities`` of type ``mention``/``text_mention``
             targeting this bot — covers cases where the client
             splits the @handle across formatting (rare but
             possible) and `text_mention` mentions of the bot's
             user id without an @handle.
        """
        if self._bot_username:
            handle = "@" + self._bot_username
            if handle.lower() in (raw_text or "").lower():
                return True
        reply_to = msg.get("reply_to_message") or {}
        reply_from = (reply_to.get("from") or {}).get("id")
        if self._bot_user_id and isinstance(reply_from, int) and reply_from == self._bot_user_id:
            return True
        for ent in msg.get("entities") or []:
            if not isinstance(ent, dict):
                continue
            etype = ent.get("type")
            if etype == "text_mention":
                target = (ent.get("user") or {}).get("id")
                if self._bot_user_id and isinstance(target, int) and target == self._bot_user_id:
                    return True
        return False

    def _strip_bot_mention(self, text: str) -> str:
        """Strip ``@<botname>`` from inbound message text.

        Telegram delivers two flavors of bot mention that the agent
        should NOT see verbatim:

        * Slash commands in groups arrive as ``/cmd@<botname> [args]``
          (group members must disambiguate which bot the command is
          for). The literal ``cmd == "/new"`` checks in
          ``_handle_command`` would miss ``/new@<botname>`` and fall
          through to "unknown command".
        * Natural-language mentions arrive as ``@<botname> rest of
          message`` — the user is addressing the bot conversationally;
          the model should see only ``rest of message``.

        We strip the @<botname> token wherever it appears (both
        forms), case-insensitive, and collapse the surrounding
        whitespace. No-op until ``getMe`` populates
        ``self._bot_username`` (early-message edge case during a
        cold restart) so we never mangle text by accident.
        """
        if not self._bot_username:
            return text
        token = "@" + self._bot_username
        token_lower = token.lower()
        text_lower = text.lower()
        if token_lower not in text_lower:
            return text
        # Locate every occurrence in the original text by walking
        # the lower-cased copy in lockstep, then splice them out.
        out: list[str] = []
        i = 0
        n = len(text)
        while i < n:
            idx = text_lower.find(token_lower, i)
            if idx == -1:
                out.append(text[i:])
                break
            out.append(text[i:idx])
            i = idx + len(token)
        stripped = "".join(out)
        # Collapse runs of whitespace introduced by the splice (e.g.
        # ``"hi @bot there"`` → ``"hi  there"`` → ``"hi there"``)
        # without touching newlines that may carry meaning.
        import re

        stripped = re.sub(r"[ \t]+", " ", stripped).strip()
        return stripped

    async def _deny(self, chat_id: int, msg: dict) -> None:
        """One-shot deny reply. Logs at WARNING the first time, then
        silently drops further messages from the same chat.
        """
        if chat_id in self._denied_chats:
            return
        self._denied_chats.add(chat_id)
        sender = msg.get("from") or {}
        # The deny template still has an ``{agent}`` slot for legacy
        # log-grep compatibility; with multi-agent routing the deny
        # decision is bot-wide (allowlist applies before agent
        # resolution), so report the route the chat *would* have
        # taken if it had been allowed in.
        would_route_to = self.tg_cfg.agent_for(chat_id)
        self._log.warning(
            DENY_LOG_TEMPLATE.format(
                chat_id=chat_id,
                username=sender.get("username") or "",
                first_name=sender.get("first_name") or "",
                config_path=str(self.tg_cfg.config_path),
                agent=would_route_to,
            ),
        )
        thread_id_raw = (msg or {}).get("message_thread_id")
        deny_thread_id: int | None = (
            int(thread_id_raw)
            if isinstance(thread_id_raw, (int, str)) and str(thread_id_raw).lstrip("-").isdigit()
            else None
        )
        await self._reply(
            chat_id,
            "⛔ This bot is private. Your chat is not on the allowlist.\n\n"
            f"Your chat_id is `{chat_id}`. The bot operator can add it "
            "to the allowlist to grant access.",
            deny_thread_id,
        )


# ---------------------------------------------------------------------------
# Shared helpers/constants owned by ``bot.py`` (imported at BOTTOM to keep the
# bot<->mixin import cycle deadlock-free — see module docstring).
# ---------------------------------------------------------------------------
from sharedagents_telegram.bot import (  # noqa: E402
    _ABORT_DOUBLE_PRESS_S,
    _MAX_IMAGE_BYTES,
    _OFFICE_EXTENSIONS,
    ACP_DOCUMENT_UNSUPPORTED_REPLY,
    ACP_IMAGE_FORMAT_UNSUPPORTED_REPLY,
    ACP_IMAGE_UNSUPPORTED_REPLY,
    ACP_STICKER_UNSUPPORTED_REPLY,
    DENY_LOG_TEMPLATE,
    TelegramConfig,
    TranscriptionUnavailableError,
    _convert_office_bytes_to_markdown,
    _truncate_for_telegram,
)
