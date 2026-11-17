"""Long-poll run loop / update dispatch / shutdown methods for :class:`AcpTelegramBot`.

Extracted from the :class:`AcpTelegramBot` god-class (de-monolith). This is a
mixin: every method still binds ``self`` and dispatches identically through the
concrete class's MRO. Behaviour is unchanged — the method bodies moved verbatim.

The module-level helpers/constants these methods read stay owned by ``bot.py``
and are imported at the BOTTOM (after the class) so the ``bot`` <-> mixin import
cycle is deadlock-free in either import order.
"""

from __future__ import annotations

import asyncio
import os


class LifecycleMixin:
    async def close(self) -> None:
        """Tear down every cached client + its child process / WS session."""
        # Cancel the live cross-interface renderer observers first so they stop
        # tailing before the transports go away (cancel-safe, FIX 1).
        for key in list(self._render_log_tasks):
            self._stop_render_key(key)
        if self._render_log_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        *self._render_log_tasks.values(),
                        return_exceptions=True,
                    ),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                pass
        self._render_log_tasks.clear()
        for client in list(self._clients.values()):
            try:
                await client.close()
            except Exception:  # noqa: BLE001
                pass
            ws_session = getattr(client, "_telegram_ws_session", None)
            if ws_session is not None:
                try:
                    await ws_session.close()
                except Exception:  # noqa: BLE001
                    pass
        self._clients.clear()
        # SIGTERM any stdio children that didn't exit on stdin EOF.
        for proc in list(self._child_procs.values()):
            try:
                if proc.returncode is None:
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=2.0)
                    except asyncio.TimeoutError:
                        proc.kill()
            except Exception:  # noqa: BLE001
                pass
        self._child_procs.clear()
        if self._api is not None:
            try:
                await self._api.close()
            except Exception:  # noqa: BLE001
                pass

    # ---- long-poll loop -----------------------------------------------

    def request_stop(self) -> None:
        """SIGTERM-handler hook. Wakes the long-poll loop."""
        self._stop.set()

    async def _handle_update(self, update: dict) -> None:
        """Resolve one ``getUpdates`` entry — text/voice/photo/etc.

        Per-chat lock is taken inside :meth:`_dispatch_blocks` so slash
        commands (cancel etc.) bypass it and stay responsive while a
        turn is running.
        """
        # Inline-keyboard taps (permission prompts) arrive as
        # ``callback_query`` updates, not messages — route them out
        # before the message-shaped handling below.
        if update.get("callback_query"):
            await self._handle_callback_query(update)
            return
        msg = update.get("message") or update.get("edited_message")
        # DEBUG: dump the message shape so we can see what Telegram sent
        # for forwarded / photo / document edge cases. Remove after we
        # diagnose the missing-photo bug.
        if msg:
            self._log.info(
                "update keys=%s update.keys=%s",
                sorted(msg.keys()),
                sorted(update.keys()),
            )
        if not msg:
            return
        chat = msg.get("chat") or {}
        chat_id = chat.get("id")
        if not isinstance(chat_id, int):
            return
        # Forum topics: messages in a non-General topic carry
        # ``message_thread_id``. Non-forum chats and the General topic
        # leave it absent; we treat ``None`` as "reply at top level".
        thread_id_raw = msg.get("message_thread_id")
        thread_id: int | None = (
            int(thread_id_raw)
            if isinstance(thread_id_raw, (int, str)) and str(thread_id_raw).lstrip("-").isdigit()
            else None
        )
        # Learn the forum topic's human name from any update that carries it
        # (creation / rename service messages, or a topic reply) so the next
        # session bound in this thread is labelled with the real topic name.
        self._remember_topic_name(chat_id, thread_id, msg)
        raw_text = msg.get("text") or msg.get("caption") or ""
        raw_text = raw_text.strip()
        # In groups Telegram appends ``@<botname>`` to slash commands
        # and users prepend ``@<botname>`` for natural-language
        # addressing. Strip both BEFORE logging + dispatch so the
        # downstream literal command checks (``cmd == "/new"`` etc.)
        # match and the model doesn't see its own handle in every
        # turn's user message. ``raw_text`` is preserved separately
        # because the mention-required gate below needs to see the
        # ORIGINAL text to detect the @handle.
        text = self._strip_bot_mention(raw_text)
        sender = msg.get("from") or {}
        if not self._is_authorized(chat_id) or not self.tg_cfg.can_read(chat_id):
            await self._deny(chat_id, msg)
            return
        # Per-user allowlist (silent drop). Distinct from
        # ``allowed_chat_ids``: a chat may be on the allowlist while
        # the operator only wants the bot to respond to a specific
        # subset of members. Drop silently — replying with "you're
        # not allowed" would advertise the bot to every group member
        # who can read messages.
        sender_id_raw = sender.get("id")
        sender_id = int(sender_id_raw) if isinstance(sender_id_raw, int) else None
        if not self._user_allowed(sender_id):
            self._log.info(
                "drop chat=%s user=%s — not in allowed_user_ids",
                chat_id,
                sender_id,
            )
            return
        preview = text[:120].replace("\n", " ")
        self._log.info(
            "RX chat=%s thread=%s user=@%s len=%d text=%r%s",
            chat_id,
            thread_id,
            sender.get("username") or sender.get("id") or "",
            len(text),
            preview,
            "…" if len(text) > 120 else "",
        )
        # Mention-required gate (silent drop) — only in non-private
        # chats. DMs are 1:1 by definition, so requiring a mention
        # there would just be friction. In groups / supergroups /
        # channels the operator can opt in to "only when called"
        # via ``mention_required_in_groups: true`` in
        # ``<telegram_home>/config.json``; we then ignore everything
        # the user didn't explicitly @-mention or reply-to-bot.
        chat_type = str(chat.get("type") or "").lower()
        is_private = chat_type == "private"
        from sharedagents_core.reply_policy import should_reply

        rule = self.tg_cfg.reply_rule(chat_id, is_private=is_private)
        if not should_reply(
            rule,
            is_private=is_private,
            addressed_to_me=self._addressed_to_bot(raw_text, msg),
            text=raw_text,
            bot_username=self._bot_username,
        ):
            self._log.info(
                "drop chat=%s thread=%s — reply rule %s did not match",
                chat_id,
                thread_id,
                rule.get("reply"),
            )
            return
        # Acknowledge receipt with a 👀 reaction so the user gets
        # immediate visual confirmation that the bot saw the message
        # (independent of however long the model takes to reply).
        # Fire-and-forget — never block the turn on a reaction.
        msg_id = msg.get("message_id")
        if isinstance(msg_id, int) and self._api is not None:
            import asyncio as _asyncio

            _asyncio.create_task(
                self._api.call(
                    "setMessageReaction",
                    chat_id=chat_id,
                    message_id=msg_id,
                    reaction=[{"type": "emoji", "emoji": _REACT_SEEN}],
                    is_big=False,
                )
            )

        # Image / document → streaming turn (reply only on pre-dispatch
        # errors); sticker → explicit unsupported reply.
        if msg.get("photo"):
            reply_text = await self.handle_image_message(
                chat_id,
                msg,
                thread_id=thread_id,
            )
            if self._api is not None and reply_text:
                await self._reply(chat_id, reply_text, thread_id)
            return
        if msg.get("sticker"):
            if self._api is not None:
                await self._reply(
                    chat_id,
                    await self.handle_sticker_message(chat_id),
                    thread_id,
                )
            return
        if msg.get("document"):
            if self._api is not None:
                reply = await self.handle_document_message(
                    chat_id,
                    msg,
                    thread_id=thread_id,
                )
                if reply is not None:
                    await self._reply(chat_id, reply, thread_id)
            return

        # Voice / audio / video_note — go through STT, then text path.
        voice = msg.get("voice") or msg.get("audio") or msg.get("video_note")
        if voice:
            if self._stt is None or self._download_audio is None:
                if self._api is not None:
                    await self._reply(
                        chat_id,
                        "🎙 voice input received but STT is not configured. "
                        "Set OPENROUTER_API_KEY in "
                        f"{self.tg_cfg.home / '.env'} (or in the agent's "
                        ".env) and restart the bot.",
                        thread_id,
                    )
                return
            file_id = voice.get("file_id")
            if not file_id:
                return
            # Telegram's cloud Bot API can't download files over 20 MB,
            # so reject oversized audio up front with an actionable
            # message rather than the generic "couldn't download voice"
            # the silent getFile failure would otherwise produce.
            file_size = voice.get("file_size")
            _local = getattr(self._api, "local_mode", False)
            _cap = TG_LOCAL_MAX_BYTES if _local else TG_GETFILE_MAX_BYTES
            if isinstance(file_size, int) and file_size > _cap:
                self._log.info(
                    "voice rejected chat=%s file_size=%d cap=%d local=%s",
                    chat_id,
                    file_size,
                    _cap,
                    _local,
                )
                if self._api is not None:
                    _human = f"{file_size / 1_048_576:.0f} MB"
                    _limit = "2 GB" if _local else "20 MB"
                    await self._reply(
                        chat_id,
                        f"⚠️ that audio is {_human} — over my {_limit} limit, "
                        "so I can't transcribe it. Send a shorter clip.",
                        thread_id,
                    )
                return
            mime = (voice.get("mime_type") or "").lower()
            original_name = voice.get("file_name")  # present for audio, absent for voice/video_note
            await self._stream_turn_for_chat(
                chat_id,
                voice_file_id=file_id,
                voice_mime=mime,
                original_filename=original_name,
                thread_id=thread_id,
                untrusted=bool(
                    msg.get("forward_origin")
                    or msg.get("forward_from")
                    or msg.get("forward_sender_name")
                    or msg.get("forward_date")
                ),
                incoming_message_id=msg_id,
            )
            return

        if not text:
            return

        # Slash commands run outside the chat lock so /stop, /cancel,
        # and /abort stay responsive while a turn is in-flight — they
        # share the imperative cancel path in ``_dispatch_abort`` and
        # MUST land even if the chat lock is held by a streaming turn.
        if text.startswith("/"):
            await self._handle_command(chat_id, text, thread_id)
            return

        # A forwarded message carries content the sender RELAYED from
        # elsewhere — external data, not their own prompt. Mark it untrusted
        # so it is fenced before reaching the model (prompt-injection defense:
        # an allow-listed sender is trusted as an author, but forwarded text
        # they relay is not).
        forwarded = bool(
            msg.get("forward_origin")
            or msg.get("forward_from")
            or msg.get("forward_sender_name")
            or msg.get("forward_date")
        )

        # Durable enqueue for plain text messages sent while a local-LLM
        # turn is in-flight. Instead of letting the message stack behind
        # the per-chat lock (which silently queues it until the turn
        # finishes), push it into the session's durable follow-up queue
        # so the user gets an immediate ack. (The /queue command was removed;
        # the durable session/queue/* API it managed
        # stays — the message is still enqueued here.)
        # Forwarded (untrusted) messages fall through to the normal path —
        # the injection-guard fence must be applied by _stream_turn_for_chat.
        # CLI-bridge engines (claude/codex/antigravity) can't accept a
        # live inject either way; they use the normal lock-wait path.
        if not forwarded and self.chat_lock(chat_id, thread_id).locked():
            _agt_cfg = self._agent_cfg(chat_id, thread_id)
            _engine = (getattr(_agt_cfg, "engine", "") or "").strip().lower()
            _is_cli = _engine in ("claude_cli", "codex_cli", "antigravity_cli", "zcode_cli")
            if not _is_cli:
                _sid = self.get_session_id(chat_id, thread_id)
                _client = self._clients.get(TelegramConfig.chat_key(chat_id, thread_id))
                if _sid and _client is not None:
                    _qadd = getattr(_client, "session_queue_add", None)
                    if callable(_qadd):
                        try:
                            _ack = await _qadd(
                                _sid,
                                text,
                                source="telegram",
                                author=str(chat_id),
                            )
                        except Exception as _exc:  # noqa: BLE001
                            self._log.debug(
                                "session_queue_add failed chat=%s: %s",
                                chat_id,
                                _exc,
                            )
                            _ack = {}
                        if hasattr(_ack, "get") and _ack.get("ok"):
                            _qid = str(_ack.get("queue_id") or "")
                            _short = (_qid[:8] if len(_qid) > 8 else _qid) or "?"
                            # Advance the incoming message's lifecycle reaction
                            # from 👀 (seen) to 🤔 (queued). The durable-enqueue
                            # path returns HERE without ever reaching
                            # ``_stream_turn_for_chat`` — the only other place
                            # that sets ``_REACT_QUEUED`` — so before this the
                            # durably-queued message stayed stuck on 👀 while a
                            # placeholder-queued one flipped to 🤔. That
                            # inconsistency is the "queued messages don't have
                            # the queued emoji anymore" regression.
                            await self._set_status_reaction(
                                chat_id,
                                msg_id,
                                _REACT_QUEUED,
                            )
                            if self._api is not None:
                                await self._api.send_message(
                                    chat_id,
                                    f"\U0001f552 queued (#{_short})",
                                    message_thread_id=thread_id,
                                )
                            return

        await self._stream_turn_for_chat(
            chat_id,
            text=text,
            thread_id=thread_id,
            untrusted=forwarded,
            incoming_message_id=msg_id,
        )

    async def _drain_backlog(self) -> int:
        """Skip every Telegram update queued from before this startup.

        If the bot was offline for any length of time, Telegram has been
        accumulating updates server-side. Replaying them would mean
        responding to questions the user might have asked hours ago and
        already given up on.
        """
        if self._api is None:
            return 0
        skipped = 0
        offset = _load_update_offset(self.tg_cfg)
        while not self._stop.is_set():
            res = await self._api.call(
                "getUpdates",
                offset=offset,
                timeout=0,
                limit=100,
                allowed_updates=["message", "edited_message", "callback_query"],
            )
            if not res.get("ok"):
                self._log.warning(
                    "drain: getUpdates failed: %s",
                    res.get("description"),
                )
                break
            batch = res.get("result") or []
            if not batch:
                break
            last = batch[-1].get("update_id")
            if isinstance(last, int):
                offset = last + 1
                _save_update_offset(self.tg_cfg, offset)
            skipped += len(batch)
        if skipped:
            self._log.info(
                "drain: skipped %d backlog update(s) — bot only replies to new messages from now on",
                skipped,
            )
        return skipped

    async def _handle_update_safe(self, update: dict) -> None:
        try:
            await self._handle_update(update)
        except asyncio.CancelledError:
            raise
        except FileNotFoundError as e:
            # AgentConfig.load() raises this when agent_routes points
            # at a name with no on-disk config — same input will crash
            # again on every retry, so tell the user once instead of
            # silently crash-looping. Other exceptions stay as a
            # log-only swallow (intermittent / transient — a Telegram
            # reply would risk spam).
            self._log.warning(
                "agent config missing on update=%s: %s",
                update.get("update_id"),
                e,
            )
            await self._notify_handler_error(update, f"⚠️ {e}")
        except Exception:  # noqa: BLE001
            self._log.exception("handler crashed on update: %s", update)

    async def _notify_handler_error(self, update: dict, text: str) -> None:
        """Best-effort Telegram reply when ``_handle_update`` raises.

        Pulls chat_id / message_thread_id out of whichever update
        variant arrived (``message``, ``edited_message``, or the
        wrapped ``callback_query.message``). Silent no-op if we can't
        find a chat — better than logging twice for the same error.
        """
        if self._api is None:
            return
        msg = (
            update.get("message")
            or update.get("edited_message")
            or (update.get("callback_query") or {}).get("message")
            or {}
        )
        chat_id = (msg.get("chat") or {}).get("id")
        if chat_id is None:
            return
        try:
            await self._api.send_message(
                int(chat_id),
                text,
                message_thread_id=msg.get("message_thread_id"),
            )
        except Exception:  # noqa: BLE001
            self._log.exception(
                "failed to notify chat=%s of handler error",
                chat_id,
            )

    async def run(self) -> int:
        """Main long-poll loop. Honors ``request_stop`` (SIGTERM)."""
        # Write the pidfile from inside the bot so both the daemonised
        # ``sharedagents telegram start`` path and the systemd-managed
        # ``ExecStart=python -m sharedagents_telegram.bot`` path produce
        # the same on-disk artefacts (status / stop subcommands rely
        # on the file existing).
        try:
            self.tg_cfg.pidfile.parent.mkdir(parents=True, exist_ok=True)
            self.tg_cfg.pidfile.write_text(str(os.getpid()), encoding="utf-8")
        except OSError:
            self._log.exception("failed to write pidfile")
        if self._api is None:
            self._api = _TelegramAPI(
                self.tg_cfg.bot_token,
                api_base=os.environ.get("SHAREDAGENTS_TELEGRAM_API_BASE", TELEGRAM_API),
                log=self._log,
            )
        # Wire default audio download (always — uses the bot-file CDN)
        # and STT (only if OPENROUTER_API_KEY is set). Tests inject
        # their own; the production path lands here.
        if self._download_audio is None:
            self._download_audio = _make_default_audio_downloader(self._api)
        if self._download_image is None:
            self._download_image = self._download_audio
        if self._stt is None:
            # Resolve STT (primary + backup) from the unified sharedagents config
            # ``stt`` block — no hardcoded host/IP, no scattered env vars.
            from sharedagents_core.services import resolve_service

            svc = resolve_service("stt")
            if svc.configured:
                mismatch = svc.model_mismatch()
                if mismatch:
                    self._log.warning("%s", mismatch)
                self._stt = _make_default_stt(svc, self._log)
                eps = ", ".join(f"{e.label}={e.base_url}({e.model})" for e in svc.candidates())
                self._log.info("STT wired: multipart failover [%s]", eps)
            else:
                self._log.warning(
                    "STT not configured — voice input will be rejected. Add an "
                    '"stt" block (primary/backup) to ~/.sharedagents/config.json.',
                )
        # Resolve TTS from the unified ``tts`` block, matching STT's
        # primary/backup service config. Failure to wire is non-fatal — voice
        # replies just don't fire; text reply path stays unaffected.
        if self._tts is None:
            from sharedagents_core.services import resolve_service

            svc = resolve_service("tts")
            if svc.configured:
                self._tts = _make_default_tts(svc, self._log)
                eps = ", ".join(f"{e.label}={e.base_url}({e.model})" for e in svc.candidates())
                self._log.info(
                    "TTS wired: speech failover [%s] voice=%s",
                    eps,
                    OPENROUTER_TTS_DEFAULT_VOICE,
                )
            else:
                self._log.info(
                    'TTS unwired — add a "tts" block (primary/backup) to '
                    "~/.sharedagents/config.json to enable voice replies for "
                    "voice-input chats.",
                )
        # ``default_agent`` shown so operators can see which agent the
        # bot will route brand-new chats to; per-(chat,thread) routes
        # in ``agent_routes`` override that and are logged on first
        # use.
        self._log.info(
            "sharedagents-telegram-acp poller started default_agent=%s allow=%s trusted=%s acp_url=%s routes=%d",
            self.tg_cfg.default_agent,
            self.tg_cfg.allowed_chat_ids,
            sorted(self.trusted_chats),
            self.acp_url or "(stdio)",
            len(self.tg_cfg.agent_routes or {}),
        )
        if not self.tg_cfg.allowed_chat_ids:
            self._log.warning(
                "allowed_chat_ids is EMPTY — every incoming message "
                "will be denied. Add your Telegram chat_id to %s "
                "and `sharedagents telegram restart` to enable.",
                self.tg_cfg.config_path,
            )
        # Identify the bot.
        me = await self._api.call("getMe")
        if me.get("ok"):
            u = me["result"]
            self._bot_username = str(u.get("username") or "")
            try:
                self._bot_user_id = int(u.get("id") or 0)
            except (TypeError, ValueError):
                self._bot_user_id = 0
            self._log.info(
                "bot connected: @%s id=%s",
                u.get("username"),
                u.get("id"),
            )
            # Register slash commands with Telegram's Bot API so the
            # "/" autocomplete popup shows the actual commands the bot
            # handles. This runs every startup — Telegram persists the
            # last set; there's no stale-state concern.
            try:
                cmds_result = await self._api.call(
                    "setMyCommands",
                    commands=_PERSISTENT_COMMANDS,
                )
                if cmds_result.get("ok"):
                    self._log.info(
                        "registered %d commands via setMyCommands",
                        len(_PERSISTENT_COMMANDS),
                    )
                else:
                    self._log.warning(
                        "setMyCommands failed: %s",
                        cmds_result.get("description"),
                    )
            except Exception:  # noqa: BLE001
                self._log.exception("setMyCommands raised")
        else:
            self._log.error(
                "getMe failed: %s — token may be invalid",
                me.get("description"),
            )
            return 1

        # Verify any persisted chat→session bindings are still alive on
        # the agent side. Failures mark the chat stale; the next message
        # starts a fresh session.
        try:
            await self.verify_sessions_on_startup()
        except Exception:  # noqa: BLE001
            self._log.exception("verify_sessions_on_startup raised")
        try:
            await self._recover_inflight_turns_on_startup()
        except Exception:  # noqa: BLE001
            self._log.exception("inflight recovery raised")

        # Skip backlog so an offline bot doesn't replay stale conversation.
        await self._drain_backlog()

        try:
            while not self._stop.is_set():
                if self._dispatched:
                    self._dispatched = {t for t in self._dispatched if not t.done()}

                offset = _load_update_offset(self.tg_cfg)
                stop_wait = asyncio.create_task(self._stop.wait())
                poll_task = asyncio.create_task(
                    self._api.call(
                        "getUpdates",
                        offset=offset,
                        timeout=LONG_POLL_TIMEOUT_S,
                        allowed_updates=["message", "edited_message", "callback_query"],
                    )
                )
                done, _pending = await asyncio.wait(
                    {stop_wait, poll_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if stop_wait in done:
                    poll_task.cancel()
                    try:
                        await poll_task
                    except (asyncio.CancelledError, Exception):  # noqa: BLE001
                        pass
                    break
                stop_wait.cancel()
                try:
                    res = poll_task.result()
                except Exception as e:  # noqa: BLE001
                    self._log.warning("getUpdates raised: %s", e)
                    await asyncio.sleep(2.0)
                    continue
                if not res.get("ok"):
                    self._log.warning(
                        "getUpdates failed: %s",
                        res.get("description"),
                    )
                    await asyncio.sleep(2.0)
                    continue
                updates = res.get("result") or []
                for upd in updates:
                    uid = upd.get("update_id")
                    if isinstance(uid, int):
                        _save_update_offset(self.tg_cfg, uid + 1)
                    task = asyncio.create_task(
                        self._handle_update_safe(upd),
                    )
                    self._dispatched.add(task)
        finally:
            for task in list(self._dispatched):
                if not task.done():
                    task.cancel()
            if self._dispatched:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(
                            *self._dispatched,
                            return_exceptions=True,
                        ),
                        timeout=5.0,
                    )
                except asyncio.TimeoutError:
                    pass
            await self.close()
            # Clean up the pidfile so ``status`` reports stopped
            # without a stale entry. Best-effort — if another process
            # already replaced it, leave the new one alone.
            try:
                if self.tg_cfg.pidfile.is_file():
                    pid_text = self.tg_cfg.pidfile.read_text(encoding="utf-8").strip()
                    if pid_text == str(os.getpid()):
                        self.tg_cfg.pidfile.unlink()
            except OSError:
                pass
            self._log.info("sharedagents-telegram-acp poller stopped")
        return 0


# ---------------------------------------------------------------------------
# Shared helpers/constants owned by ``bot.py`` (imported at BOTTOM to keep the
# bot<->mixin import cycle deadlock-free — see module docstring).
# ---------------------------------------------------------------------------
from sharedagents_telegram.bot import (  # noqa: E402
    _PERSISTENT_COMMANDS,
    _REACT_QUEUED,
    _REACT_SEEN,
    LONG_POLL_TIMEOUT_S,
    OPENROUTER_TTS_DEFAULT_VOICE,
    TELEGRAM_API,
    TG_GETFILE_MAX_BYTES,
    TG_LOCAL_MAX_BYTES,
    TelegramConfig,
    _load_update_offset,
    _make_default_audio_downloader,
    _make_default_stt,
    _make_default_tts,
    _save_update_offset,
    _TelegramAPI,
)
