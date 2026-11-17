"""One realtime voice session: protocol event loop + turn pipeline.

Owns the VAD segmenter, the conversation history and at most ONE in-flight
response. VAD has no turn-taking authority: a client wake event opens the
channel, a terminal server-transcribed ``over`` dispatches the turn, and an
explicit client interrupt event is the sole speech-driven cancellation path.

The session persists across a connection loss: when a reconnecting client
passes a resume id, the new session REBINDS to that existing gateway
conversation instead of minting a fresh one.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable

from . import audio
from .cascade import Cascade, detect_lang, looks_like_noise, parse_languages, speakable, split_sentences
from .config import RealtimeConfig
from .endword import match_end_word
from .noise import strip_cjk_artifacts
from .vad import EnergyVAD, VadConfig

_LOG = logging.getLogger("sharedagents.realtime.session")

SendFn = Callable[[dict], Awaitable[None]]

# Reasoning the voice path must never read aloud. Most models wrap it; the relay
# prompt handles untagged chain-of-thought.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# A background result that keeps getting cancelled (e.g. by repeated wake-word
# preemptions) is requeued at most this many times, then dropped — a flapping
# cancel must not loop the same result forever.
_MAX_RESULT_VOICE_ATTEMPTS = 2

_WHOLE_REVIEW_TIMEOUT_S = 15.0


def _eid() -> str:
    return uuid.uuid4().hex[:24]


class RealtimeSession:
    def __init__(
        self,
        send: SendFn,
        cascade: Cascade,
        cfg: RealtimeConfig,
        resume_session: str | None = None,
        resume_registry=None,
    ) -> None:
        self._send = send
        self._cascade = cascade
        self._cfg = cfg
        self.id = "sess_" + _eid()
        self._messages: list[dict] = [{"role": "system", "content": cfg.instructions}]
        self._vad: EnergyVAD | None = EnergyVAD(VadConfig(**cfg.vad))
        self._response_task: asyncio.Task | None = None
        self._closed = False
        # Agent mode (cfg.agent set): lazily-built gateway client + the Jarvis
        # session id this voice session drives. Both stay None on the shipped
        # raw-cascade path.
        self._gateway = None
        self._jarvis_sid: str | None = None
        # Resume: a reconnecting client passes the gateway
        # session id it was previously bound to. If that conversation still
        # exists on the gateway, _ensure_jarvis_session REBINDS to it (same
        # history/context) instead of minting a fresh one; if it's gone/expired
        # we fall back to a fresh session cleanly. Consumed once, on first use.
        self._resume_session: str | None = resume_session or None
        # On binding a gateway conversation we
        # mint an unguessable resume token via this registry and hand it to the
        # client in session.config; only THAT token lets the client rebind on a
        # later reconnect. None in raw-cascade mode / unit tests → no token
        # issued (resume is agent-mode-only). ``resume_session`` above is already
        # the sid the handler RESOLVED from an incoming token, so it is trusted.
        self._resume_registry = resume_registry
        self._resume_token: str | None = None
        # Serialize gateway-session resolution so a resume probe and the first
        # audio turn can't race into two gateway sessions (both call
        # _ensure_jarvis_session concurrently: start()'s announce vs. a fast
        # replayed audio turn).
        self._session_lock = asyncio.Lock()
        self._config_task: asyncio.Task | None = None  # bg gateway-session announce
        # Background-delegation results, voiced FIFO at the next conversational
        # pause (never interrupting). A tailer task streams the Jarvis session
        # for ``subagent_result`` events and enqueues them; ``_seen_results``
        # dedupes by job id so a stream reconnect can't double-voice.
        self._result_queue: list[dict] = []
        self._seen_results: set[str] = set()
        self._results_task: asyncio.Task | None = None
        # Monotonic generation counter — incremented each _start_response so
        # the speak() tasks can detect stale TTS output from a cancelled turn
        # and drop it instead of emitting audio the user talked over.
        self._response_gen = 0
        # Interactive microphone audio is accepted only from a client that has
        # declared local wake gating AND sent an explicit wake/open event. This
        # server check is defense in depth for the client-side privacy boundary.
        self._client_gated = False
        self._channel_open = False
        # After a turn is dispatched the channel closes, but the streaming
        # client keeps sending audio frames for the round-trip it takes to
        # receive session.wake_reset. Those trailing frames are EXPECTED — drop
        # them silently until this monotonic deadline instead of erroring per
        # frame (the 'audio channel is closed' storm on every turn).
        self._drain_until = 0.0
        self._last_channel_err_s = 0.0
        # Server-side stop word. Every VAD-bounded transcript segment is buffered
        # until a terminal match; silence alone never dispatches the turn.
        self._pending_segments: list[str] = []
        self._pending_audio = bytearray()
        self._review_overflow = False
        self._max_review_bytes = 2 * audio.WIRE_RATE * max(1, cfg.max_utterance_s)
        self._collecting = False
        cw = (getattr(cfg, "conversation_end_word", "") or "over").strip().lower()
        self._conv_end_aliases = ([cw] if cw else []) + [
            a.strip().lower() for a in getattr(cfg, "conversation_end_aliases", []) or [] if a.strip()
        ]
        self._conversation_max_segments = int(getattr(cfg, "conversation_max_segments", 40) or 0)
        # True while a user turn is being prepared (ASR in flight, before the
        # response task is spawned) — part of the "busy" / not-a-pause check.
        self._turn_pending = False
        # True after a text user item is created but before its response.create
        # spawns the reply — also a "busy" (not-a-pause) window.
        self._pending_user_item = False
        self._tailer_retry_s = 1.0  # backoff before the result tailer resubscribes
        # Languages the user speaks (from asr_language) and the locked language
        # for the current reply's TTS voice — detected once per turn so the
        # cloned voice stays consistent across the reply's chunks.
        self._langs = parse_languages(cfg.asr_language) or ["en"]
        self._reply_lang = self._langs[0]
        # Live partial transcription (cfg.transcribe_partials). One in-flight
        # re-decode at a time; a stable item id ties the partial deltas to the
        # final ``…completed`` so the client updates ONE live caption bubble.
        self._partial_task: asyncio.Task | None = None
        self._partial_item_id: str | None = None
        self._last_partial_text = ""
        self._last_partial_ms = 0.0  # monotonic ms of the last partial kick
        # Need a little audio before the first re-decode (avoids transcribing a
        # single noisy frame). ~300 ms of 16-bit mono at the wire rate.
        self._min_partial_bytes = int(2 * audio.WIRE_RATE * 0.3)

    # ------------------------------------------------------------------

    async def start(self) -> None:
        await self._emit("session.created")
        # Agent mode: create the gateway agent (Jarvis) session and tell the
        # browser its id, so the web UI can subscribe to that session's canonical
        # event stream — the SAME path a typed session uses — for full parity
        # (reasoning, tool calls, stats), with realtime voice layered on top.
        # Done in the BACKGROUND so a slow/unreachable gateway never delays the
        # voice session; on failure the web simply falls back to plain transcript
        # bubbles.
        if self._cfg.agent:
            self._config_task = asyncio.create_task(self._announce_gateway_session())

    async def _announce_gateway_session(self) -> None:
        try:
            sid = await self._ensure_jarvis_session()
            if not self._closed:
                # gateway_session_id stays the REAL sid (the web UI subscribes to
                # that gateway session's canonical event stream); resume_token is
                # the unguessable handle the client echoes on reconnect.
                await self._emit(
                    "session.config", gateway_session_id=sid, agent=self._cfg.agent, resume_token=self._resume_token
                )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — never let this kill the voice session
            _LOG.exception("could not pre-create gateway session for web parity")

    async def close(self) -> None:
        self._closed = True
        self._clear_pending_utterance()
        await self._cancel_response(emit=False)
        if self._config_task is not None and not self._config_task.done():
            self._config_task.cancel()
            try:
                await self._config_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._config_task = None
        if self._partial_task and not self._partial_task.done():
            self._partial_task.cancel()
            try:
                await self._partial_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        # Stop the background results tailer before closing the gateway client.
        if self._results_task is not None:
            self._results_task.cancel()
            try:
                await self._results_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._results_task = None
        # Agent mode: release the gateway client's connection pool (httpx).
        if self._gateway is not None:
            try:
                await self._gateway.aclose()
            except Exception:  # noqa: BLE001
                pass
            self._gateway = None

    async def handle_event(self, ev: dict) -> None:
        etype = ev.get("type") or ""
        handler = {
            "session.update": self._on_session_update,
            "input_audio_buffer.wake": self._on_wake,
            "input_audio_buffer.append": self._on_append,
            "input_audio_buffer.commit": self._on_commit,
            "input_audio_buffer.clear": self._on_clear,
            "conversation.item.create": self._on_item_create,
            "response.create": self._on_response_create,
            "response.cancel": self._on_response_cancel,
            "session.interrupt": self._on_interrupt,
        }.get(etype)
        if handler is None:
            await self._error(f"unsupported event type: {etype!r}")
            return
        try:
            await handler(ev)
        except Exception as e:  # noqa: BLE001 — session must survive a bad turn
            _LOG.exception("event %s failed", etype)
            await self._error(f"{etype} failed: {e}")

    # -- client events ---------------------------------------------------

    async def _on_session_update(self, ev: dict) -> None:
        sess = ev.get("session") or {}
        if sess.get("instructions"):
            self._cfg.instructions = sess["instructions"]
            self._messages[0] = {"role": "system", "content": sess["instructions"]}
        if sess.get("voice"):
            self._cfg.voice = sess["voice"]
        if "wake_client_gated" in sess:
            self._client_gated = bool(sess["wake_client_gated"])
            if not self._client_gated:
                self._channel_open = False
        td = sess.get("turn_detection", "unset")
        if td is None:
            self._vad = None  # manual commit mode
        elif isinstance(td, dict):
            vc = VadConfig(**self._cfg.vad)
            if "silence_duration_ms" in td:
                vc.silence_ms = int(td["silence_duration_ms"])
            if "prefix_padding_ms" in td:
                vc.prefix_ms = int(td["prefix_padding_ms"])
            if "threshold" in td:  # 0..1 → scale onto RMS
                vc.min_rms = max(50.0, float(td["threshold"]) * 2000.0)
            self._vad = EnergyVAD(vc)
        await self._emit(
            "session.updated",
            session={
                "instructions": self._cfg.instructions,
                "voice": self._cfg.voice,
                "turn_detection": None if self._vad is None else {"type": "server_vad"},
            },
        )

    async def _on_append(self, ev: dict) -> None:
        if not self._client_gated:
            await self._error("interactive audio requires client-side wake gating")
            return
        if not self._channel_open:
            now = time.monotonic()
            # Expected trailing frames after a turn dispatch: drop silently.
            if now < self._drain_until:
                return
            # Genuinely unexpected (a gated client that never waked): emit ONE
            # debounced notice, never one-per-frame.
            if (now - self._last_channel_err_s) > 1.0:
                self._last_channel_err_s = now
                await self._error("audio channel is closed; send input_audio_buffer.wake first")
            return
        pcm = audio.b64_to_pcm(ev.get("audio") or "")
        if self._vad is None:
            self._manual_buffer().extend(pcm)
            return
        for kind, utterance in self._vad.feed(pcm):
            if kind == "speech_end":
                await self._emit("input_audio_buffer.speech_stopped")
                await self._begin_audio_turn(utterance)
        # While the user is still speaking, re-decode the in-progress buffer on
        # a cadence and stream it as a live caption (display-only). Throttled and
        # single-in-flight; opt-in via cfg.transcribe_partials.
        if self._cfg.transcribe_partials and self._vad is not None and self._vad.in_speech:
            self._maybe_kick_partial()
        # After each audio chunk, if we're at a pause (no speech in/pending,
        # nothing in flight), voice a queued background result. Runs on quiet
        # frames, so a sub-threshold blip (cough/click) that never became speech
        # still gets drained on the next silent frame.
        await self._maybe_drain_when_idle()

    async def _on_wake(self, ev: dict) -> None:
        """Open the locally wake-gated channel before its first audio frame."""
        if not self._client_gated:
            await self._error("wake rejected: client-side gating was not declared")
            return
        if self._response_task and not self._response_task.done():
            await self._error("wake rejected during response; use session.interrupt")
            return
        if self._channel_open:
            await self._emit("input_audio_buffer.wake_acknowledged", request_id=ev.get("request_id"))
            return
        self._channel_open = True
        self._drain_until = 0.0
        self._clear_pending_utterance()
        await self._emit("input_audio_buffer.wake_acknowledged", request_id=ev.get("request_id"))

    async def _on_interrupt(self, ev: dict) -> None:
        """Wake-word interrupt: the only speech-driven response cancellation."""
        if not self._client_gated:
            await self._error("interrupt rejected: client-side gating was not declared")
            return
        await self._cancel_response(emit=False)
        await self._emit("response.cancelled")
        self._channel_open = True
        self._clear_pending_utterance()
        await self._emit("session.interrupt_acknowledged", request_id=ev.get("request_id"))

    async def _on_commit(self, _ev: dict) -> None:
        if not self._client_gated or not self._channel_open:
            await self._error("audio commit rejected: interactive channel is closed")
            return
        if self._vad is not None:
            for kind, utterance in self._vad.flush():
                if kind == "speech_end":
                    await self._begin_audio_turn(utterance)
            await self._maybe_drain_when_idle()
            return
        buf = bytes(self._manual_buffer())
        self._manual_buffer().clear()
        if buf:
            await self._begin_audio_turn(buf)
        # Empty/noise commit spawned no response → voice a queued result.
        await self._maybe_drain_when_idle()

    async def _on_clear(self, _ev: dict) -> None:
        self._manual_buffer().clear()
        # Drop any half-collected end-word utterance — the buffer is being reset.
        self._clear_pending_utterance()
        if self._vad is not None:
            self._vad.flush()
        # Drop any live caption tied to the cleared buffer. No terminal event is
        # needed — the next utterance gets a fresh item id (a new bubble).
        if self._partial_task and not self._partial_task.done():
            self._partial_task.cancel()
            try:
                await self._partial_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._partial_item_id = None
        self._last_partial_text = ""
        self._last_partial_ms = 0.0

    async def _on_item_create(self, ev: dict) -> None:
        item = ev.get("item") or {}
        if item.get("type") != "message" or item.get("role") != "user":
            await self._error("only user message items are supported")
            return
        text = " ".join(
            c.get("text", "") for c in item.get("content") or [] if c.get("type") in ("input_text", "text")
        ).strip()
        if text:
            self._reply_lang = detect_lang(text, self._langs)
            self._messages.append({"role": "user", "content": text})
            # Awaiting the client's response.create — busy until then so a
            # background result can't slip in and race the requested reply.
            self._pending_user_item = True
            await self._emit(
                "conversation.item.created",
                item={"id": "item_" + _eid(), "role": "user", "content": [{"type": "input_text", "text": text}]},
            )

    async def _on_response_create(self, _ev: dict) -> None:
        if self._response_task and not self._response_task.done():
            await self._error("a response is already in progress")
            return
        self._pending_user_item = False
        self._spawn_response()

    async def _on_response_cancel(self, _ev: dict) -> None:
        self._pending_user_item = False
        await self._cancel_response()

    # -- turn pipeline -----------------------------------------------------

    async def _begin_audio_turn(self, utterance: bytes) -> None:
        # _turn_pending marks the ASR-in-flight window: VAD has already cleared
        # in_speech and no _response_task exists yet, so a background-result
        # drain could otherwise slip in and collide with the turn we're about to
        # spawn (two concurrent responses). Held until the response is spawned.
        self._turn_pending = True
        try:
            await self._emit("input_audio_buffer.committed")
            t0 = time.monotonic()
            # Even quiet/noise-filtered fragments can contain a soft word whose
            # meaning is recoverable only with the surrounding utterance.
            self._keep_review_audio(utterance)
            transcript = await self._cascade.transcribe(utterance)
            asr_ms = int(1000 * (time.monotonic() - t0))
            # Drop non-speech artifacts (silence, coughs, hallucinated CJK
            # filler) BEFORE surfacing them — no ghost user bubble, no reply.
            # Strip first: a MIXED transcript ("Hello 嗯。") has real speech in
            # it, so the whole-utterance guard would keep the hallucinated CJK
            # if we only asked "is this noise?".
            transcript = strip_cjk_artifacts(transcript)
            if not transcript or looks_like_noise(transcript, self._cfg.asr_language):
                _LOG.info(
                    "ignoring non-speech transcript %r (%.1fs audio)", transcript, len(utterance) / 2 / audio.WIRE_RATE
                )
                await self._finalize_partial("")  # clear any live caption
                return
            # Resolve the live caption with the authoritative final transcript
            # (same item id, so the client replaces the partial bubble in place).
            await self._finalize_partial(transcript)
            # Lock the reply voice language from the (usually full, clear) user
            # utterance — the model answers in the user's language (system
            # prompt), keeping the cloned voice consistent across the reply.
            self._reply_lang = detect_lang(transcript, self._langs)
            terminal, remainder = match_end_word(transcript, self._conv_end_aliases)
            if not terminal:
                self._pending_segments.append(transcript)
                self._collecting = True
                if 0 < self._conversation_max_segments <= len(self._pending_segments):
                    # A resource cap may abort malformed/unbounded input, but it
                    # must never manufacture an end-of-turn decision. Only the
                    # transcribed stop word is allowed to dispatch a response.
                    _LOG.warning("stop-word buffer hit %d segments; discarding", len(self._pending_segments))
                    self._clear_pending_utterance()
                    self._channel_open = False
                    self._drain_until = time.monotonic() + 2.0
                    await self._emit("session.wake_reset")
                    await self._error("turn discarded: stop word not received")
                return
            else:
                if remainder:
                    self._pending_segments.append(remainder)
                user_text = " ".join(self._pending_segments).strip()
            # Stop-word recognition has already decided the turn boundary.
            # Close capture before the bounded review; the review cannot open
            # or dispatch another turn, even if it invents a stop word.
            review_audio = bytes(self._pending_audio)
            self._clear_pending_utterance()
            self._channel_open = False
            self._drain_until = time.monotonic() + 2.0
            await self._emit("session.wake_reset")
            if review_audio:
                user_text = await self._review_utterance(review_audio, user_text)
            asr_ms = int(1000 * (time.monotonic() - t0))
            if self._closed or not user_text:
                return
            self._reply_lang = detect_lang(user_text, self._langs)
            self._messages.append({"role": "user", "content": user_text})
            self._spawn_response(asr_ms=asr_ms)
        finally:
            self._turn_pending = False

    def _clear_pending_utterance(self) -> None:
        self._pending_segments = []
        self._pending_audio.clear()
        self._review_overflow = False
        self._collecting = False

    def _keep_review_audio(self, pcm: bytes) -> None:
        if self._review_overflow:
            return
        if len(self._pending_audio) + len(pcm) > self._max_review_bytes:
            # A partial recording could silently drop the beginning of a turn.
            # Preserve the joined captions instead when the audio cap is hit.
            self._pending_audio.clear()
            self._review_overflow = True
            return
        self._pending_audio.extend(pcm)

    async def _review_utterance(self, pcm: bytes, fallback: str) -> str:
        try:
            reviewed = await asyncio.wait_for(
                self._cascade.transcribe(pcm, whole_utterance=True),
                timeout=_WHOLE_REVIEW_TIMEOUT_S,
            )
            reviewed = strip_cjk_artifacts(reviewed)
            _, reviewed = match_end_word(reviewed, self._conv_end_aliases)
            if not reviewed or looks_like_noise(reviewed, self._cfg.asr_language):
                return fallback
            # A review may repair words/punctuation, but a short tail or a
            # largely unrelated result must not replace the collected message.
            original_words = re.findall(r"\w+", fallback.casefold())
            review_words = re.findall(r"\w+", reviewed.casefold())
            review_vocabulary = set(review_words)
            overlap = sum(w in review_vocabulary for w in original_words)
            if original_words and (
                len(review_words) < 0.6 * len(original_words)
                or (len(original_words) >= 4 and overlap < 0.4 * len(original_words))
            ):
                _LOG.info("whole-utterance review lost content; using segment text")
                return fallback
            return reviewed
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — segment text is already usable
            _LOG.warning("whole-utterance review failed; using segment text", exc_info=True)
            return fallback

    # -- live partial transcription ---------------------------------------

    def _maybe_kick_partial(self) -> None:
        """Spawn a re-decode of the in-progress utterance if one is due and none
        is in flight. Single-in-flight so partials never pile up behind the
        final; cheap on the 1.7B ASR model."""
        if self._partial_task and not self._partial_task.done():
            return
        now = time.monotonic() * 1000
        if now - self._last_partial_ms < self._cfg.partial_interval_ms:
            return
        buf = self._vad.current_utterance if self._vad else b""
        if len(buf) < self._min_partial_bytes:
            return
        self._last_partial_ms = now
        if self._partial_item_id is None:
            self._partial_item_id = "item_" + _eid()
        self._partial_task = asyncio.create_task(self._run_partial(buf, self._partial_item_id))

    async def _run_partial(self, buf: bytes, item_id: str) -> None:
        """Re-decode ``buf`` and emit the current hypothesis as a live caption.

        Sends the FULL current text (marked ``partial``) keyed by ``item_id``;
        the client REPLACES that bubble — robust to the tail rewriting that
        re-decoding causes (OpenAI flags the same partial-revision behaviour)."""
        try:
            max_s = self._cfg.max_partial_window_s
            if max_s:
                max_bytes = 2 * audio.WIRE_RATE * max_s
                if len(buf) > max_bytes:
                    buf = buf[-max_bytes:]
            text = ""
            async for piece in self._cascade.transcribe_stream(buf):
                text += piece
            text = text.strip()
            # The utterance ended (or a new one began) while we decoded → drop.
            if item_id != self._partial_item_id:
                return
            # Same strip-first rule as the final transcript: a live caption
            # must never carry hallucinated CJK out of a mixed partial.
            text = strip_cjk_artifacts(text)
            if not text or looks_like_noise(text, self._cfg.asr_language):
                return
            if text == self._last_partial_text:
                return
            self._last_partial_text = text
            await self._emit(
                "conversation.item.input_audio_transcription.delta", item_id=item_id, delta=text, partial=True
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — a partial must never break the turn
            _LOG.debug("partial transcription failed", exc_info=True)

    async def _finalize_partial(self, transcript: str) -> None:
        """Resolve the live caption with exactly ONE terminal ``…completed``: the
        final ``transcript`` for a real turn, or an empty transcript (client
        clears the ghost bubble) when the turn was dropped (noise / no wake
        word). Cancels any in-flight partial and resets state. With partials off
        (no item id) and a real transcript this is the original single event."""
        item_id = self._partial_item_id
        if self._partial_task and not self._partial_task.done():
            self._partial_task.cancel()
            try:
                await self._partial_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._partial_task = None
        self._partial_item_id = None
        self._last_partial_text = ""
        self._last_partial_ms = 0.0
        if transcript:
            fields = {"transcript": transcript}
            if item_id:
                fields["item_id"] = item_id
            await self._emit("conversation.item.input_audio_transcription.completed", **fields)
        elif item_id:
            await self._emit("conversation.item.input_audio_transcription.completed", transcript="", item_id=item_id)

    def _spawn_response(self, asr_ms: int = 0) -> None:
        self._start_response(self._run_response(asr_ms))

    def _start_response(self, coro) -> None:
        """Run ``coro`` as the single in-flight response task, with a finish
        hook that drains the next queued background result (agent mode)."""
        self._response_gen += 1
        self._response_task = asyncio.create_task(coro)
        self._response_task.add_done_callback(self._on_response_finished)

    def _on_response_finished(self, task: asyncio.Task) -> None:
        # When a turn / result finishes cleanly (not explicitly interrupted) and
        # the user is silent, voice the next queued background result — chaining
        # through the backlog until it's empty or the user speaks.
        if self._closed or task.cancelled():
            return
        if self._cfg.agent and self._result_queue:
            asyncio.ensure_future(self._on_conversational_pause())

    async def _run_response(self, asr_ms: int) -> None:
        # Agent mode: drive the Jarvis gateway session instead of the raw chat
        # model. Default (cfg.agent == "") falls through to the shipped cascade
        # path below, byte-for-byte unchanged.
        if self._cfg.agent:
            user_text = next((m["content"] for m in reversed(self._messages) if m.get("role") == "user"), "")
            await self._run_response_agent(user_text, asr_ms=asr_ms)
            return
        rid = "resp_" + _eid()
        t0 = time.monotonic()
        ttft_ms = tts_first_ms = 0
        full_text = ""
        spoken_pcm_ms = 0
        gen = self._response_gen
        try:
            await self._emit("response.created", response={"id": rid})
            pending = ""
            tts_queue: asyncio.Queue[str | None] = asyncio.Queue()

            async def speak() -> None:
                nonlocal tts_first_ms, spoken_pcm_ms
                while True:
                    chunk = await tts_queue.get()
                    if chunk is None:
                        return
                    pcm = await self._cascade.tts(chunk, language=self._reply_lang)
                    if not pcm:
                        continue
                    # Stale: the user already spoke over this turn — don't emit
                    # audio the client would have to discard (or worse, play).
                    if self._response_gen != gen:
                        return
                    if not tts_first_ms:
                        tts_first_ms = int(1000 * (time.monotonic() - t0))
                    spoken_pcm_ms += int(len(pcm) / 2 / audio.WIRE_RATE * 1000)
                    await self._emit("response.output_audio.delta", response_id=rid, delta=audio.pcm_to_b64(pcm))

            speaker = asyncio.create_task(speak())
            try:
                async for piece in self._cascade.chat_stream(self._messages):
                    if not ttft_ms:
                        ttft_ms = int(1000 * (time.monotonic() - t0))
                    full_text += piece
                    await self._emit("response.output_audio_transcript.delta", response_id=rid, delta=piece)
                    pending += piece
                    chunks, pending = split_sentences(pending)
                    for c in chunks:
                        c = speakable(c)
                        if c:
                            await tts_queue.put(c)
                tail = speakable(pending)
                if tail:
                    await tts_queue.put(tail)
                await tts_queue.put(None)
                await speaker
            finally:
                if not speaker.done():
                    speaker.cancel()

            self._messages.append({"role": "assistant", "content": full_text.strip()})
            await self._emit(
                "response.done",
                response={
                    "id": rid,
                    "status": "completed",
                    "output_text": full_text.strip(),
                    "metrics": {
                        "asr_ms": asr_ms,
                        "llm_ttft_ms": ttft_ms,
                        "tts_first_audio_ms": tts_first_ms,
                        "audio_ms": spoken_pcm_ms,
                        "total_ms": int(1000 * (time.monotonic() - t0)),
                    },
                },
            )
        except asyncio.CancelledError:
            if full_text.strip():
                self._messages.append({"role": "assistant", "content": full_text.strip() + " …(interrupted)"})
            raise
        except Exception as e:  # noqa: BLE001
            _LOG.exception("response pipeline failed")
            await self._error(f"response failed: {e}")
            await self._emit("response.done", response={"id": rid, "status": "failed"})

    # -- agent mode --------------------------------------------------------

    def _ensure_gateway(self):
        """Lazily build the loopback gateway client (agent mode only)."""
        if self._gateway is None:
            from .gateway_client import GatewayClient

            self._gateway = GatewayClient(self._cfg.gateway_url, token=self._cfg.gateway_token or None)
        return self._gateway

    async def _ensure_jarvis_session(self) -> str:
        """Return the gateway session id this voice session drives.

        First use: if the client asked to RESUME a prior conversation
        (``resume_session``) and that gateway session still exists, rebind to
        it — the reconnected voice session continues the SAME conversation
        (history, context, in-flight backlog) rather than starting over. If the
        resume id is absent, gone, or expired, mint a fresh session attributed
        to ``cfg.agent``. Cached for the lifetime of the voice session; the
        lock makes concurrent callers converge on one session id.
        """
        async with self._session_lock:
            if not self._jarvis_sid:
                gc = self._ensure_gateway()
                resume = self._resume_session
                self._resume_session = None  # consume the resume hint once
                if resume and await gc.session_exists(resume):
                    self._jarvis_sid = resume  # rebind to the live conversation
                else:
                    self._jarvis_sid = await gc.ensure_session(self._cfg.agent)
                # Mint the resume token for THIS
                # gateway sid now, so session.config can hand it to the client.
                # Rotated on every bind (fresh or resumed); the client always
                # stores whatever session.config last carried.
                if self._resume_registry is not None:
                    self._resume_token = self._resume_registry.issue(self._jarvis_sid)
                self._start_results_tailer()
            return self._jarvis_sid

    # -- background-delegation results (pause-gated voicing) ---------------

    def _start_results_tailer(self) -> None:
        """Start the background task that tails the Jarvis session for
        ``subagent_result`` events and queues them for pause-gated voicing."""
        if self._results_task is None and self._jarvis_sid:
            self._results_task = asyncio.create_task(self._tail_results())

    async def _tail_results(self) -> None:
        gc = self._ensure_gateway()
        sid = self._jarvis_sid or ""
        # Resubscribe across stream EOF/errors so a transient gateway disconnect
        # doesn't permanently stop background-result voicing. Always from offset
        # 0; the per-job dedupe (``_seen_results``) makes a replay harmless.
        while not self._closed:
            try:
                async for ev in gc.stream_events(sid, 0):
                    if ev.get("type") == "subagent_result":
                        self._enqueue_subagent_result(ev)
                        # A result may land while the session is ALREADY idle
                        # (the delegate finished long after the turn ended) — no
                        # response-done hook or VAD event will fire to drain it,
                        # so voice it now if we're at a pause.
                        await self._maybe_drain_when_idle()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — a tailer error must not kill the session
                _LOG.exception("subagent-result tailer error; will resubscribe")
            if self._closed:
                break
            await asyncio.sleep(self._tailer_retry_s)  # backoff before resubscribe

    def _enqueue_subagent_result(self, ev: dict) -> None:
        """Append a background result to the FIFO voice queue (deduped by job)."""
        job = ev.get("job_id")
        if job:
            if job in self._seen_results:
                return
            self._seen_results.add(job)
        self._result_queue.append(ev)

    def _busy(self) -> bool:
        """True when this is NOT a conversational pause: a response is in flight,
        OR a user turn is being prepared (ASR), OR the user is mid-utterance."""
        if self._response_task and not self._response_task.done():
            return True
        if self._turn_pending or self._pending_user_item:
            return True
        # Mid-utterance with the end-word: the user is between segments waiting to
        # say "over" — don't voice a background result into their command.
        if self._collecting:
            return True
        # Channel state, never a VAD edge, controls conversational availability.
        # While OPEN, VAD may segment ASR input but cannot cause a response,
        # cancellation, or background-result turn.
        if self._channel_open:
            return True
        # Manual-commit mode (turn_detection disabled): the user is recording
        # into the manual buffer until they commit — don't talk over them.
        return bool(getattr(self, "_manual_pcm", None))

    async def _maybe_drain_when_idle(self) -> None:
        """Voice a queued result IF the session is at a pause right now. Used
        when a result arrives during silence (no turn-finish / VAD event would
        otherwise drain it)."""
        if self._closed or self._busy():
            return
        await self._on_conversational_pause()

    @staticmethod
    def _format_result_for_speech(r: dict) -> str:
        if r.get("ok") is False or r.get("error"):
            agent = r.get("agent") or "background"
            return f"The {agent} task didn't complete."
        # Strip any <think> reasoning the delegate may have left in its content
        # so we never read chain-of-thought aloud. Untagged reasoning is handled
        # by the relay prompt in _voice_result.
        return _THINK_RE.sub("", r.get("content") or "").strip()

    async def _on_conversational_pause(self) -> None:
        """At a VAD pause with nothing in flight, voice ONE queued result FIFO.

        The voicing runs as the tracked ``_response_task`` so an explicit interrupt cancels
        it (and ``_voice_result`` requeues the item). Never clears the queue."""
        if self._closed or self._busy():
            return
        if not self._result_queue:
            return
        result = self._result_queue.pop(0)
        self._start_response(self._voice_result(result))

    async def _voice_result(self, result: dict) -> None:
        """Voice one background result. Short (≤150 chars) is spoken directly;
        longer is handed back to Jarvis as a relay turn so it summarizes for
        speech. On explicit interrupt the result is requeued at the FIFO front so
        it is not lost."""
        text = self._format_result_for_speech(result)
        if not text:
            return
        result["_voice_attempts"] = result.get("_voice_attempts", 0) + 1
        try:
            if len(text) <= 150:
                await self._voice_text(text)
            else:
                # Relay through Jarvis to summarize for speech. The prompt is
                # explicit: read ONLY the answer, ignore any planning/reasoning/
                # tool steps the delegate's content may contain (the raw content
                # can be chain-of-thought, not a clean answer).
                await self._run_response_agent(
                    "Read out only the result below for the user, in one or two "
                    "short spoken sentences. Ignore any planning, reasoning, or "
                    f"tool-use narration it contains:\n\n{text}"
                )
        except asyncio.CancelledError:
            # Preserve on a genuine preemption — but bounded, so a flapping
            # cancel (e.g. repeated wake-word commands) can't loop the same
            # result forever (the 7x re-voice bug).
            if not self._closed and result["_voice_attempts"] < _MAX_RESULT_VOICE_ATTEMPTS:
                self._result_queue.insert(0, result)
            raise

    async def _voice_text(self, text: str) -> None:
        """Speak a ready-made line as a self-contained response (transcript +
        TTS audio). Used for short background results."""
        rid = "resp_" + _eid()
        t0 = time.monotonic()
        gen = self._response_gen
        try:
            await self._emit("response.created", response={"id": rid})
            await self._emit("response.output_audio_transcript.delta", response_id=rid, delta=text)
            chunks, tail = split_sentences(text)
            for c in list(chunks) + ([tail] if tail else []):
                c = speakable(c)
                if not c:
                    continue
                pcm = await self._cascade.tts(c, language=self._reply_lang)
                if not pcm:
                    continue
                # Stale: the user interrupted — don't emit old audio.
                if self._response_gen != gen:
                    return
                await self._emit("response.output_audio.delta", response_id=rid, delta=audio.pcm_to_b64(pcm))
            await self._emit(
                "response.done",
                response={
                    "id": rid,
                    "status": "completed",
                    "output_text": text,
                    "metrics": {"total_ms": int(1000 * (time.monotonic() - t0))},
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            _LOG.exception("voice-text pipeline failed")
            await self._error(f"voice failed: {e}")
            await self._emit("response.done", response={"id": rid, "status": "failed"})

    async def _run_response_agent(self, user_text: str, asr_ms: int = 0) -> None:
        """Drive one voice turn through the Jarvis gateway session.

        Submits the user transcript as a prompt, then feeds the session's
        ``text_delta`` events into the SAME sentence-split → TTS pipeline the
        cascade path uses, stopping at ``turn_ended``. Kept separate from
        :meth:`_run_response` so the shipped cascade path is untouched (its
        cancel/metrics semantics differ)."""
        rid = "resp_" + _eid()
        t0 = time.monotonic()
        ttft_ms = tts_first_ms = 0
        full_text = ""
        spoken_pcm_ms = 0
        gen = self._response_gen
        try:
            gc = self._ensure_gateway()
            sid = await self._ensure_jarvis_session()
            await self._emit("response.created", response={"id": rid})
            pending = ""
            tts_queue: asyncio.Queue[str | None] = asyncio.Queue()

            async def speak() -> None:
                nonlocal tts_first_ms, spoken_pcm_ms
                while True:
                    chunk = await tts_queue.get()
                    if chunk is None:
                        return
                    pcm = await self._cascade.tts(chunk, language=self._reply_lang)
                    if not pcm:
                        continue
                    if self._response_gen != gen:
                        return
                    if not tts_first_ms:
                        tts_first_ms = int(1000 * (time.monotonic() - t0))
                    spoken_pcm_ms += int(len(pcm) / 2 / audio.WIRE_RATE * 1000)
                    await self._emit("response.output_audio.delta", response_id=rid, delta=audio.pcm_to_b64(pcm))

            speaker = asyncio.create_task(speak())
            turn_status = "completed"
            turn_error: str | None = None
            try:
                off = await gc.submit_prompt(sid, user_text)
                async for ev in gc.stream_events(sid, off):
                    etype = ev.get("type")
                    if etype == "text_delta":
                        piece = ev.get("delta") or ""
                        if not piece:
                            continue
                        if not ttft_ms:
                            ttft_ms = int(1000 * (time.monotonic() - t0))
                        full_text += piece
                        await self._emit("response.output_audio_transcript.delta", response_id=rid, delta=piece)
                        pending += piece
                        chunks, pending = split_sentences(pending)
                        for c in chunks:
                            c = speakable(c)
                            if c:
                                await tts_queue.put(c)
                    elif etype == "turn_ended":
                        # The canonical turn_ended carries aborted/error (no
                        # "ok"); a wrapped form may carry ok=False. Treat any of
                        # them as a failed turn rather than reporting success.
                        if ev.get("aborted") or ev.get("error") or ev.get("ok") is False:
                            turn_status = "failed"
                            turn_error = ev.get("error") or ("aborted" if ev.get("aborted") else None)
                        break
                tail = speakable(pending)
                if tail:
                    await tts_queue.put(tail)
                await tts_queue.put(None)
                await speaker
            finally:
                if not speaker.done():
                    speaker.cancel()

            if full_text.strip():
                self._messages.append({"role": "assistant", "content": full_text.strip()})
            if turn_status != "completed":
                await self._error(f"agent turn {turn_status}" + (f": {turn_error}" if turn_error else ""))
            await self._emit(
                "response.done",
                response={
                    "id": rid,
                    "status": turn_status,
                    "output_text": full_text.strip(),
                    "metrics": {
                        "asr_ms": asr_ms,
                        "llm_ttft_ms": ttft_ms,
                        "tts_first_audio_ms": tts_first_ms,
                        "audio_ms": spoken_pcm_ms,
                        "total_ms": int(1000 * (time.monotonic() - t0)),
                    },
                },
            )
        except asyncio.CancelledError:
            # Explicit interrupt / ws-close: the gateway turn was submitted fire-and-forget
            # and keeps running server-side, so stop it too (best-effort; one
            # CancelledError has already been consumed, so this await is safe).
            if self._jarvis_sid and self._gateway is not None:
                try:
                    await self._gateway.cancel_turn(self._jarvis_sid)
                except Exception:  # noqa: BLE001
                    pass
            if full_text.strip():
                self._messages.append({"role": "assistant", "content": full_text.strip() + " …(interrupted)"})
            raise
        except Exception as e:  # noqa: BLE001
            _LOG.exception("agent response pipeline failed")
            await self._error(f"response failed: {e}")
            await self._emit("response.done", response={"id": rid, "status": "failed"})

    async def _cancel_response(self, emit: bool = True) -> None:
        task = self._response_task
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        if emit and not self._closed:
            await self._emit("response.cancelled")

    # -- plumbing ----------------------------------------------------------

    def _manual_buffer(self) -> bytearray:
        if not hasattr(self, "_manual_pcm"):
            self._manual_pcm = bytearray()
        return self._manual_pcm

    async def _emit(self, etype: str, **fields) -> None:
        if self._closed:
            return
        await self._send({"type": etype, "event_id": "evt_" + _eid(), "session_id": self.id, **fields})

    async def _error(self, message: str) -> None:
        await self._emit("error", error={"message": message})
