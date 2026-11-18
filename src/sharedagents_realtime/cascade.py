"""The STT → LLM → TTS cascade against OpenAI-compatible REST stages.

Every stage is a plain ``base_url + model + api_key`` (see config.py), so the
configured TTS proxy profile, OpenAI and OpenRouter are the same code path.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator

import httpx

from . import audio
from .config import RealtimeConfig, Stage
from .noise import (  # noqa: F401  (re-export — session.py imports both from cascade)
    looks_like_noise,
    parse_languages,
    sanitize_transcript,
)

_LOG = logging.getLogger("sharedagents.realtime.cascade")

# The configured TTS proxy SUMMARIZES inputs longer than ~160 chars through the 27B
# (TTS_SUMMARY_THRESHOLD) — chunks must stay under that or the spoken text
# diverges from the transcript. Shorter chunks also cut time-to-first-audio.
MAX_TTS_CHARS = 150

_SENTENCE_END = re.compile(r"(?<=[.!?…])[\s\n]+|\n+")
_THINK_OPEN, _THINK_CLOSE = "<think>", "</think>"


def split_sentences(buf: str) -> tuple[list[str], str]:
    """Return ``(complete_chunks, remainder)``; chunks are ≤ MAX_TTS_CHARS."""
    parts = _SENTENCE_END.split(buf)
    if not parts:
        return [], buf
    complete, rest = parts[:-1], parts[-1]
    out: list[str] = []
    for p in complete:
        out.extend(_hard_wrap(p.strip()))
    # Oversized remainder with no sentence end yet → flush at a word boundary.
    while len(rest) > MAX_TTS_CHARS:
        cut = rest.rfind(" ", 0, MAX_TTS_CHARS)
        cut = cut if cut > 40 else MAX_TTS_CHARS
        out.append(rest[:cut].strip())
        rest = rest[cut:].lstrip()
    return [c for c in out if c], rest


def _hard_wrap(text: str) -> list[str]:
    if len(text) <= MAX_TTS_CHARS:
        return [text] if text else []
    out = []
    while len(text) > MAX_TTS_CHARS:
        cut = text.rfind(" ", 0, MAX_TTS_CHARS)
        cut = cut if cut > 40 else MAX_TTS_CHARS
        out.append(text[:cut].strip())
        text = text[cut:].lstrip()
    if text:
        out.append(text)
    return out


def speakable(text: str) -> str:
    """Light cleanup so stray markdown never reaches the synthesizer."""
    text = re.sub(r"[*_`#>|]", "", text)
    return re.sub(r"\s{2,}", " ", text).strip()


# Qwen3-ASR streams a `language <lang><asr_text>…` preamble before the actual
# transcript (the non-stream `text` field strips it server-side; the SSE token
# stream does not). The sentinel can arrive split across token deltas.
_ASR_TEXT_SENTINEL = "<asr_text>"
# If we never see the sentinel within this many buffered chars, assume the
# backend emits no preamble (a different ASR/profile) and pass text through.
_ASR_PREAMBLE_MAXLEN = 64


class AsrStreamFilter:
    """Strip the Qwen3-ASR ``language <lang><asr_text>`` preamble from a streamed
    token sequence, yielding only the transcript that follows the sentinel.

    Stateful and incremental: ``feed(token)`` returns the cleaned text to emit
    so far (``""`` while still inside the preamble), ``flush()`` returns anything
    held back when the stream ends. Tolerant of a sentinel split across tokens
    and of backends that emit no preamble at all (pass-through fallback)."""

    def __init__(self) -> None:
        self._seen = False
        self._buf = ""

    def feed(self, token: str) -> str:
        if self._seen:
            return token
        self._buf += token
        idx = self._buf.find(_ASR_TEXT_SENTINEL)
        if idx >= 0:
            self._seen = True
            out = self._buf[idx + len(_ASR_TEXT_SENTINEL) :]
            self._buf = ""
            return out
        # No sentinel yet. Once we've buffered more than a plausible preamble,
        # assume there is none and start passing text through verbatim.
        if len(self._buf) > _ASR_PREAMBLE_MAXLEN:
            self._seen = True
            out, self._buf = self._buf, ""
            return out
        return ""

    def flush(self) -> str:
        """Stream ended: a sentinel-less short buffer is the whole transcript."""
        if self._seen:
            return ""
        self._seen = True
        out, self._buf = self._buf, ""
        return out


_PT_DIACRITIC = re.compile(r"[ãõçáàâéêíóôúü]", re.I)
# PT cues that are NOT also common English words (short ambiguous ones omitted).
_PT_WORDS = frozenset(
    [
        "que",
        "com",
        "para",
        "mais",
        "como",
        "mas",
        "você",
        "voce",
        "não",
        "nao",
        "sim",
        "obrigado",
        "obrigada",
        "bom",
        "dia",
        "boa",
        "noite",
        "tarde",
        "claro",
        "posso",
        "pode",
        "podemos",
        "ajudar",
        "fazer",
        "está",
        "esta",
        "estão",
        "sou",
        "estou",
        "quero",
        "queria",
        "gostaria",
        "mercado",
        "ações",
        "dólar",
        "onde",
        "quando",
        "porque",
        "também",
        "já",
        "muito",
        "tudo",
        "bem",
        "fala",
        "falar",
        "tem",
        "têm",
        "são",
        "é",
        "então",
        "isso",
        "aqui",
        "agora",
        "hoje",
        "amanhã",
        "ontem",
        "coisa",
        "gente",
    ]
)


def detect_lang(text: str, candidates: list[str] | None = None) -> str:
    """Pick the spoken language for TTS voice routing among the configured
    candidates (default en/pt). Robust on SHORT text where the proxy's own
    heuristic fails (e.g. "Bom dia! Claro, posso ajudar." → pt). Returns the
    first candidate when undecided so a single-language setup is unaffected."""
    cands = candidates or ["en", "pt"]
    if "pt" in cands:
        t = text.lower()
        if _PT_DIACRITIC.search(t):
            return "pt"
        words = re.findall(r"[a-zà-ÿ]+", t)
        if words and sum(w in _PT_WORDS for w in words) / len(words) >= 0.15:
            return "pt"
    return "en" if "en" in cands else cands[0]


class Cascade:
    def __init__(self, cfg: RealtimeConfig) -> None:
        self.cfg = cfg
        # Cascade instances belong to one websocket session.
        self._sticky_language: str | None = None
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))

    async def aclose(self) -> None:
        await self._client.aclose()

    @staticmethod
    def _headers(stage: Stage) -> dict[str, str]:
        return {"Authorization": f"Bearer {stage.api_key}"} if stage.api_key else {}

    # -- STT ------------------------------------------------------------

    def _stt_stages(self) -> list[Stage]:
        return [s for s in [self.cfg.stt, *self.cfg.stt_fallbacks] if s.configured]

    def _languages(self) -> list[str]:
        return list(dict.fromkeys(parse_languages(self.cfg.asr_language)))

    def _pinned_language(self) -> str | None:
        """Exactly one supported language keeps the existing fixed decode."""
        langs = self._languages()
        return langs[0] if len(langs) == 1 else None

    def _stt_form(self, st: Stage, *, stream: bool = False, pin: str | None = None) -> dict:
        # A single-code ``pin`` fixes the decode and needs BOTH fields:
        # ``language`` is the OpenAI/Whisper one (the local Whisper fallback
        # obeys it), but vLLM's Qwen3-ASR never reads it — it pins only from
        # ``to_language``, which prefills the reply with ``language English
        # <asr_text>``. Sending ``language`` alone left Qwen3-ASR auto-detecting.
        data: dict = {}
        if stream:
            data["stream"] = "true"
        if st.model:
            data["model"] = st.model
        if pin and "," in pin:
            # Qwen ignores candidate lists. A known session language is its
            # best hint; Whisper performs restricted detection on the list.
            if "qwen" in st.model.lower():
                pin = self._sticky_language
            else:
                data["language"] = pin
                return data
        if pin:
            data["language"] = pin
            data["to_language"] = pin
        return data

    async def transcribe(
        self,
        pcm: bytes,
        rate: int = audio.WIRE_RATE,
        *,
        whole_utterance: bool = False,
    ) -> str:
        pin = self._pinned_language()
        langs = self._languages()
        if not pin and len(langs) > 1:
            # Short fragments cannot reliably identify language. Longer audio,
            # including the whole-turn review, may change the session hint.
            if not whole_utterance and len(pcm) < 2 * rate * 2 and self._sticky_language in langs:
                pin = self._sticky_language
            else:
                return await self._transcribe_once(pcm, rate, ",".join(langs))
        if not pin:
            return await self._transcribe_once(pcm, rate, None)
        # A pinned decoder cannot answer "nothing was said": it turns a hum or
        # a click into a word of the pinned language ("Okay."), which the
        # non-speech guard cannot tell from speech. So the speech verdict comes
        # from an unbiased decode run alongside (it answers empty or CJK filler
        # on non-speech), and the pinned decode supplies the text.
        unbiased, pinned = await asyncio.gather(
            self._transcribe_once(pcm, rate, None),
            self._transcribe_once(pcm, rate, pin),
        )
        if not sanitize_transcript(unbiased, self.cfg.asr_language):
            return unbiased
        return pinned

    async def _transcribe_once(self, pcm: bytes, rate: int, pin: str | None) -> str:
        stages = self._stt_stages()
        if not stages:
            raise RuntimeError("STT not configured")
        wav = audio.wav_wrap(pcm, rate)
        errors: list[str] = []
        for st in stages:
            try:

                async def request(language: str | None) -> httpx.Response:
                    return await self._client.post(
                        st.url("audio/transcriptions"),
                        files={"file": ("utterance.wav", wav, "audio/wav")},
                        data=self._stt_form(st, pin=language),
                        headers=self._headers(st),
                    )

                verdict = None
                if pin and "," in pin and "qwen" in st.model.lower() and self._sticky_language:
                    # Qwen's sticky hint is a fixed decode too: preserve the
                    # unbiased non-speech verdict even on a long fallback.
                    r, verdict = await asyncio.gather(request(pin), request(None))
                else:
                    r = await request(pin)
                if verdict is not None:
                    verdict.raise_for_status()
                    unbiased = (verdict.json().get("text") or "").strip()
                    if not sanitize_transcript(unbiased, self.cfg.asr_language):
                        return unbiased
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{st.base_url}: transport {exc}")
                continue
            if r.status_code != 200:
                errors.append(f"{st.base_url}: HTTP {r.status_code} {r.text[:200]}")
                continue
            if errors:
                _LOG.info(
                    "STT succeeded after fallback; prior failures: %s",
                    " | ".join(errors),
                )
            result = r.json()
            text = (result.get("text") or "").strip()
            if (
                pin
                and "," in pin
                and result.get("language") in self._languages()
                and sanitize_transcript(text, self.cfg.asr_language)
            ):
                self._sticky_language = result["language"]
            return text
        _LOG.warning("STT failed on all endpoints: %s", " | ".join(errors))
        raise RuntimeError("all STT endpoints failed - " + " | ".join(errors))

    async def transcribe_stream(
        self,
        pcm: bytes,
        rate: int = audio.WIRE_RATE,
    ) -> AsyncIterator[str]:
        """Stream a transcription token-by-token (vLLM ``stream=true``).

        The audio is still posted whole — Qwen3-ASR has no incremental-audio
        state — so this is for re-decoding a growing buffer to surface *partial*
        captions while the user speaks; the authoritative final still comes from
        the blocking :meth:`transcribe`. Yields cleaned transcript pieces with
        the ``language <lang><asr_text>`` preamble stripped (see AsrStreamFilter).
        Tries PRIMARY then BACKUP.

        With one configured language the decode is pinned to it, and — as in
        :meth:`transcribe` — an unbiased decode decides whether the buffer holds
        speech at all, so the caption never shows a word the pin made up from a
        speech-onset fragment. The pinned text is then yielded whole.
        """
        if len(self._languages()) > 1:
            # Whisper's JSON response carries the restricted detection winner;
            # its endpoint need not implement SSE. Keep partials display-only.
            text = await self.transcribe(pcm, rate)
            if sanitize_transcript(text, self.cfg.asr_language):
                yield text
            return
        pin = self._pinned_language()
        if not pin:
            async for piece in self._transcribe_stream_once(pcm, rate, None):
                yield piece
            return

        async def pinned_text() -> str:
            return "".join([p async for p in self._transcribe_stream_once(pcm, rate, pin)])

        unbiased, pinned = await asyncio.gather(self._transcribe_once(pcm, rate, None), pinned_text())
        if sanitize_transcript(unbiased, self.cfg.asr_language) and pinned:
            yield pinned

    async def _transcribe_stream_once(
        self,
        pcm: bytes,
        rate: int,
        pin: str | None,
    ) -> AsyncIterator[str]:
        stages = self._stt_stages()
        if not stages:
            raise RuntimeError("STT not configured")
        wav = audio.wav_wrap(pcm, rate)
        errors: list[str] = []
        for st in stages:
            filt = AsrStreamFilter()
            try:
                async with self._client.stream(
                    "POST",
                    st.url("audio/transcriptions"),
                    files={"file": ("utterance.wav", wav, "audio/wav")},
                    data=self._stt_form(st, stream=True, pin=pin),
                    headers=self._headers(st),
                ) as r:
                    if r.status_code != 200:
                        body = (await r.aread())[:200]
                        errors.append(f"{st.base_url}: HTTP {r.status_code} {body!r}")
                        continue
                    async for line in r.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            delta = json.loads(payload)["choices"][0]["delta"]
                        except (ValueError, LookupError):
                            continue
                        piece = delta.get("content") or ""
                        if not piece:
                            continue
                        out = filt.feed(piece)
                        if out:
                            yield out
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{st.base_url}: transport {exc}")
                continue
            tail = filt.flush()
            if tail:
                yield tail
            if errors:
                _LOG.info(
                    "STT stream succeeded after fallback; prior failures: %s",
                    " | ".join(errors),
                )
            return
        _LOG.warning("STT stream failed on all endpoints: %s", " | ".join(errors))
        raise RuntimeError("all STT endpoints failed - " + " | ".join(errors))

    # -- LLM ------------------------------------------------------------

    async def chat_stream(self, messages: list[dict]) -> AsyncIterator[str]:
        ch = self.cfg.chat
        body: dict = {
            "model": ch.model,
            "messages": messages,
            "stream": True,
        }
        if self.cfg.temperature is not None:
            body["temperature"] = self.cfg.temperature
        if self.cfg.max_tokens > 0:
            body["max_tokens"] = self.cfg.max_tokens
        if self.cfg.disable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        # Defensive: even with thinking off, strip any <think> block that a
        # template change might let through (known qwen3.6 failure mode).
        in_think = False
        carry = ""
        async with self._client.stream(
            "POST",
            ch.url("chat/completions"),
            json=body,
            headers=self._headers(ch),
        ) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    delta = json.loads(payload)["choices"][0]["delta"]
                except (ValueError, LookupError):
                    continue
                piece = delta.get("content") or ""
                if not piece:
                    continue
                carry += piece
                out = ""
                while carry:
                    if in_think:
                        idx = carry.find(_THINK_CLOSE)
                        if idx < 0:
                            carry = carry[-len(_THINK_CLOSE) :]
                            break
                        carry = carry[idx + len(_THINK_CLOSE) :]
                        in_think = False
                    else:
                        idx = carry.find(_THINK_OPEN)
                        if idx < 0:
                            # Hold back a potential split "<thi" tag prefix.
                            safe = len(carry) - len(_THINK_OPEN) + 1
                            if carry.endswith(tuple(_THINK_OPEN[:i] for i in range(1, len(_THINK_OPEN)))):
                                out += carry[:safe]
                                carry = carry[safe:]
                            else:
                                out += carry
                                carry = ""
                            break
                        out += carry[:idx]
                        carry = carry[idx + len(_THINK_OPEN) :]
                        in_think = True
                if out:
                    yield out
        if carry and not in_think:
            yield carry

    # -- TTS ------------------------------------------------------------

    def _tts_stages(self) -> list[Stage]:
        return [s for s in [self.cfg.tts, *self.cfg.tts_fallbacks] if s.configured]

    async def tts(self, text: str, language: str | None = None) -> bytes:
        """Synthesize ``text`` → PCM16 mono at WIRE_RATE.

        ``language`` (e.g. "pt"/"en") is passed so a voice-cloning backend picks
        the right reference voice for the WHOLE reply instead of re-guessing per
        chunk — short chunks otherwise flip the cloned voice mid-sentence.
        """
        stages = self._tts_stages()
        if not stages:
            raise RuntimeError("TTS not configured")

        errors: list[str] = []
        for i, ts in enumerate(stages):
            # Realtime chunks must be spoken VERBATIM - the transcript already
            # streamed to the client. The local TTS proxy honours this header by
            # skipping its >160-char summarizer; other providers ignore it.
            headers = {**self._headers(ts), "X-TTS-No-Summary": "1"}
            # Primary (index 0, local TTS proxy) serves WAV; backup (index 1+,
            # OpenRouter) only accepts mp3 or pcm. Use mp3 for backup and decode
            # via ffmpeg pipe. Stage has no label field so we use position.
            use_mp3 = i > 0
            payload = {
                "model": ts.model or "tts",
                "voice": self.cfg.voice,
                "input": text,
                "response_format": "mp3" if use_mp3 else "wav",
            }
            if language:
                payload["language"] = language
            try:
                r = await self._client.post(
                    ts.url("audio/speech"),
                    json=payload,
                    headers=headers,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{ts.base_url}: transport {exc}")
                continue
            if r.status_code != 200:
                errors.append(f"{ts.base_url}: HTTP {r.status_code} {r.text[:200]}")
                continue
            if not r.content:
                errors.append(f"{ts.base_url}: empty response body")
                continue
            try:
                if use_mp3:
                    # Decode mp3 → WAV via ffmpeg pipe
                    proc = await asyncio.create_subprocess_exec(
                        "ffmpeg",
                        "-i",
                        "pipe:0",
                        "-f",
                        "wav",
                        "pipe:1",
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    wav_bytes, _ = await proc.communicate(r.content)
                    pcm, rate = audio.wav_parse(wav_bytes)
                else:
                    pcm, rate = audio.wav_parse(r.content)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{ts.base_url}: decode failed ({len(r.content)} bytes): {exc}")
                continue
            if errors:
                _LOG.info("TTS succeeded after fallback; prior failures: %s", " | ".join(errors))
            return audio.resample(pcm, rate, audio.WIRE_RATE)

        _LOG.warning("TTS failed on all endpoints: %s", " | ".join(errors))
        raise RuntimeError("all TTS endpoints failed - " + " | ".join(errors))
