"""Telegram voice/audio subsystem — STT + TTS (extracted from ``bot.py``).

Self-contained audio pipeline used by :class:`~sharedagents_telegram.bot.AcpTelegramBot`
for voice notes and spoken replies:

* format detection + ffmpeg resolution (``_stt_format_for``,
  ``_stt_needs_transcode``, ``_resolve_ffmpeg``, ``_mp3_has_audio_frame``),
* transcode/segment/transcribe (``_transcode_to_mp3``, ``_segment_audio``,
  ``_transcribe_segments``),
* the injected STT/TTS service factories (``_make_default_stt`` /
  ``_make_default_tts``) + the :class:`TranscriptionUnavailableError` taxonomy.

This module depends on NOTHING in ``bot.py`` (one-way import), so ``bot.py``
re-exports every public name from here for callers/tests that import them from
``sharedagents_telegram.bot``. The heavyweight codec deps (subprocess/tempfile/
pathlib/shutil/imageio_ffmpeg) stay LAZY-imported inside their functions.
"""

from __future__ import annotations

import asyncio
import json
import logging

import httpx

OPENROUTER_STT_URL = "https://openrouter.ai/api/v1/audio/transcriptions"
OPENROUTER_STT_MODEL = "openai/gpt-4o-transcribe"
# OpenRouter gpt-4o-transcribe accepts only `mp3` and `wav` for the
# `input_audio.format` field. OGG / Opus / m4a aren't supported and
# return HTTP 400 on the wire. We map known formats here; OGG (the
# default Telegram voice_note container) needs transcoding before
# upload — see `_transcode_to_mp3` below.
_STT_FORMAT_BY_MIME = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
}
_STT_FORMAT_BY_EXT = {
    ".mp3": "mp3",
    ".wav": "wav",
}
# Formats we know how to transcode to mp3 if `ffmpeg` is on PATH.
_STT_TRANSCODE_FROM_MIME = frozenset(
    {
        "audio/ogg",
        "audio/oga",
        "audio/opus",
        "audio/mp4",
        "audio/x-m4a",
        "audio/m4a",
        "audio/aac",
        "video/mp4",
        "video/quicktime",
        "video/webm",
    }
)
_STT_TRANSCODE_FROM_EXT = frozenset(
    {
        ".ogg",
        ".oga",
        ".opus",
        ".m4a",
        ".aac",
        ".mp4",
        ".mov",
        ".webm",
    }
)


def _stt_format_for(mime: str, filename: str) -> str | None:
    """Pick OpenRouter input_audio.format from mime first, then file
    extension. Returns ``None`` when the input format is one OpenRouter
    can't handle natively (e.g. OGG/Opus from Telegram voice notes) —
    callers should transcode first or surface a clear error.
    """
    if mime:
        f = _STT_FORMAT_BY_MIME.get(mime.lower().split(";")[0].strip())
        if f:
            return f
    if filename:
        from pathlib import Path as _P

        f = _STT_FORMAT_BY_EXT.get(_P(filename).suffix.lower())
        if f:
            return f
    return None


def _stt_needs_transcode(mime: str, filename: str) -> bool:
    """True if the input is a known format we'd need to transcode
    (via ffmpeg) before sending to OpenRouter."""
    if mime and mime.lower().split(";")[0].strip() in _STT_TRANSCODE_FROM_MIME:
        return True
    if filename:
        from pathlib import Path as _P

        if _P(filename).suffix.lower() in _STT_TRANSCODE_FROM_EXT:
            return True
    return False


def _resolve_ffmpeg() -> str | None:
    """Locate an ``ffmpeg`` binary.

    Order: system PATH (operator install / distro package) →
    ``imageio_ffmpeg.get_ffmpeg_exe()`` (static binary shipped with
    the ``imageio-ffmpeg`` wheel that ``src/requirements.txt`` pins).
    Returns the absolute path or ``None`` if neither is available.

    The ``imageio_ffmpeg`` fallback is what makes Telegram voice notes
    work out of the box on machines without a system ffmpeg —
    ``pip install`` is the only setup step the operator has to take.
    """
    import shutil as _shutil

    p = _shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg  # type: ignore[import-not-found]

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


def _mp3_has_audio_frame(data: bytes) -> bool:
    """True iff ``data`` carries at least one MPEG audio frame.

    A header-only MP3 stub (an ID3 / Xing header with no encoded audio) — which
    ffmpeg can emit while still exiting 0 after silently truncating a moov-at-end
    container — has no frame-sync marker and returns ``False`` here, so the
    truncated output can be refused instead of forwarded to STT as valid audio.
    """
    n = len(data)
    off = 0
    # Skip a leading ID3v2 tag so its (often zero-padded) body is not scanned
    # for — or does not mask — the first real audio frame.
    if n >= 10 and data[:3] == b"ID3":
        off = 10 + ((data[6] & 0x7F) << 21 | (data[7] & 0x7F) << 14 | (data[8] & 0x7F) << 7 | (data[9] & 0x7F))
    # Scan for an MPEG frame sync: 0xFF followed by a byte whose top three bits
    # are set (11 sync bits + a non-reserved MPEG version field).
    i = data.find(b"\xff", off)
    while i != -1 and i < n - 1:
        if (data[i + 1] & 0xE0) == 0xE0:
            return True
        i = data.find(b"\xff", i + 1)
    return False


def _transcode_to_mp3(
    audio_bytes: bytes,
    *,
    sample_rate: int | None = None,
    channels: int | None = None,
    timeout: int = 60,
) -> bytes | None:
    """Best-effort: run audio through ``ffmpeg`` to produce MP3.
    Returns the transcoded bytes or ``None`` if ffmpeg is missing or
    fails. Used to normalize incoming audio for STT endpoints that
    can't decode the original container (OGG/Opus, m4a/aac, mp4).

    Optional ``sample_rate`` / ``channels`` force resampling — needed
    for Qwen3-ASR via vLLM which only accepts 16 kHz mono. Default
    None preserves the source rate (what OpenRouter wants).

    The input is written to a temp file and handed to ffmpeg by path
    rather than piped on stdin. MP4 / M4A / MOV containers from phones
    and apps put the ``moov`` index atom at the END of the file (i.e.
    NOT ``+faststart``); decoding those needs ffmpeg to seek back to
    the audio ``mdat`` after it finally reads the index — impossible
    over a non-seekable stdin pipe. On a pipe ffmpeg bails ("partial
    file" / "Error during demuxing: Invalid data found") and emits a
    header-only MP3 (~140-220 bytes) *while still exiting 0*, which the
    downstream STT endpoint rejects as "Invalid or unsupported audio
    file". A seekable temp file lets ffmpeg decode moov-at-end inputs;
    OGG/Opus voice notes (already streamable) keep working unchanged.
    """
    import subprocess as _sub
    import tempfile as _tempfile

    ffmpeg = _resolve_ffmpeg()
    if not ffmpeg:
        return None
    try:
        with _tempfile.NamedTemporaryFile(suffix=".audio") as src:
            src.write(audio_bytes)
            src.flush()
            cmd = [ffmpeg, "-loglevel", "error", "-i", src.name]
            if sample_rate is not None:
                cmd += ["-ar", str(sample_rate)]
            if channels is not None:
                cmd += ["-ac", str(channels)]
            cmd += [
                "-codec:a",
                "libmp3lame",
                "-q:a",
                "4",
                "-f",
                "mp3",
                "pipe:1",
            ]
            try:
                proc = _sub.run(cmd, capture_output=True, timeout=timeout)
            except (_sub.TimeoutExpired, FileNotFoundError, OSError):
                return None
    except OSError:
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    # Ffmpeg can bail on a moov-at-end container and still
    # exit 0 while emitting a header-only stub (~140-220 bytes) with no audio
    # frames. Such a silently-truncated exit-0 output must be treated as a
    # transcode failure, never forwarded to STT as valid audio.
    if not _mp3_has_audio_frame(proc.stdout):
        return None
    return proc.stdout


def _segment_audio(audio_bytes: bytes, *, seconds: int = 300) -> list[bytes]:
    """Split an MP3 into <=``seconds`` chunks via ffmpeg's segment muxer.
    Returns chunk bytes in order; a short clip yields a single chunk.
    Empty list if ffmpeg is unavailable or fails.
    """
    import subprocess as _sub
    import tempfile as _tf
    from pathlib import Path as _P

    ffmpeg = _resolve_ffmpeg()
    if not ffmpeg or not audio_bytes:
        return []
    with _tf.TemporaryDirectory() as d:
        src = _P(d) / "in.mp3"
        src.write_bytes(audio_bytes)
        pat = str(_P(d) / "seg_%04d.mp3")
        cmd = [
            ffmpeg,
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-f",
            "segment",
            "-segment_time",
            str(seconds),
            "-c",
            "copy",
            pat,
        ]
        try:
            proc = _sub.run(cmd, capture_output=True, timeout=600)
        except (_sub.TimeoutExpired, OSError):
            return []
        if proc.returncode != 0:
            return []
        return [p.read_bytes() for p in sorted(_P(d).glob("seg_*.mp3"))]


async def _transcribe_segments(segments, post, on_progress=None) -> str:
    """POST each segment via async ``post(bytes)->str`` sequentially and
    join the non-empty transcripts with newlines. ``on_progress(k, n)``
    is awaited (if given) before each segment. One retry per segment.

    If EVERY segment fails (no real transcript at all) this returns ``""`` so
    the caller surfaces a clean transcription failure ("empty transcription").
    Returning the ``[segment N unavailable]`` placeholders in that case would
    forward sentinel text to the agent as if it were the user's words — the
    user then "receives garbled text instead of intended voice content". A
    partial failure (some segments OK) keeps the inline placeholder so the
    real transcript is still delivered with a visible gap marker.
    """
    parts: list[str] = []
    n = len(segments)
    any_ok = False
    for i, seg in enumerate(segments, 1):
        if on_progress is not None:
            await on_progress(i, n)
        try:
            txt = await post(seg)
        except Exception:  # noqa: BLE001
            txt = ""
        if not txt:
            try:
                txt = await post(seg)
            except Exception:  # noqa: BLE001
                txt = ""
        if txt:
            any_ok = True
            parts.append(txt)
        else:
            parts.append(f"[segment {i} unavailable]")
    if not any_ok:
        # Total transcription failure — do NOT hand placeholder sentinels to
        # the caller (they would reach the agent as the user's prompt).
        return ""
    return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# TTS — voice-note replies for chats where the user spoke
# ---------------------------------------------------------------------------
#
# Mirrors the STT factory shape: a unified primary/backup service block returns
# an async callable. Used by ``_stream_turn_for_chat`` AFTER the assembled text
# reply is finalized, when the original user message was voice — so a spoken
# question gets a spoken answer alongside the text bubble.
#
# Model id requires the **dated** slug — ``openai/gpt-4o-mini-tts``
# without the date returns HTTP 400 from OpenRouter. The dated slug pins
# the upstream snapshot so behaviour stays stable across upstream
# rolling-version bumps.

OPENROUTER_TTS_DEFAULT_URL = "https://openrouter.ai/api/v1/audio/speech"
OPENROUTER_TTS_DEFAULT_MODEL = "openai/gpt-4o-mini-tts-2025-12-15"
OPENROUTER_TTS_DEFAULT_VOICE = "alloy"
# How much text to bother with — TTS for a 5-page wall is wasteful, costly,
# and Telegram caps voice-note duration anyway. Hard cap at the OpenAI
# spec input limit (4096 chars).
TTS_MAX_INPUT_CHARS = 4000


def _tts_speech_url(ep) -> str:
    base = ep.base_url.rstrip("/")
    if base.endswith("/audio/speech"):
        return base
    return ep.url("audio/speech")


def _make_default_tts(svc, log: logging.Logger):
    """Return an async ``tts(text, *, voice, model) -> Optional[bytes]``
    callable that synthesises ``text`` via the unified ``tts`` service
    config. Returns the audio bytes on success or ``None`` on any failure
    (transport error, non-200, empty body). Failure is non-fatal — the text
    bubble already landed; the voice reply is bonus.
    """

    async def _tts(
        text: str,
        *,
        voice: str = OPENROUTER_TTS_DEFAULT_VOICE,
        model: str = "",
    ) -> bytes | None:
        if not text or not text.strip():
            return None
        errors: list[str] = []
        for ep in svc.candidates():
            body = json.dumps(
                {
                    "model": model or ep.model or OPENROUTER_TTS_DEFAULT_MODEL,
                    "voice": voice,
                    "input": text[:TTS_MAX_INPUT_CHARS],
                    "response_format": "wav",
                }
            )
            headers = {
                "Content-Type": "application/json",
                "X-Title": "sharedagents-telegram",
            }
            if ep.api_key:
                headers["Authorization"] = f"Bearer {ep.api_key}"
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(120.0, connect=10.0),
                ) as client:
                    r = await client.post(
                        _tts_speech_url(ep),
                        content=body,
                        headers=headers,
                    )
            except httpx.HTTPError as exc:
                errors.append(f"{ep.label}: transport {exc}")
                continue
            if r.status_code != 200:
                errors.append(f"{ep.label}: HTTP {r.status_code} {r.text[:200]}")
                continue
            if not r.content:
                errors.append(f"{ep.label}: empty response body")
                continue
            return r.content
        if errors:
            log.warning("TTS failed on all endpoints: %s", " | ".join(errors))
        else:
            log.warning("TTS skipped: no configured endpoints")
            return None
        return None

    return _tts


# ---------------------------------------------------------------------------
# STT failure taxonomy
# ---------------------------------------------------------------------------
# Two transcription-failure classes must be DISTINGUISHABLE at the operator UI:
#   (a) transcription cannot run at all -- the audio could not be decoded,
#       typically because no ffmpeg is resolvable (or the container is
#       undecodable). A server-side dependency problem the operator can fix, so
#       it is signalled as a typed error and rendered with an actionable
#       message that names the fixable dependency.
#   (b) a genuinely silent / empty recording -- decode succeeded but no speech
#       was recovered. Nothing for the operator to fix.
# Before this taxonomy both collapsed to "empty transcription", which is
# user-hostile (SUPERSEDES). Both strings' exact wording is
# owner-tunable; the REQUIREMENT is only that the two classes render DISTINCTLY
# and that the decode-unavailable message names the dependency.
STT_DECODE_UNAVAILABLE_MSG = (
    "⚠️ couldn't transcribe — the server can't decode this audio "
    "(ffmpeg / audio decoder unavailable). Install ffmpeg or the imageio-ffmpeg "
    "package on the bot host, then resend the voice note."
)
STT_EMPTY_TRANSCRIPT_MSG = "⚠️ empty transcription — no speech detected in the recording."


class TranscriptionUnavailableError(RuntimeError):
    """Transcription could not run at all (audio undecodable / no ffmpeg).

    Distinct from a genuinely silent recording (which yields an empty
    transcript string), so the operator UI can render a distinct, actionable
    message rather than collapsing both causes to one string.
    """


def _make_default_stt(svc, log: logging.Logger):
    """Return an async ``stt(audio_bytes, *, mime, filename) -> transcript``
    callable that POSTs to an OpenAI-compatible /v1/audio/transcriptions
    endpoint (standard Whisper API, multipart/form-data), trying the PRIMARY
    then BACKUP endpoint from the unified sharedagents ``stt`` config (``svc`` is a
    ``sharedagents_core.services.ServiceConfig``).

    Audio is resampled to 16k-mono mp3 first (Qwen3-ASR rejects m4a containers;
    OpenRouter accepts mp3) so a single body works for either endpoint.
    """

    async def _stt(audio_bytes: bytes, *, mime: str = "", filename: str = "", on_progress=None) -> str:
        if not audio_bytes:
            return ""
        # Resolve the transcoder through the bot facade at call time so a test
        # patching ``sharedagents_telegram.bot._transcode_to_mp3`` still reaches this
        # relocated STT path (pre-split, both lived in bot.py).
        from . import bot as _bot

        transcoded = await asyncio.to_thread(
            _bot._transcode_to_mp3,
            audio_bytes,
            sample_rate=16000,
            channels=1,
            timeout=600,
        )
        if not transcoded:
            # A failed decode (no resolvable ffmpeg, or an
            # undecodable container) is a DISTINCT failure class from genuine
            # silence. Surface it as a typed error so the operator UI renders a
            # distinct, actionable message instead of collapsing both to
            # "empty transcription".
            log.warning("STT unavailable: audio could not be decoded (ffmpeg unavailable or input undecodable).")
            raise TranscriptionUnavailableError("audio could not be decoded for transcription")
        segments = await asyncio.to_thread(_segment_audio, transcoded, seconds=300)
        if not segments:
            segments = [transcoded]

        async def _post_one(mp3_bytes: bytes) -> str:
            # Primary → backup failover for this segment.
            errors: list[str] = []
            for ep in svc.candidates():
                url = ep.url("audio/transcriptions")
                files = {"file": ("audio.mp3", mp3_bytes, "audio/mpeg")}
                data = {"model": ep.model} if ep.model else {}
                headers = {"X-Title": "sharedagents-telegram"}
                if ep.api_key:
                    headers["Authorization"] = f"Bearer {ep.api_key}"
                log.info("STT → %s model=%s bytes=%d", url, ep.model, len(mp3_bytes))
                try:
                    async with httpx.AsyncClient(
                        timeout=httpx.Timeout(120.0, connect=10.0),
                    ) as client:
                        r = await client.post(url, files=files, data=data, headers=headers)
                except httpx.HTTPError as exc:
                    errors.append(f"{ep.label}: transport {exc}")
                    continue
                if r.status_code != 200:
                    errors.append(f"{ep.label}: HTTP {r.status_code} {r.text[:200]}")
                    continue
                try:
                    return (r.json().get("text") or "").strip()
                except json.JSONDecodeError:
                    errors.append(f"{ep.label}: non-json {r.text[:200]}")
                    continue
            if errors:
                log.warning("STT failed on all endpoints: %s", " | ".join(errors))
            return ""

        return await _transcribe_segments(segments, _post_one, on_progress)

    return _stt
