#!/usr/bin/env python3
"""YouTube transcription MCP server — built into sharedagents.

Spawned on demand by sharedagents (stdio transport, JSON-RPC 2.0). Exposes
tools for transcribing YouTube videos and configuring credentials.

Transcription strategy (in order):
  1. Try YouTube subtitles via ``yt-dlp`` (manual or auto-generated en).
  2. Fall back to audio download + chunked STT via a self-hosted,
     OpenAI-compatible STT service. Its endpoint/model come from the UNIFIED
     sharedagents config (``~/.sharedagents/config.json`` → ``"stt"`` block) — never a host
     hardcoded in source, never scattered env vars. OpenRouter was removed for
     now and may return as a backup.

Long videos are split into 5-minute chunks via ffmpeg and transcribed
independently with retry, so a single chunk failure does not abort the
whole transcript. Completed transcripts are persisted under
``~/.cache/sharedagents-youtube-mcp/transcripts/<video_id>.txt``.

Minimal deps: stdlib + the ``yt-dlp`` binary, plus ``ffmpeg`` for audio
chunking — resolved from PATH or, failing that, the static binary in the
``imageio-ffmpeg`` wheel (the same fallback the Telegram voice path uses).
Without ffmpeg, long videos can only be transcribed in a single STT shot,
which the STT endpoint silently truncates mid-transcript.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "youtube-mcp-server"
SERVER_VERSION = "0.2.0"

# STT — a self-hosted OpenAI-compatible STT service (multipart
# /v1/audio/transcriptions). Its PRIMARY + BACKUP endpoints live in the UNIFIED
# sharedagents config (~/.sharedagents/config.json → "stt" block), resolved via
# sharedagents_core.services — never hardcode a local host/IP in source, and don't
# scatter across env vars. This server is a standalone spawn (no agent context),
# so it reads the GLOBAL stt block; per-agent overrides apply to in-process callers.
from sharedagents_core.services import resolve_service, with_failover  # noqa: E402

YT_DLP_TIMEOUT_S = 900
STT_TIMEOUT_S = 1800
CHUNK_SECONDS = 300  # 5-minute audio chunks
STT_MAX_RETRIES = 2  # additional attempts after the first try (so 3 total)

CACHE_DIR = Path(os.path.expanduser("~/.cache/sharedagents-youtube-mcp"))
TRANSCRIPT_DIR = CACHE_DIR / "transcripts"

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_URL_VID_RE = re.compile(r"(?:v=|youtu\.be/|/embed/|/shorts/|/live/)([A-Za-z0-9_-]{11})")

# ------------------------------------------------------------------- state
_stt_api_key: str = ""  # optional per-session override; real key comes from config
_cookie_file: Path | None = None  # set by ``set_cookies`` tool


# ----------------------------------------------------------------------- url
def _normalise_target(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        raise ValueError("empty url/id")
    if _VIDEO_ID_RE.match(s):
        return f"https://www.youtube.com/watch?v={s}"
    return s


def _extract_video_id(target: str) -> str:
    m = _URL_VID_RE.search(target)
    if m:
        return m.group(1)
    if _VIDEO_ID_RE.match(target):
        return target
    return "unknown"


# -------------------------------------------------------------- yt-dlp call
_yt_dlp_base_cache: list[str] | None = None


def _resolve_yt_dlp_base() -> list[str]:
    """Resolve the yt-dlp invocation prefix, robust to a minimal spawn PATH.

    When an MCP client (claude / sharedagents) spawns this server, PATH is the bare
    system PATH — it excludes ``~/.local/bin`` (pipx) and the server's own
    ``.venv/bin``. So a plain ``shutil.which("yt-dlp")`` finds nothing even
    though yt-dlp is installed right next to the interpreter running us.

    Resolution order (mirrors the ffmpeg fallback strategy):
      1. ``yt-dlp`` on PATH (operator / distro install).
      2. ``yt-dlp`` binary next to the running interpreter (the venv bin).
      3. ``~/.local/bin/yt-dlp`` (pipx install).
      4. ``[sys.executable, "-m", "yt_dlp"]`` if the module is importable.
    """
    global _yt_dlp_base_cache
    if _yt_dlp_base_cache is not None:
        return _yt_dlp_base_cache

    on_path = shutil.which("yt-dlp")
    if on_path:
        _yt_dlp_base_cache = [on_path]
        return _yt_dlp_base_cache

    candidates = [
        Path(sys.executable).resolve().parent / "yt-dlp",
        Path.home() / ".local" / "bin" / "yt-dlp",
    ]
    for cand in candidates:
        if cand.exists() and os.access(cand, os.X_OK):
            _yt_dlp_base_cache = [str(cand)]
            return _yt_dlp_base_cache

    import importlib.util

    if importlib.util.find_spec("yt_dlp") is not None:
        _yt_dlp_base_cache = [sys.executable, "-m", "yt_dlp"]
        return _yt_dlp_base_cache

    raise RuntimeError(
        "yt-dlp not found on PATH, next to the interpreter, in ~/.local/bin, "
        "or as an importable module. Install with: pipx install yt-dlp"
    )


def _yt_dlp_candidates(extra_args: list[str], target: str) -> list[list[str]]:
    """Build cookies-file → cookies-from-browser → bare candidate cmds."""
    base = _resolve_yt_dlp_base()
    candidates: list[list[str]] = []
    if _cookie_file and _cookie_file.exists():
        candidates.append(base + ["--cookies", str(_cookie_file)] + extra_args + [target])
    cookie_browser = os.environ.get(
        "YOUTUBE_MCP_COOKIES_FROM_BROWSER",
        "chrome",
    ).strip()
    if cookie_browser:
        candidates.append(base + ["--cookies-from-browser", cookie_browser] + extra_args + [target])
    candidates.append(base + extra_args + [target])
    return candidates


def _ensure_yt_dlp() -> None:
    _resolve_yt_dlp_base()


def _clear_dir(d: Path) -> None:
    for f in list(d.iterdir()):
        try:
            f.unlink()
        except OSError:
            pass


def _pick_output(dest_dir: Path) -> Path:
    files = sorted(dest_dir.iterdir())
    if not files:
        raise RuntimeError("yt-dlp produced no output file")
    return files[0]


def _download_audio(target: str, dest_dir: Path) -> Path:
    """Run ``yt-dlp`` to fetch the bestaudio track into ``dest_dir``."""
    _ensure_yt_dlp()
    out_template = str(dest_dir / "audio.%(ext)s")
    extra = [
        "-f",
        "bestaudio",
        "--no-playlist",
        "--no-warnings",
        "--quiet",
        "-o",
        out_template,
    ]
    candidates = _yt_dlp_candidates(extra, target)

    last_err = ""
    for attempt in range(1, 4):  # up to 3 rounds with backoff
        for cmd in candidates:
            _clear_dir(dest_dir)
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=YT_DLP_TIMEOUT_S,
                )
            except subprocess.TimeoutExpired as e:
                last_err = f"timeout after {YT_DLP_TIMEOUT_S}s: {e}"
                continue
            if proc.returncode == 0:
                return _pick_output(dest_dir)
            last_err = (proc.stderr or proc.stdout or "").strip()[:600]
        if attempt < 3:
            time.sleep(2 * attempt)  # 2s, 4s

    raise RuntimeError(f"yt-dlp failed after 3 attempts: {last_err}")


# --------------------------------------------------------- subtitle handling
def _vtt_to_text(vtt: str) -> str:
    """Extract plain transcript text from a WebVTT string."""
    out: list[str] = []
    for raw in vtt.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s == "WEBVTT" or s.startswith(("NOTE", "Kind:", "Language:", "STYLE", "REGION")):
            continue
        if "-->" in s:
            continue
        if s.isdigit():
            continue
        s = re.sub(r"<[^>]+>", "", s)  # strip karaoke / styling tags
        s = s.replace("&nbsp;", " ")
        s = s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        s = s.strip()
        if not s:
            continue
        if out and out[-1] == s:  # dedupe roll-up captions
            continue
        out.append(s)
    return "\n".join(out).strip()


def _srt_to_text(srt: str) -> str:
    out: list[str] = []
    for raw in srt.splitlines():
        s = raw.strip()
        if not s or "-->" in s or s.isdigit():
            continue
        s = re.sub(r"<[^>]+>", "", s).strip()
        if not s:
            continue
        if out and out[-1] == s:
            continue
        out.append(s)
    return "\n".join(out).strip()


_TERMINAL_PUNCT = (".", "!", "?", "…")


def _check_subtitle_quality(
    subs_text: str,
    video_duration_seconds: float | None,
) -> tuple[bool, str]:
    """Heuristic completeness check for subtitle text.

    Returns ``(ok, reason)``. ``reason`` is empty when ``ok`` is True.
    The caller should fall through to audio STT when ``ok`` is False.
    """
    if not subs_text or not subs_text.strip():
        return False, "subtitles empty"

    lines = [ln.strip() for ln in subs_text.splitlines() if ln.strip()]
    if not lines:
        return False, "subtitles whitespace-only"

    word_counts = [len(ln.split()) for ln in lines]
    median_words = statistics.median(word_counts) if word_counts else 0
    total_words = sum(word_counts)

    # 1) Mid-sentence cutoff: last line lacks terminal punctuation AND
    #    doesn't end with a capital-letter word (proper noun escape).
    last = lines[-1]
    last_tokens = last.split()
    last_token = last_tokens[-1] if last_tokens else ""
    ends_terminal = last.endswith(_TERMINAL_PUNCT)
    ends_proper_noun = bool(last_token) and last_token[0].isupper() and last_token.isalpha()
    if not ends_terminal and not ends_proper_noun:
        return False, (f"last line lacks terminal punctuation and isn't a proper noun (last={last!r})")

    # Suspiciously short final line vs the rest — rolling captions cut
    # mid-utterance often leave a 1-3 word fragment at the tail.
    last_word_count = len(last_tokens)
    if last_word_count < 5 and median_words >= 8:
        return False, (f"last line is suspiciously short ({last_word_count} words) vs median {median_words}")

    # 2) Duration vs word count. ~150 wpm spoken English → 2.5 wps; use
    #    2 wps as a conservative expectation. Flag at <2% of that floor.
    if video_duration_seconds and video_duration_seconds > 30:
        expected_words = video_duration_seconds * 2
        threshold = max(1, int(expected_words * 0.02))
        if total_words < threshold:
            return False, (
                f"transcript word count {total_words} < 2% of expected "
                f"({int(expected_words)} words for "
                f"{video_duration_seconds:.0f}s)"
            )

    # 3) Frozen-caption / repeat detection: >30% of consecutive lines
    #    identical to their predecessor (beyond normal dedup, which
    #    already drops adjacent duplicates inside _vtt_to_text — so any
    #    that survive are non-adjacent in the source). Skip on short
    #    transcripts where one repeat trips the threshold falsely.
    if len(lines) >= 10:
        repeats = sum(1 for i in range(1, len(lines)) if lines[i] == lines[i - 1])
        repeat_ratio = repeats / (len(lines) - 1)
        if repeat_ratio > 0.30:
            return False, (f"{repeat_ratio:.0%} of consecutive lines are duplicates (rolling-caption freeze?)")

    # 4) NON-ADJACENT frozen-caption / low-diversity detection. A rolling
    #    freeze or an alternating placeholder feed (A, B, A, B, …) leaves very
    #    few DISTINCT lines while keeping NO two ADJACENT lines identical — so
    #    heuristic 3's consecutive-only count misses it entirely. Flag when a
    #    large share of lines are non-unique duplicates of some other line,
    #    regardless of adjacency (an excessive non-adjacent-
    #    duplicate / frozen-caption ratio).
    if len(lines) >= 10:
        from collections import Counter

        counts = Counter(lines)
        duplicated = sum(count for count in counts.values() if count > 1)
        dup_ratio = duplicated / len(lines)
        if dup_ratio > 0.50:
            return False, (
                f"{dup_ratio:.0%} of lines are non-unique duplicates "
                f"({len(counts)} distinct of {len(lines)} — "
                f"frozen/looping captions?)"
            )

    return True, ""


def _try_subtitles(
    target: str,
    dest_dir: Path,
) -> tuple[str | None, dict[str, Any]]:
    """Attempt to fetch English subtitles. Also returns id/title/url info.

    Uses two separate yt-dlp invocations: combining ``-O`` (print) with
    ``-o`` (output template) for subtitle writes is unreliable — yt-dlp
    can exit 0 without writing any subtitle file. So we fetch metadata
    via ``--dump-json`` first, then download subs in a second call.

    Returns ``(transcript_or_None, info)``. The info dict is best-effort
    and may be empty if the metadata call failed.
    """
    _ensure_yt_dlp()

    # 1) Metadata call: --dump-json prints one JSON line per video.
    info: dict[str, Any] = {}
    info_extra = [
        "--skip-download",
        "--no-warnings",
        "--quiet",
        "--dump-json",
    ]
    for cmd in _yt_dlp_candidates(info_extra, target):
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=YT_DLP_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            continue
        if proc.returncode != 0:
            continue
        first = (proc.stdout or "").strip().splitlines()
        if not first:
            continue
        try:
            data = json.loads(first[0])
        except json.JSONDecodeError:
            continue
        for key in ("id", "title", "webpage_url"):
            val = data.get(key)
            if isinstance(val, str) and val:
                info[key] = val
        duration = data.get("duration")
        if isinstance(duration, (int, float)) and duration > 0:
            info["duration"] = float(duration)
        break

    # 2) Subtitle download call: no -O — just --write-subs / --write-auto-subs.
    out_template = str(dest_dir / "subs.%(ext)s")
    sub_extra = [
        "--skip-download",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs",
        "en,en.*",
        "--sub-format",
        "vtt/best",
        "--no-warnings",
        "--quiet",
        "-o",
        out_template,
    ]
    sub_success = False
    for cmd in _yt_dlp_candidates(sub_extra, target):
        _clear_dir(dest_dir)
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=YT_DLP_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            continue
        if proc.returncode == 0:
            sub_success = True
            break

    if not sub_success:
        return None, info

    sub_files = sorted(p for p in dest_dir.iterdir() if p.is_file() and p.suffix.lower() in (".vtt", ".srt"))
    if not sub_files:
        return None, info

    text = ""
    for sf in sub_files:
        try:
            raw = sf.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        text = _vtt_to_text(raw) if sf.suffix.lower() == ".vtt" else _srt_to_text(raw)
        if text.strip():
            break

    if not text.strip():
        return None, info
    return text, info


# ----------------------------------------------------------- audio chunking
def _resolve_ffmpeg() -> str | None:
    """Locate an ``ffmpeg`` binary.

    Order: system PATH (operator install / distro package) →
    ``imageio_ffmpeg.get_ffmpeg_exe()`` (static binary shipped with the
    ``imageio-ffmpeg`` wheel pinned in ``src/requirements.txt``). Returns
    the absolute path, or ``None`` if neither is available.

    Mirrors ``sharedagents_telegram.bot._resolve_ffmpeg``. Without this
    fallback a host with no system ffmpeg silently skips audio chunking
    and sends the whole video to the STT endpoint in one shot — which
    truncates long videos mid-transcript with no error.
    """
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg  # type: ignore[import-not-found]

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


def _chunk_audio(audio_path: Path, chunk_dir: Path) -> list[Path]:
    """Split ``audio_path`` into ~5-minute chunks via ffmpeg.

    Returns the list of chunk files in order. Falls back to
    ``[audio_path]`` (single-shot) if ffmpeg is missing or fails — and
    warns loudly on stderr when ffmpeg is missing entirely, since a
    single-shot STT call truncates long videos mid-transcript.
    """
    ffmpeg = _resolve_ffmpeg()
    if ffmpeg is None:
        sys.stderr.write(
            "[youtube-mcp] WARNING: no ffmpeg found (PATH or imageio-ffmpeg) "
            "- cannot split audio; long videos will transcribe only "
            "partially. Install ffmpeg or `pip install imageio-ffmpeg`.\n"
        )
        sys.stderr.flush()
        return [audio_path]

    ext = audio_path.suffix.lower() or ".m4a"

    # Attempt 1: stream copy (instant, no re-encode). Works for most
    # bestaudio outputs that yt-dlp gives us.
    out_pattern = str(chunk_dir / f"chunk_%03d{ext}")
    cmd_copy = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(audio_path),
        "-f",
        "segment",
        "-segment_time",
        str(CHUNK_SECONDS),
        "-c",
        "copy",
        "-reset_timestamps",
        "1",
        out_pattern,
    ]
    try:
        proc = subprocess.run(
            cmd_copy,
            capture_output=True,
            text=True,
            timeout=YT_DLP_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        proc = None  # type: ignore[assignment]

    chunks = sorted(chunk_dir.glob(f"chunk_*{ext}"))
    if proc is not None and proc.returncode == 0 and chunks:
        return chunks

    # Attempt 2: re-encode to mp3 (handles codecs the segment muxer
    # cannot stream-copy).
    for f in chunks:
        try:
            f.unlink()
        except OSError:
            pass
    out_pattern_mp3 = str(chunk_dir / "chunk_%03d.mp3")
    cmd_enc = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(audio_path),
        "-f",
        "segment",
        "-segment_time",
        str(CHUNK_SECONDS),
        "-c:a",
        "libmp3lame",
        "-b:a",
        "64k",
        "-reset_timestamps",
        "1",
        out_pattern_mp3,
    ]
    try:
        proc = subprocess.run(
            cmd_enc,
            capture_output=True,
            text=True,
            timeout=YT_DLP_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return [audio_path]

    if proc.returncode != 0:
        return [audio_path]

    chunks = sorted(chunk_dir.glob("chunk_*.mp3"))
    return chunks if chunks else [audio_path]


# ----------------------------------------------- local STT (faster-whisper)
def _multipart_body(audio_path: Path, audio_bytes: bytes, model: str) -> tuple[bytes, str]:
    """Build a multipart/form-data body for an OpenAI-compatible STT server
    (expects a ``file`` upload + ``model``; language omitted — Qwen3-ASR rejects
    'auto', so the server auto-detects, mirroring the proven telegram path)."""
    boundary = "----sharedagentsSTT" + os.urandom(12).hex()
    crlf = b"\r\n"
    parts: list[bytes] = []
    parts.append(b"--" + boundary.encode() + crlf)
    parts.append(
        f'Content-Disposition: form-data; name="file"; filename="{audio_path.name}"'.encode()
        + crlf
        + b"Content-Type: application/octet-stream"
        + crlf
        + crlf
    )
    parts.append(audio_bytes + crlf)
    if model:
        parts.append(b"--" + boundary.encode() + crlf)
        parts.append(b'Content-Disposition: form-data; name="model"' + crlf + crlf)
        parts.append(model.encode() + crlf)
    parts.append(b"--" + boundary.encode() + b"--" + crlf)
    return b"".join(parts), boundary


# Qwen3-ASR (vLLM) accepts these directly; m4a/webm/etc. are transcoded to mp3.
_STT_DIRECT_EXT = {".mp3", ".wav", ".ogg", ".oga", ".opus"}


def _to_qwen_audio(audio_path: Path) -> tuple[Path, bytes]:
    """Return (path, bytes) in a Qwen3-ASR-accepted format. yt-dlp bestaudio is
    usually m4a/webm which Qwen3-ASR rejects, so transcode to 16k-mono mp3."""
    if audio_path.suffix.lower() in _STT_DIRECT_EXT:
        return audio_path, audio_path.read_bytes()
    ffmpeg = _resolve_ffmpeg()
    if ffmpeg is None:
        return audio_path, audio_path.read_bytes()  # best effort
    out = audio_path.with_suffix(".stt.mp3")
    proc = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(audio_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "libmp3lame",
            "-q:a",
            "5",
            str(out),
        ],
        capture_output=True,
        timeout=STT_TIMEOUT_S,
    )
    if proc.returncode == 0 and out.exists() and out.stat().st_size > 0:
        return out, out.read_bytes()
    return audio_path, audio_path.read_bytes()


def _post_stt(endpoint, send_path: Path, audio_bytes: bytes) -> str:
    """POST a multipart transcription request to ONE STT endpoint; return the
    transcript or raise (so :func:`with_failover` can try the backup)."""
    url = endpoint.url("audio/transcriptions")
    body, boundary = _multipart_body(send_path, audio_bytes, endpoint.model)
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    api_key = (_stt_api_key or endpoint.api_key).strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=body, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=STT_TIMEOUT_S) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:600] if e.fp else ""
        raise RuntimeError(f"HTTP {e.code} from {url}: {detail}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"unreachable ({url}): {e.reason}") from None
    text = (payload.get("text") or "").strip()
    if not text:
        raise RuntimeError(f"empty transcript from {url}: {payload!r}")
    return text


def _transcribe_audio(audio_path: Path) -> str:
    """Transcribe one audio file via the configured self-hosted STT service
    (OpenAI-compatible multipart upload). Tries the PRIMARY then BACKUP endpoint
    from the unified sharedagents config ``stt`` block (``sharedagents_core.services``)."""
    svc = resolve_service("stt")
    send_path, audio_bytes = _to_qwen_audio(audio_path)
    if not audio_bytes:
        raise RuntimeError(f"audio file is empty: {audio_path.name}")
    return with_failover(svc, lambda ep: _post_stt(ep, send_path, audio_bytes))


def _transcribe_one_with_retry(audio_path: Path) -> str:
    last_err = ""
    for attempt in range(STT_MAX_RETRIES + 1):
        try:
            return _transcribe_audio(audio_path)
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            if attempt < STT_MAX_RETRIES:
                time.sleep(2 ** (attempt + 1))  # 2s, 4s
    raise RuntimeError(f"transcription failed after {STT_MAX_RETRIES + 1} attempts: {last_err}")


def _transcribe_chunks(chunks: list[Path]) -> str:
    """Transcribe each chunk independently; a single chunk's failure is
    recorded inline and excluded rather than aborting the whole job.
    """
    if not chunks:
        raise RuntimeError("no audio chunks to transcribe")
    total = len(chunks)
    if total == 1:
        text = _transcribe_one_with_retry(chunks[0])
        _notify_progress(
            "transcribed chunk 1 of 1",
            chunk=1,
            total=1,
            status="ok",
        )
        return text

    parts: list[str] = []
    successes = 0
    for i, chunk in enumerate(chunks, 1):
        status = "ok"
        try:
            text = _transcribe_one_with_retry(chunk).strip()
            if text:
                parts.append(text)
                successes += 1
            else:
                parts.append(f"[chunk {i} empty]")
                status = "empty"
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"[youtube-mcp] chunk {i}/{total} failed: {e}\n")
            sys.stderr.flush()
            parts.append(f"[chunk {i} unavailable]")
            status = "failed"
        _notify_progress(
            f"transcribed chunk {i} of {total}",
            chunk=i,
            total=total,
            status=status,
        )
    if successes == 0:
        raise RuntimeError("all chunks failed transcription")
    return "\n\n".join(parts)


# ---------------------------------------------------------- persist transcript
def _safe_filename(s: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", s)[:64]
    return cleaned or "unknown"


def _save_transcript(
    video_id: str,
    title: str,
    url: str,
    source: str,
    transcript: str,
) -> Path:
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPT_DIR / f"{_safe_filename(video_id)}.txt"
    header = (
        f"# Title: {title}\n"
        f"# URL: {url}\n"
        f"# Source: {source}\n"
        f"# Saved: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
        f"\n"
    )
    try:
        path.write_text(header + transcript.rstrip() + "\n", encoding="utf-8")
    except OSError as e:
        sys.stderr.write(f"[youtube-mcp] failed to save transcript: {e}\n")
        sys.stderr.flush()
    return path


# ------------------------------------------------------------------- tools
def _tool_transcribe_youtube(arguments: dict[str, Any]) -> str:
    url = arguments.get("url")
    if not isinstance(url, str) or not url.strip():
        raise ValueError("missing required string argument 'url'")
    target = _normalise_target(url)

    with tempfile.TemporaryDirectory(prefix="yt-mcp-") as td:
        td_path = Path(td)
        info_dir = td_path / "info"
        audio_dir = td_path / "audio"
        chunk_dir = td_path / "chunks"
        for d in (info_dir, audio_dir, chunk_dir):
            d.mkdir()

        # 1) Try captions first — instant when available.
        _notify_progress("trying subtitles...", stage="subtitles")
        subs_text, info = _try_subtitles(target, info_dir)
        video_id = info.get("id") or _extract_video_id(target)
        title = info.get("title") or "(unknown)"
        canonical_url = info.get("webpage_url") or target
        duration = info.get("duration")

        subs_reason = ""
        if subs_text:
            ok, subs_reason = _check_subtitle_quality(subs_text, duration)
            if ok:
                _save_transcript(
                    video_id,
                    title,
                    canonical_url,
                    "subtitles",
                    subs_text,
                )
                return subs_text
            _notify_progress(
                f"subtitles failed quality check: {subs_reason}; falling back to audio STT",
                stage="subtitles_rejected",
                video_id=video_id,
                reason=subs_reason,
            )

        # 2) Fall back to audio download + chunked STT.
        _notify_progress(
            (
                "subtitles not usable, downloading audio..."
                if subs_reason
                else "subtitles not found, downloading audio..."
            ),
            stage="download",
            video_id=video_id,
        )
        audio = _download_audio(target, audio_dir)
        if audio.stat().st_size == 0:
            raise RuntimeError("downloaded audio is empty (no audio track?)")
        chunks = _chunk_audio(audio, chunk_dir)
        _notify_progress(
            f"audio downloaded, splitting into {len(chunks)} chunks...",
            stage="chunked",
            video_id=video_id,
            total=len(chunks),
        )
        transcript = _transcribe_chunks(chunks)
        _notify_progress(
            "combining transcript...",
            stage="combine",
            video_id=video_id,
        )
        if not transcript.strip():
            raise RuntimeError("transcription produced empty result")
        _save_transcript(video_id, title, canonical_url, "stt", transcript)
        return transcript


def _tool_set_openrouter_key(arguments: dict[str, Any]) -> str:
    # Kept for back-compat; now sets the optional STT endpoint bearer token
    # (usually unnecessary — the local STT needs no auth).
    global _stt_api_key
    key = arguments.get("key")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("missing required string argument 'key'")
    _stt_api_key = key.strip()
    return "STT endpoint API key set (note: local STT needs no key)."


def _tool_set_cookies(arguments: dict[str, Any]) -> str:
    """Set browser cookies in Netscape format for yt-dlp --cookies.

    Accepts the raw text of a ``cookies.txt`` file exported from your
    browser (via the "Get cookies.txt LOCALLY" extension or DevTools).
    The cookies are written to ``~/.cache/sharedagents-youtube-mcp/cookies.txt``
    and used as ``--cookies`` for every subsequent yt-dlp call.

    **Required for:** channel scanning, playlist listing, age-restricted
    content, and any operation that fails with anonymous requests.

    How to get the cookies::

      1. Install "Get cookies.txt LOCALLY" extension:
         Chrome: https://chrome.google.com/webstore/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc
         Firefox: https://addons.mozilla.org/firefox/addon/cookies-txt/
      2. Go to youtube.com (logged in)
      3. Click the extension icon → Export → saves cookies.txt
      4. Paste the entire file contents as the ``cookies_text`` argument
    """
    global _cookie_file
    cookies_text = arguments.get("cookies_text")
    if not isinstance(cookies_text, str) or not cookies_text.strip():
        raise ValueError("missing required string argument 'cookies_text'")

    # Validate basic Netscape format — should have tab-separated fields
    lines = [ln for ln in cookies_text.strip().split("\n") if ln and not ln.startswith("#")]
    if not lines:
        raise ValueError("cookies_text contains no cookie entries (only comments)")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cookie_file = CACHE_DIR / "cookies.txt"
    _cookie_file.write_text(cookies_text, encoding="utf-8")

    cookie_count = len(lines)
    return f"Cookies saved ({cookie_count} entries). Used for all subsequent downloads."


def _build_yt_dlp_cmd(extra_args: list[str], url: str) -> list[str]:
    """Build a yt-dlp command with cookie injection when available."""
    cmd = list(_resolve_yt_dlp_base())
    if _cookie_file and _cookie_file.exists():
        cmd += ["--cookies", str(_cookie_file)]
    cmd.extend(extra_args)
    cmd.append(url)
    return cmd


def _tool_scan_channel(arguments: dict[str, Any]) -> str:
    """Scan a YouTube channel's recent videos and return metadata.

    Returns video id, title, duration, and upload date for the most
    recent videos on a channel. Requires valid browser cookies (set via
    ``set_cookies``) — YouTube blocks anonymous channel listing.

    Args:
        channel_url: YouTube channel URL or handle
            (e.g., "https://www.youtube.com/@Bloomberg" or "@Bloomberg")
        count: Number of recent videos to return (default 10, max 50)
    """
    channel_url = arguments.get("channel_url")
    if not isinstance(channel_url, str) or not channel_url.strip():
        raise ValueError("missing required string argument 'channel_url'")

    count = min(int(arguments.get("count", 10)), 50)

    # Normalise to full URL if just a handle
    url = channel_url.strip()
    if url.startswith("@") or not url.startswith("http"):
        url = f"https://www.youtube.com/{url}/videos"

    _ensure_yt_dlp()

    cmd = _build_yt_dlp_cmd(
        [
            "--flat-playlist",
            "--no-warnings",
            "--quiet",
            "--playlistend",
            str(count),
            "--print",
            "%(id)s\t%(title)s\t%(duration_string)s\t%(upload_date)s",
        ],
        url,
    )

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=YT_DLP_TIMEOUT_S)
    if proc.returncode != 0:
        err = (proc.stderr or "").strip()[:600]
        # Check if it's a cookie issue
        if "404" in err or "Not Found" in err or "age restriction" in err.lower():
            err += (
                "\n\nHint: Channel scanning requires valid browser cookies. "
                "Export them from your browser using the 'Get cookies.txt LOCALLY' "
                "extension while logged into YouTube, then call set_cookies()."
            )
        raise RuntimeError(f"yt-dlp failed: {err}")

    output = proc.stdout.strip()
    if not output:
        return f"No videos found for {channel_url}"

    # Parse tab-separated output into structured text
    videos = []
    for line in output.split("\n"):
        parts = line.split("\t")
        if len(parts) >= 2:
            vid_id = parts[0]
            title = parts[1]
            duration = parts[2] if len(parts) > 2 else "?"
            upload = parts[3][:10] if len(parts) > 3 and parts[3] else "?"
            videos.append(f"- [{vid_id}] {title}\n  Duration: {duration} | Uploaded: {upload}")

    return f"\n{len(videos)} recent videos from {channel_url}:\n\n" + "\n".join(videos)


TOOL_DEFS: list[dict[str, Any]] = [
    {
        "name": "transcribe_youtube",
        "description": (
            "Transcribe a YouTube video. Tries YouTube subtitles first "
            "(manual or auto-generated en); falls back to chunked audio "
            "download + speech-to-text via the local Qwen3-ASR service. "
            "Long videos are split into 5-minute chunks for reliability. "
            "Completed transcripts are saved to "
            "~/.cache/sharedagents-youtube-mcp/transcripts/<video_id>.txt. "
            "Accepts any YouTube URL or video ID."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": (
                        "YouTube URL or video ID (e.g., 'dQw4w9WgXcQ' or 'https://youtube.com/watch?v=dQw4w9WgXcQ')"
                    ),
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "set_openrouter_key",
        "description": (
            "Set an optional STT-endpoint bearer token (local STT needs none). "
            "Rarely needed; the local STT requires no key. "
            "is not set in the environment."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": ("Optional STT endpoint bearer token."),
                },
            },
            "required": ["key"],
        },
    },
    {
        "name": "set_cookies",
        "description": (
            "Inject browser cookies (Netscape cookies.txt format) to "
            "bypass YouTube bot detection. Export from your browser using "
            "the 'Get cookies.txt LOCALLY' extension while logged into "
            "youtube.com, then paste the file contents here. Cookies are "
            "persisted in ~/.cache/sharedagents-youtube-mcp/cookies.txt and "
            "used for all subsequent calls. Required for channel scanning."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "cookies_text": {
                    "type": "string",
                    "description": (
                        "Full contents of cookies.txt exported from your "
                        "browser (Netscape/Mozilla cookie format). Each "
                        "line has 7 tab-separated fields: domain, flag, "
                        "path, secure, expiration, name, value."
                    ),
                },
            },
            "required": ["cookies_text"],
        },
    },
    {
        "name": "scan_channel",
        "description": (
            "Scan a YouTube channel's recent videos and return metadata "
            "(video id, title, duration, upload date). Requires valid "
            "browser cookies — set them first via set_cookies(). "
            "Useful for finding recent content on a channel like Bloomberg."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "channel_url": {
                    "type": "string",
                    "description": (
                        "YouTube channel URL or handle. Examples: "
                        "'@Bloomberg', 'https://www.youtube.com/@Bloomberg', "
                        "'@MarketSqueeze'"
                    ),
                },
                "count": {
                    "type": "integer",
                    "default": 10,
                    "description": ("Number of recent videos to return (default 10, max 50)"),
                },
            },
            "required": ["channel_url"],
        },
    },
]

TOOL_DISPATCH = {
    "transcribe_youtube": _tool_transcribe_youtube,
    "set_openrouter_key": _tool_set_openrouter_key,
    "set_cookies": _tool_set_cookies,
    "scan_channel": _tool_scan_channel,
}


# ------------------------------------------------------------ JSON-RPC loop
def _send(msg: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _notify_progress(message: str, **extra: Any) -> None:
    """Send a JSON-RPC progress notification (no ``id`` => not a response).

    Keeps the stdout pipe active during long-running tool calls so the
    client's idle timeout doesn't kill the connection. The client ignores
    notifications and keeps waiting for the response with matching id.
    """
    params: dict[str, Any] = {"message": message}
    params.update(extra)
    try:
        _send(
            {
                "jsonrpc": "2.0",
                "method": "notifications/progress",
                "params": params,
            }
        )
    except Exception:  # noqa: BLE001
        # Never let progress reporting break the actual work.
        pass


def _ok(req_id: Any, result: Any) -> None:
    _send({"jsonrpc": "2.0", "id": req_id, "result": result})


def _err(req_id: Any, code: int, message: str, data: Any = None) -> None:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    _send({"jsonrpc": "2.0", "id": req_id, "error": err})


def _handle_initialize(_params: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {"tools": {}},
        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
    }


def _handle_tools_list(_params: dict[str, Any]) -> dict[str, Any]:
    return {"tools": TOOL_DEFS}


def _handle_tools_call(params: dict[str, Any]) -> dict[str, Any]:
    name = params.get("name")
    arguments = params.get("arguments") or {}
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return {
            "content": [{"type": "text", "text": f"unknown tool: {name!r}"}],
            "isError": True,
        }
    try:
        text = fn(arguments)
    except Exception as e:  # noqa: BLE001
        return {
            "content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}],
            "isError": True,
        }
    return {"content": [{"type": "text", "text": text}], "isError": False}


def _dispatch(method: str, params: dict[str, Any]) -> dict[str, Any]:
    if method == "initialize":
        return _handle_initialize(params)
    if method == "tools/list":
        return _handle_tools_list(params)
    if method == "tools/call":
        return _handle_tools_call(params)
    if method == "ping":
        return {}
    raise LookupError(f"method not found: {method}")


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            _err(None, -32700, f"parse error: {e}")
            continue
        method = msg.get("method")
        msg_id = msg.get("id")
        params = msg.get("params") or {}
        is_notification = msg_id is None and method is not None
        if is_notification:
            continue
        if method is None:
            continue
        try:
            result = _dispatch(method, params)
        except LookupError as e:
            _err(msg_id, -32601, str(e))
            continue
        except Exception as e:  # noqa: BLE001
            _err(msg_id, -32603, f"internal error: {e}")
            continue
        _ok(msg_id, result)


if __name__ == "__main__":
    main()
