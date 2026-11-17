"""Isolated passive meeting transcription for the realtime service.

Meeting audio terminates here: VAD segments it, the configured ASR transcribes
it, and :class:`MeetingTranscriptStore` writes retrievable chunk files.  This
module deliberately does not import the interactive realtime session, gateway
client, agent runtime, LLM, tool, or TTS code.

The audible-notice handshake is part of the capture state machine.  A start
intent only issues a one-use signal challenge; capture remains closed until the
client reports that challenge completed.  Stop closes and flushes capture
before the stop-signal challenge is emitted.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from . import audio
from .noise import looks_like_noise, strip_cjk_artifacts
from .vad import EnergyVAD, VadConfig

_LOG = logging.getLogger("sharedagents.realtime.meeting")

SendFn = Callable[[dict], Awaitable[None]]

CHUNK_MAX_AGE_S = 5 * 60
CHUNK_MAX_BYTES = 16 * 1024
_CHUNK_NAME_RE = re.compile(r"chunk-\d{4}\.md\Z")


class Transcriber(Protocol):
    async def transcribe(self, pcm: bytes, rate: int = audio.WIRE_RATE) -> str: ...


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _manifest_chunk_path(directory: Path, raw_name: object) -> Path:
    """Resolve only the canonical chunk basename emitted by MeetingWriter."""
    if not isinstance(raw_name, str) or not _CHUNK_NAME_RE.fullmatch(raw_name):
        raise ValueError("invalid meeting chunk name")
    path = Path(raw_name)
    if path.is_absolute() or path.name != raw_name or len(path.parts) != 1:
        raise ValueError("invalid meeting chunk path")
    return directory / raw_name


def default_meetings_root() -> Path:
    configured = os.environ.get("SHAREDAGENTS_MEETINGS_DIR")
    return Path(configured).expanduser() if configured else Path.home() / ".sharedagents" / "meetings"


class MeetingTranscriptStore:
    """Durable meeting chunks plus read-only retrieval methods.

    Rotation is five minutes or 16 KiB of text, whichever comes first.  The
    manifest is rewritten after every durable state change so an interrupted
    process still leaves an auditable record.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or default_meetings_root()
        self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
        self.root.chmod(0o700)

    def create(self, meeting_id: str, *, start_signal_completed_at: datetime) -> MeetingWriter:
        stamp = start_signal_completed_at.astimezone(timezone.utc).strftime("%Y%m%d-%H%M%S")
        directory = self.root / f"{stamp}-{meeting_id}"
        directory.mkdir(mode=0o700)
        directory.chmod(0o700)
        return MeetingWriter(directory, meeting_id, start_signal_completed_at)

    def list_meetings(self) -> list[dict]:
        if not self.root.is_dir():
            return []
        out: list[dict] = []
        for manifest_path in sorted(self.root.glob("*/manifest.json"), reverse=True):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            out.append(manifest)
        return out

    def retrieve(self, meeting_id: str) -> dict:
        """Return manifest and chunks for an exact meeting id.

        This is deliberately a retrieval call, never an automatic prompt hook.
        """
        if not meeting_id or any(c not in "0123456789abcdef-" for c in meeting_id.lower()):
            raise KeyError(meeting_id)
        if not self.root.is_dir():
            raise KeyError(meeting_id)
        matches = [p for p in self.root.glob(f"*-{meeting_id}") if p.is_dir()]
        if len(matches) != 1:
            raise KeyError(meeting_id)
        directory = matches[0]
        try:
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            chunks = []
            for name in manifest.get("chunks", []):
                chunk_path = _manifest_chunk_path(directory, name)
                chunks.append(
                    {
                        "name": name,
                        "text": chunk_path.read_text(encoding="utf-8"),
                    }
                )
        except (OSError, ValueError) as exc:
            raise KeyError(meeting_id) from exc
        return {"manifest": manifest, "chunks": chunks}

    def search(self, query: str, *, limit: int = 6) -> list[dict]:
        """Lexically search stored chunks for explicit later-prompt recall.

        Nothing invokes this automatically: the caller must supply a query via
        the ``meeting_search`` tool or another deliberate retrieval surface.
        """
        terms = tuple(dict.fromkeys(re.findall(r"[\w'-]+", query.casefold())))
        if not terms:
            return []
        hits: list[tuple[int, str, str, str]] = []
        for manifest in self.list_meetings():
            meeting_id = str(manifest.get("meeting_id") or "")
            if not meeting_id:
                continue
            matches = [p for p in self.root.glob(f"*-{meeting_id}") if p.is_dir()]
            if len(matches) != 1:
                continue
            for name in manifest.get("chunks", []):
                try:
                    chunk_path = _manifest_chunk_path(matches[0], name)
                    text = chunk_path.read_text(encoding="utf-8")
                except (OSError, ValueError):
                    continue
                folded = text.casefold()
                score = sum(folded.count(term) for term in terms)
                if not score:
                    continue
                first = min((folded.find(t) for t in terms if t in folded), default=0)
                start = max(0, first - 240)
                end = min(len(text), first + 760)
                snippet = text[start:end].strip()
                hits.append((score, meeting_id, str(name), snippet))
        hits.sort(key=lambda hit: (-hit[0], hit[1], hit[2]))
        return [
            {"meeting_id": mid, "chunk": name, "score": score, "text": snippet}
            for score, mid, name, snippet in hits[: max(1, min(int(limit), 20))]
        ]


class MeetingWriter:
    def __init__(self, directory: Path, meeting_id: str, started_at: datetime) -> None:
        self.directory = directory
        self.meeting_id = meeting_id
        self._chunk_started_at: datetime | None = None
        self._chunk_bytes = 0
        self._chunk_count = 0
        self._closed = False
        started = started_at.isoformat()
        self.manifest = {
            "meeting_id": meeting_id,
            "start_signal_completed_at": started,
            "capture_started_at": started,
            "capture_stopped_at": None,
            "stop_signal_completed_at": None,
            "chunks": [],
        }
        self._write_manifest()

    @property
    def chunk_count(self) -> int:
        return self._chunk_count

    def append(self, transcript: str, *, captured_at: datetime | None = None) -> None:
        if self._closed:
            raise RuntimeError("meeting capture is closed")
        text = transcript.strip()
        if not text:
            return
        now = captured_at or _utc_now()
        line = f"[{now.isoformat()}] {text}\n"
        encoded_len = len(line.encode("utf-8"))
        age = (now - self._chunk_started_at).total_seconds() if self._chunk_started_at else 0
        if (
            self._chunk_started_at is None
            or age >= CHUNK_MAX_AGE_S
            or (self._chunk_bytes and self._chunk_bytes + encoded_len > CHUNK_MAX_BYTES)
        ):
            self._open_chunk(now)
        path = self.directory / self.manifest["chunks"][-1]
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
        path.chmod(0o600)
        self._chunk_bytes += encoded_len

    def stop_capture(self, *, stopped_at: datetime | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        self.manifest["capture_stopped_at"] = (stopped_at or _utc_now()).isoformat()
        self._write_manifest()

    def record_stop_signal(self, *, completed_at: datetime | None = None) -> None:
        if not self._closed:
            raise RuntimeError("cannot announce stop before capture closes")
        self.manifest["stop_signal_completed_at"] = (completed_at or _utc_now()).isoformat()
        self._write_manifest()

    def _open_chunk(self, now: datetime) -> None:
        self._chunk_count += 1
        name = f"chunk-{self._chunk_count:04d}.md"
        (self.directory / name).touch(mode=0o600)
        (self.directory / name).chmod(0o600)
        self.manifest["chunks"].append(name)
        self._chunk_started_at = now
        self._chunk_bytes = 0
        self._write_manifest()

    def _write_manifest(self) -> None:
        path = self.directory / "manifest.json"
        tmp = self.directory / ".manifest.json.tmp"
        tmp.write_text(json.dumps(self.manifest, indent=2) + "\n", encoding="utf-8")
        tmp.chmod(0o600)
        tmp.replace(path)


class MeetingSession:
    """Protocol/state boundary for one meeting on a realtime connection."""

    def __init__(
        self,
        send: SendFn,
        transcriber: Transcriber,
        *,
        store: MeetingTranscriptStore | None = None,
        vad: EnergyVAD | None = None,
        asr_language: str = "en,pt",
    ) -> None:
        self._send = send
        self._transcriber = transcriber
        self._store = store or MeetingTranscriptStore()
        self._vad = vad or EnergyVAD(VadConfig())
        # Same spec as RealtimeConfig.asr_language, handed to the shared
        # non-speech guard (en/pt only: no value disables its CJK check).
        self._asr_language = asr_language
        self._state = "idle"
        self._meeting_id: str | None = None
        self._signal_token: str | None = None
        self._signal_phase: str | None = None
        self._writer: MeetingWriter | None = None

    @property
    def active(self) -> bool:
        return self._state == "active"

    @property
    def finished(self) -> bool:
        return self._state == "finished"

    async def handle_event(self, event: dict) -> None:
        event_type = event.get("type") or ""
        handlers = {
            "meeting.start": self._start,
            "meeting.stop": self._stop,
            "meeting.signal.completed": self._signal_completed,
            "meeting.signal.failed": self._signal_failed,
            "input_audio_buffer.append": self._append,
            "input_audio_buffer.commit": self._commit,
        }
        handler = handlers.get(event_type)
        if handler is None:
            await self._error(f"unsupported meeting event type: {event_type!r}")
            return
        try:
            await handler(event)
        except Exception as exc:  # fail closed; never activate/capture after an error
            await self._fail_closed(f"{event_type} failed: {exc}")

    async def close(self) -> None:
        if self._state == "active":
            try:
                await self._close_capture()
            except Exception as exc:
                await self._fail_closed(f"meeting close failed: {exc}", emit=False)
        self._state = "finished"

    async def _start(self, _event: dict) -> None:
        if self._state != "idle":
            raise RuntimeError("meeting has already been started")
        self._meeting_id = str(uuid.uuid4())
        self._state = "start_signal"
        await self._issue_signal("start")

    async def _signal_completed(self, event: dict) -> None:
        phase = event.get("phase")
        token = event.get("token")
        if phase != self._signal_phase or not secrets.compare_digest(str(token), self._signal_token or ""):
            raise RuntimeError("invalid or stale audible-signal completion")
        self._signal_token = None
        self._signal_phase = None
        now = _utc_now()
        if phase == "start":
            if self._state != "start_signal" or not self._meeting_id:
                raise RuntimeError("start signal is not pending")
            self._writer = self._store.create(self._meeting_id, start_signal_completed_at=now)
            self._state = "active"
            await self._emit("meeting.started", meeting_id=self._meeting_id)
            return
        if phase == "stop":
            if self._state != "stop_signal" or self._writer is None:
                raise RuntimeError("stop signal is not pending")
            self._writer.record_stop_signal(completed_at=now)
            self._state = "finished"
            await self._emit("meeting.signal.acknowledged", phase="stop")
            return
        raise RuntimeError("unknown audible-signal phase")

    async def _signal_failed(self, event: dict) -> None:
        phase = event.get("phase")
        token = event.get("token")
        if phase != self._signal_phase or not secrets.compare_digest(str(token), self._signal_token or ""):
            raise RuntimeError("invalid or stale audible-signal failure")
        self._signal_token = None
        self._signal_phase = None
        if phase == "start":
            self._state = "finished"  # fail closed: writer/capture never opened
        elif phase == "stop":
            self._state = "finished"  # capture was already closed before signal request
        await self._error(f"meeting {phase} audible signal failed")

    async def _append(self, event: dict) -> None:
        if self._state != "active":
            raise RuntimeError("meeting capture is not active")
        pcm = audio.b64_to_pcm(event.get("audio") or "")
        for kind, utterance in self._vad.feed(pcm):
            if kind == "speech_end":
                await self._transcribe(utterance)

    async def _commit(self, _event: dict) -> None:
        if self._state != "active":
            raise RuntimeError("meeting capture is not active")
        await self._flush_vad()

    async def _stop(self, _event: dict) -> None:
        if self._state != "active":
            raise RuntimeError("meeting capture is not active")
        await self._close_capture()
        assert self._writer is not None
        self._state = "stop_signal"
        # The stopped event proves capture is closed before the audible stop
        # challenge is emitted.  The browser only plays the chime after this.
        await self._emit(
            "meeting.stopped",
            meeting_id=self._meeting_id,
            chunk_count=self._writer.chunk_count,
        )
        await self._issue_signal("stop")

    async def _close_capture(self) -> None:
        await self._flush_vad()
        if self._writer is not None:
            self._writer.stop_capture()

    async def _flush_vad(self) -> None:
        for kind, utterance in self._vad.flush():
            if kind == "speech_end":
                await self._transcribe(utterance)

    async def _transcribe(self, utterance: bytes) -> None:
        if not utterance or self._writer is None:
            return
        transcript = (await self._transcriber.transcribe(utterance)).strip()
        # Mirror the interactive session guard: drop non-speech artifacts
        # (silence, coughs, hallucinated CJK filler) BEFORE emitting or
        # persisting them. Strip first, so a MIXED transcript keeps its speech
        # ("Hello 嗯。" is persisted as "Hello") instead of being dropped whole.
        transcript = strip_cjk_artifacts(transcript)
        if not transcript or looks_like_noise(transcript, self._asr_language):
            _LOG.info(
                "ignoring non-speech meeting transcript %r (%.1fs audio)",
                transcript,
                len(utterance) / 2 / audio.WIRE_RATE,
            )
            return
        self._writer.append(transcript)
        await self._emit("meeting.transcription.completed", transcript=transcript)

    async def _issue_signal(self, phase: str) -> None:
        self._signal_phase = phase
        self._signal_token = secrets.token_urlsafe(24)
        await self._emit("meeting.signal", phase=phase, token=self._signal_token)

    async def _fail_closed(self, message: str, *, emit: bool = True) -> None:
        """Irreversibly close capture after any ASR/persistence/state failure."""
        was_active = self._state == "active"
        self._state = "finished"
        self._signal_phase = None
        self._signal_token = None
        if self._writer is not None:
            try:
                self._writer.stop_capture()
            except Exception:
                pass  # original failure remains authoritative; finalize best-effort
        if emit:
            if was_active:
                await self._emit("meeting.failed", meeting_id=self._meeting_id, reason=message)
            await self._error(message)

    async def _emit(self, event_type: str, **fields: object) -> None:
        await self._send({"type": event_type, **fields})

    async def _error(self, message: str) -> None:
        await self._emit("error", error={"message": message})
