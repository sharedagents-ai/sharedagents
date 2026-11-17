"""Server-side voice-activity detection (energy-based, v1).

Frame-level RMS against an adaptive noise floor. Deliberately dependency-free
(no torch/silero on the gateway host); the interface is one method —
``feed(pcm) -> list[events]`` — so a smarter VAD can drop in later.

VAD only segments audio; the session stop word alone dispatches a turn.

Events:
    ("speech_start", b"")           — emitted once when speech begins
    ("speech_end", utterance_pcm)   — full utterance incl. prefix padding
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import audio


@dataclass
class VadConfig:
    rate: int = audio.WIRE_RATE
    frame_ms: int = 20
    # Speech begins after this much consecutive above-threshold audio. Set
    # above a single short blip (cough/click) so noise doesn't open a turn.
    start_ms: int = 180
    # …and ends after this much below-threshold audio.
    silence_ms: int = 800
    # Audio kept from before the trigger so first syllables aren't clipped.
    prefix_ms: int = 500
    # Threshold = max(floor * ratio, min_rms); floor is an EMA of quiet frames.
    threshold_ratio: float = 3.5
    min_rms: float = 320.0
    floor_alpha: float = 0.05
    # Hard cap so a stuck-open mic can't grow the buffer forever.
    max_utterance_s: int = 60


@dataclass
class EnergyVAD:
    cfg: VadConfig = field(default_factory=VadConfig)

    def __post_init__(self) -> None:
        c = self.cfg
        self._frame_bytes = 2 * int(c.rate * c.frame_ms / 1000)
        self._pending = bytearray()  # bytes not yet forming a frame
        self._prefix = bytearray()  # rolling pre-speech padding
        self._utterance = bytearray()
        self._in_speech = False
        self._above_ms = 0
        self._below_ms = 0
        self._floor = c.min_rms

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    @property
    def speech_pending(self) -> bool:
        """Loud frames are accumulating toward ``speech_start`` but the
        ``start_ms`` threshold hasn't been crossed yet (the VAD start window).
        Lets callers treat the very start of an utterance as 'busy' before
        ``in_speech`` flips."""
        return not self._in_speech and self._above_ms > 0

    @property
    def threshold(self) -> float:
        return max(self._floor * self.cfg.threshold_ratio, self.cfg.min_rms)

    @property
    def current_utterance(self) -> bytes:
        """The in-progress utterance bytes (incl. prefix padding) while speech
        is active, else ``b""``. Used to re-decode a partial transcript live —
        never consumes the buffer (the final ``speech_end`` event still carries
        the complete utterance)."""
        return bytes(self._utterance) if self._in_speech else b""

    def feed(self, pcm: bytes) -> list[tuple[str, bytes]]:
        self._pending.extend(pcm)
        events: list[tuple[str, bytes]] = []
        while len(self._pending) >= self._frame_bytes:
            frame = bytes(self._pending[: self._frame_bytes])
            del self._pending[: self._frame_bytes]
            events.extend(self._frame(frame))
        return events

    def flush(self) -> list[tuple[str, bytes]]:
        """Manual end-of-turn (``input_audio_buffer.commit`` / disconnect)."""
        if not self._in_speech:
            # Clear any half-formed start window so a sub-threshold blip doesn't
            # leave ``speech_pending`` stuck true after a commit/clear.
            self._above_ms = 0
            self._prefix.clear()
            return []
        utterance = bytes(self._utterance)
        self._reset_turn()
        return [("speech_end", utterance)]

    # ------------------------------------------------------------------

    def _frame(self, frame: bytes) -> list[tuple[str, bytes]]:
        c = self.cfg
        level = audio.rms(frame)
        loud = level >= self.threshold
        events: list[tuple[str, bytes]] = []

        if not self._in_speech:
            if not loud:
                # Only quiet frames teach the noise floor.
                self._floor += c.floor_alpha * (level - self._floor)
            self._prefix.extend(frame)
            max_prefix = 2 * int(c.rate * c.prefix_ms / 1000)
            if len(self._prefix) > max_prefix:
                del self._prefix[: len(self._prefix) - max_prefix]
            self._above_ms = self._above_ms + c.frame_ms if loud else 0
            if self._above_ms >= c.start_ms:
                self._in_speech = True
                self._below_ms = 0
                self._utterance = bytearray(self._prefix)
                self._prefix.clear()
                events.append(("speech_start", b""))
            return events

        self._utterance.extend(frame)
        self._below_ms = 0 if loud else self._below_ms + c.frame_ms
        too_long = len(self._utterance) >= 2 * c.rate * c.max_utterance_s
        if self._below_ms >= c.silence_ms or too_long:
            utterance = bytes(self._utterance)
            self._reset_turn()
            events.append(("speech_end", utterance))
        return events

    def _reset_turn(self) -> None:
        self._in_speech = False
        self._above_ms = 0
        self._below_ms = 0
        self._utterance = bytearray()
        self._prefix.clear()
