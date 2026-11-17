"""PCM16 helpers: base64 framing, WAV wrap/parse, resampling, level metering.

Everything on the realtime wire is PCM16 little-endian mono. The canonical
wire rate is 24 kHz (``WIRE_RATE``); stage outputs at other rates are
resampled here with numpy linear interpolation — good enough for speech.
"""

from __future__ import annotations

import base64
import io
import wave

import numpy as np

WIRE_RATE = 24_000


def b64_to_pcm(data: str) -> bytes:
    return base64.b64decode(data)


def pcm_to_b64(pcm: bytes) -> str:
    return base64.b64encode(pcm).decode("ascii")


def pcm_to_array(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2")


def array_to_pcm(arr: np.ndarray) -> bytes:
    return np.clip(arr, -32768, 32767).astype("<i2").tobytes()


def rms(pcm: bytes) -> float:
    """Root-mean-square level of an int16 chunk (0.0 when empty)."""
    arr = pcm_to_array(pcm)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))


def wav_wrap(pcm: bytes, rate: int = WIRE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def wav_parse(data: bytes) -> tuple[bytes, int]:
    """Return ``(pcm16_mono, rate)`` from WAV bytes; stereo is downmixed."""
    with wave.open(io.BytesIO(data), "rb") as w:
        rate = w.getframerate()
        channels = w.getnchannels()
        width = w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"only 16-bit WAV supported, got {8 * width}-bit")
    arr = np.frombuffer(frames, dtype="<i2")
    if channels > 1:
        arr = arr.reshape(-1, channels).mean(axis=1).astype("<i2")
    return arr.tobytes(), rate


def resample(pcm: bytes, from_rate: int, to_rate: int) -> bytes:
    if from_rate == to_rate or not pcm:
        return pcm
    arr = pcm_to_array(pcm).astype(np.float64)
    n_out = int(round(arr.size * to_rate / from_rate))
    if n_out <= 0:
        return b""
    x_in = np.linspace(0.0, 1.0, num=arr.size, endpoint=False)
    x_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return array_to_pcm(np.interp(x_out, x_in, arr))


def silence(ms: int, rate: int = WIRE_RATE) -> bytes:
    return b"\x00\x00" * int(rate * ms / 1000)
