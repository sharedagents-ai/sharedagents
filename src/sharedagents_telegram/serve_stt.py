"""
Lightweight STT server using faster-whisper — the DISTRIBUTED DEFAULT STT for
sharedagents. OpenAI-compatible /v1/audio/transcriptions endpoint.

Served on Starlette + Hypercorn (both already core sharedagents deps), so the only
extra a `pip install sharedagents[stt]` pulls in is faster-whisper itself.

Defaults follow Open WebUI: the `base` model (~150 MB), auto device detection
(CUDA if a GPU is visible, otherwise CPU), and `int8` compute so it runs out of
the box on CPU with NO GPU required. Override via env for a GPU box:

    STT_MODEL=small STT_DEVICE=cuda STT_COMPUTE=float16 python -m sharedagents_telegram.serve_stt

For a heavier local ASR (e.g. Qwen3-ASR-1.7B) point the frontends at that
endpoint via ~/.sharedagents/config.json instead of running this server.
"""

import asyncio
import io
import os

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

# Open WebUI parity: base model, auto device, int8 (CPU-friendly) compute.
MODEL = os.environ.get("STT_MODEL", "base")
DEVICE = os.environ.get("STT_DEVICE", "auto")
COMPUTE = os.environ.get("STT_COMPUTE", "int8")
PORT = int(os.environ.get("PORT", "8002"))

# Lazy singleton — the (potentially large) Whisper model is loaded on the FIRST
# transcription request, not at import time. This keeps import + /health cheap
# and non-blocking, so a supervisor can health-check the process immediately.
_model = None


def _detect_device() -> str:
    """CUDA if a GPU is visible to CTranslate2 (faster-whisper's backend), else
    CPU — the Open WebUI auto-detect pattern, with no hard torch dependency."""
    try:
        import ctranslate2

        n = ctranslate2.get_cuda_device_count()
        if n > 0:
            print(f"[stt] device=cuda ({n} GPU{'s' if n != 1 else ''})", flush=True)
            return "cuda"
    except Exception:  # noqa: BLE001 — any probe failure means "no usable GPU"
        pass
    print("[stt] device=cpu (no CUDA)", flush=True)
    return "cpu"


def _load_model():
    """Import faster-whisper and construct the WhisperModel.

    The import lives here (not at module level) so the server module imports
    cleanly even when the optional `[stt]` extra is not installed — the error
    only surfaces on the first transcription, with an actionable message.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise RuntimeError("faster-whisper not installed. Run: pip install sharedagents[stt]") from exc

    device = _detect_device() if DEVICE == "auto" else DEVICE
    compute = COMPUTE
    # float16/float32 aren't valid CTranslate2 CPU compute types — coerce to
    # int8 so a CPU box never crashes on a GPU-shaped compute default.
    if device == "cpu" and compute in ("float16", "float32"):
        compute = "int8"
    # download_root: honor STT_DOWNLOAD_ROOT, else faster-whisper's default
    # HF cache (~/.cache/huggingface). Never hardcode an absolute home path.
    root = os.environ.get("STT_DOWNLOAD_ROOT") or None
    print(f"[stt] loading faster-whisper {MODEL} on {device}/{compute}", flush=True)
    try:
        return WhisperModel(MODEL, device=device, compute_type=compute, download_root=root)
    except Exception as exc:  # noqa: BLE001 — fall back to CPU if the GPU load fails
        if device == "cpu":
            raise
        print(f"[stt] {device} load failed ({exc}); falling back to cpu/int8", flush=True)
        return WhisperModel(MODEL, device="cpu", compute_type="int8", download_root=root)


def _get_model():
    """Return the process-wide model, loading it on first use."""
    global _model
    if _model is None:
        _model = _load_model()
        print("[stt] model loaded", flush=True)
    return _model


async def health(request: Request) -> JSONResponse:
    # Returns immediately without forcing a model load — cheap liveness check.
    return JSONResponse({"status": "ok", "model": f"faster-whisper/{MODEL}"})


async def models(request: Request) -> JSONResponse:
    status = "ready" if _model is not None else "loading..."
    return JSONResponse(
        {
            "object": "list",
            "data": [{"id": f"faster-whisper/{MODEL}", "object": "model", "status": status}],
        }
    )


def _parse_language(value) -> tuple[str | None, list[str]]:
    """``(pinned, candidates)`` from the ``language`` form field.

    ``""``/``"auto"`` → free detection; one code → pinned; a comma-separated
    list (``"pt,en"``) → detection restricted to those candidates. Whisper's
    own detection on short clips drifts to unrelated languages (French,
    Korean …); a speaker who only uses a known set is decoded in one of them.
    """
    raw = str(value or "").strip().lower()
    if not raw or raw == "auto":
        return None, []
    codes = list(dict.fromkeys(c.strip() for c in raw.split(",") if c.strip()))
    if len(codes) == 1:
        return codes[0], []
    return None, codes


def _restricted_language(model, audio, candidates: list[str]) -> str:
    """The most likely candidate per Whisper's language probabilities."""
    _lang, _prob, all_probs = model.detect_language(audio=audio)
    probs = dict(all_probs or [])
    return max(candidates, key=lambda c: probs.get(c, 0.0))


async def transcribe(request: Request) -> JSONResponse:
    form = await request.form()
    upload = form.get("file")
    if upload is None:
        return JSONResponse({"error": "missing 'file' form field"}, status_code=400)
    audio_bytes = await upload.read()
    language = form.get("language", "auto")
    lang, candidates = _parse_language(language)

    # WhisperModel.transcribe is blocking CPU/GPU work — run it off the event
    # loop so a long transcription doesn't stall other requests / health checks.
    def _run():
        model = _get_model()
        audio = io.BytesIO(audio_bytes)
        pinned = lang
        if candidates:
            from faster_whisper import decode_audio

            unknown = [c for c in candidates if c not in model.supported_languages]
            if unknown:
                raise ValueError(f"Unsupported language code(s): {', '.join(unknown)}")
            audio = decode_audio(audio, sampling_rate=16000)
            pinned = _restricted_language(model, audio, candidates)
        segments, info = model.transcribe(audio, language=pinned, beam_size=5)
        text = " ".join(seg.text for seg in segments)
        detected_lang = info.language if info else language
        return text, detected_lang

    try:
        text, detected_lang = await asyncio.to_thread(_run)
    except ValueError as exc:  # an unknown language code
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse({"text": text, "language": detected_lang})


app = Starlette(
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/v1/models", models, methods=["GET"]),
        Route("/v1/audio/transcriptions", transcribe, methods=["POST"]),
    ]
)


async def _serve() -> None:
    from hypercorn.asyncio import serve
    from hypercorn.config import Config

    config = Config()
    config.bind = [f"0.0.0.0:{PORT}"]
    config.loglevel = "warning"
    print(f"[stt] serving on 0.0.0.0:{PORT} (model loads on first request)", flush=True)
    await serve(app, config)


def main() -> None:
    asyncio.run(_serve())


if __name__ == "__main__":
    main()
