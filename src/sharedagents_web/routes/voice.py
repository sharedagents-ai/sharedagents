"""sharedagents_web.routes.voice — Speech-to-text transcription route.

Extracted from server.py (Task 2 de-monolith). Handler bodies are
byte-identical to the originals; only the decorator target changed
(``@app.route`` -> ``@bp.route``). Registered by server.create_app().
"""

import requests
from flask import Blueprint, jsonify, request

# Shared, dependency-free ASR text guard — the same filtering the realtime voice
# paths apply (sharedagents_realtime.noise re-exports this module's names). Kept in
# sharedagents_core, the leaf package both sides already contain, so the WebUI gets
# ONE copy of the rules instead of a second implementation drifting from the
# realtime one.
from sharedagents_core.asr_text import parse_languages, sanitize_transcript
from sharedagents_web.server import app, require_auth

bp = Blueprint("voice", __name__)

# Connect timeout stays short so a dead LAN primary fails over quickly;
# read timeout matches the gateway STT client (long utterances).
_STT_TIMEOUT = (10, 120)


def _prepare_stt_audio(filename: str, audio_bytes: bytes, content_type: str):
    """Normalize browser audio so primary and backup STT share one body.

    MediaRecorder typically emits WebM/Opus. Local Qwen3-ASR may accept that
    container; OpenRouter's backup does not. Transcode to 16 kHz mono MP3 when
    the container is a known non-native format, matching Telegram/gateway STT.
    """
    send_name = filename or "audio"
    send_bytes = audio_bytes
    send_type = content_type or "application/octet-stream"
    try:
        from sharedagents_telegram.stt import _stt_needs_transcode, _transcode_to_mp3
    except Exception:  # noqa: BLE001 — keep original bytes if helpers missing
        return send_name, send_bytes, send_type
    if not _stt_needs_transcode(send_type, send_name):
        return send_name, send_bytes, send_type
    transcoded = _transcode_to_mp3(
        audio_bytes,
        sample_rate=16000,
        channels=1,
    )
    if not transcoded:
        return send_name, send_bytes, send_type
    return "audio.mp3", transcoded, "audio/mpeg"


def _configured_languages() -> list[str]:
    """``stt.languages`` from the global config (list or ``"pt,en"``).

    Empty = free detection (today's behaviour). Only supported codes survive
    (``parse_languages``), so a typo never reaches the ASR request.
    """
    from sharedagents_core.kernel.config import load_global_config

    stt = load_global_config().get("stt")
    # Service chains may use a top-level list rather than a block. That shape
    # has no languages setting and must retain free detection.
    raw = stt.get("languages") if isinstance(stt, dict) else None
    if isinstance(raw, (list, tuple)):
        raw = ",".join(str(x) for x in raw)
    return parse_languages(str(raw or ""))


def _language_fields(languages: list[str], hint: str) -> dict:
    """Form fields that keep the decode inside the configured languages.

    A validated single-language hint pins (``to_language`` too: Qwen3-ASR only
    pins from that field); otherwise local faster-whisper gets the candidate
    list and detects among those languages only.
    """
    if not languages:
        return {}
    if hint in languages:
        return {"language": hint, "to_language": hint}
    if len(languages) == 1:
        return {"language": languages[0], "to_language": languages[0]}
    return {"language": ",".join(languages)}


@bp.route("/api/transcribe", methods=["POST"])
@require_auth
def api_transcribe():
    """Transcribe audio using the configured STT service.

    Accepts a multipart ``audio`` file field, proxies it to speech-to-text,
    and returns the transcribed text. Tries PRIMARY then BACKUP from the
    unified ``stt`` block in ``~/.sharedagents/config.json``.

    With ``stt.languages`` set, the decode stays inside those languages. The
    optional ``language`` field is the language an earlier chunk of the same
    recording came back in (the browser echoes it so a short chunk cannot flip
    language); it is honoured only when configured. The response carries the
    decoded ``language`` when it is one of the configured ones.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files["audio"]
    if not audio_file.filename:
        return jsonify({"error": "Empty audio file"}), 400

    audio_bytes = audio_file.read()
    if not audio_bytes:
        return jsonify({"error": "Empty audio file"}), 400

    from sharedagents_core.services import resolve_service

    candidates = resolve_service("stt").candidates()
    if not candidates:
        return jsonify(
            {
                "error": "STT is not configured; add an stt block to ~/.sharedagents/config.json",
            }
        ), 503

    send_name, send_bytes, send_type = _prepare_stt_audio(
        audio_file.filename,
        audio_bytes,
        audio_file.content_type or "",
    )

    languages = _configured_languages()
    hint = str(request.form.get("language") or "").strip().lower()
    lang_fields = _language_fields(languages, hint)

    errors: list[str] = []
    for endpoint in candidates:
        headers = {"Authorization": f"Bearer {endpoint.api_key}"} if endpoint.api_key else None
        data = {"model": endpoint.model, "response_format": "json", **lang_fields}
        try:
            resp = requests.post(
                endpoint.url("audio/transcriptions"),
                files={"file": (send_name, send_bytes, send_type)},
                data=data,
                headers=headers,
                timeout=_STT_TIMEOUT,
            )
            if resp.status_code in (400, 422) and "," in str(data.get("language", "")):
                # An endpoint without candidate-list support: free detection
                # beats no transcript; the text guard still applies.
                data.pop("language", None)
                resp = requests.post(
                    endpoint.url("audio/transcriptions"),
                    files={"file": (send_name, send_bytes, send_type)},
                    data=data,
                    headers=headers,
                    timeout=_STT_TIMEOUT,
                )
        except requests.RequestException as exc:
            errors.append(f"{endpoint.label}({endpoint.base_url}): {exc}")
            continue
        if resp.status_code != 200:
            errors.append(f"{endpoint.label}({endpoint.base_url}): HTTP {resp.status_code} {resp.text[:200]}")
            continue
        try:
            result = resp.json()
            text = (result.get("text") or "").strip()
            detected = str(result.get("language") or "").strip().lower()
        except (ValueError, AttributeError) as exc:
            app.logger.error(
                "Bad STT response from %s: %s; body=%s",
                endpoint.label,
                exc,
                resp.text[:500],
            )
            errors.append(f"{endpoint.label}({endpoint.base_url}): bad JSON")
            continue
        # Defense in depth: a silent chunk makes the
        # primary ASR model answer with invented filler ("嗯。") and the
        # browser appends every non-empty chunk verbatim, so filter here too —
        # sanitize_transcript() drops the artifact and returns "" when nothing
        # usable is left. The shared guard (sharedagents_core.asr_text) is the same
        # one the realtime voice paths use and keeps the Latin part of a MIXED
        # transcript ("Hello 嗯。" → "Hello") instead of losing the speech.
        raw_text = text
        text = sanitize_transcript(text)
        if request.form.get("final") == "1":
            # Review quality is server policy. The browser only transports its
            # draft and pre-recording prefix, so existing composer text is not
            # counted as speech. A severely truncated pass must not erase the
            # useful chunk draft; an empty reply tells the client to keep it.
            draft = str(request.form.get("draft") or "")
            prefix = str(request.form.get("prefix") or "")
            if prefix and draft.startswith(prefix):
                draft = draft[len(prefix) :]
            draft = sanitize_transcript(draft)
            if len(draft) >= 40 and len(text) < len(draft) * 0.5:
                app.logger.warning(
                    "Whole-recording STT truncated: %d chars vs draft %d via %s", len(text), len(draft), endpoint.label
                )
                text = ""
        if raw_text != text:
            app.logger.info(
                "STT artifact filtered: %r → %r via %s",
                raw_text[:200],
                text,
                endpoint.label,
            )
        if errors:
            app.logger.info(
                "STT succeeded after fallback; prior failures: %s",
                " | ".join(errors),
            )
        app.logger.info(
            "STT response: text=%r language=%s via %s",
            text[:200],
            detected or "-",
            endpoint.label,
        )
        body = {"text": text, "via": endpoint.label}
        if detected in languages:
            body["language"] = detected
        elif not detected and lang_fields.get("language") in languages:
            body["language"] = lang_fields["language"]
        return jsonify(body)

    app.logger.warning("STT failed on all endpoints: %s", " | ".join(errors))
    return jsonify(
        {
            "error": "STT failed — " + " | ".join(errors),
        }
    ), 502
