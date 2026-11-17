"""Stage configuration for the realtime cascade.

Reads the unified sharedagents config (``~/.sharedagents/config.json``):

- ``stt`` / ``tts`` service blocks via :mod:`sharedagents_core.services`
- chat from ``models[]`` (entry with ``role: "primary"``, else first)
- an optional ``realtime`` block overrides anything here::

    "realtime": {
        "chat": {"base_url": ..., "model": ..., "api_key_env": ...},
        "stt":  {...}, "tts": {...},
        "instructions": "...", "voice": "alloy",
        "disable_thinking": true,
        "vad": {"silence_ms": 800, "min_rms": 250}
    }
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

DEFAULT_INSTRUCTIONS = (
    "You are sharedagents, a helpful voice assistant. Your reply will be spoken "
    "aloud via TTS, so keep every answer to 1–3 short natural sentences "
    "(roughly ≤80 words). Prefer a brief spoken summary; offer to expand "
    "only if asked. Reply in English or Portuguese, matching whichever the "
    "user spoke; never reply in any other language. Never use markdown, "
    "bullet points, code, tables, emojis or URLs — plain speakable text only."
)


@dataclass
class Stage:
    base_url: str = ""
    model: str = ""
    api_key: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    def url(self, path: str) -> str:
        return self.base_url.rstrip("/") + "/" + path.lstrip("/")


@dataclass
class RealtimeConfig:
    stt: Stage = field(default_factory=Stage)
    stt_fallbacks: list[Stage] = field(default_factory=list)
    tts: Stage = field(default_factory=Stage)
    tts_fallbacks: list[Stage] = field(default_factory=list)
    chat: Stage = field(default_factory=Stage)
    instructions: str = DEFAULT_INSTRUCTIONS
    voice: str = "alloy"
    # Supported recognition languages (en/pt), comma-separated. One pins the
    # decode; several restrict Whisper detection to those candidates. The last
    # detected language pins short segments (<2 s) and hints Qwen fallbacks.
    # auto/empty keeps free detection. Non-speech guards still apply.
    asr_language: str = "en,pt"
    # vLLM honours chat_template_kwargs.enable_thinking; hosted providers may
    # 400 on it, so it's a profile switch rather than always-on.
    disable_thinking: bool = True
    temperature: float | None = None
    # Hard cap for the raw-cascade chat stage (agent mode uses Jarvis's own
    # sampler). TTS latency scales with tokens — keep this tight.
    max_tokens: int = 180
    vad: dict[str, Any] = field(default_factory=dict)
    # Interactive voice is the explicit Jarvis radio protocol by default. An
    # operator can still set this to "" for a typed/raw-cascade integration,
    # but microphone turns never gain a continuous/VAD-only fallback.
    agent: str = "jarvis"
    gateway_url: str = "http://127.0.0.1:8100"
    # Bearer token for a gateway whose control routes are protected by
    # gateway.auth_token. Resolved from the global ``gateway`` block unless the
    # ``realtime`` block sets its own.
    gateway_token: str = ""
    # Server protocol word. The client must open the audio channel by detecting
    # this word locally; repeating it sends the sole speech-driven interrupt.
    wake_word: str = "jarvis"
    wake_aliases: list[str] = field(default_factory=list)
    # Server-side explicit end-of-turn word. VAD only segments audio for ASR;
    # silence cannot dispatch an interactive turn. The cap is a bounded-resource
    # safety valve, not a normal turn-taking signal.
    conversation_end_word: str = "over"
    conversation_end_aliases: list[str] = field(default_factory=list)
    conversation_max_segments: int = 40
    # Cap audio retained for the stop-word-triggered whole-utterance review.
    # On overflow, keep segment text and skip review rather than truncate audio.
    max_utterance_s: int = 60
    # Live partial transcription (OPT-IN). When true, the user's in-progress
    # utterance is re-decoded every ``partial_interval_ms`` and streamed to the
    # client as ``…input_audio_transcription.delta`` (display-only "live
    # caption") before the authoritative ``…completed`` lands at end-of-speech.
    # ``max_partial_window_s`` caps the re-decode to the last N seconds so a long
    # utterance stays cheap; 0 = whole buffer. Mirrors OpenAI's transcription
    # ``delay`` knob — shorter interval = earlier, flickerier partials.
    transcribe_partials: bool = False
    partial_interval_ms: int = 700
    max_partial_window_s: int = 20
    # Client-side wake word (browser openWakeWord; DEFAULT ON). When enabled, the
    # browser detects the wake phrase locally and only streams audio upstream
    # once it fires. Failure is closed: it disables hands-free capture rather
    # than falling back to continuous streaming.
    client_wake: bool = True
    # Honest asset declaration: the bundled openWakeWord model recognizes the
    # phrase "Hey Jarvis". The settled protocol word remains literal "Jarvis";
    # a bare-Jarvis acoustic asset is not currently bundled.
    wake_keyword: str = "hey_jarvis"
    wake_phrase: str = "Hey Jarvis"
    wake_supports_literal_word: bool = False
    # openWakeWord score threshold. Lower = more sensitive. 0.4 missed real
    # "Hey Jarvis" utterances in a normal room; 0.28 matches prior operator
    # tuning. False positives are cheap (server TEXT gate / short stream).
    wake_threshold: float = 0.28
    # Short pre-roll: just enough to not clip the command that immediately follows
    # the wake word. It no longer needs to carry the wake word to the server (the
    # client-gated bypass handles that), so keep it small to avoid streaming prior
    # room chatter as part of the command.
    wake_preroll_ms: int = 700
    # Cooldown after a detect. Keep short so a missed engage (channel open
    # timeout) can be retried without waiting two full seconds.
    wake_cooldown_ms: int = 1200
    # Require Silero VAD to agree before honouring a wake score. OFF by default:
    # the hard VAD gate suppressed real wake peaks (the score lands just after
    # the phrase, often past the VAD window) and was the main sensitivity killer.
    # The server-side wake_word TEXT gate is the precision filter, so the client
    # can stay sensitive.
    wake_vad_gate: bool = False


def _as_bool(v: Any, default: bool = False) -> bool:
    """Coerce a JSON-ish value to bool. Real JSON booleans pass through; a
    string like "false"/"0"/"no"/"off" (which ``bool()`` would wrongly treat as
    True) is parsed; anything else falls back to ``bool(v)``."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(v) if v is not None else default


def _stage_from_dict(d: dict | None) -> Stage:
    if not d:
        return Stage()
    key = d.get("api_key") or ""
    env = d.get("api_key_env") or ""
    if not key and env:
        key = os.environ.get(env, "")
    return Stage(
        base_url=d.get("base_url") or d.get("baseUrl") or "", model=d.get("model") or d.get("id") or "", api_key=key
    )


def _stages_from_dict(d: dict | None) -> list[Stage]:
    if not d:
        return []
    if "primary" in d or "backup" in d:
        return [
            s
            for s in (
                _stage_from_dict(d.get("primary")),
                _stage_from_dict(d.get("backup")),
            )
            if s.configured
        ]
    s = _stage_from_dict(d)
    return [s] if s.configured else []


def _global_config() -> dict:
    home = os.environ.get("SHAREDAGENTS_HOME") or os.path.expanduser("~/.sharedagents")
    try:
        with open(os.path.join(home, "config.json"), encoding="utf-8") as fh:
            return json.load(fh) or {}
    except (OSError, ValueError):
        return {}


def _service_stage(name: str) -> Stage:
    try:
        from sharedagents_core.services import resolve_service

        ep = resolve_service(name).candidates()[0]
        return Stage(base_url=ep.base_url, model=ep.model, api_key=ep.api_key)
    except Exception:  # noqa: BLE001 — stdlib fallback keeps us standalone
        return _stage_from_dict(_global_config().get(name))


def _service_stages(name: str) -> list[Stage]:
    try:
        from sharedagents_core.services import resolve_service

        return [
            Stage(base_url=ep.base_url, model=ep.model, api_key=ep.api_key) for ep in resolve_service(name).candidates()
        ]
    except Exception:  # noqa: BLE001 - stdlib fallback keeps us standalone
        return _stages_from_dict(_global_config().get(name))


def stt_default() -> Stage:
    """Return the unified STT service, or an unconfigured stage.

    Self-hosted endpoints have no source-code default: the sole authority is
    the ``stt`` block in ``~/.sharedagents/config.json``.
    """
    return _service_stage("stt")


def _chat_stage(cfg: dict) -> Stage:
    models = cfg.get("models") or []
    chosen: dict | None = None
    for m in models:
        if m.get("role") == "primary":
            chosen = m
            break
    if chosen is None and models:
        chosen = models[0]
    if chosen:
        # NOTE: models[].apiKeyEnv historically holds junk for local vLLM
        # entries; only honour it when it names a set env var.
        env = chosen.get("apiKeyEnv") or ""
        key = os.environ.get(env, "") if env else ""
        return Stage(base_url=chosen.get("baseUrl", ""), model=chosen.get("id", ""), api_key=key)
    if cfg.get("upstream_endpoint"):
        return Stage(base_url=cfg["upstream_endpoint"], model=cfg.get("upstream_model", ""))
    return Stage()


def load_config() -> RealtimeConfig:
    raw = _global_config()
    rt = raw.get("realtime") or {}
    stt_stages = _stages_from_dict(rt.get("stt")) if rt.get("stt") else _service_stages("stt")
    tts_stages = _stages_from_dict(rt.get("tts")) if rt.get("tts") else _service_stages("tts")
    out = RealtimeConfig(
        stt=stt_stages[0] if stt_stages else stt_default(),
        stt_fallbacks=stt_stages[1:],
        tts=tts_stages[0] if tts_stages else Stage(),
        tts_fallbacks=tts_stages[1:],
        chat=_stage_from_dict(rt.get("chat")) if rt.get("chat") else _chat_stage(raw),
    )
    if rt.get("instructions"):
        out.instructions = rt["instructions"]
    if rt.get("voice"):
        out.voice = rt["voice"]
    if "asr_language" in rt:
        out.asr_language = str(rt["asr_language"])
    if "disable_thinking" in rt:
        out.disable_thinking = bool(rt["disable_thinking"])
    if "temperature" in rt:
        out.temperature = None if rt["temperature"] is None else float(rt["temperature"])
    if "max_tokens" in rt:
        out.max_tokens = int(rt["max_tokens"])
    out.vad = dict(rt.get("vad") or {})
    if "wake_word" in rt:
        out.wake_word = str(rt.get("wake_word") or "jarvis")
    if rt.get("wake_aliases"):
        out.wake_aliases = [str(a) for a in rt["wake_aliases"]]
    if "conversation_end_word" in rt:
        out.conversation_end_word = str(rt.get("conversation_end_word") or "over")
    if rt.get("conversation_end_aliases"):
        out.conversation_end_aliases = [str(a) for a in rt["conversation_end_aliases"]]
    if "conversation_max_segments" in rt:
        out.conversation_max_segments = int(rt["conversation_max_segments"])
    if "max_utterance_s" in rt:
        out.max_utterance_s = max(1, int(rt["max_utterance_s"]))
    # Live partial transcription block: {"partials": true, "interval_ms": 700,
    # "max_window_s": 20}. Flat keys are also accepted for convenience.
    tr = rt.get("transcription") or {}
    if "partials" in tr or "transcribe_partials" in rt:
        out.transcribe_partials = bool(tr.get("partials", rt.get("transcribe_partials")))
    if "interval_ms" in tr:
        out.partial_interval_ms = int(tr["interval_ms"])
    elif "partial_interval_ms" in rt:
        out.partial_interval_ms = int(rt["partial_interval_ms"])
    if "max_window_s" in tr:
        out.max_partial_window_s = int(tr["max_window_s"])
    elif "max_partial_window_s" in rt:
        out.max_partial_window_s = int(rt["max_partial_window_s"])
    # Env opt-in (ops / selftest) without editing config.json.
    env_partials = os.environ.get("SHAREDAGENTS_REALTIME_PARTIALS")
    if env_partials is not None:
        out.transcribe_partials = env_partials not in ("", "0", "false", "no")
    # Client-side wake-word block: {"enabled": true, "keyword": "hey_jarvis",
    # "threshold": 0.5, "preroll_ms": 1500, "cooldown_ms": 2000}.
    cw = rt.get("client_wake")
    if isinstance(cw, (bool, str)):
        out.client_wake = _as_bool(cw, out.client_wake)
    elif isinstance(cw, dict):
        if "enabled" in cw:
            out.client_wake = _as_bool(cw["enabled"], out.client_wake)
        if cw.get("keyword"):
            out.wake_keyword = str(cw["keyword"])
        if "threshold" in cw:
            out.wake_threshold = float(cw["threshold"])
        if "preroll_ms" in cw:
            out.wake_preroll_ms = int(cw["preroll_ms"])
        if "cooldown_ms" in cw:
            out.wake_cooldown_ms = int(cw["cooldown_ms"])
        if "vad_gate" in cw:
            out.wake_vad_gate = _as_bool(cw["vad_gate"], out.wake_vad_gate)
    if rt.get("agent"):
        out.agent = str(rt["agent"])
    if rt.get("gateway_url"):
        out.gateway_url = str(rt["gateway_url"])
    # Token precedence: an explicit realtime override wins — INCLUDING an
    # explicit empty string, which means "send no token" (don't leak the global
    # gateway secret to a different gateway_url). Only a MISSING key falls back
    # to the global gateway block.
    token_explicit = "gateway_token" in rt
    if token_explicit:
        out.gateway_token = str(rt.get("gateway_token") or "")
    else:
        out.gateway_token = str((raw.get("gateway") or {}).get("auth_token") or "")
    # Env overrides (ops / selftest): opt into agent mode without editing
    # config.json. An explicit empty SHAREDAGENTS_REALTIME_AGENT forces raw cascade.
    env_agent = os.environ.get("SHAREDAGENTS_REALTIME_AGENT")
    if env_agent is not None:
        out.agent = env_agent
    prior_url = out.gateway_url  # the config/default gateway the token belongs to
    env_gw = os.environ.get("SHAREDAGENTS_REALTIME_GATEWAY_URL")
    env_tok = os.environ.get("SHAREDAGENTS_REALTIME_GATEWAY_TOKEN")
    if env_gw:
        out.gateway_url = env_gw
        # Don't leak a globally-configured token to a DIFFERENT gateway: drop a
        # non-explicit (global-fallback) token only when the env points
        # somewhere ELSE. Pointing at the same gateway keeps it.
        if env_gw.rstrip("/") != prior_url.rstrip("/") and not token_explicit and env_tok is None:
            out.gateway_token = ""
    # An explicit gateway-token env always wins, with or without a URL override
    # (secured deployments may inject just the token for the default gateway).
    if env_tok is not None:
        out.gateway_token = env_tok
    return out
