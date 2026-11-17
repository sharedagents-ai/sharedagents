"""Spoken / TTS mode guidance for realtime voice turns only.

Realtime (main chat embed + dedicated voice page) drives Jarvis through the
gateway with ``voice_mode=true`` on the prompt submit. Guidance is
**turn-scoped** — never persisted onto the session — so typed chat / TUI /
Telegram on the same Jarvis session are unaffected.

A brief cleanup helper removes any legacy ``[VOICE-TTS]`` pinned facts left by
an earlier persistent-pin implementation.
"""

from __future__ import annotations

from typing import Any

# Sids currently executing a voice_mode turn (this process). Set around
# ``backend.prompt`` so preamble composition can inject TTS brevity without
# mutating durable session state.
_VOICE_TTS_TURNS: set[str] = set()

VOICE_TTS_PIN_PREFIX = "[VOICE-TTS]"

VOICE_TTS_GUIDANCE = (
    f"{VOICE_TTS_PIN_PREFIX} Spoken realtime turn — your reply will be read "
    "aloud via TTS. Keep this answer to 1–3 short, natural spoken sentences "
    "(roughly ≤80 words). Prefer a brief summary over detail; offer to expand "
    "only if the user asks. Never use markdown, bullet lists, code blocks, "
    "tables, URLs, or emoji. Plain speakable prose only. When delegating, "
    "acknowledge in one short spoken sentence and stay brief when relaying "
    "results."
)

# Back-compat alias used by older imports/tests.
VOICE_TTS_PIN = VOICE_TTS_GUIDANCE


def begin_voice_tts_turn(sid: str) -> None:
    """Mark ``sid`` as a spoken turn for the duration of ``backend.prompt``."""
    if sid:
        _VOICE_TTS_TURNS.add(sid)


def end_voice_tts_turn(sid: str) -> None:
    """Clear the spoken-turn mark (always pair with :func:`begin_voice_tts_turn`)."""
    if sid:
        _VOICE_TTS_TURNS.discard(sid)


def is_voice_tts_turn(sid: str | None) -> bool:
    """True while a realtime voice_mode turn is driving ``sid`` in-process."""
    return bool(sid) and sid in _VOICE_TTS_TURNS


def voice_tts_preamble_block() -> dict[str, str]:
    """User-role preamble block injected only for voice_mode turns."""
    return {"role": "user", "content": VOICE_TTS_GUIDANCE}


def strip_session_voice_tts_pin(agent: Any, sid: str) -> bool:
    """Remove a legacy durable ``[VOICE-TTS]`` pin if present. Returns True if removed."""
    if not sid or agent is None:
        return False
    getter = getattr(agent, "get_session", None)
    if not callable(getter):
        return False
    try:
        sess = getter(sid)
    except Exception:  # noqa: BLE001
        return False
    if sess is None:
        return False
    facts = [str(x) for x in (getattr(sess, "pinned_facts", None) or [])]
    cleaned = [f for f in facts if not f.startswith(VOICE_TTS_PIN_PREFIX)]
    if len(cleaned) == len(facts):
        return False
    sess.pinned_facts = cleaned
    # Also clear any in-memory engine attachment so the live turn sees it.
    try:
        eng = getattr(agent, "engine", None)
        live = getattr(eng, "_sessions", {}).get(sid) if eng is not None else None
        if live is not None and live is not sess:
            live.pinned_facts = [
                str(x)
                for x in (getattr(live, "pinned_facts", None) or [])
                if not str(x).startswith(VOICE_TTS_PIN_PREFIX)
            ]
    except Exception:  # noqa: BLE001
        pass
    sessions_dir = getattr(agent, "sessions_dir", None)
    if sessions_dir is None:
        return False
    try:
        from sharedagents_session.session_writer import get_session_writer

        get_session_writer(sessions_dir, sid).header_update(
            patch={"pinned_facts": cleaned},
        )
    except Exception:  # noqa: BLE001
        try:
            saver = getattr(sess, "save", None)
            if callable(saver):
                saver()
        except Exception:  # noqa: BLE001
            return False
    return True
