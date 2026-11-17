"""Re-export of the shared ASR text guards (see :mod:`sharedagents_core.asr_text`).

The guards moved to :mod:`sharedagents_core.asr_text` so the WebUI transcribe route
(``sharedagents_web.routes.voice``) can apply exactly the same non-speech filtering
as the voice paths: ONE copy of the rules for all three callers instead of a
second implementation that drifts from this one. Core is the leaf package both
sides already contain, so the dependency arrow stays one-way (both -> core).

Not because of an import cycle. ``sharedagents_realtime.server`` does import
``sharedagents_web``, but importing this module pulls in ``sharedagents_core.asr_text``
and nothing else — neither ``sharedagents_realtime.server`` nor the STT/LLM/TTS
cascade — so a ``sharedagents_web -> sharedagents_realtime.noise`` import would not in
fact cycle. It would still be the wrong home: the guard belongs to no single
caller.

This module keeps its name (and its dependency-free promise: ``sharedagents_core.
asr_text`` imports only ``re``, never the STT/LLM/TTS cascade or its network
client) for the existing callers — ``sharedagents_realtime.session`` and
``sharedagents_realtime.meeting`` — and for the tests that import from here.
"""

from __future__ import annotations

from sharedagents_core.asr_text import (
    SUPPORTED_LANGUAGES,
    looks_like_noise,
    parse_languages,
    sanitize_transcript,
    strip_cjk_artifacts,
)

__all__ = [
    "SUPPORTED_LANGUAGES",
    "looks_like_noise",
    "parse_languages",
    "sanitize_transcript",
    "strip_cjk_artifacts",
]
