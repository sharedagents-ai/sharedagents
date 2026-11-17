"""Shared ASR transcript guards for every voice path (dependency-free: ``re``).

Lives in :mod:`sharedagents_core` because three callers need the same verdicts and
must not each carry — and slowly drift apart with — a private copy of the rules:

* ``sharedagents_realtime.session`` — interactive voice turns (final + partials),
* ``sharedagents_realtime.meeting`` — passive meeting transcription,
* ``sharedagents_web.routes.voice`` — the WebUI push-to-talk ``/api/transcribe``
  proxy.

``sharedagents_core`` is the leaf package both sides already depend on, which keeps
the dependency arrow one-way (both sides -> core). It is NOT about an import
cycle: ``sharedagents_realtime.server`` does import ``sharedagents_web``, but
``sharedagents_realtime.noise`` — the module that re-exports the names below —
imports only this module (verified: neither ``sharedagents_realtime.server`` nor
``sharedagents_web`` lands in ``sys.modules``), so a ``sharedagents_web ->
sharedagents_realtime.noise`` import would not cycle; core is simply the right home
for a rule that belongs to no single caller. This module imports nothing but
``re`` on purpose, so the voice/meeting paths never pull the STT/LLM/TTS cascade
(and its network client) into a transcription guard.

:mod:`sharedagents_realtime.noise` re-exports the names below for its existing
callers and tests.
"""

from __future__ import annotations

import re

# Character-class body shared by the detection class (_CJK) and the stripping
# class (_CJK_OR_PUNCT) so both always agree on what "CJK" means.
#
# Han, kana and Hangul ONLY. The U+FF00-U+FFEF "Halfwidth and Fullwidth Forms"
# block used to sit in here as a whole, which silently deleted legitimate en/pt
# content: that block carries the fullwidth Latin letters/digits (``Ｆｏｏ``,
# ``１２３``) and fullwidth punctuation (``，``, ``！``) a transcript may really
# contain, so whole results came back emptied. Only its genuinely CJK
# sub-ranges belong here — halfwidth katakana and halfwidth Hangul jamo.
_CJK_RANGES = (
    "぀-ヿ"  # Hiragana + Katakana
    "㐀-鿿"  # CJK Unified Ideographs + Extension A
    "\uff61-\uff9f"  # halfwidth katakana (inside the fullwidth-forms block)
    "\uffa0-\uffdc"  # halfwidth Hangul jamo (ditto)
    "\u1100-\u11ff\u3130-\u318f\ua960-\ua97f"  # Hangul jamo/compat jamo/jamo ext-A
    "\uac00-\ud7af\ud7b0-\ud7ff"  # Hangul syllables/jamo ext-B
)
_CJK = re.compile("[" + _CJK_RANGES + "]")
#: The detection class PLUS CJK symbols/punctuation (U+3000-U+303F: ``。``
#: ``、`` ``「」`` ``〜``). Punctuation is not part of "is this whole utterance a
#: hallucination" (_CJK), but it must go when a mixed artifact is stripped, or
#: ``"Hello 嗯。"`` would leave ``"Hello 。"`` behind. Unlike the fullwidth
#: forms, this block is CJK-only punctuation no en/pt transcript uses, so
#: stripping it cannot drop content.
_CJK_OR_PUNCT = re.compile("[" + _CJK_RANGES + "\u3000-\u303f]")
#: A NON-CJK LETTER — the test that decides whether a CJK-bearing utterance also
#: carries real (supported-language) letter content. Built from the SAME range
#: body as :data:`_CJK` so the two can never disagree, and Unicode-wide rather
#: than ASCII-only: ``"Ｆｏｏ"`` is fullwidth LATIN content, so a transcript that
#: mixes it with a CJK artifact must not be mistaken for a CJK-only
#: hallucination (the ASCII-only class it replaces called ``"Ｆｏｏ 嗯。"`` noise
#: and would have discarded the whole result). ``[^\W\d_]`` = word characters
#: minus digits and ``_`` = letters, any script.
_NON_CJK_LETTER = re.compile("[^" + _CJK_RANGES + r"\W\d_]")

#: A one-code-point transcript counts as content only when that code point is a
#: letter or a digit: one-character Portuguese dictations (``é``, ``á``) and a
#: lone number (``7``) are real ASR results, while a lone symbol/emoji (``$``,
#: ``😀``) is not. ``[^\W_]`` = Unicode word characters minus the underscore
#: (letters + digits, any script). CJK is rejected before this check, so a lone
#: ``"嗯"`` is still noise.
_LETTER_OR_DIGIT = re.compile(r"[^\W_]")

#: Characters the whole-utterance checks ignore. Members are identical to the
#: class this guard has always used: ASCII/Unicode whitespace (``\s``),
#: ASCII + fullwidth + CJK punctuation, quotes and brackets. Assembled from
#: pieces so the double quote needs no backslash escape in this source file.
_FILLER_CHARS = r"\s\.,!?;:…。，！？、~\-" + '"' + r"'()\[\]"
_FILLER_STRIP = re.compile("[" + _FILLER_CHARS + "]+")

#: The only ASR languages the voice pipelines support (English and
#: Portuguese). Any other value is an invalid/unsupported config and must
#: never be able to disable the Latin-only non-speech guard.
SUPPORTED_LANGUAGES = frozenset({"en", "pt"})


def parse_languages(spec: str) -> list[str]:
    """``"en,pt"`` → ``["en","pt"]``; only :data:`SUPPORTED_LANGUAGES` survive.

    Unsupported entries (``zh``/``ja``/``ko``/…) are dropped like ``auto``/empty,
    so an invalid config never reaches the ASR language hint or reply-language
    detection. ``"auto"``/``""`` → ``[]``.
    """
    out = []
    for part in (spec or "").split(","):
        p = part.strip().lower()
        if p in SUPPORTED_LANGUAGES:
            out.append(p)
    return out


def looks_like_noise(text: str, language: str = "en") -> bool:
    """True when an ASR transcript is almost certainly a non-speech artifact.

    Catches the classic failure modes on coughs/breaths/room-tone: empty, a lone
    filler character, a lone symbol/emoji, or a hallucinated CJK filler like
    ``喵。可。`` with no letter content. Every supported language
    (:data:`SUPPORTED_LANGUAGES`) is Latin-script, so a CJK-only transcript
    (Han, kana or Hangul, with no real letter in it) is ALWAYS a hallucination:
    the ``language`` spec (kept for callers) can never disable this check — not
    an unsupported ``zh``/``ja``/``ko`` entry, ``auto`` or empty. "Letter
    content" is Unicode-wide (:data:`_NON_CJK_LETTER`), so fullwidth Latin
    (``"Ｆｏｏ 嗯。"``) counts as content and is not a CJK-only artifact.

    A one-character transcript is judged by WHAT that character is
    (:data:`_LETTER_OR_DIGIT`), not by its length alone: ``"é"``/``"á"`` (a
    one-character Portuguese dictation) and ``"7"`` are legitimate results and
    pass, while a lone ``"😀"``/``"$"`` does not.

    This is the WHOLE-UTTERANCE verdict, so a MIXED transcript that does carry
    real speech (``"Hello 嗯。"``) is deliberately not noise here — dropping it
    would throw the speech away with the artifact. Callers that surface or
    persist a transcript run :func:`sanitize_transcript` (or
    :func:`strip_cjk_artifacts`) first, which removes the hallucinated CJK and
    leaves ``"Hello"``; only a transcript with nothing usable left is dropped.
    """
    core = _FILLER_STRIP.sub("", text)
    if not core:
        return True
    if _CJK.search(core) and not _NON_CJK_LETTER.search(core):
        return True
    return bool(len(core) == 1 and not _LETTER_OR_DIGIT.match(core))


def strip_cjk_artifacts(text: str) -> str:
    """Remove hallucinated CJK code points from an ASR transcript.

    Every supported ASR language is Latin-script (:data:`SUPPORTED_LANGUAGES`),
    so Han/kana/Hangul and CJK punctuation can never be real content and
    stripping them cannot lose speech — but unlike :func:`looks_like_noise` this
    keeps the legitimate part of a MIXED transcript instead of discarding the
    whole string:

        ``"Hello 嗯。"`` → ``"Hello"`` (not ``""``)
        ``"嗯。"`` / ``"喵。可。"`` → ``""`` (nothing usable left)

    Deliberately narrow (see :data:`_CJK_RANGES`): the fullwidth Latin
    letters/digits (``Ｆｏｏ``, ``１２３``) and fullwidth punctuation (``，``,
    ``！``) of the Halfwidth and Fullwidth Forms block are presentation forms of
    legitimate en/pt text and are NOT stripped — removing them deleted whole
    transcripts.

    Text without a single CJK code point is returned byte-for-byte unchanged —
    no whitespace rewriting — so an ordinary English/Portuguese transcript is
    never touched.
    """
    if not text or not _CJK_OR_PUNCT.search(text):
        return text
    cleaned = _CJK_OR_PUNCT.sub("", text)
    # Tidy what the removed run left behind: collapse whitespace runs and drop a
    # space stranded in front of surviving Latin punctuation.
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return cleaned.strip()


def sanitize_transcript(text: str, language: str = "en") -> str:
    """Single entry point for a transcript a caller will surface or persist.

    Strips CJK hallucinations (keeping the Latin part of a mixed transcript)
    and returns ``""`` when nothing usable remains — empty input, a lone filler
    character, punctuation/whitespace only, a lone symbol or emoji, or a
    CJK-only hallucination such as the classic silent-audio ``"嗯。"``. A
    one-character LETTER OR DIGIT is content and survives (``"é"``, ``"á"``,
    ``"7"`` — see :func:`looks_like_noise`). :func:`looks_like_noise` keeps its
    whole-utterance meaning and is applied here to the *cleaned* text.
    """
    cleaned = strip_cjk_artifacts((text or "").strip())
    if not cleaned or looks_like_noise(cleaned, language):
        return ""
    return cleaned
