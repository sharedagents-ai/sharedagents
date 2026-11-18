"""Redact secrets from text destined for the session log.

The session JSON files at ``<agent_home>/sessions/<sid>.json`` are
the corpus operators feed back into fine-tune SFT pipelines, AWQ
calibration runs, and crash-debug forensics. They MUST NOT carry
credentials in the clear — once a session JSON has been generated,
it gets copied around: into training datasets, into shared
diagnostic archives, into research ingestion. A leaked OpenAI key,
GitHub PAT, AWS access key, or operator OAuth bearer there is a
real-world incident.

This module is the single chokepoint. Every text field that lands
on disk runs through :func:`scrub_text`; structured fields (dicts,
lists, JSON-serialisable values) run through :func:`scrub_value`,
which walks the tree and applies ``scrub_text`` to every string
leaf.

Design constraints:

* **Cheap.** The scrubber runs once per turn-save, of which there
  are several per turn (in-progress flushes). Pure regex over the
  text; no SAT solvers, no allocation-heavy string ops.
* **High recall over precision.** A false positive (over-redaction)
  is annoying but recoverable — operators can paste the redacted
  segment back from gateway.log if they need to debug an exact
  call. A false negative (a real credential makes it to disk) is
  an incident. So patterns are tuned to err on the side of redact.
* **No external dependencies.** stdlib ``re`` only. Easy to audit;
  easy to extend; no surprise transitive vuln surface.
* **Idempotent.** Running ``scrub_text(scrub_text(x))`` == ``scrub_text(x)``.
  Critical because the same string may transit multiple writes
  (in-progress flushes + final TurnDone).

Patterns are intentionally LIBERAL — when in doubt, redact. The
redaction sentinel ``<<REDACTED:KIND>>`` is unique enough that a
trainer / linter can spot whether a corpus has been scrubbed and
optionally exclude scrubbed rows from sensitive eval sets.

This module is the DETECTION registry only — the delegate-egress HARD BLOCK is
enforced by ``sharedagents_agent.delegation_gate``, which reuses ``scrub_text`` in
TWO places sharing this one registry: ``evaluate_preflight`` screens the
pre-turn call args at the delegation chokepoint
``delegate_shims._drive_child_agent_turn_inner``, and ``guard_egress_payload``
screens the FULLY-COMPOSED payload (system prompt + preamble + memory context +
prompt) at the true egress sink inside the child engine — ``external_cli`` (CLI
subprocess) and ``acp_engine`` (ACP session_prompt). This module stays
persistence-facing; ``delegation_gate`` is the enforcement point.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

# ---------------------------------------------------------------------
# Pattern registry. Each entry is ``(name, compiled_regex, replacement)``.
#
# Replacement uses the literal sentinel ``<<REDACTED:NAME>>`` so a
# downstream consumer can detect what was stripped without learning
# the value. ``NAME`` is the pattern key — keeping it stable makes
# diff-based audit of the corpus straightforward.
#
# Order matters within a pass: more specific patterns must match
# BEFORE the catch-all "looks-like-a-token" pattern, otherwise the
# catch-all redacts the value with a generic tag and loses the
# kind-of-secret signal.
# ---------------------------------------------------------------------


def _redact(name: str) -> str:
    return f"<<REDACTED:{name}>>"


# ---- the canonical sentinel grammar (ONE shared parser, codex r2/r4) --
#
# A COMPLETE, well-formed redaction sentinel (``<<REDACTED:KIND>>``
# closed with ``>>``) counts as "pre-existing" ONLY when its kind is one
# THIS scrubber can actually emit. ``_sub`` exempts a captured value
# from redaction ONLY when it is exactly one or more of these — a
# malformed opener (``<<REDACTED:`` without a canonical ``KIND>>``
# closure), a NON-canonical kind (arbitrary / mixed-case / colon-suffix
# / digit-suffix — i.e. a real credential wrapped as
# ``<<REDACTED:<the-value>>>``), or real material glued around a
# sentinel is credential material and must be redacted (fail closed).
# A ``[^>]+`` grammar would trust ANY kind and let a syntactically
# complete forged sentinel smuggle the real value through unchanged; a
# grammar that accepted ARBITRARY
# payload after the two dynamic prefixes (``UNKNOWN_TYPE:<identifier>``,
# ``OPAQUE_BYTES_<digits>``), so a context-detectable identifier or
# numeric secret riding a dynamic form would be exempted unchanged too —
# no rewrite, no finding, no egress block.
#
# The kind set is the scrubber's OWN emission surface, enumerated
# exactly: the registered pattern names below (``_PATTERNS``) plus the
# two FIXED dynamic kinds ``scrub_value`` produces — ``OPAQUE_BYTES``
# for non-UTF-8 bytes and ``UNKNOWN_TYPE`` for unknown objects. Both
# dynamic kinds carry NO payload: ``scrub_value`` never appends the
# byte length or ``type(value).__name__``, so there is no
# attacker-controllable suffix a credential could ride through the
# exemption (codex r4). This module's ``_sub`` and
# ``delegation_gate.screen_payload`` consume THIS SAME grammar via
# :func:`iter_complete_sentinels` — a sentinel-shaped run with a
# non-canonical kind (including the legacy payload-bearing emissions
# ``UNKNOWN_TYPE:<typename>`` / ``OPAQUE_BYTES_<n>`` from earlier
# rounds) is ordinary text everywhere.
_DYNAMIC_SENTINEL_KINDS = (
    "OPAQUE_BYTES",
    "UNKNOWN_TYPE",
)


_PATTERNS: list[tuple[str, re.Pattern[str], str]] = []


def _register(name: str, pattern: str, *, flags: int = 0) -> None:
    _PATTERNS.append((name, re.compile(pattern, flags), _redact(name)))


# ---- specific provider keys (highest priority — exact prefixes) -----
#
# OpenAI / Anthropic / xAI / Mistral / Cohere / OpenRouter / DeepSeek
# all converged on a ``<prefix>-<long_b64ish>`` shape. Catch them by
# their distinguishing prefix rather than a generic "long token".

# OpenAI: sk-… (classic) and sk-proj-… (project-scoped). Length
# varies (~32-72 chars after the prefix). Match the prefix and grab
# at least 20 token chars after it. Refuse ``sk-ant-`` so it doesn't
# steal matches that belong to the ANTHROPIC_KEY pattern below.
_register("OPENAI_KEY", r"\bsk-(?!ant-)(?:proj-)?[A-Za-z0-9_-]{20,}")

# Anthropic: sk-ant-api… or sk-ant-… (legacy). Same shape.
_register("ANTHROPIC_KEY", r"\bsk-ant-[A-Za-z0-9_-]{20,}")

# Google AI Studio / Gemini API: AIza followed by exactly 35
# token chars. No trailing ``\b`` — the value may end in ``-``,
# which is non-word, so a non-word→non-word transition into
# whitespace would not satisfy ``\b``. The exact ``{35}`` quantifier
# bounds the match.
_register("GOOGLE_API_KEY", r"\bAIza[A-Za-z0-9_-]{35}")

# GitHub PATs (classic + fine-grained + server-to-server + user-to-
# server + OAuth + refresh). All share the ghX_ prefix.
_register("GITHUB_TOKEN", r"\bgh[opusr]_[A-Za-z0-9]{36,}\b")

# AWS access key IDs (AKIA/ASIA/AGPA prefix + 16 base32). Secret
# access keys are 40 chars of base64 — caught by the password=
# context-prefix patterns below since they don't have a distinctive
# prefix to anchor on.
_register("AWS_ACCESS_KEY_ID", r"\b(?:AKIA|ASIA|AGPA|ANPA|ANVA|ASCA)[0-9A-Z]{16}\b")

# Slack tokens (bot / user / app / refresh).
_register("SLACK_TOKEN", r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")

# Stripe (live + test).
_register("STRIPE_KEY", r"\b(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{24,}\b")

# Telegram bot tokens: ``<bot_id>:<35-char>``. Used by sharedagents's
# own telegram service — catch operator copy/pastes.
_register("TELEGRAM_BOT_TOKEN", r"\b\d{8,12}:[A-Za-z0-9_-]{30,}\b")

# Discord bot tokens (Nlong.Bshort.Csigned).
_register("DISCORD_TOKEN", r"\b[A-Za-z0-9_-]{24,28}\.[A-Za-z0-9_-]{6,7}\.[A-Za-z0-9_-]{27,}\b")

# JWT — three base64url segments joined by dots, header begins with
# the alg=HS256/RS256/ES256/etc. eyJ prefix.
_register("JWT", r"\beyJ[A-Za-z0-9_-]+?\.[A-Za-z0-9_-]+?\.[A-Za-z0-9_-]+\b")

# HashiCorp Vault tokens — modern: hvs.<base62> / legacy:
# s.<base62>. The ``s.`` prefix is short enough that we anchor by
# requiring at least 24 chars of value to avoid false positives on
# everyday "s. plus identifier" prose.
_register("VAULT_TOKEN", r"\bhvs\.[A-Za-z0-9_-]{20,}\b")
_register("VAULT_LEGACY_TOKEN", r"\bs\.[A-Za-z0-9]{24,}\b")

# npm registry auth tokens (``//registry.npmjs.org/:_authToken=…``
# and ``_password=…``). The leading underscore breaks the
# ``\btoken`` boundary, so the generic CREDENTIAL pattern misses
# this idiom in .npmrc / npm config files.
_register(
    "NPM_AUTH_TOKEN",
    r"(?i)//[A-Za-z0-9._/-]+:_authtoken\s*=\s*([A-Za-z0-9_-]{8,})",
)

# Docker config (``~/.docker/config.json``) embeds basic-auth as
# ``"auth": "<base64>"``. The keyword ``auth`` alone isn't in the
# generic CREDENTIAL list (too noisy — "auth" appears in benign
# prose), so match it only in the JSON-string idiom.
_register(
    "DOCKER_AUTH",
    r'"auth"\s*:\s*"([A-Za-z0-9+/=]{12,})"',
)

# Private keys (PEM blocks). DOTALL so ``-----BEGIN … END-----``
# spanning multiple lines is captured in one match.
_register(
    "PRIVATE_KEY_BLOCK",
    r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED |PGP |)PRIVATE KEY"
    r"(?: BLOCK)?-----.*?-----END (?:RSA |EC |OPENSSH |DSA |ENCRYPTED |PGP |)"
    r"PRIVATE KEY(?: BLOCK)?-----",
    flags=re.DOTALL,
)

# SSH key fingerprints / passphrases are NOT redacted (they're not
# secrets); the PEM block above catches the keys themselves.

# ---- context-prefix patterns (kw=value, kw: value) ------------------
#
# Operator scripts and Bash tool calls often paste secrets as
# ``PASSWORD=hunter2`` / ``TOKEN: abc123`` / ``--api-key foo`` /
# ``Authorization: Bearer eyJ…``. Match the keyword, the separator,
# and grab a generous run of value chars until whitespace or shell
# metacharacter (quotes, ``$`` backtick ``& | ;``). ``<`` and ``>``
# deliberately STAY in the value class: ``<`` keeps recall for real
# secrets containing it, and ``>`` keeps a sentinel closure plus any
# trailing material (``<<REDACTED:CREDENTIAL>>>>hunter2``) inside the
# capture so ``_sub`` can screen it instead of letting it leak past the
# match.
#
# These run AFTER the provider-key patterns above so an exact key
# format gets the right ``NAME`` instead of the generic
# ``CREDENTIAL`` tag.

_CREDENTIAL_KEYWORDS = (
    "password",
    "passwd",
    "secret",
    "token",
    "apikey",
    "api_key",
    "api-key",
    "auth_token",
    "auth-token",
    "access_token",
    "access-token",
    "refresh_token",
    "refresh-token",
    "bearer",
    "private_key",
    "private-key",
    "session_token",
    "session-token",
    "credentials",
    "client_secret",
    "client-secret",
)
_KW = "|".join(re.escape(k) for k in _CREDENTIAL_KEYWORDS)
# The value capture must stop BEFORE an escaped quote. Scrubbers run over
# text that can be a JSON document (session events, delegation digests),
# where `\"` is an escaped quote, not a delimiter: a greedy value class
# swallowed the escape AND the closing quote (`...secret:hunter2\"` ->
# `...secret:<<REDACTED:CREDENTIAL>>`), leaving invalid JSON that no longer
# reloads or parses. The non-greedy capture plus lookahead stops at the
# optional `\` + quote boundary, so the escape survives; a value that
# genuinely ends in a backslash is still captured whole (the lookahead only
# stops before a QUOTE). Recall is unchanged.
_VALUE = r"([^'\"\s$`&|;]{6,}?)(?=\\?['\"]|[\s$`&|;]|$)"
# Form: `kw = value` or `kw: value` or `kw "value"` etc.
_register(
    "CREDENTIAL",
    rf"(?i)\b(?:{_KW})\s*[:=]\s*['\"]?{_VALUE}['\"]?",
)
# Form: `--api-key foo` / `--token=foo` / `-p somepassword` (very
# common in Bash tool calls).
_register(
    "CLI_FLAG_CREDENTIAL",
    rf"(?i)\B--(?:{_KW})(?:[= ])\s*['\"]?{_VALUE}['\"]?",
)
# Form: `Authorization: Bearer <token>` (HTTP header style).
_register(
    "BEARER_AUTH",
    r"(?i)\bauthorization\s*:\s*bearer\s+[A-Za-z0-9._\-+/=]{16,}",
)

# Form: an env-style line `EXPORT KEY=value` or `KEY=value` where
# KEY ends in known credential suffixes. More specific than the
# generic CREDENTIAL above because it requires SHOUTY_SNAKE_CASE
# (the env-var idiom).
_register(
    "ENV_CREDENTIAL",
    r"\b(?:[A-Z][A-Z0-9_]*_(?:KEY|TOKEN|SECRET|PASSWORD|PWD))\s*=\s*['\"]?"
    rf"{_VALUE}['\"]?",
)


# The canonical sentinel regex, built AFTER the registry is populated
# (patterns are registered at import time only — nothing registers a
# pattern at runtime; ``_PATTERNS`` stays module-private). This is the
# SINGLE source of truth for what counts as a redaction sentinel,
# shared with ``delegation_gate.screen_payload`` through
# :func:`iter_complete_sentinels`. The dynamic kinds are FIXED strings
# (``OPAQUE_BYTES`` / ``UNKNOWN_TYPE``) with no payload capture of any
# kind — no colon-suffix, no digit-suffix — so ``<<REDACTED:KIND>>``
# matches ONLY when the kind is exactly one of the 19 registered
# pattern names or one of the two fixed dynamic strings.
_CANONICAL_SENTINEL_RE = re.compile(
    r"<<REDACTED:("
    + "|".join(re.escape(k) for k in list(_DYNAMIC_SENTINEL_KINDS) + sorted(name for name, _, _ in _PATTERNS))
    + r")>>"
)


# Base64 media payloads carried in a ``data:`` URI (Telegram photo,
# Open WebUI paste → ``image_url`` part on the persisted turn). These are
# binary pixels, NEVER text credentials, yet a large blob incidentally
# contains secret-SHAPED substrings (e.g. a ``+AIza<35 alnum>`` run that
# matches GOOGLE_API_KEY). Redacting inside the blob SHORTENS the base64;
# the reloaded session then replays a truncated image and the vision
# upstream 400s with "Failed to load image: Truncated File Read", ending
# the turn. So we mask the base64
# span out before scanning and restore it verbatim afterwards. Only the
# base64 BODY is protected — a real credential sitting outside a data:
# URI is still redacted.
#
# §4) ratifies this exemption as constitutional, not an oversight — it is
# the scrubber's *only* content-class exemption. The residual risk (a real
# credential surviving inside an image blob) was weighed by the Round 3
# review and judged negligible against the demonstrated regression above:
# a credential embedded in media bytes isn't the exfiltration shape the
# secret gate defends against, since it never renders as a copyable
# plaintext token. Do not "fix" this by scanning inside the base64 body —
# that reintroduces the b7ae4e54/#425 truncation regression.
_DATA_URI_B64 = re.compile(r"data:[^\s;,]*;base64,[A-Za-z0-9+/=]+")
# Sentinel placeholder for a stashed span. NUL bytes never appear in the
# JSON-text leaves we scrub and match none of the registered patterns,
# so a masked span is inert during the scan.
_MASK_OPEN = "\x00DATAURI:"
_MASK_CLOSE = "\x00"

# A diagnostic such as ``/verify-token: Missing`` contains the word ``token``
# in a URL/path segment, not a credential assignment.  The generic
# ``CREDENTIAL`` pattern intentionally accepts punctuation before a keyword so
# it can catch JSON and shell forms; keep that recall, but do not let a path
# segment turn a harmless status word into a redaction.  Query strings remain
# eligible for screening because ``?token=...`` is a real credential-egress
# shape.
#
# Closed-set membership is the first line of defense.  Title-Case alphabetic
# words (``Failed``, ``Timeout``, ``Unauthorized``) in a path are HTTP-status
# labels, not tokens — requiring Title Case keeps ``secretpassword`` fail-closed.
_PATH_SEGMENT_SPLIT = re.compile(r"[\s\"'`<>(){}\[\]]")
_PATH_TITLE_STATUS = re.compile(r"^[A-Z][a-z]+(?:[-_][A-Z]?[a-z]+)?$")
_PATH_STATUS_MAX_LEN = 16
_PATH_DIAGNOSTIC_VALUES = frozenset(
    {
        "absent",
        "blocked",
        "canceled",
        "cancelled",
        "denied",
        "disabled",
        "empty",
        "error",
        "expired",
        "failed",
        "forbidden",
        "gone",
        "invalid",
        "missing",
        "none",
        "not-found",
        "not_found",
        "notfound",
        "null",
        "refused",
        "rejected",
        "revoked",
        "timed-out",
        "timed_out",
        "timedout",
        "timeout",
        "unauthenticated",
        "unauthorized",
        "unavailable",
        "unknown",
        "unset",
    }
)


def _is_path_segment_credential(match: re.Match[str]) -> bool:
    """Whether a generic match is a benign status diagnostic, not a secret."""
    raw = match.group(1).strip("'\"").rstrip(",:;.!?")
    # ``token: Missing`` in prose (no URL path) is the same diagnostic as
    # ``/verify-token: Missing``. Status words are never credential material.
    if raw.lower() in _PATH_DIAGNOSTIC_VALUES:
        return True
    prefix = match.string[: match.start()]
    segment = _PATH_SEGMENT_SPLIT.split(prefix)[-1]
    if not segment or "/" not in segment:
        return False
    # Do not exempt query-string or fragment values: those can carry a real
    # token even when the URL also contains a path.
    if "?" in segment or "#" in segment:
        return False
    if len(raw) > _PATH_STATUS_MAX_LEN or any(ch.isdigit() for ch in raw):
        return False
    return bool(_PATH_TITLE_STATUS.fullmatch(raw))


# ---- the public API ------------------------------------------------


def scrub_text(text: str) -> str:
    """Apply every registered pattern to ``text`` and return the
    redacted string. Idempotent: a captured value that is already a
    COMPLETE redaction sentinel with a CANONICAL kind
    (``<<REDACTED:…>>`` — a kind this scrubber emits, see
    :func:`iter_complete_sentinels`) is left untouched by ``_sub``;
    anything that is not exactly a complete canonical sentinel is
    redacted (fail closed — malformed ``<<REDACTED:`` openers, forged
    non-canonical kinds, or real material glued around a sentinel
    never pass unscreened).

    Base64 ``data:`` URI media payloads are preserved byte-for-byte —
    see :data:`_DATA_URI_B64` for why redacting inside them corrupts
    replayed images.

    Returns ``text`` unchanged when it is not a string. The empty
    string and ``None``-shaped inputs short-circuit immediately so
    the hot session-save path doesn't waste cycles.
    """
    if not isinstance(text, str) or not text:
        return text
    # Stash every data-URI base64 span behind an inert NUL-delimited
    # placeholder so the secret patterns can't fire inside it.
    stashed: list[str] = []

    def _stash(m: re.Match[str]) -> str:
        stashed.append(m.group(0))
        return f"{_MASK_OPEN}{len(stashed) - 1}{_MASK_CLOSE}"

    out = _DATA_URI_B64.sub(_stash, text) if "base64," in text else text
    for name, pattern, replacement in _PATTERNS:
        # For patterns that capture a single group, replace only the
        # group span — that keeps the surrounding context (`PASSWORD=`,
        # `--token`, etc.) visible to the reader so they know the
        # SHAPE of what was redacted. For patterns without groups
        # (raw token shapes like sk-…) replace the whole match.
        if pattern.groups == 1:

            def _sub(m: re.Match[str], _r: str = replacement) -> str:
                whole = m.group(0)
                grp = m.group(1)
                if name == "CREDENTIAL" and _is_path_segment_credential(m):
                    return whole
                # Idempotence: a captured value that is EXACTLY one or
                # more COMPLETE, CANONICAL sentinels
                # (``<<REDACTED:KIND>>`` where KIND is a kind this
                # scrubber emits — a registered pattern name or one of
                # the two FIXED dynamic strings ``OPAQUE_BYTES`` /
                # ``UNKNOWN_TYPE``, see ``_CANONICAL_SENTINEL_RE``)
                # from a prior scrub pass is left alone. ANYTHING else —
                # a malformed opener without a ``KIND>>`` closure, a
                # forged sentinel whose kind is NOT canonical
                # (arbitrary / mixed-case / colon-suffix / digit-suffix
                # — including the payload-bearing dynamic forms
                # ``UNKNOWN_TYPE:<id>`` and ``OPAQUE_BYTES_<digits>``
                # and the legacy round-1..3 emissions
                # ``UNKNOWN_TYPE:<typename>`` / ``OPAQUE_BYTES_<n>``),
                # real material glued before/after a sentinel
                # (``password=<<REDACTED:CREDENTIAL>>>>hunter2``),
                # or a plain secret — is redacted IN FULL (fail
                # closed). Looser rules — startswith/"<6-prefix"
                # exemptions, a ``[^>]+`` any-kind grammar, or a
                # payload-bearing dynamic grammar — all let real
                # secrets through unscreened:
                # ``screen_payload`` returned () and
                # ``guard_egress_payload`` allowed egress
                if not _CANONICAL_SENTINEL_RE.sub("", grp):
                    return whole
                start = m.start(1) - m.start()  # offset of capture group within whole match
                end = m.end(1) - m.start()
                return whole[:start] + _r + whole[end:]

            out = pattern.sub(_sub, out)
        else:
            out = pattern.sub(replacement, out)
    # Restore the stashed base64 media payloads verbatim.
    for i, span in enumerate(stashed):
        out = out.replace(f"{_MASK_OPEN}{i}{_MASK_CLOSE}", span)
    return out


def scrub_value(value: Any) -> Any:
    """Recursively scrub a JSON-serialisable value.

    * ``str``  → :func:`scrub_text` applied.
    * ``dict`` → keys passed through unchanged; values scrubbed.
    * ``list`` / ``tuple`` → elementwise scrub (returns a ``list``
      because the persisted shape is always JSON-array).
    * Anything else (int, float, bool, None, custom objects) →
      returned unchanged.

    Returns a NEW container; the input is never mutated. That
    matters because the in-memory ``tool_calls`` list also feeds
    the live model context — mutating it under the model would
    poison subsequent rounds.
    """
    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, dict):
        return {k: scrub_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub_value(v) for v in value]
    # ``bytes`` may carry a credential too (e.g. a copy-pasted PEM
    # block that hit a bytes-handling code path). Decode STRICTLY and
    # scrub the text; bytes that do NOT cleanly decode as UTF-8 must
    # NOT be lossy-decoded (a ``errors="replace"`` mojibake could carry
    # partial/malformed credential material to disk under-redacted) —
    # they redact to an opaque, sized sentinel instead.
    # ``set`` / ``frozenset`` get converted to a
    # sorted list so the JSON serializer downstream can persist them;
    # the original container type was already destined to fail
    # ``json.dumps``, so this is a strict upgrade. Anything else
    # (numpy / custom object / etc.) is replaced with a sentinel —
    # silent pass-through would let an engine that passes such a value
    # at some future point leak its repr to disk un-scrubbed.
    if isinstance(value, (bytes, bytearray)):
        try:
            decoded = bytes(value).decode("utf-8")
        except (UnicodeDecodeError, ValueError):
            # FIXED kind — no length payload. A sized sentinel would
            # let an attacker-controlled digit run ride the exemption;
            # the kind alone is enough for
            # audit.
            return "<<REDACTED:OPAQUE_BYTES>>"
        return scrub_text(decoded)
    if isinstance(value, (set, frozenset)):
        return [scrub_value(v) for v in sorted(value, key=repr)]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    # Unknown type — produce a FIXED sentinel rather than a
    # silently-stringified repr that might leak unscrubbed content.
    # No ``type(value).__name__`` payload: type names are
    # attacker-observable strings (and need not even be Python
    # identifiers — the ``type`` constructor accepts any string), so
    # carrying one verbatim would keep a secret-smuggling channel open.
    return "<<REDACTED:UNKNOWN_TYPE>>"


def list_pattern_names() -> Iterable[str]:
    """Return the registered pattern names (for tests + audit)."""
    return tuple(name for name, _, _ in _PATTERNS)


def iter_complete_sentinels(text: str) -> tuple[str, ...]:
    """Kinds of every COMPLETE canonical redaction sentinel in ``text``.

    THE single shared sentinel parser:
    what counts as a pre-existing redaction sentinel is exactly the kind
    set this scrubber can emit — the registered pattern names plus the
    two FIXED dynamic ``scrub_value`` kinds (``OPAQUE_BYTES``,
    ``UNKNOWN_TYPE`` — no payload of any kind: no length suffix, no
    type-name suffix). ``scrub_text``'s idempotence exemption (``_sub``)
    and ``delegation_gate.screen_payload``'s pre-existing-sentinel
    tracking both consume THIS SAME grammar, so a sentinel-shaped run
    with a non-canonical kind (arbitrary, mixed-case, colon-suffix,
    digit-suffix — including ``UNKNOWN_TYPE:<id>``,
    ``OPAQUE_BYTES_<digits>``, and the legacy round-1..3 emissions) is
    ordinary text everywhere: it is redacted, reported as a finding,
    and hard-blocked at egress (fail closed).

    Returns the kinds in order of appearance; duplicates included
    (callers build a multiset Counter to diff pre/post scrub).
    """
    if not isinstance(text, str) or not text:
        return ()
    return tuple(m.group(1) for m in _CANONICAL_SENTINEL_RE.finditer(text))


def list_sentinel_kinds() -> tuple[str, ...]:
    """The exact kind set the scrubber can emit — the canonical sentinel kinds.

    Registered pattern names plus the two FIXED dynamic ``scrub_value``
    kinds (``OPAQUE_BYTES`` / ``UNKNOWN_TYPE``). This is the grammar
    :func:`iter_complete_sentinels` / ``_CANONICAL_SENTINEL_RE`` accept
    (for audit and tests): any other ``<<REDACTED:…>>`` kind —
    including every payload-bearing dynamic variant — is non-canonical
    and is treated as ordinary text by both the scrubber and the
    egress gate.
    """
    return tuple(_DYNAMIC_SENTINEL_KINDS) + tuple(sorted(name for name, _, _ in _PATTERNS))


__all__ = (
    "scrub_text",
    "scrub_value",
    "list_pattern_names",
    "iter_complete_sentinels",
    "list_sentinel_kinds",
)
