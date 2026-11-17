"""Pre-flight delegation gate — the single shared enforcement point for
expensive (third-party / frontier) delegate calls.

This module is the ONE mechanism that decides whether an expensive delegate
turn may be driven. It is called from exactly one authoritative call site,
:func:`sharedagents_agent.agents.delegate_shims._drive_child_agent_turn_inner`,
immediately after ``factory.load_agent`` resolves the child's engine and
before any child session is created or ``run_turn`` fires — the single
convergence point every delegation route funnels through (the four
``shim_run_delegate_*`` shims and the detached background runner). One gate,
no bypass.

It enforces two properties for delegate egress:

* **Checklist gate** — an *expensive* delegate call whose pre-flight
  checklist (goal, constraints, acceptance criteria, budget) is incomplete is
  BLOCKED by tooling, not merely prompted for. Enforcement is one shared
  mechanism identical for every delegating agent.
* **Secret hard-block** — a secret detected in the outbound payload sent to a
  third-party delegate HARD-BLOCKS the call. There is no override parameter in
  this gate's interface (secrets have no override). This is
  enforced in TWO places that share one detection registry (``scrub_text``):
  (1) :func:`evaluate_preflight` screens the pre-turn CALL ARGS here — including
  non-string ``prompt``/``task``/``context`` values, which are deep-serialized
  before screening so ``prompt={"token": "..."}`` cannot slip past; and
  (2) :func:`guard_egress_payload` screens the FULLY-COMPOSED payload at the
  true egress sink (external CLI subprocess / ACP session), because the child
  engine assembles system prompt + preamble + AGENTS.md/MEMORY/pinned-facts/
  todos/skill/recall context AFTER this gate runs. The args-screen alone cannot
  see that composed context; the sink guard is where all of it converges.

"Expensive" / "third-party" is decided by the child's RESOLVED
:class:`~sharedagents_agent.agents.engine.EngineKind` (spec/delegation.md §4.2), not
by ``subagent_type`` — a sibling agent scaffolded with an external engine is
gated identically to a bare ``subagent_type="claude"``, closing the alias hole.
Local in-process subagents (``local_llm``) are the cheap side of the arbitrage
and are never gated, scrubbed, or slowed (requirements/02-delegation.md §1).

Scope note: this module deliberately implements ONLY the checklist-presence
gate and the secret hard-block. Tiered budget caps, spend reporting, the
routing-authority ``needs_owner`` round-trip, and the Agent-tool schema wiring
are separate concerns and are out of scope here; the
``PreflightResult`` shape leaves room for them without a redesign.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from sharedagents_core.secrets_scrubber import iter_complete_sentinels, scrub_text

# Engine kinds whose inference runs OUTSIDE the local vLLM boundary — the
# "expensive" / third-party set (spec/delegation.md §4.2, requirements/02 §1).
# ``*_cli`` are frontier APIs by construction; the ACP kinds front a frontier
# model by default (Cursor is frontier; OpenCode/generic-ACP usually wrap one).
# ``local_llm`` is the cheap side and is intentionally absent.
_EXPENSIVE_ENGINE_KINDS = frozenset(
    {
        "claude_cli",
        "codex_cli",
        "antigravity_cli",
        "zcode_cli",
        "acp",
        "cursor_acp",
        "opencode_acp",
        "kimi_acp",
        "hermes_acp",
        "devin_acp",
        "pi_rpc",
    }
)

# The four pre-flight checklist fields (spec/delegation.md §4.1). Carried as
# structured call args so presence is machine-checkable without NLP.
_CHECKLIST_FIELDS = ("goal", "constraints", "acceptance_criteria", "budget_usd")

# Trivial placeholder values that do NOT satisfy a prose checklist field —
# presence enforcement rejects an empty-after-strip or throwaway token
# (spec §4.1: "non-empty after strip; not a placeholder like 'n/a'").
_PLACEHOLDERS = frozenset({"n/a", "na", "none", "tbd", "todo", "-", "?", ".", "x"})

# Keys whose values leave the local boundary and must be secret-screened
# (the materialized prompt/context plus the prose checklist fields). NOTE the
# value is NOT assumed to be a string: ``_outbound_payload`` deep-serializes a
# non-string value (dict / list) before screening, because ``delegate_shims``
# stringifies whatever it finds here (``str(args.get("prompt"))``) and sends
# THAT — so a secret nested in ``prompt={"token": "..."}`` must be caught too.
_OUTBOUND_KEYS = (
    "prompt",
    "task",
    "message",
    "input",
    "context",
    "goal",
    "constraints",
    "acceptance_criteria",
)

# NOTE: there is deliberately NO sentinel regex of this module's own.
# What counts as a pre-existing ``<<REDACTED:…>>`` sentinel is decided
# by the ONE shared canonical parser in secrets_scrubber
# (``iter_complete_sentinels``), whose kind set is exactly what the
# scrubber can emit (registered pattern names + the ``scrub_value``
# dynamic kinds). The round-2 gate grammar ``[A-Z0-9_]+`` accepted ANY
# uppercase kind the scrubber could never emit, and the scrubber's
# ``[^>]+`` accepted ANY kind at all — so a real credential wrapped as
# ``<<REDACTED:<the-value>>>`` was counted as pre-existing / exempted
# unchanged, ``screen_payload`` returned () and egress was permitted.


@dataclass(frozen=True)
class PreflightResult:
    """Outcome of :func:`evaluate_preflight`.

    ``ok`` — the call may proceed to be driven. When ``ok`` is False, ``reason``
    is a complete, model-facing refusal string (returned verbatim as a failed
    ``ToolResult.content``), and the diagnostic tuples name what failed.
    """

    ok: bool
    reason: str = ""
    missing_fields: tuple[str, ...] = ()
    secret_findings: tuple[str, ...] = ()


def is_expensive(engine_kind: Any) -> bool:
    """True when a delegate's resolved engine leaves the local vLLM boundary.

    Accepts an :class:`EngineKind` enum (uses ``.value``) or a raw string. This
    is the expensive/third-party predicate the gate keys on (spec §4.2), and it
    classifies by resolved engine — never by ``subagent_type`` — so a sibling
    configured with an external engine is gated the same as ``subagent_type=
    "claude"`` (closing the alias hole).
    """
    kind = getattr(engine_kind, "value", engine_kind)
    return str(kind) in _EXPENSIVE_ENGINE_KINDS


def _field_present(args: dict[str, Any], key: str) -> bool:
    """Whether a single checklist field is present and non-trivial.

    ``budget_usd`` must parse to a positive number (a budget of ``0`` / ``n/a``
    is not a budget); prose fields must be non-empty after strip and not a bare
    placeholder token.
    """
    val = args.get(key)
    if val is None:
        return False
    if key == "budget_usd":
        s = str(val).strip().lstrip("$").strip()
        try:
            return float(s) > 0
        except (TypeError, ValueError):
            return False
    s = str(val).strip()
    if not s:
        return False
    return s.lower() not in _PLACEHOLDERS


def screen_payload(text: str) -> tuple[str, ...]:
    """Return the secret-kinds a scrub pass would redact from ``text``.

    Reuses :func:`sharedagents_core.secrets_scrubber.scrub_text` as the single
    detection registry — so it inherits every provider-key / credential pattern
    AND the ratified base64 ``data:`` URI exemption: a
    secret-shaped substring inside an image blob is preserved and does NOT flag,
    while a real credential outside a ``data:`` URI does. A non-empty result
    means a secret is present in the outbound payload (hard-block).
    A payload that already carries redaction sentinels from a prior scrub does
    not re-flag: pre-existing ``<<REDACTED:…>>`` sentinels in the input are
    tracked (as a multiset / :class:`~collections.Counter`) before scrubbing
    and excluded from the result, so only NEWLY detected secrets are reported
    (scrub_text is idempotent on sentinels, and the multiset diff closes the
    cascade where one real secret caused every prior sentinel in memory text
    to be reported as a new finding — AND closes the kind-collision hole
    where a pre-existing sentinel and a new secret share the same kind: a
    plain set-difference would see the same kind in both sets and report
    nothing, letting the new secret through unscrubbed; the Counter tracks
    per-kind *counts*, so a kind whose count increased is always reported).

    "Pre-existing sentinel" means a COMPLETE sentinel with a CANONICAL kind —
    exactly the kinds the shared parser :func:`sharedagents_core.secrets_scrubber.iter_complete_sentinels`
    accepts (the scrubber's registered pattern names plus its two
    ``scrub_value`` dynamic kinds). A sentinel-shaped run with ANY other
    kind (arbitrary / mixed-case / colon-suffix — e.g. a real credential
    wrapped as ``<<REDACTED:<the-value>>>``) is ordinary text: it is not
    counted pre-existing, scrub_text redacts it, and it surfaces as a
    finding (fail closed, codex r2).
    """
    if not text:
        return ()
    pre_existing = Counter(iter_complete_sentinels(text))
    scrubbed = scrub_text(text)
    if scrubbed == text:
        return ()
    post = Counter(iter_complete_sentinels(scrubbed))
    found = tuple(sorted(k for k, n in post.items() if n > pre_existing[k]))
    return found or ("UNCLASSIFIED",)


def _serialize_outbound(v: Any) -> str:
    """Render an outbound value to the text a secret screen must see.

    A string is returned as-is. A NON-string prompt/task/context value (dict,
    list, number, ...) is deep-serialized — ``delegate_shims`` coerces the same
    field with ``str(...)`` and ships the result to the CLI/ACP delegate, so a
    secret nested inside (e.g. ``prompt={"token": "AKIA..."}``) has to survive
    into the screened text. ``json.dumps(..., default=repr)`` walks nested
    containers and falls back to ``repr`` for exotic leaves; a total failure
    (non-JSON, non-repr) still degrades to ``repr`` rather than dropping the
    value silently (fail toward screening, never toward egress).
    """
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v, default=repr, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return repr(v)


def _outbound_payload(args: dict[str, Any]) -> str:
    """The concatenated text that would leave the local boundary for this call.

    Every :data:`_OUTBOUND_KEYS` field present in ``args`` is included, whether
    it is a string or not — a non-string value is deep-serialized by
    :func:`_serialize_outbound` first, closing the non-string-payload bypass
    where ``prompt={"token": "..."}`` slipped past a top-level string check and
    was then stringified onto the wire.
    """
    parts: list[str] = []
    for k in _OUTBOUND_KEYS:
        if k not in args:
            continue
        v = args.get(k)
        if v is None:
            continue
        text = _serialize_outbound(v)
        if text.strip():
            parts.append(text)
    return "\n".join(parts)


def evaluate_preflight(engine_kind: Any, args: dict[str, Any] | None) -> PreflightResult:
    """Decide whether an expensive delegate turn may be driven.

    Ordered validator chain (spec/delegation.md §4.4), refusing at the first
    failure:

    1. **Boundary classification** — a cheap/local delegate returns ``ok``
       immediately; nothing below runs (local fan-out pays zero overhead).
    2. **Checklist completeness** — every checklist field
       must be present and non-trivial, else BLOCK naming the missing fields.
    3. **Secret screen** — the outbound payload is screened;
       a detected secret HARD-BLOCKS with no override.

    Checklist precedes the secret screen because there is no materialized
    payload to screen until the fields exist; both precede any (out-of-scope)
    budget/authority check so a secret can never be excused by an otherwise
    perfect call.
    """
    # 1. Boundary classification — local/cheap is never gated.
    if not is_expensive(engine_kind):
        return PreflightResult(ok=True)

    args = args or {}

    # 2. Checklist completeness.
    missing = tuple(f for f in _CHECKLIST_FIELDS if not _field_present(args, f))
    if missing:
        reason = (
            "[delegation gate: REFUSED — incomplete pre-flight checklist. "
            f"Missing/invalid fields: {', '.join(missing)}. Expensive delegate "
            "calls require goal / constraints / acceptance_criteria / budget_usd "
            "(a positive number). Complete the checklist and retry, or use a "
            "local subagent (explore / general / research), which is ungated.]"
        )
        return PreflightResult(ok=False, reason=reason, missing_fields=missing)

    # 3. Secret screen — hard block, no override.
    findings = screen_payload(_outbound_payload(args))
    if findings:
        reason = (
            "[delegation gate: REFUSED — secret detected in the outbound payload "
            f"to a third-party delegate ({', '.join(findings)}). Secrets, API "
            "keys, and auth tokens are hard-blocked from leaving the local "
            "boundary; there is NO override. Remove the credential from the "
            "prompt/checklist and retry.]"
        )
        return PreflightResult(ok=False, reason=reason, secret_findings=findings)

    return PreflightResult(ok=True)


# --- Sensitive-but-sendable egress FLAG (visibility) --------
#
# Contrast with the secret hard-block above: PII has legitimate uses in
# a delegate payload (a frontier consult on a support ticket needs the ticket),
# so the safety property for THIS class is VISIBILITY, not prohibition — the owner
# is flagged that sensitive data is about to cross the local boundary BEFORE an
# expensive (third-party) delegate turn fires. Detection runs over the SAME
# outbound payload the secret screen sees and only ever classifies the data
# (never echoes a value). Local/co-located engines never leave the boundary and
# are never flagged (the arbitrage exemption, identical to is_expensive gating
# for the secret screen).
#
# Detection matches a personal-data VALUE, never vocabulary: the word
# ``address`` or ``account`` is not PII, and neither is an ISO date, an IP, a
# session id, a millisecond timeout or a line range. A flag that fires on most
# prompts tells the owner nothing. Every pattern is bounded (no nested
# unbounded quantifier), so none backtracks catastrophically.
#
# A mailbox is located by its ``@`` and its two sides are matched on short
# slices, so the scan is linear in the payload. One unanchored
# ``local+@domain`` pattern rescans a long unbroken token from every start
# position: 200 KB of hex cost ~20 s on every third-party delegation.
_PII_EMAIL_LOCAL_CHAR_RE = re.compile(r"[A-Za-z0-9._%+-]")
_PII_EMAIL_DOMAIN_RE = re.compile(r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PII_EMAIL_DOMAIN_MAX = 255
# An identifier introduced by its label: ``account 0012345678``,
# ``acct #44-221-9087``, ``phone: (11) 91234-5678``, ``passport X1234567``.
# Only label punctuation may sit between the keyword and the value, so
# ``account settings ... 2026-10-03`` does not match; six or more digits.
_PII_LABELLED_RE = re.compile(
    r"\b(?:accounts?|acct|routing|iban|ssn|passport|phone|tel|mobile|cpf|cnpj)\b"
    r"(?:\s*(?:number|num|no|nr|id)\b\.?)?\s*[:=#]?\s*"
    r"(?!\d{4}-\d{2}-\d{2})[+(]?[A-Za-z]{0,2}\d(?:[ .\-/()]{0,2}\d){5,}",
    re.IGNORECASE,
)
# Identifiers recognisable by their own shape, with or without a label.
_PII_FORMAT_RES = (
    # International phone: +55 11 91234-5678 / +1 415-555-0100 / +14155550100.
    re.compile(
        r"(?<![\w+])\+(?:\d{10,15}(?!\d)"
        r"|\d{1,3}[ .\-]?\(?\d{2,4}\)?[ .\-]\d{3,5}[ .\-]?\d{3,4}(?!\d))"
    ),
    re.compile(r"(?<![\d\-])\d{3}-\d{2}-\d{4}(?![\d\-])"),  # US SSN
    re.compile(r"(?<!\d)\d{3}\.\d{3}\.\d{3}-\d{2}(?!\d)"),  # CPF
    re.compile(r"(?<!\d)\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}(?!\d)"),  # CNPJ
    re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),  # IBAN
    re.compile(r"(?<![\d\-])(?:\d{4}[ \-]){3}\d{4}(?![\d\-])"),  # payment card
)


def _has_email_address(payload: str) -> bool:
    """True for a mailbox; an scp-style remote (``git@example.com:path``) is not one."""
    at = payload.find("@")
    while at != -1:
        if at and _PII_EMAIL_LOCAL_CHAR_RE.match(payload[at - 1]):
            m = _PII_EMAIL_DOMAIN_RE.match(
                payload,
                at + 1,
                at + 1 + _PII_EMAIL_DOMAIN_MAX,
            )
            if m and payload[m.end() : m.end() + 1] != ":":
                return True
        at = payload.find("@", at + 1)
    return False


def sensitive_egress_classes(args: dict[str, Any] | None) -> tuple[str, ...]:
    """Return the sensitive-but-sendable data CLASSES present in the outbound
    payload (``()`` when none).

    A non-empty result means the payload carries data the owner must be
    flagged about before it leaves the local boundary. The
    detection registry is deliberately the same ``_outbound_payload`` text the
    secret screen uses, so a non-string ``prompt`` cannot smuggle PII past
    a top-level string check. NEVER returns or logs a sensitive VALUE — only the
    class label (``"pii"``).
    """
    payload = _outbound_payload(args or {})
    if not payload:
        return ()
    classes: list[str] = []
    if (
        _has_email_address(payload)
        or _PII_LABELLED_RE.search(payload)
        or any(rx.search(payload) for rx in _PII_FORMAT_RES)
    ):
        classes.append("pii")
    return tuple(classes)


def sensitive_egress_flag_event(
    classes: tuple[str, ...],
    agent: str = "",
) -> dict[str, Any]:
    """Build the structured, durable pre-egress warning record for
    :func:`sensitive_egress_classes` findings.

    The event names the class(es) but never a value, and is shaped so any
    owner-visible surface (and the archival log) can recognise a sensitive
    egress flag without a bespoke schema. Emitted BEFORE the delegate turn.
    ``agent`` is the delegate about to receive the data: what tells one entry
    of a fan-out from another, for a reader and for the log.
    """
    labels = list(classes)
    return {
        "type": "sensitive_egress_flag",
        "kind": "sensitive_egress_flag",
        "sensitive": True,
        "direction": "egress",
        "boundary": "delegate",
        "classes": labels,
        "detail": (
            "sensitive data (" + ", ".join(labels) + ") is about to leave the local boundary to a third-party delegate"
        ),
        **({"agent": agent} if agent else {}),
    }


# What a reader calls each sensitive-data class.
_SENSITIVE_CLASS_PROSE = {"pii": "Personal data (PII)"}


def sensitive_egress_notice_text(classes: tuple[str, ...], agent: str = "") -> str:
    """Owner-facing prose for the pre-egress sensitive-data flag.

    One plain sentence shown in the conversation before the third-party call
    proceeds, at every ``/tools`` level. It is read by a person, so it names
    the agent receiving the data and carries no log tag, batch index or
    requirement id (those stay in the event and the session log).
    """
    known = [_SENSITIVE_CLASS_PROSE[c] for c in classes if c in _SENSITIVE_CLASS_PROSE]
    what = (
        " and ".join(known) if known and len(known) == len(classes) else "Sensitive data (" + ", ".join(classes) + ")"
    )
    to = f"{agent}, a third-party agent" if agent else "a third-party agent"
    return f"{what} in this request is being sent to {to}."


class SecretEgressBlocked(RuntimeError):
    """Raised at a third-party egress SINK when the fully-composed outbound
    payload contains a secret.

    :func:`evaluate_preflight` screens the pre-turn CALL ARGS, but the
    provider-bound payload is only fully assembled INSIDE the child engine's
    ``run_turn`` — system prompt + preamble + child AGENTS.md / MEMORY / pinned
    facts / todos / skill / memory-recall context + the user prompt. This
    exception is the hard-block signal for a secret found in that composed
    string right before it leaves the process (no override,
    no silent redaction — the turn is aborted, the payload never sent).

    ``findings`` names the secret KINDS (never the values); ``reason`` is a
    complete, model-facing refusal string safe to surface as a ``TurnError``
    message (it does not echo the secret).
    """

    def __init__(self, findings: tuple[str, ...]) -> None:
        self.findings: tuple[str, ...] = tuple(findings)
        super().__init__(self.reason)

    @property
    def reason(self) -> str:
        return (
            "[delegation gate: REFUSED at egress — secret detected in the "
            "composed outbound payload to a third-party delegate "
            f"({', '.join(self.findings)}). Secrets, API keys, and auth tokens "
            "are hard-blocked from leaving the local boundary; there is NO "
            "override. Remove the credential from the prompt / system prompt / "
            "memory context and retry.]"
        )


def guard_egress_payload(engine_kind: Any, text: str) -> str:
    """Screen the FINAL composed outbound string at the true third-party egress.

       The sink-side companion to :func:`evaluate_preflight`. The gate can only see
       the pre-turn call args; the payload actually handed to an external CLI
       subprocess / ACP session is composed later, deep inside the child engine's
       ``run_turn`` (system prompt + folded preamble/memory context + the user
       prompt). Both CLI and ACP egress call THIS one helper immediately before
       that composed string leaves the process, so every context source is
       covered by an identical screen — closing the "wrong-place scrub" gap where
       composed context reached a frontier vendor unscreened (one screen, no
       bypass).

       Policy (HARD BLOCK, no override):

       * A LOCAL engine is never a third party → ``text`` is returned unchanged and
         is never screened (the cheap side of the arbitrage pays zero overhead;
         the block is scoped to third-party delegates).
       * A clean third-party payload is returned unchanged.
       * A third-party payload containing a secret raises
         :class:`SecretEgressBlocked` — the caller aborts the turn (fail-safe),
         it does NOT redact-and-send. The ratified base64 ``data:`` URI exemption
    is inherited from :func:`screen_payload`/``scrub_text``.
    """
    if not text or not is_expensive(engine_kind):
        return text
    findings = screen_payload(text)
    if findings:
        raise SecretEgressBlocked(findings)
    return text


__all__ = (
    "PreflightResult",
    "SecretEgressBlocked",
    "evaluate_preflight",
    "guard_egress_payload",
    "is_expensive",
    "screen_payload",
    "sensitive_egress_classes",
    "sensitive_egress_flag_event",
    "sensitive_egress_notice_text",
)
