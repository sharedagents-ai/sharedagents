"""Task-completion recap side call (pure logic, no I/O).

Mirror of :mod:`sharedagents_session.session_title` for the Tasks-board recap:
this module holds the *decisions* — how to build the digest prompt, how to
parse the strict-JSON verdict — so they are unit-testable without a running
agent or a network. The async LLM call lives in
:meth:`sharedagents_agent.agents.agent.Agent._make_hidden_agent_caller`
(``"summary"`` profile, configured selection or Jarvis default) and the trigger /
persistence in :mod:`sharedagents_web.task_recap` + the gateway control route.

The model must return one strict JSON object (see
``profile_agents/summary/AGENTS.md``)::

    {"summary": "...", "verdict": "complete|follow_up",
     "follow_up": "...", "confidence": 0.0}

``parse_recap_reply`` is deliberately forgiving: a fenced block, an
unclosed ``<think>``, or prose around the object still parse; a plain-text
reply degrades to ``verdict="unknown"``. Malformed structured
replies return an empty summary so the caller can retry instead of storing
raw JSON as a finished recap. A current follow-up verdict moves an idle
task to Needs input on the board; session runtime state is unchanged.

Long final replies are clipped head + tail, never head-only: the closing
questions, approval requests and hand-offs that decide the verdict live at
the end of the reply.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Digest input caps (chars). The side call must never turn a long task into
# a long prompt: opening request 4k, the last two user turns 2k each, the
# final assistant reply 6k, joined user part capped at ~12k.
RECAP_OPENING_MAX_CHARS = 4_000
RECAP_USER_TEXT_MAX_CHARS = 2_000
RECAP_ASSISTANT_MAX_CHARS = 6_000
#: Share of a clipped final reply kept from its END. Questions, approval
#: requests and "left for you" hand-offs close the reply, so most of the
#: budget goes to the tail.
RECAP_ASSISTANT_TAIL_SHARE = 0.75
RECAP_TOTAL_MAX_CHARS = 12_000
RECAP_SUMMARY_MAX_CHARS = 2_000
RECAP_FOLLOW_UP_MAX_CHARS = 1_000
RECAP_GOAL_MAX_CHARS = 1_500
RECAP_SUMMARY_FALLBACK_MAX_CHARS = 1_200
#: Bump when the prompt or digest changes what a verdict means: stored
#: recaps from an older version are re-run once (``task_board``).
RECAP_VERSION = 2

VERDICT_COMPLETE = "complete"
VERDICT_FOLLOW_UP = "follow_up"
VERDICT_UNKNOWN = "unknown"


def recap_model_rows(cfg: Any) -> list[dict]:
    """Configured compatible deployments; keys are aliases or model IDs."""
    from sharedagents_gateway.codex_responses import is_codex_responses_endpoint

    rows = list(getattr(cfg, "models", None) or []) + [
        {
            "id": getattr(cfg, "upstream_model", ""),
            "baseUrl": getattr(cfg, "upstream_endpoint", ""),
        }
    ]
    result, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        model = str(row.get("id") or "").strip()
        endpoint = str(row.get("baseUrl") or "").rstrip("/")
        key = str(row.get("alias") or model).strip()
        if not key or not model or not endpoint or key in seen or is_codex_responses_endpoint(endpoint):
            continue
        seen.add(key)
        result.append({**row, "key": key, "id": model, "baseUrl": endpoint})
    return result


_COMPLETE_ALIASES = frozenset({"complete", "completed", "done"})
_FOLLOW_UP_ALIASES = frozenset(
    {
        "follow_up",
        "followup",
        "follow-up",
        "continue",
        "off_track",
        "off-track",
        "looping",
        "stuck",
        "needs_human",
        "needs-human",
        "incomplete",
        "needs_input",
        "needs-input",
        "blocked",
        "pending",
    }
)
# Follow-up notes that only say "nothing remains".
_EMPTY_FOLLOW_UPS = frozenset(
    {
        "",
        "none",
        "n/a",
        "na",
        "-",
        "nothing",
        "no",
        "null",
        "nil",
        "nothing remains",
        "no follow-up",
        "no follow up",
        "none needed",
    }
)

_THINK_BLOCK_RE = re.compile(
    r"<\s*\|?\s*think\b[^>]*>.*?(?:<\s*\|?\s*/?\s*think\s*\|?\s*>|<\s*\|\s*end\s*\|\s*>)",
    re.DOTALL | re.IGNORECASE,
)
_THINK_OPEN_RE = re.compile(
    r"<\s*\|?\s*think\b[^>]*>.*$",
    re.DOTALL | re.IGNORECASE,
)


def _clip(text: Any, limit: int, *, tail_share: float = 0.0) -> str:
    """Neutralize and cap ``text``; ``tail_share`` of the budget keeps its end."""
    from sharedagents_core.injection_guard import neutralize

    s = neutralize(str(text or "").strip())
    if len(s) <= limit:
        return s
    if limit < 2:
        return "…"[:limit]
    tail = int((limit - 1) * tail_share)
    head = limit - 1 - tail
    if not tail:
        return s[:head].rstrip() + "…"
    return s[:head].rstrip() + "…" + s[-tail:].lstrip()


def build_recap_messages(
    *,
    label: str = "",
    opening_prompt: str = "",
    latest_user_texts: list[str] | tuple[str, ...] = (),
    last_assistant_text: str = "",
    active_goal: str = "",
    session_summary: str = "",
) -> list[dict[str, str]]:
    """Build the recap side-call chat messages (system + single user digest).

    Every part is individually capped; the joined digest is capped at
    ``RECAP_TOTAL_MAX_CHARS`` so the call stays bounded no matter how long
    the parent task ran.
    """
    from sharedagents_session.hidden_agent import load_hidden_agent_system

    parts: list[str] = []
    task = _clip(label, RECAP_USER_TEXT_MAX_CHARS)
    parts.append("Task name: " + (task or "(unnamed task)"))
    goal = _clip(active_goal, RECAP_GOAL_MAX_CHARS)
    if goal:
        parts.append("Active goal:\n" + goal)
    opening = _clip(opening_prompt, RECAP_OPENING_MAX_CHARS)
    if opening:
        parts.append("Opening request:\n" + opening)
    texts = [t for t in (latest_user_texts or ()) if str(t or "").strip()]
    # Newest last, dedup against the opening request by identity only —
    # paraphrases are cheap to keep and valuable to the recap.
    for i, t in enumerate(texts[-2:], start=1):
        parts.append(f"Latest request {i}:\n" + _clip(t, RECAP_USER_TEXT_MAX_CHARS))
    reply = _clip(last_assistant_text, RECAP_ASSISTANT_MAX_CHARS, tail_share=RECAP_ASSISTANT_TAIL_SHARE)
    reply_index = None
    if reply:
        reply_index = len(parts)
        parts.append("Final assistant reply:\n" + reply)
    prior = _clip(session_summary, RECAP_USER_TEXT_MAX_CHARS)
    if prior:
        parts.append("Earlier summary:\n" + prior)
    contract = (
        "Recap this task and judge its completion. Return ONLY one strict "
        'JSON object: {"summary": "...", "verdict": "complete|follow_up", '
        '"follow_up": "...", "confidence": 0.0}. '
        "Describe tokenizer control markers by name, without reproducing "
        "their delimiter spellings."
    )
    # Budget every section before joining: truncating the joined prefix can
    # discard the final result (the evidence for the verdict) and contract.
    available = RECAP_TOTAL_MAX_CHARS - len(contract) - 2 * len(parts)
    total = sum(map(len, parts))
    if total > available:
        # Keep short sections intact, sharing the remainder among long ones.
        remaining = len(parts)
        for index in sorted(range(len(parts)), key=lambda i: len(parts[i])):
            share = RECAP_ASSISTANT_TAIL_SHARE if index == reply_index else 0.0
            parts[index] = _clip(parts[index], available // remaining, tail_share=share)
            available -= len(parts[index])
            remaining -= 1
    body = "\n\n".join([*parts, contract])
    return [
        {"role": "system", "content": load_hidden_agent_system("summary")},
        {"role": "user", "content": body},
    ]


def _strip_think(raw: str) -> str:
    s = _THINK_BLOCK_RE.sub(" ", raw)
    return _THINK_OPEN_RE.sub(" ", s)


def _first_json_object(raw: str) -> dict[str, Any] | None:
    """First balanced ``{...}`` JSON object in ``raw`` (string-aware).

    Tolerates code fences and prose around the object. Returns None when no
    brace-balanced JSON object parses.
    """
    start = raw.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(raw)):
            ch = raw[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = raw[start : i + 1]
                    try:
                        obj = json.loads(chunk)
                    except (TypeError, ValueError, RecursionError):
                        break
                    if isinstance(obj, dict):
                        return obj
                    break
        start = raw.find("{", start + 1)
    return None


def normalize_verdict(value: Any) -> str:
    s = str(value or "").strip().lower().replace(" ", "_")
    if s in _COMPLETE_ALIASES:
        return VERDICT_COMPLETE
    if s in _FOLLOW_UP_ALIASES:
        return VERDICT_FOLLOW_UP
    return VERDICT_UNKNOWN


def _clean_raw_text(raw: str) -> str:
    s = _strip_think(raw)
    s = re.sub(r"^```[a-zA-Z]*\s*|```\s*$", "", s.strip()).strip()
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s


def parse_recap_reply(raw: Any) -> dict[str, Any]:
    """Parse the recap side-call reply into the board's contract.

    Returns ``{"summary", "verdict", "follow_up", "confidence"}`` with a
    verdict of ``complete`` / ``follow_up`` / ``unknown``. Never raises: a
    plain-text reply degrades to ``unknown`` + the cleaned raw text. A broken
    JSON object or an object with no summary returns an empty summary so the
    gateway treats it as a retryable failure.
    """
    text = raw if isinstance(raw, str) else str(raw or "")
    obj = _first_json_object(_strip_think(text))
    summary = ""
    verdict = VERDICT_UNKNOWN
    follow_up = ""
    confidence = 0.0
    if obj is not None:
        value = obj.get("summary") or obj.get("recap") or obj.get("progress")
        summary = value.strip() if isinstance(value, str) else ""
        verdict = normalize_verdict(obj.get("verdict"))
        follow_up = str(obj.get("follow_up") or obj.get("followup") or "").strip()
        try:
            confidence = float(obj.get("confidence", 0.0))
        except (TypeError, ValueError, OverflowError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        if verdict == VERDICT_COMPLETE:
            # A complete verdict carrying a real follow-up note is
            # contradictory. The note is evidence of remaining work, so the
            # task waits on the owner rather than hiding it as Completed.
            if follow_up.strip(" .").lower() in _EMPTY_FOLLOW_UPS:
                follow_up = ""
            else:
                verdict = VERDICT_FOLLOW_UP
    if obj is None:
        cleaned = _clean_raw_text(text)
        if not cleaned.startswith(("{", "[")):
            summary = cleaned[:RECAP_SUMMARY_FALLBACK_MAX_CHARS]
    if not summary:
        verdict = VERDICT_UNKNOWN
    return {
        "summary": summary[:RECAP_SUMMARY_MAX_CHARS],
        "verdict": verdict,
        "follow_up": follow_up[:RECAP_FOLLOW_UP_MAX_CHARS],
        "confidence": confidence,
    }


__all__ = [
    "RECAP_ASSISTANT_MAX_CHARS",
    "RECAP_FOLLOW_UP_MAX_CHARS",
    "RECAP_OPENING_MAX_CHARS",
    "RECAP_SUMMARY_MAX_CHARS",
    "RECAP_TOTAL_MAX_CHARS",
    "RECAP_USER_TEXT_MAX_CHARS",
    "RECAP_VERSION",
    "VERDICT_COMPLETE",
    "VERDICT_FOLLOW_UP",
    "VERDICT_UNKNOWN",
    "build_recap_messages",
    "normalize_verdict",
    "parse_recap_reply",
]
