"""Agent questions and permission prompts as numbered WhatsApp text.

An ask-family tool call (``question`` / ``AskUserQuestion`` / …) or a
``permission_request`` parks the turn until a human answers. WhatsApp has no
buttons in a self-chat, so each prompt is rendered as a numbered list and the
operator's next message is parsed into the gateway's answer payload: numbers,
an option label, yes/no words (English and Portuguese), or — for questions —
free text, which the gateway renders as a typed "Other" answer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

PERMISSION_LABELS = {
    "allow_once": "Allow once",
    "allow_always": "Always allow",
    "reject_once": "Deny",
    "reject_always": "Always deny",
}
YES_WORDS = frozenset({"y", "yes", "ok", "allow", "approve", "sim", "s"})
NO_WORDS = frozenset({"n", "no", "deny", "reject", "cancel", "nao", "não"})
# Reply meaning "cancel the gate" when no deny option was offered; the channel
# resolves it as the ACP ``cancelled`` outcome (fail-closed: the tool won't run).
CANCEL = "\x00cancel"
_NUMBERS_RE = re.compile(r"^\s*\d+(\s*[,\s]\s*\d+)*\s*$")
_DETAIL_MAX = 600


@dataclass(frozen=True)
class Question:
    text: str
    header: str = ""
    options: tuple[str, ...] = ()
    descriptions: tuple[str, ...] = ()
    multi: bool = False


@dataclass
class PendingQuestion:
    """A parked ask call; its questions are asked (and answered) one at a time."""

    call_id: str
    questions: list[Question]
    answers: list[dict] = field(default_factory=list)

    @property
    def key(self) -> str:
        return self.call_id

    @property
    def current(self) -> Question:
        return self.questions[len(self.answers)]

    @property
    def done(self) -> bool:
        return len(self.answers) >= len(self.questions)


@dataclass
class PendingPermission:
    """A parked permission gate; ``options`` are the option ids the gateway offered."""

    request_id: str
    options: list[str]
    summary: str = ""
    detail: str = ""

    @property
    def key(self) -> str:
        return self.request_id


def parse_questions(arguments: Any) -> list[Question]:
    """Normalize an ask-family tool's arguments (dict or JSON string)."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return []
    if not isinstance(arguments, dict):
        return []
    raw = arguments.get("questions")
    if raw is None and arguments.get("question"):
        raw = [arguments]
    if not isinstance(raw, list):
        return []
    out: list[Question] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("question") or "").strip()
        if not text:
            continue
        labels: list[str] = []
        descriptions: list[str] = []
        options = item.get("options")
        if isinstance(options, str):
            try:
                options = json.loads(options)
            except ValueError:
                options = [options]
        if options is None:
            options = []
        elif not isinstance(options, list):
            options = [options]
        for option in options:
            if isinstance(option, dict):
                label = str(option.get("label") or "").strip()
                description = str(option.get("description") or "").strip()
            else:
                label, description = str(option).strip(), ""
            if label:
                labels.append(label)
                descriptions.append(description)
        multi = bool(item.get("multiSelect", item.get("multiple", False)))
        header = str(item.get("header") or "").strip()
        out.append(Question(text, header, tuple(labels), tuple(descriptions), multi))
    return out


def render_question(question: Question, index: int = 1, total: int = 1) -> str:
    """The numbered prompt for one question (Markdown; converted on send)."""
    head = f"❓ **{question.header}** — {question.text}" if question.header else f"❓ {question.text}"
    if total > 1:
        head = f"{head} ({index}/{total})"
    lines = [head]
    for i, (label, description) in enumerate(
        zip(question.options, question.descriptions),
        1,
    ):
        lines.append(f"{i}. {label}" + (f" — {description}" if description else ""))
    if question.options:
        pick = "numbers separated by commas" if question.multi else "a number"
        lines.append(f"\nReply with {pick}, or type your own answer.")
    else:
        lines.append("\nType your answer.")
    return "\n".join(lines)


def parse_answer(question: Question, reply: str) -> dict | None:
    """Map a reply onto ``{question, selected, notes}``; ``None`` means re-prompt."""
    text = (reply or "").strip()
    if not text:
        return None
    if question.options and _NUMBERS_RE.match(text):
        picks = [int(n) for n in re.split(r"[,\s]+", text) if n]
        if not all(1 <= n <= len(question.options) for n in picks):
            return None
        if len(picks) > 1 and not question.multi:
            return None
        selected: list[str] = []
        for n in picks:
            label = question.options[n - 1]
            if label not in selected:
                selected.append(label)
        return {"question": question.text, "selected": selected, "notes": ""}
    for label in question.options:
        if label.casefold() == text.casefold():
            return {"question": question.text, "selected": [label], "notes": ""}
    return {"question": question.text, "selected": [], "notes": text}


def render_permission(pending: PendingPermission) -> str:
    """The numbered permission prompt (Markdown; converted on send)."""
    lines = [f"🔐 Permission needed: **{pending.summary or 'tool call'}**"]
    detail = pending.detail.strip().replace("```", "'''")
    if detail:
        if len(detail) > _DETAIL_MAX:
            detail = detail[:_DETAIL_MAX] + "…"
        lines.append(f"```\n{detail}\n```")
    for i, option in enumerate(pending.options, 1):
        lines.append(f"{i}. {PERMISSION_LABELS.get(option, option)}")
    if pending.options:
        lines.append(f"\nReply 1–{len(pending.options)} (or yes / no).")
    else:
        lines.append("\nReply no to cancel it, or answer in the web UI.")
    return "\n".join(lines)


def parse_permission_reply(pending: PendingPermission, reply: str) -> str | None:
    """Map a reply onto an OFFERED option id; ``None`` means re-prompt.

    The gateway also refuses an option it did not offer; checking here keeps
    the operator's typo from ever reaching it as a wider grant.
    """
    text = (reply or "").strip().casefold()
    if not text:
        return None
    if pending.options:
        if text.isdigit():
            n = int(text)
            return pending.options[n - 1] if 1 <= n <= len(pending.options) else None
        for option in pending.options:
            if text in (option.casefold(), PERMISSION_LABELS.get(option, "").casefold()):
                return option
        if text in YES_WORDS:
            return _first_offered(pending.options, ("allow_once", "allow_always"))
    if text in NO_WORDS:
        # Deny when a deny option was offered; otherwise cancel the gate.
        return _first_offered(pending.options, ("reject_once", "reject_always")) or CANCEL
    return None


def _first_offered(offered: list[str], preferred: tuple[str, ...]) -> str | None:
    for want in preferred:
        if want in offered:
            return want
    return None
