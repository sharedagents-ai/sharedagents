"""Machine-readable resume brief for round-capped turns (OpenCode parity).

When a turn hits the hard per-turn round cap it ends terminally with
``completion_reason='round_cap'``. Recovery is an *explicit* next turn —
the engine NEVER auto-resumes. The capped turn leaves a structured resume
brief:

* delivered IN the Agent result as a fenced ``[resume-brief]`` block
* written atomically to ``~/.sharedagents/jobs/resume:<session>:<turn>.json``

A subsequent turn whose WHOLE prompt is that rendered brief gets a compact
context (brief + bounded recent tail) — never a full history replay.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

#: Ordered fields of a resume brief (machine contract — do not rename).
RESUME_BRIEF_FIELDS = (
    "session_id",
    "turn_id",
    "worktree",
    "branch",
    "final_summary",
    "remaining_todos",
    "last_round_idx",
    "next_step",
    "reason",
)


def _jobs_root() -> Path:
    """``~/.sharedagents/jobs`` honoring ``SHAREDAGENTS_HOME``."""
    base = os.environ.get("SHAREDAGENTS_HOME") or os.path.join(str(Path.home()), ".sharedagents")
    return Path(base) / "jobs"


def build_resume_brief(
    *,
    session_id: str,
    turn_id: Any,
    worktree: str = "",
    branch: str = "",
    final_summary: str = "",
    remaining_todos: Sequence[str] = (),
    last_round_idx: int = 0,
    next_step: str = "",
    reason: str = "round_cap",
) -> dict[str, Any]:
    """Assemble the structured resume brief (machine contract)."""
    return {
        "session_id": str(session_id),
        "turn_id": turn_id,
        "worktree": str(worktree or ""),
        "branch": str(branch or ""),
        "final_summary": str(final_summary or ""),
        "remaining_todos": [str(t) for t in (remaining_todos or ())],
        "last_round_idx": int(last_round_idx or 0),
        "next_step": str(next_step or ""),
        "reason": reason,
    }


def render_resume_brief(brief: dict[str, Any]) -> str:
    """Render the brief as a fenced, machine-readable block."""
    payload = json.dumps(brief, indent=2, sort_keys=True)
    return f"[resume-brief]\n{payload}\n[/resume-brief]"


#: Line-anchored "Next step(s):" markers (singular and plural).
_NEXT_STEP_LINE_RE = re.compile(r"(?im)^\s*(?:next\s*steps?|next|up\s*next)\s*:?\s*([^\n]+?)\s*$")
#: Non-anchored "next step(s)" phrase inside a single-line wrap-up.
_NEXT_STEP_PHRASE_RE = re.compile(r"(?i)next\s*steps?\s*:?\s*([^.;\n]+)")
#: Section (2) "remaining tasks" header (line-anchored).
_REMAINING_SECTION_RE = re.compile(r"(?im)^\s*(?:\(2\)\s*)?(?:remaining\s*tasks?|remaining|to\s*do|todos?)\b[^\n]*")
#: Non-anchored "(2) … remaining …" phrasing inside a single-line wrap-up.
_REMAINING_PHRASE_RE = re.compile(r"(?i)\(2\)[^()\n]*?(?:remaining|to\s*do|todos?)[^()\n]*?[:\-]\s*([^.;\n]+)")
_NEXT_SECTION_RE = re.compile(r"(?im)^\s*\([123]\)")
_BULLET_STRIP_RE = re.compile(r"^[\s\-*•·◦‣]+")

_MD_STRIP_RE = re.compile(r"(?m)(\*\*|__|`|#{1,6}\s*)")
_MD_EMPH_STRIP_RE = re.compile(r"\*([^*_\s][^*_]*?)\*")
_SECTION_LABEL_RE = re.compile(
    r"(?i)^\s*\([123]\)\s*(?:work\s*completed|work\s+done|completed|"
    r"remaining\s*tasks?|remaining|blockers?|next\s*steps?|next)\b.*$"
)
_REMAINING_NONE_RE = re.compile(r"(?i)(?:remaining\s*tasks?|remaining|to\s*do|todos?)\s*:\s*none\b")
_NEXT_STEP_NONE = "none"


def _strip_markdown(text: str) -> str:
    """Remove Markdown formatting markers from wrap-up text."""
    text = _MD_STRIP_RE.sub("", text or "")
    return _MD_EMPH_STRIP_RE.sub(r"\1", text)


def _clean_step(cand: str) -> str:
    cand = _BULLET_STRIP_RE.sub("", cand or "").strip().strip(".").strip()
    cand = _strip_markdown(cand).strip().strip(".").strip()
    return cand[:240]


def extract_next_step(
    wrap_up_text: str,
    remaining_todos: Sequence[str] = (),
) -> str:
    """Extract the single next actionable step from a round-cap wrap-up.

    ``next_step`` is never an empty string. Cascade:

    1. explicit ``Next step:`` / ``Next steps:`` marker (line then inline);
    2. persisted pending todos (preferred over a textual ``remaining: none``);
    3. explicit ``remaining tasks: none`` → documented ``"none"`` sentinel;
    4. first bullet of the ``(2) remaining tasks`` section;
    5. wrap-up's first non-empty, non-label line;
    6. generic continue directive (last resort).
    """
    text = _strip_markdown(wrap_up_text or "").strip()
    pending = [str(t).strip() for t in (remaining_todos or ()) if str(t).strip()]

    if not text:
        if pending:
            return _clean_step(pending[0])
        return "Continue the task described in the resume brief"

    for m in _NEXT_STEP_LINE_RE.finditer(text):
        cand = _clean_step(m.group(1))
        if cand:
            return cand
    m = _NEXT_STEP_PHRASE_RE.search(text)
    if m:
        cand = _clean_step(m.group(1))
        if cand:
            return cand

    # Prefer real persisted todos over a wrap-up that claims "none".
    if pending:
        return _clean_step(pending[0])

    if _REMAINING_NONE_RE.search(text):
        return _NEXT_STEP_NONE

    m = _REMAINING_SECTION_RE.search(text)
    if m:
        section = text[m.end() :]
        end = _NEXT_SECTION_RE.search(section)
        if end:
            section = section[: end.start()]
        for line in section.splitlines():
            if _SECTION_LABEL_RE.match(line):
                continue
            cand = _clean_step(line)
            if cand:
                return cand
    m = _REMAINING_PHRASE_RE.search(text)
    if m:
        cand = _clean_step(m.group(1))
        if cand:
            return cand

    first = next(
        (_clean_step(ln) for ln in text.splitlines() if ln.strip() and not _SECTION_LABEL_RE.match(ln)),
        "",
    )
    return first or "Continue the task described in the resume brief"


def write_resume_brief_sidecar(brief: dict[str, Any], *, home: Path | None = None) -> Path:
    """Persist the brief atomically to ``jobs/resume:<session>:<turn>.json``."""
    root = Path(home) if home is not None else _jobs_root()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"resume:{brief['session_id']}:{brief['turn_id']}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(brief, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return path
