"""Session-scoped SDLC gate board — the todo list, but you cannot skip.

Like ``todowrite``, this persists on the session and returns a compact card
the orchestrator keeps in context. Unlike todos, a gate only moves when
this tool CLEARS or LOOPS it. Jumping ahead is refused. Jarvis holds this
card, not the child transcript.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GATES: tuple[str, ...] = (
    "describe",
    "clarify",
    "contract",
    "plan",
    "design",
    "code",
    "tests",
    "deploy",
    "ship",
)

#: A failed generate/verify hops back to the producer of the artifact.
PRODUCER: dict[str, str] = {
    "describe": "describe",
    "clarify": "clarify",
    "contract": "clarify",
    "plan": "plan",
    "design": "design",
    "code": "code",
    "tests": "code",
    "deploy": "code",
    "ship": "tests",
}

MAX_ATTEMPTS = 3
MAX_NO_PROGRESS = 2

#: Independent reviewers. The conductor (Jarvis) cannot CLEAR ``tests``.
REVIEW_ROLES: tuple[str, ...] = ("validation", "verification")
REVIEW_ROLE_ALIASES = {
    "validation": "validation",
    "verify-a": "validation",
    "verify_a": "validation",
    "verification": "verification",
    "verify-b": "verification",
    "verify_b": "verification",
}

_VAGUE_CONDITION = (
    "fix it",
    "implement the feature",
    "make it work",
    "as discussed",
    "do the thing",
    "the change",
    "ship it",
)


def sdlc_path_for_session(sessions_dir: Path, sid: str) -> Path:
    return sessions_dir / f"{sid}.sdlc.json"


def resolve_board_path(sessions_dir: Path, sid: str) -> Path:
    """The board for this session, or the aliased conductor board.

    Web conductor ids and gateway conversation hashes are different strings.
    Submit must find the run either way — never spawn a child to hunt the file.
    """
    return _canonical_board_path(sessions_dir, sid)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _canonical_board_path(sessions_dir: Path, sid: str) -> Path:
    """Always the conductor file. Alias files are pointers, never copies."""
    primary = sdlc_path_for_session(sessions_dir, sid)
    data = _read_json(primary) if primary.exists() else {}
    alias = str(data.get("alias_of") or "").strip()
    if alias:
        return sdlc_path_for_session(sessions_dir, alias)
    if data.get("run_id") and data.get("conductor_session_id"):
        conductor = str(data.get("conductor_session_id") or sid)
        return sdlc_path_for_session(sessions_dir, conductor)
    for other in sessions_dir.glob("*.sdlc.json"):
        if other == primary:
            continue
        row = _read_json(other)
        if str(row.get("alias_of") or "") == sid:
            continue
        if str(row.get("conductor_session_id") or "") == sid:
            return other
        aliases = row.get("session_aliases") or []
        if isinstance(aliases, list) and sid in {str(item) for item in aliases}:
            conductor = str(row.get("conductor_session_id") or other.stem)
            return sdlc_path_for_session(sessions_dir, conductor)
    return primary


def _alias_board(path: Path, board: dict[str, Any], extra_sids: list[str]) -> None:
    """Pointer files only. One writable board — the conductor's."""
    aliases = []
    for sid in extra_sids:
        token = str(sid or "").strip()
        if token and token not in aliases:
            aliases.append(token)
    board["session_aliases"] = aliases
    conductor = str(board.get("conductor_session_id") or path.stem)
    board["conductor_session_id"] = conductor
    canonical = sdlc_path_for_session(path.parent, conductor)
    _save_board(canonical, board)
    for sid in aliases:
        dest = sdlc_path_for_session(path.parent, sid)
        if dest.resolve() == canonical.resolve():
            continue
        dest.write_text(
            json.dumps({"alias_of": conductor}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def set_board_worktree(sessions_dir: Path, sid: str, worktree: str) -> Path:
    """Record the run worktree on the one canonical board."""
    path = resolve_board_path(sessions_dir, sid)
    board = load_board(path)
    board["worktree"] = str(worktree or "")
    if not board.get("conductor_session_id"):
        board["conductor_session_id"] = sid
    extras = list(board.get("session_aliases") or [])
    extras.extend(_session_aliases(sid))
    _alias_board(path, board, extras)
    return resolve_board_path(sessions_dir, sid)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _empty_board() -> dict[str, Any]:
    return {
        "run_id": "",
        "headline": "",
        "repo": "",
        "kind": "feature",
        "conductor_session_id": "",
        "issue_number": None,
        "issue_url": "",
        "open": "",
        "blocked": False,
        "requirements": [],
        "criteria": [],
        "gates": {
            name: {
                "status": "pending",
                "attempts": 0,
                "no_progress": 0,
                "fingerprint": "",
                "note": "",
                "sha": "",
            }
            for name in GATES
        },
        "history": [],
        "drive_ready": False,
        "drive_evidence": "",
        "fuse_plan_design": True,
        "worktree": "",
        "session_aliases": [],
        "reviews": {
            "validation": _empty_review(),
            "verification": _empty_review(),
        },
        "updated_at": _now(),
    }


def _empty_review() -> dict[str, Any]:
    return {
        "status": "pending",
        "verdict": "",
        "note": "",
        "sha": "",
        "session_id": "",
    }


def load_board(path: Path, *, _seen: frozenset[str] | None = None) -> dict[str, Any]:
    if not path.exists():
        return _empty_board()
    key = str(path.resolve())
    seen = set(_seen or ())
    if key in seen:
        return _empty_board()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_board()
    if not isinstance(data, dict):
        return _empty_board()
    alias = str(data.get("alias_of") or "").strip()
    if alias:
        return load_board(
            sdlc_path_for_session(path.parent, alias),
            _seen=frozenset(seen | {key}),
        )
    board = _empty_board()
    board.update({k: data[k] for k in board if k in data and k != "gates"})
    raw_gates = data.get("gates") if isinstance(data.get("gates"), dict) else {}
    for name in GATES:
        row = raw_gates.get(name) if isinstance(raw_gates.get(name), dict) else {}
        board["gates"][name].update({k: row[k] for k in board["gates"][name] if k in row})
    if not isinstance(board.get("history"), list):
        board["history"] = []
    if not isinstance(board.get("requirements"), list):
        board["requirements"] = []
    if not isinstance(board.get("criteria"), list):
        board["criteria"] = []
    raw_reviews = data.get("reviews") if isinstance(data.get("reviews"), dict) else {}
    reviews = {
        "validation": _empty_review(),
        "verification": _empty_review(),
    }
    for name in REVIEW_ROLES:
        row = raw_reviews.get(name) if isinstance(raw_reviews.get(name), dict) else {}
        reviews[name].update({k: row[k] for k in reviews[name] if k in row})
    board["reviews"] = reviews
    board["drive_ready"] = bool(data.get("drive_ready") or board.get("drive_ready"))
    if "fuse_plan_design" in data:
        board["fuse_plan_design"] = bool(data.get("fuse_plan_design"))
    raw_aliases = data.get("session_aliases")
    if isinstance(raw_aliases, list):
        board["session_aliases"] = [str(item) for item in raw_aliases if str(item).strip()]
    return board


def _session_aliases(sid: str) -> list[str]:
    """Return the canonical session id used by one run."""
    out = [sid]
    try:
        from sharedagents_web.server import store

        sdata = store.get(sid) or {}
    except Exception:  # noqa: BLE001 — aliases are best-effort
        return out
    for key in ("id",):
        token = str(sdata.get(key) or "").strip()
        if token:
            out.append(token)
    return out


def seed_board_for_run(sessions_dir: Path, sid: str, run: dict[str, Any]) -> Path:
    """Point this session's sdlc card at an existing SdlcStore run."""
    path = sdlc_path_for_session(sessions_dir, sid)
    board = load_board(path)
    store_id = str(run.get("run_id") or "").strip()
    if board.get("run_id") and board.get("open") and board["run_id"] != store_id:
        return path
    if store_id:
        board["run_id"] = store_id
    if run.get("headline"):
        board["headline"] = str(run["headline"])
    if run.get("repo"):
        board["repo"] = str(run["repo"])
    if run.get("kind"):
        board["kind"] = str(run["kind"])
    if run.get("worktree"):
        board["worktree"] = str(run["worktree"])
    board["conductor_session_id"] = sid
    if board.get("headline") and board["gates"]["describe"].get("status") != "cleared":
        board["gates"]["describe"]["status"] = "cleared"
        board["gates"]["describe"]["note"] = str(board["headline"])
        if not board.get("open"):
            _open_gate(board, "clarify")
    extras = _session_aliases(sid)
    _alias_board(path, board, extras)
    return path


def sync_board_after_ratify(sessions_dir: Path, sid: str, run: dict[str, Any]) -> Path:
    """Contract is frozen; the next Jarvis prompt is the plan ``/goal``."""
    path = sdlc_path_for_session(sessions_dir, sid)
    board = load_board(path)
    store_id = str(run.get("run_id") or "").strip()
    if store_id:
        board["run_id"] = store_id
    if run.get("headline"):
        board["headline"] = str(run["headline"])
    if run.get("repo"):
        board["repo"] = str(run["repo"])
    if run.get("kind"):
        board["kind"] = str(run["kind"])
    board["conductor_session_id"] = sid
    board["issue_number"] = run.get("issue_number")
    board["issue_url"] = str(run.get("issue_url") or "")
    board["requirements"] = list(run.get("requirements") or [])
    board["criteria"] = list(run.get("criteria") or [])
    if run.get("worktree"):
        board["worktree"] = str(run["worktree"])
    for name in ("describe", "clarify", "contract"):
        board["gates"][name]["status"] = "cleared"
    _open_gate(board, "plan")
    extras = _session_aliases(sid)
    _alias_board(path, board, extras)
    return path


def _save_board(path: Path, board: dict[str, Any]) -> None:
    board["updated_at"] = _now()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(board, indent=2, ensure_ascii=False) + "\n"
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _strings(value: Any, *, limit: int = 40) -> list[str]:
    out: list[str] = []
    for item in value if isinstance(value, list) else []:
        text = str(item if not isinstance(item, dict) else item.get("text") or "").strip()
        if text:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _is_condition_line(text: str) -> bool:
    low = " ".join((text or "").lower().split())
    if not low or low in _VAGUE_CONDITION:
        return False
    if "fail-if" in low or "fail if" in low:
        return False
    return any(
        token in low
        for token in (
            "shall ",
            "must ",
            "when ",
            "if ",
            "the system",
            "non-goal",
        )
    )


def _is_evidence_line(text: str) -> bool:
    low = " ".join((text or "").lower().split())
    if not low:
        return False
    if any(
        part in low
        for part in (
            "works as expected",
            "looks good",
            "no regressions",
            "tests pass",
            "should work",
            "as discussed",
        )
    ):
        return False
    has_given = "given " in low or low.startswith("given")
    has_when = " when " in low or low.startswith("when ")
    has_then = " then " in low or " pass " in low or "/ then" in low
    has_fail = "fail-if" in low or "fail if" in low
    return has_given and has_when and has_then and has_fail


def contract_defects(requirements: list[str], criteria: list[str]) -> list[str]:
    defects: list[str] = []
    if not requirements:
        defects.append("no condition for plan")
    else:
        for index, line in enumerate(requirements, 1):
            if not _is_condition_line(line):
                defects.append(f"condition {index} is not WHEN/SHALL")
    if not criteria:
        defects.append("no clearing evidence for test")
    else:
        for index, line in enumerate(criteria, 1):
            if not _is_evidence_line(line):
                defects.append(f"evidence {index} is not given/when/then/fail-if")
    return defects


def _fingerprint(stage: str, sha: str, defects: list[str]) -> str:
    return "|".join((stage, sha.strip(), ";".join(sorted(defects))))


def _open_gate(board: dict[str, Any], stage: str) -> None:
    for _name, row in board["gates"].items():
        if row.get("status") == "open":
            row["status"] = "pending"
    board["gates"][stage]["status"] = "open"
    board["open"] = stage
    board["blocked"] = False
    if stage == "tests":
        _reset_reviews(board)


def _next_after(stage: str, board: dict[str, Any] | None = None) -> str:
    index = GATES.index(stage)
    if index + 1 >= len(GATES):
        return ""
    nxt = GATES[index + 1]
    if stage == "plan" and nxt == "design" and (board or {}).get("fuse_plan_design", True):
        return "code"
    return nxt


def _hint(stage: str) -> str:
    return {
        "describe": "Name the object, repo, and kind. Then submit describe PASS.",
        "clarify": (
            "Interview until WHEN/SHALL condition + given/when/then/fail-if "
            "evidence parse. websearch is a tool of this gate, not a stage. "
            "Submit clarify with requirements and criteria."
        ),
        "contract": (
            "Operator ratifies the two lists. Submit contract PASS with the "
            "same requirements and criteria. Misshapen lines LOOP to clarify."
        ),
        "plan": (
            "This step is a /goal. Delegate ONE plan child for the fused brief "
            "(requirements + file map). You submit. Do not ask for the full text."
        ),
        "design": (
            "Skipped on the fused path. If this gate is open, write spec/ only because the operator split the V."
        ),
        "code": (
            "Delegate ONE build child. You submit with the SHA. "
            "Child reports path + SHA + checks, never the full source."
        ),
        "tests": (
            "Drive each frozen evidence line, then submit verdict=READY. "
            "You cannot CLEAR tests. The driver launches verify A and verify B; "
            "do not spawn them. Tests CLEAR only when both decide PASS and agree. "
            "Red code → FAIL with rewind=previous; bad definition → rewind=plan."
        ),
        "deploy": (
            "Re-check the frozen evidence. The driver opens and merges the PR. "
            "Do not open a PR, do not merge, do not deploy."
        ),
        "ship": ("Merge when verify A + verify B + LSP are green. Deploy stays an operator click (CANON-A)."),
    }.get(stage, f"Work {stage}, then submit.")


def format_sdlc_md(board: dict[str, Any], *, verdict: str = "") -> str:
    """Compact card — this is what Jarvis keeps in context."""
    if not board.get("run_id") and not board.get("open"):
        return "## SDLC\n\n(no run)\n\nCall action=start with headline (and repo if known)."
    lines = ["## SDLC", ""]
    meta = [
        f"run `{board.get('run_id') or '?'}`",
        str(board.get("kind") or "feature"),
    ]
    if board.get("headline"):
        meta.append(str(board["headline"]))
    if board.get("repo"):
        meta.append(str(board["repo"]))
    if board.get("issue_number"):
        meta.append(f"#{board['issue_number']}")
    lines.append(" · ".join(meta))
    if board.get("blocked"):
        lines.append("**BLOCKED** — change the contract or abort. Do not invent a pass.")
    elif board.get("open"):
        lines.append(f"**OPEN** `{board['open']}` — {_hint(str(board['open']))}")
    else:
        lines.append("**DONE** — every gate cleared.")
    if str(board.get("open") or "") == "tests" or board.get("drive_ready"):
        reviews = board.get("reviews") or {}
        va = reviews.get("validation") or {}
        vb = reviews.get("verification") or {}
        lines.append(
            f"verify A: {va.get('verdict') or va.get('status') or 'pending'} · "
            f"verify B: {vb.get('verdict') or vb.get('status') or 'pending'}"
        )
        if board.get("drive_ready") and str(board.get("open") or "") == "tests":
            lines.append("Jarvis is waiting — tests CLEAR only when both reviewers agree PASS.")
    lines.append("")
    for name in GATES:
        row = board["gates"][name]
        status = str(row.get("status") or "pending")
        mark = {"cleared": "x", "open": "~", "looped": "!", "blocked": "!"}.get(status, " ")
        extra = ""
        if row.get("attempts"):
            extra += f" · try {row['attempts']}/{MAX_ATTEMPTS}"
        if row.get("note") and status in {"looped", "blocked", "open"}:
            extra += f" — {row['note']}"
        lines.append(f"- [{mark}] {name}{extra}")
    reqs = board.get("requirements") or []
    crits = board.get("criteria") or []
    if reqs or crits:
        lines += [
            "",
            f"### Frozen contract — {len(reqs)} condition(s), "
            f"{len(crits)} evidence line(s). Full lists are on the issue. "
            "Do not paste them into child prompts or chat.",
        ]
        if reqs:
            lines.append(f"- first condition: {reqs[0]}")
    if verdict:
        lines += ["", verdict]
    lines += [
        "",
        "Keep this card. Do not keep the child transcript. "
        "You have this tool — submit yourself. Never delegate submit. "
        "Delegate only the OPEN gate. Jarvis cannot CLEAR tests.",
    ]
    return "\n".join(lines)


def _try_post_issue(board: dict[str, Any], body: str) -> str:
    repo = str(board.get("repo") or "")
    number = board.get("issue_number")
    if not repo or not number:
        return ""
    try:
        from sharedagents_web.forge_client import comment_issue, configured

        if not configured():
            return ""
        status, payload = comment_issue(repo, int(number), body)
        if status in (200, 201):
            return str(payload.get("html_url") or payload.get("id") or "posted")
    except Exception:  # noqa: BLE001 — forge outage must not stall the gate
        return ""
    return ""


def _store_run_id_for_session() -> str:
    """Reuse the SdlcStore run already bound to this Jarvis chat."""
    sid = os.environ.get("SHAREDAGENTS_SESSION_ID") or ""
    if not sid:
        return ""
    try:
        from sharedagents_web.dev_sdlc import get_sdlc_store
        from sharedagents_web.server import store

        mapped = get_sdlc_store().conductor_run_ids([sid])
        if mapped.get(sid):
            return mapped[sid]
        sdata = store.get(sid)
        if sdata and sdata.get("sdlc_run_id"):
            return str(sdata["sdlc_run_id"])
        if sdata:
            gw = str(sdata.get("gw_sid") or "")
            if gw:
                mapped = get_sdlc_store().conductor_run_ids([gw])
                if mapped.get(gw):
                    return mapped[gw]
    except Exception:  # noqa: BLE001
        return ""
    return ""


def _try_register_task(board: dict[str, Any]) -> None:
    """Put this Jarvis chat on the Tasks board so Dev / Chat / Tasks agree."""
    sid = os.environ.get("SHAREDAGENTS_SESSION_ID") or ""
    if not sid:
        return
    try:
        from sharedagents_web.dev_task import register_dev_conductor
        from sharedagents_web.server import store

        sdata = store.get(sid)
        if not sdata:
            return
        bound = register_dev_conductor(
            owner=str(sdata.get("owner") or ""),
            headline=str(board.get("headline") or "Dev"),
            session_id=sid,
        )
        board["conductor_session_id"] = bound["conductor_session_id"]
    except Exception:  # noqa: BLE001 — board file is enough if web is down
        return


def _try_mint_issue(board: dict[str, Any]) -> str:
    repo = str(board.get("repo") or "")
    headline = str(board.get("headline") or "").strip()
    if not repo or not headline or board.get("issue_number"):
        return ""
    reqs = board.get("requirements") or []
    crits = board.get("criteria") or []
    req_lines = [f"{i}. {text}" for i, text in enumerate(reqs, 1)] or ["_none_"]
    crit_lines = [f"- [ ] {text}" for text in crits] or ["- [ ] _none_"]
    body_lines = [
        f"## Object\n{headline}\n",
        "## Condition (for plan)",
        *req_lines,
        "",
        "## Clearing evidence (for test)",
        *crit_lines,
        "",
        f"SDLC_RUN: `{board.get('run_id')}`",
    ]
    try:
        from sharedagents_web.forge_client import configured, create_issue

        if not configured():
            return "forge not configured — contract frozen locally only"
        status, issue = create_issue(
            repo,
            title=headline,
            body="\n".join(body_lines),
            labels=["status/in-progress", "sdlc/intake"],
        )
        number = issue.get("number") if isinstance(issue, dict) else None
        if status in (200, 201) and number:
            board["issue_number"] = int(number)
            board["issue_url"] = str(issue.get("html_url") or "")
            return f"issue #{number} opened"
    except Exception as exc:  # noqa: BLE001
        return f"issue not minted: {type(exc).__name__}"
    return "issue not minted"


def _record(board: dict[str, Any], *, stage: str, verdict: str, note: str) -> None:
    board["history"].append(
        {
            "at": _now(),
            "stage": stage,
            "verdict": verdict,
            "note": note,
        }
    )
    board["history"] = board["history"][-40:]
    posted = _try_post_issue(
        board,
        f"**`sdlc/{stage}`** — {verdict}\n\n{note}".strip(),
    )
    if posted:
        board["history"][-1]["issue"] = posted


def _start(board: dict[str, Any], args: dict[str, Any]) -> str:
    if board.get("run_id") and board.get("open"):
        return "run already open — call status. Do not start a second board."
    headline = str(args.get("headline") or args.get("object") or "").strip()
    existing_id = _store_run_id_for_session()
    board.update(
        {
            "run_id": existing_id or uuid.uuid4().hex[:12],
            "headline": headline,
            "repo": str(args.get("repo") or "").strip(),
            "kind": str(args.get("kind") or "feature").strip() or "feature",
            "issue_number": args.get("issue_number") or None,
            "issue_url": str(args.get("issue_url") or ""),
        }
    )
    if headline:
        board["gates"]["describe"]["status"] = "cleared"
        board["gates"]["describe"]["note"] = headline
        _open_gate(board, "clarify")
        _try_register_task(board)
        return "CLEAR describe — OPEN clarify"
    _open_gate(board, "describe")
    _try_register_task(board)
    return "OPEN describe — name the object"


def _session_id() -> str:
    return str(os.environ.get("SHAREDAGENTS_SESSION_ID") or "").strip()


def _caller_role(board: dict[str, Any], args: dict[str, Any]) -> str:
    """Who is submitting. The conductor session can never claim a review role."""
    sid = _session_id()
    conductor = str(board.get("conductor_session_id") or "").strip()
    if sid and conductor and sid == conductor:
        return "conductor"
    if not conductor and sid:
        # Fresh board with no conductor yet — this session is Jarvis.
        return "conductor"
    reviews = board.get("reviews") if isinstance(board.get("reviews"), dict) else {}
    for role in REVIEW_ROLES:
        row = reviews.get(role) if isinstance(reviews.get(role), dict) else {}
        if sid and sid == str(row.get("session_id") or "").strip():
            return role
    wanted = REVIEW_ROLE_ALIASES.get(str(args.get("role") or "").strip().lower(), "")
    if wanted and sid and sid != conductor:
        return wanted
    return "conductor"


def _reset_reviews(board: dict[str, Any]) -> None:
    board["drive_ready"] = False
    board["reviews"] = {
        "validation": _empty_review(),
        "verification": _empty_review(),
    }


def reopen_tests(board: dict[str, Any], *, note: str) -> None:
    """HEAD moved after A/B — drive and review again. Tests is the OPEN gate."""
    _reset_reviews(board)
    row = board["gates"]["tests"]
    row["status"] = "open"
    row["note"] = (note or "HEAD moved after A/B")[:400]
    row["sha"] = ""
    if str(board.get("open") or "") == "deploy":
        board["gates"]["deploy"]["status"] = "pending"
        board["gates"]["deploy"]["note"] = ""
        board["gates"]["deploy"]["sha"] = ""
    _open_gate(board, "tests")


def _submit_review(
    board: dict[str, Any],
    *,
    role: str,
    verdict: str,
    note: str,
    sha: str,
    session_id: str = "",
) -> str:
    """Record one independent reviewer. CLEAR tests only when both agree PASS."""
    if role not in REVIEW_ROLES:
        return f"REFUSED — `{role}` is not a review role"
    if not board.get("drive_ready"):
        return "REFUSED — drive the frozen evidence first (conductor submits verdict=READY) before a reviewer PASS."
    if verdict == "PASS" and not str(sha or "").strip():
        return "REFUSED — reviewer PASS needs the commit SHA that was judged"
    if str(board.get("open") or "") != "tests":
        return (
            f"JUMP refused — OPEN is `{board.get('open') or 'none'}`, "
            f"not `tests`. Reviewers submit only while tests is OPEN."
        )
    reviews = board.setdefault(
        "reviews",
        {
            "validation": _empty_review(),
            "verification": _empty_review(),
        },
    )
    row = reviews.setdefault(role, _empty_review())
    row["status"] = "decided"
    row["verdict"] = verdict
    row["note"] = (note or verdict)[:400]
    row["sha"] = sha
    if session_id:
        row["session_id"] = session_id
    label = "verify A" if role == "validation" else "verify B"
    other_role = "verification" if role == "validation" else "validation"
    other = reviews.get(other_role) or _empty_review()
    if str(other.get("status") or "") != "decided":
        board["gates"]["tests"]["note"] = f"{label} {verdict} — waiting for the other reviewer"
        _record(board, stage="tests", verdict=f"REVIEW:{role}", note=row["note"])
        return f"{label} recorded {verdict}. Tests still OPEN. Jarvis cannot CLEAR. Waiting for the other reviewer."
    ok_a = str((reviews["validation"] or {}).get("verdict") or "").upper() == "PASS"
    ok_b = str((reviews["verification"] or {}).get("verdict") or "").upper() == "PASS"
    sha_a = str((reviews["validation"] or {}).get("sha") or "").strip()
    sha_b = str((reviews["verification"] or {}).get("sha") or "").strip()
    sha_mismatch = bool(sha_a and sha_b and sha_a != sha_b)
    notes_a = [str((reviews["validation"] or {}).get("note") or "")]
    notes_b = [str((reviews["verification"] or {}).get("note") or "")]
    if ok_a and ok_b and not sha_mismatch:
        agreed = sha_a or sha_b
        if not agreed:
            return (
                "REFUSED — verify A and verify B both PASS but neither recorded the SHA they judged. Tests stay OPEN."
            )
        return _clear(
            board,
            stage="tests",
            note=("verify A and verify B agreed PASS" + (f" on {agreed}" if agreed else "")),
            sha=agreed,
        )
    defects = []
    if sha_mismatch:
        defects.append(f"reviewers judged different SHAs ({sha_a} vs {sha_b})")
    if not ok_a:
        defects.append("verify A: " + (notes_a[0] or "FAIL"))
    if not ok_b:
        defects.append("verify B: " + (notes_b[0] or "FAIL"))
    if ok_a != ok_b:
        defects.append("verify A and verify B did not agree — tests cannot pass")
    elif not ok_a and not ok_b:
        defects.append("verify A and verify B both rejected — tests cannot pass")
    return _loop(
        board,
        stage="tests",
        defects=defects or ["reviewers did not agree"],
        sha=sha_a or sha_b,
        evidence="",
        rewind="code",
    )


def apply_review(
    sessions_dir: Path,
    conductor_sid: str,
    *,
    role: str,
    verdict: str,
    note: str = "",
    sha: str = "",
    session_id: str = "",
) -> str:
    """Record a reviewer on the conductor board. Used by the driver and the tool."""
    path = sdlc_path_for_session(sessions_dir, conductor_sid)
    board = load_board(path)
    wanted = REVIEW_ROLE_ALIASES.get(str(role or "").strip().lower(), "")
    if wanted not in REVIEW_ROLES:
        return f"REFUSED — `{role}` is not a review role"
    msg = _submit_review(
        board,
        role=wanted,
        verdict=str(verdict or "").strip().upper() or "FAIL",
        note=note,
        sha=sha,
        session_id=session_id,
    )
    _save_board(path, board)
    return msg


def apply_drive_ready(
    sessions_dir: Path,
    conductor_sid: str,
    *,
    evidence: str,
) -> str:
    """Record driver-executed evidence before independent reviewers decide.

    This cannot clear tests. It only marks the already-open tests gate READY;
    verify A and verify B must still agree on one commit through
    :func:`apply_review` before the board advances.
    """
    path = sdlc_path_for_session(sessions_dir, conductor_sid)
    board = load_board(path)
    if str(board.get("open") or "") != "tests":
        return f"JUMP refused — OPEN is `{board.get('open') or 'none'}`, not `tests`."
    blob = str(evidence or "").strip()
    if not blob:
        return "REFUSED — driver readiness needs executed evidence"
    board["drive_ready"] = True
    board["drive_evidence"] = blob[:800]
    board["gates"]["tests"]["note"] = blob[:400]
    _record(board, stage="tests", verdict="READY", note=blob[:400])
    _save_board(path, board)
    return "DRIVE recorded — tests remain OPEN for verify A and verify B."


def apply_stage_pass(
    sessions_dir: Path,
    conductor_sid: str,
    *,
    stage: str,
    note: str,
    sha: str,
) -> str:
    """Advance one verified producer gate on the conductor's board.

    The driver calls this only after structural and semantic verification of
    the committed branch. Tests are excluded because only the A/B reviewer
    path may clear that gate.
    """
    path = sdlc_path_for_session(sessions_dir, conductor_sid)
    board = load_board(path)
    name = str(stage or "").strip().lower()
    gate = (board.get("gates") or {}).get(name) or {}
    if gate.get("status") == "cleared":
        return f"`{name}` already cleared"
    if name == "tests":
        return "REFUSED — tests clear only through verify A and verify B"
    if str(board.get("open") or "") != name:
        return f"JUMP refused — OPEN is `{board.get('open') or 'none'}`, not `{name}`."
    result = _submit(
        board,
        {
            "stage": name,
            "verdict": "PASS",
            "note": str(note or "driver verified the committed artifact"),
            "sha": str(sha or ""),
        },
    )
    _save_board(path, board)
    return result


def _submit(board: dict[str, Any], args: dict[str, Any]) -> str:
    if board.get("blocked"):
        return "BLOCKED — this run is closed. Change the contract or abort."
    stage = str(args.get("stage") or board.get("open") or "").strip().lower()
    if stage not in GATES:
        return f"JUMP refused — unknown gate `{stage or '?'}`"
    if stage != board.get("open"):
        return f"JUMP refused — OPEN is `{board.get('open') or 'none'}`, not `{stage}`. Delegate only the open gate."
    verdict = str(args.get("verdict") or "PASS").strip().upper()
    note = str(args.get("note") or "").strip()
    evidence = str(args.get("evidence") or "").strip()
    sha = str(args.get("head_sha") or args.get("sha") or "").strip()
    defects = _strings(args.get("defects"))
    requirements = _strings(args.get("requirements"))
    criteria = _strings(args.get("criteria"))
    row = board["gates"][stage]
    if stage == "tests":
        role = _caller_role(board, args)
        if role in REVIEW_ROLES:
            return _submit_review(
                board,
                role=role,
                verdict=verdict if verdict in {"PASS", "FAIL"} else "FAIL",
                note=note or evidence,
                sha=sha,
                session_id=_session_id(),
            )
        if verdict in {"READY", "DRIVE"}:
            blob = f"{note}\n{evidence}".strip()
            if _looks_like_source_dump(blob) or not _looks_like_driven_evidence(blob):
                return (
                    "REFUSED — READY needs the observation "
                    "(screenshot path, curl, given/when/then/fail-if). "
                    "A SHA is not a drive."
                )
            board["drive_ready"] = True
            board["drive_evidence"] = blob[:800]
            row["note"] = blob[:400]
            _record(board, stage="tests", verdict="READY", note=row["note"])
            return "DRIVE recorded — tests still OPEN. Jarvis cannot CLEAR. The driver launches verify A and verify B."
        if verdict == "PASS" and not defects:
            return (
                "REFUSED — tests CLEAR only from verify A and verify B. "
                "Drive the frozen evidence, submit verdict=READY, and wait. "
                "You cannot CLEAR this gate."
            )

    if stage in {"clarify", "contract"}:
        if requirements:
            board["requirements"] = requirements
        if criteria:
            board["criteria"] = criteria
        holes = contract_defects(
            list(board.get("requirements") or []),
            list(board.get("criteria") or []),
        )
        if holes:
            defects = holes
            verdict = "FAIL"

    if stage == "describe":
        headline = str(args.get("headline") or note or board.get("headline") or "").strip()
        if verdict == "PASS" and not headline:
            defects = ["no object named"]
            verdict = "FAIL"
        if headline:
            board["headline"] = headline
        if args.get("repo"):
            board["repo"] = str(args.get("repo") or "").strip()

    if stage == "plan" and verdict == "PASS" and not sha and not defects:
        sha = _controller_commit_plan(board)
        args["sha"] = sha
    if stage in {"plan", "design", "code"} and verdict == "PASS" and not defects:
        if not sha:
            defects = ["PASS needs the commit SHA — not a pasted file"]
            verdict = "FAIL"
        dump = f"{note}\n{evidence}"
        if _looks_like_source_dump(dump):
            defects = ["do not paste full source into submit; report path + SHA only"]
            verdict = "FAIL"

    if verdict != "PASS" or defects:
        rewind = str(args.get("rewind") or args.get("back") or "").strip().lower()
        return _loop(
            board,
            stage=stage,
            defects=defects or [note or "unnamed failure"],
            sha=sha,
            evidence=evidence,
            rewind=rewind,
        )

    return _clear(board, stage=stage, note=note or evidence or "passed", sha=sha)


def _looks_like_sha(text: str) -> bool:
    token = (text or "").strip()
    return bool(re.fullmatch(r"[0-9a-fA-F]{7,40}", token))


def _looks_like_driven_evidence(text: str) -> bool:
    blob = (text or "").strip()
    if not blob or _looks_like_sha(blob) or "```" in blob:
        return False
    low = blob.lower()
    tokens = set(re.findall(r"[a-z0-9]+", low))
    if tokens <= {
        "screenshot",
        "headed",
        "curl",
        "http",
        "https",
        "file",
        "artifacts",
        "com",
        "www",
        "example",
    }:
        return False

    def _clause(start: str, stop: str) -> str:
        match = re.search(start + r"(.+?)" + stop, low, flags=re.S)
        return match.group(1) if match else ""

    given = _clause(r"\bgiven\b", r"\bwhen\b")
    when = _clause(r"\bwhen\b", r"\bthen\b")
    then = _clause(r"\bthen\b", r"\bfail[- ]if\b")
    fail = re.search(r"\bfail[- ]if\b(.+)", low, flags=re.S)
    structured = all(re.search(r"[a-z0-9#]", part or "") for part in (given, when, then, fail.group(1) if fail else ""))
    artifact = (
        bool(
            re.search(
                r"(?:file://)?(/[^\s]+?\.(?:png|jpe?g|webp|gif|json|txt|html))\b",
                low,
            )
        )
        or "/.sharedagents/sessions/artifacts/" in low
    )
    observed = bool(
        re.search(
            r"\bcurl\b.+\b(?:20\d|30\d|40\d|50\d)\b",
            low,
        )
    )
    return structured or artifact or observed


def _looks_like_source_dump(text: str) -> bool:
    """True when a submit note is a file dump instead of a gate record."""
    body = text or ""
    low = body.lstrip().lower()
    if low.startswith("<!doctype") or low.startswith("<html"):
        return True
    if "<html" in low and "</html>" in low:
        return True
    if "```" in body:
        return True
    if body.count("\n") >= 15 and ("def " in body or "function " in body or "const " in body):
        return True
    return len(body) > 4000


def _controller_commit_plan(board: dict[str, Any]) -> str:
    worktree = str(board.get("worktree") or "").strip()
    if not worktree:
        return ""
    try:
        from sharedagents_web.dev_sdlc import commit_plan_if_uncommitted
    except Exception:  # noqa: BLE001
        return ""
    return commit_plan_if_uncommitted(
        {
            "worktree": worktree,
            "issue_number": board.get("issue_number") or "issue",
        }
    )


def _previous_phase(stage: str, board: dict[str, Any] | None = None) -> str:
    """One gate back on the lock. Used when a definition in the last phase is wrong."""
    if stage not in GATES:
        return PRODUCER.get(stage, stage)
    index = GATES.index(stage)
    if index <= 0:
        return stage
    prev = GATES[index - 1]
    if prev == "design" and (board or {}).get("fuse_plan_design", True):
        return "plan"
    return prev


def _rewind_target(stage: str, rewind: str, board: dict[str, Any] | None = None) -> str:
    """Jarvis names where a bad definition goes: plan, previous phase, or a gate."""
    token = str(rewind or "").strip().lower()
    fused = (board or {}).get("fuse_plan_design", True)
    if token == "design" and fused:
        return "plan"
    if token in {"plan", "clarify", "design", "code", "tests", "contract"}:
        return token
    if token in {"previous", "last", "last-phase", "last_phase"}:
        return _previous_phase(stage, board)
    return PRODUCER.get(stage, stage)


def _loop(board: dict[str, Any], *, stage: str, defects: list[str], sha: str, evidence: str, rewind: str = "") -> str:
    if not defects:
        defects = ["unnamed failure — not a loop, not a pass"]
    back = _rewind_target(stage, rewind, board)
    row = board["gates"][stage]
    row["attempts"] = int(row.get("attempts") or 0) + 1
    fp = _fingerprint(stage, sha, defects)
    if fp and fp == str(row.get("fingerprint") or ""):
        row["no_progress"] = int(row.get("no_progress") or 0) + 1
    else:
        row["no_progress"] = 0
        row["fingerprint"] = fp
    row["sha"] = sha
    row["note"] = "; ".join(defects)[:400]
    row["status"] = "looped"

    bound = int(row["attempts"]) >= MAX_ATTEMPTS or int(row["no_progress"]) >= MAX_NO_PROGRESS
    if bound:
        row["status"] = "blocked"
        board["blocked"] = True
        board["open"] = ""
        why = (
            f"BLOCKED {stage} after {row['attempts']} attempts "
            f"({row['no_progress']} no-progress). Defects: {row['note']}"
        )
        _record(board, stage=stage, verdict="BLOCKED", note=why)
        return why

    _open_gate(board, back)
    board["gates"][back]["status"] = "open"
    why = (
        f"LOOP {stage} → OPEN {back}. Defects: {row['note']}. "
        f"Next child gets the frozen contract plus these defects only."
    )
    if evidence:
        why += f" Evidence: {evidence[:240]}"
    _record(board, stage=stage, verdict="LOOP", note=why)
    return why


def _clear(board: dict[str, Any], *, stage: str, note: str, sha: str) -> str:
    row = board["gates"][stage]
    row["status"] = "cleared"
    row["note"] = note[:400]
    row["sha"] = sha
    row["no_progress"] = 0
    minted = ""
    if stage == "contract":
        minted = _try_mint_issue(board)
    nxt = _next_after(stage, board)
    if stage == "plan" and nxt == "code":
        board["gates"]["design"]["status"] = "cleared"
        board["gates"]["design"]["note"] = "fused into plan"
    if nxt:
        _open_gate(board, nxt)
        verdict = f"CLEAR {stage} — OPEN {nxt}"
    else:
        board["open"] = ""
        verdict = f"CLEAR {stage} — DONE"
    if minted:
        verdict += f" ({minted})"
    _record(board, stage=stage, verdict="CLEAR", note=note or verdict)
    return verdict


__all__ = [
    "GATES",
    "PRODUCER",
    "contract_defects",
    "format_sdlc_md",
    "load_board",
    "sdlc_path_for_session",
    "seed_board_for_run",
    "sync_board_after_ratify",
    "resolve_board_path",
    "set_board_worktree",
    "reopen_tests",
    "apply_review",
    "apply_drive_ready",
    "apply_stage_pass",
    "REVIEW_ROLES",
]
