"""Closed-loop contract for the Dev tab (Devin-shaped, forge-backed).

Inner laps (commits, tests, steers) are free. Publish is once: one branch,
one PR, human merge. A retry that opens a second PR is an open loop.

Playbooks: plan/usage-flows/dev.md
(create / continue / forge overlay / no merge / no Tasks board).
"""

from __future__ import annotations

import re
from typing import Any

# Every list/detail/create/continue payload MUST carry these keys. A fast path
# or overlay that omits one is an open loop.
PROJECTION_KEYS = (
    "id",
    "object",
    "done_when",
    "repo",
    "branch",
    "kind",
    "agent",
    "session_id",
    "state",
    "inner_laps",
    "pr_number",
    "pr_url",
    "ci_status",
    "last_sha",
    "session_live",
    "forge_configured",
    "live_pr",
    "issue_number",
    "issue_url",
)

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def project_loop(
    row: dict[str, Any],
    *,
    session_live: bool = False,
    forge_configured: bool | None = None,
) -> dict[str, Any]:
    """Fill the contract keys. Extra fields (last_error, …) may remain."""
    out = dict(row)
    out["session_live"] = bool(session_live)
    if forge_configured is not None:
        out["forge_configured"] = bool(forge_configured)
    else:
        out.setdefault("forge_configured", False)
    out.setdefault("live_pr", None)
    out.setdefault("pr_number", None)
    out.setdefault("pr_url", "")
    out.setdefault("ci_status", "")
    out.setdefault("last_sha", "")
    out.setdefault("inner_laps", 0)
    out.setdefault("session_id", "")
    out.setdefault("state", "inner")
    out.setdefault("issue_number", None)
    out.setdefault("issue_url", "")
    return out


def has_issue(row: dict[str, Any] | None) -> bool:
    """A Dev loop does not exist until it has a forge issue."""
    if not row:
        return False
    try:
        number = int(row.get("issue_number") or 0)
    except (TypeError, ValueError):
        number = 0
    url = str(row.get("issue_url") or "").strip()
    return number > 0 and url.startswith("http")


def tracker_preamble(*, loop_id: str, issue_url: str = "", issue_number: str | int = "") -> str:
    """Machine-readable header that lands in the session JSONL as the user prompt.

    Every Dev-loop turn SHALL start with DEV_ID and ISSUE so the log links
    the driving conversation to the forge issue and the Dev card.
    """
    loop_id = str(loop_id or "").strip() or "unset"
    url = str(issue_url or "").strip()
    if not url and str(issue_number or "").strip():
        url = f"#{issue_number}"
    if not url:
        url = "unset"
    return f"TRACKER\nDEV_ID: {loop_id}\nISSUE: {url}\n\n"


def branch_name(kind: str, object_name: str) -> str:
    """One branch per object. No suffixes. ``fix/`` vs ``feature/``."""
    raw = (object_name or "").strip().lower()
    slug = _SLUG_RE.sub("-", raw).strip("-")[:40] or "loop"
    prefix = "fix" if str(kind).strip().lower() == "fix" else "feature"
    return f"{prefix}/{slug}"


def closed_loop_prompt(
    *,
    object_name: str,
    done_when: str,
    repo: str,
    branch: str,
    kind: str = "feature",
    pr_url: str = "",
    issue_number: str = "",
    issue_url: str = "",
    loop_id: str = "",
) -> str:
    """Devin-like playbook the session runs. Intent, not today's HTTP surface."""
    is_fix = str(kind).strip().lower() == "fix"
    playbook = "FIX" if is_fix else "FEATURE"
    issue_type = "type/bug" if is_fix else "type/feature"
    cited = str(issue_number or "").strip()
    issue_cite = (
        f"Tracker: forge issue #{cited}"
        + (f" ({issue_url})" if issue_url else "")
        + ". Reuse it. Never open another issue."
        if cited or issue_url
        else f"Open ONE forge issue in `{repo}` (or reuse a matching open one). "
        f"Title = OBJECT. Body = DONE WHEN as acceptance. "
        f"Labels: {issue_type}, status/in-progress. Never a second issue."
    )
    if pr_url:
        publish = (
            f"PR already exists: {pr_url}\nPush more commits to this branch. Do not open another PR. Do not merge."
        )
    else:
        publish = (
            "When DONE WHEN is true, open ONE pull request against main. "
            "PR body: `Fixes #<issue>`. Commits: `Refs #<issue>` only "
            "(never Fixes on inner-lap commits). Do not merge. "
            "Do not open a second PR if this session is retried."
        )
    if is_fix:
        procedure = """PLAYBOOK FIX:
1. SETUP — worktree on this branch (reuse if it exists). The TRACKER issue is already minted; cite it. Post a short plan on the session before any edit.
2. REPRODUCE the failure. If you cannot, report blocked and stop. Do not invent a different bug.
3. PIN with a regression that fails on that reproduction (when a test seam exists).
4. FIX only what the reproduction needs. No new features. No drive-by refactors.
5. PROVE the new traced reproduction test and only the smallest checks for the
   affected requirements/functions pass. Do not run a broadly "relevant" or
   full suite; deployment/CI owns integration and regression.
6. PUBLISH — one PR, then wait. You merge nothing."""
    else:
        procedure = """PLAYBOOK FEATURE:
1. SETUP — worktree on this branch (reuse if it exists). The TRACKER issue is already minted; cite it.
2. PLAN against DONE WHEN. Name files and checks. Post the plan on the session before any edit.
3. BUILD only the named object. Unrelated bugs are a separate FIX loop, not this PR.
4. PROVE the checks named in DONE WHEN. Do not open a PR until they pass. No test, no PR.
5. SELF-CHECK for leftover debug / edge cases, then PUBLISH — one PR. You merge nothing."""
    body = f"""You are running a Devin-style closed loop on forge repo `{repo}`.

PLAYBOOK: {playbook}
OBJECT: {object_name}
BRANCH: `{branch}`
DONE WHEN (the only exit):
{done_when}

{issue_cite}

{procedure}

LOOP:
- Inner laps are free: edit, test, commit, push on `{branch}` as many times as needed.
- One branch. Reuse it. Never suffix (_2, -b). Worktree: `~/src/<repo>/.worktrees/{branch}`. Never edit local main.
- {publish}
- CI red or review comments: same session, same branch, same issue. Read the logs, push a fix, comment what changed.
- Do not create Tasks-board rows.

Start: setup the worktree, post the plan, then run the playbook until DONE WHEN is true.
"""
    return (
        tracker_preamble(
            loop_id=loop_id,
            issue_url=issue_url,
            issue_number=issue_number,
        )
        + body
    )


def continue_prompt(
    *,
    object_name: str,
    done_when: str,
    branch: str,
    reason: str,
    kind: str = "feature",
    loop_id: str = "",
    issue_url: str = "",
    issue_number: str = "",
) -> str:
    playbook = "FIX" if str(kind).strip().lower() == "fix" else "FEATURE"
    body = f"""Wake the SAME Devin-style {playbook} session. Do not start a new loop.

OBJECT: {object_name}
BRANCH: `{branch}` — push here only; no new PR; no new issue; do not merge
DONE WHEN: {done_when}

WAKE: {reason}

This is an inner lap (CI / review / steer). Read the failure, fix on this branch, push, comment on the existing issue/PR. Stop when DONE WHEN is true or you are blocked.
"""
    return (
        tracker_preamble(
            loop_id=loop_id,
            issue_url=issue_url,
            issue_number=issue_number,
        )
        + body
    )
