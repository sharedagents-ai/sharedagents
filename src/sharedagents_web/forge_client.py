"""Thin forge API client for the Dev tab, with Gitea and GitHub backends.

Gitea copied GitHub's issue/PR surface; the same calls work on both. Never
logs the token. Failures return structured errors, not exceptions the UI
cannot show.

Configuration (environment or the web secrets store):

- ``FORGE_KIND`` — ``gitea`` or ``github``. Optional: inferred from
  ``FORGE_URL`` (``github.com`` means GitHub, anything else Gitea).
- ``FORGE_URL`` — the forge's web root, e.g. ``https://git.example.com``.
  GitHub defaults to ``https://github.com``. When unset, it is read from the
  ``origin`` remote of the local source clone.
- ``FORGE_TOKEN`` — API token.
- ``FORGE_API_URL`` — optional API root override (GitHub Enterprise).
- ``FORGE_USER_AGENT`` — optional HTTP ``User-Agent``.

Older names are still read as fallbacks: ``GITEA_BASE`` / ``GITEA_TOKEN`` /
``GITEA_USER_AGENT`` / ``GITEA_SRC`` for Gitea, ``GITHUB_TOKEN`` /
``GH_TOKEN`` / ``GITHUB_BASE`` / ``GITHUB_API`` for GitHub, and
``SHAREDAGENTS_FORGE`` for the kind.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

KINDS = ("gitea", "github")
GITHUB_URL = "https://github.com"
GITHUB_API = "https://api.github.com"
#: Pre-FORGE_* names per backend, still read so existing setups keep working.
_FALLBACK_ENV = {
    "gitea": {"url": ("GITEA_BASE",), "token": ("GITEA_TOKEN",)},
    "github": {"url": ("GITHUB_BASE",), "token": ("GITHUB_TOKEN", "GH_TOKEN")},
}
# Some forges sit behind Cloudflare, which rejects non-browser clients with
# error 1010, so the default looks like a browser. FORGE_USER_AGENT overrides.
_BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
_inferred_remote_base: str | None = None


def _env(*names: str) -> str:
    """The first non-empty value among ``names``."""
    for name in names:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return ""


def _is_github(url: str) -> bool:
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    return host in {"github.com", "www.github.com"}


def _base_from_remote(url: str) -> str:
    """Turn a clone URL into the forge web root (the URL minus ``owner/repo``)."""
    raw = str(url or "").strip()
    if not raw:
        return ""
    if "://" in raw:
        parsed = urllib.parse.urlparse(raw)
        host = (parsed.netloc or "").split("@")[-1]
        parts = [p for p in (parsed.path or "").split("/") if p]
        scheme = "https" if parsed.scheme in {"ssh", "git"} else (parsed.scheme or "https")
    elif ":" in raw:  # scp form: git@host:prefix/owner/repo.git
        host, _, path = raw.partition(":")
        host = host.split("@")[-1]
        parts = [p for p in path.split("/") if p]
        scheme = "https"
    else:
        return ""
    if not host or len(parts) < 2:
        return ""
    return "/".join([f"{scheme}://{host}", *parts[:-2]])


def _remote_url(root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "remote", "get-url", "origin"],
            check=False,
            capture_output=True,
            text=True,
            timeout=4,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (proc.stdout or "").strip() if proc.returncode == 0 else ""


def _remote_base() -> str:
    """The forge web root read from the local source clone's ``origin`` (cached)."""
    global _inferred_remote_base
    if _inferred_remote_base is not None:
        return _inferred_remote_base
    candidates: list[Path] = []
    for raw in (
        _env("FORGE_SRC", "GITEA_SRC"),
        _env("SHAREDAGENTS_SRC"),
        _env("SHAREDAGENTS_CODE_ROOT"),
        str(Path.home() / "src" / "sharedagents"),
        str(Path.cwd()),
    ):
        path = Path(raw).expanduser() if raw else None
        if path and path not in candidates:
            candidates.append(path)
    found = ""
    for root in candidates:
        if not (root / ".git").exists():
            continue
        found = _base_from_remote(_remote_url(root))
        if found:
            break
    _inferred_remote_base = found
    return found


def reset_inferred_remote_base_for_tests() -> None:
    global _inferred_remote_base
    _inferred_remote_base = None


def _config(forge: str) -> tuple[str, str]:
    """``(web root, token)`` for one backend; either may be empty."""
    names = _FALLBACK_ENV[forge]
    token = _env("FORGE_TOKEN", *names["token"])
    url = _env("FORGE_URL", *names["url"]).rstrip("/")
    if not url and token:
        if forge == "github":
            url = GITHUB_URL
        else:
            remote = _remote_base()
            url = "" if _is_github(remote) else remote
    return url, token


def _forced_kind() -> str:
    return _env("FORGE_KIND", "SHAREDAGENTS_FORGE").lower()


def kind() -> str:
    """Active forge backend. Empty string means no usable configuration."""
    forced = _forced_kind()
    if forced:
        return forced if forced in KINDS and all(_config(forced)) else ""
    if _env("FORGE_TOKEN"):
        url = _env("FORGE_URL")
        if not url:
            # Same order as _config(): a legacy per-backend URL beats the clone
            # remote, so a GITEA_BASE setup never sends FORGE_TOKEN to GitHub.
            for forge in KINDS:
                if _env(*_FALLBACK_ENV[forge]["url"]):
                    return forge
            url = _remote_base()
        return "github" if not url or _is_github(url) else "gitea"
    url = _env("FORGE_URL")
    if url:
        # An explicit FORGE_URL names the backend: never pair it with the
        # other backend's legacy token (a GitHub token sent to a Gitea host).
        forge = "github" if _is_github(url) else "gitea"
        return forge if all(_config(forge)) else ""
    for forge in KINDS:
        if all(_config(forge)):
            return forge
    return ""


def configured() -> bool:
    return bool(kind())


def configured_reason() -> str:
    """Why the forge client is off — empty when it is usable."""
    if kind():
        return ""
    forced = _forced_kind()
    if forced and forced not in KINDS:
        return f"FORGE_KIND={forced} is not one of: {', '.join(KINDS)}"
    if forced:
        return f"FORGE_KIND={forced} but FORGE_URL or FORGE_TOKEN is missing"
    url = _env("FORGE_URL")
    if url:
        forge = "github" if _is_github(url) else "gitea"
        names = " or ".join(("FORGE_TOKEN", *_FALLBACK_ENV[forge]["token"]))
        return f"FORGE_URL points at a {forge} forge but no {forge} token ({names}) is set"
    if _config("gitea")[1]:
        return "FORGE_TOKEN is set but FORGE_URL is missing and no local clone remote could supply it"
    return "FORGE_TOKEN is not set"


def forge_url() -> str:
    """Web root of the active forge (``''`` when unconfigured)."""
    forge = kind()
    return _config(forge)[0] if forge else ""


def forge_token() -> str:
    forge = kind()
    return _config(forge)[1] if forge else ""


def html_base() -> str:
    return forge_url()


def _api_root() -> str:
    if kind() == "github":
        url = forge_url()
        return _env("FORGE_API_URL", "GITHUB_API").rstrip("/") or (GITHUB_API if _is_github(url) else url + "/api/v3")
    return _env("FORGE_API_URL").rstrip("/") or (forge_url() + "/api/v1")


def _headers() -> dict[str, str]:
    token = forge_token()
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": _env("FORGE_USER_AGENT", "GITEA_USER_AGENT") or _BROWSER_UA,
    }
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def _page_size_key() -> str:
    return "per_page" if kind() == "github" else "limit"


def _request(method: str, path: str, body: dict | None = None, timeout: float = 12.0) -> tuple[int, Any]:
    url = _api_root() + path
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method, headers=_headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8") or "null"
            payload: Any
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {"raw": raw[:500]}
            return int(resp.status), payload
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")[:500]
        log.warning("forge %s %s -> %s", method, path, exc.code)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"error": raw or exc.reason}
        return int(exc.code), payload
    except Exception as exc:  # noqa: BLE001 — Dev tab must degrade, not 500
        log.warning("forge %s %s failed: %s", method, path, type(exc).__name__)
        return 0, {"error": type(exc).__name__}


def split_repo(repo: str) -> tuple[str, str]:
    parts = (repo or "").strip().strip("/").split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("repo must be owner/name")
    return parts[0], parts[1]


def list_pulls(repo: str, *, state: str = "open") -> tuple[int, list[dict]]:
    owner, name = split_repo(repo)
    want = (state or "open").strip().lower()
    if want not in {"open", "closed", "all"}:
        want = "open"
    status, payload = _request_pages(
        f"/repos/{owner}/{name}/pulls",
        params={"state": want},
    )
    out = []
    for pr in payload:
        if not isinstance(pr, dict):
            continue
        head = pr.get("head") or {}
        out.append(
            {
                "number": pr.get("number"),
                "title": pr.get("title") or "",
                "html_url": pr.get("html_url") or pr.get("url") or "",
                "state": pr.get("state") or "",
                "merged": bool(pr.get("merged")),
                "draft": bool(pr.get("draft")),
                "head": (head.get("ref") if isinstance(head, dict) else "") or "",
                "sha": (head.get("sha") if isinstance(head, dict) else "") or "",
                "user": ((pr.get("user") or {}).get("login") if isinstance(pr.get("user"), dict) else ""),
                "updated_at": pr.get("updated_at") or "",
            }
        )
    return status, out


def get_branch(repo: str, branch: str) -> tuple[int, dict]:
    owner, name = split_repo(repo)
    enc = urllib.parse.quote(branch, safe="")
    status, payload = _request("GET", f"/repos/{owner}/{name}/branches/{enc}")
    if status != 200 or not isinstance(payload, dict):
        return status, {}
    commit = payload.get("commit") or {}
    sha = ""
    if isinstance(commit, dict):
        sha = str(commit.get("id") or commit.get("sha") or "")
    return status, {"name": payload.get("name") or branch, "sha": sha}


def commit_status(repo: str, sha: str) -> tuple[int, dict]:
    if not sha:
        return 0, {"state": "", "total_count": 0}
    owner, name = split_repo(repo)
    if kind() == "github":
        status, payload = _request("GET", f"/repos/{owner}/{name}/commits/{sha}/status")
        if status != 200 or not isinstance(payload, dict):
            return status, {"state": "", "total_count": 0}
        return status, {
            "state": str(payload.get("state") or ""),
            "total_count": int(payload.get("total_count") or 0),
        }
    status, payload = _request("GET", f"/repos/{owner}/{name}/statuses/{sha}")
    if status != 200:
        return status, {"state": "", "total_count": 0}
    states = []
    if isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict) and row.get("status"):
                states.append(str(row["status"]))
    # Gitea: success / pending / failure / error / warning
    if any(s in ("failure", "error") for s in states):
        combined = "failure"
    elif states and all(s == "success" for s in states):
        combined = "success"
    elif any(s == "pending" for s in states):
        combined = "pending"
    else:
        combined = states[0] if states else ""
    return status, {"state": combined, "total_count": len(states)}


def create_pull(repo: str, *, title: str, head: str, body: str, base: str = "main") -> tuple[int, dict]:
    owner, name = split_repo(repo)
    status, payload = _request(
        "POST",
        f"/repos/{owner}/{name}/pulls",
        {"title": title, "head": head, "base": base, "body": body},
    )
    if not isinstance(payload, dict):
        payload = {"error": "bad_payload"}
    return status, payload


def issue_html_url(repo: str, number: int | str) -> str:
    owner, name = split_repo(repo)
    return f"{html_base()}/{owner}/{name}/issues/{int(number)}"


def _issue_record(repo: str, payload: dict) -> dict:
    number = payload.get("number")
    url = str(payload.get("html_url") or payload.get("url") or "")
    if number and not url:
        url = issue_html_url(repo, number)
    labels = []
    raw_labels = payload.get("labels") or []
    if isinstance(raw_labels, list):
        for lab in raw_labels:
            if isinstance(lab, dict) and lab.get("name"):
                labels.append(str(lab["name"]))
            elif isinstance(lab, str) and lab.strip():
                labels.append(lab.strip())
    user = payload.get("user")
    login = ""
    if isinstance(user, dict):
        login = str(user.get("login") or user.get("username") or "")
    return {
        "number": number,
        "html_url": url,
        "title": payload.get("title") or "",
        "state": payload.get("state") or "",
        "body": payload.get("body") or "",
        "labels": labels,
        "user": login,
        "updated_at": payload.get("updated_at") or "",
        "pull_request": bool(payload.get("pull_request")),
    }


def _repo_record(payload: dict) -> dict:
    owner = payload.get("owner") if isinstance(payload.get("owner"), dict) else {}
    full = str(payload.get("full_name") or "").strip()
    if not full:
        login = str(owner.get("login") or owner.get("username") or "")
        name = str(payload.get("name") or "")
        if login and name:
            full = f"{login}/{name}"
    html = str(payload.get("html_url") or payload.get("clone_url") or "")
    if full and not html:
        html = f"{html_base()}/{full}"
    return {
        "full_name": full,
        "name": str(payload.get("name") or full.split("/")[-1] if full else ""),
        "owner": str(owner.get("login") or owner.get("username") or (full.split("/")[0] if "/" in full else "")),
        "html_url": html,
        "private": bool(payload.get("private")),
        "fork": bool(payload.get("fork")),
        "updated_at": payload.get("updated_at") or "",
    }


def _request_pages(
    path: str,
    *,
    params: dict[str, Any] | None = None,
    item_limit: int = 400,
) -> tuple[int, list[Any]]:
    """Paginate a forge list endpoint. Returns last status + accumulated rows."""
    page = 1
    out: list[Any] = []
    last_status = 0
    params = dict(params or {})
    size_key = _page_size_key()
    while len(out) < item_limit:
        q = dict(params)
        q["page"] = page
        q[size_key] = min(50, item_limit - len(out))
        status, payload = _request("GET", f"{path}?{urllib.parse.urlencode(q)}")
        last_status = status
        if status != 200 or not isinstance(payload, list):
            return status, out
        if not payload:
            break
        out.extend(payload)
        if len(payload) < int(q[size_key]):
            break
        page += 1
    return last_status, out


def list_accessible_repos() -> tuple[int, list[dict]]:
    """Repos the token can see: user-owned plus every org membership."""
    status, owned = _request_pages("/user/repos")
    rows: list[dict] = []
    seen: set[str] = set()
    for raw in owned:
        if not isinstance(raw, dict):
            continue
        rec = _repo_record(raw)
        key = rec.get("full_name") or ""
        if key and key not in seen:
            seen.add(key)
            rows.append(rec)
    org_status, orgs = _request_pages("/user/orgs", item_limit=100)
    if org_status == 200:
        for org in orgs:
            if not isinstance(org, dict):
                continue
            name = str(org.get("username") or org.get("login") or "").strip()
            if not name:
                continue
            _st, org_repos = _request_pages(f"/orgs/{urllib.parse.quote(name)}/repos")
            for raw in org_repos:
                if not isinstance(raw, dict):
                    continue
                rec = _repo_record(raw)
                key = rec.get("full_name") or ""
                if key and key not in seen:
                    seen.add(key)
                    rows.append(rec)
    rows.sort(key=lambda r: (str(r.get("updated_at") or ""), str(r.get("full_name") or "")), reverse=True)
    # Prefer a 200 from /user/repos; empty+error still surfaces that status.
    return status, rows


def list_issues(repo: str, *, state: str = "open") -> tuple[int, list[dict]]:
    """List issues (not PRs) for a repo. ``state`` is open / closed / all."""
    owner, name = split_repo(repo)
    want = (state or "open").strip().lower()
    if want not in {"open", "closed", "all"}:
        want = "open"
    params: dict[str, Any] = {"state": want}
    if kind() != "github":
        params["type"] = "issues"
    status, payload = _request_pages(
        f"/repos/{owner}/{name}/issues",
        params=params,
    )
    out = []
    for row in payload:
        if not isinstance(row, dict) or row.get("pull_request"):
            continue
        rec = _issue_record(repo, row)
        rec.pop("body", None)  # list payload stays small; detail fetches body
        rec.pop("pull_request", None)
        out.append(rec)
    return status, out


def get_issue(repo: str, number: int | str) -> tuple[int, dict]:
    owner, name = split_repo(repo)
    try:
        idx = int(number)
    except (TypeError, ValueError):
        return 400, {"error": "invalid_issue_number"}
    status, payload = _request("GET", f"/repos/{owner}/{name}/issues/{idx}")
    if status != 200 or not isinstance(payload, dict):
        return status, payload if isinstance(payload, dict) else {}
    rec = _issue_record(repo, payload)
    if rec.get("pull_request"):
        return 400, {"error": "not_an_issue", "number": rec.get("number")}
    rec.pop("pull_request", None)
    return status, rec


def kind_from_labels(labels: list[str] | None) -> str:
    names = {str(x).strip().lower() for x in (labels or [])}
    if "type/bug" in names or "bug" in names:
        return "fix"
    if "type/feature" in names or "enhancement" in names:
        return "feature"
    return "feature"


def find_open_issue(repo: str, *, title: str) -> tuple[int, dict]:
    """Return the first open issue whose title matches exactly."""
    owner, name = split_repo(repo)
    want = (title or "").strip()
    if not want:
        return 0, {}
    q = urllib.parse.urlencode(
        {"state": "open", _page_size_key(): 50}
        if kind() == "github"
        else {"state": "open", "type": "issues", "limit": 50}
    )
    status, payload = _request("GET", f"/repos/{owner}/{name}/issues?{q}")
    if status != 200 or not isinstance(payload, list):
        return status, {}
    for row in payload:
        if not isinstance(row, dict) or row.get("pull_request"):
            continue
        if str(row.get("title") or "").strip() == want:
            return status, _issue_record(repo, row)
    return status, {}


def create_issue(
    repo: str,
    *,
    title: str,
    body: str,
    labels: list[str] | None = None,
) -> tuple[int, dict]:
    owner, name = split_repo(repo)
    # Gitea's create-issue endpoint takes label **ids**. Names are accepted and
    # then silently dropped, so a brand-new repo -- which has none of these
    # labels yet -- opened every issue untagged while the code read as if it
    # had tagged them. Mint the ids first, exactly as `add_issue_labels` does.
    ids = [lid for lid in (ensure_label(repo, one) for one in (labels or []) if one) if lid]
    payload_body: dict[str, Any] = {"title": title, "body": body}
    if ids:
        payload_body["labels"] = ids
    status, payload = _request("POST", f"/repos/{owner}/{name}/issues", payload_body)
    if status in (400, 422) and ids:
        return create_issue(repo, title=title, body=body, labels=None)
    if not isinstance(payload, dict):
        return status, {}
    if status not in (200, 201):
        return status, payload if isinstance(payload, dict) else {}
    return status, _issue_record(repo, payload)


# ---------------------------------------------------------------------------
# Forge writes used by the Dev-tab SDLC pipeline.
#
# Every stage of a pipeline run lands on the issue, so these must degrade the
# same way the read paths do: a forge outage returns a falsey result and the
# caller keeps going, it never raises into a request handler.
# ---------------------------------------------------------------------------

#: Colour per ``sdlc/`` tag so a glance at the issue list shows how far a run
#: got.  Gitea creates a label on first use; the colour is only set then.
_SDLC_LABEL_COLORS = {
    "sdlc/intake": "#c5def5",
    "sdlc/spec": "#bfd4f2",
    "sdlc/code": "#d4c5f9",
    "sdlc/tests": "#c2e0c6",
    "sdlc/deploy": "#fef2c0",
    "sdlc/pr": "#0e8a16",
    "sdlc/merged": "#5319e7",
    "sdlc/archived": "#6e7781",
    "sdlc/deployed": "#1d76db",
    "sdlc/blocked": "#e11d21",
}


def comment_issue(repo: str, number: int | str, body: str) -> tuple[int, dict]:
    """Post one comment on an issue. Returns ``(status, comment)``."""
    owner, name = split_repo(repo)
    status, payload = _request(
        "POST",
        f"/repos/{owner}/{name}/issues/{number}/comments",
        {"body": body},
    )
    return status, payload if isinstance(payload, dict) else {}


def list_labels(repo: str) -> tuple[int, list[dict]]:
    owner, name = split_repo(repo)
    status, payload = _request("GET", f"/repos/{owner}/{name}/labels?limit=200")
    return status, payload if isinstance(payload, list) else []


def ensure_label(repo: str, label: str) -> int | None:
    """Return the id of *label*, creating it if the repo does not have it yet."""
    wanted = (label or "").strip()
    if not wanted:
        return None
    owner, name = split_repo(repo)
    _status, existing = list_labels(repo)
    for row in existing:
        if isinstance(row, dict) and str(row.get("name") or "") == wanted:
            try:
                return int(row.get("id"))
            except (TypeError, ValueError):
                return None
    status, created = _request(
        "POST",
        f"/repos/{owner}/{name}/labels",
        {
            "name": wanted,
            "color": _SDLC_LABEL_COLORS.get(wanted, "#ededed"),
            "description": "sharedagents Dev-tab SDLC pipeline",
        },
    )
    if status in (200, 201) and isinstance(created, dict):
        try:
            return int(created.get("id"))
        except (TypeError, ValueError):
            return None
    return None


def add_issue_labels(repo: str, number: int | str, labels: list[str]) -> bool:
    """Attach labels by name, creating any the repo lacks.

    Gitea's issue-label endpoint takes label **ids**, not names; posting names
    is accepted by some versions and silently ignored by others, which is how a
    pipeline ends up with untagged issues that look tagged in the code.
    """
    ids = [lid for lid in (ensure_label(repo, name) for name in labels if name) if lid]
    if not ids:
        return False
    owner, name = split_repo(repo)
    status, _payload = _request(
        "POST",
        f"/repos/{owner}/{name}/issues/{number}/labels",
        {"labels": ids},
    )
    return status in (200, 201)


def close_issue(repo: str, number: int | str) -> tuple[int, dict]:
    """Close a forge issue. Idempotent if it is already closed."""
    owner, name = split_repo(repo)
    status, payload = _request(
        "PATCH",
        f"/repos/{owner}/{name}/issues/{int(number)}",
        {"state": "closed"},
    )
    return status, payload if isinstance(payload, dict) else {}


def delete_branch(repo: str, branch: str) -> tuple[int, dict]:
    """Delete a forge branch. 204/404 both mean it is gone."""
    owner, name = split_repo(repo)
    enc = urllib.parse.quote(str(branch or "").strip(), safe="")
    if not enc:
        return 400, {"error": "branch is required"}
    status, payload = _request("DELETE", f"/repos/{owner}/{name}/branches/{enc}")
    return status, payload if isinstance(payload, dict) else {}


def get_pull(repo: str, number: int | str) -> tuple[int, dict]:
    owner, name = split_repo(repo)
    status, payload = _request("GET", f"/repos/{owner}/{name}/pulls/{number}")
    return status, payload if isinstance(payload, dict) else {}


def merge_pull(
    repo: str,
    number: int | str,
    *,
    method: str = "merge",
    title: str = "",
    message: str = "",
    head_sha: str = "",
    delete_branch_after_merge: bool = True,
) -> tuple[int, dict]:
    """Merge a pull request.

    Gitea's merge body is **CamelCase**: ``Do`` / ``MergeTitleField`` /
    ``MergeMessageField``.  snake_case or a lowercase ``do`` returns a
    ``[Do]: Required`` 405 that reads exactly like a branch-protection block,
    so the field names here are load-bearing, not style.

    ``head_sha`` binds the merge to the commit the checks were run against.
    Without it the candidate is merely SAMPLED: a force-push between the check
    and the merge means something else ships, and every assurance gathered
    about the old head silently describes content nobody reviewed.
    Gitea rejects the merge if the head has moved.
    """
    owner, name = split_repo(repo)
    do = method if method in ("merge", "rebase", "rebase-merge", "squash") else "merge"
    body: dict[str, Any] = {"Do": do}
    if delete_branch_after_merge:
        body["delete_branch_after_merge"] = True
    if head_sha:
        body["head_commit_id"] = str(head_sha)
    if title:
        body["MergeTitleField"] = title
    if message:
        body["MergeMessageField"] = message
    status, payload = _request(
        "POST",
        f"/repos/{owner}/{name}/pulls/{number}/merge",
        body,
    )
    return status, payload if isinstance(payload, dict) else {}
