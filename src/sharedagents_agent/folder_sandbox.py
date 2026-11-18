"""Kernel-enforced folder confinement for user-directed execution subprocesses.

A Web user's turn whose grants limit ``folders`` runs its host-acting tools
(shell, file read/write/edit/search, MCP and anything unknown) through
bubblewrap with only the granted folder binds, a private scratch home, and the
minimum runtime paths needed to run Python. sharedagents's own tools (todos,
skills, delegation, the web tools) run outside: they touch sharedagents state
through sharedagents code, never a user-chosen host path, and a delegated child's
host tools are confined again through its own turn's pin. No stamp, an
unrestricted stamp, or All folders launches plainly. If bubblewrap is missing
or the policy cannot be applied, a confined spawn is refused instead of
falling open.
"""

from __future__ import annotations

import os
import shutil
import sys
import sysconfig
from collections.abc import Mapping, Sequence
from pathlib import Path

from sharedagents_core.access_grants import AccessStamp
from sharedagents_core.folder_policy import FolderPolicyError, normalize_folder_grants

_SECRET_MARKERS = (
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "PASSWD",
    "API_KEY",
    "APIKEY",
    "CREDENTIAL",
    "PRIVATE_KEY",
    "AWS_",
    "BEARER",
    "AUTHORIZATION",
    "GITEA",
    "OPENAI",
    "ANTHROPIC",
    "DASHBOARD_AUTH",
)
_KEEP_EXACT = frozenset(
    {
        "PATH",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TERM",
        "USER",
        "LOGNAME",
        "TMPDIR",
        "TZ",
        "PYTHONPATH",
        "PYTHONUNBUFFERED",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "HOME",
        "SHAREDAGENTS_ACCESS_PIN",
        "SHAREDAGENTS_SESSION_ID",
        "SHAREDAGENTS_SESSIONS_DIR",
        "SHAREDAGENTS_AGENT_HOME",
        "SHAREDAGENTS_AGENT_NAME",
        "SHAREDAGENTS_AGENT_PROFILE",
        "SHAREDAGENTS_PERMISSION_MODE",
        "SHAREDAGENTS_WORKSPACE_ORIGIN",
        "SHAREDAGENTS_WORKSPACE_CWD",
        "SHAREDAGENTS_UPSTREAM_ENDPOINT",
        "SHAREDAGENTS_UPSTREAM_MODEL",
        "SHAREDAGENTS_FOLDER_SANDBOX",
    }
)
# System software only: never an application prefix such as /opt, whose
# data a read-only mount would still disclose.
_RUNTIME_FIXED = (
    "/usr",
    "/lib",
    "/lib64",
    "/bin",
    "/sbin",
    "/etc/ssl",
    "/etc/pki",
    "/etc/alternatives",
)

SCRATCH_HOME = "/tmp/sharedagents-scratch"

# sharedagents's own tools: they read and write sharedagents state (todos, skills, the
# child session of a delegation) through sharedagents code, or only reach the
# network, so they run unconfined. Every other tool — including MCP tools and
# ones added later — runs in the sandbox (default deny).
UNCONFINED_TOOLS = frozenset(
    {
        "todowrite",
        "Skill",
        "compact_context",
        "ExitPlanMode",
        "AgentSync",
        "AgentAsync",
        "BatchAgent",
        "webfetch",
        "websearch",
        "x_tweets",
        "meeting_search",
        "decision",
        # First-party state capabilities: no model-supplied filesystem paths.
        # The ontology backend derives its private namespace from pinned grants.
        "ontology_search",
        "ontology_source",
        "ontology_update",
    }
)


class SandboxRefused(RuntimeError):
    """A restricted turn cannot start an execution subprocess."""


def needs_sandbox(stamp: AccessStamp | None, tool: str | None = None) -> bool:
    """Whether ``tool`` (``None`` = any) runs confined under ``stamp``."""
    if stamp is None or not stamp.restricted or stamp.grants.folders is None:
        return False
    return tool not in UNCONFINED_TOOLS


def _secret_key(key: str) -> bool:
    upper = key.upper()
    if upper in _KEEP_EXACT:
        return False
    return any(mark in upper for mark in _SECRET_MARKERS)


def narrow_env(env: Mapping[str, str]) -> dict[str, str]:
    """Drop credentials and unrelated host environment from a child."""
    out = {}
    for key, value in env.items():
        if value is None:
            continue
        if _secret_key(key) or key == "SHAREDAGENTS_ACP_MCP_SERVERS":
            continue
        if key in _KEEP_EXACT or key.startswith("LC_") or key.startswith("SHAREDAGENTS_"):
            out[key] = str(value)
    out["HOME"] = SCRATCH_HOME
    out["TMPDIR"] = "/tmp"
    out["SHAREDAGENTS_FOLDER_SANDBOX"] = "1"
    return out


def _existing_dir(path: str) -> str | None:
    """Keep the path the kernel looks up; do not collapse /lib64 → /usr/lib64."""
    p = Path(path)
    try:
        if p.is_dir():
            return str(p)
    except OSError:
        return None
    return None


def _symlink_hop_dirs(path: str) -> list[str]:
    """``bin`` and install root of every hop the interpreter symlink takes.

    A uv venv's ``bin/python`` points through a version-alias directory link
    (``cpython-3.12-…`` → ``cpython-3.12.13-…``) and Python derives its stdlib
    from that alias path; binding only the resolved install leaves the alias
    dangling inside the sandbox.
    """
    dirs: list[str] = []
    p = Path(path)
    for _ in range(40):
        if not p.is_symlink():
            break
        target = Path(os.readlink(p))
        p = target if target.is_absolute() else p.parent / target
        # /bin/python3 -> python3.12 would otherwise add "/" itself.
        dirs += [str(d) for d in (p.parent, p.parent.parent) if d != d.parent]
    return dirs


def _exposes_home(path: str, home: str) -> bool:
    """True for "/", the operator home, or any directory that contains it."""
    p = Path(path)
    if p == p.parent:
        return True
    if not home:
        return False
    h = Path(home)
    return p == h or p in h.parents


def runtime_ro_paths(env: Mapping[str, str] | None = None) -> list[str]:
    """Host directories the interpreter needs, not the operator home."""
    paths: list[str] = []
    for raw in _RUNTIME_FIXED:
        hit = _existing_dir(raw)
        if hit:
            paths.append(hit)
    for raw in (
        sys.prefix,
        sys.base_prefix,
        sys.exec_prefix,
        str(Path(sys.executable).resolve().parent),
        str(Path(sys.executable).resolve().parent.parent),
        *_symlink_hop_dirs(sys.executable),
    ):
        hit = _existing_dir(raw)
        if hit:
            paths.append(hit)
    for key in ("stdlib", "platstdlib", "purelib", "platlib"):
        try:
            hit = _existing_dir(sysconfig.get_path(key))
        except (KeyError, TypeError, ValueError):
            hit = None
        if hit:
            paths.append(hit)
    env = env or {}
    for part in str(env.get("PYTHONPATH") or "").split(os.pathsep):
        part = part.strip()
        if not part or part in {".", ".."}:
            continue
        if not Path(part).is_absolute():
            continue
        # Only sharedagents's own source: another project on the path is not the
        # runner's to show.
        if not (Path(part) / "sharedagents_agent").is_dir():
            continue
        hit = _existing_dir(part)
        if hit:
            paths.append(hit)
    # Unique, keep order. Never bind the operator home as a whole.
    try:
        home = str(Path.home().resolve())
    except (OSError, RuntimeError):
        home = ""
    seen: set[str] = set()
    out: list[str] = []
    for path in paths:
        if path in seen or _exposes_home(path, home):
            continue
        seen.add(path)
        out.append(path)
    return out


def _bwrap() -> str:
    path = shutil.which("bwrap")
    if not path:
        raise SandboxRefused("folder isolation requires bubblewrap (bwrap); refusing to run without a kernel sandbox")
    return path


def sandbox_argv(
    argv: Sequence[str],
    env: Mapping[str, str],
    stamp: AccessStamp | None,
    *,
    tool: str | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Prefix ``argv`` with bubblewrap when ``tool`` runs confined under ``stamp``."""
    if not needs_sandbox(stamp, tool):
        return list(argv), dict(env)
    try:
        grants = normalize_folder_grants(stamp.grants.folders or ())
    except FolderPolicyError as exc:
        raise SandboxRefused(f"folder policy is invalid: {exc}") from exc
    bwrap = _bwrap()
    child_env = narrow_env(env)
    cmd = [
        bwrap,
        "--die-with-parent",
        "--unshare-pid",
        "--unshare-uts",
        "--unshare-ipc",
        # No network: unauthenticated LAN services and local MCP servers over
        # HTTP would act outside the granted folders.
        "--unshare-net",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--dir",
        SCRATCH_HOME,
        "--chdir",
        SCRATCH_HOME,
    ]
    for path in runtime_ro_paths(env):
        cmd.extend(["--ro-bind", path, path])
    for raw in (
        "/etc/passwd",
        "/etc/group",
        "/etc/nsswitch.conf",
        "/etc/hosts",
        "/etc/resolv.conf",
        "/etc/ld.so.cache",
        "/etc/localtime",
    ):
        p = Path(raw)
        if p.exists() and not p.is_dir():
            cmd.extend(["--ro-bind", raw, raw])
    for grant in grants:
        flag = "--bind" if grant.mode == "rw" else "--ro-bind"
        cmd.extend([flag, grant.path, grant.path])
    cmd.extend(["--", *argv])
    return cmd, child_env


def spawn_stamp(session_id: str, sessions_dir: str = "") -> AccessStamp | None:
    """The running turn's pin, else the session sidecar, else unrestricted."""
    from sharedagents_session.access_binding import session_access_stamp

    sid = str(session_id or "")
    if not sid:
        return None
    stamp = session_access_stamp(sessions_dir or os.environ.get("SHAREDAGENTS_SESSIONS_DIR") or "", sid)
    return stamp
