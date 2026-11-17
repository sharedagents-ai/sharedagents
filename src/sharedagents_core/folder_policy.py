"""Validate per-user folder grants against approved host roots.

Admins pick canonical absolute paths from ``SHAREDAGENTS_FOLDER_GRANT_ROOTS``
(path-separated). When that is unset, ``~/src`` is the only default root if
it exists. ``/``, the operator home, and system directories are never
grantable. Roots must not overlap (a parent write grant cannot hide a child
read-only/no-access entry).
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from sharedagents_core.access_grants import FolderGrant

ENV_ROOTS = "SHAREDAGENTS_FOLDER_GRANT_ROOTS"
_BLOCKED_NAMES = frozenset(
    {
        "/",
        "/etc",
        "/usr",
        "/bin",
        "/sbin",
        "/lib",
        "/lib64",
        "/var",
        "/root",
        "/boot",
        "/proc",
        "/sys",
        "/dev",
        "/run",
        "/tmp",
        "/home",
    }
)


class FolderPolicyError(ValueError):
    """A folder grant the Users tab must refuse."""


def approved_roots() -> tuple[Path, ...]:
    raw = str(os.environ.get(ENV_ROOTS) or "").strip()
    if raw:
        out = []
        for part in raw.split(os.pathsep):
            text = part.strip()
            if not text:
                continue
            path = Path(text).expanduser()
            if path.is_absolute():
                out.append(path.resolve())
        return tuple(dict.fromkeys(out))
    src = (Path.home() / "src").resolve()
    return (src,) if src.is_dir() else ()


def _blocked(path: Path) -> bool:
    text = str(path)
    if text in _BLOCKED_NAMES:
        return True
    try:
        if path == Path.home().resolve():
            return True
    except (OSError, RuntimeError):
        return True
    return False


def _under_root(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _overlaps(a: Path, b: Path) -> bool:
    return a == b or a in b.parents or b in a.parents


def normalize_folder_grants(
    folders: Iterable[FolderGrant] | None,
    *,
    roots: Iterable[Path] | None = None,
) -> tuple[FolderGrant, ...]:
    """Return canonical non-overlapping grants, or raise ``FolderPolicyError``."""
    if folders is None:
        return ()
    allowed = tuple(roots) if roots is not None else approved_roots()
    cleaned: list[FolderGrant] = []
    resolved: list[Path] = []
    for grant in folders:
        raw = str(grant.path or "").strip()
        if not raw:
            raise FolderPolicyError("folder path is required")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            raise FolderPolicyError(f"{raw!r} must be an absolute path")
        try:
            path = path.resolve()
        except (OSError, RuntimeError) as exc:
            raise FolderPolicyError(f"cannot resolve {raw!r}") from exc
        if not path.is_dir():
            raise FolderPolicyError(f"{str(path)!r} must be an existing directory")
        if _blocked(path):
            raise FolderPolicyError(f"{str(path)!r} cannot be granted")
        if not any(_under_root(path, root) for root in allowed):
            raise FolderPolicyError(f"{str(path)!r} is outside the approved folder roots")
        mode = str(grant.mode or "").strip().lower()
        if mode not in ("ro", "rw"):
            raise FolderPolicyError("folder access must be read-only or read-write")
        for other in resolved:
            if _overlaps(path, other):
                raise FolderPolicyError(
                    f"folder grants must be non-overlapping roots; {str(path)!r} overlaps {str(other)!r}"
                )
        resolved.append(path)
        cleaned.append(FolderGrant(str(path), mode))
    return tuple(cleaned)
