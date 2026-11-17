"""One-shot 'unify legacy per-agent sessions/logs' migration.

Operator answered "Fresh start (delete legacy)" in the 2026-05-26
brainstorming, so this command:

1. Tars every legacy ``agents/<name>/sessions`` and
   ``agents/<name>/logs`` into
   ``~/.sharedagents/.legacy_backup/<timestamp>.tar.gz`` for forensics.
2. Removes the legacy directories in place.

Leaves the agent-local CONFIG files (``agents/<name>/config.json``,
``hooks.json``, ``gateway.pid``, ``SOUL.md``, ``secrets.env``,
``session_reviews/``) untouched — those are not data, only
per-agent configuration that stays per-agent in the unified layout.

Public surface:

* :func:`has_legacy` — quick existence check used by the runtime
  startup guard.
* :func:`migrate` — perform the archive + delete; idempotent and
  safe to call when no legacy data is present.
"""

from __future__ import annotations

import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sharedagents_core.layout import agents_root, sharedagents_root

_LEGACY_SUBDIRS: tuple[str, ...] = ("sessions", "logs")


def _legacy_dirs() -> list[Path]:
    """Per-agent ``sessions`` and ``logs`` entries that need archiving.

    Treats both real directories AND symlinks pointing at one as
    legacy entries. ``shutil.rmtree(symlink)`` is a no-op (it raises
    ``NotADirectoryError`` on most filesystems), so a dangling
    ``agents/<name>/sessions`` symlink left over from an early-deploy
    workaround would otherwise survive ``migrate`` and keep the
    gateway's startup guard tripping forever.
    """
    out: list[Path] = []
    root = agents_root()
    if not root.is_dir():
        return out
    for agent_dir in sorted(root.iterdir()):
        # Skip symlinked agent dirs entirely; archiving them would
        # follow the link to a shared dir we have no business touching.
        if agent_dir.is_symlink() or not agent_dir.is_dir():
            continue
        for sub in _LEGACY_SUBDIRS:
            p = agent_dir / sub
            if p.is_symlink() or p.is_dir():
                out.append(p)
    return out


def has_legacy() -> bool:
    """True iff at least one per-agent legacy sessions/logs dir exists.

    This is the boot-refusal gate: the gateway (``gateway.py``) calls
    this at startup and hard-refuses to boot (``raise SystemExit(2)``)
    when it returns True, rather than let two writers (legacy per-agent
    layout and the unified layout) silently disagree on where
    session/log data lives. Do not make this check optional or
    best-effort without re-examining that boot guard.
    """
    return bool(_legacy_dirs())


def migrate() -> dict[str, Any]:
    """Archive every legacy dir to ``.legacy_backup/<ts>.tar.gz`` then delete.

    Idempotent: returns ``{"agents_migrated": 0, "tarball": None}`` when
    nothing legacy is present. On success returns
    ``{"agents_migrated": N, "tarball": "<absolute path>"}``.
    """
    legacy = _legacy_dirs()
    if not legacy:
        return {"agents_migrated": 0, "tarball": None}
    home = sharedagents_root()
    backup_dir = home / ".legacy_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tarball = backup_dir / f"{ts}.tar.gz"
    # Re-running after an interrupted DELETION must not lose
    # data. A rerun in the SAME second resolves to the same timestamped path;
    # overwriting ("w:gz") would clobber the already-complete pre-delete
    # archive with a smaller one covering only the not-yet-deleted remainder,
    # permanently losing the entries the first run already deleted. Any tarball
    # surviving at this path is guaranteed COMPLETE — it was staged entirely
    # before any delete, and a crash mid-tar unlinks its own partial (below) —
    # and captures a SUPERSET of the legacy data still on disk now. So reuse it
    # rather than clobber it, then proceed to (re)deletion.
    if not tarball.exists():
        # Stage the tarball BEFORE any destructive op so a crash mid-tar
        # leaves the legacy data intact.
        try:
            with tarfile.open(tarball, "w:gz") as tf:
                for p in legacy:
                    tf.add(p, arcname=str(p.relative_to(home)))
        except BaseException:
            # Drop the partial tarball — re-running migrate after a fix
            # should produce a clean archive.
            try:
                tarball.unlink()
            except FileNotFoundError:
                pass
            raise
    for p in legacy:
        # symlink: unlink the link itself (rmtree would either raise
        # NotADirectoryError or, with ignore_errors=True, leave the
        # dangling link in place). Directory: rmtree.
        try:
            if p.is_symlink():
                p.unlink()
            else:
                shutil.rmtree(p, ignore_errors=True)
        except OSError:
            # Best-effort: the next migrate run will re-discover the
            # entry and try again; the tarball already preserves it.
            pass
    agents_migrated = len({p.parent for p in legacy})
    return {"agents_migrated": agents_migrated, "tarball": str(tarball)}


__all__ = ["has_legacy", "migrate"]
