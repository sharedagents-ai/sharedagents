"""Durable storage for inbound uploads, independent of document extraction."""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path


def upload_directory(identity: str, *, admin: bool, destination: str = "") -> Path:
    """Resolve a destination; regular users are confined to their own folder.

    Explicit destinations are directories, relative to the default download
    folder unless absolute. Only administrators may select an outside folder.
    """
    root = Path(os.environ.get("SHAREDAGENTS_DOWNLOADS_DIR") or Path.home() / "Downloads").expanduser()
    if not admin:
        owner = hashlib.sha256(identity.strip().lower().encode()).hexdigest()
        root = root / "users" / owner
    root = root.resolve()
    target = Path(destination).expanduser() if destination else root
    if not target.is_absolute():
        target = root / target
    target = target.resolve()
    if not admin and not target.is_relative_to(root):
        raise PermissionError("destination must be within your downloads folder")
    return target


def save_upload(data: bytes, filename: str, directory: Path) -> Path:
    """Save original bytes without overwriting files or following file symlinks."""
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(ch for ch in name if ch.isprintable()).strip()
    # Leave room for a collision suffix and stay below common NAME_MAX limits.
    name = name.encode("utf-8")[:180].decode("utf-8", "ignore")
    if not name or name in {".", ".."}:
        name = "attachment"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = directory / name
    while True:
        try:
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            break
        except FileExistsError:
            base = Path(name)
            target = directory / f"{base.stem}-{uuid.uuid4().hex[:12]}{base.suffix}"
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    return target
