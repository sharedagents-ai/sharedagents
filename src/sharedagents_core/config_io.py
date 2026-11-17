"""Serialize Windows config migration with application config writers."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

from sharedagents_core import oscompat

IS_WINDOWS = os.name == "nt"


@contextmanager
def config_write_lock(path: Path):
    """Hold a stable sidecar lock across a config read/replace or write.

    Locking the config itself would stop protecting it after atomic replacement.
    Keep the sidecar on disk so all processes always lock the same file. POSIX
    keeps its existing behavior, with no extra files or locking.
    """
    if not IS_WINDOWS:
        yield
        return
    resolved = path.resolve()
    lock_path = resolved.with_name(resolved.name + ".write.lock")
    with lock_path.open("a+b") as handle, oscompat.file_lock(handle):
        yield


def write_config_text(path: Path, text: str) -> None:
    """Write UTF-8 config text without racing a legacy-encoding migration."""
    with config_write_lock(path):
        path.write_text(text, encoding="utf-8")
