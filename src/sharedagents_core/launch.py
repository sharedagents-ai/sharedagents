"""Process setup for entry points: Windows UTF-8 mode and server output."""

from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from sharedagents_core.config_io import config_write_lock

IS_WINDOWS = os.name == "nt"


def _in_utf8_mode() -> bool:
    return bool(sys.flags.utf8_mode)


def _ansi_code_page() -> str:
    import _winapi

    return f"cp{_winapi.GetACP()}"


def ensure_utf8_mode() -> None:
    """Run this Windows process in Python UTF-8 mode; call first in every entry point.

    Without it Python 3.11 reads and writes text without an explicit encoding in
    the ANSI code page (cp1252), while sharedagents's readers mostly decode UTF-8, so
    one em dash in a saved file broke the next start ("'utf-8' codec can't
    decode byte 0x97"). UTF-8 mode can only be chosen at interpreter start, so
    the command is run again with PYTHONUTF8=1 and this process exits with its
    code; servers and tools started from it inherit the variable. POSIX and an
    explicit PYTHONUTF8 setting are left alone. Then configs that an older
    version saved in the code page are converted.
    """
    if not IS_WINDOWS:
        return
    if not _in_utf8_mode() and "PYTHONUTF8" not in os.environ:
        argv = [sys.executable, *sys.orig_argv[1:]]
        previous = signal.signal(signal.SIGINT, signal.SIG_IGN)  # The child handles Ctrl+C.
        try:
            code = subprocess.call(argv, env={**os.environ, "PYTHONUTF8": "1"})
        except OSError:
            signal.signal(signal.SIGINT, previous)
        else:
            raise SystemExit(code)
    repair_legacy_config_encoding()


def repair_legacy_config_encoding() -> list[Path]:
    """Rewrite the JSON configs older Windows releases saved in the code page as UTF-8.

    Only files that are not UTF-8 but decode and parse as JSON in the ANSI code
    page are touched; anything else is left for its reader to report.
    """
    if not IS_WINDOWS:
        return []
    from sharedagents_core.config import config_path
    from sharedagents_core.kernel.config import global_config_path
    from sharedagents_core.kernel.layout import agents_root

    legacy = _ansi_code_page()
    converted = []
    for path in (global_config_path(), config_path(), *sorted(agents_root().glob("*/config.json"))):
        try:
            # Do not create sidecars for missing configs (including --help).
            if not path.is_file():
                continue
            with config_write_lock(path):
                if not _repair_config_encoding(path, legacy):
                    continue
        except OSError:
            continue
        print(f"sharedagents: converted {path} from {legacy} to UTF-8", file=sys.stderr, flush=True)
        converted.append(path)
    return converted


def _repair_config_encoding(path: Path, legacy: str) -> bool:
    raw = path.read_bytes()
    try:
        raw.decode("utf-8")
        return False
    except UnicodeDecodeError:
        pass
    try:
        text = raw.decode(legacy)
        json.loads(text)
    except ValueError:
        return False
    tmp = path.with_name(f"{path.name}.utf8.{os.getpid()}")
    try:
        tmp.write_bytes(text.encode("utf-8"))
        # Also leave an edit from a non-cooperating writer alone if observed.
        if path.read_bytes() != raw:
            return False
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return True


def component_stdio(role: str) -> None:
    """Reopen redirected Python output in the actual server, before app imports.

    Console/venv wrappers may replace inherited streams. A write-through append
    stream also avoids buffering and stale offsets when JSON log writers append
    through a separate descriptor. Ordinary CLI/systemd runs are unchanged.
    The launcher rotates the previous log before spawning this process.
    """
    path = os.environ.get("SHAREDAGENTS_COMPONENT_LOG")
    if not path or os.environ.get("SHAREDAGENTS_COMPONENT_ROLE") != role:
        return
    logfile = Path(path)
    if not logfile.is_absolute():
        raise ValueError("SHAREDAGENTS_COMPONENT_LOG must be absolute")
    stream = io.TextIOWrapper(
        logfile.open("ab", buffering=0), encoding="utf-8", errors="backslashreplace", write_through=True
    )
    sys.stdout = sys.stderr = stream
    try:
        release = version("sharedagents")
    except PackageNotFoundError:
        release = "source"
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"=== SHAREDAGENTS START {stamp} role={role} pid={os.getpid()} version={release} ===", flush=True)
