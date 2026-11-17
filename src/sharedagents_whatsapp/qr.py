"""Copy the live WhatsApp link QR into a chat-deliverable artifact.

The pairing QR rotates about every 20 s and is invalid once the linker
exits. This module never returns a cached image: it copies the current
``link-qr.png`` (starting ``sharedagents whatsapp link`` when needed) to a
timestamped file under the session artifacts dir.

Used by the on-demand ``whatsapp`` skill — not an always-loaded builtin tool.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sharedagents_whatsapp.config import WhatsAppConfig
from sharedagents_whatsapp.neonize_transport import linked_account, mask_user

_INSTALL_HINT = "pip install 'sharedagents[whatsapp]'"
_QR_WAIT_S = 20.0
_LINKER_TIMEOUT_S = 180.0


@dataclass(frozen=True)
class QrShow:
    """Result of :func:`show_qr`."""

    linked: bool
    account: str
    image_path: Path | None
    detail: str
    started_linker: bool


def artifacts_dir() -> Path:
    """Where the operator-visible QR copy is written."""
    from sharedagents_core.kernel.layout import sessions_root

    dest = sessions_root() / "artifacts"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def linker_pid_path(home: Path) -> Path:
    return Path(home) / "link.pid"


def _pid_state(pid: int) -> str:
    """``alive``, ``zombie``, or ``dead``. Zombies stay killable; treat them as gone."""
    if pid <= 0:
        return "dead"
    try:
        os.kill(pid, 0)
    except OSError:
        return "dead"
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return "dead"
    rpar = stat.rfind(")")
    if rpar == -1:
        return "alive"
    parts = stat[rpar + 2 :].split()
    if parts and parts[0] == "Z":
        return "zombie"
    return "alive"


def _pid_alive(pid: int) -> bool:
    return _pid_state(pid) == "alive"


def _is_our_linker(pid: int) -> bool:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    cmd = raw.replace(b"\x00", b" ").decode("utf-8", "replace")
    return "cmd_link" in cmd or "sharedagents_whatsapp.cli" in cmd


def linker_running(home: Path) -> bool:
    """True when a ``whatsapp link`` process we started (or recorded) is alive."""
    path = linker_pid_path(home)
    if not path.is_file():
        return False
    try:
        pid = int(path.read_text().strip() or "0")
    except ValueError:
        return False
    return _pid_alive(pid)


def _default_spawn(cfg: WhatsAppConfig, phone: str | None) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-c",
        "from argparse import Namespace\n"
        "from sharedagents_whatsapp.cli import cmd_link\n"
        f"cmd_link(Namespace(phone={phone!r}, timeout={_LINKER_TIMEOUT_S}))\n",
    ]
    log = (cfg.home / "link.log").open("ab")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=os.environ.copy(),
        )
    finally:
        log.close()
    threading.Thread(target=proc.wait, name="wa-link-reap", daemon=True).start()
    return proc


def stop_linker(home: Path) -> None:
    """Drop a recorded linker so the next generate can start a new one."""
    path = linker_pid_path(home)
    if not path.is_file():
        return
    try:
        pid = int(path.read_text().strip() or "0")
    except ValueError:
        pid = 0
    if _pid_alive(pid) and _is_our_linker(pid):
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGTERM)
    with contextlib.suppress(OSError):
        path.unlink()


def start_linker(
    cfg: WhatsAppConfig,
    phone: str | None = None,
    *,
    spawn: Callable[[WhatsAppConfig, str | None], subprocess.Popen] = _default_spawn,
    force: bool = False,
) -> int:
    """Start one ``whatsapp link`` if none is running. Returns the pid."""
    if force or not linker_running(cfg.home):
        stop_linker(cfg.home)
        proc = spawn(cfg, phone)
        linker_pid_path(cfg.home).write_text(f"{proc.pid}\n")
        return int(proc.pid)
    return int(linker_pid_path(cfg.home).read_text().strip())


def _copy_qr(src: Path, dest_dir: Path) -> Path:
    dest = dest_dir / f"whatsapp-qr-{int(time.time())}.png"
    shutil.copy2(src, dest)
    with contextlib.suppress(OSError):
        os.chmod(dest, 0o600)
    return dest


def link_status(*, home: Path | None = None) -> QrShow:
    """Read-only: already linked or not. Never starts the linker."""
    cfg = WhatsAppConfig.load(home)
    account = linked_account(cfg.store_path)
    if account:
        return QrShow(
            linked=True,
            account=mask_user(account["user"]),
            image_path=None,
            detail=f"already linked as {mask_user(account['user'])}",
            started_linker=False,
        )
    return QrShow(
        linked=False,
        account="",
        image_path=None,
        detail="not linked",
        started_linker=False,
    )


def peek_qr(
    *,
    home: Path | None = None,
    dest_dir: Path | None = None,
) -> QrShow:
    """Copy the current QR if one is already on disk. Never starts the linker."""
    status = link_status(home=home)
    if status.linked:
        return status
    cfg = WhatsAppConfig.load(home)
    if not (cfg.qr_png_path.is_file() and cfg.qr_png_path.stat().st_size > 0):
        return status
    dest = dest_dir or artifacts_dir()
    dest.mkdir(parents=True, exist_ok=True)
    return QrShow(
        linked=False,
        account="",
        image_path=_copy_qr(cfg.qr_png_path, dest),
        detail="scan this QR now; it expires in about 20s — generate again for a fresh one",
        started_linker=False,
    )


def show_qr(
    *,
    home: Path | None = None,
    dest_dir: Path | None = None,
    phone: str | None = None,
    wait_s: float = _QR_WAIT_S,
    spawn: Callable[[WhatsAppConfig, str | None], subprocess.Popen] | None = None,
    force: bool = False,
) -> QrShow:
    """Return a fresh QR artifact path, or report that the device is linked.

    ``spawn`` is the test seam — omit it in production to start the real linker.
    ``force`` restarts the linker so Refresh QR is not stuck on a dead/zombie
    leftover after the previous pairing window timed out.
    """
    cfg = WhatsAppConfig.load(home)
    account = linked_account(cfg.store_path)
    if account:
        return QrShow(
            linked=True,
            account=mask_user(account["user"]),
            image_path=None,
            detail=f"already linked as {mask_user(account['user'])}",
            started_linker=False,
        )
    cfg.ensure_home()
    dest = dest_dir or artifacts_dir()
    dest.mkdir(parents=True, exist_ok=True)
    started = False
    need_start = force or not cfg.qr_png_path.is_file() or not linker_running(cfg.home)
    if need_start:
        if spawn is None:
            if importlib.util.find_spec("neonize") is None:
                return QrShow(
                    linked=False,
                    account="",
                    image_path=None,
                    detail=f"neonize is not installed — {_INSTALL_HINT}",
                    started_linker=False,
                )
            spawn = _default_spawn
        started_at = time.time()
        start_linker(cfg, phone, spawn=spawn, force=force or not linker_running(cfg.home))
        started = True
    else:
        started_at = 0.0
    deadline = time.time() + max(0.1, float(wait_s))
    while time.time() < deadline:
        if cfg.qr_png_path.is_file() and cfg.qr_png_path.stat().st_size > 0:
            if started and cfg.qr_png_path.stat().st_mtime < started_at - 0.5:
                time.sleep(0.2)
                continue
            copied = _copy_qr(cfg.qr_png_path, dest)
            return QrShow(
                linked=False,
                account="",
                image_path=copied,
                detail="scan this QR now; it expires in about 20s — generate again for a fresh one",
                started_linker=started,
            )
        time.sleep(0.2)
    return QrShow(
        linked=False,
        account="",
        image_path=None,
        detail="QR not ready — WhatsApp link did not write link-qr.png in time",
        started_linker=started,
    )


def render_snapshot(
    result: QrShow,
) -> str:
    """Markdown for the Skill appendix / script stdout."""
    lines = ["## Live WhatsApp link"]
    if result.linked:
        lines.append(
            f"Already linked as **{result.account}**. No QR. Start the channel with `sharedagents whatsapp run`."
        )
        return "\n".join(lines) + "\n"
    if result.image_path is not None:
        path = result.image_path.resolve()
        lines.append(result.detail)
        lines.append("")
        lines.append(f"[Screenshot](<{path}>)")
        lines.append("")
        lines.append(
            "Paste that Screenshot line in the user-visible reply. "
            "Do not reuse a QR from an earlier turn — it is already stale."
        )
        return "\n".join(lines) + "\n"
    lines.append(f"Not linked: {result.detail}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    del argv
    result = show_qr()
    sys.stdout.write(render_snapshot(result))
    return 0 if result.linked or result.image_path is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
