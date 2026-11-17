"""``sharedagents whatsapp …`` commands and the channel service entry point.

    sharedagents whatsapp link [--phone +55…]   pair this host as a linked device
    sharedagents whatsapp status                link / agent / service status
    sharedagents whatsapp run                   run the channel in the foreground
    sharedagents whatsapp logout                unlink and delete the credentials
    sharedagents whatsapp install-service       install the systemd user unit

The service (``python -m sharedagents_whatsapp``) fails closed with exit 78 when the
device is not linked, logged out, replaced or banned; the unit's
``RestartPreventExitStatus=78`` keeps systemd from restart-looping it. neonize's
Go worker thread never returns after ``stop()`` (it even blocks
``asyncio.run``'s executor shutdown), so every entry point that started it ends
with ``os._exit`` once state is flushed. This module is imported by the main
``sharedagents`` parser, so anything heavy is imported lazily.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib.util
import logging
import os
import signal
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover — Windows has no flock
    fcntl = None  # type: ignore[assignment]

from sharedagents_whatsapp.config import WhatsAppConfig
from sharedagents_whatsapp.neonize_transport import (
    EX_CONFIG,
    linked_account,
    mask_user,
    remove_store,
)

SERVICE_NAME = "sharedagents-whatsapp.service"
DEFAULT_STT: Any = object()  # sentinel: wire the shared STT service
_INSTALL_HINT = "pip install 'sharedagents[whatsapp]'"
_log = logging.getLogger("sharedagents.whatsapp")


# -- user-systemd primitives (shared with the gateway / telegram units) --------


def systemd_available() -> bool:
    from sharedagents_gateway.gateway_cli import systemd_available as available

    return available()


def _user_unit_dir() -> Path:
    from sharedagents_gateway.gateway_cli import _user_unit_dir as unit_dir

    return unit_dir()


def _run_user_systemctl(argv: list[str]) -> Any:
    from sharedagents_gateway.gateway_cli import _run_user_systemctl as run

    return run(argv)


def render_whatsapp_unit(home: Path) -> str:
    """The ``sharedagents-whatsapp.service`` user unit for THIS install.

    Mirrors ``render_telegram_unit`` (same PATH composition and env files, no
    secrets inline) plus the fail-closed restart policy.
    """
    venv_bin = str(Path(sys.executable).parent)
    parts = [
        venv_bin,
        str(Path.home() / ".local" / "bin"),
        "/usr/local/sbin",
        "/usr/local/bin",
        "/usr/sbin",
        "/usr/bin",
        "/sbin",
        "/bin",
        "/snap/bin",
    ]
    seen: set[str] = set()
    path_env = os.pathsep.join(p for p in parts if p and not (p in seen or seen.add(p)))
    return f"""\
[Unit]
Description=sharedagents WhatsApp channel (linked device, self-chat)
After=network-online.target sharedagents-gateway.service
Wants=network-online.target sharedagents-gateway.service
StartLimitIntervalSec=300
StartLimitBurst=10

[Service]
Type=simple
WorkingDirectory={home}
EnvironmentFile=-/etc/environment
EnvironmentFile=-{home}/.env
Environment="PATH={path_env}"
Environment="SHAREDAGENTS_WHATSAPP_HOME={home}"
ExecStart={sys.executable} -m sharedagents_whatsapp
Restart=on-failure
RestartSec=15
# 78 = not linked / logged out / replaced / banned — needs a human; never loop.
RestartPreventExitStatus=78
TimeoutStopSec=20
KillMode=mixed
StandardOutput=journal
StandardError=journal
SyslogIdentifier=sharedagents-whatsapp

[Install]
WantedBy=default.target
"""


def install_service(*, enable: bool = True, start: bool = True) -> dict:
    """Write + enable the user unit; start it only when a device is linked."""
    cfg = WhatsAppConfig.load()
    result: dict[str, Any] = {
        "ok": False,
        "error": None,
        "unit_path": None,
        "enabled": False,
        "started": False,
        "linked": False,
    }
    if not systemd_available():
        result["error"] = "systemd (user instance) is not available on this host"
        return result
    cfg.ensure_home()
    unit_path = _user_unit_dir() / SERVICE_NAME
    unit_path.parent.mkdir(parents=True, exist_ok=True)
    unit_path.write_text(render_whatsapp_unit(cfg.home), encoding="utf-8")
    result["unit_path"] = str(unit_path)
    _run_user_systemctl(["daemon-reload"])
    if enable:
        result["enabled"] = _run_user_systemctl(["enable", SERVICE_NAME]).returncode == 0
    result["linked"] = linked_account(cfg.store_path) is not None
    if start and result["linked"]:
        _run_user_systemctl(["reset-failed", SERVICE_NAME])
        started = _run_user_systemctl(["restart", SERVICE_NAME])
        result["started"] = started.returncode == 0
        if not result["started"]:
            result["error"] = (started.stderr or "").strip() or "systemctl restart failed"
            return result
    result["ok"] = True
    return result


# -- service -------------------------------------------------------------------


def _acquire_pid_lock(path: Path) -> int | None:
    """Exclusive, non-blocking lock on ``path``; ``None`` when another instance holds it."""
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o600)
    if fcntl is not None:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return None
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n".encode())
    return fd


def _release_pid_lock(fd: int | None) -> None:
    if fd is not None:
        with contextlib.suppress(OSError):
            os.close(fd)


def _default_stt(log: logging.Logger) -> Any:
    """The shared STT pipeline (primary/backup from ``~/.sharedagents/config.json``).

    Reuses the Telegram voice-note pipeline (transcode → segment → Whisper API
    failover) rather than a second copy of it.
    """
    try:
        from sharedagents_core.services import resolve_service

        service = resolve_service("stt")
        if not service.configured:
            log.warning(
                'STT not configured — voice notes will be refused (add an "stt" block to ~/.sharedagents/config.json)'
            )
            return None
        from sharedagents_telegram.stt import _make_default_stt

        return _make_default_stt(service, log)
    except Exception as exc:  # noqa: BLE001
        log.warning("STT unavailable (%s) — voice notes will be refused", exc)
        return None


def _wire_allowlist_gates(cfg: WhatsAppConfig, transport: Any) -> None:
    """Gate the transport on config.json as it is now, not as it was at startup."""

    def gate(access: str) -> Callable[[str, bool], bool]:
        def allowed(ident: str, is_group: bool) -> bool:
            cfg.refresh_allowlist()
            if cfg.allows(ident, group=is_group, access=access) or is_group:
                return cfg.allows(ident, group=is_group, access=access)
            cfg.refresh_lid_aliases()  # a first message may bring a new LID↔phone mapping
            return cfg.allows(ident, group=is_group, access=access)

        return allowed

    transport.may_read = gate("read")
    transport.may_send = gate("send")


async def run_service(
    cfg: WhatsAppConfig,
    *,
    transport: Any = None,
    gateway: Any = None,
    stt: Any = DEFAULT_STT,
    install_signals: bool = True,
) -> int:
    """Run the channel until SIGTERM/SIGINT (→ 0) or a transport fatal (→ 78)."""
    from sharedagents_whatsapp.channel import WhatsAppChannel
    from sharedagents_whatsapp.gateway import GatewayClient
    from sharedagents_whatsapp.state import ChannelState

    cfg.ensure_home()
    lock = _acquire_pid_lock(cfg.pid_path)
    if lock is None:
        _log.error("another sharedagents-whatsapp is already running (lock %s)", cfg.pid_path)
        return 1
    channel = None
    try:
        if transport is None:
            if linked_account(cfg.store_path) is None:
                _log.error("WhatsApp is not linked — run `sharedagents whatsapp link` first")
                return EX_CONFIG
            if importlib.util.find_spec("neonize") is None:
                _log.error("neonize is not installed — %s", _INSTALL_HINT)
                return EX_CONFIG
            from sharedagents_whatsapp.neonize_transport import NeonizeTransport

            transport = NeonizeTransport(cfg.store_path, home=cfg.home, log=_log)
            _wire_allowlist_gates(cfg, transport)
        if gateway is None:
            gateway = GatewayClient(cfg.resolved_gateway_url(), token=WhatsAppConfig.gateway_token())
        if stt is DEFAULT_STT:
            stt = _default_stt(_log)
        state = ChannelState.load(cfg.state_path, cfg.default_agent)
        channel = WhatsAppChannel(cfg, transport, gateway, state, stt=stt, log=_log)
        stop = asyncio.Event()
        if install_signals:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                with contextlib.suppress(NotImplementedError, RuntimeError):
                    loop.add_signal_handler(sig, stop.set)
        failed = getattr(transport, "failed", None)
        waiters = [asyncio.ensure_future(stop.wait()), asyncio.ensure_future(transport.fatal.wait())]
        if isinstance(failed, asyncio.Event):
            waiters.append(asyncio.ensure_future(failed.wait()))
        startup = asyncio.create_task(channel.start())
        try:
            done, _ = await asyncio.wait([startup, *waiters], return_when=asyncio.FIRST_COMPLETED)
            if startup in done and not (
                stop.is_set() or transport.fatal.is_set() or (isinstance(failed, asyncio.Event) and failed.is_set())
            ):
                await startup  # propagate readiness / initialization errors
                await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for waiter in [startup, *waiters]:
                waiter.cancel()
            await asyncio.gather(startup, *waiters, return_exceptions=True)
        if transport.fatal.is_set():
            _log.error("stopping: %s", transport.fatal_reason)
            return EX_CONFIG
        if isinstance(failed, asyncio.Event) and failed.is_set():
            _log.error("connection failed: %s", getattr(transport, "failure_reason", "") or "unknown")
            return 1
        return 0
    finally:
        if channel is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(channel.stop(), 15)
        _release_pid_lock(lock)


def _configure_logging() -> None:
    level = os.environ.get("SHAREDAGENTS_WHATSAPP_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Channel debugging must never enable raw traffic from the other chats.
    for name in ("whatsmeow", "Whatsmeow", "neonize", "neonize.utils.log"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _hard_exit(code: int) -> None:
    """Exit without joining neonize's never-returning Go worker thread."""
    with contextlib.suppress(Exception):
        sys.stdout.flush()
        sys.stderr.flush()
    logging.shutdown()
    os._exit(int(code))


def service_main(argv: list[str] | None = None) -> None:
    """``python -m sharedagents_whatsapp`` / ``sharedagents-whatsapp`` — the systemd ExecStart."""
    del argv
    _configure_logging()
    try:
        cfg = WhatsAppConfig.load()
    except ValueError as exc:
        _log.error("%s", exc)
        _hard_exit(EX_CONFIG)
        return
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        code = loop.run_until_complete(run_service(cfg))
    except KeyboardInterrupt:
        code = 0
    except Exception:  # noqa: BLE001
        _log.exception("sharedagents-whatsapp crashed")
        code = 1
    _hard_exit(code)


# -- commands --------------------------------------------------------------------


def _neonize_ready() -> bool:
    if importlib.util.find_spec("neonize") is None:
        print(f"neonize is not installed — {_INSTALL_HINT}")
        return False
    try:
        import neonize  # noqa: F401 — loads the bundled Go library + python-magic
    except ImportError as exc:
        hint = " (install libmagic: apt install libmagic1)" if "magic" in str(exc).lower() else ""
        print(f"neonize failed to import: {exc}{hint}")
        return False
    return True


def cmd_link(args: argparse.Namespace) -> int:
    from sharedagents_whatsapp.neonize_transport import link_device

    cfg = WhatsAppConfig.load()
    cfg.ensure_home()
    account = linked_account(cfg.store_path)
    if account:
        print(
            f"Already linked as {mask_user(account['user'])}. "
            "Run `sharedagents whatsapp logout` first to link a different account."
        )
        return 1
    if not _neonize_ready():
        return 1
    _configure_logging()
    print("Linking sharedagents as a WhatsApp device…")
    loop = asyncio.new_event_loop()
    try:
        code, detail = loop.run_until_complete(
            link_device(
                cfg.store_path,
                home=cfg.home,
                phone=getattr(args, "phone", None),
                qr_png=cfg.qr_png_path,
                timeout=float(getattr(args, "timeout", 180.0)),
            )
        )
    except KeyboardInterrupt:
        with contextlib.suppress(OSError):
            cfg.qr_png_path.unlink()
        _hard_exit(130)
        return 130
    if code == 0:
        print(
            f"✅ Linked as {detail}. Start the channel with "
            "`sharedagents whatsapp install-service` (or `sharedagents whatsapp run`)."
        )
    else:
        print(f"✗ Not linked: {detail}")
    _hard_exit(code)
    return code


def cmd_status(args: argparse.Namespace) -> int:
    del args
    from sharedagents_whatsapp.state import SELF_CHAT, ChannelState

    cfg = WhatsAppConfig.load()
    account = linked_account(cfg.store_path)
    print("WhatsApp channel")
    print(f"  home:     {cfg.home}")
    if account:
        name = f" ({account['push_name']})" if account["push_name"] else ""
        print(f"  linked:   yes — {mask_user(account['user'])}{name}")
    else:
        print("  linked:   no — not linked; run `sharedagents whatsapp link`")
    state = ChannelState.load(cfg.state_path, cfg.default_agent)
    sid = state.session_for(SELF_CHAT)
    print(f"  agent:    {state.agent_for(SELF_CHAT)}" + (f" (session {sid[:8]})" if sid else ""))
    print(f"  gateway:  {cfg.resolved_gateway_url()}")
    installed = importlib.util.find_spec("neonize") is not None
    print(f"  neonize:  {'installed' if installed else 'missing — ' + _INSTALL_HINT}")
    if systemd_available():
        active = (_run_user_systemctl(["is-active", SERVICE_NAME]).stdout or "").strip()
        print(f"  service:  {active or 'unknown'} ({SERVICE_NAME})")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    del args
    service_main()
    return 0


def cmd_logout(args: argparse.Namespace) -> int:
    del args
    from sharedagents_whatsapp.neonize_transport import logout_device

    cfg = WhatsAppConfig.load()
    if linked_account(cfg.store_path) is None:
        remove_store(cfg.store_path)
        print("Not linked — nothing to log out.")
        return 0
    if not _neonize_ready():
        return 1
    _configure_logging()
    loop = asyncio.new_event_loop()
    unlinked = loop.run_until_complete(logout_device(cfg.store_path, home=cfg.home))
    if unlinked:
        print("✅ Unlinked from WhatsApp and removed the local credentials.")
    else:
        print(
            'Removed the local credentials. If your phone still lists "sharedagents" '
            "under Linked devices, remove it there."
        )
    _hard_exit(0)
    return 0


def cmd_install_service(args: argparse.Namespace) -> int:
    result = install_service(
        enable=not getattr(args, "no_enable", False),
        start=not getattr(args, "no_start", False),
    )
    if result["error"]:
        print(f"✗ {result['error']}")
        return 1
    print(f"Installed {result['unit_path']}")
    if result["enabled"]:
        print(f"Enabled {SERVICE_NAME} (starts with your user session / at boot with linger).")
    if result["started"]:
        print(f"Started {SERVICE_NAME} — message yourself on WhatsApp to talk to your agent.")
    elif not result["linked"]:
        print(
            f"Not started: link first with `sharedagents whatsapp link`, then `systemctl --user start {SERVICE_NAME}`."
        )
    return 0


def add_subparser(sub: argparse._SubParsersAction) -> None:
    """Wire ``sharedagents whatsapp <cmd>`` into the top-level parser."""
    parser = sub.add_parser(
        "whatsapp",
        help="WhatsApp self-chat channel: link/status/run/logout/install-service",
    )
    parser.set_defaults(_dispatch=lambda a: parser.print_help() or 0)
    inner = parser.add_subparsers(dest="whatsapp_cmd")

    p_link = inner.add_parser("link", help="Link this host as a WhatsApp device (QR or pairing code)")
    p_link.add_argument(
        "--phone",
        default=None,
        help="Your number in international format (e.g. +5511987654321) to get an "
        "8-character pairing code instead of a QR",
    )
    p_link.add_argument("--timeout", type=float, default=180.0, help="Seconds to wait (default 180)")
    p_link.set_defaults(_dispatch=cmd_link)

    p_status = inner.add_parser("status", help="Link, agent and service status")
    p_status.set_defaults(_dispatch=cmd_status)

    p_run = inner.add_parser("run", help="Run the channel in the foreground")
    p_run.set_defaults(_dispatch=cmd_run)

    p_logout = inner.add_parser("logout", help="Unlink this device and delete its credentials")
    p_logout.set_defaults(_dispatch=cmd_logout)

    p_install = inner.add_parser(
        "install-service",
        help="Install + enable the sharedagents-whatsapp systemd user service",
    )
    p_install.add_argument("--no-enable", action="store_true", help="Don't enable the unit at login/boot")
    p_install.add_argument("--no-start", action="store_true", help="Don't start the service now")
    p_install.set_defaults(_dispatch=cmd_install_service)


__all__ = [
    "SERVICE_NAME",
    "add_subparser",
    "cmd_install_service",
    "cmd_link",
    "cmd_logout",
    "cmd_run",
    "cmd_status",
    "install_service",
    "render_whatsapp_unit",
    "run_service",
    "service_main",
]
