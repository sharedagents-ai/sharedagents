"""``sharedagents web`` — lifecycle for the localhost web frontend.

Mirrors ``sharedagents gateway`` / ``sharedagents telegram``: a daemon-control CLI for
the Flask web server bundled as ``sharedagents_web`` (the browser frontend that
renders the shared command runtime — see ``sharedagents_web/README.md``). Subcommands:

    sharedagents web start            # daemonize sharedagents_web.server (pidfile + log)
    sharedagents web stop             # SIGTERM the daemon
    sharedagents web restart          # stop + start
    sharedagents web status           # pid / uptime / url
    sharedagents web logs             # tail -f web.log
    sharedagents web install-service  # ~/.config/systemd/user/sharedagents-web.service

Bind is **loopback-only** ``127.0.0.1:8110`` (the server itself rejects a
non-loopback host outside debug mode); it's meant to sit behind the
dashboard reverse proxy. ``SHAREDAGENTS_WEB_SECRET`` is auto-generated once and
persisted to ``~/.sharedagents/.env`` so both the foreground daemon and the
systemd unit (via ``EnvironmentFile``) pick it up.

Daemon control uses a pidfile at ``~/.sharedagents/web.pid`` (separate from the
gateway/telegram pidfiles). The web server is independent of the gateway
process; it only *proxies* to it over HTTP, so it's safe to start/stop on
its own — but the systemd unit orders itself ``After`` the gateway.
"""

from __future__ import annotations

import argparse
import getpass
import os
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path

from sharedagents_core import oscompat

_SERVICE_NAME = "sharedagents-web.service"
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8110


# ---- path / config helpers -------------------------------------------


def _sharedagents_home() -> Path:
    return Path(os.environ.get("SHAREDAGENTS_HOME") or (Path.home() / ".sharedagents"))


def _web_dir() -> Path:
    """Locate the packaged web dir (holds ``server.py`` + assets).

    ``$SHAREDAGENTS_WEB_DIR`` overrides; otherwise resolve the installed
    ``sharedagents_web`` package location. Works identically for an installed
    wheel (site-packages/sharedagents_web) and an editable checkout
    (``<repo>/src/sharedagents_web``). Raising here with a clear message beats
    a cryptic FileNotFound deep in Popen.
    """
    env = os.environ.get("SHAREDAGENTS_WEB_DIR")
    if env:
        return Path(env).expanduser()
    try:
        import sharedagents_web

        return Path(sharedagents_web.__file__).resolve().parent
    except Exception:  # noqa: BLE001 — fall back to the source-tree layout
        return Path(__file__).resolve().parents[1] / "sharedagents_web"


def _server_py() -> Path:
    return _web_dir() / "server.py"


def web_argv(python: Path) -> list[str]:
    """Use this interpreter's console script, else its installed package module.

    Never select a different environment's entry point from PATH. Windows
    console and venv wrappers can spawn descendants; this is not a PID lookup.
    """
    exe = python.with_name("sharedagents-web.exe" if oscompat.IS_WINDOWS else "sharedagents-web")
    if exe.is_file():
        return [str(exe)]
    return [str(python), "-m", "sharedagents_web"]


def _systemd_argv(argv: list[str]) -> str:
    # systemd parses double quotes, not shell quoting. Escape specifiers and
    # variable expansion too, so paths containing spaces/%/$ stay literal.
    return " ".join(
        '"' + arg.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%").replace("$", "$$") + '"' for arg in argv
    )


def _web_pidfile() -> Path:
    return _sharedagents_home() / "web.pid"


def _web_logfile() -> Path:
    try:
        from sharedagents_core.layout import logs_root

        return logs_root() / "web.log"
    except Exception:  # noqa: BLE001
        return _sharedagents_home() / "logs" / "web.log"


def _web_config() -> dict:
    """Resolve host/port/gateway. Loopback-only by design."""
    host = os.environ.get("SHAREDAGENTS_WEB_HOST") or _DEFAULT_HOST
    try:
        port = int(os.environ.get("SHAREDAGENTS_WEB_PORT") or _DEFAULT_PORT)
    except (TypeError, ValueError):
        port = _DEFAULT_PORT
    gateway = os.environ.get("SHAREDAGENTS_GATEWAY_URL") or "http://127.0.0.1:8100"
    return {"host": host, "port": port, "gateway": gateway}


def _env_file() -> Path:
    return _sharedagents_home() / ".env"


def _ensure_web_secret() -> str:
    """Return ``SHAREDAGENTS_WEB_SECRET``, generating + persisting one once.

    Read order: process env (operator override) → ``~/.sharedagents/.env`` →
    freshly minted (and appended to ``.env`` so the systemd unit's
    ``EnvironmentFile`` and future starts reuse the same value). A stable
    secret keeps browser sessions valid across restarts.
    """
    cur = os.environ.get("SHAREDAGENTS_WEB_SECRET")
    if cur:
        return cur
    env_path = _env_file()
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("SHAREDAGENTS_WEB_SECRET="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val:
                    return val
    secret = secrets.token_hex(32)
    try:
        env_path.parent.mkdir(parents=True, exist_ok=True)
        with open(env_path, "a", encoding="utf-8") as fh:
            fh.write(f'SHAREDAGENTS_WEB_SECRET="{secret}"\n')
        try:
            env_path.chmod(0o600)
        except OSError:
            pass
    except OSError:
        pass  # best-effort; we still return the secret for this run
    return secret


def _web_python() -> Path:
    """Interpreter that can run the Flask server (flask + requests + gunicorn).

    Now that ``sharedagents_web`` ships inside the wheel, the web deps live in
    the ``[web]`` extra (``pip install sharedagents[web]``) and run under the
    SAME interpreter as the gateway — no separate ``web/venv``. Prefer the
    current interpreter when the web deps import; otherwise fall back to a
    legacy ``web/venv`` if one exists (older source checkouts), and finally
    to ``sys.executable`` so the caller sees a clear ImportError rather than
    a silent misconfiguration.

    ``$SHAREDAGENTS_WEB_PYTHON`` overrides everything.
    """
    override = os.environ.get("SHAREDAGENTS_WEB_PYTHON")
    if override:
        return Path(override).expanduser()
    try:
        import flask  # noqa: F401
        import requests  # noqa: F401

        if oscompat.IS_WINDOWS:
            import waitress  # noqa: F401
        else:
            import gunicorn  # noqa: F401
        return Path(sys.executable)
    except Exception:  # noqa: BLE001 — deps not in the current interpreter
        legacy_venv = oscompat.venv_python(_web_dir() / "venv")
        if legacy_venv.exists():
            return legacy_venv
        return Path(sys.executable)


def _web_env(*, debug: bool = False) -> dict:
    """Environment for the web process: secret + loopback host/port + gateway.

    ``debug=True`` sets ``SHAREDAGENTS_WEB_DEBUG=1`` — the server then SKIPS the
    dashboard-JWT auth (open access). Loopback-only still holds (host stays
    127.0.0.1 unless SHAREDAGENTS_WEB_HOST overrides), so this is "no auth on a
    localhost-bound UI", intended for local use behind the box's own
    boundary — NOT for an exposed bind.
    """
    cfg = _web_config()
    env = os.environ.copy()
    env["SHAREDAGENTS_WEB_SECRET"] = _ensure_web_secret()
    env["SHAREDAGENTS_WEB_HOST"] = cfg["host"]
    env["SHAREDAGENTS_WEB_PORT"] = str(cfg["port"])
    env.setdefault("SHAREDAGENTS_GATEWAY_URL", cfg["gateway"])
    if debug or os.environ.get("SHAREDAGENTS_WEB_DEBUG") == "1":
        env["SHAREDAGENTS_WEB_DEBUG"] = "1"
    return env


# ---- pidfile / liveness ----------------------------------------------


def _read_pid() -> int | None:
    pf = _web_pidfile()
    if not pf.exists():
        return None
    try:
        return int(pf.read_text().strip())
    except (ValueError, OSError):
        return None


def _is_alive(pid: int) -> bool:
    return oscompat.process_alive(pid)


def _running() -> tuple[bool, int | None]:
    pid = _read_pid()
    if pid is not None and _is_alive(pid):
        return True, pid
    if pid is not None:
        try:
            _web_pidfile().unlink()
        except OSError:
            pass
    return False, None


def _wait_for_listener(host: str, port: int, *, timeout: float = 15.0) -> bool:
    """Wait until something is listening on host:port. A bound port is
    enough — the server requires auth so any HTTP status (incl. 401)
    proves it's up; we just probe the TCP connect."""
    import socket

    deadline = time.monotonic() + timeout
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            if s.connect_ex((probe_host, port)) == 0:
                return True
        time.sleep(0.3)
    return False


# ---- start / stop -----------------------------------------------------


def start_web_background(*, debug: bool = False) -> dict:
    """Daemonize the packaged ``sharedagents_web`` server. Returns a result dict (no stdout).

    ``debug=True`` runs the server with ``SHAREDAGENTS_WEB_DEBUG=1`` (auth
    skipped)."""
    cfg = _web_config()
    url = f"http://{cfg['host']}:{cfg['port']}"
    server = _server_py()
    if not server.exists():
        return {
            "ok": False,
            "already_running": False,
            "pid": None,
            "url": url,
            "error": f"web server not found at {server} (set $SHAREDAGENTS_WEB_DIR if the repo layout differs)",
        }
    running, pid = _running()
    if running:
        return {"ok": True, "already_running": True, "pid": pid, "url": url, "error": None}
    try:
        web_python = _web_python()
    except (OSError, subprocess.SubprocessError) as e:
        return {
            "ok": False,
            "already_running": False,
            "pid": None,
            "url": url,
            "error": f"failed to prepare web venv: {e}",
        }
    log_path = _web_logfile()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fp = open(log_path, "ab", buffering=0)
    except OSError as e:
        return {
            "ok": False,
            "already_running": False,
            "pid": None,
            "url": url,
            "error": f"could not open log {log_path}: {e}",
        }
    try:
        proc = subprocess.Popen(
            web_argv(web_python),
            cwd=str(_web_dir()),
            stdin=subprocess.DEVNULL,
            stdout=log_fp,
            stderr=log_fp,
            env=_web_env(debug=debug),
            start_new_session=not oscompat.IS_WINDOWS,
            **oscompat.detach_kwargs(),
        )
    except OSError as e:
        log_fp.close()
        return {
            "ok": False,
            "already_running": False,
            "pid": None,
            "url": url,
            "error": f"failed to spawn web server: {e}",
        }
    log_fp.close()
    try:
        _web_pidfile().parent.mkdir(parents=True, exist_ok=True)
        _web_pidfile().write_text(str(proc.pid))
    except OSError:
        pass
    if not _wait_for_listener(cfg["host"], cfg["port"], timeout=20.0):
        return {
            "ok": False,
            "already_running": False,
            "pid": proc.pid,
            "url": url,
            "error": f"web server did not bind {url} within 20s (pid={proc.pid}). Check {log_path}.",
        }
    return {"ok": True, "already_running": False, "pid": proc.pid, "url": url, "error": None}


def stop_web() -> dict:
    running, pid = _running()
    if not running:
        return {"ok": True, "was_running": False, "pid": None}
    try:
        oscompat.kill_process(pid, force=False)
    except ProcessLookupError:
        pass
    for _ in range(30):
        if not _is_alive(pid):
            break
        time.sleep(0.2)
    try:
        _web_pidfile().unlink()
    except OSError:
        pass
    return {"ok": True, "was_running": True, "pid": pid}


# ---- systemd (user-instance) -----------------------------------------


def systemd_available() -> bool:
    return Path("/run/systemd/system").is_dir() and shutil.which("systemctl") is not None


def _user_unit_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    base_p = Path(base) if base else (Path.home() / ".config")
    return base_p / "systemd" / "user"


def _unit_path() -> Path:
    return _user_unit_dir() / _SERVICE_NAME


def render_web_unit(*, debug: bool = False) -> str:
    """Render the ``sharedagents-web.service`` user-instance unit for THIS install.

    ``debug=True`` bakes ``SHAREDAGENTS_WEB_DEBUG=1`` into the unit (auth
    skipped) — convenient for a local-only deployment that sits behind the
    box's own boundary."""
    home = _sharedagents_home()
    web_python = _web_python()
    command = _systemd_argv(web_argv(web_python))
    cfg = _web_config()
    venv_bin = str(Path(web_python).parent)
    path_env = os.pathsep.join(
        [
            venv_bin,
            str(Path.home() / ".local" / "bin"),
            "/usr/local/sbin",
            "/usr/local/bin",
            "/usr/sbin",
            "/usr/bin",
            "/sbin",
            "/bin",
        ]
    )
    debug_line = 'Environment="SHAREDAGENTS_WEB_DEBUG=1"\n' if debug else ""
    return f"""\
[Unit]
Description=sharedagents web frontend (localhost Flask UI, proxies the gateway)
After=network-online.target sharedagents-gateway.service
Wants=network-online.target
StartLimitIntervalSec=300
StartLimitBurst=10

[Service]
Type=simple
WorkingDirectory={_web_dir()}
EnvironmentFile=-/etc/environment
EnvironmentFile=-{home}/.env
Environment="PATH={path_env}"
Environment="SHAREDAGENTS_WEB_HOST={cfg["host"]}"
Environment="SHAREDAGENTS_WEB_PORT={cfg["port"]}"
{debug_line}ExecStart={command}
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=sharedagents-web

[Install]
WantedBy=default.target
"""


def _run_user_systemctl(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["systemctl", "--user", *argv], capture_output=True, text=True, timeout=120)


def _try_enable_linger() -> tuple[bool, str | None]:
    if not shutil.which("loginctl"):
        return False, "loginctl not found on PATH"
    try:
        cp = subprocess.run(
            ["loginctl", "enable-linger", getpass.getuser()], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"loginctl invocation failed: {e}"
    if cp.returncode == 0:
        return True, None
    return False, (cp.stderr or cp.stdout or "").strip() or f"exit {cp.returncode}"


def _linger_enabled() -> bool:
    try:
        return (Path("/var/lib/systemd/linger") / getpass.getuser()).exists()
    except OSError:
        return False


def install_systemd_service(*, enable: bool = True, start: bool = True, debug: bool = False) -> dict:
    """Install + enable + start ``sharedagents-web.service`` as a user unit.

    ``debug=True`` bakes ``SHAREDAGENTS_WEB_DEBUG=1`` into the unit.
    """
    cfg = _web_config()
    url = f"http://{cfg['host']}:{cfg['port']}"
    unit_path = _unit_path()
    result: dict = {
        "ok": False,
        "error": None,
        "unit_path": str(unit_path),
        "url": url,
        "enabled": False,
        "started": False,
        "linger": False,
        "linger_hint": None,
    }
    if not systemd_available():
        result["error"] = "systemd not available on this host"
        return result
    if not _server_py().exists():
        result["error"] = f"web server not found at {_server_py()}"
        return result
    # Ensure the secret exists in ~/.sharedagents/.env BEFORE the unit starts
    # (the server raises without SHAREDAGENTS_WEB_SECRET).
    _ensure_web_secret()
    try:
        _sharedagents_home().mkdir(parents=True, exist_ok=True)
        web_python = _web_python()  # bootstraps venv if needed
        unit = render_web_unit(debug=debug)
        unit_path.parent.mkdir(parents=True, exist_ok=True)
        unit_path.write_text(unit)
        try:
            unit_path.chmod(0o644)
        except OSError:
            pass
        cp = _run_user_systemctl(["daemon-reload"])
        if cp.returncode != 0:
            result["error"] = f"daemon-reload failed: {(cp.stderr or '').strip()}"
            return result
        if enable:
            cp = _run_user_systemctl(["enable", _SERVICE_NAME])
            result["enabled"] = cp.returncode == 0
            if cp.returncode != 0:
                result["error"] = f"enable failed: {(cp.stderr or '').strip()}"
                return result
        if start:
            cp = _run_user_systemctl(["restart", _SERVICE_NAME])
            if cp.returncode != 0:
                result["error"] = f"restart failed: {(cp.stderr or '').strip()}"
                return result
            if not _wait_for_listener(cfg["host"], cfg["port"], timeout=20.0):
                result["error"] = (
                    f"{_SERVICE_NAME} started but {url} did not come up within 20s "
                    f"— check `journalctl --user -u {_SERVICE_NAME}`."
                )
                return result
            result["started"] = True
        if _linger_enabled():
            result["linger"] = True
        else:
            ok, hint = _try_enable_linger()
            result["linger"] = ok
            result["linger_hint"] = None if ok else hint
    except (OSError, subprocess.SubprocessError) as e:
        result["error"] = f"install failed: {e}"
        return result
    _ = web_python
    result["ok"] = True
    return result


# ---- subcommands ------------------------------------------------------


def cmd_start(args: argparse.Namespace) -> int:
    r = start_web_background(debug=getattr(args, "debug", False))
    if r["already_running"]:
        sys.stdout.write(f"web already running (pid={r['pid']}) at {r['url']}\n")
        return 0
    if not r["ok"]:
        sys.stderr.write(f"web start failed: {r['error']}\n")
        return 1
    sys.stdout.write(f"web started (pid={r['pid']}) at {r['url']}\n")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:  # noqa: ARG001
    r = stop_web()
    if not r["was_running"]:
        sys.stdout.write("web not running\n")
        return 0
    sys.stdout.write(f"web stopped (pid={r['pid']})\n")
    return 0


def cmd_restart(args: argparse.Namespace) -> int:
    stop_web()
    return cmd_start(args)


def cmd_status(args: argparse.Namespace) -> int:  # noqa: ARG001
    cfg = _web_config()
    running, pid = _running()
    if running:
        sys.stdout.write(f"web: running (pid={pid}) at http://{cfg['host']}:{cfg['port']}\n")
    else:
        sys.stdout.write("web: stopped\n")
    if systemd_available() and _unit_path().exists():
        cp = _run_user_systemctl(["is-enabled", _SERVICE_NAME])
        sys.stdout.write(f"systemd unit: installed (is-enabled: {(cp.stdout or cp.stderr).strip()})\n")
    return 0 if running else 1


def cmd_logs(args: argparse.Namespace) -> int:  # noqa: ARG001
    log = _web_logfile()
    if not log.exists():
        sys.stderr.write(f"no log yet at {log}\n")
        return 1
    try:
        os.execvp("tail", ["tail", "-f", str(log)])
    except OSError as e:
        sys.stderr.write(f"could not tail {log}: {e}\n")
        return 1
    return 0


def cmd_install_service(args: argparse.Namespace) -> int:
    r = install_systemd_service(start=not getattr(args, "no_start", False), debug=getattr(args, "debug", False))
    if not r["ok"]:
        sys.stderr.write(f"install-service failed: {r['error']}\n")
        return 1
    sys.stdout.write(f"installed {r['unit_path']}\n")
    sys.stdout.write(f"enabled={r['enabled']} started={r['started']} at {r['url']}\n")
    if not r["linger"]:
        sys.stdout.write(
            f"note: boot-persistence needs linger — run "
            f"`loginctl enable-linger {getpass.getuser()}` "
            f"({r['linger_hint']})\n"
        )
    return 0


def cmd_fetch_voice_models(args: argparse.Namespace) -> int:
    from sharedagents_web import voice_models

    argv = []
    if args.dir:
        argv += ["--dir", args.dir]
    for kw in args.keyword or []:
        argv += ["--keyword", kw]
    if args.list:
        argv.append("--list")
    return voice_models.main(argv)


# ---- parser wiring ----------------------------------------------------


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "web",
        help="Web frontend lifecycle: start/stop/restart/status/logs/install-service",
    )
    inner = p.add_subparsers(dest="web_cmd")
    p_start = inner.add_parser("start", help="Daemonize the web server (loopback :8110)")
    p_start.add_argument(
        "--debug", action="store_true", help="Run with SHAREDAGENTS_WEB_DEBUG=1 — skip dashboard-JWT auth (local use)"
    )
    p_start.set_defaults(_dispatch=cmd_start)
    inner.add_parser("stop", help="SIGTERM the web server").set_defaults(_dispatch=cmd_stop)
    p_restart = inner.add_parser("restart", help="Stop + start the web server")
    p_restart.add_argument(
        "--debug", action="store_true", help="Run with SHAREDAGENTS_WEB_DEBUG=1 — skip auth (local use)"
    )
    p_restart.set_defaults(_dispatch=cmd_restart)
    inner.add_parser("status", help="Pid / uptime / url").set_defaults(_dispatch=cmd_status)
    inner.add_parser("logs", help="tail -f web.log").set_defaults(_dispatch=cmd_logs)
    psvc = inner.add_parser(
        "install-service",
        help="Install + enable + start the sharedagents-web systemd user service",
    )
    psvc.add_argument("--no-start", action="store_true", help="Install + enable but don't start now")
    psvc.add_argument(
        "--debug", action="store_true", help="Bake SHAREDAGENTS_WEB_DEBUG=1 into the unit — skip auth (local use)"
    )
    psvc.set_defaults(_dispatch=cmd_install_service)
    pf = inner.add_parser(
        "fetch-voice-models",
        help="Download the CC BY-NC-SA wake-word ONNX models into <home>/voice-models",
    )
    pf.add_argument(
        "--dir", type=str, default=None, help="target directory (default: <sharedagents home>/voice-models)"
    )
    pf.add_argument(
        "--keyword",
        action="append",
        default=[],
        help="also fetch this wake-word model (e.g. --keyword hey_mycroft_v0.1)",
    )
    pf.add_argument("--list", action="store_true", help="list models and exit")
    pf.set_defaults(_dispatch=cmd_fetch_voice_models)


def dispatch(args: argparse.Namespace) -> int:
    fn = getattr(args, "_dispatch", None)
    if fn is None:
        # bare `sharedagents web` — print help-ish summary, exit 0
        sys.stdout.write(
            "usage: sharedagents web {start,stop,restart,status,logs,install-service,fetch-voice-models}\n"
        )
        return 0
    return int(fn(args))


__all__ = [
    "add_subparser",
    "dispatch",
    "start_web_background",
    "stop_web",
    "install_systemd_service",
    "render_web_unit",
]
