"""CLI subcommands for the north gateway (+ paired south orchestrator).

Wired into the top-level ``sharedagents`` parser as the ``gateway`` subcommand:

    sharedagents gateway start            # daemonize orchestrator then gateway
    sharedagents gateway stop             # SIGTERM gateway then orchestrator
    sharedagents gateway restart
    sharedagents gateway status           # both pids/listeners
    sharedagents gateway serve            # foreground NORTH only
    sharedagents gateway logs             # tail -f gateway.log

Daemon control uses pidfile signals. The gateway pidfile lives at
``~/.sharedagents/gateway.pid``; the orchestrator at
``~/.sharedagents/orchestrator.pid``. ``gateway start`` always brings up
both (south first).

Cancel cascade: SIGTERM on the north gateway drains the edge listener
without aborting south turns. ``gateway stop`` then SIGTERMs the
orchestrator so operators still have a full-stack stop button.
"""

from __future__ import annotations

import argparse
import getpass
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

from sharedagents_core import oscompat

from .gateway import (
    _gateway_config,
    _gateway_pidfile,
    _orchestrator_config,
    _orchestrator_pidfile,
)

# ---- systemd service (user-instance) ---------------------------------
#
# The setup wizard (and ``sharedagents gateway install-service``) generate a
# unit tailored to THIS install — current venv, current ``SHAREDAGENTS_HOME``
# — and write it under the invoking user's systemd-user directory
# (``~/.config/systemd/user``). Everything happens through
# ``systemctl --user``: no sudo, no system unit, no root-owned files.
# Reboot persistence is provided by ``loginctl enable-linger`` (also
# best-effort, no sudo on distros that allow self-linger via polkit).
# ``deploy/systemd/sharedagents-gateway.service`` stays in the repo as a
# readable reference, but it is NOT what gets installed.

_SERVICE_NAME = "sharedagents-gateway.service"
_ORCH_SERVICE_NAME = "sharedagents-orchestrator.service"


def _user_unit_dir() -> Path:
    """``~/.config/systemd/user`` — the per-user systemd unit dir.

    Honours ``XDG_CONFIG_HOME`` so test fixtures (and operators who
    relocate their XDG dirs) get the same answer systemd does.
    """
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    base = Path(xdg) if xdg else (Path.home() / ".config")
    return base / "systemd" / "user"


def _unit_path() -> Path:
    """Absolute path of the user-instance gateway unit file."""
    return _user_unit_dir() / _SERVICE_NAME


# ---- helpers ---------------------------------------------------------


def _read_pid() -> int | None:
    return _read_pidfile(_gateway_pidfile())


def _read_pidfile(pidfile: Path) -> int | None:
    if not pidfile.exists():
        return None
    try:
        return int(pidfile.read_text().strip())
    except (OSError, ValueError):
        return None


def _is_alive(pid: int) -> bool:
    return oscompat.process_alive(pid, permission_denied_alive=False)


def _port_holder(host: str, port: int) -> int | None:
    """Return the pid holding ``host:port`` in LISTEN state, or None.

    Parses ``/proc/net/tcp`` directly so we don't depend on ``ss`` /
    ``lsof`` being on PATH and don't require sudo. Only matches IPv4 +
    IPv6 LISTEN sockets bound to the requested host (or 0.0.0.0 / ::).
    Returns the pid if the inode-to-pid lookup succeeds; if the socket
    is owned by another user (kernel exposes inode 0), returns -1 as a
    sentinel meaning "port is in use but pid not visible to us" so the
    caller can still surface a clean error.
    """
    if oscompat.IS_WINDOWS:
        return -1 if oscompat.port_in_use(host, port) else None
    import socket as _socket

    try:
        ip_int = int.from_bytes(
            _socket.inet_aton("127.0.0.1" if host in ("localhost", "127.0.0.1") else host),
            "big",
        )
    except OSError:
        return None
    target_local_hex = f"{ip_int:08X}".upper()
    target_any_hex = "00000000"
    target_port_hex = f"{port:04X}".upper()
    inode_to_match: set[str] = set()
    for tcp_path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(tcp_path) as fh:
                next(fh, None)  # header
                for line in fh:
                    cols = line.split()
                    if len(cols) < 10:
                        continue
                    state = cols[3]
                    if state != "0A":  # 0A = LISTEN
                        continue
                    local = cols[1]
                    if ":" not in local:
                        continue
                    addr_hex, port_hex = local.split(":", 1)
                    if port_hex.upper() != target_port_hex:
                        continue
                    if "tcp6" in tcp_path:
                        # IPv6 — accept any address since dual-stack
                        # listeners on `::` cover IPv4-mapped too.
                        inode_to_match.add(cols[9])
                    else:
                        if addr_hex.upper() in (target_local_hex, target_any_hex):
                            inode_to_match.add(cols[9])
        except OSError:
            continue
    if not inode_to_match:
        return None
    # Walk /proc/*/fd/* looking for socket:[<inode>] symlinks. Skip
    # processes whose fd dir we can't read (other users).
    visible_pid: int | None = None
    for proc_dir in os.listdir("/proc"):
        if not proc_dir.isdigit():
            continue
        fd_dir = f"/proc/{proc_dir}/fd"
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(f"{fd_dir}/{fd}")
            except OSError:
                continue
            if target.startswith("socket:[") and target[8:-1] in inode_to_match:
                visible_pid = int(proc_dir)
                return visible_pid
    # Inode found in /proc/net/tcp but no /proc/*/fd link visible —
    # the listener is owned by another user.
    return -1


def _running() -> tuple[bool, int | None]:
    """Return ``(alive, pid)`` for the north gateway tracked by the pidfile."""
    return _running_pidfile(_gateway_pidfile(), "gateway")


def _orch_running() -> tuple[bool, int | None]:
    """Return ``(alive, pid)`` for the south orchestrator pidfile."""
    return _running_pidfile(_orchestrator_pidfile(), "orchestrator")


def _pid_is_sharedagents(pid: int, role: str) -> bool:
    """True when ``pid`` really is our ``sharedagents <role> serve``.

    ``os.kill(pid, 0)`` only proves SOMETHING holds that number. Pids are
    recycled -- most sharply in a container, where a persisted ``$SHAREDAGENTS_HOME``
    carries a pidfile from a previous boot and the new namespace hands the same
    low number to an unrelated process. The stale entry then reads as "already
    running", the daemon is never started, and every turn fails against a port
    nothing is listening on.

    ``/proc`` is the cheap authority, and this only ever answers False when it
    can point at a process that is positively NOT this daemon: an unreadable
    ``/proc`` entry, or a platform without ``/proc`` at all, keeps the old
    permissive answer. Deployments name their ExecStart in more ways than this
    can enumerate, and a false "not running" would spawn a second daemon that
    cannot bind -- the exact failure the port guard exists to prevent.
    """
    if oscompat.IS_WINDOWS:
        return True  # No /proc identity; pidfile + liveness + bind probe only.
    cmdline = Path(f"/proc/{pid}/cmdline")
    try:
        argv = cmdline.read_bytes().decode("utf-8", "replace").split("\0")
    except OSError:
        return True  # cannot tell; assume the pidfile is honest
    joined = " ".join(part for part in argv if part)
    if not joined:
        return True
    if "sharedagents" not in joined:
        return False  # a recycled pid now held by something else entirely
    other = "orchestrator" if role == "gateway" else "gateway"
    # Both daemons match "sharedagents", so only reject when the cmdline names the
    # SIBLING role and not this one -- that is the pidfiles crossed over.
    return not (other in joined and role not in joined)


def _running_pidfile(pidfile: Path, role: str = "") -> tuple[bool, int | None]:
    pid = _read_pidfile(pidfile)
    if pid is None:
        return False, None
    if _is_alive(pid) and (not role or _pid_is_sharedagents(pid, role)):
        return True, pid
    try:
        pidfile.unlink()
    except FileNotFoundError:
        pass
    return False, pid


def _wait_for_listener(url: str, *, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{url}/health", timeout=1.0)
            # Require an explicit 200 — any other status (404/500, or a
            # *different* service that happens to hold the port) is NOT
            # our gateway being ready. The old "any status_code" check
            # would treat a wrong service on the port as alive.
            if r.status_code == 200:
                return True
        except httpx.ConnectError:
            pass
        time.sleep(0.2)
    return False


def _gateway_logfile() -> Path:
    """Path to the unified gateway log shared by every agent."""
    from sharedagents_core.layout import gateway_logfile

    return gateway_logfile()


def _sharedagents_home() -> Path:
    return Path(os.environ.get("SHAREDAGENTS_HOME") or Path.home() / ".sharedagents")


def _stack_argv(kind: str) -> list[str]:
    """Argv to spawn south (``orchestrator``) or north (``gateway``) serve."""
    exe = shutil.which("sharedagents")
    if kind == "orchestrator":
        if exe:
            return [exe, "orchestrator", "serve"]
        return [sys.executable, "-m", "sharedagents_tui", "orchestrator", "serve"]
    if kind == "gateway":
        if exe:
            return [exe, "gateway", "serve"]
        return [sys.executable, "-m", "sharedagents_tui", "gateway", "serve"]
    raise ValueError(f"unknown stack kind {kind!r}")


def _port_conflict_error(url: str, host: str, port: int, holder: int, *, label: str) -> str:
    if holder == -1:
        return (
            f"another process already holds {url} ({label}; owned by a different "
            f"user — pid not visible). Stop it as that user with "
            f"`systemctl --user stop sharedagents-{label}` (if user-systemd "
            f"managed) or whatever started it."
        )
    return (
        f"another process already holds {url} ({label}; pid={holder}). The "
        f"CLI's pidfile doesn't track it — stop it with `sharedagents "
        f"gateway stop`, or `systemctl --user restart sharedagents-{label}` "
        f"if user-systemd managed."
    )


def _spawn_serve(kind: str, log_path: Path) -> tuple[subprocess.Popen | None, str | None]:
    """Spawn a serve process; return ``(proc, error)``."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = open(log_path, "ab", buffering=0)
    try:
        proc = subprocess.Popen(
            _stack_argv(kind),
            stdin=subprocess.DEVNULL,
            stdout=log_fp,
            stderr=log_fp,
            env=os.environ.copy(),
            start_new_session=not oscompat.IS_WINDOWS,
            **oscompat.detach_kwargs(),
        )
    except OSError as e:
        log_fp.close()
        return None, f"failed to spawn {kind}: {e}"
    log_fp.close()
    return proc, None


def start_gateway_background() -> dict:
    """Daemonize south orchestrator then north gateway; wait for both listeners.

    Returns ``{"ok", "already_running", "pid", "pid_orchestrator", "url",
    "orchestrator_url", "error"}``. ``pid`` is the north gateway pid
    (back-compat for callers). Kept free of stdout chatter so the setup
    wizard can call it directly; :func:`cmd_start` wraps it for the CLI.

    The whole check→spawn→wait sequence runs under an exclusive
    per-user file lock (``$SHAREDAGENTS_HOME/gateway.start.lock``).
    """
    gw = _gateway_config()
    orch = _orchestrator_config()
    url = f"http://{gw['host']}:{gw['port']}"
    orch_url = f"http://{orch['host']}:{orch['port']}"

    home = _sharedagents_home()
    lock_fp = None
    try:
        home.mkdir(parents=True, exist_ok=True)
        # Truncation can fail before locking on Windows' mandatory byte lock.
        lock_fp = open(home / "gateway.start.lock", "a" if oscompat.IS_WINDOWS else "w")
        oscompat.file_lock(lock_fp)
    except OSError:
        lock_fp = None

    try:
        gw_running, gw_pid = _running()
        orch_running, orch_pid = _orch_running()

        # Both already up → idempotent success.
        if gw_running and orch_running:
            return {
                "ok": True,
                "already_running": True,
                "pid": gw_pid,
                "pid_orchestrator": orch_pid,
                "url": url,
                "orchestrator_url": orch_url,
                "error": None,
            }

        # Port conflicts for whichever side we still need to start.
        if not orch_running:
            holder = _port_holder(orch["host"], int(orch["port"]))
            if holder is not None:
                err = _port_conflict_error(
                    orch_url,
                    orch["host"],
                    int(orch["port"]),
                    holder,
                    label="orchestrator",
                )
                return {
                    "ok": False,
                    "already_running": False,
                    "pid": gw_pid,
                    "pid_orchestrator": holder if holder and holder > 0 else None,
                    "url": url,
                    "orchestrator_url": orch_url,
                    "error": err,
                }
        if not gw_running:
            holder = _port_holder(gw["host"], int(gw["port"]))
            if holder is not None:
                err = _port_conflict_error(
                    url,
                    gw["host"],
                    int(gw["port"]),
                    holder,
                    label="gateway",
                )
                return {
                    "ok": False,
                    "already_running": False,
                    "pid": holder if holder and holder > 0 else None,
                    "pid_orchestrator": orch_pid,
                    "url": url,
                    "orchestrator_url": orch_url,
                    "error": err,
                }

        log_path = _gateway_logfile()

        if not orch_running:
            proc, err = _spawn_serve("orchestrator", log_path)
            if err:
                return {
                    "ok": False,
                    "already_running": False,
                    "pid": None,
                    "pid_orchestrator": None,
                    "url": url,
                    "orchestrator_url": orch_url,
                    "error": err,
                }
            assert proc is not None
            if not _wait_for_listener(orch_url, timeout=20.0):
                return {
                    "ok": False,
                    "already_running": False,
                    "pid": None,
                    "pid_orchestrator": proc.pid,
                    "url": url,
                    "orchestrator_url": orch_url,
                    "error": (
                        f"orchestrator listener at {orch_url} did not respond "
                        f"within 20s (pid={proc.pid}). Check {log_path}."
                    ),
                }
            orch_pid = proc.pid
        else:
            orch_pid = orch_pid

        if not gw_running:
            proc, err = _spawn_serve("gateway", log_path)
            if err:
                return {
                    "ok": False,
                    "already_running": False,
                    "pid": None,
                    "pid_orchestrator": orch_pid,
                    "url": url,
                    "orchestrator_url": orch_url,
                    "error": err,
                }
            assert proc is not None
            if not _wait_for_listener(url, timeout=20.0):
                return {
                    "ok": False,
                    "already_running": False,
                    "pid": proc.pid,
                    "pid_orchestrator": orch_pid,
                    "url": url,
                    "orchestrator_url": orch_url,
                    "error": (
                        f"gateway listener at {url} did not respond within 20s (pid={proc.pid}). Check {log_path}."
                    ),
                }
            gw_pid = proc.pid

        return {
            "ok": True,
            "already_running": False,
            "pid": gw_pid,
            "pid_orchestrator": orch_pid,
            "url": url,
            "orchestrator_url": orch_url,
            "error": None,
        }
    finally:
        if lock_fp is not None:
            try:
                oscompat.file_unlock(lock_fp)
            except OSError:
                pass
            lock_fp.close()


def systemd_available() -> bool:
    """True iff this host runs systemd and ``systemctl`` is on PATH.

    ``/run/systemd/system`` exists only when systemd is PID 1, so this
    cleanly returns False inside most containers / dev sandboxes — the
    caller then falls back to a plain background gateway.
    """
    return Path("/run/systemd/system").is_dir() and shutil.which("systemctl") is not None


def _resolve_gateway_exec(role: str = "gateway") -> str:
    """Absolute ExecStart line for a unit, by deployment ``role`` (§13 split).

    Prefers the installed ``sharedagents`` console script (a ``pip install``
    venv puts one on PATH); falls back to ``<python> -m sharedagents_tui …``
    so an editable / PYTHONPATH-only install still produces a working unit.

    * ``"gateway"`` (north) → ``sharedagents gateway serve``
    * ``"orchestrator"`` (south) → ``sharedagents orchestrator serve``
    """
    if role not in ("gateway", "orchestrator"):
        raise ValueError(f"invalid systemd role {role!r}: must be 'gateway' or 'orchestrator'")
    exe = shutil.which("sharedagents")
    if role == "orchestrator":
        if exe:
            return f"{exe} orchestrator serve"
        return f"{sys.executable} -m sharedagents_tui orchestrator serve"
    if exe:
        return f"{exe} gateway serve"
    return f"{sys.executable} -m sharedagents_tui gateway serve"


def _npm_prefix_bins() -> list[str]:
    """Candidate bin dirs where ``npm install -g`` may have landed CLIs.

    The wizard (``setup_external_agents``) installs the external coding
    agents via ``npm install -g <pkg>``; the destination depends on
    ``npm config get prefix``. We probe npm if it's on PATH and always
    include ``~/.npm-global/bin`` (the prefix the wizard's
    ``_ensure_user_npm_prefix`` sets when the default prefix is not
    user-writable). Returning a list — caller dedupes.
    """
    out: list[str] = []
    npm = shutil.which("npm")
    if npm:
        try:
            cp = subprocess.run(  # noqa: S603
                [npm, "config", "get", "prefix"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if cp.returncode == 0:
                prefix = (cp.stdout or "").strip()
                if prefix and prefix not in ("undefined", "null"):
                    out.append(str(Path(prefix) / "bin"))
        except (OSError, subprocess.SubprocessError):
            pass
    out.append(str(Path.home() / ".npm-global" / "bin"))
    return out


def render_gateway_unit(role: str = "gateway") -> str:
    """Render the ``sharedagents-{gateway,orchestrator}.service`` *user-instance*
    unit for THIS install, by deployment ``role`` (§13 split).

    ``role="gateway"`` (north, default) and ``role="orchestrator"`` (south)
    produce the two split units.

    User units run as the invoking user automatically — no ``User=`` /
    ``Group=`` directives — and target ``default.target`` (the user
    instance's analog of ``multi-user.target``). Everything the shipped
    template hard-codes (home, venv path, PATH) is resolved at call
    time so the unit is correct on any box a fresh ``pip install``
    lands on.

    Reboot-survival is provided by ``loginctl enable-linger`` (driven
    by :func:`install_systemd_service`), not by the unit itself.
    """
    if role not in ("gateway", "orchestrator"):
        raise ValueError(f"invalid systemd role {role!r}: must be 'gateway' or 'orchestrator'")
    home = _sharedagents_home()
    # Build a deterministic PATH rather than inheriting the (often
    # polluted) interactive-shell PATH: the venv bin dir first — so the
    # gateway's child processes (bundled MCP servers, CLI engines)
    # resolve against the SAME interpreter the gateway runs under —
    # then ``~/.local/bin`` (where ``pip install --user`` / pipx land
    # the ``sharedagents`` script), the npm-global bin (where the wizard
    # installs claude/codex/antigravity via ``npm install -g``), and
    # the standard system dirs.
    venv_bin = str(Path(sys.executable).parent)
    _path_parts = [
        venv_bin,
        str(Path.home() / ".local" / "bin"),
        # CLI-agent install destinations: query npm's configured prefix
        # (the wizard's ``_ensure_user_npm_prefix`` points this at
        # ``~/.npm-global`` on first install), and always include the
        # well-known default as a defensive fallback. Without these the
        # gateway-launched ``claude`` / ``codex`` / ``antigravity``
        # children FileNotFoundError out even though ``sharedagents agents``
        # reports them as installed.
        *_npm_prefix_bins(),
        str(Path.home() / ".local" / "share" / "cursor-agent" / "bin"),
        "/usr/local/sbin",
        "/usr/local/bin",
        "/usr/sbin",
        "/usr/bin",
        "/sbin",
        "/bin",
        "/snap/bin",
    ]
    _seen: set[str] = set()
    path_env = os.pathsep.join(p for p in _path_parts if p and not (p in _seen or _seen.add(p)))
    if role == "orchestrator":
        descr = "sharedagents SOUTH orchestrator (write-master; runs agent turns; loopback :8099)"
        deps = "After=network-online.target\nWants=network-online.target"
        ident = "sharedagents-orchestrator"
    else:
        descr = "sharedagents NORTH gateway (interfaces + reads; proxies commands to the orchestrator)"
        # Soft dep: the gateway still serves reads from the shared log even if
        # the orchestrator is momentarily down, and is restartable WITHOUT
        # killing in-flight turns (they live in the orchestrator).
        deps = (
            "After=network-online.target sharedagents-orchestrator.service\n"
            "Wants=network-online.target\n"
            "Wants=sharedagents-orchestrator.service"
        )
        ident = "sharedagents-gateway"
    return f"""\
[Unit]
Description={descr}
{deps}
# Disable the start rate-limit so ``Restart=always`` NEVER gives up:
# this unit is also a self-protection supervisor — if an agent inside
# the process repeatedly signal-kills it (e.g. a cross-call
# ``kill <pid>`` that no static command guard can catch), systemd must
# keep bringing it back rather than latch into ``failed`` after a burst.
# (An intentional ``systemctl stop`` is still honoured — that path is
# blocked at the agent's command guard instead.)
StartLimitIntervalSec=0

[Service]
Type=simple
WorkingDirectory={home}
EnvironmentFile=-/etc/environment
EnvironmentFile=-{home}/.env
Environment="PATH={path_env}"
ExecStart={_resolve_gateway_exec(role)}
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier={ident}

[Install]
WantedBy=default.target
"""


def _run_user_systemctl(argv: list[str]) -> subprocess.CompletedProcess:
    """Run ``systemctl --user <argv>``. No sudo, no special privileges
    — talks to the caller's user-instance manager via ``$XDG_RUNTIME_DIR``.
    """
    return subprocess.run(  # noqa: S603, S607
        ["systemctl", "--user", *argv],
        capture_output=True,
        text=True,
        timeout=120,
    )


def _try_enable_linger() -> tuple[bool, str | None]:
    """Best-effort ``loginctl enable-linger`` for the current user.

    Linger is what makes user-instance services start at boot rather
    than at login — without it, ``WantedBy=default.target`` only kicks
    in when the user actually logs in via a session. Many distros allow
    the user to enable linger for themselves via polkit
    (``org.freedesktop.login1.set-self-linger``); others require admin
    auth. We attempt it without sudo and return ``(False, hint)`` if it
    fails, so the caller can surface a one-line note instead of
    pretending boot-persistence is guaranteed.
    """
    if not shutil.which("loginctl"):
        return False, "loginctl not found on PATH"
    user = getpass.getuser()
    try:
        cp = subprocess.run(  # noqa: S603, S607
            ["loginctl", "enable-linger", user],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"loginctl invocation failed: {e}"
    if cp.returncode == 0:
        return True, None
    hint = (cp.stderr or cp.stdout or "").strip()
    return False, hint or f"loginctl enable-linger exit {cp.returncode}"


def _linger_enabled() -> bool:
    """True iff the current user already has linger enabled.

    Reads ``/var/lib/systemd/linger/<user>`` directly: cheap, doesn't
    require ``loginctl show-user`` (which can fail without a logind
    session — e.g. inside a containerised CI runner).
    """
    try:
        return (Path("/var/lib/systemd/linger") / getpass.getuser()).exists()
    except OSError:
        return False


def install_systemd_service(*, enable: bool = True, start: bool = True, interactive: bool = True) -> dict:
    """Install + enable + start ``sharedagents-gateway.service`` as a
    *user-instance* systemd service — no sudo, no system unit.

    Generates a unit tailored to this install (see
    :func:`render_gateway_unit`), writes it to
    ``~/.config/systemd/user/sharedagents-gateway.service``,
    ``daemon-reload``s the user manager, then ``enable``s and
    ``restart``s the unit (``restart`` covers both the cold-install and
    the re-run cases). Best-effort ``loginctl enable-linger`` ensures
    the unit also starts at boot rather than only at next login.

    Returns ``{"ok", "error", "unit_path", "url", "enabled", "started",
    "linger", "linger_hint"}``. ``linger`` is True when linger is
    already on (or this call turned it on); ``linger_hint`` carries the
    short message to surface if linger could not be enabled.

    ``interactive`` is accepted for API symmetry with the legacy
    sudo-based signature but is unused: nothing in this code path
    prompts for a password.
    """
    del interactive  # see docstring
    gw = _gateway_config()
    url = f"http://{gw['host']}:{gw['port']}"
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
    # WorkingDirectory must exist before systemd will start the unit.
    try:
        _sharedagents_home().mkdir(parents=True, exist_ok=True)
    except OSError as e:
        result["error"] = f"could not create {_sharedagents_home()}: {e}"
        return result

    # §13 hard cutover: install the SOUTH orchestrator unit FIRST (the gateway
    # depends on it), then the NORTH gateway. The orchestrator owns the agents
    # and runs turns; the gateway proxies commands to it.
    try:
        orch_path = _user_unit_dir() / _ORCH_SERVICE_NAME
        orch_path.parent.mkdir(parents=True, exist_ok=True)
        orch_path.write_text(render_gateway_unit(role="orchestrator"))
        try:
            orch_path.chmod(0o644)
        except OSError:
            pass
        _run_user_systemctl(["daemon-reload"])
        if enable:
            _run_user_systemctl(["enable", _ORCH_SERVICE_NAME])
        if start:
            cp = _run_user_systemctl(["restart", _ORCH_SERVICE_NAME])
            if cp.returncode != 0:
                result["error"] = f"systemctl --user restart {_ORCH_SERVICE_NAME} failed: {(cp.stderr or '').strip()}"
                return result
            from .gateway import _orchestrator_config

            orch = _orchestrator_config()
            if not _wait_for_listener(f"http://{orch['host']}:{orch['port']}", timeout=20.0):
                result["error"] = (
                    f"{_ORCH_SERVICE_NAME} started but its loopback listener "
                    f"did not come up within 20s — check "
                    f"`journalctl --user -u {_ORCH_SERVICE_NAME}`."
                )
                return result
        result["orchestrator_unit_path"] = str(orch_path)
    except OSError as e:
        result["error"] = f"could not install orchestrator unit: {e}"
        return result

    unit = render_gateway_unit(role="gateway")
    try:
        # Write the unit straight into the user systemd dir — no sudo,
        # no tee, no temp file.
        unit_path.parent.mkdir(parents=True, exist_ok=True)
        unit_path.write_text(unit)
        try:
            unit_path.chmod(0o644)
        except OSError:
            # chmod failure is non-fatal (the write above already
            # created it with the umask default); don't abort over it.
            pass
        cp = _run_user_systemctl(["daemon-reload"])
        if cp.returncode != 0:
            result["error"] = f"systemctl --user daemon-reload failed: {(cp.stderr or '').strip()}"
            return result
        if enable:
            cp = _run_user_systemctl(["enable", _SERVICE_NAME])
            result["enabled"] = cp.returncode == 0
            if cp.returncode != 0:
                result["error"] = f"systemctl --user enable failed: {(cp.stderr or '').strip()}"
                return result
        if start:
            # `restart` starts a stopped unit and bounces a running one,
            # so it's correct whether or not the gateway was already up.
            cp = _run_user_systemctl(["restart", _SERVICE_NAME])
            if cp.returncode != 0:
                result["error"] = f"systemctl --user restart failed: {(cp.stderr or '').strip()}"
                return result
            if not _wait_for_listener(url, timeout=20.0):
                result["error"] = (
                    f"{_SERVICE_NAME} started but the listener at {url} did "
                    f"not come up within 20s — check "
                    f"`journalctl --user -u {_SERVICE_NAME}`."
                )
                return result
            result["started"] = True
        # Best-effort: ensure the user-instance survives reboot. We do
        # this AFTER enable+start so the install isn't blocked when
        # linger needs admin auth on this distro.
        if _linger_enabled():
            result["linger"] = True
        else:
            ok, hint = _try_enable_linger()
            result["linger"] = ok
            result["linger_hint"] = None if ok else hint
    except OSError as e:
        result["error"] = f"could not write {unit_path}: {e}"
        return result
    except subprocess.SubprocessError as e:
        result["error"] = f"systemctl invocation failed: {e}"
        return result
    result["ok"] = True
    return result


# ---- subcommands -----------------------------------------------------


def _sigterm_wait(pid: int, *, label: str, timeout: float = 15.0) -> int:
    """SIGTERM ``pid``, wait, escalate to SIGKILL. Return 0 on clean stop."""
    try:
        oscompat.kill_process(pid, force=False)
    except ProcessLookupError:
        return 0
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _is_alive(pid):
            sys.stdout.write(f"{label} stopped (pid={pid})\n")
            return 0
        time.sleep(0.2)
    sys.stderr.write(f"{label} pid={pid} did not exit within {int(timeout)}s; sending SIGKILL\n")
    try:
        oscompat.kill_process(pid, force=True)
    except ProcessLookupError:
        pass
    return 1


def _stop_one(
    *,
    running: bool,
    pid: int | None,
    host: str,
    port: int,
    label: str,
) -> int:
    """Stop one stack member by pidfile or port-holder orphan reap."""
    if not running:
        holder = _port_holder(host, int(port))
        if holder and holder > 0:
            sys.stdout.write(f"{label} pidfile clean but port {port} held by orphan pid={holder}; reaping\n")
            return _sigterm_wait(holder, label=label, timeout=10.0)
        if holder == -1:
            sys.stderr.write(
                f"port {port} held by a different user — cannot "
                f"reap. Stop it as that user with "
                f"`systemctl --user stop sharedagents-{label}`.\n"
            )
            return 1
        sys.stdout.write(f"{label} is not running\n")
        return 0
    assert pid is not None
    return _sigterm_wait(pid, label=label, timeout=15.0)


def cmd_start(args: argparse.Namespace) -> int:  # noqa: ARG001
    """Daemonize south orchestrator then north gateway; wait for both listeners.

    Idempotent: a no-op when both are already alive. Port-conflict guard
    refuses to spawn a non-binding zombie on either side.
    """
    result = start_gateway_background()
    if result["already_running"]:
        sys.stdout.write(
            f"stack already running "
            f"(gateway pid={result['pid']}, "
            f"orchestrator pid={result.get('pid_orchestrator')}) "
            f"at {result['url']}\n"
        )
        return 0
    if not result["ok"]:
        sys.stderr.write(f"refusing to start: {result['error']}\n")
        return 1
    sys.stdout.write(
        f"stack started "
        f"(gateway pid={result['pid']}, "
        f"orchestrator pid={result.get('pid_orchestrator')}) "
        f"gateway={result['url']} "
        f"orchestrator={result.get('orchestrator_url')}\n"
    )
    return 0


def cmd_stop(args: argparse.Namespace) -> int:  # noqa: ARG001
    """SIGTERM north gateway first, then south orchestrator.

    North-first so clients disconnect before turns are aborted. Port-holder
    orphan reaping applies to both sides.
    """
    gw = _gateway_config()
    orch = _orchestrator_config()
    gw_running, gw_pid = _running()
    orch_running, orch_pid = _orch_running()
    rc_gw = _stop_one(
        running=gw_running,
        pid=gw_pid,
        host=gw["host"],
        port=int(gw["port"]),
        label="gateway",
    )
    rc_orch = _stop_one(
        running=orch_running,
        pid=orch_pid,
        host=orch["host"],
        port=int(orch["port"]),
        label="orchestrator",
    )
    return 1 if (rc_gw or rc_orch) else 0


def cmd_restart(args: argparse.Namespace) -> int:
    # the units are loaded. Default is NORTH only so a confirmed bounce
    # cannot SIGTERM the south orchestrator (2026-09-14 hung-shutdown).
    # ``--stack`` / ``--hard`` restarts orchestrator then gateway via
    # systemd. The pidfile fallback still chains stop+start for
    # non-systemd hosts.
    stack = bool(getattr(args, "stack", False) or getattr(args, "hard", False))
    if systemd_available():
        units = [_ORCH_SERVICE_NAME, _SERVICE_NAME] if stack else [_SERVICE_NAME]
        present = []
        for unit in units:
            cat = _run_user_systemctl(["cat", unit])
            if cat.returncode == 0:
                present.append(unit)
        if present:
            _run_user_systemctl(["reset-failed", *present])
            cp = _run_user_systemctl(["restart", *present])
            if cp.returncode != 0:
                sys.stderr.write(f"systemctl --user restart {' '.join(present)} failed: {(cp.stderr or '').strip()}\n")
                return 1
            gw = _gateway_config()
            url = f"http://{gw['host']}:{gw['port']}"
            if not _wait_for_listener(url, timeout=20.0):
                sys.stderr.write(f"gateway listener at {url} did not return after restart\n")
                return 1
            sys.stdout.write(f"restarted via systemd: {', '.join(present)} → {url}\n")
            return 0
    rc = cmd_stop(args)
    if rc not in (0, 1):
        return rc
    return cmd_start(args)


def cmd_status(args: argparse.Namespace) -> int:  # noqa: ARG001
    """Print both stack members' pid + listener health."""
    gw = _gateway_config()
    orch = _orchestrator_config()
    gw_url = f"http://{gw['host']}:{gw['port']}"
    orch_url = f"http://{orch['host']}:{orch['port']}"
    gw_running, gw_pid = _running()
    orch_running, orch_pid = _orch_running()

    exit_rc = 0
    if orch_running:
        sys.stdout.write(f"orchestrator: running (pid={orch_pid}) at {orch_url}\n")
    else:
        sys.stdout.write(f"orchestrator: not running (configured: {orch_url})\n")
        exit_rc = 1

    if gw_running:
        sys.stdout.write(f"gateway: running (pid={gw_pid}) at {gw_url}\n")
        try:
            r = httpx.get(f"{gw_url}/", timeout=2.0)
            if r.status_code == 200:
                data = r.json()
                agents = data.get("agents") or []
                live = data.get("live_backends") or data.get("live_children") or []
                role = data.get("role") or "gateway"
                sys.stdout.write(f"  role:          {role}\n")
                sys.stdout.write(f"  agents:        {', '.join(agents) or '(none)'}\n")
                sys.stdout.write(f"  live backends: {', '.join(live) or '(none)'}\n")
            else:
                sys.stdout.write(f"  index route returned HTTP {r.status_code}\n")
        except Exception as e:  # noqa: BLE001
            sys.stdout.write(f"  index route unreachable: {e}\n")
    else:
        sys.stdout.write(f"gateway: not running (configured: {gw_url})\n")
        exit_rc = 1
    return exit_rc


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the NORTH gateway in the foreground (no daemonize).

    South is ``sharedagents orchestrator serve``. ``gateway start`` brings both up.
    """
    _augment_path_for_cli_agents()
    from .gateway import serve_gateway_main

    return serve_gateway_main(role="gateway")


def _augment_path_for_cli_agents() -> None:
    """Prepend wizard-installed CLI-agent bin dirs to ``$PATH`` in this
    process so foreground/manual gateway launches see the same external
    coding CLIs that ``render_gateway_unit()`` already grants to the
    systemd-installed gateway.

    Without this, ``sharedagents gateway serve`` started from a terminal
    inherits whatever PATH the parent shell had — which on a fresh
    install is the bare system PATH with no ``~/.npm-global/bin`` or
    cursor-agent dir, and the ``claude``/``codex``/``antigravity``
    engines fail with ``FileNotFoundError`` despite the wizard reporting
    them as installed.
    """
    current = os.environ.get("PATH", "")
    existing = current.split(os.pathsep) if current else []
    seen = {p for p in existing if p}
    extras: list[str] = []
    for cand in (
        *_npm_prefix_bins(),
        str(Path.home() / ".local" / "share" / "cursor-agent" / "bin"),
        str(Path.home() / ".local" / "bin"),
    ):
        if cand and cand not in seen:
            extras.append(cand)
            seen.add(cand)
    if extras:
        os.environ["PATH"] = os.pathsep.join([*extras, *existing])


def cmd_logs(args: argparse.Namespace) -> int:  # noqa: ARG001
    """``tail -f`` the gateway log."""
    log_path = _gateway_logfile()
    if not log_path.is_file():
        sys.stderr.write(f"no gateway log at {log_path}\n")
        return 1
    try:
        os.execvp("tail", ["tail", "-f", str(log_path)])
    except FileNotFoundError:
        sys.stderr.write("`tail` not found on PATH\n")
        return 1


def cmd_install_service(args: argparse.Namespace) -> int:
    """Install + enable + start the ``sharedagents-gateway`` user-instance
    systemd service.

    Generates a unit tailored to this install and writes it under
    ``~/.config/systemd/user/`` — no sudo, no root-owned files.
    ``systemctl --user enable`` + ``restart`` brings it up; a
    best-effort ``loginctl enable-linger`` makes it survive reboot
    (the operator gets a printed pointer if linger needs admin auth
    on this distro). The bare ``sharedagents`` command then "just works"
    — no manual ``systemctl`` step.

    The setup wizard offers this automatically; this subcommand is the
    same path for operators who skipped it then or want to re-apply the
    unit after an upgrade.
    """
    if not systemd_available():
        sys.stderr.write(
            "systemd is not available on this host — nothing to install.\n"
            "run the gateway directly instead: `sharedagents gateway start`.\n"
        )
        return 1
    enable = not getattr(args, "no_enable", False)
    start = not getattr(args, "no_start", False)
    sys.stderr.write(f"installing {_SERVICE_NAME} → {_unit_path()}\n")
    result = install_systemd_service(enable=enable, start=start)
    if not result["ok"]:
        sys.stderr.write(f"service install failed: {result['error']}\n")
        return 1
    bits = [f"installed {_SERVICE_NAME} (user-instance)"]
    if result["enabled"]:
        if result["linger"]:
            bits.append("enabled (starts on boot, linger on)")
        else:
            bits.append("enabled (starts on next login)")
    if result["started"]:
        bits.append(f"started → {result['url']}")
    sys.stdout.write("✓ " + ", ".join(bits) + "\n")
    if not result["linger"]:
        # Linger could not be self-enabled. Tell the operator what this
        # means without recommending sudo — the whole point of the
        # install path is sudo-free. Enabling linger is an admin /
        # one-time host setup; on distros where polkit allows
        # self-linger this same `sharedagents gateway install-service` will
        # succeed without help.
        sys.stdout.write(
            "  note: linger is off — the unit will start at your next "
            "login, but not on a cold boot. Boot-start needs `loginctl "
            "enable-linger` to be allowed for your user (one-time host "
            "setup; on many distros it is allowed by default via polkit "
            "and re-running install-service will pick it up).\n"
        )
        if result["linger_hint"]:
            sys.stdout.write(f"  (loginctl said: {result['linger_hint']})\n")
    return 0


# ---- argparse wiring -------------------------------------------------


def add_subparser(sub: argparse._SubParsersAction) -> None:
    """Wire ``sharedagents gateway <op>`` into the top-level parser."""
    p = sub.add_parser(
        "gateway",
        help="NORTH gateway (+ paired SOUTH orchestrator lifecycle)",
    )
    inner = p.add_subparsers(dest="gw_op")

    p_start = inner.add_parser(
        "start",
        help="Daemonize orchestrator then gateway (paired stack)",
    )
    p_start.set_defaults(_gw_dispatch=cmd_start)

    p_stop = inner.add_parser(
        "stop",
        help="SIGTERM gateway then orchestrator",
    )
    p_stop.set_defaults(_gw_dispatch=cmd_stop)

    p_restart = inner.add_parser(
        "restart",
        help="Bounce the NORTH gateway (systemd) or the paired stack (pidfile)",
    )
    p_restart.add_argument(
        "--stack",
        "--hard",
        dest="stack",
        action="store_true",
        help="Also bounce the SOUTH orchestrator (full stack)",
    )
    p_restart.set_defaults(_gw_dispatch=cmd_restart)

    p_status = inner.add_parser(
        "status",
        help="Pid/listener status for gateway and orchestrator",
    )
    p_status.set_defaults(_gw_dispatch=cmd_status)

    p_serve = inner.add_parser(
        "serve",
        help="Foreground NORTH gateway only (south: sharedagents orchestrator serve)",
    )
    p_serve.set_defaults(_gw_dispatch=cmd_serve)

    p_logs = inner.add_parser("logs", help="tail -f gateway.log")
    p_logs.set_defaults(_gw_dispatch=cmd_logs)

    p_install = inner.add_parser(
        "install-service",
        help="Install + enable + start sharedagents-orchestrator + sharedagents-gateway",
    )
    p_install.add_argument(
        "--no-enable",
        action="store_true",
        help="Install + start but don't enable the unit on boot",
    )
    p_install.add_argument(
        "--no-start",
        action="store_true",
        help="Install + enable but don't start the gateway now",
    )
    p_install.set_defaults(_gw_dispatch=cmd_install_service)


def dispatch(args: argparse.Namespace) -> int:
    """Run the chosen ``sharedagents gateway <op>`` handler."""
    handler = getattr(args, "_gw_dispatch", None)
    if handler is None:
        sys.stderr.write("usage: sharedagents gateway <start|stop|restart|status|serve|logs|install-service>\n")
        return 2
    return int(handler(args))


__all__ = [
    "add_subparser",
    "dispatch",
    "cmd_start",
    "cmd_stop",
    "cmd_restart",
    "cmd_status",
    "cmd_serve",
    "cmd_logs",
    "cmd_install_service",
    "start_gateway_background",
    "systemd_available",
    "render_gateway_unit",
    "install_systemd_service",
    "_unit_path",
    "_user_unit_dir",
    "_linger_enabled",
    "_try_enable_linger",
]
