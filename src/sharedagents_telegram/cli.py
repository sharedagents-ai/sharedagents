"""``sharedagents telegram`` CLI subcommand.

Mirrors the shape of ``sharedagents agent`` (init / start / stop / restart
/ status / logs / setup). Bridges to the same lifecycle primitives —
pidfile-based daemonization, systemd-friendly start_new_session.

Wired into the top-level ``sharedagents`` parser by the same module that
wires ``sharedagents agent`` (``sharedagents_gateway.agent_cli``).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Module-level aliases for the clock primitives so tests can replace
# them via ``monkeypatch.setattr(tgcli, "_sleep", …)``. Passing them as
# default keyword arguments would NOT work: Python binds default kwarg
# values at function-definition time, so post-import ``setattr`` on
# the ``time`` module wouldn't re-route the captured reference. Module-
# level names are looked up at call time, so the patch lands.
_sleep = time.sleep
_monotonic = time.monotonic

from sharedagents_core import oscompat  # noqa: E402

# User-systemd primitives reused from the gateway side. They are
# distro-agnostic (no gateway-specific state); the gateway file just
# happens to be where they were first needed. Importing here avoids
# duplicating ~30 lines of XDG / loginctl / systemctl glue.
from sharedagents_gateway.gateway_cli import (  # noqa: E402
    _linger_enabled,
    _run_user_systemctl,
    _try_enable_linger,
    _user_unit_dir,
    systemd_available,
)
from sharedagents_telegram.config import TelegramConfig, telegram_home  # noqa: E402

_SERVICE_NAME = "sharedagents-telegram.service"


def _unit_path() -> Path:
    """Absolute path of the user-instance Telegram unit file."""
    return _user_unit_dir() / _SERVICE_NAME


def _resolve_telegram_exec() -> str:
    """ExecStart line for the unit.

    Uses ``<python> -m sharedagents_telegram.bot`` — the canonical entry the
    deploy template uses. The interpreter is ``sys.executable`` so the
    unit runs in the SAME venv that the CLI runs in (an editable
    install lands here too, with no console-script needed).
    """
    return f"{sys.executable} -m sharedagents_telegram.bot"


def render_telegram_unit() -> str:
    """Render the ``sharedagents-telegram.service`` *user-instance* unit for
    THIS install.

    Mirrors ``sharedagents_gateway.gateway_cli.render_gateway_unit`` — same
    PATH composition, no ``User=`` / ``Group=`` (user-instance units
    run as the invoking user), targets ``default.target``. The bot
    ``Wants=``/``After=`` the gateway so a single ``systemctl --user
    start sharedagents-telegram.service`` brings the whole stack up in
    dependency order on a cold boot; bouncing the gateway alone never
    drags telegram down (the bot reconnects on its own).
    """
    home = telegram_home()
    venv_bin = str(Path(sys.executable).parent)
    _path_parts = [
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
    _seen: set[str] = set()
    path_env = os.pathsep.join(p for p in _path_parts if p and not (p in _seen or _seen.add(p)))
    return f"""\
[Unit]
Description=sharedagents Telegram Bot (standalone)
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
ExecStart={_resolve_telegram_exec()}
Restart=always
RestartSec=15
StandardOutput=journal
StandardError=journal
SyslogIdentifier=sharedagents-telegram

[Install]
WantedBy=default.target
"""


def _is_bot_connected_row(line: bytes) -> bool:
    """True iff ``line`` is the bot's own ``bot connected:`` boot row.

    The log row is emitted by ``sharedagents_telegram.bot._log.info('bot
    connected: @%s id=%s', …)`` (see ``bot.py``) and lands in the
    JSONL stream as ``{"service":"telegram","agent":"telegram",
    "msg":"bot connected: @<name> id=<n>", …}``. We MUST anchor on
    those JSON fields rather than substring-grep the raw line — TX/RX
    rows embed arbitrary user-controlled chat text in their ``msg``
    field (``"msg":"RX chat=… text='…'"``), so a user message
    containing the literal ``bot connected:`` would otherwise false-
    positive the install verifier. The JSON-parsed match is the
    "trustworthy signal" the install path's docstring promises.

    ``service`` is matched with ``startswith("telegram")`` rather
    than exact-equal so a future multi-bot deploy (``service`` =
    ``telegram-bot-1``) doesn't silently break install verification.
    """
    try:
        data = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return False
    if not isinstance(data, dict):
        return False
    msg = data.get("msg")
    if not isinstance(msg, str) or not msg.startswith("bot connected:"):
        return False
    # Belt-and-suspenders: also require the service tag start with
    # "telegram". A rogue agent writing into the shared JSONL stream
    # under a different service name is not us.
    service = data.get("service")
    return isinstance(service, str) and service.startswith("telegram")


def _wait_for_bot_connected(
    logfile: Path,
    baseline_size: int,
    timeout: float = 30.0,
) -> bool:
    """Poll ``logfile`` (JSONL) for the bot's ``bot connected:`` boot
    row appended AFTER ``baseline_size``.

    The Telegram bot has no listening port to connect to, so install
    success is verified via the bot's own boot signal. Per the
    ``feedback_sharedagents_smart_quote_edit_and_restart_verify`` lesson,
    ``systemctl is-active`` returns ``active`` for ~76ms before an
    import-time crash — trusting it would yield a false success. The
    bot's main loop only emits ``bot connected: @<username>`` after
    the Telegram API handshake succeeds, so that's the trustworthy
    signal.

    ``baseline_size`` MUST be captured BEFORE the legacy daemon is
    reaped (let alone the unit started) so:

    1. a stale ``bot connected:`` line from a previous boot doesn't
       fake success;
    2. any line a SIGTERM'd legacy daemon flushes during shutdown
       (buffered handler, reconnect race) does not fake success.

    Clock primitives go through module-level ``_sleep`` / ``_monotonic``
    aliases so tests can replace them via
    ``monkeypatch.setattr(tgcli, "_sleep", …)``. Default-kwarg
    injection was tried and removed: Python binds defaults at
    def-time, so a post-import patch wouldn't reach the function.
    """
    deadline = _monotonic() + timeout
    while _monotonic() < deadline:
        try:
            current = logfile.stat().st_size
        except FileNotFoundError:
            current = 0
        if current > baseline_size:
            try:
                with open(logfile, "rb") as fh:
                    fh.seek(baseline_size)
                    tail = fh.read()
            except OSError:
                tail = b""
            for raw in tail.splitlines():
                if _is_bot_connected_row(raw):
                    return True
        _sleep(0.5)
    return False


def _proc_is_telegram_bot(pid: int) -> bool:
    """True iff ``/proc/<pid>/cmdline`` is a python process running
    the telegram bot module.

    Defence against a recycled pid: if the legacy pidfile is stale and
    the kernel has handed the same pid to an unrelated process — the
    operator's interactive shell, a `vim` editing a path containing
    "sharedagents_telegram", a `grep` over the source tree — ``kill(pid,
    0)`` still says "alive" but we MUST NOT SIGTERM it.

    Substring-matching the full cmdline (as round-1 did) is too
    permissive: a `cd /opt/sharedagents_telegram/...` would license a
    SIGTERM on a shell. We split the NUL-delimited cmdline into argv
    tokens and require BOTH (a) ``argv[0]`` to be a python interpreter
    (basename contains ``python``) AND (b) some argv token to name
    the bot module explicitly — either as the `-m` arg or as a
    ``.../sharedagents_telegram/bot.py`` script path. Returns False on any
    OSError (proc gone / permission denied) so the reaper falls
    through cleanly.
    """
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    argv = [tok for tok in raw.split(b"\0") if tok]
    if not argv:
        return False
    argv0_name = argv[0].rsplit(b"/", 1)[-1]
    if b"python" not in argv0_name:
        return False
    return any(
        tok == b"sharedagents_telegram.bot" or tok.endswith(b"/sharedagents_telegram/bot.py") for tok in argv[1:]
    )


def _stop_legacy_cli_daemon(
    tg_cfg: TelegramConfig,
    *,
    grace_seconds: float = 8.0,
) -> int | None:
    """Reap a CLI-daemonized bot (legacy ``sharedagents telegram start``)
    before systemd takes over.

    Telegram allows ONE long-poll per bot token: two parallel
    ``getUpdates`` loops yield random 409 Conflicts and the bot
    randomly misses updates. So when ``install-service`` is run while a
    legacy daemon is alive, we SIGTERM it (then SIGKILL on grace-period
    expiry) and clean up the pidfile so the systemd-managed unit boots
    on a clean slate.

    Safety contract:

    1. The pidfile's pid is cross-checked against ``/proc/<pid>/cmdline``
       to confirm it is actually a telegram bot — never SIGTERM a
       recycled pid that happens to be the operator's shell.
    2. The pidfile is unlinked ONLY when it still holds the killed pid
       (so a fresh ``cmd_start`` that wrote its own pidfile mid-reap is
       not stomped on). Holding ``_spawn_lock`` around the install path
       is the broader fix; this is the inner belt-and-suspenders.

    Clock primitives go through module-level ``_sleep`` /
    ``_monotonic`` so tests can replace them via
    ``monkeypatch.setattr(tgcli, "_sleep", …)`` and exercise the
    SIGKILL branch deterministically (default kwargs wouldn't work —
    Python binds them at def-time). ``grace_seconds`` is kept as a
    proper kwarg because it's not a function reference and binds fine.

    Returns the killed pid, or None when nothing was reaped.
    """
    running, pid = _running(tg_cfg)
    if not running or not pid:
        return None
    if not _proc_is_telegram_bot(pid):
        # Pidfile points at a process that isn't the bot. Don't kill
        # it. Drop the stale pidfile so the next start can proceed.
        try:
            if tg_cfg.pidfile.read_text(encoding="utf-8").strip() == str(pid):
                tg_cfg.pidfile.unlink()
        except (FileNotFoundError, OSError):
            pass
        return None
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    # Bounded wait for clean shutdown. Mirrors ``cmd_stop``'s grace.
    deadline = _monotonic() + grace_seconds
    alive_at_deadline = True
    while _monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            alive_at_deadline = False
            break
        _sleep(0.2)
    if alive_at_deadline:
        # Final probe — only SIGKILL if it really survived the grace.
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    # Ownership-checked unlink: don't stomp a fresh pidfile that a
    # racing ``cmd_start`` may have just written. The install caller
    # holds ``_spawn_lock`` for the outer guarantee.
    try:
        current = tg_cfg.pidfile.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError):
        current = ""
    if current == str(pid):
        try:
            tg_cfg.pidfile.unlink()
        except FileNotFoundError:
            pass
    return pid


def install_systemd_service(*, enable: bool = True, start: bool = True, interactive: bool = True) -> dict:
    """Install + enable + start ``sharedagents-telegram.service`` as a
    *user-instance* systemd service — no sudo, no system unit.

    Mirrors :func:`sharedagents_gateway.gateway_cli.install_systemd_service`
    (rendered template → ``~/.config/systemd/user`` → daemon-reload →
    enable → restart → best-effort linger), with one telegram-specific
    twist: a legacy CLI-daemonized bot (started by ``sharedagents telegram
    start``) is reaped first so the new systemd-managed unit doesn't
    fight it for ``getUpdates`` — Telegram permits ONE long-poll per
    bot token.

    Returns ``{"ok", "error", "unit_path", "enabled", "started",
    "linger", "linger_hint", "stopped_legacy_pid"}``.

    ``interactive`` is accepted for parity with the gateway signature
    but is unused: nothing on this code path prompts for a password.
    """
    del interactive  # see docstring
    tg_cfg = TelegramConfig.load()
    unit_path = _unit_path()
    result: dict = {
        "ok": False,
        "error": None,
        "unit_path": str(unit_path),
        "enabled": False,
        "started": False,
        "linger": False,
        "linger_hint": None,
        "stopped_legacy_pid": None,
    }
    if not tg_cfg.bot_token:
        result["error"] = f"no bot_token in {tg_cfg.config_path}. Run `sharedagents telegram setup` first."
        return result
    if not systemd_available():
        result["error"] = "systemd not available on this host"
        return result
    # WorkingDirectory must exist before systemd will start the unit.
    try:
        tg_cfg.home.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        result["error"] = f"could not create {tg_cfg.home}: {e}"
        return result
    # Single spawn-lock for the whole install. Without it, a racing
    # ``sharedagents telegram start`` from another shell could (a) win the
    # window between SIGTERM-grace and pidfile-unlink and have its
    # own pidfile clobbered by the reaper, or (b) spawn a second bot
    # process the systemd unit then fights for `getUpdates`. Mirrors
    # the same flock the daemon-start path uses, so the two cannot
    # interleave by construction.
    #
    # LOCK-HOLD DURATION: the lock spans the full install — legacy
    # reap (up to 8s SIGTERM grace) + systemctl daemon-reload/enable/
    # reset-failed/restart + the bot-connected log poll (up to 30s).
    # Worst case ~45s. A concurrent ``sharedagents telegram start`` will
    # block for that whole window before checking ``_running()`` and
    # short-circuiting to "already running". Reducing scope (e.g.
    # releasing before the log poll) was considered and rejected: a
    # concurrent ``cmd_start`` between unlock and the verifier could
    # spawn a SECOND bot while the bot under systemd hasn't yet
    # written its pidfile, restarting the dual-getUpdates conflict
    # this PR exists to fix. The serialisation here IS the fix.
    try:
        with _spawn_lock(tg_cfg):
            # Capture log baseline BEFORE the legacy reap. A SIGTERM'd
            # daemon can flush buffered handler output during shutdown
            # (reconnect retries, late "connected" lines from a
            # re-handshake); we don't want any of that to fake the
            # post-install verifier.
            try:
                baseline_size = tg_cfg.logfile.stat().st_size
            except FileNotFoundError:
                baseline_size = 0
            result["stopped_legacy_pid"] = _stop_legacy_cli_daemon(tg_cfg)

            unit = render_telegram_unit()
            unit_path.parent.mkdir(parents=True, exist_ok=True)
            unit_path.write_text(unit)
            try:
                unit_path.chmod(0o644)
            except OSError:
                # chmod failure is non-fatal (the write above already
                # created it with the umask default); don't abort.
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
                # `reset-failed` first so a unit stuck in
                # `start-limit-hit` (the crash-loop end-state) can
                # still be brought back; without it `restart` refuses
                # inside the start-limit window.
                _run_user_systemctl(["reset-failed", _SERVICE_NAME])
                cp = _run_user_systemctl(["restart", _SERVICE_NAME])
                if cp.returncode != 0:
                    result["error"] = f"systemctl --user restart failed: {(cp.stderr or '').strip()}"
                    _rollback_unit(enable_was_set=result["enabled"])
                    return result
                if not _wait_for_bot_connected(
                    tg_cfg.logfile,
                    baseline_size,
                    timeout=30.0,
                ):
                    # Roll back: stop + disable the unit so it doesn't
                    # crash-loop toward `start-limit-hit` while the
                    # operator is investigating. Half-installed +
                    # ENABLED is the worst end-state — next boot would
                    # crash-loop forever. Stderr is routed to journal
                    # by `StandardError=journal`, not a sibling file,
                    # so the failure hint points at journalctl only.
                    _rollback_unit(enable_was_set=result["enabled"])
                    result["error"] = (
                        f"{_SERVICE_NAME} started but no `bot "
                        f"connected:` row appeared in {tg_cfg.logfile} "
                        f"within 30s — check "
                        f"`journalctl --user -u {_SERVICE_NAME}`. "
                        f"Unit has been stopped and disabled to "
                        f"prevent a crash-loop."
                    )
                    return result
                result["started"] = True
            # Best-effort: ensure the user-instance survives reboot.
            # Done AFTER enable+start so the install isn't blocked
            # when linger needs admin auth on this distro.
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


def _rollback_unit(*, enable_was_set: bool) -> None:
    """Stop + (optionally) disable the unit on a failed install.

    Best-effort: failures are swallowed because the caller is already
    reporting the install error and shouldn't be derailed by a
    follow-on systemctl glitch. We at minimum SIGTERM the bot via the
    unit so it stops crash-looping; if ``enable`` succeeded earlier in
    this install attempt we also undo it so a reboot doesn't bring the
    broken bot back.

    A defensive ``daemon-reload`` is issued before ``disable`` so the
    rollback works even if the install path is ever refactored to
    skip the upstream reload — without it, ``disable`` on a unit
    systemd doesn't yet know about would no-op and leave the broken
    enable symlink behind.
    """
    try:
        _run_user_systemctl(["stop", _SERVICE_NAME])
    except subprocess.SubprocessError:
        pass
    if enable_was_set:
        try:
            _run_user_systemctl(["daemon-reload"])
        except subprocess.SubprocessError:
            pass
        try:
            _run_user_systemctl(["disable", _SERVICE_NAME])
        except subprocess.SubprocessError:
            pass


def cmd_install_service(args: argparse.Namespace) -> int:
    """Install + enable + start the ``sharedagents-telegram`` user-instance
    systemd service.

    Mirrors ``sharedagents gateway install-service``: renders a unit tailored
    to this install (current venv, current ``SHAREDAGENTS_TELEGRAM_HOME``),
    drops it under ``~/.config/systemd/user/``, ``daemon-reload``s,
    ``enable``s + ``restart``s it, then best-effort ``loginctl
    enable-linger`` so it survives reboot.

    When a legacy CLI-daemonized bot is already running (started via
    ``sharedagents telegram start``), it is SIGTERM'd first so the new
    systemd-managed unit doesn't fight it for ``getUpdates``.
    """
    if not systemd_available():
        sys.stderr.write(
            "systemd is not available on this host — nothing to install.\n"
            "run the bot directly instead: `sharedagents telegram start`.\n",
        )
        return 1
    enable = not getattr(args, "no_enable", False)
    start = not getattr(args, "no_start", False)
    if not enable and not start:
        # Operator typo / muscle memory — combining both flags leaves
        # the unit written but inert (not enabled, not running). Print
        # a clear warning so the success line isn't mistaken for "the
        # bot is up now".
        sys.stderr.write(
            f"warning: --no-enable + --no-start — the unit will be "
            f"written but neither enabled nor started. Run "
            f"`systemctl --user enable --now {_SERVICE_NAME}` to bring "
            f"it up.\n",
        )
    sys.stderr.write(f"installing {_SERVICE_NAME} → {_unit_path()}\n")
    result = install_systemd_service(enable=enable, start=start)
    if not result["ok"]:
        sys.stderr.write(f"service install failed: {result['error']}\n")
        return 1
    bits = [f"installed {_SERVICE_NAME} (user-instance)"]
    if result["stopped_legacy_pid"]:
        bits.append(
            f"stopped legacy CLI-daemon (pid {result['stopped_legacy_pid']})",
        )
    if result["enabled"]:
        if result["linger"]:
            bits.append("enabled (starts on boot, linger on)")
        else:
            bits.append("enabled (starts on next login)")
    if result["started"]:
        bits.append("started (bot connected)")
    sys.stdout.write("✓ " + ", ".join(bits) + "\n")
    if not result["linger"]:
        sys.stdout.write(
            "  note: linger is off — the unit will start at your next "
            "login, but not on a cold boot. Boot-start needs `loginctl "
            "enable-linger` to be allowed for your user (one-time host "
            "setup; on many distros it is allowed by default via polkit "
            "and re-running install-service will pick it up).\n",
        )
        if result["linger_hint"]:
            sys.stdout.write(f"  (loginctl said: {result['linger_hint']})\n")
    return 0


def _running(tg_cfg: TelegramConfig) -> tuple[bool, int | None]:
    """Return (running, pid) by reading the pidfile + ``kill -0``."""
    pidfile = tg_cfg.pidfile
    if not pidfile.exists():
        return False, None
    try:
        pid_text = pidfile.read_text(encoding="utf-8").strip()
        pid = int(pid_text)
    except (OSError, ValueError):
        return False, None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False, pid
    except PermissionError:
        return True, pid
    return True, pid


@contextlib.contextmanager
def _spawn_lock(tg_cfg: TelegramConfig):
    """``flock``-based mutex serialising any operation that could
    spawn (or take ownership of) the bot process.

    Held by both ``cmd_start`` (short — until the pidfile appears or
    a 5s timeout) and ``install_systemd_service`` (longer — see the
    LOCK-HOLD DURATION note on that function). The shared lock means
    a concurrent ``sharedagents telegram start`` and ``install-service``
    cannot interleave; whichever takes the lock first finishes before
    the other proceeds, at which point ``_running()`` makes the
    second one a no-op. Locks ``<home>/telegram.spawn.lock`` (a
    sibling of the pidfile so the lock fd doesn't race with the
    daemon's own pidfile open).
    """
    lock_path = tg_cfg.spawn_lock
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        oscompat.file_lock(fd)
        yield
    finally:
        try:
            oscompat.file_unlock(fd)
        except OSError:
            pass
        os.close(fd)


def cmd_setup(args: argparse.Namespace) -> int:
    """Interactive setup for the standalone Telegram bot.

    Writes ``~/.sharedagents/telegram/config.json``. Reuses any existing
    values as defaults, including the legacy per-agent fields if a
    fresh ``TelegramConfig`` is empty (one-time migration).
    """
    print("sharedagents-telegram setup")
    print(f"  config → {TelegramConfig.load().config_path}")
    print()

    tg_cfg = TelegramConfig.load()

    # One-time pickup of legacy per-agent telegram_bot_token /
    # telegram_allowed_chat_ids if the bot's own config is empty.
    if not tg_cfg.bot_token:
        try:
            from sharedagents_agent.agent import AgentConfig as _Acfg

            legacy = _Acfg.load("jarvis")
        except (FileNotFoundError, Exception):  # noqa: BLE001
            legacy = None
        if legacy is not None:
            legacy_token = str(getattr(legacy, "telegram_bot_token", "") or "")
            legacy_ids = list(getattr(legacy, "telegram_allowed_chat_ids", []) or [])
            if legacy_token and not tg_cfg.bot_token:
                print(f"  (carrying over legacy bot token from {legacy.config_path})")
                tg_cfg.bot_token = legacy_token
            if legacy_ids and not tg_cfg.allowed_chat_ids:
                tg_cfg.allowed_chat_ids = legacy_ids

    def ask(prompt: str, default: str = "") -> str:
        suffix = f" [{default}]" if default else ""
        try:
            ans = input(f"  {prompt}{suffix}: ").strip()
        except EOFError:
            ans = ""
        return ans or default

    masked = tg_cfg.bot_token[:8] + "…" + tg_cfg.bot_token[-4:] if tg_cfg.bot_token else ""
    new_token = ask("bot token (from @BotFather)", masked)
    if new_token and new_token != masked:
        tg_cfg.bot_token = new_token
    if not tg_cfg.bot_token:
        print("ERROR: bot_token is required. Re-run when you have one.")
        return 2

    cur_ids = ",".join(str(x) for x in tg_cfg.allowed_chat_ids)
    raw_ids = ask("allowed chat_ids (comma-separated)", cur_ids)
    if raw_ids:
        try:
            tg_cfg.allowed_chat_ids = [int(x.strip()) for x in raw_ids.split(",") if x.strip()]
        except ValueError:
            print("ERROR: chat_ids must be integers.")
            return 2

    tg_cfg.default_agent = ask(
        "default agent for new chats",
        tg_cfg.default_agent or "jarvis",
    )

    tg_cfg.save()
    print(f"  ✓ saved → {tg_cfg.config_path}")
    return 0


def cmd_start(args: argparse.Namespace) -> int:
    """Daemonize ``python -m sharedagents_telegram.bot``.

    Same pattern as ``sharedagents agent start``: open the logfile, fork
    via ``Popen(start_new_session=True)``, wait for the pidfile to
    appear (the bot writes it from its main loop), then return.
    """
    tg_cfg = TelegramConfig.load()
    if not tg_cfg.bot_token:
        sys.stderr.write(
            f"no bot_token in {tg_cfg.config_path}. Run `sharedagents telegram setup` first.\n",
        )
        return 2

    with _spawn_lock(tg_cfg):
        running, pid = _running(tg_cfg)
        if running:
            print(f"telegram bot already running (pid {pid})")
            return 0

        tg_cfg.logfile.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        # Emit the boot row through the unified JSONL writer so the
        # shared ``telegram.log`` stays valid JSONL on every line.
        # (The legacy "=== sharedagents-
        # telegram starting ===" banner was plaintext and corrupted
        # the file.)
        try:
            from sharedagents_core.jsonlog import get_writer

            _w = get_writer(tg_cfg.logfile, service="telegram")
            _w.write(
                agent="telegram",
                level="INFO",
                msg="sharedagents-telegram starting",
                kind="TELEGRAM_BOOT",
            )
        except (OSError, ValueError):
            pass
        # The subprocess's stdout/stderr capture any pre-logging
        # tracebacks; redirect them to a SIBLING plaintext file so the
        # shared JSONL log stays clean. The bot itself logs via the
        # JsonlLogHandler, so this file only holds python-startup
        # noise / unhandled tracebacks.
        stderr_path = tg_cfg.logfile.with_suffix(".stderr.log")
        with open(stderr_path, "a", buffering=1) as log_fh:
            proc = subprocess.Popen(  # noqa: S603
                [sys.executable, "-m", "sharedagents_telegram.bot"],
                stdin=subprocess.DEVNULL,
                stdout=log_fh,
                stderr=log_fh,
                start_new_session=True,
                env=env,
            )

        # Bot writes its own pidfile from ``run()`` at first
        # iteration; wait for it (or up to 5s) so the next
        # ``status`` / ``stop`` call sees the file.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if tg_cfg.pidfile.is_file():
                break
            time.sleep(0.1)
        if not tg_cfg.pidfile.is_file():
            sys.stderr.write(
                f"telegram bot did not write its pidfile within 5s. Check {tg_cfg.logfile} for errors.\n",
            )
            return 1
        print(
            f"telegram bot started (pid {proc.pid}). Logs: {tg_cfg.logfile}",
        )
        return 0


def _systemd_manages_telegram() -> bool:
    """True iff this host has the sharedagents-telegram user unit installed
    AND it is the manager of the running bot (loaded + active).

    Routes ``cmd_stop`` / ``cmd_restart`` away from a futile SIGTERM
    chase: under systemd's ``Restart=always``, killing the pid only
    earns a respawn 15s later, leaving the operator with a misleading
    "telegram bot stopped" while the bot is in fact alive. Same shape
    as ``sharedagents_core/restart._systemd_unit_present`` but also gates
    on ``is-active`` so a unit that's installed-but-disabled (e.g.
    ``--no-start`` install, or after rollback) still allows the CLI
    path to operate normally.

    Fail-safe: when the unit FILE is present but ``is-active`` times
    out or errors, return True (conservative). The opposite — return
    False — would route ``cmd_stop`` back to SIGTERM-the-pid, which
    ``Restart=always`` would undo. That's the exact UX bug the guard
    was added to prevent, and it's most likely under load — which is
    the same scenario this whole PR exists to address (post-OOM
    cascade). The user-manager probe timeout is raised to 5s so a
    cold-cache ``is-active`` call has room to complete normally.
    """
    if not systemd_available():
        return False
    if not _unit_path().exists():
        return False
    try:
        cp = subprocess.run(  # noqa: S603, S607
            ["systemctl", "--user", "is-active", "--quiet", _SERVICE_NAME],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5.0,
            check=False,
        )
        return cp.returncode == 0
    except (OSError, subprocess.SubprocessError):
        # See docstring — unit file present + probe failed → assume
        # systemd is in charge. Better an over-cautious refusal than a
        # SIGTERM that the supervisor immediately undoes.
        return True


def cmd_stop(args: argparse.Namespace) -> int:  # noqa: ARG001
    tg_cfg = TelegramConfig.load()
    if _systemd_manages_telegram():
        # SIGTERM'ing the pid here just hands the bot to systemd's
        # ``Restart=always`` 15s later — operator sees "stopped" while
        # the bot keeps running. Refuse and point at the right tool.
        sys.stderr.write(
            f"telegram bot is managed by systemd ({_SERVICE_NAME}); "
            f"`sharedagents telegram stop` would be undone by Restart=always.\n"
            f"Use: systemctl --user stop {_SERVICE_NAME}\n",
        )
        return 1
    running, pid = _running(tg_cfg)
    if not running:
        if tg_cfg.pidfile.exists():
            try:
                tg_cfg.pidfile.unlink()
            except OSError:
                pass
        print("telegram bot not running")
        return 0
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    # Bounded wait for clean shutdown; SIGKILL after the grace period.
    # Explicit alive_at_deadline guard — same shape as
    # _stop_legacy_cli_daemon so the two paths read identically.
    deadline = time.time() + 8.0
    alive_at_deadline = True
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            alive_at_deadline = False
            break
        time.sleep(0.2)
    if alive_at_deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    try:
        tg_cfg.pidfile.unlink()
    except FileNotFoundError:
        pass
    print(f"telegram bot stopped (pid {pid})")
    return 0


def cmd_restart(args: argparse.Namespace) -> int:
    if _systemd_manages_telegram():
        # Two CLI ``restart`` calls in a row against a systemd-managed
        # bot would each get respawned; the operator gets the wrong
        # mental model. Use ``systemctl --user restart`` for the
        # graceful, dependency-correct bounce.
        sys.stderr.write(
            f"telegram bot is managed by systemd ({_SERVICE_NAME}); "
            f"`sharedagents telegram restart` would fight Restart=always.\n"
            f"Use: systemctl --user restart {_SERVICE_NAME}\n",
        )
        return 1
    rc = cmd_stop(args)
    if rc != 0:
        return rc
    return cmd_start(args)


def cmd_status(args: argparse.Namespace) -> int:
    tg_cfg = TelegramConfig.load()
    running, pid = _running(tg_cfg)
    print(f"telegram bot: {'running' if running else 'stopped'}")
    print(f"  home:        {tg_cfg.home}")
    print(f"  config:      {tg_cfg.config_path}")
    print(f"  default agent: {tg_cfg.default_agent}")
    print(f"  routes:      {len(tg_cfg.agent_routes)} configured")
    print(f"  allowlist:   {len(tg_cfg.allowed_chat_ids)} chat(s)")
    if running:
        print(f"  pid:         {pid}")
        try:
            uptime_s = max(0.0, time.time() - tg_cfg.pidfile.stat().st_mtime)
            print(f"  uptime:      {uptime_s:.0f}s")
        except OSError:
            pass
    return 0 if running else 1


def cmd_logs(args: argparse.Namespace) -> int:
    tg_cfg = TelegramConfig.load()
    if not tg_cfg.logfile.exists():
        print(f"(no log yet at {tg_cfg.logfile})")
        return 0
    return subprocess.call(["tail", "-n", "200", "-f", str(tg_cfg.logfile)])


def add_subparser(sub: argparse._SubParsersAction) -> None:
    """Wire ``sharedagents telegram <cmd>`` into the top-level parser.

    Called from ``sharedagents_gateway.agent_cli`` next to the existing
    ``agent`` and ``memory`` subcommands.
    """
    p = sub.add_parser(
        "telegram",
        help=("Telegram bot lifecycle: setup/start/stop/restart/status/logs"),
    )
    p.set_defaults(_dispatch=lambda a: p.print_help() or 0)
    inner = p.add_subparsers(dest="telegram_cmd")

    p_setup = inner.add_parser("setup", help="Interactive setup wizard")
    p_setup.set_defaults(_dispatch=cmd_setup)

    p_start = inner.add_parser("start", help="Start the bot daemon")
    p_start.set_defaults(_dispatch=cmd_start)

    p_stop = inner.add_parser("stop", help="Stop the bot daemon (SIGTERM)")
    p_stop.set_defaults(_dispatch=cmd_stop)

    p_rest = inner.add_parser("restart", help="Stop + start the bot")
    p_rest.set_defaults(_dispatch=cmd_restart)

    p_stat = inner.add_parser("status", help="pid + uptime + routes")
    p_stat.set_defaults(_dispatch=cmd_status)

    p_logs = inner.add_parser("logs", help="tail -f logs/telegram.log")
    p_logs.set_defaults(_dispatch=cmd_logs)

    p_install = inner.add_parser(
        "install-service",
        help=("Install + enable + start the sharedagents-telegram systemd user service (no sudo)"),
    )
    p_install.add_argument(
        "--no-enable",
        action="store_true",
        help="Install + start but don't enable the unit on boot",
    )
    p_install.add_argument(
        "--no-start",
        action="store_true",
        help="Install + enable but don't start the bot now",
    )
    p_install.set_defaults(_dispatch=cmd_install_service)


__all__ = [
    "add_subparser",
    "cmd_setup",
    "cmd_start",
    "cmd_stop",
    "cmd_restart",
    "cmd_status",
    "cmd_logs",
    "cmd_install_service",
    "install_systemd_service",
    "render_telegram_unit",
    "_unit_path",
    "systemd_available",
]
