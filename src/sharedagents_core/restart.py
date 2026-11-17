"""Cross-frontend gateway-restart helper.

``/restart`` from the TUI, the Telegram bot, or the CLI all collapse
to one operation: bounce the sharedagents stack (south orchestrator + north
gateway) without taking the calling client down with it.

Resolution order:

1. **systemd-managed**: ``sharedagents-orchestrator.service`` and
   ``sharedagents-gateway.service`` are loaded in the *user* manager →
   ``systemctl --user restart`` both units. No sudo: sharedagents installs
   only user-instance units (see ``sharedagents gateway install-service``).
   This MUST come first when applicable: the in-process daemonizer
   below races systemd's ``Restart=always`` during its ``RestartSec``
   window and can create orphan serve processes that win the bind race
   against systemd's next attempt.
2. **CLI-managed** (no systemd): live pidfiles → ``sharedagents gateway restart``.
3. **CLI fallback**: neither systemd nor live pidfiles → ``sharedagents gateway start``.

The actual work happens in a *detached* subprocess that sleeps ~1s
before doing the kill — that gap lets the calling wire response
flush before the daemon tears down its connections, which prevents
the caller from seeing an ugly "connection reset". Returns metadata
the calling frontend can use to render a clean confirmation.
"""

from __future__ import annotations

import logging
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path


def _gateway_pidfile() -> Path:
    """North gateway pidfile path.

    Mirrors ``sharedagents_gateway.gateway._gateway_pidfile`` byte for byte —
    duplicated here so this module doesn't pull a runtime import on
    ``sharedagents_gateway`` (which itself imports ``sharedagents_core``). The
    contract is the path; if it ever changes there, change it here.
    """
    root = os.environ.get("SHAREDAGENTS_HOME")
    return (Path(root) if root else Path.home() / ".sharedagents") / "gateway.pid"


def _gateway_running() -> bool:
    """True iff the gateway is reachable.

    Primary: the gateway pidfile points at a live pid. Fallback: a
    successful ``GET /health`` against the configured bind URL —
    covers a missing/raced pidfile while the listener is healthy
    (e.g. early in a systemd restart). Stale pidfile (exists, pid
    dead) only triggers the HTTP fallback; if both fail, the caller
    falls through to the gateway-start path.
    """
    pidfile = _gateway_pidfile()
    if pidfile.is_file():
        try:
            pid = int(pidfile.read_text().strip())
            if pid > 0:
                try:
                    os.kill(pid, 0)
                    return True
                except ProcessLookupError:
                    pass
                except PermissionError:
                    return True
        except (OSError, ValueError):
            pass
    # HTTP fallback. Pull host/port from the global config; on import
    # error or unset config fall back to the gateway's own defaults
    # (``sharedagents_gateway.gateway._DEFAULT_PORT`` — kept in sync by hand
    # here since this module deliberately avoids importing
    # ``sharedagents_gateway``; see ``_gateway_pidfile`` above for the same
    # duplication contract).
    host, port = "127.0.0.1", 8100
    try:
        from sharedagents_agent.agent import load_global_config

        gw = (load_global_config() or {}).get("gateway") or {}
        host = str(gw.get("host") or host)
        port = int(gw.get("port") or port)
    except Exception:  # noqa: BLE001
        pass
    try:
        import httpx as _httpx

        r = _httpx.get(f"http://{host}:{port}/health", timeout=1.0)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def _systemd_managed() -> bool:
    """True iff ``sharedagents-gateway.service`` is loaded and active in the
    *user* systemd instance.

    Calls ``systemctl --user is-active --quiet`` synchronously
    (~10ms). When True, ``/restart`` must route through ``systemctl
    --user restart`` so systemd remains the single owner of the
    gateway port — see module docstring for the orphan-spawn bug this
    guards against.

    Returns False on any error (systemctl missing, no user manager
    reachable, timeout, unit not loaded, unit loaded-but-inactive).
    The fall-through is deliberate: a stopped or absent unit means
    no race risk, so the in-process daemonizer below is safe to use.
    """
    try:
        result = subprocess.run(  # noqa: S603, S607
            ["systemctl", "--user", "is-active", "--quiet", "sharedagents-gateway.service"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
            check=False,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


# Canonical unit names. ``--hard`` bounces both; plain ``/restart``
# only ever touches the gateway.
_GATEWAY_UNIT = "sharedagents-gateway.service"
_TELEGRAM_UNIT = "sharedagents-telegram.service"


def _systemd_unit_present(unit: str) -> bool:
    """True iff a user-instance systemd unit file is installed
    (loaded or not).

    Gate for whether ``--hard`` should also bounce
    ``sharedagents-telegram.service``: only include it when it actually
    exists, so a ``--hard`` on a gateway-only box doesn't fail the
    whole restart on an unknown unit. ``systemctl --user cat`` exits 0
    when the unit exists, non-zero otherwise. Returns False on any
    error (systemctl missing / no user manager / timeout) — the
    conservative choice that keeps ``--hard`` degrading to
    gateway-only rather than erroring.
    """
    try:
        result = subprocess.run(  # noqa: S603, S607
            ["systemctl", "--user", "cat", unit],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
            check=False,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _build_restart_command(
    agent_name: str,
    *,
    hard: bool = False,
) -> tuple[str, str, str | None]:
    """Pure-function side: pick the (cmd, mode, unit) tuple for
    ``restart_agent_detached``, with no side effects so tests can
    pin the resolution order without spawning processes.

    ``agent_name`` is informational — the gateway restart bounces
    every agent at once. Kept in the signature so frontends don't
    have to special-case the call.

    ``hard=True`` extends a systemd-managed restart to also bounce the
    standalone Telegram bot (``sharedagents-telegram.service``) when that
    unit is installed, so ``/restart --hard`` brings the whole stack
    back in one graceful, dependency-ordered sweep. ``hard`` has no
    effect off systemd: the bot has no in-process daemonizer to drive,
    so the non-systemd arms fall back to the gateway-only paths and the
    caller is told the bot was left alone (see ``restart_agent_detached``).
    """
    if _systemd_managed():
        # User-systemd managed: bounce the unit(s), never spawn our
        # own daemon. No sudo: sharedagents installs only user-instance
        # units, so the calling user always has permission to drive
        # them via ``systemctl --user``.
        units = [_GATEWAY_UNIT]
        if hard and _systemd_unit_present(_TELEGRAM_UNIT):
            units.append(_TELEGRAM_UNIT)
        joined = " ".join(units)
        # ``reset-failed`` first so a unit stuck in ``start-limit-hit``
        # (the crash-loop end-state — e.g. an orphan held the port long
        # enough to exhaust StartLimitBurst) can still be brought back:
        # without it ``systemctl restart`` refuses inside the
        # start-limit window. systemd honours ``After=`` across the
        # listed units regardless of argv order — it stops
        # telegram→gateway and starts gateway→telegram, the graceful,
        # dependency-correct order for the bot (which ``Wants=``/
        # ``After=`` the gateway).
        #
        # Two direct ``systemctl --user`` calls rather than one
        # ``sh -c '…'`` wrapper: keeps the dispatch readable in tests
        # and avoids depending on a shell quoting layer. The
        # ``reset-failed`` is best-effort — its stderr is swallowed and
        # the ``;`` runs the restart regardless.
        cmd = f"sleep 1 && systemctl --user reset-failed {joined} 2>/dev/null; exec systemctl --user restart {joined}"
        unit = ",".join(u.removesuffix(".service") for u in units)
        return cmd, "systemd", unit
    # The gateway lifecycle CLI lives under the `sharedagents` console script
    # (`sharedagents gateway {restart,start}`), not `python -m sharedagents_gateway` —
    # `sharedagents_gateway`'s argparse only knows init/list/status, so the old
    # `python -m sharedagents_gateway gateway restart` exited 2 silently in the
    # detached shell, leaving the gateway untouched on every non-systemd box.
    # Derive the console-script path from sys.executable so we do not depend
    # on the detached shell's PATH.
    sharedagents_bin = str(Path(sys.executable).parent / "sharedagents")
    if _gateway_running():
        cmd = f"sleep 1 && exec {shlex.quote(sharedagents_bin)} gateway restart"
        return cmd, "gateway", "sharedagents-gateway"
    # Gateway not running — start it. Same effect from the user's
    # perspective ("agent comes back"), surface the mode honestly so
    # the frontend can render a different confirmation if it wants.
    cmd = f"sleep 1 && exec {shlex.quote(sharedagents_bin)} gateway start"
    return cmd, "gateway-start", "sharedagents-gateway"


# ---- child-process sweep ---------------------------------------------
#
# Observed: ``/restart`` rebinds the TUI's ACP client to a
# (logically) fresh gateway but does NOT enumerate the current gateway
# PID's recursive children and SIGTERM them. The gateway-side
# ``cmd_stop`` SIGTERMs the parent only; the parent's signal handler
# stops per-agent schedulers + drains the HTTP listener, and its
# ``TurnState.stop`` SIGTERMs the *currently-tracked* worker / tool
# runner per active session — but any tool_runner / subagent stdio
# child that is no longer the active ``turn.proc`` (parallel batch
# leftovers, mid-flight subagent runners, anything whose registration
# was cleared) survives. From the operator's POV the TUI shows idle
# but the model keeps running; worse, a new prompt can race the
# orphaned subprocess. ``kill_gateway_children`` walks
# ``/proc/<pid>/task/*/children`` (depth-first, recursive) and reaps
# every descendant — SIGTERM, 3 s grace, SIGKILL fallback. Pure-stdlib
# (psutil is NOT a dependency of sharedagents) and idempotent: zero children
# is a no-op.


_LOG = logging.getLogger("sharedagents.restart")


def _proc_children(pid: int) -> list[int]:
    """Return immediate child PIDs of ``pid`` via ``/proc``.

    Reads every ``/proc/<pid>/task/<tid>/children`` file and unions the
    whitespace-separated child PIDs. Returns ``[]`` on any error so
    callers can treat "process gone / not visible" the same as "no
    children" without try/except scaffolding at every call site.
    """
    task_root = f"/proc/{pid}/task"
    children: set[int] = set()
    try:
        tids = os.listdir(task_root)
    except OSError:
        return []
    for tid in tids:
        if not tid.isdigit():
            continue
        try:
            with open(f"{task_root}/{tid}/children") as fh:
                raw = fh.read()
        except OSError:
            continue
        for tok in raw.split():
            try:
                children.add(int(tok))
            except ValueError:
                continue
    return sorted(children)


def _proc_descendants(pid: int) -> list[int]:
    """DFS walk of ``pid``'s recursive descendants. Excludes ``pid``
    itself. Order is deterministic (children-first ordering by PID)
    so test goldens stay stable.
    """
    out: list[int] = []
    stack = list(_proc_children(pid))
    seen: set[int] = set()
    while stack:
        cur = stack.pop(0)
        if cur in seen or cur == pid:
            continue
        seen.add(cur)
        out.append(cur)
        stack.extend(_proc_children(cur))
    return out


def _proc_cmdline(pid: int) -> str:
    """Cheap label for log lines — ``argv[0]`` (basename) + ``argv[1]``
    when available. Returns ``"<unknown>"`` on any error. Strictly
    cosmetic; the kill decision never depends on this.
    """
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            raw = fh.read()
    except OSError:
        return "<unknown>"
    parts = [p.decode("utf-8", "replace") for p in raw.split(b"\x00") if p]
    if not parts:
        return "<unknown>"
    head = os.path.basename(parts[0]) or parts[0]
    if len(parts) > 1:
        return f"{head} {parts[1]}"
    return head


def _classify(cmdline: str) -> str:
    """Best-effort kind label for the post-kill log line. Pure cosmetic
    — operators reading the gateway log want "tool_runner" / "subagent"
    rather than full argvs. Falls back to ``"unknown"``.
    """
    if "sharedagents_agent.tool_runner" in cmdline:
        return "tool_runner"
    if "sharedagents_gateway" in cmdline and "acp_stdio_main" in cmdline:
        return "subagent"
    if "sharedagents_gateway" in cmdline:
        return "agent_child"
    return "unknown"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def kill_gateway_children(
    gw_pid: int | None,
    *,
    grace_s: float = 3.0,
) -> dict:
    """Reap every recursive descendant of the gateway PID.

    Idempotent: a missing / dead / childless gateway returns
    ``{"ok": True, "killed": [], "note": "..."}`` with no side
    effects. ``SIGTERM`` everyone, poll for ``grace_s`` seconds, then
    ``SIGKILL`` anything still alive. Logs every killed PID + best-
    effort kind classification (``tool_runner`` / ``subagent`` /
    ``unknown``) at INFO so the gateway log carries the evidence trail.
    """
    if not gw_pid or gw_pid <= 0:
        return {"ok": True, "killed": [], "note": "no-pid"}
    try:
        os.kill(gw_pid, 0)
    except (ProcessLookupError, PermissionError):
        return {"ok": True, "killed": [], "note": "gateway-gone-or-foreign"}
    descendants = _proc_descendants(gw_pid)
    if not descendants:
        return {"ok": True, "killed": [], "note": "no-children"}
    killed: list[dict] = []
    for cpid in descendants:
        cmdline = _proc_cmdline(cpid)
        kind = _classify(cmdline)
        try:
            os.kill(cpid, signal.SIGTERM)
            killed.append({"pid": cpid, "kind": kind, "cmd": cmdline, "signal": "TERM"})
            _LOG.info(
                "restart-sweep SIGTERM pid=%d kind=%s cmd=%s",
                cpid,
                kind,
                cmdline,
            )
        except ProcessLookupError:
            continue
        except PermissionError as e:
            _LOG.warning("restart-sweep cannot signal pid=%d: %s", cpid, e)
    deadline = time.monotonic() + max(0.1, grace_s)
    while time.monotonic() < deadline:
        if not any(_pid_alive(rec["pid"]) for rec in killed):
            break
        time.sleep(0.1)
    for rec in killed:
        if _pid_alive(rec["pid"]):
            try:
                os.kill(rec["pid"], signal.SIGKILL)
                rec["signal"] = "KILL"
                _LOG.warning(
                    "restart-sweep SIGKILL pid=%d kind=%s cmd=%s (survived %.1fs grace)",
                    rec["pid"],
                    rec["kind"],
                    rec["cmd"],
                    grace_s,
                )
            except ProcessLookupError:
                continue
    return {"ok": True, "killed": killed, "note": "swept"}


def gateway_pid() -> int | None:
    """Return the current gateway PID from ``$SHAREDAGENTS_HOME/gateway.pid``
    (mirrors the duplication contract documented on
    :func:`_gateway_pidfile`). Returns ``None`` on missing /
    unparseable / dead pidfile.
    """
    pidfile = _gateway_pidfile()
    if not pidfile.is_file():
        return None
    try:
        pid = int(pidfile.read_text().strip())
    except (OSError, ValueError):
        return None
    if pid <= 0:
        return None
    try:
        os.kill(pid, 0)
        return pid
    except ProcessLookupError:
        return None
    except PermissionError:
        # Pid exists but owned by another user — return it so callers
        # can surface "cannot signal" rather than silently no-op.
        return pid


def restart_agent_detached(agent_name: str, *, hard: bool = False) -> dict:
    """Schedule a gateway restart and return immediately.

    ``agent_name`` is informational — bouncing the gateway cascades
    to every agent (every child stdio process) at once. The signature
    accepts it for symmetry with the legacy per-agent helper that
    callers (bot, TUI, CLI) already pass.

    ``hard=True`` (``/restart --hard``) additionally bounces the
    standalone Telegram bot service when systemd manages the stack and
    the unit is installed — one detached command brings gateway + bot
    back gracefully (see :func:`_build_restart_command`). The detached
    child outlives its caller (``start_new_session``), so this works
    even when the *caller is the bot itself*: the bot flushes its reply
    during the command's ~1s ``sleep`` before systemd stops it.

    Returns:
        ``{"ok": True, "mode": "systemd"|"gateway"|"gateway-start",
        "unit": str, "pid": int, "agent": str, "hard": bool,
        "telegram": bool}`` on a successfully *scheduled* restart (the
        actual bounce happens ~1s later in the detached child).
        ``telegram`` is True only when the bot service is actually part
        of this restart, so frontends can render an honest message.
        ``{"ok": False, "error": "..."}`` if the restart could not be
        scheduled (e.g. spawn failure).

    The caller should drop any cached connection / session state
    BEFORE returning to the user — when the gateway respawns it will
    have a different PID and possibly a different listening URL, and
    a stale client would issue requests against the going-away
    instance for a few hundred milliseconds.
    """
    cmd, mode, unit = _build_restart_command(agent_name, hard=hard)
    telegram = bool(hard and mode == "systemd" and "sharedagents-telegram" in (unit or ""))

    try:
        proc = subprocess.Popen(  # noqa: S603
            ["/bin/sh", "-c", cmd],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except OSError as e:
        return {"ok": False, "error": f"spawn failed: {e}"}

    return {
        "ok": True,
        "mode": mode,
        "unit": unit,
        "pid": proc.pid,
        "agent": agent_name,
        "hard": hard,
        "telegram": telegram,
    }


__all__ = [
    "restart_agent_detached",
    "kill_gateway_children",
    "gateway_pid",
    "_build_restart_command",
    "_gateway_running",
    "_systemd_managed",
    "_systemd_unit_present",
]
