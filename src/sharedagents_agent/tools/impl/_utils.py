"""Extracted tool-runner family: _utils (de-monolith of runners.py).

Behavior-preserving split — code moved verbatim.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from dataclasses import dataclass
from typing import Any

from sharedagents_agent.tools.widget_adapt import is_ask_user_tool
from sharedagents_core import oscompat

# Opt-in routing trace: when SHAREDAGENTS_ROUTE_TRACE=1, the delegation shim prints one
# ``[route-dispatch] -> <agent>`` line per delegation (and ``[route-reroute]``
# when a generic target is corrected) to stderr. Off by default — pure
# observability for routing verification, never on in production.
_ROUTE_TRACE = os.environ.get("SHAREDAGENTS_ROUTE_TRACE") == "1"


# ---- subprocess group helpers (killability) -------------------------
#
# Every long-running subprocess we spawn — exec, delegate_claude,
# delegate_codex — runs in its own process group via
# `start_new_session=True`. On cancel/timeout we send the signal to the
# whole group with `os.killpg`, so a tool that itself shells out (the
# `claude` CLI launches `Bash`/`Read`/etc.; an `exec` `bash -c '…'`
# spawns commands behind the pipeline) doesn't leave orphaned
# descendants running after ESC. Calling `proc.kill()` alone only
# delivers the signal to the immediate child, which is the foundational
# killability bug this addresses.


async def _feed_stdin(proc: asyncio.subprocess.Process, data: bytes) -> None:
    """Write `data` to `proc.stdin` and close it.

    Used by the delegate runners to pass the prompt via stdin instead
    of argv — argv would hit Linux's ARG_MAX (~128 KiB) on large
    prompts. Errors are swallowed: a closed pipe means the child
    already exited (we'll surface its rc), and the caller's cancel
    path also closes the proc, both of which would otherwise raise
    here and shadow the real failure.
    """
    if proc.stdin is None:
        return
    try:
        proc.stdin.write(data)
        await proc.stdin.drain()
    except (BrokenPipeError, ConnectionResetError, OSError):
        pass
    finally:
        try:
            proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass


_SIGNAL_MAP: dict[int, str] | None = None


def _signal_name(returncode: int) -> str:
    """Map a process return code > 128 to the signal name that killed it.

    POSIX convention: when a process is killed by signal N, waitpid()
    reports exit status 128+N. This helper returns ``"SIGTERM"`` for 143,
    ``"SIGKILL"`` for 137, etc.

    Returns:
    * ``"exit {rc}"`` — return code ≤ 128 (normal exit / non-signal crash).
    * ``"rc={rc} ({sig_name})"`` — known signal N = rc-128 (e.g. 143 → SIGTERM).
    * ``"rc={rc}"`` — unknown signal number (rc > 128 but signal number
      not in ``signal.Signals``).

    Builds ``_SIGNAL_MAP`` on first call by iterating ``signal.Signals`` so
    we don't hardcode a table and pick up platform-specific signals.
    """
    # Windows exit codes (including NTSTATUS) are not POSIX 128+signal.
    if oscompat.IS_WINDOWS or returncode <= 128:
        return f"exit {returncode}"
    sig_num = returncode - 128
    # Lazy-build the reverse map from signal number → name.
    global _SIGNAL_MAP  # noqa: PLW0603
    if _SIGNAL_MAP is None:
        _SIGNAL_MAP = {}
        for s in signal.Signals:
            if s.value not in _SIGNAL_MAP:
                _SIGNAL_MAP[s.value] = s.name
    sig_name = _SIGNAL_MAP.get(sig_num)
    if sig_name:
        return f"rc={returncode} ({sig_name})"
    return f"rc={returncode}"


def _kill_group(proc: asyncio.subprocess.Process, sig: int = signal.SIGTERM) -> None:
    """Best-effort send `sig` to the entire process group of `proc`.

    The shared kill chokepoint for every spawned process group (the Bash exec
    AND the spawned ``*_cli`` delegate agents). When the group carries a
    SandboxBroker lease the kill is first routed through the capability layer
    (downward-only, gateway-guarded, §10.3) by pgid; an un-leased group — or a
    failed capability kill — falls through to the raw ``killpg`` so isolation
    can never cost the existing kill behaviour.

    Falls back to per-process kill if the group lookup fails (the process
    already exited, or the platform doesn't have process groups). All
    exceptions are swallowed — this is called from cancel paths where
    re-raising would mask the original cancellation.
    """
    if proc.pid is None:
        return
    try:
        pgid = oscompat.process_group(proc.pid)
    except (ProcessLookupError, PermissionError, OSError):
        # Process gone or we lost permission. Try the immediate kill.
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            pass
        return
    try:
        from sharedagents_agent import sandbox_broker

        status = sandbox_broker.enforce_kill_pgid(pgid, sig)
    except Exception:  # noqa: BLE001 — never block the kill path on the broker
        status = "no_lease"
    if status != "no_lease":
        # A LEASED group: the capability layer is the enforcement boundary —
        # "killed"/"gone" are handled, and "denied" must NOT fall through to a
        # raw killpg (that would bypass the boundary, §10.3). The raw fallback
        # below is reserved for genuinely un-leased (legacy) groups only.
        return
    try:
        oscompat.signal_process_group(pgid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            pass


async def _terminate_group(
    proc: asyncio.subprocess.Process,
    *,
    grace_s: float = 1.0,
) -> None:
    """SIGTERM the group, wait up to `grace_s` seconds, then SIGKILL.

    Used in cancel/timeout paths so well-behaved children get a chance
    to clean up but uncooperative ones don't pin the runner forever.
    """
    # See _kill_tree: honour a monkeypatch of ``runners._kill_group`` post-split.
    import sharedagents_agent.tools.runners as _runners

    _kill_group = getattr(_runners, "_kill_group", globals()["_kill_group"])
    _kill_group(proc, signal.SIGTERM)
    try:
        await asyncio.wait_for(proc.wait(), timeout=grace_s)
        return
    except asyncio.TimeoutError:
        pass
    _kill_group(proc, oscompat.FORCE_KILL)
    try:
        await asyncio.wait_for(proc.wait(), timeout=grace_s)
    except asyncio.TimeoutError:
        pass


def _descendant_pids(root_pid: int) -> list[int]:
    """All transitive child PIDs of ``root_pid`` (Linux ``/proc`` walk).

    Used by :func:`_kill_tree` to reach a runner's whole subtree on a
    hard cancel — including grandchildren that started their OWN session
    (which a single ``killpg`` would miss) — while never returning the
    runner's ANCESTORS (so the gateway is never in the kill set).
    Best-effort: returns whatever it can read; never raises.
    """
    if oscompat.IS_WINDOWS:
        return []  # Best effort: no descendant enumeration without Job Objects.
    children: dict[int, list[int]] = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return []
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as fh:
                data = fh.read()
            # Field 2 (comm) is paren-wrapped and may contain spaces or
            # ')'; ppid is the 2nd whitespace field AFTER the last ')'.
            rparen = data.rfind(b")")
            after = data[rparen + 2 :].split()
            ppid = int(after[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(int(entry))
    out: list[int] = []
    stack = [root_pid]
    while stack:
        cur = stack.pop()
        for ch in children.get(cur, []):
            out.append(ch)
            stack.append(ch)
    return out


def _kill_tree(proc: asyncio.subprocess.Process, sig: int = oscompat.FORCE_KILL) -> None:
    """INSTANT, best-effort kill of ``proc`` AND its entire descendant tree.

    The hard-cancel chokepoint for ``/abort`` / a second ESC. Unlike
    :func:`_kill_group` — which signals only ``proc``'s OWN process group
    via ``killpg`` — this also walks ``/proc`` for every transitive
    descendant and signals each directly, so a *subagent* the delegated
    CLI (or the subagent ``tool_runner``) spawned in its OWN session
    (``start_new_session=True`` → a DIFFERENT pgid that a lone ``killpg``
    misses) dies too instead of orphaning to init. Synchronous and
    signal-only: there is no grace window and no ``await`` — the whole
    tree gets ``SIGKILL`` on the same tick the operator pressed abort.

    Order matters: descendants are signalled FIRST so a killed parent
    can't reparent them past the sweep, THEN the proc's own group, THEN
    the proc itself.

    Safety — the ``killpg`` is gated on ``proc`` being its OWN group
    leader (``getpgid(pid) == pid``, i.e. it was spawned
    ``start_new_session=True``). A proc that SHARES the gateway's group
    (the subagent ``tool_runner`` is spawned WITHOUT a new session) has
    ``getpgid(pid) != pid``; ``killpg`` is skipped for it so the sweep can
    never signal the gateway's group. Its own-session children are still
    reached via the descendant walk. Never raises.
    """
    pid = getattr(proc, "pid", None)
    if pid is None:
        return
    # Resolve the killability helpers through the ``runners`` re-export so tests
    # that ``monkeypatch.setattr(runners, "_descendant_pids"/"_kill_group", …)``
    # still steer this sweep after the de-monolith split (local shadow keeps the
    # reference sites below unchanged; falls back to this module's own copy).
    import sharedagents_agent.tools.runners as _runners

    _descendant_pids = getattr(_runners, "_descendant_pids", globals()["_descendant_pids"])
    _kill_group = getattr(_runners, "_kill_group", globals()["_kill_group"])
    if oscompat.IS_WINDOWS:
        _kill_group(proc, sig)
        return  # Best effort: Windows grandchildren can survive cancellation.
    # Descendants first (grandchildren in their own sessions included), so
    # a killed parent can't reparent them past the sweep. Raw os.kill is
    # downward-only by construction — these ARE our descendants — so it
    # honours the capability model (§10.3) without a per-pid broker round.
    for dpid in _descendant_pids(pid):
        try:
            os.kill(dpid, sig)
        except OSError:
            pass
    # The proc's own group — ONLY when proc LEADS its group (spawned
    # start_new_session=True → getpgid(pid)==pid). A proc that SHARES the
    # gateway's group (the subagent tool_runner is spawned WITHOUT a new
    # session) has getpgid(pid)!=pid; skipping killpg there means the
    # sweep can NEVER signal the gateway's group. The group kill routes
    # through the broker-aware _kill_group so a leased group still goes via
    # the capability layer; _kill_group also signals proc itself.
    try:
        leads_group = os.getpgid(pid) == pid
    except OSError:
        leads_group = False
    if leads_group:
        _kill_group(proc, sig)
    else:
        try:
            os.kill(pid, sig)
        except OSError:
            pass


# --- Delegate-CLI read timeouts -------------------------------------
# A spawned delegate CLI (`claude -p` / `codex exec` / antigravity) can
# stall before emitting ANY output — a hung SessionStart hook, an auth or
# network wedge, a spawn that never reaches the model. codex and antigravity
# also had no read timeout at all and would block on `readline()` forever.
# These bound the wait so a stall fails fast with a clear error instead of a
# silent freeze that needs an ESC / gateway restart to clear. Critically the
# idle default sits BELOW Telegram's 300s no-progress watchdog so the engine
# reports the stall FIRST (a clean turn error) instead of losing the race to
# the coarse watchdog; callers can raise it for unusually long silent tool
# calls. (The claude path has its own knob — see ``_claude_idle_timeout_s``.)
def _delegate_first_token_timeout_s() -> float:
    """Seconds to wait for a delegate CLI's FIRST stream event before
    treating its startup as stalled. CLI engines (claude / codex /
    antigravity) emit a JSON line long before the first model token, so
    60s is a hung-spawn detector, not a TTFT budget. ACP engines use
    :func:`_acp_first_token_timeout_s` instead. Env override:
    ``SHAREDAGENTS_DELEGATE_FIRST_TOKEN_TIMEOUT_S`` (default 60)."""
    try:
        return float(os.environ.get("SHAREDAGENTS_DELEGATE_FIRST_TOKEN_TIMEOUT_S", "60") or "60")
    except ValueError:
        return 60.0


# OpenCode/big-pickle silent prefill is ~58s on a ~10k-token injected
# prompt. The shared CLI first-event default of 60s races that. ACP
# servers often emit nothing translatable until the first token
# (usage_update is protocol-only), so ACP needs its own budget.
_ACP_FIRST_TOKEN_TIMEOUT_DEFAULT_S = 180.0


def _acp_first_token_timeout_s() -> float:
    """Seconds to wait for the first translatable ACP output (or a
    usage_update that proves the peer accepted the prompt). Env override:
    ``SHAREDAGENTS_ACP_FIRST_TOKEN_TIMEOUT_S`` (default 180)."""
    try:
        return float(
            os.environ.get(
                "SHAREDAGENTS_ACP_FIRST_TOKEN_TIMEOUT_S",
                str(_ACP_FIRST_TOKEN_TIMEOUT_DEFAULT_S),
            )
            or _ACP_FIRST_TOKEN_TIMEOUT_DEFAULT_S
        )
    except ValueError:
        return _ACP_FIRST_TOKEN_TIMEOUT_DEFAULT_S


def _delegate_idle_timeout_s() -> float:
    """Seconds of silence AFTER output has started before treating a
    delegate CLI as hung (one long tool call is legitimate). Defaults BELOW
    Telegram's 540s no-progress watchdog (Antigravity print 480 + 60s
    buffer) so a mid-stream stall surfaces as a clean engine error before
    the coarse watchdog hard-cancels the turn. Env override:
    ``SHAREDAGENTS_DELEGATE_IDLE_TIMEOUT_S`` (default 240)."""
    try:
        return float(os.environ.get("SHAREDAGENTS_DELEGATE_IDLE_TIMEOUT_S", "240") or "240")
    except ValueError:
        return 240.0


def _acp_reasoning_stall_s() -> float:
    """Deprecated. ACP no longer time-gates thinking.

    After a reasoning block the next act must be a tool or visible
    output; a finished think-then-stop nudges immediately. Kept so
    existing env docs and imports do not break. Env override:
    ``SHAREDAGENTS_ACP_REASONING_STALL_S`` (default 180)."""
    try:
        return float(os.environ.get("SHAREDAGENTS_ACP_REASONING_STALL_S", "180") or "180")
    except ValueError:
        return 180.0


# Seconds of silence AFTER claude has emitted its first stream event before
# the runner gives up. This MUST sit BELOW Telegram's no-progress watchdog
# (``_DEFAULT_TURN_IDLE_TIMEOUT_S`` = 540s in ``sharedagents_telegram.bot`` —
# Antigravity print 480 + 60s buffer) so a mid-stream claude stall surfaces
# as a clean ``claude_runner_idle_timeout`` turn error (a real
# ``turn_ended``) BEFORE the coarse watchdog hard-cancels the turn. The old
# 900s default sat ABOVE the watchdog, so the watchdog always won the race
# — the turn was aborted with no ``turn_ended`` and the engine idle timeout
# "never fired in time" (recurring ``delegate_stall:claude`` incident,
# fp 9e94ede5). 270s keeps clear margin under 540s while leaving headroom
# for a legitimately long single tool call. Env override:
# ``SHAREDAGENTS_CLAUDE_IDLE_TIMEOUT_S``.
_CLAUDE_IDLE_TIMEOUT_DEFAULT_S = 270.0


def _claude_idle_timeout_s() -> float:
    """Idle (post-first-event) read timeout for ``run_delegate_claude_async``.
    Defaults BELOW the Telegram turn watchdog so a claude stall fails fast and
    cleanly instead of being hard-cancelled by the coarse watchdog. Env
    override: ``SHAREDAGENTS_CLAUDE_IDLE_TIMEOUT_S``."""
    try:
        return float(
            os.environ.get(
                "SHAREDAGENTS_CLAUDE_IDLE_TIMEOUT_S",
                str(_CLAUDE_IDLE_TIMEOUT_DEFAULT_S),
            )
            or _CLAUDE_IDLE_TIMEOUT_DEFAULT_S
        )
    except ValueError:
        return _CLAUDE_IDLE_TIMEOUT_DEFAULT_S


# Antigravity ``--print-timeout`` default is 5m0s. That wall is what
# surfaces as ``timeout waiting for response`` after ~303s — not
# sharedagents's 240s idle kill. Bump slightly so a long tool chain can
# finish one checkpoint; operators raise it via
# ``SHAREDAGENTS_ANTIGRAVITY_PRINT_TIMEOUT_S``.
_ANTIGRAVITY_PRINT_TIMEOUT_DEFAULT_S = 480.0


def _format_go_duration(seconds: float) -> str:
    """Format seconds as a Go duration (``8m0s``) for Antigravity flags."""
    secs = max(1, int(round(float(seconds))))
    minutes, rem = divmod(secs, 60)
    if minutes and rem:
        return f"{minutes}m{rem}s"
    if minutes:
        return f"{minutes}m0s"
    return f"{secs}s"


def _antigravity_print_timeout_s() -> float:
    """Wall-clock budget for one Antigravity ``-p`` invocation.

    Env override: ``SHAREDAGENTS_ANTIGRAVITY_PRINT_TIMEOUT_S`` (default 480).
    """
    try:
        return max(
            1.0,
            float(
                os.environ.get(
                    "SHAREDAGENTS_ANTIGRAVITY_PRINT_TIMEOUT_S",
                    str(_ANTIGRAVITY_PRINT_TIMEOUT_DEFAULT_S),
                )
                or _ANTIGRAVITY_PRINT_TIMEOUT_DEFAULT_S
            ),
        )
    except ValueError:
        return _ANTIGRAVITY_PRINT_TIMEOUT_DEFAULT_S


def _antigravity_idle_timeout_s() -> float:
    """Idle read timeout for Antigravity — at least the print-mode wall.

    Silent thinking must not be killed at the generic 240s idle while
    ``--print-timeout`` is still running.
    """
    return max(_delegate_idle_timeout_s(), _antigravity_print_timeout_s())


# Wall budget for one ZCode headless invocation. The runner now streams
# ``--output-format stream-json`` (legacy ``--json`` fallback prints NOTHING
# until the turn finishes), but a healthy multi-minute turn can still sit
# between events longer than the generic idle timeout allows, so the wall
# budget stays the authority (plus the runner's own no-line stall guard).
# Kept BELOW Telegram's 540s no-progress watchdog (like Antigravity's 480) so
# a stall surfaces as a clean engine error before the coarse watchdog cancels
# the turn. Env override: ``SHAREDAGENTS_ZCODE_TIMEOUT_S``.
_ZCODE_TIMEOUT_DEFAULT_S = 480.0


def _zcode_timeout_s() -> float:
    """Wall-clock budget for one ZCode ``-p --json`` invocation.

    Env override: ``SHAREDAGENTS_ZCODE_TIMEOUT_S`` (default 480).
    """
    try:
        return max(
            1.0,
            float(
                os.environ.get(
                    "SHAREDAGENTS_ZCODE_TIMEOUT_S",
                    str(_ZCODE_TIMEOUT_DEFAULT_S),
                )
                or _ZCODE_TIMEOUT_DEFAULT_S
            ),
        )
    except ValueError:
        return _ZCODE_TIMEOUT_DEFAULT_S


def _antigravity_print_timeout_argv() -> list[str]:
    """``--print-timeout <Go duration>`` for the Antigravity CLI."""
    return [
        "--print-timeout",
        _format_go_duration(_antigravity_print_timeout_s()),
    ]


async def _delegate_readline(
    stream: asyncio.StreamReader,
    *,
    got_first_event: bool,
    first_token_timeout_s: float,
    idle_timeout_s: float,
) -> bytes:
    """One delegate-CLI stdout line, bounded by a startup or idle timeout.

    Before the first event arrives the wait is bounded by
    ``first_token_timeout_s`` (catches a CLI that never starts emitting);
    afterwards by ``idle_timeout_s``. Raises :class:`asyncio.TimeoutError`
    when the stream stalls past the applicable bound so the caller can fail
    the turn fast instead of blocking on ``readline()`` until ESC (claude)
    or forever (codex / antigravity, which had no read timeout).
    """
    timeout = first_token_timeout_s if not got_first_event else idle_timeout_s
    return await asyncio.wait_for(stream.readline(), timeout=timeout)


MAX_OUTPUT_BYTES = 64 * 1024  # 64 KiB cap per tool result

# Hard upper bound on the bytes a single `read_file` call will pull off
# disk before refusing. The runner truncates the *returned* content to
# `MAX_OUTPUT_BYTES`, but without a pre-read cap a multi-GB log file
# would still be slurped into memory and OOM-kill the runner. Source
# code rarely needs more than this; for genuinely-large files use
# `grep` or split the read into ranges.
READ_FILE_MAX_BYTES = 50 * 1024 * 1024  # 50 MiB

# OpenCode ``read`` output-shape caps (mirror ``tool/read.ts``):
#   DEFAULT_READ_LIMIT — lines returned when ``limit`` is omitted.
#   READ_MAX_LINE_LENGTH — a single line longer than this is clipped with a
#     ``... (line truncated to N chars)`` suffix.
#   READ_MAX_OUTPUT_BYTES — 50 KB cap on the returned content, with a
#     resumable ``offset=`` footer so the model can continue.
DEFAULT_READ_LIMIT = 2000
READ_MAX_LINE_LENGTH = 2000
READ_MAX_LINE_SUFFIX = f"... (line truncated to {READ_MAX_LINE_LENGTH} chars)"
READ_MAX_OUTPUT_BYTES = 50 * 1024  # 50 KB
READ_MAX_BYTES_LABEL = f"{READ_MAX_OUTPUT_BYTES // 1024} KB"

# Hard upper bound on the bytes `web_fetch` will pull from the
# upstream HTTP response. trafilatura is happy with much less than
# this; the cap exists to prevent a single misdirected fetch from
# materialising a 500 MB response into memory before the runner gets a
# chance to truncate.
WEB_FETCH_MAX_BYTES = 8 * 1024 * 1024  # 8 MiB

# Directories `run_grep` skips by default. The previous implementation
# walked everything (including `.git/`, `node_modules/`, `__pycache__/`,
# `.venv/`, `dist/`, `build/`, …) and decoded every file in full, which
# turned a single grep on a real repo into a 2-minute disk-bound spin.
# Pruning these is the single biggest win.
_GREP_IGNORE_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".bzr",
        "node_modules",
        "bower_components",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        ".env",
        "env",
        "dist",
        "build",
        "target",
        "out",
        ".output",
        ".tox",
        ".cache",
        ".idea",
        ".vscode",
        ".next",
        ".nuxt",
        ".gradle",
        ".sharedagents",
        ".claude",
    }
)
# Per-file cap: above this, skip rather than decode. Source code is
# rarely this big; this dodges multi-GB log files / model dumps that
# dominate runtime when grep accidentally points at one.
_GREP_MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MiB
# Hard ceiling on files examined per grep call. Without this, pointing
# grep at `~` or `/` walks the entire tree (millions of inodes under
# `.cache`, `.local/share`, etc.) before any other limit applies.
_GREP_MAX_FILES = 10000
# Max recursion depth from the root path. Caps pathological deeply
# nested trees (e.g. node_modules-style fan-out we somehow didn't prune).
_GREP_MAX_DEPTH = 15
# Wall-clock budget for one grep invocation. After this, we stop and
# return what we have plus a warning, rather than block the agent loop.
_GREP_TIMEOUT_S = 30.0
# Claude Code ``Bash`` timeout surface (ms). Telegram's no-progress watchdog
# and tool headers import these so operator-visible limits match runtime.
BASH_TIMEOUT_MS_DEFAULT = 120_000
BASH_TIMEOUT_MS_MAX = 600_000
MONITOR_TIMEOUT_MS_DEFAULT = 300_000
MONITOR_TIMEOUT_MS_MAX = 3_600_000
TELEGRAM_IDLE_BUFFER_S = 60.0

# Foreground Bash idle timeout when ``timeout`` is omitted (120000 ms = 2 min).
EXEC_TIMEOUT_DEFAULT_S = BASH_TIMEOUT_MS_DEFAULT / 1000.0


# Hard upper bound regardless of output activity. A perfectly chatty
# infinite loop (`while True: print('.')`) would never trip the idle
# timer, so this is the safety net. 30 min is comfortable for big test
# suites and ML training scripts; the model can override per-call.
EXEC_WALL_TIMEOUT_DEFAULT_S = 1800.0


def _timeout_ms_to_seconds(
    raw: Any,
    *,
    default_ms: float,
    max_ms: float,
) -> float:
    """Parse ``timeout`` / ``timeout_ms`` — always milliseconds (Claude Code)."""
    if raw is None:
        return default_ms / 1000.0
    try:
        ms = float(raw)
    except (TypeError, ValueError):
        return default_ms / 1000.0
    if ms <= 0:
        return default_ms / 1000.0
    return min(ms / 1000.0, max_ms / 1000.0)


def bash_exec_timeouts(args: dict[str, Any]) -> tuple[float, float]:
    """Return ``(idle_timeout_s, wall_timeout_s)`` for foreground ``Bash``."""
    idle = _timeout_ms_to_seconds(
        args.get("timeout"),
        default_ms=BASH_TIMEOUT_MS_DEFAULT,
        max_ms=BASH_TIMEOUT_MS_MAX,
    )
    wall = float(args.get("wall_timeout") or EXEC_WALL_TIMEOUT_DEFAULT_S)
    return idle, wall


def bash_background_timeout_s(args: dict[str, Any]) -> float | None:
    """Wall budget for ``Bash(run_in_background=true)`` (``None`` = unbounded)."""
    if args.get("persistent"):
        return None
    raw = args.get("timeout_ms")
    if raw is None:
        raw = args.get("timeout")
    if raw is None:
        return EXEC_WALL_TIMEOUT_DEFAULT_S
    return _timeout_ms_to_seconds(
        raw,
        default_ms=EXEC_WALL_TIMEOUT_DEFAULT_S * 1000.0,
        max_ms=MONITOR_TIMEOUT_MS_MAX,
    )


def monitor_timeout_s(args: dict[str, Any]) -> float | None:
    if args.get("persistent"):
        return None
    return _timeout_ms_to_seconds(
        args.get("timeout_ms"),
        default_ms=MONITOR_TIMEOUT_MS_DEFAULT,
        max_ms=MONITOR_TIMEOUT_MS_MAX,
    )


def is_operator_wait_tool(tool_name: str) -> bool:
    """True for AskUserQuestion-family parks that wait on a human.

    Delegates to the shared widget dictionary (``widget_adapt``). A miss here is
    expensive: ``worker_idle_timeout_s_for_pending`` then lets the worker
    idle-kill a turn that is parked on a human, and ``telegram_idle_budget_s``
    hands the call the turn's watchdog. Every engine's spelling must park, so
    this can never be a local name list.
    """
    return is_ask_user_tool(tool_name)


def worker_idle_timeout_s_for_pending(
    tool_names: list[str] | tuple[str, ...] | None,
    default_s: float,
) -> float:
    """0 while an ask-family tool is pending — do not idle-kill a parked question."""
    for name in tool_names or []:
        if is_operator_wait_tool(name):
            return 0.0
    return float(default_s)


def telegram_idle_budget_s(tool_name: str, args: dict[str, Any] | None) -> float | None:
    """No-progress budget while a blocking tool runs (``None`` = use turn default).

    Ask/question parks wait for a human and must not inherit the turn
    idle watchdog — ``inf`` disables it for that call.
    """
    if is_operator_wait_tool(tool_name):
        return float("inf")
    a = args or {}
    if tool_name == "bash":
        if a.get("run_in_background"):
            return None
        _idle, wall = bash_exec_timeouts(a)
        return wall + TELEGRAM_IDLE_BUFFER_S
    if tool_name == "Monitor":
        t = monitor_timeout_s(a)
        return (t + TELEGRAM_IDLE_BUFFER_S) if t is not None else None
    return None


# Primary input field(s) per tool for Telegram one-liner headers. Keys are
# casefolded tool names; values are arg key(s) tried in order.
_TOOL_TELEGRAM_PRIMARY: dict[str, tuple[str, ...]] = {
    "bash": ("command",),
    "websearch": ("query", "search_term"),
    "grep": ("pattern",),
    "glob": ("glob_pattern", "pattern"),
    "read": ("path", "file_path", "filePath", "target_file"),
    "read file": ("path", "file_path", "filePath", "target_file"),
    "semanticsearch": ("query",),
    "webfetch": ("url",),
    "memory search": ("query",),
    "memory get": ("path",),
    "toolsearch": ("query",),
    "skillsearch": ("query",),
    "write": ("path", "file_path", "filePath"),
    "edit": ("path", "file_path", "filePath"),
}

# Fallback arg keys when the tool name is unknown (OpenCode MCP plugins).
_TOOL_TELEGRAM_ARG_FALLBACK: tuple[str, ...] = (
    "command",
    "pattern",
    "path",
    "filePath",
    "file_path",
    "url",
    "query",
    "search_term",
    "glob_pattern",
)


def _tool_telegram_primary_value(
    tool_name: str,
    args: dict[str, Any],
) -> str | None:
    """Return the primary scalar input for a tool, if present."""
    key = tool_name.strip().casefold().replace("_", " ")
    fields = _TOOL_TELEGRAM_PRIMARY.get(key)
    if not fields:
        # OpenCode MCP tools arrive as ``brave-search_brave_web_search`` etc.
        if "web search" in key or key.endswith(" websearch"):
            fields = ("query", "search_term")
        elif key.endswith(" webfetch") or key == "webfetch":
            fields = ("url",)
    if fields:
        for field in fields:
            val = args.get(field)
            if val is not None and str(val).strip():
                return str(val)
    for field in _TOOL_TELEGRAM_ARG_FALLBACK:
        val = args.get(field)
        if val is not None and str(val).strip():
            return str(val)
    return None


def format_tool_telegram_header(
    tool_name: str,
    args: dict[str, Any] | None = None,
    *,
    max_len: int = 320,
) -> str:
    """Telegram 🔧 block one-liner: ``Tool: primary-input`` (≤320 chars).

    Used by ``_BlockStream.tool_call`` so operators see *what* the agent
    read/grepped/searched — especially for ACP agents whose tool names
    arrive as ``grep`` / ``Read File`` instead of sharedagents builtins.
    Falls back to :func:`format_tool_header` when no primary input exists.
    """
    a = args or {}
    primary = _tool_telegram_primary_value(tool_name, a)
    if primary:
        one_liner = " ".join(primary.split())
        if len(one_liner) > max_len:
            one_liner = one_liner[: max_len - 1] + "…"
        return f"{tool_name}: {one_liner}"
    try:
        return format_tool_header(tool_name, a)
    except Exception:  # noqa: BLE001
        return tool_name


def format_tool_header(tool_name: str, args: dict[str, Any] | None) -> str:
    """Telegram/TUI-facing tool label with advertised timeout."""
    a = args or {}
    if tool_name == "bash":
        if a.get("run_in_background"):
            t = bash_background_timeout_s(a)
            if t is None:
                return "Bash · background · no timeout"
            return f"Bash · background · timeout {int(t)}s max"
        idle, wall = bash_exec_timeouts(a)
        return f"Bash · idle {int(idle)}s · wall {int(wall)}s max"
    if tool_name == "Monitor":
        t = monitor_timeout_s(a)
        if t is None:
            return "Monitor · persistent · session length"
        return f"Monitor · timeout {int(t)}s max"
    return tool_name


def _background_usage_footer(
    bash_id: str,
    *,
    timeout_s: float | None,
    kind: str = "Bash",
) -> str:
    if timeout_s is None:
        timeout_line = "timeout: none (session / until KillShell)"
    else:
        timeout_line = f"timeout: {int(timeout_s)}s wall max"
    return (
        f"\n\n--- {kind} background ---\n"
        f"{timeout_line}\n"
        f"poll: BashOutput(bash_id={bash_id!r}) — repeat; reply between polls\n"
        f"stop: KillShell(shell_id={bash_id!r})\n"
        "list: monitor_list"
    )


@dataclass(frozen=True)
class ToolResult:
    """One tool invocation's result.

    `content` goes back to the model as the `tool` message body. `ok` is
    metadata for the agent runtime / TUI to render success vs failure
    indicators; `outcome` and `member_outcomes` distinguish refused dispatches
    from failed executions on live and replayed cards. The model sees
    `content`, plus `images` when the tool returned any.
    """

    content: str
    ok: bool = True
    truncated: bool = False
    background_handle: bool = False
    # Display metadata, never inferred from model-facing refusal prose.
    outcome: str | None = None
    member_outcomes: tuple[str | None, ...] = ()
    # OpenAI ``image_url`` parts (inline data URLs) for the model to view,
    # e.g. ``read`` on a PNG. Chat ``tool`` messages are text-only, so the
    # engine delivers these in a ``user`` message after the round's tool
    # results — the same shape a user attachment has.
    images: tuple[dict, ...] = ()


def _truncate(text: str) -> tuple[str, bool]:
    enc = text.encode("utf-8")
    if len(enc) <= MAX_OUTPUT_BYTES:
        return text, False
    head = enc[:MAX_OUTPUT_BYTES].decode("utf-8", "ignore")
    return (
        head + f"\n\n[... truncated, {len(enc) - MAX_OUTPUT_BYTES} more bytes ...]",
        True,
    )


def _err(msg: str) -> ToolResult:
    return ToolResult(content=f"error: {msg}", ok=False)


def _build_delegate_spawn_env(args: dict[str, Any]) -> dict[str, str]:
    """Return the env dict to hand to ``create_subprocess_exec`` for a
    CLI delegate spawn (claude / codex / antigravity).

    Every caller of this helper is a THIRD-PARTY external CLI delegate
    (``run_delegate_{antigravity,claude,codex}_async``), so the subprocess
    gets ONLY the least-privilege allowlist (:func:`delegate_env.
    build_third_party_spawn_env`) — never the full parent ``os.environ``,
    which can carry infra secrets (API tokens) an
    agentic delegate could exfiltrate. The per-agent ``_spawn_env_overrides``
    (API keys lifted from ``~/.sharedagents/agents/<name>/secrets.env``) are the
    CLI's OWN provider credential by design and are always overlaid on top,
    winning any collision.

    NOTE: unlike the pre-DATA-007 behaviour this ALWAYS returns a concrete
    filtered dict (never ``None``) — returning ``None`` would make the runner
    inherit the parent env verbatim, the exact leak this requirement closes.
    """
    from sharedagents_agent.delegate_env import build_third_party_spawn_env

    overrides = args.get("_spawn_env_overrides")
    if not isinstance(overrides, dict):
        overrides = None
    return build_third_party_spawn_env(overrides)


# Well-known user-level bin dirs where the coding CLIs land (npm global,
# pipx / ``--user`` installs, the native installer). A detached daemon
# (systemd / nohup) frequently starts with a minimal PATH that omits these
# even when the operator installed the CLI, so we widen ``shutil.which``'s
# search before giving up with a "not on PATH" error.
_CLI_FALLBACK_BIN_DIRS = (
    "~/.npm-global/bin",
    "~/.local/bin",
    "~/.claude/local",
    "~/.opencode/bin",  # SST OpenCode native installer (`opencode acp`)
    "~/.bun/bin",
    "~/node_modules/.bin",
    "/usr/local/bin",
)


def _windows_cli_fallback_bin_dirs() -> tuple[str, ...]:
    """Common per-user/package-manager bins on a native Windows install."""
    import ntpath

    candidates = (
        r"%APPDATA%\npm",
        r"%LOCALAPPDATA%\Microsoft\WinGet\Links",
        r"%USERPROFILE%\scoop\shims",
        r"%USERPROFILE%\.cargo\bin",
        r"%USERPROFILE%\.bun\bin",
        r"%ProgramFiles%\nodejs",
        ntpath.join(sys.prefix, "Scripts"),
    )
    expanded = (ntpath.expandvars(p) for p in candidates)
    return tuple(p for p in expanded if "%" not in p)


def _which_cli(name: str) -> str | None:
    """Resolve a delegate CLI (claude/codex/antigravity) to an absolute path.

    Tries the inherited ``PATH`` first, then the well-known user bin
    directories in :data:`_CLI_FALLBACK_BIN_DIRS` so a gateway launched as a
    detached daemon with a stripped-down ``PATH`` still finds a CLI the
    operator actually installed. Returns ``None`` when nothing resolves.

    This is the SINGLE source of truth for "is the CLI launchable": both the
    spawn site (:func:`_resolve_cli_executable`) and the launch gate
    (``ExternalCliEngine.is_available``) consult it, so the availability
    check and the actual spawn can never disagree about a CLI that lives in a
    fallback bin dir.
    """
    import shutil

    # Read the fallback-dirs constant through the ``runners`` re-export so a
    # test that does ``monkeypatch.setattr(runners, "_CLI_FALLBACK_BIN_DIRS", …)``
    # is still honoured after the de-monolith split (the constant's canonical
    # home is this module; runners re-exports it). Falls back to the local copy.
    import sharedagents_agent.tools.runners as _runners

    fallback_dirs = getattr(_runners, "_CLI_FALLBACK_BIN_DIRS", _CLI_FALLBACK_BIN_DIRS)
    if oscompat.IS_WINDOWS:
        fallback_dirs = _windows_cli_fallback_bin_dirs()
    found = shutil.which(name)
    if found:
        return found
    extra = os.pathsep.join(os.path.expanduser(d) for d in fallback_dirs)
    cur = os.environ.get("PATH") or ""
    search = extra + os.pathsep + cur if cur else extra
    return shutil.which(name, path=search)


def _resolve_cli_executable(name: str) -> str:
    """Best-effort absolute path for a delegate CLI spawn.

    Delegates to :func:`_which_cli`, falling back to the bare ``name`` when
    nothing resolves so the existing ``FileNotFoundError`` → install-hint
    path still fires for a genuinely-missing CLI.
    """
    # Resolve _which_cli through the ``runners`` re-export so a test patching
    # ``runners._which_cli`` (e.g. a spy) is honoured post-split; fall back to
    # the local definition.
    import sharedagents_agent.tools.runners as _runners

    which = getattr(_runners, "_which_cli", _which_cli)
    return which(name) or name


__all__ = [
    "_ROUTE_TRACE",
    "_feed_stdin",
    "_SIGNAL_MAP",
    "_signal_name",
    "_kill_group",
    "_terminate_group",
    "_descendant_pids",
    "_kill_tree",
    "_delegate_first_token_timeout_s",
    "_ACP_FIRST_TOKEN_TIMEOUT_DEFAULT_S",
    "_acp_first_token_timeout_s",
    "_delegate_idle_timeout_s",
    "_acp_reasoning_stall_s",
    "_CLAUDE_IDLE_TIMEOUT_DEFAULT_S",
    "_claude_idle_timeout_s",
    "_ANTIGRAVITY_PRINT_TIMEOUT_DEFAULT_S",
    "_ZCODE_TIMEOUT_DEFAULT_S",
    "_zcode_timeout_s",
    "_format_go_duration",
    "_antigravity_print_timeout_s",
    "_antigravity_idle_timeout_s",
    "_antigravity_print_timeout_argv",
    "_delegate_readline",
    "MAX_OUTPUT_BYTES",
    "READ_FILE_MAX_BYTES",
    "DEFAULT_READ_LIMIT",
    "READ_MAX_LINE_LENGTH",
    "READ_MAX_LINE_SUFFIX",
    "READ_MAX_OUTPUT_BYTES",
    "READ_MAX_BYTES_LABEL",
    "WEB_FETCH_MAX_BYTES",
    "_GREP_IGNORE_DIRS",
    "_GREP_MAX_FILE_BYTES",
    "_GREP_MAX_FILES",
    "_GREP_MAX_DEPTH",
    "_GREP_TIMEOUT_S",
    "BASH_TIMEOUT_MS_DEFAULT",
    "BASH_TIMEOUT_MS_MAX",
    "MONITOR_TIMEOUT_MS_DEFAULT",
    "MONITOR_TIMEOUT_MS_MAX",
    "TELEGRAM_IDLE_BUFFER_S",
    "is_operator_wait_tool",
    "worker_idle_timeout_s_for_pending",
    "EXEC_TIMEOUT_DEFAULT_S",
    "EXEC_WALL_TIMEOUT_DEFAULT_S",
    "_timeout_ms_to_seconds",
    "bash_exec_timeouts",
    "bash_background_timeout_s",
    "monitor_timeout_s",
    "telegram_idle_budget_s",
    "_TOOL_TELEGRAM_PRIMARY",
    "_TOOL_TELEGRAM_ARG_FALLBACK",
    "_tool_telegram_primary_value",
    "format_tool_telegram_header",
    "format_tool_header",
    "_background_usage_footer",
    "ToolResult",
    "_truncate",
    "_err",
    "_build_delegate_spawn_env",
    "_CLI_FALLBACK_BIN_DIRS",
    "_which_cli",
    "_resolve_cli_executable",
]
