"""Extracted tool-runner family: shell (de-monolith of runners.py).

Behavior-preserving split — code moved verbatim.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sharedagents_core import oscompat

from ._utils import (
    MAX_OUTPUT_BYTES,
    ToolResult,
    _err,
    _kill_tree,
    _terminate_group,
    _truncate,
    bash_background_timeout_s,
    bash_exec_timeouts,
)

# ---- exec ----


def _bash_executable() -> str | None:
    """Path to a ``bash`` binary to run shell command *strings*, or
    ``None`` to fall back to the platform default (``/bin/sh``).

    The ``Bash`` tool's command strings come from Claude Code skills and
    the user's CLAUDE.md, which assume *bash* semantics — ``source``,
    ``[[ … ]]``, arrays, ``${arr[@]}``. But ``create_subprocess_shell``
    and ``Popen(shell=True)`` run via ``/bin/sh``, which on Debian/Ubuntu
    is ``dash`` and lacks those builtins. A literal
    ``source venv/bin/activate && …`` then dies with
    ``/bin/sh: 1: source: not found`` (exit 127) — observed wrecking a
    real research session's tweet-search calls. Resolve a real bash so
    the "Bash" tool actually behaves like bash. ``SHAREDAGENTS_EXEC_SHELL``
    overrides the choice. On Windows resolve cmd.exe through oscompat;
    on POSIX retain the historical ``/bin/sh`` fallback if Bash is absent.
    """
    if oscompat.IS_WINDOWS:
        return oscompat.shell_executable()
    if os.name != "posix":
        return None
    override = os.environ.get("SHAREDAGENTS_EXEC_SHELL")
    if override:
        return override
    import shutil

    found = shutil.which("bash")
    if found:
        return found
    for cand in ("/bin/bash", "/usr/bin/bash"):
        if os.path.exists(cand):
            return cand
    return None


async def run_exec_async(
    args: dict[str, Any],
    *,
    on_delta: Callable[[str], None] | None = None,
) -> ToolResult:
    """Run a shell command, streaming its stdout/stderr line-by-line.

    Two timeouts:

    - `timeout` (idle, default 120s): kill if NO output arrives for this
      many seconds. A long-running command that keeps printing progress
      is NOT killed when it crosses 120s wall time. Each line resets the
      idle timer. This is what users actually want — the previous code
      killed a healthy, streaming exec at exactly 60s wall time.
    - `wall_timeout` (absolute, default 30 min): hard ceiling regardless
      of output activity. Catches a chatty infinite loop that would
      otherwise dodge the idle timer forever.

    Lines are pushed to `on_delta` (one call per line) so the parent
    tool_runner emits `delta` events the agent forwards as `tool_delta`
    SSE — the TUI shows live exec output the same way it shows
    `delegate_claude` text. stderr lines are tagged inline so the user
    can tell the streams apart.

    Killability stays unchanged: if the awaiting task is cancelled (ESC
    → SIGTERM on the tool_runner), the `finally` block kills the child
    subprocess so it doesn't keep running orphaned.
    """
    command = args.get("command")
    # OpenCode's ``bash`` steers the model to ``workdir`` instead of ``cd``.
    # Accept it as the primary working-directory arg, falling back to the
    # legacy ``cwd`` alias.
    cwd = args.get("workdir") or args.get("cwd")
    # OpenCode's ``bash`` carries ``timeout`` in milliseconds (max 600000).
    # sharedagents keeps a background mode (``run_in_background``) as an extension.
    idle_timeout, wall_timeout = bash_exec_timeouts(args)
    if not isinstance(command, str) or not command:
        return _err("command required")
    # Gateway self-protection (defense-in-depth). The permission gate
    # already classifies these FORBIDDEN, but refuse at the exec boundary
    # too so a gateway-kill can never spawn even if the permission layer
    # is bypassed (a PreToolUse hook that force-allows, a future code
    # path that skips criticality, etc.). ``restart`` is NOT matched, so
    # the agent can still bounce the gateway — it just can't leave it
    # down. See ``sharedagents_agent.tools.safety.kills_sharedagents_gateway``.
    from ..safety import kills_sharedagents_gateway

    if kills_sharedagents_gateway(command):
        return _err(
            "refused: this command would terminate the sharedagents gateway "
            "the agent runs inside. Killing the gateway is never allowed; "
            "use `sharedagents gateway restart` to bounce it instead."
        )
    if args.get("run_in_background"):
        # run_monitor_start_async lives in the monitor/background family; import
        # it from the runners re-export point at call time (behaviour-preserving
        # cross-module reference — was a same-module global before the split).
        from sharedagents_agent.tools.runners import run_monitor_start_async

        bg_timeout = bash_background_timeout_s(args)
        return await run_monitor_start_async(
            {
                "command": command,
                "cwd": cwd,
                "description": args.get("description") or command[:60],
                "timeout_ms": int(bg_timeout * 1000) if bg_timeout is not None else None,
                "persistent": bool(args.get("persistent")),
                "_footer_kind": "Bash",
            }
        )
    cwd_path: str | None = None
    if isinstance(cwd, str) and cwd:
        cwd_path = str(Path(cwd).expanduser())
    try:
        # Bump the StreamReader buffer above asyncio's 64 KiB default so
        # a single huge line (e.g. an unwrapped JSON dump from a Python
        # script) doesn't blow up `readline()` with LimitOverrunError.
        #
        # Run via bash, NOT /bin/sh. `create_subprocess_shell` uses
        # `/bin/sh -c`, which on Debian/Ubuntu is dash — dash lacks
        # `source`, `[[ … ]]` and arrays that Claude Code skills /
        # CLAUDE.md routinely emit, so `source venv/bin/activate` dies
        # with "source: not found" (exit 127). `bash -c <cmd>` is the
        # exec-form equivalent. Falls back to the platform default shell
        # when no bash is found (see `_bash_executable`).
        bash = _bash_executable()
        spawn_kwargs = dict(
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd_path,
            limit=4 * 1024 * 1024,
            # New session = own process group. ESC / timeout sends
            # SIGTERM to the group via `_kill_group` so any commands
            # the shell spawned (a build, a `find`, a piped chain) get
            # the signal too, instead of being orphaned.
            start_new_session=not oscompat.IS_WINDOWS,
            **oscompat.detach_kwargs(),
        )
        if oscompat.IS_WINDOWS:
            proc = await asyncio.create_subprocess_exec(
                *oscompat.spawn_shell_argv(command),
                **spawn_kwargs,
            )
        elif bash is not None:
            proc = await asyncio.create_subprocess_exec(
                bash,
                "-c",
                command,
                **spawn_kwargs,
            )
        else:
            proc = await asyncio.create_subprocess_shell(
                command,
                **spawn_kwargs,
            )
    except OSError as e:
        return _err(f"spawn failed: {e}")

    # Execution lease (spec §10): this Bash exec is the canonical untrusted
    # unit (§10.4 "sandbox the thing that runs arbitrary commands"). Lease its
    # process group (start_new_session=True → pgid == pid); the now broker-aware
    # ``_kill_group`` routes the cancel through the capability layer for us, so
    # the cancel path below is unchanged. Best-effort: leasing never blocks the
    # tool and the raw group kill stays the guaranteed fallback.
    from sharedagents_agent import sandbox_broker

    _lease_id = sandbox_broker.lease_process(pgid=proc.pid, kind="exec")

    out_chunks: list[str] = []
    err_chunks: list[str] = []
    loop = asyncio.get_running_loop()
    started_at = loop.time()
    last_output_at = started_at
    kill_reason: str | None = None

    async def _drain(
        stream: asyncio.StreamReader | None,
        sink: list[str],
        is_stderr: bool,
    ) -> None:
        """Read one line at a time, append to `sink`, push as a delta.

        Returns naturally when the stream hits EOF (process exited and
        all buffered output has been drained).
        """
        nonlocal last_output_at
        if stream is None:
            return
        while True:
            try:
                line = await stream.readline()
            except ValueError:
                # Single line exceeded the StreamReader buffer. Skip
                # (don't drop the rest of the stream) — `readline`
                # raises but the buffer is left in a recoverable state.
                continue
            if not line:
                return
            text = line.decode("utf-8", "replace")
            sink.append(text)
            last_output_at = loop.time()
            if on_delta is not None:
                try:
                    on_delta(f"[stderr] {text}" if is_stderr else text)
                except Exception:  # noqa: BLE001
                    pass

    out_task = asyncio.create_task(_drain(proc.stdout, out_chunks, False))
    err_task = asyncio.create_task(_drain(proc.stderr, err_chunks, True))
    wait_task = asyncio.create_task(proc.wait())

    try:
        # Poll loop: sleep until the process exits OR until the next
        # idle/wall deadline. `asyncio.shield(wait_task)` prevents the
        # `wait_for` timeout from cancelling the underlying `proc.wait()`,
        # so we can keep re-awaiting it across iterations.
        while not wait_task.done():
            now = loop.time()
            idle_left = idle_timeout - (now - last_output_at)
            wall_left = wall_timeout - (now - started_at)
            if idle_left <= 0:
                kill_reason = f"killed: no output for {idle_timeout:.0f}s (idle timeout)"
                break
            if wall_left <= 0:
                kill_reason = f"killed: wall-clock {wall_timeout:.0f}s exceeded"
                break
            sleep_for = min(idle_left, wall_left, 5.0)
            try:
                await asyncio.wait_for(
                    asyncio.shield(wait_task),
                    timeout=sleep_for,
                )
            except asyncio.TimeoutError:
                # Loop again to re-check the deadlines.
                continue
        if kill_reason:
            await _terminate_group(proc, grace_s=2.0)
    except asyncio.CancelledError:
        # Hard cancel → SIGKILL this proc's ENTIRE descendant tree
        # synchronously so nothing outlives the cancellation — including a
        # pipeline child that started its OWN session, which a lone killpg
        # misses. Don't `await proc.wait()` from inside the cancelled
        # coroutine — let the loop's child watcher reap. ``_kill_tree``
        # routes its group kill through the broker-aware ``_kill_group``
        # (downward-only, gateway-guarded, §10.3) and never signals the
        # gateway's group.
        _kill_tree(proc, oscompat.FORCE_KILL)
        sandbox_broker.release_process(_lease_id)
        raise

    # Wait for the drain tasks AND the wait_task to finish so we don't
    # leak pending tasks into the loop's shutdown phase. Drain tasks
    # return on EOF (kernel delivers it as soon as the child exits)
    # so this should be near-instant; small timeout is a safety net.
    try:
        await asyncio.wait_for(
            asyncio.gather(
                out_task,
                err_task,
                wait_task,
                return_exceptions=True,
            ),
            timeout=2.0,
        )
    except asyncio.TimeoutError:
        for t in (out_task, err_task, wait_task):
            if not t.done():
                t.cancel()

    rc = proc.returncode if proc.returncode is not None else -9
    out = "".join(out_chunks)
    err = "".join(err_chunks)
    parts = [f"$ {command}"]
    if kill_reason:
        parts.append(f"[{kill_reason}]")
        parts.append("[hint] for long work use Bash(run_in_background=true) and poll with BashOutput(bash_id=...).")
    else:
        parts.append(f"[exit {rc}]")
    if out.strip():
        parts.append("--- stdout ---")
        parts.append(out.rstrip())
    if err.strip():
        parts.append("--- stderr ---")
        parts.append(err.rstrip())
    body = "\n".join(parts)

    # OpenCode truncate-to-file: when combined output overflows the cap, spill
    # the FULL body to a temp file and prefix the (tail-)clipped result with a
    # pointer to it, so the model can retrieve the rest instead of losing it.
    full_body = body
    truncated = False
    if len(body.encode("utf-8")) > MAX_OUTPUT_BYTES:
        saved_path: str | None = None
        try:
            import tempfile

            fd, saved_path = tempfile.mkstemp(prefix="sharedagents-bash-", suffix=".log")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(full_body)
        except OSError:
            saved_path = None
        clipped, _ = _truncate(body)
        body = f"...output truncated...\n\nFull output saved to: {saved_path}\n\n" + clipped if saved_path else clipped
        truncated = True

    # OpenCode appends a <shell_metadata> block describing an abnormal exit
    # (timeout / kill) with a retry-with-larger-timeout hint.
    if kill_reason:
        meta = (
            f"shell tool terminated command: {kill_reason}. If this command "
            "is expected to take longer and is not waiting for interactive "
            "input, retry with a larger timeout value in milliseconds."
        )
        body += "\n\n<shell_metadata>\n" + meta + "\n</shell_metadata>"

    # The exec is done — release its lease so the broker's spawn-tree + kill
    # scope stay bounded (§10.5).
    sandbox_broker.release_process(_lease_id)
    return ToolResult(
        content=body,
        ok=(kill_reason is None and rc == 0),
        truncated=truncated,
    )


__all__ = [
    "_bash_executable",
    "run_exec_async",
]
