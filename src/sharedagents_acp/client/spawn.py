"""Spawn a stdio ACP agent subprocess and wrap it in an ``AcpClient``.

The lifecycle contract here is intentionally minimal:

- ``spawn_stdio_agent(cmd, ...)`` launches ``cmd`` as a child process
  with piped stdin/stdout/stderr, wraps its stdio in
  ``StdioTransport``, builds an ``AcpClient`` over that transport, and
  returns ``(client, process)``.
- The caller owns the returned ``process``. Closing the client (via
  ``client.close()``) closes the transport, which closes the child's
  stdin — agents are expected to exit cleanly on stdin EOF. For a hard
  kill (the sharedagents ``TurnState.stop`` SIGTERM path), the caller does
  ``process.terminate()`` (or ``process.kill()``) directly on the
  ``Process`` handle.
- Stderr is left open and piped; the caller can drain it for
  diagnostics. We don't pump it ourselves to avoid taking ownership of
  yet another background task.

This honors the ``CLAUDE.md`` killability rule: every long-running
operation runs in its own OS process, and the PID is exposed to the
caller for direct signal control. ``AcpClient.close`` is the orderly
shutdown path; ``process.terminate`` is the hard kill path. Neither
goes through threads or in-loop coroutines.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from ..protocol.jsonrpc import ACP_STREAM_LIMIT
from ..protocol.transport import StdioTransport
from .client import AcpClient


def _resolve_child_env(env: dict[str, str] | None, inherit_parent_env: bool) -> dict[str, str] | None:
    """Decide the child process's environment.

    Pure + side-effect free so it is unit-testable without a live spawn.

    * ``inherit_parent_env=True`` (INTERNAL / co-located child): merge ``env``
      onto the full parent ``os.environ`` (legacy). ``None`` → ``None`` so the
      subprocess inherits the parent env unchanged.
    * ``inherit_parent_env=False`` (THIRD-PARTY child): return ``env``
      VERBATIM as the child's ENTIRE env — the parent ``os.environ`` is NEVER
      merged in. ``None`` fails CLOSED to an empty env, never the parent's.
    """
    if inherit_parent_env:
        return None if env is None else {**os.environ, **env}
    return dict(env) if env is not None else {}


async def spawn_stdio_agent(
    cmd: list[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    inherit_parent_env: bool = True,
    client_capabilities=None,
    client_info=None,
    return_process: bool = True,
    frontend=None,
):
    """Launch ``cmd`` and wrap its stdio in an ``AcpClient``.

    Args:
        cmd: argv list, e.g. ``["claude", "--acp"]``.
        cwd: working directory for the child; defaults to the parent's
            cwd if ``None``.
        env: environment variables for the child process. Interpreted
            per ``inherit_parent_env``.
        inherit_parent_env: gate for passing the parent environment to the
            child.

            * ``True`` (default) — the INTERNAL / co-located spawn: ``env``
              is MERGED onto the full parent ``os.environ`` (legacy
              behaviour). This is correct only when the child is trusted —
              e.g. the TUI spawning sharedagents's OWN agent over stdio
              (``python -m sharedagents_gateway acp-stdio``).
            * ``False`` — the THIRD-PARTY spawn (external ACP server:
              cursor / opencode / generic ACP): ``env`` is used VERBATIM as
              the child's ENTIRE environment; the parent ``os.environ`` is
              NOT merged in. Third-party callers pass an env already reduced
              to the third-party env allowlist (``sharedagents_agent.delegate_env.
              build_third_party_spawn_env``) so no infra secret leaks into an
              autonomous, prompt-injectable delegate. Passing ``env=None``
              here yields an EMPTY child env (fail-closed) rather than the
              parent's.
        client_capabilities: passed through to ``AcpClient``.
        client_info: passed through to ``AcpClient``.
        return_process: if True (default) return ``(client, process)``;
            if False return only ``client``. The process handle is the
            killability anchor — most callers want it.

    Returns:
        ``(AcpClient, asyncio.subprocess.Process)`` or just
        ``AcpClient`` depending on ``return_process``. The client's
        read loop has been started; ``await client.initialize()`` is
        the next step.
    """

    full_env = _resolve_child_env(env, inherit_parent_env)

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd) if cwd is not None else None,
        env=full_env,
        # NDJSON framing reads a whole line before the ``\n`` delimiter;
        # base64 image blocks overflow asyncio's 64 KiB default reader
        # buffer, so size the child's stdout reader off ACP_STREAM_LIMIT.
        limit=ACP_STREAM_LIMIT,
    )
    if process.stdin is None or process.stdout is None:
        # Should not happen with PIPE on both sides, but the type is
        # ``Optional`` so we narrow it explicitly.
        raise RuntimeError("subprocess stdio not available after spawn")

    transport = StdioTransport(process.stdout, process.stdin)
    client = AcpClient(
        transport,
        client_capabilities=client_capabilities,
        client_info=client_info,
        frontend=frontend,
    )
    await client.start()

    if return_process:
        return client, process
    return client


__all__ = ["spawn_stdio_agent", "_resolve_child_env"]
