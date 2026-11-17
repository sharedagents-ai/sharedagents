"""Client-role callback handlers: fs + terminal + capability advertisement.

When sharedagents acts as an ACP *client* driving an external agent (cursor /
opencode / generic ACP), the agent may call BACK into sharedagents for
filesystem reads/writes and terminal execution — but only for the
capabilities sharedagents advertised in ``ClientCapabilities``. This module
builds that capability set and the concrete callback handlers
that back it, so the engine registers the full advertised
``CLIENT_METHODS`` surface rather than leaving fs/terminal unhandled.

By default, fs paths and terminal working directories are confined to the
session workspace ``cwd`` and a missing workspace is refused. The engine may
explicitly opt out for Devin's allow-all mode; missing cwd then uses the
process directory. Terminal output is bounded in either mode. Directory
confinement checks paths and working directories, not commands' file access.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import acp.schema as _acp

from .handlers import TerminalHandlers

# Default cap on a terminal's captured output (bytes) when the agent
# doesn't set ``outputByteLimit``. Bounds memory for a chatty command.
_DEFAULT_OUTPUT_BYTE_LIMIT = 1 << 20  # 1 MiB


def build_client_capabilities() -> _acp.ClientCapabilities:
    """Advertise the full client-callback surface: fs read+write, terminal.

    Registering a handler for a capability sharedagents did NOT advertise raises
    (``AcpClient.register_*``); advertising here is what lets the engine
    register the fs/terminal handlers below. ``session/request_permission``
    is always available and needs no capability flag.
    """
    return _acp.ClientCapabilities.model_validate(
        {
            "fs": {"readTextFile": True, "writeTextFile": True},
            "terminal": True,
        }
    )


def _confined_path(cwd: str | None, raw: Any, *, confine: bool = True) -> Path:
    """Resolve ``raw`` against ``cwd``, enforcing confinement by default.

    Fail-closed: when ``confine=True``, a blank ``cwd`` or a path resolving
    outside the workspace raises ``PermissionError`` (surfaced to the peer as
    a JSON-RPC error), never a silent read/write of an arbitrary host path.
    When ``confine=False``, paths outside cwd are allowed and a missing cwd
    falls back to Path.cwd().
    """
    if confine and not cwd:
        raise PermissionError("no workspace configured for fs callback")
    if not isinstance(raw, str) or not raw:
        raise ValueError("fs callback requires a non-empty 'path'")
    base = Path(cwd).resolve() if cwd else Path.cwd()
    target = (base / raw).resolve()
    if confine and target != base and base not in target.parents:
        raise PermissionError(f"path escapes workspace: {raw!r}")
    return target


def make_fs_read_handler(cwd: str | None, *, confine: bool = True):
    async def handler(params: dict[str, Any]) -> dict[str, Any]:
        target = _confined_path(cwd, params.get("path"), confine=confine)
        text = await asyncio.to_thread(target.read_text, "utf-8")
        line = params.get("line")
        limit = params.get("limit")
        if isinstance(line, int) or isinstance(limit, int):
            lines = text.splitlines(keepends=True)
            start = (line - 1) if isinstance(line, int) and line > 0 else 0
            end = (start + limit) if isinstance(limit, int) and limit >= 0 else None
            text = "".join(lines[start:end])
        return {"content": text}

    return handler


def make_fs_write_handler(cwd: str | None, *, confine: bool = True):
    async def handler(params: dict[str, Any]) -> dict[str, Any]:
        target = _confined_path(cwd, params.get("path"), confine=confine)
        content = params.get("content")
        if not isinstance(content, str):
            raise ValueError("fs/write_text_file requires string 'content'")

        def _write() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, "utf-8")

        await asyncio.to_thread(_write)
        return {}

    return handler


class _Terminal:
    """A single subprocess-backed terminal with a bounded output buffer."""

    def __init__(self, proc: asyncio.subprocess.Process, limit: int) -> None:
        self.proc = proc
        self.limit = limit if limit and limit > 0 else _DEFAULT_OUTPUT_BYTE_LIMIT
        self._buf = bytearray()
        self.truncated = False
        self._pump = asyncio.create_task(self._drain())

    async def _drain(self) -> None:
        stream = self.proc.stdout
        if stream is None:
            return
        try:
            while True:
                chunk = await stream.read(4096)
                if not chunk:
                    break
                self._buf.extend(chunk)
                if len(self._buf) > self.limit:
                    del self._buf[: len(self._buf) - self.limit]
                    self.truncated = True
        except asyncio.CancelledError:
            return
        except Exception:  # noqa: BLE001 - best-effort capture
            return

    def output(self) -> str:
        return self._buf.decode("utf-8", "replace")

    def exit_status(self) -> dict[str, Any] | None:
        code = self.proc.returncode
        if code is None:
            return None
        if code < 0:
            return {"exitCode": None, "signal": str(-code)}
        return {"exitCode": code, "signal": None}

    async def kill(self) -> None:
        if self.proc.returncode is None:
            try:
                self.proc.kill()
            except ProcessLookupError:
                pass

    async def release(self) -> None:
        await self.kill()
        if not self._pump.done():
            self._pump.cancel()


def make_terminal_handlers(
    cwd: str | None,
    env: dict[str, str] | None = None,
    *,
    confine: bool = True,
) -> TerminalHandlers:
    """Build the five ``terminal/*`` handlers over a shared terminal table.

    Each terminal runs in the session ``cwd`` (or process cwd if unconfined)
    with a bounded captured-output buffer, so a chatty or runaway command can't
    exhaust memory and can always be killed/released by id.
    """
    terminals: dict[str, _Terminal] = {}
    counter = {"n": 0}

    async def create(params: dict[str, Any]) -> dict[str, Any]:
        if confine and not cwd:
            raise PermissionError("no workspace configured for terminal callback")
        command = params.get("command")
        if not isinstance(command, str) or not command:
            raise ValueError("terminal/create requires a 'command'")
        args = params.get("args") or []
        if args:
            argv = [command, *[str(a) for a in args if a is not None]]
        else:
            from sharedagents_core import oscompat

            argv = oscompat.spawn_shell_argv(command)
        run_cwd = params.get("cwd")
        target_cwd = str(_confined_path(cwd, run_cwd, confine=confine)) if run_cwd else cwd or str(Path.cwd())
        child_env = dict(os.environ if env is None else env)
        for item in params.get("env") or []:
            if isinstance(item, dict) and "name" in item:
                child_env[str(item["name"])] = str(item.get("value", ""))
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=target_cwd,
            env=child_env,
        )
        counter["n"] += 1
        tid = f"term-{counter['n']}"
        terminals[tid] = _Terminal(proc, int(params.get("outputByteLimit") or 0))
        return {"terminalId": tid}

    def _get(params: dict[str, Any]) -> _Terminal:
        tid = params.get("terminalId")
        term = terminals.get(tid) if isinstance(tid, str) else None
        if term is None:
            raise KeyError(f"unknown terminalId: {tid!r}")
        return term

    async def output(params: dict[str, Any]) -> dict[str, Any]:
        term = _get(params)
        return {
            "output": term.output(),
            "truncated": term.truncated,
            "exitStatus": term.exit_status(),
        }

    async def wait_for_exit(params: dict[str, Any]) -> dict[str, Any]:
        term = _get(params)
        await term.proc.wait()
        status = term.exit_status() or {"exitCode": None, "signal": None}
        return status

    async def kill(params: dict[str, Any]) -> dict[str, Any]:
        await _get(params).kill()
        return {}

    async def release(params: dict[str, Any]) -> dict[str, Any]:
        tid = params.get("terminalId")
        term = terminals.pop(tid, None) if isinstance(tid, str) else None
        if term is not None:
            await term.release()
        return {}

    return TerminalHandlers(
        create=create,
        output=output,
        wait_for_exit=wait_for_exit,
        kill=kill,
        release=release,
    )


__all__ = [
    "build_client_capabilities",
    "make_fs_read_handler",
    "make_fs_write_handler",
    "make_terminal_handlers",
]
