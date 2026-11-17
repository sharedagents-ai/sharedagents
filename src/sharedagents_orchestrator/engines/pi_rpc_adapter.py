"""Adapter for Pi coding agent's JSONL RPC subprocess interface.

Pi RPC is a long-lived, bidirectional JSONL protocol (``pi --mode rpc``),
distinct from both ACP and one-shot CLI execution. This adapter owns the
process and translates Pi records into stable orchestrator events.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

_log = logging.getLogger(__name__)


@dataclass
class PiRunResult:
    """Summary collected from a Pi RPC turn."""

    status: str  # "completed" | "error" | "aborted"
    content: str
    session_id: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def parse_pi_record(line: bytes | str) -> dict[str, Any] | None:
    """Decode one strict JSONL record; ignore blanks and malformed lines."""
    text = line.decode("utf-8", errors="replace") if isinstance(line, bytes) else line
    text = text.strip()
    if not text:
        return None
    try:
        record = json.loads(text)
    except json.JSONDecodeError:
        _log.debug("ignoring non-JSON Pi RPC stdout record")
        return None
    return record if isinstance(record, dict) else None


def translate_pi_record(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize Pi event records for orchestrator consumers.

    The source record is retained as ``raw`` so later consumers can use
    Pi-specific metadata without forcing protocol details into core state.
    """
    kind = str(record.get("type") or "")
    event: dict[str, Any] = {"type": kind, "raw": record}
    if kind == "message_update":
        update = record.get("assistantMessageEvent")
        if isinstance(update, dict):
            subtype = str(update.get("type") or "")
            event["type"] = subtype
            if subtype == "text_delta" or subtype == "thinking_delta":
                event["chunk"] = str(update.get("delta") or "")
            elif subtype in {"toolcall_start", "toolcall_delta", "toolcall_end"}:
                event["tool_call"] = update
    elif kind == "response":
        event["success"] = bool(record.get("success"))
        event["command"] = str(record.get("command") or "")
        event["error"] = str(record.get("error") or "")
    elif kind == "agent_settled":
        event["settled"] = True
    elif kind == "message_end":
        event["message"] = record.get("message")
    return event


class PiRpcAdapter:
    """Manage a Pi RPC process, command correlation and JSONL event stream."""

    def __init__(
        self,
        *,
        command: list[str] | None = None,
        model: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        no_session: bool = False,
        max_record_bytes: int = 16 * 1024 * 1024,
        stderr_tail_bytes: int = 64 * 1024,
        process_factory: Any = None,
    ) -> None:
        self.command = list(command or ["pi"])
        self.model = model
        self.cwd = cwd
        self.env = env
        self.no_session = no_session
        if max_record_bytes <= 0 or stderr_tail_bytes <= 0:
            raise ValueError("Pi stream limits must be positive")
        self.max_record_bytes = max_record_bytes
        self.stderr_tail_bytes = stderr_tail_bytes
        self.stderr_tail = b""
        self._process_factory = process_factory
        self.process: Any = None
        self._process_group_id: int | None = None
        self._request_id = 0
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._reader_error: Exception | None = None
        self._events: asyncio.Queue[dict[str, Any] | Exception] = asyncio.Queue()
        self._write_lock = asyncio.Lock()
        self._start_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._prompt_active = False

    @staticmethod
    def resolve_executable(executable: str = "pi") -> str | None:
        """Resolve Pi from PATH and common user-local CLI locations."""
        from sharedagents_agent.tools.runners import _which_cli

        return _which_cli(executable)

    def build_command(self) -> list[str]:
        """Return Pi's RPC-mode startup argv."""
        argv = list(self.command)
        resolved = self.resolve_executable(argv[0])
        if resolved:
            argv[0] = resolved
        argv.extend(["--mode", "rpc"])
        if self.no_session:
            argv.append("--no-session")
        if self.model:
            argv.extend(["--model", self.model])
        return argv

    def is_available(self) -> tuple[bool, str]:
        """Report whether the configured Pi executable is launchable."""
        executable = self.resolve_executable(self.command[0])
        if executable:
            return True, ""
        return False, f"Pi CLI `{self.command[0]}` is not installed or launchable"

    async def start(self) -> None:
        """Start the long-lived process and its stdout reader."""
        async with self._start_lock:
            await self._start()

    async def _start(self) -> None:
        if self.process is not None:
            return
        self._events = asyncio.Queue()
        self._reader_error = None
        self.stderr_tail = b""
        child_env = None if self.env is None else {**os.environ, **self.env}
        spawn = self._process_factory or asyncio.create_subprocess_exec
        self.process = await spawn(
            *self.build_command(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.cwd,
            env=child_env,
            start_new_session=(os.name != "nt"),
        )
        pid = getattr(self.process, "pid", None)
        self._process_group_id = int(pid) if pid is not None and os.name != "nt" else None
        self._reader_task = asyncio.create_task(self._read_stdout())
        self._stderr_task = asyncio.create_task(self._read_stderr())

    async def _read_stderr(self) -> None:
        """Drain diagnostics without retaining unbounded logs or logging secrets."""
        assert self.process is not None
        stream = self.process.stderr
        if stream is None:
            return
        while chunk := await stream.read(8192):
            self.stderr_tail = (self.stderr_tail + chunk)[-self.stderr_tail_bytes :]

    async def _records(self) -> AsyncIterator[dict[str, Any]]:
        """Frame on LF without StreamReader.readline's 64 KiB limit."""
        assert self.process is not None and self.process.stdout is not None
        stream = self.process.stdout
        buffered = bytearray()
        while chunk := await stream.read(65536):
            buffered.extend(chunk)
            start = 0
            while (end := buffered.find(b"\n", start)) != -1:
                if end - start > self.max_record_bytes:
                    raise ValueError("Pi RPC record exceeds configured size limit")
                record = parse_pi_record(bytes(buffered[start:end]))
                if record is not None:
                    yield record
                start = end + 1
            del buffered[:start]
            if len(buffered) > self.max_record_bytes:
                raise ValueError("Pi RPC record exceeds configured size limit")
        if buffered:
            raise ConnectionError("Pi RPC stdout ended inside a JSONL record")

    async def _read_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        try:
            async for record in self._records():
                if record.get("type") == "response" and record.get("id") is not None:
                    request_id = str(record["id"])
                    future = self._pending.pop(request_id, None)
                    if future is not None and not future.done():
                        future.set_result(record)
                await self._events.put(translate_pi_record(record))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._reader_error = exc
        finally:
            error = self._reader_error or ConnectionError("Pi RPC process exited")
            self._reader_error = error
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(error)
            self._pending.clear()
            await self._events.put(error)

    async def send(
        self, command: dict[str, Any], *, wait: bool = False, timeout: float = 30.0
    ) -> dict[str, Any] | None:
        """Write one JSONL command, optionally awaiting its correlated response."""
        if self.process is None:
            await self.start()
        if self._reader_error is not None:
            raise self._reader_error
        assert self.process is not None and self.process.stdin is not None
        request_id = str(command.get("id") or self._next_id())
        payload = {**command, "id": request_id}
        future: asyncio.Future[dict[str, Any]] | None = None
        if wait:
            if request_id in self._pending:
                raise ValueError(f"Duplicate Pi request ID: {request_id}")
            future = asyncio.get_running_loop().create_future()
            self._pending[request_id] = future
        try:
            async with self._write_lock:
                self.process.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode())
                await self.process.stdin.drain()
            return await asyncio.wait_for(future, timeout) if future is not None else None
        finally:
            if future is not None:
                self._pending.pop(request_id, None)
                if not future.done():
                    future.cancel()
                elif not future.cancelled():
                    future.exception()

    def _next_id(self) -> str:
        self._request_id += 1
        return f"sharedagents-{self._request_id}"

    async def prompt(
        self, message: str, *, images: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Send a prompt and stream normalized events through agent_settled."""
        if self._prompt_active:
            raise RuntimeError("A Pi prompt is already active")
        self._prompt_active = True
        try:
            await self.start()
            request_id = self._next_id()
            # Earlier command responses and idle events belong to the previous
            # operation. There is only one prompt consumer for this process.
            while not self._events.empty():
                self._events.get_nowait()
            payload: dict[str, Any] = {"id": request_id, "type": "prompt", "message": message}
            if images:
                payload["images"] = images
            await self.send(payload)
            accepted = settled = False
            while True:
                event = await self._events.get()
                if isinstance(event, Exception):
                    raise event
                raw = event["raw"]
                if event.get("type") == "response" and raw.get("id") == request_id:
                    if not event.get("success"):
                        raise RuntimeError(event.get("error") or "Pi rejected prompt")
                    accepted = True
                yield event
                if event.get("type") == "agent_settled":
                    settled = True
                if settled and accepted:
                    break
        finally:
            self._prompt_active = False

    async def abort(self) -> None:
        """Ask Pi to abort its active turn."""
        if self.process is None:
            return
        try:
            await self.send({"type": "abort"})
        except (BrokenPipeError, ConnectionError, OSError):
            pass

    async def close(self, *, force: bool = False) -> None:
        """Close stdin, then bound shutdown with TERM/KILL if it stalls."""
        async with self._close_lock:
            await self._close(force=force)

    async def _close(self, *, force: bool) -> None:
        process = self.process
        if process is None:
            return
        if process.stdin is not None and not process.stdin.is_closing():
            process.stdin.close()
        try:
            if callable(getattr(process, "wait", None)):
                if force:
                    self._signal_process(process, force=True)
                try:
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    self._signal_process(process, force=False)
                    try:
                        await asyncio.wait_for(process.wait(), timeout=2.0)
                    except asyncio.TimeoutError:
                        self._signal_process(process, force=True)
                        await asyncio.wait_for(process.wait(), timeout=2.0)
        finally:
            try:
                for task in (self._reader_task, self._stderr_task):
                    if task is not None:
                        try:
                            await asyncio.wait_for(task, timeout=0.5)
                        except (asyncio.TimeoutError, asyncio.CancelledError):
                            pass
            finally:
                self._reader_task = self._stderr_task = None
                self.process = None
                self._process_group_id = None

    def _signal_process(self, process: Any, *, force: bool) -> None:
        """Signal this owned subprocess group without touching its parent."""
        try:
            if self._process_group_id is not None:
                os.killpg(
                    self._process_group_id,
                    signal.SIGKILL if force else signal.SIGTERM,
                )
            elif force:
                process.kill()
            else:
                process.terminate()
        except (ProcessLookupError, OSError):
            pass


__all__ = ["PiRpcAdapter", "PiRunResult", "parse_pi_record", "translate_pi_record"]
