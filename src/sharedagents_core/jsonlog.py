"""JSONL log writer with origin tags + cross-process flock.

Single writer used by every sharedagents service (gateway, heartbeat, tui,
telegram, self-improver) so the file lock + JSONL encoding + origin-
tag discipline lives in one place.

Module-level name is ``jsonlog`` deliberately — not ``logging`` — so
this module never shadows the stdlib ``logging`` package for other
sharedagents modules that ``import logging``.

Line schema
-----------

Mandatory keys: ``ts``, ``service``, ``agent``, ``level``, ``msg``.

Optional keys:

* ``kind`` — structured tag for grep-ability (e.g. ``TURN_METRICS``,
  ``MONITOR_RC``, ``DELEGATE_AGENT``, ``HEARTBEAT``).
* ``session_id``, ``parent_session_id`` — link a log line to one
  session in the unified ``sessions.db``.
* ``round_idx``, ``turn_idx`` — narrow to one turn/round.
* ``k`` — free-form ``{str: scalar}`` payload for line-specific facts.

Concurrency
-----------

* In-process: a threading ``RLock`` serialises writes within the
  process so two engines in the same gateway can't interleave bytes.
* Cross-process: every line is written under
  ``fcntl.flock(fd, LOCK_EX)`` so two agents whose gateways share
  ``logs/gateway.log`` serialise too. On Linux ext4/xfs this is true
  serialisation; on NFS the lock is advisory + client-local
  (documented limitation — production runs on local ext4).
* ``get_writer(path, service=…)`` is a per-(path, pid) singleton so
  one process owns one file descriptor per log file.
  ``reset_writers_for_test()`` clears the cache between pytest
  fixtures.

Corruption tolerance
--------------------

Each line is a single ``os.write`` of a ``\\n``-terminated JSON
string with ``ensure_ascii=False`` and compact separators. A killed
writer at worst loses its last unflushed bytes; readers tolerant of
partial last lines (which any JSONL consumer should be) work without
recovery.
"""

from __future__ import annotations

import json
import logging as _stdlib_logging  # aliased: this module is also ``jsonlog``
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sharedagents_core import oscompat


def _now_iso_ms() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class JsonLogWriter:
    """Append-only JSONL log file with cross-process flock per line."""

    def __init__(self, path: Path, *, service: str) -> None:
        if not service:
            raise ValueError("JsonLogWriter requires a non-empty service tag")
        self.path = Path(path)
        self.service = service
        self._rlock = threading.RLock()
        self._fd = -1
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _open(self) -> int:
        """Open (or return the cached) fd for this log file, mode 0o600.

        The log carries tracebacks, tool arguments, and hook/worker
        stderr that can contain credentials or PII, so the file is
        created owner-only (``0o600``) — never world-readable — the
        same discipline ``errors.py`` applies to captured error files.
        ``O_APPEND`` preserves the append/rotation semantics; only the
        creation mode changed.
        """
        if self._fd >= 0:
            return self._fd
        self._fd = os.open(
            str(self.path),
            os.O_WRONLY | os.O_APPEND | os.O_CREAT | oscompat.O_BINARY,
            0o600,
        )
        return self._fd

    def write(
        self,
        *,
        agent: str,
        level: str,
        msg: str,
        kind: str | None = None,
        session_id: str | None = None,
        parent_session_id: str | None = None,
        round_idx: int | None = None,
        turn_idx: int | None = None,
        k: dict | None = None,
        ts: str | None = None,
    ) -> None:
        """Write one JSONL line under the in-process lock + cross-process flock.

        The serialized line passes through
        :func:`secrets_scrubber.scrub_text` at this single write
        chokepoint before it reaches disk, so a credential-shaped value
        landing in ``msg``/``k`` (hook or worker stderr, delegate error
        previews) receives the same redaction the session-JSON corpus
        gets. The scrubber respects its ``data:`` base64 exemption and
        is idempotent. NOTE (hot path): scrubbing runs the full regex
        registry (~two dozen patterns) on every line write; correctness
        over the credential-leak risk takes priority here.
        """
        for name, val in (("agent", agent), ("level", level), ("msg", msg)):
            if not val:
                raise ValueError(f"JsonLogWriter.write: {name} is required")
        row: dict[str, Any] = {
            "ts": ts or _now_iso_ms(),
            "service": self.service,
            "agent": agent,
            "level": level,
            "msg": msg,
        }
        if kind:
            row["kind"] = kind
        if session_id:
            row["session_id"] = session_id
        if parent_session_id:
            row["parent_session_id"] = parent_session_id
        if round_idx is not None:
            row["round_idx"] = int(round_idx)
        if turn_idx is not None:
            row["turn_idx"] = int(turn_idx)
        if k:
            row["k"] = k
        # ``default=str`` is the difference between "logging is best-
        # effort observability" and "logging crashes the call site":
        # an Exception / object() / threading.Lock that lands in
        # ``k`` from a fast-path caller (acp_bridge cancel paths,
        # delegate error rows) MUST not raise. Coerce to repr() and
        # let the operator see a degraded row.
        # Lazy import: keeps this foundational, early-imported module
        # free of any import-order coupling to the scrubber (which is
        # stdlib-only, so there is no cycle — this is defensive). The
        # module is cached after first use, so the per-line cost is a
        # dict lookup plus the regex pass documented above.
        from sharedagents_core.secrets_scrubber import scrub_text

        line = (
            scrub_text(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                )
            )
            + "\n"
        )
        data = line.encode("utf-8")
        with self._rlock:
            fd = self._open()
            try:
                oscompat.file_lock(fd)
                os.write(fd, data)
            finally:
                try:
                    oscompat.file_unlock(fd)
                except OSError:
                    pass

    def close(self) -> None:
        with self._rlock:
            if self._fd >= 0:
                try:
                    os.close(self._fd)
                finally:
                    self._fd = -1


# Per-(path, pid) singleton so two callers in one process share one fd
# + lock for a given file. PID is in the key so a fork doesn't reuse
# the parent's stale fd.
_WRITERS: dict[tuple[str, int], JsonLogWriter] = {}
_WRITERS_LOCK = threading.Lock()


def get_writer(path: Path, *, service: str) -> JsonLogWriter:
    """Singleton JsonLogWriter for ``(resolved-path, current-pid)``."""
    key = (str(Path(path).resolve()), os.getpid())
    with _WRITERS_LOCK:
        w = _WRITERS.get(key)
        if w is None:
            w = JsonLogWriter(path, service=service)
            _WRITERS[key] = w
        return w


def reset_writers_for_test() -> None:
    """Drop the per-pid writer cache and close any open fds.

    Used by pytest fixtures so each test gets a fresh writer pointing
    at the test's tmp_path.
    """
    with _WRITERS_LOCK:
        for w in _WRITERS.values():
            try:
                w.close()
            except OSError:
                pass
        _WRITERS.clear()


# ---- stdlib logging adapter ---------------------------------------


class JsonlLogHandler(_stdlib_logging.Handler):
    """``logging.Handler`` that emits JSONL via :class:`JsonLogWriter`.

    Attaches to the ``sharedagents`` (or any) stdlib logger tree so plain
    ``_LOG.info("…")`` calls land in the SAME unified
    ``layout.gateway_logfile()`` (or ``telegram_logfile``) as the
    structured ``log_event`` calls — no parallel plaintext sink, no
    drift between writers.

    Parameters
    ----------
    path
        Target log file (e.g. ``layout.gateway_logfile()``).
    service
        Service tag stamped on every emitted line (e.g. ``"gateway"``,
        ``"telegram"``).
    default_agent
        Fallback ``agent`` value when the calling logger didn't set
        one via the LogRecord's ``agent`` attr. ``layout.current_agent_name()``
        is queried first; this is the absolute fallback.

    Concurrency
    -----------
    Delegates to the underlying ``JsonLogWriter`` singleton, which
    holds the in-process RLock + ``fcntl.flock`` and re-uses one fd
    per (path, pid). Idempotent — attaching the same handler twice
    to the same logger emits each line once (the writer's lock
    serialises, the stdlib ``logging`` machinery dedups handlers).
    """

    def __init__(
        self,
        path: Path,
        *,
        service: str,
        default_agent: str = "",
    ) -> None:
        super().__init__()
        self._path = Path(path)
        self._service = service
        self._default_agent = default_agent

    # ``baseFilename`` lets :func:`_attach_gateway_logfile_handler`
    # short-circuit a duplicate attach the same way it did for the
    # legacy ``RotatingFileHandler``.
    @property
    def baseFilename(self) -> str:  # noqa: N802 — stdlib convention
        return str(self._path.resolve())

    def emit(self, record: _stdlib_logging.LogRecord) -> None:
        try:
            agent = getattr(record, "agent", None) or _resolve_default_agent() or self._default_agent or "unknown"
            msg = record.getMessage()
            extras: dict[str, Any] = {}
            for k in ("session_id", "parent_session_id", "round_idx", "turn_idx", "kind"):
                v = getattr(record, k, None)
                if v not in (None, ""):
                    extras[k] = v
            # Stash logger name in ``k`` so a fine-grained reader
            # can still see WHERE the line came from.
            kpayload: dict[str, Any] = {"logger": record.name}
            if record.exc_info:
                kpayload["exc"] = self.format(record).splitlines()[-1][:500]
            writer = get_writer(self._path, service=self._service)
            writer.write(
                agent=str(agent),
                level=record.levelname,
                msg=msg,
                kind=extras.pop("kind", None),
                session_id=extras.pop("session_id", None),
                parent_session_id=extras.pop("parent_session_id", None),
                round_idx=extras.pop("round_idx", None),
                turn_idx=extras.pop("turn_idx", None),
                k=kpayload,
            )
        except (OSError, ValueError):
            # Never let logging-machinery failure raise into the
            # calling code path. The stdlib ``Handler`` base would
            # otherwise call ``self.handleError(record)``, which
            # writes a traceback to stderr.
            self.handleError(record)


def _resolve_default_agent() -> str:
    """Best-effort: read ``$SHAREDAGENTS_AGENT_NAME`` lazily."""
    return (os.environ.get("SHAREDAGENTS_AGENT_NAME") or "").strip()


__all__ = [
    "JsonLogWriter",
    "JsonlLogHandler",
    "get_writer",
    "reset_writers_for_test",
]
