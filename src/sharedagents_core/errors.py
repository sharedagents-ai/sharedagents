"""Structured error log at `~/.sharedagents/errors/`.

When something goes wrong inside sharedagents — a Claude-bridge readline
overflow, a worker that crashes, an unhandled exception in the agent
runtime — we write a Markdown file describing the failure: timestamp,
component, exception type, traceback, captured context. The agent can
later read these (`list_dir` + `read_file`) and propose fixes; once a
fix ships, the file is renamed with a `[fixed]_` prefix to keep the
"open" set small.

## File layout

```
~/.sharedagents/errors/
  2026-04-26T14-32-15Z_a1b2c3d4_claude-bridge-overflow.md   ← open
  [fixed]_2026-04-25T10-00-00Z_d4e5f6a7_dispatch-race.md    ← resolved
```

Filename: `<UTC-ISO-ts>_<fp>_<slug>.md`, `[fixed]_` prefix when resolved.

`<fp>` is an 8-char SHA-1 of (component, exception type, top traceback
frame, first arg of the exception). Same fingerprint = same bug — so a
recurring crash appends `## Occurrence — <ts>` sections to one file
instead of flooding the directory.

## Frontmatter

YAML-style header at the top of every file:

```yaml
---
timestamp: 2026-04-26T14-32-15Z
component: claude_bridge
exception: ValueError
fingerprint: a1b2c3d4
status: open
---
```

The agent can grep these fields to triage without reading the body.

## API

- `log_error(component, exc, context=None, summary=None)` — write or
  append. Returns the file path.
- `mark_fixed(path)` — rename with `[fixed]_` prefix and append a
  resolution timestamp. Returns the new path.
- `list_errors(status="open" | "fixed" | "all")` — sorted recency-first.
- `errors_dir()` — `~/.sharedagents/errors`, created lazily.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def errors_dir() -> Path:
    """`<root>/errors` for the active workspace.

    Resolves the workspace root the same way as
    :func:`sharedagents_agent.agent.sharedagents_root` so a test that sets
    ``$SHAREDAGENTS_HOME`` (or the more-specific ``$SHAREDAGENTS_AGENTS_HOME``)
    gets an isolated errors dir alongside the rest of the runtime tree.
    """
    base = os.environ.get("SHAREDAGENTS_AGENTS_HOME")
    if base:
        root = Path(base).parent
    else:
        home = os.environ.get("SHAREDAGENTS_HOME")
        root = Path(home).expanduser() if home else Path.home() / ".sharedagents"
    return root / "errors"


def _write_private(path: Path, body: str) -> None:
    """Write `body` to `path` with mode 0600.

    Tracebacks may include partial user prompts, env-var values, or
    captured tool args (which can contain API keys / file paths). The
    default umask makes new files world-readable on most systems; on a
    shared host that's a quiet PII / secret leak. Write 0600 so only the
    owner reads.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(str(path), flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(body)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def _open_private_append(path: Path):
    """Append-mode open with 0600 perms on first create. Returns a
    text-mode file object (caller `close`s it).
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    fd = os.open(str(path), flags, 0o600)
    return os.fdopen(fd, "a", encoding="utf-8")


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _fingerprint(component: str, exc: BaseException) -> str:
    """8-char SHA-1 over (component, exc_type, top_frame_file:line, arg[0]).

    Stable enough to dedupe recurring failures, brittle enough that an
    edit which moves the failing line gets a fresh fingerprint — which
    is what we want; a relocated bug IS a new bug.
    """
    tb = exc.__traceback__
    while tb is not None and tb.tb_next is not None:
        tb = tb.tb_next
    if tb is not None:
        top_file = (tb.tb_frame.f_code.co_filename or "").rsplit("/", 1)[-1]
        top_line = tb.tb_lineno or 0
    else:
        top_file, top_line = "", 0
    arg0 = ""
    if exc.args:
        try:
            arg0 = str(exc.args[0])[:120]
        except Exception:  # noqa: BLE001
            arg0 = ""
    key = f"{component}:{type(exc).__name__}:{top_file}:{top_line}:{arg0}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]


def _slug(text: str, max_len: int = 60) -> str:
    """File-safe lowercase kebab. Always non-empty."""
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s[:max_len] or "error"


def _scrub(text: str) -> str:
    """Redact secrets from captured error content before it is persisted.

    Error records land in cleartext at ``~/.sharedagents/errors/*.md``. An
    exception message, its formatted traceback (which echoes that
    message), and a captured context blob can each carry credentials —
    a bare ``sk-…`` token, an ``Authorization: Bearer …`` header, a
    ``PASSWORD=…`` env line — exactly the classes the session-JSON path
    already runs through :func:`secrets_scrubber.scrub_text`. Route the
    same content through the same chokepoint so a secret in an
    exception cannot land unscrubbed on disk.

    Late-bound like :func:`_get_hook_engine`: ``errors.py`` is imported
    very early during agent boot, so it keeps its own import graph
    minimal. ``secrets_scrubber`` is stdlib-only, so this cannot fail
    in practice; and if scrubbing ever did raise, we would rather no
    error file be written than a secret written verbatim.

    The scrubber preserves base64 ``data:`` URI media byte-for-byte —
    that exemption is deliberate; do not defeat it.
    """
    from sharedagents_core.secrets_scrubber import scrub_text  # lazy: keep boot import graph minimal

    return scrub_text(text)


def _format_traceback(exc: BaseException, *, limit: int = 8000) -> str:
    """Format the exception traceback, scrubbed and length-capped.

    The traceback's final line echoes the exception message
    (``ExcType: <arg0>``), so scrubbing the message alone would still
    leak a credential here — the scrub happens before the truncation so
    the whole text is scanned, then capped at ``limit``.
    """
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return _scrub(text)[:limit]


def _format_context(context: dict[str, Any] | None, *, limit: int = 2000) -> str:
    """Serialize the captured context to JSON, scrubbed and length-capped.

    Scrub the full serialized blob before truncating so a secret is
    never split across the ``limit`` boundary and thereby missed.
    """
    if not context:
        return ""
    try:
        body = json.dumps(context, indent=2, default=str, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        body = repr(context)
    return _scrub(body)[:limit]


def _get_hook_engine(agent_name: str | None = None):  # pragma: no cover - thin lookup, tested via patches
    """Return the registered HookEngine for ``agent_name`` (default ``jarvis``).

    Late-bound (instead of a top-level import) for two reasons:

    1. ``sharedagents_core.errors`` is imported very early during agent
       boot — circular-import risk if it pulls in ``sharedagents_agent.hooks``
       and the hook engine's transitive deps.
    2. Tests can ``patch.object(errors, '_get_hook_engine', ...)``
       without touching the real engine.
    """
    try:
        from sharedagents_agent.hooks import get_active_engine  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    try:
        return get_active_engine(agent_name)
    except Exception:  # noqa: BLE001
        return None


def _maybe_fire_session_error_hook(
    *,
    component: str,
    exc: BaseException,
    fp: str,
    path: Path,
    session_id: str | None,
    agent_name: str | None = None,
) -> None:
    """Fire SessionError on the active hook engine, swallowing all errors.

    Logging-side hook invocation is strictly best-effort — a failure
    here must not break the caller, and must not recurse back into
    ``log_error`` (or we'd loop on a broken hook engine).

    ``HookEngine.run`` is ``async def``; ``log_error`` is sync and may
    be called from either sync code (TUI worker handlers) or inside a
    running event loop (turn_runtime). We dispatch the coroutine in a
    short-lived daemon thread with its own ``asyncio.run`` so the
    behaviour is identical from either context, and the caller never
    waits.

    ``async: true`` hooks are scheduled by ``engine.run`` via
    ``asyncio.ensure_future`` and the call returns *without* awaiting
    them. In a short-lived ``asyncio.run`` loop those tasks would be
    cancelled before they ever started, so we explicitly
    ``drain_async_hooks()`` afterwards to make sure the spawns landed.
    """
    engine = _get_hook_engine(agent_name)
    if engine is None:
        return
    payload = {
        "component": component,
        "exception": type(exc).__name__,
        "fingerprint": fp,
        "file_path": str(path),
        "session_id": session_id,
        "agent_name": agent_name,
    }

    import asyncio
    import inspect
    import threading

    def _runner() -> None:
        # Call ``engine.run`` synchronously first so that even a
        # MagicMock-patched engine (which returns a non-awaitable) still
        # records the call before we attempt to drive the event loop.
        try:
            result = engine.run(event="SessionError", payload=payload)
        except Exception:  # noqa: BLE001
            return

        async def _drive() -> None:
            if inspect.isawaitable(result):
                await result
            drain = getattr(engine, "drain_async_hooks", None)
            if drain is not None:
                try:
                    drained = drain()
                    if inspect.isawaitable(drained):
                        await drained
                except Exception:  # noqa: BLE001
                    pass

        try:
            asyncio.run(_drive())
        except Exception:  # noqa: BLE001
            # Swallow — best-effort, must never propagate.
            pass

    try:
        threading.Thread(target=_runner, daemon=True).start()
    except Exception:  # noqa: BLE001
        pass


def log_error(
    component: str,
    exc: BaseException,
    *,
    context: dict[str, Any] | None = None,
    summary: str | None = None,
    session_id: str | None = None,
    agent_name: str | None = None,
) -> Path:
    """Record an error in `~/.sharedagents/errors/`. See module docstring.

    If an OPEN file with the same fingerprint already exists, append a
    new `## Occurrence — <ts>` section to it. This way a flapping
    failure doesn't flood the dir; one fix retires one file.

    The exception message, formatted traceback, and serialized context
    are secret-scrubbed before they are written — see :func:`_scrub`.
    """
    d = errors_dir()
    d.mkdir(parents=True, exist_ok=True)
    ts = _utc_ts()
    fp = _fingerprint(component, exc)

    # Merge session_id into the recorded context so the self-improver
    # can read the originating session JSON off the error file alone.
    # An explicit ``session_id`` key in ``context`` wins (caller knows
    # best); otherwise the kwarg fills it in.
    effective_context = dict(context or {})
    if session_id and "session_id" not in effective_context:
        effective_context["session_id"] = session_id

    # Look for an existing OPEN file with this fingerprint.
    for existing in d.glob(f"*_{fp}_*.md"):
        if existing.name.startswith("[fixed]"):
            continue
        with _open_private_append(existing) as f:
            f.write(f"\n\n## Occurrence — {ts}\n\n")
            if effective_context:
                f.write("**Context**:\n\n```json\n")
                f.write(_format_context(effective_context))
                f.write("\n```\n\n")
            f.write("**Traceback**:\n\n```\n")
            f.write(_format_traceback(exc))
            f.write("\n```\n")
        _maybe_fire_session_error_hook(
            component=component,
            exc=exc,
            fp=fp,
            path=existing,
            session_id=session_id,
            agent_name=agent_name,
        )
        return existing

    # New file.
    summary_slug = _slug(summary or f"{component}-{type(exc).__name__}")
    path = d / f"{ts}_{fp}_{summary_slug}.md"
    msg = ""
    if exc.args:
        try:
            # Scrub before truncating so a secret spanning the 400-char
            # cap is fully scanned rather than sliced past detection.
            msg = _scrub(str(exc.args[0]))[:400]
        except Exception:  # noqa: BLE001
            msg = ""
    body: list[str] = [
        "---",
        f"timestamp: {ts}",
        f"component: {component}",
        f"exception: {type(exc).__name__}",
        f"fingerprint: {fp}",
        "status: open",
        "---",
        "",
        f"# {component}: {type(exc).__name__}",
        "",
    ]
    if summary:
        body += [f"> {summary}", ""]
    if msg:
        body += ["**Message**:", "", f"> {msg}", ""]
    if effective_context:
        body += [
            "## Context",
            "",
            "```json",
            _format_context(effective_context),
            "```",
            "",
        ]
    body += ["## Traceback", "", "```", _format_traceback(exc), "```", ""]
    _write_private(path, "\n".join(body))
    _maybe_fire_session_error_hook(
        component=component,
        exc=exc,
        fp=fp,
        path=path,
        session_id=session_id,
        agent_name=agent_name,
    )
    return path


def mark_fixed(path: Path | str, *, resolution: str | None = None) -> Path:
    """Rename an error file with the `[fixed]_` prefix and append a
    resolution stamp. Idempotent — re-marking a `[fixed]_` file is a no-op.

    `resolution` is a one-line summary of how it got fixed (e.g. the
    commit hash, or a sentence). Stored at the bottom of the file so a
    future agent reading the dir can learn from past fixes.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.name.startswith("[fixed]_"):
        return p
    new = p.with_name(f"[fixed]_{p.name}")
    p.rename(new)
    # Update frontmatter `status: open` → `status: fixed` and append a
    # resolution section.
    text = new.read_text(encoding="utf-8")
    text = re.sub(r"^status:\s*open\s*$", "status: fixed", text, count=1, flags=re.MULTILINE)
    ts = _utc_ts()
    text += f"\n---\n\n## Fixed — {ts}\n"
    if resolution:
        text += f"\n{resolution}\n"
    _write_private(new, text)
    return new


def list_errors(*, status: str = "open") -> list[Path]:
    """List error files, sorted most-recent-first.

    `status`: `"open"` (default), `"fixed"`, or `"all"`.
    """
    d = errors_dir()
    if not d.exists():
        return []
    paths = sorted(d.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    if status == "open":
        return [p for p in paths if not p.name.startswith("[fixed]_")]
    if status == "fixed":
        return [p for p in paths if p.name.startswith("[fixed]_")]
    return paths
