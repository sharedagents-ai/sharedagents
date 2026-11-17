"""`remember` — the only tool that can write to MEMORY.md / memory/*.md.

Memory-only. The agent's curated notes are private continuity,
not external source storage.

Pipeline (every successful call):

  1. Validate args (action / target / required content / required old_text)
  3. Resolve paths (target=memory → MEMORY.md ;
                    target=daily  → memory/YYYY-MM-DD.md)
  4. Markdown write
       - target=daily  action=add → atomic-append a dated bullet (no LLM)
       - target=memory action=add → SYNTHESIZE (small LLM merge of
         related existing entries + the new fact) → atomic-replace
         section
       - replace/remove           → substring-match, atomic-rewrite
  5. memory.db reindex (chunk + embed; per-agent SQLite)

Failure semantics: markdown writes happen FIRST and are atomic
(tmp+fsync+rename). If memory.db indexing fails after the markdown
landed, we surface the error in `RememberResult.error` but the file
is durable; the next heartbeat tick reindexes everything from disk.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sharedagents_mcp.memory.engine.memory_index import MemoryIndex

# ---- per-file write locks -------------------------------------------
#
# Concurrent `remember` calls (e.g. user turn + heartbeat tick) racing
# on the same daily file would each read the baseline, append, and
# write — and one of the two writes would silently win, dropping the
# other entry. We serialise read-modify-write per file with an
# `asyncio.Lock`; locks are stored weakly per-path so repeated calls
# to the same daily share a lock. Lock acquisition lives in the async
# entry point (`remember`) — the synchronous helpers stay unchanged.
_DAILY_LOCKS: dict[str, asyncio.Lock] = {}


def _lock_for(path: Path) -> asyncio.Lock:
    key = str(path)
    lock = _DAILY_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _DAILY_LOCKS[key] = lock
    return lock


# ---- security scanner -----------------------------------------------

# Invisible / format Unicode that's been used as a prompt-injection
# channel. A "fact" like `innocent text​<malicious>` looks fine
# on screen but smuggles instructions into the model on rehydrate.
# Rejecting on add is the simplest way to keep the daily log clean.
_INVISIBLE_UNICODE: tuple[str, ...] = (
    "​",  # ZERO WIDTH SPACE
    "‌",  # ZERO WIDTH NON-JOINER
    "‍",  # ZERO WIDTH JOINER
    "﻿",  # ZERO WIDTH NO-BREAK SPACE / BOM
    "⁠",  # WORD JOINER
    "‎",  # LEFT-TO-RIGHT MARK
    "‏",  # RIGHT-TO-LEFT MARK
)

# Credential-shaped patterns. Lossy on purpose — we'd rather refuse a
# legitimate fact that happens to look like a key than persist a real
# one to disk where it'd land in `git diff` and embeddings.
_CREDENTIAL_RE = re.compile(
    r"(sk-[a-zA-Z0-9]{20,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|-----BEGIN (RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----)"
)


def _scan_content(content: str) -> str | None:
    """Return a refusal reason iff `content` trips a security check.

    Pure function — easy to unit-test. Returns None when content is
    clean. Scoped to the high-confidence patterns the spec calls out so
    we don't false-positive on legitimate operator notes.
    """
    if not isinstance(content, str):
        return None
    for ch in _INVISIBLE_UNICODE:
        if ch in content:
            return f"rejected: contains invisible Unicode character U+{ord(ch):04X}"
    if _CREDENTIAL_RE.search(content):
        return "rejected: appears to contain credentials"
    return None


# ---- result type -----------------------------------------------------


@dataclass(frozen=True)
class RememberResult:
    ok: bool
    summary: str
    sections_synthesized: int = 0
    entries_added: int = 0
    entities_new: int = 0
    entities_updated: int = 0
    concepts_new: int = 0
    concepts_updated: int = 0
    relations_added: int = 0
    duration_s: float = 0.0
    error: str | None = None


# ---- valid categories the agent may pass to remember -----------------

CATEGORIES: tuple[str, ...] = (
    "fact",
    "preference",
    "decision",
    "lesson",
    "todo",
    "context",
    "profile",
    "project",
)


# ---- the public entry point -----------------------------------------


async def remember(
    *,
    action: str,
    content: str = "",
    old_text: str = "",
    category: str = "",
    cfg,  # AgentConfig
    workspace_origin: str,
    chat_id: str = "",
    upstream_endpoint: str = "",  # unused; kept for back-compat callers
    upstream_model: str = "",  # unused
    backend: str = "vllm",  # unused
) -> RememberResult:
    """Daily-only memory writer. MEMORY.md is owned by the heartbeat
    synthesizer; the agent never touches it directly.

    Actions:
      - `add`     — append a dated bullet to memory/YYYY-MM-DD.md.
      - `replace` — substring-match in today's daily, atomic-rewrite.
      - `remove`  — substring-match in today's daily, delete that span.
    """
    started = time.monotonic()

    # 1. Validate args.
    action = (action or "").strip().lower()
    category = (category or "").strip().lower() or "context"
    if action not in ("add", "replace", "remove"):
        return RememberResult(
            ok=False,
            summary=f"invalid action: {action!r}",
            error="invalid_action",
        )
    if action == "add" and not (content or "").strip():
        return RememberResult(
            ok=False,
            summary="add: content required.",
            error="missing_content",
        )
    if action in ("replace", "remove") and not (old_text or "").strip():
        return RememberResult(
            ok=False,
            summary=f"{action}: old_text required.",
            error="missing_old_text",
        )
    if action == "replace" and not (content or "").strip():
        return RememberResult(
            ok=False,
            summary="replace: content required.",
            error="missing_content",
        )

    # 2.5. Security scan. `add` and `replace` write `content` into the
    # daily log; `remove` only consumes `old_text` and leaves no new
    # text on disk, so we skip the scan there. Failures here are NOT
    # silent — the agent / operator should see the rejection so they
    # can sanitize the input rather than wonder why the entry didn't
    # land.
    if action in ("add", "replace"):
        reason = _scan_content(content)
        if reason is not None:
            return RememberResult(
                ok=False,
                summary=f"remember refused: {reason}",
                error="security_refused",
            )

    # 3. Resolve daily path.
    home = Path(cfg.home)
    home.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    (home / "memory").mkdir(exist_ok=True)
    target_path = home / "memory" / f"{today}.md"

    # 4. Markdown write — daily-only. MEMORY.md is curated by the
    #    heartbeat synthesizer (see heartbeat.py / synth_memory). The
    #    agent only ever writes raw observations to today's daily;
    #    the synthesizer lifts general principles up to MEMORY.md
    #    out-of-band, dropping specifics (paths, branches, perf
    #    numbers) along the way.
    entries_added = 0
    write_error: str | None = None
    # Serialise the entire read-modify-write so two concurrent
    # `remember` calls on the same daily file can't lose entries.
    duplicate_skipped = False
    async with _lock_for(target_path):
        try:
            if action == "add":
                wrote = _append_daily_entry(
                    target_path,
                    category=category,
                    content=content,
                )
                if wrote:
                    entries_added = 1
                else:
                    duplicate_skipped = True
            elif action == "replace":
                _substring_replace(target_path, old_text=old_text, new_text=content)
                entries_added = 1
            elif action == "remove":
                _substring_replace(target_path, old_text=old_text, new_text="")
                entries_added = 1
        except _MarkdownWriteError as e:
            return RememberResult(
                ok=False,
                summary=str(e),
                error="markdown_write_failed",
            )
        except Exception as e:  # noqa: BLE001
            write_error = f"{type(e).__name__}: {e}"

    # 6. memory.db reindex.
    try:
        from sharedagents_mcp.memory.engine.memory_embed import resolve_embedder

        embedder = None
        try:
            embedder = resolve_embedder()
        except Exception:  # noqa: BLE001
            embedder = None
        mem_idx = MemoryIndex(home, embedder=embedder)
        mem_idx.index_all()
    except Exception as e:  # noqa: BLE001
        write_error = (write_error or "") + f"; memory.index_all: {e}"

    duration = round(time.monotonic() - started, 3)

    if duplicate_skipped:
        head = f"skipped duplicate add to memory/{today}.md"
    elif entries_added:
        head = f"saved {entries_added} entry to memory/{today}.md"
    else:
        head = f"{action} memory/{today}.md"
    summary_bits = [head, f"t={duration}s"]
    summary = " · ".join(summary_bits)

    return RememberResult(
        ok=write_error is None,
        summary=summary,
        sections_synthesized=0,
        entries_added=entries_added,
        entities_new=0,
        entities_updated=0,
        concepts_new=0,
        concepts_updated=0,
        relations_added=0,
        duration_s=duration,
        error=write_error,
    )


# ---- daily-log append ------------------------------------------------


def _append_daily_entry(
    target_path: Path,
    *,
    category: str,
    content: str,
) -> bool:
    """Append `- {ISO ts} [{category}] {content}` to the daily log.

    Returns True iff a new entry was actually written; False when an
    identical `[category] content` line already exists in today's log
    and we skipped the write to avoid duplicate noise (heartbeats firing
    twice within a tick on the same observation, the model re-asserting
    a fact we already captured, etc.). Same-day dedup only — a
    yesterday-vs-today repeat is intentional, those land in different
    files.

    Atomic-append: read, append, fsync-rename. Fires `daily_appended`
    on the process-wide event bus so the memory synthesizer (subscribed
    elsewhere, debounced) can lift any new general principles up to
    MEMORY.md out-of-band.
    """
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    norm_content = content.rstrip()
    duplicate_marker = f"[{category}] {norm_content}"
    if target_path.exists():
        body = target_path.read_text(encoding="utf-8")
        # Same-day dedup: any prior bullet whose text after the
        # timestamp matches `[category] content` is treated as the
        # same observation. We check substring match so the
        # `- {ts} ` prefix doesn't trip us up.
        if duplicate_marker in body:
            return False
        if not body.endswith("\n"):
            body += "\n"
        body += f"- {ts} {duplicate_marker}\n"
    else:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        body = f"# {date_str}\n\n- {ts} {duplicate_marker}\n"
    _atomic_write(target_path, body)
    # Notify subscribers. Best-effort: import + publish failures must
    # never break the write path. Payload now also carries `action` /
    # `content` / `category` so providers wired via
    # `MemoryManager.on_memory_write` can mirror the write without a
    # second disk read.
    try:
        from sharedagents_core.kernel.event_bus import get_bus

        get_bus().publish(
            "daily_appended",
            {
                "home": str(target_path.parent.parent),
                "daily_path": str(target_path),
                "ts": ts,
                "action": "add",
                "category": category,
                "content": norm_content,
            },
        )
    except Exception:  # noqa: BLE001
        pass
    return True


# ---- substring replace (action=replace/remove) -----------------------


class _MarkdownWriteError(Exception):
    pass


def _substring_replace(
    target_path: Path,
    *,
    old_text: str,
    new_text: str,
) -> None:
    if not target_path.is_file():
        raise _MarkdownWriteError(f"file does not exist: {target_path.name}")
    body = target_path.read_text(encoding="utf-8")
    n = body.count(old_text)
    if n == 0:
        raise _MarkdownWriteError(
            f"old_text not found in {target_path.name!r} — give more specific context or use action=add."
        )
    if n > 1:
        raise _MarkdownWriteError(
            f"old_text matches {n} entries in {target_path.name!r} — "
            "give more specific (longer / unique) context to identify "
            "exactly one entry."
        )
    new_body = body.replace(old_text, new_text, 1)
    # Cleanup: collapse any double blank lines created by removal.
    new_body = re.sub(r"\n{3,}", "\n\n", new_body)
    _atomic_write(target_path, new_body)


# ---- helpers --------------------------------------------------------


def _atomic_write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(body)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


__all__ = [
    "CATEGORIES",
    "RememberResult",
    "remember",
]
