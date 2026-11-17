"""SQLite-backed FTS5 search over saved session turns.

Sessions are JSON files at `<agent_home>/sessions/<hash>.json`; nothing
indexes their *contents* today, so "did we discuss X last week?" has no
fast answer. This module fixes that with a separate `sessions.db`
keyed by session hash + turn index, indexed via FTS5 on user/assistant
text combined per turn.

Two-table layout (sessions_meta + turns) instead of letting the FTS
table double as storage. Reasons:

- A future migration that wants role splitting / per-side scoring
  doesn't need a re-import — the `turns` table holds the durable
  rows; FTS is a derived index that can be rebuilt.
- `pin_fact` / future "summarize this session" tools can hit the
  `turns` table directly without going through MATCH.

Memory and session indexes are deliberately split into different
DBs so the memory index's chunking + embedding pipeline (see
`memory_index.py`) doesn't get polluted with conversational noise.

Lifecycle:

- `Session.save()` (in-process) publishes `session_saved` on the
  shared event bus.
- `subscribe_for_sessions(home)` wires that event to a non-blocking
  `index_session()` call and runs a one-shot incremental `refresh()`
  to backfill any un-indexed history on first wire.
- The `session_search` tool (subprocess-side) opens a fresh
  `SessionIndex` per call — same lazy pattern `memory_tools.py`
  uses for `MemoryIndex`. The tool runner subprocess inherits
  `SHAREDAGENTS_AGENT_HOME` and `SHAREDAGENTS_WORKSPACE_ORIGIN` so the privacy
  gate enforces the same trusted-origin set memory tools do.

FTS-only by design — vector search over conversational text added
little signal in early prototypes (turns are short, terms repeat,
BM25 + recency does the heavy lifting). Re-introduce it later by
mirroring the `chunks_vec` virtual table from `memory_index.py` if
the recall numbers ever justify it.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Recall primitives now live in the kernel leaf. Importing
# them from ``sharedagents_core.kernel.recall`` instead of the memory engine deletes
# the top-level core→memory back-edge (C1) that would otherwise close a startup
# import cycle now that the memory engine lives under ``sharedagents_mcp.memory``
# (Phase B, done).
from sharedagents_core.kernel.recall import (
    DEFAULT_SNIPPET_CHARS,
    RecallResult,
    bm25_rank_to_score,
    build_fts_query,
    format_recall_hits,
)
from sharedagents_session.ask_pending import is_ask_tool_name
from sharedagents_session.session_activity import (
    ATTENTION_STATES,
    INTERNAL_SIDE_CALL_AGENTS,
    LEGACY_STATES,
    OUTCOMES,
    OWN_LIVE_STATES,
    PHASES,
    TREE_BUSY_STATES,
    legacy_phase,
)

_log = logging.getLogger(__name__)


DEFAULT_MAX_RESULTS: int = 6
DEFAULT_MIN_SCORE: float = 0.35

#: Abort reasons that mean the work was interrupted by the runtime, not
#: cancelled by the owner.
_INTERRUPTED_ABORT_REASONS = frozenset(
    {
        "orchestrator_restart",
        "stale_live_reconcile",
        "gateway_restart",
        "worker_lost",
        "process_exit",
    }
)
#: A clean turn that ends with the remaining work blocked on the owner or an
#: outside event is not a completed task.
_NON_SUCCESS_DISPOSITIONS = frozenset({"waiting_approval", "waiting_external"})


def turn_outcome(event: Mapping[str, Any]) -> str:
    """Classify one ``turn_ended`` event's outcome.

    Order matters: an error is ``failed``; an abort is ``interrupted`` for a
    runtime restart and ``cancelled`` for an owner stop; a clean turn whose
    durable disposition is blocked on the owner/external is ``interrupted``;
    everything else is ``succeeded``.
    """
    metrics = event.get("metrics") or {}
    abort_reason = str(event.get("abort_reason") or metrics.get("abort_reason") or "").strip().lower()
    disposition = str(metrics.get("workflow_disposition") or "").strip().lower()
    if event.get("error"):
        return "failed"
    if event.get("aborted"):
        return "interrupted" if abort_reason in _INTERRUPTED_ABORT_REASONS else "cancelled"
    if disposition in _NON_SUCCESS_DISPOSITIONS:
        return "interrupted"
    return "succeeded"


# ---- text extraction ------------------------------------------------

_COMPACT_SENTINEL_PREFIX = "__SHAREDAGENTS_DELEGATE__:compact_context"
_COMPACT_SLASH = "/compact"


def _is_compact_user_text(user: Any) -> bool:
    """True when this turn is an operator ``/compact`` (sentinel or slash).

    Used by the projector so ``session_state`` becomes ``compacting`` at
    ``turn_started`` — not later, and not ``tool_running``. A refresh of
    the WebUI mid-compact then still sees a live compacting session.
    """
    text = _extract_user_text(user).strip()
    if not text:
        return False
    if text.startswith(_COMPACT_SENTINEL_PREFIX):
        return True
    if text == _COMPACT_SLASH or text.startswith(_COMPACT_SLASH + " "):
        return True
    return bool(text.startswith(_COMPACT_SLASH + "\n"))


def _extract_user_text(user: Any) -> str:
    """Pull plain text out of a turn's `user` field.

    Vision-capable models accept multi-part `user` content (a list of
    `{type: image_url|text, ...}` dicts). For search purposes we only
    care about the text parts; image URLs are skipped. A legacy saved-
    attachment block is left out: previews and titles show the typed text.
    """
    from sharedagents_core.attachment_context import display_user

    user, _files = display_user(user)
    if isinstance(user, str):
        return user
    if isinstance(user, list):
        parts: list[str] = []
        for part in user:
            if isinstance(part, dict):
                kind = str(part.get("type") or "").lower()
                if kind == "text":
                    txt = part.get("text") or ""
                    if txt:
                        parts.append(str(txt))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    if user is None:
        return ""
    return str(user)


def _combine_turn(user: Any, assistant: str) -> str:
    """Combine user + assistant text into a single FTS document.

    A leading separator + role tag would help downstream consumers
    that want to slice the snippet, but FTS5's `snippet()` works
    over the raw string — keep the markup minimal.
    """
    user_text = _extract_user_text(user).strip()
    assistant_text = (assistant or "").strip()
    if user_text and assistant_text:
        return f"{user_text}\n---\n{assistant_text}"
    return user_text or assistant_text


def _derive_title(user: Any, limit: int | None = None) -> str:
    """First non-empty line of the user prompt, capped at
    ``SESSION_TITLE_MAX_CHARS``. Delegates to ``Session._derive_title``
    so a title derived by the projector matches what the in-memory
    dataclass would have produced."""
    from sharedagents_session.session import SESSION_TITLE_MAX_CHARS, Session

    if limit is None:
        limit = SESSION_TITLE_MAX_CHARS
    return Session._derive_title(user, limit=limit)


# Header fields stored as JSON TEXT blobs — ``header_update`` patches may
# carry Python lists; sqlite needs a string bind.
_JSON_TEXT_HEADER_COLS = frozenset(
    {
        "pinned_facts",
        "goal_queue",
        "compaction_last_stats",
    }
)


def _coerce_header_value(name: str, v: Any) -> Any:
    if name in _JSON_TEXT_HEADER_COLS and not isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, bool):
        return 1 if v else 0
    return v


# ---- structured event extraction ------------------------------------


# Recognised event kinds. Stored as TEXT in turn_events.kind so
# downstream queries can JOIN / GROUP BY without a separate enum.
EVENT_KINDS = (
    "loop",  # loop detector fired (metrics.loop_detected==True
    # OR finish_reason=='tool_args_loop')
    "nudge",  # silent-nudge after premature end / loop
    "distill",  # partial-reasoning recovery
    "compaction",  # mid-turn compaction
    "pause_resume",  # operator-driven pause/resume cycle
    "error",  # turn-level error with classification
    "finish",  # finish_reason (always emitted once per turn)
    "tool_call",  # one row per tool dispatched
    "tool_result_fail",  # one row per tool_result with ok=False
)


def _extract_turn_events(
    session_hash: str,
    turn_idx: int,
    turn: dict,
    default_ts_ms: int,
) -> list[tuple[str, str]]:
    """Fan one turn dict into `(kind, payload_json)` tuples.

    Empty result is fine — many turns have nothing to index here.
    Each emitted payload is a small JSON document with the
    actionable fields for that kind; the FULL turn record stays in
    the on-disk JSON for callers who need the surrounding context.

    Self-improvement use case (fine-tune / eval / regression):
    queryers JOIN ``turn_events`` to ``sessions_meta`` + ``turns``
    to get session metadata + conversational text alongside the
    structured event payload.
    """
    out: list[tuple[str, str]] = []
    if not isinstance(turn, dict):
        return out
    metrics = turn.get("metrics") or {}
    if not isinstance(metrics, dict):
        metrics = {}

    # loop — metrics.loop_detected (flattened) OR a literal
    # tool_args_loop finish_reason OR token_loop_suspected heuristic.
    finish_reason = str(metrics.get("finish_reason") or turn.get("finish_reason") or "")
    if (
        metrics.get("loop_detected")
        or metrics.get("loop_event")
        or metrics.get("token_loop_suspected")
        or finish_reason == "tool_args_loop"
    ):
        payload = {
            "tier": metrics.get("loop_tier") or "",
            "evidence": metrics.get("loop_evidence") or "",
            "channel": metrics.get("loop_channel") or "",
            "tool_name": metrics.get("loop_tool_name") or "",
            "args_chars": metrics.get("loop_args_chars"),
            "checked_at": metrics.get("loop_checked_at"),
            "finish_reason": finish_reason,
            "token_loop_suspected": bool(
                metrics.get("token_loop_suspected") or False,
            ),
            # Preserve the raw nested dict too when present — it's
            # the legacy shape from pre-PR-#242 records.
            "raw": metrics.get("loop_event"),
        }
        out.append(("loop", json.dumps(payload, ensure_ascii=False)))

    # finish — emitted whenever the turn terminated (real
    # finish_reason OR explicit aborted flag), so forensics
    # queries like "how many turns aborted before finishing" can
    # count without scanning every JSON. ``in_progress`` is NOT
    # part of this predicate: ``index_session`` skips in_progress
    # turns up-front (so live-flush checkpoints don't churn the
    # index against the same row N times per turn), so we'd never
    # reach this branch with ``in_progress=True`` from production.
    aborted = bool(turn.get("aborted"))
    if finish_reason or aborted:
        out.append(
            (
                "finish",
                json.dumps(
                    {
                        "finish_reason": finish_reason,
                        "aborted": aborted,
                    },
                    ensure_ascii=False,
                ),
            )
        )

    # error — turn-level error + classification (engine_exception,
    # rate_limit, needs_memory_flush, etc.).
    err = turn.get("error")
    err_class = turn.get("error_classification") or ""
    if err or err_class:
        out.append(
            (
                "error",
                json.dumps(
                    {
                        "message": str(err or "")[:1000],
                        "classification": err_class,
                        "finish_reason": finish_reason,
                    },
                    ensure_ascii=False,
                ),
            )
        )

    # tool_call — one row per dispatched tool. Cap the serialised
    # arguments payload at ``_TOOL_ARGS_CAP`` chars so a single
    # multi-megabyte pasted blob (long SQL / base64 / file body)
    # can't balloon sessions.db at "thousands of sessions" scale.
    # When the truncation fires, store a marker + the original
    # length so the row still tells the trainer "this turn passed
    # a giant payload" without keeping it verbatim. Match the
    # 1000-char policy on error.message / tool_result_fail.content
    # so the truncation budget is consistent.
    #
    # NOTE: this column is an INDEX, not a content store. For full
    # fidelity (fine-tune SFT pipelines, etc.) read the on-disk
    # session JSON — ``turn_events`` only carries enough to drive
    # SELECTs that pinpoint which turn to load.
    _TOOL_ARGS_CAP = 1000
    for call in turn.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        raw_args = call.get("arguments") or {}
        try:
            args_dumped = json.dumps(raw_args, ensure_ascii=False)
            args_unserializable = False
        except (TypeError, ValueError):
            # ``arguments`` carried a non-JSON-serialisable value
            # (an engine bug — the wire shape should always be a
            # dict-of-primitives, but defensive code keeps the
            # whole turn from disappearing from the index). Anchor
            # the outer dump on the stringified repr instead.
            args_dumped = str(raw_args)
            args_unserializable = True
        if args_unserializable or len(args_dumped) > _TOOL_ARGS_CAP:
            args_field: Any = {
                "_truncated": True,
                "_full_chars": len(args_dumped),
                "preview": args_dumped[:_TOOL_ARGS_CAP],
            }
            if args_unserializable:
                args_field["_unserializable"] = True
        else:
            args_field = raw_args
        out.append(
            (
                "tool_call",
                json.dumps(
                    {
                        "call_id": str(call.get("call_id") or ""),
                        "name": str(call.get("name") or ""),
                        "arguments": args_field,
                        "started_at": str(call.get("started_at") or ""),
                    },
                    ensure_ascii=False,
                ),
            )
        )

    # tool_result_fail — only failures (ok=False). Successes are
    # already implied by the matching tool_call row. Note that
    # tool_results carry only metadata after session_log.record_turn
    # strips the content body (fix_20260519 round-4) — the
    # ``content_chars`` size hint is enough for an indexer; the
    # actual body lives in gateway.log if a debugger needs it.
    for result in turn.get("tool_results") or []:
        if not isinstance(result, dict):
            continue
        if result.get("ok"):
            continue
        # Tolerate both the legacy shape (with ``content``) AND the
        # post-stripping shape (with ``content_chars``).
        legacy_content = result.get("content")
        out.append(
            (
                "tool_result_fail",
                json.dumps(
                    {
                        "call_id": str(result.get("call_id") or ""),
                        "content_chars": int(
                            result.get("content_chars")
                            if result.get("content_chars") is not None
                            else (len(legacy_content) if isinstance(legacy_content, str) else 0)
                        ),
                        "ended_at": str(result.get("ended_at") or ""),
                    },
                    ensure_ascii=False,
                ),
            )
        )

    # Forward-compat: the four lifecycle audit lists on TurnRecord
    # (nudge_events / distill_events / compaction_events /
    # pause_resume_log). Today's engine doesn't write them; once it
    # does, the index picks them up automatically with no schema
    # bump. Empty lists produce zero rows.
    _LIFECYCLE_KINDS = (
        ("nudge_events", "nudge"),
        ("distill_events", "distill"),
        ("compaction_events", "compaction"),
        ("pause_resume_log", "pause_resume"),
    )
    for field_name, kind in _LIFECYCLE_KINDS:
        for entry in turn.get(field_name) or []:
            if not isinstance(entry, dict):
                # Tolerate engines that wrote a bare string — wrap
                # it so the payload column is always valid JSON.
                entry = {"value": str(entry)}
            out.append((kind, json.dumps(entry, ensure_ascii=False)))

    return out


# ---- main class -----------------------------------------------------


class SessionIndex:
    """One `sessions.db` per agent home.

    Three-table layout:
      - `sessions_meta` — one row per session (title, created_at,
        frontend, updated_at).
      - `turns` — one row per non-empty turn (combined_text, ts).
        rowid is autoincrement; FK references `sessions_meta`. The
        `title` column is populated ONLY on the turn-0 row (the
        session title, mirrored from `sessions_meta.title`) so a
        title-only word is searchable without repeating the title in
        every turn row.
      - `turns_fts` — FTS5 virtual table over `combined_text` +
        `title`. Kept in sync manually (FTS5 triggers are fragile
        across INSERT-OR-REPLACE rowid churn — same reason
        `MemoryIndex` avoids them).

    Thread safety: NOT shared across threads. Open a fresh instance
    per call.
    """

    def __init__(self, sessions_dir: Path) -> None:
        """Open / migrate the unified ``sessions.db`` at ``sessions_dir``.

        ``sessions_dir`` is the ONE shared sessions folder (typically
        ``sharedagents_core.layout.sessions_root()``) — all agents write
        their per-session JSONL files into it and read/write rows in
        the single ``sessions.db`` it contains.

        ``self.agent_home`` is kept as a back-compat alias pointing at
        the parent dir so legacy callers that still spell the runtime
        root that way keep working (memory index initialisation,
        session_log save helpers).
        """
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.sessions_dir / "sessions.db"
        # Back-compat alias for callers that still spell the runtime
        # root ``agent_home`` (they're not strictly agent-specific
        # anymore but the name is too widely used to rename in one PR).
        self.agent_home = self.sessions_dir.parent
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return connect_sessions_db(self.db_path)

    def _init_db(self) -> None:
        ensure_schema(self.db_path)

    # -- write path (tail JSONL into the index) --------------------

    def tail(self, sid: str, *, projector: EventProjector | None = None) -> int:
        """Project new ``<sid>.jsonl`` events into the db, from the
        session's ``indexed_bytes`` cursor forward.

        Idempotent and crash-safe: each event advances ``indexed_bytes``
        in the same transaction, and the per-event ``turn_events`` writes
        use INSERT-OR-IGNORE on a dedup UNIQUE index. Resets the cursor
        to 0 when it has run past EOF (operator truncated the JSONL).

        This is the fallback / reconcile path. In the steady state the
        ``SessionWriter`` projects each event in-line as it appends, so
        ``tail`` usually finds nothing new. Returns the count applied.
        """
        from sharedagents_session.session_reader import tail_jsonl

        sid = str(sid)
        path = self.sessions_dir / f"{sid}.jsonl"
        if not path.exists():
            return 0
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        con = self._connect()
        try:
            proj = projector or EventProjector()
            applied = 0
            # Read the cursor INSIDE the write transaction so two
            # concurrent tail()s can't both fold from the same stale
            # offset and double-apply the live-column read-modify-append
            # (TOCTOU on indexed_bytes). The
            # second tailer blocks on BEGIN IMMEDIATE, then sees the
            # advanced cursor and folds nothing.
            con.execute("BEGIN IMMEDIATE")
            try:
                # Guarantee a cursor row. ``apply()`` advances ``indexed_bytes``
                # with an UPDATE, which is a SILENT no-op when the row does not
                # exist — so a JSONL with no ``sessions_meta`` row re-decoded
                # from offset 0 on EVERY ``refresh()``, forever (measured:
                # 14 such files in the live store, ~60ms per
                # refresh). ``rebuild()`` seeds the identical row.
                con.execute(
                    "INSERT OR IGNORE INTO sessions_meta(session_hash, indexed_bytes) VALUES (?, 0)",
                    (sid,),
                )
                row = con.execute(
                    "SELECT indexed_bytes FROM sessions_meta WHERE session_hash=?",
                    (sid,),
                ).fetchone()
                start = int(row["indexed_bytes"]) if row is not None else 0
                if start > size:
                    # JSONL truncated/rewritten under us — re-fold from
                    # the top AND purge the rows projected from the OLD
                    # file content. Without the purge, error rows from a
                    # previous run of the same session_hash lingered as
                    # orphans (turn_events is INSERT-only), so the DB
                    # over-counted errors for re-run sessions (one carried 3 orphan
                    # tool_result_fail + 1 orphan error).
                    start = 0
                    con.execute(
                        "DELETE FROM turn_events WHERE session_hash=?",
                        (sid,),
                    )
                    # By-rowid via the indexed ``turns`` — MUST run before the
                    # ``turns`` rows are deleted (the subquery reads them).
                    con.execute(
                        "DELETE FROM turns_fts WHERE rowid IN (SELECT rowid FROM turns WHERE session_hash=?)",
                        (sid,),
                    )
                    con.execute(
                        "DELETE FROM turns WHERE session_hash = ?",
                        (sid,),
                    )
                end = start
                for event, end_off in tail_jsonl(path, start):
                    proj.apply(con, sid, event, end_off)
                    applied += 1
                    end = end_off
                # ``apply()`` only advances the cursor on a good event, so a
                # file ENDING in complete corrupt lines was re-read (and
                # re-logged) on every refresh. Bytes below ``size`` existed
                # before the scan, so any LF-terminated line there that was
                # not yielded is corrupt: step past it. A torn last line
                # (no LF yet) stays unconsumed for the writer to finish.
                if end < size:
                    with open(path, "rb") as fp:
                        fp.seek(end)
                        nl = fp.read(size - end).rfind(b"\n")
                    if nl >= 0:
                        con.execute(
                            "UPDATE sessions_meta SET indexed_bytes=? WHERE session_hash=?",
                            (end + nl + 1, sid),
                        )
                con.commit()
            except BaseException:
                con.rollback()
                raise
            return applied
        finally:
            con.close()

    def refresh(self) -> int:
        """Tail every session whose JSONL has grown past its indexed
        cursor — the read-side lazy-index trigger.

        The writer appends to the JSONL only (no realtime SQL); this
        brings ``sessions.db`` current before a list/search read. Cheap:
        one ``stat`` + one indexed_bytes lookup per session, and a real
        tail only for the ones that actually grew. Returns the number of
        sessions refreshed.
        """
        if not self.sessions_dir.is_dir():
            return 0
        con = self._connect()
        try:
            cur = {
                row["session_hash"]: int(row["indexed_bytes"])
                for row in con.execute("SELECT session_hash, indexed_bytes FROM sessions_meta")
            }
        except Exception:  # noqa: BLE001
            cur = {}
        finally:
            con.close()
        refreshed = 0
        for entry in self.sessions_dir.iterdir():
            if not entry.is_file() or entry.suffix != ".jsonl":
                continue
            sid = entry.stem
            try:
                size = entry.stat().st_size
            except OSError:
                continue
            # Tail on GROWTH (new events) and on SHRINK (the JSONL was
            # truncated/rewritten under the same session_hash) — the
            # shrink case is what routes into tail()'s purge-and-refold
            # branch; gating on growth alone left stale rows orphaned
            # until an explicit tail()/rebuild. ``size == cursor`` stays a no-op.
            if size != cur.get(sid, 0):
                try:
                    self.tail(sid)
                    refreshed += 1
                except Exception:  # noqa: BLE001
                    pass
        return refreshed

    def rebuild(self) -> dict:
        """Recreate the index for every ``<sid>.jsonl`` from offset 0.

        Power-loss recovery: ``sessions.db`` is a derived cache; the
        JSONL files are the source of truth. Wipes the materialised
        rows for each session, resets its cursor, and re-tails.
        """
        stats = {"files": 0, "events": 0, "errors": 0}
        if not self.sessions_dir.is_dir():
            return stats
        for entry in sorted(self.sessions_dir.iterdir()):
            if not entry.is_file() or entry.suffix != ".jsonl":
                continue
            sid = entry.stem
            stats["files"] += 1
            con = self._connect()
            try:
                # FTS first, by rowid via the indexed ``turns`` — before the
                # ``turns`` rows it references are deleted below.
                con.execute(
                    "DELETE FROM turns_fts WHERE rowid IN (SELECT rowid FROM turns WHERE session_hash=?)",
                    (sid,),
                )
                for tbl in ("turn_events", "turns", "sessions_meta"):
                    con.execute(f"DELETE FROM {tbl} WHERE session_hash=?", (sid,))
                con.execute(
                    "INSERT OR IGNORE INTO sessions_meta(session_hash, indexed_bytes) VALUES (?, 0)",
                    (sid,),
                )
                con.commit()
            finally:
                con.close()
            try:
                stats["events"] += self.tail(sid)
            except Exception:  # noqa: BLE001
                stats["errors"] += 1
        return stats

    def backfill_parent_session_ids(self, limit: int | None = None) -> dict:
        """Repair ``sessions_meta.parent_session_id`` from the JSONL headers.

        Repairs rows blanked by the legacy ``rebuild()`` bug (the
        bare-row INSERT defeated the ``session_started`` upsert, so every
        subagent's parent back-link was dropped). The JSONL ``session_started``
        header is the source of truth and still carries the real parent — we
        read it and fill any row whose stored parent is empty.

        Idempotent and cheap: reads only the header line of each ``<sid>.jsonl``
        and touches ONLY rows where ``parent_session_id`` is currently empty, so
        it never re-runs work or clobbers a good link. Returns counts.

        ``limit`` — when set, only the most-recently-modified ``limit`` session
        files are scanned. Used by orchestrator startup to self-heal the recent
        delegation tree without paying a full-corpus scan on every boot.
        """
        from sharedagents_session.session_reader import tail_jsonl

        stats = {"scanned": 0, "filled": 0}
        if not self.sessions_dir.is_dir():
            return stats
        entries = [e for e in self.sessions_dir.iterdir() if e.is_file() and e.suffix == ".jsonl"]
        if limit is not None:
            # Most recent first so a bounded startup pass heals what matters.
            entries.sort(key=lambda e: e.stat().st_mtime, reverse=True)
            entries = entries[: max(0, int(limit))]
        else:
            entries.sort()
        for entry in entries:
            sid = entry.stem
            stats["scanned"] += 1
            parent = ""
            try:
                # The session_started header is the first event; stop there.
                for event, _end in tail_jsonl(entry, 0):
                    if str(event.get("type") or "") in ("session_started", "session_meta"):
                        parent = str(event.get("parent_session_id") or "")
                        break
            except Exception:  # noqa: BLE001
                continue
            if not parent:
                continue
            con = self._connect()
            try:
                cur = con.execute(
                    "UPDATE sessions_meta SET parent_session_id=? "
                    "WHERE session_hash=? AND "
                    "(parent_session_id IS NULL OR parent_session_id='')",
                    (parent, sid),
                )
                con.commit()
                if cur.rowcount:
                    stats["filled"] += int(cur.rowcount)
            finally:
                con.close()
        return stats

    def index_session_file(self, path: Path) -> int:
        """Back-compat shim: tail the JSONL for ``path``'s session.

        The legacy whole-file JSON re-index is gone (is append-only
        JSONL). Accepts a ``<sid>.jsonl`` path and tails it.
        """
        sid = Path(path).stem
        try:
            return self.tail(sid)
        except Exception:  # noqa: BLE001
            return 0

    def reindex_all(self) -> dict:
        """Back-compat alias for :meth:`rebuild`."""
        r = self.rebuild()
        return {"files": r["files"], "indexed_turns": r["events"], "errors": r["errors"]}

    def remove_session(self, session_hash: str) -> None:
        """Drop one session's rows. Used when a session is deleted."""
        if not session_hash:
            return
        con = self._connect()
        try:
            # FTS by rowid via the indexed ``turns`` — before ``turns`` rows go.
            con.execute(
                "DELETE FROM turns_fts WHERE rowid IN (SELECT rowid FROM turns WHERE session_hash=?)",
                (session_hash,),
            )
            con.execute("DELETE FROM turns WHERE session_hash = ?", (session_hash,))
            con.execute("DELETE FROM turn_events WHERE session_hash = ?", (session_hash,))
            con.execute("DELETE FROM sessions_meta WHERE session_hash = ?", (session_hash,))
            con.commit()
        finally:
            con.close()

    # -- read path -----------------------------------------------------

    def search(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_MAX_RESULTS,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> list[RecallResult]:
        """FTS5 search across saved session turns.

        Returns `RecallResult`s with `source="session"` so callers
        can rank session hits alongside other recall sources.

        Refreshes the index before reading so JSONL-only writes —
        the write side that hasn't yet been folded by the bus
        subscriber, or written by a process whose subscriber lagged
        — are visible to the search.
        Cheap when the cursor is already at EOF for every session.
        """
        query = (query or "").strip()
        if not query:
            return []
        match = build_fts_query(query)
        if match is None:
            return []
        try:
            self.refresh()
        except Exception:  # noqa: BLE001
            # Best-effort: if refresh dies (locked DB, transient
            # filesystem error), we'd still rather return the index's
            # current view than blow up the search call.
            pass
        con = self._connect()
        try:
            rows = con.execute(
                "SELECT f.session_hash AS session_hash, "
                "       f.turn_idx     AS turn_idx, "
                "       snippet(turns_fts, 0, '', '', '…', 64) AS snippet, "
                "       bm25(turns_fts) AS rank, "
                "       m.title        AS title, "
                "       m.created_at   AS created_at, "
                "       m.workspace_origin AS frontend "
                "FROM turns_fts AS f "
                "LEFT JOIN sessions_meta AS m "
                "  ON m.session_hash = f.session_hash "
                "WHERE turns_fts MATCH ? "
                "ORDER BY rank LIMIT ?",
                (match, max(1, int(top_k)) * 4),
            ).fetchall()
        finally:
            con.close()

        out: list[RecallResult] = []
        for r in rows:
            score = bm25_rank_to_score(float(r["rank"]))
            if score < min_score:
                continue
            session_hash = r["session_hash"]
            title = r["title"] or f"untitled-{(session_hash or '')[-8:]}"
            created_at = str(r["created_at"] or "")
            date_part = created_at.split("T", 1)[0] if created_at else ""
            out.append(
                RecallResult(
                    source="session",
                    path_or_id=session_hash,
                    title=title,
                    snippet=(r["snippet"] or "")[:DEFAULT_SNIPPET_CHARS],
                    score=score,
                    metadata={
                        "turn_idx": int(r["turn_idx"]),
                        "date": date_part,
                        "frontend": r["frontend"] or "",
                    },
                )
            )
        # Dedup on (session_hash, turn_idx) — defensive; with the new
        # combined-text schema each turn is one row and won't appear
        # twice, but keep the guard for forward-compatibility with any
        # future role-split index.
        by_key: dict[tuple[str, int], RecallResult] = {}
        for h in out:
            key = (h.path_or_id, int(h.metadata.get("turn_idx", 0)))
            existing = by_key.get(key)
            if existing is None or h.score > existing.score:
                by_key[key] = h
        deduped = sorted(by_key.values(), key=lambda x: x.score, reverse=True)
        return deduped[: max(1, int(top_k))]

    def stats(self) -> dict:
        con = self._connect()
        try:
            sess_n = con.execute("SELECT count(*) AS n FROM sessions_meta").fetchone()["n"]
            turn_n = con.execute("SELECT count(*) AS n FROM turns").fetchone()["n"]
            fts_n = con.execute("SELECT count(*) AS n FROM turns_fts").fetchone()["n"]
        finally:
            con.close()
        try:
            db_size = self.db_path.stat().st_size
        except OSError:
            db_size = 0
        return {
            "db_path": str(self.db_path),
            "db_size_bytes": db_size,
            "sessions": sess_n,
            "turns": turn_n,
            "turn_rows": fts_n,  # back-compat alias
        }

    def clear(self) -> None:
        """Drop all data — for `sharedagents session index --force` (future)."""
        con = self._connect()
        try:
            con.executescript(
                "DELETE FROM turns_fts;\nDELETE FROM turns;\nDELETE FROM turn_events;\nDELETE FROM sessions_meta;"
            )
            con.commit()
        finally:
            con.close()


def _iso_to_ms(ts: Any) -> int:
    """Best-effort ISO-8601 → epoch milliseconds."""
    if not ts:
        return 0
    if isinstance(ts, (int, float)):
        return int(ts)
    if not isinstance(ts, str):
        return 0
    try:
        from datetime import datetime

        # Strip any trailing 'Z' for fromisoformat (3.10 doesn't
        # accept it); 3.11+ does.
        cleaned = ts.replace("Z", "+00:00")
        return int(datetime.fromisoformat(cleaned).timestamp() * 1000)
    except (ValueError, TypeError):
        return 0


# ---- public tool entry --------------------------------------------


def session_search(
    query: str,
    *,
    top_k: int = DEFAULT_MAX_RESULTS,
    min_score: float = DEFAULT_MIN_SCORE,
    workspace_origin: str = "",
    agent_home: Path | None = None,
    chat_id: str = "",
) -> str:
    """FTS-based search over saved sessions. Returns a markdown blob.

    Saved transcripts are a shared diagnostic/recall surface for this
    single-owner runtime. Native agents may search every agent's sessions.
    The identity arguments remain for wire compatibility with older callers.
    """
    _ = (workspace_origin, agent_home, chat_id)
    if not isinstance(query, str) or not query.strip():
        return "session_search failed: query required."
    try:
        top_k = max(1, min(20, int(top_k)))
    except (TypeError, ValueError):
        top_k = DEFAULT_MAX_RESULTS
    try:
        min_score = max(0.0, min(1.0, float(min_score)))
    except (TypeError, ValueError):
        min_score = DEFAULT_MIN_SCORE
    # ``home`` here is the agent home (per-agent config dir). The
    # unified sessions DB lives under ``layout.sessions_root()`` for
    # ALL agents, so resolve via layout — the legacy ``home/sessions``
    # path no longer holds session data.
    from sharedagents_core.layout import sessions_root as _sroot

    idx = SessionIndex(_sroot())
    hits = idx.search(query, top_k=top_k, min_score=min_score)
    if not hits:
        return f"session_search: no matches for {query!r} (min_score={min_score})."
    return format_recall_hits(
        hits,
        header=f"# session_search: {query!r}\n_{len(hits)} hit(s)_",
    )


def _resolve_agent_home() -> Path | None:
    """Same resolution as `memory_tools._resolve_agent_home`."""
    explicit = os.environ.get("SHAREDAGENTS_AGENT_HOME")
    if explicit:
        return Path(explicit).expanduser()
    sdir = os.environ.get("SHAREDAGENTS_SESSIONS_DIR")
    if sdir:
        p = Path(sdir).expanduser()
        if p.name == "sessions":
            return p.parent
    return None


# ---- event-bus subscriber ------------------------------------------

# Per-agent registration state (keyed by agent home path) so
# `subscribe_for_sessions()` is idempotent across multiple calls.
_REGISTERED: set[str] = set()


def subscribe_for_sessions(home: Path, *, quiet_seconds: float = 0.5) -> bool:
    """Wire `session_saved` → `SessionIndex.tail` on the process bus.

    All agents now share ONE ``sessions.db`` under
    ``layout.sessions_root()``, so the handler indexes EVERY
    ``session_saved`` event regardless of which agent fired it —
    cross-contamination is impossible because there's only one DB.
    (A per-agent ``home``-equality filter would drop every event: the
    writer publishes ``sessions_dir.parent`` while heartbeats subscribe
    with ``cfg.home == agents_root() / <name>``, so the two never match
    and the live indexer silently freezes.)

    Debounce is short (0.5s default) — saves are cheap to index and
    we want recent turns searchable as quickly as possible. The bus's
    late-publish-merging serializes per-subscriber so two saves of
    the same session can't run the indexer concurrently.

    First-subscribe also runs a one-shot `refresh()` (incremental —
    folds any session whose JSONL grew past / was written since its
    indexed cursor) so sessions written by other processes get
    backfilled on the next heartbeat tick, WITHOUT re-folding the whole
    corpus from offset 0 (`reindex_all()`), which froze boot at scale.

    The ``home`` argument is now only used for idempotency keying +
    subscriber-naming (so two callers from the same agent home
    de-dupe) — it does NOT scope the events the handler sees.

    Returns True iff newly subscribed.
    """
    key = str(Path(home).resolve())
    if key in _REGISTERED:
        return False

    async def handler(payload: dict[str, Any]) -> None:
        # payload carries ``session_hash`` directly; tolerate the
        # legacy ``path`` (a ``<sid>.jsonl`` path) too. No per-agent
        # filter — see docstring.
        sid = (payload or {}).get("session_hash", "")
        if not sid:
            path_str = (payload or {}).get("path", "")
            if path_str:
                sid = Path(path_str).stem
        if not sid:
            return
        try:
            from sharedagents_core.layout import sessions_root as _sroot

            idx = SessionIndex(_sroot())
            n = idx.tail(str(sid))
            _log.debug("session_index: tailed %d event(s) for %s", n, sid)
        except Exception:  # noqa: BLE001
            _log.exception("session_index handler crashed for %s", sid)

    from sharedagents_core.event_bus import get_bus

    bus = get_bus()
    bus.subscribe(
        "session_saved",
        handler,
        quiet_seconds=quiet_seconds,
        name=f"session_index[{Path(home).name}]",
        # Coalesce bursts per session_hash so concurrent writes from
        # N agents don't collapse into the LATEST session only.
        coalesce_key="session_hash",
    )
    _REGISTERED.add(key)
    # First-subscribe backfill — INCREMENTAL (``refresh``), not a full
    # ``reindex_all``. refresh() folds every session whose JSONL has grown
    # past (or was written entirely since) its indexed cursor, which is
    # exactly the backfill intent: pick up sessions this process's bus
    # subscriber never saw (a session missing from the index has cursor 0 <
    # size → folded from the top). reindex_all() re-folds ALL sessions from
    # offset 0 every boot — O(all events); at 3k sessions that froze the
    # orchestrator's event loop for ~70s (this runs synchronously inside the
    # heartbeat tick), blocking the :8099 bind. refresh() only touches what
    # actually changed (a full rebuild here once caused 2-minute restarts).
    try:
        from sharedagents_core.layout import sessions_root as _sroot

        SessionIndex(_sroot()).refresh()
    except Exception:  # noqa: BLE001
        _log.exception("session_index: backfill error for %s", home)
    return True


# Back-compat alias — earlier code (heartbeat.py) imports the old
# name. Keeping both spellings avoids touching unrelated wiring.
subscribe_for_agent = subscribe_for_sessions


# ---- DDL ------------------------------------------------------------

# v6: add a `title` column to `turns` and an indexed `title` column to
# `turns_fts` so a session's title words are FTS-searchable even when a
# title word never appears in any turn's text. The title is carried ONLY
# on the turn-0 row (empty on every other row) so BM25 is not diluted by
# repeating the title once per turn. Existing DBs migrate additively via
# ALTER (+ backfill from sessions_meta.title) in `_apply_migrations`.
SCHEMA_USER_VERSION = 6

# Full sessions_meta column set. Used both to CREATE a fresh table
# and to drive the additive ALTER-TABLE migration on existing DBs
# (``_apply_migrations``). Order: (name, sql_type_with_default).
_SESSIONS_META_COLUMNS: list[tuple[str, str]] = [
    # identity (immutable after session_started)
    ("session_hash", "TEXT PRIMARY KEY"),
    ("created_at", "TEXT NOT NULL DEFAULT ''"),
    ("agent_name", "TEXT NOT NULL DEFAULT ''"),
    ("footer_state_json", "TEXT NOT NULL DEFAULT '{}'"),
    ("engine", "TEXT NOT NULL DEFAULT ''"),
    # Parent sharedagents session id when this session is a subagent (an
    # agent spawned by another session via Agent / BatchAgent /
    # delegate_*). Empty for top-level sessions. Folded from the
    # ``session_started`` header; lets queries walk the parent↔child
    # subagent tree without parsing the loose ``delegate_*`` sidecars.
    ("parent_session_id", "TEXT NOT NULL DEFAULT ''"),
    # The engine's OWN CLI session/resume token (claude/codex/antigravity
    # session id) for CLI-engine sessions — the sharedagents↔engine session
    # link. Folded from a `header_update` at TurnDone; empty for
    # local-LLM sessions. Mutable (set once the first CLI turn lands).
    ("engine_session_id", "TEXT NOT NULL DEFAULT ''"),
    ("workspace_cwd", "TEXT NOT NULL DEFAULT ''"),
    ("workspace_origin", "TEXT NOT NULL DEFAULT ''"),  # was `frontend`
    ("schema_version", "TEXT NOT NULL DEFAULT '1'"),
    # mutable session-config
    ("title", "TEXT NOT NULL DEFAULT ''"),
    # Count of LLM-based title refinements already applied. Titles now
    # lock after the first-turn generation; this column stays for older
    # sessions that spent budget under the retired 4/8/16 schedule.
    ("title_refinements", "INTEGER NOT NULL DEFAULT 0"),
    # Pinned title flag: True suppresses every title-agent trigger for this
    # session (operator /title rename or Tasks-board pinned task name).
    # Additive column — older DBs migrate via ALTER.
    ("title_locked", "INTEGER NOT NULL DEFAULT 0"),
    ("model", "TEXT NOT NULL DEFAULT ''"),
    # The REAL model an external CLI/ACP engine reported running
    # (claude-opus-4-…/gpt-5.5-codex/composer-2.5/antigravity's concrete
    # id). Folded from a `header_update` at TurnDone; empty for local-LLM
    # sessions. Kept SEPARATE from ``model`` so engines with a user-facing
    # selector dimension (antigravity's auto/pro/flash alias lives in
    # ``model``) can surface the concrete model WITHOUT clobbering the
    # picker. Every channel prefers this when set.
    ("engine_model", "TEXT NOT NULL DEFAULT ''"),
    ("endpoint", "TEXT NOT NULL DEFAULT ''"),
    ("backend", "TEXT NOT NULL DEFAULT ''"),
    ("context_window", "INTEGER NOT NULL DEFAULT 0"),
    ("system_mode", "TEXT NOT NULL DEFAULT ''"),
    ("reasoning_visibility", "TEXT NOT NULL DEFAULT 'off'"),
    # Per-session reasoning depth / model-sheet controls. These are additive
    # header fields so older JSONL sessions remain loadable with empty values.
    ("effort", "TEXT NOT NULL DEFAULT ''"),
    ("speed", "TEXT NOT NULL DEFAULT ''"),
    ("tool_output_visible", "INTEGER NOT NULL DEFAULT 0"),
    ("persist_reasoning_traces", "INTEGER NOT NULL DEFAULT 1"),
    ("persist_tool_result_bodies", "INTEGER NOT NULL DEFAULT 1"),
    # system prompt
    ("system_prompt", "TEXT NOT NULL DEFAULT ''"),
    ("system_prompt_sha256", "TEXT NOT NULL DEFAULT ''"),
    # mutable session-state
    ("pinned_facts", "TEXT NOT NULL DEFAULT '[]'"),
    ("compaction_count", "INTEGER NOT NULL DEFAULT 0"),
    ("compaction_summary", "TEXT NOT NULL DEFAULT ''"),
    # Stats of the most recent manual compaction (JSON dict — the WebUI
    # collapsed widget's source of truth; see Session.compaction_last_stats).
    # Additive column — older DBs migrate via ALTER.
    ("compaction_last_stats", "TEXT NOT NULL DEFAULT '{}'"),
    # OpenCode-style PR blurb from the hidden summary agent (distinct from
    # compaction_summary). Additive column — older DBs migrate via ALTER.
    ("session_summary", "TEXT NOT NULL DEFAULT ''"),
    ("memory_flush_compaction_count", "INTEGER NOT NULL DEFAULT -1"),
    ("last_memory_flush_at", "TEXT NOT NULL DEFAULT ''"),
    # Most recent authoritative prompt_tokens — persisted so the compaction
    # gate survives a session reload (see Session.last_prompt_tokens).
    ("last_prompt_tokens", "INTEGER NOT NULL DEFAULT 0"),
    ("last_turn_idx", "INTEGER NOT NULL DEFAULT -1"),
    ("last_turn_ts", "INTEGER NOT NULL DEFAULT 0"),
    # live streaming snapshot (NULL/empty between turns)
    ("live_turn_idx", "INTEGER"),
    ("live_round_idx", "INTEGER"),
    ("live_started_at", "TEXT"),
    ("live_content_so_far", "TEXT NOT NULL DEFAULT ''"),
    ("live_thinking_so_far", "TEXT NOT NULL DEFAULT ''"),
    ("live_tool_calls_so_far", "TEXT NOT NULL DEFAULT '[]'"),
    ("live_last_event_offset", "INTEGER NOT NULL DEFAULT 0"),
    # discrete session state (the durable state-machine register the
    # orchestrator queries to answer "what is every agent doing right
    # now?"). Values:
    #
    #   idle | prompting | prefill | reasoning | streaming | tool_running |
    #   compacting | ask | retrying | paused | waiting | error | closed
    #
    # ``waiting`` = this session's own turn ended, but background Agent
    # children / open async jobs are still running (parent stays live for
    # Stop / monitor until the tree drains).
    #
    # Maintained by :meth:`EventProjector._set_state` as the projector
    # folds JSONL events. See ``orchestrator.list_live_sessions``.
    ("session_state", "TEXT NOT NULL DEFAULT 'idle'"),
    ("session_state_updated_at", "INTEGER NOT NULL DEFAULT 0"),
    # V2 activity projection. ``session_state`` remains the compatibility
    # register; these columns keep own-turn phase, operator attention,
    # lifecycle, and last outcome independent so a failed parent can coexist
    # with live descendants without losing either fact.
    ("session_phase", "TEXT NOT NULL DEFAULT 'idle'"),
    ("session_attention", "TEXT NOT NULL DEFAULT 'none'"),
    ("session_lifecycle", "TEXT NOT NULL DEFAULT 'open'"),
    ("session_last_outcome", "TEXT NOT NULL DEFAULT 'none'"),
    ("session_phase_changed_at", "INTEGER NOT NULL DEFAULT 0"),
    ("session_heartbeat_at", "INTEGER NOT NULL DEFAULT 0"),
    ("session_outcome_at", "INTEGER NOT NULL DEFAULT 0"),
    ("session_state_version", "INTEGER NOT NULL DEFAULT 0"),
    # Goal state for the orchestrator to observe "what is each agent working
    # toward, and is it under teacher supervision?". ``active_goal`` is the
    # current goal (empty = none → teacher dormant), folded from the session
    # header's ``active_goal`` key via ``_apply_header_patch``.
    ("active_goal", "TEXT NOT NULL DEFAULT ''"),
    # Per-session teacher toggle override ("on"/"off"/""), folded from the
    # header's ``teacher_override`` key. Lets the orchestrator see whether an
    # operator forced supervision on/off independent of the goal gate.
    ("teacher_override", "TEXT NOT NULL DEFAULT ''"),
    # Per-session supervising-model override (models[] alias/id, "" = configured
    # teacher), folded from the header's ``teacher_model_override`` key.
    ("teacher_model_override", "TEXT NOT NULL DEFAULT ''"),
    # Goal outer-loop state. ``goal_turn_count`` tracks how many auto-re-drive
    # turns have been consumed (bounded by ``max_goal_turns`` config).
    # ``goal_terminal`` is the last goal cycle's terminal state: "" (none),
    # "met", "blocked", or "paused" (cap exhausted).
    ("goal_turn_count", "INTEGER NOT NULL DEFAULT 0"),
    ("goal_terminal", "TEXT NOT NULL DEFAULT ''"),
    # Durable goal entity projection (source of truth: goals.db).
    ("goal_id", "TEXT NOT NULL DEFAULT ''"),
    # Token-cost attribution for delegated child sessions: the immutable root
    # session + ancestor goal this session's spend rolls up to (folded from the
    # header so a reloaded child carries it). Empty on a normal session.
    ("cost_root_session_id", "TEXT NOT NULL DEFAULT ''"),
    ("cost_ancestor_goal_id", "TEXT NOT NULL DEFAULT ''"),
    # Session-tree USD cap. 0 = use the $25 default at read time.
    ("session_budget_usd", "REAL NOT NULL DEFAULT 0"),
    # Conversation topic (mirrors a Telegram forum-topic name; default
    # 'general'). Folded from the header's ``topic`` key on session_started
    # and every header_update, so the WebUI Topics page can group sessions by
    # topic. ``_apply_migrations`` backfills existing DBs with 'general'.
    ("topic", "TEXT NOT NULL DEFAULT 'general'"),
    # Deprecated (one goal per session; no queue). Column kept so existing DBs
    # match the schema list; always '[]'.
    ("goal_queue", "TEXT NOT NULL DEFAULT '[]'"),
    ("tools_state_json", "TEXT NOT NULL DEFAULT '{}'"),
    # indexer cursor + bookkeeping
    ("indexed_bytes", "INTEGER NOT NULL DEFAULT 0"),
    ("updated_at", "INTEGER NOT NULL DEFAULT 0"),
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions_meta (
{cols}
);
CREATE INDEX IF NOT EXISTS idx_sessions_meta_agent
  ON sessions_meta(agent_name);
CREATE INDEX IF NOT EXISTS idx_sessions_meta_last_turn_ts
  ON sessions_meta(last_turn_ts DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_meta_state
  ON sessions_meta(session_state, last_turn_ts DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_meta_parent
  ON sessions_meta(parent_session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_meta_cost_root
  ON sessions_meta(cost_root_session_id);
-- Sidebar ``session_roots``: parent-empty rows ordered by recency.
CREATE INDEX IF NOT EXISTS idx_sessions_meta_roots_ts
  ON sessions_meta(last_turn_ts DESC)
  WHERE (parent_session_id IS NULL OR parent_session_id = '');

CREATE TABLE IF NOT EXISTS turns (
  rowid         INTEGER PRIMARY KEY AUTOINCREMENT,
  session_hash  TEXT NOT NULL,
  turn_idx      INTEGER NOT NULL,
  combined_text TEXT NOT NULL,
  ts            INTEGER NOT NULL,
  -- Session title, carried ONLY on the turn-0 row (empty on every other
  -- row) so a title-only word is FTS-searchable without diluting BM25 by
  -- repeating the title once per turn. Kept in sync with
  -- sessions_meta.title by the projector (turn_ended / header_update).
  title         TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_hash);
CREATE INDEX IF NOT EXISTS idx_turns_ts      ON turns(ts);
CREATE UNIQUE INDEX IF NOT EXISTS idx_turns_session_turn
  ON turns(session_hash, turn_idx);

CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
  combined_text,
  title,
  session_hash UNINDEXED,
  turn_idx     UNINDEXED,
  tokenize='unicode61'
);

CREATE TABLE IF NOT EXISTS session_turns (
  session_hash     TEXT NOT NULL,
  turn_idx         INTEGER NOT NULL,
  start_row_idx    INTEGER NOT NULL,
  end_row_idx      INTEGER NOT NULL,
  agent_name       TEXT NOT NULL DEFAULT '',
  user_prompt      TEXT NOT NULL DEFAULT '',
  assistant_reply  TEXT NOT NULL DEFAULT '',
  status           TEXT NOT NULL DEFAULT 'completed',
  created_at       TEXT NOT NULL DEFAULT '',
  completed_at     TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (session_hash, turn_idx)
);
CREATE INDEX IF NOT EXISTS idx_session_turns_sid ON session_turns(session_hash);

CREATE TABLE IF NOT EXISTS ai_generations (
  id               INTEGER PRIMARY KEY AUTOINCREMENT,
  session_hash     TEXT NOT NULL,
  turn_idx         INTEGER NOT NULL,
  round_idx        INTEGER NOT NULL,
  agent_name       TEXT NOT NULL,
  model            TEXT NOT NULL,
  provider         TEXT NOT NULL,
  input_tokens     INTEGER NOT NULL DEFAULT 0,
  output_tokens    INTEGER NOT NULL DEFAULT 0,
  thinking_tokens  INTEGER NOT NULL DEFAULT 0,
  cost_usd         REAL NOT NULL DEFAULT 0.0,
  duration_s       REAL NOT NULL DEFAULT 0.0,
  speed_tok_s      REAL NOT NULL DEFAULT 0.0,
  ttft_s           REAL NOT NULL DEFAULT 0.0,
  overhead_s       REAL NOT NULL DEFAULT 0.0,
  finish_reason    TEXT NOT NULL DEFAULT 'stop',
  start_row_idx    INTEGER NOT NULL,
  end_row_idx      INTEGER NOT NULL,
  created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ai_generations_sid ON ai_generations(session_hash);
CREATE INDEX IF NOT EXISTS idx_ai_generations_created ON ai_generations(created_at DESC);

-- Structured per-turn events for self-improvement queries
-- (fine-tune dataset extraction, regression tests, incident triage).
-- projected from JSONL events by ``EventProjector.apply`` inside
-- the writer's transaction. ``UNIQUE(session_hash, turn_idx, kind,
-- payload_hash)`` + INSERT-OR-IGNORE makes re-tail idempotent.
CREATE TABLE IF NOT EXISTS turn_events (
  rowid         INTEGER PRIMARY KEY AUTOINCREMENT,
  session_hash  TEXT NOT NULL,
  turn_idx      INTEGER NOT NULL,
  kind          TEXT NOT NULL,
  ts            INTEGER NOT NULL,
  payload_json  TEXT NOT NULL DEFAULT '{{}}',
  payload_hash  TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_turn_events_kind_ts
  ON turn_events(kind, ts DESC);
CREATE INDEX IF NOT EXISTS idx_turn_events_session
  ON turn_events(session_hash, turn_idx);
CREATE UNIQUE INDEX IF NOT EXISTS idx_turn_events_dedup
  ON turn_events(session_hash, turn_idx, kind, payload_hash);
""".format(cols=",\n".join(f"  {name:<30} {decl}" for name, decl in _SESSIONS_META_COLUMNS))


# ---- connection + schema (module-level so SessionWriter shares them) -


def connect_sessions_db(db_path: Any) -> sqlite3.Connection:
    """Open a WAL-mode connection to a session ``sessions.db``.

    Shared by :class:`SessionIndex` and ``session_writer.SessionWriter``
    so both speak to the db with identical pragmas. ``wal_autocheckpoint``
    is bounded so the per-delta UPDATE cadence doesn't grow the ``-wal``
    file without limit (the spec, §7.5).
    """
    # ``check_same_thread=False``: a ``SessionWriter`` may be driven from
    # a different thread than the one that opened its connection. All
    # access to a given connection is serialised by the writer's RLock
    # (in-process) and the per-session flock (cross-process), so no two
    # threads ever touch the same connection concurrently.
    con = sqlite3.connect(str(db_path), check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA temp_store=MEMORY")
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA wal_autocheckpoint=400")
        # Explicit busy timeout so a long lock-hold on one session (a
        # multi-MB model_input append) doesn't SQLITE_BUSY-abort another
        # session's BEGIN IMMEDIATE on the shared db.
        con.execute("PRAGMA busy_timeout=15000")
    except sqlite3.DatabaseError:
        pass
    return con


def list_child_sessions(sessions_dir: Any, parent_hash: str) -> list[dict[str, str]]:
    """Children of ``parent_hash`` from ``sessions_meta.parent_session_id``.

    Used by ``/subagents`` and the session-tree live snapshot so neither path
    has to open every JSONL just to walk the parent→child edge.
    """
    parent = str(parent_hash or "").strip()
    if not parent:
        return []
    db = Path(sessions_dir) / "sessions.db"
    if not db.exists():
        return []
    try:
        con = connect_sessions_db(db)
        try:
            rows = con.execute(
                "SELECT session_hash, title FROM sessions_meta WHERE parent_session_id=? ORDER BY created_at DESC",
                (parent,),
            ).fetchall()
        finally:
            con.close()
    except Exception:  # noqa: BLE001 — UI/command paths degrade to empty
        return []
    out: list[dict[str, str]] = []
    for row in rows:
        hid = str(row["session_hash"] if not isinstance(row, tuple) else row[0] or "")
        title = str(row["title"] if not isinstance(row, tuple) else (row[1] if len(row) > 1 else "") or "")
        if not hid:
            continue
        out.append(
            {
                "id": hid,
                "title": title or hid[:12],
                "subtitle": hid[:12],
            }
        )
    return out


def _sessions_db_path(
    *,
    sessions_dir: Any = None,
    db_path: Any = None,
) -> Path | None:
    if db_path:
        return Path(db_path)
    if sessions_dir:
        return Path(sessions_dir) / "sessions.db"
    try:
        from sharedagents_core.layout import sessions_db_path

        return sessions_db_path()
    except Exception:  # noqa: BLE001
        return None


def ancestor_ids(
    session_hash: str,
    *,
    sessions_dir: Any = None,
    db_path: Any = None,
    max_hops: int = 64,
) -> list[str] | None:
    """Nearest-parent-first ancestor hashes of ``session_hash``.

    Does not include ``session_hash`` itself. Returns ``[]`` when the
    session row exists and is a root (empty parent). Returns ``None``
    when the starting row is missing, the db is unavailable, or a
    query error occurs — callers must not treat that as a valid root.
    Caps at ``max_hops`` so a cycle cannot run away.
    """
    sid = str(session_hash or "").strip()
    if not sid:
        return None
    db = _sessions_db_path(sessions_dir=sessions_dir, db_path=db_path)
    if db is None or not db.exists():
        return None
    try:
        con = connect_sessions_db(db)
    except Exception:  # noqa: BLE001
        return None
    try:
        out: list[str] = []
        seen = {sid}
        cur = sid
        first = True
        for _ in range(max(1, int(max_hops))):
            row = con.execute(
                "SELECT parent_session_id FROM sessions_meta WHERE session_hash=?",
                (cur,),
            ).fetchone()
            if row is None:
                if first:
                    return None
                break
            first = False
            parent = str(row["parent_session_id"] if not isinstance(row, tuple) else row[0] or "").strip()
            if not parent or parent in seen:
                break
            out.append(parent)
            seen.add(parent)
            cur = parent
        return out
    except Exception:  # noqa: BLE001
        return None
    finally:
        try:
            con.close()
        except Exception:  # noqa: BLE001
            pass


def set_session_parent(
    session_hash: str,
    parent_session_id: str,
    *,
    sessions_dir: Any = None,
    db_path: Any = None,
    only_if_parent: str | None = None,
) -> bool:
    """Rewrite ``sessions_meta.parent_session_id``. True if a row changed."""
    sid = str(session_hash or "").strip()
    parent = str(parent_session_id or "").strip()
    if not sid:
        return False
    db = _sessions_db_path(sessions_dir=sessions_dir, db_path=db_path)
    if db is None or not db.exists():
        return False
    try:
        con = connect_sessions_db(db)
    except Exception:  # noqa: BLE001
        return False
    try:
        if only_if_parent is None:
            cur = con.execute(
                "UPDATE sessions_meta SET parent_session_id=? WHERE session_hash=?",
                (parent, sid),
            )
        else:
            cur = con.execute(
                "UPDATE sessions_meta SET parent_session_id=? WHERE session_hash=? AND parent_session_id=?",
                (parent, sid, str(only_if_parent)),
            )
        con.commit()
        return int(cur.rowcount or 0) > 0
    except Exception:  # noqa: BLE001
        return False
    finally:
        try:
            con.close()
        except Exception:  # noqa: BLE001
            pass


# States a finalized session may be forced into directly (the projector's
# ``_set_state`` LIVE_STATES minus every busy/intermediate state).
_TERMINAL_STATES = ("idle", "error", "closed")


def mark_session_terminal(sessions_dir: Any, sid: str, *, state: str = "error") -> bool:
    """Directly fold one ``sessions_meta`` row to a terminal state (backstop).

    Abort/finalize paths call this AFTER the best-effort writer-based
    ``turn_ended(aborted)`` so the row is GUARANTEED to leave the live set
    even when no turn event could be emitted. The gap it closes: a
    mid-tool-call child whose ``live_turn_idx`` is NULL (trailing stream
    delta after a prior turn ended re-stamped its state to
    ``streaming``/``prompting`` without re-arming the live idx) is skipped
    by every writer-based finalizer, leaving a ghost live row that wedges
    its waiting parent forever. Also NULLs the
    live-snapshot columns (``live_turn_idx``/``live_round_idx``/
    ``live_started_at``) so the row reads fully terminal.

    Idempotent: re-finalizing an already-terminal row is a no-op UPDATE.
    Never raises; returns True when a row was updated.
    """
    if state not in _TERMINAL_STATES:
        raise ValueError(f"mark_session_terminal: {state!r} is not terminal")
    try:
        from pathlib import Path

        db = Path(sessions_dir) / "sessions.db"
        if not db.exists():
            return False
        con = connect_sessions_db(db)
        try:
            cur = con.execute(
                "UPDATE sessions_meta SET session_state=?, "
                "live_turn_idx=NULL, live_round_idx=NULL, "
                "live_started_at=NULL, session_state_updated_at=?, "
                "session_phase='idle', session_attention='none', "
                "session_lifecycle=?, session_last_outcome=?, "
                "session_phase_changed_at=?, session_heartbeat_at=?, "
                "session_outcome_at=?, "
                "session_state_version=session_state_version+1 "
                "WHERE session_hash=? AND session_state != ?",
                (
                    state,
                    int(time.time() * 1000),
                    "detached" if state == "closed" else "open",
                    "failed" if state == "error" else "none",
                    int(time.time() * 1000),
                    int(time.time() * 1000),
                    int(time.time() * 1000) if state == "error" else 0,
                    sid,
                    state,
                ),
            )
            con.commit()
            return cur.rowcount > 0
        finally:
            con.close()
    except Exception:  # noqa: BLE001 — never let the backstop fail the abort
        return False


def ensure_schema(db_path: Any) -> None:
    """Create the schema if absent and run additive migrations.

    Idempotent. Safe to call on every writer/index construction.
    """
    con = connect_sessions_db(db_path)
    try:
        # Migrate FIRST so a legacy table gains the columns before
        # ``_SCHEMA`` creates indexes that reference them. On a fresh DB
        # the migration is a no-op (no table yet) and ``_SCHEMA`` creates
        # the full shape.
        # Only stamp the new schema version if EVERY migration landed. A v4
        # FTS realign that failed (and rolled itself back) must be retried on
        # the next boot, not skipped forever behind a bumped version (which
        # would leave legacy misaligned rows the by-rowid deletes can't reach).
        migrations_ok = _apply_migrations(con)
        con.executescript(_SCHEMA)
        # The FTS index is a derived, always-droppable
        # projection of ``turns``. If ``turns`` carries rows the FTS index is
        # missing (e.g. ``turns_fts`` was dropped and ``_SCHEMA`` just recreated
        # it empty), rebuild those rows by-rowid from the durable table without
        # mutating ``turns``. Cheap no-op when the two are already aligned.
        try:
            n_turns = int(con.execute("SELECT count(*) FROM turns").fetchone()[0])
            n_fts = int(con.execute("SELECT count(*) FROM turns_fts").fetchone()[0])
            if n_turns > n_fts:
                # Carry the turn-0 title across too (v6+); fall back to the
                # 4-column insert if the v6 migration hasn't landed yet.
                if "title" in {r[1] for r in con.execute("PRAGMA table_info(turns)")}:
                    con.execute(
                        "INSERT INTO turns_fts(rowid, combined_text, "
                        "session_hash, turn_idx, title) SELECT rowid, "
                        "combined_text, session_hash, turn_idx, title "
                        "FROM turns WHERE rowid NOT IN "
                        "(SELECT rowid FROM turns_fts)"
                    )
                else:
                    con.execute(
                        "INSERT INTO turns_fts(rowid, combined_text, "
                        "session_hash, turn_idx) SELECT rowid, combined_text, "
                        "session_hash, turn_idx "
                        "FROM turns WHERE rowid NOT IN "
                        "(SELECT rowid FROM turns_fts)"
                    )
        except sqlite3.DatabaseError:
            pass
        if migrations_ok:
            con.execute(f"PRAGMA user_version={SCHEMA_USER_VERSION}")
        con.commit()
    finally:
        con.close()


def _apply_migrations(con: sqlite3.Connection) -> bool:
    """Bring an existing ``sessions_meta`` up to the column set.

    Returns ``True`` when every migration landed (so ``ensure_schema`` may
    stamp the new ``user_version``), ``False`` only when the v4 FTS realign
    was attempted and failed — that case must be retried next boot rather
    than sealed behind a bumped version.

    SQLite can't ``ALTER TABLE ... ADD COLUMN`` with a non-constant
    default, but all defaults are constants, so an additive sweep
    works. The legacy ``frontend`` column (pre-) is preserved in
    place (SQLite can't cleanly drop columns pre-3.35) and its value
    copied into the new ``workspace_origin`` once.
    """
    cols = {row[1] for row in con.execute("PRAGMA table_info(sessions_meta)")}
    if not cols:
        return True  # table didn't exist; _SCHEMA creates the v5 shape fresh
    try:
        initial_uv = int(con.execute("PRAGMA user_version").fetchone()[0])
    except Exception:  # noqa: BLE001
        initial_uv = 0
    had_workspace_origin = "workspace_origin" in cols
    for name, decl in _SESSIONS_META_COLUMNS:
        if name == "session_hash" or name in cols:
            continue
        # Strip PRIMARY KEY (only valid at CREATE); keep type+default.
        add_decl = decl.replace(" PRIMARY KEY", "")
        try:
            con.execute(f"ALTER TABLE sessions_meta ADD COLUMN {name} {add_decl}")
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise
    # One-time copy frontend -> workspace_origin for pre-rows.
    if "frontend" in cols and not had_workspace_origin:
        con.execute(
            "UPDATE sessions_meta SET workspace_origin = frontend WHERE workspace_origin = '' AND frontend <> ''"
        )

    # v5: initialize the additive activity projection for rows written by a
    # pre-V2 projector.  The ALTERs above fill the new columns with defaults,
    # which would make an old ``streaming`` row look idle to a V2 reader until
    # its next event.  This backfill is deliberately fill-once: a partially
    # migrated database with a real V2 version must never be overwritten.
    migrated_cols = {row[1] for row in con.execute("PRAGMA table_info(sessions_meta)")}
    v2_cols = {
        "session_phase",
        "session_attention",
        "session_lifecycle",
        "session_last_outcome",
        "session_phase_changed_at",
        "session_heartbeat_at",
        "session_outcome_at",
        "session_state_version",
    }
    if initial_uv < 5 and v2_cols.issubset(migrated_cols):
        con.execute(
            "UPDATE sessions_meta SET "
            "session_phase=CASE session_state "
            "  WHEN 'prompting' THEN 'prompting' "
            "  WHEN 'streaming' THEN 'streaming' "
            "  WHEN 'tool_running' THEN 'tool_running' "
            "  WHEN 'compacting' THEN 'compacting' "
            "  ELSE 'idle' END, "
            "session_attention=CASE WHEN session_state='paused' "
            "  THEN 'operator_paused' ELSE 'none' END, "
            "session_lifecycle=CASE WHEN session_state='closed' "
            "  THEN 'detached' ELSE 'open' END, "
            "session_last_outcome=CASE WHEN session_state='error' "
            "  THEN 'failed' ELSE 'none' END, "
            "session_phase_changed_at=COALESCE(session_state_updated_at, 0), "
            "session_heartbeat_at=COALESCE(session_state_updated_at, 0), "
            "session_outcome_at=CASE WHEN session_state='error' "
            "  THEN COALESCE(session_state_updated_at, 0) ELSE 0 END, "
            "session_state_version=1 "
            "WHERE session_state_version=0"
        )
    # turn_events.payload_hash (added here for the dedup UNIQUE index).
    te_cols = {row[1] for row in con.execute("PRAGMA table_info(turn_events)")}
    if te_cols and "payload_hash" not in te_cols:
        con.execute("ALTER TABLE turn_events ADD COLUMN payload_hash TEXT NOT NULL DEFAULT ''")
        # Backfill the hash from existing payloads, THEN drop duplicate
        # rows, so the UNIQUE(session_hash,turn_idx,kind,payload_hash)
        # index in _SCHEMA can build. A pre-turn that dispatched ≥2
        # tools has multiple (sid,turn,'tool_call') rows that would all
        # carry payload_hash='' and collide, which would crash ensure_schema on any populated
        # sessions.db.
        rows = con.execute("SELECT rowid, payload_json FROM turn_events").fetchall()
        for r in rows:
            con.execute(
                "UPDATE turn_events SET payload_hash=? WHERE rowid=?",
                (_payload_hash(str(r[1] or "{}")), r[0]),
            )
        con.execute(
            "DELETE FROM turn_events WHERE rowid NOT IN ("
            "  SELECT MIN(rowid) FROM turn_events "
            "  GROUP BY session_hash, turn_idx, kind, payload_hash)"
        )

    # v4: align turns_fts.rowid with turns.rowid so an FTS row delete is a
    # by-rowid O(1) hit instead of a full-index SCAN on the UNINDEXED
    # session_hash column. The old shape made EVERY turn_ended — and every
    # boot re-fold of a re-pruned (shrunk) session — scan the whole
    # turns_fts index; at scale (a 384MB index) that wedged orchestrator
    # boot for ~2.5min while it re-folded a backlog. combined_text is
    # derived text, so a one-time bulk realign is safe and cheap (no
    # per-session scan). Guarded on user_version so it runs exactly once;
    # ``ensure_schema`` stamps the new version immediately after.
    try:
        uv = int(con.execute("PRAGMA user_version").fetchone()[0])
    except Exception:  # noqa: BLE001
        uv = 0
    if uv < 4:

        def has(n: str) -> bool:
            return con.execute("SELECT 1 FROM sqlite_master WHERE name=?", (n,)).fetchone() is not None

        if has("turns_fts") and has("turns"):
            # SAVEPOINT so a mid-realign failure rolls back the DELETE too —
            # never leave the FTS emptied. On failure return False so
            # ``ensure_schema`` withholds the version stamp and the realign is
            # retried next boot (rather than sealed off with legacy misaligned
            # rows the by-rowid deletes would then never reach).
            try:
                con.execute("SAVEPOINT fts_realign_v4")
                con.execute("DELETE FROM turns_fts")
                con.execute(
                    "INSERT INTO turns_fts(rowid, combined_text, session_hash, "
                    "turn_idx) SELECT rowid, combined_text, session_hash, "
                    "turn_idx FROM turns"
                )
                con.execute("RELEASE fts_realign_v4")
            except sqlite3.DatabaseError:
                try:
                    con.execute("ROLLBACK TO fts_realign_v4")
                    con.execute("RELEASE fts_realign_v4")
                except sqlite3.DatabaseError:
                    pass
                return False  # don't stamp v4 → retry on the next boot

    # v6: add a `title` column to `turns` and an indexed `title` column to
    # `turns_fts` so a session's title words are FTS-searchable even when a
    # title word never appears in any turn's text. The title is carried ONLY
    # on the turn-0 row (empty on every other row) so BM25 is not diluted by
    # repeating the title once per turn.
    #
    # ``turns`` is a regular table, so its column is added additively with
    # ALTER. ``turns_fts`` is an FTS5 *virtual* table: SQLite only permits
    # ALTER on FTS5 from 3.47 on, and this runtime ships older, so a legacy
    # index is rebuilt in place — DROP + CREATE with the new column set, then
    # re-populate every row from ``turns`` (the durable source). The FTS
    # index is a derived, always-rebuildable cache, so a one-time full
    # re-fold is safe and is exactly what ``rebuild()`` already does.
    #
    # Guarded on user_version so it runs exactly once; on failure the version
    # stamp is withheld and the step retries next boot (idempotent). The
    # guard is on ``turns`` ONLY: if ``turns_fts`` is missing, ``_SCHEMA``
    # (run right after the migrations) recreates it with the v6 column set
    # and the self-heal re-folds its rows carrying title. Guarding on both
    # tables would stamp v6 without ever adding the column to ``turns`` and
    # leave that DB broken forever.
    try:
        uv6 = int(con.execute("PRAGMA user_version").fetchone()[0])
    except Exception:  # noqa: BLE001
        uv6 = 0
    if uv6 < 6:

        def has6(n: str) -> bool:
            return con.execute("SELECT 1 FROM sqlite_master WHERE name=?", (n,)).fetchone() is not None

        if has6("turns"):
            try:
                con.execute("SAVEPOINT fts_title_v6")
                # 1. Additive column on the durable table.
                if "title" not in {r[1] for r in con.execute("PRAGMA table_info(turns)")}:
                    con.execute("ALTER TABLE turns ADD COLUMN title TEXT NOT NULL DEFAULT ''")
                # 2. Backfill turn-0 rows from the canonical stored title
                #    (COALESCE: an orphan turns row with no sessions_meta
                #    row must not violate the NOT NULL constraint).
                con.execute(
                    "UPDATE turns SET title=COALESCE(("
                    "SELECT m.title FROM sessions_meta m "
                    "WHERE m.session_hash = turns.session_hash"
                    "), '') WHERE turn_idx=0"
                )
                # 3. Bring the FTS index to the v6 column set + title data.
                if has6("turns_fts"):
                    if "title" not in {r[1] for r in con.execute("PRAGMA table_info(turns_fts)")}:
                        # Legacy 4-column index: drop + recreate (FTS5
                        # virtual tables can't be ALTERed on this SQLite).
                        con.execute("DROP TABLE turns_fts")
                        con.execute(
                            "CREATE VIRTUAL TABLE turns_fts USING fts5("
                            "combined_text, title, session_hash UNINDEXED, "
                            "turn_idx UNINDEXED, tokenize='unicode61')"
                        )
                    # One-time re-fold from the durable table. Covers both
                    # the just-recreated index and a title-column index
                    # whose rows predate title content (re-folded by the v4
                    # realign / self-heal with empty titles).
                    con.execute("DELETE FROM turns_fts")
                    con.execute(
                        "INSERT INTO turns_fts(rowid, combined_text, "
                        "session_hash, turn_idx, title) SELECT rowid, "
                        "combined_text, session_hash, turn_idx, title "
                        "FROM turns"
                    )
                # (turns_fts missing: _SCHEMA recreates it with the v6
                #  column set; the self-heal re-folds rows from turns,
                #  which now carries title.)
                con.execute("RELEASE fts_title_v6")
            except sqlite3.DatabaseError:
                try:
                    con.execute("ROLLBACK TO fts_title_v6")
                    con.execute("RELEASE fts_title_v6")
                except sqlite3.DatabaseError:
                    pass
                return False  # don't stamp v6 → retry on the next boot
    return True


# ---- event projection (JSONL event -> sessions.db rows) -------------


# Indexable lifecycle event kinds -> turn_events.kind.
_LIFECYCLE_INDEX_KINDS = {
    "loop_detected": "loop",
    "nudge": "nudge",
    "distill": "distill",
    "compaction": "compaction",
    "pause_resume": "pause_resume",
    "approval_decision": "approval_decision",
    "hook_outcome": "hook_outcome",
}


def _payload_hash(payload_json: str) -> str:
    import hashlib

    return hashlib.sha1(payload_json.encode("utf-8")).hexdigest()[:16]


class EventProjector:
    """Projects JSONL events into ``sessions.db`` rows, in order.

    Stateful: tracks each in-flight turn's user + accumulated content so
    ``turn_ended`` can build the FTS ``combined_text`` without a cross-
    event SQL read. One instance per writer (live path) and a fresh
    instance per ``tail``/``rebuild`` replay. ``apply`` runs inside the
    caller's open transaction; it never commits.

    Drives the session state machine: each event branch that changes
    the session's lifecycle phase calls :meth:`_set_state` to write
    the new value into ``sessions_meta.session_state`` (+ its
    ``session_state_updated_at`` timestamp), giving cross-agent
    consumers a single SQL predicate for "what is every agent doing
    right now?".
    """

    # Compatibility values accepted by ``sessions_meta.session_state``.
    # ``waiting``/``error``/``closed`` remain queryable for older clients, but
    # the V2 columns below carry the independent activity facts.
    LIVE_STATES: tuple[str, ...] = LEGACY_STATES

    # States that mean a child is still doing work (parent may stay in waiting).
    _CHILD_BUSY_STATES: tuple[str, ...] = TREE_BUSY_STATES

    def __init__(self) -> None:
        # turn_idx -> {"user": str, "content": str}
        self._scratch: dict[int, dict[str, str]] = {}
        # (sid, call_id) of tool_result_fail rows awaiting their body —
        # the failing BODY lives on the tool_result sibling event that
        # follows the tool_result_meta. Bounded FIFO; a miss just means
        # the row keeps content_chars only (pre-H6a behaviour).
        self._pending_fail_bodies: dict[tuple[str, str], bool] = {}

    def _set_state(
        self,
        con: sqlite3.Connection,
        sid: str,
        state: str,
        ts_ms: int,
        *,
        outcome: str | None = None,
        activity_phase: str | None = None,
        attention: str | None = None,
        lifecycle: str | None = None,
        expected_turn_idx: int | None = None,
    ) -> None:
        """Update ``sessions_meta.session_state`` + its timestamp.

        ``state`` MUST be a member of :attr:`LIVE_STATES`; a typo would
        otherwise silently produce a state no compatibility query matches.
        ``ts_ms`` is the projector's event-fold wall clock. V2 fields are
        updated atomically with the legacy register so API readers never see
        a half-upgraded snapshot.
        """
        if state not in LEGACY_STATES:
            raise ValueError(f"unknown session_state: {state!r}")
        row = con.execute(
            "SELECT session_phase, session_attention, session_lifecycle, "
            "session_last_outcome, session_phase_changed_at, "
            "session_outcome_at, live_turn_idx "
            "FROM sessions_meta WHERE session_hash=?",
            (sid,),
        ).fetchone()
        current_phase = "idle"
        current_attention = "none"
        current_lifecycle = "open"
        current_outcome = "none"
        phase_changed_at = 0
        outcome_at = 0
        live_turn_idx = None
        if row is not None:
            current_phase = str(row["session_phase"] or "idle")
            current_attention = str(row["session_attention"] or "none")
            current_lifecycle = str(row["session_lifecycle"] or "open")
            current_outcome = str(row["session_last_outcome"] or "none")
            phase_changed_at = int(row["session_phase_changed_at"] or 0)
            outcome_at = int(row["session_outcome_at"] or 0)
            live_turn_idx = row["live_turn_idx"]
        if activity_phase not in PHASES:
            # A pause keeps the prior own phase in V2 while its compatibility
            # state says ``paused``. Other terminal/tree states are idle.
            if state == "paused" and live_turn_idx is not None:
                activity_phase = legacy_phase(
                    state,
                    current_phase,
                    live_turn_idx=live_turn_idx,
                )
            else:
                activity_phase = legacy_phase(
                    state,
                    live_turn_idx=live_turn_idx,
                )
        # Compaction/pause-resume may pass a live V2 phase while folding the
        # compatibility register to idle/waiting. Once the own turn is gone,
        # that leftover must not survive as a live own-turn phase.
        if state not in OWN_LIVE_STATES and live_turn_idx is None and activity_phase in OWN_LIVE_STATES:
            activity_phase = "idle"
        if attention not in ATTENTION_STATES:
            if state == "ask":
                attention = "question"
            elif state == "paused":
                attention = "operator_paused"
            else:
                attention = "none"
        if lifecycle not in ("open", "detached"):
            lifecycle = "detached" if state == "closed" else current_lifecycle
        if outcome not in OUTCOMES:
            outcome = current_outcome
            if state == "error" and outcome == "none":
                outcome = "failed"
        phase_changed = (
            activity_phase != current_phase or attention != current_attention or lifecycle != current_lifecycle
        )
        if phase_changed or not phase_changed_at:
            phase_changed_at = int(ts_ms)
        if outcome != current_outcome and outcome != "none":
            outcome_at = int(ts_ms)
        where = "WHERE session_hash=?"
        params: list[Any] = [
            state,
            int(ts_ms),
            activity_phase,
            attention,
            lifecycle,
            outcome,
            int(phase_changed_at),
            int(ts_ms),
            int(outcome_at),
            sid,
        ]
        if expected_turn_idx is not None:
            where += " AND live_turn_idx=?"
            params.append(int(expected_turn_idx))
        con.execute(
            "UPDATE sessions_meta SET session_state=?, "
            "session_state_updated_at=?, session_phase=?, "
            "session_attention=?, session_lifecycle=?, "
            "session_last_outcome=?, session_phase_changed_at=?, "
            "session_heartbeat_at=?, session_outcome_at=?, "
            "session_state_version=session_state_version+1 " + where,
            tuple(params),
        )
        # Child activity keeps a waiting parent's freshness stamp alive so
        # the monitor freshness gate doesn't age the parent out mid-tree.
        self._touch_parent_waiting(con, sid, ts_ms)

    def _set_live_state(self, con: sqlite3.Connection, sid: str, state: str, ts_ms: int, ti) -> None:
        """Set a *live-phase* state only when ``ti`` is the active turn.

        A trailing / out-of-order ``text_delta`` / ``thinking_delta`` /
        ``tool_call`` / ``tool_result_meta`` for a turn that has already
        ended must NOT resurrect the session to ``streaming`` /
        ``tool_running``. ``turn_ended`` clears ``live_turn_idx`` to NULL
        (and sets the terminal state), so guarding the write on
        ``live_turn_idx = ti`` makes a stale event a no-op (NULL never
        equals ``ti``) and also ignores a delta whose turn isn't the one
        currently live.

        This is the projector-side fix for the Agent Monitor "subagent
        stuck streaming after parent cancel" bug, where a buffered delta
        from the still-draining engine stream landed after the cancel
        finalized the turn and flipped ``session_state`` back to a live
        value with a fresh timestamp — leaving ``active_sessions=0`` while
        ``subagents_calling>0``. The guard is DB-based (not in-memory) so
        it holds even when the finalize and the trailing delta are
        projected by different writers. ``reset_stale_sessions`` remains
        the boot-time backstop for rows that predate this guard.

        Scope: this covers the engine-streamed delta/tool events, which
        are what the engine emits while draining. It deliberately does NOT
        guard ``compaction`` / ``pause_resume`` (orchestrator-emitted, not
        engine-buffered) nor the live-snapshot columns
        (``live_content_so_far`` etc., cosmetic — the monitor keys off
        ``session_state``). Those residual vectors are tracked as a
        follow-up, not part of this fix.
        """
        if state not in LEGACY_STATES:
            raise ValueError(f"unknown session_state: {state!r}")
        if ti is None:
            return
        attention = "question" if state == "ask" else "none"
        phase = state if state in PHASES else "idle"
        con.execute(
            "UPDATE sessions_meta SET session_state=?, "
            "session_state_updated_at=?, session_phase=?, "
            "session_attention=?, session_heartbeat_at=?, "
            "session_phase_changed_at=CASE WHEN session_phase != ? "
            "THEN ? ELSE session_phase_changed_at END, "
            "session_state_version=session_state_version+1 "
            "WHERE session_hash=? AND live_turn_idx=?",
            (
                state,
                int(ts_ms),
                phase,
                attention,
                int(ts_ms),
                phase,
                int(ts_ms),
                sid,
                int(ti),
            ),
        )
        self._touch_parent_waiting(con, sid, ts_ms)

    def _parent_still_has_work(
        self,
        con: sqlite3.Connection,
        sid: str,
        *,
        freshness_ms: int | None = None,
    ) -> bool:
        """True if ``sid`` still has live descendants or open async jobs.

        ``freshness_ms`` (optional) adds a staleness window: only children
        whose ``session_state_updated_at`` is >= ``freshness_ms`` count as
        busy. A child stuck at a live value with an old stamp is a GHOST —
        an abort/crash left its state live with no terminal event, so it
        must not keep the parent waiting forever (2026-08-11 wedge: a
        'waiting' parent held 20+ min by 2 ghost 'streaming'/'prompting'
        explore children). Open async jobs ALWAYS count: the job registry
        only returns jobs that are still open. ``None`` = no gate
        (historical behaviour for the projector's own hot paths).

        Hidden internal side-call children (``title``/``summary`` recaps)
        never count — the same exclusion ``count_fresh_live_descendants``
        applies. A recap fires ~30s after every turn end; treating it as
        work flipped the finished parent to ``waiting`` for the whole side
        call, which the WebUI renders as a Stop button on a session whose
        last turn already succeeded.
        """
        ph = ",".join("?" for _ in self._CHILD_BUSY_STATES)
        hidden = ",".join("?" for _ in INTERNAL_SIDE_CALL_AGENTS)
        agent_not_hidden = f"LOWER(TRIM(COALESCE(agent_name, ''))) NOT IN ({hidden})"
        if freshness_ms is not None:
            # An OPEN child turn (live_turn_idx set) is work regardless of
            # the stamp age: a single long silent tool call ages
            # session_state_updated_at past the cutoff while genuinely
            # running. Ghost rows (the 2026-08-11 wedge) have a NULL
            # live_turn_idx and still release the parent.
            row = con.execute(
                f"SELECT 1 FROM sessions_meta WHERE parent_session_id=? "
                f"AND session_state IN ({ph}) "
                f"AND {agent_not_hidden} "
                f"AND (session_state_updated_at >= ? "
                f"     OR live_turn_idx IS NOT NULL) LIMIT 1",
                (sid, *self._CHILD_BUSY_STATES, *INTERNAL_SIDE_CALL_AGENTS, freshness_ms),
            ).fetchone()
        else:
            row = con.execute(
                f"SELECT 1 FROM sessions_meta WHERE parent_session_id=? "
                f"AND session_state IN ({ph}) "
                f"AND {agent_not_hidden} LIMIT 1",
                (sid, *self._CHILD_BUSY_STATES, *INTERNAL_SIDE_CALL_AGENTS),
            ).fetchone()
        if row is not None:
            return True
        # Race: parent turn_ended can fold before the child's
        # session_started is indexed. Open job records are written at
        # Agent dispatch, so they cover that window.
        try:
            from sharedagents_agent.agents.async_jobs import open_jobs_for

            if open_jobs_for(sid):
                return True
        except Exception:  # noqa: BLE001 — jobs store optional at fold time
            pass
        return False

    def _maybe_clear_parent_waiting(
        self,
        con: sqlite3.Connection,
        sid: str,
        ts_ms: int,
        *,
        child_outcome: str = "",
    ) -> None:
        """If this child finished and its parent is ``waiting``, maybe idle it.

        A child that ended ``interrupted`` or ``failed`` must not leave the
        parent stamped ``succeeded``: the parent's last turn delivered a
        status message, not a completed task. Propagate the child's outcome
        when the parent has no other work left.
        """
        parent = self._parent_of(con, sid)
        if not parent:
            return
        prow = con.execute(
            "SELECT session_state FROM sessions_meta WHERE session_hash=?",
            (parent,),
        ).fetchone()
        if prow is None:
            return
        pstate = prow["session_state"] if not isinstance(prow, tuple) else prow[0]
        if pstate != "waiting":
            return
        if not self._parent_still_has_work(con, parent):
            parent_outcome = child_outcome if child_outcome in ("interrupted", "failed") else None
            self._set_state(
                con,
                parent,
                "idle",
                ts_ms,
                outcome=parent_outcome,
            )
            # Nested trees: clearing a mid-level waiter may free its parent.
            self._maybe_clear_parent_waiting(
                con,
                parent,
                ts_ms,
                child_outcome=child_outcome,
            )

    def _maybe_mark_parent_waiting(self, con: sqlite3.Connection, sid: str, ts_ms: int) -> None:
        """If a child is busy and its parent is idle, flip parent to ``waiting``.

        Covers the race where the parent already folded to ``idle`` before
        the child's ``session_started`` / ``turn_started`` was indexed.
        """
        parent = self._parent_of(con, sid)
        if not parent:
            return
        prow = con.execute(
            "SELECT session_state FROM sessions_meta WHERE session_hash=?",
            (parent,),
        ).fetchone()
        if prow is None:
            return
        pstate = prow["session_state"] if not isinstance(prow, tuple) else prow[0]
        if pstate not in ("idle", "waiting"):
            return
        if self._parent_still_has_work(con, parent):
            self._set_state(con, parent, "waiting", ts_ms)
            # Nested trees: a mid-level waiter also keeps its parent waiting.
            self._maybe_mark_parent_waiting(con, parent, ts_ms)

    def _touch_parent_waiting(self, con: sqlite3.Connection, sid: str, ts_ms: int) -> None:
        parent = self._parent_of(con, sid)
        if not parent:
            return
        con.execute(
            "UPDATE sessions_meta SET session_state_updated_at=?, "
            "session_heartbeat_at=? "
            "WHERE session_hash=? AND session_state='waiting'",
            (int(ts_ms), int(ts_ms), parent),
        )

    @staticmethod
    def _parent_of(con: sqlite3.Connection, sid: str) -> str:
        row = con.execute(
            "SELECT parent_session_id FROM sessions_meta WHERE session_hash=?",
            (sid,),
        ).fetchone()
        if row is None:
            return ""
        parent = row["parent_session_id"] if not isinstance(row, tuple) else row[0]
        return str(parent or "")

    def apply(self, con: sqlite3.Connection, sid: str, event: dict, end_offset: int) -> None:
        etype = str(event.get("type") or "")
        now_ms = int(time.time() * 1000)
        # Indexed ``turn_events`` rows must carry the event's REAL time, not
        # the materialisation wall-clock. ``apply()`` is also the replay path
        # for ``rebuild()`` / re-tail, where ``now_ms`` would stamp every
        # historical row with the rebuild time and collapse the timeline.
        # ``now_ms`` stays for live/bookkeeping columns (updated_at,
        # session_state_updated_at, live_state) which legitimately want the
        # wall clock.
        event_ms = _iso_to_ms(event.get("ts")) or now_ms
        # The cursor advances for EVERY event so a re-tail resumes past
        # already-applied lines.
        con.execute(
            "UPDATE sessions_meta SET indexed_bytes=?, updated_at=? WHERE session_hash=?",
            (int(end_offset), now_ms, sid),
        )
        if etype in ("session_started", "session_meta"):
            existing = con.execute(
                "SELECT session_state, session_state_version, live_turn_idx FROM sessions_meta WHERE session_hash=?",
                (sid,),
            ).fetchone()
            self._upsert_header_from_started(con, sid, event, now_ms, end_offset)
            # A freshly seen session is idle until the first turn lands. A
            # duplicate/replayed header is metadata-only: it must never reset a
            # running, failed, or detached session back to idle.
            if existing is None or (
                int(existing["session_state_version"] or 0) == 0
                and existing["live_turn_idx"] is None
                and str(existing["session_state"] or "idle") == "idle"
            ):
                self._set_state(con, sid, "idle", now_ms)
            # Child header may land after the parent already folded idle —
            # open jobs / later turn_started will promote the parent; if an
            # open job already exists, mark waiting now.
            self._maybe_mark_parent_waiting(con, sid, now_ms)
            return
        if etype == "agent_switched":
            agent = str(event.get("to_agent") or "").strip()
            if agent:
                row = con.execute(
                    "SELECT footer_state_json, model, endpoint, backend FROM sessions_meta WHERE session_hash=?",
                    (sid,),
                ).fetchone()
                try:
                    footer = json.loads((row["footer_state_json"] if row is not None else "{}") or "{}")
                except (TypeError, ValueError):
                    footer = {}
                if not isinstance(footer, dict):
                    footer = {}
                sequence = footer.get("sequence")
                if not isinstance(sequence, list):
                    sequence = []
                model = str(event.get("model") or ((row["model"] if row is not None else "") or ""))
                endpoint = str(event.get("endpoint") or ((row["endpoint"] if row is not None else "") or ""))
                backend = str(event.get("backend") or ((row["backend"] if row is not None else "") or ""))
                activation = {
                    "agent": agent,
                    "model": model,
                    "timestamp": str(event.get("ts") or ""),
                }
                if not sequence or sequence[-1].get("agent") != agent:
                    sequence.append(activation)
                else:
                    sequence[-1] = activation
                footer = {"current": activation, "sequence": sequence}
                con.execute(
                    "UPDATE sessions_meta SET agent_name=?, model=?, endpoint=?, backend=?, "
                    "footer_state_json=?, updated_at=? "
                    "WHERE session_hash=?",
                    (agent, model, endpoint, backend, json.dumps(footer), event_ms, sid),
                )
            return
        ti = event.get("turn_idx")

        if etype == "queue_msg_injected":
            # A follow-up injected into an existing turn
            # also contributes an author, without changing the original
            # prompt's bubble label. The JSONL event survives a rebuild.
            self._index_turn_actors(con, sid, ti, event.get("actor"), now_ms)

        if etype == "turn_started":
            ti_int = int(ti or 0)
            self._index_turn_actors(con, sid, ti_int, event.get("actors"), now_ms)
            current = con.execute(
                "SELECT session_state, session_lifecycle, live_turn_idx, "
                "last_turn_idx FROM sessions_meta WHERE session_hash=?",
                (sid,),
            ).fetchone()
            if current is not None:
                current_state = str(current["session_state"] or "idle")
                current_lifecycle = str(current["session_lifecycle"] or "open")
                current_live = current["live_turn_idx"]
                current_last = int(current["last_turn_idx"] or -1)
                # A detached session is terminal. A buffered/replayed
                # turn_started must not resurrect it or reset its outcome.
                if current_lifecycle == "detached" or current_state == "closed":
                    return
                # Turn indices are monotonic per session. Ignore a late
                # header for a completed/older turn, and never replace a
                # newer live turn with an older buffered event.
                if (current_live is None and current_last >= ti_int) or (
                    current_live is not None and ti_int < int(current_live)
                ):
                    return
            user_text = _extract_user_text(event.get("user"))
            self._scratch[ti_int] = {
                "user": user_text,
                "content": "",
            }
            con.execute(
                "UPDATE sessions_meta SET live_turn_idx=?, live_round_idx=NULL, "
                "live_started_at=?, live_content_so_far='', "
                "live_thinking_so_far='', live_tool_calls_so_far='[]', "
                "live_last_event_offset=? WHERE session_hash=?",
                (ti_int, str(event.get("ts") or ""), int(end_offset), sid),
            )
            # Derive title from the first turn if still untitled.
            row = con.execute("SELECT title FROM sessions_meta WHERE session_hash=?", (sid,)).fetchone()
            if row is not None:
                cur_title = row["title"] if not isinstance(row, tuple) else row[0]
                if not cur_title or str(cur_title).startswith("untitled-"):
                    if not _is_compact_user_text(user_text):
                        con.execute(
                            "UPDATE sessions_meta SET title=? WHERE session_hash=?",
                            (_derive_title(event.get("user")), sid),
                        )
            next_state = "compacting" if _is_compact_user_text(user_text) else "prompting"
            self._set_state(con, sid, next_state, now_ms)
            self._maybe_mark_parent_waiting(con, sid, now_ms)
            return

        if etype == "round_started":
            con.execute(
                "UPDATE sessions_meta SET live_round_idx=?, live_last_event_offset=? WHERE session_hash=?",
                (int(event.get("round_idx") or 0), int(end_offset), sid),
            )
            self._set_live_state(con, sid, "prefill", now_ms, ti)
            return

        if etype == "text_delta":
            self._scratch.setdefault(int(ti), {"user": "", "content": ""})
            self._scratch[int(ti)]["content"] += str(event.get("delta") or "")
            # Cap the live column to a trailing window so a long turn
            # doesn't rewrite an ever-growing TEXT value per delta
            # (O(n^2) WAL churn). The full content lives in the JSONL;
            # the live column only feeds the "what's it typing now"
            # poll.
            con.execute(
                "UPDATE sessions_meta SET "
                "live_content_so_far = substr(live_content_so_far || ?, ?), "
                "live_last_event_offset=? WHERE session_hash=?",
                (str(event.get("delta") or ""), -_LIVE_SNAPSHOT_CAP, int(end_offset), sid),
            )
            self._set_live_state(con, sid, "streaming", now_ms, ti)
            return

        if etype == "thinking_delta":
            con.execute(
                "UPDATE sessions_meta SET "
                "live_thinking_so_far = substr(live_thinking_so_far || ?, ?), "
                "live_last_event_offset=? WHERE session_hash=?",
                (str(event.get("delta") or ""), -_LIVE_SNAPSHOT_CAP, int(end_offset), sid),
            )
            self._set_live_state(con, sid, "reasoning", now_ms, ti)
            return

        if etype == "tool_call":
            self._append_live_tool_call(con, sid, event, end_offset)
            name = str(event.get("name") or "")
            self._index_event(
                con,
                sid,
                ti,
                "tool_call",
                {
                    "call_id": str(event.get("call_id") or ""),
                    "name": name,
                    "arguments": _cap_args(event.get("arguments") or {}),
                    "started_at": str(event.get("started_at") or ""),
                },
                event_ms,
            )
            if is_ask_tool_name(name):
                live_state = "ask"
            elif name == "compact_context":
                live_state = "compacting"
            else:
                live_state = "tool_running"
            self._set_live_state(con, sid, live_state, now_ms, ti)
            return

        if etype == "tool_result_meta":
            if not event.get("ok"):
                cid = str(event.get("call_id") or "")
                self._index_event(
                    con,
                    sid,
                    ti,
                    "tool_result_fail",
                    {
                        "call_id": cid,
                        "content_chars": int(event.get("content_chars") or 0),
                        "ended_at": str(event.get("ended_at") or ""),
                    },
                    event_ms,
                )
                # Remember to attach the failing body head when the
                # tool_result sibling arrives. The body is what SI triage
                # keys on (launch-failure regex) — content_chars alone
                # made DB-side error views untriageable.
                if cid:
                    if len(self._pending_fail_bodies) >= 64:
                        self._pending_fail_bodies.pop(next(iter(self._pending_fail_bodies)), None)
                    self._pending_fail_bodies[(sid, cid)] = True
            con.execute(
                "UPDATE sessions_meta SET live_last_event_offset=? WHERE session_hash=?",
                (int(end_offset), sid),
            )
            # Do not infer prefill/streaming here. A tool result may continue
            # the same round, start another model call, or end the turn.
            # The next round_started (model-call boundary) or delta sets
            # the live phase.
            return

        if etype == "tool_result":
            # Attach the failing body head to the tool_result_fail
            # row indexed from the meta sibling. Body events are
            # otherwise not indexed (full body stays in the JSONL).
            cid = str(event.get("call_id") or "")
            if cid and self._pending_fail_bodies.pop((sid, cid), None):
                head = str(event.get("content") or "")[:600]
                if head:
                    rows = con.execute(
                        "SELECT rowid, payload_json FROM turn_events "
                        "WHERE session_hash=? AND kind='tool_result_fail' "
                        "ORDER BY rowid DESC LIMIT 16",
                        (sid,),
                    ).fetchall()
                    for r in rows:
                        try:
                            p = json.loads(r["payload_json"])
                        except (TypeError, ValueError):
                            continue
                        if p.get("call_id") == cid and "content_head" not in p:
                            p["content_head"] = head
                            pj = json.dumps(p, ensure_ascii=False)
                            # OR IGNORE: the enriched payload can hash onto an
                            # EXISTING row for the same (session, turn, kind) —
                            # a re-tail after a cursor reset re-processes the
                            # event while the enriched row from the first pass
                            # is already present. A plain UPDATE raises
                            # IntegrityError on the dedup UNIQUE index, which
                            # aborts the whole tail() and rolls the cursor back
                            # to its pre-tail value; the session then re-decodes
                            # from 0 on EVERY refresh() (measured:
                            # 6 such sessions cost ~830ms per refresh()/tree
                            # read — the ~1s session-load tax).
                            # Ignoring the collision is correct: the enriched
                            # row is already there, so the event is captured.
                            con.execute(
                                "UPDATE OR IGNORE turn_events SET payload_json=?, payload_hash=? WHERE rowid=?",
                                (pj, _payload_hash(pj), r["rowid"]),
                            )
                            break
            return

        if etype == "round_ended":
            finish = str(event.get("finish_reason") or "")
            metrics = event.get("metrics") or {}
            if finish or metrics.get("loop_detected"):
                self._index_event(
                    con,
                    sid,
                    ti,
                    "finish",
                    {
                        "finish_reason": finish,
                        "loop_detected": bool(metrics.get("loop_detected")),
                        "round_idx": event.get("round_idx"),
                    },
                    event_ms,
                )
            # Loop fires live in per-ROUND metrics, but the batch
            # extractor only reads TURN-level metrics — so the canonical
            # ``loop`` rows never materialized on the live path (394
            # gateway.log fires vs 0 DB rows). Emit the
            # same kind/payload shape as ``_extract_turn_events`` here.
            if (
                metrics.get("loop_detected")
                or metrics.get("token_loop_suspected")
                or finish in ("tool_args_loop", "content_loop")
            ):
                self._index_event(
                    con,
                    sid,
                    ti,
                    "loop",
                    {
                        "tier": metrics.get("loop_tier") or "",
                        "evidence": str(metrics.get("loop_evidence") or "")[:200],
                        "channel": metrics.get("loop_channel") or "",
                        "tool_name": metrics.get("loop_tool_name") or "",
                        "args_chars": metrics.get("loop_args_chars"),
                        "checked_at": metrics.get("loop_checked_at"),
                        "finish_reason": finish,
                        "token_loop_suspected": bool(metrics.get("token_loop_suspected") or False),
                        "round_idx": event.get("round_idx"),
                    },
                    event_ms,
                )
            return

        if etype == "turn_ended":
            self._on_turn_ended(con, sid, event, now_ms)
            return

        if etype == "turn_retry":
            self._set_state(con, sid, "retrying", now_ms)
            return

        if etype == "header_update":
            patch = event.get("patch") or {}
            self._apply_header_patch(con, sid, patch)
            # A title change must be searchable immediately (no manual
            # rebuild): re-emit the turn-0 FTS row with the new title.
            if "title" in patch:
                self._sync_turn0_title(con, sid)
            model = str(patch.get("engine_model") or patch.get("model") or "")
            if model:
                row = con.execute(
                    "SELECT agent_name, footer_state_json FROM sessions_meta WHERE session_hash=?",
                    (sid,),
                ).fetchone()
                try:
                    footer = json.loads((row or ["", "{}"])[1] or "{}")
                except (TypeError, ValueError):
                    footer = {}
                if not isinstance(footer, dict):
                    footer = {}
                sequence = footer.get("sequence")
                if not isinstance(sequence, list):
                    sequence = []
                agent = str((row["agent_name"] if row is not None else "") or "")
                current = dict(footer.get("current") or {})
                current.update({"agent": agent, "model": model})
                if sequence and sequence[-1].get("agent") == agent:
                    sequence[-1] = {**sequence[-1], "model": model}
                footer = {"current": current, "sequence": sequence}
                con.execute(
                    "UPDATE sessions_meta SET footer_state_json=? WHERE session_hash=?",
                    (json.dumps(footer), sid),
                )
            return

        if etype == "session_closed":
            self._set_state(con, sid, "closed", now_ms, lifecycle="detached")
            return

        if etype in _LIFECYCLE_INDEX_KINDS:
            payload = {k: v for k, v in event.items() if k not in ("v", "ts", "type", "engine")}
            self._index_event(
                con,
                sid,
                ti,
                _LIFECYCLE_INDEX_KINDS[etype],
                payload,
                event_ms,
            )
            # State transitions for compaction / pause_resume — the
            # only two lifecycle events with a phase-bearing payload.
            # Other lifecycle kinds (loop_detected, nudge, distill,
            # approval_decision, hook_outcome) don't change session
            # phase, only stash a row in turn_events.
            phase = str(event.get("phase") or "").strip().lower()
            if etype == "compaction":
                ti_int = int(ti) if ti is not None else None
                if phase in ("done", "ended", "complete"):
                    end = "waiting" if self._parent_still_has_work(con, sid) else "idle"
                    # Preserve the own-turn phase in V2 while keeping the
                    # legacy idle/waiting fold for existing SQL consumers.
                    live = con.execute(
                        "SELECT live_turn_idx FROM sessions_meta WHERE session_hash=?",
                        (sid,),
                    ).fetchone()
                    resume = (
                        "streaming"
                        if live is not None and ti_int is not None and live["live_turn_idx"] == ti_int
                        else "idle"
                    )
                    self._set_state(
                        con,
                        sid,
                        end,
                        now_ms,
                        activity_phase=resume,
                        expected_turn_idx=ti_int,
                    )
                else:
                    self._set_state(
                        con,
                        sid,
                        "compacting",
                        now_ms,
                        expected_turn_idx=ti_int,
                    )
            elif etype == "pause_resume":
                ti_int = int(ti) if ti is not None else None
                if phase in ("paused", "pause"):
                    self._set_state(
                        con,
                        sid,
                        "paused",
                        now_ms,
                        expected_turn_idx=ti_int,
                    )
                elif phase in ("resumed", "resume"):
                    end = "waiting" if self._parent_still_has_work(con, sid) else "idle"
                    self._set_state(
                        con,
                        sid,
                        end,
                        now_ms,
                        attention="none",
                        activity_phase=("streaming" if ti_int is not None else "idle"),
                        expected_turn_idx=ti_int,
                    )
            return
        # Anything else: cursor already advanced.

    # -- helpers --

    def _upsert_header_from_started(self, con, sid, event, now_ms, end_offset):
        cols = [name for name, _ in _SESSIONS_META_COLUMNS]
        vals: dict[str, Any] = {}
        for name in cols:
            if name in event:
                v = event[name]
                # Defense-in-depth: never let a (re-)emitted
                # ``session_started`` clobber an already-populated identity
                # field with an empty value. The ON CONFLICT UPDATE below
                # would otherwise blank a good agent_name / engine /
                # created_at if some session-creator emits a second
                # ``session_started`` with those unset. Mirrors the same
                # guard already in ``_apply_header_patch`` — this is the
                # layer that makes the "blank attribution can never
                # self-heal" bug class impossible regardless of which
                # creator runs.
                if name in ("engine", "created_at", "agent_name") and not v:
                    continue
                if name == "parent_session_id" and not v:
                    continue
                # Cost attribution is immutable once set:
                # a re-emitted ``session_started`` never re-stamps it — see the
                # fill-only CASE in ``_assign`` below. Skip only an empty value
                # here (nothing to fill).
                if name in ("cost_root_session_id", "cost_ancestor_goal_id") and not v:
                    continue
                # Same monotonic guard as ``_apply_header_patch`` for the
                # once-per-session memory-flush gate: a re-projected historical
                # ``session_started`` legitimately carries ``last_memory_flush_at=""``
                # (empty at creation), and must not wipe a stamp that a later
                # ``header_update`` set — else the end-of-session flush loop recurs
                # on any index rebuild/re-tail (gap).
                if name == "last_memory_flush_at" and not v:
                    continue
                vals[name] = _coerce_header_value(name, v)
        vals["session_hash"] = sid
        if "footer_state_json" not in vals and vals.get("agent_name"):
            activation = {
                "agent": vals["agent_name"],
                "model": str(vals.get("model") or ""),
                "timestamp": str(event.get("ts") or event.get("created_at") or ""),
            }
            vals["footer_state_json"] = json.dumps(
                {
                    "current": activation,
                    "sequence": [activation],
                }
            )
        vals.setdefault("schema_version", str(event.get("v") or "1"))
        # Cursor advances to the end of THIS line so a re-tail resumes
        # past it.
        vals["indexed_bytes"] = int(end_offset)
        vals["updated_at"] = now_ms
        colnames = list(vals.keys())
        placeholders = ",".join("?" for _ in colnames)

        # ``parent_session_id`` MUST stay in the ON CONFLICT update set so the
        # subagent back-link gets POPULATED when ``session_started`` is projected
        # onto a pre-existing row. Excluding it used to drop the parent whenever
        # the row already existed — the exact case ``rebuild()`` creates (it
        # pre-inserts a bare ``(session_hash, indexed_bytes)`` row, then
        # re-tails), which silently flattened the orchestrator delegation tree on
        # every full re-index. (Logs carried the parent,
        # sqlite didn't.) It is updated FILL-ONLY: a stored non-empty link is
        # preserved, so even a duplicate/corrupt re-emitted header can never
        # change a good parent — only fill an empty one. (codex 2026-06-23.)
        def _assign(c: str) -> str:
            if c == "title_locked":
                return "title_locked=MAX(sessions_meta.title_locked, excluded.title_locked)"
            if c == "title":
                return "title=CASE WHEN sessions_meta.title_locked=1 THEN sessions_meta.title ELSE excluded.title END"
            if c in ("cost_root_session_id", "cost_ancestor_goal_id"):
                # Immutable once set: fill an empty stored
                # value, never overwrite a populated one.
                return (
                    f"{c}=CASE "
                    f"WHEN sessions_meta.{c} IS NULL OR sessions_meta.{c}='' "
                    f"THEN excluded.{c} ELSE sessions_meta.{c} END"
                )
            if c == "parent_session_id":
                return (
                    "parent_session_id=CASE "
                    "WHEN sessions_meta.parent_session_id IS NULL "
                    "OR sessions_meta.parent_session_id='' "
                    "THEN excluded.parent_session_id "
                    "ELSE sessions_meta.parent_session_id END"
                )
            if c == "session_budget_usd":
                # A replayed session_started carries the create-time 0.
                # Never let that wipe a cap /budget already persisted.
                return (
                    f"{c}=CASE WHEN excluded.{c} IS NULL OR excluded.{c}=0 THEN sessions_meta.{c} ELSE excluded.{c} END"
                )
            return f"{c}=excluded.{c}"

        updates = ",".join(_assign(c) for c in colnames if c not in ("session_hash",))
        con.execute(
            f"INSERT INTO sessions_meta({','.join(colnames)}) "
            f"VALUES ({placeholders}) "
            f"ON CONFLICT(session_hash) DO UPDATE SET {updates}",
            tuple(vals[c] for c in colnames),
        )

    def _apply_header_patch(self, con, sid, patch: dict):
        valid = {name for name, _ in _SESSIONS_META_COLUMNS} - {"session_hash"}
        sets = []
        args = []
        for k, v in patch.items():
            if k not in valid:
                continue
            # A title pin is sticky across stale Session instances and restart
            # hydration. Only an explicit locked rename may replace it.
            if k == "title_locked" and not v:
                continue
            # Never let a header_update clobber a real (turn-derived)
            # title with the in-memory ``untitled-…`` placeholder — the
            # in-memory Session.title may still be the placeholder when a
            # config mutation triggers save(), but the projector already
            # derived the title from the first turn.
            if k == "title" and str(v or "").startswith("untitled-"):
                continue
            if k == "title" and not patch.get("title_locked"):
                sets.append("title=CASE WHEN title_locked=1 THEN title ELSE ? END")
                args.append(_coerce_header_value(k, v))
                continue
            # Identity fields set at session_started — never mutable via
            # header_update (parent_session_id is the subagent back-link).
            if k in ("parent_session_id",):
                continue
            # Cost attribution is IMMUTABLE ONCE SET: a
            # delegated child's ``cost_root_session_id`` / ``cost_ancestor_goal_id``
            # identify the root its spend rolls up to for the child's lifetime. A
            # later ``header_update`` may FILL an empty value but must never
            # re-stamp a populated one, so spend can't be silently re-attributed.
            if k in ("cost_root_session_id", "cost_ancestor_goal_id"):
                sets.append(f"{k}=CASE WHEN {k} IS NULL OR {k}='' THEN ? ELSE {k} END")
                args.append(_coerce_header_value(k, v))
                continue
            # Likewise don't overwrite a known engine/created_at with an
            # empty value (ACP new_session doesn't stamp engine).
            #
            # ``last_memory_flush_at`` is MONOTONIC: it is assigned exactly once
            # (agent.maybe_memory_flush stamps a timestamp; nothing ever resets
            # it to ""), so an incoming empty value is always a STALE in-memory
            # session instance clobbering the durable once-per-session flush
            # gate. Dropping it keeps the gate sticky, which is what makes the
            # end-of-session flush provably at-most-once (2026-07-13 jarvis
            # flush-loop: the stamp went ""→set→""→set→… as a stale instance
            # wiped it, re-arming the idle flush every ~15 min).
            if k in ("engine", "created_at", "agent_name", "last_memory_flush_at") and not v:
                continue
            # 0 means "unset" on a stale in-memory Session. An intentional
            # reset is /budget 0 → persisted $25, never a raw 0 overwrite.
            if k == "session_budget_usd":
                try:
                    incoming = float(v or 0)
                except (TypeError, ValueError):
                    incoming = 0.0
                if incoming <= 0:
                    continue
            sets.append(f"{k}=?")
            args.append(_coerce_header_value(k, v))
        if not sets:
            return
        args.append(sid)
        con.execute(
            f"UPDATE sessions_meta SET {','.join(sets)} WHERE session_hash=?",
            tuple(args),
        )

    def _sync_turn0_title(self, con, sid: str) -> None:
        """Keep the turn-0 row's title in ``turns`` + ``turns_fts`` in sync
        with ``sessions_meta.title`` after a title-changing
        ``header_update``, so the new title is searchable immediately
        (no manual rebuild).

        Only the turn-0 FTS row carries the title (every other row stays
        empty so BM25 is not diluted by repeating the title once per
        turn). FTS5 rows are updated by delete+re-insert at the same
        rowid, matching the rest of the projector. No-op when the turn-0
        row does not exist yet — ``_on_turn_ended`` will stamp the title
        when it lands (and a rebuild re-plays the whole stream, so the
        final title always wins).
        """
        mrow = con.execute("SELECT title FROM sessions_meta WHERE session_hash=?", (sid,)).fetchone()
        title = str(mrow[0] if isinstance(mrow, tuple) else mrow["title"] or "") if mrow is not None else ""
        trow = con.execute(
            "SELECT rowid, title FROM turns WHERE session_hash=? AND turn_idx=0",
            (sid,),
        ).fetchone()
        if trow is None:
            return
        rowid = int(trow[0] if isinstance(trow, tuple) else trow["rowid"])
        cur_title = str(trow[1] if isinstance(trow, tuple) else trow["title"] or "")
        if cur_title == title:
            return  # already in sync
        con.execute("UPDATE turns SET title=? WHERE rowid=?", (title, rowid))
        crow = con.execute("SELECT combined_text FROM turns WHERE rowid=?", (rowid,)).fetchone()
        combined = str(crow[0] if isinstance(crow, tuple) else crow["combined_text"]) if crow is not None else ""
        con.execute("DELETE FROM turns_fts WHERE rowid=?", (rowid,))
        con.execute(
            "INSERT INTO turns_fts(rowid, combined_text, session_hash, turn_idx, title) VALUES (?,?,?,?,?)",
            (rowid, combined, sid, 0, title),
        )

    def _append_live_tool_call(self, con, sid, event, end_offset):
        row = con.execute(
            "SELECT live_tool_calls_so_far FROM sessions_meta WHERE session_hash=?",
            (sid,),
        ).fetchone()
        cur = "[]"
        if row is not None:
            cur = (row["live_tool_calls_so_far"] if not isinstance(row, tuple) else row[0]) or "[]"
        try:
            arr = json.loads(cur)
            if not isinstance(arr, list):
                arr = []
        except (ValueError, TypeError):
            arr = []
        arr.append(
            {
                "call_id": str(event.get("call_id") or ""),
                "name": str(event.get("name") or ""),
                "arguments": event.get("arguments") or {},
                "started_at": str(event.get("started_at") or ""),
            }
        )
        con.execute(
            "UPDATE sessions_meta SET live_tool_calls_so_far=?, live_last_event_offset=? WHERE session_hash=?",
            (json.dumps(arr, ensure_ascii=False), int(end_offset), sid),
        )

    def _on_turn_ended(self, con, sid, event, now_ms):
        ti = int(event.get("turn_idx") or 0)
        self._index_turn_actors(con, sid, ti, event.get("actors"), now_ms)
        live_row = con.execute(
            "SELECT live_turn_idx, last_turn_idx, session_state FROM sessions_meta WHERE session_hash=?",
            (sid,),
        ).fetchone()
        if live_row is not None:
            live_idx = live_row["live_turn_idx"]
            last_idx = int(live_row["last_turn_idx"] or -1)
            cur_state = str(live_row["session_state"] or "")
            # A late terminal event for an older turn must not clear or
            # overwrite a newer live turn. Replayed terminal events on an
            # already-idle/error/closed row are idempotent no-ops.
            if live_idx is not None and int(live_idx) != ti:
                return
            if live_idx is None and last_idx >= ti:
                # ``last_turn_idx`` normally means "highest completed".
                # Ghost live rows (prompting/streaming/ask/… with NULL
                # ``live_turn_idx``) reuse that column for the in-flight
                # turn that never closed. Boot/abort recovery must still
                # fold those to a terminal state.
                if cur_state not in OWN_LIVE_STATES and cur_state != "paused":
                    return
        scratch = self._scratch.pop(ti, {"user": "", "content": ""})
        user_text = scratch.get("user") or _extract_user_text(event.get("user"))
        final = event.get("final_content")
        content = final if final is not None else scratch.get("content", "")
        combined = _combine_turn(user_text, content or "")
        ts = _iso_to_ms(event.get("ts")) or now_ms
        if combined.strip():
            # The turn-0 row ALSO carries the session title (mirrored from
            # sessions_meta.title) so a title-only word is FTS-searchable;
            # every other turn row stays '' so BM25 is not diluted by
            # repeating the title once per turn.
            title = ""
            if ti == 0:
                mrow = con.execute(
                    "SELECT title FROM sessions_meta WHERE session_hash=?",
                    (sid,),
                ).fetchone()
                title = str(mrow[0] if isinstance(mrow, tuple) else mrow["title"] or "") if mrow is not None else ""
            con.execute(
                "INSERT INTO turns(session_hash, turn_idx, combined_text, ts, "
                "title) VALUES (?,?,?,?,?) "
                "ON CONFLICT(session_hash, turn_idx) DO UPDATE SET "
                "combined_text=excluded.combined_text, ts=excluded.ts, "
                "title=excluded.title",
                (sid, ti, combined, ts, title),
            )
            # Keep the FTS row at the SAME rowid as its turns row so the
            # delete below (and every bulk delete) is a by-rowid O(1) hit,
            # never a full-index SCAN on the UNINDEXED session_hash column.
            trow = con.execute(
                "SELECT rowid FROM turns WHERE session_hash=? AND turn_idx=?",
                (sid, ti),
            ).fetchone()
            if trow is not None:
                turn_rowid = int(trow[0] if isinstance(trow, tuple) else trow["rowid"])
                con.execute("DELETE FROM turns_fts WHERE rowid=?", (turn_rowid,))
                con.execute(
                    "INSERT INTO turns_fts(rowid, combined_text, session_hash, turn_idx, title) VALUES (?,?,?,?,?)",
                    (turn_rowid, combined, sid, ti, title),
                )
        # turn_metrics index row. ``ts`` (above) is the event's real time.
        self._index_event(
            con,
            sid,
            ti,
            "turn_metrics",
            {
                "aborted": bool(event.get("aborted")),
                "error": event.get("error"),
                "error_classification": event.get("error_classification") or "",
                "metrics": event.get("metrics") or {},
            },
            ts,
        )
        # A discrete ``error`` event row when the turn errored or was
        # aborted, so health/forensics queries (cmd_health) can count
        # them by kind.
        if event.get("error") or event.get("aborted"):
            self._index_event(
                con,
                sid,
                ti,
                "error",
                {
                    "message": str(event.get("error") or "")[:1000],
                    "classification": event.get("error_classification") or "",
                    "aborted": bool(event.get("aborted")),
                },
                ts,
            )
        # Clear live + advance last_turn_*. A memory-flush turn is
        # housekeeping run after the task went idle: it advances the index
        # but keeps ``last_turn_ts`` on the last real turn, so it never counts
        # as new task activity (Tasks-board recap stamp, ordering, age).
        housekeeping = str(event.get("source") or "") == "memory_flush"
        con.execute(
            "UPDATE sessions_meta SET live_turn_idx=NULL, live_round_idx=NULL, "
            "live_started_at=NULL, live_content_so_far='', live_thinking_so_far='', "
            "live_tool_calls_so_far='[]', last_turn_idx=?, "
            "last_turn_ts=CASE WHEN ? THEN last_turn_ts ELSE ? END "
            "WHERE session_hash=?",
            (ti, int(housekeeping), ts, sid),
        )
        # Discrete session_state transition for the orchestrator:
        # errored/aborted → error; clean turn with live children/jobs →
        # waiting (not idle — Stop/monitor must still see the tree as live);
        # otherwise idle.
        metrics = event.get("metrics") or {}
        outcome = turn_outcome(event)
        retrying = bool(
            event.get("retrying")
            or metrics.get("retrying")
            or str(event.get("finish_reason") or "").lower() == "retrying"
        )
        if retrying:
            end_state = "retrying"
        elif event.get("error") or event.get("aborted"):
            end_state = "error"
        elif self._parent_still_has_work(con, sid):
            end_state = "waiting"
        else:
            end_state = "idle"
        self._set_state(con, sid, end_state, now_ms, outcome=outcome)
        # If THIS session is a child, maybe clear the parent's waiting state.
        self._maybe_clear_parent_waiting(
            con,
            sid,
            now_ms,
            child_outcome=outcome,
        )
        # Backfill the header engine from the turn when the session row
        # was created without one (ACP new_session doesn't stamp engine;
        # per-turn events carry it).
        ev_engine = str(event.get("engine") or "")
        if ev_engine:
            con.execute(
                "UPDATE sessions_meta SET engine=? WHERE session_hash=? AND (engine='' OR engine IS NULL)",
                (ev_engine, sid),
            )

    def _index_turn_actors(self, con, sid, turn_idx, raw_actors, now_ms) -> None:
        """One ``turn_actor`` row per human author of a turn.

        Called BEFORE the live-state early returns in the ``turn_started`` and
        ``turn_ended`` branches: authorship is history, not live state, so a
        replayed / late / closed-session event still records it. Both call
        sites build the payload with ``actor_index_payload`` (fixed key order),
        so the two copies hash identically and INSERT OR IGNORE keeps one row.
        A delegating agent (``via == "agent"``) is not a human author and
        gets no row.
        """
        from sharedagents_session.actors import (
            ACTOR_VIA_AGENT,
            TURN_ACTOR_KIND,
            actor_index_payload,
            normalize_actors,
        )

        for actor in normalize_actors(raw_actors):
            if actor["via"] == ACTOR_VIA_AGENT:
                continue
            self._index_event(
                con,
                sid,
                turn_idx,
                TURN_ACTOR_KIND,
                actor_index_payload(actor),
                now_ms,
            )

    def _index_event(self, con, sid, turn_idx, kind, payload, now_ms):
        pj = json.dumps(payload, ensure_ascii=False)
        con.execute(
            "INSERT OR IGNORE INTO turn_events"
            "(session_hash, turn_idx, kind, ts, payload_json, payload_hash) "
            "VALUES (?,?,?,?,?,?)",
            (sid, int(turn_idx or 0), kind, now_ms, pj, _payload_hash(pj)),
        )


# Live snapshot columns keep only a trailing window — the full content
# is in the JSONL; the live column only serves the realtime poll.
_LIVE_SNAPSHOT_CAP = 16000

_TOOL_ARGS_INDEX_CAP = 1000


def _cap_args(args: Any) -> Any:
    """Cap tool-call arguments for the INDEX column only. The full args
    live verbatim in the JSONL ``tool_call`` event."""
    try:
        dumped = json.dumps(args, ensure_ascii=False)
    except (TypeError, ValueError):
        dumped = str(args)
    if len(dumped) <= _TOOL_ARGS_INDEX_CAP:
        return args
    return {"_truncated": True, "_full_chars": len(dumped), "preview": dumped[:_TOOL_ARGS_INDEX_CAP]}


__all__ = [
    "DEFAULT_MAX_RESULTS",
    "DEFAULT_MIN_SCORE",
    "DEFAULT_SNIPPET_CHARS",
    "EventProjector",
    "RecallResult",
    "SessionIndex",
    "connect_sessions_db",
    "ensure_schema",
    "session_search",
    "subscribe_for_agent",
    "subscribe_for_sessions",
]
