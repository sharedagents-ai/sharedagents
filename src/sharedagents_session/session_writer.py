"""Write side of the session log — the single sink for every engine.

``SessionWriter`` owns, for one session:

* the append-only ``<sid>.jsonl`` event log,
* the per-session ``.<sid>.lock`` flock,
* a connection to the shared ``sessions.db`` (materialised header +
  live snapshot + FTS index).

Every engine and every header mutator goes through this class. Direct
``Session.add_turn`` / ``Session.save`` / raw ``path.write_text`` on a
session file is forbidden post-cutover (the spec, §7).

Per-event critical section (the flock serialises writers; the kernel
serialises nothing on its own for >PIPE_BUF appends, so the flock IS
the atomicity primitive):

    acquire .<sid>.lock LOCK_EX   (fstat-verify inode; reopen if swept)
      os.write(jsonl_fd, "<crc8> <json>\\n")   # single write(2)
      if fsync_mode == always or boundary-event: os.fsync(jsonl_fd)
      BEGIN IMMEDIATE on sessions.db
        EventProjector.apply(con, sid, event, end_offset)   # §8 projection
      COMMIT
      fsync(sessions_dir_fd)        # parent-dir fsync for create/rename durability
    release lock

JSONL write+fsync happens BEFORE the SQL commit, so a crash leaves the
JSONL ahead of ``sessions_meta`` but never behind; ``reconcile_on_open``
repairs the db from the JSONL on the next writer start.

Threading: synchronous, guarded by a re-entrant in-process lock plus the
flock. Delta events (``text_delta`` / ``thinking_delta``) are debounced
in-memory and flushed when (a) ``flush_ms`` has elapsed since the first
buffered chunk, (b) the turn/round changes, or (c) any non-delta event
arrives. (The spec's separate writer-thread + bounded queue for
backpressure is a documented deferral — today's runtime already does
synchronous fsync in the loop, so this matches the existing threading
model; the debounce already bounds write frequency.)
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sharedagents_core import oscompat
from sharedagents_core.secrets_scrubber import scrub_value
from sharedagents_session.ask_pending import is_ask_tool_name
from sharedagents_session.session_index import EventProjector, SessionIndex
from sharedagents_session.session_reader import SCHEMA_VERSION, encode_line

# Events that always fsync the JSONL even in ``boundaries`` mode —
# anything whose loss would break recovery or lose a structured signal.
_FSYNC_BOUNDARY_EVENTS = frozenset(
    {
        "session_started",
        "turn_started",
        "round_started",
        "round_ended",
        # ``tool_result_meta`` is the STRUCTURAL tool-result record (call_id/ok/
        # content_chars) that is emitted even when the raw ``tool_result`` body is
        # gated off by ``persist_tool_result_bodies=False``; losing it would drop a
        # structured training/recovery signal, so it always fsyncs regardless of
        # mode.
        "turn_ended",
        "turn_retry",
        "tool_call",
        "tool_result",
        "tool_result_meta",
        "header_update",
        "session_closed",
        "loop_detected",
        "nudge",
        "distill",
        "compaction",
        "pause_resume",
        "approval_decision",
        "artifact_reference",
        "hook_outcome",
        "subagent_spawned",
        "subagent_linked",
        "subagent_done",
        # A background delegation's terminal result: losing it would leave a
        # consumer (Jarvis / a front end) waiting forever, so fsync at emit.
        "subagent_result",
        # The follow-up queue is also a durable acceptance/consumption ledger.
        # In particular, queue_msg_added is the background-result receipt returned
        # to a detached runner. Every transition must reach stable storage so a
        # restart neither loses an acknowledged result nor resurrects one whose
        # removal/injection had already completed.
        "queue_msg_added",
        "queue_msg_updated",
        "queue_msg_removed",
        "queue_msg_injected",
        "goal_auto_turn_started",
        # A user Stop gates every automatic turn start until the next prompt.
        "operator_stop",
    }
)

# Events whose session_state transition must be visible to monitoring
# endpoints (ACP bridge, TUI task list) immediately — not after the
# debounced bus subscriber fires. The writer projects these inline
# (via SessionIndex.tail) after the JSONL write.
_IMMEDIATE_PROJECTION_EVENTS = frozenset(
    {
        "session_started",  # → idle
        "turn_started",  # → prompting
        "round_started",  # → prefill (model-call boundary)
        "turn_ended",  # → idle / error / waiting / retrying
        "turn_retry",  # → retrying (between-attempt sleep)
        "session_closed",  # → closed
        "compaction",  # → compacting / idle / waiting
        "pause_resume",  # → paused / idle / waiting
        "agent_switched",  # → sessions_meta.agent_name + footer_state.current
    }
)


def _needs_immediate_projection(event: dict) -> bool:
    """Boundary events, plus compacting/ask tool_call phase changes."""
    et = event.get("type")
    if et in _IMMEDIATE_PROJECTION_EVENTS:
        return True
    if et != "tool_call":
        return False
    name = str(event.get("name") or "")
    return name == "compact_context" or is_ask_tool_name(name)


def _now_iso_ms() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _now_ms() -> int:
    return int(time.time() * 1000)


class SessionWriter:
    def __init__(
        self,
        sessions_dir: Path,
        sid: str,
        *,
        engine: str = "",
        fsync_mode: str | None = None,
        flush_ms: int | None = None,
    ) -> None:
        self.sessions_dir = Path(sessions_dir)
        self.sid = str(sid)
        self.engine = engine
        self.jsonl_path = self.sessions_dir / f"{self.sid}.jsonl"
        self.lock_path = self.sessions_dir / f".{self.sid}.lock"
        self.fsync_mode = (fsync_mode or os.environ.get("SHAREDAGENTS_LOG_FSYNC", "boundaries")).strip().lower()
        try:
            self.flush_ms = int(
                flush_ms if flush_ms is not None else os.environ.get("SHAREDAGENTS_LOG_FLUSH_MS", "100")
            )
        except (TypeError, ValueError):
            self.flush_ms = 100

        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        # Ensure the schema exists / is migrated before any write.
        # SessionIndex now takes the sessions_dir directly (the unified
        # ``~/.sharedagents/sessions/`` for every agent) — the old per-agent
        # ``agent_home`` signature has been retired.
        self._index = SessionIndex(self.sessions_dir)
        self._projector = EventProjector()

        self._rlock = threading.RLock()
        self._jsonl_fd = -1
        self._lock_fd = -1
        self._dir_fd = -1
        self._dir_synced = False  # parent-dir fsync only needed on create
        self._poison = False
        # Current turn's user prompt, cached from turn_started so
        # turn_ended ALWAYS carries it — the FTS combined_text then
        # pairs user+content even when a fresh projector (post-restart
        # tail) has no scratch for this turn.
        self._cur_user: Any = ""
        self._cur_attachments: list = []
        # Current turn's permission mode, cached from turn_started so the
        # terminal turn_ended record self-describes the mode too — the
        # fine-tune pipeline wants permission_mode on EVERY TurnRecord
        # (see agents/session_log.TurnRecord), not only inferred from the
        # turn_started event during reconstruction.
        self._cur_permission_mode: str = ""
        self._cur_actors: list[dict[str, str]] = []
        self._cur_source: str = ""
        self._current_physical_turn_id: str = ""
        self._terminated_physical_turn_ids: set[str] = set()
        self._terminal_turns_seen: set[int] = set()
        # Authoritative, in-memory turn index. The writer is the single
        # serialized appender for this session (flock), so it — not a
        # possibly-lagging sqlite projection — owns the next durable
        # ``turn_idx``. Seeded once from the log/projection on (re)open
        # (``_seed_turn_state``), then advanced purely in memory. This
        # avoids the per-turn sqlite read that let consecutive turns reuse
        # a stale ``last_turn_idx`` and collapse in the fold.
        self._max_turn_seen: int = -1
        # Highest turn whose terminal ``turn_ended`` is already on disk —
        # used to drop a duplicate finalize WITHOUT a full-file rescan.
        self._last_terminal_turn: int = -1
        # Lease counter (guarded by ``_WRITERS_LOCK``). A caller about to
        # resolve a turn index and append to this writer pins it so the LRU
        # cannot evict + close it in the window BEFORE ``_open_turn`` is set
        # — a TOCTOU that would otherwise diverge the registry from the
        # writer the caller still holds (checkout -> open eviction race).
        self._inuse: int = 0
        # Incremental-stream state for the facade. The facade hands us
        # the FULL accumulated content/thinking/tool lists on each flush;
        # we diff against what's already on disk and append only the new
        # suffix as a JSONL delta — realtime streaming straight to the
        # file, no SQL on the hot path.
        self._open_turn: int | None = None
        self._emitted_content_len = 0
        self._emitted_thinking_len = 0
        self._emitted_tool_calls = 0
        self._emitted_tool_results = 0
        # Round indices for which a ``round_ended`` was already emitted
        # (live, at the round boundary). Lets the terminal flush skip
        # re-emitting them so per-round stats stream INTERLEAVED with the
        # round's reasoning/tools instead of batching at turn end.
        self._emitted_round_indices: set[int] = set()
        # Round indices for which ``round_started`` was already emitted
        # so ``open_turn`` + first ``record_stream`` cannot double-fire it.
        self._emitted_round_started: set[int] = set()

        # Privacy gates — refreshed from sessions_meta on first use or
        # set by session_started's header.
        self._persist_thinking = True
        self._persist_tool_bodies = True
        self._gates_loaded = False

        # Delta debounce buffers.
        self._buf_content: list[str] = []
        self._buf_thinking: list[str] = []
        self._buf_turn: int | None = None
        self._buf_round: int | None = None
        self._buf_started_ms: int | None = None

    # ---- public API: lifecycle --------------------------------------

    def session_started(self, header: dict[str, Any]) -> None:
        """Stamp the initial header. Writes the ``session_started`` event
        (full header, self-describing) and INSERTs the sessions_meta row.
        """
        hdr = dict(header or {})
        self.engine = self.engine or str(hdr.get("engine") or "")
        self._persist_thinking = bool(hdr.get("persist_reasoning_traces", True))
        self._persist_tool_bodies = bool(hdr.get("persist_tool_result_bodies", True))
        self._gates_loaded = True
        ev = {"type": "session_started"}
        ev.update(hdr)
        self._emit(ev)

    def record_sensitive_egress_flag(self, flag: dict[str, Any]) -> None:
        """Durably archive a pre-egress sensitive-data flag on this session.

        Positions / PII may accompany a genuinely-needed third-party delegate
        payload, but the owner must be flagged BEFORE the call proceeds and the
        flag itself must be recoverable from the durable log
        (archival discipline). The record names only the
        sensitive-data CLASS, never a value. fsync'd so it is on disk before the
        delegate turn starts.
        """
        ev = dict(flag or {})
        ev["type"] = "sensitive_egress_flag"
        self._emit(ev, fsync=True)

    def turn_started(
        self,
        *,
        turn_idx: int,
        user: Any,
        permission_mode: str = "",
        cwd: str = "",
        attachments: list | None = None,
        agent_name: str = "",
        actors: list | None = None,
        source: str = "",
        stop_guard: Any = None,
    ) -> None:
        self._flush_deltas()
        self._cur_user = user
        # Saved-file references, kept apart from the typed ``user`` text so
        # transcripts show what the person wrote; replay adds them back.
        self._cur_attachments = list(attachments or [])
        self._cur_permission_mode = permission_mode or ""
        from sharedagents_session.actors import normalize_actors

        # Human author(s) + non-user origin of this turn.
        self._cur_actors = normalize_actors(actors)
        self._cur_source = "" if str(source or "user") == "user" else str(source)
        self._max_turn_seen = max(self._max_turn_seen, int(turn_idx))
        agent = agent_name or self.engine or getattr(self, "_agent_name", "") or ""
        ev: dict[str, Any] = {
            "type": "turn_started",
            "turn_idx": int(turn_idx),
            "agent_name": agent,
            "user": user,
            "permission_mode": permission_mode or "",
            "cwd": cwd or "",
        }
        # Writer identity: a boot sweep in another process closes this turn
        # only once this process is gone (see sharedagents_session.turn_runner).
        from sharedagents_session.turn_runner import runner_stamp

        ev["runner"] = runner_stamp()
        if attachments:
            ev["attachments"] = attachments
        # Omitted when empty so legacy / non-web turns stay byte-identical.
        if self._cur_actors:
            ev["actors"] = [dict(a) for a in self._cur_actors]
        if self._cur_source:
            ev["source"] = self._cur_source
        self._emit(ev, stop_guard=stop_guard)

    def open_turn(
        self,
        *,
        turn_idx: int,
        user: Any,
        permission_mode: str = "",
        cwd: str = "",
        physical_turn_id: str = "",
        attachments: list | None = None,
        agent_name: str = "",
        actors: list | None = None,
        source: str = "",
        stop_guard: Any = None,
    ) -> None:
        """Admit a turn before the first model token (Codex: after identity mint).

        Emits ``turn_started`` only. ``round_started`` (→ prefill) is the
        model-call boundary and stays on the first ``record_stream`` / engine
        ``TurnStarted`` flush.
        """
        with self._rlock:
            pid = str(physical_turn_id or "")
            if pid and pid in self._terminated_physical_turn_ids:
                return
            if self._open_turn == int(turn_idx):
                if physical_turn_id and not self._current_physical_turn_id:
                    self._current_physical_turn_id = str(physical_turn_id)
                return
            if self._open_turn is not None:
                try:
                    self.turn_ended(
                        turn_idx=int(self._open_turn),
                        aborted=True,
                        error="turn orphaned by next turn_started",
                        error_classification="turn_orphaned",
                    )
                except Exception:  # noqa: BLE001
                    pass
            self._open_turn = int(turn_idx)
            self._current_physical_turn_id = str(physical_turn_id or "")
            self._emitted_content_len = 0
            self._emitted_thinking_len = 0
            self._emitted_tool_calls = 0
            self._emitted_tool_results = 0
            self._emitted_round_indices = set()
            self._emitted_round_started = set()
            try:
                self.turn_started(
                    turn_idx=int(turn_idx),
                    user=user,
                    permission_mode=permission_mode,
                    cwd=cwd,
                    attachments=attachments,
                    agent_name=agent_name,
                    actors=actors,
                    source=source,
                    stop_guard=stop_guard,
                )
            except Exception:
                # Rejected admission did not open a durable turn. Cleanup
                # must not manufacture a terminal or an unguarded fallback.
                self._open_turn = None
                self._current_physical_turn_id = ""
                raise

    def round_started(
        self,
        *,
        turn_idx: int,
        round_idx: int,
        model_input: list | None = None,
        effective_system_prompt: str = "",
        effective_system_prompt_sha256: str = "",
        effective_preamble: list | None = None,
        upstream_endpoint: str = "",
        model: str = "",
        engine_session_id: str = "",
    ) -> None:
        self._flush_deltas()
        self._emit(
            {
                "type": "round_started",
                "turn_idx": int(turn_idx),
                "round_idx": int(round_idx),
                "model_input": _strip_tool_msg_bodies(model_input or []),
                "effective_system_prompt": effective_system_prompt or "",
                "effective_system_prompt_sha256": effective_system_prompt_sha256 or "",
                "effective_preamble": list(effective_preamble or []),
                "upstream_endpoint": upstream_endpoint or "",
                "model": model or "",
                "engine_session_id": engine_session_id or "",
            }
        )

    def text_delta(self, *, turn_idx: int, round_idx: int, chunk: str) -> None:
        self._buffer_delta("content", turn_idx, round_idx, chunk)

    def thinking_delta(self, *, turn_idx: int, round_idx: int, chunk: str) -> None:
        """Buffer a reasoning-trace delta; drop it when persistence is off."""
        if not self._persist_thinking:
            return
        self._buffer_delta("thinking", turn_idx, round_idx, chunk)

    def tool_call(
        self,
        *,
        turn_idx: int,
        round_idx: int,
        call_id: str,
        name: str,
        arguments: dict | None = None,
        raw_arguments: str = "",
        started_at: str = "",
    ) -> None:
        self._flush_deltas()
        self._emit(
            {
                "type": "tool_call",
                "turn_idx": int(turn_idx),
                "round_idx": int(round_idx),
                "call_id": str(call_id),
                "name": str(name),
                "arguments": dict(arguments or {}),
                "raw_arguments": str(raw_arguments or ""),
                "started_at": started_at or "",
            }
        )

    def tool_result(
        self,
        *,
        turn_idx: int,
        round_idx: int,
        call_id: str,
        ok: bool,
        content: str = "",
        content_chars: int | None = None,
        ended_at: str = "",
        outcome: str | None = None,
        member_outcomes: tuple[str | None, ...] = (),
        images: tuple[dict, ...] = (),
    ) -> None:
        self._flush_deltas()
        n_chars = content_chars if content_chars is not None else (len(content) if isinstance(content, str) else 0)
        self._emit(
            {
                "type": "tool_result_meta",
                "turn_idx": int(turn_idx),
                "round_idx": int(round_idx),
                "call_id": str(call_id),
                "ok": bool(ok),
                "ended_at": ended_at or "",
                "content_chars": int(n_chars),
                **({"outcome": outcome} if outcome else {}),
                **({"member_outcomes": list(member_outcomes)} if member_outcomes else {}),
            }
        )
        if self._persist_tool_bodies and content:
            # Images a tool showed the model ride the body event, under the
            # same privacy gate as the body, once — the round/turn
            # ``model_input`` snapshots stub them (``_strip_tool_msg_bodies``).
            self._emit(
                {
                    "type": "tool_result",
                    "turn_idx": int(turn_idx),
                    "round_idx": int(round_idx),
                    "call_id": str(call_id),
                    "content": content,
                    **({"images": [dict(p) for p in images]} if images else {}),
                }
            )

    def lifecycle(
        self, *, kind: str, payload: dict | None = None, turn_idx: int | None = None, round_idx: int | None = None
    ) -> None:
        """Emit a lifecycle event: loop_detected / nudge / distill /
        compaction / pause_resume / approval_decision / hook_outcome."""
        self._flush_deltas()
        ev: dict[str, Any] = {"type": str(kind)}
        if turn_idx is not None:
            ev["turn_idx"] = int(turn_idx)
        if round_idx is not None:
            ev["round_idx"] = int(round_idx)
        ev.update(dict(payload or {}))
        self._emit(ev)

    def round_ended(
        self,
        *,
        turn_idx: int,
        round_idx: int,
        metrics: dict | None = None,
        finish_reason: str = "",
        tokenizer_id: str = "",
    ) -> None:
        self._flush_deltas()
        self._emit(
            {
                "type": "round_ended",
                "turn_idx": int(turn_idx),
                "round_idx": int(round_idx),
                "metrics": dict(metrics or {}),
                "finish_reason": finish_reason or "",
                "tokenizer_id": tokenizer_id or "",
            }
        )

    def turn_ended(
        self,
        *,
        turn_idx: int,
        user: Any = "",
        final_content: str = "",
        final_thinking: str = "",
        aborted: bool = False,
        error: str | None = None,
        error_classification: str = "",
        metrics: dict | None = None,
        permission_mode: str = "",
        engine_session_id: str = "",
        model_input_messages: list | None = None,
        effective_system_prompt: str = "",
        effective_system_prompt_sha256: str = "",
        effective_preamble: list | None = None,
        upstream_endpoint: str = "",
        agent_name: str = "",
        actors: list | None = None,
        source: str = "",
        engine: str = "",
    ) -> dict[str, Any]:
        """Terminal turn event. Returns the materialised turn dict so the
        caller (observe_turn_memory) need not read ``session.turns``.

        ``engine`` stamps this event only (default: the writer's engine);
        an out-of-band closer passes the engine that ran the turn without
        repointing a cached writer the live turn may still be using."""
        self._flush_deltas()
        # Always carry the user prompt (cached from turn_started) so the
        # FTS combined_text pairs user+content even when re-tailed by a
        # fresh projector after a restart.
        eff_user = user if user not in ("", None) else self._cur_user
        # Stamp the permission mode on the terminal record too (cached
        # from turn_started if the caller didn't repeat it) so the
        # durable TurnRecord self-describes the mode it ran under.
        eff_mode = permission_mode or self._cur_permission_mode or ""
        from sharedagents_session.actors import normalize_actors

        # Repeat the author(s) like ``user`` (fresh projector, compact fallback).
        eff_actors = normalize_actors(actors) or list(self._cur_actors)
        eff_source = ("" if str(source or "user") == "user" else str(source)) or self._cur_source
        if not eff_source and self._open_turn is None:
            # Boot/stale-turn recovery uses a fresh writer, whose current
            # source cache is empty. Recover it from this turn's durable
            # start so aborting a memory flush cannot advance task activity.
            # Normal in-process finalization uses the cache above (no scan).
            from sharedagents_session.session_reader import reverse_jsonl

            for prior, _, _ in reverse_jsonl(self.sessions_dir / f"{self.sid}.jsonl"):
                if prior.get("type") not in {"turn_started", "turn_ended"}:
                    continue
                if int(prior.get("turn_idx") or 0) != int(turn_idx):
                    break
                eff_source = str(prior.get("source") or "")
                if eff_source or prior.get("type") == "turn_started":
                    break
        agent = agent_name or self.engine or getattr(self, "_agent_name", "") or ""
        ev: dict[str, Any] = {
            "type": "turn_ended",
            "turn_idx": int(turn_idx),
            "agent_name": agent,
            "user": eff_user,
            "permission_mode": eff_mode,
            "final_content": final_content or "",
            "aborted": bool(aborted),
            "error": error,
            "error_classification": error_classification or "",
            "metrics": dict(metrics or {}),
        }
        if engine:
            ev["engine"] = str(engine)
        if eff_actors:
            ev["actors"] = [dict(a) for a in eff_actors]
        if eff_source:
            ev["source"] = eff_source
        if self._cur_attachments and (user in ("", None) or user == self._cur_user):
            ev["attachments"] = [dict(a) for a in self._cur_attachments]
        # Final-round replay-fidelity fields. The turn was opened (and its
        # ``round_started`` frozen) on the FIRST streaming flush; the authoritative
        # FINAL round's request + the engine's terminal resume token only arrive on
        # this terminal flush, so they ride ``turn_ended``. Emitted only when set so
        # an orphan-close (aborted) or a body-less turn stays lean and never clobbers
        # a good ``round_started`` value in the fold.
        if engine_session_id:
            ev["engine_session_id"] = str(engine_session_id)
        if model_input_messages:
            ev["model_input_messages"] = _strip_tool_msg_bodies(list(model_input_messages))
        if effective_system_prompt:
            ev["effective_system_prompt"] = effective_system_prompt
        if effective_system_prompt_sha256:
            ev["effective_system_prompt_sha256"] = effective_system_prompt_sha256
        if effective_preamble:
            ev["effective_preamble"] = list(effective_preamble)
        if upstream_endpoint:
            ev["upstream_endpoint"] = upstream_endpoint
        if self._persist_thinking:
            ev["final_thinking"] = final_thinking or ""
        self._emit_turn_ended_once(ev)
        self._cur_user = ""
        self._cur_attachments = []
        self._cur_permission_mode = ""
        self._cur_actors = []
        self._cur_source = ""
        # Scrub the returned dict too — it feeds observe_turn_memory,
        # a second on-disk corpus path that must not bypass the
        # scrubber.
        from sharedagents_core.secrets_scrubber import scrub_text, scrub_value

        return {
            "user": scrub_value(eff_user),
            "content": scrub_text(final_content or ""),
            "thinking": scrub_text(final_thinking or "") if self._persist_thinking else "",
            "aborted": bool(aborted),
            "error": scrub_text(error) if error else error,
            "metrics": scrub_value(dict(metrics or {})),
            "engine": self.engine,
        }

    def turn_retry(
        self,
        *,
        retry: int,
        retry_max: int,
        interval_s: float,
        classification: str = "",
        message: str = "",
    ) -> None:
        """Operator-visible between-attempt sleep (session_state=retrying)."""
        self._flush_deltas()
        self._emit(
            {
                "type": "turn_retry",
                "retry": int(retry),
                "retry_max": int(retry_max),
                "interval_s": float(interval_s),
                "classification": classification or "",
                "message": (message or "")[:200],
            }
        )

    def header_update(self, *, patch: dict[str, Any]) -> None:
        """Apply a header mutation: write the ``header_update`` event and
        UPDATE the matching sessions_meta columns (in the projector)."""
        self._flush_deltas()
        # Refresh local privacy gates if they changed.
        if "persist_reasoning_traces" in patch:
            self._persist_thinking = bool(patch["persist_reasoning_traces"])
        if "persist_tool_result_bodies" in patch:
            self._persist_tool_bodies = bool(patch["persist_tool_result_bodies"])
        self._emit({"type": "header_update", "patch": dict(patch or {})})

    def agent_switched(
        self,
        *,
        from_agent: str,
        to_agent: str,
        model: str = "",
        endpoint: str = "",
        backend: str = "",
        reason: str = "",
    ) -> None:
        """Record an in-stream agent switch (spec FR-08).

        The active agent for a session is the ``to_agent`` of the last
        ``agent_switched`` (or the ``agent_name`` of the last ``turn_started``
        when no switch event exists) — derived by scanning this one file.
        """
        payload: dict[str, Any] = {
            "type": "agent_switched",
            "from_agent": str(from_agent or ""),
            "to_agent": str(to_agent or ""),
            "model": str(model or ""),
            "reason": str(reason or ""),
        }
        if endpoint:
            payload["endpoint"] = str(endpoint)
        if backend:
            payload["backend"] = str(backend)
        self._emit(payload)

    def session_closed(self, *, reason: str = "") -> None:
        self._flush_deltas()
        self._emit({"type": "session_closed", "reason": reason or ""})

    def flush_deltas(self) -> None:
        self._flush_deltas()

    def reconcile_on_open(self) -> None:
        """Sync ``sessions_meta`` to the JSONL on (re)open.

        If the process died between a JSONL append and its SQL commit,
        ``sessions_meta`` is behind the file. Replaying the unindexed
        tail through THIS writer's projector both advances the cursor
        and re-seeds the projector's per-turn scratch (so an in-flight
        turn's user prompt survives into the FTS combined_text). Cheap
        no-op when the cursor is already at EOF.
        """
        try:
            self._index.tail(self.sid, projector=self._projector)
        except Exception:  # noqa: BLE001
            pass
        self._seed_turn_state()

    def _seed_turn_state(self) -> None:
        """Seed the in-memory turn counters from the (now-current) projection.

        Called once on (re)open AFTER ``reconcile_on_open``'s tail, so
        ``sessions_meta`` reflects the file. We read it ONCE here instead of
        per turn; thereafter ``_max_turn_seen`` advances purely in memory as
        turns are written. ``last_turn_idx`` is the highest COMPLETED turn;
        ``live_turn_idx`` (if any) is an in-flight turn that has no
        ``turn_ended`` yet.
        """
        try:
            con = self._index._connect()
            try:
                row = con.execute(
                    "SELECT last_turn_idx, live_turn_idx FROM sessions_meta WHERE session_hash=?",
                    (self.sid,),
                ).fetchone()
            finally:
                con.close()
        except Exception:  # noqa: BLE001
            return
        if row is None:
            return
        last = row["last_turn_idx"] if not isinstance(row, tuple) else row[0]
        live = row["live_turn_idx"] if not isinstance(row, tuple) else row[1]
        last_i = int(last) if last is not None else -1
        self._last_terminal_turn = max(self._last_terminal_turn, last_i)
        self._max_turn_seen = max(self._max_turn_seen, last_i)
        if live is not None:
            try:
                self._max_turn_seen = max(self._max_turn_seen, int(live))
            except (TypeError, ValueError):
                pass
        self._seed_terminated_physical_ids()

    def _seed_terminated_physical_ids(self) -> None:
        """If EOF already finalized a UUID, a recreated writer must not reopen it.

        Abort writes ``turn_ended`` then drops the cached writer. The next
        flush constructs a new instance with an empty in-memory set; the
        last complete JSONL row is that ``turn_ended``.
        """
        from sharedagents_session.session_reader import decode_line

        path = self.sessions_dir / f"{self.sid}.jsonl"
        try:
            data = path.read_bytes()
        except OSError:
            return
        if not data:
            return
        end = len(data) - 1 if data.endswith(b"\n") else len(data)
        start = data.rfind(b"\n", 0, end) + 1
        ev = decode_line(data[start:end])
        if not ev or ev.get("type") != "turn_ended":
            return
        pid = str(ev.get("physical_turn_id") or "")
        if pid:
            self._terminated_physical_turn_ids.add(pid)

    def next_turn_idx(self) -> int:
        """Authoritative next ``turn_idx`` for this session.

        Returns the open turn while one is in flight (so continuation flushes
        keep the same index), otherwise one past the highest turn this writer
        has ever seen. Pure in-memory — no sqlite, no file scan — which is
        what stops consecutive turns from colliding on a stale projection
        value and keeps long-session finalization O(1).
        """
        with self._rlock:
            if self._open_turn is not None:
                try:
                    return int(self._open_turn)
                except (TypeError, ValueError):
                    pass
            return self._max_turn_seen + 1

    def current_turn_idx(self) -> int | None:
        """The integer idx of the turn currently OPEN (``turn_started`` already
        written), or ``None`` if no turn is in flight.

        Unlike :meth:`next_turn_idx`, this never speculates a future index — a
        caller that needs to emit a turn-scoped event mid-turn (e.g. a durable
        ``queue_msg_injected``) uses this to ensure it does NOT write before
        the turn's ``turn_started`` lands in the log, which would split the
        fold.
        """
        with self._rlock:
            if self._open_turn is None:
                return None
            try:
                return int(self._open_turn)
            except (TypeError, ValueError):
                return None

    def close(self) -> None:
        with self._rlock:
            # Orphan-close a still-open turn on graceful shutdown so the
            # session record never ends on a bare turn_started (SIGKILL /
            # OOM paths can't run this — the turn-transition orphan-close
            # in append_turn_delta repairs those on the next write, and
            # SI triage's session-active gate covers the rest).
            if self._open_turn is not None:
                try:
                    self.turn_ended(
                        turn_idx=int(self._open_turn),
                        aborted=True,
                        error="turn open at writer close",
                        error_classification="turn_orphaned",
                    )
                except Exception:  # noqa: BLE001
                    pass
                self._open_turn = None
            try:
                self._flush_deltas_locked()
            except Exception:
                pass
            # Close ALL fds incl. the lock fd (a leaked lock fd exhausts the
            # gateway's fd table).
            self._drop_lock_fd()
            for fd_attr in ("_jsonl_fd", "_dir_fd"):
                fd = getattr(self, fd_attr)
                if fd >= 0:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                    setattr(self, fd_attr, -1)

    # ---- internals --------------------------------------------------

    def _buffer_delta(self, channel: str, turn_idx: int, round_idx: int, chunk: str) -> None:
        if not chunk:
            return
        with self._rlock:
            if self._buf_turn != turn_idx or self._buf_round != round_idx:
                self._flush_deltas_locked()
                self._buf_turn = turn_idx
                self._buf_round = round_idx
            if self._buf_started_ms is None:
                self._buf_started_ms = _now_ms()
            if channel == "content":
                self._buf_content.append(chunk)
            else:
                self._buf_thinking.append(chunk)
            if _now_ms() - self._buf_started_ms >= self.flush_ms:
                self._flush_deltas_locked()

    def _flush_deltas(self) -> None:
        with self._rlock:
            self._flush_deltas_locked()

    def _flush_deltas_locked(self) -> None:
        ti, ri = self._buf_turn, self._buf_round
        if self._buf_content:
            delta = "".join(self._buf_content)
            self._buf_content = []
            self._emit(
                {
                    "type": "text_delta",
                    "turn_idx": ti,
                    "round_idx": ri,
                    "delta": delta,
                },
                fsync=(self.fsync_mode == "always"),
            )
        if self._buf_thinking:
            delta = "".join(self._buf_thinking)
            self._buf_thinking = []
            self._emit(
                {
                    "type": "thinking_delta",
                    "turn_idx": ti,
                    "round_idx": ri,
                    "delta": delta,
                },
                fsync=(self.fsync_mode == "always"),
            )
        self._buf_started_ms = None

    def _emit(
        self,
        event: dict[str, Any],
        *,
        fsync: bool | None = None,
        stop_guard: Any = None,
    ) -> int:
        if self._poison:
            raise RuntimeError(f"SessionWriter[{self.sid}] is poisoned")
        event.setdefault("v", SCHEMA_VERSION)
        event.setdefault("ts", _now_iso_ms())
        if event.get("turn_idx") is not None and self._current_physical_turn_id:
            event.setdefault("physical_turn_id", self._current_physical_turn_id)
        if event.get("type") != "session_started":
            event.setdefault("engine", self.engine)
        safe = scrub_value(event)
        from sharedagents_session.jsonl_protocol import prepare_session_event

        safe = prepare_session_event(safe)
        line = encode_line(safe)
        do_fsync = (
            fsync if fsync is not None else (self.fsync_mode == "always" or event.get("type") in _FSYNC_BOUNDARY_EVENTS)
        )
        with self._rlock:
            self._acquire_flock()
            try:
                if stop_guard is not None:
                    stop_guard.check(self.sessions_dir, self.sid)
                jfd = self._open_jsonl()
                _write_all(jfd, line)
                if do_fsync:
                    try:
                        os.fsync(jfd)
                    except OSError as e:
                        self._poison = True
                        raise RuntimeError(f"SessionWriter[{self.sid}] fsync failed: {e}") from e
                end_offset = os.lseek(jfd, 0, os.SEEK_CUR)
                self._fsync_dir()
            finally:
                self._release_flock()
        # The sessions.db index is NOT written on the realtime hot path
        # (per-token SQL was the perf problem). The JSONL is the durable
        # write target + source of truth; sessions.db is a derived index
        # rebuilt by lazily tailing the JSONL — driven by this bus
        # publish (debounced subscriber → SessionIndex.tail) and by
        # read-triggered tail(sid). Published AFTER the flock release so
        # a subscriber that tails immediately sees the committed bytes.
        #
        # EXCEPTION: boundary events that change session_state are
        # projected INLINE here so monitoring endpoints (ACP bridge,
        # TUI task list) see the update immediately — not after the
        # debounced bus subscriber fires. The bus subscriber's later
        # tail() is a no-op (cursor already advanced).
        if _needs_immediate_projection(event):
            try:
                self._index.tail(self.sid, projector=self._projector)
            except Exception:  # noqa: BLE001
                pass  # Best-effort — bus subscriber will catch up
        self._publish_appended()
        return end_offset

    def _terminal_event_exists_locked(self, turn_idx: int) -> bool:
        """True if the log already has a terminal event for ``turn_idx``.

        Caller must hold this session's flock. Normal events may repeat, but a
        second ``turn_ended`` for the same turn makes recovery and frontends
        disagree about the turn outcome.

        In-memory only — no full-file rescan. ``_terminal_turns_seen`` records
        terminals written by THIS writer; ``_last_terminal_turn`` is seeded
        from the projection on (re)open (``_seed_turn_state``) so a terminal
        written before an LRU eviction is still recognised after recreation.
        With the monotonic ``next_turn_idx`` every new turn is strictly
        greater than ``_last_terminal_turn``, so this only ever drops a true
        duplicate finalize (keeps finalization O(1) rather than O(n²)).
        """
        ti = int(turn_idx)
        return ti in self._terminal_turns_seen or ti <= self._last_terminal_turn

    def _emit_turn_ended_once(self, event: dict[str, Any]) -> bool:
        if self._poison:
            raise RuntimeError(f"SessionWriter[{self.sid}] is poisoned")
        event.setdefault("v", SCHEMA_VERSION)
        event.setdefault("ts", _now_iso_ms())
        event.setdefault("engine", self.engine)
        if self._current_physical_turn_id:
            event.setdefault("physical_turn_id", self._current_physical_turn_id)
        safe = scrub_value(event)
        from sharedagents_session.jsonl_protocol import prepare_session_event

        safe = prepare_session_event(safe)
        line = encode_line(safe)
        do_fsync = self.fsync_mode == "always" or event.get("type") in _FSYNC_BOUNDARY_EVENTS
        with self._rlock:
            self._acquire_flock()
            try:
                if self._terminal_event_exists_locked(int(event["turn_idx"])):
                    return False
                jfd = self._open_jsonl()
                _write_all(jfd, line)
                if do_fsync:
                    try:
                        os.fsync(jfd)
                    except OSError as e:
                        self._poison = True
                        raise RuntimeError(f"SessionWriter[{self.sid}] fsync failed: {e}") from e
                os.lseek(jfd, 0, os.SEEK_CUR)
                self._fsync_dir()
                ti = int(event["turn_idx"])
                self._terminal_turns_seen.add(ti)
                self._last_terminal_turn = max(self._last_terminal_turn, ti)
                pid = str(event.get("physical_turn_id") or self._current_physical_turn_id or "")
                if pid:
                    self._terminated_physical_turn_ids.add(pid)
            finally:
                self._release_flock()
        # Inline projection for turn_ended (always a boundary event).
        if _needs_immediate_projection(event):
            try:
                self._index.tail(self.sid, projector=self._projector)
            except Exception:  # noqa: BLE001
                pass  # Best-effort — bus subscriber will catch up
        self._publish_appended()
        return True

    def _publish_appended(self) -> None:
        try:
            from sharedagents_core.event_bus import get_bus

            agent_home = self.sessions_dir.parent if self.sessions_dir.name == "sessions" else self.sessions_dir
            get_bus().publish(
                "session_saved",
                {
                    "agent_home": str(agent_home),
                    "session_hash": self.sid,
                    "path": str(self.jsonl_path),
                },
            )
        except Exception:  # noqa: BLE001
            pass

    def _open_jsonl(self) -> int:
        if self._jsonl_fd < 0:
            existed = self.jsonl_path.exists()
            self._jsonl_fd = os.open(
                self.jsonl_path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND | oscompat.O_CLOEXEC | oscompat.O_BINARY,
                0o600,
            )
            # A parent-dir fsync is only needed to make the file's
            # CREATE durable; pure appends don't change the dirent.
            # (Was fsync'ing the dir per event.)
            if not existed:
                self._dir_synced = False
        return self._jsonl_fd

    def _acquire_flock(self) -> None:
        # Open (lazily) and lock. After acquiring, verify the inode
        # hasn't been swept out from under us; if it has, reopen+relock.
        for _ in range(3):
            if self._lock_fd < 0:
                self._lock_fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT | oscompat.O_CLOEXEC, 0o600)
            oscompat.file_lock(self._lock_fd)
            try:
                disk = os.stat(self.lock_path)
                held = os.fstat(self._lock_fd)
            except FileNotFoundError:
                # Swept while held — drop and retry.
                self._drop_lock_fd()
                continue
            if (disk.st_ino, disk.st_dev) == (held.st_ino, held.st_dev):
                return
            # Inode changed underneath: release, reopen, retry.
            self._drop_lock_fd()
        # Best effort: proceed with whatever we have.
        if self._lock_fd < 0:
            self._lock_fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT | oscompat.O_CLOEXEC, 0o600)
            oscompat.file_lock(self._lock_fd)

    def _drop_lock_fd(self) -> None:
        if self._lock_fd >= 0:
            try:
                oscompat.file_unlock(self._lock_fd)
            except OSError:
                pass
            try:
                os.close(self._lock_fd)
            except OSError:
                pass
            self._lock_fd = -1

    def _release_flock(self) -> None:
        if self._lock_fd >= 0:
            try:
                oscompat.file_unlock(self._lock_fd)
            except OSError:
                pass

    def _fsync_dir(self) -> None:
        # Only needed once, to make the JSONL file's directory entry
        # durable after create. Appends don't touch the dirent.
        if self._dir_synced:
            return
        try:
            if self._dir_fd < 0:
                self._dir_fd = os.open(self.sessions_dir, os.O_RDONLY | oscompat.O_CLOEXEC)
            os.fsync(self._dir_fd)
            self._dir_synced = True
        except OSError:
            pass

    # ---- facade: incremental streaming (used by record_turn) --------

    def record_stream(
        self,
        *,
        turn_idx: int,
        user: Any,
        content: str = "",
        thinking: str = "",
        tool_calls: list | None = None,
        tool_results: list | None = None,
        terminal: bool = False,
        permission_mode: str = "",
        cwd: str = "",
        model: str = "",
        finish_reason: str = "",
        metrics: dict | None = None,
        aborted: bool = False,
        error: str | None = None,
        error_classification: str = "",
        engine_session_id: str = "",
        model_input_messages: list | None = None,
        round_metrics: list | None = None,
        live_round_metrics: list | None = None,
        effective_system_prompt: str = "",
        effective_system_prompt_sha256: str = "",
        effective_preamble: list | None = None,
        upstream_endpoint: str = "",
        hook_outcomes: list | None = None,
        approval_decisions: list | None = None,
        artifact_references: list | None = None,
        compaction_events: list | None = None,
        distill_events: list | None = None,
        nudge_events: list | None = None,
        pause_resume_log: list | None = None,
        physical_turn_id: str = "",
        round_idx: int = 0,
        agent_name: str = "",
        actors: list | None = None,
        source: str = "",
        attachments: list | None = None,
    ) -> dict[str, Any] | None:
        """Append the NEW suffix of an in-flight (or finished) turn.

        The facade hands the full accumulated ``content`` / ``thinking`` /
        tool lists on each flush; this diffs against what's already on
        disk and appends only the new part as JSONL deltas — realtime
        streaming straight to the file, no SQL. On the first flush of a
        turn it opens it (turn_started + round_started). When
        ``terminal`` it emits round_ended + turn_ended (carrying the
        replay-fidelity fields) and returns the materialised turn dict;
        otherwise returns ``None``.
        """
        with self._rlock:
            pid = str(physical_turn_id or self._current_physical_turn_id or "")
            if pid and pid in self._terminated_physical_turn_ids:
                return None
            if self._open_turn != turn_idx:
                # A prior turn is still open: close it as orphaned BEFORE
                # opening the new one. Without this, an abort/crash between
                # flushes leaves an interior turn_started with no
                # turn_ended — the exact mid-flight gap the SI lifecycle
                # auditor flags (5 genuine cases in 2650 production
                # sessions). In-process
                # close on the transition is crash-independent: the next
                # flush of ANY later turn repairs the record.
                if self._open_turn is not None:
                    try:
                        self.turn_ended(
                            turn_idx=int(self._open_turn),
                            aborted=True,
                            error="turn orphaned by next turn_started",
                            error_classification="turn_orphaned",
                        )
                    except Exception:  # noqa: BLE001
                        pass
                # New turn — flush any stale buffers, open it (turn_started
                # only). round_started is the model-call boundary below.
                self._open_turn = turn_idx
                self._current_physical_turn_id = str(physical_turn_id or "")
                self._emitted_content_len = 0
                self._emitted_thinking_len = 0
                self._emitted_tool_calls = 0
                self._emitted_tool_results = 0
                self._emitted_round_indices = set()
                self._emitted_round_started = set()
                self.turn_started(
                    turn_idx=turn_idx,
                    user=user,
                    permission_mode=permission_mode,
                    cwd=cwd,
                    agent_name=agent_name,
                    actors=actors,
                    source=source,
                    attachments=attachments,
                )
            elif physical_turn_id and not self._current_physical_turn_id:
                self._current_physical_turn_id = str(physical_turn_id)

            round_idx = max(0, int(round_idx or 0))
            if round_idx not in self._emitted_round_started:
                self._emitted_round_started.add(round_idx)
                self.round_started(
                    turn_idx=turn_idx,
                    round_idx=round_idx,
                    model_input=model_input_messages or [],
                    effective_system_prompt=effective_system_prompt,
                    effective_system_prompt_sha256=effective_system_prompt_sha256,
                    effective_preamble=effective_preamble or [],
                    upstream_endpoint=upstream_endpoint,
                    model=model,
                    engine_session_id=engine_session_id,
                )

            content = content or ""
            if len(content) > self._emitted_content_len:
                self.text_delta(
                    turn_idx=turn_idx,
                    round_idx=round_idx,
                    chunk=content[self._emitted_content_len :],
                )
                self._emitted_content_len = len(content)
            thinking = thinking or ""
            if self._persist_thinking and len(thinking) > self._emitted_thinking_len:
                self.thinking_delta(
                    turn_idx=turn_idx,
                    round_idx=round_idx,
                    chunk=thinking[self._emitted_thinking_len :],
                )
                self._emitted_thinking_len = len(thinking)

            tcs = list(tool_calls or [])
            for tc in tcs[self._emitted_tool_calls :]:
                if not isinstance(tc, dict):
                    continue
                self.tool_call(
                    turn_idx=turn_idx,
                    round_idx=round_idx,
                    call_id=str(tc.get("call_id") or ""),
                    name=str(tc.get("name") or ""),
                    arguments=tc.get("arguments") or {},
                    raw_arguments=str(tc.get("raw_arguments") or ""),
                    started_at=str(tc.get("started_at") or ""),
                )
            self._emitted_tool_calls = len(tcs)

            trs = list(tool_results or [])
            for tr in trs[self._emitted_tool_results :]:
                if not isinstance(tr, dict):
                    continue
                self.tool_result(
                    turn_idx=turn_idx,
                    round_idx=round_idx,
                    call_id=str(tr.get("call_id") or ""),
                    ok=bool(tr.get("ok")),
                    content=str(tr.get("content") or ""),
                    content_chars=tr.get("content_chars"),
                    ended_at=str(tr.get("ended_at") or ""),
                    outcome=tr.get("outcome"),
                    member_outcomes=tuple(tr.get("member_outcomes") or ()),
                    images=tuple(tr.get("images") or ()),
                )
            self._emitted_tool_results = len(trs)

            # Per-round stats, emitted LIVE at the round boundary so the
            # stats row streams INTERLEAVED right after the round's
            # reasoning/tools — not batched at turn end. Each round is
            # emitted at most once (tracked in ``_emitted_round_indices``)
            # so the terminal flush below never re-emits a round already
            # streamed. The metrics dicts carry ``round_index`` (stamped by
            # the engine's ``_record_round_metrics``); the live caller
            # stamps it from ``RoundComplete.round_idx``.
            def _emit_round_ended(rm: dict) -> None:
                ridx = int(rm.get("round_index", rm.get("round_idx", 0)) or 0)
                if ridx in self._emitted_round_indices:
                    return
                self._emitted_round_indices.add(ridx)
                self.round_ended(
                    turn_idx=turn_idx,
                    round_idx=ridx,
                    metrics=rm,
                    finish_reason=str(rm.get("finish_reason") or ""),
                    tokenizer_id=str(rm.get("tokenizer_id") or ""),
                )

            for rm in live_round_metrics or []:
                if isinstance(rm, dict):
                    _emit_round_ended(rm)

            if not terminal:
                return None

            for rm in round_metrics or []:
                if isinstance(rm, dict):
                    _emit_round_ended(rm)
            if not round_metrics and not self._emitted_round_indices:
                self.round_ended(
                    turn_idx=turn_idx,
                    round_idx=0,
                    metrics=metrics or {},
                    finish_reason=finish_reason,
                )
            for ho in hook_outcomes or []:
                self.lifecycle(kind="hook_outcome", payload=ho, turn_idx=turn_idx)
            from sharedagents_agent.artifact_links import (
                _canonical_artifact_references,
                _canonical_permission_decisions,
            )

            safe_decisions = _canonical_permission_decisions(list(approval_decisions or []))
            safe_artifacts = _canonical_artifact_references(list(artifact_references or []))
            for ad in safe_decisions:
                self.lifecycle(kind="approval_decision", payload=ad, turn_idx=turn_idx)
            for artifact in safe_artifacts:
                self.lifecycle(
                    kind="artifact_reference",
                    payload=artifact,
                    turn_idx=turn_idx,
                )
            for cv in compaction_events or []:
                self.lifecycle(kind="compaction", payload=cv, turn_idx=turn_idx)
            for dv in distill_events or []:
                self.lifecycle(kind="distill", payload=dv, turn_idx=turn_idx)
            for nv in nudge_events or []:
                self.lifecycle(kind="nudge", payload=nv, turn_idx=turn_idx)
            for pv in pause_resume_log or []:
                self.lifecycle(kind="pause_resume", payload=pv, turn_idx=turn_idx)
            rec = self.turn_ended(
                turn_idx=turn_idx,
                user=user,
                final_content=content,
                actors=actors,
                source=source,
                final_thinking=thinking,
                aborted=aborted,
                error=error,
                error_classification=error_classification,
                metrics=metrics or {},
                permission_mode=permission_mode,
                engine_session_id=engine_session_id,
                model_input_messages=model_input_messages or [],
                effective_system_prompt=effective_system_prompt,
                effective_system_prompt_sha256=effective_system_prompt_sha256,
                effective_preamble=effective_preamble or [],
                upstream_endpoint=upstream_endpoint,
            )
            self._open_turn = None
            self._current_physical_turn_id = ""
            return rec

    def write_full_turn(self, **kw: Any) -> dict[str, Any]:
        """One-shot complete turn (tests / non-streaming callers).

        Thin wrapper over :meth:`record_stream` with ``terminal=True``.
        ``thinking`` is accepted as a kwarg alias for the stream's
        thinking channel; ``started_at`` / ``ended_at`` are ignored
        (event ``ts`` is authoritative).
        """
        kw.pop("started_at", None)
        kw.pop("ended_at", None)
        kw["terminal"] = True
        return self.record_stream(**kw)


# ---- process-wide writer registry -----------------------------------
#
# A SessionWriter holds an append fd + per-turn incremental-stream
# state, so it MUST persist across the many ``record_turn`` calls of
# one session (not be rebuilt per
# call — that would lose the projector scratch and reopen fds every
# event). Keyed by the resolved jsonl path.

_WRITERS: OrderedDict[str, SessionWriter] = OrderedDict()
_WRITERS_LOCK = threading.Lock()
# Cap the registry so a long-lived gateway serving many sessions doesn't
# leak fds (each writer holds jsonl + dir + lock fds). LRU-evict + close
# the least-recently-used beyond the cap.
_WRITERS_MAX = int(os.environ.get("SHAREDAGENTS_SESSION_WRITERS_MAX", "128"))


def get_session_writer(
    sessions_dir: Any,
    sid: str,
    *,
    engine: str = "",
    pin: bool = False,
) -> SessionWriter:
    """Return a cached :class:`SessionWriter` for ``sid``, creating one
    on first use. Thread-safe. LRU-bounded.

    ``pin=True`` leases the writer: it cannot be LRU-evicted until a matching
    :func:`release_session_writer`. A turn-writing caller MUST pin across the
    whole resolve-index + append sequence so the writer it holds stays the
    registry's writer (otherwise an eviction in that window orphans the turn
    and skips an index).
    """
    key = str(Path(sessions_dir) / f"{sid}.jsonl")
    evicted: list[SessionWriter] = []
    with _WRITERS_LOCK:
        w = _WRITERS.get(key)
        if w is None:
            w = SessionWriter(sessions_dir, sid, engine=engine)
            # Resuming an existing session: replay any unindexed tail so
            # sessions_meta + the projector scratch are current.
            if w.jsonl_path.exists():
                w.reconcile_on_open()
            _WRITERS[key] = w
        else:
            _WRITERS.move_to_end(key)
            if engine and w.engine != engine:
                w.engine = engine
        if pin:
            w._inuse += 1
        # Evict the OLDEST writer that is neither leased (``_inuse``) nor
        # mid-turn (``_open_turn``). Recreating such a writer would reopen
        # the in-flight turn under a fresh ``turn_idx`` (orphaning the
        # original as a stuck ``in_progress`` turn) and reset the
        # emitted-delta counters (duplicating streamed content). If every
        # writer is busy we let the cache grow transiently rather than
        # corrupt a live turn — leases are held only for the brief append
        # window, so this is bounded.
        while len(_WRITERS) > _WRITERS_MAX:
            victim_key = None
            for k, cand in _WRITERS.items():  # OrderedDict: oldest first
                if k == key:
                    continue
                if cand._inuse > 0 or getattr(cand, "_open_turn", None) is not None:
                    continue
                victim_key = k
                break
            if victim_key is None:
                break
            evicted.append(_WRITERS.pop(victim_key))
    # Close evicted writers OUTSIDE the registry lock (close takes the
    # writer's own rlock + flock; never nest under _WRITERS_LOCK).
    for old in evicted:
        try:
            old.close()
        except Exception:  # noqa: BLE001
            pass
    return w


def release_session_writer(writer: SessionWriter) -> None:
    """Release a lease taken with ``get_session_writer(..., pin=True)``."""
    if writer is None:
        return
    with _WRITERS_LOCK:
        if writer._inuse > 0:
            writer._inuse -= 1


def close_session_writer(sessions_dir: Any, sid: str) -> None:
    """Close + drop a cached writer (e.g. on session close)."""
    key = str(Path(sessions_dir) / f"{sid}.jsonl")
    with _WRITERS_LOCK:
        w = _WRITERS.pop(key, None)
    if w is not None:
        w.close()


def _write_all(fd: int, data: bytes) -> None:
    """Write every byte; loop on short writes (large lines > pipe buf)."""
    mv = memoryview(data)
    while mv:
        n = os.write(fd, mv)
        mv = mv[n:]


def _strip_tool_msg_bodies(msgs: list) -> list:
    """Stub ``role=tool`` message bodies inside a model_input list.

    Tool-result content is persisted (when enabled) via the dedicated
    ``tool_result`` event, not duplicated inside every round's
    model_input — the round event would otherwise carry the full tool
    output N times (once per subsequent round). Preserves the
    structural slot so SFT pipelines see "a tool result of N chars was
    here."

    The ``user`` message that carries tool images (``read`` on a PNG) is
    kept with its captions but its image parts are stubbed for the same
    reason: the image itself is persisted once, on its ``tool_result``.
    """
    from sharedagents_core.attachments import is_image_url_part, is_tool_image_message

    out: list = []
    for m in msgs or []:
        if is_tool_image_message(m):
            out.append(
                {
                    **m,
                    "content": [
                        {"type": "text", "text": "<<TOOL_IMAGE_OMITTED>>"} if is_image_url_part(p) else p
                        for p in m.get("content") or []
                    ],
                }
            )
            continue
        if isinstance(m, dict) and str(m.get("role") or "") == "tool":
            content = m.get("content")
            stub = {
                "role": "tool",
                "content": "<<TOOL_RESULT_OMITTED>>",
                "content_chars": len(content) if isinstance(content, str) else 0,
            }
            if m.get("tool_call_id") is not None:
                stub["tool_call_id"] = m["tool_call_id"]
            if m.get("name") is not None:
                stub["name"] = m["name"]
            out.append(stub)
        else:
            out.append(m)
    return out
