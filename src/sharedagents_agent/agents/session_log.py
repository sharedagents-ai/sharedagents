"""Uniform session log shape across every engine.

The single source of truth for "what does a session JSON file look
like, regardless of which engine ran it." Used for:

* Replay (TUI / Telegram render past sessions identically whether the
  engine was ``local_llm`` or ``claude_cli``).
* Cross-frontend sync (the ``_poll_session_for_external_turns`` path
  in ``sharedagents_tui/app.py``).
* **Fine-tuning datasets** — operator harvests
  ``~/.sharedagents/agents/*/sessions/*.json`` to train successor local
  models. Every turn MUST carry enough structured detail to be a
  training row on its own; no relying on inline ``🔧 <name> <json>``
  text markers to reconstruct tool calls.

Where today's data lives (legacy → new mapping):

* ``Session`` dataclass fields (``sharedagents_session.session.Session``,
  L113–190) — preserved verbatim. New fields are additive.
* ``Session.add_turn`` (session.py L225) — current turn shape preserved;
  three new top-level fields are added (``engine``, ``tool_calls``,
  ``tool_results``) so external-engine turns carry structured detail
  that today is buried in the body's ``🔧`` text markers.
* The delegate runners' ``metrics.source = "claude"`` convention used
  by ``/v1/sessions/{sid}/delegate`` to tag a turn as engine-produced
  (see ``sharedagents_tui/app.py`` L5371) — promoted to a first-class
  ``engine`` field so it doesn't depend on parsing ``metrics``.

Backward compat: every reader of the legacy shape (TUI history,
Telegram resume, ACP session/get) MUST continue to work. New fields
are optional and absent on pre-refactor saves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---- canonical turn record ------------------------------------------


@dataclass(slots=True)
class TurnRecord:
    """One turn in a session, engine-agnostic.

    The exact JSON shape every engine writes. Field-by-field:

    Legacy fields (already present in ``Session.add_turn``):

    * ``user`` — the user prompt. Plain string OR OpenAI-style vision
      multi-part list (preserved from session.py L228).
    * ``thinking`` — the engine's reasoning trace, if any. Empty for
      CLI engines that don't surface reasoning (Codex hides it by
      default; Claude/Antigravity emit it as separate stream blocks).
    * ``content`` — the assistant's reply text. For CLI engines this
      is the concatenated ``TextDelta`` stream; for local LLM it's
      the upstream's content channel.
    * ``started_at`` / ``ended_at`` — ISO timestamps (session.py L233).
    * ``aborted`` / ``error`` / ``in_progress`` — lifecycle flags.
    * ``metrics`` — per-turn numbers. Today carries
      ``prompt_tokens`` / ``completion_tokens`` / ``ttft`` / ``gen_tps``
      / ``wall_time`` for local LLM; ``source="claude"`` plus an
      estimated chars-based proxy for delegate turns. Preserved
      as-is; the new ``engine`` field below makes the ``source``
      tag redundant but we keep both for one release cycle.

    **NEW fields** (additive — engines MUST populate):

    * ``engine`` — ``"local_llm"`` | ``"claude_cli"`` | ``"codex_cli"`` |
      ``"antigravity_cli"``. Engine that produced the assistant content of
      this turn. Allows mid-session agent switches in the future
      without losing per-turn attribution.
    * ``model`` — engine-specific model id. Local LLM: the upstream
      model id (today persisted at session level only; promoted to
      per-turn so a mid-session ``/model`` switch is captured). Claude:
      the CLI's effective model (``claude-sonnet-4-6`` etc., from
      ``init`` event). Codex: the configured model. Antigravity: same.
    * ``engine_session_id`` — the engine's own resume token at the
      time of this turn (Claude session uuid / Codex thread id /
      Antigravity session uuid; empty for local LLM). Recorded so the
      training pipeline can correlate cross-turn state.
    * ``tool_calls`` — structured list of every tool call dispatched
      this turn. Each entry: ``{call_id, name, arguments, started_at}``.
      For local LLM this is what the model emitted; for CLI engines
      it's the inner Read/Bash/TodoWrite/etc. blocks that today live
      only as inline ``🔧 <name> <json>`` markers in ``content``.
    * ``tool_results`` — content-stripped per-result metadata. Each
      entry: ``{call_id, ok, ended_at, content_chars}``. The actual
      ``content`` body is NOT persisted — tool outputs commonly
      carry sensitive material (file bodies, command stdout,
      scraped HTML, API responses) that has no place in a corpus
      destined for fine-tune / AWQ / shared archives. Engines see
      the full content in-memory during the turn so model context
      is unaffected; only the on-disk record loses the body.
    * ``permission_mode`` — ``"plan"`` | ``"ask"`` | ``"auto"`` at
      the moment the turn ran. Recorded because the same prompt
      behaves differently under different modes; fine-tuning needs
      to know.
    """

    user: Any
    thinking: str
    content: str
    started_at: str
    ended_at: str
    engine: str
    model: str
    aborted: bool = False
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    in_progress: bool = False
    engine_session_id: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    permission_mode: str = ""

    # ---- additional fine-tune fields ------
    #
    # All default to empty/None so legacy session readers ignore them
    # silently. Each field captures information the fine-tune pipeline
    # would otherwise reconstruct from sparse hints.

    # Working directory at turn time. Critical for code-context
    # fine-tuning (model behavior is heavily cwd-dependent).
    cwd: str = ""

    # Round index within the turn. Local-LLM turns can have many
    # rounds (model → tool → model → tool → final answer); each is
    # ONE record. Setting this lets the trainer reconstruct the
    # multi-round trajectory.
    round_index: int = 0

    # Why the upstream stopped. Sourced from worker
    # ``metrics.finish_reason`` for local LLM; CLI engines synthesize
    # ``"stop"|"tool_calls"|"length"|"error"`` from terminal events.
    finish_reason: str = ""

    # Classified upstream error when the round ended in TurnError.
    # See ``turn_runtime._classify_upstream_error``.
    error_classification: str = ""

    # Approval decisions per tool call:
    # ``[{call_id, decision, reason, source}, ...]``  where
    # ``source ∈ {policy|hook|user|auto}``.
    approval_decisions: list[dict[str, Any]] = field(default_factory=list)
    artifact_references: list[dict[str, Any]] = field(default_factory=list)

    # Raw streamed tool arguments before json_repair (per call_id).
    # The repaired version lives in ``tool_calls[i].arguments``; this
    # preserves the original wire data for trainer truncation studies.
    raw_tool_arguments: dict[str, str] = field(default_factory=dict)

    # In-turn lifecycle audit events.
    compaction_events: list[dict[str, Any]] = field(default_factory=list)
    distill_events: list[dict[str, Any]] = field(default_factory=list)
    nudge_events: list[dict[str, Any]] = field(default_factory=list)
    pause_resume_log: list[dict[str, Any]] = field(default_factory=list)

    # Tool-result truncation flags (list of call_ids that were
    # truncated).
    truncated_tool_results: list[str] = field(default_factory=list)

    # CLI engine: effective argv (sans secrets) for reproducibility,
    # plus the active permission flags. Empty for local LLM.
    cli_argv: list[str] = field(default_factory=list)
    cli_permission_flags: list[str] = field(default_factory=list)

    # Multimodal attachment metadata (mime, size, sha256; raw bytes
    # live in a separate side-store keyed by sha).
    attachments: list[dict[str, Any]] = field(default_factory=list)

    # ---- replay-fidelity fields (debug / fine-tune / AWQ) ----------
    #
    # Added in fix_20260519 round-4. Goal: a single turn record
    # carries everything an offline pipeline needs to (a) recreate
    # the exact upstream request for AWQ activation collection,
    # (b) build a clean ``messages → completion`` SFT pair, or
    # (c) diagnose a regression after the fact without needing
    # access to gateway.log. Tool RESULTS are intentionally NOT
    # persisted — see ``record_turn`` for the rationale.

    # The exact message list (system + preamble + prior turns +
    # live user) the engine sent to the upstream for the FINAL
    # round of this turn. Replay this through the same model to
    # reproduce the activations / completions for AWQ calibration.
    model_input_messages: list[dict[str, Any]] = field(default_factory=list)

    # Per-round metrics dicts (prompt_tokens, completion_tokens,
    # ttft_ms, tps, finish_reason, loop_detected, …) in dispatch
    # order. The top-level ``metrics`` field holds the TERMINAL
    # round's metrics only; this captures the full multi-round
    # trajectory the operator needs to find "loop fired on round
    # 3 of 7" without scanning gateway.log.
    round_metrics: list[dict[str, Any]] = field(default_factory=list)

    # Snapshot of the system prompt that was actually sent to the
    # upstream this turn (sessions.system_prompt may drift across
    # SOUL/AGENTS edits; this pins the per-turn value). The sha256
    # field is the same hash the engine uses for CLI session
    # invalidation — putting it on each turn lets a fine-tune
    # pipeline group turns that shared a system prompt.
    effective_system_prompt: str = ""
    effective_system_prompt_sha256: str = ""

    # The below-cache-boundary preamble blocks (skills / memory
    # recall / pinned facts / todos / compaction summary) injected
    # ahead of the conversation history this turn. SFT pairs need
    # these to reconstruct the model's full input.
    effective_preamble: list[dict[str, Any]] = field(default_factory=list)

    # The actual upstream endpoint URL the request hit. Lets a
    # post-hoc trainer separate sessions that ran on different
    # backends (e.g. an A/B between a 27B and a 70B model).
    upstream_endpoint: str = ""

    # Claude Code hook lifecycle outcomes during this turn —
    # PreToolUse / PostToolUse / PreCompact / Stop / Notification /
    # UserPromptSubmit. Shape: ``[{event, matcher, decision,
    # additional_context_chars, exit_code, fired_at}, …]``. Lets
    # debug pipelines distinguish "model refused to call X" vs
    # "hook denied X" without re-running.
    hook_outcomes: list[dict[str, Any]] = field(default_factory=list)


# ---- session header (top-of-file fields) ----------------------------


@dataclass(slots=True)
class SessionHeader:
    """Top-level fields on a session JSON, engine-agnostic.

    Mirrors ``Session.save`` payload (session.py L300–) plus the new
    attribution fields. Existing readers tolerate extra keys.
    """

    # Legacy (preserved verbatim from Session.save)
    hash: str
    title: str
    created_at: str
    system_prompt: str
    system_mode: str = ""
    workspace_cwd: str = ""
    workspace_origin: str = ""
    model: str = ""
    endpoint: str = ""
    backend: str = ""
    context_window: int = 0
    reasoning_visibility: str = "off"
    tool_output_visible: bool = False
    pinned_facts: list[str] = field(default_factory=list)
    compaction_summary: str = ""
    session_summary: str = ""
    compaction_count: int = 0
    memory_flush_compaction_count: int = -1
    last_memory_flush_at: str = ""

    # NEW attribution fields:
    agent_name: str = ""  # which agent owned this session
    engine: str = ""  # the agent's engine at session-creation time
    # System prompt's stable hash — captured at session creation so a
    # training pipeline can group sessions that shared a SOUL/AGENTS
    # configuration even after the agent's files have evolved.
    system_prompt_sha256: str = ""
    # Whether reasoning traces persist into ``TurnRecord.thinking``.
    # "ALWAYS persisted" can be
    # a privacy footgun. Default True (fine-tune-friendly); operators
    # with regulatory constraints flip this to False per session.
    # Live reasoning rendering in the TUI is governed independently by
    # ``reasoning_visibility``.
    persist_reasoning_traces: bool = True


# ---- engine-side writer surface -------------------------------------


def record_turn(
    session: Any,  # actually sharedagents_session.session.Session
    *,
    user: Any,
    thinking: str,
    content: str,
    started_at: str,
    ended_at: str,
    engine: str,
    model: str,
    sessions_dir: Any | None = None,  # path-like
    engine_session_id: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
    tool_results: list[dict[str, Any]] | None = None,
    permission_mode: str = "",
    aborted: bool = False,
    error: str | None = None,
    metrics: dict[str, Any] | None = None,
    in_progress: bool = False,
    # ---- fine-tune fields ----
    cwd: str = "",
    round_index: int = 0,
    finish_reason: str = "",
    error_classification: str = "",
    approval_decisions: list[dict[str, Any]] | None = None,
    artifact_references: list[dict[str, Any]] | None = None,
    raw_tool_arguments: dict[str, str] | None = None,
    compaction_events: list[dict[str, Any]] | None = None,
    distill_events: list[dict[str, Any]] | None = None,
    nudge_events: list[dict[str, Any]] | None = None,
    pause_resume_log: list[dict[str, Any]] | None = None,
    truncated_tool_results: list[str] | None = None,
    cli_argv: list[str] | None = None,
    cli_permission_flags: list[str] | None = None,
    attachments: list[dict[str, Any]] | None = None,
    # ---- replay-fidelity fields (debug / fine-tune / AWQ) ----------
    # Added in fix_20260519 round-4. Captures the exact context the
    # engine sent to the upstream so SFT / AWQ pipelines can re-
    # construct the prompt verbatim without scraping gateway.log.
    model_input_messages: list[dict[str, Any]] | None = None,
    round_metrics: list[dict[str, Any]] | None = None,
    effective_system_prompt: str = "",
    effective_system_prompt_sha256: str = "",
    effective_preamble: list[dict[str, Any]] | None = None,
    upstream_endpoint: str = "",
    hook_outcomes: list[dict[str, Any]] | None = None,
    physical_turn_id: str = "",
    actors: list[dict[str, Any]] | None = None,
    source: str = "",
) -> None:
    """Single sink for "a turn just finished." Every engine calls this.

    Equivalent to today's ``Session.add_turn`` (session.py L225) plus
    the new attribution fields. Implementation:

    1. Build a :class:`TurnRecord` from the args. Apply the
       ``session.persist_reasoning_traces`` privacy gate: when False,
       ``thinking`` is dropped from the record (the live render still
       sees it; only the on-disk record is sanitized).
    2. Call ``session.add_turn(...)`` with legacy fields so existing
       readers keep working, AND inject the new fields directly onto
       the resulting dict (last element of ``session.turns``) so the
       JSON file carries them too.
    3. ``session.save(force=False)`` per the legacy contract.

    This function MUST be the ONLY place engines persist turn data —
    direct manipulation of ``session.turns`` is forbidden in the new
    architecture, so the on-disk shape stays uniform.

    Defaults for the new fields are conservative (empty / 0 / ""); a
    CLI-engine turn legitimately has e.g. ``round_index=0`` and no
    nudge events.
    """
    # facade — route to the append-only SessionWriter.
    #
    # in_progress flushes update ONLY the live snapshot (sessions_meta
    # live_* columns, SQL-only) so the durable JSONL stays append-only;
    # the terminal call (in_progress=False) appends the turn's events.
    # An in-memory mirror is kept on ``session.turns`` so same-process
    # readers (observe_turn_memory, /context token math) keep working
    # until they're migrated to read sessions.db directly.
    from sharedagents_session.session_writer import (
        get_session_writer,
        release_session_writer,
    )

    if sessions_dir is None:
        from sharedagents_session.session import SESSIONS_DIR

        sdir = SESSIONS_DIR
    else:
        sdir = sessions_dir

    sid = getattr(session, "hash", "") or ""
    eng = (
        engine
        or getattr(session, "agent", "")
        or getattr(session, "agent_name", "")
        or getattr(session, "engine", "")
        or ""
    )
    writer = get_session_writer(sdir, sid, engine=eng, pin=True)
    if eng and writer.engine != eng:
        writer.engine = eng
    try:
        # Keep the writer's privacy gates in sync with the live session
        # (they may have been toggled mid-session via set_config_option).
        writer._persist_thinking = bool(getattr(session, "persist_reasoning_traces", True))
        writer._persist_tool_bodies = bool(getattr(session, "persist_tool_result_bodies", True))

        turn_idx = _resolve_turn_idx_from_index(sdir, sid, session, writer)

        if in_progress:
            # Append the new suffix of the in-flight turn straight to the
            # JSONL (realtime, no SQL). The first flush opens the turn.
            writer.record_stream(
                turn_idx=turn_idx,
                user=user,
                actors=actors,
                source=source,
                attachments=list(attachments or []),
                content=content or "",
                thinking=thinking or "",
                tool_calls=list(tool_calls or []),
                tool_results=list(tool_results or []),
                terminal=False,
                permission_mode=permission_mode,
                cwd=cwd,
                model=model,
                engine_session_id=engine_session_id,
                model_input_messages=list(model_input_messages or []),
                # Per-round metrics for the round that just completed —
                # emitted LIVE as a ``round_ended`` so the stats row streams
                # interleaved with the round's reasoning/tools (not batched
                # at turn end).
                live_round_metrics=list(round_metrics or []),
                effective_system_prompt=effective_system_prompt,
                effective_system_prompt_sha256=effective_system_prompt_sha256,
                effective_preamble=list(effective_preamble or []),
                upstream_endpoint=upstream_endpoint,
                physical_turn_id=physical_turn_id,
                round_idx=round_index,
                agent_name=str(getattr(session, "agent_name", "") or ""),
            )
            return

        rec = (
            writer.record_stream(
                turn_idx=turn_idx,
                user=user,
                actors=actors,
                source=source,
                attachments=list(attachments or []),
                thinking=thinking or "",
                content=content or "",
                terminal=True,
                permission_mode=permission_mode,
                cwd=cwd,
                model=model,
                finish_reason=finish_reason,
                metrics=metrics or {},
                aborted=aborted,
                error=error,
                error_classification=error_classification,
                tool_calls=list(tool_calls or []),
                tool_results=list(tool_results or []),
                engine_session_id=engine_session_id,
                model_input_messages=list(model_input_messages or []),
                round_metrics=list(round_metrics or []),
                effective_system_prompt=effective_system_prompt,
                effective_system_prompt_sha256=effective_system_prompt_sha256,
                effective_preamble=list(effective_preamble or []),
                upstream_endpoint=upstream_endpoint,
                hook_outcomes=list(hook_outcomes or []),
                approval_decisions=list(approval_decisions or []),
                artifact_references=list(artifact_references or []),
                compaction_events=list(compaction_events or []),
                distill_events=list(distill_events or []),
                nudge_events=list(nudge_events or []),
                pause_resume_log=list(pause_resume_log or []),
                physical_turn_id=physical_turn_id,
                round_idx=round_index,
                agent_name=str(getattr(session, "agent_name", "") or ""),
            )
            or {}
        )
    finally:
        release_session_writer(writer)

    # In-memory mirror (scrubbed, ephemeral) for same-process readers.
    from sharedagents_core.secrets_scrubber import scrub_value

    try:
        # ``Session.add_turn`` only appends the durable turn now. Title
        # lifecycle is owned by the detached title agent, whose explicit
        # header_update reaches the projector + live WebUI/TUI streams.
        session.add_turn(
            user=rec.get("user"),
            thinking=rec.get("thinking", ""),
            content=rec.get("content", ""),
            started_at=started_at,
            ended_at=ended_at,
            aborted=aborted,
            error=rec.get("error"),
            metrics=scrub_value(metrics or {}),
            in_progress=False,
        )
        last = session.turns[-1]
        if attachments:
            # Saved-file references beside the typed text; model replay
            # folds them back in (sharedagents_core.attachment_context.model_user).
            last["attachments"] = [dict(a) for a in attachments]
        last["engine"] = eng
        last["model"] = model
        last["permission_mode"] = permission_mode
        last["cwd"] = cwd
        last["source"] = str(source or "")
        last["tool_calls"] = scrub_value(list(tool_calls or []))
        last["tool_results"] = [
            {
                "call_id": str(r.get("call_id") or ""),
                "ok": bool(r.get("ok")),
                "ended_at": str(r.get("ended_at") or ""),
                "content_chars": (
                    len(r.get("content")) if isinstance(r.get("content"), str) else int(r.get("content_chars") or 0)
                ),
                **{k: r[k] for k in ("outcome", "member_outcomes") if k in r},
            }
            for r in (tool_results or [])
            if isinstance(r, dict)
        ]
        # WebUI footer reads last_prompt_tokens / context_window from the
        # session HEADER, not live turn_ended metrics. Local-LLM already
        # stamps last_prompt_tokens in cost.py; ACP/CLI engines (cursor,
        # opencode, kimi, claude, codex) only put the count on
        # TurnDone.metrics, so without this copy the footer sat at 0/200K
        # for every cursor session.
        _stamp_session_context_meter(session, metrics or {})
        try:
            session.save(root=sdir)
        except Exception:  # noqa: BLE001 — header persist is best-effort
            pass
    except Exception:  # noqa: BLE001
        pass

    # Publish ``turn_completed`` so memory providers auto-extract facts.
    #
    # DECISION: neither an aborted turn NOR an errored turn is eligible for
    # memory-fact extraction. This gate is a DELIBERATE, documented exclusion
    # policy — not an incidental byproduct of the abort check. A ``state="error"``
    # turn reaches ``record_turn`` with ``aborted=False`` but a non-empty
    # ``error`` (e.g. ``agent.py`` "engine raised" / last-``TurnError`` paths);
    # an ``aborted``-only gate would let that failed-turn text (partial or failed
    # reasoning, tool errors) seed durable memory. Both ``aborted=True`` and any
    # truthy ``error`` therefore skip the ``turn_completed`` publish that drives
    # ``provider.sync_turn`` — extraction runs ONLY for a clean (ok) turn.
    if not aborted and not error:
        try:
            from sharedagents_agent.agent import AgentConfig
            from sharedagents_agent.turn_runtime import _publish_turn_completed

            agent_name = getattr(session, "agent_name", "") or ""
            if agent_name:
                cfg = AgentConfig.load(agent_name)
                _publish_turn_completed(session, user, content, cfg)
        except Exception:  # noqa: BLE001
            pass
    return


def open_in_progress_turn(
    session: Any,
    *,
    user: Any = "",
    physical_turn_id: str = "",
    permission_mode: str = "",
    cwd: str = "",
    engine: str = "",
    sessions_dir: Any | None = None,
    actors: list | None = None,
    source: str = "",
    stop_guard: Any = None,
    attachments: list | None = None,
) -> int:
    """Admit ``turn_started`` after physical_turn_id is minted, before the model.

    Must not run before admission has a physical identity.
    ``record_stream`` later emits ``round_started`` once (prefill) and must
    not emit a second ``turn_started``.
    """
    from sharedagents_session.session_writer import (
        get_session_writer,
        release_session_writer,
    )

    if sessions_dir is None:
        from sharedagents_session.session import SESSIONS_DIR

        sdir = SESSIONS_DIR
    else:
        sdir = sessions_dir
    eng = (
        engine
        or getattr(session, "agent", "")
        or getattr(session, "agent_name", "")
        or getattr(session, "engine", "")
        or ""
    )
    sid = str(getattr(session, "hash", "") or "")
    writer = get_session_writer(sdir, sid, engine=eng, pin=True)
    if eng and writer.engine != eng:
        writer.engine = eng
    try:
        writer._persist_thinking = bool(getattr(session, "persist_reasoning_traces", True))
        writer._persist_tool_bodies = bool(getattr(session, "persist_tool_result_bodies", True))
        turn_idx = _resolve_turn_idx_from_index(sdir, sid, session, writer)
        writer.open_turn(
            turn_idx=turn_idx,
            user=user,
            actors=actors,
            source=source,
            stop_guard=stop_guard,
            permission_mode=permission_mode,
            cwd=cwd,
            physical_turn_id=physical_turn_id,
            agent_name=str(getattr(session, "agent_name", "") or ""),
            attachments=list(attachments or []) or None,
        )
        return turn_idx
    finally:
        release_session_writer(writer)


def _resolve_turn_idx_from_index(
    sessions_dir: Any,
    sid: str,
    session: Any,
    writer: Any,
) -> int:
    """Return the durable turn index for the next record_stream call.

    ``session.turns`` is an in-memory mirror used for prompt/context math
    and may be compacted, so its length is NOT a reliable index. The
    ``SessionWriter`` is the single serialized appender for this session and
    owns the authoritative monotonic ``turn_idx`` in memory (seeded once from
    the projection on open, advanced per turn). Delegating here keeps the hot
    path free of the per-turn sqlite read that — when the projection lagged —
    let consecutive turns reuse the same ``turn_idx`` and collapse in the
    fold. ``session.turns`` length is only a
    last-ditch fallback if the writer has no counter.
    """
    try:
        return int(writer.next_turn_idx())
    except Exception:  # noqa: BLE001
        return len(getattr(session, "turns", []) or [])


def record_session_creation(
    session: Any,  # sharedagents_session.session.Session
    *,
    agent_name: str,
    engine: str,
    system_prompt: str,
    sessions_dir: Any | None = None,
    parent_session_id: str = "",
    announce_in_parent: bool = True,
) -> None:
    """Stamp attribution fields on a fresh Session AND emit the
    ``session_started`` event (which inserts the sessions_meta row).

    Called by ``Agent.create_session`` once. Computes
    ``system_prompt_sha256``, writes ``agent_name`` + ``engine`` onto the
    in-memory ``Session``, and persists the full initial header to the
    JSONL + sessions.db via the SessionWriter.

    ``parent_session_id`` is the sharedagents session id of the agent that
    spawned this one (set when this session is a subagent). It rides the
    ``session_started`` header so the parent↔child link is durable in
    the JSONL + sessions.db, not just the loose ``delegate_*`` sidecar.

    The ``agent_name`` is validated FIRST against
    :func:`agent_profiles.is_known_agent_name` (delegation roster ∪
    profile identities ∪ built-in agent defaults ∪ on-disk configs) so
    fixture-style phantom names (``rt`` / ``reload-test``) FAIL LOUDLY
    with a ``ValueError`` instead of silently polluting the sessions
    store and rendering as fake agents on the monitor. Empty names are
    the deliberate legacy/unattributed edge and pass through untouched.
    """
    from sharedagents_agent.agent_profiles import is_known_agent_name

    if (agent_name or "").strip() and not is_known_agent_name(agent_name):
        raise ValueError(
            f"refusing to stamp session {session.hash} with unknown agent "
            f"name {agent_name!r}: not a delegation-roster name, not a "
            "resolvable on-disk agent, and not a profile identity. To "
            "create this custom sibling, scaffold "
            "~/.sharedagents/agents/<name>/config.json (factory.load_agent "
            "must resolve the name)."
        )
    session.agent_name = agent_name
    session.engine = engine
    if parent_session_id:
        session.parent_session_id = parent_session_id
    session.system_prompt_sha256 = compute_system_prompt_hash(system_prompt)

    if sessions_dir is None:
        from sharedagents_session.session import SESSIONS_DIR

        sessions_dir = SESSIONS_DIR
    from sharedagents_session.session_writer import get_session_writer

    writer = get_session_writer(sessions_dir, session.hash, engine=engine)
    writer.session_started(_session_header(session))

    # Surface the spawn in the PARENT's log too (spec §3.2 / §12): a
    # ``lifecycle(subagent_spawned)`` event pointing at this child hash lets a
    # monitor render the subagent tree live, from the parent stream alone,
    # with zero joins against sessions_meta. The child↔parent link is already
    # durable via ``parent_session_id`` on the child header; this is the
    # parent-side counterpart. Advisory — never fail child creation if the
    # parent log write hiccups.
    # ``announce_in_parent=False`` for hidden utility children (title): the
    # parent<->child link is already durable in the child header, and the live
    # ``subagent_spawned`` line only makes the monitor's counter flicker.
    if parent_session_id and announce_in_parent:
        try:
            parent_writer = get_session_writer(sessions_dir, parent_session_id)
            parent_writer.lifecycle(
                kind="subagent_spawned",
                payload={
                    "child_session_id": session.hash,
                    "child_agent_name": agent_name,
                    "child_engine": engine,
                },
            )
        except Exception:  # noqa: BLE001
            pass


def _session_header(session: Any) -> dict[str, Any]:
    """Build the materialised-header dict (all sessions_meta config /
    identity fields) from a live ``Session`` for ``session_started`` and
    ``header_update`` events."""
    return {
        "session_hash": session.hash,
        "created_at": getattr(session, "created_at", "") or "",
        "agent_name": getattr(session, "agent_name", "") or "",
        "engine": getattr(session, "engine", "") or "",
        "parent_session_id": getattr(session, "parent_session_id", "") or "",
        "workspace_cwd": getattr(session, "workspace_cwd", "") or "",
        "workspace_origin": getattr(session, "workspace_origin", "") or "",
        "title": getattr(session, "title", "") or "",
        "title_refinements": int(getattr(session, "title_refinements", 0) or 0),
        "model": getattr(session, "model", "") or "",
        "endpoint": getattr(session, "endpoint", "") or "",
        "backend": getattr(session, "backend", "") or "",
        "context_window": int(getattr(session, "context_window", 0) or 0),
        "system_mode": getattr(session, "system_mode", "") or "",
        "reasoning_visibility": getattr(session, "reasoning_visibility", "off") or "off",
        "tool_output_visible": bool(getattr(session, "tool_output_visible", False)),
        "persist_reasoning_traces": bool(getattr(session, "persist_reasoning_traces", True)),
        "persist_tool_result_bodies": bool(getattr(session, "persist_tool_result_bodies", True)),
        "system_prompt": getattr(session, "system_prompt", "") or "",
        "system_prompt_sha256": getattr(session, "system_prompt_sha256", "") or "",
        "pinned_facts": list(getattr(session, "pinned_facts", []) or []),
        "compaction_count": int(getattr(session, "compaction_count", 0) or 0),
        "compaction_summary": getattr(session, "compaction_summary", "") or "",
        "session_summary": getattr(session, "session_summary", "") or "",
        "memory_flush_compaction_count": int(getattr(session, "memory_flush_compaction_count", -1) or -1),
        "last_memory_flush_at": getattr(session, "last_memory_flush_at", "") or "",
        "last_prompt_tokens": int(getattr(session, "last_prompt_tokens", 0) or 0),
    }


# ---- helpers --------------------------------------------------------


def _stamp_session_context_meter(
    session: Any,
    metrics: dict[str, Any],
) -> None:
    """Copy this turn's occupancy onto the persisted session header.

    The WebUI footer seeds ``used / max`` from ``session.last_prompt_tokens``
    and ``session.context_window`` (see ``syncContextMeterFromHeader``).
    Engines that only report occupancy on ``TurnDone.metrics`` (ACP/CLI)
    never reached that header, so the meter stayed ``0/<window>``.
    """
    if not isinstance(metrics, dict):
        return
    try:
        pt = int(metrics.get("prompt_tokens") or 0)
    except (TypeError, ValueError):
        pt = 0
    try:
        cw = int(metrics.get("context_window") or 0)
    except (TypeError, ValueError):
        cw = 0
    if pt > 0:
        prev = int(getattr(session, "last_prompt_tokens", 0) or 0)
        # Estimated occupancy is a running total. A process restart used to
        # reset the in-memory char counter, so the next turn stamped 580
        # over 15889 and the WebUI footer jumped to 580/200K. Never let an
        # estimate shrink the header; real usage (tokens_estimated=False)
        # may drop after compaction.
        if bool(metrics.get("tokens_estimated")) and prev > 0 and pt < prev:
            pt = prev
        session.last_prompt_tokens = pt
    if cw > 0:
        session.context_window = cw


def make_tool_call_record(
    call_id: str,
    name: str,
    arguments: dict[str, Any],
    started_at: str,
) -> dict[str, Any]:
    """Canonical ``tool_calls[i]`` row.

    Engines build these from their stream parsers — Claude's
    ``content_block_start`` + ``content_block_stop`` pair, Codex's
    ``item.started`` with ``command_execution``, Antigravity's ``tool_use``
    event. Local LLM builds them from the model's emitted tool calls.
    """
    return {
        "call_id": str(call_id),
        "name": str(name),
        "arguments": dict(arguments or {}),
        "started_at": str(started_at),
    }


def make_tool_result_record(
    call_id: str,
    ok: bool,
    content: str,
    ended_at: str,
    outcome: str | None = None,
    member_outcomes: tuple[str | None, ...] = (),
    images: tuple[dict, ...] = (),
) -> dict[str, Any]:
    """Canonical ``tool_results[i]`` row. Pairs with the call by id.

    ``images`` (``image_url`` parts a tool showed the model) is present only
    when non-empty, so every other row keeps its shape.
    """
    return {
        "call_id": str(call_id),
        "ok": bool(ok),
        "content": str(content or ""),
        "ended_at": str(ended_at),
        **({"outcome": outcome} if outcome else {}),
        **({"member_outcomes": list(member_outcomes)} if member_outcomes else {}),
        **({"images": [dict(p) for p in images]} if images else {}),
    }


def compute_system_prompt_hash(system_prompt: str) -> str:
    """SHA-256 of the system prompt. Stable across sessions that share
    a SOUL/AGENTS/TOOLS configuration. Empty input → empty string so a
    not-yet-composed session has a sentinel rather than the sha of the
    empty string."""
    if not system_prompt:
        return ""
    import hashlib

    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()


# ---- session → legacy dict rendering --------------------------------


def _loads_list(v: Any) -> list:
    """Parse a JSON-array TEXT column (e.g. sessions_meta.pinned_facts)
    into a list; tolerate already-list / null / malformed values."""
    if isinstance(v, list):
        return v
    if not v:
        return []
    try:
        import json as _json

        out = _json.loads(v)
        return out if isinstance(out, list) else []
    except (ValueError, TypeError):
        return []


def _loads_json_dict(v: Any) -> dict:
    """Parse a JSON-object TEXT column (e.g. sessions_meta.compaction_last_stats)
    into a dict; tolerate already-dict / null / malformed values."""
    if isinstance(v, dict):
        return v
    if not v:
        return {}
    try:
        import json as _json

        out = _json.loads(v)
        return out if isinstance(out, dict) else {}
    except (ValueError, TypeError):
        return {}


def agent_session_to_dict(sess: Any) -> dict[str, Any]:
    """Render a :class:`Session` (header + folded turns) to the legacy
    ``<sid>.json``-shaped dict that pre-readers expect."""
    return {
        "hash": getattr(sess, "hash", ""),
        "title": getattr(sess, "title", ""),
        "created_at": getattr(sess, "created_at", ""),
        "system_prompt": getattr(sess, "system_prompt", ""),
        "system_mode": getattr(sess, "system_mode", ""),
        "model": getattr(sess, "model", ""),
        "endpoint": getattr(sess, "endpoint", ""),
        "backend": getattr(sess, "backend", ""),
        "context_window": int(getattr(sess, "context_window", 0) or 0),
        "workspace_cwd": getattr(sess, "workspace_cwd", ""),
        "workspace_origin": getattr(sess, "workspace_origin", ""),
        "agent_name": getattr(sess, "agent_name", ""),
        "engine": getattr(sess, "engine", ""),
        "pinned_facts": list(getattr(sess, "pinned_facts", []) or []),
        "compaction_count": int(getattr(sess, "compaction_count", 0) or 0),
        "compaction_summary": getattr(sess, "compaction_summary", ""),
        "session_summary": getattr(sess, "session_summary", ""),
        "turns": list(getattr(sess, "turns", []) or []),
    }
