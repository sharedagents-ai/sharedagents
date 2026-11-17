"""Claude CLI delegate runner."""

from __future__ import annotations

import asyncio
import json
import os
import signal
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sharedagents_agent import sandbox_broker
from sharedagents_core.cwdutil import safe_getcwd

from .base import (
    _CLAUDE_MODE_MAP,
    _REASONING_BLOCK_TYPES,
    _REASONING_DELTA_TYPES,
    CliAnswerSink,
    ReasoningTextRelay,
    _claude_effort_flags,
    _claude_error_is_abort,
    _claude_result_context_window,
    _claude_result_is_task_notification,
    _claude_thinking_display_flags,
    _delegate_state_path,
    _format_cli_tool_use,
    _load_tool_utils,
    _log_delegate_error,
    _read_delegate_state,
    _reasoning_signal_chars,
    _resolve_delegate_cwd,
    _write_delegate_state,
    log_event,
)

if TYPE_CHECKING:
    # Bound into this module at call time by ``_load_tool_utils(globals())``;
    # a top-level import would close the tools <-> delegates import cycle.
    from sharedagents_agent.tools.impl._utils import (
        ToolResult,
        _build_delegate_spawn_env,
        _claude_idle_timeout_s,
        _delegate_first_token_timeout_s,
        _delegate_readline,
        _err,
        _feed_stdin,
        _kill_tree,
        _resolve_cli_executable,
        _signal_name,
        _terminate_group,
        _truncate,
    )


def _claude_state_path(
    sa_sid: str | None = None,
    sa_sdir: str | None = None,
) -> Path | None:
    """Resolve the per-sharedagents-session state file for `delegate_claude`.

    Callers (the engine pipeline) pass ``sa_sid`` / ``sa_sdir``
    explicitly so two concurrent CLI turns against the same agent
    can't race on process-wide ``os.environ``. Legacy direct callers
    (no agent context) get an env fallback for back-compat — same
    behavior as before. Returns ``None`` when neither args nor env
    yield a sid/sdir, in which case ``--resume`` is skipped and the
    call starts a fresh Claude session.
    """
    return _delegate_state_path("claude", sa_sid, sa_sdir)


def _read_claude_session_id(
    sa_sid: str | None = None,
    sa_sdir: str | None = None,
) -> str | None:
    sid = _read_delegate_state("claude", sa_sid, sa_sdir).get("claude_session_id")
    return sid if isinstance(sid, str) and sid else None


def _write_claude_session_id(
    claude_sid: str,
    sa_sid: str | None = None,
    sa_sdir: str | None = None,
    usage_totals: dict[str, int] | None = None,
) -> None:
    state: dict[str, Any] = {"claude_session_id": claude_sid}
    if usage_totals:
        state["usage_totals"] = usage_totals
    _write_delegate_state("claude", state, sa_sid, sa_sdir)


# ``result.modelUsage`` field -> the ``usage`` key the engine prices from.
_CLAUDE_USAGE_TOTAL_FIELDS = (
    ("inputTokens", "input_tokens"),
    ("outputTokens", "output_tokens"),
    ("cacheReadInputTokens", "cache_read_input_tokens"),
    ("cacheCreationInputTokens", "cache_creation_input_tokens"),
)


def _claude_usage_totals(event: dict[str, Any]) -> dict[str, int]:
    """Running token totals from a `result` event's ``modelUsage``.

    ``modelUsage`` and ``total_cost_usd`` are the session's RUNNING totals:
    on ``--resume`` the CLI restores them from the transcript, so they cover
    every earlier turn as well as this one. Summed across models.
    """
    totals = {key: 0 for _, key in _CLAUDE_USAGE_TOTAL_FIELDS}
    usage = event.get("modelUsage")
    if not isinstance(usage, dict):
        return {}
    seen = False
    for entry in usage.values():
        if not isinstance(entry, dict):
            continue
        for field, key in _CLAUDE_USAGE_TOTAL_FIELDS:
            value = entry.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] += int(value)
                seen = True
    return totals if seen else {}


def _claude_usage_since(
    baseline: dict[str, Any] | None,
    totals: dict[str, int],
) -> dict[str, int] | None:
    """This turn's share of the running totals; ``None`` when unknowable."""
    if baseline is None or not totals:
        return None
    turn: dict[str, int] = {}
    for _, key in _CLAUDE_USAGE_TOTAL_FIELDS:
        before = baseline.get(key) or 0
        if not isinstance(before, (int, float)) or isinstance(before, bool):
            return None
        delta = totals.get(key, 0) - int(before)
        if delta < 0:
            return None
        turn[key] = delta
    return turn if any(turn.values()) else None


def _claude_usage_has_tokens(usage: Any) -> bool:
    if not isinstance(usage, dict):
        return False
    return any(isinstance(usage.get(key), (int, float)) and usage.get(key) for _, key in _CLAUDE_USAGE_TOTAL_FIELDS)


def _claude_compact_usage_estimate(meta: dict[str, Any]) -> dict[str, int] | None:
    """Upper-bound usage for a compaction with no usable running totals.

    The whole pre-compact context is read once and the kept context is
    written once; priced as plain input and output.
    """
    pre = meta.get("pre_tokens")
    post = meta.get("post_tokens")
    if not isinstance(pre, int) or isinstance(pre, bool) or pre <= 0:
        return None
    return {
        "input_tokens": pre,
        "output_tokens": post if isinstance(post, int) and post > 0 else 0,
    }


def _claude_local_command_reply(
    event: dict[str, Any],
    compact_meta: dict[str, Any],
    compact_error: str,
) -> str:
    """Reply text for a `result` that ran a built-in command, not the model.

    The CLI answers ``/compact`` itself: the `result` carries
    ``local_command`` and an empty ``result`` string on success (the summary
    goes into the transcript, not into the reply).
    """
    text = event.get("result")
    if isinstance(text, str) and text.strip():
        return text.strip()
    if compact_error:
        return compact_error
    if str(event.get("local_command") or "") != "compact":
        return f"/{event.get('local_command')} finished."
    pre = compact_meta.get("pre_tokens")
    post = compact_meta.get("post_tokens")
    if isinstance(pre, int) and isinstance(post, int) and pre > 0:
        return f"Context compacted: {pre:,} → {post:,} tokens."
    if isinstance(pre, int) and pre > 0:
        return f"Context compacted ({pre:,} tokens before)."
    return "Context compacted."


def _format_claude_tool_use(name: str, inp: Any) -> str:
    """Render an in-band marker for a `tool_use` block Claude emitted.

    Surfaces "Claude is calling Read foo.py" to the sharedagents user without
    a separate event channel — the marker is appended to the same text
    stream Claude's prose flows through, so the existing TUI / Telegram
    consumers render it for free. Stable, compact format (no timestamps,
    no ids) so the marker also lands cleanly in the saved tool result
    that gets sent back to the local model on the next round.
    """
    return _format_cli_tool_use(name, inp)


def _format_claude_tool_result(block: dict) -> str:
    """Render an in-band marker for a `tool_result` Claude received.

    Anthropic's tool_result `content` is either a plain string or a list
    of `{type: text, text: ...}` blocks; both shapes are flattened to a
    single line + char count so the user can see what came back without
    inflating context.
    """
    is_err = bool(block.get("is_error"))
    raw = block.get("content")
    if isinstance(raw, list):
        parts: list[str] = []
        for b in raw:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text") or "")
        text = "".join(parts)
    elif isinstance(raw, str):
        text = raw
    else:
        text = "" if raw is None else str(raw)
    n = len(text)
    head = next((line for line in text.splitlines() if line.strip()), "")
    summary = head.strip()[:140]
    arrow = "✗" if is_err else "↳"
    if summary:
        return f"{arrow} {summary} ({n} chars)\n"
    return f"{arrow} ({n} chars)\n"


# ---- delegate_antigravity helpers ----


async def run_delegate_claude_async(
    args: dict[str, Any],
    *,
    on_delta: Callable[[str], None] | None = None,
    on_event: Callable[[dict], None] | None = None,
    on_spawn: Callable[[Any], None] | None = None,
) -> ToolResult:
    """Spawn `claude -p ...` and return its full reply as the tool result.

    Driver shared between two callers:

    * :class:`sharedagents_agent.agents.claude_engine.ClaudeCliEngine` via
      :meth:`delegate_runner` — drives a turn against the Claude CLI
      when an Agent's engine is ``claude_cli``.
    * The ``delegate_claude`` tool registry entry (via the shared
      :func:`sharedagents_agent.agents.delegate_shims.shim_run_delegate_cli_async`)
      — lets any agent dispatch a child turn to Claude as
      a regular tool call.

    Streams `--output-format stream-json --include-partial-messages` and
    forwards each text delta (Claude's prose) plus a compact in-band
    marker for every `tool_use` / `tool_result` block Claude emits, all
    through the same `on_delta` callback. The slash-command bridge
    (`/claude`) and the autonomous-tool path (`tool_delta` SSE) both
    consume that callback, so live progress — including which tools
    Claude itself is running under the hood — shows up in the TUI,
    Telegram, and any other SSE consumer with no per-frontend changes.

    When ``on_event`` is wired, Claude's inner ``tool_use`` /
    ``tool_result`` blocks are ALSO surfaced as structured events
    (``{"kind": "tool_use", "id", "name", "input"}`` /
    ``{"kind": "tool_result", "tool_use_id", "ok", "text"}``) so the
    frontend can mount native tool-call widgets — e.g., a
    ``TodoWrite`` inside ``/claude`` ticks the sharedagents plan checklist
    the same way a local-model ``TodoWrite`` would. The inline text
    marker still lands in ``text_buf`` (so the persisted reply the
    local model replays on its next round retains a record of what
    Claude did), but it is NOT also sent through ``on_delta`` to the
    frontend — that would double-render. When ``on_event`` is
    ``None``, behaviour is unchanged: markers stream through
    ``on_delta`` just like before.

    Per-sharedagents-session state: reads `claude_session_id` from
    `<sessions_dir>/<sid>.delegate_claude.json` and passes
    `--resume <id>` so a follow-up `delegate_claude` call continues the
    same Claude conversation. The new id is written back on success.

    Killability: a SIGTERM landing on this coroutine (via the parent
    `tool_runner`'s signal handler) propagates through CancelledError;
    the `finally` block kills the claude subprocess so it doesn't keep
    running orphaned.
    """
    _load_tool_utils(globals())
    prompt = args.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return _err("prompt required")

    # Engine-pipeline callers pass session identity + permission mode
    # through args so concurrent CLI turns on the same agent can't race
    # on process-wide ``os.environ``. Legacy direct callers leave these
    # absent and fall back to env reads — same behaviour as before.
    sa_sid_arg = args.get("_sharedagents_session_id")
    sa_sid: str | None = sa_sid_arg if isinstance(sa_sid_arg, str) and sa_sid_arg else None
    sa_sdir_arg = args.get("_sharedagents_sessions_dir")
    sa_sdir: str | None = sa_sdir_arg if isinstance(sa_sdir_arg, str) and sa_sdir_arg else None
    sa_perm_arg = args.get("_sharedagents_permission_mode")
    sa_perm: str = sa_perm_arg if isinstance(sa_perm_arg, str) and sa_perm_arg else ""

    mode_raw = args.get("mode")
    if isinstance(mode_raw, str) and mode_raw in _CLAUDE_MODE_MAP:
        canonical_mode = mode_raw
        claude_mode = _CLAUDE_MODE_MAP[mode_raw]
    else:
        # Inherit from the sharedagents permission mode the agent is running
        # in. Engine callers pass it via ``_sharedagents_permission_mode``;
        # legacy direct callers leave it absent and we fall back to the
        # env var. Falls back to ``default`` if neither yields a known
        # mode — same fallback as the slash command.
        sa_mode = sa_perm or os.environ.get("SHAREDAGENTS_PERMISSION_MODE", "")
        canonical_mode = sa_mode if sa_mode in _CLAUDE_MODE_MAP else "ask"
        claude_mode = _CLAUDE_MODE_MAP.get(sa_mode, "default")
    # Claude refuses bypassPermissions / --dangerously-skip-permissions
    # as root. A verify-A gate only needs to return JSON from the prompt.
    if claude_mode == "bypassPermissions" and hasattr(os, "geteuid"):
        try:
            if os.geteuid() == 0:
                claude_mode = "default"
        except OSError:
            pass

    # Optional `cwd`: if the caller passed one, validate it BEFORE
    # spawning so a typo fails loud with a clear tool-result error
    # instead of crashing the subprocess with a cryptic FileNotFoundError.
    spawn_cwd, cwd_err = _resolve_delegate_cwd(args)
    if cwd_err is not None:
        _log_delegate_error(
            "claude",
            "bad_cwd",
            prompt=prompt,
            cwd=safe_getcwd(),
            mode=claude_mode,
            message=cwd_err,
        )
        return _err(cwd_err)

    force_fresh = bool(args.get("_sharedagents_claude_no_resume"))
    retry_of = args.get("_sharedagents_claude_retry_of")
    retry_of = retry_of if isinstance(retry_of, str) and retry_of else None
    # Resume-id override (new Agent/Engine pipeline). When the caller
    # is :class:`sharedagents_agent.agents.external_cli.ExternalCliEngine`,
    # the engine manages its own sidecar (``<sid>.claude_cli.json``)
    # and passes the resume id explicitly via ``_force_resume_id``
    # rather than letting us re-read the legacy
    # ``<sid>.delegate_claude.json``. ``force_fresh`` still wins.
    forced_resume = args.get("_force_resume_id")
    if isinstance(forced_resume, str) and forced_resume:
        resume_id = None if force_fresh else forced_resume
    else:
        resume_id = None if force_fresh else _read_claude_session_id(sa_sid, sa_sdir)

    # Prompt is piped on stdin, NOT passed as argv. argv would hit
    # Linux's ARG_MAX (~128 KiB) and fail with OSError E2BIG
    # ("Argument list too long") on large prompts (full session
    # context, long files, etc.). `claude -p` reads stdin when no
    # positional prompt arg is given.
    cmd = [
        _resolve_cli_executable("claude"),
        "-p",
        "--output-format",
        "stream-json",
        "--include-partial-messages",
        "--verbose",
        "--permission-mode",
        claude_mode,
        # `claude -p` is headless: AskUserQuestion auto-denies with
        # "Answer questions?" instead of round-tripping to the user.
        # Disallow it so Claude falls back to asking in its text reply.
        "--disallowed-tools",
        "AskUserQuestion",
    ]
    cmd += _claude_effort_flags(args)
    cmd += _claude_thinking_display_flags()
    if resume_id:
        cmd += ["--resume", resume_id]
    # System-prompt priming on the FIRST call of an agent session
    # (new Agent/Engine pipeline). ``ExternalCliEngine.run_turn``
    # composes the agent's SOUL.md + AGENTS.md + TOOLS.md (+ live
    # preamble blocks) into ``_append_system_prompt``; we pass it
    # to ``claude -p`` via the documented ``--append-system-prompt``
    # flag so it lands in the bound Claude session's state and
    # rides on every ``--resume`` after. Without this flag (antigravity
    # final audit #1) the engine's whole agent context is dropped
    # on the floor — Claude runs vanilla without the agent's
    # persona.
    appended_sys = args.get("_append_system_prompt")
    if isinstance(appended_sys, str) and appended_sys.strip():
        cmd += ["--append-system-prompt", appended_sys]
    # If the caller wants AskUserQuestion blocked (same `claude -p`
    # has no user available, so the model would dead-end on it),
    # respect a ``_disallowed_tools`` arg as a list of tool names.
    disallowed = args.get("_disallowed_tools")
    if isinstance(disallowed, (list, tuple)) and disallowed:
        cmd += ["--disallowedTools", ",".join(str(t) for t in disallowed)]
    # Optional model override. The Engine resolves the user's ``/model``
    # pick (or a discovered model from the Anthropic catalogue) to a wire
    # id and passes it as ``args['model']``; ``claude -p --model`` accepts
    # an alias (opus/sonnet/haiku) or a full id.
    #
    # MODEL GATE (owner round-2 scope change — the incident is not codex-
    # specific): direct callers of this runner bypass the engine gate, so
    # the SAME fail-fast gate applies here. The literal engine label
    # "claude" (``display_model()`` degrades to the engine name, never a
    # wire id — the Claude CLI rejects it) is refused BEFORE the spawn.
    # Only the CANONICAL catalogue spelling is built into ``--model``
    # (a catalogue match returns the catalogue's own spelling; any other
    # non-label value passes through verbatim — the Claude CLI is the
    # authority for its own model syntax: aliases, ``[1m]``/``[2m]``
    # forms, operator deployments). With
    # no override on a FRESH call, the claude engine's effective default
    # (operator-explicit agent ``upstream_model`` → ANTHROPIC_MODEL env →
    # ~/.claude/settings.json → ``""`` — the builtin pin is GONE, finding
    # 1a) is passed EXPLICITLY so sharedagents's resolution drives the spawn —
    # the codex OPTION B precedent, generalized. Empty default → no flag
    # → the CLI's own default applies (Claude Code precedence). A resumed
    # thread keeps its own model unless the caller supplies a selection.
    from sharedagents_agent.discovery._common import (
        external_cli_effective_default,
        external_cli_model_refusal,
        validate_external_cli_model,
    )

    model_override = args.get("model")
    if isinstance(model_override, str) and model_override.strip():
        _canonical_model = validate_external_cli_model(
            "claude_cli",
            model_override,
        )
        if _canonical_model is None:
            return ToolResult(
                content=external_cli_model_refusal(
                    model_override,
                    "claude_cli",
                    source="explicit",
                ),
                ok=False,
            )
        model_override = _canonical_model
    elif not resume_id:
        model_override = external_cli_effective_default("claude_cli")
    if isinstance(model_override, str) and model_override.strip():
        cmd += ["--model", model_override.strip()]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=16 * 1024 * 1024,
            # Own process group so SIGTERM reaches Claude's spawned
            # `Bash`/`Read`/etc. subtools when the user hits ESC.
            start_new_session=True,
            cwd=spawn_cwd,
            env=_build_delegate_spawn_env(args),
        )
        # Lease the spawned *_cli agent's process group (§10.4): the broker-
        # aware _kill_group then capability-validates its cancel; released in
        # this runner's ``finally``.
        _lease_id = sandbox_broker.lease_process(pgid=proc.pid, kind="cli")
        if on_spawn is not None:
            try:
                on_spawn(proc)
            except Exception:  # noqa: BLE001
                pass
    except FileNotFoundError:
        _log_delegate_error(
            "claude",
            "spawn_failed",
            prompt=prompt,
            cwd=spawn_cwd,
            mode=claude_mode,
            resume_id=resume_id,
            message="`claude` CLI not on PATH",
        )
        return _err("`claude` CLI not on PATH. Install: https://claude.com/code")

    stdin_task = asyncio.create_task(_feed_stdin(proc, prompt.encode("utf-8")))

    text_buf: list[str] = []
    answer_sink = CliAnswerSink()
    reasoning_chars = 0
    final_session_id: str | None = resume_id
    is_error = False
    got_result = False  # True when a `result` event was received (success or error)
    # True once this stream carried a ``system``/``task_notification``
    # event — gates the compatibility fallback in
    # ``_claude_result_is_task_notification`` (see its docstring).
    saw_task_notification = False
    error_subtype: str | None = None
    error_event: dict[str, Any] = {}
    # ``/compact`` is answered by the CLI itself; these carry its outcome.
    compact_meta: dict[str, Any] = {}
    compact_error = ""
    # Running totals as of the previous turn, to isolate this turn's share.
    # A fresh process starts from zero; a resumed one from the sidecar.
    usage_totals: dict[str, int] = {}
    usage_baseline: dict[str, Any] | None = {}
    if resume_id:
        _prior = _read_delegate_state("claude", sa_sid, sa_sdir)
        usage_baseline = (
            _prior.get("usage_totals")
            if _prior.get("claude_session_id") == resume_id and isinstance(_prior.get("usage_totals"), dict)
            else None
        )
    # Per-index buffer for tool_use blocks streaming via stream_event:
    # content_block_start gives us the name + id, content_block_delta
    # streams the input as input_json_delta partial_json chunks, and
    # content_block_stop is when we finally have the full args ready
    # to emit a marker. Without this we'd either fire `🔧 Read {}` at
    # block_start (no args yet) or wait for the consolidated `assistant`
    # event (less live).
    tool_use_pending: dict[int, dict] = {}
    # Dedup against the consolidated `assistant` / `user` events that
    # restate the same blocks the stream channel already delivered.
    seen_tool_uses: set[str] = set()
    seen_tool_results: set[str] = set()
    # Per-message usage from the LAST `assistant` event of the turn. The
    # terminal `result` event's `usage` is CUMULATIVE across every internal
    # API round-trip claude made (each tool-calling step re-reads the growing
    # context), so its input+cache sum balloons to millions on a long agentic
    # turn — correct for BILLING, but nonsense as the ctx-meter numerator
    # ("in <pt>/<window>" then renders e.g. 7.0M/256K). The last assistant
    # message's usage is the TRUE context occupancy at end-of-turn (one API
    # call, ≤ the window). We capture it here and hand it to the engine as the
    # context-display token count, leaving cost priced from the cumulative
    # totals / the CLI's reported total_cost_usd.
    last_assistant_usage: dict | None = None
    # The ``system/init`` model id; ``result.modelUsage`` is keyed by it.
    init_model = ""
    # One trace per assistant message (deltas stream, the consolidated block
    # forwards only their unseen tail).
    thinking_relay = ReasoningTextRelay()

    def _note_reasoning(value: Any) -> None:
        """Record only the size of private reasoning; never its text."""
        nonlocal reasoning_chars
        if isinstance(value, int) and value > 0:
            reasoning_chars += value

    def _emit_thinking(text: str, *, consolidated: bool = False) -> None:
        """Forward Claude thinking text once per assistant message.

        The CLI runs with ``--include-partial-messages``, so ``thinking_delta``
        fragments stream first and the consolidated ``assistant`` message then
        restates the same block: the relay forwards only the part the stream
        has not already produced.
        """
        if not text or on_event is None:
            return
        kept = thinking_relay.consolidated("message", text) if consolidated else thinking_relay.delta("message", text)
        if not kept:
            return
        try:
            on_event({"kind": "thinking", "text": kept})
        except Exception:  # noqa: BLE001
            pass

    def _emit_answer(kept: str) -> None:
        if not kept:
            return
        text_buf.append(kept)
        if on_delta is not None:
            try:
                on_delta(kept)
            except Exception:  # noqa: BLE001
                pass

    def _push(chunk: str) -> None:
        """Append an incremental Claude token."""
        _emit_answer(answer_sink.push_delta(chunk))

    def _push_consolidated(text: str) -> None:
        """Append a consolidated assistant body, skipping restatements."""
        _emit_answer(answer_sink.push_consolidated(text))

    def _push_tool_use(tu_id: str, name: str, inp: Any) -> None:
        """Surface a tool_use block to the frontend.

        The compact ``🔧 <name> <args>`` text marker ALWAYS lands in
        ``text_buf`` so the persisted body the local model replays on
        its next round retains a record of what Claude did. Streaming
        delivery splits on whether the frontend asked for structured
        events: with ``on_event`` wired, only the structured payload
        goes out so the TUI can mount a native ToolCallMessage / route
        TodoWrite to the plan checklist; without, the text marker
        flows through ``on_delta`` so older consumers still see the
        inline pill.
        """
        marker = _format_claude_tool_use(name, inp)
        text_buf.append(marker)
        if on_event is not None:
            try:
                on_event(
                    {
                        "kind": "tool_use",
                        "id": tu_id or "",
                        "name": name or "?",
                        "input": inp,
                    }
                )
            except Exception:  # noqa: BLE001
                pass
        elif on_delta is not None:
            try:
                on_delta(marker)
            except Exception:  # noqa: BLE001
                pass

    def _push_tool_result(block: dict) -> None:
        """Surface a tool_result block. Mirror of ``_push_tool_use``."""
        marker = _format_claude_tool_result(block)
        text_buf.append(marker)
        if on_event is not None:
            raw = block.get("content")
            if isinstance(raw, list):
                parts: list[str] = []
                for b in raw:
                    if isinstance(b, dict) and b.get("type") == "text":
                        parts.append(b.get("text") or "")
                text = "".join(parts)
            elif isinstance(raw, str):
                text = raw
            else:
                text = "" if raw is None else str(raw)
            try:
                on_event(
                    {
                        "kind": "tool_result",
                        "tool_use_id": block.get("tool_use_id") or "",
                        "ok": not bool(block.get("is_error")),
                        "text": text,
                    }
                )
            except Exception:  # noqa: BLE001
                pass
        elif on_delta is not None:
            try:
                on_delta(marker)
            except Exception:  # noqa: BLE001
                pass

    # Hard backstop in case stdout never reaches EOF (a tool grandchild
    # inherited claude's stdout fd and kept it open after claude itself
    # exited — observed in sharedagents session 6a0af106cd41 where claude
    # finished at 11:03:15 but readline blocked for 15+ min until ESC).
    # Configurable via env; defaults BELOW Telegram's 300s no-progress
    # watchdog so a mid-stream stall fails fast with a clean turn error
    # before the coarse watchdog hard-cancels the turn (see
    # ``_claude_idle_timeout_s``).
    idle_timeout_s = _claude_idle_timeout_s()
    # A `claude -p` that never emits its first stream event (hung
    # SessionStart hook, auth/network wedge, a spawn that never reaches the
    # model) was previously invisible until this 900s idle timer — the
    # operator saw a frozen turn and restarted the gateway. Bound the wait
    # for the FIRST event separately so a startup stall fails fast.
    first_token_timeout_s = _delegate_first_token_timeout_s()
    got_first_event = False
    session_id_emitted = False
    model_emitted = False
    try:
        if proc.stdout is None:
            raise RuntimeError("subprocess stdout pipe is None — spawn config bug")
        while True:
            try:
                line = await _delegate_readline(
                    proc.stdout,
                    got_first_event=got_first_event,
                    first_token_timeout_s=first_token_timeout_s,
                    idle_timeout_s=idle_timeout_s,
                )
            except asyncio.TimeoutError:
                if not got_first_event:
                    # Stalled startup — fail fast with a clear error rather
                    # than waiting out the full idle window. The bounded
                    # finally below reaps the process group.
                    _log_delegate_error(
                        "claude",
                        "first_token_timeout",
                        prompt=prompt,
                        cwd=spawn_cwd,
                        mode=claude_mode,
                        resume_id=resume_id,
                        remote_session_id=final_session_id,
                        returncode=proc.returncode,
                    )
                    return _err(
                        f"claude produced no output within "
                        f"{int(first_token_timeout_s)}s (startup timeout) — the "
                        "CLI appears stalled (e.g. a hung SessionStart hook, "
                        "auth, or network). Aborted instead of hanging; tune "
                        "via SHAREDAGENTS_DELEGATE_FIRST_TOKEN_TIMEOUT_S."
                    )
                # Output started then went silent past the idle window.
                # Surface as an error instead of spinning forever. The
                # finally below SIGTERMs the process group so any
                # pipe-holding grandchild dies too.
                is_error = True
                error_subtype = "claude_runner_idle_timeout"
                break
            except ValueError:
                # Single event over the 16 MiB readline limit — skip it
                # but keep reading rather than failing the whole call.
                got_first_event = True
                continue
            if not line:
                break
            got_first_event = True
            try:
                event = json.loads(line.decode("utf-8", "replace"))
            except json.JSONDecodeError:
                continue
            sid = event.get("session_id")
            if isinstance(sid, str) and sid:
                final_session_id = sid
                # Surface the session id to the engine as soon as we
                # see it (instead of only on the success terminal
                # path) so an ESC/idle-timeout still leaves the
                # engine sidecar primed for `--resume` on the next
                # turn. Without this, an aborted turn 0 erases the
                # claude session id and the user loses context.
                if not session_id_emitted and on_event is not None:
                    try:
                        on_event({"kind": "session_id", "id": sid})
                    except Exception:  # noqa: BLE001
                        pass
                    session_id_emitted = True
            # The first ``system``/``init`` event carries the real model
            # id (e.g. ``claude-opus-4-…``). Surface it once so the engine
            # can record it and the TUI footer can show the actual model
            # instead of the ``claude`` placeholder.
            if not model_emitted and on_event is not None:
                model_id = event.get("model")
                if isinstance(model_id, str) and model_id:
                    try:
                        on_event({"kind": "model", "id": model_id})
                    except Exception:  # noqa: BLE001
                        pass
                    model_emitted = True
            etype = event.get("type")
            if etype == "system" and event.get("subtype") == "init":
                _init_model = event.get("model")
                if isinstance(_init_model, str) and _init_model:
                    init_model = _init_model
            if etype == "system" and event.get("subtype") == "task_notification":
                # Precedes the CLI's background-task drain pseudo-turn.
                saw_task_notification = True
            if etype == "system" and event.get("subtype") == "compact_boundary":
                _meta = event.get("compact_metadata")
                if isinstance(_meta, dict):
                    compact_meta = _meta
            if etype == "system" and event.get("subtype") == "status" and event.get("compact_result") == "failed":
                compact_error = str(event.get("compact_error") or "").strip()
            if etype == "system" and event.get("subtype") == "init" and on_event is not None:
                native_mode = event.get("permissionMode")
                if not isinstance(native_mode, str):
                    native_mode = event.get("permission_mode")
                if isinstance(native_mode, str) and native_mode:
                    try:
                        on_event(
                            {
                                "kind": "native_mode",
                                "mode": canonical_mode,
                                "native_value": native_mode,
                                "expected_native_value": claude_mode,
                                "verified": native_mode == claude_mode,
                                "session_id": final_session_id or "",
                            }
                        )
                    except Exception:  # noqa: BLE001
                        pass
            if etype == "stream_event":
                inner = event.get("event") or {}
                inner_type = inner.get("type")
                if inner_type == "message_start":
                    # Content-block numbering restarts with each assistant
                    # message. Resetting here (not at message_stop) keeps the
                    # deltas buffered while the consolidated `assistant` event
                    # that closes them is still in flight.
                    thinking_relay.forget("message")
                if inner_type == "content_block_delta":
                    delta = inner.get("delta") or {}
                    dtype = delta.get("type")
                    if dtype == "text_delta":
                        txt = delta.get("text") or ""
                        if txt:
                            # Push immediately so the SSE consumer
                            # (TUI ClaudeMessage / bash bridge stdout)
                            # sees Claude's text as it's generated,
                            # not just at end-of-call.
                            _push(txt)
                    elif dtype in _REASONING_DELTA_TYPES:
                        _note_reasoning(_reasoning_signal_chars(delta))
                        thought = delta.get("thinking") or delta.get("text") or delta.get("reasoning") or ""
                        if thought:
                            _emit_thinking(thought)  # streamed increment
                    elif dtype == "input_json_delta":
                        idx = inner.get("index")
                        partial = delta.get("partial_json") or ""
                        if isinstance(idx, int) and idx in tool_use_pending and partial:
                            tool_use_pending[idx]["json"] += partial
                elif inner_type == "content_block_start":
                    cb = inner.get("content_block") or {}
                    if cb.get("type") in _REASONING_BLOCK_TYPES:
                        _note_reasoning(_reasoning_signal_chars(cb))
                        thought = cb.get("thinking") or cb.get("text") or cb.get("reasoning") or ""
                        if thought:
                            _emit_thinking(thought, consolidated=True)
                    elif cb.get("type") == "tool_use":
                        idx = inner.get("index")
                        if isinstance(idx, int):
                            tool_use_pending[idx] = {
                                "id": cb.get("id") or "",
                                "name": cb.get("name") or "?",
                                "json": "",
                            }
                elif inner_type == "content_block_stop":
                    idx = inner.get("index")
                    pending = tool_use_pending.pop(idx, None) if isinstance(idx, int) else None
                    if pending and pending["id"] not in seen_tool_uses:
                        seen_tool_uses.add(pending["id"])
                        try:
                            inp = json.loads(pending["json"]) if pending["json"] else {}
                        except json.JSONDecodeError:
                            inp = pending["json"]
                        _push_tool_use(pending["id"], pending["name"], inp)
            elif etype == "assistant":
                # Consolidated assistant message — fill in text not yet
                # streamed AND surface any tool_use blocks we haven't
                # already seen via `content_block_start`.
                # Track this message's own usage (NOT cumulative) — the last
                # one wins and becomes the honest ctx-meter numerator below.
                _msg_usage = (event.get("message") or {}).get("usage")
                if isinstance(_msg_usage, dict) and _msg_usage:
                    last_assistant_usage = _msg_usage
                content = (event.get("message") or {}).get("content") or []
                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        btype = block.get("type")
                        if btype == "text":
                            full = block.get("text") or ""
                            _push_consolidated(full)
                        elif btype in _REASONING_BLOCK_TYPES:
                            _note_reasoning(_reasoning_signal_chars(block))
                            thought = block.get("thinking") or block.get("text") or block.get("reasoning") or ""
                            if thought:
                                # The consolidated `assistant` message
                                # restates the blocks the deltas streamed.
                                _emit_thinking(thought, consolidated=True)
                        elif btype == "tool_use":
                            tu_id = block.get("id") or ""
                            if tu_id and tu_id in seen_tool_uses:
                                continue
                            if tu_id:
                                seen_tool_uses.add(tu_id)
                            _push_tool_use(
                                tu_id,
                                block.get("name") or "?",
                                block.get("input") or {},
                            )
            elif etype == "user":
                # Tool results coming back to Claude. Surface a brief
                # marker so the sharedagents user (and the local model on
                # replay) sees what Claude's sub-tool returned.
                content = (event.get("message") or {}).get("content") or []
                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") != "tool_result":
                            continue
                        tu_id = block.get("tool_use_id") or ""
                        if tu_id and tu_id in seen_tool_results:
                            continue
                        if tu_id:
                            seen_tool_results.add(tu_id)
                        _push_tool_result(block)
            elif etype in _REASONING_BLOCK_TYPES:
                _note_reasoning(_reasoning_signal_chars(event))
                thought = event.get("thinking") or event.get("text") or event.get("reasoning") or ""
                if thought:
                    _emit_thinking(thought, consolidated=True)
            elif etype == "result":
                # NOT every `result` ends OUR turn. When ``--resume`` lands
                # on a session whose previous run left a background task
                # unreaped (claude -p is one-shot: it exits right after the
                # final `result`, killing any `run_in_background` Bash /
                # backgrounded Task with it), the CLI reconciles on startup
                # by enqueueing a `<task-notification>` and answering it as
                # its OWN zero-cost pseudo-turn BEFORE it ever reads our
                # prompt. That drain emits a full `result` event carrying
                # ``origin.kind == "task-notification"``, ``num_turns: 0``
                # and an EMPTY ``result`` string, and is followed by a
                # second `system/init` + the real turn.
                #
                # Breaking on it (as this loop did) made the runner report
                # ``claude returned empty response`` and SIGTERM the group
                # mid-prompt — the user's message was already enqueued but
                # never answered, so the CLI transcript recorded
                # ``[Request interrupted by user]``. Re-sending the SAME
                # text then worked, because the failed run had already
                # drained the notification. In every production case seen
                # (4 turns) the failing turn was one whose
                # predecessor had left an orphaned background task.
                #
                # So skip drain results entirely — don't set ``got_result``
                # (a later non-zero exit must still surface stderr via the
                # ``nonzero_exit`` path), don't let ``is_error`` fire off
                # them, and don't emit their all-zero usage. Keep reading
                # for the real terminal result.
                if _claude_result_is_task_notification(
                    event,
                    saw_task_notification=saw_task_notification,
                ):
                    log_event(
                        "DELEGATE_CLAUDE_TASK_NOTIFICATION_DRAINED",
                        sharedagents_session_id=sa_sid,
                        claude_session_id=final_session_id,
                        resume_id=resume_id,
                    )
                    continue
                got_result = True
                # A result whose subtype says SUCCESS is a success no
                # matter what is_error claims — the claude CLI sets
                # is_error truthy on some benign terminal results, which
                # produced 30 production "claude reported error: success"
                # rows that downstream error views and the SI had to
                # special-case (this is the fix at the SOURCE).
                _res_subtype = str(event.get("subtype") or "").lower()
                if event.get("is_error") and _res_subtype != "success":
                    is_error = True
                    error_event = event
                    error_subtype = event.get("subtype") or "claude reported an error"
                turn_usage = event.get("usage")
                turn_cost = event.get("total_cost_usd")
                context_usage = last_assistant_usage
                usage_totals = _claude_usage_totals(event)
                if resume_id:
                    # A running total for the whole resumed session, never
                    # this turn's cost.
                    turn_cost = None
                if event.get("local_command") and not is_error:
                    if not "".join(text_buf).strip():
                        _emit_answer(
                            _claude_local_command_reply(
                                event,
                                compact_meta,
                                compact_error,
                            )
                        )
                    if not _claude_usage_has_tokens(turn_usage):
                        # The command's own model call (the compaction) is in
                        # the running totals only; ``usage`` is all zero.
                        turn_usage = (
                            _claude_usage_since(usage_baseline, usage_totals)
                            or _claude_compact_usage_estimate(compact_meta)
                            or turn_usage
                        )
                    _post = compact_meta.get("post_tokens")
                    if isinstance(_post, int) and not isinstance(_post, bool) and _post > 0:
                        # The context meter shows what is left after the
                        # compaction, not the size of the call that made it.
                        context_usage = {"input_tokens": _post}
                # Surface REAL token usage / cost so the engine prices this
                # delegate turn from actual tokens (not a char estimate).
                if on_event is not None and (
                    isinstance(turn_usage, dict) or turn_cost is not None or context_usage is not None
                ):
                    try:
                        on_event(
                            {
                                "kind": "usage",
                                "usage": turn_usage,
                                # Honest end-of-turn context occupancy (last single
                                # API call), used for the ctx-meter numerator instead
                                # of the cumulative `usage` sum. See the capture site.
                                "context_usage": context_usage,
                                # The window the CLI ran the model with — the
                                # ctx-meter denominator (see the helper).
                                "context_window": _claude_result_context_window(
                                    event,
                                    init_model,
                                )
                                or None,
                                "total_cost_usd": turn_cost,
                            }
                        )
                    except Exception:  # noqa: BLE001
                        pass
                # `result` is claude's logical end-of-turn marker. Trust
                # it instead of waiting for stdout EOF — a tool
                # grandchild can keep the stdout pipe open long after
                # claude itself exits, which would otherwise leave the
                # runner blocked on readline forever (and the TUI's
                # worker indicator spinning until the user hits ESC).
                break
    except asyncio.CancelledError:
        _log_delegate_error(
            "claude",
            "cancelled",
            prompt=prompt,
            cwd=spawn_cwd,
            mode=claude_mode,
            resume_id=resume_id,
            remote_session_id=final_session_id,
            returncode=proc.returncode,
        )
        # Hard cancel → SIGKILL the WHOLE tree so claude's spawned tools
        # (Bash, Read, …) AND any subagents it launched (Task tool — which
        # start their own session) die too instead of being orphaned.
        _kill_tree(proc, signal.SIGKILL)
        raise
    finally:
        sandbox_broker.release_process(locals().get("_lease_id"))
        if not stdin_task.done():
            stdin_task.cancel()
        try:
            await stdin_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        # Bounded reap: wait for natural exit first (cleanly-exiting
        # claude races against any signal we send and would otherwise
        # come back with a negative returncode → spurious _err). If
        # the proc is still alive after the grace window — usually a
        # tool grandchild holding claude's stdout pipe open — escalate
        # to SIGTERM/SIGKILL on the whole group so the finally itself
        # can't pin the engine indefinitely.
        try:
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            try:
                await _terminate_group(proc, grace_s=2.0)
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            pass

    if proc.returncode != 0 and not is_error and not got_result:
        stderr_bytes = b""
        if proc.stderr is not None:
            try:
                stderr_bytes = await proc.stderr.read()
            except Exception:  # noqa: BLE001
                pass
        stderr_text = stderr_bytes.decode("utf-8", "replace")
        err_msg = stderr_text[:400] or _signal_name(proc.returncode)
        _log_delegate_error(
            "claude",
            "nonzero_exit",
            prompt=prompt,
            cwd=spawn_cwd,
            mode=claude_mode,
            resume_id=resume_id,
            remote_session_id=final_session_id,
            returncode=proc.returncode,
            stderr=stderr_text,
        )
        # Stale-resume recovery. When ``--resume <id>`` points at a
        # Claude CLI conversation that no longer exists, claude exits
        # NON-ZERO with stderr like "No conversation found with session
        # ID: <id>" (it never emits a structured ``result`` event, so
        # the ``error_during_execution`` retry above can't catch it).
        # This is exactly what happens after an orchestrator restart:
        # the resume id persisted in our sidecar outlives the Claude
        # CLI's own transcript (different spawn cwd, pruned project
        # store, …), so every continued session would otherwise die
        # with ``claude_session_not_continued_after_orchestrator_restart``.
        # Recover by transparently retrying once WITHOUT ``--resume`` —
        # a fresh Claude session re-primed from our system prompt keeps
        # the sharedagents-level conversation alive instead of failing the turn.
        err_lower = stderr_text.lower()
        is_stale_resume = (
            bool(resume_id)
            and not force_fresh
            and ("no conversation found" in err_lower or "session id" in err_lower or "--resume" in err_lower)
        )
        if is_stale_resume and not args.get("_native_readiness_probe") and not args.get("_force_resume_id"):
            log_event(
                "DELEGATE_CLAUDE_RETRY_FRESH",
                sharedagents_session_id=os.environ.get("SHAREDAGENTS_SESSION_ID"),
                stale_claude_session_id=resume_id,
                cwd=spawn_cwd or safe_getcwd(),
                mode=claude_mode,
            )
            retry_args = dict(args)
            retry_args["_sharedagents_claude_no_resume"] = True
            retry_args["_sharedagents_claude_retry_of"] = resume_id
            retry = await run_delegate_claude_async(
                retry_args,
                on_delta=on_delta,
                on_event=on_event,
            )
            log_event(
                "DELEGATE_CLAUDE_RETRY_FRESH_DONE",
                sharedagents_session_id=os.environ.get("SHAREDAGENTS_SESSION_ID"),
                stale_claude_session_id=resume_id,
                ok=retry.ok,
                truncated=retry.truncated,
            )
            if retry.ok:
                return retry
            return _err(f"claude exited {proc.returncode}: {err_msg}; fresh retry also failed: {retry.content}")
        return _err(f"claude exited {proc.returncode}: {err_msg}")

    if is_error:
        _log_delegate_error(
            "claude",
            "reported_error",
            prompt=prompt,
            cwd=spawn_cwd,
            mode=claude_mode,
            resume_id=resume_id,
            remote_session_id=final_session_id,
            returncode=proc.returncode,
            event=error_event,
            claude_session_id=final_session_id,
            retry_no_resume=force_fresh,
            retry_of=retry_of,
        )
        if (
            error_subtype == "error_during_execution"
            and resume_id
            and not force_fresh
            and not args.get("_native_readiness_probe")
            and not args.get("_force_resume_id")
            and not _claude_error_is_abort(error_event)
        ):
            log_event(
                "DELEGATE_CLAUDE_RETRY_FRESH",
                sharedagents_session_id=os.environ.get("SHAREDAGENTS_SESSION_ID"),
                stale_claude_session_id=resume_id,
                cwd=spawn_cwd or safe_getcwd(),
                mode=claude_mode,
            )
            retry_args = dict(args)
            retry_args["_sharedagents_claude_no_resume"] = True
            retry_args["_sharedagents_claude_retry_of"] = resume_id
            retry = await run_delegate_claude_async(
                retry_args,
                on_delta=on_delta,
                on_event=on_event,
            )
            log_event(
                "DELEGATE_CLAUDE_RETRY_FRESH_DONE",
                sharedagents_session_id=os.environ.get("SHAREDAGENTS_SESSION_ID"),
                stale_claude_session_id=resume_id,
                ok=retry.ok,
                truncated=retry.truncated,
            )
            if retry.ok:
                return retry
            return _err(f"claude reported error: {error_subtype}; fresh retry also failed: {retry.content}")
        return _err(f"claude reported error: {error_subtype}")

    body_text = "".join(text_buf).strip()
    if not body_text and reasoning_chars <= 0:
        _log_delegate_error(
            "claude",
            "empty_response",
            prompt=prompt,
            cwd=spawn_cwd,
            mode=claude_mode,
            resume_id=resume_id,
            remote_session_id=final_session_id,
            returncode=proc.returncode,
        )
        return _err("claude returned empty response")

    if reasoning_chars > 0 and on_event is not None:
        try:
            on_event({"kind": "reasoning", "chars": reasoning_chars})
        except Exception:  # noqa: BLE001
            pass

    if final_session_id:
        _write_claude_session_id(
            final_session_id,
            sa_sid,
            sa_sdir,
            usage_totals=usage_totals,
        )

    header_bits = []
    if final_session_id:
        header_bits.append(f"session={final_session_id[:12]}")
    header_bits.append(f"mode={claude_mode}")
    if resume_id:
        header_bits.append("resumed")
    header = f"[claude {' · '.join(header_bits)}]"

    # No "send only NEW context" footer: continuity is owned by the
    # session-id link (``--resume <id>`` via the persisted sidecar),
    # not a per-result note. A footer can't know whether the link
    # actually survived — after an orchestrator restart the stale-resume
    # path falls back to a FRESH session, so "Claude already has it"
    # was false exactly when it mattered. The orchestrator-side contract
    # ("on follow-ups send only the new turn") lives once in the
    # delegate-strategy skill instead of riding every tool result (where
    # it also leaked into user-facing output).
    body = f"{header}\n\n{body_text}" if body_text else header
    body, truncated = _truncate(body)
    return ToolResult(content=body, ok=True, truncated=truncated)
