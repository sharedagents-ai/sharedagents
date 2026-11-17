"""Codex CLI delegate runner."""

from __future__ import annotations

import asyncio
import json
import os
import signal
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sharedagents_agent import sandbox_broker
from sharedagents_core.cwdutil import safe_getcwd

from .base import (
    _REASONING_BLOCK_TYPES,
    _REASONING_DELTA_TYPES,
    CliAnswerSink,
    ReasoningTextRelay,
    _codex_effort_flags,
    _codex_local_artifact_references,
    _codex_raw_cumulative,
    _codex_reasoning_summary_flags,
    _codex_usage_delta,
    _delegate_state_path,
    _load_tool_utils,
    _log_delegate_error,
    _looks_like_context_overflow,
    _read_delegate_state,
    _reasoning_signal_chars,
    _resolve_delegate_cwd,
    _update_delegate_state,
    log_event,
)

if TYPE_CHECKING:
    # Bound into this module at call time by ``_load_tool_utils(globals())``;
    # a top-level import would close the tools <-> delegates import cycle.
    from sharedagents_agent.tools.impl._utils import (
        ToolResult,
        _build_delegate_spawn_env,
        _delegate_first_token_timeout_s,
        _delegate_idle_timeout_s,
        _delegate_readline,
        _err,
        _feed_stdin,
        _kill_group,
        _kill_tree,
        _resolve_cli_executable,
        _signal_name,
        _truncate,
    )

# Map sharedagents permission modes → codex flags. The codex CLI's resume
# subcommand doesn't accept `-s/--sandbox` or `-a/--ask-for-approval`
# directly, so we use `-c sandbox_mode=...` / `-c approval_policy=...`
# overrides which work for both first-call (`codex exec`) and resume
# (`codex exec resume`). `auto` uses the canonical
# `--dangerously-bypass-approvals-and-sandbox` flag because in headless
# mode there's no human to answer an approval prompt.
_CODEX_MODE_FLAGS: dict[str, list[str]] = {
    "plan": ["-c", "sandbox_mode=read-only", "-c", "approval_policy=never"],
    "ask": ["-c", "sandbox_mode=workspace-write", "-c", "approval_policy=on-request"],
    "auto": ["--dangerously-bypass-approvals-and-sandbox"],
}


def _codex_state_path(
    sa_sid: str | None = None,
    sa_sdir: str | None = None,
) -> Path | None:
    """Resolve the per-sharedagents-session state file for `delegate_codex`.

    Mirrors :func:`_claude_state_path`: engine-pipeline callers pass
    ``sa_sid`` / ``sa_sdir`` explicitly so concurrent CLI turns can't
    race on ``os.environ``; legacy direct callers fall back to env.
    """
    return _delegate_state_path("codex", sa_sid, sa_sdir)


def _read_codex_state(
    sa_sid: str | None = None,
    sa_sdir: str | None = None,
) -> dict[str, Any]:
    """Read the whole per-session codex sidecar (missing/corrupt → ``{}``).

    The sidecar is a JSON object; historically exactly
    ``{"codex_thread_id": <str>}``, now a superset that also carries the
    cumulative-usage baseline (``last_cumulative_usage`` /
    ``last_cumulative_thread_id``) for delta billing. Missing keys are
    simply absent — callers treat them as ``None``.
    """
    return _read_delegate_state("codex", sa_sid, sa_sdir)


def _update_codex_state(
    updates: dict[str, Any],
    sa_sid: str | None = None,
    sa_sdir: str | None = None,
) -> None:
    """Merge ``updates`` into the codex sidecar (read-modify-write).

    Preserves keys written by other callers — in particular, writing
    ``codex_thread_id`` must not clobber the cumulative-usage baseline
    and vice versa. Best-effort: a non-writable location is swallowed
    (the sidecar is a resume/billing convenience, not a correctness gate).
    """
    _update_delegate_state("codex", updates, sa_sid, sa_sdir)


def _read_codex_thread_id(
    sa_sid: str | None = None,
    sa_sdir: str | None = None,
) -> str | None:
    data = _read_codex_state(sa_sid, sa_sdir)
    tid = data.get("codex_thread_id")
    return tid if isinstance(tid, str) and tid else None


def _write_codex_thread_id(
    thread_id: str,
    sa_sid: str | None = None,
    sa_sdir: str | None = None,
) -> None:
    # Read-modify-write so the cumulative-usage baseline keys survive a
    # thread-id write (a whole-file overwrite would clobber them and
    # silently re-bill the whole thread on the next resume).
    _update_codex_state({"codex_thread_id": thread_id}, sa_sid, sa_sdir)


def _codex_rollout_model(thread_id: str | None) -> str:
    """Read the latest model from this native thread's rollout, best-effort.

    ``exec --json`` normally omits model metadata. The local rollout includes
    it in ``turn_context`` even on resumed turns. Match the native thread id,
    not ``session_id``: subagents can share that session while using different
    models. Only this thread's files are opened; Codex's files stay read-only.
    """
    try:
        native_id = str(UUID(str(thread_id)))
    except (ValueError, TypeError, AttributeError):
        return ""
    root = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex").expanduser()
    latest: tuple[str, str] = ("", "")
    try:
        for path in (root / "sessions").glob(f"**/rollout-*-{native_id}.jsonl"):
            try:
                with path.open("rb") as stream:
                    metadata = json.loads(next(stream, b"{}"))
                    payload = metadata.get("payload") or {}
                    if (
                        metadata.get("type") != "session_meta"
                        or (payload.get("id") or payload.get("session_id")) != native_id
                    ):
                        continue
                    for line in stream:
                        if b'"turn_context"' not in line:
                            continue
                        try:
                            event = json.loads(line)
                        except (ValueError, UnicodeError):
                            continue  # a live writer may leave a partial line
                        if not isinstance(event, dict) or event.get("type") != "turn_context":
                            continue
                        context = event.get("payload") or {}
                        model = context.get("model") if isinstance(context, dict) else None
                        if isinstance(model, str) and model.strip() and model.strip().lower() != "codex":
                            timestamp = str(event.get("timestamp") or "")
                            if timestamp >= latest[0]:
                                latest = (timestamp, model.strip())
            except (OSError, ValueError, AttributeError):
                continue  # files may move while Codex reorganizes sessions
    except OSError:
        pass
    return latest[1]


def _format_codex_command_started(cmd: str) -> str:
    """In-band marker for a `command_execution` item codex started running."""
    cmd_disp = cmd.strip()
    if len(cmd_disp) > 160:
        cmd_disp = cmd_disp[:160] + "…"
    return f"\n\n🔧 bash {cmd_disp}\n"


def _format_codex_command_completed(item: dict) -> str:
    """In-band marker for a completed `command_execution` item.

    Uses the same `↳ <head> (N chars)` shape as `_format_claude_tool_result`
    so the two delegates look uniform in saved transcripts.
    """
    out = item.get("aggregated_output") or ""
    if not isinstance(out, str):
        out = str(out)
    exit_code = item.get("exit_code")
    n = len(out)
    head = next((line for line in out.splitlines() if line.strip()), "")
    summary = head.strip()[:140]
    arrow = "✗" if isinstance(exit_code, int) and exit_code != 0 else "↳"
    if summary:
        return f"{arrow} {summary} ({n} chars)\n"
    return f"{arrow} ({n} chars)\n"


async def _run_codex_compact_resume(
    *,
    thread_id: str,
    mode_flags: list[str],
    spawn_cwd: str | None,
    args: dict[str, Any],
    model_flags: tuple[str, ...] | list[str] = (),
) -> tuple[bool, str]:
    """Run a hidden ``/compact`` turn against an existing Codex thread.

    Codex owns its conversation state internally, so sharedagents cannot trim
    the prompt the way ``LocalLLMEngine`` trims OpenAI-style message
    lists. When Codex reports a context-window overflow on a resumed
    thread, the only safe recovery is to ask Codex itself to compact that
    thread, then retry the original prompt once. ``model_flags`` are the
    turn's own ``--model`` flags: without them ``exec resume`` compacts on
    codex's config default, whose window may be smaller than the thread's.
    """
    _load_tool_utils(globals())
    cmd = (
        [_resolve_cli_executable("codex"), "exec", "--json", "--skip-git-repo-check"]
        + ["resume", thread_id]
        + list(model_flags)
        + list(mode_flags)
    )
    proc: asyncio.subprocess.Process | None = None
    lease_id: str | None = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=16 * 1024 * 1024,
            start_new_session=True,
            cwd=spawn_cwd,
            env=_build_delegate_spawn_env(args),
        )
        lease_id = sandbox_broker.lease_process(pgid=proc.pid, kind="cli")
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(b"/compact\n"),
                timeout=min(max(30.0, _delegate_idle_timeout_s()), 300.0),
            )
        except asyncio.TimeoutError:
            _kill_tree(proc, signal.SIGKILL)
            return False, "codex /compact timed out"

        if proc.returncode != 0:
            msg = stderr.decode("utf-8", "replace")[:400]
            return False, msg or _signal_name(proc.returncode)

        failed_msg = ""
        for raw in stdout.splitlines():
            try:
                event = json.loads(raw.decode("utf-8", "replace"))
            except json.JSONDecodeError:
                continue
            if event.get("type") != "turn.failed":
                continue
            err = event.get("error") or {}
            if isinstance(err, dict):
                failed_msg = str(err.get("message") or err.get("type") or "codex /compact failed")
            else:
                failed_msg = str(err or "codex /compact failed")
            break
        if failed_msg:
            return False, failed_msg
        return True, ""
    except FileNotFoundError:
        return False, "`codex` CLI not on PATH"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    finally:
        sandbox_broker.release_process(lease_id)
        if proc is not None and proc.returncode is None:
            try:
                _kill_tree(proc, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass


async def run_delegate_codex_async(
    args: dict[str, Any],
    *,
    on_delta: Callable[[str], None] | None = None,
    on_event: Callable[[dict], None] | None = None,
    on_spawn: Callable[[Any], None] | None = None,
) -> ToolResult:
    """Spawn `codex exec ...` and return its full reply as the tool result.

    .. deprecated:: refactor-phase-4
       Legacy delegate-as-tool entry point. The new
       :class:`sharedagents_agent.agents.codex_engine.CodexCliEngine`
       reuses this via :meth:`delegate_runner`; the function stays
       until Phase 7's legacy sweep.


    Streams `codex exec --json` and forwards each `agent_message` text
    plus a compact in-band marker for every `command_execution` item
    Codex emits, all through the same `on_delta` callback. The
    slash-command bridge (`/codex`) and the autonomous-tool path
    (`tool_delta` SSE) both consume that callback, so live progress —
    including which shell commands Codex itself is running under the
    hood — shows up in the TUI, Telegram, and any other SSE consumer
    with no per-frontend changes.

    Per-sharedagents-session state: reads `codex_thread_id` from
    `<sessions_dir>/<sid>.delegate_codex.json` and uses
    `codex exec resume <id>` so a follow-up `delegate_codex` call
    continues the same Codex conversation. The new id is written back
    on success.

    Note: codex doesn't stream agent text token-by-token like
    `claude -p` does — `agent_message` arrives as one consolidated
    `item.completed` event. We surface it as a single chunk on
    `on_delta`, which is fine for the SSE consumer (the user just sees
    a delayed burst rather than a typewriter effect).

    Killability: a SIGTERM landing on this coroutine (via the parent
    `tool_runner`'s signal handler) propagates through CancelledError;
    the `finally` block kills the codex subprocess so it doesn't keep
    running orphaned.
    """
    _load_tool_utils(globals())
    prompt = args.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return _err("prompt required")

    # Engine-pipeline session identity + mode — see :func:`run_delegate_claude_async`
    # for rationale. Concurrent CLI turns on the same agent must not
    # race on process-wide ``os.environ``.
    sa_sid_arg = args.get("_sharedagents_session_id")
    sa_sid: str | None = sa_sid_arg if isinstance(sa_sid_arg, str) and sa_sid_arg else None
    sa_sdir_arg = args.get("_sharedagents_sessions_dir")
    sa_sdir: str | None = sa_sdir_arg if isinstance(sa_sdir_arg, str) and sa_sdir_arg else None
    sa_perm_arg = args.get("_sharedagents_permission_mode")
    sa_perm: str = sa_perm_arg if isinstance(sa_perm_arg, str) and sa_perm_arg else ""

    mode_raw = args.get("mode")
    if isinstance(mode_raw, str) and mode_raw in _CODEX_MODE_FLAGS:
        mode = mode_raw
    else:
        # Inherit from the sharedagents permission mode the agent is running
        # in. Engine callers pass it via ``_sharedagents_permission_mode``;
        # legacy direct callers fall back to env. ``ask`` is the safe
        # fallback when neither yields a known mode.
        sa_mode = sa_perm or os.environ.get("SHAREDAGENTS_PERMISSION_MODE", "")
        mode = sa_mode if sa_mode in _CODEX_MODE_FLAGS else "ask"

    mode_flags = _CODEX_MODE_FLAGS[mode]

    # Optional `cwd`: validate before spawn (same rationale as
    # delegate_claude — fail loud on typos rather than crash subprocess).
    spawn_cwd, cwd_err = _resolve_delegate_cwd(args)
    if cwd_err is not None:
        _log_delegate_error(
            "codex",
            "bad_cwd",
            prompt=prompt,
            cwd=safe_getcwd(),
            mode=mode,
            message=cwd_err,
        )
        return _err(cwd_err)

    # Resume-id override (new Agent/Engine pipeline). When the caller
    # is :class:`ExternalCliEngine`, the engine manages its own
    # sidecar and passes the resume id explicitly via
    # ``_force_resume_id``. ``_sharedagents_codex_no_resume`` (if added)
    # is also respected for parity with the claude branch.
    forced_resume = args.get("_force_resume_id")
    force_fresh = bool(args.get("_sharedagents_codex_no_resume"))
    if isinstance(forced_resume, str) and forced_resume:
        resume_id = None if force_fresh else forced_resume
    else:
        resume_id = None if force_fresh else _read_codex_thread_id(sa_sid, sa_sdir)

    # System-prompt priming for the new Agent/Engine pipeline. Codex
    # has no ``--system-prompt`` flag, so we inject the composed
    # SOUL.md / AGENTS.md / TOOLS.md as a leading user-message
    # fragment in the stdin payload.
    appended_sys = args.get("_append_system_prompt")
    if isinstance(appended_sys, str) and appended_sys.strip():
        # Prepend with a clear separator so codex treats the agent
        # context as instruction prelude rather than user request.
        prompt = f"<system>\n{appended_sys.strip()}\n</system>\n\n---\n\n{prompt}"

    # `--skip-git-repo-check` so codex doesn't refuse to run when the
    # cwd isn't a git repo (sharedagents agents may launch from anywhere).
    # Prompt is piped on stdin (see delegate_claude rationale): argv
    # would hit ARG_MAX (~128 KiB) and crash with E2BIG. `codex exec`
    # reads stdin when no positional PROMPT is given.
    base = [
        _resolve_cli_executable("codex"),
        "exec",
        "--json",
        "--skip-git-repo-check",
        *_codex_effort_flags(args),
        *_codex_reasoning_summary_flags(),
    ]
    # ``--model`` rides EVERY spawn, fresh or resumed. A resumed thread does
    # NOT keep the model it was created with: ``codex exec resume`` applies
    # codex's own config.toml default unless ``--model`` is passed (2026-09-23:
    # 20 of 32 multi-turn threads silently ran that default; thread 01a0ca5e
    # ran gpt-6-sol once, then gpt-5.6-sol for 7 turns while the picker showed
    # gpt-6-sol). The Engine resolves the user's ``/model`` pick to a wire id
    # in ``args['model']``.
    from sharedagents_agent.discovery._common import (
        codex_effective_default_model,
        codex_model_refusal,
        validate_codex_model,
    )

    model_override = args.get("model")
    model_flags: list[str] = []
    if isinstance(model_override, str) and model_override.strip():
        # A9 (plan v2): fail fast on foreign model ids BEFORE the spawn — the
        # ChatGPT-backend codex CLI rejects them with a 400 "model not
        # supported" (a codex delegate must never inherit a local parent's
        # model override). The literal engine label "codex" is refused here
        # too, so it can never reach ``codex --model``. validate_codex_model
        # returns the CANONICAL slug from the installed codex's own model
        # list, and ONLY that canonical value is
        # built into ``--model``, never the raw user string ("GPT-5.6-SOL"
        # spawns as ``--model gpt-5.6-sol``). The refusal copy names the
        # actual source: direct callers pass an explicit override.
        _canonical_model = validate_codex_model(model_override)
        if _canonical_model is None:
            return ToolResult(
                content=codex_model_refusal(model_override, source="explicit"),
                ok=False,
            )
        if _canonical_model:
            model_flags = ["--model", _canonical_model]
    else:
        # No override. A resumed thread continues on the model it last ran
        # (read from its rollout), so a follow-up call keeps its model instead
        # of drifting to codex's config default; a model codex no longer
        # offers is skipped. Otherwise (fresh thread, or nothing usable
        # recorded) OPTION B: the effective default
        # is passed EXPLICITLY so the stamped model is the model that runs.
        # There is no pinned fallback: ``""`` builds no ``--model`` and codex
        # applies its own default.
        _thread_model = _codex_rollout_model(resume_id) if resume_id else ""
        _canonical_thread = validate_codex_model(_thread_model) if _thread_model else ""
        if _canonical_thread:
            model_flags = ["--model", _canonical_thread]
        else:
            # The default rides the SAME gate as an
            # explicit override and FAILS CLOSED if an invalid id ever gets
            # here, instead of a raw id reaching ``codex --model`` argv.
            _default_model = codex_effective_default_model()
            _canonical_default = validate_codex_model(_default_model)
            if _canonical_default is None:
                return ToolResult(
                    content=codex_model_refusal(_default_model, source="default"),
                    ok=False,
                )
            if _canonical_default:
                model_flags = ["--model", _canonical_default]
    if resume_id:
        # `codex exec resume <id>`. Mode and model flags come AFTER
        # `resume <id>` because of how the codex argparse subcommand is
        # structured: positional `[SESSION_ID]` then options on the `resume`
        # subcommand (which accepts ``-m/--model``).
        cmd = base + ["resume", resume_id] + model_flags + list(mode_flags)
    else:
        cmd = base + model_flags + list(mode_flags)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=16 * 1024 * 1024,
            # Own process group; see delegate_claude.
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
            "codex",
            "spawn_failed",
            prompt=prompt,
            cwd=spawn_cwd,
            mode=mode,
            resume_id=resume_id,
            message="`codex` CLI not on PATH",
        )
        return _err("`codex` CLI not on PATH. Install: https://github.com/openai/codex")

    stdin_task = asyncio.create_task(_feed_stdin(proc, prompt.encode("utf-8")))

    text_buf: list[str] = []
    answer_sink = CliAnswerSink()
    reasoning_chars = 0
    final_thread_id: str | None = resume_id
    reported_model = ""
    # Codex's ``turn.completed`` usage is CUMULATIVE across the whole
    # thread (openai/codex#17539), and sharedagents resumes the same thread
    # across invocations (idle-empty continues within a turn, sidecar
    # resume across turns). Billing the raw event would re-bill every
    # prior call, so we bill only the DELTA vs the last-seen cumulative
    # and persist that baseline in the sidecar. When RESUMING an existing
    # thread, carry the prior baseline from the sidecar; a fresh thread
    # starts at zero (last_cumulative stays None).
    _codex_state = _read_codex_state(sa_sid, sa_sdir)
    last_cumulative: dict[str, int] | None = None
    last_cum_thread_id: str | None = None
    if resume_id is not None:  # resuming → carry the prior cumulative baseline
        _stored_thread = _codex_state.get("last_cumulative_thread_id")
        _stored_cum = _codex_state.get("last_cumulative_usage")
        if _stored_thread == resume_id and isinstance(_stored_cum, dict):
            last_cumulative = _stored_cum
            last_cum_thread_id = _stored_thread
    is_error = False
    error_msg: str | None = None
    error_event: dict[str, Any] = {}
    # Per-id buffers for command_execution items so we don't print the
    # `🔧 bash …` marker twice when both `item.started` and
    # `item.completed` carry the command.
    cmd_announced: set[str] = set()
    reasoning_seen_ids: set[str] = set()
    # One trace per reasoning item: deltas stream, the restatement forwards
    # only what the deltas have not already produced.
    thinking_relay = ReasoningTextRelay()

    def _emit_model(value: Any) -> None:
        nonlocal reported_model
        if not isinstance(value, str):
            return
        model = value.strip()
        if not model or model.lower() == "codex" or model == reported_model:
            return
        reported_model = model
        _update_codex_state({"actual_model": model}, sa_sid, sa_sdir)
        if on_event is not None:
            try:
                # ExternalCliEngine persists this in <sid>.codex_cli.json
                # and uses it for both live checkpoints and terminal metrics.
                on_event({"kind": "model", "id": model})
            except Exception:  # noqa: BLE001
                pass

    def _note_reasoning(value: Any, *, identity: str = "") -> None:
        """Record only the size of private reasoning; never its text."""
        nonlocal reasoning_chars
        if identity:
            if identity in reasoning_seen_ids:
                return
            reasoning_seen_ids.add(identity)
        count = _reasoning_signal_chars(value)
        if count > 0:
            reasoning_chars += count

    def _emit_thinking(
        text: str,
        *,
        identity: str = "",
        consolidated: bool = False,
    ) -> None:
        """Forward Codex reasoning text once per item.

        Codex restates a reasoning item on every ``item.*`` event (each payload
        is the block's cumulative text, and ``item.completed`` repeats the
        last one), so each restatement only contributes its unseen tail.
        """
        if not text:
            return
        key = identity or "reasoning"
        kept = thinking_relay.consolidated(key, text) if consolidated else thinking_relay.delta(key, text)
        if not kept or on_event is None:
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
        """Append an incremental answer token if Codex ever streams one."""
        _emit_answer(answer_sink.push_delta(chunk))

    def _push_consolidated(text: str) -> None:
        """Append a completed agent_message, skipping restatements."""
        _emit_answer(answer_sink.push_consolidated(text))

    def _push_marker(chunk: str) -> None:
        """Append a tool marker to text_buf so the persisted body
        keeps its inline trace, but skip ``on_delta`` when
        ``on_event`` is wired — the structured event already drives
        a native widget on the frontend, so duplicating it as inline
        text would render twice.
        """
        if not chunk:
            return
        text_buf.append(chunk)
        if on_event is None and on_delta is not None:
            try:
                on_delta(chunk)
            except Exception:  # noqa: BLE001
                pass

    try:
        if proc.stdout is None:
            raise RuntimeError("subprocess stdout pipe is None — spawn config bug")
        first_token_timeout_s = _delegate_first_token_timeout_s()
        idle_timeout_s = _delegate_idle_timeout_s()
        got_first_event = False
        while True:
            try:
                line = await _delegate_readline(
                    proc.stdout,
                    got_first_event=got_first_event,
                    first_token_timeout_s=first_token_timeout_s,
                    idle_timeout_s=idle_timeout_s,
                )
            except asyncio.TimeoutError:
                stage = "startup" if not got_first_event else "idle"
                secs = int(first_token_timeout_s if not got_first_event else idle_timeout_s)
                # codex's finally does an UNBOUNDED proc.wait(); kill the
                # group first so it returns instead of hanging on a stalled
                # child.
                _kill_group(proc, signal.SIGTERM)
                _kill_group(proc, signal.SIGKILL)
                _log_delegate_error(
                    "codex",
                    f"{stage}_timeout",
                    prompt=prompt,
                    cwd=spawn_cwd,
                    mode=mode,
                    resume_id=resume_id,
                    remote_session_id=final_thread_id,
                    returncode=proc.returncode,
                )
                return _err(
                    f"codex produced no output for {secs}s ({stage} timeout) "
                    "— the CLI appears stalled. Aborted instead of hanging; "
                    "tune via SHAREDAGENTS_DELEGATE_FIRST_TOKEN_TIMEOUT_S / "
                    "SHAREDAGENTS_DELEGATE_IDLE_TIMEOUT_S."
                )
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
            etype = event.get("type")
            if etype == "thread.started":
                tid = event.get("thread_id")
                if isinstance(tid, str) and tid:
                    final_thread_id = tid
                    # Record the thread the moment codex opens it, not only
                    # when the turn completes: a Stop mid-turn otherwise
                    # leaves no id to resume, the next turn opens a fresh
                    # thread, and codex never sees the stopped request.
                    # The sidecar write does not depend on the callback;
                    # the engine ignores an id it already holds.
                    _write_codex_thread_id(tid, sa_sid, sa_sdir)
                    if on_event is not None:
                        try:
                            on_event({"kind": "session_id", "id": tid})
                        except Exception:  # noqa: BLE001
                            pass
                _emit_model(event.get("model"))
                continue
            if etype in {"turn.started", "turn_context"}:
                context = event.get("payload") or event
                if isinstance(context, dict):
                    _emit_model(context.get("model"))
                continue
            if etype == "turn.completed":
                # Resolve before emitting usage: the checkpoint must carry
                # the model that ran, including a resumed thread's model.
                # This turn's own evidence wins (its event, the model the
                # stream reported, then the --model flag, which rides every
                # spawn that has a model); the rollout is last because its
                # newest record can still be the PREVIOUS turn's.
                _emit_model(
                    event.get("model")
                    or reported_model
                    or (model_flags[-1] if model_flags else "")
                    or _codex_rollout_model(final_thread_id)
                )
                # Codex reports the WHOLE thread's cumulative totals here,
                # not this invocation's delta — bill only what is new.
                # ``final_thread_id`` holds the current thread (set from
                # ``thread.started``, which always precedes
                # ``turn.completed``; falls back to ``resume_id``).
                raw_cum = _codex_raw_cumulative(event)
                if raw_cum is not None:
                    if final_thread_id and last_cum_thread_id not in (None, final_thread_id):
                        last_cumulative = None  # thread changed mid-run — reset baseline
                    delta = _codex_usage_delta(raw_cum, last_cumulative)
                    if delta is not None and on_event is not None:
                        try:
                            on_event({"kind": "usage", "usage": delta})
                        except Exception:  # noqa: BLE001
                            pass
                    last_cumulative = raw_cum
                    last_cum_thread_id = final_thread_id
                    _update_codex_state(
                        {
                            "last_cumulative_usage": raw_cum,
                            "last_cumulative_thread_id": final_thread_id,
                        },
                        sa_sid,
                        sa_sdir,
                    )
                continue
            if etype in ("item.started", "item.updated", "item.completed"):
                item = event.get("item") or {}
                itype = item.get("type")
                iid = item.get("id") or ""
                if itype in _REASONING_BLOCK_TYPES:
                    # Codex emits reasoning as private ``agent_reasoning`` /
                    # ``reasoning`` items.  Count a signal only; never push
                    # its text into the answer stream or persisted result.
                    _note_reasoning(
                        {"type": itype, **item},
                        identity=str(iid or ""),
                    )
                    thought = item.get("text") or item.get("summary") or item.get("reasoning") or ""
                    if isinstance(thought, list):
                        thought = "\n".join(str(x) for x in thought if x)
                    if thought:
                        # Item payloads are cumulative snapshots of the block
                        # (``item.updated`` restates what it already sent), so
                        # every one of them is a restatement, never an
                        # increment: only the unseen tail is forwarded.
                        _emit_thinking(
                            str(thought),
                            identity=str(iid or ""),
                            consolidated=True,
                        )
                elif itype == "agent_message":
                    # Agent text arrives only on `item.completed` (codex
                    # doesn't stream tokens). Push the full body once,
                    # unless it restates a prior agent_message.
                    if etype == "item.completed":
                        txt = item.get("text") or ""
                        if txt:
                            _push_consolidated(txt)
                            if on_event is not None:
                                for reference in _codex_local_artifact_references(
                                    txt,
                                    spawn_cwd,
                                ):
                                    try:
                                        on_event(
                                            {
                                                "kind": "artifact_reference",
                                                "reference": reference,
                                            }
                                        )
                                    except Exception:  # noqa: BLE001
                                        pass
                elif itype in {"todo_list", "update_plan"}:
                    # Codex plan tool → exec JSON ``todo_list`` items
                    # ({text, completed}) or raw ``update_plan``
                    # ({plan:[{step,status}]}). Surface as a todo widget
                    # so Web/TUI/Telegram + HarnessTaskMirror see it.
                    if on_event is not None:
                        raw_items = item.get("items")
                        raw_plan = item.get("plan")
                        inp: dict[str, Any] = {}
                        if isinstance(raw_items, list):
                            inp["items"] = raw_items
                        if isinstance(raw_plan, list):
                            inp["plan"] = raw_plan
                        try:
                            on_event(
                                {
                                    "kind": "tool_use",
                                    "id": iid or "todo_list",
                                    "name": "todo_list" if itype == "todo_list" else "update_plan",
                                    "input": inp,
                                }
                            )
                            on_event(
                                {
                                    "kind": "tool_result",
                                    "tool_use_id": iid or "todo_list",
                                    "ok": True,
                                    "text": "",
                                }
                            )
                        except Exception:  # noqa: BLE001
                            pass
                elif itype == "command_execution":
                    if etype == "item.updated":
                        continue
                    cmd = item.get("command") or "?"
                    if etype == "item.started":
                        if iid not in cmd_announced:
                            cmd_announced.add(iid)
                            _push_marker(_format_codex_command_started(cmd))
                            if on_event is not None:
                                try:
                                    on_event(
                                        {
                                            "kind": "tool_use",
                                            "id": iid,
                                            "name": "bash",
                                            "input": {"command": cmd},
                                        }
                                    )
                                except Exception:  # noqa: BLE001
                                    pass
                    else:  # item.completed
                        # Announce here too if we missed the start event
                        # (race / dropped frame), then push the result.
                        if iid not in cmd_announced:
                            cmd_announced.add(iid)
                            _push_marker(_format_codex_command_started(cmd))
                            if on_event is not None:
                                try:
                                    on_event(
                                        {
                                            "kind": "tool_use",
                                            "id": iid,
                                            "name": "bash",
                                            "input": {"command": cmd},
                                        }
                                    )
                                except Exception:  # noqa: BLE001
                                    pass
                        _push_marker(_format_codex_command_completed(item))
                        if on_event is not None:
                            out = item.get("aggregated_output") or ""
                            if not isinstance(out, str):
                                out = str(out)
                            exit_code = item.get("exit_code")
                            ok = not (isinstance(exit_code, int) and exit_code != 0)
                            try:
                                on_event(
                                    {
                                        "kind": "tool_result",
                                        "tool_use_id": iid,
                                        "ok": ok,
                                        "text": out,
                                    }
                                )
                            except Exception:  # noqa: BLE001
                                pass
                # Other item types (web_search, file_search, mcp tool
                # calls) are intentionally ignored:
                # reasoning bloats context and the others don't have a
                # stable enough shape to render usefully yet.
                continue
            if etype in _REASONING_BLOCK_TYPES or etype in _REASONING_DELTA_TYPES:
                _note_reasoning(event, identity=str(event.get("id") or ""))
                thought = (
                    event.get("text") or event.get("summary") or event.get("reasoning") or event.get("delta") or ""
                )
                if isinstance(thought, dict):
                    thought = thought.get("text") or thought.get("reasoning") or thought.get("delta") or ""
                if isinstance(thought, list):
                    thought = "\n".join(str(x) for x in thought if x)
                if thought:
                    # A bare ``*_delta`` event is an increment; a bare
                    # ``agent_reasoning``/``reasoning`` event restates the
                    # block, so only its unseen tail is forwarded.
                    _emit_thinking(
                        str(thought),
                        identity=str(event.get("id") or ""),
                        consolidated=etype in _REASONING_BLOCK_TYPES,
                    )
                continue
            if etype == "turn.failed":
                is_error = True
                err = event.get("error") or {}
                if isinstance(err, dict):
                    error_msg = err.get("message") or err.get("type") or "codex turn failed"
                    error_event = event
                else:
                    error_msg = str(err) or "codex turn failed"
                    error_event = event
                continue
    except asyncio.CancelledError:
        _log_delegate_error(
            "codex",
            "cancelled",
            prompt=prompt,
            cwd=spawn_cwd,
            mode=mode,
            resume_id=resume_id,
            remote_session_id=final_thread_id,
            returncode=proc.returncode,
            codex_thread_id=final_thread_id,
        )
        # Same whole-tree hard-kill rationale as delegate_claude.
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
        try:
            await proc.wait()
        except Exception:  # noqa: BLE001
            pass

    async def _retry_after_context_compact(
        message_preview: str,
    ) -> ToolResult | None:
        if not final_thread_id or bool(args.get("_sharedagents_codex_compact_retry_done")):
            return None
        log_event(
            "DELEGATE_CODEX_CONTEXT_COMPACT_RETRY",
            sharedagents_session_id=sa_sid or os.environ.get("SHAREDAGENTS_SESSION_ID"),
            agent_name=os.environ.get("SHAREDAGENTS_AGENT_NAME"),
            mode=mode,
            resume_id=resume_id,
            codex_thread_id=final_thread_id,
            prompt_chars=len(prompt),
            message_preview=message_preview[:1000],
        )
        compact_ok, compact_msg = await _run_codex_compact_resume(
            thread_id=final_thread_id,
            mode_flags=mode_flags,
            spawn_cwd=spawn_cwd,
            args=args,
            model_flags=model_flags,
        )
        log_event(
            "DELEGATE_CODEX_CONTEXT_COMPACT_DONE",
            sharedagents_session_id=sa_sid or os.environ.get("SHAREDAGENTS_SESSION_ID"),
            agent_name=os.environ.get("SHAREDAGENTS_AGENT_NAME"),
            mode=mode,
            resume_id=resume_id,
            codex_thread_id=final_thread_id,
            ok=compact_ok,
            message_preview=str(compact_msg or "")[:1000],
        )
        if not compact_ok:
            return None
        retry_args = dict(args)
        retry_args["_force_resume_id"] = final_thread_id
        retry_args["_sharedagents_codex_compact_retry_done"] = True
        return await run_delegate_codex_async(
            retry_args,
            on_delta=on_delta,
            on_event=on_event,
            on_spawn=on_spawn,
        )

    if proc.returncode != 0 and not is_error:
        stderr_bytes = b""
        if proc.stderr is not None:
            try:
                stderr_bytes = await proc.stderr.read()
            except Exception:  # noqa: BLE001
                pass
        stderr_text = stderr_bytes.decode("utf-8", "replace")
        err_msg = stderr_text[:400] or _signal_name(proc.returncode)
        if _looks_like_context_overflow(stderr_text or err_msg):
            retry_result = await _retry_after_context_compact(err_msg)
            if retry_result is not None:
                return retry_result
        _log_delegate_error(
            "codex",
            "nonzero_exit",
            prompt=prompt,
            cwd=spawn_cwd,
            mode=mode,
            resume_id=resume_id,
            remote_session_id=final_thread_id,
            returncode=proc.returncode,
            stderr=stderr_text,
            codex_thread_id=final_thread_id,
        )
        return _err(f"codex exited {proc.returncode}: {err_msg}")

    if is_error:
        overflow_text = f"{error_msg or ''} {json.dumps(error_event, ensure_ascii=False)}"
        if _looks_like_context_overflow(overflow_text):
            retry_result = await _retry_after_context_compact(str(error_msg or ""))
            if retry_result is not None:
                return retry_result
        _log_delegate_error(
            "codex",
            "reported_error",
            prompt=prompt,
            cwd=spawn_cwd,
            mode=mode,
            resume_id=resume_id,
            remote_session_id=final_thread_id,
            returncode=proc.returncode,
            message=error_msg,
            event=error_event,
            codex_thread_id=final_thread_id,
        )
        return _err(f"codex reported error: {error_msg or 'unknown'}")

    body_text = "".join(text_buf).strip()
    if not body_text and reasoning_chars <= 0:
        _log_delegate_error(
            "codex",
            "empty_response",
            prompt=prompt,
            cwd=spawn_cwd,
            mode=mode,
            resume_id=resume_id,
            remote_session_id=final_thread_id,
            returncode=proc.returncode,
            codex_thread_id=final_thread_id,
        )
        return _err("codex returned empty response")

    if reasoning_chars > 0 and on_event is not None:
        try:
            on_event({"kind": "reasoning", "chars": reasoning_chars})
        except Exception:  # noqa: BLE001
            pass

    if final_thread_id:
        _write_codex_thread_id(final_thread_id, sa_sid, sa_sdir)

    header_bits = []
    if final_thread_id:
        header_bits.append(f"thread={final_thread_id[:12]}")
    header_bits.append(f"mode={mode}")
    if resume_id:
        header_bits.append("resumed")
    header = f"[codex {' · '.join(header_bits)}]"

    # No "send only NEW context" footer: continuity is owned by the
    # session-id link (``codex exec resume <id>``), not a per-result
    # note. The orchestrator-side contract lives once in the
    # delegate-strategy skill instead of riding every tool result (where
    # it also leaked into user-facing output). Consistent with
    # delegate_claude / delegate_antigravity.
    body = f"{header}\n\n{body_text}" if body_text else header
    body, truncated = _truncate(body)
    return ToolResult(content=body, ok=True, truncated=truncated)
