"""ZCode CLI delegate runner (Z.AI GLM Coding Plan).

Drives the official Z.AI **ZCode** CLI (``zcode``, npm package
``zcode-app-cli``) in headless print mode::

    zcode --json --no-color --mode <build|edit|yolo> -p <prompt>

There is deliberately NO ``--model`` on that argv: ``zcode`` exposes no such
flag for headless prompts (``Unknown option '--model'``, exit 1 — verified
against ``zcode 0.16.9``). The model is selected by the CLI's own
provider config (``~/.zcode/v2/provider_config.json`` →
``config.defaultModelSelection.modelId``), so the delegate reports the
configured model and refuses an override it cannot apply.

Auth is the CLI's OWN Z.AI OAuth login stored in
``~/.zcode/v2/credentials.json`` (``zcode login``) — the same pattern the
claude / codex runners use with ``~/.claude`` / ``~/.codex``. There is no
sharedagents-side token and no hardcoded credential. When the operator also
exports ``ZCODE_API_KEY`` (the coding-plan key ZCode provisions for API
access), it is passed to THIS subprocess only, through the per-delegate
override channel — see :func:`_zcode_spawn_args` and ``delegate_env``.

Output contract (verified against ``zcode-app-cli 3.14.4-30`` /
``zcode-runtime 0.16.9``): headless prompts stream NDJSON
events on stdout via ``--output-format stream-json`` (one JSON object per
line, optionally preceded by plain-text notice lines such as ``ZCode
Built-in skipped (not-due)``). Event taxonomy observed live::

    {"type": "model.streaming", "payload": {"kind": "reasoning_start" |
        "reasoning_delta" | "reasoning_end" | "text_start" | "text_delta" |
        "text_end" | "tool_input_start" | "tool_input_delta" |
        "tool_input_end" | "tool_call" | "start" | "finish",
        "delta": "…", "toolCallId": "…", "toolName": "Bash",
        "input": {…}, "assistantMessageId": "…"}, "seq": n,
        "sessionId": "sess_…", "turnId": "turn_…", "timestamp": ms}
    {"type": "tool.updated", "payload": {"kind": "scheduled" | "started" |
        "result" | "batch", "toolCallId": "…", "toolName": "…",
        "input": {…}, "result": {"success": bool, "content": "…"}}}
    {"type": "model.request.started" | "model.request.completed",
        "payload": {"modelId": "GLM-5.3", "providerId": "…"}}
    {"type": "turn.completed", "payload": {"usage": {…camelCase…}}}
    {"type": "result", "sessionId": "sess_…", "response": "…",
        "usage": {"inputTokens": …, "outputTokens": …},
        "eventCount": 46,
        "projection": {"status": "idle", "contextWindow": 200000, …}}

The terminal ``result`` line is the SAME envelope the legacy ``--json``
mode prints as its whole output, so the final payload handling is shared.
``sessionId`` (``sess_<uuid>``) is the durable engine session id handed to
``--resume`` on the next call, persisted in the per-sharedagents-session
sidecar; it rides EVERY event line, so the id is captured (and the sidecar
primed) from the FIRST line — an abort mid-turn still resumes cleanly.

Streaming: this runner maps the events onto the shared delegate callback
contract as they arrive — ``reasoning_delta`` → ``on_event(thinking)``
(the model's visible thinking), ``text_delta`` → ``on_delta`` (the
answer), ``tool_call`` → ``on_event(tool_use)``, ``tool.updated/result``
→ ``on_event(tool_result)``, ``turn.completed`` → ``on_event(usage)``.
An older CLI without ``--output-format`` falls back to the legacy
``--json`` one-shot mode and emits a single delta, exactly as before.
The read loop is bounded by the ZCode wall budget plus a no-line idle
guard, and a cancel hard-kills the whole process tree.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sharedagents_agent import sandbox_broker
from sharedagents_core.cwdutil import safe_getcwd

from .base import (
    _delegate_state_path,
    _load_tool_utils,
    _log_delegate_error,
    _read_delegate_state,
    _resolve_delegate_cwd,
    _update_delegate_state,
)

if TYPE_CHECKING:
    # Bound into this module at call time by ``_load_tool_utils(globals())``;
    # a top-level import would close the tools <-> delegates import cycle.
    from sharedagents_agent.tools.impl._utils import (
        ToolResult,
        _build_delegate_spawn_env,
        _err,
        _kill_tree,
        _resolve_cli_executable,
        _signal_name,
        _truncate,
        _zcode_timeout_s,
    )

#: ``zcode --mode`` accepts exactly these values
#: (``zcode 0.16.9 --help``: "Permission mode for prompts: build, edit, or
#: yolo (default: yolo for --prompt)").
_ZCODE_MODE_FLAG_VALUES: frozenset[str] = frozenset({"build", "edit", "yolo"})

#: sharedagents requested mode id (``args['mode']`` / ``SHAREDAGENTS_PERMISSION_MODE``)
#: → ``zcode --mode`` value. ``auto`` is the autonomous posture sharedagents runs
#: delegates in and matches the CLI's own ``--prompt`` default.
_ZCODE_MODE_MAP: dict[str, str] = {
    "auto": "yolo",
    "yolo": "yolo",
    "ask": "edit",
    "default": "edit",
    "edit": "edit",
    "accept_edits": "edit",
    "plan": "build",
    "build": "build",
}

#: Where the CLI keeps its OAuth credentials + selected model. ZCODE_HOME-style
#: relocation is honoured for the model read the same way ``~/.zcode`` is.
_ZCODE_HOME_ENV = ("ZCODE_HOME", "ZCODE_STORAGE_DIR")


def _zcode_home() -> Path:
    for var in _ZCODE_HOME_ENV:
        raw = (os.environ.get(var) or "").strip()
        if raw:
            return Path(raw).expanduser()
    return Path.home() / ".zcode"


def _zcode_state_path(
    sa_sid: str | None = None,
    sa_sdir: str | None = None,
) -> Path | None:
    """Resolve the per-sharedagents-session state file for ``delegate_zcode``."""
    return _delegate_state_path("zcode", sa_sid, sa_sdir)


def _read_zcode_session_id(
    sa_sid: str | None = None,
    sa_sdir: str | None = None,
) -> str | None:
    sid = _read_delegate_state("zcode", sa_sid, sa_sdir).get("zcode_session_id")
    return sid if isinstance(sid, str) and sid else None


def _write_zcode_session_id(
    zcode_sid: str,
    sa_sid: str | None = None,
    sa_sdir: str | None = None,
) -> None:
    _update_delegate_state("zcode", {"zcode_session_id": zcode_sid}, sa_sid, sa_sdir)


def _zcode_configured_model() -> str:
    """The model the CLI itself will run, from its own config.

    ``~/.zcode/v2/provider_config.json`` carries
    ``config.defaultModelSelection.modelId`` (``GLM-5.3`` for a GLM Coding
    Plan login). Read-only, best effort: an unreadable/absent file yields
    ``""`` and the caller simply does not stamp an actual model.
    """
    try:
        raw = (_zcode_home() / "v2" / "provider_config.json").read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return ""
    config = data.get("config")
    if not isinstance(config, dict):
        return ""
    selection = config.get("defaultModelSelection")
    if not isinstance(selection, dict):
        return ""
    model = selection.get("modelId")
    return model.strip() if isinstance(model, str) and model.strip() else ""


def _zcode_write_model_config(model_id: str) -> tuple[Path | None, str]:
    """Per-run provider config selecting ``model_id`` (shared config untouched).

    ``zcode -p`` has no ``--model`` flag and ``/model`` is not processed in
    headless prompts, but ``ZCODE_PERSONAL_PROVIDER_CONFIG_FILE`` points a single
    process at its own provider config. Copy the operator's config and swap
    ``config.defaultModelSelection.modelId``; verified live (the
    run's model requests carried the swapped id and ``~/.zcode`` was unchanged).
    Returns ``(path, "")`` or ``(None, reason)``. The caller removes the
    parent directory.
    """
    try:
        data = json.loads((_zcode_home() / "v2" / "provider_config.json").read_text(encoding="utf-8"))
        data["config"]["defaultModelSelection"]["modelId"] = model_id
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, f"the CLI's provider config is unreadable ({exc})"
    tmp = Path(tempfile.mkdtemp(prefix="sharedagents-zcode-model-"))
    path = tmp / "provider_config.json"
    try:
        path.write_text(json.dumps(data), encoding="utf-8")
        path.chmod(0o600)
    except OSError as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        return None, f"could not write the per-run config ({exc})"
    return path, ""


def _zcode_extract_result(stdout_text: str) -> dict[str, Any] | None:
    """Return the CLI's ``--json`` payload from a stdout blob, or ``None``.

    ``zcode --json`` may prefix the payload with plain-text notices and
    pretty-prints the object across many lines, so neither line-wise
    parsing nor ``json.loads(whole_blob)`` works. ``raw_decode`` from every
    ``{`` offset is notice- and nesting-proof; the OUTERMOST object (the
    earliest offset that decodes AND carries a payload marker) wins, so a
    nested ``projection`` object can never be mistaken for the envelope.
    """
    decoder = json.JSONDecoder()
    candidates: list[tuple[int, dict[str, Any]]] = []
    index = stdout_text.find("{")
    while index != -1:
        try:
            obj, _end = decoder.raw_decode(stdout_text, index)
        except ValueError:
            index = stdout_text.find("{", index + 1)
            continue
        if isinstance(obj, dict):
            candidates.append((index, obj))
        index = stdout_text.find("{", index + 1)
    if not candidates:
        return None
    for _offset, obj in candidates:  # earliest offset == outermost object
        if "sessionId" in obj or "response" in obj:
            return obj
    return candidates[0][1]


def _zcode_usage_to_snake(usage: Any) -> dict[str, int]:
    """Normalize the CLI's camelCase usage buckets to sharedagents metric keys.

    ``metrics_from_provider_usage`` reads snake_case provider fields
    (``input_tokens`` / ``cache_read_input_tokens`` / …), while ``zcode
    --json`` reports ``inputTokens`` / ``cacheReadTokens``.
    """
    if not isinstance(usage, dict) or not usage:
        return {}
    mapping = {
        "input_tokens": ("inputTokens", "input_tokens", "prompt_tokens"),
        "output_tokens": ("outputTokens", "output_tokens", "completion_tokens"),
        "cache_read_input_tokens": (
            "cacheReadTokens",
            "cache_read_input_tokens",
            "cached_read_tokens",
        ),
        "cache_creation_input_tokens": (
            "cacheWriteTokens",
            "cache_creation_input_tokens",
            "cached_write_tokens",
        ),
    }
    out: dict[str, int] = {}
    for target, sources in mapping.items():
        for key in sources:
            value = usage.get(key)
            if value is None:
                continue
            try:
                number = int(value)
            except (TypeError, ValueError):
                continue
            if number > 0:
                out[target] = number
            break
    return out


def _zcode_spawn_args(args: dict[str, Any]) -> dict[str, Any]:
    """``args`` plus this engine's OWN provider credential.

    ``zcode`` authenticates from ``~/.zcode/v2/credentials.json`` (Z.AI
    OAuth); the coding-plan key an operator exports as ``ZCODE_API_KEY`` is
    the SAME account's API credential and belongs to THIS delegate alone.
    It therefore rides the per-delegate override channel
    (``build_third_party_spawn_env``'s ``overrides``), which
    :mod:`sharedagents_agent.delegate_env` documents as "the delegate's OWN
    per-agent provider credentials … always included" — never the global
    allowlist, which would leak it to every other third-party CLI
    """
    key = (os.environ.get("ZCODE_API_KEY") or "").strip()
    if not key:
        return args
    merged = dict(args)
    overrides = merged.get("_spawn_env_overrides")
    overrides = dict(overrides) if isinstance(overrides, dict) else {}
    overrides.setdefault("ZCODE_API_KEY", key)
    merged["_spawn_env_overrides"] = overrides
    return merged


#: No-stdout-line stall guard for the streaming read loop. The CLI itself
#: retries stalled model streams (its own ``modelStream.idleTimeoutMs``, 60s
#: by default) and re-prints progress, so this only fires when the CLI is
#: truly wedged. Wall budget stays ``_zcode_timeout_s()``.
_ZCODE_STREAM_IDLE_S = 180.0

#: Longest stderr we keep for diagnostics.
_ZCODE_STDERR_KEEP = 4000


def _zcode_emit_stream_event(
    event: dict[str, Any],
    *,
    on_delta: Callable[[str], None] | None,
    on_event: Callable[[dict], None] | None,
    state: dict[str, Any],
) -> None:
    """Map ONE ``--output-format stream-json`` event onto the delegate contract.

    ``state`` accumulates the turn (streamed answer, first session id, the
    terminal ``result`` envelope, usage) so the caller can finish the runner's
    bookkeeping from it. Every callback is best-effort: a raising consumer must
    not kill the CLI read loop.
    """
    etype = str(event.get("type") or "")

    # Session id rides every line; the FIRST one primes --resume even if the
    # turn later aborts (mirrors the "session id FIRST" contract below).
    sid = event.get("sessionId")
    if isinstance(sid, str) and sid.strip() and not state.get("session_id"):
        state["session_id"] = sid.strip()
        if on_event is not None:
            try:
                on_event({"kind": "session_id", "id": sid.strip()})
            except Exception:  # noqa: BLE001
                pass

    if etype == "model.streaming":
        payload = event.get("payload")
        if not isinstance(payload, dict):
            return
        kind = str(payload.get("kind") or "")
        delta = payload.get("delta")
        delta_text = delta if isinstance(delta, str) else ""
        if kind == "text_delta" and delta_text:
            state["answer_parts"].append(delta_text)
            if on_delta is not None:
                try:
                    on_delta(delta_text)
                except Exception:  # noqa: BLE001
                    pass
        elif kind == "reasoning_delta" and delta_text:
            state["reasoning_chars"] += len(delta_text)
            if on_event is not None:
                try:
                    on_event({"kind": "thinking", "text": delta_text})
                except Exception:  # noqa: BLE001
                    pass
        elif kind == "tool_call":
            tool_input = payload.get("input")
            if on_event is not None:
                try:
                    on_event(
                        {
                            "kind": "tool_use",
                            "id": str(payload.get("toolCallId") or ""),
                            "name": str(payload.get("toolName") or "?"),
                            "input": (dict(tool_input) if isinstance(tool_input, dict) else {}),
                        }
                    )
                except Exception:  # noqa: BLE001
                    pass
        return

    if etype == "tool.updated":
        payload = event.get("payload")
        if not isinstance(payload, dict):
            return
        if str(payload.get("kind") or "") != "result":
            return
        result = payload.get("result")
        ok = bool(result.get("success")) if isinstance(result, dict) else True
        content = ""
        if isinstance(result, dict):
            raw_content = result.get("content")
            if isinstance(raw_content, str):
                content = raw_content
        if on_event is not None:
            try:
                on_event(
                    {
                        "kind": "tool_result",
                        "tool_use_id": str(payload.get("toolCallId") or ""),
                        "ok": ok,
                        # Bounded: a huge tool body must not double the event size.
                        "text": content[:8000],
                    }
                )
            except Exception:  # noqa: BLE001
                pass
        return

    if etype == "session.updated":
        # zcode-runtime 0.16.9 reports the model that actually runs the turn
        # here (``payload.modelId``, also nested in model_request_started).
        payload = event.get("payload")
        model = payload.get("modelId") if isinstance(payload, dict) else None
        if isinstance(model, str) and model.strip() and model.strip() != state.get("model_id"):
            state["model_id"] = model.strip()
            if on_event is not None:
                try:
                    on_event({"kind": "model", "id": model.strip()})
                except Exception:  # noqa: BLE001
                    pass
        return

    if etype == "model.request.started":
        payload = event.get("payload")
        model = payload.get("modelId") if isinstance(payload, dict) else None
        if isinstance(model, str) and model.strip() and on_event is not None:
            try:
                on_event({"kind": "model", "id": model.strip()})
            except Exception:  # noqa: BLE001
                pass
        return

    if etype == "turn.completed":
        payload = event.get("payload")
        usage = payload.get("usage") if isinstance(payload, dict) else None
        if isinstance(usage, dict) and usage:
            state["turn_usage"] = usage
            snake = _zcode_usage_to_snake(usage)
            if snake and on_event is not None:
                try:
                    on_event({"kind": "usage", "usage": snake})
                except Exception:  # noqa: BLE001
                    pass
        return

    if etype == "result":
        # Terminal envelope — the same object legacy ``--json`` prints.
        state["result"] = event
        return


async def run_delegate_zcode_async(
    args: dict[str, Any],
    *,
    on_delta: Callable[[str], None] | None = None,
    on_event: Callable[[dict], None] | None = None,
    on_spawn: Callable[[Any], None] | None = None,
) -> ToolResult:
    """Spawn ``zcode -p <prompt> --json`` and return its reply.

    Reused by :class:`sharedagents_agent.agents.zcode_engine.ZCodeCliEngine` via
    :meth:`delegate_runner`, so the autonomous-tool path
    (``delegate_zcode`` emitted as a tool call) and the agent-turn path
    converge on one backend.

    Per-sharedagents-session state: reads ``zcode_session_id`` from
    ``<sessions_dir>/<sid>.delegate_zcode.json`` and passes
    ``--resume <id>``, then writes the id back on success.

    Killability: a SIGTERM landing on this coroutine propagates as
    ``CancelledError``; the ``finally`` block hard-kills the whole
    subprocess tree so no orphaned ``zcode`` keeps burning plan quota.
    """
    _load_tool_utils(globals())
    prompt = args.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return _err("prompt required")

    # Engine-pipeline session identity + mode. Concurrent CLI turns on the
    # same agent must not race on process-wide ``os.environ``, so explicit
    # args win over the env fallback.
    sa_sid_arg = args.get("_sharedagents_session_id")
    sa_sid: str | None = sa_sid_arg if isinstance(sa_sid_arg, str) and sa_sid_arg else None
    sa_sdir_arg = args.get("_sharedagents_sessions_dir")
    sa_sdir: str | None = sa_sdir_arg if isinstance(sa_sdir_arg, str) and sa_sdir_arg else None
    sa_perm_arg = args.get("_sharedagents_permission_mode")
    sa_perm: str = sa_perm_arg if isinstance(sa_perm_arg, str) and sa_perm_arg else ""

    def _log_error(reason: str, **fields: Any) -> None:
        _log_delegate_error("zcode", reason, sharedagents_session_id=sa_sid, **fields)

    mode_raw = args.get("mode")
    if isinstance(mode_raw, str) and mode_raw in _ZCODE_MODE_MAP:
        zcode_mode = _ZCODE_MODE_MAP[mode_raw]
    else:
        sa_mode = sa_perm or os.environ.get("SHAREDAGENTS_PERMISSION_MODE", "")
        zcode_mode = _ZCODE_MODE_MAP.get(sa_mode, "yolo")
    if zcode_mode not in _ZCODE_MODE_FLAG_VALUES:  # pragma: no cover - map-guarded
        zcode_mode = "yolo"

    spawn_cwd, cwd_err = _resolve_delegate_cwd(args)
    if cwd_err is not None:
        _log_error(
            "bad_cwd",
            prompt=prompt,
            cwd=safe_getcwd(),
            mode=zcode_mode,
            message=cwd_err,
        )
        return _err(cwd_err)

    force_fresh = bool(args.get("_sharedagents_zcode_no_resume"))
    # Resume-id override (Agent/Engine pipeline): ExternalCliEngine owns its
    # own sidecar and passes the resume id explicitly.
    forced_resume = args.get("_force_resume_id")
    if isinstance(forced_resume, str) and forced_resume:
        resume_id = None if force_fresh else forced_resume
    else:
        resume_id = None if force_fresh else _read_zcode_session_id(sa_sid, sa_sdir)

    # System-prompt priming. ``zcode`` exposes no ``--system-prompt`` flag,
    # so the composed SOUL/AGENTS/TOOLS context rides as a leading
    # user-message fragment (codex/antigravity precedent).
    appended_sys = args.get("_append_system_prompt")
    appended_sys_text = appended_sys.strip() if isinstance(appended_sys, str) and appended_sys.strip() else ""
    if appended_sys_text:
        prompt = f"<system>\n{appended_sys_text}\n</system>\n\n---\n\n{prompt}"

    # Model gate. ``zcode`` exposes NO ``--model`` flag for headless prompts:
    # ``zcode --model X -p …`` is rejected by the CLI's own arg parser
    # (``Unknown option '--model'``, exit 1, usage on stderr), verified against
    # ``zcode 0.16.9`` on 2026-10-01. The model is chosen by the CLI's OWN
    # provider config (``~/.zcode/v2/provider_config.json`` →
    # ``config.defaultModelSelection.modelId``; package ``docs/CONFIGURATION.md``
    # §"defaultModelSelection"), so this gate must NOT build a flag — its job is
    # to refuse what the engine cannot honour:
    #
    # * a FOREIGN id (the parent's ``upstream_model``, e.g. a local vLLM id, or
    #   the literal engine label) would otherwise silently run GLM — refuse,
    #   fail closed;
    # * a valid GLM id that is not the CLI's configured model cannot be applied
    #   headlessly — refuse with the remedy rather than run (and report) a
    #   different model than the one asked for.
    from sharedagents_agent.discovery._common import (
        external_cli_model_refusal,
        validate_external_cli_model,
    )

    requested_model = args.get("model")
    configured_model = _zcode_configured_model()
    model_cfg_dir: Path | None = None
    if isinstance(requested_model, str) and requested_model.strip():
        canonical = validate_external_cli_model("zcode_cli", requested_model)
        if canonical is None:
            return ToolResult(
                content=external_cli_model_refusal(
                    requested_model,
                    "zcode_cli",
                    source="explicit",
                ),
                ok=False,
            )
        if canonical:
            from sharedagents_agent.discovery.zcode import zcode_catalog_spelling

            exact = zcode_catalog_spelling(canonical)
            if not exact and configured_model and (canonical.lower() == configured_model.lower()):
                exact = configured_model
            if not exact:
                return ToolResult(
                    content=external_cli_model_refusal(
                        requested_model,
                        "zcode_cli",
                        source="explicit",
                    ),
                    ok=False,
                )
            if exact.lower() != (configured_model or "").lower():
                model_cfg_path, cfg_err = _zcode_write_model_config(exact)
                if model_cfg_path is None:
                    return _err(f"zcode model '{exact}' could not be selected: {cfg_err}")
                model_cfg_dir = model_cfg_path.parent
                spawn_args = dict(args)
                overrides = dict(spawn_args.get("_spawn_env_overrides") or {})
                overrides["ZCODE_PERSONAL_PROVIDER_CONFIG_FILE"] = str(model_cfg_path)
                spawn_args["_spawn_env_overrides"] = overrides
                args = spawn_args

    # ``zcode`` takes the prompt as the ``-p`` ARGUMENT and has no stdin or
    # file prompt path (``zcode 0.16.9 --help``), so argv is the only
    # transport — same constraint as the antigravity runner. There is no
    # ``--model`` here on purpose: the CLI rejects it (see the gate above), and
    # a resumed session keeps its own bound model regardless.
    def _build_argv(stream: bool) -> list[str]:
        cmd = [_resolve_cli_executable("zcode")]
        # ``--output-format stream-json`` (zcode-runtime >= 0.16.9) streams
        # NDJSON turn events — thinking, answer text, tool calls — as they
        # happen; ``--json`` is the legacy one-envelope-at-exit mode kept as
        # the fallback for older installs.
        cmd += ["--output-format", "stream-json"] if stream else ["--json"]
        cmd += ["--no-color", "--mode", zcode_mode]
        if resume_id:
            cmd += ["--resume", resume_id]
        cmd += ["-p", prompt]
        return cmd

    budget_s = _zcode_timeout_s()

    async def _attempt(
        stream: bool,
    ) -> tuple[
        int | None,
        str,
        str,
        bool,
        bool,
        dict[str, Any],
    ]:
        """Spawn one CLI run; stream-map events when ``stream``.

        Returns ``(returncode, stdout_text, stderr_text, timed_out, idle_out,
        state)``. ``state`` carries the accumulated turn (``answer_parts``,
        ``session_id``, terminal ``result`` envelope, ``turn_usage``).
        """
        state: dict[str, Any] = {
            "answer_parts": [],
            "session_id": "",
            "session_id_emitted": False,
            "result": None,
            "turn_usage": None,
            "reasoning_chars": 0,
            "model_id": "",
        }
        cmd = _build_argv(stream)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                # No prompt on stdin; DEVNULL keeps the CLI from waiting on a tty.
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=16 * 1024 * 1024,
                start_new_session=True,
                cwd=spawn_cwd,
                env=_build_delegate_spawn_env(_zcode_spawn_args(args)),
            )
        except FileNotFoundError:
            _log_error(
                "spawn_failed",
                prompt=prompt,
                cwd=spawn_cwd,
                mode=zcode_mode,
                resume_id=resume_id,
                message="`zcode` CLI not on PATH",
            )
            return None, "", "", False, False, state

        lease_id: str | None = None
        timed_out = False
        idle_out = False
        stdout_parts: list[str] = []
        stderr_task: asyncio.Future | None = None
        try:
            # Lease the spawned *_cli agent's process group (§10.4): the
            # broker-aware tree kill then capability-validates its cancel;
            # released in this attempt's ``finally``.
            lease_id = sandbox_broker.lease_process(pgid=proc.pid, kind="cli")
            if on_spawn is not None:
                try:
                    on_spawn(proc)
                except Exception:  # noqa: BLE001
                    pass
            deadline = time.monotonic() + budget_s
            if stream:
                # Only the streaming loop needs a dedicated stderr drain;
                # legacy mode uses ``communicate()`` which reads BOTH pipes
                # itself (two readers on one pipe would race).
                stderr_task = asyncio.ensure_future(proc.stderr.read())
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        timed_out = True
                        break
                    try:
                        line = await asyncio.wait_for(
                            proc.stdout.readline(),
                            timeout=min(_ZCODE_STREAM_IDLE_S, remaining),
                        )
                    except asyncio.TimeoutError:
                        if deadline - time.monotonic() <= 0:
                            timed_out = True
                        else:
                            idle_out = True
                        break
                    if not line:
                        break  # EOF — the CLI printed its final ``result``.
                    text = line.decode("utf-8", "replace").strip()
                    if not text:
                        continue
                    stdout_parts.append(text)
                    try:
                        event = json.loads(text)
                    except ValueError:
                        continue  # plain-text notice line (e.g. not-due skip)
                    if isinstance(event, dict):
                        if state["session_id_emitted"] is False and event.get("sessionId"):
                            state["session_id_emitted"] = True
                        _zcode_emit_stream_event(
                            event,
                            on_delta=on_delta,
                            on_event=on_event,
                            state=state,
                        )
            else:
                try:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(
                        proc.communicate(),
                        timeout=budget_s,
                    )
                except asyncio.TimeoutError:
                    timed_out = True
                else:
                    return (
                        proc.returncode,
                        stdout_bytes.decode("utf-8", "replace"),
                        stderr_bytes.decode("utf-8", "replace"),
                        False,
                        False,
                        state,
                    )
        except asyncio.CancelledError:
            _log_error(
                "cancelled",
                prompt=prompt,
                cwd=spawn_cwd,
                mode=zcode_mode,
                resume_id=resume_id,
                returncode=proc.returncode,
            )
            # Whole-tree hard kill: a cancelled delegate must not leave an
            # autonomous ``zcode`` burning the coding-plan quota.
            _kill_tree(proc, signal.SIGKILL)
            raise
        finally:
            sandbox_broker.release_process(locals().get("lease_id"))
            if timed_out or idle_out:
                reason = "timeout" if timed_out else "idle_no_output"
                _kill_tree(proc, signal.SIGKILL)
                _log_error(
                    reason,
                    prompt=prompt,
                    cwd=spawn_cwd,
                    mode=zcode_mode,
                    resume_id=resume_id,
                    budget_s=budget_s,
                    stream=stream,
                )
            try:
                await asyncio.wait_for(proc.wait(), timeout=10.0)
            except Exception:  # noqa: BLE001
                pass

        stderr_text = ""
        if stderr_task is not None:
            try:
                stderr_text = (await asyncio.wait_for(stderr_task, timeout=5.0)).decode("utf-8", "replace")
            except Exception:  # noqa: BLE001
                stderr_text = ""
        return (
            proc.returncode,
            "\n".join(stdout_parts),
            stderr_text[:_ZCODE_STDERR_KEEP],
            timed_out,
            idle_out,
            state,
        )

    try:
        # Streaming attempt first; an older CLI that rejects ``--output-format``
        # (``Unknown option '--output-format'``, exit 1) retries once in legacy
        # ``--json`` mode with the original one-shot semantics.
        (
            returncode,
            stdout_text,
            stderr_text,
            timed_out,
            idle_out,
            state,
        ) = await _attempt(stream=True)
        if (
            state.get("result") is None
            and returncode not in (0, None)
            and "--output-format" in stderr_text
            and "unknown option" in stderr_text.lower()
        ):
            (
                returncode,
                stdout_text,
                stderr_text,
                timed_out,
                idle_out,
                state,
            ) = await _attempt(stream=False)
            streamed = False
        else:
            streamed = True

    finally:
        if model_cfg_dir is not None:
            shutil.rmtree(model_cfg_dir, ignore_errors=True)

    if timed_out or idle_out:
        if idle_out:
            return _err(f"zcode produced no output for {_ZCODE_STREAM_IDLE_S:.0f}s and was stopped (mode={zcode_mode})")
        return _err(f"zcode exceeded the {budget_s:.0f}s delegate budget and was stopped (mode={zcode_mode})")

    if returncode is None:
        _log_error(
            "spawn_failed",
            prompt=prompt,
            cwd=spawn_cwd,
            mode=zcode_mode,
            resume_id=resume_id,
            message="`zcode` CLI not on PATH",
        )
        return _err("`zcode` CLI not on PATH. Install: npm install -g zcode-app-cli  (then: zcode login)")

    payload = state.get("result") if isinstance(state.get("result"), dict) else None
    if payload is None:
        payload = _zcode_extract_result(stdout_text)
    if payload is None:
        if returncode != 0:
            err_msg = stderr_text[:400].strip() or _signal_name(returncode)
            _log_error(
                "nonzero_exit",
                prompt=prompt,
                cwd=spawn_cwd,
                mode=zcode_mode,
                resume_id=resume_id,
                returncode=returncode,
                stderr=stderr_text,
                stdout=stdout_text[:2000],
            )
            return _err(f"zcode exited {returncode}: {err_msg}")
        _log_error(
            "unparsable_output",
            prompt=prompt,
            cwd=spawn_cwd,
            mode=zcode_mode,
            resume_id=resume_id,
            returncode=returncode,
            stdout=stdout_text[:2000],
            stderr=stderr_text[:2000],
        )
        return _err("zcode produced no parsable --json payload")

    response = payload.get("response")
    envelope_answer = response.strip() if isinstance(response, str) else ""
    streamed_answer = "".join(state.get("answer_parts") or []).strip()
    answer = envelope_answer or streamed_answer
    if not answer:
        _log_error(
            "empty_response",
            prompt=prompt,
            cwd=spawn_cwd,
            mode=zcode_mode,
            resume_id=resume_id,
            returncode=returncode,
            payload_keys=sorted(payload.keys()),
            stderr=stderr_text[:1000],
        )
        return _err("zcode returned empty response")

    final_session_id = payload.get("sessionId")
    if not isinstance(final_session_id, str) or not final_session_id.strip():
        final_session_id = state.get("session_id") or resume_id
    final_session_id = final_session_id.strip() if isinstance(final_session_id, str) else ""

    usage = _zcode_usage_to_snake(payload.get("usage"))
    if not usage and isinstance(state.get("turn_usage"), dict):
        usage = _zcode_usage_to_snake(state["turn_usage"])
    # Session id FIRST: ExternalCliEngine persists it as
    # ``state.engine_session_id`` and locks the spawn cwd from this event, so
    # an ESC / idle abort right after the answer still leaves ``--resume``
    # primed for the next turn. In streaming mode the id already rode the
    # first event line; emitting it again is harmless (same id) but skipped.
    if final_session_id:
        _write_zcode_session_id(final_session_id, sa_sid, sa_sdir)
        if on_event is not None and not state.get("session_id_emitted"):
            try:
                on_event({"kind": "session_id", "id": final_session_id})
            except Exception:  # noqa: BLE001
                pass
    if on_event is not None:
        # Streaming mode reported usage per ``turn.completed`` already; the
        # terminal envelope carries the same numbers, so emit it once.
        if usage and not streamed:
            try:
                on_event({"kind": "usage", "usage": usage})
            except Exception:  # noqa: BLE001
                pass
        # Streaming mode reported the model from ``model.request.started``
        # (authoritative even on a resumed session); the legacy path has only
        # the CLI's configured default to offer.
        if not streamed:
            if configured_model and not resume_id:
                try:
                    on_event({"kind": "model", "id": configured_model})
                except Exception:  # noqa: BLE001
                    pass
            reasoning_tokens = payload.get("usage")
            if isinstance(reasoning_tokens, dict):
                try:
                    chars = int(reasoning_tokens.get("reasoningTokens") or 0)
                except (TypeError, ValueError):
                    chars = 0
                if chars > 0:
                    try:
                        on_event({"kind": "reasoning", "chars": chars})
                    except Exception:  # noqa: BLE001
                        pass
        elif state.get("reasoning_chars"):
            try:
                on_event(
                    {
                        "kind": "reasoning",
                        "chars": int(state["reasoning_chars"]),
                    }
                )
            except Exception:  # noqa: BLE001
                pass

    # The streamed deltas were already delivered; emit a whole-answer delta
    # only when nothing streamed (legacy mode, or a stream that produced no
    # text deltas but a result envelope with text).
    if answer and not state.get("answer_parts") and on_delta is not None:
        try:
            on_delta(answer)
        except Exception:  # noqa: BLE001
            pass

    header_bits = []
    if final_session_id:
        # 12 chars, matching the claude/antigravity/codex header convention:
        # ExternalCliEngine._extract_session_id_from_result treats a captured
        # ``session=<id>`` of <=12 chars as a TRUNCATED display stub and
        # recovers the full id from this engine's sidecar. A longer slice
        # would be mistaken for the real id and corrupt ``--resume``.
        header_bits.append(f"session={final_session_id[:12]}")
    header_bits.append(f"mode={zcode_mode}")
    if resume_id:
        header_bits.append("resumed")
    if usage:
        header_bits.append(f"tokens={usage.get('input_tokens', 0)}in/{usage.get('output_tokens', 0)}out")
    header = f"[zcode {' · '.join(header_bits)}]"

    body = f"{header}\n\n{answer}"
    body, truncated = _truncate(body)
    return ToolResult(content=body, ok=True, truncated=truncated)


__all__ = [
    "run_delegate_zcode_async",
    "_ZCODE_MODE_MAP",
    "_ZCODE_STREAM_IDLE_S",
    "_zcode_emit_stream_event",
    "_zcode_extract_result",
    "_zcode_home",
    "_read_zcode_session_id",
    "_write_zcode_session_id",
    "_zcode_spawn_args",
    "_zcode_state_path",
    "_zcode_usage_to_snake",
    "_zcode_configured_model",
]
