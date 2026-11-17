"""``ExternalCliEngine`` — shared base for CLI-backed engines.

Hosts the shared ``run_turn`` driver that adapts every CLI engine's
:meth:`delegate_runner` (a wrapper around the legacy
``run_delegate_*_async`` helper) into the uniform
:data:`AgentTurnEvent` stream. Subclasses
(``ClaudeCliEngine`` / ``CodexCliEngine`` / ``AntigravityCliEngine``)
configure the per-CLI argv assembly, permission-mode mapping, and
event-parsing details. The CLI subprocess plumbing duplicated across
the three legacy runners lives there:

* ``run_delegate_claude_async`` (runners.py)
* ``run_delegate_codex_async``  (runners.py)
* ``run_delegate_antigravity_async`` (runners.py)

Subclasses (``ClaudeCliEngine`` / ``CodexCliEngine`` / ``AntigravityCliEngine``)
override the parts that actually differ: CLI argv assembly, permission-
mode mapping, stream-JSON event parser, and the engine-specific tool-name
dictionary.

Every helper here corresponds to a named function in today's
``runners.py``; the docstring cites the source so the implementer can
re-read the authoritative version while filling in the body.
"""

from __future__ import annotations

import asyncio
import os
import re
import signal
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

# Grace window for a friendly CLI-bridge abort: on ``/abort`` (first
# press) we send SIGINT and let the delegated ``claude -p`` / ``codex`` /
# ``antigravity`` subprocess wind down — flushing its reply and its own
# ``--resume`` session file — for up to this many seconds before
# escalating to SIGTERM→SIGKILL. Tunable via env for slow models.
CLI_GRACEFUL_ABORT_S: float = float(os.environ.get("SHAREDAGENTS_CLI_GRACEFUL_ABORT_S", "15") or "15")

# Map each CLI engine kind to the pricing-table provider so a delegate turn's
# tokens resolve to that provider's rate (usage_pricing matches by provider when
# no model/alias entry hits). Keys are EngineKind.value strings.
_PROVIDER_FOR_KIND: dict[str, str] = {
    "claude_cli": "anthropic",
    "codex_cli": "openai",
    "antigravity_cli": "google",
    "zcode_cli": "zai",
}


def _foreground_instruction() -> str:
    """Operator requirement appended to every external-CLI prompt.

    The external CLI agents (codex / claude / antigravity) can hand their
    work off to their OWN background subprocess / thread / cloud agent and
    return before the final result exists, so the CLI exits and sharedagents
    never receives the real result. The prompt itself asks the agent to
    run at least one task in the FOREGROUND of its CLI session.
    """
    return (
        "\n\nIMPORTANT (operator requirement): run at least one task in the "
        "FOREGROUND of this CLI session. Do NOT hand your work off to your "
        "own background subprocess, thread, cloud/background agent, or "
        "scheduled feature and return before it completes — the caller waits "
        "for your complete final result in this session before the CLI exits. "
        "If you delegate work, run the delegation inside this session and "
        "return the final result before exiting."
    )


def _outbound_prompt_text(request: TurnRequest, composed: str) -> str:
    """The text handed to the CLI: ``composed`` plus the foreground note.

    An engine command (``/compact``) goes out bare: anything added around
    it becomes the command's arguments or turns it into plain chat.
    """
    if request.extra.get("_engine_command"):
        return request.prompt
    return f"{composed}{_foreground_instruction()}"


def _apply_delegate_usage(
    metrics: dict,
    usage: dict | None,
    context_usage: dict | None = None,
) -> dict:
    """Overwrite char-estimated token fields with REAL CLI-reported usage.

    ``usage`` follows the Anthropic/claude result shape
    (``input_tokens`` / ``output_tokens`` / ``cache_read_input_tokens`` /
    ``cache_creation_input_tokens``); codex/antigravity runners normalize to the
    same keys before calling this. No-op (returns ``metrics`` unchanged, so the
    char estimate stands) when usage is missing/empty or carries no usable
    tokens. Sets ``tokens_estimated=False`` ONLY when real tokens are applied —
    callers rely on that flag to decide whether to fall back to a cost override.

    ``usage`` is the turn's CUMULATIVE token counts (summed over every internal
    API round-trip) — the right basis for BILLING, but wrong for the ctx-meter,
    whose numerator must be the context window's actual end-of-turn occupancy.
    When ``context_usage`` (the last single API call's usage) is supplied, the
    displayed ``prompt_tokens`` family is taken from it instead, so a long
    agentic claude turn shows e.g. 231K/256K rather than a cumulative 7.0M/256K.
    Billing still reads ``prompt_tokens_billed*`` (cumulative) — see below.
    """
    if not isinstance(metrics, dict) or not isinstance(usage, dict) or not usage:
        return metrics
    inp = int(usage.get("input_tokens") or 0)
    cached = int(usage.get("cache_read_input_tokens") or 0)
    cache_w = int(usage.get("cache_creation_input_tokens") or 0)
    out = int(usage.get("output_tokens") or 0)
    if inp <= 0 and out <= 0:
        return metrics
    # Cumulative totals — preserved separately so the pricing layer bills the
    # full turn (every cache read/write across all internal steps) even when the
    # DISPLAY prompt_tokens are narrowed to the last call's occupancy below.
    cum_prompt = inp + cached + cache_w
    metrics["prompt_tokens_billed"] = cum_prompt
    metrics["prompt_tokens_billed_cached"] = cached
    metrics["prompt_tokens_billed_cache_write"] = cache_w
    metrics["prompt_tokens"] = cum_prompt
    metrics["prompt_tokens_cached"] = cached
    metrics["prompt_tokens_cache_write"] = cache_w
    metrics["completion_tokens"] = out
    # One extractor for every provider's reasoning-token shape (see
    # ``delegates.base``); imported lazily because the delegates package
    # reaches back into ``tools.runners`` at import time.
    from sharedagents_agent.delegates.base import _reasoning_tokens_from_usage

    think = _reasoning_tokens_from_usage(usage)
    metrics["reasoning_tokens"] = think
    metrics["content_tokens"] = max(0, out - think) if think > 0 else out
    metrics["tokens_estimated"] = False
    # Narrow the DISPLAYED context size to the last API call's real occupancy.
    if isinstance(context_usage, dict) and context_usage:
        c_inp = int(context_usage.get("input_tokens") or 0)
        c_cached = int(context_usage.get("cache_read_input_tokens") or 0)
        c_cache_w = int(context_usage.get("cache_creation_input_tokens") or 0)
        c_prompt = c_inp + c_cached + c_cache_w
        if c_prompt > 0:
            metrics["prompt_tokens"] = c_prompt
            metrics["prompt_tokens_cached"] = c_cached
            metrics["prompt_tokens_cache_write"] = c_cache_w
    metrics["total_tokens"] = int(metrics.get("prompt_tokens") or 0) + out
    return metrics


def _apply_codex_effective_throughput(metrics: dict, turn_s: float) -> dict:
    """Label Codex's only measurable throughput honestly.

    ``codex exec --json`` emits assistant messages as consolidated items, so
    there is no progressive decode window. Its meaningful rate is authoritative
    output tokens divided by whole-turn wall time (including tool waits), which
    is effective throughput rather than model decode throughput.
    """
    tokens = metrics.get("completion_tokens") if isinstance(metrics, dict) else None
    if isinstance(tokens, (int, float)) and tokens > 0 and isinstance(turn_s, (int, float)) and turn_s >= 0.25:
        metrics["tps"] = round(float(tokens) / float(turn_s), 1)
        metrics["tps_kind"] = "effective"
    return metrics


_CLI_TOOL_RESULT_MARKER_RE = re.compile(
    r"^\s*[↳✗](?:\s+.*)?\(\d+\s+chars\)\s*$",
)


def _accumulate_delegate_usage(
    previous: dict | None,
    current: dict | None,
) -> dict | None:
    """Add one CLI subprocess's usage to the turn total.

    ``ExternalCliEngine`` may run several CLI subprocesses for one logical
    turn when an idle-empty continuation is nudged.  Each legacy runner
    reports usage for its own subprocess, so replacing ``previous`` would
    under-report billing after the first round.  Context usage is deliberately
    *not* handled here: the engine keeps the last single-call context snapshot
    separately for the context-window meter.
    """
    if not isinstance(current, dict) or not current:
        return previous

    fields = (
        "input_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "thinking_tokens",
    )
    merged = dict(previous) if isinstance(previous, dict) else {}
    changed = False
    for field in fields:
        if field not in current or current[field] is None:
            continue
        try:
            value = int(current[field])
        except (TypeError, ValueError):
            continue
        if value < 0:
            continue
        try:
            prior = int(merged.get(field) or 0)
        except (TypeError, ValueError):
            prior = 0
        merged[field] = prior + value
        changed = True

    return merged if changed else previous


def _cli_visible_result_chars(result_text: str) -> int:
    """Return visible answer chars in a CLI result, ignoring scaffolding.

    Legacy runners prefix ``[claude session=… · mode=…]`` (or Codex
    ``thread=``), and persist inline tool markers such as ``🔧 Bash …`` and
    ``↳ output (N chars)`` in the returned body.  Neither is user answer
    content: an idle-empty stop can carry both while the operator saw
    nothing.  The stream callback is the authoritative path in production;
    this narrow parser keeps classification correct for legacy callers and
    test doubles that only return the persisted body.
    """
    body = result_text or ""
    lines = body.splitlines()
    if (
        lines
        and lines[0].startswith("[")
        and (
            "session=" in lines[0]
            or "thread=" in lines[0]
            or (lines[0].startswith(("[claude ", "[codex ", "[antigravity ")) and "mode=" in lines[0])
        )
    ):
        lines = lines[1:]
    visible = [
        line for line in lines if not line.lstrip().startswith("🔧 ") and not _CLI_TOOL_RESULT_MARKER_RE.match(line)
    ]
    return len("\n".join(visible).strip())


def _cli_tool_result_is_abort(result_text: str) -> bool:
    """Best-effort abort sniff on legacy delegate error bodies."""
    low = (result_text or "").lower()
    if not low:
        return False
    for needle in (
        "aborted_streaming",
        "aborted_tools",
        "user_abort",
        "terminal_reason: aborted",
        "cancelled",
        "canceled",
        "interrupted",
    ):
        if needle in low:
            return True
    # Do NOT fall back to a fuzzy both-words
    # ("aborted" + "error") text sniff. A genuine CLI error that merely
    # mentions a prior aborted operation contains both words and would be
    # swallowed as an abort, hiding the TurnError the user must see. Real
    # aborts are recognized by the structured needles above plus the
    # caller's structured signals (``_operator_stop_turns`` and
    # ``_claude_error_is_abort`` on the result metadata). Absent a
    # structured abort marker, an errored result is a genuine error.
    return False


from sharedagents_model.usage_pricing import metrics_from_provider_usage  # noqa: E402

from .engine import (  # noqa: E402
    AgentTurnEvent,
    ArtifactReference,
    Engine,
    EngineKind,
    EngineSessionState,
    HandoffAccepted,
    TextDelta,
    ToolCallResult,
    ToolCallStarted,
    TurnDone,
    TurnError,
    TurnRequest,
    TurnStarted,
)

CLAUDE_AUTO_MEMORY_OFF_ENV = "CLAUDE_CODE_DISABLE_AUTO_MEMORY"


def spawn_env_overrides(kind: Any, extra: Any) -> dict[str, str]:
    """Env overrides for one CLI spawn: the agent's ``api_keys`` plus, for
    claude_cli with ``cli_auto_memory`` False, the auto-memory off switch.

    The switch is set even when ``api_keys`` is empty and wins over a
    same-named ``secrets.env`` entry: the config field is the authority.
    """
    extra = extra if isinstance(extra, dict) else {}
    overrides: dict[str, str] = {}
    api_keys = extra.get("api_keys")
    if isinstance(api_keys, dict):
        overrides.update(api_keys)
    if kind is EngineKind.CLAUDE_CLI and extra.get("cli_auto_memory") is False:
        overrides[CLAUDE_AUTO_MEMORY_OFF_ENV] = "1"
    return overrides


_QUOTA_EXHAUSTED_MARKERS = (
    "usage limit reached",
    "providerbusinesserror",
    "[1308]",
    "insufficient_quota",
    "quota exceeded",
    # Antigravity's account wall. It is not one of the tracked percent pools,
    # and "quota reached" is not "quota exceeded".
    "individual quota reached",
)


def _cli_error_is_quota_exhausted(text: str) -> bool:
    """True when a CLI error body is a plan/usage-limit wall (not transient)."""
    low = (text or "").lower()
    return any(m in low for m in _QUOTA_EXHAUSTED_MARKERS)


def _remember_cli_quota(engine_kind: Any, text: str) -> None:
    """Keep a parsed CLI reset so routing can refuse the engine until it passes.

    Best-effort: a disk or import failure must not change the turn's error.
    """
    try:
        from sharedagents_agent.external_registry import short_name_for
        from sharedagents_web.subscription_limits import note_cli_quota_exhaustion

        note_cli_quota_exhaustion(short_name_for(engine_kind), text)
    except Exception:  # noqa: BLE001 — the turn error is already decided
        return


def _match_model_row(rows: list[dict[str, Any]], value: str) -> str:
    """The wire id of the row ``value`` names (alias, then id, then display,
    case-insensitive), or ``""``."""
    vlow = value.lower()
    for key in ("alias", "id", "display"):
        for row in rows:
            if str(row.get(key) or "").lower() == vlow and row.get("id"):
                return str(row["id"])
    return ""


class ExternalCliEngine(Engine):
    """Shared scaffolding for CLI-backed engines.

    Each subclass is responsible for telling this base class:

    * **what CLI to spawn** — via :meth:`build_argv`
    * **how to pipe in the prompt** — default is stdin (mirrors all three
      delegates today; see runners.py L1565–1576 on the ARG_MAX
      rationale). Subclasses can override if a CLI behaves differently.
    * **what permission mode flags to pass** — via :meth:`permission_flags`
    * **how to parse stream-JSON lines** — via :meth:`parse_stream_event`
    * **how to extract the engine's own session id** from those events —
      via :meth:`extract_session_id`
    * **how to format the in-band tool markers** that get appended to
      ``text_buf`` so the persisted body is identical to today — via
      :meth:`format_tool_use_marker` / :meth:`format_tool_result_marker`

    The base class owns:

    * Process spawn + ``start_new_session=True`` (runners.py L1589, L2100,
      L1294).
    * Stdin feed via ``_feed_stdin`` (runners.py L42–63).
    * Stdout read-loop with the 16 MiB readline-bypass and JSONDecode
      tolerance (runners.py L1696–1714).
    * Cancellation → SIGTERM-group → SIGKILL-group fallthrough
      (runners.py L1803–1813).
    * Exit-code + error-event triage → tool-result error formatting
      (runners.py L1827–1899, L2225–2289, L1430–1489).
    * Sidecar I/O with legacy-key tolerance (see ``EngineSessionState``).
    * Result body assembly (header / body / footer) and truncation
      (runners.py L1904–1926, L2269–2289, L1473–1489).
    * Structured logging via ``_log_delegate_error`` (runners.py L970).
    """

    # Subclasses MUST set ``kind``; ``cli_command`` / ``install_url`` come from
    # :data:`EXTERNAL_REGISTRY` at init time.
    kind: EngineKind
    cli_command: str  # populated from registry in ``__init__``
    install_url: str = ""  # populated from registry in ``__init__``

    def __init__(self) -> None:
        from sharedagents_agent.external_registry import spec_for

        spec = spec_for(self.kind)
        self.cli_command = spec.cli_command
        self.install_url = spec.install_url or ""
        # Per-engine in-memory state. Each ``run_turn`` registers
        # its driver-task in ``_active_procs[turn_id]`` so ESC can
        # route through :meth:`abort`. The legacy ``run_delegate_*``
        # runners handle their own subprocess group-kill on
        # ``CancelledError``, so cancelling the driver task is
        # sufficient — the legacy runner's ``finally`` block does
        # the SIGTERM/SIGKILL on the spawned CLI.
        self._active_procs: dict[str, Any] = {}
        # Per-turn handle to the live CLI subprocess (populated by the
        # runner via its ``on_spawn`` callback) so :meth:`abort` can send
        # a friendly SIGINT to the process group and wait for a clean
        # wind-down before escalating to a hard kill.
        self._active_cli_procs: dict[str, Any] = {}
        # Reverse index ``sharedagents_session_id -> {turn_id, …}`` so a cancel can
        # abort EVERY live turn for a session even when the caller has lost the
        # turn_id (e.g. the bridge's active-turn map was already popped). Paired
        # with the durable ``engine_pidfile`` sidecar for the across-restart
        # orphan case. Populated/cleared alongside ``_active_cli_procs``.
        self._session_turns: dict[str, set[str]] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}
        # Turn ids for which the operator requested a stop (graceful OR hard).
        # A graceful SIGINT can make Codex/Claude/Antigravity return either a
        # nonzero error result or a clean empty result. Both are the requested
        # terminal outcome: treating the former as retryable ``cli_error``
        # restarts the stopped process, while treating the latter as an idle
        # stop starts an automatic continuation. Keep the intent until the
        # outer generator emits its one terminal TurnDone.
        self._operator_stop_turns: set[str] = set()

    def _session_model(self, sharedagents_sid: str) -> str:
        """Return the per-session model override an engine subclass has
        stored via :meth:`set_config_option`, or ``""`` when none.

        :meth:`run_turn` consults this before falling back to the
        engine's built-in default (no ``--model`` flag). Subclasses
        that implement model selection expose a ``session_model(sid)``
        accessor (see :class:`AntigravityCliEngine`); subclasses that don't
        return ``""``.
        """
        getter = getattr(self, "session_model", None)
        if callable(getter):
            try:
                return str(getter(sharedagents_sid) or "")
            except Exception:  # noqa: BLE001
                return ""

    def context_window(self, model: str | None = None) -> int:
        """Best-known context window for this CLI engine.

        When ``model`` (wire id / alias / display) matches a discovery row
        that carries a per-model ``contextWindow`` (codex: the ChatGPT
        catalogue's ``max_context_window``; claude: the Models API's
        ``max_input_tokens``), that wins — the spec default is the fallback
        for engines/models without a published window. The Claude CLI can
        report the id with its context tag (``claude-opus-5-5[1m]``), so the
        bare id is tried after the exact name.
        """
        m = (model or "").strip()
        if m:
            try:
                names = [m.lower()]
                base = m.split("[", 1)[0].strip().lower() if m.endswith("]") else ""
                if base and base not in names:
                    names.append(base)
                rows = self.available_models() or []
                for mlow in names:
                    for row in rows:
                        if mlow in {
                            str(row.get("id") or "").lower(),
                            str(row.get("alias") or "").lower(),
                            str(row.get("display") or "").lower(),
                        }:
                            win = int(row.get("maxContextWindow") or row.get("contextWindow") or 0)
                            if win > 0:
                                return win
            except Exception:  # noqa: BLE001
                pass
        from sharedagents_agent.external_registry import spec_for

        try:
            spec = spec_for(self.kind)
            if spec and spec.default_context_window:
                return spec.default_context_window
        except Exception:
            pass
        return 0

    # ---- per-session model selection (shared by all CLI engines) -----
    #
    # The ``/models`` picker resolves a row by alias and calls
    # ``set_config_option(sid, "model", <alias>)`` over the wire. We
    # resolve that alias against :meth:`available_models` to the WIRE
    # model id and stash it per session; :meth:`run_turn` reads it back
    # via :meth:`_session_model` → ``legacy_args["model"]`` → the CLI's
    # ``--model`` flag. Before this, ``set_config_option`` was the base
    # no-op and no CLI engine stored a selection, so the picker only
    # updated the footer cosmetically while the subprocess kept its
    # default model.

    def _session_models_map(self) -> dict[str, str]:
        """Lazy per-engine ``{sid: wire_model_id}`` store.

        Lazily initialised (rather than in ``__init__``) so this mix-in is
        independent of subclass constructors.
        """
        m = getattr(self, "_session_models", None)
        if m is None:
            m = {}
            self._session_models = m  # type: ignore[attr-defined]
        return m

    def session_model(self, sharedagents_sid: str) -> str:
        """The wire model id selected for ``sharedagents_sid``, or ``""`` (CLI
        default). Consulted by :meth:`_session_model`."""
        return self._session_models_map().get(sharedagents_sid, "")

    def _resolve_model_value(self, value: str) -> str:
        """Resolve a picker value (alias / wire id / display) to the wire
        model id, using :meth:`available_models`.

        When the live list has no match (empty after a failed fetch, or
        missing the row), the provider's static catalogue and the engine
        profile's own ``upstream_alias`` → ``upstream_model`` are tried
        before giving up: the claude CLI rejects a picker alias such as
        ``opus-5-5`` as an unknown model.

        Unknown values pass through verbatim — the CLI is the authority on
        valid model ids, and the whole point of discovery is that a NEW model
        works without anyone updating a constant first. Resolution is
        best-effort: a discovery failure never blocks selection.
        """
        v = (value or "").strip()
        if not v:
            return ""
        try:
            rows = self.available_models() or []
        except Exception:  # noqa: BLE001
            rows = []
        hit = _match_model_row(rows, v)
        if hit:
            return hit
        try:
            from sharedagents_agent.discovery._common import (
                _cli_profile_model_row,
                _fallback_models,
            )
            from sharedagents_agent.external_registry import spec_for

            provider = str(spec_for(self.kind).provider or "")
            offline = _fallback_models(provider) if provider else []
            profile_row = _cli_profile_model_row(self.kind.value)
            if profile_row:
                offline.append(profile_row)
            hit = _match_model_row(offline, v)
        except Exception:  # noqa: BLE001
            hit = ""
        return hit or v  # pass-through

    def set_config_option(self, sharedagents_sid: str, option_id: str, value: Any) -> None:
        """Per-session live config toggle. CLI engines act only on
        ``"model"`` (everything else is a no-op — permission mode rides on the
        next ``--resume`` call's flag).

        ``value`` is the picker's chosen alias / id; ``"auto"`` / ``"default"``
        / ``"reset"`` / empty clears the override so the CLI uses its own
        default model.
        """
        if option_id != "model":
            return
        raw = str(value or "").strip()
        store = self._session_models_map()
        if not raw or raw.lower() in ("auto", "default", "reset", "none"):
            store.pop(sharedagents_sid, None)
            return
        store[sharedagents_sid] = self._resolve_model_value(raw)

    def _materialize_image_parts(
        self,
        prompt_parts: list[dict[str, Any]],
    ) -> tuple[list[str], list[Path]]:
        """Resolve every ``image_url`` part to a local filesystem path.

        Returns ``(paths, temp_files)``:

        * ``paths`` — every resolved local path, in prompt order, for
          :meth:`run_turn` to inline into the prompt text. The CLI
          subprocess can't see an inline base64 blob; the supported
          CLIs (claude / codex / antigravity) all read images off disk via
          their own file-reading tool, so we hand them a path instead.
        * ``temp_files`` — the subset of ``paths`` this call created
          (``data:`` decode / ``http(s)`` fetch). The caller MUST
          unlink these after the CLI subprocess exits. ``file://`` and
          bare local paths are inlined but never returned here — we
          didn't create them, so there's nothing to clean up.

        Mirrors the URL handling in
        :meth:`ClaudeCliEngine.prime_first_turn_prompt` — kept here as
        the live ``run_turn`` path's own copy because that method is
        only consumed by the not-yet-wired native-priming path.
        """
        import base64
        import tempfile
        import uuid

        paths: list[str] = []
        temp_files: list[Path] = []
        for part in prompt_parts or []:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image_url = part.get("image_url")
            url: str | None = None
            if isinstance(image_url, dict):
                u = image_url.get("url")
                if isinstance(u, str):
                    url = u
            elif isinstance(image_url, str):
                url = image_url
            if not url:
                continue

            if url.startswith("data:"):
                # data:image/<ext>;base64,<payload>
                try:
                    header, _, payload = url.partition(",")
                    meta = header[len("data:") :]
                    mime = meta.split(";")[0] if meta else "image/png"
                    ext = mime.split("/")[-1] if "/" in mime else "png"
                    raw_bytes = base64.b64decode(payload)
                    tmp = tempfile.NamedTemporaryFile(
                        prefix=f"sharedagents-attach-{uuid.uuid4().hex[:8]}-",
                        suffix=f".{ext}",
                        delete=False,
                    )
                    try:
                        tmp.write(raw_bytes)
                    finally:
                        tmp.close()
                    paths.append(tmp.name)
                    temp_files.append(Path(tmp.name))
                except (ValueError, base64.binascii.Error, OSError):
                    continue
            elif url.startswith("http://") or url.startswith("https://"):
                try:
                    import httpx  # local import — only on http(s) urls

                    resp = httpx.get(url, timeout=5.0, follow_redirects=True)
                    resp.raise_for_status()
                    ctype = resp.headers.get("content-type", "image/png")
                    ext = ctype.split(";")[0].split("/")[-1] or "png"
                    tmp = tempfile.NamedTemporaryFile(
                        prefix=f"sharedagents-attach-{uuid.uuid4().hex[:8]}-",
                        suffix=f".{ext}",
                        delete=False,
                    )
                    try:
                        tmp.write(resp.content)
                    finally:
                        tmp.close()
                    paths.append(tmp.name)
                    temp_files.append(Path(tmp.name))
                except Exception:  # noqa: BLE001
                    continue
            elif url.startswith("file://"):
                paths.append(url[len("file://") :])
            else:
                # Bare local path — inline as-is, nothing to clean up.
                paths.append(url)
        return paths, temp_files

    # ---- Engine ABC methods (concrete here, parametrized by subclass) -

    @property
    def name(self) -> str:
        from sharedagents_agent.external_registry import spec_for

        return spec_for(self.kind).short_name

    def state_path(self, sessions_dir: Path, sharedagents_sid: str) -> Path:
        """``<sessions_dir>/<sid>.<engine>.json``."""
        return Path(sessions_dir) / f"{sharedagents_sid}.{self.kind.value}.json"

    def read_state(self, sessions_dir: Path, sharedagents_sid: str) -> EngineSessionState:
        """Read sidecar JSON. Tolerates the legacy single-key shapes
        (``claude_session_id``, ``codex_thread_id``,
        ``antigravity_session_id``) so an in-flight chain from before the
        refactor keeps resuming.
        """
        import json

        state = EngineSessionState(
            sharedagents_session_id=sharedagents_sid,
            engine=self.kind,
        )
        path = self.state_path(Path(sessions_dir), sharedagents_sid)
        # Owner turns must not resume a leftover ``delegate_<cli>``
        # native id. That sidecar is a child thread, not this
        # conversation; treating it as primed skips the switch
        # transcript (Codex → Antigravity on the same SID).
        if not path.exists():
            return state
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return state
        if not isinstance(data, dict):
            return state
        # New normalized shape.
        eid = data.get("engine_session_id") or ""
        # Legacy fallbacks (pre-refactor sidecars).
        if not eid:
            eid = (
                data.get("claude_session_id") or data.get("codex_thread_id") or data.get("antigravity_session_id") or ""
            )
        state.engine_session_id = str(eid or "")
        state.priming_status = str(data.get("priming_status") or ("completed" if eid else "never"))
        state.priming_system_prompt_sha256 = str(data.get("priming_system_prompt_sha256") or "")
        state.last_used_at = str(data.get("last_used_at") or "")
        state.last_error = str(data.get("last_error") or "")
        state.spawn_cwd = str(data.get("spawn_cwd") or "")
        state.actual_model = str(data.get("actual_model") or "")
        return state

    def write_state(self, sessions_dir: Path, state: EngineSessionState) -> None:
        """Persist sidecar atomically (best-effort)."""
        import json

        path = self.state_path(Path(sessions_dir), state.sharedagents_session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sharedagents_session_id": state.sharedagents_session_id,
            "engine": state.engine.value,
            "engine_session_id": state.engine_session_id,
            "priming_status": state.priming_status,
            "priming_system_prompt_sha256": state.priming_system_prompt_sha256,
            "last_used_at": state.last_used_at,
            "last_error": state.last_error,
            "spawn_cwd": state.spawn_cwd,
            "actual_model": state.actual_model,
        }
        try:
            path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass

    # ---- registry hooks for the vendor runner and discovery --------
    #
    # Phase 4 strategy: rather than reimplement the full subprocess /
    # sidecar / SIGTERM / stream-parse / error-retry pipeline from
    # scratch, each CLI engine delegates to its corresponding legacy
    # ``sharedagents_agent.tools.runners.run_delegate_*_async`` helper.
    # That helper already implements every behavior the user said
    # "can't lose" — stdin priming, ARG_MAX avoidance, session
    # resume sidecar, group-kill on CancelledError, stale-resume
    # retry (Claude), structured logging, truncation. The Engine
    # wraps it with ``on_delta`` / ``on_event`` callbacks that yield
    # :class:`AgentTurnEvent` instances to the Agent layer.

    def delegate_runner(self) -> Any:
        """Return this engine kind's registered CLI runner."""
        from sharedagents_agent.external_registry import (
            delegate_runner_for_short_name,
            spec_for,
        )

        return delegate_runner_for_short_name(spec_for(self.kind).short_name)

    def supports_on_event(self) -> bool:
        """Whether this engine's registered runner accepts ``on_event``."""
        from sharedagents_agent.external_registry import spec_for

        return spec_for(self.kind).supports_on_event

    def available_models(self) -> list[dict[str, Any]]:
        """Discover picker rows through the external-agent registry."""
        from sharedagents_agent.discovery import discover_models

        return discover_models(self.kind)

    async def run_turn(  # type: ignore[override]
        self,
        request: TurnRequest,
        state: EngineSessionState,
        *,
        sessions_dir: Path,
        turn_id: str,
        env: dict[str, str] | None = None,
    ) -> AsyncIterator[AgentTurnEvent]:
        """Drive one CLI turn end-to-end.

        Phase 4 implementation strategy: delegate to the legacy
        ``run_delegate_*_async`` helper via :meth:`delegate_runner`,
        translating its callbacks to :data:`AgentTurnEvent` yields.
        This guarantees byte-identical CLI-side behavior to the
        legacy delegate-as-tool path; the Engine wrapper adds:

        * The first-call vs resume policy (system prompt rides on
          ``--append-system-prompt`` / first-message injection on the
          first call; later calls use ``--resume <id>``).
        * sha-change detection so an operator who edits SOUL.md
          between turns gets a fresh CLI session.
        * Uniform :data:`AgentTurnEvent` stream for the gateway / ACP
          translator.
        * Per-session state sidecar (replaces ``<sid>.delegate_<kind>.json``).
        """
        import asyncio as _asyncio

        sharedagents_sid = state.sharedagents_session_id
        # Snapshot for the failed-downgrade check in the finally below:
        # we only want to mark priming_status="failed" when a *prior*
        # session id (loaded from disk) failed to resume — never when
        # we just committed a fresh id mid-stream this turn (which is
        # how an aborted first turn now reaches the finally block).
        initial_engine_session_id = state.engine_session_id

        yield TurnStarted(
            turn_id=turn_id,
            sharedagents_session_id=sharedagents_sid,
            engine_session_id=state.engine_session_id,
            resumed=bool(state.engine_session_id),
        )

        # Decide whether this is a first-turn priming call. The Agent
        # has already consulted ``system_prompt_policy`` and either
        # populated ``request.system_prompt`` (first_turn / sha_changed)
        # or left it ``None`` (never — pure resume).
        policy = self.system_prompt_policy(
            state,
            current_system_prompt_sha256=request.system_prompt_sha256,
        )

        # Resume id: ALWAYS honor a recorded engine_session_id when one
        # exists, even when ``policy == "sha_changed"``. Re-priming a
        # CLI session mid-conversation trades every prior user turn,
        # tool result, and decision for an updated system prompt — a
        # bad trade for an opaque CLI conversation we can't replay.
        #
        # In particular, the agent's own tool calls (editing
        # ``config.json``'s ``mcp_servers`` / ``acp_agents``, editing
        # MEMORY.md or any .md file, creating skills) flow into
        # ``_compose_system_prompt`` and shift the sha. Until this fix
        # those self-edits caused the very next turn to spawn a fresh
        # Claude CLI session, dropping the conversation cold. Drift is
        # now logged below but not acted on; operators who want the
        # new prompt to take effect start a new session.
        effective_resume_id = state.engine_session_id or ""

        # Build the prompt payload. We only send
        # ``--append-system-prompt`` on the TRUE first call of a CLI
        # session (no engine_session_id to resume into). On resume the
        # bound CLI session already has its priming snapshot baked in;
        # passing a different prompt alongside ``--resume`` either
        # gets ignored or fights the session's own state.
        user_prompt_text = request.prompt or ""
        if request.handoff is not None and getattr(request.handoff, "rendered", ""):
            user_prompt_text = (
                "<sharedagents-canonical-handoff>\n"
                f"{request.handoff.rendered}\n"
                "</sharedagents-canonical-handoff>\n\n"
                f"{user_prompt_text}"
            )
        manual_budget_exhausted = bool(
            request.extra.get("_budget_exhausted_manual") if isinstance(request.extra, dict) else False
        )
        manual_budget_notice = (
            str(request.extra.get("_manual_budget_notice") or "").strip() if isinstance(request.extra, dict) else ""
        )
        if manual_budget_notice and effective_resume_id:
            # Resumed native CLI sessions cannot replace their baked system
            # prompt. Carry the same framework notice in a clearly delimited
            # user-message prelude for this one manual turn.
            user_prompt_text = (
                "<sharedagents-manual-budget-notice>\n"
                f"{manual_budget_notice}\n"
                "</sharedagents-manual-budget-notice>\n\n"
                f"{user_prompt_text}"
            )
        # A resumed opaque CLI session cannot replace its system channel.
        # In particular, replaying a changed Plan/Auto prompt as user text is
        # both semantically false and correctly looks like prompt injection to
        # Claude Code.  Per-invocation native flags carry mode transitions;
        # the bound system-prompt snapshot stays unchanged until an explicit
        # lane reset.
        effective_prompt_delivered = bool(request.system_prompt and not effective_resume_id)
        # Multimodal: the user attached image(s) (Telegram photo, TUI
        # paste, …). The CLI subprocess can't see an inline base64
        # blob, so materialize each image to a temp file and inline the
        # paths into the prompt text — every supported CLI reads images
        # off disk via its own file-reading tool. Temp files are
        # unlinked in the ``finally`` below once the subprocess exits.
        # Native per-CLI flags (antigravity ``--image`` etc.) are a later
        # optimization; inlining the path works for all three today.
        image_temp_files: list[Path] = []
        if request.has_multimodal_content():
            image_paths, image_temp_files = self._materialize_image_parts(request.prompt_parts)
            if image_paths:
                n = len(image_paths)
                listing = "\n".join(f"  - {p}" for p in image_paths)
                plural = "s" if n != 1 else ""
                them = "them" if n != 1 else "it"
                user_prompt_text = (
                    f"{user_prompt_text}\n\n"
                    f"[The user attached {n} image{plural}. Read {them} "
                    f"from the local filesystem with your file-reading "
                    f"tool:\n{listing}\n]"
                ).strip()
        # Operator requirement: the external CLI agents (codex / claude /
        # antigravity) may hand work off to their OWN background subprocess /
        # thread / cloud agent and return before the result exists, closing
        # the CLI before sharedagents harvests the real result. Ask the agent in
        # the prompt itself to run in the foreground. Appended here — the
        # single shared choke point every external-CLI turn passes through
        # (fresh prime AND resume) — so no external engine can be dispatched
        # without it. Native local_llm agents never pass through this engine,
        # so their prompts stay byte-identical. The appended text rides the
        # composed payload through the secret egress screen below.
        user_prompt_text = _outbound_prompt_text(request, user_prompt_text)
        if request.system_prompt and not effective_resume_id:
            # First-turn priming — fold preamble blocks into the
            # composed system prompt so the CLI session has them in
            # its ``--append-system-prompt`` baseline.
            composed_sys = request.system_prompt
            if request.preamble:
                tail = "\n\n".join(str(b.get("content") or "") for b in request.preamble if b.get("content"))
                if tail:
                    composed_sys = f"{composed_sys}\n\n{tail}"
        else:
            composed_sys = None

        # Model resolution: caller-supplied wins; otherwise fall back to the
        # engine's per-session selection (set via ``set_config_option``).
        # Resolved HERE — ahead of the egress guard — because the model is
        # itself model-controllable (``Agent(model=...)``) AND rides to the CLI
        # as a ``--model`` argv parameter, OUTSIDE the prompt payload. A secret
        # placed in ``model`` therefore bypassed the prompt-only screen below
        # and reached the third-party CLI's argv unscreened (Codex re-review
        # finding). It must pass the SAME egress screen.
        effective_model = request.model or self._session_model(sharedagents_sid)
        if not effective_model and not effective_resume_id:
            # OPTION B, GENERALIZED (owner scope
            # change): no explicit override on a FRESH spawn → the engine's
            # effective default drives the spawn EXPLICITLY as ``--model`` —
            # the CLI's own config default never silently applies. Per
            # engine: codex → the operator-EXPLICIT per-agent upstream_model
            # else codex's config.toml model else ``""`` (no pin; hardcoded
            # fallbacks removed); claude → the operator-EXPLICIT per-agent
            # upstream_model else ANTHROPIC_MODEL env else
            # ~/.claude/settings.json (env BEFORE settings — Claude Code
            # precedence) else ``""`` — the
            # builtin pin is GONE: with no explicit sharedagents
            # model selection the Claude CLI applies its own default and no
            # ``--model`` flag is built; antigravity → ``""`` (no safe
            # sharedagents pin: the CLI's own settings file governs its default,
            # so no ``--model`` flag is built). ``configured`` is the ACTUAL
            # agent's upstream_model (carried via TurnRequest.extra) so the
            # spawn wire id agrees with the session stamp by construction;
            # ``configured_explicit`` marks the operator-EXPLICIT per-agent
            # value — without it the merged global/local-LLM
            # upstream would leak into ``claude --model``.
            # On a RESUMED thread no default is resolved — the CLI resumes
            # its own session model. An explicit empty string override is
            # treated as "use default". Resolved here so the egress guard
            # below screens it like every other model value.
            from sharedagents_agent.discovery._common import (
                external_cli_effective_default,
            )

            effective_model = external_cli_effective_default(
                self.kind.value,
                configured=(request.extra.get("upstream_model") if isinstance(request.extra, dict) else None),
                configured_explicit=bool(
                    request.extra.get("upstream_model_explicit") if isinstance(request.extra, dict) else False
                ),
            )

        # --- Secret egress SINK guard (hard block, no override) ----
        # The pre-flight delegation gate (delegation_gate.evaluate_preflight)
        # screens only the CALL ARGS, upstream in delegate_shims. The payload
        # actually handed to the CLI subprocess is composed HERE — the folded
        # system prompt (system_prompt + preamble: child AGENTS.md / MEMORY /
        # pinned facts / todos / skills / memory-recall) plus the user prompt,
        # PLUS every model-controllable value that leaves as an argv parameter
        # (today: the resolved ``model``, screened HERE; and — screened in a
        # SECOND metadata guard once resolved below — the spawn ``cwd`` and the
        # permission ``mode``. ``cwd`` IS a foreign-secret egress vector: the
        # agentic CLIs (Claude Code / Codex / Antigravity) report their working
        # directory to the model as environment context ("Working directory:
        # <path>"), so a secret embedded in a checkout path (e.g. under
        # ``/tmp/<token>/``) egresses to the vendor unless screened — this is
        # the CLI-side analog of the ACP cwd/mode metadata guard in
        # ``acp_engine``. ``cwd`` is resolved AFTER this point (the resume-cwd-
        # locking block below picks WHICH cwd), so it is screened by a dedicated
        # ``guard_egress_payload`` immediately before the spawn rather than
        # folded into this call. The per-agent ``api_keys`` env override is the
        # CLI's OWN auth credential by design — see ``_spawn_env_overrides``
        # below — so it is NOT foreign-secret egress and is not screened.)
        # Screen the fully-composed string at
        # the true third-party egress so a secret in ANY source — not just the
        # top-level args, and not just the prompt — is caught (closes the
        # "wrong-place scrub" gap AND the argv/model gap). Hard-block per
        # Abort the turn and NEVER spawn the CLI; do not
        # redact-and-send. ``guard_egress_payload`` no-ops for local engines,
        # but every ``external_cli`` engine is third-party by construction, so
        # this always screens here. The base64 data: URI exemption
        # is inherited from the shared scrubber.
        from sharedagents_agent import delegation_gate as _dgate

        try:
            _dgate.guard_egress_payload(
                self.kind,
                "\n\n".join(p for p in (composed_sys, user_prompt_text, effective_model) if p),
            )
        except _dgate.SecretEgressBlocked as _seb:
            # Abort cleanly: unlink any image temp files we materialized above
            # (the finally that normally reaps them is downstream of this early
            # return) and never reach the subprocess spawn.
            for _tmp in image_temp_files:
                try:
                    _tmp.unlink()
                except Exception:  # noqa: BLE001
                    pass
            try:
                from sharedagents_agent.tools._log import log_event as _log_event

                _log_event(
                    "DELEGATE_SECRET_EGRESS_BLOCKED",
                    engine=self.kind.value,
                    sharedagents_session_id=sharedagents_sid,
                    findings=list(_seb.findings),
                    turn_id=turn_id,
                )
            except Exception:  # noqa: BLE001
                pass
            yield TurnError(
                message=_seb.reason,
                classification="secret_egress_blocked",
            )
            yield TurnDone(state="error")
            return

        # Surface prompt drift without acting on it. Lets ops see
        # divergence between the bound CLI session's baked-in priming
        # prompt and the current composed prompt (e.g., the agent
        # mutated config.json mid-session) without silently abandoning
        # the conversation.
        if policy == "sha_changed" and effective_resume_id:
            from sharedagents_agent.tools._log import log_event as _log_event

            _log_event(
                "CLI_ENGINE_PROMPT_DRIFT_RESUMED",
                engine=self.kind.value,
                sharedagents_session_id=sharedagents_sid,
                engine_session_id=effective_resume_id,
                prior_sha=state.priming_system_prompt_sha256,
                current_sha=request.system_prompt_sha256,
                turn_id=turn_id,
            )

        # Marshall args for the legacy runner. Pre-refactor this path
        # mutated process-wide ``os.environ`` to convey session identity,
        # which raced between concurrent CLI turns on the same agent
        # (two TUIs against ``claude`` hit the same engine instance and
        # cross-wired the ``<sid>.delegate_claude.json`` sidecar). The
        # runner now reads ``_sharedagents_session_id`` /
        # ``_sharedagents_sessions_dir`` / ``_sharedagents_permission_mode``
        # directly from ``args`` and falls back to env only when
        # absent (back-compat for direct callers outside the engine).
        legacy_args: dict[str, Any] = {
            "prompt": user_prompt_text,
            # A manual prompt after the cap is still answerable, but native
            # CLI tools must be read-only if the model ignores the notice.
            # CLI engines cannot round-trip permission prompts. Always
            # force auto unless a spent budget requires a read-only turn.
            "mode": "plan" if manual_budget_exhausted else "auto",
            "effort": str(request.extra.get("effort") or "max").strip().lower(),
            "_sharedagents_session_id": sharedagents_sid,
            "_sharedagents_sessions_dir": str(sessions_dir),
            "_sharedagents_permission_mode": ("plan" if manual_budget_exhausted else "auto"),
            "_native_readiness_probe": bool(request.extra.get("_native_readiness_probe")),
            "_sharedagents_manual_budget_exhausted": manual_budget_exhausted,
        }
        # Per-agent API keys → spawn env override for the CLI
        # subprocess. Lets each sharedagents agent's claude/codex/antigravity
        # CLI hit its OWN provider sub-account so per-agent cost +
        # rate-limit attribution works (the same headline feature
        # LocalLLMEngine gets via cfg.upstream_headers). Empty dict
        # = CLI inherits os.environ only. Runners (runners.py
        # run_delegate_*_async) read ``_spawn_env_overrides`` and
        # pass it as ``env={**os.environ, **overrides}`` to
        # create_subprocess_exec.
        overrides = spawn_env_overrides(self.kind, request.extra)
        if overrides:
            legacy_args["_spawn_env_overrides"] = overrides
        # Spawn cwd resolution. The CLIs (Claude Code / Codex / Antigravity)
        # all key ``--resume <id>`` lookups by an encoded form of the
        # spawn cwd; if a resume runs from a different cwd than the
        # one that minted the engine_session_id, the lookup misses
        # silently and we'd fall back to a fresh CLI session — losing
        # the entire prior conversation. So when we have a recorded
        # ``state.spawn_cwd`` AND we're resuming (engine_session_id
        # set), we use the recorded cwd unconditionally; only when we
        # have no recorded spawn cwd yet do we honor request.cwd.
        # ``turn_spawn_cwd`` is closed over by the ``session_id``
        # event handler below so we know what cwd to lock when the
        # CLI first emits its session id.
        if state.spawn_cwd and effective_resume_id:
            turn_spawn_cwd = state.spawn_cwd
        else:
            turn_spawn_cwd = request.cwd or ""
            if not turn_spawn_cwd:
                # Claude/Codex resume stores are cwd-scoped; persist the
                # inherited cwd too so restart-from-elsewhere still resumes.
                from sharedagents_core.cwdutil import safe_getcwd

                turn_spawn_cwd = safe_getcwd()
        if turn_spawn_cwd:
            legacy_args["cwd"] = turn_spawn_cwd

        # --- Secret egress SINK guard: pre-spawn METADATA ----------
        # ``cwd`` and the permission ``mode`` reach the third-party CLI OUTSIDE
        # the composed-prompt payload screened at :674 — ``cwd`` as the
        # subprocess working dir the agentic CLI echoes back to the model as
        # environment context, ``mode`` as a permission flag — and ``cwd`` is
        # resolved only HERE, downstream of that guard (the resume-cwd-locking
        # block above). A secret-shaped path or mode would therefore have
        # egressed unscreened. Screen both through the SAME shared guard before
        # the spawn, mirroring the ACP cwd/mode metadata guard. Hard-block per
        # Abort the turn and NEVER spawn — fail CLOSED,
        # replicating the composed-prompt guard's cleanup (unlink image temp
        # files, log, error).
        _meta_mode = request.mode or "auto"
        try:
            _dgate.guard_egress_payload(
                self.kind,
                "\n\n".join(p for p in (turn_spawn_cwd, _meta_mode) if p),
            )
        except _dgate.SecretEgressBlocked as _seb:
            for _tmp in image_temp_files:
                try:
                    _tmp.unlink()
                except Exception:  # noqa: BLE001
                    pass
            try:
                from sharedagents_agent.tools._log import log_event as _log_event

                _log_event(
                    "DELEGATE_SECRET_EGRESS_BLOCKED",
                    engine=self.kind.value,
                    sharedagents_session_id=sharedagents_sid,
                    findings=list(_seb.findings),
                    turn_id=turn_id,
                    phase="pre_spawn_metadata",
                )
            except Exception:  # noqa: BLE001
                pass
            yield TurnError(
                message=_seb.reason,
                classification="secret_egress_blocked",
            )
            yield TurnDone(state="error")
            return

        # Model resolution happened ahead of the egress guard above (the
        # model is model-controllable and rides to the CLI as argv, so it is
        # screened there). ``effective_model`` is already bound (with no
        # override on a fresh spawn it resolves to the engine's effective
        # default — see the OPTION B block in the model-resolution section
        # above). Empty / "auto" → no ``--model`` flag, letting the CLI pick
        # its own default.
        #
        # MODEL GATE — GENERALIZED (owner round-2 scope change): every
        # external-CLI engine (claude / codex / antigravity) fails fast
        # BEFORE the spawn instead of burning the delegation on the CLI's
        # own rejection (the deepseek-v4-flash incident for codex; the
        # literal "claude" label incident for the Claude CLI — both are
        # foreign ids the vendor CLI rejects). The engine's own display
        # label is NEVER a wire model id and is refused here, so it can
        # never reach ``--model``. validate_external_cli_model returns the
        # CANONICAL slug — the catalogue entry's OWN spelling
        # (case-preserving; for codex the
        # installed codex's own model list) — and ONLY
        # that canonical value is built into ``--model``, never the raw
        # user string. Claude passes any non-label value through verbatim
        # (the Claude CLI is the authority for its own model syntax —
        # aliases / ``[1m]`` forms / operator deployments).
        # The refusal copy reflects the model's actual source (explicit
        # override / inherited parent override / picker pass-through /
        # configured default). Antigravity without an operator
        # catalogue degrades to the label invariant (documented limitation
        # — its CLI remains the authority).
        if self.kind in (
            EngineKind.CLAUDE_CLI,
            EngineKind.CODEX_CLI,
            EngineKind.ANTIGRAVITY_CLI,
            EngineKind.ZCODE_CLI,
        ):
            from sharedagents_agent.discovery._common import (
                external_cli_model_refusal,
                validate_external_cli_model,
            )

            if request.model:
                _model_source = str(request.extra.get("model_source") or "explicit")
            elif self._session_model(sharedagents_sid):
                _model_source = "picker"
            else:
                _model_source = "default"
            _canonical_model = validate_external_cli_model(
                self.kind.value,
                effective_model,
            )
            if _canonical_model is None:
                # Inherited local-LLM / vLLM ids (session stamp after
                # ``sharedagents configure --model qwen…``) are not an operator
                # Gemini pick. Drop them so auto omits ``--model``.
                inherited_local = self.kind is EngineKind.ANTIGRAVITY_CLI and not str(
                    effective_model or ""
                ).lower().startswith("gemini")
                if not inherited_local and self.kind is EngineKind.ZCODE_CLI:
                    # Same leak class as antigravity: a local vLLM id in the
                    # session stamp is not a GLM pick, so drop it (auto omits
                    # ``--model``) instead of failing the turn.
                    from sharedagents_agent.discovery.zcode import is_zcode_chat_model

                    inherited_local = not is_zcode_chat_model(effective_model)
                if inherited_local:
                    effective_model = ""
                    _canonical_model = ""
                else:
                    yield TurnError(
                        message=external_cli_model_refusal(
                            effective_model,
                            self.kind.value,
                            source=_model_source,
                        ),
                        classification="model_not_allowed",
                    )
                    yield TurnDone(state="error")
                    return
            effective_model = _canonical_model
            # A selected model applies to resumed turns too. Dropping it
            # acknowledges picker changes that the CLI never receives. With
            # no selection, the runner retains its native resume behavior.
            legacy_args["model"] = effective_model
        elif effective_model:
            legacy_args["model"] = effective_model
        if composed_sys:
            legacy_args["_append_system_prompt"] = composed_sys
        if effective_resume_id:
            legacy_args["_force_resume_id"] = effective_resume_id
        elif policy == "sha_changed":
            # We land here only when ``sha_changed`` AND no
            # engine_session_id is recorded — i.e., turn 0's session
            # id wasn't successfully captured. Treat as a fresh prime:
            # the legacy runners gate session resume on a per-engine
            # ``_sharedagents_<cli>_no_resume`` flag (see runners.py); set
            # it so the runner doesn't silently pick up a stale
            # legacy sidecar id.
            _no_resume_flag = {
                EngineKind.CLAUDE_CLI: "_sharedagents_claude_no_resume",
                EngineKind.CODEX_CLI: "_sharedagents_codex_no_resume",
                EngineKind.ANTIGRAVITY_CLI: "_sharedagents_antigravity_no_resume",
                EngineKind.ZCODE_CLI: "_sharedagents_zcode_no_resume",
            }.get(self.kind)
            if _no_resume_flag:
                legacy_args[_no_resume_flag] = True

        # Event channel from runner callbacks → our generator.
        ev_queue: _asyncio.Queue[AgentTurnEvent | None] = _asyncio.Queue()

        # Mirror Claude Code's IN-SUBPROCESS Task*/TodoWrite tools into sharedagents's
        # own session stores. For a claude_cli session those tools run inside the
        # CLI process, so `<sid>.tasks.json` / `<sid>.todos.json` stay empty and
        # the frontends' turn_ended reconcile (`/task list`, `/todo`) wipes the
        # live task widget. Reflecting the observed tool calls keeps the commands
        # authoritative. Best-effort: never fatal to the turn.
        _task_mirror = None
        if sharedagents_sid:
            try:
                from sharedagents_agent.tools.harness_task_mirror import (
                    HarnessTaskMirror,
                )
                from sharedagents_agent.tools.runners import todos_path_for_session

                _task_mirror = HarnessTaskMirror(
                    tasks_path=sessions_dir / f"{sharedagents_sid}.tasks.json",
                    todos_path=todos_path_for_session(sessions_dir, sharedagents_sid),
                )
            except Exception:  # noqa: BLE001
                _task_mirror = None

        from sharedagents_agent.continuation import TurnEventTracker

        nudge_tracker = TurnEventTracker()
        round_streamed_chunks: list[str] = []
        round_thinking_chunks: list[str] = []
        round_reasoning_chars = 0

        def _on_delta(text: str) -> None:
            nonlocal round_streamed_chunks
            try:
                if request.handoff is not None and not request.extra.get("_handoff_consumed"):
                    request.extra["_handoff_consumed"] = True
                    ev_queue.put_nowait(
                        HandoffAccepted(
                            handoff_id=request.handoff.handoff_id,
                            through_seq=request.handoff.through_seq,
                        )
                    )
                if text:
                    round_streamed_chunks.append(text)
                    nudge_tracker.on_content(text)
                ev_queue.put_nowait(TextDelta(text=text, channel="content"))
            except Exception:  # noqa: BLE001
                pass

        # Real per-turn usage/cost reported by the CLI (claude `result` event,
        # codex/antigravity equivalents). Captured here and merged into the
        # turn metrics below so record_turn_cost prices REAL tokens, not the
        # char-based estimate. Closure over run_turn locals → `nonlocal`.
        delegate_usage: dict | None = None
        delegate_context_usage: dict | None = None
        delegate_context_window = 0
        delegate_total_cost_usd: float | None = None
        # A persisted ``state.actual_model`` is evidence from some PRIOR turn.
        # Keep current-turn runtime evidence separate so a newly selected model
        # is not masked when this invocation emits no fresh model event.
        turn_actual_model = ""

        def _on_event(payload: dict) -> None:
            nonlocal delegate_usage, delegate_context_usage
            nonlocal delegate_context_window
            nonlocal delegate_total_cost_usd
            nonlocal turn_actual_model
            nonlocal round_tool_starts
            nonlocal round_reasoning_chars
            nonlocal round_thinking_chunks
            nonlocal turn_thinking_chunks
            nonlocal turn_reasoning_chars
            try:
                kind_ = (payload or {}).get("kind") or ""
                if (
                    kind_ in {"tool_use", "tool_result", "artifact_reference"}
                    and request.handoff is not None
                    and not request.extra.get("_handoff_consumed")
                ):
                    request.extra["_handoff_consumed"] = True
                    ev_queue.put_nowait(
                        HandoffAccepted(
                            handoff_id=request.handoff.handoff_id,
                            through_seq=request.handoff.through_seq,
                        )
                    )
                if kind_ == "usage":
                    u = payload.get("usage")
                    checkpoint = request.extra.get("_usage_checkpoint")
                    checkpointed_tokens = False
                    if isinstance(u, dict) and u:
                        # Feed each provider-reported subprocess usage into
                        # the shared live ledger checkpoint. The terminal
                        # metrics still carry the cumulative total; this hook
                        # prevents a long CLI turn from appearing unbilled
                        # until its final result arrives.
                        if callable(checkpoint):
                            usage_metrics = metrics_from_provider_usage(
                                u,
                                model=(turn_actual_model or effective_model or state.actual_model or self.name),
                                provider=_PROVIDER_FOR_KIND.get(self.kind.value, ""),
                            )
                            if usage_metrics:
                                checkpoint(usage_metrics)
                                checkpointed_tokens = True
                        delegate_usage = _accumulate_delegate_usage(
                            delegate_usage,
                            u,
                        )
                    if callable(checkpoint) and not checkpointed_tokens and payload.get("total_cost_usd") is not None:
                        # Some CLI integrations expose only their
                        # authoritative dollar total. Keep that cost live
                        # too; token buckets remain zero until a later
                        # real-usage or terminal snapshot arrives.
                        try:
                            _cost_only = float(payload["total_cost_usd"])
                        except (TypeError, ValueError):
                            _cost_only = -1.0
                        if _cost_only >= 0:
                            checkpoint(
                                {
                                    "model": (turn_actual_model or effective_model or state.actual_model or self.name),
                                    "provider": _PROVIDER_FOR_KIND.get(self.kind.value, ""),
                                    "cost_usd_override": _cost_only,
                                    "tokens_estimated": False,
                                }
                            )
                    cu = payload.get("context_usage")
                    if isinstance(cu, dict) and cu:
                        # This is a single-call occupancy snapshot, not a
                        # billing total. Keep the latest snapshot across any
                        # continuation subprocesses.
                        delegate_context_usage = dict(cu)
                    cw = payload.get("context_window")
                    if isinstance(cw, int) and cw > 0:
                        # The window the CLI reports running the model with
                        # beats the catalogue maximum (claude: modelUsage).
                        delegate_context_window = cw
                    if payload.get("total_cost_usd") is not None:
                        try:
                            cost = float(payload["total_cost_usd"])
                            if cost >= 0:
                                delegate_total_cost_usd = (delegate_total_cost_usd or 0.0) + cost
                        except (TypeError, ValueError):
                            pass
                    return
                if kind_ == "thinking":
                    text = str(payload.get("text") or "")
                    if text:
                        round_thinking_chunks.append(text)
                        turn_thinking_chunks.append(text)
                        round_reasoning_chars += len(text)
                        turn_reasoning_chars += len(text)
                        nudge_tracker.on_thinking(text)
                        ev_queue.put_nowait(
                            TextDelta(text=text, channel="thinking"),
                        )
                    return
                if kind_ == "reasoning":
                    try:
                        chars = int(payload.get("chars") or 0)
                    except (TypeError, ValueError):
                        chars = 0
                    if chars > 0:
                        round_reasoning_chars += chars
                        turn_reasoning_chars += chars
                        nudge_tracker.on_reasoning_chars(chars)
                    return
                if kind_ == "tool_use":
                    _call_id = str(payload.get("id") or "")
                    _name = str(payload.get("name") or "?")
                    _args = dict(payload.get("input") or {})
                    round_tool_starts += 1
                    nudge_tracker.on_tool(_name, arguments=_args)
                    if _task_mirror is not None:
                        _task_mirror.on_tool_call(_name, _call_id, _args)
                    ev_queue.put_nowait(
                        ToolCallStarted(
                            call_id=_call_id,
                            name=_name,
                            arguments=_args,
                        )
                    )
                elif kind_ == "tool_result":
                    _res_id = str(payload.get("tool_use_id") or "")
                    _res_ok = bool(payload.get("ok", True))
                    _res_text = str(payload.get("text") or "")
                    # A completion-only tool event still proves this round
                    # performed tool work (ACP already counts ToolCallResult
                    # the same way). Antigravity can omit the matching start.
                    round_tool_starts += 1
                    nudge_tracker.on_tool("", ok=_res_ok, result=_res_text)
                    if _task_mirror is not None:
                        _task_mirror.on_tool_result(_res_id, _res_text, _res_ok)
                    ev_queue.put_nowait(
                        ToolCallResult(
                            call_id=_res_id,
                            ok=_res_ok,
                            content=_res_text,
                        )
                    )
                elif kind_ == "artifact_reference":
                    reference = payload.get("reference") or {}
                    if isinstance(reference, dict):
                        ev_queue.put_nowait(
                            ArtifactReference(
                                uri=str(reference.get("uri") or ""),
                                path=str(reference.get("path") or ""),
                                display_name=str(reference.get("display_name") or ""),
                                media_type=str(reference.get("media_type") or ""),
                                size_bytes=int(reference.get("size_bytes") or 0),
                                sha256=str(reference.get("sha256") or ""),
                                storage_scope=str(reference.get("storage_scope") or ""),
                                locator=str(reference.get("locator") or ""),
                            )
                        )
                elif kind_ == "session_id":
                    # Runner saw the CLI's session id on a streamed
                    # event. Persist immediately so an ESC / idle
                    # timeout before TurnDone still leaves the
                    # sidecar primed for `--resume` on the next
                    # turn. Without this, an aborted first turn
                    # erases the id and the user loses context.
                    new_eid = str(payload.get("id") or "")
                    if new_eid and new_eid != state.engine_session_id:
                        state.engine_session_id = new_eid
                        state.priming_status = "completed"
                        state.last_error = ""
                        # Lock the spawn cwd alongside the session id.
                        # This is the cwd the CLI just used to encode
                        # its on-disk session record; every future
                        # resume MUST use the same cwd or the lookup
                        # will silently miss. Recorded only on the
                        # first sighting (``not state.spawn_cwd``) so
                        # an unrelated mid-stream session-id rebind
                        # doesn't relocate the lock.
                        if not state.spawn_cwd and turn_spawn_cwd:
                            state.spawn_cwd = turn_spawn_cwd
                        try:
                            self.write_state(sessions_dir, state)
                        except Exception:  # noqa: BLE001
                            pass
                elif kind_ == "model":
                    # The CLI's init/system event reports the real model id
                    # (e.g. ``claude-opus-4-…``). Persist it so the TUI
                    # footer shows the actual model rather than the
                    # engine-name placeholder. Metadata only — not surfaced
                    # as an AgentTurnEvent.
                    new_model = str(payload.get("id") or "")
                    if new_model:
                        turn_actual_model = new_model
                    if new_model and new_model != state.actual_model:
                        state.actual_model = new_model
                        try:
                            self.write_state(sessions_dir, state)
                        except Exception:  # noqa: BLE001
                            pass
                elif kind_ == "native_mode":
                    # Claude's structured system/init event reports the
                    # permission mode the CLI actually activated. This is
                    # stronger evidence than asking the model to attest its
                    # own launch flags in prose.
                    native_sid = str(payload.get("session_id") or "")
                    if (
                        payload.get("verified") is True
                        and payload.get("mode") == request.mode
                        and native_sid
                        and native_sid == state.engine_session_id
                    ):
                        request.extra["_native_mode_evidenced"] = request.mode
                        request.extra["_native_mode_receipt"] = {
                            "mode": request.mode,
                            "native_value": str(payload.get("native_value") or ""),
                            "session_id": native_sid,
                        }
            except Exception:  # noqa: BLE001
                pass

        runner = self.delegate_runner()
        if runner is None:
            yield TurnError(
                message=f"{self.name} engine has no delegate_runner wired",
                classification="not_yet_implemented",
            )
            yield TurnDone(state="error")
            return

        # Mark priming status as in_progress + write state so a crash
        # before TurnDone preserves a recoverable engine_session_id.
        state.priming_status = "in_progress"
        self.write_state(sessions_dir, state)

        def _register_proc(proc: Any) -> None:
            # Called by the runner the instant the CLI subprocess is
            # spawned; lets :meth:`abort` reach the live process group.
            self._active_cli_procs[turn_id] = proc
            self._session_turns.setdefault(sharedagents_sid, set()).add(turn_id)
            # Durable backstop: persist pid/pgid so a HARD abort can SIGKILL
            # this engine even with zero in-memory state (e.g. after the
            # orchestrator restarts and orphans it). Best-effort.
            try:
                from . import engine_pidfile

                pid = getattr(proc, "pid", None)
                if pid is not None:
                    engine_pidfile.record(
                        sessions_dir,
                        sharedagents_sid,
                        int(pid),
                        cli=self.cli_command,
                    )
            except Exception:  # noqa: BLE001
                pass

        async def _drive() -> Any:
            kwargs: dict[str, Any] = {
                "on_delta": _on_delta,
                "on_spawn": _register_proc,
            }
            if self.supports_on_event():
                kwargs["on_event"] = _on_event
            try:
                return await runner(legacy_args, **kwargs)
            finally:
                self._active_cli_procs.pop(turn_id, None)
                s = self._session_turns.get(sharedagents_sid)
                if s is not None:
                    s.discard(turn_id)
                    if not s:
                        self._session_turns.pop(sharedagents_sid, None)
                # Drop the durable pidfile — the subprocess is being torn down
                # here, so a later abort must not chase a dead/recycled pid.
                try:
                    from . import engine_pidfile

                    engine_pidfile.clear(sessions_dir, sharedagents_sid)
                except Exception:  # noqa: BLE001
                    pass
                # Sentinel signals end-of-stream to the consumer below.
                await ev_queue.put(None)

        terminal_state = "error"
        terminal_metrics: dict[str, Any] = {}
        result_text = ""
        nudge_count = 0
        timeout_nudge_count = 0
        tool_rounds_so_far = 0
        turn_visible_chars = 0
        round_tool_starts = 0
        turn_thinking_chunks: list[str] = []
        turn_reasoning_chars = 0
        driver_task: Any = None

        # Timing instrumentation for metrics (ttft, tps, wt).
        turn_start = time.monotonic()
        first_token_at: float | None = None
        last_token_at: float | None = None
        delta_count = 0

        # Every exit path (success, runner-error, ESC-cancel) MUST yield
        # exactly one terminal TurnDone — otherwise the upstream ACP
        # server logs "backend.prompt iterator finished without yielding
        # a PromptResponse" and returns JSON-RPC -32603 to the TUI, the
        # sharedagents session record never gets an aborted=true row (since
        # Agent.run_turn only persists the turn on TurnDone), and the
        # operator's next prompt on the same sid hits the same error
        # against a stale priming_status=failed sidecar until they give
        # up and create a fresh session — losing all context. So the
        # `try`/`except`/`else` ladder below only sets ``terminal_state``;
        # the unconditional `yield TurnDone(...)` after the `finally`
        # block fires for every path.
        try:
            while True:
                if turn_id in self._operator_stop_turns:
                    # Stop landed before this spawn (or between idle-empty
                    # nudges). Do not start another CLI process.
                    terminal_state = "aborted"
                    break
                round_tool_starts = 0
                round_streamed_chunks = []
                round_thinking_chunks = []
                round_reasoning_chars = 0
                nudge_tracker.begin_round()
                driver_task = _asyncio.create_task(_drive())
                # Track in active map for ESC routing.
                self._active_procs[turn_id] = driver_task
                try:
                    while True:
                        ev = await ev_queue.get()
                        if ev is None:
                            break
                        # Capture timing for TextDelta events.
                        if isinstance(ev, TextDelta):
                            if first_token_at is None:
                                first_token_at = time.monotonic()
                            last_token_at = time.monotonic()
                            delta_count += 1
                        yield ev
                except _asyncio.CancelledError:
                    # Generator cancelled by the consumer (Agent.run_turn
                    # task cancel). Cancel the driver so the legacy runner's
                    # own CancelledError handler SIGTERMs the CLI process
                    # group, then fall through to the terminal yield.
                    driver_task.cancel()
                    terminal_state = "aborted"
                    break
                else:
                    # Driver signalled end-of-stream (sentinel None). Harvest
                    # the ToolResult to classify success / error.
                    try:
                        tool_result = await driver_task
                    except _asyncio.CancelledError:
                        # External ESC → engine.abort(turn_id) cancelled the
                        # driver mid-stream; the runner's finally drained
                        # the sentinel before propagating CancelledError.
                        terminal_state = "aborted"
                        break
                    except FileNotFoundError as exc:
                        if turn_id in self._operator_stop_turns:
                            terminal_state = "aborted"
                            break
                        # A bare ``FileNotFoundError`` ([Errno 2]) escaping the
                        # runner is the single most model-unrecoverable failure
                        # this engine can surface: the message is just
                        # ``[Errno 2] No such file or directory`` with no hint
                        # of WHICH file — the CLI binary (not installed / not on
                        # PATH) or the spawn cwd (a long-lived gateway whose
                        # launch dir was removed — the dead-cwd class the
                        # ``_safe_getcwd`` fix hardened one layer up in
                        # ``Agent._hook_kwargs``; any residual unguarded
                        # ``os.getcwd()`` in the delegate runner re-surfaces it
                        # HERE). A small / local model fed ``codex runner
                        # raised: [Errno 2]`` loops or gives up — it can't tell
                        # whether to retry, switch ``subagent_type``, or report
                        # inability. Translate it into an ACTIONABLE message
                        # (names the binary + both likely causes + a recovery
                        # path) under a distinct ``cli_not_available``
                        # classification so the turn runtime / model can recover
                        # instead of leaking the raw errno.
                        yield TurnError(
                            message=(
                                f"{self.name} CLI could not be launched "
                                f"(FileNotFoundError: {exc}). The "
                                f"'{self.cli_command}' binary may not be "
                                f"installed or on PATH, or the working "
                                "directory was removed. Install it, or "
                                "dispatch a different subagent_type "
                                "(e.g. general-purpose)."
                            ),
                            classification="cli_not_available",
                        )
                        terminal_state = "error"
                        break
                    except Exception as exc:  # noqa: BLE001
                        if turn_id in self._operator_stop_turns:
                            terminal_state = "aborted"
                            break
                        yield TurnError(
                            message=f"{self.name} runner raised: {exc}",
                            classification="engine_exception",
                        )
                        terminal_state = "error"
                        break
                    else:
                        ok = bool(getattr(tool_result, "ok", False))
                        result_text = str(getattr(tool_result, "content", "") or "")
                        if turn_id in self._operator_stop_turns:
                            # The runner finished because Stop/abort asked it
                            # to. Do this before both error classification and
                            # idle-empty continuation so operator intent can
                            # never become another automatic attempt.
                            terminal_state = "aborted"
                            break
                        if not ok:
                            from sharedagents_agent.continuation import (
                                MAX_TIMEOUT_CONTINUES,
                                cli_timeout_continue_body,
                                cli_timeout_continue_notice,
                                is_cli_print_timeout,
                            )
                            from sharedagents_agent.tools.impl._utils import (
                                _antigravity_print_timeout_s,
                            )

                            from ..tools.runners import _claude_error_is_abort

                            is_abort = _cli_tool_result_is_abort(result_text)
                            if not is_abort:
                                meta = getattr(tool_result, "metadata", None)
                                if isinstance(meta, dict):
                                    is_abort = _claude_error_is_abort(meta)
                            if is_abort:
                                terminal_state = "aborted"
                                break
                            resume_id = state.engine_session_id or str(legacy_args.get("_force_resume_id") or "")
                            if (
                                is_cli_print_timeout(result_text)
                                and resume_id
                                and timeout_nudge_count < MAX_TIMEOUT_CONTINUES
                            ):
                                timeout_nudge_count += 1
                                wall_s = _antigravity_print_timeout_s()
                                yield TextDelta(
                                    text=cli_timeout_continue_notice(
                                        attempt=timeout_nudge_count,
                                        timeout_s=wall_s,
                                    ),
                                    channel="content",
                                )
                                legacy_args["prompt"] = cli_timeout_continue_body(
                                    wall_s,
                                )
                                legacy_args["_force_resume_id"] = resume_id
                                legacy_args.pop("_append_system_prompt", None)
                                continue
                            if is_cli_print_timeout(result_text):
                                classification = "cli_timeout"
                            elif _cli_error_is_quota_exhausted(result_text):
                                # Plan quota is deterministic until the reset
                                # time: the 5 s retry burst would only burn
                                # more calls against the wall.
                                classification = "quota_exhausted"
                                _remember_cli_quota(self.kind, result_text)
                            else:
                                classification = "cli_error"
                            yield TurnError(
                                message=result_text or f"{self.name} reported error",
                                classification=classification,
                            )
                            terminal_state = "error"
                            break
                        # Success: the CLI session is now primed. Extract
                        # the engine_session_id from the result body
                        # header (``[claude session=<sid:8>... ...]``)
                        # and persist.
                        new_sid = self._extract_session_id_from_result(
                            result_text,
                            sharedagents_sid=sharedagents_sid,
                            sessions_dir=sessions_dir,
                        )
                        if new_sid:
                            state.engine_session_id = new_sid
                        state.priming_status = "completed"
                        if effective_prompt_delivered:
                            state.priming_system_prompt_sha256 = request.system_prompt_sha256
                        state.last_error = ""
                        # Lock spawn_cwd on the success terminal too —
                        # covers CLIs whose ``session_id`` event the
                        # runner doesn't surface mid-stream (only the
                        # claude runner does today). Without this, a
                        # codex/antigravity session would only get its cwd
                        # locked once a subsequent turn happens to
                        # spawn from a different cwd, which is exactly
                        # the failure mode we're trying to prevent.
                        if not state.spawn_cwd and turn_spawn_cwd:
                            state.spawn_cwd = turn_spawn_cwd
                        self.write_state(sessions_dir, state)
                        terminal_state = "ok"
                        if round_tool_starts:
                            tool_rounds_so_far += 1
                        # Structured runners keep tool markers in the persisted
                        # ToolResult for replay but omit them from on_delta.
                        # Count the streamed answer first. Legacy runners or
                        # test doubles that only return the persisted body use
                        # the narrow marker-aware parser instead.
                        streamed_text = "".join(round_streamed_chunks)
                        round_produced = _cli_visible_result_chars(
                            streamed_text if streamed_text else result_text,
                        )
                        if not streamed_text and round_produced:
                            nudge_tracker.on_content(result_text)
                        turn_visible_chars += round_produced
                        from sharedagents_agent.continuation import (
                            MAX_IDLE_EMPTY_CONTINUES,
                            apply_nudge_hooks,
                            idle_empty_continue_notice,
                        )
                        from sharedagents_agent.hooks import fire_hook_event

                        _hook_engine = request.extra.get("hook_engine")
                        _hook_ctx = request.extra.get("hook_ctx") or {}
                        for _t in list(nudge_tracker.tools_this_round):
                            await fire_hook_event(
                                _hook_engine,
                                _hook_ctx,
                                "PreToolUse",
                                {"tool_name": _t.get("name") or "", "tool_input": _t.get("arguments") or {}},
                            )
                            await fire_hook_event(
                                _hook_engine,
                                _hook_ctx,
                                "PostToolUse",
                                {
                                    "tool_name": _t.get("name") or "",
                                    "tool_input": _t.get("arguments") or {},
                                    "tool_response": _t.get("result") or "",
                                },
                            )
                        plan = await apply_nudge_hooks(
                            nudge_tracker,
                            hook_engine=_hook_engine,
                            hook_ctx=_hook_ctx,
                            finish_reason="stop",
                            cancelled=False,
                            nudge_count=nudge_count,
                            max_nudges=MAX_IDLE_EMPTY_CONTINUES,
                        )
                        nudge_tracker.commit_round()
                        if not plan.is_idle_resume:
                            break
                        body = plan.body
                        reason_code = plan.decision.reason_code
                        nudge_count += 1
                        for sm in plan.system_messages:
                            yield TextDelta(
                                text=f"[hook] {sm}\n",
                                channel="content",
                            )
                        yield TextDelta(
                            text=idle_empty_continue_notice(
                                reason_code,
                                attempt=nudge_count,
                            ),
                            channel="content",
                        )
                        legacy_args["prompt"] = body
                        if state.engine_session_id:
                            legacy_args["_force_resume_id"] = state.engine_session_id
                        legacy_args.pop("_append_system_prompt", None)
                        continue
        finally:
            # ``async_generator.aclose()`` injects ``GeneratorExit`` at a
            # suspended ``yield``.  It does not take the CancelledError branch
            # above, so without an explicit reap here the private driver keeps
            # running after every upstream layer has forgotten the turn.  In
            # the Codex case that leaves ``codex exec`` detached: it may finish
            # and even write the legacy thread sidecar, but SharedAgents never harvests
            # the result or emits TurnDone.  Cancel and await on every
            # non-terminal exit; awaiting a completed task also retrieves any
            # exception so asyncio cannot report an orphan later.
            if driver_task is not None and not driver_task.done():
                driver_task.cancel()
            if driver_task is not None:
                try:
                    await driver_task
                except _asyncio.CancelledError:
                    pass
                except Exception:  # noqa: BLE001 — cleanup must preserve exit cause
                    pass
            self._operator_stop_turns.discard(turn_id)
            self._active_procs.pop(turn_id, None)
            self._active_cli_procs.pop(turn_id, None)
            # Drop the temp image files materialized for this turn.
            # The CLI subprocess has exited (success / error) or is
            # being group-killed (abort) by the time we get here, and
            # it read the images off disk at turn start — so unlinking
            # now is safe. Every terminal path (including the cancel
            # branches above) routes through this ``finally``.
            for _tmp in image_temp_files:
                try:
                    _tmp.unlink()
                except OSError:
                    pass
            # If we exited with anything other than "ok" (and we
            # have an engine_session_id), mark priming_status=failed
            # so the next call can decide to retry-with-resume vs
            # start fresh. EXCEPT: a session id we just acquired
            # mid-stream this turn (via the on_event "session_id"
            # path — first turn aborted between session-id sighting
            # and end-of-stream) is almost certainly still resumable
            # on the CLI side; downgrading it would force a fresh
            # prime and discard the conversation.
            if (
                terminal_state != "ok"
                and state.engine_session_id
                and state.engine_session_id == initial_engine_session_id
            ):
                state.priming_status = "failed"
                self.write_state(sessions_dir, state)

        # Build terminal metrics via the shared estimator so this engine
        # renders the SAME full field set as the local (vLLM) path. We measure
        # what a CLI gives us — wall time, ttft/gen windows (only when the
        # stream was progressive), the sent prompt + received body char counts
        # — and complete_metrics() derives the rest (tokens, prefill, total),
        # flagging everything ``≈``.
        turn_end = time.monotonic()
        turn_thinking = "".join(turn_thinking_chunks)
        effective_thinking_chars = (
            len(turn_thinking) if turn_thinking else (turn_reasoning_chars or round_reasoning_chars)
        )
        reported_model = turn_actual_model or effective_model or state.actual_model or self.name
        # ttft/gen windows are only honest on a progressive (multi-delta)
        # stream — a single terminal blob labelled "ttft" would mislead.
        ttft_s = (first_token_at - turn_start) if (first_token_at is not None and delta_count >= 2) else None
        gen_window_s = (
            (last_token_at - first_token_at)
            if (first_token_at is not None and last_token_at is not None and delta_count >= 3)
            else None
        )
        from sharedagents_agent.metrics_format import complete_metrics

        terminal_metrics = complete_metrics(
            {"engine": self.kind.value, "model": reported_model},
            prompt_chars=len(user_prompt_text or ""),
            thinking_chars=effective_thinking_chars,
            content_chars=len(result_text),
            model=reported_model,
            context_window=(delegate_context_window or self.context_window(reported_model) or None),
            turn_s=turn_end - turn_start,
            ttft_s=ttft_s,
            gen_window_s=gen_window_s,
        )
        # Cost accuracy: prefer the CLI's REAL reported usage over the
        # char-based estimate, and tag the provider so the pricing table
        # resolves to this engine's rate. When the CLI reports a total cost
        # but no usable tokens, carry it as an explicit override.
        terminal_metrics.setdefault("provider", _PROVIDER_FOR_KIND.get(self.kind.value, ""))
        terminal_metrics = _apply_delegate_usage(terminal_metrics, delegate_usage, delegate_context_usage)
        if (turn_reasoning_chars > 0 or round_reasoning_chars > 0) and not terminal_metrics.get("reasoning_tokens"):
            terminal_metrics["reasoning_tokens"] = max(1, round(max(turn_reasoning_chars, round_reasoning_chars) / 4))
        if self.kind is EngineKind.CODEX_CLI:
            terminal_metrics = _apply_codex_effective_throughput(terminal_metrics, turn_end - turn_start)
        if delegate_total_cost_usd is not None and terminal_metrics.get("tokens_estimated") is not False:
            terminal_metrics["cost_usd_override"] = delegate_total_cost_usd

        if not turn_thinking and terminal_metrics.get("reasoning_tokens"):
            rt = terminal_metrics["reasoning_tokens"]
            turn_thinking = f"Reasoning trace ({rt:,} tokens processed by model)."

        request.extra["_engine_audit"] = {
            "thinking": turn_thinking,
            "content": result_text,
            "tool_calls": [],
            "tool_results": [],
            "metrics": dict(terminal_metrics),
            "started_at": "",
            "ended_at": "",
            "state": terminal_state,
        }
        if terminal_state == "ok":
            native_mode_receipt = request.extra.get("_native_mode_receipt")
            structured_native_mode = bool(
                request.extra.get("_native_readiness_probe")
                and self.kind is EngineKind.CLAUDE_CLI
                and isinstance(native_mode_receipt, dict)
                and native_mode_receipt.get("mode") == request.mode
                and native_mode_receipt.get("session_id") == state.engine_session_id
                and native_mode_receipt.get("native_value")
            )
            if not (request.extra.get("_native_readiness_probe") and self.kind is EngineKind.CLAUDE_CLI):
                request.extra["_native_mode_evidenced"] = request.mode
            request.extra["_native_acceptance_receipt"] = {
                "kind": ("cli_native_mode_terminal" if structured_native_mode else "cli_success_terminal"),
                "mode": request.mode,
                # Report the prompt snapshot actually bound to the native
                # session. A resumed mode transition must not pretend the
                # newly composed prompt replaced that immutable snapshot.
                "effective_prompt_sha256": state.priming_system_prompt_sha256,
                "session_id": state.engine_session_id,
            }
            if structured_native_mode:
                request.extra["_native_acceptance_receipt"]["native_value"] = native_mode_receipt["native_value"]
        yield TurnDone(
            state=terminal_state,
            engine_session_id=state.engine_session_id,
            metrics=terminal_metrics,
        )

    def _extract_session_id_from_result(
        self,
        body: str,
        *,
        sharedagents_sid: str = "",
        sessions_dir: Path | None = None,
    ) -> str:
        """Pull ``session=<id>`` (Claude/Antigravity) or ``thread=<id>``
        (Codex) out of the legacy result-body header.

        Legacy runners produce a header like
        ``[claude session=abc12345…1234 · mode=…]`` at the top of
        the tool-result body. Parsing that out is the cheapest way
        to recover the CLI's own session id without re-reading the
        legacy sidecar file (which the runner already wrote).
        ``sharedagents_sid`` / ``sessions_dir`` are passed through to the
        sidecar fallback so concurrent CLI turns don't cross-wire
        through ``os.environ``.
        """
        import re

        m = re.search(r"(?:session|thread)=([A-Za-z0-9_\-]+)", body or "")
        if not m:
            return ""
        head = m.group(1)
        # Header truncates to 12 chars w/ ellipsis; the FULL id is
        # in the sidecar. Fall back to the sidecar read.
        if "…" in head or len(head) <= 12:
            return self._read_legacy_sidecar_session_id(
                sharedagents_sid=sharedagents_sid,
                sessions_dir=sessions_dir,
            )
        return head

    def _read_legacy_sidecar_session_id(
        self,
        *,
        sharedagents_sid: str = "",
        sessions_dir: Path | None = None,
    ) -> str:
        """Read the legacy ``<sid>.delegate_<cli>.json`` sidecar to
        recover the full engine session id when the result header
        carries only a truncated form. Explicit ``sharedagents_sid`` /
        ``sessions_dir`` args are required for the engine pipeline so
        concurrent CLI turns on the same agent don't cross-wire
        through process-wide ``os.environ``; absent args fall back to
        env for legacy direct callers (none today, but defensive)."""
        import json
        import os

        sid = sharedagents_sid or os.environ.get("SHAREDAGENTS_SESSION_ID") or ""
        sdir = str(sessions_dir) if sessions_dir else (os.environ.get("SHAREDAGENTS_SESSIONS_DIR") or "")
        if not sid or not sdir:
            return ""
        path = Path(sdir) / f"{sid}.delegate_{self.cli_command}.json"
        if not path.exists():
            return ""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ""
        from sharedagents_agent.external_registry import spec_for

        key = spec_for(self.kind).session_id_key or ""
        return str(data.get(key) or "")

    async def abort(self, turn_id: str, *, force: bool = False) -> None:
        """Stop the in-flight CLI turn, gracefully when possible.

        First press (``force=False``): send a friendly ``SIGINT`` to the
        delegated CLI's process group — the documented "wind down
        cleanly" signal for ``claude -p`` (it flushes its reply and its
        own ``--resume`` session file, then exits) — and wait up to
        :data:`CLI_GRACEFUL_ABORT_S` for the driver to reach its terminal
        stream event. If it finishes in time, NO hard kill is sent and
        the turn (plus the CLI's resume state) is preserved. On timeout
        we escalate ``SIGTERM`` → grace → ``SIGKILL`` via the runner's
        :func:`_terminate_group` and cancel the driver.

        We wait on the driver task (which completes when the event
        stream ends), NOT on process exit — ``claude -p`` in stream-json
        mode is known to linger after emitting its final ``result``
        event, so keying off the stream avoids a needless full-grace
        wait when the turn is already captured.

        Second press (``force=True``): skip the grace window and cancel
        the driver immediately; the runner's own ``CancelledError``
        handler SIGTERM/SIGKILLs the subprocess group right away.

        Idempotent — abort on an unknown / already-finished turn_id is a
        silent no-op.
        """
        # Imported lazily to avoid a core↔tools import cycle at module load.
        from ..tools.runners import _kill_group, _kill_tree, _terminate_group

        # Record intent first even when no driver is registered yet
        # (abort-before-spawn) or the map was already popped. A later
        # harvest of this turn_id must still see operator-requested stop.
        if turn_id:
            self._operator_stop_turns.add(turn_id)
        task = self._active_procs.get(turn_id)
        if task is None:
            return
        if task.done():
            # The driver may have queued its sentinel while run_turn has not
            # harvested/classified the result yet. There is no process left to
            # signal, but Stop still owns that pending terminal outcome.
            return

        if force:
            # Hard kill NOW (``/abort`` / second ESC). Don't merely cancel
            # the driver and let its CancelledError handler reap — that is
            # neither instant (the cancel only lands at the driver's next
            # await) nor complete (the handler's sweep is killpg-only and
            # misses subagents the CLI spawned in their OWN sessions). So
            # SIGKILL the CLI subprocess's ENTIRE descendant tree
            # synchronously on this tick, THEN cancel the driver so the
            # engine's round loop unwinds.
            proc = self._active_cli_procs.get(turn_id)
            if proc is not None:
                _kill_tree(proc, signal.SIGKILL)
            try:
                task.cancel()
            except Exception:  # noqa: BLE001
                pass
            return

        proc = self._active_cli_procs.get(turn_id)
        if proc is None:
            # No live subprocess to be gentle with (between rounds, or it
            # already exited) — fall back to a plain driver cancel.
            try:
                task.cancel()
            except Exception:  # noqa: BLE001
                pass
            return

        try:
            # Close stdin (EOF) + friendly SIGINT to the whole group.
            stdin = getattr(proc, "stdin", None)
            if stdin is not None:
                try:
                    stdin.close()
                except Exception:  # noqa: BLE001
                    pass
            _kill_group(proc, signal.SIGINT)
            try:
                # ``shield`` so the wait_for timeout doesn't cancel the
                # driver out from under us — we own the escalation below.
                await asyncio.wait_for(
                    asyncio.shield(task),
                    CLI_GRACEFUL_ABORT_S,
                )
            except asyncio.TimeoutError:
                # Didn't wind down in time → hard escalation.
                await _terminate_group(proc, grace_s=2.0)
                if not task.done():
                    task.cancel()
            except asyncio.CancelledError:
                # Driver was cancelled elsewhere mid-grace — nothing more
                # to do; the runner's own handler kills the subprocess.
                pass
        except Exception:  # noqa: BLE001
            if not task.done():
                task.cancel()

    async def request_graceful_stop(self, turn_id: str) -> None:
        """Cooperative first-tier stop for a CLI-backed turn.

        The graceful counterpart the frontends probe for on a first
        ESC / ``/stop`` (see ``AgentACPBackend.cancel`` and
        ``CoreBackend.cancel``): without it, a graceful cancel on a
        codex_cli / claude_cli agent is a signal-free no-op, so the
        operator's first stop does NOTHING and the turn keeps running
        until they escalate to a hard ``/abort`` (which SIGKILLs and
        discards all in-progress work). A local-LLM engine flips a
        cooperative flag here; a CLI engine has no in-process round
        loop to poll, so the cooperative equivalent is the SIGINT-then-
        grace path already implemented by ``abort(force=False)`` —
        SIGINT is the CLEAN stop for ``claude -p`` / ``codex exec``
        (they flush their reply + resume state, then exit), and we wait
        up to ``CLI_GRACEFUL_ABORT_S`` for the driver's terminal event
        before any hard escalation. No SIGKILL is sent on this path
        unless the CLI ignores the grace window.
        """
        await self.abort(turn_id, force=False)

    async def abort_session(
        self,
        session_id: str,
        *,
        sessions_dir: Path | None = None,
        force: bool = True,
    ) -> bool:
        """Abort EVERY live turn for ``session_id`` even without a turn_id,
        then SIGKILL any orphaned engine subprocess via the durable pidfile.

        This is the by-session counterpart to :meth:`abort` (which needs a
        turn_id). Two layers:

        1. **In-memory:** abort each turn_id the reverse index still holds for
           the session — reuses :meth:`abort`'s correct kill path.
        2. **Durable:** ``engine_pidfile.kill_session_engine`` SIGKILLs the
           recorded subprocess tree using ONLY the sidecar, so an engine
           ORPHANED across an orchestrator restart (in-memory maps gone) is
           still reaped. Identity-guarded against pid reuse.

        Returns ``True`` iff something live was signalled. Idempotent; never
        raises — an abort must not be failable.
        """
        signalled = False
        for tid in list(self._session_turns.get(session_id, ())):
            try:
                await self.abort(tid, force=force)
                signalled = True
            except Exception:  # noqa: BLE001
                pass
        if force and sessions_dir is not None:
            try:
                from . import engine_pidfile

                if engine_pidfile.kill_session_engine(sessions_dir, session_id):
                    signalled = True
            except Exception:  # noqa: BLE001
                pass
        return signalled

    async def session_lock(self, sharedagents_sid: str) -> Any:
        """Lazy-allocate a per-session asyncio.Lock."""
        lock = self._session_locks.get(sharedagents_sid)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[sharedagents_sid] = lock
        return lock

    def is_available(self) -> tuple[bool, str]:
        """Report whether the CLI is launchable. Returns
        ``(False, install-url-hint)`` when missing.

        Resolution goes through :func:`sharedagents_agent.tools.runners._which_cli`
        — the SAME helper the delegate runners use to spawn — rather than a
        bare ``shutil.which``. This is the launch gate for subagent dispatch
        (``delegate_shims``): a detached gateway daemon started with a
        stripped-down ``PATH`` that omits the operator's npm/local bin dir
        would otherwise reject the launch here with "CLI not on PATH" even
        though the runner could (and now would) find and spawn the CLI from a
        fallback bin dir. Sharing the resolver keeps the gate and the spawn
        from ever disagreeing (fp 836b1239)."""
        from sharedagents_agent.tools.runners import _which_cli

        if _which_cli(self.cli_command):
            return True, ""
        return False, (
            f"`{self.cli_command}` CLI not on PATH" + (f". Install: {self.install_url}" if self.install_url else "")
        )

    # ---- system-prompt policy ---------------------------------------

    def system_prompt_policy(
        self,
        state: EngineSessionState,
        *,
        current_system_prompt_sha256: str,
    ) -> str:
        """CLI engines prime once, then resume.

        Returns ``"first_turn"`` when ``state.priming_status in
        {"never", "in_progress", "failed"}``; ``"sha_changed"`` when
        ``current_system_prompt_sha256 !=
        state.priming_system_prompt_sha256``; ``"never"`` once the
        session is fully primed AND the prompt hash matches.

        Compare against :meth:`LocalLLMEngine.system_prompt_policy`
        which always returns ``"every_turn"``.

        Concrete here on the base class — every CLI engine shares
        this state-machine logic (no subclass override). Per codex
        Leaving this as
        ``NotImplementedError`` would crash every CLI agent at the
        policy step.
        """
        if state.priming_status in ("never", "in_progress", "failed"):
            return "first_turn"
        if state.priming_system_prompt_sha256 and state.priming_system_prompt_sha256 != current_system_prompt_sha256:
            return "sha_changed"
        return "never"
