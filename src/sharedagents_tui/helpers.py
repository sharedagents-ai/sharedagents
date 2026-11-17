"""Module-level globals and helper functions for the sharedagents TUI.

Extracted verbatim from ``app.py`` (behavior-preserving de-monolith).
Every symbol is re-exported from ``sharedagents_tui.app`` for import compatibility."""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from textual.app import App

from sharedagents_agent.metrics_format import metrics_line as _metrics_line


# SSH detection — when the TUI runs over a remote shell, throttle paint
# rate and reduce the per-tick chatter. Each escape sequence is a
# round-trip in latency-bound terminal data, and Textual's default
# 60 FPS is overkill once you pay an extra 30-150 ms RTT per frame.
# Override with `SHAREDAGENTS_TUI_FPS=<n>` if you want to force a value.
def _detect_ssh() -> bool:
    return bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT") or os.environ.get("SSH_TTY"))


def _detect_plain_keys() -> bool:
    """True when Enter is likely to arrive as a bare CR/LF (tmux, SSH).

    Those transports fragment kitty CSI-u reports and often deliver
    Enter as LF (``ctrl+j``). Treat them like SSH for keyboard policy.
    """
    if (os.environ.get("SHAREDAGENTS_TUI_PLAIN_KEYS") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return True
    return _detect_ssh() or bool(os.environ.get("TMUX"))


_IS_SSH = _detect_ssh()
_PLAIN_KEYS = _detect_plain_keys()
_TUI_FPS = int(os.environ.get("SHAREDAGENTS_TUI_FPS") or (20 if _IS_SSH else 60))

# Kitty keyboard protocol policy. `auto` (default) disables it over SSH —
# where fragmented multi-byte CSI-u key reports leak into the input box as
# literal text — and enables shift+enter disambiguation on a local TTY.
# Force with on|off. See `_enable_kitty_keyboard_enhancements`.
_KITTY_KEYBOARD_MODE = (os.environ.get("SHAREDAGENTS_KITTY_KEYBOARD") or "auto").strip().lower()


# Footer-spinner frames. Animated by `_animate_status` while
# `_worker_state` is non-empty so the operator sees motion even when the
# upstream pauses between chunks (e.g. right before a long tool call,
# or while a tool runs and no chunks are flowing). Braille frames are
# terminal-safe, low-bandwidth, and read at a glance.
_SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


# asyncio's StreamReader has a 64 KiB default per-line buffer
# (`asyncio.streams._DEFAULT_LIMIT`). Claude Code's stream-json output
# can emit single events > 64 KiB when a `tool_use` block carries a
# large input (e.g. a Read on a multi-page file, or a Write with a long
# document body). Same goes for the local `worker.py` when it emits a
# `tool_calls` event whose accumulated arguments are large. Bump the
# limit to 16 MiB so realistic agentic workloads never trip
# `LimitOverrunError` mid-stream.
_SUBPROCESS_STDIO_LIMIT = 16 * 1024 * 1024


def _alias_for_endpoint(
    models: list | None,
    endpoint: str,
    model_id: str = "",
) -> str:
    """Look up the configured alias for `endpoint` in the agent's
    `models[]` list. Returns "" when nothing matches (caller falls back
    to hostname). Trailing slashes are normalized so a session bound
    to `https://openrouter.ai/api/v1` matches a configured row whose
    `baseUrl` is the same with or without a trailing slash.

    When `model_id` is given, an endpoint that fronts SEVERAL model ids
    (e.g. the vLLM mux router on :8000) is disambiguated by also matching
    the id. Endpoint-only fallback is used ONLY when it can't mislead:
    either no `model_id` was supplied (back-compat), or exactly one row
    shares the endpoint (unambiguous). When an id WAS supplied but matched
    no row at a MULTI-row mux endpoint, return "" rather than guess a
    sibling's alias — the caller then falls back to the honest hostname.
    """
    if not endpoint or not models:
        return ""
    target = endpoint.rstrip("/")
    ep_aliases: list[str] = []  # aliases of every row sharing the endpoint
    for m in models:
        if (m.get("baseUrl") or "").rstrip("/") != target:
            continue
        if model_id and str(m.get("id") or "") == str(model_id):
            return m.get("alias") or ""
        ep_aliases.append(m.get("alias") or "")
    if not ep_aliases:
        return ""
    # Safe to name by endpoint alone only when there is nothing to
    # disambiguate (no id) or no ambiguity (single row on this endpoint).
    if not model_id or len(ep_aliases) == 1:
        return ep_aliases[0]
    return ""


def _hostname_of(endpoint: str) -> str:
    """Pull the hostname out of an upstream URL, e.g. `host` from
    `http://host:8001/v1`. Returns "" on a malformed / empty URL.

    Used as a fallback for the status bar's provider alias when the
    user hasn't set `AgentConfig.upstream_alias` explicitly.
    """
    if not endpoint:
        return ""
    try:
        from urllib.parse import urlparse

        return urlparse(endpoint).hostname or ""
    except Exception:  # noqa: BLE001
        return ""


def _truncate_smart(s: str, n: int, ellipsis: str = "…") -> str:
    """Truncate ``s`` to at most ``n`` characters at a word boundary.

    The transcript's tool-call preview, tool-result peek, and subagent-
    box summary all used to do ``s[:n] + "…"`` which cuts mid-word —
    operators saw ``shar…`` (mid-``sharedagents``) and ``independently
    researc…`` (mid-``research``) in the ``/goal`` fan-out. Walk back
    from ``n`` to the nearest whitespace within a 40-char window; fall
    back to the hard cut if no breakpoint exists (e.g. a 200-char JSON
    blob with no spaces). ``rstrip`` keeps trailing whitespace out of
    the displayed string before the ellipsis.
    """
    if len(s) <= n:
        return s
    window = min(40, n)
    for i in range(n, max(n - window, 0), -1):
        if s[i].isspace():
            return s[:i].rstrip() + ellipsis
    return s[:n].rstrip() + ellipsis


def _engine_models_for(agent_cfg) -> list[dict] | None:
    """Return the per-engine model menu for an agent, or ``None``.

    The command runtime is the single source of truth for all frontends.
    External engines reach their registry-registered discovery hook through
    the engine capability; local engines return ``None`` so callers fall back
    to the agent's configured upstream rows. Discovery may perform a
    TTL-cached network or CLI probe, so UI-thread callers should offload this
    helper via ``asyncio.to_thread``.
    """
    from sharedagents_command.commands import engine_model_rows

    return engine_model_rows(agent_cfg)


def _model_alias_id_label(alias: object = "", model_id: object = "") -> str:
    """Render ``alias (id)`` unless both values are the same."""
    alias_s = str(alias or "")
    model_s = str(model_id or "")
    if alias_s and model_s and alias_s != model_s:
        return f"{alias_s} ({model_s})"
    return alias_s or model_s


# In-band tool-use / tool-result markers the legacy claude/antigravity CLI
# runners inline into the assistant content body (see
# ``sharedagents_agent.tools.runners._format_claude_tool_use`` and
# ``_format_claude_tool_result``). When the saved turn carries
# structured ``tool_calls[]`` / ``tool_results[]`` lists we mount real
# ``ToolCallMessage`` / ``ToolResultMessage`` widgets and strip these
# markers from the prose so the operator doesn't see both.
#
# Marker shape (per runners.py):
#     "\n\n🔧 <name> <json-args>\n"
#     "↳ <summary> (<n> chars)\n"   (or "✗ ..." on error, or "↳ (<n> chars)\n")
_INLINE_TOOL_MARKER_RE = re.compile(
    r"\n*🔧 [^\n]*\n(?:[↳✗] [^\n]*\n)?",
)


# Wire-sentinel that the TUI prepends to a delegate-class slash command's
# user message so the bridge can intercept it before any LLM round (see
# ``agent_client.delegate()`` at agent_client.py:1979 and the bridge's
# ``_parse_direct_delegate_marker`` at acp_bridge.py:1326-1364). The
# sentinel is for the wire only — every renderer that displays the user
# message must humanise it via :func:`humanise_delegate_marker` first so
# the operator sees ``/<tool> <prompt>`` instead of the raw sentinel.
_DELEGATE_MARKER_PREFIX = "__SHAREDAGENTS_DELEGATE__:"


def humanise_delegate_marker(text: str) -> str:
    """Render an ``__SHAREDAGENTS_DELEGATE__:<tool>:<json>`` sentinel as the
    operator's actual slash invocation.

    Mapping mirrors the slash dispatchers:

    - ``delegate_claude``       → ``/claude <prompt>``
    - ``delegate_codex``        → ``/codex <prompt>``
    - ``delegate_antigravity``  → ``/antigravity <prompt>``
    - ``delegate_agent``        → ``/<name> <task>`` (matches the
      sibling-agent autodiscovery path at app.py:5138-5152 which
      builds the marker on a ``/<sibling> <task>`` slash command).
    - ``compact_context``       → ``/compact``
    - other well-formed tools   → ``/<tool> <prompt>`` (never the wire).

    Strings that don't start with the sentinel are returned unchanged
    (idempotent on already-humanised text). Malformed sentinels —
    truncated body, non-JSON args — fall back to returning the raw
    text so debugging the wire is still possible if a renderer leaks.
    """
    if not isinstance(text, str) or not text.startswith(_DELEGATE_MARKER_PREFIX):
        return text
    rest = text[len(_DELEGATE_MARKER_PREFIX) :]
    # Body shape: ``<tool>:<json-args>``. Split on the first colon so a
    # JSON value containing ``:`` (e.g. a URL) doesn't truncate the
    # parse.
    tool, sep, raw_args = rest.partition(":")
    if not sep:
        return text
    try:
        args = json.loads(raw_args)
    except (ValueError, TypeError):
        return text
    if not isinstance(args, dict):
        return text
    if tool == "delegate_claude":
        return f"/claude {str(args.get('prompt') or '').strip()}".rstrip()
    if tool == "delegate_codex":
        return f"/codex {str(args.get('prompt') or '').strip()}".rstrip()
    if tool == "delegate_antigravity":
        return f"/antigravity {str(args.get('prompt') or '').strip()}".rstrip()
    if tool == "delegate_agent":
        name = str(args.get("name") or "").strip()
        task = str(args.get("task") or args.get("prompt") or "").strip()
        if name:
            return f"/{name} {task}".rstrip()
        return f"/agent {task}".rstrip()
    if tool == "compact_context":
        return "/compact"
    extra = str(args.get("prompt") or args.get("task") or "").strip()
    if tool.startswith("delegate_"):
        name = tool[len("delegate_") :] or "agent"
        return f"/{name} {extra}".rstrip()
    return f"/{tool} {extra}".rstrip()


_TASK_SUMMARY_LINE_RE = re.compile(
    r"^#(?P<id>\d+)\s+\[(?P<status>[a-zA-Z_]+)\]\s+(?P<rest>.*)$",
)


def _parse_task_list_text(body: str) -> list[dict]:
    """Decode the plain-text rows emitted by the harness
    ``TaskList`` / ``TaskGet`` runner.

    Mirrors ``sharedagents_core/tools/runners.py:_task_summary`` —
    each row is ``#<id> [<status>] <subject>[ owner=<name>]
    [ blockedBy=<csv>]``. Returns a list of
    ``{"id", "subject", "status"}`` dicts the
    :class:`HarnessTaskListMessage`'s ``apply_tasklist_payload``
    consumes.

    Best-effort: malformed lines (a missing ``#<id>``, an
    unexpected status word) are skipped; the empty-store sentinel
    ``"(no tasks)"`` returns ``[]``. Idempotent on whitespace
    around lines.
    """
    if not body:
        return []
    body = body.strip()
    if body == "(no tasks)":
        return []
    out: list[dict] = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or not line.startswith("#"):
            continue
        m = _TASK_SUMMARY_LINE_RE.match(line)
        if m is None:
            continue
        rest = (m.group("rest") or "").strip()
        # Strip trailing ``owner=…`` / ``blockedBy=…`` segments so
        # ``subject`` carries just the human-readable text.
        for tag in (" owner=", " blockedBy="):
            idx = rest.find(tag)
            if idx >= 0:
                rest = rest[:idx].rstrip()
        out.append(
            {
                "id": m.group("id"),
                "subject": rest,
                "status": m.group("status"),
            }
        )
    return out


def _strip_inline_tool_markers(content: str) -> str:
    """Remove the in-band 🔧 / ↳ tool markers from a saved assistant body.

    Idempotent and safe on strings that don't carry any markers
    (returns the input unchanged). Collapses the runs of blank lines
    the markers leave behind so the remaining prose reads cleanly.
    """
    if not content or "🔧" not in content:
        return content
    cleaned = _INLINE_TOOL_MARKER_RE.sub("\n", content)
    # Collapse 3+ newlines to 2 (one blank line between paragraphs).
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip("\n")


def _build_delegate_metrics(
    *,
    label: str,
    t0: float,
    first_chunk_at: float | None,
    chars: int,
) -> dict | None:
    """Synthesize a native-style metrics row for a delegated turn.

    Mirrors what the bridge ships for local turns (``model``,
    ``ttft_ms``, ``tps``, ``completion_tokens``, ``wt_ms``) so the
    operator can compare a /claude / /codex run against a gpu-a-served
    one at a glance. The body of the delegate is opaque to us — we
    don't see prompt-token / cache-hit numbers, so those fields stay
    absent and the row is marked ``tokens_estimated`` (the ≈ glyph
    on ``gen``/``out``).

    Returns ``None`` when nothing useful can be reported (no chunks
    arrived, no wall time accumulated).
    """
    now = time.monotonic()
    wt_ms = max(0.0, (now - t0) * 1000.0)
    if wt_ms <= 0 and chars == 0:
        return None
    metrics: dict[str, Any] = {
        "model": label,
        "wt_ms": wt_ms,
        "tokens_estimated": True,
    }
    if first_chunk_at is not None:
        metrics["ttft_ms"] = (first_chunk_at - t0) * 1000.0
        if chars > 0:
            stream_s = max(now - first_chunk_at, 1e-3)
            est_tok = max(int(chars / 4), 1)
            metrics["completion_tokens"] = est_tok
            metrics["content_tokens"] = est_tok
            metrics["tps"] = est_tok / stream_s
    return metrics


def _gateway_url_and_token() -> tuple[str, str | None]:
    """Resolve the sharedagents gateway URL + bearer token for AgentClient.

    Order of precedence:

    1. ``$SHAREDAGENTS_TUI_ACP_URL`` / ``$SHAREDAGENTS_TUI_ACP_TOKEN`` — direct
       env override (also what :class:`AgentClient` reads as a default).
    2. ``~/.sharedagents/config.json`` ``gateway`` block (host/port/auth_token).
    3. Hard default ``http://127.0.0.1:8100`` with no auth.

    The TUI uses this once on startup to populate the ``gateway_url``
    + ``gateway_token`` kwargs on every :class:`AgentClient` it
    constructs, so all agent traffic flows through the systemd-managed
    gateway and we never spawn a per-agent stdio child from the TUI.

    Steps 2 + 3 delegate to :func:`sharedagents_gateway.gateway._gateway_config`
    — the gateway's OWN resolver — so the TUI can never disagree with
    the port the gateway actually binds. Hand-rolling this resolution
    with a stale ``8700`` default is exactly what stranded a fresh
    install with "connection refused": the gateway bound :8100, the
    TUI dialed :8700.
    """
    env_url = os.environ.get("SHAREDAGENTS_TUI_ACP_URL", "").strip()
    env_token = os.environ.get("SHAREDAGENTS_TUI_ACP_TOKEN", "").strip() or None
    if env_url:
        return env_url.rstrip("/"), env_token
    try:
        from sharedagents_gateway.gateway import _gateway_config

        gw = _gateway_config()
    except Exception:  # noqa: BLE001
        gw = {"host": "127.0.0.1", "port": 8100, "auth_token": None}
    return f"http://{gw['host']}:{gw['port']}", (gw.get("auth_token") or None)


def _gateway_alive() -> bool:
    """True iff the sharedagents gateway is reachable.

    Primary signal: ``$SHAREDAGENTS_HOME/gateway.pid`` points at a live
    pid. Fallback: a successful ``GET /health`` against the gateway's
    bind URL — covers the case where the pidfile is missing (e.g.
    early in a systemd restart, or a stale-cleanup race) but the
    listener is healthy.
    """
    try:
        from sharedagents_gateway.gateway import _gateway_pidfile
    except Exception:  # noqa: BLE001
        return False
    pidfile = _gateway_pidfile()
    if pidfile.is_file():
        try:
            pid = int(pidfile.read_text().strip())
            os.kill(pid, 0)
            return True
        except (OSError, ValueError, PermissionError, ProcessLookupError):
            pass
    gateway_url, _ = _gateway_url_and_token()
    try:
        r = httpx.get(f"{gateway_url}/health", timeout=1.0)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


# Thinking-title elapsed-time tick. Local: 5 Hz feels live. SSH: 2 Hz
# is plenty (humans don't read sub-half-second updates on a header
# anyway) and saves bandwidth during the multi-second think phase.
_THINK_TICK_INTERVAL_S = 0.5 if _IS_SSH else 0.2


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _enable_kitty_keyboard_enhancements(app: App) -> None:
    """Tune the kitty keyboard protocol for the current transport.

    Textual's Linux/Windows drivers write `\\x1b[>1u` to push flag 1
    (disambiguate escape codes) onto the kitty keyboard stack. On a
    LOCAL TTY we add flags 2/4 (report event types / alternate keys)
    via `\\x1b[>7u` so terminals that fully implement the protocol
    deliver `shift+enter` as a distinct key event instead of collapsing
    it into plain `enter`.

    PREVIOUSLY this wrote `\\x1b[=8;u`, which is SET-mode (mode `1`
    being the default) — that REPLACED Textual's flag 1 with 8 alone,
    nuking disambiguation, so `shift+enter` started arriving as plain
    `enter` and the input box submitted instead of inserting a newline.
    A later revision pushed `\\x1b[>15u` (flags 1+2+4+8); flag 8
    ("report *all* keys as escape codes") made every plain letter arrive
    as a CSI sequence and, under load (session replay / scroll), timed-
    out partial sequences were reissued as literal ``[``/``u`` garbage
    in the input box. `\\x1b[>7u` (push 7 = 1+2+4) stopped that for
    unmodified letters, but flag 1 (which Textual itself pushes) still
    encodes Esc/Backspace/Enter as multi-byte CSI-u reports.

    OVER SSH that is the remaining failure mode: the transport fragments
    those multi-byte sequences, and under load the event loop is starved
    long enough that Textual's escape-sequence parser times out
    mid-sequence — consuming the lone `\\x1b` as a bare Escape and
    flushing the tail (``[27;1;3u``, ``[127;1;2u``, …) into the input
    box as literal text. Pressing Backspace/Escape to clear it emits
    more CSI-u, so the garbage cascades and typing is impossible.

    So over SSH (or when forced off) we push `\\x1b[>0u` — flags=0,
    which DISABLES every enhancement including Textual's disambiguation.
    Special keys then revert to legacy single-byte encodings (`\\x1b`,
    `\\x7f`) that cannot fragment. `shift+enter` is no longer
    distinguishable, but `MessageInput.BINDINGS` includes a `ctrl+j`
    fallback (literal LF) that any terminal can deliver, so multi-line
    composition still works.

    Override the transport heuristic with `SHAREDAGENTS_KITTY_KEYBOARD`:
    ``on`` forces `>7u`, ``off`` forces `>0u`, ``auto`` (default) picks
    `>0u` on SSH and `>7u` locally. On terminals that don't speak the
    protocol at all the sequence is ignored. Textual's own teardown
    (`\\x1b[<u`) pops the stack on exit.

    Tracking upstream: https://github.com/Textualize/textual/issues/6074
    """
    driver = getattr(app, "_driver", None)
    if driver is None:
        return
    write = getattr(driver, "write", None)
    flush = getattr(driver, "flush", None)
    if write is None or flush is None:
        return
    # Resolve SSH state through the app facade at call time so a test patching
    # ``sharedagents_tui.app._IS_SSH`` still reaches this relocated helper (pre-split,
    # the read lived in app.py). Falls back to the local module value.
    from sharedagents_tui import app as _app_mod

    _is_ssh = getattr(_app_mod, "_IS_SSH", _IS_SSH)
    _plain = getattr(_app_mod, "_PLAIN_KEYS", _PLAIN_KEYS)
    if _KITTY_KEYBOARD_MODE == "off" or (_KITTY_KEYBOARD_MODE != "on" and (_is_ssh or _plain)):
        seq = "\x1b[>0u"  # disable kitty enhancements; legacy keys can't fragment
    else:
        seq = "\x1b[>7u"  # flags 1+2+4: shift+enter disambiguation on a local TTY
    try:
        write(seq)
        flush()
    except Exception:  # noqa: BLE001 — never crash boot over a TTY quirk
        pass


async def discover_model(endpoint: str, headers: dict | None = None) -> str:
    async with httpx.AsyncClient(timeout=10, headers=headers or {}) as client:
        r = await client.get(f"{endpoint.rstrip('/')}/models")
        r.raise_for_status()
        data = r.json().get("data") or []
    if not data:
        raise RuntimeError(f"no models advertised by {endpoint}")
    return data[0]["id"]


def _build_metrics_markup(metrics: dict) -> str:
    """Build the one-line dim telemetry footer markup from a metrics dict.

    Shared by :meth:`AssistantMessage.set_metrics` (inline, end-of-turn
    cumulative + replay) and :class:`RoundMetricsMessage` (the standalone
    per-round line mounted AFTER a round's tool widgets). Returns ``""``
    when no renderable field is present.

    Order (left → right):
        <model> · classify <ms> · plan <ms>
          · prefill <ms> @ <prefill_tps> tok/s · ttft <ms>
          · gen <gen_tps> tok/s
          · ctx in <pt> (<cached> cached, <fresh> fresh) /<window>
          · out <ct> tok (think <rt> / result <ot>)
          · total <total> tok · wt <s>
    """
    # Field selection / ordering / number formatting lives in the shared
    # ``sharedagents_agent.metrics_format`` so the TUI, Telegram and Web stats
    # rows can never drift (the sharedagents "identical rendering" property).
    # The TUI's only job is to wrap the plain segments in Rich dim markup —
    # the historical ``[dim]─ a · b · c[/dim]`` shape, byte-identical to
    # the pre-extraction inline builder.
    line = _metrics_line(metrics)
    if not line:
        return ""
    return f"[dim]{line}[/dim]"


CLAUDE_MODE_MAP: dict[str, str] = {
    "plan": "plan",  # claude plan: research-only, doesn't edit
    "ask": "default",  # claude default: prompts; in -p mode denies non-allowed
    "auto": "bypassPermissions",  # claude bypassPermissions: full bypass (Edit/Write/Bash/...)
}

# Display-only mapping for Codex. The actual flags are picked inside the
# `delegate_codex` runner; this is just the human-readable label the
# status bar shows ("mode=read-only", "mode=bypass") so the user knows
# what permission level Codex is running with.
CODEX_MODE_LABELS: dict[str, str] = {
    "plan": "read-only",
    "ask": "workspace-write",
    "auto": "bypass",
}


def _codex_sandbox_label(sharedagents_mode: str) -> str:
    """Map sharedagents's plan/ask/auto to a short Codex sandbox label.

    Used by the TUI / bash bridge to render the active permission level
    in status messages. Falls back to `workspace-write` (the codex
    default) when the mode is unknown — mirrors `_claude_permission_mode`'s
    fallback to claude `default`.
    """
    if not sharedagents_mode:
        return "workspace-write"
    return CODEX_MODE_LABELS.get(sharedagents_mode, "workspace-write")


def _claude_permission_mode(sharedagents_mode: str) -> str:
    """Map sharedagents's plan/ask/auto to Claude Code's --permission-mode.

    `auto` maps to `bypassPermissions`: in headless `-p` mode there is
    no human to answer permission prompts, so `acceptEdits` (which only
    auto-approves Edit) leaves Bash/Write/etc. deadlocked. The user
    opts into "do whatever you need" by cycling sharedagents to `auto`, so
    bypass is the right Claude analogue.

    Unknown / empty modes fall back to claude's `default` (asks for
    everything) — safest landing for a None mode coming through the CLI
    when no `--mode` was passed.
    """
    if not sharedagents_mode:
        return "default"
    return CLAUDE_MODE_MAP.get(sharedagents_mode, "default")


def _default_tool_result_collapsed(content: str, *, was_streaming: bool) -> bool:
    """Shared collapse decision for a tool result, used by BOTH the live
    render and the replay/render so a passive frontend's default
    collapse state matches the driver exactly. A result that streamed
    live starts collapsed (the live widget auto-collapsed it); otherwise
    only long outputs fold away. (turn-replication: exact parity)
    """
    c = content or ""
    return was_streaming or (c.count("\n") + 1 > 12) or len(c) > 800


# Map from runtime-warning ``event`` slug (the first field
# `_format_runtime_warning` writes) to the human-readable headline shown
# in the transcript banner. Keeping it explicit means the operator sees
# wording tuned to the recovery being taken instead of a raw kebab-case
# event name. Unknown slugs fall back to the slug itself plus the raw
# fields, so a new bridge-side warning doesn't silently disappear.
_RUNTIME_WARNING_HEADLINES: dict[str, str] = {
    "loop-detected": "loop detected — pausing worker",
    "loop-distill": "distill recovery — resuming with summary",
    "loop-force-terminate": "loop force-terminate — turn aborted",
    "post-force-terminate-deny": "post-force-terminate — tool calls denied",
    "nudge": "premature finish — nudging model to continue",
}


# Prefix-sniff helpers for the chunk dispatcher. Kept at module scope so
# the dispatch loop's hot path doesn't re-allocate the prefix tuples on
# every chunk.
_RUNTIME_WARNING_GLYPHS = ("⚠", "↻")
_RUNTIME_WARNING_PREFIX = " runtime: "


def _parse_runtime_warning_chunk(
    text: str,
) -> tuple[str, str, dict[str, str]] | None:
    """Parse a chunk body that *might* be a ``_format_runtime_warning``
    line. Returns ``(glyph, event, fields_dict)`` on a hit; ``None``
    otherwise — in which case the caller should treat the chunk as
    normal model output.

    The bridge formats warnings exactly as ``\\n\\n{glyph} runtime:
    {event} · k1=v1 · k2=…\\n\\n`` (single chunk per warning, see
    acp_bridge.py L1153). We accept leading/trailing whitespace and
    only commit if both the glyph and the ``runtime:`` token are
    present, so legitimate model output that happens to contain a
    similar substring deeper in the chunk won't be misclassified.
    """
    if not text:
        return None
    stripped = text.lstrip()
    if not stripped or stripped[0] not in _RUNTIME_WARNING_GLYPHS:
        return None
    if _RUNTIME_WARNING_PREFIX not in stripped[:32]:
        return None
    glyph = stripped[0]
    body = stripped.split("\n", 1)[0].rstrip()
    parts = [p.strip() for p in body.split(" · ") if p.strip()]
    if not parts:
        return None
    # First piece is "<glyph> runtime: <event>" — slice off the prefix
    # to land on just the event slug.
    head = parts[0]
    if " runtime: " not in head:
        return None
    event = head.split(" runtime: ", 1)[1].strip()
    fields: dict[str, str] = {}
    for piece in parts[1:]:
        if "=" not in piece:
            continue
        k, _, v = piece.partition("=")
        k = k.strip()
        v = v.strip()
        if k:
            fields[k] = v
    return glyph, event, fields


def _humanize_age(ts: str) -> str:
    """Render an ISO-8601 timestamp as `"X min ago"` / `"X hour(s) ago"` /
    `"X day(s) ago"` / `"just now"`. Returns `"?"` on a parse failure so
    the picker stays usable even with malformed session JSON.
    """
    if not ts:
        return "?"
    try:
        # `fromisoformat` accepts the ISO-8601 we save (`+00:00` offset).
        from datetime import datetime, timezone

        when = datetime.fromisoformat(ts)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        delta_s = (datetime.now(timezone.utc) - when).total_seconds()
    except (ValueError, TypeError):
        return "?"
    if delta_s < 0:
        # Future timestamps shouldn't happen in normal flow, but a clock
        # skew between agent host and TUI host could put us here. Fall
        # back to "just now" rather than rendering a negative age.
        return "just now"
    if delta_s < 45:
        return "just now"
    if delta_s < 60 * 90:
        return f"{int(delta_s / 60)} min ago"
    if delta_s < 60 * 60 * 36:
        return f"{int(delta_s / 3600)} hour" + ("s ago" if delta_s >= 7200 else " ago")
    days = int(delta_s / 86400)
    return f"{days} day" + ("s ago" if days != 1 else " ago")


def _rows_signature(rows: list[dict]) -> tuple:
    """Stable signature for the visible state of the shells list. Two
    refreshes that produce the same signature render the same widget
    output, so the modal can skip the clear+rebuild — that's what
    stops the periodic refresh from flickering the highlighted row.

    Mirror the fields ``MonitorPanelModal._build_items`` actually
    displays (alive, pid, monitor_id, command) — fields the modal
    doesn't render (log_path, env, …) don't belong in the signature
    because they'd cause spurious repaints on backend churn that
    isn't visible to the operator anyway.
    """
    return tuple(
        (
            bool(r.get("alive")),
            r.get("pid"),
            r.get("monitor_id"),
            r.get("command"),
        )
        for r in rows
    )


def _reconcile_subagent_sidecars(
    sessions_dir: Any,
    sid: str,
    reason: str = "aborted",
) -> list[str]:
    """Flip orphaned subagent sidecars out of ``status == "running"``.

    The per-call result write (:meth:`ChatApp._persist_subagent_result`)
    only fires when the matching ``tool_result`` is processed by the live
    TUI *while the box is still tracked in* ``_subagent_boxes_by_call``.
    A process restart, a reattach from another frontend, or a hard turn
    abort can drop that path — leaving ``<sid>.subagent.<call_id>.json``
    stuck at ``"running"`` forever, since no other code path reconciles
    it. The subagent-navigation panel then shows a phantom run that has
    actually been dead for hours.

    This sweep — run at end-of-turn and on session attach — rewrites any
    still-``"running"`` sidecar for ``sid`` with a terminal ``status``
    (``reason``), ``ok=False`` and ``ts_end`` / ``elapsed_s`` so the run
    reads as finished. Best-effort, atomic, and idempotent (a sidecar
    already in a terminal state is skipped). Returns the reconciled
    call_ids for logging / tests.
    """
    if not sid:
        return []
    try:
        sdir = Path(sessions_dir)
    except Exception:  # noqa: BLE001
        return []
    reconciled: list[str] = []
    ts = time.time()
    ts_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))
    for path in sorted(sdir.glob(f"{sid}.subagent.*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(payload, dict) or payload.get("status") != "running":
            continue
        start = payload.get("ts_start") or ts
        payload["ts_end"] = ts
        payload["ts_end_iso"] = ts_iso
        try:
            payload["elapsed_s"] = round(float(ts) - float(start), 3)
        except (TypeError, ValueError):
            payload["elapsed_s"] = 0.0
        payload["ok"] = False
        payload["status"] = reason
        payload.setdefault(
            "result",
            f"subagent run reconciled as {reason}: no terminal "
            "tool_result was recorded (turn ended or session reattached "
            "before completion).",
        )
        try:
            tmp = path.with_name(f".{path.name}.tmp")
            tmp.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            tmp.replace(path)
            reconciled.append(str(payload.get("call_id") or path.stem))
        except Exception:  # noqa: BLE001
            pass
    return reconciled


def user_prompt_text(user, attachment_names=()) -> str:
    """Render backend-projected user text and file names in the terminal."""
    if isinstance(user, list):
        text = "\n".join(
            str(part.get("text") or "") for part in user if isinstance(part, dict) and part.get("type") == "text"
        )
    else:
        text = str(user or "")
    return "\n".join(
        part
        for part in (
            humanise_delegate_marker(text),
            "\n".join(f"📎 {name}" for name in attachment_names),
        )
        if part
    )
