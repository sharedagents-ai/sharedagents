"""Claude Code–compatible hook engine.

sharedagents's hooks are *exactly* Claude Code's hooks — same config
shape, same event names, same stdin payloads, same exit-code
semantics, same JSON-stdout control protocol. A ``hooks.json`` written
for Claude Code (or a Claude Code plugin such as
everything-claude-code) loads and runs here unchanged.

## Config shape

Loaded from a JSON file (the agent's ``hooks_file``, default
``<agent_home>/hooks.json``). Either the Claude Code *settings* shape
(an object with a top-level ``"hooks"`` key) or a bare event map is
accepted — unknown top-level keys (``$schema`` …) are ignored::

    {
      "hooks": {
        "PreToolUse": [
          {
            "matcher": "Bash",
            "hooks": [
              { "type": "command", "command": "...", "timeout": 60 }
            ]
          }
        ],
        ...
      }
    }

Recognised events (:data:`HOOK_EVENTS`): the full Claude Code set —
``PreToolUse``, ``PostToolUse``, ``PostToolUseFailure``,
``UserPromptSubmit``, ``Notification``, ``Stop``, ``SubagentStop``,
``PreCompact``, ``SessionStart``, ``SessionEnd`` — plus sharedagents
turn-runtime extensions that expose hardcoded recovery policies as
operator-overridable hooks: ``ReasoningLoopDetected`` (fires when the
streaming or token-ratio loop detector trips, before distill /
force-terminate fires), ``PrematureTurnEnd`` (fires when the model
emits ``done`` but the round looks premature — silent / cutoff /
colon-cliffhanger — before the hardcoded nudge body is injected),
``DistillRecovery`` (fires before a distill round runs so a hook can
veto / substitute a recovery body), and ``ForceTerminate`` (fires
before the last-ditch force-terminate body is injected). Each
extension event uses the **exact same** stdin/stdout/exit-code
protocol as the standard Claude Code events — an ECC plugin can
target them with a plain ``hooks.json`` entry without learning any
sharedagents-specific machinery.

``matcher`` is a regex ``re.fullmatch``'d against the event's match
target — the tool name for tool events, ``source`` for
``SessionStart``, ``trigger`` for ``PreCompact``, the loop ``tier``
for ``ReasoningLoopDetected``, the premature ``kind`` for
``PrematureTurnEnd``, ``""`` otherwise.
``"*"``, ``""``, or a missing matcher matches everything. Each
group's ``hooks`` list holds hook entries; the spec defines five
types but sharedagents executes two today:

* ``{"type": "command", "command": <shell string>}`` — shell-exec,
  pipes the stdin payload, parses exit code + stdout (the full
  spec behaviour).
* ``{"type": "http", "url": <url>, "http": {"method": "POST",
  "headers": {...}}}`` — POSTs the stdin JSON to the URL. HTTP 2xx
  becomes exit-0 (response body as stdout, parsed as a control
  block if it starts with ``{``); HTTP 4xx/5xx becomes exit-1
  (non-blocking). Network/timeout errors land as spawn failures.
* ``{"type": "mcp_tool" | "prompt" | "agent", ...}`` — recognised
  by the loader so they show up in ``HookEngine.events()`` /
  ``hook_count()`` (which the operator inspects via ``/doctor``)
  but skipped at fire time with a single WARN per hook. Strict
  improvement over the previous silent-drop behaviour.

Optional ``timeout`` (seconds, default 60) and ``async``
(fire-and-forget — the runtime doesn't wait for or read the result,
matching Claude Code).

## Runtime contract

:meth:`HookEngine.run` builds the Claude Code stdin JSON (common
fields ``session_id`` / ``transcript_path`` / ``cwd`` /
``hook_event_name`` / ``permission_mode`` plus the per-event fields
the caller passes in ``payload``), runs every matching command hook,
and folds the results into a :class:`HookOutcome`.

## Exit-code semantics (exactly Claude Code)

* **0** — success. ``stdout`` is surfaced; for ``UserPromptSubmit``
  and ``SessionStart`` it is added to the model's context, for other
  events it is transcript-only.
* **2** — blocking error. ``stderr`` is fed back to the model. For
  ``PreToolUse`` it denies the tool; for ``UserPromptSubmit`` it
  blocks the prompt; for ``Stop`` / ``SubagentStop`` it blocks
  stopping; for ``PostToolUse`` it surfaces the message to the model.
* **other** — non-blocking error. ``stderr`` goes to the user/log;
  the turn continues.

## JSON stdout control protocol

If a hook's ``stdout`` parses as a JSON object, these fields are
honoured (a superset folded across every matching hook):

* ``continue`` (bool) — ``false`` halts the agent after hooks run.
* ``stopReason`` (str) — shown to the user when ``continue`` is false.
* ``suppressOutput`` (bool) — keep ``stdout`` out of the transcript.
* ``systemMessage`` (str) — a warning surfaced to the user.
* ``decision`` (``"block"``) + ``reason`` — block the event, feed
  ``reason`` to the model (``PostToolUse`` / ``Stop`` /
  ``SubagentStop`` / ``UserPromptSubmit``). ``"approve"`` is the
  deprecated Claude Code alias for a ``PreToolUse`` allow.
* ``hookSpecificOutput``:
  * ``PreToolUse``: ``permissionDecision`` (``allow`` / ``deny`` /
    ``ask``) + ``permissionDecisionReason``.
  * ``UserPromptSubmit`` / ``SessionStart`` / ``PostToolUse``:
    ``additionalContext`` — text injected into the model's context.

## Shell-execution security note

A hook's ``command`` is operator-controlled config (it ships in the
agent's ``hooks.json`` or a vetted plugin bundle) and is executed via
``/bin/sh``. The LLM-controlled tool args reach the hook ONLY as a
JSON document on **stdin** — never spliced into the command string —
so a hostile ``tool_input`` cannot break out into shell
metacharacters. This matches Claude Code's model: trust the hook
config, treat the event payload as data.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sharedagents_core import oscompat
from sharedagents_core.cwdutil import safe_getcwd

_log = logging.getLogger("sharedagents.hooks")


# The Claude Code hook lifecycle events sharedagents recognises. An entry
# under any other key in a `hooks.json` is ignored (forward-compatible
# with future Claude Code events without crashing the loader).
#
# The first block is the Claude Code spec set. The second block is
# sharedagents turn-runtime extensions that expose hardcoded recovery
# policies as operator-overridable hooks; they share the standard
# stdin/stdout/exit-code protocol so an ECC plugin's ``hooks.json``
# can target them with no sharedagents-specific shape.
HOOK_EVENTS: frozenset[str] = frozenset(
    {
        # --- Claude Code standard events ----------------------------------
        "PreToolUse",
        "PostToolUse",
        "PostToolUseFailure",
        "PostToolBatch",
        "UserPromptSubmit",
        "UserPromptExpansion",
        "Notification",
        "Stop",
        "StopFailure",
        "SubagentStart",
        "SubagentStop",
        "PreCompact",
        "SessionStart",
        "SessionEnd",
        "Setup",
        "PermissionRequest",
        "PermissionDenied",
        "Elicitation",
        "ElicitationResult",
        "InstructionsLoaded",
        # --- sharedagents turn-runtime extension events ------------------------
        # Each fires at the decision point of a previously hardcoded
        # policy (loop recovery, premature-end nudging, distill, force-
        # terminate); the hardcoded body / behaviour is the FALLBACK when
        # no hook expresses an opinion.
        "ReasoningLoopDetected",
        "PrematureTurnEnd",
        "DistillRecovery",
        "ForceTerminate",
        # Pre-decision hook for the premature-end gate itself — fires
        # BEFORE ``_should_nudge`` is consulted so a plugin can enable
        # nudges the hardcoded gate would suppress (e.g. enable a nudge
        # when ``tool_rounds_so_far == 0`` for a chat-only model).
        "PrematureTurnEndGate",
        # Per-block preamble injection control. Fires once per built-in
        # injector (pinned_facts / compaction_summary / current_todos /
        # skills_preamble / memory_recall) BEFORE the block is appended
        # to the preamble. A ``decision: block`` from a hook suppresses
        # the block; a ``reason`` substitutes the block body. Lets ECC
        # plugins replace sharedagents's hardcoded preamble injectors instead
        # of only wrapping them.
        "PreambleInject",
        # Upstream-error classifier override — fires when a worker error
        # arrives and ``_classify_upstream_error`` has produced a default
        # tag (``ctx_overflow`` / ``rate_limited`` / ``upstream_error``).
        # A plugin can return ``hookSpecificOutput.classification`` to
        # override the tag, which steers the engine's reactive-compact
        # vs hard-error retry decision.
        "UpstreamError",
        # Synthetic event fired by sharedagents_core.errors.log_error() after
        # a structured error file is written under ~/.sharedagents/errors/.
        # Lets the self-improver react to captured exceptions the same
        # way it reacts to soft-failure events (ReasoningLoopDetected
        # etc.). Payload: {component, exception, fingerprint, file_path,
        # session_id}. Not block-capable — error is already logged.
        "SessionError",
    }
)

# Events where a hook can actually *block* the operation (exit 2 /
# `decision: block`). For every other event exit-2 is just "surface
# stderr" — no block semantics — so we do NOT flip `outcome.blocked`
# there (it would be a misleading signal to a generic caller). This
# mirrors Claude Code's event-specific exit-2 behaviour.
#
# sharedagents extension events are block-capable too: a ``decision:
# block`` from a ``ReasoningLoopDetected`` hook suppresses the
# detection (treat as false positive), from ``PrematureTurnEnd``
# supplies a custom nudge body, from ``DistillRecovery`` /
# ``ForceTerminate`` substitutes the canned body. The turn runtime
# reads ``block_reason`` to pull out the operator's text.
_BLOCK_CAPABLE_EVENTS: frozenset[str] = frozenset(
    {
        "PreToolUse",
        "PostToolUse",
        "UserPromptSubmit",
        "UserPromptExpansion",
        "Stop",
        "SubagentStop",
        "PermissionRequest",
        "Elicitation",
        "ReasoningLoopDetected",
        "PrematureTurnEnd",
        "PrematureTurnEndGate",
        "DistillRecovery",
        "ForceTerminate",
        "PreambleInject",
        "UpstreamError",
        # PreCompact carries operator-overridable substitution: a hook
        # can return a replacement message list via
        # ``hookSpecificOutput.replacementMessages``. The substitution
        # happens via ``has_effect``, but a ``decision: block`` is also
        # honoured to short-circuit compaction.
        "PreCompact",
    }
)

# Events whose plain (non-JSON, exit-0) stdout is added to the model's
# context instead of just the transcript — Claude Code's behaviour.
# (PostToolUse plain stdout is transcript-only; it reaches the model
# ONLY via the JSON `additionalContext` field — see below.)
_CONTEXT_STDOUT_EVENTS: frozenset[str] = frozenset(
    {
        "UserPromptSubmit",
        "UserPromptExpansion",
        "SessionStart",
        "Setup",
        "InstructionsLoaded",
    }
)

# Events that honour `hookSpecificOutput.additionalContext` (text the
# hook injects into the model's context). A superset of the
# plain-stdout set — PostToolUse can inject context via the JSON
# field but not via bare stdout.
_ADDITIONAL_CONTEXT_EVENTS: frozenset[str] = frozenset(
    {
        "UserPromptSubmit",
        "UserPromptExpansion",
        "SessionStart",
        "Setup",
        "InstructionsLoaded",
        "PostToolUse",
    }
)

# Claude Code's default per-hook timeout.
_DEFAULT_TIMEOUT_S: float = 60.0


# --------------------------------------------------------------------------
# Parsed-config primitives
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _CommandHook:
    """One ``{"type": "command", "command": ...}`` entry, carrying the
    matcher of the group it was declared under plus the plugin /
    config root the hook should run with.

    ``hook_type`` distinguishes the (currently three) supported hook
    types — ``command`` (shell), ``http`` (POST to ``command`` as a
    URL), and ``unsupported`` (parsed-and-recorded but skipped at
    fire time with a one-line warning, so ``/hooks`` / ``/doctor``
    can surface the issue instead of silently dropping the entry).
    The Claude Code spec defines ``mcp_tool`` / ``prompt`` / ``agent``
    types too; sharedagents recognises them in the loader (so the operator
    sees they're declared) but doesn't yet execute them.
    """

    event: str
    matcher: str  # raw matcher string ("" / "*" => match all)
    command: str
    timeout_s: float
    is_async: bool
    hook_id: str  # the group's `id` (or a synthesized one) + index
    hook_type: str = "command"  # "command" | "http" | "unsupported"
    # Per-hook ``CLAUDE_PLUGIN_ROOT``. Set when a hooks.json comes from
    # a plugin bundle (so plugin hooks resolve their plugin dir as
    # CLAUDE_PLUGIN_ROOT, not the agent's). "" => fall through to the
    # caller-supplied ``plugin_root`` argument on :meth:`HookEngine.run`.
    plugin_root: str = ""
    # For ``http`` hooks, the HTTP method + extra headers (declared via
    # ``"http": {"method": "POST", "headers": {...}}`` in the hook
    # entry; defaults to POST). Empty for non-http hooks.
    http_method: str = "POST"
    http_headers: tuple[tuple[str, str], ...] = ()


@dataclass
class HookCommandResult:
    """One command hook's raw result + parsed JSON control block."""

    hook_id: str
    command: str
    exit_code: int | None  # None => spawn failure / timeout
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    error: str = ""  # spawn/timeout error string, if any
    json_output: dict | None = None  # parsed stdout, if it was a JSON object


@dataclass
class HookOutcome:
    """Folded result of every matching hook for one event dispatch.

    The runtime reads this to decide what to do — block a tool, inject
    context, halt the agent, surface a warning. An engine with no
    matching hooks returns an all-defaults outcome (``ran == 0``,
    nothing blocked) so callers can treat "hooks disabled" and "hooks
    ran, no effect" identically.
    """

    event: str
    ran: int = 0
    # A hook blocked the event (exit 2, `decision: block`, or
    # `permissionDecision: deny`). `block_reason` is fed to the model.
    blocked: bool = False
    block_reason: str = ""
    # PreToolUse permission control. "" => no hook expressed an opinion;
    # the caller falls back to its normal permission policy.
    permission_decision: str = ""  # "" | "allow" | "deny" | "ask"
    permission_reason: str = ""
    # Text a hook asked to inject into the model's context
    # (UserPromptSubmit / SessionStart / PostToolUse additionalContext,
    # or plain stdout from a context-stdout event).
    additional_context: str = ""
    # `continue: false` => the agent should stop after hooks run.
    should_continue: bool = True
    stop_reason: str = ""
    suppress_output: bool = False
    # `systemMessage`s — operator-facing warnings.
    system_messages: list[str] = field(default_factory=list)
    # Non-suppressed plain stdout, for the transcript / debug surface.
    stdout_blocks: list[str] = field(default_factory=list)
    # Per-hook raw results, for the session events audit log.
    results: list[HookCommandResult] = field(default_factory=list)
    # Claude Code ``hookSpecificOutput.updatedInput`` — a PreToolUse
    # hook can rewrite the model's tool args before dispatch. ``None``
    # means no hook touched the input. The LAST hook to set it wins
    # (Claude Code semantics: linter-style mutators are chained by the
    # operator in declaration order). The turn-runtime caller merges
    # this into the live ``args`` before the permission gate so the
    # downstream criticality check sees the rewritten command.
    updated_input: dict[str, Any] | None = None

    @property
    def has_effect(self) -> bool:
        """True when the outcome carries anything the caller must act on."""
        return bool(
            self.blocked
            or not self.should_continue
            or self.additional_context
            or self.permission_decision
            or self.system_messages
            or self.updated_input is not None
        )

    def audit(self) -> list[dict[str, Any]]:
        """Compact per-hook audit dicts for the session ``events`` log."""
        out: list[dict[str, Any]] = []
        for r in self.results:
            rec: dict[str, Any] = {
                "hook": r.hook_id,
                "event": self.event,
                "rc": r.exit_code,
            }
            if r.timed_out:
                rec["timeout"] = True
            if r.error:
                rec["error"] = r.error
            if r.stdout:
                rec["stdout_chars"] = len(r.stdout)
            if r.stderr:
                rec["stderr_tail"] = r.stderr.strip()[-200:]
            if r.json_output is not None:
                rec["json"] = True
            out.append(rec)
        return out


async def fire_hook_event(
    hook_engine: Any,
    hook_ctx: dict[str, Any] | None,
    event: str,
    payload: dict[str, Any],
) -> HookOutcome:
    """Run one hook event; never raises. Shared by every engine harness."""
    if hook_engine is None or not hasattr(hook_engine, "has_event"):
        return HookOutcome(event=event)
    if not hook_engine.has_event(event):
        return HookOutcome(event=event)
    try:
        return await hook_engine.run(event, payload, **(hook_ctx or {}))
    except Exception:  # noqa: BLE001
        return HookOutcome(event=event)


# --------------------------------------------------------------------------
# Folding helpers
# --------------------------------------------------------------------------


def _join_reason(a: str, b: str) -> str:
    """Concatenate two block reasons (newline-separated, drop blanks)."""
    a, b = (a or "").strip(), (b or "").strip()
    return f"{a}\n{b}" if a and b else (a or b)


def _join_context(a: str, b: str) -> str:
    """Concatenate two context blocks (blank-line-separated)."""
    a, b = (a or "").rstrip(), (b or "").strip()
    return f"{a}\n\n{b}" if a and b else (a or b)


# Permission strength ordering — when several PreToolUse hooks fire,
# the strongest opinion wins (deny > ask > allow), matching Claude
# Code's "any deny denies" behaviour.
_PERM_RANK = {"": 0, "allow": 1, "ask": 2, "deny": 3}


def _strongest_perm(a: str, b: str) -> str:
    return a if _PERM_RANK.get(a, 0) >= _PERM_RANK.get(b, 0) else b


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------


class HookEngine:
    """In-memory Claude Code hook table for one agent.

    Loaded once at agent/backend construction from the agent's
    ``hooks.json``. The table is small (a handful of groups per
    event) so per-event linear scan is fine. Reload-on-edit is NOT
    supported on purpose — hot-swapping a security gate mid-session is
    a footgun; edit ``hooks.json`` and restart the agent.
    """

    def __init__(self, by_event: dict[str, list[_CommandHook]] | None = None) -> None:
        self._by_event: dict[str, list[_CommandHook]] = by_event or {}
        # In-flight ``async: true`` hook tasks. Tracked so graceful
        # shutdown can drain them (and so the per-spawn add/discard
        # callback dance doesn't lose its anchor).
        self._async_tasks: set[asyncio.Task[Any]] = set()

    # ---- construction ----------------------------------------------------

    @classmethod
    def load(cls, path: Path | None) -> HookEngine:
        """Load from ``path``; return an empty engine if absent/broken.

        A missing file, unreadable file, malformed JSON, or non-object
        root all yield an empty engine (logged, never raised) — a
        broken ``hooks.json`` must never brick the agent at startup.

        The plugin root for each loaded hook is derived from the file
        location (``<root>/hooks/hooks.json`` → ``<root>``; plain
        ``<root>/hooks.json`` → ``<root>``) so plugin hooks see the
        right ``CLAUDE_PLUGIN_ROOT`` even after :meth:`merge` combines
        them with the agent's own hooks.
        """
        if path is None:
            return cls()
        p = Path(path)
        if not p.exists():
            _log.warning("hooks file not found at %s; using empty (no-op) hook engine", p)
            return cls()
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            _log.warning("hooks file failed to parse at %s: %s", p, e)
            return cls()
        # Derive the bundle root from the hooks file location.
        plugin_root = str(p.parent.parent) if p.parent.name == "hooks" else str(p.parent)
        return cls._from_obj(raw, source=str(p), plugin_root=plugin_root)

    @classmethod
    def from_dict(cls, raw: Any) -> HookEngine:
        """Build directly from an already-parsed config object (tests)."""
        return cls._from_obj(raw, source="<dict>")

    @classmethod
    def merge(cls, engines: list[HookEngine]) -> HookEngine:
        """Concatenate the hook lists of multiple engines.

        Used to fold the agent's primary ``hooks.json`` together with
        each enabled plugin's ``hooks.json`` so a Claude-Code-shape
        plugin bundle can ship hooks that fire in this agent.
        Per-event lists are concatenated IN ORDER — the operator's
        own hooks run first, then plugins in plugin-discovery order;
        an exit-2 / decision:block from any engine still blocks.
        """
        by_event: dict[str, list[_CommandHook]] = {}
        for e in engines or []:
            for ev, hooks in e._by_event.items():
                by_event.setdefault(ev, []).extend(hooks)
        return cls(by_event)

    @classmethod
    def _from_obj(
        cls,
        raw: Any,
        *,
        source: str,
        plugin_root: str = "",
    ) -> HookEngine:
        if not isinstance(raw, dict):
            _log.warning("hooks config must be a JSON object at %s", source)
            return cls()
        # Accept the Claude Code settings shape ({"hooks": {...}, ...})
        # or a bare event map. Unknown sibling keys ($schema, …) ignored.
        events = raw.get("hooks")
        if not isinstance(events, dict):
            events = raw
        by_event: dict[str, list[_CommandHook]] = {}
        for event, groups in events.items():
            if event not in HOOK_EVENTS or not isinstance(groups, list):
                continue
            collected: list[_CommandHook] = []
            for gi, group in enumerate(groups):
                if not isinstance(group, dict):
                    continue
                matcher = group.get("matcher")
                matcher = "" if matcher is None else str(matcher)
                gid = str(group.get("id") or f"{event}[{gi}]")
                hook_entries = group.get("hooks")
                if not isinstance(hook_entries, list):
                    continue
                multi = len(hook_entries) > 1
                for hi, hook in enumerate(hook_entries):
                    if not isinstance(hook, dict):
                        continue
                    # Claude Code defines five hook types:
                    #   * ``command``  — shell exec (default; fully supported)
                    #   * ``http``     — POST the stdin payload to a URL
                    #                    (fully supported below)
                    #   * ``mcp_tool`` — invoke an MCP server tool
                    #   * ``prompt``   — ask the model to evaluate inline
                    #   * ``agent``    — spawn a subagent to evaluate
                    # sharedagents currently executes ``command`` + ``http``.
                    # The other three are parsed and KEPT in the engine
                    # (so introspection / ``/doctor`` can show them) but
                    # marked ``unsupported`` and skipped at fire time
                    # with a one-line warning per hook. This is strictly
                    # better than the previous silent-drop behaviour:
                    # an ECC plugin shipping a ``"type": "mcp_tool"``
                    # hook is now visible instead of vanishing.
                    raw_type = str(hook.get("type") or "command").lower()
                    if raw_type == "command":
                        hook_type = "command"
                        command = hook.get("command")
                        if not isinstance(command, str) or not command.strip():
                            continue
                        http_method = "POST"
                        http_headers: tuple[tuple[str, str], ...] = ()
                    elif raw_type == "http":
                        hook_type = "http"
                        # ``"command"`` field carries the URL for
                        # consistency with the command type's storage;
                        # Claude Code's spec also uses an explicit
                        # ``url`` field — accept either.
                        command = hook.get("url") or hook.get("command") or ""
                        if not isinstance(command, str) or not command.strip():
                            continue
                        http_cfg = hook.get("http")
                        http_method = "POST"
                        http_headers = ()
                        if isinstance(http_cfg, dict):
                            m = http_cfg.get("method")
                            if isinstance(m, str) and m.strip():
                                http_method = m.strip().upper()
                            h = http_cfg.get("headers")
                            if isinstance(h, dict):
                                http_headers = tuple((str(k), str(v)) for k, v in h.items())
                    elif raw_type in ("mcp_tool", "prompt", "agent"):
                        # Preserve so /doctor can surface them, but mark
                        # unsupported. Spawn at fire time logs a single
                        # WARN and skips the call.
                        hook_type = "unsupported"
                        command = str(
                            hook.get("command")
                            or hook.get("url")
                            or hook.get("tool")
                            or hook.get("agent")
                            or hook.get("prompt")
                            or f"<{raw_type}>"
                        )
                        http_method = "POST"
                        http_headers = ()
                    else:
                        # Truly unknown type — log + drop to keep
                        # forward-compat. Unlike the listed-but-
                        # unimplemented types, these have no
                        # convention we can faithfully preserve.
                        _log.warning(
                            "hook %s: unknown type %r, dropping",
                            f"{gid}#{hi}",
                            raw_type,
                        )
                        continue
                    try:
                        timeout_s = float(hook.get("timeout") or _DEFAULT_TIMEOUT_S)
                    except (TypeError, ValueError):
                        timeout_s = _DEFAULT_TIMEOUT_S
                    if timeout_s <= 0:
                        timeout_s = _DEFAULT_TIMEOUT_S
                    collected.append(
                        _CommandHook(
                            event=event,
                            matcher=matcher,
                            command=command,
                            timeout_s=timeout_s,
                            is_async=bool(hook.get("async")),
                            hook_id=f"{gid}#{hi}" if multi else gid,
                            hook_type=hook_type,
                            plugin_root=plugin_root,
                            http_method=http_method,
                            http_headers=http_headers,
                        )
                    )
            if collected:
                by_event[event] = collected
        return cls(by_event)

    # ---- introspection ---------------------------------------------------

    def is_empty(self) -> bool:
        return not self._by_event

    def has_event(self, event: str) -> bool:
        return bool(self._by_event.get(event))

    def events(self) -> list[str]:
        return sorted(self._by_event)

    def hook_count(self, event: str | None = None) -> int:
        if event is not None:
            return len(self._by_event.get(event, ()))
        return sum(len(v) for v in self._by_event.values())

    # ---- matching --------------------------------------------------------

    @staticmethod
    def _matches(matcher: str, target: str) -> bool:
        """Claude Code matcher semantics.

        ``"*"`` / ``""`` / whitespace matches everything; otherwise the
        matcher is a regex ``re.fullmatch``'d against ``target`` (so
        ``"Edit|Write"`` matches ``Edit`` or ``Write`` but NOT
        ``MultiEdit`` — callers list ``MultiEdit`` explicitly, exactly
        as Claude Code plugins do). A malformed regex falls back to a
        literal-equality check rather than raising.
        """
        m = (matcher or "").strip()
        if m in ("", "*"):
            return True
        target = target or ""
        # Backward-compat: a matcher written for a pre-lean-agent-deck TitleCase
        # tool id (``"Bash"``, ``"Edit|Write"``) must still fire against the renamed
        # lowercase tool. Test the matcher against the target AND, if the target is
        # a renamed builtin, its LEGACY id — so existing operator deny/rewrite hooks
        # keep matching.
        candidates = [target]
        try:
            from sharedagents_agent.tools.registry import LEGACY_TOOL_IDS

            for _legacy, _canon in LEGACY_TOOL_IDS.items():
                if _canon == target and _legacy != target:
                    candidates.append(_legacy)
        except Exception:  # noqa: BLE001 — matching must never break on import
            pass
        try:
            return any(re.fullmatch(m, t) is not None for t in candidates)
        except re.error as e:
            _log.warning("hook matcher %r is not a valid regex: %s", m, e)
            return any(m == t for t in candidates)

    @staticmethod
    def _match_target(event: str, payload: dict[str, Any]) -> str:
        """The string a ``matcher`` is tested against for ``event``."""
        if event in (
            "PreToolUse",
            "PostToolUse",
            "PostToolUseFailure",
            "PostToolBatch",
        ):
            return str(payload.get("tool_name") or "")
        if event == "SessionStart":
            return str(payload.get("source") or "")
        if event == "SessionEnd":
            return str(payload.get("reason") or "")
        if event == "Setup":
            return str(payload.get("source") or "")
        if event == "PreCompact":
            return str(payload.get("trigger") or "")
        if event in ("SubagentStart", "SubagentStop"):
            return str(payload.get("subagent_name") or "")
        if event == "Notification":
            return str(payload.get("kind") or "")
        if event in (
            "PermissionRequest",
            "PermissionDenied",
        ):
            # Plugin scopes by tool name (e.g. only audit Bash perms).
            return str(payload.get("tool_name") or "")
        if event == "UserPromptExpansion":
            # Slash-command name (e.g. "/compact"), or "" for free text.
            return str(payload.get("command") or "")
        if event == "StopFailure":
            return str(payload.get("reason") or "")
        if event in ("Elicitation", "ElicitationResult"):
            return str(payload.get("name") or "")
        if event == "InstructionsLoaded":
            # The source file path / virtual name (so a plugin can
            # scope to specific instruction files, e.g. "CLAUDE.md").
            return str(payload.get("source") or "")
        if event == "ReasoningLoopDetected":
            # Tier is "tight" | "ngram" | "entropy" | "ratio"; lets a
            # plugin scope a hook to (say) only post-stream ratio
            # loops vs streaming tight-token-collapse.
            return str(payload.get("tier") or "")
        if event in ("PrematureTurnEnd", "PrematureTurnEndGate"):
            # Kind is "silent" | "cutoff" | "colon"; lets a plugin
            # plug only the colon-cliffhanger nudge without touching
            # the silent / cutoff variants.
            return str(payload.get("kind") or "")
        if event == "PreambleInject":
            # Block kind — "pinned_facts" | "compaction_summary" |
            # "current_todos" | future kinds. Scopes a plugin to one
            # specific injector.
            return str(payload.get("kind") or "")
        if event == "UpstreamError":
            # Default classification tag. A plugin matcher of
            # "ctx_overflow" overrides only context-overflow handling.
            return str(payload.get("classification") or "")
        return ""

    # ---- execution -------------------------------------------------------

    async def drain_async_hooks(self) -> None:
        """Await every outstanding async-hook spawn task.

        ``run()`` schedules ``async: true`` hooks via ``asyncio.ensure_future``
        and returns without waiting for them — the hook contract is
        fire-and-forget at the *agent's* perspective. But callers that
        invoke ``run()`` in a short-lived event loop (e.g. the
        ``log_error`` daemon-thread + ``asyncio.run`` path in
        ``sharedagents_core.errors``) need the spawns to actually start before
        the loop closes, otherwise the tasks get cancelled. This drain is
        the explicit "make sure my spawns landed" knob.
        """
        if not self._async_tasks:
            return
        pending = [t for t in self._async_tasks if not t.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def run(
        self,
        event: str,
        payload: dict[str, Any] | None = None,
        *,
        session_id: str = "",
        transcript_path: str = "",
        cwd: str = "",
        permission_mode: str = "",
        plugin_root: str = "",
        extra_env: dict[str, str] | None = None,
    ) -> HookOutcome:
        """Run every hook matching ``event`` and fold the results.

        ``payload`` carries the per-event fields — ``tool_name`` /
        ``tool_input`` / ``tool_response`` for tool events, ``prompt``
        for ``UserPromptSubmit``, ``trigger`` / ``custom_instructions``
        for ``PreCompact``, ``source`` for ``SessionStart``,
        ``reason`` for ``SessionEnd``, ``message`` / ``kind`` for
        ``Notification``, ``stop_hook_active`` for ``Stop`` /
        ``SubagentStop``, ``subagent_name`` / ``state`` / ``output``
        for ``SubagentStop``. The sharedagents extension events carry their
        own payload shape: ``ReasoningLoopDetected`` has ``tier`` (the
        layered detector tier — ``tight`` / ``ngram`` / ``entropy`` /
        ``ratio``) + ``evidence`` + ``round_idx`` + ``thinking_chars``
        + ``content_chars`` + ``channel`` + ``distill_recoveries``;
        ``PrematureTurnEnd`` has ``kind`` (``silent`` / ``cutoff`` /
        ``colon``) + ``tail`` + ``tool_rounds_so_far`` + ``nudge_count``
        + ``final_text_chars`` + ``round_idx`` + ``should_fire_default``;
        ``PrematureTurnEndGate`` has ``kind`` + ``tail`` +
        ``tool_rounds_so_far`` + ``nudge_count`` + ``final_text_chars`` +
        ``round_idx``; ``DistillRecovery`` has ``attempt`` + ``tier`` +
        ``thinking_chars`` + ``evidence`` + ``max_recoveries``;
        ``ForceTerminate`` has ``tier`` + ``round_idx``;
        ``PreambleInject`` has ``kind`` + ``description`` +
        ``user_text_chars``; ``UpstreamError`` has ``classification`` +
        ``message`` + ``round_idx``; ``SessionError`` has ``session_id``
        + ``agent_name`` + ``component`` + ``exception`` + ``file_path``
        + ``fingerprint``.
        The common Claude Code fields (``session_id`` /
        ``transcript_path`` / ``cwd`` / ``hook_event_name`` /
        ``permission_mode``) are merged in here.

        Never raises — a misbehaving hook is captured as an audit
        field, never propagated into the turn.
        """
        outcome = HookOutcome(event=event)
        hooks = self._by_event.get(event)
        if not hooks:
            return outcome
        payload = payload or {}
        target = self._match_target(event, payload)
        matched = [h for h in hooks if self._matches(h.matcher, target)]
        if not matched:
            return outcome

        # Build the Claude Code stdin JSON once — every matching hook
        # for this dispatch receives the same document.
        stdin_obj: dict[str, Any] = {
            "session_id": session_id,
            "transcript_path": transcript_path,
            "cwd": cwd,
            "hook_event_name": event,
        }
        if permission_mode:
            stdin_obj["permission_mode"] = permission_mode
        for k, v in payload.items():
            if v is not None:
                stdin_obj[k] = v
        try:
            stdin_json = json.dumps(stdin_obj, ensure_ascii=False)
        except (TypeError, ValueError):
            # A non-serialisable payload value — fall back to a
            # best-effort string-coerced dump rather than dropping the
            # whole dispatch.
            stdin_json = json.dumps(
                {k: (v if _json_safe(v) else str(v)) for k, v in stdin_obj.items()},
                ensure_ascii=False,
            )

        # Base env: shared by every matching hook. CLAUDE_PLUGIN_ROOT
        # is set PER-HOOK in `_spawn` (each hook brings its own
        # plugin root via `_CommandHook.plugin_root`, falling back to
        # the caller-supplied `plugin_root`) so a merged engine
        # holding hooks from multiple plugin bundles resolves each to
        # its own dir instead of all sharing the agent's root.
        env = dict(os.environ)
        if cwd:
            env["CLAUDE_PROJECT_DIR"] = cwd
        env.setdefault("CLAUDE_HOOK_EVENT_NAME", event)
        if extra_env:
            env.update({str(k): str(v) for k, v in extra_env.items()})
        env.pop("CLAUDE_PLUGIN_ROOT", None)
        fallback_plugin_root = plugin_root or ""

        sync_hooks = [h for h in matched if not h.is_async]
        async_hooks = [h for h in matched if h.is_async]

        # `async: true` hooks are fire-and-forget — Claude Code does
        # not wait for them or read their result. Spawn detached with
        # output discarded (nothing reads it, so there's no reason to
        # buffer a chatty hook's stdout/stderr in-process).
        for h in async_hooks:
            try:
                task = asyncio.ensure_future(
                    self._spawn(
                        h,
                        stdin_json,
                        env,
                        capture=False,
                        fallback_plugin_root=fallback_plugin_root,
                    ),
                )
                # Swallow the result so a failing async hook doesn't
                # surface as an "exception never retrieved" warning.
                task.add_done_callback(_drain_task)
                # Track the background task so a graceful shutdown
                # / process exit can see/await them, addressing the
                # "untracked tasks" nit.
                self._async_tasks.add(task)
                task.add_done_callback(self._async_tasks.discard)
            except RuntimeError:  # pragma: no cover - run() is always async
                pass

        # Sync hooks run concurrently; we await them all (each bounded
        # by its own timeout) and fold in declared order.
        if sync_hooks:
            results = await asyncio.gather(
                *(
                    self._spawn(
                        h,
                        stdin_json,
                        env,
                        fallback_plugin_root=fallback_plugin_root,
                    )
                    for h in sync_hooks
                ),
                return_exceptions=True,
            )
            for h, res in zip(sync_hooks, results):
                if isinstance(res, BaseException):
                    _log.warning("hook %s raised: %s", h.hook_id, res)
                    continue
                outcome.results.append(res)
                outcome.ran += 1
                self._fold(outcome, event, res)
        return outcome

    async def _spawn(
        self,
        hook: _CommandHook,
        stdin_json: str,
        env: dict[str, str],
        *,
        capture: bool = True,
        fallback_plugin_root: str = "",
    ) -> HookCommandResult:
        """Run one hook (shell command, HTTP POST, or no-op for
        unsupported types) and capture the result for ``_fold``.

        Dispatch by ``hook.hook_type``:
        * ``command`` — original shell-exec path; pipe ``stdin_json``,
          read stdout/stderr, parse exit code.
        * ``http`` — POST the stdin JSON to ``hook.command`` with the
          declared method + headers. HTTP 2xx is exit-0 success with
          response body folded as stdout. HTTP ≥ 400 is exit-1
          (non-blocking error). Network/timeout errors land in
          ``res.error``.
        * ``unsupported`` — single WARN per fire; result records the
          skip so ``/doctor`` sees something.

        ``capture=False`` (used for fire-and-forget ``async`` hooks)
        routes stdout/stderr to ``subprocess.DEVNULL`` — the result is never
        folded into a :class:`HookOutcome`.
        """
        res = HookCommandResult(
            hook_id=hook.hook_id,
            command=hook.command,
            exit_code=None,
        )
        if hook.hook_type == "unsupported":
            res.error = (
                f"hook type not yet implemented; declared as "
                f"{hook.command!r} (sharedagents supports `command` + `http`)"
            )
            _log.warning(
                "hook %s: skipped (unsupported type — declared as %r)",
                hook.hook_id,
                hook.command,
            )
            return res
        if hook.hook_type == "http":
            return await self._spawn_http(hook, stdin_json, res)
        # Fall through: command-type hook (original behaviour).
        out_target = asyncio.subprocess.PIPE if capture else subprocess.DEVNULL
        # Per-hook CLAUDE_PLUGIN_ROOT: each hook may originate from a
        # different plugin (merged engines), so set the env var from
        # the hook's recorded root, falling back to the caller's.
        eff_root = hook.plugin_root or fallback_plugin_root
        spawn_env = dict(env)
        if eff_root:
            spawn_env["CLAUDE_PLUGIN_ROOT"] = eff_root
        else:
            spawn_env.pop("CLAUDE_PLUGIN_ROOT", None)
        try:
            spawn_kwargs = dict(
                stdin=asyncio.subprocess.PIPE,
                stdout=out_target,
                stderr=out_target,
                env=spawn_env,
                # New session => own process group, so a timeout kill
                # cascades to any children the hook spawned.
                start_new_session=not oscompat.IS_WINDOWS,
                **oscompat.detach_kwargs(),
            )
            if oscompat.IS_WINDOWS:
                proc = await asyncio.create_subprocess_exec(
                    *oscompat.spawn_shell_argv(hook.command),
                    **spawn_kwargs,
                )
            else:
                proc = await asyncio.create_subprocess_shell(
                    hook.command,
                    **spawn_kwargs,
                )
        except OSError as e:
            res.error = f"spawn: {type(e).__name__}: {e}"
            _log.warning("hook %s spawn failed: %s", hook.hook_id, e)
            return res
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=stdin_json.encode("utf-8")),
                timeout=hook.timeout_s,
            )
        except asyncio.TimeoutError:
            await _kill_process_group(proc)
            res.timed_out = True
            res.error = f"timeout after {hook.timeout_s:.0f}s"
            _log.warning(
                "hook %s timed out after %.0fs",
                hook.hook_id,
                hook.timeout_s,
            )
            return res
        res.exit_code = proc.returncode
        res.stdout = (stdout or b"").decode("utf-8", "replace")
        res.stderr = (stderr or b"").decode("utf-8", "replace")
        # Parse a JSON *object* stdout — that is the control protocol.
        s = res.stdout.strip()
        if s.startswith("{"):
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                res.json_output = parsed
        return res

    async def _spawn_http(
        self,
        hook: _CommandHook,
        stdin_json: str,
        res: HookCommandResult,
    ) -> HookCommandResult:
        """POST ``stdin_json`` to the hook's URL; map HTTP status to
        Claude Code exit-code semantics.

        HTTP 2xx → exit_code 0, response body as stdout (parsed as a
        JSON control block if it starts with ``{``). HTTP 4xx/5xx →
        exit_code 1 (non-blocking error; response body in stderr) so
        the engine surfaces the failure without blocking the turn — a
        plugin author that wants a hard block should return 2xx with
        ``{"decision":"block","reason":"..."}`` in the body instead.

        Transport errors / DNS / timeouts populate ``res.error`` and
        leave ``exit_code=None`` (matches command-hook spawn-failure
        semantics for downstream auditing).

        Implementation uses ``urllib.request`` to avoid pulling in
        ``httpx`` / ``aiohttp`` as new hard dependencies — hooks must
        stay startup-cheap. The blocking call is dispatched to a
        thread executor so a slow webhook can't pin the event loop.
        """
        import urllib.error
        import urllib.request

        url = hook.command
        method = hook.http_method or "POST"
        headers: dict[str, str] = {
            "Content-Type": "application/json; charset=utf-8",
        }
        for k, v in hook.http_headers:
            headers[k] = v

        def _do_request() -> tuple[int, str, str]:
            req = urllib.request.Request(
                url=url,
                method=method,
                data=stdin_json.encode("utf-8"),
                headers=headers,
            )
            try:
                with urllib.request.urlopen(req, timeout=hook.timeout_s) as resp:
                    status = resp.getcode()
                    body = resp.read().decode("utf-8", "replace")
                    return status, body, ""
            except urllib.error.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode("utf-8", "replace")
                except Exception:  # noqa: BLE001
                    body = ""
                return int(e.code or 500), "", body or str(e)

        loop = asyncio.get_running_loop()
        try:
            status, stdout, stderr = await asyncio.wait_for(
                loop.run_in_executor(None, _do_request),
                timeout=hook.timeout_s + 5.0,
            )
        except asyncio.TimeoutError:
            res.timed_out = True
            res.error = f"http hook timeout after {hook.timeout_s:.0f}s"
            _log.warning(
                "hook %s (http): timeout after %.0fs",
                hook.hook_id,
                hook.timeout_s,
            )
            return res
        except urllib.error.URLError as e:
            res.error = f"http: {type(e).__name__}: {e}"
            _log.warning("hook %s (http) transport failed: %s", hook.hook_id, e)
            return res
        except Exception as e:  # noqa: BLE001
            res.error = f"http: {type(e).__name__}: {e}"
            _log.warning("hook %s (http) failed: %s", hook.hook_id, e)
            return res

        res.exit_code = 0 if 200 <= status < 300 else 1
        res.stdout = stdout
        res.stderr = stderr
        s = (stdout or "").strip()
        if s.startswith("{"):
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                res.json_output = parsed
        return res

    def _fold(self, outcome: HookOutcome, event: str, res: HookCommandResult) -> None:
        """Interpret one hook result into ``outcome`` per Claude Code."""
        # Spawn failure / timeout — non-blocking error, audit only.
        if res.timed_out or res.error:
            return
        rc = res.exit_code

        # --- exit code 2 — blocking error ---------------------------------
        if rc == 2:
            msg = (res.stderr or res.stdout or "").strip()
            if msg:
                outcome.block_reason = _join_reason(outcome.block_reason, msg)
            # Only flip `blocked` for events that can actually block an
            # operation — for the rest (Notification, SessionStart/End,
            # PreCompact, PostToolUseFailure) exit-2 just surfaces
            # stderr, exactly like Claude Code.
            if event in _BLOCK_CAPABLE_EVENTS:
                outcome.blocked = True
            if event == "PreToolUse":
                outcome.permission_decision = _strongest_perm(
                    outcome.permission_decision,
                    "deny",
                )
                outcome.permission_reason = _join_reason(
                    outcome.permission_reason,
                    msg,
                )
            # An exit-2 hook does not also emit a JSON control block.
            return

        # --- other non-zero — non-blocking error --------------------------
        if rc not in (0, None):
            tail = res.stderr.strip() or res.stdout.strip()
            if tail:
                _log.info(
                    "hook %s exited rc=%s: %s",
                    res.hook_id,
                    rc,
                    tail[:200],
                )
            return

        # --- exit 0 -------------------------------------------------------
        j = res.json_output
        if j is None:
            # Plain stdout.
            out = res.stdout.strip()
            if out:
                if event in _CONTEXT_STDOUT_EVENTS:
                    outcome.additional_context = _join_context(
                        outcome.additional_context,
                        out,
                    )
                else:
                    outcome.stdout_blocks.append(out)
            return

        # --- exit 0 with a JSON control block -----------------------------
        if j.get("continue") is False:
            outcome.should_continue = False
            sr = j.get("stopReason")
            if isinstance(sr, str) and sr:
                outcome.stop_reason = _join_reason(outcome.stop_reason, sr)
        if j.get("suppressOutput") is True:
            outcome.suppress_output = True
        sm = j.get("systemMessage")
        if isinstance(sm, str) and sm.strip():
            outcome.system_messages.append(sm.strip())

        decision = j.get("decision")
        reason = j.get("reason") if isinstance(j.get("reason"), str) else ""
        if decision == "block":
            # `block_reason` (the model-facing feedback) is always
            # recorded; `blocked` only flips for block-capable events.
            if reason:
                outcome.block_reason = _join_reason(outcome.block_reason, reason)
            if event in _BLOCK_CAPABLE_EVENTS:
                outcome.blocked = True
            if event == "PreToolUse":
                outcome.permission_decision = _strongest_perm(
                    outcome.permission_decision,
                    "deny",
                )
                if reason:
                    outcome.permission_reason = _join_reason(
                        outcome.permission_reason,
                        reason,
                    )
        elif decision == "approve" and event == "PreToolUse":
            # Deprecated Claude Code alias for a PreToolUse allow.
            outcome.permission_decision = _strongest_perm(
                outcome.permission_decision,
                "allow",
            )
            if reason and not outcome.permission_reason:
                outcome.permission_reason = reason

        hso = j.get("hookSpecificOutput")
        if isinstance(hso, dict):
            if event == "PreToolUse":
                pd = hso.get("permissionDecision")
                if pd in ("allow", "deny", "ask"):
                    outcome.permission_decision = _strongest_perm(
                        outcome.permission_decision,
                        pd,
                    )
                    pdr = hso.get("permissionDecisionReason")
                    if isinstance(pdr, str) and pdr:
                        outcome.permission_reason = _join_reason(
                            outcome.permission_reason,
                            pdr,
                        )
                    if pd == "deny":
                        outcome.blocked = True
                        outcome.block_reason = _join_reason(
                            outcome.block_reason,
                            hso.get("permissionDecisionReason") or "",
                        )
                # ``updatedInput`` — Claude Code's tool-input rewrite
                # hook (e.g. an auto-linter that rewrites ``rm -rf``
                # into ``rm -rfI``, or a normaliser that resolves a
                # relative path). Last writer wins so chained mutators
                # can compose. Non-dict / non-object payloads are
                # ignored — better to no-op than ship a malformed
                # dispatch to the runner.
                ui = hso.get("updatedInput")
                if isinstance(ui, dict):
                    outcome.updated_input = ui
            # `additionalContext` only reaches the model for the
            # events Claude Code injects it into — UserPromptSubmit,
            # SessionStart, PostToolUse. Elsewhere it's ignored.
            if event in _ADDITIONAL_CONTEXT_EVENTS:
                ac = hso.get("additionalContext")
                if isinstance(ac, str) and ac.strip():
                    outcome.additional_context = _join_context(
                        outcome.additional_context,
                        ac,
                    )


# --------------------------------------------------------------------------
# Module helpers
# --------------------------------------------------------------------------


def _json_safe(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool, type(None), list, dict))


def _drain_task(task: asyncio.Future[Any]) -> None:
    """done_callback for fire-and-forget async hooks — swallow result."""
    try:
        task.result()
    except Exception:  # noqa: BLE001 - async hook failures are non-fatal
        pass


async def _kill_process_group(proc: asyncio.subprocess.Process) -> None:
    """SIGTERM the hook's process group, escalate to SIGKILL.

    The hook ran with ``start_new_session=True`` so it owns a process
    group; killing the group reaps any children it spawned (ECC hooks
    routinely shell out to ``node`` which spawns more).
    """
    if oscompat.IS_WINDOWS:
        # Best effort: terminate the immediate hook, not its grandchildren.
        if proc.pid and proc.returncode is None:
            try:
                oscompat.kill_process(proc.pid, force=True)
                await asyncio.wait_for(proc.wait(), timeout=1.0)
            except (OSError, asyncio.TimeoutError):
                pass
        return
    pgid = None
    try:
        if proc.pid:
            pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError, OSError):
        pgid = None

    def _signal(sig: int) -> bool:
        """Signal the whole group when we own its pgid, else just the child."""
        try:
            if pgid is not None:
                os.killpg(pgid, sig)
            else:
                proc.send_signal(sig)
        except (ProcessLookupError, PermissionError, OSError, ValueError):
            return False
        return True

    # SIGTERM the group and give it a grace period to exit cleanly.
    if _signal(signal.SIGTERM):
        try:
            await asyncio.wait_for(proc.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pass
    # ALWAYS escalate to SIGKILL on the group — the direct child exiting
    # after SIGTERM does not prove the group is empty: a SIGTERM-ignoring
    # grandchild (ECC hooks shell out to ``node`` which spawns more) keeps
    # the group alive and must still be reaped. Returning early once the
    # child exits would orphan it. When we only have the child handle
    # (no pgid) there is nothing further to reap, so this is a no-op then.
    if pgid is not None:
        _signal(signal.SIGKILL)
        try:
            await asyncio.wait_for(proc.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pass


# --------------------------------------------------------------------------
# Process-wide active engine registry
# --------------------------------------------------------------------------
#
# Name-keyed registry of the HookEngines attached to a running sharedagents
# process. Each Agent constructs its engine lazily in
# ``_load_hook_engine`` and wires it here so logging-side callers (e.g.
# ``sharedagents_core.errors.log_error`` firing ``SessionError``) can fire
# hooks without having to thread an Agent reference through every call
# site. A gateway that hosts multiple Agents (each with its own
# ``~/.sharedagents/agents/<name>/hooks.json``) needs the dict — a singleton
# would let agent B's hook receive agent A's error.

_ACTIVE_ENGINES: dict[str, HookEngine] = {}
_DEFAULT_AGENT = "jarvis"


def set_active_engine(
    engine: HookEngine | None,
    agent_name: str | None = None,
) -> None:
    """Register an engine under ``agent_name`` (default ``jarvis``).

    Passing ``None`` engine removes the registration. The gateway-multi-agent
    case (each Agent calls this at boot) is the load-bearing path; the
    single-agent shim (``agent_name=None``) preserves the previous
    behaviour for callers that don't know their own name yet.
    """
    key = agent_name or _DEFAULT_AGENT
    if engine is None:
        _ACTIVE_ENGINES.pop(key, None)
    else:
        _ACTIVE_ENGINES[key] = engine


def get_active_engine(agent_name: str | None = None) -> HookEngine | None:
    """Look up the engine for ``agent_name`` (default ``jarvis``).

    Returns ``None`` if no engine is registered for that name.
    """
    key = agent_name or _DEFAULT_AGENT
    return _ACTIVE_ENGINES.get(key)


# --------------------------------------------------------------------------
# Agent-side hook dispatch mixin
# --------------------------------------------------------------------------
#
# The per-agent glue that loads/caches an agent's ``HookEngine`` and fires
# Claude Code hook events with the common hook context. Lives here (next to
# the engine it drives) rather than in ``agents/agent.py`` so the Agent
# composition root stays lean. ``Agent`` mixes this in; every method
# resolves against the Agent instance via normal MRO (``self.cfg``,
# ``self.name``, ``self.sessions_dir``, ``self._effective_mode`` etc.).
#
# ``__slots__ = ()`` is REQUIRED: ``Agent`` is ``@dataclass(slots=True)`` and
# relies on undeclared-attribute assignment raising ``AttributeError``. A
# mixin base without ``__slots__`` would re-introduce ``__dict__`` and
# silently break that guarantee.


class HookDispatchMixin:
    """Claude Code hook-engine load + per-event dispatch for :class:`Agent`."""

    __slots__ = ()

    def _hook_kwargs(self, session: Any, cwd: str = "") -> dict[str, Any]:
        """Common Claude Code hook fields for this agent + session."""
        sid = str(getattr(session, "hash", "") or "")
        transcript = ""
        try:
            if sid:
                transcript = str(self.sessions_dir / f"{sid}.jsonl")
        except Exception:  # noqa: BLE001
            transcript = ""
        return {
            "session_id": sid,
            "transcript_path": transcript,
            "cwd": cwd or str(getattr(session, "workspace_cwd", "") or safe_getcwd()),
            "permission_mode": self._effective_mode(session),
            "plugin_root": getattr(self, "_hook_plugin_root", "") or "",
        }

    async def _run_hook_event(
        self,
        event: str,
        session: Any,
        payload: dict[str, Any],
        *,
        cwd: str = "",
    ) -> Any:
        """Dispatch one Claude Code hook event; never raises.

        Returns an empty-effect
        :class:`~sharedagents_agent.hooks.HookOutcome` when no engine is
        loaded, the event has no matching hooks, or the engine errors
        — so callers treat "hooks disabled" and "hooks ran, no
        effect" identically.
        """
        from sharedagents_agent.hooks import HookOutcome

        engine = self._load_hook_engine()
        if engine is None or not engine.has_event(event):
            return HookOutcome(event=event)
        try:
            return await engine.run(
                event,
                payload,
                **self._hook_kwargs(session, cwd),
            )
        except Exception as exc:  # noqa: BLE001
            import logging

            logging.getLogger("sharedagents.agents.hooks").info(
                "hook event %s failed: %s",
                event,
                exc,
            )
            return HookOutcome(event=event)

    def _load_hook_engine(self) -> Any:
        """Lazy-load + cache the agent's Claude Code :class:`HookEngine`.

        Sourced from ``cfg.hooks_path`` (the resolved
        ``<agent_home>/hooks.json`` by default; ``cfg.hooks_file`` for
        older configs). Also derives ``_hook_plugin_root`` — the
        ``CLAUDE_PLUGIN_ROOT`` exported to every hook command — from
        the hooks file's location: a file inside a ``hooks/`` dir (the
        Claude Code plugin layout, e.g. ``…/ecc/hooks/hooks.json``)
        roots at its grandparent, otherwise at its parent. Returns
        ``None`` on any failure so callers always treat hooks as
        best-effort. The cached instance is reused for the agent's
        lifetime — operators reloading hooks should bounce the agent.
        """
        if self._hook_engine is not None:
            return self._hook_engine
        try:
            from sharedagents_agent.hooks import HookEngine

            hooks_path = getattr(self.cfg, "hooks_path", None) or getattr(self.cfg, "hooks_file", None)
            if hooks_path:
                hp = Path(hooks_path)
                engine = HookEngine.load(hp)
                root = hp.parent.parent if hp.parent.name == "hooks" else hp.parent
                self._hook_plugin_root = str(root)
            else:
                engine = HookEngine.load(None)
                self._hook_plugin_root = ""
            # Merge enabled-plugin hooks.json files so a Claude-Code
            # plugin bundle that ships ``hooks/hooks.json`` (or a
            # top-level ``hooks.json``) fires here too — agent-local
            # rules first, then plugins in discovery order.
            try:
                from sharedagents_agent.plugins import list_plugins

                plug_engines: list[Any] = []
                for p in list_plugins():
                    if not p.enabled or p.hooks_file is None:
                        continue
                    pe = HookEngine.load(p.hooks_file)
                    if not pe.is_empty():
                        plug_engines.append(pe)
                if plug_engines:
                    engine = HookEngine.merge([engine] + plug_engines)
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            return None
        self._hook_engine = engine
        # Wire the process-wide active engine so logging-side callers
        # (e.g. ``sharedagents_core.errors.log_error`` firing ``SessionError``)
        # can fire hooks without needing an Agent reference. A gateway
        # may host multiple Agents — register under our own ``name`` so
        # an error from agent A does NOT trigger agent B's hooks.
        try:
            from sharedagents_agent.hooks import set_active_engine

            set_active_engine(engine, agent_name=self.name)
        except Exception:  # noqa: BLE001
            pass
        return engine
