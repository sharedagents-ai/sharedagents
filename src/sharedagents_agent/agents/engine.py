"""``Engine`` ABC — how an agent runs one turn.

The contract here is the union of behaviors observed in:

* ``sharedagents_agent.tools.runners.run_delegate_claude_async``
  (~ runners.py L1492–1926)
* ``sharedagents_agent.tools.runners.run_delegate_codex_async``  (~ L2014–2289)
* ``sharedagents_agent.tools.runners.run_delegate_antigravity_async`` (~ L1207–1489)
* ``sharedagents_agent.turn_runtime`` (local-LLM happy path)
* ``sharedagents_acp.server`` event shapes consumed by every frontend

If a behavior is in any of those today it MUST appear here or in a
subclass; otherwise we are silently dropping functionality.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class EngineKind(str, enum.Enum):
    """How a turn is executed for an agent.

    Stored as a string on ``AgentConfig.engine`` so configs stay JSON-
    friendly. Values mirror what the user types in
    ``~/.sharedagents/agents/<name>/config.json``::

        {"name": "dashboard_dev", "engine": "claude_cli", ...}
    """

    LOCAL_LLM = "local_llm"
    CLAUDE_CLI = "claude_cli"
    CODEX_CLI = "codex_cli"
    ANTIGRAVITY_CLI = "antigravity_cli"
    # Z.AI's ZCode CLI (``zcode``, ``zcode-app-cli``), driven in headless
    # print mode ``zcode -p <prompt> --json``. Auth is the CLI's own Z.AI
    # OAuth login (``zcode login``), covering a GLM Coding Plan quota.
    ZCODE_CLI = "zcode_cli"
    PI_RPC = "pi_rpc"
    # Drives ANY external ACP *server* (the agent) as an ACP *client*,
    # over newline-delimited JSON-RPC stdio via ``sharedagents_acp.client``.
    # Generic by design: the spawn command + auth/model come from the
    # agent's config (``acp_command`` etc.), so one engine serves Cursor
    # (``cursor-agent acp``), Gemini, Zed agents, or any ACP server.
    # Unlike the *_CLI kinds above (native print mode), this talks ACP.
    ACP = "acp"
    # Registry-configured ACP kinds. Their command, mode map, auth, model,
    # and context defaults are data passed to the generic AcpEngine.
    CURSOR_ACP = "cursor_acp"
    OPENCODE_ACP = "opencode_acp"
    KIMI_ACP = "kimi_acp"
    HERMES_ACP = "hermes_acp"
    # Devin CLI (``devin acp``). Mode is a session config option
    # (``accept-edits`` / ``ask`` / ``plan`` / ``bypass``); ``session/set_mode``
    # acknowledges and does not change it.
    DEVIN_ACP = "devin_acp"
    # Qwen Code CLI (``qwen --acp``), the QwenLM terminal agent (Gemini-CLI
    # lineage). Modes are native ACP ``session/set_mode`` ids (``plan`` /
    # ``default`` / ``auto-edit`` / ``auto`` / ``yolo``); the model route is a
    # ``session/set_config_option`` select (``model``). Auth and the model
    # catalogue come from ``~/.qwen/settings.json`` (openai-compatible routes,
    # e.g. a local vllm-mux pool), so the engine itself carries no cloud
    # credential.
    QWEN_ACP = "qwen_acp"

    @classmethod
    def is_kind_label(cls, value: str) -> bool:
        """True when ``value`` is a backend kind (cursor_acp), not a model id."""
        raw = str(value or "").strip()
        return bool(raw) and raw in {member.value for member in cls}


# ---- normalized turn event stream -----------------------------------
#
# Every engine emits the same shape so frontends (TUI ClaudeMessage,
# Telegram bot, ACP bridge) don't have to care which engine produced
# the turn. The shapes below intentionally mirror the existing ACP
# events (``tool_call`` / ``tool_call_update`` / ``tool_delta`` /
# ``tool_result`` / ``done``) that ``sharedagents_tui/app.py:_run_claude``
# already consumes (app.py L5350–5616).


@dataclass(slots=True, frozen=True)
class TurnStarted:
    """Emitted once at the top of every turn.

    Mirrors the ``turn.started`` ACP event today. ``sharedagents_session_id``
    is the parent sharedagents session; ``engine_session_id`` is whatever
    the engine itself uses to resume (Claude session uuid, Codex thread
    id, Antigravity session uuid, or empty for the first call).
    """

    turn_id: str
    sharedagents_session_id: str
    engine_session_id: str = ""
    resumed: bool = False


@dataclass(slots=True, frozen=True)
class TextDelta:
    """Streaming prose chunk. Equivalent to today's SSE ``delta``.

    ``channel`` tags the emission source so the gateway/frontend can
    route the chunk correctly:

    * ``"content"`` — the assistant's visible reply. ACP wire shape:
      ``agent_message_chunk``. CLI engines emit ONLY this.
    * ``"thinking"`` — the model's reasoning trace. Local LLM and
      Antigravity CLI emit this channel; other CLI engines still hide
      private CoT. ACP wire shape: ``agent_thought_chunk``. Visible
      to the user only when ``cfg.reasoning_visibility != "off"``.
      Persistence is GATED by ``session.persist_reasoning_traces``,
      default True
      (fine-tune-friendly), but operators with regulatory constraints
      flip it to False per session. The live render is independent —
      flipping persistence off does NOT hide the trace in the TUI when
      reasoning_visibility is on.
    * ``"tool_delta"`` — a chunk streaming inside an in-flight tool
      call. Carries ``call_id`` to route to the right widget. ACP
      wire shape: ``tool_call_update`` with ``content=[{"type":"text",
      "text": ...}]``. Used by CLI engines while their inner
      Read/Bash/etc. tool is running, and by LocalLLMEngine when a
      sharedagents tool runner is streaming progress (e.g. delegate_*).

    ``call_id`` is required when ``channel == "tool_delta"``; ignored
    otherwise.
    """

    text: str
    channel: str = "content"  # "content" | "thinking" | "tool_delta"
    call_id: str = ""


@dataclass(slots=True, frozen=True)
class ToolCallStarted:
    """The engine is calling a sub-tool (Claude's Read/Bash/TodoWrite,
    Codex's command_execution, Antigravity's tool_use). Mirrors today's
    ACP ``tool_call`` event the TUI mounts as ``ToolCallMessage``
    (``sharedagents_tui/app.py`` L5491–5515).

    For ``LocalLLMEngine`` this also covers the model's own tool calls
    dispatched through ``tool_runner`` — same event shape.
    """

    call_id: str
    name: str
    arguments: dict[str, Any]
    # True when this call is the *outer* delegate wrapper (the engine
    # call itself) rather than an inner CLI tool. Today's runners
    # surface a single outer call_id per turn — ``sharedagents_tui/app.py``
    # L5491–5499 relies on this to skip rendering the wrapper.
    is_outer: bool = False
    # True when this is a SUBAGENT's inner tool call surfaced into the parent
    # turn stream for live display (the Agent/delegate tool runs a child in
    # process). The parent session log must NOT persist these — the child has
    # its own session log (viewable via the monitor). Distinct from is_outer
    # (the delegate WRAPPER call): a from_subagent call is a real inner tool of
    # the CHILD, not the parent's wrapper.
    from_subagent: bool = False


@dataclass(slots=True, frozen=True)
class ToolCallResult:
    """A sub-tool finished. Mirrors ACP ``tool_result``."""

    call_id: str
    ok: bool
    content: str
    # See ToolCallStarted.from_subagent — a subagent's inner tool result the
    # parent log must not persist.
    from_subagent: bool = False
    outcome: str | None = None
    member_outcomes: tuple[str | None, ...] = ()
    # ``image_url`` parts the tool showed the model (``read`` on an image).
    # Persisted with the result so a replay shows the model the same image;
    # never forwarded to frontends (the ACP translation sends text only).
    images: tuple[dict, ...] = ()


@dataclass(slots=True, frozen=True)
class ArtifactReference:
    """One explicit, metadata-only artifact output from an engine.

    Engines emit it when their native protocol carries a structured artifact
    reference (for example an ACP ``resource_link`` content block) or when a
    CLI adapter recognizes its explicit user-facing local-output link. Raw
    bodies, data URIs, credentials, and native session ids are deliberately
    absent.
    """

    uri: str = ""
    path: str = ""
    display_name: str = ""
    media_type: str = ""
    size_bytes: int = 0
    sha256: str = ""
    storage_scope: str = ""
    locator: str = ""


@dataclass(slots=True, frozen=True)
class TurnError:
    """Engine reported a recoverable error mid-turn. Equivalent to
    ``sharedagents_agent.tools.runners._log_delegate_error`` + a tool-result
    error today. Non-fatal events (e.g. gemini ``error`` items, see
    runners.py L1391–1401) MUST NOT use this class; log them at the
    engine level and keep streaming.
    """

    message: str
    classification: str = ""  # see ``turn_runtime._classify_upstream_error``


@dataclass(slots=True, frozen=True)
class HandoffAccepted:
    """The target accepted the complete prompt carrying a handoff.

    This is an internal durability boundary, not a frontend event. It occurs
    after a prompt-correlated protocol acknowledgement and before terminal
    success/error/abort, allowing the canonical lane watermark to advance
    without conflating prompt acceptance with turn completion.
    """

    handoff_id: str
    through_seq: int


@dataclass(slots=True, frozen=True)
class TurnDone:
    """End-of-turn marker. The TUI uses ``state != "error"`` to decide
    finalize vs mark_error (``sharedagents_tui/app.py`` L5555–5566).

    **Contract:** an engine MUST yield exactly ONE :class:`TurnDone`
    per call to :meth:`Engine.run_turn`, and it MUST be the LAST event.
    Even on error / abort. The ACP translator turns this into the
    single ``stopReason`` wire event for the frontend; :class:`TurnError`
    emits prose only, never a stopReason, so we never double-terminate
    a turn.
    """

    state: str  # "ok" | "error" | "aborted"
    engine_session_id: str = ""
    # Wall-clock metrics that today's TUI stamps onto delegate turns
    # (see ``_build_delegate_metrics`` for shape). Engines compute this
    # so the metrics row reads identically for native and CLI turns.
    metrics: dict[str, Any] = field(default_factory=dict)
    # Structured finish reason — "stop" for natural completion, "round_cap"
    # when the per-turn round budget was exhausted. state keeps its engine
    # contract (ok/error/aborted); callers that must distinguish capped from
    # completed read this field (and the engine audit / turn record).
    completion_reason: str = "stop"


@dataclass(slots=True)
class PermissionRequested:
    """Engine wants the user to approve a tool call before it dispatches.

    The previous design had
    ``PermissionRequest`` as an out-of-band sentinel that couldn't
    actually travel through the ``AgentTurnEvent`` stream. This is the
    proper event variant.

    Flow (cooperative; same event loop, no threads):

    1. :class:`LocalLLMEngine` encounters a tool call whose policy is
       ``ask``. It constructs this event with a freshly-created
       ``asyncio.Future`` on ``response`` and ``yield``s it.
    2. :class:`Agent.run_turn` receives the event. It does NOT forward
       it to the gateway translator. Instead it runs
       :meth:`Agent.resolve_permission` which consults the
       :class:`PermissionPolicy`, optionally surfaces the
       ``request_permission`` ACP outbound call, and sets the future.
    3. The engine awaits the future, then proceeds with allow / deny.

    CLI engines never yield this event — the CLI's own permission
    system runs independently.
    """

    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    criticality: str  # "read_only" | "mutating" | "destructive" | "network"
    response: Any = None  # asyncio.Future[str] — "allow" | "deny"
    reason: str = ""  # populated by Agent on deny
    # A ``PreToolUse`` hook's permission decision, when one fired
    # before this event was yielded: "" (no opinion) | "ask" (force
    # the confirmation path even if policy would allow). The engine
    # handles "allow"/"deny" itself and never yields this event for
    # those, so only "" / "ask" reach the Agent's resolver.
    hook_decision: str = ""
    # Populated by Agent.resolve_permission after the cooperative decision.
    # These fields contain no tool arguments or free-form reason text.
    decision: str = ""
    decision_source: str = ""


@dataclass(slots=True, frozen=True)
class RoundComplete:
    """A worker round finished (tool results have been appended).

    Emitted between rounds so the Agent layer can checkpoint the in-
    progress turn to disk (``record_turn(in_progress=True)``) and so
    frontends can render a "round N done" marker. Carries the just-
    completed round index — the next round, if any, is ``round_idx+1``.

    NOT a terminal event: the turn loop continues. ``TurnDone`` is the
    only terminal signal; engines MUST still emit exactly one TurnDone
    regardless of how many RoundComplete events fired.

    ``metrics`` carries THIS round's worker telemetry (``prompt_tokens``,
    ``context_window``, token splits, …) — the same dict the engine
    logs as ``TURN_METRICS``. The ACP translator turns it into a
    per-round ``usage_update`` so the frontend's ``ctx X/Y`` indicator
    grows after every tool round, not just once at ``TurnDone``. ``None``
    when the round produced no metrics (e.g. an aborted round).
    """

    round_idx: int
    metrics: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class UserMessageInjected:
    """A mid-turn user message was drained from the inject buffer.

    Yielded by the engine when a queued prompt from
    :meth:`LocalLLMEngine.inject_user_message` lands at a round
    boundary and gets appended to the live message list. Lets the
    frontend confirm "the steering message you typed mid-plan was
    picked up" instead of guessing from later behavior. Carries the
    text exactly as queued so the UI can render it inline above the
    next assistant round.
    """

    text: str
    round_idx: int


@dataclass(slots=True, frozen=True)
class QueueMessageInjected:
    """A durable follow-up-queue item was drained at a round boundary.

    Like :class:`UserMessageInjected` but sourced from the persistent
    cross-interface follow-up queue (not the in-memory steer buffer) and
    drained GENTLY — no soft-abort. Carries ``queue_id`` so a UI can
    reconcile it with its pending-queue widget. The durable
    ``queue_msg_injected`` log event is written by the drain callback; this
    event is the live-stream signal for ACP consumers (TUI / Telegram).
    """

    text: str
    queue_id: str
    round_idx: int


AgentTurnEvent = (
    TurnStarted
    | TextDelta
    | ToolCallStarted
    | ToolCallResult
    | ArtifactReference
    | TurnError
    | HandoffAccepted
    | TurnDone
    | PermissionRequested
    | RoundComplete
    | UserMessageInjected
    | QueueMessageInjected
)


# ---- run-turn input --------------------------------------------------


@dataclass(slots=True)
class TurnRequest:
    """One inbound prompt from the user (or the runtime).

    Carries everything the engine needs to dispatch a turn without
    reaching into globals. Fields mirror what the existing slash bridge
    + delegate runners read today:

    * ``prompt`` — the user message text. May be empty when the engine
      is being primed-only (``system_prompt`` set, no user turn yet).
      Also empty when ``prompt_parts`` carries multipart content (see
      below).
    * ``prompt_parts`` — multimodal content in OpenAI vision shape::

          [
            {"type": "text", "text": "what's in this image?"},
            {"type": "image_url", "image_url": {"url": "data:..."}}
          ]

      When non-empty,
      ``prompt_parts`` wins over ``prompt``. The engine is responsible
      for translating this into the upstream wire format (local LLM:
      pass through; Claude CLI: base64-encode each image and use
      ``--attach``; Codex: encode via ``--image``; Antigravity: same).
    * ``system_prompt`` / ``system_prompt_sha256`` — composed by the
      Agent's context-builder; see
      :meth:`Agent.compose_system_prompt`. The engine's
      :meth:`system_prompt_policy` decides what to do with it:
      ``"every_turn"`` (local LLM) prepends every round;
      ``"first_turn"`` (CLI engines) primes on the FIRST call and
      relies on resume thereafter. The sha256 hash lets the engine
      detect when the prompt has CHANGED (operator edited SOUL.md /
      AGENTS.md) — that invalidates the CLI session and forces a
      fresh prime even though ``priming_status`` says completed.
    * ``preamble`` — pre-composed below-cache-boundary blocks (skills /
      memory recall / pinned facts / todos / compaction summary).
      Built by :meth:`Agent.compose_round_preamble`. Agent owns the
      composition exclusively; engines NEVER reach into the agent's
      .md files themselves (single
      composition owner). Local LLM injects this every round; CLI
      engines fold it into the first-call system prompt.
    * ``mode`` — sharedagents permission mode (``plan`` / ``ask`` / ``auto``).
      Each subclass maps this to its CLI flag set; today see
      ``runners._CLAUDE_MODE_MAP`` / ``_CODEX_MODE_FLAGS`` /
      ``_GEMINI_MODE_MAP`` (runners.py L916–931, L1936–1940).
    * ``cwd`` — optional working directory; engines call
      ``_resolve_delegate_cwd`` (runners.py L934–967) shape.
    * ``model`` — engine-specific override (Antigravity today accepts
      ``--model``; runners.py L1268–1283). Ignored by engines that
      don't surface a model knob.
    """

    prompt: str = ""
    prompt_parts: list[dict[str, Any]] = field(default_factory=list)
    system_prompt: str | None = None
    system_prompt_sha256: str = ""
    preamble: list[dict[str, Any]] = field(default_factory=list)
    mode: str = ""  # "plan" | "ask" | "auto" | ""
    cwd: str | None = None
    model: str | None = None
    # Reserved for future per-engine kwargs (extra CLI flags, headers,
    # etc.). Engines pop their known keys and warn on unknown.
    extra: dict[str, Any] = field(default_factory=dict)
    handoff: Any = None

    def has_multimodal_content(self) -> bool:
        """True when ``prompt_parts`` carries non-text blocks."""
        return any((p.get("type") or "") != "text" for p in self.prompt_parts)


# ---- session state ---------------------------------------------------


@dataclass(slots=True)
class EngineSessionState:
    """Persisted per-session engine state.

    Replaces the three sidecar shapes today (``<sid>.delegate_claude.json``
    holding ``claude_session_id``; ``<sid>.delegate_codex.json`` holding
    ``codex_thread_id``; ``<sid>.delegate_gemini.json`` holding
    ``antigravity_session_id``) with one normalized shape::

        {
          "sharedagents_session_id": "<parent>",
          "engine": "claude_cli" | "codex_cli" | "antigravity_cli" | "local_llm",
          "engine_session_id": "<cli's own id>",
          "priming_status": "never" | "in_progress" | "completed" | "failed",
          "priming_system_prompt_sha256": "<hex>",
          "last_used_at": "<iso>",
          "last_error": ""
        }

    For backwards compat, ``read_state`` MUST also recognize the legacy
    key names (``claude_session_id``, ``codex_thread_id``,
    ``antigravity_session_id``) and the legacy ``first_call_completed``
    boolean (mapped to ``priming_status="completed"`` when True).
    Legacy callsites: ``runners._read_claude_session_id`` (L1040),
    ``_read_codex_thread_id`` (L1961), ``_read_antigravity_session_id`` (L1132).

    The previous design
    used a single ``first_call_completed: bool`` which conflated two
    distinct semantics and lost information on partial failures.

    The new four-state ``priming_status`` machine:

    * ``"never"``        — engine has never been primed; treat next call
                           as the first call.
    * ``"in_progress"``  — a prime is currently running (engine_session_id
                           MAY already be set if Claude allocated mid-
                           stream). Concurrent callers MUST await the
                           session lock (:meth:`Engine.session_lock`)
                           and re-read state before deciding.
    * ``"completed"``    — engine session is ready; subsequent calls
                           pass system_prompt=None and use the engine's
                           resume mechanism.
    * ``"failed"``       — the first call started but didn't reach
                           a clean TurnDone(ok). The engine_session_id
                           IS persisted (so we can ``--resume`` it on
                           retry) but the system prompt may not have
                           been fully consumed. The engine subclass
                           decides whether to retry-with-resume or
                           force a fresh session.

    ``priming_system_prompt_sha256`` lets the engine detect when the
    operator has edited SOUL/AGENTS/TOOLS.md between turns — a sha
    mismatch invalidates the CLI session and forces a fresh prime
    even if ``priming_status == "completed"``.
    """

    sharedagents_session_id: str
    engine: EngineKind
    engine_session_id: str = ""
    priming_status: str = "never"  # "never"|"in_progress"|"completed"|"failed"
    priming_system_prompt_sha256: str = ""
    last_used_at: str = ""
    last_error: str = ""
    # The cwd that was used to spawn the CLI subprocess when the
    # ``engine_session_id`` was first minted. Claude Code / Codex /
    # Antigravity all key their on-disk session lookup by an encoded form
    # of the spawn cwd (Claude Code: ``~/.claude/projects/<encoded-cwd>/
    # <engine_session_id>.jsonl``). If a later turn spawns with a
    # different cwd, ``--resume`` silently misses the saved
    # conversation and sharedagents retries from scratch, dropping context.
    # Capturing the cwd here and forcing every subsequent resume to
    # use it makes the resume contract resilient to frontend reattach
    # from a different directory.
    spawn_cwd: str = ""
    # The real model id the CLI reported at runtime. Claude's stream-json
    # ``init``/``system`` event carries it (e.g. ``claude-opus-4-…``);
    # empty until the first turn surfaces it. Persisted so the TUI footer
    # can show the actual model instead of the engine-name placeholder
    # ("claude"), rather than the meaningless ``claude:claude``.
    actual_model: str = ""


# ---- Engine ABC ------------------------------------------------------


class Engine(ABC):
    """One turn-dispatch backend (local-LLM or one of three CLIs).

    Mostly stateless across turns — per-session state lives in the
    sidecar file via :class:`EngineSessionState`. The same Engine
    instance can serve every session for a given agent; the sidecar is
    keyed by ``sharedagents_session_id``.

    Per-session in-memory bookkeeping (``_session_locks``,
    ``_active_turns`` mapping turn_id → live subprocess, etc.) DOES
    live on the engine. Frontends only keep
    ``session_id → turn_id``; the engine owns the proc map.

    Lifecycle expected by :meth:`Agent.run_turn`:

    1. :meth:`read_state` — return prior state or fresh blank.
    2. :meth:`session_lock` — per-session asyncio.Lock to serialize
       concurrent prompts.
    3. :meth:`system_prompt_policy` — tells the Agent whether this
       turn needs the composed system prompt (every_turn for local
       LLM; first_turn for CLI; never for resume).
    4. :meth:`run_turn` — async generator yielding
       :data:`AgentTurnEvent`. The engine writes updated state via
       :meth:`write_state` at appropriate points (NOT only on success
       — partial-failure writes preserve engine_session_id with
       ``priming_status="failed"`` so the next call can ``--resume``).
    5. :meth:`abort` — best-effort kill of the in-flight turn's
       subprocess group. The engine owns the turn_id → proc map.

    Optional capability methods (default no-op so CLI engines don't
    have to implement them):

    * :meth:`attach_session` / :meth:`detach_session` — for engines
      that need per-session in-memory state (LocalLLMEngine tracks
      promoted_tools / mode overrides / cancellation flag).
    * :meth:`set_mode` / :meth:`set_config_option` — runtime config
      changes. Local LLM acts on them immediately; CLI engines
      generally ignore (the next ``--resume`` call's permission flag
      reflects the new mode).
    """

    # Engine kind tag — subclasses set this as a class attribute.
    kind: EngineKind

    @property
    @abstractmethod
    def name(self) -> str:
        """Short human label for status/logs. e.g. "claude" / "codex" /
        "antigravity" / "local_llm"."""

    # ---- session sidecar I/O ----------------------------------------

    @abstractmethod
    def state_path(self, sessions_dir: Path, sharedagents_sid: str) -> Path:
        """Resolve ``<sessions_dir>/<sharedagents_sid>.<engine>.json``.

        Maps to today's ``_claude_state_path`` / ``_codex_state_path``
        / ``_gemini_state_path``. Returns the path even if it doesn't
        exist yet — the engine creates it on first successful turn.
        """

    @abstractmethod
    def read_state(self, sessions_dir: Path, sharedagents_sid: str) -> EngineSessionState:
        """Read prior state or return a fresh blank one.

        MUST tolerate the legacy key names (``claude_session_id`` etc.)
        for unmigrated sidecars — see :class:`EngineSessionState`.
        """

    @abstractmethod
    def write_state(self, sessions_dir: Path, state: EngineSessionState) -> None:
        """Persist updated state atomically. Best-effort: failures are
        logged but do not abort the turn (parity with today's
        ``_write_claude_session_id`` swallowing ``OSError``,
        runners.py L1062)."""

    # ---- turn dispatch ----------------------------------------------

    @abstractmethod
    def run_turn(
        self,
        request: TurnRequest,
        state: EngineSessionState,
        *,
        sessions_dir: Path,
        turn_id: str,
        env: dict[str, str] | None = None,
    ) -> AsyncIterator[AgentTurnEvent]:
        """Stream a single turn's events.

        The implementation MUST:

        * Yield exactly one :class:`TurnStarted` first.
        * Yield :class:`TextDelta` for each prose chunk as it arrives
          (Claude streams per-token; Codex's ``agent_message`` arrives
          consolidated, see runners.py L2163–2169).
        * Yield :class:`ToolCallStarted` / :class:`ToolCallResult`
          pairs for every inner tool block the engine surfaces. The
          format string for the IN-BAND text marker
          (``\\n\\n🔧 <name> <json>\\n``) MUST match
          ``runners._format_claude_tool_use`` (L1066) byte-for-byte so
          the persisted body the local model replays on its next round
          is identical.
        * Yield :class:`PermissionRequested` and await
          ``response.set_result(...)`` whenever a tool needs an
          ASK-mode approval (local-LLM only in practice).
        * Yield exactly one :class:`TurnDone` last, even on error
          (state="error") or abort (state="aborted"). :class:`TurnError`
          carries the error text but DOES NOT terminate the turn — only
          :class:`TurnDone` does.
        * Handle ``CancelledError`` by SIGTERM/SIGKILLing the
          subprocess group (mirror runners.py L1803–1813), then
          ``yield`` :class:`TurnDone(state="aborted")` and RETURN.
          The engine MUST NOT
          re-raise ``CancelledError`` after yielding TurnDone — that
          would violate the "TurnDone is terminal" contract since
          the Agent's ``async for`` would see TurnDone (a normal
          terminator) and then CancelledError propagating from the
          generator's next ``__anext__``. The engine has handled
          the cancellation; the right terminal event has been
          emitted; cleanup runs in ``finally``.

        State-persistence rules:

        * Set ``priming_status = "in_progress"`` + write state AS SOON
          AS the engine has either (a) bound a CLI session id OR
          (b) consumed the system prompt — whichever happens first.
          DO NOT wait for TurnDone(ok); a mid-stream failure must leave
          enough state behind for the next call to resume.
        * On clean completion: set ``priming_status = "completed"``,
          stamp ``priming_system_prompt_sha256``, write.
        * On error / abort with engine_session_id non-empty: set
          ``priming_status = "failed"``, record ``last_error``, write.
          The next call's :meth:`run_turn` decides whether to
          ``--resume`` from the failed state or start fresh.
        """

    @abstractmethod
    async def abort(self, turn_id: str, *, force: bool = False) -> None:
        """Best-effort kill of the named turn's subprocess group.

        Engine owns the ``turn_id → proc`` map;
        the gateway / Agent only stores ``session_id → turn_id``.
        Called when the user hits ESC or on shutdown. Mirror of
        ``TurnState.stop(force=…)`` — ``force=True`` propagates the
        two-press hard-kill escalation (SIGKILL + ``killpg`` on the
        process group) that ``CoreBackend.cancel`` triggers when a
        second cancel arrives inside ``_HARD_KILL_WINDOW_S``.
        """

    # ---- introspection ----------------------------------------------

    @abstractmethod
    def is_available(self) -> tuple[bool, str]:
        """Whether the engine can actually run on this host.

        For CLI engines: check ``shutil.which("claude")`` etc — same
        check as the ``FileNotFoundError`` branches in each
        ``run_delegate_*_async`` (runners.py L1592–1601, L2103–2112,
        L1297–1307). For ``LocalLLMEngine``: ``True`` always (the
        upstream check happens later via the HTTP request).

        Returns ``(True, "")`` when available, ``(False, "<reason>")``
        otherwise — the reason text is surfaced to the user via the
        gateway / factory.
        """

    # ---- system-prompt policy ---------------------------------------

    def system_prompt_policy(
        self,
        state: EngineSessionState,
        *,
        current_system_prompt_sha256: str,
    ) -> str:
        """When does this engine need the agent's composed system prompt?

        Args:
            state: persisted per-session engine state.
            current_system_prompt_sha256: sha256 of the system prompt
                Agent JUST composed for this turn. CLI engines compare
                against ``state.priming_system_prompt_sha256`` to
                detect operator edits to SOUL/AGENTS/TOOLS .md files
                between turns (the policy needs SOMETHING to compare
                against).

        Returns one of:

        * ``"every_turn"`` — local LLM: stateless across rounds; always
          include the system prompt.
        * ``"first_turn"`` — CLI engines: prime once via
          ``--append-system-prompt`` (or first-message injection), then
          resume thereafter. Returned whenever
          ``state.priming_status in {"never", "in_progress", "failed"}``.
        * ``"sha_changed"`` — CLI engines: ``priming_status="completed"``
          but the current hash doesn't match
          ``state.priming_system_prompt_sha256`` → operator edited
          SOUL.md / AGENTS.md; force a fresh prime. The Agent treats
          this the same as ``"first_turn"`` (passes system_prompt)
          but also resets the engine session.
        * ``"never"`` — CLI engines on a resume call where the sha
          matches: skip the prompt; the engine already has it via
          its own session state.

        Default is ``"every_turn"`` (local-LLM-safe). CLI engines
        override.

        Replaces the old ``first_call_completed`` bool.
        """
        return "every_turn"

    # ---- display ----------------------------------------------------

    def display_model(self) -> str:
        """Human label for the model this engine runs, shown in the TUI
        footer / per-turn stats and stamped on the session.

        Defaults to :attr:`name` so the native CLI engines read
        ``"claude"`` / ``"codex"`` / ``"antigravity"``. ACP engines
        override this to surface the upstream agent's *actual* model
        (e.g. Cursor's ``"composer-2.5"``) instead of a generic label.
        """
        return self.name

    def context_window(self) -> int:
        """Context window (tokens) for the ctx-meter denominator, or 0
        for "unknown" (the agent layer then uses the global default).

        Default 0: local-LLM and native-CLI engines let the global
        ``upstream_max_context`` apply. ACP engines override to surface
        the model's real window (parsed from the ACP model id, or a
        per-agent default like composer's 200k)."""
        return 0

    def available_models(self) -> list[dict[str, Any]] | None:
        """Discover the models this engine can be switched to, as picker rows.

        Returns a list of ``{"alias", "id", "display"}`` dicts, or ``None``
        when the engine has no engine-owned catalogue (the caller then falls
        back to the operator's ``cfg.models[]`` deployment list — correct for
        :class:`LocalLLMEngine`, whose models ARE the configured upstreams).

        External-CLI engines override this to DISCOVER their provider's
        catalogue at runtime (Anthropic / OpenAI ``/v1/models``,
        antigravity's curated Google list) so a newly-released model becomes
        selectable without anyone editing a constant. Default ``None``.
        """
        return None

    # ---- per-session in-memory bookkeeping (optional) ---------------

    def attach_session(self, sharedagents_sid: str, session: Any) -> None:
        """Optional: allocate per-session live state.

        Default no-op. Local LLM overrides to populate
        ``_cancelled[sid] = False`` / ``_promoted_tools[sid] = set()`` /
        ``_mode_overrides[sid]``. CLI engines don't need this — all
        their per-session state is the sidecar file.

        Required so :meth:`Agent.create_session` can forward uniformly
        regardless of engine kind.
        """
        return None

    def detach_session(self, sharedagents_sid: str) -> None:
        """Optional: drop per-session live state on close_session.

        Default no-op. Local LLM overrides to clean up its dicts.
        """
        return None

    async def session_lock(self, sharedagents_sid: str) -> Any:  # asyncio.Lock
        """Per-session async lock to serialize concurrent prompts.

        Returns the same Lock for the same sid across calls (the
        engine MUST cache it in a private dict). Used by
        :meth:`Agent.run_turn` so two simultaneous prompts on the same
        session don't both observe ``priming_status="never"`` and race
        on creating distinct engine sessions.

        Default uses ``self._session_locks: dict[str, asyncio.Lock]``
        with lazy allocation. Concrete subclasses still ``__init__``
        the dict for clarity; this default makes the call safe even
        when a subclass forgot to.
        """
        import asyncio as _asyncio

        locks = getattr(self, "_session_locks", None)
        if locks is None:
            locks = {}
            try:
                self._session_locks = locks  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
        lock = locks.get(sharedagents_sid)
        if lock is None:
            lock = _asyncio.Lock()
            locks[sharedagents_sid] = lock
        return lock

    def set_mode(self, sharedagents_sid: str, mode_id: str) -> None:
        """Optional: per-session permission-mode override.

        Default no-op. Local LLM overrides; CLI engines treat it as
        the next ``--resume`` call's permission flag (handled when
        :meth:`run_turn` reads ``request.mode``).
        """
        return None

    def set_config_option(self, sharedagents_sid: str, option_id: str, value: Any) -> None:
        """Optional: live config toggle (``/reasoning``, ``/tool``, etc.).

        Default no-op. Local LLM overrides; CLI engines mostly ignore
        (model selection is fixed per CLI agent).
        """
        return None

    def steer_now(self, turn_id: str) -> bool:
        """Optional: deliver the session's queued follow-up to ``turn_id`` now.

        The operator pressed Steer on a queued message. An engine that can
        interrupt its in-flight step and continue the SAME turn with the
        queued text returns ``True`` and takes the queue itself. Default
        ``False``: the caller stops the turn and sends the text as a new
        prompt instead.
        """
        return False

    def inject_user_message(self, sharedagents_sid: str, text: str) -> bool:
        """Optional: forward a mid-turn user message into the live turn.

        Frontends call this (via :meth:`Agent.inject_user_message`, which
        probes for this hook) when the operator submits a fresh prompt
        while a turn is already streaming, so the steering message lands
        at the next round boundary instead of after the in-flight plan
        terminates.

        Default returns ``False`` — the engine does not implement the
        injection protocol, so the caller falls back to a local queue +
        post-turn dispatch. Local LLM overrides to queue the message (and
        soft-abort the active round); CLI/ACP bridges leave the default.

        Declared here (rather than left as a bare ``getattr`` duck-typed
        probe) so a new engine author reading the ``Engine`` contract can
        discover the recognized hook.
        """
        return False
