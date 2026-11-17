"""Session dataclass: identity, header fields, and header persistence."""

from __future__ import annotations

import os
import secrets
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sharedagents_core import oscompat

# Global default — used only by callers that don't pass an explicit
# `sessions_dir`/`root` (legacy direct-spawn paths). Agents always pass
# their own per-agent sessions dir, so this default never fires for the
# agent flow. Kept cwd-independent so we never leak a `.sharedagents/` into
# the project tree.
SESSIONS_DIR = Path.home() / ".sharedagents" / "sessions"

# Max length for a session title derived from the first user prompt line.
# Shown in the TUI header (word-wrapped), /sessions picker, and CLI list.
SESSION_TITLE_MAX_CHARS = 160

# Legacy prompt-derived title helpers remain available for old records and
# compatibility callers. New sessions use the timestamp placeholder below.
TEMP_TITLE_MAX_CHARS = 50

# A new session has a stable, human-readable placeholder until the title
# agent finishes.  Keep this helper in the session layer so the gateway and
# WebUI mint exactly the same shape instead of each growing its own heuristic.
NEW_CHAT_TITLE_PREFIX = "New Chat - "


def build_new_chat_title(timestamp: datetime | str | None = None) -> str:
    """Return the temporary title shown while LLM titling is pending.

    ``timestamp`` is optional to make the value deterministic for callers
    that already have a session creation time.  ISO timestamps are rendered
    in the local wall-clock timezone because the visible ``HHMMSS`` is meant
    for the operator, while invalid/missing values simply use ``now``.
    """
    stamp: datetime
    if isinstance(timestamp, datetime):
        stamp = timestamp
    elif isinstance(timestamp, str) and timestamp.strip():
        try:
            stamp = datetime.fromisoformat(timestamp.strip())
        except ValueError:
            stamp = datetime.now(timezone.utc)
    else:
        stamp = datetime.now(timezone.utc)
    if stamp.tzinfo is not None:
        stamp = stamp.astimezone()
    return f"{NEW_CHAT_TITLE_PREFIX}{stamp:%H%M%S}"


@contextmanager
def _exclusive_session_write(path: Path) -> Iterator[Path]:
    """Write-side guard for ``Session.save``.

    Yields a *per-writer-unique* tmp path that the caller writes,
    fsyncs, then ``os.replace``s into ``path``. The body of the
    ``with`` block runs under an exclusive ``fcntl.flock(LOCK_EX)``
    on a sibling ``.<filename>.lock`` so concurrent writers — within
    the process, across threads, or across separate processes (e.g.
    gateway + a legacy TUI ChatApp save in ``sharedagents_tui/app.py``,
    gateway + a stdio-fallback one-shot, or ``set_mode`` in the ACP
    bridge racing a streaming flush from ``Agent.run_turn``) — line
    up cleanly instead of issuing parallel writes against the same
    visible file.

    Two pieces, with distinct jobs — don't conflate them:

    1. **Per-writer unique tmp filename** (``PID.TID.NONCE``) opened
       with ``O_EXCL`` (``Session.save`` body, below). This is what
       prevents two writers from truncating each other's in-flight
       tmp file: each writer gets its own private inode. Combined
       with the atomicity of ``rename(2)`` (``os.replace``), the
       visible ``<sid>.json`` ends up as exactly ONE writer's
       complete payload — no torn / interleaved JSON is possible
       even if the flock were skipped.
    2. **fcntl.flock(LOCK_EX)** (this helper). Adds *ordering*:
       writers run one at a time in the critical section, so the
       final on-disk state is deterministic last-writer-wins
       instead of "whichever ``os.replace`` happened to land
       second." It also gives downstream readers a predictable
       sequence point (e.g. the ``session_saved`` bus publish
       fires AFTER the lock is released, so subscribers can't
       reach into a half-written FD). On Linux ``flock(2)`` is
       advisory and per-open-file-description, so two distinct
       processes (each calling ``os.open`` on the lockfile) get
       their own OFDs and contend on the same lock.

    Readers are intentionally NOT gated by this lock. They open
    ``<sid>.json`` directly and rely on the atomic ``os.replace``
    rename to always see a fully-formed JSON document — that
    contract is preserved unchanged.

    The lockfile is created on first write and left behind on
    cleanup; deleting it would re-open the race window (proc A
    unlinks the lockfile while proc B has it open → next writers
    create a fresh lockfile and the old holder no longer serialises
    with them). No code path in this repo deletes session JSONs
    today (see ``Agent.close_session`` — it explicitly preserves
    the file), so the ``.<sid>.json.lock`` files accumulate
    one-per-session alongside their JSON, at the same growth rate.
    An operator-side ``sessions/`` sweep is the only cleanup.

    NFS note: ``flock(2)`` on local ext4/xfs/btrfs (the production
    deployment per the operator's ops notes) gives true cross-
    process serialisation. On NFS mounts the lock is advisory and
    client-local — two clients on different machines would NOT
    serialise. If you ever point ``~/.sharedagents`` at an NFS volume,
    revisit this primitive.
    """
    lock_path = path.with_name(f".{path.name}.lock")
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.{secrets.token_hex(4)}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        oscompat.file_lock(lock_fd)
        try:
            yield tmp_path
        finally:
            try:
                oscompat.file_unlock(lock_fd)
            except OSError:
                # Best-effort unlock; ``os.close`` below will also
                # release the lock on the OFD.
                pass
    finally:
        os.close(lock_fd)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_session_id() -> str:
    """Return a fresh session id that sorts into creation order under
    **any** sort — lexical *and* numeric.

    Format (UTC)::

        YYYYMMDD 'T' HHMMSS ffffff '-' <12 hex random>

    The leading ``YYYYMMDDTHHMMSSffffff`` is a fixed-width, zero-padded
    UTC timestamp (8 + 6 + 6 digits). Because every field is fixed
    width and the only non-digit before the random tail is the ``T``
    date/time divider, the id orders chronologically under:

    * **lexical / byte sort** — ``ls``, Python ``sorted()``, SQLite
      ``ORDER BY hash`` (BINARY collation); and
    * **numeric / natural sort** — what VS Code's Explorer and many
      file managers use *by default*: maximal digit runs are compared
      by value, so ``YYYYMMDD`` and ``HHMMSSffffff`` (both fixed width)
      still compare correctly and the newest session lands **last**.

    Sorting right under *both* is the whole point of the format. The
    previous UUIDv7-hex id only sorted right under a *lexical* sort;
    under a numeric sorter the hex timestamp gets chopped into
    variable-length decimal runs (``…d91…`` → 91, ``…d152…`` → 152)
    and the order scrambles — so in VS Code a freshly created session
    did **not** appear last. UTC (not local time) guarantees
    monotonicity across DST transitions; the 12-hex random tail keeps
    ids unique within a single microsecond.

    Single-use contract: every caller MUST treat the returned id as
    opaque and set it exactly once at ``Session.new`` time.
    Reassignment elsewhere is a bug — the id is the file name (its
    ``Path.stem``), the dict key, and the index key. It contains no
    ``.``, so ``"<id>.jsonl"`` round-trips through ``Path.stem`` cleanly.
    """
    import secrets
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    return f"{now:%Y%m%dT%H%M%S%f}-{secrets.token_hex(6)}"


@dataclass
class Session:
    hash: str
    title: str
    created_at: str
    system_prompt: str
    model: str
    endpoint: str
    # Canonical directory that owns this session's JSONL + derived index.
    # Runtime-only: never serialized into the header.  A Session can be
    # created/loaded from an SHAREDAGENTS_SESSIONS_DIR that differs from the
    # process user's default ~/.sharedagents/sessions; remembering that binding
    # prevents a later no-argument save from escaping into the live store.
    _storage_root: Path | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    # The REAL model an external CLI/ACP engine reported running
    # (claude-opus-4-…/gpt-5.5-codex/composer-2.5). Kept SEPARATE from
    # ``model`` so engines with a selector dimension (antigravity's
    # auto/pro/flash alias lives in ``model``) surface the concrete model
    # without clobbering the picker. Persisted ONLY via the explicit
    # TurnDone ``header_update`` (deliberately NOT in ``to_dict`` so a
    # plain ``save()`` can't reset it); reloaded in ``Agent.load_session``.
    engine_model: str = ""
    turns: list[dict[str, Any]] = field(default_factory=list)
    # Permission mode the saved `system_prompt` was composed with. Used
    # by `agent_runtime` to decide whether the cached prompt is still
    # valid for the next turn — a mode switch (Shift+Tab in the TUI)
    # forces a fresh `load_system_prompt(...)` so the right MODE-*.md
    # file is in effect.
    system_mode: str = ""
    # The frontend-reported workspace context, captured at session
    # creation. `workspace_cwd` is the absolute filesystem directory the
    # frontend was launched from (TUI / `-p` / Telegram poller); the
    # agent injects it as a `# WORKSPACE-CONTEXT` section at the tail of
    # the system prompt so the model knows which folder the user is
    # operating in. `workspace_origin` is a short tag identifying the
    # frontend ("tui", "oneshot", "telegram", "api"). Both default to
    # empty strings; nothing is injected when both are empty (preserves
    # byte-stable prompts for callers that don't care).
    workspace_cwd: str = ""
    workspace_origin: str = ""
    # Tokenizer/protocol family of the upstream this session is bound
    # to. Empty = use cfg.backend at turn time. Stored alongside
    # `endpoint` so a session created against an alternate endpoint
    # (e.g. `sharedagents --model gpu-c`) keeps the right backend tag for
    # every subsequent turn — without this, switching endpoints
    # mid-config would silently mismatch the tokenizer.
    backend: str = ""
    # Max context window in tokens for the bound model. 0 = use
    # cfg.upstream_max_context fallback. OpenClaw-aligned name; per
    # session because two configured `models[]` rows can advertise
    # different contextWindows (e.g. gpu-a 256k vs gpu-c 32k) and the
    # compaction guard must trigger at the right threshold for each.
    context_window: int = 0
    # Number of successful compactions this session has run. Bumps on
    # each `_compact_via_llm` success. The pre-compaction memory-flush
    # gate uses this to avoid re-flushing within the same cycle —
    # mirrors OpenClaw's `entry.compactionCount`.
    compaction_count: int = 0
    # Compaction count at which the most recent pre-compaction memory
    # flush ran. Equals `compaction_count` after a flush; differs after
    # a fresh compaction has happened, which re-arms the flush gate.
    memory_flush_compaction_count: int = -1
    # ISO timestamp of the last pre-compaction memory flush — purely
    # informational, surfaced via `sharedagents memory status`.
    last_memory_flush_at: str = ""
    # Turn watermark at which compaction was executed. Turns at or before
    # this watermark are represented by compaction_summary; turns after it
    # retain full tool results in replay.
    compaction_turn: int = 0
    # Most recent AUTHORITATIVE prompt_tokens vLLM reported for this session
    # (stamped each round by ``_record_round_metrics``). Persisted so the
    # compaction gate (``_last_authoritative_prompt_tokens``) can still size
    # off the real token count after a reload — without it the in-memory-only
    # ``round_metrics`` is empty on a fresh load and the gate falls back to the
    # char estimate, which over-counts replayed (wire-squeezed) context ~17x
    # and spuriously compacts a session that's nowhere near its window.
    last_prompt_tokens: int = 0
    # --- pinned facts + rolling compaction summary ------------------
    # Facts the model has explicitly pinned via the `pin_fact` tool, or
    # that an operator pinned out-of-band. These survive every
    # compaction stage (deterministic and LLM-based), are re-injected
    # into the message list right after the system prompt, and persist
    # across turns. Use sparingly — every pinned fact costs prompt
    # tokens on every round.
    pinned_facts: list[str] = field(default_factory=list)
    # Rolling cross-compaction summary. The LLM summary produced by
    # `_compact_via_llm` is normally session-local (one summary per
    # compaction event, replacing the prior one). When a session
    # compacts repeatedly, facts established in compaction N can be
    # lost by compaction N+1's summary if the model focused on more
    # recent material. This field accumulates a SECOND-ORDER summary
    # across compactions so old context isn't silently dropped. Bounded
    # by `_COMPACTION_SUMMARY_MAX_CHARS` in agent_runtime.
    compaction_summary: str = ""
    # Stats of the MOST RECENT manual compaction (the WebUI collapsed
    # widget reads these): dict {at: ISO-ts, before_tokens, after_tokens,
    # removed_tokens, removed_chars, messages_trimmed, context_window}.
    # Written by the ``compact_context`` direct-dispatch path so the
    # widget survives reloads; JSON TEXT in sessions_meta. Empty dict on
    # sessions that never compacted.
    compaction_last_stats: dict[str, Any] = field(default_factory=dict)
    # OpenCode-style PR blurb from the hidden ``summary`` agent (2–3
    # sentences: "I added… / I fixed…"). Distinct from
    # ``compaction_summary`` (anchored context rollup). Empty until the
    # first completed turn refreshes it. Additive — pre-feature sessions
    # default to "".
    session_summary: str = ""
    # Per-session frontend visibility flags. These dataclass defaults are the
    # CONSERVATIVE / legacy-load values (an unknown record stays quiet on tool
    # output); a fresh session opts INTO showing tool output explicitly in
    # ``Session.new`` (the operator wants the full picture out of the box, and
    # can quiet it per session via `/tool off`). Keeping the dataclass +
    # ``from_dict`` defaults aligned means flipping the new-session default
    # never silently re-interprets a legacy record missing the field. Telegram
    # (and any other stream-aware client) reads these from the saved session
    # record so each session carries its own preference instead of a per-chat
    # default unrelated to which session is currently bound. The model's
    # THINKING behaviour is independent (`cfg.enable_thinking` / `/thinking`).
    reasoning_visibility: str = "stream"
    tool_output_visible: bool = False
    # ---- agents/ refactor attribution fields (Phase 2) -------------
    #
    # Set by ``sharedagents_agent.agents.session_log.record_session_creation``
    # at agent.create_session() time. Additive — sessions saved before
    # the refactor lack these fields and the loader / index tolerates
    # their absence via dataclass defaults.
    agent_name: str = ""
    engine: str = ""  # "local_llm" | "claude_cli" | …
    system_prompt_sha256: str = ""  # stable hash for fine-tune grouping
    # Privacy gate for reasoning traces. When False, ``record_turn``
    # drops the ``thinking`` channel from the on-disk record; live
    # render in the TUI still receives it (gated separately by
    # ``reasoning_visibility``).
    persist_reasoning_traces: bool = True
    # Parent sharedagents session id, set when this session IS a subagent —
    # i.e. an agent spawned by another session via the ``Agent`` /
    # ``BatchAgent`` tool (or the ``delegate_*`` slash-commands). Empty
    # for top-level (operator-initiated) sessions. This is the
    # first-class encoding of "a subagent is nothing more than an agent
    # with a parent session": the same Session/Agent machinery, plus a
    # back-link to whoever dispatched it. Additive — pre-refactor
    # sessions lack the field and default to "" (top-level). Persisted
    # to the JSONL header + sessions.db so the parent↔child tree is
    # queryable without parsing the loose ``delegate_*`` sidecars.
    parent_session_id: str = ""
    # Active goal/completion-condition for this session. ONE goal at a time;
    # ``/goal <cond>`` sets it (OVERRIDING any previous goal), ``/goal clear``
    # stops it. The teacher-model supervisor only fires while ``active_goal`` is
    # non-empty — supervision is for goal-directed autonomous work. Persisted
    # (additive; pre-feature sessions default to "").
    active_goal: str = ""
    # Per-session teacher toggle override set by ``/teacher on|off``. "on" forces
    # supervision even with no active goal; "off" forces it off; "" = default
    # (follow the teacher policy + goal gate). Persisted so the operator's
    # choice survives a restart, symmetric with ``active_goal``.
    teacher_override: str = ""
    # Per-session supervising-model override set by ``/teacher model <alias>``.
    # The models[] alias/id that should review THIS session, or "" for the
    # config's role==teacher row. Persisted (additive; defaults to "").
    teacher_model_override: str = ""
    # Per-session reasoning-depth override. Empty means follow the agent
    # default, which in turn may be empty to preserve upstream behavior.
    effort: str = ""
    # Reserved for model-sheet speed capabilities. Empty means default.
    speed: str = ""
    # Number of LLM-based title refinements already applied to this
    # session. Titles now generate at lifecycle turns 1 and 10; this legacy
    # counter stays 0 unless an older session already spent budget under the
    # retired 4/8/16 schedule. Persisted for compatibility. See
    # ``sharedagents_session.session_title``.
    title_refinements: int = 0
    # True when the title is PINNED and the title agent must never
    # regenerate it: an operator ``/title`` rename, or a Tasks-board task
    # whose title is the pinned task name (parent session only). Gates
    # ``session_title.should_create_title`` — the single gate point covering
    # turn 1, the unconditional turn-10 refresh, the boot reconciler and
    # ``maybe_ensure_title``. Additive; pre-feature sessions default False
    # (today's behavior).
    title_locked: bool = False
    # Number of goal re-drive turns consumed for this session. Incremented
    # each time the runtime auto-re-drives after TurnDone (the outer goal
    # loop). Bounded by ``max_goal_turns`` config. Persisted so a restart
    # doesn't reset the budget. Default 0 (pre-feature sessions start fresh).
    goal_turn_count: int = 0
    # Terminal state of the last goal cycle: "" (none), "met", "blocked",
    # or "paused" (cap exhausted). Persisted so the UI can show the goal
    # status on resume.
    goal_terminal: str = ""
    # Durable goal entity id (projection from GoalRegistry). Empty when no
    # goal is bound to this session.
    goal_id: str = ""
    # Cost-attribution context for token-cost accounting. A delegated child
    # session carries the immutable root session + ancestor goal it spends
    # against, so its priced ledger rows roll up to the parent goal's budget.
    # Empty on a normal (non-delegated) session — the finalizer falls back to
    # this session's own hash / goal_id.
    cost_root_session_id: str = ""
    cost_ancestor_goal_id: str = ""
    # Session-tree USD cost cap. 0/unset floors to DEFAULT_BUDGET_USD ($25)
    # at every read site — a session always has a budget, goal or not.
    session_budget_usd: float = 0.0
    # Conversation topic this session belongs to. Mirrors a Telegram forum
    # topic name (the bot pushes the topic name via the ``topic`` config
    # option when it binds a session to a thread), so the same grouping the
    # operator sees in Telegram surfaces on the WebUI Topics page. Every
    # session has one; the default — and what the Telegram "General" topic /
    # any non-forum chat maps to — is ``"general"``. Persisted (additive;
    # pre-feature sessions default to "general").
    topic: str = "general"

    @classmethod
    def new(
        cls,
        *,
        system_prompt: str,
        model: str,
        endpoint: str,
        sessions_dir: Path | None = None,
        system_mode: str = "",
        workspace_cwd: str = "",
        workspace_origin: str = "",
        backend: str = "",
        context_window: int = 0,
        parent_session_id: str = "",
        session_id: str = "",
    ) -> Session:
        # Resolve at call time so test/runtime monkey-patches of the module
        # constant take effect.
        if sessions_dir is None:
            sessions_dir = SESSIONS_DIR
        h = session_id or new_session_id()
        created_at = _now()
        session = cls(
            hash=h,
            # The title agent replaces this temporary value asynchronously;
            # keeping the creation timestamp in the title makes a freshly
            # opened chat identifiable without echoing user prompt content.
            title=build_new_chat_title(created_at),
            created_at=created_at,
            system_prompt=system_prompt,
            model=model,
            endpoint=endpoint,
            system_mode=system_mode,
            workspace_cwd=workspace_cwd,
            workspace_origin=workspace_origin,
            backend=backend,
            context_window=context_window,
            parent_session_id=parent_session_id,
            # New sessions show tool output by default (operator preference);
            # reasoning already streams via the dataclass default. Set here, NOT
            # on the dataclass default, so legacy records missing the field
            # still load conservatively as off.
            tool_output_visible=True,
        )
        session._storage_root = Path(sessions_dir)
        return session

    def add_turn(
        self,
        *,
        user: Any,
        thinking: str,
        content: str,
        started_at: str,
        ended_at: str,
        aborted: bool = False,
        error: str | None = None,
        metrics: dict[str, Any] | None = None,
        in_progress: bool = False,
    ) -> None:
        new_entry = {
            "user": user,
            "thinking": thinking,
            "content": content,
            "started_at": started_at,
            "ended_at": ended_at,
            "aborted": aborted,
            "error": error,
            "metrics": metrics,
            "in_progress": in_progress,
        }
        # In-progress checkpoint updates the last turn IN PLACE when
        # it's still an in-progress entry for the SAME user prompt —
        # the engine flushes one of these per round boundary, and
        # appending each as a fresh entry would explode ``turns[]`` into
        # N near-duplicate rows. The terminal record_turn (with
        # ``in_progress=False``) likewise overwrites the trailing
        # in-progress marker so the saved turn is the final state,
        # not stacked partials. New user prompts always append.
        if self.turns and self.turns[-1].get("in_progress") and self.turns[-1].get("user") == user:
            self.turns[-1] = new_entry
            return
        self.turns.append(new_entry)

    @staticmethod
    def _derive_title(
        text: Any,
        limit: int = SESSION_TITLE_MAX_CHARS,
    ) -> str:
        """Derive a session title from the opening user prompt."""
        # Title = first non-empty line of the user's opening prompt, capped
        # at ``limit`` characters (default ``SESSION_TITLE_MAX_CHARS``).
        # No trailing ellipsis — the TUI header word-wraps the full stored
        # title instead of clipping with "…".
        #
        # `text` is either a plain string OR an OpenAI-style multi-part
        # content list (vision turns ship `[image_url, text]`). Pull the
        # first non-empty text part out before deriving the title; fall
        # back to "[image]" when the user sent a bare photo with no caption.
        if isinstance(text, list):
            text_part = ""
            for part in text:
                if isinstance(part, dict) and str(part.get("type") or "").lower() == "text":
                    text_part = part.get("text") or ""
                    if text_part:
                        break
            text = text_part or "[image]"
        if not isinstance(text, str):
            text = str(text or "")
        first = next((line for line in text.strip().splitlines() if line.strip()), "")
        if not first:
            return "session"
        return first[:limit]

    @staticmethod
    def _derive_temp_title(
        text: Any,
        limit: int = TEMP_TITLE_MAX_CHARS,
    ) -> str:
        """The turn-1 TEMP title: the opening line, clipped to title width.

        Same extraction as :meth:`_derive_title` (incl. the ``[image]``
        fallback), then clipped on a word boundary with an ellipsis so the
        sidebar shows ``"now on the sharedagents webui the indication of…"``
        instead of 160 unbroken characters of raw prompt.

        ``session_title.build_temp_title`` delegates here for compatibility
        with legacy records; ``Session.add_turn`` no longer mutates titles.
        """
        full = Session._derive_title(text, limit=SESSION_TITLE_MAX_CHARS)
        if len(full) <= limit:
            return full
        cut = full[: limit - 1]
        # Prefer a word boundary, but never clip away more than half the
        # budget chasing one (a single long token just gets hard-cut).
        space = cut.rfind(" ")
        if space >= limit // 2:
            cut = cut[:space]
        return cut.rstrip(" ,;:.-–—") + "…"

    def _header_dict(self) -> dict[str, Any]:
        """Materialised-header fields for the sessions.db row."""
        return {
            "session_hash": self.hash,
            "created_at": self.created_at,
            "agent_name": self.agent_name,
            "engine": self.engine,
            "parent_session_id": getattr(self, "parent_session_id", "") or "",
            "active_goal": getattr(self, "active_goal", "") or "",
            "goal_id": getattr(self, "goal_id", "") or "",
            "cost_root_session_id": getattr(self, "cost_root_session_id", "") or "",
            "cost_ancestor_goal_id": getattr(self, "cost_ancestor_goal_id", "") or "",
            "session_budget_usd": float(getattr(self, "session_budget_usd", 0) or 0),
            "goal_turn_count": int(getattr(self, "goal_turn_count", 0) or 0),
            "goal_terminal": getattr(self, "goal_terminal", "") or "",
            "topic": (getattr(self, "topic", "") or "general").strip() or "general",
            "teacher_override": getattr(self, "teacher_override", "") or "",
            "teacher_model_override": getattr(self, "teacher_model_override", "") or "",
            "effort": getattr(self, "effort", "") or "",
            "speed": getattr(self, "speed", "") or "",
            "workspace_cwd": self.workspace_cwd,
            "workspace_origin": self.workspace_origin,
            "title": self.title,
            "title_refinements": int(getattr(self, "title_refinements", 0) or 0),
            "title_locked": bool(getattr(self, "title_locked", False)),
            "model": self.model,
            "endpoint": self.endpoint,
            "backend": self.backend,
            "context_window": int(self.context_window or 0),
            "system_mode": self.system_mode,
            "reasoning_visibility": self.reasoning_visibility,
            "tool_output_visible": bool(self.tool_output_visible),
            "persist_reasoning_traces": bool(self.persist_reasoning_traces),
            "persist_tool_result_bodies": bool(getattr(self, "persist_tool_result_bodies", True)),
            "system_prompt": self.system_prompt,
            "system_prompt_sha256": self.system_prompt_sha256,
            "pinned_facts": list(self.pinned_facts),
            "compaction_count": int(self.compaction_count or 0),
            "compaction_summary": self.compaction_summary,
            "compaction_last_stats": dict(getattr(self, "compaction_last_stats", None) or {}),
            "session_summary": getattr(self, "session_summary", "") or "",
            "memory_flush_compaction_count": int(self.memory_flush_compaction_count),
            "last_memory_flush_at": self.last_memory_flush_at,
            "compaction_turn": int(getattr(self, "compaction_turn", 0) or 0),
            "last_prompt_tokens": int(getattr(self, "last_prompt_tokens", 0) or 0),
        }

    def save(
        self,
        root: Path | None = None,
        *,
        force: bool = False,
        force_fields: Iterable[str] | None = None,
    ) -> Path:
        """Persist the session HEADER to ``sessions.db``.

        Turns are persisted as append-only JSONL events by
        ``session_log.record_turn`` — ``save()`` now only flushes the
        materialised header (title / model / pinned_facts / compaction /
        config) that header-mutating callers (set_mode, set_config_option,
        memory_flush, acp new_session) change. On the first call for a
        session it emits ``session_started`` (which inserts the
        sessions_meta row); thereafter it emits ``header_update``.

        Returns the ``<sid>.jsonl`` path (the durable artifact). The
        legacy whole-document JSON write is gone.

        A poisoned writer / durable-write failure raised by ``header_update`` /
        ``session_started`` below propagates UNIFORMLY out of ``save()`` — the
        same observable channel as a turn write via ``record_turn`` — rather
        than being swallowed for header-only mutations (the old asymmetric
        ``except Exception: pass`` this requirement retires). Failures are
        loud everywhere, with no carve-out.

        ``force_fields`` is a field-scoped forced patch for derived projections:
        those named mutable fields are emitted even when this instance's
        snapshot already contains the desired value. Other fields retain the
        stale-writer protection above, so repairing one projection cannot
        replay an unrelated stale title/model/config value.
        """
        if root is None:
            root = self._storage_root or SESSIONS_DIR
        root = Path(root)
        # An explicit root establishes the canonical binding for subsequent
        # scoped/no-argument header saves on this in-memory Session instance.
        self._storage_root = root
        jsonl_path = root / f"{self.hash}.jsonl"
        # Immutable, potentially-large header fields — never re-appended on a
        # header-only persist. The initial system_prompt rode the ``session_started`` event.
        _IMMUTABLE = (
            "system_prompt",
            "system_prompt_sha256",
            "created_at",
            "agent_name",
            "parent_session_id",
        )
        # A freshly-loaded instance often still has the create-time default
        # (0 / $25). Re-emitting that on the first save clobbers a cap another
        # writer just raised via /budget. Only persist this field when this
        # instance mutated it (snapshot diff) or the caller forced it.
        _STALE_UNSAFE = ("session_budget_usd",)
        from sharedagents_session.session_writer import get_session_writer

        writer = get_session_writer(root, self.hash, engine=self.engine or "")
        header = self._header_dict()
        forced = {
            str(field_name)
            for field_name in (force_fields or ())
            if str(field_name) in header and str(field_name) not in _IMMUTABLE
        }
        # Per-instance snapshot of the header THIS Session last persisted. A
        # ``header_update`` carries ONLY the fields this instance actually
        # mutated since its last save (diffed against the snapshot), never the
        # full mutable set: a one-field ``/model`` toggle appends a one-field
        # patch, and a stale in-memory value for a field this
        # instance never touched can no longer clobber a column another writer
        # advanced (no stale-writer clobber).
        snapshot = getattr(self, "_persisted_header", None)
        if jsonl_path.exists() and snapshot is not None:
            _MISSING = object()
            patch = {
                k: v
                for k, v in header.items()
                if k not in _IMMUTABLE and (k in forced or v != snapshot.get(k, _MISSING))
            }
            if patch:
                writer.header_update(patch=patch)
        elif jsonl_path.exists() and forced:
            # A freshly-loaded instance has no local diff baseline. A scoped
            # projection write must remain scoped even here: another writer may
            # have advanced unrelated fields after this instance was loaded.
            writer.header_update(patch={k: header[k] for k in forced})
        elif jsonl_path.exists():
            # First save from a freshly-loaded instance (no snapshot yet):
            # emit the full mutable patch so a resumed session self-heals its
            # header, then seed the snapshot below for subsequent diffs.
            # Omit stale-unsafe fields unless this save explicitly forced them.
            patch = {k: v for k, v in header.items() if k not in _IMMUTABLE and k not in _STALE_UNSAFE}
            writer.header_update(patch=patch)
        else:
            writer.session_started(header)
        # A poisoned writer / durable-write failure raises out of the calls
        # above and MUST propagate uniformly — the same observable channel as a
        # turn write via ``record_turn``. The old bare
        # ``except Exception: pass`` swallowed exactly this, leaving
        # header-only mutations ( /mode, /model, pin_fact ) silently dropped
        # while turn writes failed loudly. The snapshot advances ONLY after a
        # successful persist, so a mutated field re-emits on the next retry.
        self._persisted_header = dict(header)
        return jsonl_path

    def save_header_fields(
        self,
        fields: Iterable[str],
        root: Path | None = None,
    ) -> Path:
        """Durably emit selected mutable header fields.

        This keeps field-scoped projection repair explicit at call sites while
        preserving ordinary :meth:`save` diff semantics everywhere else.
        """
        return self.save(root=root, force_fields=fields)
