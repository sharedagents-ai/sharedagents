"""Heartbeat scheduler — OpenClaw-style background poll loop.

A periodic turn fired into a DEDICATED heartbeat session. The agent reads
`HEARTBEAT.md` (a model-edited checklist) and decides whether to do
background work (summarize recent user-session activity, check
email/calendar/mentions, sweep `memory/`…) or reply `HEARTBEAT_OK` and
stay quiet.

Three design constraints:

1.  **Cache-friendly prefix.** The heartbeat session's prompt grows
    monotonically (each tick appends one turn). vLLM APC / SGLang
    RadixAttention reuse KV up to the divergence point, so the
    re-prefill cost per tick is just the prior assistant content +
    the new heartbeat user message. Compaction will eventually
    fire — that's fine; same as any session.

2.  **Don't redo work.** State at `<home>/heartbeat-state.json` records
    `session_id` (the heartbeat session itself) plus `session_progress`
    keyed by user-session id with a `last_analyzed_turn_idx`. The model
    reads / writes this via `read_file` / `write_file` tool calls so
    a tick-2 heartbeat knows what tick-1 already analyzed.

3.  **OpenClaw byte-exact for the contract.** `HEARTBEAT_PROMPT` and
    the "effectively empty" check match the canonical OpenClaw runtime
    (extracted from `dist/heartbeat-DKgfsHcO.js`) so behavior is the
    same: an empty / comments-only `HEARTBEAT.md` skips the API call
    entirely; an `HEARTBEAT_OK` reply is treated as "no-op, don't
    surface."
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---- OpenClaw byte-exact constants -----------------------------------
#
# Verbatim from `dist/heartbeat-DKgfsHcO.js`. Don't drift unless the
# upstream OpenClaw runtime changes — frontends (and the AGENTS.md
# template) reference this exact wording.

HEARTBEAT_PROMPT = (
    "Read HEARTBEAT.md if it exists (workspace context). Follow it "
    "strictly. Do not infer or repeat old tasks from prior chats. If "
    "nothing needs attention, reply HEARTBEAT_OK."
)

HEARTBEAT_TOKEN = "HEARTBEAT_OK"
HEARTBEAT_FILENAME = "HEARTBEAT.md"
HEARTBEAT_STATE_FILENAME = "heartbeat-state.json"

# Mirrors `isHeartbeatContentEffectivelyEmpty` from the OpenClaw bundle.
# A HEARTBEAT.md that contains only headers, fence markers, or empty
# list-item stubs has nothing actionable; we skip the API call so we
# don't spend tokens on a no-op tick.
_HB_FENCE_RE = re.compile(r"^```[A-Za-z0-9_-]*$")
_HB_HEADER_RE = re.compile(r"^#+(\s|$)")
_HB_EMPTY_LIST_RE = re.compile(r"^[-*+]\s*(\[[\sXx]?\]\s*)?$")


def is_heartbeat_content_effectively_empty(content: str | None) -> bool:
    """Return True when `HEARTBEAT.md` has no actionable content.

    Matches OpenClaw's check: whitespace, ATX headers, empty list items,
    markdown fence markers, and comment-only lines count as empty. Lines
    inside fenced blocks are ignored (template docs/examples).
    """
    if content is None:
        return False
    if not isinstance(content, str):
        return False
    in_fence = False
    for line in content.split("\n"):
        trimmed = line.strip()
        if _HB_FENCE_RE.match(trimmed):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if not trimmed:
            continue
        if _HB_HEADER_RE.match(trimmed):
            continue
        if _HB_EMPTY_LIST_RE.match(trimmed):
            continue
        return False
    return True


# Max age (seconds) for a TurnState entry on the heartbeat session
# before the scheduler evicts it as stale. Set just below the heartbeat
# interval (default 1800s) so an entry leaked one cycle ago doesn't
# block the next. 1500s = 25 min.
_HEARTBEAT_STALE_TURN_S: float = 1500.0


# Cap on the memory-synth failure backoff. After N consecutive failures
# the scheduler skips the synth call for ``2 ** (streak-1)`` ticks,
# clamped to this many — so a persistently misconfigured upstream stops
# hammering it every tick without disabling synth forever (a key fix +
# SIGHUP reload still recovers on the next attempt after the cooldown).
_SYNTH_MAX_BACKOFF_TICKS: int = 16


def _synth_backoff_decision(
    *,
    now: float,
    interval_s: float,
    fail_streak: int,
    last_error: str | None,
    new_error: str,
) -> tuple[int, str, float, bool]:
    """Pure backoff/log-dedup policy for a memory-synth failure.

    Given the current failure streak + last logged error and a new
    error string, return ``(new_streak, new_last_error, cooldown_until,
    should_log)``:

    * ``new_streak`` — incremented streak.
    * ``cooldown_until`` — monotonic deadline before which the next
      synth attempt is skipped (exponential in the streak, capped by
      :data:`_SYNTH_MAX_BACKOFF_TICKS`).
    * ``should_log`` — True only when this error differs from the last
      one logged, so a persistent identical error (e.g. a 401) is
      logged once per distinct signature instead of every tick.
    """
    new_streak = fail_streak + 1
    backoff_ticks = min(2 ** (new_streak - 1), _SYNTH_MAX_BACKOFF_TICKS)
    cooldown_until = now + backoff_ticks * max(0.0, interval_s)
    should_log = new_error != last_error
    return new_streak, new_error, cooldown_until, should_log


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _log_heartbeat_line(
    *,
    agent: str,
    level: str,
    msg: str,
    kind: str = "HEARTBEAT",
    session_id: str = "",
    k: dict | None = None,
) -> None:
    """Append one origin-tagged JSONL line to the unified heartbeat.log.

    Shared by :class:`HeartbeatScheduler._log` and any out-of-class
    helper that needs to emit a heartbeat-scoped line (memory-synth
    failures, backoff messages). All lines land in the ONE shared
    ``layout.heartbeat_logfile()`` regardless of which agent fired
    them; the ``agent`` field on each row carries attribution.
    """
    from sharedagents_core.jsonlog import get_writer
    from sharedagents_core.layout import heartbeat_logfile

    w = get_writer(heartbeat_logfile(), service="heartbeat")
    w.write(
        agent=agent or "unknown",
        level=level,
        msg=msg,
        kind=kind or None,
        session_id=session_id or None,
        k=k or None,
    )


# ---- State file ------------------------------------------------------


def state_path(home: Path) -> Path:
    return home / HEARTBEAT_STATE_FILENAME


def load_state(home: Path) -> dict[str, Any]:
    """Read `<home>/heartbeat-state.json`. Returns an empty dict on
    missing or unparseable files — caller seeds defaults. Never
    raises (the heartbeat loop must keep running even if state is
    corrupt; we just lose continuity, not the loop)."""
    p = state_path(home)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_state(home: Path, state: dict[str, Any]) -> None:
    """Atomic write of the heartbeat state file. Same pattern as the
    Telegram poller's state — write to `.tmp` then rename so a
    crash mid-write doesn't leave a half-truncated JSON."""
    p = state_path(home)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps(state, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(p)


def read_heartbeat_md(home: Path) -> str | None:
    p = home / HEARTBEAT_FILENAME
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return None


HEARTBEAT_MD_TEMPLATE = """\
# Heartbeat tasks

# Keep this file empty (comments only) to skip periodic API calls.
# Add actionable tasks below when you want the agent to check something.
"""


def ensure_heartbeat_md(home: Path) -> Path:
    """Create ``HEARTBEAT.md`` when heartbeat is activated.

    Idempotent — never overwrites an existing file so operator edits
    survive a disable/enable cycle.
    """
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    path = home / HEARTBEAT_FILENAME
    if not path.exists():
        path.write_text(HEARTBEAT_MD_TEMPLATE, encoding="utf-8")
    return path


def remove_heartbeat_md(home: Path) -> None:
    """Remove ``HEARTBEAT.md`` when heartbeat is deactivated."""
    path = Path(home) / HEARTBEAT_FILENAME
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def sync_heartbeat_files(home: Path, *, enabled: bool) -> None:
    """Keep ``HEARTBEAT.md``'s presence in sync with ``heartbeat_enabled``.

    Presence tracks the flag on every save/load: the file is created when
    heartbeat is enabled and removed when it is disabled, so an operator or
    tool can read ``HEARTBEAT.md``'s presence as ground truth for whether
    heartbeat is on. Best-effort — ``remove_heartbeat_md`` swallows IO errors.
    """
    if enabled:
        ensure_heartbeat_md(home)
    else:
        remove_heartbeat_md(home)


# ---- Scheduler -------------------------------------------------------


class HeartbeatScheduler:
    """Background async task that fires periodic turns into a dedicated
    heartbeat session.

    Wired up in `agent_runtime.serve()` alongside the Telegram poller.
    The scheduler:

    1. Ensures a heartbeat session exists (creates one on first run,
       persists its id in `heartbeat-state.json`).
    2. Sleeps `cfg.heartbeat_interval_s` between ticks.
    3. On each tick: if `HEARTBEAT.md` is effectively empty, skips
       (matches OpenClaw — empty file = no API call). Otherwise fires
       a turn with `user=HEARTBEAT_PROMPT` against the heartbeat
       session via the agent's own `session_turn` handler.
    4. Updates `last_run_at` after a successful tick.

    The scheduler does NOT render the turn anywhere — the saved turn
    lands on disk and the user can browse it via `/sessions`.
    """

    def __init__(
        self,
        cfg: Any,  # AgentConfig
        app: Any,  # AgentApp — for spawning a turn locally
    ) -> None:
        self.cfg = cfg
        self.app = app
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None
        # Heartbeat lines land in the unified ``logs/heartbeat.log``
        # (one file across every agent). ``_log_path`` is exposed for
        # legacy callers / tests but the live writer is the shared
        # ``JsonLogWriter`` so cross-process flock + origin tags are
        # the same as gateway.log.
        from sharedagents_core.layout import heartbeat_logfile

        self._log_path = heartbeat_logfile()
        # Memory-synth failure backoff state. A misconfigured key (e.g.
        # DeepSeek 401) never self-resolves, so without backoff the synth
        # fires + logs the identical error EVERY tick (38 such lines in
        # one day). Track the failure streak + last error to
        # back off exponentially and log each distinct error once.
        self._synth_fail_streak = 0
        self._synth_last_error: str | None = None
        self._synth_cooldown_until = 0.0  # monotonic deadline

    def request_stop(self) -> None:
        self._stop.set()

    async def start(self) -> None:
        if not self.cfg.heartbeat_enabled:
            self._log("heartbeat: disabled (heartbeat_enabled=False)")
            return
        self._task = asyncio.create_task(self._run(), name="heartbeat-scheduler")

    async def stop(self) -> None:
        if self._task is None:
            return
        self.request_stop()
        try:
            await asyncio.wait_for(self._task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            self._task.cancel()

    def _log(self, msg: str) -> None:
        try:
            _log_heartbeat_line(
                agent=str(getattr(self.cfg, "name", "") or "unknown"),
                level="INFO",
                msg=str(msg),
            )
        except (OSError, ValueError):
            # Logging must never break the scheduler tick.
            pass

    async def _ensure_session(self) -> str | None:
        """Resolve the dedicated heartbeat session id.

        Persisted across agent restarts in `heartbeat-state.json`. If
        the saved id no longer exists on disk (user deleted the
        session) we create a fresh one and update state. Returns
        `None` only if creation itself fails (upstream blip, etc.) —
        the loop catches that and skips the tick.
        """
        state = load_state(self.cfg.home)
        sid = state.get("session_id")
        if sid:
            try:
                # Cheap existence check — read_session_file is sync.
                if self.app._read_session_file(sid) is not None:
                    return sid
            except Exception:  # noqa: BLE001
                pass
        # Create a new heartbeat session via the same path the create
        # endpoint uses, but in-process (no HTTP round-trip needed).
        try:
            sid = await self._create_dedicated_session()
        except Exception as e:  # noqa: BLE001
            self._log(f"heartbeat: create_session failed: {e}")
            return None
        state["session_id"] = sid
        state.setdefault("created_at", _now_iso())
        save_state(self.cfg.home, state)
        return sid

    async def _create_dedicated_session(self) -> str:
        """Mint a new heartbeat session via the same primitives
        `create_session` uses. Tagged `workspace_origin="heartbeat"` so
        it's identifiable in `/sessions` listings."""
        from sharedagents_agent.turn_runtime import _compose_system_prompt
        from sharedagents_session.session import Session

        upstream_model = self.cfg.upstream_model or ""
        if not upstream_model:
            try:
                upstream_model = await self.app._resolve_model()
            except Exception:  # noqa: BLE001
                upstream_model = ""
        active_mode = self.cfg.permission_mode or "auto"
        sess = Session.new(
            system_prompt=_compose_system_prompt(
                self.cfg,
                active_mode,
                workspace_cwd=str(self.cfg.home),
                workspace_origin="heartbeat",
            ),
            system_mode=active_mode,
            workspace_cwd=str(self.cfg.home),
            workspace_origin="heartbeat",
            model=upstream_model,
            endpoint=self.cfg.upstream_endpoint,
            sessions_dir=self.cfg.sessions_dir,
            backend=getattr(self.cfg, "backend", "") or "",
            context_window=int(getattr(self.cfg, "upstream_max_context", 0) or 0),
        )
        sess.title = "heartbeat"
        # Stamp agent_name / engine + emit the initial ``session_started``
        # event via the canonical sink (same as ``Agent.create_session``
        # and ``CoreBackend.new_session``). Plain ``sess.save`` here left
        # ``agent_name`` / ``backend`` blank in ``sessions_meta`` — and
        # save()'s ``header_update`` path excludes ``agent_name``, so the
        # blank could never self-heal. Set ``title`` first so it rides
        # the ``session_started`` header.
        try:
            from sharedagents_agent.agents.session_log import record_session_creation

            record_session_creation(
                sess,
                agent_name=getattr(self.cfg, "name", "") or "",
                engine=getattr(self.cfg, "engine", "") or "local_llm",
                system_prompt=sess.system_prompt,
                sessions_dir=self.cfg.sessions_dir,
            )
        except Exception as exc:  # noqa: BLE001
            # Loud in logs: attribution stamping refused (e.g. an unknown
            # agent_name — record_session_creation validates the identity).
            # Fall back to a plain save so the heartbeat still lands, with a
            # BLANK agent_name rather than a phantom one.
            self._log(
                f"heartbeat: record_session_creation failed "
                f"(agent_name={getattr(self.cfg, 'name', '') or ''!r}): {exc} — "
                "falling back to unattributed save"
            )
            sess.save(root=self.cfg.sessions_dir)
        return sess.hash

    async def _run(self) -> None:
        self._log(f"heartbeat: started interval={self.cfg.heartbeat_interval_s}s")
        while not self._stop.is_set():
            try:
                await self._tick_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                self._log(f"heartbeat: tick error: {e}")
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=self.cfg.heartbeat_interval_s,
                )
            except asyncio.TimeoutError:
                continue
        self._log("heartbeat: stopped")

    async def _tick_once(self) -> None:
        """One heartbeat fire. Skips if HEARTBEAT.md is effectively
        empty (matches OpenClaw). Evicts a stale in-flight turn so a hung
        turn cannot pin the session busy past the interval.
        """
        # Cheap lazy reindex: stat all tracked memory files, only re-
        # chunk those whose mtime/hash has drifted. Costs ~O(N stat)
        # when nothing changed. Keeps `memory_search` results fresh
        # for the heartbeat tick without paying upstream embedding
        # cost on every interval.
        # Run the (blocking) wire reindex in a thread so it never stalls the
        # gateway event loop (heartbeat must not block).
        await asyncio.to_thread(self._maybe_index_memory)
        # Synthesize MEMORY.md from recent daily logs. Runs in the
        # parent process (unlike `remember` which runs in a subprocess
        # and can't reach the parent's event bus). Fires on every
        # heartbeat tick — the synthesizer is idempotent and returns
        # fast when nothing changed.
        await self._maybe_synthesize_memory()
        # First-tick wiring for in-process bus subscribers (session
        # index, memory provider). Idempotent — subsequent ticks
        # short-circuit through the per-home `_REGISTERED` guard
        # in each module. Done here rather than in agent_runtime so
        # the boot path stays untouched; a missed first tick (agent
        # exits before the heartbeat fires) just means the next
        # save's events go un-consumed, which is recoverable on the
        # next start.
        await self._maybe_subscribe_memory_providers()
        content = read_heartbeat_md(self.cfg.home)
        if is_heartbeat_content_effectively_empty(content):
            self._log("heartbeat: HEARTBEAT.md empty — skipping tick")
            return
        sid = await self._ensure_session()
        if sid is None:
            return
        # Skip if a turn is already in flight on the heartbeat session
        # (slow upstream + short interval could overlap). With a
        # staleness floor — a TurnState leaks (SSE peer hung up,
        # dispatch loop didn't pop the entry, etc.) we don't want it
        # to block heartbeat indefinitely. After
        # `_HEARTBEAT_STALE_TURN_S` (25 min by default) we evict the
        # entry and proceed.
        import time as _time

        now_mono = _time.monotonic()
        STALE_S = _HEARTBEAT_STALE_TURN_S
        for turn in list(self.app.turns.values()):
            if getattr(turn, "session_hash", None) != sid:
                continue
            tstart = getattr(turn, "started_mono", None)
            if isinstance(tstart, (int, float)) and (now_mono - tstart) > STALE_S:
                # Reap the stale entry. We don't have a clean state-
                # transition API for this, so just drop the dict
                # entry — anything still referencing the TurnState
                # holds its own reference.
                self.app.turns.pop(turn.id, None)
                self._log(
                    f"heartbeat: evicted stale turn {turn.id} "
                    f"(age {now_mono - tstart:.0f}s) on heartbeat "
                    f"session {sid[:12]}"
                )
                continue
            self._log(f"heartbeat: tick skipped — turn {turn.id} still in flight on heartbeat session {sid[:12]}")
            return
        self._log(f"heartbeat: firing tick on session {sid[:12]}")
        await self._fire_turn(sid)
        state = load_state(self.cfg.home)
        state["last_run_at"] = _now_iso()
        save_state(self.cfg.home, state)

    def _maybe_index_memory(self) -> None:
        """Run a cheap incremental memory reindex if anything changed.

        Best-effort — failures are logged but never abort the tick. The
        actual indexing work is thrown into a thread so a slow disk /
        big-corpus reindex doesn't block the event loop.
        """
        if not getattr(self.cfg, "memory_search_enabled", True):
            return
        # Reindex over the memory WIRE (index_rebuild if_needed),
        # via the persistent sync bridge — never the in-process engine. The
        # server-side op does the cheap needs_reindex skip.
        try:
            from sharedagents_mcp.memory.client import index_rebuild_sync

            res = index_rebuild_sync(self.cfg.home, if_needed=True)
            stats = res.get("stats") or {}
            if res.get("ok") and not res.get("skipped"):
                self._log(
                    "memory: reindex "
                    f"files={stats.get('indexed_files', 0)} "
                    f"chunks={stats.get('indexed_chunks', 0)} "
                    f"deleted={stats.get('deleted_files', 0)} "
                    f"embedded={stats.get('embedded_chunks', 0)} "
                    f"cache_hits={stats.get('cache_hits', 0)} "
                    f"t={stats.get('duration_s')}s"
                )
        except Exception as e:  # noqa: BLE001
            self._log(f"memory: reindex error: {e}")

    async def _maybe_subscribe_memory_providers(self) -> None:
        """Wire `session_saved` subscribers exactly once per agent.

        Both subscriptions are idempotent (each module guards a
        per-home `_REGISTERED` set), so the cost on subsequent
        heartbeat ticks is one set lookup. Failures here never
        propagate — the agent works fine without these subscribers,
        we just lose auto-indexing / auto-fact-extraction until the
        next successful tick.
        """
        try:
            from sharedagents_session.session_index import subscribe_for_agent as _si_subscribe

            _si_subscribe(home=self.cfg.home)
        except Exception as e:  # noqa: BLE001
            self._log(f"memory: session_index subscribe error: {e}")
        try:
            # Subscribe the provider over the WIRE on
            # the SAME background-loop server that prefetch uses, so the provider
            # it activates lives where recall looks (was gateway-loop client ->
            # a different server/registry than prefetch_sync). Runs in a thread
            # so it never blocks the gateway loop.
            from sharedagents_mcp.memory.client import subscribe_sync

            await asyncio.to_thread(
                subscribe_sync,
                self.cfg.home,
                memory_provider=str(getattr(self.cfg, "memory_provider", "") or ""),
                memory_recall_mode=str(getattr(self.cfg, "memory_recall_mode", "") or ""),
            )
        except Exception as e:  # noqa: BLE001
            self._log(f"memory: memory_manager subscribe error: {e}")

    async def _maybe_synthesize_memory(self) -> None:
        """Synthesize MEMORY.md from recent daily logs.

        Runs on every heartbeat tick. The synthesizer is idempotent —
        if no daily is newer than MEMORY.md, it returns immediately
        without calling the LLM.

        Gated on `cfg.memory_synth_enabled` (default True). This is
        the canonical place where MEMORY.md gets updated — the old
        event-bus subscription (`daily_appended` → `synth_memory`)
        never worked because `remember` runs in a subprocess that
        can't reach the parent's event bus.
        """
        if not getattr(self.cfg, "memory_synth_enabled", True):
            return
        # Skip the LLM call while backing off after recent failures — a
        # misconfigured upstream (e.g. DeepSeek 401) would otherwise be
        # re-hit every tick. Cooldown grows exponentially per failure.
        now = time.monotonic()
        if now < self._synth_cooldown_until:
            return
        err: str | None = None
        try:
            # Synth over the memory WIRE on the same background
            # server (via a thread so it never blocks the gateway loop).
            from sharedagents_mcp.memory.client import synth_sync

            stats = await asyncio.to_thread(
                synth_sync,
                self.cfg.home,
                upstream_endpoint=self.cfg.upstream_endpoint,
                upstream_model=getattr(self.cfg, "memory_synth_model", "") or self.cfg.upstream_model or "",
            )
            stats = stats.get("stats") if isinstance(stats.get("stats"), dict) else stats
            if stats.get("error"):
                err = str(stats["error"])
            elif stats.get("changed"):
                self._log(
                    f"memory: synth changed "
                    f"before={stats.get('before_chars', 0)} "
                    f"after={stats.get('after_chars', 0)} "
                    f"n_dailies={stats.get('n_dailies', 0)} "
                    f"t={stats.get('duration_s')}s"
                )
            else:
                self._log(f"memory: synth unchanged reason={stats.get('reason', '')} t={stats.get('duration_s', 0)}s")
        except Exception as e:  # noqa: BLE001
            err = str(e)

        if err is not None:
            streak, last_err, cooldown_until, should_log = _synth_backoff_decision(
                now=now,
                interval_s=float(getattr(self.cfg, "heartbeat_interval_s", 1800) or 1800),
                fail_streak=self._synth_fail_streak,
                last_error=self._synth_last_error,
                new_error=err,
            )
            self._synth_fail_streak = streak
            self._synth_last_error = last_err
            self._synth_cooldown_until = cooldown_until
            if should_log:
                skipped = min(2 ** (streak - 1), _SYNTH_MAX_BACKOFF_TICKS)
                self._log(f"memory: synth error: {err} (fail #{streak}; backing off {skipped} tick(s))")
            return
        # Success / no-op clears the failure state so a recovered
        # upstream resumes normal cadence and logs the next error afresh.
        self._synth_fail_streak = 0
        self._synth_last_error = None
        self._synth_cooldown_until = 0.0

    async def _fire_turn(self, sid: str) -> None:
        """In-process call to `session_turn` — same code path the HTTP
        endpoint takes, just without the HTTP layer. Streams events
        through a NullSink that drops them; the saved session JSON is
        the only persistence we care about for heartbeats.

        We use httpx against our own gateway URL because the alternative
        (calling the AgentApp method directly with a fake aiohttp
        request) is fragile across aiohttp versions.
        """
        import httpx

        url = f"http://127.0.0.1:{self.cfg.port}/v1/sessions/{sid}/turn"
        timeout = httpx.Timeout(connect=5.0, read=None, write=10.0, pool=5.0)
        try:
            async with (
                httpx.AsyncClient(timeout=timeout) as c,
                c.stream(
                    "POST",
                    url,
                    json={"user": HEARTBEAT_PROMPT},
                ) as resp,
            ):
                if resp.status_code != 200:
                    body = await resp.aread()
                    self._log(f"heartbeat: turn POST failed http={resp.status_code} body={body[:200]!r}")
                    return
                # Drain the SSE stream so the server-side dispatch
                # loop completes (and the turn lands in the saved
                # JSON). We don't render anything; status updates
                # go to the heartbeat log.
                final_state: str | None = None
                async for line in resp.aiter_lines():
                    if line.startswith("event: done"):
                        final_state = "done"
                    elif line.startswith("event: error"):
                        final_state = "error"
                self._log(f"heartbeat: tick finished state={final_state}")
        except httpx.HTTPError as e:
            self._log(f"heartbeat: HTTP error: {e}")
