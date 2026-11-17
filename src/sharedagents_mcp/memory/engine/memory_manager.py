"""Active memory provider lifecycle + bus wiring.

A thin orchestrator: pick the provider named in `cfg.memory_provider`,
initialize it, and route bus events to it. It doesn't replace the
built-in memory path — `remember()`, MEMORY.md, and `memory_search`
keep working unchanged. The provider runs alongside.

Single-select by design. Two concurrent providers fight over the
recall budget and produce muddy auto-injection; if you want to
compose providers, write one provider that internally fans out.

Provider registry is a plain dict — no plugin discovery, no YAML.
Three-line additions for the second provider; if/when there are
six, we'll revisit.

Class-based interface (`MemoryManager`) plus thin module-level
functions (`subscribe_for_agent`, `prefetch_for_query`,
`shutdown_all`) for callers that just want the default per-home
behaviour.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sharedagents_core.kernel.injection_guard import wrap_untrusted
from sharedagents_mcp.memory.engine.memory_provider import MemoryProvider

_log = logging.getLogger(__name__)


# ---- recall mode + context fencing ---------------------------------

# The auto-recall wire round-trip is on the turn hot path, so it
# is hard-bounded (never block a turn indefinitely) and its warm latency is
# measured against a stated budget. Timeout is generous enough for a warm local
# provider yet short enough that a hung server degrades to "no recall this turn".
MEMORY_RECALL_TIMEOUT_S = 8.0  # hard cap; overridable via SHAREDAGENTS_MEMORY_RECALL_TIMEOUT_S
MEMORY_RECALL_BUDGET_MS = 1500.0  # warm-latency budget; over-budget is logged at WARNING

# Recall modes that callers may request via cfg.memory_recall_mode.
# - "context": auto-inject only (no provider tools surfaced).
# - "tools":   provider tools surfaced; no auto-inject.
# - "hybrid":  both. Default.
VALID_RECALL_MODES: frozenset[str] = frozenset({"context", "tools", "hybrid"})

# Cap on the auto-injected context block so a chatty provider can't
# eat the entire turn budget. Aligns with the spec's ~2000-char cap.
DEFAULT_CONTEXT_CHAR_BUDGET: int = 2000

# Context fencing: when we capture a completed turn for persistence,
# strip any `[RECALLED CONTEXT ...]` block we previously injected.
# Otherwise re-saving the assistant text re-feeds the model its own
# recalled context, which compounds across turns and pollutes
# downstream search.
_RECALLED_BLOCK_RE = re.compile(
    r"\[RECALLED\s+CONTEXT[^\]]*\][^\n]*(?:\n(?!\n).*)*",
    re.MULTILINE,
)

# Recalled memory is now injected inside an injection-guard untrusted-data
# fence (``⟦UNTRUSTED-DATA … source=memory:…⟧ … ⟦/UNTRUSTED-DATA …⟧``). Strip
# whole fences too, so injected/recalled content the assistant may have echoed
# never gets re-ingested as a long-term "fact" (which would be a memory-
# poisoning vector). The legacy ``[RECALLED CONTEXT]`` form is kept for
# backward-compatible cleanup of turns captured before this change.
_UNTRUSTED_FENCE_RE = re.compile(
    r"⟦UNTRUSTED-DATA[^\n⟧]*⟧.*?⟦/UNTRUSTED-DATA[^\n⟧]*⟧",
    re.DOTALL,
)


def strip_recalled_blocks(text: str) -> str:
    """Remove recalled-memory segments from a turn capture.

    Strips both the legacy ``[RECALLED CONTEXT ...]`` blocks and the
    injection-guard untrusted-data fences. Pure function — easy to
    unit-test. Idempotent: text without recalled blocks returns unchanged.
    """
    if not isinstance(text, str) or not text:
        return text or ""
    if "[RECALLED CONTEXT" not in text and "⟦UNTRUSTED-DATA" not in text:
        return text
    cleaned = _UNTRUSTED_FENCE_RE.sub("", text)
    cleaned = _RECALLED_BLOCK_RE.sub("", cleaned)
    # Collapse the empty lines the regex leaves behind.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def truncate_for_budget(text: str, *, budget_chars: int) -> str:
    """Trim a recalled block to the configured char budget.

    Truncation is char-based, not token-based — a tokenizer-aware
    trim would tighten the bound but adds a dependency we don't
    want on the recall hot path.
    """
    if not text:
        return ""
    if len(text) <= budget_chars:
        return text
    head = text[:budget_chars].rstrip()
    return f"{head}\n[…recall truncated to {budget_chars} chars…]"


# ---- built-in providers --------------------------------------------

# Provider factory registry keyed by name. Empty by default (file-only
# memory); a provider opts in by inserting its factory here under a name.
PROVIDER_REGISTRY: dict[str, Any] = {}

# Back-compat alias under the old name.
PROVIDER_FACTORIES = PROVIDER_REGISTRY


# ---- main class ----------------------------------------------------


@dataclass
class _ManagerState:
    """Container the manager carries between activate / wire / dispatch.

    Held outside `MemoryManager` itself so the public class stays
    focused on the lifecycle methods; this state is internal
    bookkeeping the bus subscriptions need to close over.
    """

    home: Path
    cfg: Any
    provider: MemoryProvider
    recall_mode: str
    context_budget: int
    bus_subs_wired: bool = False


class MemoryManager:
    """Lifecycle for a single active memory provider.

    Used two ways:
      - Direct: `mm = MemoryManager(); mm.activate(...); mm.wire_events()`
        — what the spec describes for the heartbeat bootstrap.
      - Per-home registry: `subscribe_for_agent(home, cfg)` resolves
        an existing manager (or creates one) so two callers in the
        same process don't double-wire the bus.

    Single-select: `activate` refuses when a provider is already
    bound on this instance. `deactivate()` clears it before re-
    activating with a different name.
    """

    def __init__(self) -> None:
        self._state: _ManagerState | None = None

    # -- public API --------------------------------------------------

    @property
    def provider(self) -> MemoryProvider | None:
        return None if self._state is None else self._state.provider

    def is_active(self) -> bool:
        return self._state is not None

    async def activate(
        self,
        provider_name: str,
        agent_home: Path,
        cfg: Any,
        workspace_origin: str = "",
    ) -> bool:
        """Bind a provider to this manager. Returns True on success.

        - `provider_name=""` (or any name not in `PROVIDER_REGISTRY`)
          activates no provider and returns False — memory is file-only
          (MEMORY.md + dailies). No provider is bundled, so the
          registry ships empty; only an
          external provider (Honcho, Mem0, …) registered by the operator
          resolves to a real backend.
        - `provider_name="none"` deliberately skips activation —
          the operator opts out and the manager stays inert.
        """
        if self._state is not None:
            _log.warning(
                "memory_manager: refusing reactivation; current=%s",
                self._state.provider.name,
            )
            return False

        name = (provider_name or "").strip().lower()
        if name == "none":
            _log.info("memory_manager: provider explicitly disabled")
            return False
        factory = PROVIDER_REGISTRY.get(name)
        if factory is None:
            _log.info(
                "memory_manager: no memory provider registered (name=%r) — file-only memory",
                name,
            )
            return False

        provider = factory()
        if not provider.is_available():
            _log.warning(
                "memory_manager: provider %r reports unavailable",
                provider.name,
            )
            return False

        try:
            await provider.initialize(Path(agent_home), cfg)
        except Exception:  # noqa: BLE001
            _log.exception(
                "memory_manager: provider %r initialize failed",
                provider.name,
            )
            return False

        recall_mode = (
            (
                getattr(cfg, "memory_recall_mode", "")
                or (cfg.get("memory_recall_mode") if isinstance(cfg, dict) else "")
                or "hybrid"
            )
            .strip()
            .lower()
        )
        if recall_mode not in VALID_RECALL_MODES:
            _log.warning(
                "memory_manager: invalid recall_mode %r; defaulting to hybrid",
                recall_mode,
            )
            recall_mode = "hybrid"
        budget = int(
            getattr(cfg, "memory_context_char_budget", 0)
            or (cfg.get("memory_context_char_budget") if isinstance(cfg, dict) else 0)
            or DEFAULT_CONTEXT_CHAR_BUDGET
        )

        self._state = _ManagerState(
            home=Path(agent_home),
            cfg=cfg,
            provider=provider,
            recall_mode=recall_mode,
            context_budget=budget,
        )
        _log.info(
            "memory_manager: active provider=%s home=%s mode=%s",
            provider.name,
            agent_home,
            recall_mode,
        )
        return True

    def deactivate(self) -> None:
        """Best-effort shutdown. Idempotent."""
        if self._state is None:
            return
        try:
            self._state.provider.shutdown()
        except Exception:  # noqa: BLE001
            _log.exception(
                "memory_manager: provider %r shutdown raised",
                self._state.provider.name,
            )
        self._state = None

    def wire_events(self) -> None:
        """Subscribe the active provider to the process-wide bus.

        Safe to call multiple times — repeat calls are no-ops.
        """
        if self._state is None or self._state.bus_subs_wired:
            return
        self._wire_bus(self._state)
        self._state.bus_subs_wired = True

    def get_extra_tools(self) -> list[dict]:
        """Tool schemas the provider wants surfaced.

        Empty list when no provider is active — the default, since no
        provider ships in `PROVIDER_REGISTRY`.
        External providers (Honcho, Mem0, …) registered by the operator
        would return their schemas here.
        """
        if self._state is None:
            return []
        if self._state.recall_mode == "context":
            # Auto-inject only — don't surface explicit tools to the
            # model, the recall already happens out-of-band.
            return []
        return list(self._state.provider.get_tool_schemas() or [])

    async def dispatch_tool(self, name: str, args: dict) -> Any:
        """Route a tool call to the active provider.

        Returns the provider's tool-result body; raises if no
        provider is active.
        """
        if self._state is None:
            raise RuntimeError(f"memory_manager: cannot dispatch {name!r}; no active provider")
        return await self._state.provider.handle_tool_call(name, args or {})

    # -- recall composition -----------------------------------------

    async def prefetch_and_format(self, query: str) -> str:
        """Run `provider.prefetch(query)` and format for injection.

        Honors `recall_mode`:
          - "context" / "hybrid": call prefetch, format inside a
            `[RECALLED CONTEXT — <provider>]` envelope, truncate to
            the configured char budget.
          - "tools": return "" — recall happens via explicit tool
            calls, not auto-injection.

        The caller is responsible for the actual splice into the
        message list (today: `agent_runtime` would call this and
        place the block after the system prompt's cache boundary;
        the wiring is intentionally not done in this commit since
        the runtime hot path is fragile).
        """
        if self._state is None:
            return ""
        if self._state.recall_mode == "tools":
            return ""
        # The memory access is wire-only, so a hung/slow provider
        # must NEVER block the turn indefinitely. Bound the round-trip with a hard
        # timeout and MEASURE + REPORT the warm latency against a stated budget; on
        # timeout, skip auto-injection (the turn proceeds without recall) rather
        # than stall. SHAREDAGENTS_MEMORY_RECALL_TIMEOUT_S overrides the default.
        import asyncio as _asyncio
        import os as _os
        import time as _time

        try:
            _timeout = float(_os.environ.get("SHAREDAGENTS_MEMORY_RECALL_TIMEOUT_S", "") or MEMORY_RECALL_TIMEOUT_S)
        except ValueError:
            _timeout = MEMORY_RECALL_TIMEOUT_S
        _t0 = _time.monotonic()
        try:
            body = await _asyncio.wait_for(self._state.provider.prefetch(query or ""), timeout=_timeout)
        except _asyncio.TimeoutError:
            _log.warning(
                "memory_manager: recall wire timed out after %.1fs (provider=%s) — "
                "skipping auto-injection; turn NOT blocked",
                _timeout,
                self._state.provider.name,
            )
            return ""
        except Exception:  # noqa: BLE001
            _log.exception(
                "memory_manager: prefetch raised for %s",
                self._state.provider.name,
            )
            return ""
        _elapsed_ms = (_time.monotonic() - _t0) * 1000.0
        _over = _elapsed_ms > MEMORY_RECALL_BUDGET_MS
        (_log.warning if _over else _log.debug)(
            "memory_manager: recall wire round-trip %.0fms (provider=%s, budget=%.0fms%s)",
            _elapsed_ms,
            self._state.provider.name,
            MEMORY_RECALL_BUDGET_MS,
            ", OVER BUDGET" if _over else "",
        )
        body = (body or "").strip()
        if not body:
            return ""
        # Recalled memory is UNTRUSTED: a poisoned fact (written by an earlier
        # injected turn, a shared-workspace peer, or a compromised provider)
        # must not be able to issue instructions to the model. Wrap it in an
        # injection-guard fence so the model treats it as data. Truncate the
        # body FIRST, then wrap — truncating the wrapped envelope could sever
        # the closing fence and re-open an instruction context.
        body = truncate_for_budget(body, budget_chars=self._state.context_budget)
        return wrap_untrusted(body, source=f"memory:{self._state.provider.name}")

    # -- bus wiring -------------------------------------------------

    def _wire_bus(self, state: _ManagerState) -> None:
        """Register the bus subscriptions a provider needs.

        The unified-layout writer publishes one ``session_saved`` per
        session, regardless of which agent owns it. Each agent's
        memory manager subscribes once; without a content filter,
        every manager would ingest every other agent's session and
        cross-pollute facts/memory.

        The right filter is the session's own ``agent_name`` (from
        ``sessions_meta``) compared to this manager's owner —
        ``state.home.name`` is the per-agent config dir's basename,
        which matches the writer's ``agent_name`` for every legit
        producer. ``workspace_origin`` then scopes to private
        origins as before.
        """
        from sharedagents_core.kernel.event_bus import get_bus

        bus = get_bus()
        provider = state.provider
        owner_agent = state.home.name
        home_resolved = state.home.resolve()

        async def on_session_saved(payload: dict[str, Any]) -> None:
            path_str = payload.get("path") or ""
            session_hash = str(payload.get("session_hash") or "")
            if not path_str and not session_hash:
                return
            # the payload points at a <sid>.jsonl event log. Fold it
            # for turns, and read agent_name from sessions_meta (the
            # bus payload no longer carries it) for per-agent attribution.
            try:
                from pathlib import Path as _P

                from sharedagents_core.kernel.layout import sessions_root as _sroot
                from sharedagents_session.session_index import (
                    SessionIndex,
                    connect_sessions_db,
                )
                from sharedagents_session.session_reader import fold_jsonl_into_turns

                jpath = _P(path_str) if path_str else (_sroot() / f"{session_hash}.jsonl")
                sid = session_hash or jpath.stem
                idx = SessionIndex(_sroot())
                idx.tail(sid)
                session_agent = ""
                con = connect_sessions_db(idx.db_path)
                try:
                    r = con.execute(
                        "SELECT agent_name FROM sessions_meta WHERE session_hash=?",
                        (sid,),
                    ).fetchone()
                    if r is not None:
                        session_agent = r["agent_name"] or ""
                finally:
                    con.close()
                # Only ingest sessions that BELONG to this manager's
                # agent. Empty session_agent is treated as a match so
                # legacy / ACP sessions without explicit attribution
                # still reach the configured manager (preserving the
                # pre-unified behaviour for those edges); a populated
                # agent_name that differs from the owner is rejected
                # to prevent cross-agent fact pollution.
                if session_agent and owner_agent and session_agent != owner_agent:
                    return
                turns = fold_jsonl_into_turns(jpath)
            except Exception:  # noqa: BLE001
                return
            try:
                await provider.on_session_end(turns, session_hash)
            except Exception:  # noqa: BLE001
                _log.exception(
                    "memory_manager: provider %r on_session_end raised",
                    provider.name,
                )

        async def on_turn_completed(payload: dict[str, Any]) -> None:
            # turn_completed is published per-agent by
            # turn_runtime._publish_turn_completed with
            # ``agent_home == cfg.home`` (the PER-AGENT path, unlike
            # session_saved's shared-root agent_home). Filter on it
            # so a unified-bus burst across N agents doesn't replay
            # every other agent's turns through this manager's
            # provider.
            ev_home = (payload or {}).get("agent_home", "")
            if ev_home:
                try:
                    if Path(ev_home).resolve() != home_resolved:
                        return
                except OSError:
                    return
            user = str((payload or {}).get("user") or "")
            assistant = str((payload or {}).get("assistant") or "")
            # Context fencing: any [RECALLED CONTEXT] block that the
            # manager auto-injected before this turn must not be
            # captured back into long-term storage. Otherwise the
            # provider learns "facts" that are just its own previous
            # recall echoed back through the model.
            assistant_clean = strip_recalled_blocks(assistant)
            if not user.strip() and not assistant_clean.strip():
                return
            try:
                await provider.sync_turn(user, assistant_clean)
            except Exception:  # noqa: BLE001
                _log.exception(
                    "memory_manager: provider %r sync_turn raised",
                    provider.name,
                )

        async def on_daily_appended(payload: dict[str, Any]) -> None:
            # NOTE: `daily_appended` fires from the `remember()`
            # subprocess in the current sharedagents build, so parent-
            # process subscribers don't see it (heartbeat.py:392-397
            # documents this for memory_synth). The subscription is
            # still wired here so the contract is in place — when
            # `remember()` runs in-process (or its events bridge into
            # the parent), providers pick up writes for free.
            ev_home = (payload or {}).get("home", "")
            try:
                if ev_home and Path(ev_home).resolve() != home_resolved:
                    return
            except Exception:  # noqa: BLE001
                return
            action = str(payload.get("action") or "add")
            content = str(payload.get("content") or "")
            category = str(payload.get("category") or "")
            try:
                await provider.on_memory_write(action, content, category)
            except Exception:  # noqa: BLE001
                _log.exception(
                    "memory_manager: provider %r on_memory_write raised",
                    provider.name,
                )

        bus.subscribe(
            "session_saved",
            on_session_saved,
            quiet_seconds=2.0,
            name=f"memory_manager[{provider.name}]/session_saved",
            # Coalesce per session_hash: a burst across N agents
            # turns into ONE handler invocation PER session, not
            # one global invocation that drops every session but
            # the last.
            coalesce_key="session_hash",
        )
        bus.subscribe(
            "turn_completed",
            on_turn_completed,
            quiet_seconds=0.5,
            name=f"memory_manager[{provider.name}]/turn_completed",
            coalesce_key="session_hash",
        )
        bus.subscribe(
            "daily_appended",
            on_daily_appended,
            quiet_seconds=2.0,
            name=f"memory_manager[{provider.name}]/daily_appended",
        )


# ---- per-home registry of managers --------------------------------

# Keyed by resolved home path. Each manager owns its provider +
# bus subscriptions; two callers in the same process get the same
# instance.
_MANAGERS: dict[str, MemoryManager] = {}


def get_active_manager(home: Path) -> MemoryManager | None:
    return _MANAGERS.get(str(Path(home).resolve()))


def get_active_provider(home: Path) -> MemoryProvider | None:
    """The provider initialized for `home`, or None if none."""
    mgr = _MANAGERS.get(str(Path(home).resolve()))
    return mgr.provider if mgr is not None else None


async def subscribe_for_agent(
    *,
    home: Path,
    cfg: Any,
    workspace_origin: str = "",
) -> MemoryProvider | None:
    """Idempotent per-home: pick + initialize the active provider, then
    wire it to the bus.

    Reads `cfg.memory_provider` (string) and `cfg.memory_recall_mode`.
    An empty/unregistered provider name activates nothing (file-only
    memory) — see `activate`.

    Returns the active provider, or None when activation refused
    (operator-disabled or init failure).
    """
    key = str(Path(home).resolve())
    existing = _MANAGERS.get(key)
    if existing is not None and existing.is_active():
        return existing.provider

    mgr = existing or MemoryManager()
    name = (getattr(cfg, "memory_provider", "") or "").strip().lower()
    activated = await mgr.activate(
        provider_name=name,
        agent_home=Path(home),
        cfg=cfg,
        # Bootstrap path: the heartbeat fires from origin "heartbeat",
        # which is private. If the caller passes their own origin
        # (e.g. one-shot CLI under "tui"), we honor it.
        workspace_origin=workspace_origin or "heartbeat",
    )
    if activated:
        mgr.wire_events()
        _MANAGERS[key] = mgr
    return mgr.provider


async def prefetch_for_query(
    *,
    home: Path,
    query: str,
    workspace_origin: str = "",
    chat_id: str = "",
) -> str:
    """Synchronous-from-caller's-POV recall hook.

    Returns the recalled-context block to splice after the cache
    boundary, or "" when:
    - no provider is active for this home,
    - the recall mode is "tools",
    - the provider returns nothing.
    """
    mgr = _MANAGERS.get(str(Path(home).resolve()))
    if mgr is None or not mgr.is_active():
        return ""
    return await mgr.prefetch_and_format(query or "")


def shutdown_all() -> None:
    """Best-effort shutdown for tests + agent-restart paths."""
    for mgr in list(_MANAGERS.values()):
        try:
            mgr.deactivate()
        except Exception:  # noqa: BLE001
            _log.exception("memory_manager: deactivate raised")
    _MANAGERS.clear()


__all__ = [
    "DEFAULT_CONTEXT_CHAR_BUDGET",
    "MemoryManager",
    "PROVIDER_FACTORIES",
    "PROVIDER_REGISTRY",
    "VALID_RECALL_MODES",
    "get_active_manager",
    "get_active_provider",
    "prefetch_for_query",
    "shutdown_all",
    "strip_recalled_blocks",
    "subscribe_for_agent",
    "truncate_for_budget",
]
