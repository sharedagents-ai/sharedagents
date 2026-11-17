"""Pluggable memory provider — minimum surface to swap recall backends.

The default sharedagents memory pipeline is filesystem-first: `remember()`
writes to `memory/YYYY-MM-DD.md`, the synthesizer curates `MEMORY.md`,
the index searches over both. That's the built-in path and it's not
going anywhere.

This ABC sits *alongside* the built-in path, not in front of it. A
provider plugs in to:

- expose extra recall surfaces to the model (`get_tool_schemas` +
  `handle_tool_call`),
- auto-inject recalled context before each turn (`prefetch`),
- consume save events for downstream learning (`sync_turn`,
  `on_session_end`).

We deliberately derived the interface from one concrete implementation
(`SqliteFactsProvider`) instead of porting every Hermes hook
speculatively. New hooks land here when a *second* provider needs
them.

Lifecycle:

1. `memory_manager.subscribe_for_agent(home, cfg)` reads
   `cfg.memory_provider`, instantiates the matching provider, calls
   `await provider.initialize(home, cfg)`.
2. The manager subscribes the provider to `session_saved` /
   `daily_appended` on the shared event bus, routing the payload to
   `sync_turn` / on_session_end.
3. `prefetch(query)` is invoked synchronously from the runtime hot
   path (when `recall_mode in {"context", "hybrid"}`); the inline
   call is required because the result must reach the next API
   request as injected context.

Privacy: providers are dispatched only when the calling session's
`workspace_origin` is in the trusted set (`""`, `"tui"`, `"oneshot"`,
`"heartbeat"`, `"telegram"`). Enforcement lives in the manager so a
provider author can't forget the gate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class MemoryProvider(ABC):
    """Minimal contract every recall backend implements.

    Required methods are kept tight. Optional hooks (`prefetch`,
    `sync_turn`, `on_session_end`, `shutdown`) default to no-op so a
    simple provider doesn't have to stub them. We'll grow the surface
    only when the second concrete provider needs more.
    """

    # -- required -----------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier matching the `cfg.memory_provider` value."""

    @abstractmethod
    def is_available(self) -> bool:
        """Cheap readiness check — no network calls.

        Return True when the provider can serve recall today (deps
        importable, on-disk paths writable). Manager skips dispatch
        when this is False, falling back to the built-in path.
        """

    @abstractmethod
    async def initialize(self, agent_home: Path, cfg: Any) -> None:
        """One-shot setup. Open DBs, register handlers, etc.

        `cfg` is whatever the caller passes (today: `AgentConfig`).
        Providers should read only attributes they actually need —
        going through `getattr(cfg, ..., default)` keeps them robust
        against the agent config evolving.
        """

    @abstractmethod
    def get_tool_schemas(self) -> list[dict]:
        """OpenAI-compatible tool schemas the provider exposes.

        Returned schemas are merged into `BUILTIN_TOOLS` at request
        time. Empty list = provider exposes no model-callable tools.
        """

    @abstractmethod
    async def handle_tool_call(self, name: str, args: dict) -> Any:
        """Dispatch a tool call by name. Return the tool result body
        (string the model sees) or raise to surface an error."""

    # -- optional -----------------------------------------------------

    async def prefetch(self, query: str) -> str:
        """Return text to inject as `[RECALLED CONTEXT]` for `query`.

        Called inline before each model API call when
        `cfg.memory_recall_mode` is `context` or `hybrid`. Empty
        string means "nothing relevant" — the manager skips the
        injection.
        """
        return ""

    async def sync_turn(self, user: str, assistant: str) -> None:
        """Mirror a completed turn to provider-side storage.

        Non-blocking: the manager runs this on the event bus's
        per-subscriber executor so a slow provider doesn't slow the
        turn loop.
        """

    async def on_session_end(self, turns: list[dict]) -> None:
        """Consume the full turn list at session boundary.

        Useful for session-level summarization / knowledge extraction.
        Today only fired on `session_saved` (close enough to "session
        end" for fact extraction; an agent that wants exact end-of-
        session semantics should layer that on top).
        """

    def shutdown(self) -> None:
        """Best-effort cleanup. Called from `memory_manager.shutdown()`."""


__all__ = ["MemoryProvider"]
