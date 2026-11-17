"""Memory-provider integration for :class:`Agent`.

The memory half of the Agent: recall prefetch (building the
``[RECALLED CONTEXT]`` block pre-turn, sync + async), post-turn provider
observation (fact extraction), and the debounced end-of-session
MEMORY.md flush + re-index. Split out of ``agents/agent.py`` as a mixin
so the composition root keeps only turn orchestration.

Methods resolve against the ``Agent`` instance via normal MRO — including
cross-mixin references (``self.run_turn``, ``self._effective_mode``) and
the per-Agent fields (``self._memory_flush_inflight``,
``self._memory_flush_timers``).

``__slots__ = ()`` is REQUIRED: ``Agent`` is ``@dataclass(slots=True)`` and
relies on undeclared-attribute assignment raising ``AttributeError``. A
mixin base without ``__slots__`` would re-introduce ``__dict__`` and break
that guarantee.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _flush_timestamp(value: Any) -> datetime | None:
    """Parse comparable UTC timestamps, including legacy offset-free values."""
    try:
        parsed = datetime.fromisoformat(str(value or ""))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
    except ValueError:
        return None


def _flush_completed_at(turn: dict[str, Any]) -> datetime | None:
    """Maintenance and retry attempts never authorize more maintenance."""
    if str(turn.get("user") or "").startswith("[MEMORY FLUSH]"):
        return None
    if str(turn.get("source") or "") == "memory_flush":
        return None  # failed flush retries can have an empty prompt
    return _flush_timestamp(turn.get("ended_at"))


class MemoryIntegrationMixin:
    """Recall prefetch, post-turn observation, and end-of-session flush."""

    __slots__ = ()

    async def _append_memory_recall_async(
        self,
        out: list[dict[str, Any]],
        session: Any,
        user_text: str,
    ) -> None:
        """Async home of memory auto-recall. The sync ``memory_recall_context`` returns "" under a running loop
        (can't block it), so on async turns recall must happen HERE: prefetch
        over the warm bridge in a thread (non-blocking on the gateway loop, same
        background server the heartbeat subscribes the provider on). Fail-soft."""
        try:
            import asyncio

            from sharedagents_agent.subagent_policy import profile_skips_memory
            from sharedagents_mcp.memory.client import prefetch_sync

            if profile_skips_memory(self.cfg):
                return
            # Only recall in context/hybrid modes — NOT tools/off/invalid. (The
            # server would normalize an invalid mode to hybrid and recall anyway,
            # so the mode gate must be enforced client-side too.)
            if str(getattr(self.cfg, "memory_recall_mode", "") or "") not in ("context", "hybrid"):
                return
            if not (user_text or "").strip():
                return
            workspace_origin = str(getattr(session, "workspace_origin", "") or "")
            chat_id = str(getattr(session, "hash", "") or "")
            recall = await asyncio.to_thread(
                prefetch_sync,
                self.cfg.home,
                user_text,
                chat_id=chat_id,
                workspace_origin=workspace_origin,
            )
            if recall:
                out.append({"role": "user", "content": recall})
        except Exception:  # noqa: BLE001
            pass

    def memory_recall_context(
        self,
        session: Any,
        user_text: str,
    ) -> str:
        """Build the ``[RECALLED CONTEXT]`` block for a turn.

        Reference: existing ``memory_manager.prefetch_for_query``
        contract (``sharedagents_core.memory_manager``). Returns the
        formatted block (or empty string when recall is disabled or
        the provider has nothing relevant). Local-LLM engine calls
        this every round; CLI engines call it once and fold into the
        composed system prompt.
        """
        mode = str(getattr(self.cfg, "memory_recall_mode", "") or "off").lower()
        if mode not in ("context", "hybrid"):
            return ""
        from sharedagents_agent.subagent_policy import profile_skips_memory

        if profile_skips_memory(self.cfg):
            return ""
        if not (user_text or "").strip():
            return ""
        try:
            import asyncio

            # Recall over the memory WIRE via the PERSISTENT
            # background-loop bridge (prefetch_sync) — never the in-process engine
            # and never a per-call server spawn. Bounded by the client's hot
            # deadline; fail-safe to "".
            from sharedagents_mcp.memory.client import prefetch_sync

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None and loop.is_running():
                # This sync method must not block a live event loop thread on the
                # bridge's result; fall back to empty recall (unchanged behaviour).
                return ""
            workspace_origin = str(getattr(session, "workspace_origin", "") or "")
            chat_id = str(getattr(session, "hash", "") or "")
            return prefetch_sync(
                self.cfg.home,
                user_text,
                chat_id=chat_id,
                workspace_origin=workspace_origin,
            )
        except Exception as exc:  # noqa: BLE001
            try:
                import logging

                logging.getLogger("sharedagents.agents.agent").warning(
                    "memory_recall_context failed: %s",
                    exc,
                )
            except Exception:  # noqa: BLE001
                pass
            return ""

    def observe_turn_memory(
        self,
        session: Any,
        turn_record: dict[str, Any],
    ) -> None:
        """Memory-provider observation of a finished turn.

        Split out of the old ``apply_post_turn_hooks``: operator
        ``Stop`` hooks now run via :meth:`run_stop_hooks` (real Claude
        Code hooks), while this keeps the memory side-effect — the
        active provider's ``observe_turn`` — that must fire on every
        turn regardless of hook config. No provider is active by default
        (file-only memory), so this is a no-op
        unless the operator registers an external provider.
        Best-effort; failures never block the turn save.
        """

        # 2. Memory provider observation (best-effort; failures here
        #    must never block the turn save).
        # Forward the finished turn to the active provider's
        # observe_turn over the memory WIRE (observe_turn RPC), never the
        # in-process engine. No-op by default (no concrete provider registered).
        # A minimal serializable summary is sent; a future provider extends it.
        try:
            import threading

            from sharedagents_mcp.memory.client import observe_turn_sync

            _summary = {
                "session_id": str(getattr(session, "hash", "") or getattr(session, "id", "") or ""),
            }
            # Fire-and-forget on the BACKGROUND BRIDGE server (where the provider
            # is subscribed). Scheduling on the gateway loop's client would
            # target a different server where the provider is inactive. A
            # daemon thread is non-blocking in both sync and async
            # contexts and reaches the same bridge server as subscribe/prefetch.
            threading.Thread(
                target=observe_turn_sync,
                args=(self.cfg.home,),
                kwargs={"turn": _summary},
                daemon=True,
            ).start()
        except Exception:  # noqa: BLE001
            pass

    def _cancel_pending_flush(self, sid: str) -> None:
        """Cancel and forget any pending end-of-session flush timer for ``sid``."""
        t = self._memory_flush_timers.pop(sid, None)
        if t is not None:
            try:
                t.cancel()
            except Exception:  # noqa: BLE001
                pass

    def _live_subagents(self, sid: str) -> int:
        """Number of live delegated-agent (subagent) leases under ``sid``.

        Every delegated-agent spawn — ``Agent``/``BatchAgent`` tool and the
        ``delegate_*`` shims, foreground or background — books an AGENT-class
        reservation in the process-wide :mod:`sharedagents_agent.sandbox_broker`
        under the parent session's key and holds it until the delegated turn
        completes (``tools/impl/subagents.py`` reserve/release, ``agents/
        delegate_shims.py`` reserve/release). A count > 0 therefore means the
        parent agent has live in-flight subagents it is waiting on, and the
        end-of-session flush must NOT run. Fail-closed: a lookup error reports
        "busy" (1) so we never flush into live subagent work.
        """
        try:
            from sharedagents_agent import sandbox_broker

            return int(sandbox_broker.get_broker().session_live(sid) or 0)
        except Exception:  # noqa: BLE001
            return 1

    def _session_is_live(self, sid: str) -> bool:
        """True if the session still has real in-flight work that must block flush.

        Dual gate:

        1. **Broker leases** — in-process AGENT reservations (``_live_subagents``).
        2. **Durable waiting-with-work** — ``sessions_meta.session_state ==
           "waiting"`` AND the parent still has busy descendants or open async
           jobs (same predicate the projector uses for ``_parent_still_has_work``).

        Deliberately NOT "any LIVE_STATES row": a bare ``waiting`` /
        wedged ``streaming`` with no children/jobs would starve every later
        flush forever. Active parent turns cancel the
        idle timer via ``_cancel_pending_flush``, so the timer path only reaches
        here after a quiet window — mid-turn states without waiting-work are
        treated as not live for flush purposes.
        """
        # Gate 1: broker leases (existing fast path)
        if self._live_subagents(sid) > 0:
            return True
        # Gate 2: durable waiting WITH remaining work (not bare LIVE_STATES)
        try:
            from sharedagents_session.session_index import (
                EventProjector,
                connect_sessions_db,
            )

            db_path = self.sessions_dir / "sessions.db"
            if not db_path.exists():
                return False
            _con = connect_sessions_db(db_path)
            try:
                row = _con.execute(
                    "SELECT session_state FROM sessions_meta WHERE session_hash=?",
                    (sid,),
                ).fetchone()
                if row is None:
                    return False
                state = str((row["session_state"] if hasattr(row, "keys") else row[0]) or "").strip()
                if state != "waiting":
                    return False
                # Freshness window:
                # a waiting parent is live only while it still has FRESH busy
                # descendants or open async jobs. Children stuck at a live
                # value with an old stamp are GHOSTS — an abort/crash never
                # folded them (the 2026-08-11 wedge) — and must not keep
                # deferring the flush forever.
                import time as _time

                from sharedagents_agent.orchestrator import LIVE_FRESHNESS_SECONDS

                now_ms = int(_time.time() * 1000)
                return bool(
                    EventProjector()._parent_still_has_work(
                        _con,
                        sid,
                        freshness_ms=now_ms - int(LIVE_FRESHNESS_SECONDS) * 1000,
                    )
                )
            finally:
                _con.close()
        except Exception:  # noqa: BLE001
            # Fail-closed: transient lock/lookup errors defer the flush.
            return True

    def _turns_since_flush(self, session: Any) -> int:
        """Count session turns NOT yet captured by a previous flush.

        Only completed turns with a valid timestamp can authorize paid
        maintenance. After a flush, only turns that ENDED after the stamp
        count — the flush turn itself is identified by its ``[MEMORY FLUSH]``
        prompt marker, and its retry attempts by ``source == "memory_flush"``;
        neither is ever counted (they run AFTER the stamp, so a
        timestamp-only filter would otherwise miscount them as new work and
        the next flush could re-process them). Replay fragments without a
        parseable ``ended_at`` never count: treating them as new on every
        idle window creates an unbounded flush loop.
        """
        try:
            turns = list(getattr(session, "turns", None) or [])
        except Exception:  # noqa: BLE001
            return 0
        stamp = str(getattr(session, "last_memory_flush_at", "") or "").strip()
        boundary = _flush_timestamp(stamp)
        if stamp and boundary is None:
            return 0  # unknown coverage must not authorize repeat spending
        n = 0
        for t in turns:
            ended = _flush_completed_at(t)
            if ended is None or (boundary is not None and ended <= boundary):
                continue
            n += 1
        return n

    def _schedule_memory_flush(self, session: Any) -> None:
        """Debounce an end-of-session MEMORY.md flush.

        Called on non-maintenance turn completion. (Re)arms the idle timer to
        ``cfg.memory_flush_idle_seconds`` (default 900 = 15 min) from now; the
        flush (:meth:`maybe_memory_flush`) only runs once the session has been
        quiet that long — i.e. it has ENDED. A new turn within the window
        cancels and reschedules, so an active multi-turn session never flushes
        mid-work. Further conversation after a flush arms another idle window,
        covering only newly completed turns. Flush completion itself does not
        re-arm maintenance. No-op when disabled or without an event loop.
        """
        if not getattr(self.cfg, "memory_flush_enabled", False):
            return
        sid = str(getattr(session, "hash", "") or "")
        if not sid:
            return
        # A real queued follow-up can complete during a flush's queue drain.
        # Let that follow-up arm the next window even while the flush is inflight.
        try:
            import asyncio as _aio

            loop = _aio.get_running_loop()
        except RuntimeError:
            return  # no event loop (sync/test context) — nothing to schedule
        delay = float(getattr(self.cfg, "memory_flush_idle_seconds", 900) or 900)
        # Cancel any pending timer for this session and arm a fresh one so the
        # 15-min idle window restarts on every turn (true debounce).
        self._cancel_pending_flush(sid)

        async def _fire_after_idle() -> None:
            try:
                await _aio.sleep(delay)
            except _aio.CancelledError:
                return
            # Only the CURRENT registered timer may fire — a cancelled-but-
            # already-resumed task must not race a freshly armed one.
            me = _aio.current_task()
            if self._memory_flush_timers.get(sid) is not me:
                return
            self._memory_flush_timers.pop(sid, None)
            # Never flush while the session still has real in-flight work
            # (broker leases, or durable waiting with busy children / open
            # jobs). DEFER instead of firing: re-arm the idle timer so the
            # flush happens on the next quiet window once work drains.
            if self._session_is_live(sid):
                self._schedule_memory_flush(session)
                return
            try:
                await self.maybe_memory_flush(session)
            except Exception:  # noqa: BLE001
                pass

        try:
            self._memory_flush_timers[sid] = loop.create_task(
                _fire_after_idle(),
                name=f"memory_flush_idle_{sid[:12]}",
            )
        except Exception:  # noqa: BLE001
            pass

    async def maybe_memory_flush(self, session: Any) -> tuple[bool, str]:
        """Run the end-of-session silent MEMORY.md flush turn if eligible.

        Called from the idle-debounced :meth:`_schedule_memory_flush` timer —
        after the session has gone quiet for ``cfg.memory_flush_idle_seconds``
        (default 900 = 15 min) — and from the pre-compaction
        ``needs_memory_flush`` intercept. Fires when ALL of:

        * ``cfg.memory_flush_enabled`` is True.
        * No live delegated agents (subagents) are in flight under this
          session (``_live_subagents``) — the parent must not be blocked
          waiting on subagent work.
        * The session has NEW work since the last flush
          (``>= memory_flush_min_turns`` turns added AFTER the persisted
          ``last_memory_flush_at`` boundary). A session that already flushed
          MAY flush again on a later idle window, but only the turns added
          after the stamp are processed — already-flushed turns are never
          re-flushed or double-counted.

        DELIBERATELY decoupled from ``compaction_count``: the old
        compaction-armed gate ("flush once per compaction cycle") degenerated
        into "flush every turn" on any session that compacted every turn — the
        flush turn ran against the full context, itself triggered a compaction,
        and re-armed the next flush: an unbounded loop a restart couldn't break
        (2026-06-22 runaway on agent 'main'). The idle trigger + monotonic
        stamp-before-run gate + completed-turn accounting bounds idle repeats.

        Spawns a silent turn through ``self.engine.run_turn`` with a directive
        prompt instructing the model to save context to MEMORY.md /
        memory/YYYY-MM-DD.md, then re-indexes so the notes are searchable.

        Returns ``(fired, audit_message)``.
        """
        if not getattr(self.cfg, "memory_flush_enabled", False):
            return False, "memory_flush_disabled"

        sid = str(getattr(session, "hash", "") or "")
        if sid and sid in self._memory_flush_inflight:
            return False, "flush_inflight"

        # A user Stop stands until the user prompts again: the flush is a
        # turn, and no automatic turn may start on a stopped session. The
        # Stop already cancelled the pending timer; this covers a timer the
        # stopped turn's own end re-armed.
        if sid:
            try:
                from sharedagents_session.operator_stop import operator_stopped

                if operator_stopped(self.sessions_dir, sid):
                    return False, "session_stopped"
            except Exception:  # noqa: BLE001
                pass

        # LIVE-WORK gate: never flush while delegated agents are still
        # performing work under this session (broker leases) or the durable
        # parent is ``waiting`` with busy children / open jobs. The timer
        # path additionally defers by re-arming (see ``_fire_after_idle``);
        # this guard covers every other caller.
        if sid and self._session_is_live(sid):
            return False, "session_live"

        # LAST-FLUSH BOUNDARY (persisted): ``last_memory_flush_at`` is the
        # boundary between already-flushed turns and new ones — NOT a
        # once-per-session "never again" gate. A session that already flushed
        # MAY flush again, but only the turns added AFTER the stamp are
        # eligible (``_turns_since_flush``), so already-flushed turns are
        # never re-processed or double-counted. The stamp is only ever
        # assigned (monotonic; nothing resets it to empty), which keeps the
        # historical flush loop bounded: re-flushing still requires
        # ``memory_flush_min_turns`` NEW turns, and a flush turn can never be
        # one of them (it is excluded by its own ``[MEMORY FLUSH]`` marker).
        # First flush (no stamp yet): only completed non-flush turns count.

        # Belt-and-suspenders: consult the DURABLE stamp too, so a long-lived
        # STALE in-memory session instance (one created before the stamp and
        # never re-synced) cannot re-flush. The persisted header is the source
        # of truth; if it says flushed, sync the in-memory object so the
        # boundary applies — then fall through to the new-turns gate (a second
        # flush is allowed iff turns were added after the durable stamp).
        # Pairs with the session_index sticky guard that keeps the durable
        # stamp from being clobbered to "" (2026-07-13 jarvis flush-loop root
        # cause).
        try:
            from sharedagents_session.session_index import connect_sessions_db

            _db = self.sessions_dir / "sessions.db"
            if sid and _db.exists():
                _con = connect_sessions_db(_db)
                try:
                    _row = _con.execute(
                        "SELECT last_memory_flush_at FROM sessions_meta WHERE session_hash=?",
                        (sid,),
                    ).fetchone()
                finally:
                    _con.close()
                _durable = str((_row[0] if _row else "") or "").strip()
                _current = _flush_timestamp(getattr(session, "last_memory_flush_at", ""))
                _persisted = _flush_timestamp(_durable)
                if _durable and (_current is None or (_persisted is not None and _persisted > _current)):
                    session.last_memory_flush_at = _durable
        except Exception:  # noqa: BLE001
            pass

        # Only flush when there is REAL new work since the last flush — skip
        # trivial one-off chats and idle repeats. ``compacted or not``
        # (operator directive): we do NOT gate on compaction, only on having
        # accumulated conversation since the last flush.
        min_turns = int(getattr(self.cfg, "memory_flush_min_turns", 3) or 3)
        n_new = self._turns_since_flush(session)
        if n_new < max(1, min_turns):
            if str(getattr(session, "last_memory_flush_at", "") or "").strip():
                return False, "no_new_turns_since_flush"
            return False, "session_too_small"

        # Fire a silent agentic turn instructing the model to save
        # context to MEMORY.md. The local-LLM engine implements the
        # actual saved-prompt drive; from the Agent we open one turn
        # against ``session.hash`` with a flush directive prompt. When
        # a previous flush exists, the prompt names the boundary so the
        # model persists ONLY the turns added after it — no re-writing
        # of already-flushed state (no double-counting).
        _last_flush = str(getattr(session, "last_memory_flush_at", "") or "").strip()
        if _last_flush:
            flush_prompt = (
                "[MEMORY FLUSH] This session has ended. Persist the most "
                f"important durable state from the turns AFTER {_last_flush} "
                "to the daily memory file memory/<YYYY-MM-DD>.md using the "
                "`remember` tool. Everything before that timestamp was already "
                "flushed — do not re-write it. Reply with [MEMORY_FLUSH_DONE] "
                "when finished."
            )
        else:
            flush_prompt = (
                "[MEMORY FLUSH] This session has ended. Persist the most "
                "important durable state from it to the daily memory file "
                "memory/<YYYY-MM-DD>.md using the `remember` tool. "
                "Reply with [MEMORY_FLUSH_DONE] when finished."
            )
        if sid:
            self._memory_flush_inflight.add(sid)
        # Persist the coverage boundary BEFORE spending. Keep the in-memory
        # boundary if persistence fails; an older durable stamp must never
        # roll it back. Across restarts persistence is still best-effort.
        covered_through = datetime.now(timezone.utc)
        # A clock rollback or imported future timestamp must not make the
        # same completed turn appear new on every attempt.
        for turn in getattr(session, "turns", None) or []:
            ended = _flush_completed_at(turn)
            if ended is not None and ended > covered_through:
                covered_through = ended
        session.last_memory_flush_at = covered_through.isoformat()
        try:
            session.save(root=self.sessions_dir, force=True)
        except Exception:  # noqa: BLE001
            pass
        try:
            from .engine import TurnDone

            done_ok = None
            async for ev in self.run_turn(
                session.hash,
                flush_prompt,
                mode=self._effective_mode(session),
                source="memory_flush",
            ):
                if isinstance(ev, TurnDone) and done_ok is None:
                    done_ok = ev.state == "ok"
                # Exhaust the stream so post-terminal cleanup and queue drains
                # run. Only the flush's own terminal determines its result;
                # a follow-up drain can emit additional terminals.
            # Re-index notes so the flushed content is searchable.
            self.memory_index_all()
            return bool(done_ok), "flush_completed" if done_ok else "flush_failed"
        except Exception as exc:  # noqa: BLE001
            return False, f"flush_error:{exc}"
        finally:
            if sid:
                self._memory_flush_inflight.discard(sid)

    def memory_index_all(self) -> None:
        """Re-build the agent's local memory index.

        Reference: ``sharedagents_core.memory_manager`` post-flush rebuild
        (today driven by ``force_post_compaction_index``).
        """
        # Rebuild the index over the memory wire (index_rebuild
        # RPC), never the in-process engine. Best-effort — a running loop or any
        # error is swallowed so this never blocks the caller.
        try:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None and loop.is_running():
                # async context (post-flush runs on the live loop): SCHEDULE the
                # reindex on it (non-blocking) instead of skipping — else flushed
                # memory would never be reindexed.
                from sharedagents_mcp.memory.client import get_memory_client

                loop.create_task(
                    get_memory_client(self.cfg.home).index_rebuild(if_needed=True),
                )
            else:
                from sharedagents_mcp.memory.client import index_rebuild_sync

                index_rebuild_sync(self.cfg.home, if_needed=True)  # warm bridge
        except Exception:  # noqa: BLE001
            # Indexing is best-effort; never block the caller.
            return
