"""Durable per-session follow-up (type-ahead) queue for :class:`Agent`.

The follow-up queue half of the Agent: append/list/update/remove pending
items, record inline injection into an open turn, and the round-boundary
+ turn-end drain callbacks the engine consumes. Backed by an in-memory
mirror (``self._queue_mirror``) folded from the on-disk JSONL and kept in
sync with durable ``queue_msg_*`` lifecycle records. Split out of
``agents/agent.py`` as a mixin so the composition root keeps only turn
orchestration.

Methods resolve against the ``Agent`` instance via normal MRO
(``self.sessions_dir`` and the per-Agent ``self._queue_mirror`` field).

``__slots__ = ()`` is REQUIRED: ``Agent`` is ``@dataclass(slots=True)`` and
relies on undeclared-attribute assignment raising ``AttributeError``. A
mixin base without ``__slots__`` would re-introduce ``__dict__`` and break
that guarantee.
"""

from __future__ import annotations

from typing import Any


class FollowupQueueMixin:
    """Durable per-session follow-up (type-ahead) message queue."""

    __slots__ = ()

    def _queue_writer(self, sid: str) -> Any:
        """Return a PINNED SessionWriter for ``sid`` (this agent's sessions_dir).

        ``pin=True`` so the get/`release_session_writer` pair is balanced — a
        queue write from another interface during an active turn must not
        decrement the turn's pinned lease (which would expose the live writer
        to LRU eviction and orphan the turn). Callers MUST release in a finally.
        """
        from sharedagents_session.session_writer import get_session_writer

        return get_session_writer(self.sessions_dir, sid, pin=True)

    def _queue_load(self, sid: str) -> dict[str, Any]:
        """Return the in-memory mirror for ``sid``, building it from the
        on-disk JSONL if this is the first access for this session."""
        import collections

        from sharedagents_session.session_reader import fold_queue_state, tail_jsonl

        if sid in self._queue_mirror:
            return self._queue_mirror[sid]  # type: ignore[return-value]

        m: dict[str, Any] = collections.OrderedDict()
        # CORRECTION 2: use self.sessions_dir (the Agent property), not
        # sessions_root(), so tests can inject an arbitrary path via cfg.
        p = self.sessions_dir / f"{sid}.jsonl"
        if p.exists():
            evs = (ev for ev, _ in tail_jsonl(p, 0))
            for item in fold_queue_state(evs):
                m[item["queue_id"]] = item
        self._queue_mirror[sid] = m
        return m

    def _queue_resync(self, sid: str) -> dict[str, Any]:
        """Re-fold pending queue items from the session JSONL.

        Cross-process writers (parent ``Agent(action=steer)``) append
        ``queue_msg_added`` to the child's log; the child's in-memory mirror
        would otherwise miss them. Cheap: one fold of the JSONL.
        """
        self._queue_mirror.pop(sid, None)
        return self._queue_load(sid)

    def queue_add(
        self,
        sid: str,
        text: str,
        *,
        source: str = "",
        author: str = "",
        actor: dict[str, Any] | None = None,
        access: Any = None,
    ) -> str:
        """Append a follow-up item.  Returns the new queue_id.

               ``author`` keeps its meaning (queue-source identity / background
               receipt key); ``actor`` is the WebUI user who typed it
        and is stored only when present; ``access`` is that
               user's grants (an ``AccessStamp``), stored likewise.
        """
        import uuid

        from sharedagents_session.actors import normalize_actor
        from sharedagents_session.session_writer import release_session_writer

        text = (text or "").strip()
        if not text:
            raise ValueError("empty queue text")
        qid = uuid.uuid4().hex[:12]
        who = normalize_actor(actor)
        stamp = access.to_obj() if access is not None else None
        payload: dict[str, Any] = {"queue_id": qid, "text": text, "source": source, "author": author}
        if who is not None:
            payload["actor"] = who
        if stamp is not None:
            payload["access"] = stamp
        m = self._queue_load(sid)
        w = self._queue_writer(sid)
        try:
            w.lifecycle(kind="queue_msg_added", payload=payload)
        finally:
            release_session_writer(w)
        m[qid] = {
            "queue_id": qid,
            "text": text,
            "source": source,
            "author": author,
            "actor": who,
            "access": stamp,
            "added_ts": "",
        }
        return qid

    def queue_list(self, sid: str) -> list:
        """Return pending items in submission order."""
        return list(self._queue_load(sid).values())

    def queue_update(
        self,
        sid: str,
        queue_id: str,
        text: str,
        *,
        access: Any = None,
    ) -> bool:
        """Update a pending item's text.  Returns False if not found.

        ``access`` is the editor's grants: the item then runs under the
        intersection of its author's and its editor's grants, so editing
        someone's queued message never borrows their wider grants
        """
        from sharedagents_core.access_grants import AccessStamp, merge_stamps
        from sharedagents_session.session_writer import release_session_writer

        text = (text or "").strip()
        m = self._queue_load(sid)
        if not text or queue_id not in m:
            return False
        payload: dict[str, Any] = {"queue_id": queue_id, "text": text}
        stamp = None
        if access is not None:
            stamp = merge_stamps([AccessStamp.from_obj(m[queue_id].get("access")), access])
            if stamp is not None:
                payload["access"] = stamp.to_obj()
        w = self._queue_writer(sid)
        try:
            w.lifecycle(kind="queue_msg_updated", payload=payload)
        finally:
            release_session_writer(w)
        m[queue_id]["text"] = text
        if stamp is not None:
            m[queue_id]["access"] = stamp.to_obj()
        return True

    def queue_remove(
        self,
        sid: str,
        queue_id: str,
        *,
        reason: str = "user_deleted",
    ) -> bool:
        """Remove a pending item. Returns False if not found.

        ``reason`` distinguishes an explicit operator deletion from successful
        runtime consumption. Older callers keep the historical
        ``user_deleted`` default; turn-end auto-drain records ``user_drained``;
        gateway child-result consumption records ``background_consumed``.
        """
        from sharedagents_session.queue_remove import (
            QUEUE_REMOVE_USER_DELETED,
            normalize_queue_remove_reason,
        )
        from sharedagents_session.session_writer import release_session_writer

        m = self._queue_load(sid)
        if queue_id not in m:
            return False
        w = self._queue_writer(sid)
        try:
            w.lifecycle(
                kind="queue_msg_removed",
                payload={
                    "queue_id": queue_id,
                    "reason": (normalize_queue_remove_reason(reason) or QUEUE_REMOVE_USER_DELETED),
                },
            )
        finally:
            release_session_writer(w)
        m.pop(queue_id, None)
        return True

    def queue_inject_record(self, sid: str, queue_id: str, text: str) -> int | None:
        """Record a pending item as injected INLINE into the CURRENTLY-OPEN turn.

        Returns the turn_idx used, or ``None`` (a no-op) when NO turn is open
        yet. The ``None`` case is the load-bearing guard: a turn-scoped
        ``queue_msg_injected`` written before this turn's ``turn_started`` would
        split the fold. ``round_idx`` is NOT a reliable proxy for
        "turn open" (some retry paths advance the round without a turn-opening
        flush), so we consult the writer's actual open-turn state. When ``None``
        the item stays pending → drained at a later round boundary or by the
        turn-end auto-drain.
        """
        from sharedagents_session.session_writer import release_session_writer

        m = self._queue_load(sid)
        item = dict(m.get(queue_id) or {})
        w = self._queue_writer(sid)
        try:
            ti = w.current_turn_idx()
            if ti is None:
                return None
            payload: dict[str, Any] = {
                "queue_id": queue_id,
                "text": text,
                "inline": True,
                # Preserve provenance in the semantic turn fold. In
                # particular, background-task output is external data while
                # an operator's queued follow-up is authoritative input.
                "source": str(item.get("source") or ""),
                "author": str(item.get("author") or ""),
            }
            who = item.get("actor")
            if isinstance(who, dict) and who:
                payload["actor"] = dict(who)
            w.lifecycle(kind="queue_msg_injected", turn_idx=int(ti), payload=payload)
        finally:
            release_session_writer(w)
        m.pop(queue_id, None)
        return int(ti)

    def _auto_drain_payload(
        self,
        sid: str,
        source: str,
    ) -> tuple[str, list[dict[str, str]]]:
        """Turn-end auto-drain text + authors (see :meth:`_auto_drain_turn`)."""
        text, actors, _access = self._auto_drain_turn(sid, source)
        return text, actors

    def _auto_drain_turn(
        self,
        sid: str,
        source: str,
    ) -> tuple[str, list[dict[str, str]], Any]:
        """Turn-end auto-drain: joined pending text, its authors, and the
        grants the drain turn runs under, in queue order.

        The grants are the intersection of the restricted authors' stamps
        (``merge_stamps``): one person's follow-up can never
        ride another's wider grants.

        Several users' follow-ups can merge into one drain turn; every author
        is returned (deduplicated) so the drain turn records them all.
        Returns ``("", [], None)`` when nothing is pending. Drain turns can
        receive new follow-ups too, so their source must not skip this check.

        Pending items are REMOVED from the queue here (``queue_msg_removed``):
        they leave the pending widget and the new turn's ``turn_started.user``
        (= the returned text) is their durable record — so they render once, as
        a normal user turn, with no inline double-render.
        """
        from sharedagents_core.access_grants import AccessStamp, merge_stamps
        from sharedagents_session.actors import normalize_actors

        try:
            # The enqueue may have come from the web/gateway process after the
            # last round-boundary drain. Refresh the mirror before the final
            # turn-end check, otherwise a late durable queue item is stranded
            # until the operator presses Steer.
            pending = list(self._queue_resync(sid).values())
        except Exception:  # noqa: BLE001
            return "", [], None
        if not pending:
            return "", [], None
        kept = [p for p in pending if (p.get("text") or "").strip()]
        from sharedagents_core.attachment_context import merge_queue_text

        text = merge_queue_text(kept)
        actors = normalize_actors([p.get("actor") for p in kept])
        access = merge_stamps(AccessStamp.from_obj(p.get("access")) for p in kept)
        for p in pending:
            try:
                from sharedagents_session.queue_remove import QUEUE_REMOVE_USER_DRAINED

                self.queue_remove(
                    sid,
                    p.get("queue_id", ""),
                    reason=QUEUE_REMOVE_USER_DRAINED,
                )
            except Exception:  # noqa: BLE001
                pass
        return text, actors, access

    def _auto_drain_text(self, sid: str, source: str) -> str:
        """Turn-end auto-drain text only (kept for existing callers/docs)."""
        return self._auto_drain_payload(sid, source)[0]

    def _make_queue_drain(self, sid: str):
        """Return the GENTLE round-boundary drain callback the engine invokes.

                Records every pending follow-up as injected (durable + mirror pop) and
                returns ``[(queue_id, text, source)]`` for the engine to append as user
                turns. No soft-abort. Self-gating: ``queue_inject_record`` returns
                ``None`` when no turn is OPEN yet (the authoritative guard against
                writing ``queue_msg_injected`` before ``turn_started``), in
                which case we stop and leave the rest pending for a later boundary or
                the turn-end auto-drain. So this is safe to call at EVERY round
                boundary, including round 0.

                An item stamped with a user's grants joins the running turn only when
                that turn already runs under grants no wider than the item's author
        ; otherwise it — and everything after it, to keep
                order — waits for the turn-end drain, which runs under the authors'
                grants.
        """

        def _drain() -> list[tuple[str, str, str]]:
            from sharedagents_core.access_grants import AccessStamp
            from sharedagents_session.access_binding import session_access_grants

            # Resync so a parent-process steer (queue_msg_added on disk) is
            # visible
            out: list[tuple[str, str, str]] = []
            bound = _MISSING = object()
            for it in list(self._queue_resync(sid).values()):
                qid, text = it["queue_id"], it["text"]
                source = str(it.get("source") or "")
                stamp = AccessStamp.from_obj(it.get("access"))
                if stamp is not None and stamp.restricted:
                    if bound is _MISSING:
                        bound = session_access_grants(self.sessions_dir, sid)
                    if bound is None or not bound.is_subset_of(stamp.grants):
                        break  # wider running turn — defer to the drain turn
                if self.queue_inject_record(sid, qid, text) is None:
                    break  # turn not open yet — defer this and the rest
                out.append((qid, text, source))
            return out

        return _drain

    def queue_has_pending(self, sid: str) -> bool:
        """True iff the session has at least one pending follow-up item."""
        return bool(self._queue_load(sid))
