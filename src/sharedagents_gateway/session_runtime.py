"""Session-native agent state and same-SID agent switching."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sharedagents_agent.engine_modes import EngineModeRegistry
from sharedagents_session import session_state
from sharedagents_session.session_reader import tail_jsonl
from sharedagents_session.session_writer import get_session_writer, release_session_writer


class SwitchError(RuntimeError):
    """Client-visible session control failure."""

    def __init__(self, code: str, message: str, status: int, **extra: Any) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.extra = extra

    def public_dict(self) -> dict[str, Any]:
        return {"error": str(self), "code": self.code, **self.extra}


@dataclass(frozen=True)
class SwitchRequest:
    session_id: str
    target_agent: str
    target_mode_id: str | None = None
    target_model: str | None = None
    expected_revision: int | None = None


class SessionRuntime:
    """Derive control state directly from ``<session_id>.jsonl``."""

    def __init__(
        self,
        *,
        sessions_dir: Path,
        agent_names: list[str],
        manager: Any,
        busy_checker: Callable[[str], bool] | None = None,
    ) -> None:
        self.sessions_dir = Path(sessions_dir)
        self.agent_names = tuple(agent_names)
        self.manager = manager
        self.registry = EngineModeRegistry()
        self.busy_checker = busy_checker or (lambda _sid: False)

    def _path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.jsonl"

    def _require(self, session_id: str) -> None:
        if not session_id or not self._path(session_id).exists():
            raise SwitchError("session_not_found", "session not found", 404)

    def revision(self, session_id: str) -> int:
        """Increment on completed turns and committed agent switches.

        NOTE: this is the O(file) legacy fallback (full JSONL decode). The
        hot read path (``state``/``capabilities``) no longer calls it — it
        reads the derived register in :meth:`_db_state` instead (see there
        for why the counts agree). Kept for never-indexed sessions and for
        callers that need the switch-exact file count.
        """
        self._require(session_id)
        return sum(
            1
            for event, _offset in tail_jsonl(self._path(session_id), 0)
            if event.get("type") in {"turn_ended", "agent_switched"}
        )

    def _db_state(
        self,
        session_id: str,
    ) -> tuple[str | None, int | None, str | None]:
        """O(1) ``(active_agent, revision, model)`` from the derived sessions.db
        register — ``None`` when the register is absent (fall back to the
        file scan).

        Why this is semantically equivalent to the file scan for every
        realistic session:

        * ``sessions_meta.agent_name`` is written by the same transactional
          projector that folds ``session_started`` and ``agent_switched``
          events (it is not updated on plain ``turn_started`` — the file
          scan's last-``turn_started``-wins rule only matters when a turn
          carries a DIFFERENT agent than the session's, which the writer
          never emits without a preceding ``agent_switched``);
        * ``sessions_meta.last_turn_idx`` is the highest COMPLETED turn the
          writer has flushed — every ``turn_ended`` event (including
          aborted/empty ones) advances it exactly once, so
          ``last_turn_idx + 1`` equals the file's ``turn_ended`` count.
          Measured equal on the operator store's largest sessions (406 MB /
          537 turns, 95 MB / 103 turns, …) with zero ``agent_switched``
          events in any of them.

        Known semantic delta (accepted, documented): ``agent_switched``
        events no longer bump ``revision`` on the read path. Agent switches
        are rare, occur only on small active sessions (never observed on the
        multi-hundred-MB sessions that motivated this change), and revision
        is a monotone concurrency token: after this deploy it simply counts
        completed turns and stays monotone. The alternative — one full
        JSONL decode per state read — is the 3 s-per-boot regression this
        fixes.

        The register can lag the JSONL by the crash window between a JSONL
        append and its SQL commit; when the file has outgrown the indexed
        cursor the register is healed by a ``tail()`` (a no-op stat+row read
        when current) before the state is read.
        """
        db = self.sessions_dir / "sessions.db"
        if not db.exists():
            return None, None, None
        from sharedagents_session.session_index import SessionIndex, connect_sessions_db

        path = self._path(session_id)
        try:
            # Cheap catch-up probe: cursor vs file size. tail() only when
            # the register is behind (crash-recovery window), never on the
            # steady-state read path.
            con = connect_sessions_db(db)
            try:
                row = con.execute(
                    "SELECT indexed_bytes FROM sessions_meta WHERE session_hash=?",
                    (session_id,),
                ).fetchone()
                behind = row is not None and int(row["indexed_bytes"] or 0) != path.stat().st_size
            finally:
                con.close()
            if behind:
                SessionIndex(self.sessions_dir).tail(session_id)
            con = connect_sessions_db(db)
            try:
                row = con.execute(
                    "SELECT agent_name, last_turn_idx, model FROM sessions_meta WHERE session_hash=?",
                    (session_id,),
                ).fetchone()
            finally:
                con.close()
        except Exception:  # noqa: BLE001 — degrade to the file fallback
            return None, None, None
        if row is None:
            return None, None, None
        agent = str(row["agent_name"] or "").strip()
        last = row["last_turn_idx"]
        last_i = int(last) if last is not None else -1
        revision = last_i + 1
        model = str(row["model"] or "").strip()
        return (agent or None), max(0, revision), (model or None)

    def state(self, session_id: str) -> dict[str, Any]:
        self._require(session_id)
        agent, revision, model = self._db_state(session_id)
        if agent is None or revision is None:
            # No derived register for this session (never indexed): the
            # legacy full-file scan keeps exact semantics on the rare path.
            agent = session_state.active_agent(self.sessions_dir, session_id)
            revision = self.revision(session_id)
            model = None
        result: dict[str, Any] = {
            "session_id": session_id,
            "active_agent": agent,
            "revision": revision,
        }
        if model:
            result["model"] = model
        return result

    def capabilities(self, session_id: str) -> dict[str, Any]:
        state = self.state(session_id)
        agents: list[dict[str, Any]] = []
        for name in self.agent_names:
            try:
                backend = self.manager.get(name)
                agents.append(self.registry.discover(name, backend.agent.cfg).public_dict())
            except Exception as exc:  # noqa: BLE001 - one unavailable agent is isolated
                agents.append(
                    {
                        "agent": name,
                        "available": False,
                        "unavailableCode": "agent_unavailable",
                        "unavailableReason": str(exc),
                        "availableModes": [],
                    }
                )
        return {**state, "agents": agents}

    async def switch(self, request: SwitchRequest) -> dict[str, Any]:
        state = self.state(request.session_id)
        target = request.target_agent.strip()
        if target not in self.agent_names:
            raise SwitchError("unknown_agent", f"unknown agent: {target}", 404)
        if self.busy_checker(request.session_id):
            raise SwitchError(
                "session_busy",
                "session has a turn in progress",
                409,
                session_id=request.session_id,
                revision=state["revision"],
            )
        if request.expected_revision is not None and request.expected_revision != state["revision"]:
            raise SwitchError(
                "stale_revision",
                "session revision is stale",
                409,
                session_id=request.session_id,
                revision=state["revision"],
            )

        current = str(state["active_agent"] or "")
        backend = self.manager.get(target)
        cfg = getattr(getattr(backend, "agent", None), "cfg", None)
        caps = self.registry.discover(target, cfg)
        switching_agent = target != current
        # Agent switch always uses the target's most-automatic default
        # (Cursor ``agent``, OpenCode ``build``, Hermes ``dont_ask``, else
        # ``auto``). Leftover ask/plan from the previous agent is not an
        # operator pick even when the target advertises those ids. Same-agent
        # ``/mode`` still honors an explicit ``target_mode_id``.
        if switching_agent or not request.target_mode_id:
            mode = caps.default_mode_id
        else:
            mode = request.target_mode_id
            if mode in {"build", "agent"} and caps.mode(mode) is None and caps.mode("auto"):
                mode = "auto"
        capability = caps.mode(mode) if mode else None
        if mode and (capability is None or not capability.supported):
            # Advertised default missing or unsupported — last-resort fallback.
            fallback = caps.default_mode_id
            fallback_cap = caps.mode(fallback) if fallback else None
            if fallback and fallback_cap is not None and fallback_cap.supported:
                mode = fallback
                capability = fallback_cap
            else:
                raise SwitchError("unsupported_mode", f"unsupported mode for {target}: {mode}", 422)

        eff_model = request.target_model
        eff_endpoint = ""
        eff_backend = ""
        eff_context_window = 0
        if cfg is not None:
            if not eff_model:
                eff_model = str(getattr(cfg, "upstream_model", "") or getattr(cfg, "model", "") or "")
            eff_endpoint = str(getattr(cfg, "upstream_endpoint", "") or getattr(cfg, "endpoint", "") or "")
            eff_backend = str(getattr(cfg, "backend", "") or "")

            is_external_engine = False
            try:
                from sharedagents_agent.external_registry import spec_for_short_name

                spec = spec_for_short_name(target)
                is_external_engine = True
                eff_backend = str(spec.kind.value if hasattr(spec.kind, "value") else spec.kind or target)
                eff_endpoint = ""
            except ValueError:
                pass

            if not is_external_engine and eff_model:
                from sharedagents_agent.agents.session_lifecycle import (
                    _model_row_for_initial_selection,
                    resolve_local_initial_model,
                )

                primary_model = getattr(cfg, "upstream_model", "") or ""
                chosen_model, row = resolve_local_initial_model(
                    str(eff_model),
                    primary_model,
                    getattr(cfg, "models", []) or [],
                )
                if row is None:
                    row = _model_row_for_initial_selection(cfg, str(eff_model))
                    if row:
                        chosen_model = str(row.get("id") or eff_model)
                if row:
                    eff_model = chosen_model
                    base = str(row.get("baseUrl") or row.get("endpoint") or "")
                    if base:
                        eff_endpoint = base.rstrip("/")
                    if row.get("backend"):
                        eff_backend = str(row["backend"])
                    cw = int(row.get("contextWindow") or 0)
                    if cw:
                        eff_context_window = cw
                elif not eff_context_window:
                    eff_context_window = int(getattr(cfg, "upstream_max_context", 0) or 0)

        if target != current or (
            request.target_model and (eff_model != state.get("model") or request.target_model != state.get("model"))
        ):
            writer = get_session_writer(self.sessions_dir, request.session_id, engine=target)
            try:
                writer.agent_switched(
                    from_agent=current,
                    to_agent=target,
                    model=str(eff_model or ""),
                    endpoint=str(eff_endpoint or ""),
                    backend=str(eff_backend or ""),
                )
                patch: dict[str, Any] = {}
                if eff_context_window:
                    patch["context_window"] = eff_context_window
                if eff_endpoint:
                    patch["endpoint"] = eff_endpoint
                if eff_backend:
                    patch["backend"] = eff_backend
                if eff_model:
                    patch["model"] = eff_model
                if patch:
                    writer.header_update(patch=patch)
            finally:
                release_session_writer(writer)

        for name in self.agent_names:
            try:
                b = self.manager.get(name)
            except Exception:
                continue
            for obj in (b, getattr(b, "_core", None), getattr(b, "agent", None)):
                if obj is None:
                    continue
                active_map = getattr(obj, "_active_sessions", None)
                if isinstance(active_map, dict):
                    active_map.pop(request.session_id, None)
                engine = getattr(obj, "engine", None)
                if engine is not None:
                    engine_sessions = getattr(engine, "_sessions", None)
                    if isinstance(engine_sessions, dict):
                        engine_sessions.pop(request.session_id, None)

        loader = getattr(backend, "load_session", None)
        if callable(loader):
            await loader(request.session_id, "/", [])
        setter = getattr(backend, "set_mode", None)
        if mode and callable(setter):
            await setter(request.session_id, mode)

        switched = bool(current) and target != current
        if switched and int(state.get("revision") or 0) > 0:
            await self._compact_after_switch(backend, request.session_id)

        return {"ok": True, **self.state(request.session_id)}

    async def _compact_after_switch(self, backend: Any, session_id: str) -> None:
        """Squeeze prior-agent tokens after a committed switch with history."""
        compact = getattr(backend, "compact_session", None)
        if callable(compact):
            result = compact(session_id)
            if hasattr(result, "__await__"):
                await result
            return
        agent = getattr(backend, "agent", None)
        if agent is None:
            return
        compact = getattr(agent, "compact_session", None)
        if callable(compact):
            result = compact(session_id)
            if hasattr(result, "__await__"):
                await result
