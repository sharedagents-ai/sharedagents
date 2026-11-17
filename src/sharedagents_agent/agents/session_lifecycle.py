"""Session lifecycle (create / list / get / close / inject) for :class:`Agent`.

The session-CRUD half of the Agent: create a fresh session (composing the
system prompt + stamping attribution), list this agent's sessions from the
shared ``sessions.db``, load a session (header from db + turns folded from
JSONL), close it (detach engine state + SessionEnd hook), and forward a
mid-turn user message to the engine. Split out of ``agents/agent.py`` as a
mixin so the composition root keeps only turn orchestration.

Methods resolve against the ``Agent`` instance via normal MRO
(``self.cfg``, ``self.engine``, ``self.name``, ``self.sessions_dir``,
``self.permission_mode``, ``self.compose_system_prompt``,
``self._run_hook_event``, and the per-Agent ``self._system_prompt_cache``
field).

``__slots__ = ()`` is REQUIRED: ``Agent`` is ``@dataclass(slots=True)`` and
relies on undeclared-attribute assignment raising ``AttributeError``. A
mixin base without ``__slots__`` would re-introduce ``__dict__`` and break
that guarantee.

new sessions — merged-config consumption documented at ``create_session``)
"""

from __future__ import annotations

import contextvars
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .session_log import _loads_json_dict, _loads_list

# In-flight loop mint: origin → new-tick is not Agent-class delegation.
# ``(origin, tick)``; tick is filled after create_session returns the id.
_LOOP_MINT: contextvars.ContextVar[tuple[str, str]] = contextvars.ContextVar(
    "sharedagents_loop_mint",
    default=("", ""),
)


# Permission-mode ids that the web picker also uses. They are never
# OpenAI-compatible model names (DeepSeek 400: "you passed auto").
_MODE_SENTINELS = frozenset({"auto", "ask", "plan"})


def resolve_local_initial_model(
    model: str,
    primary_model: str,
    models: list[Any] | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Resolve a web/loop ``initial_model`` for a local_llm session.

    Mode sentinels and aliases that miss ``models[]`` fall back to
    ``primary_model`` (the agent's ``upstream_model``). A catalogue hit
    returns that row's wire id so the caller can also adopt endpoint.
    """
    raw = str(model or "").strip()
    primary = str(primary_model or "").strip()
    if not raw or raw.lower() in _MODE_SENTINELS:
        return primary, None
    row: dict[str, Any] | None = None
    for item in models or []:
        md = dict(item) if isinstance(item, dict) else {}
        if raw in (str(md.get("alias") or ""), str(md.get("id") or "")):
            row = md
            break
    if row is None:
        return primary, None
    return str(row.get("id") or raw), row


def _effort_row_for_session(
    cfg: Any,
    agent: str,
    model: str,
    row: dict[str, Any] | None,
    *,
    is_external: bool,
) -> dict[str, Any] | None:
    """The model row a new session runs on, from saved data only (no network)."""
    target = str(model or "").strip().lower()
    if not target:
        return None
    # Use the same canonical catalogue rows as commands and the picker,
    # including native provider-discovery rows absent from cfg.models.
    try:
        from sharedagents_agent.discovery.picker import saved_models

        for item in saved_models(agent):
            if target in {str(item.get(key) or "").strip().lower() for key in ("id", "alias", "wire_id", "model_id")}:
                resolved = {**item, **(row or {})}
                if not resolved.get("provider"):
                    resolved["provider"] = item.get("provider") or ""
                return resolved
    except Exception:  # noqa: BLE001
        pass
    if row:
        return row
    for item in getattr(cfg, "models", None) or []:
        if isinstance(item, dict) and target in {str(item.get(key) or "").strip().lower() for key in ("id", "alias")}:
            return dict(item)
    try:
        primary = dict(cfg._primary_model_row())
    except Exception:  # noqa: BLE001
        primary = {}
    if target in {str(primary.get(key) or "").strip().lower() for key in ("id", "alias")}:
        return primary
    return None


def initial_session_effort(
    cfg: Any,
    agent: str,
    model: str,
    row: dict[str, Any] | None,
    *,
    is_external: bool,
    picked: str | None = None,
) -> str:
    """The effort a new session starts with, recording a picked level.

    ``picked`` (a new chat's ``initial_effort``): a level the model lists is
    stored on the session and remembered for the model; ``default`` clears
    both. Without a usable pick, the level remembered for the model applies
    when the model still lists it. ``""`` = no session override.
    """
    from sharedagents_command.commands import normalize_effort
    from sharedagents_model import effort_prefs
    from sharedagents_model.models import derive_effort_capability

    effort_row = _effort_row_for_session(
        cfg,
        agent,
        model,
        row,
        is_external=is_external,
    )
    if effort_row is None:
        return ""
    provider = effort_prefs.engine_provider(cfg, agent)
    resolved = dict(effort_row)
    try:
        from sharedagents_model.effort_discovery import discover_effort_capability

        # Cache only: creating a session never waits on a provider.
        found = discover_effort_capability(resolved, cfg, probe=False)
        if found is not None:
            base = resolved.get("effort") if isinstance(resolved.get("effort"), dict) else {}
            resolved["effort"] = {**base, **found}
    except Exception:  # noqa: BLE001
        pass
    values = list(derive_effort_capability(resolved).get("values") or [])
    if picked is not None and str(picked).strip():
        level = normalize_effort(str(picked))
        if level == "default":
            effort_prefs.set_row_pref(effort_row, "default", provider)
            return ""
        if level and level in values:
            effort_prefs.set_row_pref(effort_row, level, provider)
            return level
    return effort_prefs.remembered_level_for(effort_row, values, provider)


def remember_model_effort(agent: str, model: str, effort: Any) -> bool:
    """Remember ``effort`` for ``model`` of ``agent``.

    Used where the picker applies a level without a ``/model`` command: "Set
    default" and Save on a new chat that has no session yet.

    The row is resolved from configuration and saved catalogues only. Returns
    True when the store changed. ``default`` (or empty) clears the entry.
    """
    from sharedagents_agent.external_registry import is_external_engine_kind
    from sharedagents_command.commands import normalize_effort
    from sharedagents_model import effort_prefs

    target = str(model or "").strip()
    if not agent or not target:
        return False
    level = normalize_effort(str(effort or ""))
    if level is None:
        return False
    from sharedagents_agent.agent import AgentConfig

    cfg = AgentConfig.load(agent, ambient_discovery=False)
    row = _effort_row_for_session(
        cfg,
        agent,
        target,
        None,
        is_external=is_external_engine_kind(getattr(cfg, "engine", "") or ""),
    )
    if row is None:
        return False
    return effort_prefs.set_row_pref(
        row,
        level,
        effort_prefs.engine_provider(cfg, agent),
    )


def _sandbox_env_depth() -> int:
    try:
        return int(os.environ.get("SHAREDAGENTS_SANDBOX_AGENT_DEPTH", "0") or "0")
    except (TypeError, ValueError):
        return 0


def _model_row_for_initial_selection(cfg: Any, model: str) -> dict[str, Any] | None:
    """Resolve a fresh session's explicit model against current credentials.

    Native agents are long-lived. Their in-memory catalogue can predate a
    provider discovery performed by the separate Web process. Merge the shared
    native discovery cache first, then reload disk once on a miss. This ensures
    the selected alias carries its wire id, endpoint, backend, and auth row
    into session creation instead of riding the worker's unrelated default.
    """
    target = str(model or "").strip()
    if not target:
        return None

    def _scan() -> dict[str, Any] | None:
        configured = [dict(item) for item in (getattr(cfg, "models", None) or []) if isinstance(item, dict)]
        rows = configured
        try:
            from sharedagents_agent.discovery.cache_lock import native_models_with_cache

            rows = native_models_with_cache(
                str(getattr(cfg, "name", "") or "jarvis"),
                configured,
            )
        except Exception:  # noqa: BLE001 — retain the current config on cache failure
            pass
        for row in rows:
            if target not in (
                str(row.get("alias") or ""),
                str(row.get("id") or ""),
            ):
                continue
            selected = dict(row)
            # Persist only this selected provider row into the live config so
            # upstream_headers()/endpoint lookup can resolve its auth metadata.
            selected_key = (
                str(selected.get("id") or ""),
                str(selected.get("baseUrl") or selected.get("endpoint") or "").rstrip("/"),
            )
            configured_keys = {
                (
                    str(item.get("id") or ""),
                    str(item.get("baseUrl") or item.get("endpoint") or "").rstrip("/"),
                )
                for item in configured
            }
            if selected_key not in configured_keys:
                try:
                    cfg.models = [*configured, selected]
                except Exception:  # noqa: BLE001 — selection still resolves without mutation
                    pass
            return selected
        return None

    row = _scan()
    if row is not None:
        return row
    reload_models = getattr(cfg, "reload_models", None)
    if callable(reload_models):
        try:
            reload_models()
        except Exception:  # noqa: BLE001 — create_session retains fail-soft fallback
            return None
    return _scan()


class _LoopMintHandle:
    """Register the minted tick id on an in-flight ``loop_mint_context``."""

    def register_tick(self, tick_session_id: str) -> None:
        origin, _prev = _LOOP_MINT.get()
        _LOOP_MINT.set((origin, str(tick_session_id or "").strip()))


@contextmanager
def loop_mint_context(origin_session_id: str) -> Iterator[_LoopMintHandle]:
    """Mark an in-flight loop mint so origin→tick is not counted twice.

    ``create_session(parent=origin)`` during ``prepare_loop_fire`` is a
    tick edge, even before ``session_rebound`` is written. Nested
    ordinary children of the origin are not exempt.
    """
    origin = str(origin_session_id or "").strip()
    token = _LOOP_MINT.set((origin, ""))
    try:
        yield _LoopMintHandle()
    finally:
        _LOOP_MINT.reset(token)


def _loop_tick_origin_edges() -> dict[str, str]:
    try:
        from sharedagents_loop.loop_registry import get_loop_registry

        return get_loop_registry().tick_origin_edges()
    except Exception:  # noqa: BLE001 — no edges: count every hop
        return {}


def _in_flight_tick_edges() -> dict[str, str]:
    origin, tick = _LOOP_MINT.get()
    origin = str(origin or "").strip()
    tick = str(tick or "").strip()
    if origin and tick:
        return {tick: origin}
    return {}


def _is_verified_tick_edge(
    child: str,
    parent: str,
    tick_edges: dict[str, str],
) -> bool:
    """True only when the exact tick → origin mapping holds.

    Missing or failed edge lookup is fail-closed: the hop counts.
    Set membership is never consulted — a durable tick under a
    different durable origin must not be exempted.
    """
    if not child or not parent:
        return False
    return tick_edges.get(child) == parent


def create_session_child_depth(
    parent_session_id: str,
    *,
    sessions_dir: Any = None,
    env_depth: int | None = None,
    origins: set[str] | None = None,
    ticks: set[str] | None = None,
    tick_edges: dict[str, str] | None = None,
) -> int:
    """Agent-class depth of a new session parented to ``parent_session_id``.

    Matches the existing gates: root = 0, first child = 1, refuse when
    the result is greater than ``MAX_DELEGATION_DEPTH``. Only a verified
    tick→origin edge is stripped — an ordinary child of a loop origin
    still counts. When the chain cannot be read, only the
    ``SHAREDAGENTS_SANDBOX_AGENT_DEPTH`` lower bound applies
    (``env_depth + 1``).
    """
    if env_depth is None:
        env_depth = _sandbox_env_depth()
    env_bound = int(env_depth) + 1

    from sharedagents_session.session_index import ancestor_ids

    chain = ancestor_ids(
        parent_session_id,
        sessions_dir=sessions_dir,
    )
    if chain is None:
        return env_bound

    # One snapshot of the exact tick→origin map. Membership sets, when
    # not injected, come from that same map — never a second registry
    # call — so they cannot disagree. Failed/empty lookup → no
    # exemptions (fail closed). The sets are not an exemption source.
    if tick_edges is None:
        tick_edges = _loop_tick_origin_edges()
    if origins is None:
        origins = set(tick_edges.values())
    if ticks is None:
        ticks = set(tick_edges.keys())

    edges = {**tick_edges, **_in_flight_tick_edges()}
    parent = str(parent_session_id or "").strip()
    mint_origin, mint_tick = _LOOP_MINT.get()
    mint_origin = str(mint_origin or "").strip()
    mint_tick = str(mint_tick or "").strip()

    # NEW → parent, then parent → … → root. Skip only verified tick edges.
    counted = 0
    creating_in_flight_tick = bool(mint_origin and parent == mint_origin and not mint_tick)
    if not creating_in_flight_tick:
        counted += 1

    nodes = [parent, *chain]
    for i in range(len(nodes) - 1):
        child, anc = nodes[i], nodes[i + 1]
        if _is_verified_tick_edge(child, anc, edges):
            continue
        counted += 1
    return max(counted, env_bound)


class SessionLifecycleMixin:
    """create / list / get / close / inject-user-message session CRUD."""

    __slots__ = ()

    def create_session(
        self,
        *,
        workspace_cwd: str = "",
        workspace_origin: str = "",
        mode: str | None = None,
        parent_session_id: str = "",
        session_id: str = "",
        model: str = "",
        effort: str | None = None,
    ) -> Any:
        """Create a fresh session under ``sessions_dir``.

        ``effort`` is the level picked for the first turn (a new chat's
        ``initial_effort``); it is remembered for the chosen model. Without
        it a top-level session starts with the level remembered for its
        model, if any (:mod:`sharedagents_model.effort_prefs`).

        Returns the :class:`sharedagents_session.session.Session` object.
        Composes the system prompt at create-time (so the same prompt
        rides every round of this session) and stamps the new
        attribution fields via
        :func:`session_log.record_session_creation`. The first
        ``session.save()`` picks up those fields.

        Also calls ``self.engine.attach_session(sid, session)`` so the
        local-LLM engine can allocate its per-session live state
        (cancellation flags, promoted-tools set, mode overrides).
        No-op for CLI engines.
        """
        from sharedagents_session.session import Session

        from . import session_log

        parent_session_id = str(parent_session_id or "").strip()
        if parent_session_id:
            from sharedagents_agent.sandbox_broker import MAX_DELEGATION_DEPTH

            child_depth = create_session_child_depth(
                parent_session_id,
                sessions_dir=getattr(self, "sessions_dir", None),
            )
            if child_depth > MAX_DELEGATION_DEPTH:
                raise ValueError(
                    f"delegation-depth cap ({MAX_DELEGATION_DEPTH + 1} levels) "
                    f"reached — refused create_session parent="
                    f"{parent_session_id} (child depth {child_depth})."
                )

        mode = mode or self.permission_mode
        from sharedagents_agent.subagent_policy import profile_skips_memory

        skip_memory = profile_skips_memory(self.cfg)
        system_prompt = self.compose_system_prompt(
            mode=mode,
            skip_memory=skip_memory,
        )

        # Default model/endpoint sourced from cfg (the local-LLM hot
        # path expects them present on the saved session). CLI-engine
        # agents stamp the EFFECTIVE WIRE MODEL and leave the HTTP
        # fields empty so the TUI footer / per-turn stats row identify
        # the CLI's actual model ("claude-opus-4-8[1m]" / "gpt-5.6-sol" /
        # the live ACP model) instead of leaking the global config's
        # ``upstream_model`` (e.g. ``gpu-a/qwen3.6-27b``) — which the
        # CLI doesn't use and which made every CLI session falsely
        # advertise as if it ran on the local vLLM upstream.
        #
        # ENGINE-WIDE exception (the bug was never codex-only:
        # ``engine.display_model()`` defaults to ``self.name``, so claude
        # sessions stamped the literal label "claude", which the Claude
        # CLI rejects as a model id): CLI-engine sessions stamp the
        # effective wire id that will be passed as ``--model`` — the
        # agent's validated ``upstream_model``, else the engine's
        # validated default resolution (profile config → engine config
        # file → safe pin), else ``""`` — NEVER the engine label. The
        # label is not a real model id: stamped on the session it
        # (a) made the TUI header advertise "claude"/"codex" instead of
        # the actual model, and (b) was treated by model inheritance
        # (``delegate_shims._is_agent_default_model``) as an off-default
        # override, so a CLI parent session propagated its label to every
        # delegated child, where the spawn gate refused it. Stamping the
        # wire id keeps both the header and the inheritance clean. ACP
        # engines report a REAL wire id via ``display_model()`` (the
        # server-reported / configured ``acp_model``); only their bare
        # name fallback ("cursor" when nothing is known) is label-guarded
        # into the generic resolution (which yields ``""`` for ACP kinds —
        # the ACP server owns the model).
        engine_kind = self.engine.kind
        from sharedagents_agent.external_registry import is_external_engine_kind

        is_external_engine = is_external_engine_kind(engine_kind)
        if is_external_engine:
            from sharedagents_agent.discovery import _common as _md

            _reported = str(self.engine.display_model() or "").strip()
            if _reported and _reported.lower() not in _md.CLI_ENGINE_LABELS:
                # A real wire id (ACP engines report the server-acknowledged
                # / configured model); native CLI engines never reach here —
                # their display_model() IS the label.
                primary_model = _reported
            else:
                # The stamp is the canonical id the spawn will PASS; the
                # generic resolution validates every candidate and ends in a
                # safe pin (codex) or ``""`` (claude without an explicit
                # sharedagents selection — the Claude CLI applies its own default
                # and no ``--model`` is built; antigravity, ACP) — never a
                # value the gate refuses and never the engine label.
                primary_model = _md.external_cli_effective_default(
                    engine_kind.value,
                    configured=(getattr(self.cfg, "upstream_model", "") or ""),
                    # Only an operator-EXPLICIT
                    # per-agent upstream_model may seed the claude chain —
                    # the merged global/local-LLM upstream must never ride
                    # to the Claude CLI as ``--model``.
                    configured_explicit=bool(
                        getattr(
                            self.cfg,
                            "_upstream_model_is_per_agent",
                            False,
                        )
                    ),
                )
            primary_endpoint = ""
            backend = ""
            # ACP engines know the model's real window (e.g. composer
            # 200k); other CLI engines return 0 → global default applies.
            context_window = int(self.engine.context_window(model or primary_model) or 0)
        else:
            # Local engines read the MERGED cfg: per-agent upstream_model
            # (a Builder override) wins for new sessions;
            # without an override the merged global/builtin value is the
            # default — exactly the AgentConfig.load merge order.
            primary_model = getattr(self.cfg, "upstream_model", "") or ""
            primary_endpoint = getattr(self.cfg, "upstream_endpoint", "") or ""
            backend = getattr(self.cfg, "backend", "") or ""
            context_window = int(getattr(self.cfg, "upstream_max_context", 0) or 0)

        chosen_model = model or primary_model
        chosen_endpoint = primary_endpoint
        chosen_backend = backend
        chosen_context = context_window
        chosen_row: dict[str, Any] | None = None
        # ``initial_model`` from the web may be an operator alias (e.g. ``local-fp8``).
        # Stamp the wire id + that row's endpoint, not the alias on the agent's
        # primary endpoint — otherwise DeepSeek gets the local alias and 400s.
        # Mode sentinels (``auto``) and unresolved leftovers (``gpt-5.6-sol``)
        # must not stamp: fall back to the agent pin.
        if model and not is_external_engine:
            chosen_model, row = resolve_local_initial_model(
                str(model),
                primary_model,
                self.cfg.models or [],
            )
            if row is None:
                refreshed = _model_row_for_initial_selection(self.cfg, str(model))
                if refreshed:
                    row = refreshed
                    chosen_model = str(refreshed.get("id") or model)
            if row:
                chosen_row = row
                base = str(row.get("baseUrl") or row.get("endpoint") or "")
                if base:
                    chosen_endpoint = base.rstrip("/")
                if row.get("backend"):
                    chosen_backend = str(row["backend"])
                cw = int(row.get("contextWindow") or 0)
                if cw:
                    chosen_context = cw

        session = Session.new(
            system_prompt=system_prompt,
            model=chosen_model,
            endpoint=chosen_endpoint,
            sessions_dir=self.sessions_dir,
            system_mode=mode,
            workspace_cwd=workspace_cwd,
            workspace_origin=workspace_origin,
            backend=chosen_backend,
            context_window=chosen_context,
            parent_session_id=parent_session_id,
            session_id=session_id,
        )
        if effort is not None or not parent_session_id:
            try:
                session.effort = initial_session_effort(
                    self.cfg,
                    str(getattr(self, "name", "") or ""),
                    chosen_model,
                    chosen_row,
                    is_external=is_external_engine,
                    picked=effort,
                )
            except Exception as exc:  # noqa: BLE001 — a preference never blocks a session
                logging.getLogger("sharedagents.agents.session").warning(
                    "initial session effort failed (%s: %s)",
                    type(exc).__name__,
                    exc,
                )
        session_log.record_session_creation(
            session,
            agent_name=self.name,
            engine=self.engine.kind.value,
            system_prompt=system_prompt,
            sessions_dir=self.sessions_dir,
            parent_session_id=parent_session_id,
        )
        if getattr(session, "effort", ""):
            # ``session_started`` does not carry the effort; persist it so a
            # turn that reloads the session from its log runs the same level.
            try:
                session.save_header_fields(["effort"], root=self.sessions_dir)
            except Exception as exc:  # noqa: BLE001
                logging.getLogger("sharedagents.agents.session").warning(
                    "initial session effort not persisted (%s)",
                    type(exc).__name__,
                )
        # Engine-side per-session in-memory state (no-op on CLI engines
        # per the Engine ABC's default).
        try:
            self.engine.attach_session(session.hash, session)
        except Exception:  # noqa: BLE001
            # attach_session is best-effort scaffolding; a buggy
            # override must not block session creation.
            pass
        # Seed the per-session prompt cache with the build
        # just composed above (mode=mode, default workspace) so the session's
        # first ordinary turn reuses it instead of recomposing.
        self._system_prompt_cache[session.hash] = (
            (mode, "", ""),
            system_prompt,
        )
        # session_started was already emitted by record_session_creation
        # above (it inserts the sessions_meta row + writes the
        # self-describing header line). No legacy whole-JSON save.
        return session

    def list_sessions(self) -> list[dict[str, Any]]:
        """Return one row per session from ``sessions.db``.

        Schema matches what ``CoreBackend.list_sessions`` returns —
        hash, title, created_at, turn_count, agent_name, engine,
        workspace_*. Most-recent first. A single SQL query replaces the
        legacy per-file JSON glob+parse.
        """
        from sharedagents_session.session_index import SessionIndex, connect_sessions_db

        rows: list[dict[str, Any]] = []
        db = self.sessions_dir / "sessions.db"
        if not db.exists() and not self.sessions_dir.is_dir():
            return rows
        try:
            # Lazy index: bring sessions.db current with any JSONL writes
            # before listing (the writer appends JSONL only).
            SessionIndex(self.sessions_dir).refresh()
            con = connect_sessions_db(db)
            try:
                # Scope to this agent's own sessions. The shared
                # sessions.db means a bare SELECT would return every
                # other agent's rows too — leaks them into the TUI
                # picker / CLI ``sharedagents session list``. Empty
                # agent_name stays visible (preserves the legacy /
                # unattributed edge).
                own_agent = str(getattr(self, "name", "") or "")
                cur = con.execute(
                    "SELECT session_hash, title, created_at, agent_name, "
                    "engine, workspace_cwd, workspace_origin, last_turn_idx "
                    "FROM sessions_meta "
                    "WHERE agent_name = ? OR agent_name = '' "
                    "ORDER BY last_turn_ts DESC, created_at DESC",
                    (own_agent,),
                )
                for r in cur.fetchall():
                    rows.append(
                        {
                            "hash": r["session_hash"],
                            "title": r["title"] or "",
                            "created_at": r["created_at"] or "",
                            "turn_count": max(0, int(r["last_turn_idx"]) + 1),
                            "agent_name": r["agent_name"] or "",
                            "engine": r["engine"] or "",
                            "workspace_cwd": r["workspace_cwd"] or "",
                            "workspace_origin": r["workspace_origin"] or "",
                        }
                    )
            finally:
                con.close()
        except Exception:  # noqa: BLE001
            return rows
        return rows

    def get_session(self, sid: str) -> Any:
        """Load a session by id from ``sessions.db`` + the JSONL event log.

        Header fields come from the ``sessions_meta`` row; ``turns`` are
        folded from ``<sid>.jsonl``. Returns ``None`` when the session
        is unknown. Defense-in-depth: if there's no db row but the engine
        has the session attached (the bridge calls ``attach_session``
        right after ACP ``session/new``), recover from the engine cache.
        """
        from sharedagents_session.session import Session, build_new_chat_title
        from sharedagents_session.session_index import SessionIndex, connect_sessions_db
        from sharedagents_session.session_reader import fold_jsonl_into_turns

        db = self.sessions_dir / "sessions.db"
        jsonl = self.sessions_dir / f"{sid}.jsonl"
        row = None
        if jsonl.exists() or db.exists():
            try:
                # Lazy index: tail this session's JSONL into sessions.db
                # so the header row reflects the latest writes.
                SessionIndex(self.sessions_dir).tail(sid)
                con = connect_sessions_db(db)
                try:
                    row = con.execute("SELECT * FROM sessions_meta WHERE session_hash=?", (sid,)).fetchone()
                finally:
                    con.close()
            except Exception:  # noqa: BLE001
                row = None
        if row is None:
            return getattr(self.engine, "_sessions", {}).get(sid)

        d = dict(row)
        # Reject a known-sid lookup that crosses agent boundaries
        # (the unified sessions.db now shares the table across N
        # agents, so a CLI-engine ``load_session`` / prompt with
        # someone else's sid would otherwise attach the wrong
        # agent's session here). Empty agent_name keeps unattributed
        # / legacy sessions reachable.
        sess_agent = str(d.get("agent_name") or "")
        own_agent = str(getattr(self, "name", "") or "")
        if sess_agent and own_agent and sess_agent != own_agent:
            from sharedagents_session.session_state import active_agent

            if (
                active_agent(
                    self.sessions_dir,
                    sid,
                    default=sess_agent,
                )
                != own_agent
            ):
                return getattr(self.engine, "_sessions", {}).get(sid)
            d["agent_name"] = own_agent
        sess = Session(
            hash=d.get("session_hash") or sid,
            title=d.get("title") or build_new_chat_title(d.get("created_at")),
            created_at=d.get("created_at") or "",
            system_prompt=d.get("system_prompt") or "",
            model=d.get("model") or "",
            endpoint=d.get("endpoint") or "",
            turns=[],
            system_mode=d.get("system_mode") or "",
            workspace_cwd=d.get("workspace_cwd") or "",
            workspace_origin=d.get("workspace_origin") or "",
            backend=d.get("backend") or "",
            context_window=int(d.get("context_window") or 0),
            compaction_count=int(d.get("compaction_count") or 0),
            memory_flush_compaction_count=int(d.get("memory_flush_compaction_count") or -1),
            last_memory_flush_at=d.get("last_memory_flush_at") or "",
            last_prompt_tokens=int(d.get("last_prompt_tokens") or 0),
            pinned_facts=_loads_list(d.get("pinned_facts")),
            compaction_summary=d.get("compaction_summary") or "",
            compaction_turn=int(d.get("compaction_turn") or 0),
            compaction_last_stats=_loads_json_dict(d.get("compaction_last_stats")),
            session_summary=d.get("session_summary") or "",
            reasoning_visibility=d.get("reasoning_visibility") or "stream",
            tool_output_visible=bool(d.get("tool_output_visible") or 0),
            effort=d.get("effort") or "",
            speed=d.get("speed") or "",
            # Restore the legacy title-refinement counter; active title
            # lifecycle triggers are derived from the turn number.
            title_refinements=int(d.get("title_refinements") or 0),
            # Pinned title (operator /title rename or Tasks-board task
            # name): suppresses every title-agent trigger while set.
            title_locked=bool(d.get("title_locked") or False),
        )
        # TIER 2c — ONE shared hydration path for the goal/cost projection
        # (also used by the gateway's CoreBackend lazy loader, so both
        # restore goal_id/goal_turn_count/goal_terminal/cost identically and
        # reconcile disagreements from the durable goal registry).
        try:
            from sharedagents_session.session_hydration import (
                hydrate_session_goal_fields,
            )

            hydrate_session_goal_fields(
                sess,
                d,
                agent_name=own_agent or self.name,
                sessions_dir=self.sessions_dir,
            )
        except Exception:  # noqa: BLE001 — hydration must never break a load
            pass
        # Turns folded from the append-only event log.
        try:
            sess.turns = fold_jsonl_into_turns(self.sessions_dir / f"{sid}.jsonl")
            # OpenCode ``compaction.prune`` parity: OFF by default. Pruning on
            # load rewrites earlier prompt messages and costs the upstream
            # prefix cache; tool results are pruned at compaction instead.
            # Opt in with ``compact_prune_on_load``. Best-effort.
            try:
                from sharedagents_agent.compaction import prune_session_on_load

                prune_session_on_load(
                    sess,
                    enabled=bool(getattr(self.cfg, "compact_prune_on_load", False)),
                )
            except Exception:  # noqa: BLE001 — load must not fail on a prune error
                pass
        except Exception:  # noqa: BLE001
            sess.turns = []
        try:
            self.engine.attach_session(sess.hash, sess)
        except Exception:  # noqa: BLE001
            pass
        return sess

    def close_session(self, sid: str) -> None:
        """Drop per-session live state on the engine.

        Reference: ``CoreBackend.close_session`` (acp_bridge.py L1644).
        Forwards to ``engine.detach_session(sid)`` (no-op default on
        the Engine ABC). The on-disk session file is NOT deleted.

        Fires Claude Code's ``SessionEnd`` hook on the way out so an
        ECC plugin can flush state, persist conversation summary, or
        emit a notification. The hook is fired fire-and-forget when an
        event loop is available — ``close_session`` itself stays sync
        for callers that don't want to await. When no loop is running
        (process-shutdown / direct sync caller) the hook is skipped
        rather than blocking on a new loop; SessionEnd is informational
        and should never gate cleanup.
        """
        # Cross-agent guard: under the unified sessions.db, a known
        # sid for one agent would otherwise be closeable through
        # another agent's ACP backend — get_session() rejects the
        # row but the writer side here still appends a
        # ``session_closed`` event to the foreign JSONL. Read the
        # row's ``agent_name`` directly and skip every write side
        # (writer emit + writer cache eviction) when it disagrees.
        own_agent = str(getattr(self, "name", "") or "")
        sess_agent = ""
        try:
            from sharedagents_session.session_index import (
                SessionIndex,
                connect_sessions_db,
            )

            idx = SessionIndex(self.sessions_dir)
            try:
                idx.tail(sid)
            except Exception:  # noqa: BLE001
                pass
            con = connect_sessions_db(idx.db_path)
            try:
                r = con.execute(
                    "SELECT agent_name FROM sessions_meta WHERE session_hash=?",
                    (sid,),
                ).fetchone()
            finally:
                con.close()
            if r is not None:
                sess_agent = (r["agent_name"] or "") if r is not None else ""
        except Exception:  # noqa: BLE001
            sess_agent = ""
        if sess_agent and own_agent and sess_agent != own_agent:
            # Foreign session — leave the writer alone. The engine's
            # in-memory cache for this sid (if any) is also foreign;
            # detach is a no-op when the engine never attached it.
            try:
                self.engine.detach_session(sid)
            except Exception:  # noqa: BLE001
                pass
            return

        session = None
        try:
            session = self.get_session(sid)
        except Exception:  # noqa: BLE001
            session = None
        try:
            self.engine.detach_session(sid)
        except Exception:  # noqa: BLE001
            # detach is best-effort cleanup; a buggy override must not
            # block close.
            pass
        # Emit session_closed + evict the cached SessionWriter so its
        # jsonl/dir/lock fds are released (otherwise the process-global
        # writer registry never evicts and leaks fds).
        try:
            from sharedagents_session.session_writer import (
                close_session_writer,
                get_session_writer,
            )

            try:
                get_session_writer(self.sessions_dir, sid).session_closed(
                    reason="close_session",
                )
            except Exception:  # noqa: BLE001
                pass
            close_session_writer(self.sessions_dir, sid)
        except Exception:  # noqa: BLE001
            pass
        # SessionEnd — Claude Code's lifecycle hook. Fire-and-forget.
        try:
            import asyncio as _asyncio

            loop = _asyncio.get_event_loop()
            if loop.is_running():
                _asyncio.ensure_future(
                    self._run_hook_event(
                        "SessionEnd",
                        session,
                        {
                            "reason": "close_session",
                            "session_id": sid,
                        },
                    ),
                )
        except RuntimeError:
            # No running loop — best-effort skip (process is shutting
            # down or caller is in a thread without an asyncio loop).
            pass
        except Exception:  # noqa: BLE001
            pass

    def inject_user_message(self, sid: str, text: str) -> bool:
        """Forward a mid-turn user message to the engine.

        The frontend pushes the operator's fresh prompt here when a
        turn is already streaming so the new instruction lands at the
        next round boundary and can steer the in-flight plan without
        waiting for it to terminate.

        Returns ``True`` when queued. Engines that don't implement the
        injection protocol (CLI bridges, gemini-cli) return ``False``
        — caller should fall back to local queue + post-turn dispatch.
        """
        inj = getattr(self.engine, "inject_user_message", None)
        if inj is None:
            return False
        try:
            return bool(inj(sid, text))
        except Exception:  # noqa: BLE001
            return False
