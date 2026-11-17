"""North gateway and south orchestrator runtime.

Two cooperating processes (§13 split):

* **NORTH gateway** (default ``:8100``) — ACP/REST/SSE edge; proxies
  commands south; serves reads from the shared session filesystem.
* **SOUTH orchestrator** (``:8099``) — owns agents, SessionWriter,
  turn execution, and in-process backends.

Each side binds one Streamable HTTP listener and mounts
``/acp/<agent>`` for every discovered agent.

ACP-compliance:

* Each process IS an ACP server externally — same Starlette app a
  per-agent daemon would expose, mounted once per agent name.
* South backends are in-process; north backends proxy over ACP-HTTP.
  For local-LLM agents :class:`AgentACPBackend` holds an in-process
  :class:`CoreBackend` wrapper that delegates the multi-round turn
  loop to ``Agent.run_turn`` → :class:`LocalLLMEngine.run_turn`. For
  CLI agents (claude / codex / antigravity) it routes through
  :class:`Agent` → :class:`ExternalCliEngine` → the per-turn CLI
  subprocess; the CLI process itself runs only for the duration of
  one turn.
* No protocol additions. ``new_session`` against ``/acp/jarvis``
  allocates a session via the ``jarvis`` agent's backend; subsequent
  calls for that session id route to the same backend.

Cancel cascade:

* North SIGTERM drains the edge listener without aborting south turns.
* South SIGTERM stops every per-agent heartbeat scheduler, then drains
  its HTTP listener.
* ACP ``cancel(session_id)`` routes through
  :meth:`AgentACPBackend.cancel` → :meth:`Engine.abort(turn_id)`
  which SIGTERMs the active subprocess (worker OR tool runner) via
  :class:`TurnState.stop`. Two ESC presses inside
  ``_HARD_KILL_WINDOW_S`` escalate to SIGKILL + ``killpg``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import secrets
import signal
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sharedagents_agent.agent import (
    AgentConfig,
    agents_root,
    load_global_config,
    sharedagents_root,
)

_LOG = logging.getLogger("sharedagents.gateway")


# ---- defaults --------------------------------------------------------

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8100
# Hosts that keep the OpenAI surface inside the local perimeter. Any bind
# outside this set exposes /openai/v1 to the network, which makes a bearer
# token mandatory.
_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "::1", "localhost"})
# The orchestrator (south) binds loopback by default — only the north gateway
# (same host) talks to it (§13 same-host split).
_DEFAULT_ORCH_HOST = "127.0.0.1"
_DEFAULT_ORCH_PORT = 8099
# How many of the most-recent sessions the orchestrator re-derives subagent
# parent links for on boot (delegation-tree self-heal). Bounded so startup
# stays cheap; the one-time full repair is a manual backfill.
_PARENT_BACKFILL_ON_BOOT = 50
# Revisit a bounded recent window on every orchestrator boot.  This repairs
# sessions created before the detached turn-1 title task was available (and
# includes delegated child sessions) without making startup walk the archive.
_TITLE_RECONCILE_ON_BOOT = 100
_TITLE_RECONCILE_CONCURRENCY = 4


def _attach_gateway_logfile_handler(role: str = "gateway") -> None:
    """Wire a JSONL handler on the ``sharedagents`` logger tree.

    The gateway process has historically relied on bare ``_LOG.info``
    calls reaching stdout via stdlib logging's defaults. There are no
    defaults: a logger with no handler falls through to the
    ``lastResort`` handler, which only emits WARNING+ to stderr. So
    every INFO event from gateway.py / openai_endpoint.py /
    local_llm.py / acp_bridge.py was silently dropped — operators
    tailing the gateway log saw nothing between boot and shutdown.

    Attach a :class:`JsonlLogHandler` at INFO level pointed at the
    unified ``layout.gateway_logfile()``. Every line is JSONL with
    origin tags (matches the JsonLogWriter contract used by
    ``tools/_log.log_event``, heartbeat, tui, telegram), so the
    shared gateway.log stays one consistent format. Idempotent: a
    handler whose ``baseFilename`` already matches is skipped, so
    test harnesses + any future re-call don't double-write.

    (A plain RotatingFileHandler would emit plaintext
    ``%(asctime)s %(levelname)s …`` lines into the same JSONL file,
    corrupting the format.)
    """
    from sharedagents_core.jsonlog import JsonlLogHandler
    from sharedagents_core.layout import gateway_logfile

    log_path = gateway_logfile()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    root = logging.getLogger("sharedagents")
    target = str(log_path.resolve())
    for h in root.handlers:
        if getattr(h, "baseFilename", None) == target:
            return
    handler = JsonlLogHandler(log_path, service=role)
    handler.setLevel(logging.INFO)
    root.addHandler(handler)
    root.setLevel(logging.INFO)


# ---- AgentRegistry --------------------------------------------------
#
# In-process :class:`AgentRegistry` — every agent is loaded via
# :func:`factory.load_agent` and exposed through a single
# :class:`AgentACPBackend` instance.
#
# ``AgentRegistry.get(name)`` returns that backend (lazy-allocated on
# first access so the boot sequence doesn't have to pay the
# load-every-agent cost for agents that are never used). All callers
# (the ACP HTTP mount, the openai_endpoint helper) talk to backends
# directly without any subprocess hop.


class AgentRegistry:
    """In-process agent registry. Replaces ChildManager.

    Holds one :class:`AgentACPBackend` instance per discovered agent
    for the gateway process's lifetime. No subprocess spawn, no idle
    GC, no LRU eviction — backends are lightweight Python objects.
    """

    def __init__(self, agent_names: list[str]) -> None:
        self.agent_names = list(agent_names)
        self._backends: dict[str, Any] = {}

    def get(self, name: str) -> Any:
        """Return the in-process :class:`AgentACPBackend` for ``name``."""
        if name in self._backends:
            return self._backends[name]
        from sharedagents_agent.agents.factory import load_agent
        from sharedagents_agent.agents.gateway_shim import AgentACPBackend

        agent = load_agent(name)
        backend = AgentACPBackend(agent)
        self._backends[name] = backend
        return backend

    def keys(self) -> list[str]:
        return list(self.agent_names)

    async def shutdown(self) -> None:
        """No-op — backends hold only in-memory state."""
        return None


# ---- startup title recovery -----------------------------------------


def _find_startup_title_candidates(
    limit: int = _TITLE_RECONCILE_ON_BOOT,
) -> list[dict[str, Any]]:
    """Return prompted sessions whose title is still a New Chat placeholder.

    Placeholder matching runs *before* the recency cap so a burst of
    already-titled children cannot hide older stranded ``New Chat`` rows.
    Unused empty placeholders (no completed turn) stay out.  JSONL is
    folded first so a session written just before a restart is eligible.
    """
    from sharedagents_core.layout import sessions_root
    from sharedagents_session.session_index import SessionIndex, connect_sessions_db
    from sharedagents_session.session_title import is_default_title

    sessions_dir = sessions_root()
    index = SessionIndex(sessions_dir)
    index.refresh()
    db = sessions_dir / "sessions.db"
    if not db.exists():
        return []

    con = connect_sessions_db(db)
    try:
        rows = con.execute(
            "SELECT session_hash, title, agent_name, last_turn_idx "
            "FROM sessions_meta "
            "WHERE last_turn_idx >= 0 "
            "ORDER BY COALESCE(NULLIF(last_turn_ts, 0), updated_at) DESC, "
            "         created_at DESC",
        ).fetchall()
    finally:
        con.close()

    candidates = [dict(row) for row in rows if is_default_title(row["title"])]
    return candidates[: int(max(1, limit))]


async def _reconcile_startup_session_titles(
    manager: Any,
    *,
    limit: int = _TITLE_RECONCILE_ON_BOOT,
) -> int:
    """Best-effort recovery for recent sessions stranded on ``New Chat``.

    The scan and each title call run off the gateway's serving path.  We call
    the existing ``Agent.schedule_title_creation`` with lifecycle turn 1 so
    it uses the normal prompt validation, persistence, and Jarvis default
    title model.  A small semaphore prevents a restart from issuing 100
    side calls at once.
    """
    try:
        candidates = await asyncio.to_thread(
            _find_startup_title_candidates,
            limit,
        )
    except Exception as exc:  # noqa: BLE001 — title recovery is best-effort
        _LOG.warning("startup: title recovery scan failed: %s", exc)
        return 0
    if not candidates:
        return 0

    from sharedagents_session.session_title import is_default_title

    try:
        registry_names = list(manager.keys())
    except Exception:  # noqa: BLE001
        registry_names = []
    fallback_names = [name for name in registry_names if name]
    semaphore = asyncio.Semaphore(_TITLE_RECONCILE_CONCURRENCY)
    recovered = 0

    async def _one(row: dict[str, Any]) -> None:
        nonlocal recovered
        sid = str(row.get("session_hash") or "").strip()
        if not sid:
            return
        owner = str(row.get("agent_name") or "").strip()
        agent_names = [owner] if owner else list(dict.fromkeys(fallback_names))

        async with semaphore:
            agent_obj: Any = None
            session: Any = None
            for agent_name in agent_names:
                try:
                    backend = manager.get(agent_name)
                    agent_obj = getattr(backend, "agent", None)
                    if agent_obj is None:
                        continue
                    session = agent_obj.get_session(sid)
                    if session is not None:
                        break
                except Exception:  # noqa: BLE001 — one bad legacy row is skippable
                    agent_obj = None
                    session = None
            if agent_obj is None or session is None:
                return

            turns = list(getattr(session, "turns", None) or [])
            if not turns or not isinstance(turns[0], dict):
                return
            from sharedagents_session.session_title import opening_prompt_text

            opening_prompt = None
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                user = turn.get("user")
                text = opening_prompt_text(user)
                if not text or text == "(empty)":
                    continue
                opening_prompt = user
                if text != "[image]":
                    break
            if opening_prompt is None:
                return

            try:
                task = agent_obj.schedule_title_creation(
                    session,
                    # Recovery uses the same opening-prompt path as turn 1,
                    # even when the stranded session already has many turns.
                    turn_number=1,
                    prompt=opening_prompt,
                )
                if task is None:
                    return
                await task
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — a failed title is retryable
                _LOG.warning(
                    "startup: title recovery failed for session %s",
                    sid[:12],
                    exc_info=True,
                )
                return
            if not is_default_title(getattr(session, "title", "")):
                recovered += 1

    tasks = [
        asyncio.create_task(
            _one(row),
            name=f"startup_session_title_{str(row.get('session_hash') or '')[:12]}",
        )
        for row in candidates
    ]
    # Keep one malformed/retired legacy row from aborting recovery for the
    # rest of the recent window.  Individual failures are already logged by
    # ``_one`` and remain retryable on the next boot.
    await asyncio.gather(*tasks, return_exceptions=True)
    _LOG.info(
        "startup: repaired %d/%d recent placeholder session title(s)",
        recovered,
        len(candidates),
    )
    return recovered


# ---- gateway server --------------------------------------------------


#: Keys ``load_shared_secrets_into_environ`` put into ``os.environ`` and the
#: value it put there. Lets a later reload rotate or remove exactly those keys
#: while a value the process already had (systemd ``EnvironmentFile``, shell)
#: keeps winning.
_SHARED_SECRETS_APPLIED: dict[str, str] = {}
#: ``_config_sig`` of the shared file at the last load (watcher change probe).
_SHARED_SECRETS_SIG: tuple | None = None


def load_shared_secrets_into_environ(home: Path) -> int:
    """Merge ``<home>/secrets.env`` into ``os.environ``.

    Returns the number of keys added, changed or removed. Called at gateway
    boot, on every SIGHUP and by the config watcher when the file changes, so
    an edit (including one saved from the web UI) takes effect without
    restarting the systemd unit.

    Precedence: whatever the process env held that this loader did NOT put
    there wins — systemd ``EnvironmentFile`` / ``/etc/environment`` are
    deliberate overrides. Keys the loader itself set follow the file: a new
    value replaces the old one and a key removed from the file is removed from
    the env. Don't pollute the process env with per-agent overrides — those
    stay in ``cfg.api_keys`` so each agent gets its own keys and one agent's
    value doesn't leak into another agent's MCP subprocess.
    """
    global _SHARED_SECRETS_SIG
    sec = home / "secrets.env"
    sig = _config_sig(sec)
    kvs: dict[str, str] = {}
    if sec.is_file():
        try:
            from sharedagents_agent.agent import _parse_dotenv, _warn_if_world_readable

            _warn_if_world_readable(sec)
            kvs = _parse_dotenv(sec.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("failed to load shared secrets.env (%s)", exc)
            return 0
    changed = 0
    for k, v in kvs.items():
        cur = os.environ.get(k)
        if cur is not None and _SHARED_SECRETS_APPLIED.get(k) != cur:
            continue  # set outside this loader: it wins
        if cur != v:
            os.environ[k] = v
            changed += 1
        _SHARED_SECRETS_APPLIED[k] = v
    for k in [k for k in _SHARED_SECRETS_APPLIED if k not in kvs]:
        if os.environ.get(k) == _SHARED_SECRETS_APPLIED.pop(k):
            del os.environ[k]
            changed += 1
    # A failed read/apply must remain eligible for the next watcher poll.
    _SHARED_SECRETS_SIG = sig
    if changed:
        _LOG.info(
            "[secrets] applied %d key change(s) from %s to process env "
            "(systemd EnvironmentFile / shell wins on collision)",
            changed,
            sec,
        )
    return changed


def refresh_shared_secrets_if_changed(home: Path) -> int:
    """Reload the shared ``secrets.env`` when its signature moved since the
    last load; ``0`` (no work) otherwise."""
    if _config_sig(home / "secrets.env") == _SHARED_SECRETS_SIG:
        return 0
    return load_shared_secrets_into_environ(home)


def reload_registry_configs(registry: Any) -> dict[str, Any]:
    """Re-read every in-process agent's on-disk config and apply the
    hot-reloadable fields in place (the SIGHUP config-reload path).

    Walks ``registry._backends`` (SOUTH :class:`AgentRegistry`).
    A NORTH ``RemoteAgentRegistry`` has no ``_backends`` — turns run on the
    orchestrator there, so this is a no-op and returns ``{}``. Each backend
    reload is isolated: one bad agent never blocks the others.

    Returns ``{agent_name: ConfigReloadReport}`` for every reloaded backend.
    """
    reports: dict[str, Any] = {}
    backends = getattr(registry, "_backends", None)
    if not isinstance(backends, dict):
        return reports
    for name, backend in list(backends.items()):
        reload = getattr(backend, "reload_config", None)
        if reload is None:
            continue
        try:
            reports[name] = reload()
        except Exception as exc:  # noqa: BLE001 - isolate one bad agent
            _LOG.warning("SIGHUP config reload failed for agent %r: %s", name, exc)
    return reports


# ---- ambient config-file watcher (mtime poll) ------------------------
#
# Complements the SIGHUP path: instead of requiring `kill -HUP`, a
# background task polls the global + per-agent config.json mtimes and
# hot-reloads changed agents automatically. Reuses
# ``AgentACPBackend.reload_config`` (idempotent + error-swallowing), so:
#   * the runtime's OWN config writes (`/model`, `cfg.save()`) reload to
#     an identical state → cheap no-op, no feedback storm;
#   * a torn read of a non-atomic write raises inside reload_from_disk →
#     swallowed → the tick is skipped and retried next poll.
# Uncontrolled timing (a poll can fire mid-turn) is handled by DEFERRING a
# busy agent to the next tick — the guarantee the operator-timed SIGHUP
# path leaves to the operator.


def _config_sig(path: Path) -> tuple | None:
    """(mtime_ns, size) signature for a config file, or ``None`` if absent.

    ``st_mtime_ns`` (not float ``st_mtime``) + ``st_size`` so two edits inside
    one coarse-mtime tick still register as a change on filesystems with 1 s
    mtime resolution.
    """
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _agent_config_sig(
    backend: Any,
    global_sig: tuple | None,
    shared_secrets_sig: tuple | None = None,
) -> tuple:
    """Combined signature for one agent: (global config, its own config,
    shared ``secrets.env``, its own ``secrets.env``).

    An agent's effective config merges the global file over its per-agent
    file, and its ``api_keys`` layer the per-agent secrets over the shared
    ones, so any of them changing means the agent must reload.
    """
    cfg = getattr(getattr(backend, "agent", None), "cfg", None)
    cpath = getattr(cfg, "config_path", None)
    own = _config_sig(cpath) if cpath is not None else None
    name = getattr(cfg, "name", None)
    own_secrets = None
    if isinstance(name, str) and name:
        from sharedagents_agent.agent import sharedagents_root

        own_secrets = _config_sig(sharedagents_root() / "agents" / name / "secrets.env")
    return (global_sig, own, shared_secrets_sig, own_secrets)


def _shared_secrets_sig() -> tuple | None:
    from sharedagents_agent.agent import sharedagents_root

    return _config_sig(sharedagents_root() / "secrets.env")


def config_watch_seed(registry: Any) -> dict[str, tuple]:
    """Snapshot the CURRENT per-agent config signatures.

    Called once before the watch loop so tick 1 doesn't spuriously reload
    every agent — boot already loaded each config fresh. Returns
    ``{name: sig}`` (empty for a NORTH proxy with no ``_backends``).
    """
    from sharedagents_agent.agent import global_config_path

    acked: dict[str, tuple] = {}
    backends = getattr(registry, "_backends", None)
    if not isinstance(backends, dict):
        return acked
    gsig = _config_sig(global_config_path())
    ssig = _shared_secrets_sig()
    for name, backend in list(backends.items()):
        acked[name] = _agent_config_sig(backend, gsig, ssig)
    return acked


def config_watch_scan(registry: Any, acked: dict[str, tuple]) -> tuple[list[str], list[str]]:
    """One poll pass. Reload every agent whose (global, own) config signature
    changed since ``acked`` AND which is idle; mutate ``acked`` in place.

    A busy agent (``has_active_turn()``) is left UN-acked so it's re-checked
    on the next pass and reloaded once it goes idle — a poll never mutates cfg
    mid-round. Walks ``registry._backends``; a NORTH proxy (no ``_backends``)
    is a no-op. Returns ``(reloaded, deferred)`` agent-name lists.
    """
    from sharedagents_agent.agent import global_config_path, sharedagents_root

    reloaded: list[str] = []
    deferred: list[str] = []
    # Shared secrets reach local tool subprocesses through os.environ; the
    # per-agent reload below refreshes cfg.api_keys (external CLI spawn env).
    try:
        refresh_shared_secrets_if_changed(sharedagents_root())
    except Exception as exc:  # noqa: BLE001 - never let secrets break the scan
        _LOG.warning("shared secrets refresh failed: %s", exc)
    backends = getattr(registry, "_backends", None)
    if not isinstance(backends, dict):
        return reloaded, deferred
    gsig = _config_sig(global_config_path())
    ssig = _shared_secrets_sig()
    for name, backend in list(backends.items()):
        sig = _agent_config_sig(backend, gsig, ssig)
        if acked.get(name) == sig:
            continue
        has_active = getattr(backend, "has_active_turn", None)
        if callable(has_active):
            try:
                if has_active():
                    deferred.append(name)
                    continue  # retry next tick (sig stays un-acked)
            except Exception:  # noqa: BLE001 - treat a flaky probe as idle
                pass
        reload = getattr(backend, "reload_config", None)
        if reload is not None:
            try:
                report = reload()
                reloaded.append(name)
                applied = getattr(report, "applied", []) or []
                restart_only = getattr(report, "restart_only", []) or []
                if applied or restart_only:
                    _LOG.info(
                        "[config] watch reload agent=%s applied=%s restart_only=%s",
                        name,
                        ",".join(applied) or "-",
                        ",".join(restart_only) or "-",
                    )
                if restart_only:
                    _LOG.warning(
                        "[config] agent=%s changed restart-only field(s) %s — a "
                        "gateway restart is required to apply them",
                        name,
                        ",".join(restart_only),
                    )
            except Exception as exc:  # noqa: BLE001 - isolate one bad agent
                _LOG.warning("config watch reload failed for agent %r: %s", name, exc)
        # Ack even on reload failure so a persistently-bad file doesn't hot-loop
        # every tick; the next genuine edit bumps the sig and retries.
        acked[name] = sig
    return reloaded, deferred


async def _config_watch_loop(registry: Any, interval_s: float, stop_event: asyncio.Event) -> None:
    """Poll config files every ``interval_s`` and hot-reload changed agents.

    Cancellation-friendly: waits on ``stop_event`` with an ``interval_s``
    timeout, so shutdown returns promptly instead of after a full interval.
    """
    acked = config_watch_seed(registry)
    _LOG.info(
        "[config] mtime watch started (interval=%.1fs, %d agent(s))",
        interval_s,
        len(acked),
    )
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
            break  # stop signalled
        except asyncio.TimeoutError:
            pass  # interval elapsed — poll
        except asyncio.CancelledError:
            break
        try:
            config_watch_scan(registry, acked)
        except Exception as exc:  # noqa: BLE001 - never let a tick kill the loop
            _LOG.warning("config watch tick failed: %s", exc)
    _LOG.info("[config] mtime watch loop stopped")


def _install_sighup_secrets_reload(registry_getter: Any = None) -> None:
    """SIGHUP → re-read ``~/.sharedagents/secrets.env`` into ``os.environ`` AND
    re-apply every live agent's hot-reloadable config from disk.

    Lets the operator rotate provider keys / edit model + agent config
    without restarting the systemd-managed gateway:

      1. Edit ``~/.sharedagents/secrets.env`` and/or ``config.json`` /
         ``agents/<name>/config.json`` with the new values
      2. ``kill -HUP $(systemctl --user show -p MainPID --value sharedagents-gateway.service)``
      3. The next turn uses the new key + config

    Config reload applies scalar/hot fields (models, sampling, system prompt,
    permission_mode, cost limits, memory settings, …) to the SAME live ``cfg``
    object the mounted ACP app holds; structural changes (engine kind, agent
    profile, bind host/port, heartbeat) are logged as restart-only.
    ``registry_getter`` is a late-bound callable returning the agent registry
    (the ``manager`` local isn't constructed until later in boot); ``None``
    keeps the secrets-only behaviour for callers that don't have a registry.

    Per-agent ``secrets.env`` files are also re-read here, so a key rotation
    lands on SIGHUP for shared keys, per-agent keys, and any external CLI
    subprocess (claude/codex/antigravity) that already inherited the old env.

    Only the boot path installs the handler — the test suite doesn't
    need a real signal handler, and a missing-signal-support platform
    (Windows) silently no-ops.
    """
    import signal

    try:

        def _handler(_signum: int, _frame: Any) -> None:
            try:
                added = load_shared_secrets_into_environ(sharedagents_root())
                _LOG.info(
                    "[secrets] SIGHUP reload: refreshed %d key(s) from shared secrets.env into process env",
                    added,
                )
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("SIGHUP secrets reload failed: %s", exc)
            # Config hot-reload — best-effort, never let it break secrets.
            if registry_getter is not None:
                try:
                    reports = reload_registry_configs(registry_getter())
                    for name, rep in reports.items():
                        applied = getattr(rep, "applied", []) or []
                        restart_only = getattr(rep, "restart_only", []) or []
                        if applied or restart_only:
                            _LOG.info(
                                "[config] SIGHUP reload agent=%s applied=%s restart_only=%s",
                                name,
                                ",".join(applied) or "-",
                                ",".join(restart_only) or "-",
                            )
                        if restart_only:
                            _LOG.warning(
                                "[config] agent=%s changed restart-only field(s) "
                                "%s — a gateway restart is required to apply them",
                                name,
                                ",".join(restart_only),
                            )
                except Exception as exc:  # noqa: BLE001
                    _LOG.warning("SIGHUP config reload failed: %s", exc)

        signal.signal(signal.SIGHUP, _handler)
        _LOG.info(
            "[secrets] SIGHUP handler installed — send `kill -HUP <pid>` to reload shared secrets.env + agent config"
        )
    except (AttributeError, OSError, ValueError):
        # SIGHUP unsupported on this platform OR signal already in
        # use; silently skip.
        pass


def _gateway_config(global_cfg: dict | None = None) -> dict[str, Any]:
    """Resolve gateway config from ``~/.sharedagents/config.json``.

    Defaults: host=127.0.0.1, port=8100. ``host`` / ``port`` /
    ``auth_token`` overridable via the ``gateway`` block in the
    global config.
    """
    if global_cfg is None:
        global_cfg = load_global_config() or {}
    raw = global_cfg.get("gateway") or {}
    host = str(raw.get("host") or _DEFAULT_HOST)
    auth_token = raw.get("auth_token") or None
    # The OpenAI-surface bearer check is mandatory whenever the
    # endpoint is reachable from outside loopback. The "no token → accept every
    # request" path in openai_endpoint is safe ONLY under a loopback-only
    # perimeter; if the operator binds a non-loopback host without configuring a
    # token, mint an ephemeral one so the surface fails CLOSED rather than
    # serving an open, unauthenticated OpenAI API to the network. Config can set
    # the token but cannot silently leave a non-loopback bind unauthenticated.
    if auth_token is None and host not in _LOOPBACK_HOSTS:
        auth_token = secrets.token_urlsafe(32)
        _LOG.warning(
            "gateway bound to non-loopback host %r with no auth_token configured; "
            "generated an ephemeral bearer token to keep /openai/v1 authenticated. "
            "Set gateway.auth_token in the global config to pin it.",
            host,
        )
    return {
        "host": host,
        "port": int(raw.get("port") or _DEFAULT_PORT),
        "auth_token": auth_token,
    }


def _orchestrator_config(global_cfg: dict | None = None) -> dict[str, Any]:
    """Resolve the orchestrator (south) service config from
    ``~/.sharedagents/config.json``.

    Defaults: host=127.0.0.1, port=8099. ``host`` / ``port`` / ``auth_token``
    overridable via an ``orchestrator`` block. ``auth_token`` falls back to the
    gateway's token so a single shared bearer secures both surfaces unless the
    operator splits them.
    """
    if global_cfg is None:
        global_cfg = load_global_config() or {}
    raw = global_cfg.get("orchestrator") or {}
    gw = global_cfg.get("gateway") or {}
    return {
        "host": str(raw.get("host") or _DEFAULT_ORCH_HOST),
        "port": int(raw.get("port") or _DEFAULT_ORCH_PORT),
        "auth_token": raw.get("auth_token") or gw.get("auth_token") or None,
    }


# The picker mirrors the on-disk agents folder (see _discover_agents), so jarvis
# (the sole orchestrator + default) and every configured agent are shown. (The
# built-in `main` orchestrator was retired 2026-07-21 — nothing to mount for it.)


def _read_agent_configs() -> list[tuple[str, dict]]:
    """``[(name, config_dict), …]`` for every ``agents_root()/<name>/config.json``.

    Sorted by name for deterministic mount/list order. A dir without a readable
    ``config.json`` is skipped.
    """
    import json

    root = agents_root()
    if not root.is_dir():
        return []
    out: list[tuple[str, dict]] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        config_path = entry / "config.json"
        if not config_path.is_file():
            continue
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        out.append((entry.name, data if isinstance(data, dict) else {}))
    return out


def _discover_mountable_agents() -> list[str]:
    """Every configured agent that should get a live ``/acp/<name>`` route.

    The gateway MOUNT set (not the picker): a superset of the user-facing list
    that keeps the subagent utilities routable so no in-flight/legacy session
    404s. (`main` retired 2026-07-21 — no longer scaffolded or mounted.) Hidden
    runtime profiles (compaction/title/summary/teacher) are excluded — they carry no public
    ``config.json`` and are runtime-invoked only.
    """
    from sharedagents_agent.agent_profiles import resolve_profile

    out: list[str] = []
    for name, data in _read_agent_configs():
        prof_key = str(data.get("agent_profile") or "").strip()
        prof = resolve_profile(prof_key or name)
        if prof is not None and prof.opencode_mode == "hidden":
            continue
        out.append(name)
    return out


def _discover_agents() -> list[str]:
    """The user-facing agent list rendered by the picker (``/openai/v1/models``).

    The picker MIRRORS the on-disk agents folder: every agent under
    ``agents_root()`` with a readable ``config.json`` is shown, so adding or
    removing an agent directory is reflected directly (operator directive: "all
    agents in ~/.sharedagents shall be shown, and removed ones removed").

    Excluded are the agents a human never picks — they are delegation targets the
    orchestrators route to, not chat agents: the **subagent-only** utilities
    (explore/general/docs — ``opencode_mode == "subagent"``) and the **hidden**
    runtime agents (compaction/title/summary/teacher — ``opencode_mode ==
    "hidden"``). ``research`` is primary and shown under Native. ``main`` is
    SHOWN (it is not name-hidden any more); custom agents
    and external CLI/ACP agents (blank/absent profile) are SHOWN — the folder is
    the source of truth, not a curated subset.

    Discovered fresh on every call. Render it LIVE per request (see
    ``_handle_models``), never a boot-time snapshot, so a newly-added or deleted
    agent appears/disappears with no gateway restart.

    makes the whole on-disk roster visible in the picker, retiring the earlier
    curated-subset / picker-hidden-``main`` behavior; the list is live, not
    boot-cached, so add/remove reflects with no restart — operator 2026-07-15).
    """
    from sharedagents_agent.agent_profiles import resolve_profile

    out: list[str] = []
    for name, data in _read_agent_configs():
        prof_key = str(data.get("agent_profile") or "").strip()
        prof = resolve_profile(prof_key or name)
        # Subagent-only utilities (explore/general/docs) and the
        # hidden runtime agents (compaction/title/summary/teacher) are delegation targets
        # the orchestrators route to — never human-picked, so keep them out.
        if prof is not None and prof.opencode_mode in ("subagent", "hidden"):
            continue
        out.append(name)
    return out


def _retire_managed_main_agent() -> Path | None:
    """Quarantine the retired built-in ``main`` agent on upgraded installs.

    The old orchestrator is identified by its explicit managed marker,
    ``agents/main/config.json`` containing ``agent_profile == "main"``.  A
    custom operator agent that merely uses the directory/name ``main`` is not
    managed and is therefore left untouched.

    The complete home is moved below ``agents/.retired``.  Gateway discovery
    is intentionally one level deep, so the archived config cannot be mounted,
    shown in the picker, or offered as a delegation target.  Moving instead of
    deleting preserves operator data, and the missing source directory makes
    subsequent boots idempotent.  Migration failures propagate so boot fails
    closed instead of mounting the now-profileless orchestrator with the full
    tool registry.
    """
    import json
    from datetime import datetime, timezone

    root = agents_root()
    main_home = root / "main"
    config_path = main_home / "config.json"
    try:
        raw_config = config_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        data = json.loads(raw_config)
    except json.JSONDecodeError:
        # Only an exact, readable managed marker authorizes moving operator
        # data.  Malformed custom homes remain untouched.  Read failures are
        # deliberately not caught: boot must fail closed rather than let
        # discovery reinterpret an unreadable managed config as unrestricted.
        return None
    if not isinstance(data, dict):
        return None
    if str(data.get("agent_profile") or "").strip() != "main":
        return None

    archive_root = root / ".retired"
    archive_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    destination = archive_root / f"main-{stamp}"
    suffix = 2
    while destination.exists():
        destination = archive_root / f"main-{stamp}-{suffix}"
        suffix += 1
    try:
        main_home.rename(destination)
    except FileNotFoundError:
        # Concurrency: the split NORTH/SOUTH services both run _boot_scaffold_agents()
        # (both Type=simple, so `After=` does not serialize app startup). Two callers
        # can read the managed marker before either renames; the winner moves the home,
        # the loser then finds the source gone. A vanished source == the peer already
        # quarantined it → treat as success, not a boot-aborting error.
        return None
    _LOG.warning(
        "quarantined retired managed main agent: %s -> %s",
        main_home,
        destination,
    )
    return destination


def _gateway_pidfile() -> Path:
    return Path(os.environ.get("SHAREDAGENTS_HOME") or Path.home() / ".sharedagents") / "gateway.pid"


def _orchestrator_pidfile() -> Path:
    return Path(os.environ.get("SHAREDAGENTS_HOME") or Path.home() / ".sharedagents") / "orchestrator.pid"


def _resolve_port_holder(host: str, port: int) -> int | None:
    """Pid (or ``-1``) holding ``host:port`` in LISTEN state, else None.

    Reuses ``gateway_cli._port_holder`` via a deferred import — that
    module imports :func:`_gateway_pidfile` from here, so a top-level
    import would be circular. Any failure degrades to ``None`` ("looks
    free") so the pre-flight never blocks a legitimate start.
    """
    try:
        from sharedagents_gateway.gateway_cli import _port_holder
    except Exception:  # noqa: BLE001
        return None
    try:
        return _port_holder(host, int(port))
    except Exception:  # noqa: BLE001
        return None


async def _await_port_free(
    host: str,
    port: int,
    timeout: float = 10.0,
    *,
    holder_fn: Callable[[str, int], int | None] | None = None,
    sleep: Callable[[float], Any] | None = None,
) -> int | None:
    """Wait up to ``timeout`` s for a live listener on ``host:port`` to go.

    Returns ``None`` once no listener holds the port (free, or it freed
    within the window), or the pid (``-1`` when owned by another user)
    still holding it at timeout. On a gateway restart the previous
    process may still be draining its socket; polling lets the successor
    bind cleanly once it exits instead of crashing with ``OSError(98)``.
    Only LISTEN-state sockets count — ``SO_REUSEADDR`` (hypercorn's
    default) already covers a TIME_WAIT predecessor.
    """
    holder = holder_fn or _resolve_port_holder
    napper = sleep or asyncio.sleep
    deadline = time.monotonic() + max(0.0, timeout)
    pid = holder(host, int(port))
    while pid is not None and time.monotonic() < deadline:
        await napper(0.25)
        pid = holder(host, int(port))
    return pid


async def _await_serve_or_stop(
    hc_task: asyncio.Task[Any],
    stop_event: asyncio.Event,
) -> BaseException | None:
    """Block until the HTTP serve task ends OR a stop is requested.

    The port pre-flight (:func:`_await_port_free`) only NARROWS the
    zombie-gateway window — hypercorn still binds asynchronously inside
    ``hc_task``, so a bind ``OSError(98)`` can still fire there if the
    port is taken after the pre-flight check or two starts race and both
    saw it free. Awaiting ``stop_event`` alone would then hang forever as
    a non-serving zombie (the original pathology) and never retrieve the
    serve task's exception.

    Wait on BOTH; if ``hc_task`` finished first, return its exception (or
    ``None`` if it somehow returned cleanly) so the caller can surface the
    failure and exit non-zero. A normal stop returns ``None``.
    """
    stop_wait = asyncio.create_task(stop_event.wait())
    try:
        await asyncio.wait(
            {hc_task, stop_wait},
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        stop_wait.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await stop_wait
    if hc_task.done():
        # .exception() is safe here: the task is done and (since we only
        # cancel it later, in serve_gateway's cleanup) was not cancelled.
        return hc_task.exception()
    return None


def _discover_mcp_tools_blocking() -> int:
    """Run MCP discovery to completion on a FRESH event loop. Called on a
    worker thread (see ``_warm_mcp_discovery``) so the subprocess spawn/reap
    that ``stdio_client`` performs for every configured server can never
    freeze the orchestrator's main event loop — the loop that binds and
    serves :8099. Returns the number of tools cached.

    A broken MCP server (e.g. a plugin that maps to bare ``echo``, which
    exits before the JSON-RPC handshake) makes the anyio subprocess
    teardown block the *calling* loop thread for as long as the OS takes
    to reap the child — observed at boot as a 140s wedge where :8099 never
    bound and no turn ran (the whole loop was frozen, so an ``asyncio``
    timeout could not even fire). Isolating discovery on its own thread's
    loop means that freeze, if it recurs, is invisible to prod: the main
    loop stays responsive and the cache simply fills a little later.

    The discovery cache (``_DISCOVERED_TOOLS``) is a plain module-global
    list read synchronously elsewhere (``discovered_mcp_tools_sync``); the
    only async writer is this warm path, so populating it from a worker
    loop is race-free. The module's ``asyncio.Lock`` is reset to ``None``
    first so ``discover_all_tools`` recreates it bound to THIS loop rather
    than reusing a stale one from a prior thread's loop (which would raise
    'got Future attached to a different loop')."""
    import sharedagents_mcp.mcp_client as _mc
    from sharedagents_agent.tools.registry import discovered_mcp_tools

    _mc._DISCOVERED_TOOLS_LOCK = None
    tools = asyncio.run(discovered_mcp_tools())
    return len(tools)


async def _warm_mcp_discovery() -> int:
    """Populate the process-global MCP tool-discovery cache so the namespaced
    ``mcp__<server>__<tool>`` BASE tools actually reach the model.

    The worker request is built SYNCHRONOUSLY via ``discovered_mcp_tools_sync``,
    which returns nothing until an async discovery has populated the cache. That
    async discovery was never invoked anywhere, so the configured MCP servers
    (e.g. youtube) were invisible as direct tools — the model fell back to the
    generic mcp_list_tools/mcp_call path after empty ToolSearches. Warming once
    at orchestrator boot fixes it. Returns the number of tools cached; fail-safe
    (never raises, never blocks boot).

    Discovery runs on a WORKER THREAD (``asyncio.to_thread``) — never on the
    main loop — because a broken/slow MCP server's subprocess spawn/reap can
    synchronously freeze whichever loop drives it (see
    ``_discover_mcp_tools_blocking``). Off-thread, the main loop binds :8099
    and serves turns regardless of how long a wedged server takes to fail."""
    try:
        n = await asyncio.to_thread(_discover_mcp_tools_blocking)
        _LOG.info("startup: MCP discovery warmed — %d tool(s) cached", n)
        return n
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("startup: MCP discovery warm failed: %s", exc)
        return 0


async def _warm_effort_discovery() -> int:
    """Prefetch provider ``GET /models`` effort lists into the process cache.

    The sessionless WebUI picker reads that cache with ``probe=False`` so
    opening the sheet never waits on HTTP, but still shows each model's
    advertised ``supported_efforts`` once this warm has landed.
    """
    try:
        from sharedagents_agent.agent import AgentConfig
        from sharedagents_model.effort_discovery import discover_effort_capabilities

        cfg = AgentConfig.load("jarvis")
        rows = [dict(row) for row in (getattr(cfg, "models", None) or []) if isinstance(row, dict)]
        if not rows:
            return 0
        caps = await asyncio.to_thread(discover_effort_capabilities, rows, cfg)
        n = sum(1 for cap in caps if cap)
        _LOG.info(
            "startup: effort catalogue warmed — %d/%d model(s) with listed efforts",
            n,
            len(rows),
        )
        return n
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("startup: effort catalogue warm failed: %s", exc)
        return 0


async def _warm_model_catalogue() -> bool:
    """Refresh the models.dev output-limit catalogue when it is stale.

    ``max_tokens`` = min(the model's output limit, the cap) -- OpenCode's rule
    -- and the request path only READS the cached catalogue, so this keeps it
    current off the hot path.
    """
    try:
        from sharedagents_model import catalogue

        ok = bool(await asyncio.to_thread(catalogue.refresh))
        _LOG.info(
            "startup: model output-limit catalogue %s",
            "ready" if ok else "unavailable (the cap applies alone)",
        )
        return ok
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("startup: model catalogue warm failed: %s", exc)
        return False


def _boot_scaffold_agents() -> None:
    """Idempotently scaffold the default agent set the gateway boots with.

    Order (kept stable — the picker/mount tests assert against it): quarantine
    an upgraded install's retired managed ``main`` home → built-in CLI-engine
    agents (claude/codex/antigravity, when on PATH) → the ``jarvis`` orchestrator
    (local_llm, no CLI dependency) → the OpenCode profile agents
    (explore/plan/code/…). Every scaffold step is idempotent and never clobbers
    operator edits; a failing scaffold logs and continues so boot is never
    blocked.  The security migration is fail-closed and propagates failures.

    The built-in ``main`` orchestrator was **retired 2026-07-21** — it is
    deliberately NOT scaffolded here. jarvis is the sole orchestrator + the
    user-facing default. This function is the single boot-scaffold entrypoint so
    the "boot never materializes `main`" invariant is testable without standing up
    the whole async server.
    """
    _retire_managed_main_agent()

    try:
        from sharedagents_agent.agents.factory import ensure_default_external_agents

        scaffolded = ensure_default_external_agents()
        if scaffolded:
            _LOG.info(
                "auto-scaffolded built-in CLI-engine agents: %s",
                ", ".join(scaffolded),
            )
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("auto-scaffold failed (continuing): %s", exc)

    try:
        from sharedagents_agent.agents.default_scaffolds import ensure_voice_agent

        if ensure_voice_agent():
            _LOG.info("auto-scaffolded voice agent: jarvis")
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("failed to scaffold voice agent on boot: %s", exc)

    try:
        from sharedagents_agent.agents.default_scaffolds import ensure_builtin_profile_agents

        profile_agents = ensure_builtin_profile_agents()
        if profile_agents:
            _LOG.info(
                "auto-scaffolded OpenCode profile agents: %s",
                ", ".join(profile_agents),
            )
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("failed to scaffold OpenCode profile agents on boot: %s", exc)


async def serve_gateway(role: str = "gateway") -> None:
    """Run the gateway (NORTH) or orchestrator (SOUTH) until SIGTERM/INT.

    Binds one Streamable HTTP listener; mounts ``/acp/<agent>`` for
    every discovered agent; lazy-spawns each child on first
    ``session/new``; cascades shutdown to every live child on stop.

    ``role`` selects north vs south (§13 split):

    * ``"orchestrator"`` (SOUTH) — owns the agents / SessionWriter / Sandbox;
      binds the loopback orchestrator port; runs turns; keeps the in-process
      bus.
    * ``"gateway"`` (NORTH, default) — proxies commands to the orchestrator over
      ACP-HTTP (RemoteAgentRegistry), serves reads from the shared filesystem,
      and wakes its streams via the filesystem watch. Restartable without
      killing in-flight turns (they live in the orchestrator).
    """
    if role not in ("gateway", "orchestrator"):
        raise ValueError(f"invalid role {role!r}: must be 'gateway' or 'orchestrator'")
    from sharedagents_core.launch import component_stdio

    component_stdio(role)
    is_north = role == "gateway"
    is_south = role == "orchestrator"
    from hypercorn.asyncio import serve as _h_serve
    from hypercorn.config import Config as _HCConfig
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, PlainTextResponse
    from starlette.routing import Mount, Route

    from sharedagents_acp.server.http_server import make_acp_app
    from sharedagents_gateway.acp_stdio_main import (
        _build_agent_info,
        _build_capabilities,
    )

    # Refuse to start when legacy per-agent ``sessions/`` or ``logs/``
    # dirs still exist — the unified layout requires the operator to
    # run ``sharedagents migrate-layout`` once to archive + delete them.
    # Surfacing the requirement here (loudly) is safer than letting
    # two writers silently disagree on where data lives.
    from sharedagents_gateway.migrate_layout import has_legacy as _has_legacy

    if _has_legacy():
        sys.stderr.write(
            "sharedagents: legacy per-agent sessions/logs detected. Run "
            "`sharedagents migrate-layout` once to archive + delete them, "
            "then restart the gateway.\n"
        )
        raise SystemExit(2)

    # Wire the on-disk sink for ``sharedagents.*`` loggers before any
    # ``_LOG.*`` call below — otherwise the boot banner / heartbeat-
    # start lines silently drop (see ``_attach_gateway_logfile_handler``
    # for the full rationale).
    _attach_gateway_logfile_handler(role)

    # Load shared secrets.env into os.environ so MCP subprocesses
    # (spawned later by mcp_client with env={**os.environ, ...}) and
    # external-CLI engines (claude/codex/antigravity) that read os.environ
    # directly both inherit the operator's keys. Per-agent secrets
    # stay in cfg.api_keys only — pushing those into the process env
    # would mean whichever agent loaded last wins for any other
    # subprocess, which defeats per-agent attribution. The shared
    # file is the right granularity to lift into os.environ.
    load_shared_secrets_into_environ(sharedagents_root())

    # Install a SIGHUP handler so the operator can hot-reload edits to
    # ~/.sharedagents/secrets.env AND ~/.sharedagents/{config.json,agents/*/config.json}
    # without restarting the systemd service.
    # `kill -HUP $(pidof sharedagents)` re-reads the shared secrets file, refreshes
    # os.environ for new MCP subprocesses, and re-applies every live agent's
    # hot-reloadable config (models, sampling, system prompt, cost limits, …)
    # in place. The `manager` local is assigned its real value below (late-bound
    # getter — the closure reads it at signal time, not now). Pre-bind it to
    # None first so a SIGHUP arriving during the boot window between here and
    # that assignment resolves cleanly to None: reload_registry_configs(None)
    # no-ops (no `_backends`), so config reload is skipped silently while the
    # shared-secrets reload still runs — no NameError, no noisy warning.
    manager = None
    _install_sighup_secrets_reload(registry_getter=lambda: manager)

    # Reset sessions stuck in a live state from the previous gateway
    # process. When the gateway restarts, every session that was mid-turn
    # (streaming, tool_running, etc.) is now dead — the turn runner
    # process is gone. Without this reset the freshness gate would keep
    # them visible as "active" for up to 5 minutes, and the projector
    # would re-stamp session_state_updated_at on re-tail, effectively
    # resetting the clock and keeping them "active" forever.
    # Only the session owner (orchestrator / south) resets stale sessions —
    # the north gateway doesn't own them. On the orchestrator this also closes
    # any turn left in-flight by a previous orchestrator process (the §13
    # blast-radius: an orchestrator restart aborts its in-flight turns).
    if not is_north:
        try:
            from sharedagents_agent.orchestrator import reset_stale_sessions

            reset_count = reset_stale_sessions()
            if reset_count > 0:
                _LOG.info(
                    "startup: reset %d stale session(s) to idle",
                    reset_count,
                )
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("failed to reset stale sessions on boot: %s", exc)

        # Purge promptless sessions (only session_started, no turn ever) so
        # they stop littering the monitor as phantom, age-mis-sorted rows.
        # Runs AFTER reset_stale_sessions so any in-flight turn is already
        # aborted-closed (last_turn_idx >= 0) and thus preserved. Prompted /
        # errored / crashed-mid-turn sessions are kept (see the function's
        # JSONL turn_started guard).
        try:
            from sharedagents_agent.orchestrator import delete_promptless_sessions

            purged = delete_promptless_sessions()
            if purged > 0:
                _LOG.info("startup: deleted %d promptless session(s)", purged)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("failed to purge promptless sessions on boot: %s", exc)

        # Re-derive subagent parent links for the recent sessions so the agent
        # monitor's delegation tree renders after a restart. A legacy
        # full re-index could blank ``parent_session_id`` (the bare-row INSERT
        # defeated the session_started upsert); the JSONL headers still carry
        # the real parent, so we heal the most-recent window each boot. Bounded
        # + idempotent (only fills empty links) so it never slows startup.
        try:
            from sharedagents_agent.orchestrator import backfill_parent_links

            _bf = backfill_parent_links(limit=_PARENT_BACKFILL_ON_BOOT)
            if _bf.get("filled"):
                _LOG.info(
                    "startup: re-linked %d subagent parent(s) across %d recent session(s)",
                    _bf["filled"],
                    _bf["scanned"],
                )
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("failed to backfill subagent parent links on boot: %s", exc)

        # Loop "new session each run" used to parent each tick to the previous
        # tick, building an uncapped monitor tree (AI-usage loop: 23 levels).
        # Flatten those minted links onto the loop origin. Idempotent.
        try:
            from sharedagents_loop.loop_service import flatten_fresh_session_parent_chains

            _flat = flatten_fresh_session_parent_chains()
            if _flat.get("rewritten"):
                _LOG.info(
                    "startup: flattened %d loop-minted parent link(s) across %d loop(s)",
                    _flat["rewritten"],
                    _flat["loops"],
                )
        except Exception as exc:  # noqa: BLE001
            _LOG.warning(
                "failed to flatten loop fresh-session parents: %s",
                exc,
            )

        # Background-delegation recovery: any job whose detached runner died
        # while we were down never delivered its result. Reconcile fails those
        # (idempotently — a delivered-but-unfinalized result is adopted, not
        # re-emitted) so no consumer waits forever. Orchestrator-only, same
        # ownership as the stale-session reset above. Fail-safe.
        try:
            from sharedagents_agent.agents import async_jobs

            _resume_intents = async_jobs.reconcile()
            # A runner can die AFTER delivering its result but BEFORE resuming
            # the idle parent — reconcile returns those stranded parents so we
            # re-invoke them (once the control endpoint is bound). Idempotently
            # claimed inside reconcile, so this never double-wakes. Deferred +
            # best-effort; the durable subagent_result is already on the log.
            if _resume_intents:
                from sharedagents_agent.agents import async_delegate_runner

                asyncio.create_task(async_delegate_runner.fire_resume_intents(_resume_intents))
                _LOG.info(
                    "startup: scheduled %d stranded-parent resume(s)",
                    len(_resume_intents),
                )
            # Recover crashes and ambiguous/failed continuation POSTs while the
            # gateway remains up; boot-only repair can leave Jarvis stranded for
            # an arbitrarily long time.
            from sharedagents_agent.agents import async_delegate_runner

            async_delegate_runner.start_live_reconciler()
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("failed to reconcile background jobs on boot: %s", exc)

        # Self-improver was removed from sharedagents.

        # Seed the ALWAYS-ON builtin MCP tools (web/memory/delegate)
        # SYNCHRONOUSLY from their static TOOL_DEFS — no subprocess spawn, so
        # it's safe on the hot path. This makes them present in the FIRST worker
        # request (tools[] + strict-allowlist expansion) even before the async
        # discovery warm below lands; without it a prompt in the startup window
        # would miss them. The warm still runs and
        # re-discovers everything (external servers + a refresh of the builtins).
        try:
            from sharedagents_mcp.mcp_client import seed_builtin_mcp_tools

            _seeded = seed_builtin_mcp_tools()
            _LOG.info("startup: seeded %d builtin MCP tool(s) pre-warm", _seeded)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("startup: builtin MCP seed skipped: %s", exc)

        # Warm the MCP discovery cache off the hot path so the namespaced MCP
        # base tools are present in the FIRST real worker request (see
        # _warm_mcp_discovery). Only the turn-running role (south)
        # need it; the north gateway proxies and never builds worker requests.
        try:
            globals()["_MCP_WARM_TASK"] = asyncio.create_task(_warm_mcp_discovery())
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("startup: could not schedule MCP discovery warm: %s", exc)

        try:
            globals()["_EFFORT_WARM_TASK"] = asyncio.create_task(_warm_effort_discovery())
        except Exception as exc:  # noqa: BLE001
            _LOG.warning(
                "startup: could not schedule effort catalogue warm: %s",
                exc,
            )

        try:
            globals()["_CATALOGUE_WARM_TASK"] = asyncio.create_task(_warm_model_catalogue())
        except Exception as exc:  # noqa: BLE001
            _LOG.warning(
                "startup: could not schedule model catalogue warm: %s",
                exc,
            )

    gw = _orchestrator_config() if is_south else _gateway_config()
    bearer = gw["auth_token"]
    # Publish THIS turn-running instance's own control bind so a detached
    # async-delegate runner (which inherits our env) resumes the parent against
    # the gateway the parent actually lives on — NOT the hardcoded prod :8100
    # default, which on a dev/usersim instance (SHAREDAGENTS_GATEWAY_URL unset) would
    # misroute the resume to prod for a parent_sid that doesn't exist there.
    # North never runs turns / spawns runners, so it advertises no self-URL.
    if not is_north:
        _self_host = gw["host"]
        if _self_host in ("0.0.0.0", "::", ""):
            _self_host = "127.0.0.1"
        os.environ["SHAREDAGENTS_SELF_GATEWAY_URL"] = f"http://{_self_host}:{gw['port']}"
    # Where the north gateway reaches the orchestrator (commands down).
    _orch = _orchestrator_config()
    orchestrator_url = f"http://{_orch['host']}:{_orch['port']}" if is_north else None
    orchestrator_token = (_orch["auth_token"] or bearer) if is_north else None

    # Retire an upgraded install's managed `main`, then scaffold the default
    # agent set (external CLI engines → jarvis → OpenCode profile agents).
    # Extracted so the boot invariant is unit-testable — a test can run exactly
    # this sequence on a fresh SHAREDAGENTS_HOME and assert what is (and is NOT, e.g.
    # the retired `main`) scaffolded.
    _boot_scaffold_agents()

    # MOUNT set — a superset of the picker list. Every configured agent
    # (incl. the subagent utilities) gets a live /acp/<name> route so no
    # legacy/in-flight session 404s. (`main` retired 2026-07-21 — not mounted.)
    # The user-facing
    # PICKER list (_discover_agents) is a curated subset rendered by
    # /openai/v1/models only.
    agent_names = _discover_mountable_agents()
    picker_names = _discover_agents()
    if not agent_names:
        _LOG.warning(
            "no agents discovered under %s; gateway will run but every "
            "/acp/<name> route will 404 until an agent is configured",
            agents_root(),
        )

    # Agent registry. SOUTH hosts in-process backends (turns run here).
    # NORTH proxies to the orchestrator over ACP-HTTP via RemoteAgentRegistry,
    # so the gateway holds NO agent/engine state and can restart freely.
    if is_north:
        from sharedagents_gateway.remote_backend import RemoteAgentRegistry

        manager = RemoteAgentRegistry(
            agent_names,
            orchestrator_url,
            token=orchestrator_token,
        )
    else:
        manager = AgentRegistry(agent_names)

    # Mount one ACP app per discovered agent under /acp/<name>. EVERY
    # agent is exposed through :class:`AgentACPBackend` regardless of
    # engine kind. AgentACPBackend internally:
    #
    #   * For local_llm agents — holds an in-process CoreBackend
    #     wrapper around the same Agent + LocalLLMEngine (single source
    #     of truth for session-lifecycle / cancel-routing / outbound
    #     channel binding). The multi-round dispatch / compaction /
    #     loop detection / distill recovery / premature-turn-end
    #     nudges all live in :class:`LocalLLMEngine.run_turn`.
    #   * For claude_cli / codex_cli / antigravity_cli agents — routes
    #     through Agent.run_turn → Engine.run_turn →
    #     :func:`run_delegate_*_async` (the CLI subprocess driver
    #     shared between :meth:`ExternalCliEngine.delegate_runner`
    #     and the delegate-tool entry points in
    #     :mod:`sharedagents_agent.agents.delegate_shims`). The new engine
    #     layer adds agent-context priming (--append-system-prompt) and
    #     sha-change session-resume override.
    routes: list[Any] = []
    for name in agent_names:
        try:
            cfg = AgentConfig.load(name)
        except FileNotFoundError:
            continue
        # Boot audit: one INFO line per agent listing every apiKeyEnv
        # the operator's models[] referenced + where the key was
        # resolved from. Catches the kimi-style bug (OPENROUTER_API_KEY
        # set only inside a per-MCP env block so the gateway process
        # itself never saw it) at restart instead of on the first
        # vision-transcription 401. Missing keys render as ``✗(missing)``
        # in red-ish prefix; present keys as ``✓(agent-local)`` /
        # ``✓(env)`` so the operator can see whether per-agent
        # attribution is actually wired up.
        try:
            audit = cfg.audit_api_keys()
        except Exception:  # noqa: BLE001
            audit = {}
        if audit:
            bits: list[str] = []
            missing = False
            for env_name, src in audit.items():
                if src == "missing":
                    bits.append(f"{env_name}=✗(missing)")
                    missing = True
                else:
                    bits.append(f"{env_name}=✓({src})")
            line = f"[boot] agent={name} api_keys: " + " ".join(bits)
            (_LOG.warning if missing else _LOG.info)(line)
        # AgentACPBackend construction; lazy via the registry so the
        # boot sequence doesn't pay the full agent-load cost for
        # agents never used by a frontend.
        backend = manager.get(name)
        sub_app = make_acp_app(
            backend=backend,
            agent_capabilities=_build_capabilities(cfg),
            agent_info=_build_agent_info(cfg),
            auth_methods=[],
            bearer_token=bearer,
        )
        routes.append(Mount(f"/acp/{name}", app=sub_app))

        # Start the per-agent heartbeat scheduler when the config asks for it.
        # South must start it explicitly so background memory-synth / idle
        # cleanup keeps running.
        try:
            agent_obj = getattr(backend, "agent", None)
            if agent_obj is not None and getattr(cfg, "heartbeat_enabled", False):
                hb_task = agent_obj.start_heartbeat()
                if hb_task is not None:
                    _LOG.info(
                        "heartbeat scheduler started for agent %r",
                        name,
                    )
        except Exception as exc:  # noqa: BLE001
            _LOG.warning(
                "heartbeat scheduler failed to start for %r: %s",
                name,
                exc,
            )

    _service_name = "sharedagents-orchestrator" if is_south else "sharedagents-gateway"

    async def _index(_request: Any) -> Any:
        return JSONResponse(
            {
                "service": _service_name,
                "role": role,
                "agents": agent_names,
                "live_backends": sorted(manager._backends.keys()),
            }
        )

    async def _health(_request: Any) -> Any:
        return PlainTextResponse("ok")

    routes.append(Route("/", _index, methods=["GET"]))
    routes.append(Route("/health", _health, methods=["GET"]))

    # OpenAI-compatible HTTP API at /openai/v1 — point Open WebUI (or any
    # OpenAI-spec client) here. Same gateway, same auth, same lifecycle;
    # routes the chat/completions field 'model' → AgentRegistry.get(name)
    # so each agent shows up as one OpenAI 'model'.
    # OpenAI-compatible API is a NORTH interface surface — the orchestrator
    # (south) does not serve it; the gateway does.
    if not is_south:
        from sharedagents_gateway.openai_endpoint import make_openai_app

        routes.append(
            Mount(
                "/openai/v1",
                app=make_openai_app(
                    agent_names=agent_names,
                    manager=manager,
                    bearer_token=bearer,
                    # /v1/models lists ONLY the curated picker set; chat/completions
                    # can still resolve any MOUNTED agent (incl. `main`).
                    listed_agent_names=picker_names,
                ),
            )
        )

    # REST control surface at /control/v1 — the synchronous HTTP transport
    # over the shared server-side command runtime. The web frontend proxies
    # this; the ACP ``session/command`` method is the other shell over the
    # same ``backend.run_command``. Single command implementation, two
    # transports.
    from sharedagents_gateway.control_endpoint import make_control_app

    routes.append(
        Mount(
            "/control/v1",
            app=make_control_app(
                agent_names=agent_names,
                manager=manager,
                bearer_token=bearer,
                # NORTH proxies submit to the orchestrator so the turn runs there
                # (crash-isolation); SOUTH drives it locally.
                orchestrator_url=orchestrator_url,
                orchestrator_token=orchestrator_token,
            ),
        )
    )

    app = Starlette(routes=routes)

    stop_event = asyncio.Event()
    # Gate Hypercorn's listener drain behind heartbeat teardown. The
    # SIGTERM handler sets ``stop_event`` (which unblocks this coroutine's
    # cleanup); the drain itself waits on ``drain_gate`` and is only
    # released AFTER every heartbeat scheduler is stopped, so no scheduler
    # outlives the point the socket stops accepting connections.
    # Using ``stop_event.wait`` as
    # the trigger raced: the one SIGTERM both unblocked cleanup and started
    # the drain concurrently.
    drain_gate = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _on_signal() -> None:
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _on_signal)
        except NotImplementedError:
            pass

    _hc_cfg = _HCConfig()
    _hc_cfg.bind = [f"{gw['host']}:{gw['port']}"]
    _hc_cfg.alpn_protocols = ["h2", "http/1.1"]
    _hc_cfg.accesslog = None
    _hc_cfg.errorlog = "-"

    # Pre-flight bind guard. hypercorn binds INSIDE the serve task
    # below; if the configured port is still held (a restart racing the
    # predecessor's socket drain, or an accidental double-start) the bind
    # raises OSError(98) in that task, the exception is never retrieved,
    # and THIS process lives on awaiting ``stop_event`` as a non-serving
    # zombie that has already clobbered the pidfile to point at itself —
    # the "zombie gateway" pathology (14 such crashes on 2026-05-21).
    # Wait briefly for a draining predecessor to release the socket; if
    # it stays held, refuse to start and leave the existing pidfile
    # (owned by the live holder) intact.
    _holder_pid = await _await_port_free(gw["host"], int(gw["port"]), 10.0)
    if _holder_pid is not None:
        _LOG.error(
            "%s: %s:%d still held by pid=%s after 10s; refusing to start (leaving existing pidfile intact)",
            _service_name,
            gw["host"],
            gw["port"],
            _holder_pid,
        )
        print(
            f"[{_service_name}] {gw['host']}:{gw['port']} already in use (pid={_holder_pid}); not starting",
            flush=True,
        )
        return

    # AgentRegistry has no GC loop — backends live for the process
    # lifetime (no per-agent subprocess to reap, no idle-eviction
    # window to enforce).
    pidfile = _orchestrator_pidfile() if is_south else _gateway_pidfile()
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    pidfile.write_text(str(os.getpid()))

    startup_title_task: asyncio.Task[Any] | None = None
    hc_task = asyncio.create_task(
        _h_serve(
            app,  # type: ignore[arg-type]
            _hc_cfg,
            shutdown_trigger=drain_gate.wait,
        ),
        name=f"{_service_name}-http",
    )
    _LOG.info(
        "%s listening on http://%s:%d/  (agents: %s)",
        _service_name,
        gw["host"],
        gw["port"],
        ", ".join(agent_names) or "(none)",
    )
    print(
        f"[{_service_name}] listening on http://{gw['host']}:{gw['port']}/  "
        f"agents: {', '.join(agent_names) or '(none)'}",
        flush=True,
    )

    # Repair recent sessions left on the visible startup placeholder without
    # delaying the listener or the first request.  The task owns its bounded
    # scan and uses the same detached title scheduler as a live turn.
    if not is_north:
        startup_title_task = asyncio.create_task(
            _reconcile_startup_session_titles(manager),
            name="sharedagents-startup-title-reconcile",
        )

    # A durable background-result receipt may have been accepted immediately
    # before the previous process stopped, leaving the queue item pending but
    # its lightweight parent kick unstarted. The session owner repairs that
    # handoff from the queue itself; north holds no turn-running state.
    if not is_north:
        from sharedagents_gateway.control_endpoint import (
            recover_pending_background_prompts,
            start_background_queue_retry_loop,
        )

        try:
            _recovered = await recover_pending_background_prompts(
                agent_names=agent_names,
                manager=manager,
            )
            if _recovered:
                _LOG.info(
                    "startup: scheduled %d pending background queue parent turn(s)",
                    len(_recovered),
                )
        except Exception as exc:  # noqa: BLE001 — periodic loop will retry
            _LOG.warning(
                "initial pending background queue recovery failed: %s",
                exc,
            )
        try:
            start_background_queue_retry_loop(
                agent_names=agent_names,
                manager=manager,
            )
        except Exception as exc:  # noqa: BLE001 — gateway must still start
            _LOG.warning(
                "failed to start background queue retry loop: %s",
                exc,
            )
        # Stale-waiting resolver: a 'waiting' parent whose children turned
        # ghost (abort/crash never folded them — the 2026-08-11 wedge) is
        # released to idle within ~6 min of the last real child activity.
        # The boot sweep (reset_stale_sessions, above) only heals on restart;
        # this loop heals at runtime. Session owner only — north holds no
        # turn-running state.
        try:
            from sharedagents_agent.orchestrator import start_stale_waiting_sweep

            start_stale_waiting_sweep()
        except Exception as exc:  # noqa: BLE001 — gateway must still start
            _LOG.warning(
                "failed to start stale-waiting sweep: %s",
                exc,
            )

    # Ambient config-file watcher (mtime poll). Only where backends live
    # in-process — a NORTH gateway proxies to the orchestrator and holds no
    # cfg to reload. ``config_watch_interval_s`` in the global config tunes
    # the cadence; <= 0 disables (SIGHUP still works as the manual path).
    config_watch_task: asyncio.Task[Any] | None = None
    if not is_north:
        try:
            _watch_interval = float((load_global_config() or {}).get("config_watch_interval_s", 5.0))
        except (TypeError, ValueError):
            _watch_interval = 5.0
        if _watch_interval > 0:
            config_watch_task = asyncio.create_task(
                _config_watch_loop(manager, _watch_interval, stop_event),
                name="sharedagents-config-watch",
            )

    serve_exc: BaseException | None = None
    try:
        # Wait on the serve task too, not just stop_event — otherwise a
        # serve-task bind failure (the zombie pathology) would hang here
        # forever. _await_serve_or_stop returns the serve task's
        # exception when it died on its own.
        serve_exc = await _await_serve_or_stop(hc_task, stop_event)
        if serve_exc is not None:
            _LOG.error(
                "%s HTTP serve task exited early: %r",
                _service_name,
                serve_exc,
            )
            print(
                f"[{_service_name}] serve task failed: {serve_exc!r}",
                flush=True,
            )
    finally:
        # Stop owner-side retries AND any recovery turns they started before
        # tearing down backends. Ordinary prompt-submit tasks are not touched.
        if not is_north:
            from sharedagents_gateway.control_endpoint import (
                shutdown_background_queue_recovery,
            )

            with contextlib.suppress(Exception):
                await shutdown_background_queue_recovery()
        # Stop the ambient config watcher first — cheap, and it must not
        # fire a reload against backends the shutdown below is tearing down.
        if config_watch_task is not None:
            config_watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await config_watch_task
        if startup_title_task is not None:
            startup_title_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await startup_title_task
        # Stop every running heartbeat scheduler before the listener
        # drains so background work doesn't outlive the gateway.
        for backend in list(manager._backends.values()):
            agent_obj = getattr(backend, "agent", None)
            if agent_obj is None:
                continue
            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    agent_obj.stop_heartbeat(),
                    timeout=5.0,
                )
        # Every heartbeat scheduler is now stopped; only now release
        # Hypercorn's listener drain so a scheduler can never outlive the
        # socket-close.
        drain_gate.set()
        # Cancel-cascade: shutdown manager first so all children get
        # SIGTERM before the HTTP listener fully drains. This honors
        # the "abort reaches all child subagents" contract.
        with contextlib.suppress(Exception):
            await asyncio.wait_for(manager.shutdown(), timeout=10.0)
        if not hc_task.done():
            try:
                await asyncio.wait_for(hc_task, timeout=5.0)
            except asyncio.TimeoutError:
                hc_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await hc_task
        # Only remove the pidfile if it still points at THIS process. In a
        # double-start race the winner may have overwritten it to its own
        # pid; deleting it then would orphan the live holder's pidfile.
        with contextlib.suppress(FileNotFoundError, OSError, ValueError):
            if pidfile.read_text().strip() == str(os.getpid()):
                pidfile.unlink()
        print(f"[{_service_name}] stopped", flush=True)
    # Propagate a serve-task failure so the process exits non-zero (the
    # supervisor can restart) instead of returning 0 like a clean stop.
    if serve_exc is not None:
        raise serve_exc


def serve_gateway_main(role: str = "gateway") -> int:
    """Sync wrapper for ``sharedagents gateway serve`` (NORTH) or orchestrator serve (SOUTH)."""
    if role not in ("gateway", "orchestrator"):
        print(
            f"[sharedagents] invalid role {role!r}: must be 'gateway' or 'orchestrator'",
            flush=True,
        )
        return 2
    label = "sharedagents-orchestrator" if role == "orchestrator" else "sharedagents-gateway"
    try:
        asyncio.run(serve_gateway(role=role))
    except KeyboardInterrupt:
        return 0
    except Exception as e:  # noqa: BLE001
        # serve_gateway propagated a serve-task failure (e.g. bind
        # OSError(98) that slipped past the port pre-flight). Exit
        # non-zero so the supervisor restarts instead of treating a dead
        # gateway as a clean shutdown. Route the unhandled-exception path
        # through structured capture so this top-level process crash
        # boundary exercises the SessionError-hook machinery uniformly,
        # not only in the TUI; best-effort so a logging
        # failure can never mask the original crash.
        try:
            from sharedagents_core import errors as _errors

            _errors.log_error(
                label,
                e,
                summary=f"{label} serve loop crashed",
                context={"role": role},
            )
        except Exception:  # noqa: BLE001
            pass
        print(f"[{label}] exiting after serve failure: {e!r}", flush=True)
        return 1
    return 0


def serve_orchestrator_main() -> int:
    """Sync wrapper for the ``sharedagents orchestrator serve`` subcommand (south)."""
    return serve_gateway_main(role="orchestrator")


__all__ = [
    "AgentRegistry",
    "serve_gateway",
    "serve_gateway_main",
    "serve_orchestrator_main",
]
