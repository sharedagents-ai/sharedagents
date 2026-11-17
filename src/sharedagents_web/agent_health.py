"""Integrity of the external agents: installed / version / authenticated.

The Usage > Subscriptions page and the chat agent selector both list the
external engines (antigravity, claude, codex, cursor, devin, hermes, kimi,
mimo, opencode, pi, qwen, zcode). Until now an engine that EXISTS on the
server but cannot spend — because its CLI was never signed in — simply
vanished from the catalog (``discovered_provider_rows`` drops everything
without a reusable credential). An operator could not tell "agent removed"
from "agent needs authentication".

This module answers, for every registered external agent:

* ``installed``   — resolved through ``_which_cli``, the SAME resolver the
  delegate runners and the launch gate share, so the health row and the
  spawn can never disagree about a CLI living in a fallback bin dir.
* ``version``     — ``<cli> --version``, cached on disk; never re-run per
  page load.
* ``authenticated`` — three-valued. Reuses the authoritative credential
  probes of ``subscription_limits.subscription_available`` for the eight
  subscription engines; config/key probes for the rest; and the ledger
  itself as proof — an agent that SPENT within the lookback window is
  authenticated, whatever the credential heuristics say.

The alert the operator asked for ("instructed to authenticate") binds on
``needs_auth`` only: no credential, no recent spend, and no config-based
auth story. Engines whose ``acp_auth_method`` is wired (cursor, kimi)
already re-authenticate themselves at dispatch (``AcpEngine`` calls
``authenticate(methodId)`` when the server answers AUTH_REQUIRED) — those
carry ``auto_auth=True`` so the UI can promise a retry instead of only a
terminal command.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)

#: ``<cli> --version`` budget. A hung CLI must never block the picker.
_VERSION_TIMEOUT_S = 10.0
#: Cached version probes (installs are rare; the Update action forces one).
VERSION_TTL_S = 6 * 3600.0
#: Credential probes are cheap file reads; re-check after this many seconds.
AUTH_TTL_S = 60.0
#: An agent that spent within this window proved it can authenticate.
SPEND_PROOF_LOOKBACK_S = 30 * 24 * 3600.0

#: ACP auth-method ids sharedagents's ``AcpEngine`` already replays itself
#  (``_new_session_with_auth`` / ``session_load`` re-auth path). For these
#  the operator-facing action is "re-check", not "go sign in by hand".
AUTO_AUTH_METHODS = {"cursor_login", "login"}

#: CLI kinds whose auth is a settings file the operator edits — absence of
#  an OAuth blob is NOT an alarm (there is no login step to run).
_CONFIG_AUTH_KINDS = {"qwen_acp"}

_UPDATE_OVERRIDES = {
    # The AGY CLI ships its own updater subcommand.
    "antigravity_cli": "antigravity update",
    # Hermes owns an ``update`` subcommand (``uv tool upgrade`` is wrong for
    # a git-checkout install, which is how it ships here).
    "hermes_acp": "hermes update",
}

# Kinds whose CLI cannot print its own version (hermes --version hangs) —
# probe the installer/checkout instead, extracting only the agent's line.
_VERSION_OVERRIDE = {
    "hermes_acp": (
        ("uv tool list", r"hermes-agent\s+v?(\d+\.\d+\.\d+[^ \n]*)"),
        ("git -C {home}/.hermes/hermes-agent describe --tags", r"v?(\d+\.\d+\.\d+[^ \n]*)"),
    ),
}

_VERSION_PATTERNS = (re.compile(r"(\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.\-]+)?)"),)


def _cache_path():
    from sharedagents_core.layout import usage_root

    return Path(usage_root()) / "agent_health.json"


def _utc_now() -> float:
    return time.time()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------
# Agent → engine spec, with the subscription-provider cross-links.
# --------------------------------------------------------------------------


def _agent_rows() -> list[dict[str, Any]]:
    """One row per external agent (the picker's External lane, + MiMo).

    Built from ``EXTERNAL_REGISTRY`` — the runtime's own dispatch table —
    so a new built-in engine appears in the health view without another
    list to keep in sync. MiMo is not a CLI engine (it dispatches on the
    Token Plan key), but it owns a subscription card, so it joins the scan.
    """
    from sharedagents_agent.external_registry import EXTERNAL_REGISTRY
    from sharedagents_web.subscription_limits import PROVIDER_AGENTS

    provider_by_agent = {agent: pid for pid, agent in PROVIDER_AGENTS.items()}
    rows: list[dict[str, Any]] = []
    for spec in EXTERNAL_REGISTRY.values():
        rows.append(
            {
                "agent": spec.short_name,
                "kind": spec.kind.value,
                "display_name": spec.display_name,
                "cli": spec.cli_command,
                "adapter": spec.adapter,
                "provider": provider_by_agent.get(spec.short_name, ""),
                "auth_method": spec.default_auth or "",
                "install_cmd": spec.install_cmd,
                "install_hint": spec.install_hint,
                "install_url": spec.install_url or "",
            }
        )
    rows.append(
        {
            "agent": "mimo",
            "kind": "local_llm",
            "display_name": "MiMo (Xiaomi)",
            "cli": "",
            "adapter": "api",
            "provider": "mimo",
            "auth_method": "",
            "install_cmd": "",
            "install_hint": "",
            "install_url": "",
        }
    )
    rows.sort(key=lambda row: row["agent"])
    return rows


def agent_names() -> list[str]:
    return [row["agent"] for row in _agent_rows()]


def _spec_for_agent(agent: str) -> dict[str, Any] | None:
    agent = str(agent or "").strip().lower()
    for row in _agent_rows():
        if row["agent"] == agent:
            return row
    return None


def _update_cmd(row: dict[str, Any]) -> str:
    override = _UPDATE_OVERRIDES.get(row["kind"], "")
    if override:
        return override
    install = str(row.get("install_cmd") or "").strip()
    if install.startswith("npm install"):
        # Re-running the installer with @latest refreshes the global.
        pkg = ""
        for token in install.split():
            if token.startswith("@") and token.count("/") == 1:
                pkg = token
            elif (
                "/" not in token
                and not token.startswith(("npm", "install", "-g", "--ignore-scripts"))
                and not token.startswith("@")
            ):
                pkg = pkg or token
        if pkg:
            return f"npm install -g {pkg}@latest"
    return install


# --------------------------------------------------------------------------
# Installed + version probes.
# --------------------------------------------------------------------------


def _resolve_cli(cli: str) -> str:
    if not cli:
        return ""
    try:
        from sharedagents_agent.tools.impl._utils import _which_cli

        return _which_cli(cli) or ""
    except Exception:  # noqa: BLE001 — a health probe must never raise
        return ""


_VERSION_CLEAN = re.compile(r"^\s*(?:v(?:ersion)?:?\s*)?", re.IGNORECASE)


def _parse_version(raw: str) -> str:
    text = _VERSION_CLEAN.sub("", str(raw or "").strip())
    for pattern in _VERSION_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1)
    first = text.splitlines()[0].strip() if text else ""
    # A CLI that prints prose ("`version` is not a command") has no version
    # to report; never ship that to a chip.
    if not re.search(r"\d", first):
        return ""
    return first[:40]


def _probe_version(path: str, kind: str = "") -> str:
    """Run ``<cli> --version`` once, best effort. ``""`` when unknowable.

    Some CLIs (hermes) cannot print their own version; those kinds get one
    fallback probe of the installer tool with a line-scoped regex.
    """
    for args in (["--version"], ["version"]):
        if not path:
            break
        try:
            proc = subprocess.run(  # noqa: S603 — argv, no shell
                [path, *args],
                capture_output=True,
                text=True,
                timeout=_VERSION_TIMEOUT_S,
                stdin=subprocess.DEVNULL,
                env={**os.environ, "NO_COLOR": "1", "CI": "1"},
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        out = (proc.stdout or "") + (proc.stderr or "")
        version = _parse_version(out)
        if version:
            return version
    override = _VERSION_OVERRIDE.get(kind)
    for command, pattern in override or ():
        command = command.format(home=str(Path.home()))
        head = command.split(None, 1)
        if head:
            resolved_head = _resolve_cli(head[0])
            if resolved_head:
                command = resolved_head + command[len(head[0]) :]
        try:
            proc = subprocess.run(  # noqa: S602 — fixed installer command
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=_VERSION_TIMEOUT_S,
                stdin=subprocess.DEVNULL,
            )
            match = re.search(pattern, (proc.stdout or "") + (proc.stderr or ""))
            if match:
                return match.group(1)
        except (OSError, subprocess.TimeoutExpired):
            pass
    return ""


class _HealthCache:
    """Process-wide cache of last-known rows with a disk mirror.

    Disk persistence matters: the web server answers ``/api/models`` with a
    cache-only projection for the picker, and a server restart must not
    empty the versions the chat page shows.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: dict[str, dict[str, Any]] = {}
        self._disk_mtime: float | None = None
        self._refreshing = False

    # -- disk mirror -----------------------------------------------------
    def _load_disk(self) -> None:
        """(Re)load when the disk mirror changed since the last read.

        gunicorn runs several worker processes: the picker worker must see
        the scan results the health-route worker wrote, so a one-shot load
        per process is not enough. A stat() per access is the cheap way.
        A missing file never clears the in-process rows — they are the
        source of truth until someone actually writes a scan.
        """
        try:
            path = _cache_path()
            mtime = path.stat().st_mtime
        except OSError:
            return
        if self._disk_mtime is not None and mtime == self._disk_mtime:
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            rows = data.get("rows") if isinstance(data, dict) else None
            if isinstance(rows, dict):
                self._rows = {str(k): dict(v) for k, v in rows.items() if isinstance(v, dict)}
                self._disk_mtime = mtime
        except (OSError, ValueError):
            pass

    def _write_disk(self) -> None:
        try:
            path = _cache_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            # Unique tmp name per writing thread: sibling gunicorn workers
            # may save concurrently, and os.replace must stay atomic.
            tmp = path.with_suffix(f".json.tmp.{os.getpid()}.{threading.get_ident()}")
            tmp.write_text(
                json.dumps({"updated_at": _iso(_utc_now()), "rows": self._rows}),
                encoding="utf-8",
            )
            os.replace(tmp, path)
        except OSError:
            _LOG.debug("agent_health: disk cache write failed", exc_info=True)

    # -- access ----------------------------------------------------------
    def cached(self, agent: str) -> dict[str, Any] | None:
        with self._lock:
            self._load_disk()
            row = self._rows.get(agent)
            return dict(row) if row else None

    def merge(self, agent: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            self._load_disk()
            row = self._rows.get(agent) or {}
            row.update({k: v for k, v in fields.items() if v is not None or k in row})
            if "authenticated" in fields:
                row["checked_at"] = _utc_now()
            self._rows[agent] = row
            self._write_disk()
            return dict(row)

    def force_refresh(self, agents: list[str] | None = None) -> None:
        """Drop the TTL stamps so the next ``snapshot`` re-probes."""
        with self._lock:
            self._load_disk()
            targets = agents or list(self._rows)
            for agent in targets:
                row = self._rows.get(agent)
                if row is not None:
                    row.pop("version", None)
                    row.pop("version_checked_at", None)
                    row.pop("checked_at", None)
            self._write_disk()


_CACHE = _HealthCache()


# --------------------------------------------------------------------------
# Auth probes.
# --------------------------------------------------------------------------


def _spend_proof(agent: str) -> float:
    """Epoch of the agent's newest ``turn_costs`` row, 0.0 when never."""
    try:
        import sqlite3

        from sharedagents_core.layout import usage_ledger_db_path

        db = Path(usage_ledger_db_path())
        if not db.exists():
            return 0.0
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5.0)
        try:
            row = conn.execute(
                "SELECT MAX(created_at) FROM turn_costs WHERE lower(agent_name) = ?",
                (str(agent or "").strip().lower(),),
            ).fetchone()
        finally:
            conn.close()
        raw = row[0] if row else None
        if not raw:
            return 0.0
        text = str(raw).strip()
        try:
            return float(text)
        except ValueError:
            pass
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
            try:
                return datetime.strptime(text, fmt).timestamp()
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0
    except Exception:  # noqa: BLE001 — ledger absence is data, not an error
        return 0.0


def _kimi_has_credential() -> bool:
    """Kimi Code stores its login under ``~/.kimi`` (OAuth session) or runs
    on a plain API key; anything else is 'we cannot tell' (unknown)."""
    env = (os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY") or "").strip()
    if env:
        return True
    root = Path.home() / ".kimi"
    try:
        if not root.is_dir():
            return None  # type: ignore[return-value]
        for candidate in ("auth.json", "credentials.json", "config.toml"):
            if (root / candidate).is_file():
                return True
        return None  # type: ignore[return-value]
    except OSError:
        return None  # type: ignore[return-value]


def _hermes_has_credential() -> bool | None:
    """Hermes stores a provider credential pool in ``~/.hermes/auth.json``."""
    path = Path.home() / ".hermes" / "auth.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    pool = data.get("credential_pool")
    providers = data.get("providers")
    if isinstance(pool, dict) and any(pool.values()):
        return True
    if isinstance(pool, list) and pool:
        return True
    if isinstance(providers, dict) and providers:
        return True
    return None


def _pi_has_credential() -> bool | None:
    """Pi keeps provider logins in ``~/.pi/agent/auth.json`` (keyed by
    provider) and can also run on plain ``OPENAI_API_KEY``-style env."""
    env = (os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if env:
        return True
    path = Path.home() / ".pi" / "agent" / "auth.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(data, dict) and data:
        return True
    return None


def _qwen_authenticated() -> bool | None:
    """Qwen Code authenticates through ``~/.qwen/settings.json`` routes —
    there is no CLI login step, so an auth probe is a config probe."""
    path = Path.home() / ".qwen" / "settings.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    providers = data.get("modelProviders")
    if isinstance(providers, dict) and any(isinstance(rows, list) and rows for rows in providers.values()):
        return True
    return None


def _raw_credential(agent: str, provider: str, kind: str) -> bool | None:
    """True / False / None(unknown) from the credential stores alone.

    The eight subscription engines defer to ``subscription_available`` — the
    same authoritative probes the Usage meters use. The others get the
    config probes above.
    """
    if provider:
        try:
            from sharedagents_web.subscription_limits import PROVIDER_IDS, subscription_available

            if provider in PROVIDER_IDS:
                return bool(subscription_available(provider))
        except Exception:  # noqa: BLE001
            return None
    if kind == "kimi_acp":
        return _kimi_has_credential()
    if kind == "hermes_acp":
        return _hermes_has_credential()
    if kind == "pi_rpc":
        return _pi_has_credential()
    if kind in _CONFIG_AUTH_KINDS:
        return _qwen_authenticated()
    return None


#: Human-run sign-in command per agent ("" = nothing to run / unknown).
_LOGIN_CMDS = {
    "claude": "claude",
    "codex": "codex login",
    "cursor": "cursor-agent login",
    "antigravity": "antigravity",
    "zcode": "zcode login",
    "opencode": "opencode auth login",
    "devin": "devin auth login",
    "kimi": "kimi  (then /login, or set KIMI_API_KEY)",
    "hermes": "hermes auth  (or set the provider API key)",
    "pi": "pi  (then /login, or set OPENAI_API_KEY)",
}


def _login_hint(row: dict[str, Any]) -> str:
    hint = _LOGIN_CMDS.get(row["agent"], "")
    if not hint and row.get("install_url"):
        hint = f"see {row['install_url']}"
    return hint


def _authenticate_headless(row: dict[str, Any]) -> dict[str, Any]:
    """Try the ACP ``authenticate`` handshake for engines with a configured
    auth method — the same call ``AcpEngine`` replays at dispatch. Returns
    ``{"attempted", "ok", "detail"}``.

    Browser-driven methods (cursor) still need the operator on a terminal;
    the call is time-boxed so a hung OAuth flow never wedges a health check.
    """
    method = str(row.get("auth_method") or "").strip()
    if not method:
        return {"attempted": False, "ok": False, "detail": "no automatic auth method"}
    try:
        from sharedagents_agent.external_registry import spec_for

        spec = spec_for(row["kind"])
        command = spec.default_acp_command
    except Exception:  # noqa: BLE001
        command = None
    if not command:
        return {"attempted": False, "ok": False, "detail": "no ACP command"}
    from sharedagents_agent.tools.impl._utils import _which_cli

    argv = [str(part) for part in command]
    resolved = _which_cli(argv[0])
    if not resolved:
        return {"attempted": False, "ok": False, "detail": "CLI not installed"}
    argv[0] = resolved
    try:
        import asyncio

        async def _run() -> str:
            from sharedagents_acp.client import spawn_stdio_agent
            from sharedagents_agent.delegate_env import build_third_party_spawn_env

            # Same third-party spawn contract as ``AcpEngine._default_spawn``
            # (allowlisted env, never the full parent environ).
            client, _proc = await spawn_stdio_agent(
                argv,
                cwd=str(Path.home()),
                env=build_third_party_spawn_env(None),
                inherit_parent_env=False,
                client_info={"name": "sharedagents-health", "version": "1.0.0"},
                return_process=True,
            )
            try:
                await client.initialize()
                advertised = [str(getattr(m, "id", "") or "") for m in (client.auth_methods or [])]
                if method not in advertised:
                    raise RuntimeError(f"agent does not advertise auth method {method!r} (has {advertised})")
                await client.authenticate(method)
                return f"authenticated via method {method!r}"
            finally:
                try:
                    await client.close()
                except Exception:  # noqa: BLE001
                    pass

        try:
            detail = asyncio.run(asyncio.wait_for(_run(), timeout=45))
        except Exception as exc:  # noqa: BLE001 — the flow is best effort
            return {"attempted": True, "ok": False, "detail": str(exc) or exc.__class__.__name__}
        return {"attempted": True, "ok": True, "detail": detail}
    except Exception as exc:  # noqa: BLE001
        return {"attempted": True, "ok": False, "detail": str(exc) or exc.__class__.__name__}


# --------------------------------------------------------------------------
# Snapshot assembly.
# --------------------------------------------------------------------------


def _auth_state_for(installed: bool, authenticated: bool | None, kind: str) -> str:
    if not installed:
        return "not_installed"
    if authenticated is True:
        return "ok"
    if authenticated is False:
        return "missing"
    return "unknown"


def _probe_row(row: dict[str, Any], *, force: bool = False, live: bool = True) -> dict[str, Any]:
    """One agent's health row; never raises.

    ``live=False`` keeps vendor CLIs off the path — versions come from
    the cache only — while still reading credential files and stores.
    That is the mode the catalog/picker hot paths use.
    """
    agent = row["agent"]
    cached = _CACHE.cached(agent) or {}
    now = _utc_now()

    path = _resolve_cli(row["cli"]) if row["cli"] else ""
    installed = bool(path) if row["cli"] else True  # non-CLI rows: config-gated
    _CACHE.merge(agent, installed=installed, path=path)

    version = str(cached.get("version") or "")
    version_age = now - float(cached.get("version_checked_at") or 0.0)
    if live and (force or not version or version_age > VERSION_TTL_S) and row["cli"]:
        version = _probe_version(path, row["kind"]) if installed else ""
        if version or force:
            _CACHE.merge(agent, version=version, version_checked_at=now)

    cred: bool | None = None
    checked_age = now - float(cached.get("checked_at") or 0.0)
    if force or checked_age > AUTH_TTL_S or "authenticated" not in cached:
        cred = _raw_credential(agent, row.get("provider", ""), row["kind"])
        spent = _spend_proof(agent)
        spent_recent = bool(spent) and (now - spent) <= SPEND_PROOF_LOOKBACK_S
        # "Integrity is spending authentication": RECENT SPEND proves an
        # agent CAN authenticate when no credential store is inspectable.
        # It never overrules an authoritative probe — Antigravity spent two
        # days ago yet has no token today, and an agent that cannot spend
        # NOW must be alerted on, however recently it worked.
        if cred is True:
            authenticated: bool | None = True
        elif cred is False:
            authenticated = False
        else:  # unknown credential story — only spend can vouch for it
            authenticated = True if spent_recent else None
        row_state = _CACHE.merge(
            agent,
            authenticated=authenticated,
            auth_state=_auth_state_for(installed, authenticated, row["kind"]),
            last_spend=_iso(spent) if spent else "",
        )
    else:
        authenticated = cached.get("authenticated")
        row_state = cached

    auth_state = _auth_state_for(installed, authenticated, row["kind"])

    auto_auth = str(row.get("auth_method") or "") in AUTO_AUTH_METHODS
    return {
        "agent": agent,
        "kind": row["kind"],
        "display_name": row["display_name"],
        "cli": row["cli"],
        "path": path,
        "installed": installed,
        "adapter": row["adapter"],
        "provider": row.get("provider", ""),
        "version": version,
        "authenticated": authenticated,
        "auth_state": auth_state,
        "needs_auth": auth_state in {"missing", "not_installed"},
        "auth_hint": (
            _login_hint(row)
            if auth_state == "missing"
            else (row.get("install_hint") or row.get("install_url") or "")
            if auth_state == "not_installed"
            else ""
        ),
        "auto_auth": auto_auth and row["adapter"] == "acp",
        "install_cmd": row.get("install_cmd", ""),
        "update_cmd": _update_cmd(row),
        "last_spend": row_state.get("last_spend", ""),
        "checked_at": row_state.get("checked_at"),
    }


def snapshot(*, force: bool = False, live: bool = True) -> dict[str, Any]:
    """All external agents. Live probes run in parallel; total latency is
    the slowest single CLI (``--version``), bounded by ``_VERSION_TIMEOUT_S``."""
    rows = _agent_rows()
    if force:
        _CACHE.force_refresh()
    results: list[dict[str, Any]] = [None] * len(rows)  # type: ignore[list-item]

    def _one(index: int, row: dict[str, Any]) -> None:
        try:
            results[index] = _probe_row(row, force=force, live=live)
        except Exception as exc:  # noqa: BLE001 — a probe failure is data
            results[index] = {
                "agent": row["agent"],
                "kind": row["kind"],
                "display_name": row["display_name"],
                "cli": row["cli"],
                "path": "",
                "installed": False,
                "adapter": row["adapter"],
                "provider": row.get("provider", ""),
                "version": "",
                "authenticated": None,
                "auth_state": "unknown",
                "needs_auth": False,
                "auth_hint": "",
                "auto_auth": False,
                "install_cmd": row.get("install_cmd", ""),
                "update_cmd": "",
                "last_spend": "",
                "checked_at": _utc_now(),
                "error": str(exc) or exc.__class__.__name__,
            }

    threads = [threading.Thread(target=_one, args=(i, row), daemon=True) for i, row in enumerate(rows)]
    for thread in threads:
        thread.start()
    deadline = _utc_now() + _VERSION_TIMEOUT_S + 15.0
    for thread in threads:
        thread.join(timeout=max(0.5, deadline - _utc_now()))
    agents = [row for row in results if row]
    return {
        "updated_at": _iso(_utc_now()),
        "agents": agents,
        "alerts": [
            {"agent": row["agent"], "auth_state": row["auth_state"], "hint": row["auth_hint"]}
            for row in agents
            if row.get("needs_auth")
        ],
    }


def cached_rows() -> dict[str, dict[str, Any]]:
    """Cache-only projection (never probes) — safe on the picker's hot path."""
    out: dict[str, dict[str, Any]] = {}
    for row in _agent_rows():
        cached = _CACHE.cached(row["agent"])
        if cached:
            out[row["agent"]] = cached
    return out


def picker_chips_map() -> dict[str, dict[str, Any]]:
    """``{agent: {version, auth_state, auth_hint, installed}}`` from cache.

    The chat composer card and the boot payload render these chips without
    ever touching a subprocess.
    """
    out: dict[str, dict[str, Any]] = {}
    for row in _agent_rows():
        cached = _CACHE.cached(row["agent"])
        if not cached:
            continue
        state = str(cached.get("auth_state") or "")
        if not state:
            authenticated = cached.get("authenticated")
            state = "ok" if authenticated is True else "missing" if authenticated is False else "unknown"
        out[row["agent"]] = {
            "version": str(cached.get("version") or ""),
            "auth_state": state,
            "auth_hint": _login_hint(row) if state == "missing" else "",
            "installed": bool(cached.get("installed", True)),
        }
    return out


def _spawn_background_refresh() -> None:
    with _CACHE._lock:  # noqa: SLF001 — module owns its lock
        if _CACHE._refreshing:  # noqa: SLF001
            return
        _CACHE._refreshing = True  # noqa: SLF001

    def _run() -> None:
        try:
            snapshot()
        except Exception:  # noqa: BLE001
            _LOG.debug("agent_health: background refresh failed", exc_info=True)
        finally:
            with _CACHE._lock:  # noqa: SLF001
                _CACHE._refreshing = False  # noqa: SLF001

    threading.Thread(target=_run, daemon=True).start()


def picker_projection(agent: str) -> dict[str, Any]:
    """Fields the chat agent selector needs, cache-only.

    Answers from the disk/memory cache and kicks a background refresh when
    nothing has been probed yet, so ``/api/models`` never blocks on a CLI.
    """
    row = _spec_for_agent(agent)
    if row is None:
        return {}
    cached = _CACHE.cached(agent)
    if cached is None:
        _spawn_background_refresh()
        return {}
    if _utc_now() - float(cached.get("checked_at") or 0.0) > AUTH_TTL_S:
        _spawn_background_refresh()
    state = str(cached.get("auth_state") or "")
    if not state:
        authenticated = cached.get("authenticated")
        state = "ok" if authenticated is True else "missing" if authenticated is False else "unknown"
    return {
        "version": str(cached.get("version") or ""),
        "auth_state": state,
        "auth_hint": _login_hint(row) if state == "missing" else "",
        "installed": bool(cached.get("installed")),
        "update_cmd": _update_cmd(row),
        "auto_auth": str(row.get("auth_method") or "") in AUTO_AUTH_METHODS,
    }


def refresh_agent(agent: str, *, force: bool = True) -> dict[str, Any] | None:
    """Force one agent's re-probe (the UI's re-check button)."""
    row = _spec_for_agent(agent)
    if row is None:
        return None
    return _probe_row(row, force=force)


def auto_auth(agent: str) -> dict[str, Any]:
    """Attempt the engine's own automatic authentication when one exists."""
    row = _spec_for_agent(agent)
    if row is None:
        return {"attempted": False, "ok": False, "detail": f"unknown agent {agent!r}"}
    result = _authenticate_headless(row)
    if result.get("ok"):
        refresh_agent(agent)
    return result


# --------------------------------------------------------------------------
# Catalog decoration for the Usage subscriptions tab.
# --------------------------------------------------------------------------


def provider_id_for_agent(agent: str) -> str:
    row = _spec_for_agent(agent)
    return str((row or {}).get("provider") or "")


def agent_for_provider_id(provider_id: str) -> str:
    provider_id = str(provider_id or "").strip().lower()
    for row in _agent_rows():
        if row.get("provider") == provider_id:
            return str(row["agent"])
    return ""


def decorate_catalog(catalog: dict[str, Any]) -> dict[str, Any]:
    """Add version/auth to every subscription row and re-admit engines the
    credential probes dropped — with ``auth_required`` so the tab can render
    the sign-in alert instead of pretending the agent does not exist.
    """
    rows = [dict(row) for row in (catalog.get("providers") or [])]
    by_id = {str(row.get("id")): row for row in rows}
    snapshot_data = snapshot(live=False)  # credential reads only; no version probes
    _spawn_background_refresh()  # warm versions for the next paint
    for health in snapshot_data.get("agents") or []:
        provider = health.get("provider") or ""
        if not provider:
            continue
        card = by_id.get(provider)
        if card is not None:
            card["version"] = health.get("version", "")
            card["auth_state"] = health.get("auth_state", "")
            card["installed"] = bool(health.get("installed"))
            if health.get("needs_auth"):
                card["auth_required"] = True
                card["auth_hint"] = health.get("auth_hint", "")
            continue
        # Dropped by the credential probes but still installed: show it.
        if health.get("adapter") == "api" and not health.get("installed"):
            continue
        if not health.get("installed") or health.get("auth_state") != "missing":
            # Not-installed engines stay out of the provider tab (the agent
            # selector carries the install hint); "unknown" is not an alarm.
            continue
        from sharedagents_web.subscription_limits import PROVIDER_NAMES

        rows.append(
            {
                "id": provider,
                "name": PROVIDER_NAMES.get(provider, health.get("display_name", provider)),
                "kind": "subscription",
                "version": health.get("version", ""),
                "auth_state": "missing",
                "auth_required": True,
                "auth_hint": health.get("auth_hint", ""),
                "installed": True,
            }
        )
    out = dict(catalog)
    out["providers"] = rows
    return out


def unauthenticated_provider_card(provider_id: str, *, refresh: bool = False) -> dict[str, Any] | None:
    """Card for a provider the probes dropped (installed, not signed in).

    ``probe_provider`` would raise ``ProviderUnavailable`` and the route
    would answer 404; the tab wants a RENDERABLE card with the sign-in
    instruction instead.
    """
    agent = agent_for_provider_id(provider_id)
    row = _spec_for_agent(agent) if agent else None
    health = _probe_row(row, force=refresh, live=False) if row else None
    if not health or health.get("authenticated") is True or health.get("installed") is False:
        return None
    if health.get("auth_state") != "missing":
        # Only a confidently-missing credential gets an alert card; the
        # "unknown" trio (idle, unprobed store) must not be shouted at.
        return None
    from sharedagents_web.subscription_limits import PROVIDER_NAMES

    hint = health.get("auth_hint") or ""
    error = f"Not signed in — run `{hint}` in a terminal to authenticate" if hint else "Not signed in on this machine"
    return {
        "id": provider_id,
        "name": PROVIDER_NAMES.get(provider_id, health.get("display_name", provider_id)),
        "kind": "subscription",
        "plan": "authentication required",
        "ok": False,
        "error": error,
        "windows": [],
        "auth_required": True,
        "auth_hint": hint,
        "version": health.get("version", ""),
        "updated_at": _iso(_utc_now()),
    }


# --------------------------------------------------------------------------
# Update jobs.
# --------------------------------------------------------------------------


class _UpdateJob:
    def __init__(self, job_id: str, agent: str, command: str) -> None:
        self.job_id = job_id
        self.agent = agent
        self.command = command
        self.status = "running"
        self.output = ""
        self.exit_code: int | None = None
        self.started_at = _utc_now()
        self.finished_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "agent": self.agent,
            "command": self.command,
            "status": self.status,
            "output": self.output[-4000:],
            "exit_code": self.exit_code,
            "started_at": _iso(self.started_at),
            "finished_at": _iso(self.finished_at) if self.finished_at else None,
        }


_JOBS: dict[str, _UpdateJob] = {}
_JOBS_LOCK = threading.Lock()


def start_update(agent: str) -> dict[str, Any]:
    """Re-run the engine's installer to update it (admin action)."""
    import uuid

    row = _spec_for_agent(agent)
    if row is None:
        raise ValueError(f"unknown agent {agent!r}")
    command = _update_cmd(row)
    if not command:
        raise ValueError(f"{agent} has no update command (no local CLI to update)")
    job = _UpdateJob(uuid.uuid4().hex[:12], str(row["agent"]), command)
    with _JOBS_LOCK:
        _JOBS[job.job_id] = job
        # Keep the map small; finished jobs only.
        finished = [j for j in _JOBS.values() if j.status != "running"]
        for stale in sorted(finished, key=lambda j: j.started_at)[:-20]:
            _JOBS.pop(stale.job_id, None)

    def _run() -> None:
        try:
            # Pin the leading token to its resolved path (the web daemon may
            # run with a stripped PATH that omits e.g. ~/.local/bin where uv
            # lives), while keeping the installer's own pipe syntax intact.
            run_cmd = command
            first = command.split(None, 1)[0] if command.strip() else ""
            resolved = _resolve_cli(first) if first else ""
            if resolved and first and not first.startswith(("/", ".")):
                run_cmd = resolved + command[len(first) :]
            proc = subprocess.run(  # noqa: S602,S607 — operator-reviewed installer
                run_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=900,
                stdin=subprocess.DEVNULL,
            )
            job.exit_code = proc.returncode
            job.output = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
            job.status = "ok" if proc.returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            job.status = "failed"
            job.output = "update timed out after 900s"
            job.exit_code = -1
        except Exception as exc:  # noqa: BLE001
            job.status = "failed"
            job.output = str(exc) or exc.__class__.__name__
            job.exit_code = -1
        finally:
            job.finished_at = _utc_now()
            if job.status == "ok":
                # Refresh version right after a successful install.
                try:
                    refresh_agent(agent)
                except Exception:  # noqa: BLE001
                    pass

    threading.Thread(target=_run, daemon=True).start()
    return job.to_dict()


def update_status(job_id: str) -> dict[str, Any] | None:
    with _JOBS_LOCK:
        job = _JOBS.get(str(job_id or ""))
        return job.to_dict() if job else None


__all__ = [
    "agent_for_provider_id",
    "agent_names",
    "auto_auth",
    "cached_rows",
    "decorate_catalog",
    "picker_chips_map",
    "picker_projection",
    "provider_id_for_agent",
    "refresh_agent",
    "snapshot",
    "start_update",
    "unauthenticated_provider_card",
    "update_status",
]
