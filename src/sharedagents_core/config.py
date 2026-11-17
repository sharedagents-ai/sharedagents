"""Persistent configuration for sharedagents — TUI-side, endpoint resolution only.

Resolution order (highest wins): env vars > config file > built-in defaults.

File location (TUI-only):
    $XDG_CONFIG_HOME/sharedagents/config.json   (if XDG_CONFIG_HOME is set)
    ~/.config/sharedagents/config.json          (otherwise)

This file holds the standalone TUI's `--endpoint <alias|url>` mappings
and a few prompt knobs. It is NOT where sharedagents agents read model
deployments from. The agent gateway reads:

    ~/.sharedagents/config.json                 (the GLOBAL agent config —
        contains the `models[]` array with `alias`/`baseUrl`/`id`/
        `apiKeyEnv`/`backend`/`contextWindow`. Owner: sharedagents_agent.agent)

If you're adding a new model/provider for the agent, edit
``~/.sharedagents/config.json`` — NOT the file managed here. The two
schemas are intentionally different and the agent ignores this file
entirely.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from sharedagents_core.config_io import write_config_text

DEFAULTS: dict[str, Any] = {
    # No hardcoded endpoint — first run via `sharedagents configure --endpoint …`
    # or the agent wizard. Hardcoding leaks the author's env and makes a
    # fresh install silently point at a server that doesn't exist.
    "endpoint": "",
    "model": "",
    "system": "You are sharedagents, a practical local-first assistant. Be concise and accurate.",
    "temperature": None,
    "enable_thinking": True,
    # Named endpoints: [{"alias": "gpu-a", "endpoint": "http://gpu-a:8001/v1"}, ...]
    # If set, `sharedagents --endpoint <alias>` resolves to the matching URL.
    "endpoints": [],
    # ---- MCP / ACP (see `mcp_client.py`, `acp_server.py`, `acp_client.py`) ----
    # Map: server-name → MCPServerConfig dict. Every entry is parsed via
    # `mcp_client.MCPServerConfig.from_dict` at use-time; bad rows are
    # logged and skipped, the rest still work.
    "mcp_servers": {},
    # Map: agent-name → ACPAgentConfig dict. Used by the `acp_call` tool
    # to route prompts to remote ACP-compatible peers.
    "acp_agents": {},
    # ACP-server section. When `enabled` is true the gateway exposes
    # `/acp/*` endpoints (mounted by `acp_server.register_acp_routes`).
    # When `auth_token` is set, every endpoint requires
    # `Authorization: Bearer <token>` — set this in any deployment that
    # exposes the gateway beyond localhost.
    "acp_server": {
        "enabled": False,
        "auth_token": None,
        "description": "sharedagents AI agent",
    },
}

ENV_OVERRIDES: dict[str, str] = {
    "endpoint": "SHAREDAGENTS_ENDPOINT",
    "model": "SHAREDAGENTS_MODEL",
    "system": "SHAREDAGENTS_SYSTEM",
    "temperature": "SHAREDAGENTS_TEMPERATURE",
    "enable_thinking": "SHAREDAGENTS_ENABLE_THINKING",
}

_TRUTHY = {"1", "true", "yes", "on", "y", "t"}
_FALSY = {"0", "false", "no", "off", "n", "f"}

FIELDS = tuple(DEFAULTS.keys())


def config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / "sharedagents" / "config.json"


def load_file() -> dict[str, Any]:
    p = config_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_file(data: dict[str, Any]) -> Path:
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    write_config_text(p, json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return p


def _coerce(field: str, raw: Any) -> Any:
    if field == "temperature":
        # None (or "null"/"none"/"") means "send NO temperature — let the
        # upstream apply its own default". The worker drops a None
        # temperature off the wire (forward_sampling_fields).
        if raw is None:
            return None
        if isinstance(raw, str) and raw.strip().lower() in ("null", "none", ""):
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return DEFAULTS["temperature"]
    if field == "enable_thinking":
        if isinstance(raw, bool):
            return raw
        s = str(raw).strip().lower()
        if s in _TRUTHY:
            return True
        if s in _FALSY:
            return False
        return DEFAULTS["enable_thinking"]
    return raw


def effective() -> dict[str, Any]:
    cfg = dict(DEFAULTS)
    file_cfg = load_file()
    for k in FIELDS:
        if k not in file_cfg:
            continue
        v = file_cfg[k]
        if v is None:
            # Explicit null is honored ONLY for temperature (means "omit
            # the param, let the upstream default"). Every other field
            # treats null as "unset" and keeps the built-in default.
            if k == "temperature":
                cfg[k] = None
            continue
        cfg[k] = v
    for k, env_var in ENV_OVERRIDES.items():
        v = os.environ.get(env_var)
        if v is not None and v != "":
            cfg[k] = _coerce(k, v)
    return cfg


def _is_valid_http_url(s: str) -> bool:
    """True if `s` parses as a complete `http`/`https` URL with a host.

    `startswith("http")` accepts garbage like `httpx://`, `http:` (no
    `//`), `http:// ` (whitespace host), and `https:nope`. Any of those
    reaching the worker produces a cryptic httpx error. Validate up
    front so a typo in the wizard fails loudly here instead.
    """
    if not s:
        return False
    try:
        from urllib.parse import urlparse

        u = urlparse(s)
    except (ValueError, TypeError):
        return False
    return u.scheme in ("http", "https") and bool(u.netloc)


def resolve_endpoint(alias_or_url: str, cfg: dict[str, Any] | None = None) -> str:
    """Resolve an alias or raw URL to an upstream endpoint.

    If *alias_or_url* parses as a full `http(s)://host[:port][/path]`
    URL, it's returned as-is (trailing slash stripped). Otherwise it's
    looked up in the ``endpoints`` list of *cfg* (or ``effective()``
    when *cfg* is ``None``) by ``alias``. Falls back to the single
    ``endpoint`` field if not found.

    Strings that *look* like URLs but aren't valid (e.g. `http:nope`,
    bare `https:`) raise `ValueError` rather than being silently
    accepted and producing a cryptic httpx error downstream.
    """
    if not alias_or_url:
        return (cfg or effective()).get("endpoint", "")
    if alias_or_url.startswith(("http://", "https://", "http:", "https:")):
        if not _is_valid_http_url(alias_or_url):
            raise ValueError(
                f"endpoint looks like a URL but isn't valid: {alias_or_url!r}. Expected http(s)://host[:port][/path]."
            )
        return alias_or_url.rstrip("/")
    c = cfg or effective()
    for ep in c.get("endpoints") or []:
        if ep.get("alias") == alias_or_url:
            return (ep.get("endpoint") or "").rstrip("/")
    return c.get("endpoint", "")
