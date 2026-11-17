"""Provider-specific model discovery implementation."""

from __future__ import annotations

import json
import logging
import os
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from sharedagents_agent.external_registry import EXTERNAL_REGISTRY
from sharedagents_model.models import effort_capability_from_metadata

_LOG = logging.getLogger("sharedagents.discovery.common")

_TTL_S = 3600.0

_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}

ANTHROPIC_MODELS_URL = "https://api.anthropic.com/v1/models"

CLAUDE_OAUTH_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"

CLAUDE_OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"

OPENAI_MODELS_URL = "https://api.openai.com/v1/models"

GEMINI_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"

CODEX_CHATGPT_MODELS_URL = "https://chatgpt.com/backend-api/codex/models"

CODEX_RESPONSES_CLIENT_VERSION_FALLBACK = "99.0.0"

CODEX_RESPONSES_API = "codex-responses"

CODEX_OAUTH_AUTH = "codex-oauth"

CODEX_TRANSPORT_ALIAS_PREFIX = "codex:"

CODEX_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"

CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"

_SHIPPED_EXTERNAL_MODELS: dict[str, tuple[dict[str, str], ...]] = {
    "anthropic": (
        {
            "alias": "opus-5-5",
            "id": "claude-opus-5-5",
            "display": "Claude Opus 5.5",
            "provider": "anthropic",
            "backend": "anthropic",
            "contextWindow": 1_000_000,
        },
        {
            "alias": "sonnet-5-5",
            "id": "claude-sonnet-5-5",
            "display": "Claude Sonnet 5.5",
            "provider": "anthropic",
            "backend": "anthropic",
            "contextWindow": 1_000_000,
        },
        {
            "alias": "fable-5-1",
            "id": "claude-fable-5-1",
            "display": "Claude Fable 5.1",
            "provider": "anthropic",
            "backend": "anthropic",
            "contextWindow": 1_000_000,
        },
        {
            "alias": "opus-5",
            "id": "claude-opus-5",
            "display": "Claude Opus 5",
            "provider": "anthropic",
            "backend": "anthropic",
        },
        {
            "alias": "sonnet-5",
            "id": "claude-sonnet-5",
            "display": "Claude Sonnet 5",
            "provider": "anthropic",
            "backend": "anthropic",
        },
        {
            "alias": "fable-5",
            "id": "claude-fable-5",
            "display": "Claude Fable 5",
            "provider": "anthropic",
            "backend": "anthropic",
        },
        {
            "alias": "opus-4-8",
            "id": "claude-opus-4-8",
            "display": "Claude Opus 4.8",
            "provider": "anthropic",
            "backend": "anthropic",
        },
        {
            "alias": "opus-4-7",
            "id": "claude-opus-4-7",
            "display": "Claude Opus 4.7",
            "provider": "anthropic",
            "backend": "anthropic",
        },
        {
            "alias": "sonnet-4-6",
            "id": "claude-sonnet-4-6",
            "display": "Claude Sonnet 4.6",
            "provider": "anthropic",
            "backend": "anthropic",
        },
        {
            "alias": "opus-4-6",
            "id": "claude-opus-4-6",
            "display": "Claude Opus 4.6",
            "provider": "anthropic",
            "backend": "anthropic",
        },
        {
            "alias": "opus-4-5-20251101",
            "id": "claude-opus-4-5-20251101",
            "display": "Claude Opus 4.5",
            "provider": "anthropic",
            "backend": "anthropic",
        },
        {
            "alias": "haiku-4-5-20251001",
            "id": "claude-haiku-4-5-20251001",
            "display": "Claude Haiku 4.5",
            "provider": "anthropic",
            "backend": "anthropic",
        },
        {
            "alias": "sonnet-4-5-20250929",
            "id": "claude-sonnet-4-5-20250929",
            "display": "Claude Sonnet 4.5",
            "provider": "anthropic",
            "backend": "anthropic",
        },
        {
            "alias": "opus-4-1-20250805",
            "id": "claude-opus-4-1-20250805",
            "display": "Claude Opus 4.1",
            "provider": "anthropic",
            "backend": "anthropic",
        },
    ),
    "antigravity": (
        {
            "alias": "3.8-flash-high",
            "id": "gemini-3.8-flash-high",
            "display": "Gemini 3.8 Flash (High)",
            "provider": "antigravity",
            "backend": "antigravity",
            "contextWindow": 1_000_000,
        },
        {
            "alias": "3.8-flash-medium",
            "id": "gemini-3.8-flash-medium",
            "display": "Gemini 3.8 Flash (Medium)",
            "provider": "antigravity",
            "backend": "antigravity",
            "contextWindow": 1_000_000,
        },
        {
            "alias": "3.8-flash-low",
            "id": "gemini-3.8-flash-low",
            "display": "Gemini 3.8 Flash (Low)",
            "provider": "antigravity",
            "backend": "antigravity",
            "contextWindow": 1_000_000,
        },
        {
            "alias": "3.8-flash",
            "id": "gemini-3.8-flash",
            "display": "Gemini 3.8 Flash",
            "provider": "antigravity",
            "backend": "antigravity",
            "contextWindow": 1_000_000,
        },
        {
            "alias": "3.7-flash",
            "id": "gemini-3.7-flash",
            "display": "Gemini 3.7 Flash",
            "provider": "antigravity",
            "backend": "antigravity",
            "contextWindow": 1_000_000,
        },
        {
            "alias": "3.7-flash-thinking",
            "id": "gemini-3.7-flash-thinking",
            "display": "Gemini 3.7 Flash Thinking",
            "provider": "antigravity",
            "backend": "antigravity",
            "contextWindow": 1_000_000,
        },
        {
            "alias": "3.1-pro",
            "id": "gemini-3.1-pro",
            "display": "Gemini 3.1 Pro",
            "provider": "antigravity",
            "backend": "antigravity",
            "contextWindow": 1_000_000,
        },
        {
            "alias": "2.5-pro",
            "id": "gemini-2.5-pro",
            "display": "Gemini 2.5 Pro",
            "provider": "antigravity",
            "backend": "antigravity",
            "contextWindow": 1_000_000,
        },
        {
            "alias": "2.5-flash",
            "id": "gemini-2.5-flash",
            "display": "Gemini 2.5 Flash",
            "provider": "antigravity",
            "backend": "antigravity",
            "contextWindow": 1_000_000,
        },
    ),
    "cursor": (
        {
            "alias": "composer-2.5",
            "id": "composer-2.5",
            "display": "Composer 2.5",
            "provider": "cursor",
            "backend": "cursor",
            "contextWindow": 200_000,
        },
        {
            "alias": "composer-2.5-fast",
            "id": "composer-2.5-fast",
            "display": "Composer 2.5 Fast",
            "provider": "cursor",
            "backend": "cursor",
            "contextWindow": 200_000,
        },
        {
            "alias": "auto",
            "id": "auto",
            "display": "Auto",
            "provider": "cursor",
            "backend": "cursor",
            "contextWindow": 1_000_000,
        },
    ),
    # No "codex" rows: codex models come only from the installed codex's own
    # list (see ``codex_catalogue``), never from a shipped table.
    "kimi": (
        {
            "alias": "kimi-k3",
            "id": "moonshot-ai/kimi-k3",
            "display": "Kimi K3",
            "provider": "kimi",
            "backend": "kimi",
            "contextWindow": 1_000_000,
        },
    ),
    "hermes": (
        {
            "alias": "sonnet-5",
            "id": "anthropic:claude-sonnet-5",
            "display": "Claude Sonnet 5 (Hermes native)",
            "provider": "hermes",
            "backend": "hermes",
            "contextWindow": 1_000_000,
        },
    ),
}


def _fallback_models(provider: str) -> list[dict[str, Any]]:
    """Fallback model rows for ``external_models.<provider>``: the operator's
    configured rows UNIONED with the shipped provider catalogue (operator
    rows first, deduped by lowercased id — an operator list never EVICTS a
    shipped id). Rows whose id is a bare engine
    label are dropped so every picker row can pass the spawn gate (the
    label invariant is unconditional). Returns shipped rows (or ``[]``)
    when the config section is absent / not a list."""
    operator: list[dict[str, Any]] = []
    try:
        from sharedagents_agent.agent import load_global_config

        cfg = load_global_config()
        em = cfg.get("external_models") or {}
        if isinstance(em, dict) and provider in em:
            rows = em.get(provider)
            if isinstance(rows, list):
                operator = [dict(r) for r in rows if isinstance(r, dict)]
    except Exception:  # noqa: BLE001
        pass
    shipped = [dict(row) for row in _SHIPPED_EXTERNAL_MODELS.get(provider, ())]
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in [*operator, *shipped]:
        mid = str(row.get("id") or "").strip()
        if not mid or mid.lower() in CLI_ENGINE_LABELS:
            continue
        key = mid.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _mono() -> float:
    return time.monotonic()


def _cache_get(key: str) -> list[dict[str, Any]] | None:
    hit = _cache.get(key)
    if hit is None:
        return None
    deadline, rows = hit
    if _mono() >= deadline:
        _cache.pop(key, None)
        return None
    return [dict(r) for r in rows]


def _cache_put(key: str, rows: list[dict[str, Any]]) -> None:
    _cache[key] = (_mono() + _TTL_S, [dict(r) for r in rows])


def _cache_put_ttl(key: str, rows: list[dict[str, Any]], ttl_s: float) -> None:
    """Cache ``rows`` under ``key`` with a custom TTL (seconds). Used for the
    config fallback when the codex OAuth path was attempted but failed — a
    short TTL re-tries discovery soon (the codex CLI may refresh the token in
    the meantime) rather than serving a stale list for the full hour."""
    _cache[key] = (_mono() + ttl_s, [dict(r) for r in rows])


# The persisted cache checks this immediately after a forced discovery.
# Keep that result local to the calling context: direct engine discovery in
# another thread must not change whether this refresh is safe to persist.
_fallback_served: ContextVar[frozenset[str]] = ContextVar(
    "discovery_fallback_served",
    default=frozenset(),
)


def served_fallback(provider: str) -> bool:
    """Whether this context's latest uncached discovery used the static list."""
    return provider in _fallback_served.get()


def _set_fallback_served(provider: str, served: bool) -> None:
    providers = _fallback_served.get()
    if served:
        _fallback_served.set(providers | {provider})
    else:
        _fallback_served.set(providers - {provider})


def _http_get_json(
    url: str,
    headers: dict[str, str],
    timeout: float = 4.0,
) -> dict[str, Any]:
    import httpx

    resp = httpx.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _row_with_effort(
    row: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Attach model-specific effort levels published by a catalogue API."""
    capability = effort_capability_from_metadata(metadata)
    if capability is not None:
        row["effort"] = capability
    return row


def _codex_cache_model_map() -> dict[str, str]:
    """lowercased id → the backend's own spelling, for every model the
    INSTALLED codex CLI is offered (:func:`.codex_catalogue.load_codex_catalogue`:
    the shared ``models_cache.json`` only when that same codex version wrote
    it, else the list saved for that version). Case-preserving so a
    case-sensitive slug reaches ``--model`` verbatim.
    Network-free; ``{}`` while the installed version's list is unknown — there
    is no hardcoded fallback list. Never raises.
    """
    try:
        from .codex_catalogue import load_codex_catalogue

        return load_codex_catalogue().ids()
    except Exception:  # noqa: BLE001
        return {}


def _codex_cache_models() -> frozenset[str]:
    """Lowercased ids of :func:`_codex_cache_model_map`. Never raises."""
    return frozenset(_codex_cache_model_map())


def _codex_cache_context_windows() -> dict[str, int]:
    """lowercased id → max context window (tokens) for the installed codex's
    models, when the backend publishes one.

    The backend list carries ``context_window`` (the model's default window)
    and ``max_context_window`` (the absolute max) per model. The picker's
    ``contextWindow`` badge and the session context meter both want the MAX,
    so the larger of the two wins. Same source as
    :func:`_codex_cache_model_map`. Never raises.
    """
    out: dict[str, int] = {}
    try:
        from .codex_catalogue import load_codex_catalogue

        out = load_codex_catalogue().context_windows()
    except Exception:  # noqa: BLE001
        pass
    return out


CLI_ENGINE_LABELS: frozenset[str] = frozenset(
    {
        "acp",
        *(spec.short_name for spec in EXTERNAL_REGISTRY.values()),
    }
)

_CLI_ENGINE_PROVIDERS: dict[str, str] = {
    spec.kind.value: spec.provider for spec in EXTERNAL_REGISTRY.values() if spec.provider
}

_CLI_ENGINE_AGENT_PROFILES: dict[str, str] = {
    "claude_cli": "claude",
    "codex_cli": "codex",
    "antigravity_cli": "antigravity",
    "zcode_cli": "zcode",
}

_ACP_ENGINE_KINDS: frozenset[str] = frozenset(
    {
        "acp",
        *(spec.kind.value for spec in EXTERNAL_REGISTRY.values() if spec.adapter == "acp"),
    }
)


def _engine_allowlist(engine_kind: str) -> frozenset[str]:
    """Deterministic, network-free set of passable wire model ids for an
    external engine — or ``frozenset()`` when no catalogue is discoverable.
    Ids keep their ORIGINAL spelling (an operator
    catalogue row ``MyCaseSensitiveDeployment`` must round-trip verbatim,
    never lowercased).

    * ``codex_cli`` — the INSTALLED codex's own model list
      (:func:`_codex_gate_lookup`); no hardcoded ids;
    * ``claude_cli`` — operator ``external_models.anthropic`` rows ∪ the
      shipped anthropic catalogue (live API discovery is deliberately NOT
      part of the gate: the gate must be deterministic and offline);
    * antigravity / cursor / opencode / kimi — operator
      ``external_models.<provider>`` rows ONLY (no shipped catalogue). An
      empty result means the gate degrades to the label invariant — the
      vendor CLI / ACP server remains the authority for catalogue ids
      (documented limitation).
    * ``zcode_cli`` — the CLI's own synced catalogue for this login (plus its
      configured model); the runner applies the choice per run via an
      isolated provider config, not a ``--model`` flag.
    """
    if engine_kind == "codex_cli":
        return frozenset(_codex_gate_lookup().values())
    if engine_kind == "zcode_cli":
        # `zcode -p` has no --model flag: the CLI runs the model in its own
        # provider config, so that id is the entire passable set and the
        # refusal copy must not claim a catalogue it cannot offer.
        from .zcode import discover_zcode_models

        return frozenset(r["id"] for r in discover_zcode_models())
    provider = _CLI_ENGINE_PROVIDERS.get(engine_kind)
    if not provider:
        return frozenset()
    rows = _fallback_models(provider)
    return frozenset(str(r.get("id") or "").strip() for r in rows if r.get("id"))


def _codex_gate_lookup() -> dict[str, str]:
    """The spawn gate's ``{lowercased id: spelling}`` for the installed codex:
    its version-matched cache plus the list saved for the current codex
    login, else what the installed codex reports when asked once
    (``codex debug models``). ``{}`` when the list is still unknown (offline,
    codex version unreadable); the gate then defers to codex itself instead
    of refusing a model it cannot judge. A refusal is re-checked against
    codex (:func:`validate_external_cli_model`). Never raises.
    """
    try:
        from .codex_catalogue import load_codex_catalogue

        catalogue = load_codex_catalogue()
        if not catalogue.known:
            catalogue = load_codex_catalogue(ask_codex=True)
        return catalogue.ids()
    except Exception:  # noqa: BLE001
        return {}


def _engine_catalog_lookup(engine_kind: str) -> dict[str, str]:
    """Case-insensitive catalogue lookup for an external engine:
    ``{lowercased id: ORIGINAL spelling}``.

    Shares the SAME source as the discovery picker (the operator ∪ shipped
    ``_fallback_models`` union; for codex the installed codex's own list,
    which the picker's live fetch also saves), so every row a picker can
    offer validates and a catalog match returns the
    catalogue's original spelling, never a lowercased mutation.
    Empty for engines with no discoverable catalogue.
    """
    if engine_kind == "codex_cli":
        return _codex_gate_lookup()
    provider = _CLI_ENGINE_PROVIDERS.get(engine_kind)
    if not provider:
        return {}
    lookup = {
        str(r.get("id") or "").strip().lower(): str(r.get("id") or "").strip()
        for r in _fallback_models(provider)
        if r.get("id")
    }
    # Cursor's picker lists ``cursor-agent --list-models`` live; the spawn
    # gate must accept every discovered id or the picker lies.
    if engine_kind == "cursor_acp":
        try:
            from .acp_list import discover_cursor_models

            for row in discover_cursor_models():
                mid = str(row.get("id") or "").strip()
                if mid:
                    lookup[mid.lower()] = mid
        except Exception:  # noqa: BLE001
            pass
    return lookup


def validate_external_cli_model(
    engine_kind: str,
    model: str | None,
) -> str | None:
    """Fail-fast gate for an external engine's effective wire model.

    Returns the CANONICAL passable id — for a catalogue match, the
    catalogue's ORIGINAL spelling (a mixed-case ``"CLAUDE-OPUS-5"``
    validates AND spawns as ``claude-opus-5`` because the catalog entry's
    spelling is returned verbatim; ``"Opus[1m]"`` with no catalog match
    passes through untouched) — ``""`` for an
    empty override (no flag), and ``None`` when the model MUST be refused
    before it reaches the wire.

    Invariant (owner scope change): the engine's own display label
    (``"claude"`` / ``"codex"`` / ``"antigravity"`` / …) is NEVER a wire
    model id and is ALWAYS refused.

    Claude-specific: after the label
    invariant the Claude CLI is the authority for its own model syntax —
    documented aliases (``opus`` / ``sonnet`` / ``haiku``), ``[1m]`` /
    ``[2m]`` extended forms, and operator deployments pass through verbatim
    when no catalogue entry matches. For every OTHER engine with a
    discoverable catalogue the gate still fails CLOSED on anything outside
    it (the deepseek-v4-flash incident shape: a foreign provider id, a
    local vLLM id, a typo).     When no catalogue is discoverable the label
    invariant still holds; anything else passes through because the vendor
    CLI / ACP server is the authority for ids (documented limitation) —
    EXCEPT antigravity, which still refuses non-Gemini ids. A local vLLM
    id such as ``qwen3.8-27b-fp8`` must never become ``antigravity --model``
    just because ``external_models.antigravity`` is empty (fresh-install
    auto-default leak).
    """
    if model is None:
        return ""
    mid = str(model).strip()
    if not mid:
        return ""
    if mid.lower() in CLI_ENGINE_LABELS:
        return None
    if engine_kind == "antigravity_cli" and mid.lower() == "auto":
        # Picker "auto" means the CLI's own settings default — no flag.
        return ""
    if engine_kind == "zcode_cli":
        # ZCode runs GLM coding models on the Z.AI coding plan. Fail CLOSED
        # on anything that is not a GLM chat id: the merged global/local-LLM
        # ``upstream_model`` (a vLLM id such as ``qwen3.8-27b-fp8`` after
        # ``sharedagents configure``) must never become ``zcode --model`` on a
        # fresh install. Returns the CLI's own casing for a known id.
        from .zcode import canonical_zcode_model

        return canonical_zcode_model(mid) or None
    from .google import _is_gemini_chat_model_id

    lookup = _engine_catalog_lookup(engine_kind)
    if not lookup:
        if engine_kind == "antigravity_cli":
            return mid if _is_gemini_chat_model_id(mid) else None
        return mid
    original = lookup.get(mid.lower())
    if original is not None:
        return original
    if engine_kind == "claude_cli":
        # The Claude CLI is the authority for its own model syntax: any
        # non-label value from settings.json / ANTHROPIC_MODEL / an
        # operator-configured agent is trusted verbatim.
        return mid
    if engine_kind == "antigravity_cli" and _is_gemini_chat_model_id(mid):
        # Live Gemini ids the CLI accepts even before they land in config.
        return mid
    if engine_kind == "codex_cli":
        # Refuse only what the installed codex, asked now, does not list: the
        # list at hand may be a same-version cache from another login or older
        # than a newly added model (codex_catalogue.confirm_codex_catalogue).
        try:
            from .codex_catalogue import confirm_codex_catalogue

            catalogue, vouched = confirm_codex_catalogue()
            original = catalogue.ids().get(mid.lower())
        except Exception:  # noqa: BLE001
            original, vouched = None, False
        if original is not None:
            return original
        if not vouched:
            # Codex could not vouch for its list just now (refresh failed,
            # offline): let codex itself judge the model instead of refusing.
            return mid
    return None


def _claude_env_model() -> str:
    """``ANTHROPIC_MODEL`` from the process environment.

    Checked BEFORE ``~/.claude/settings.json`` in the claude default chain
    — Claude Code precedence: ``--model`` > ``ANTHROPIC_MODEL`` env >
    settings.json. Returns ``""`` when unset.
    """
    return str(os.environ.get("ANTHROPIC_MODEL") or "").strip()


def _claude_settings_model() -> str:
    """The model the Claude CLI itself is configured with via its settings
    FILE: ``~/.claude/settings.json`` ``model`` (or ``env.ANTHROPIC_MODEL``
    within it). Honors ``CLAUDE_CONFIG_DIR``.

    NEVER consults the process environment — that is
    :func:`_claude_env_model`, which OUTRANKS this value (Claude Code
    precedence). Returns ``""`` when unset / unreadable; the
    caller validates whatever comes back (a value the Claude CLI accepts —
    alias, ``[1m]`` form, deployment id — passes through the gate)."""
    candidate = ""
    try:
        base = os.environ.get("CLAUDE_CONFIG_DIR") or str(Path.home() / ".claude")
        raw = Path(base) / "settings.json"
        if raw.is_file():
            data = json.loads(raw.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                candidate = str(data.get("model") or "").strip()
                if not candidate:
                    env_block = data.get("env")
                    if isinstance(env_block, dict):
                        candidate = str(env_block.get("ANTHROPIC_MODEL") or "").strip()
    except Exception:  # noqa: BLE001
        pass
    return candidate


def _claude_profile_upstream_model(agent_name: str) -> str:
    """The engine profile's operator-EXPLICIT ``upstream_model`` — read from
    the per-agent config file ONLY (``~/.sharedagents/agents/<name>/config.json``).

    The merged ``AgentConfig.load(...)`` value would ALSO absorb the GLOBAL
    config's ``upstream_model`` — the local-LLM deployment default, which
    belongs to a different engine entirely and must never ride to the
    Claude CLI as ``--model`` (only an operator
    EXPLICIT choice seeds the chain; a builtin pin no longer exists for
    claude). Returns ``""`` when the agent file does not set one.
    """
    try:
        from sharedagents_agent.agent import agents_root

        path = agents_root() / agent_name / "config.json"
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return str(data.get("upstream_model") or "").strip()
    except Exception:  # noqa: BLE001
        pass
    return ""


def _cli_profile_model_row(engine_kind: str) -> dict[str, str]:
    """``{"alias", "id"}`` from the engine profile's own config file
    (``upstream_alias`` → ``upstream_model``), or ``{}`` when it sets no
    model. Lets a picker alias resolve to the operator's configured id when no
    catalogue lists it. Never raises."""
    profile = _CLI_ENGINE_AGENT_PROFILES.get(engine_kind)
    if not profile:
        return {}
    try:
        from sharedagents_agent.agent import agents_root

        path = agents_root() / profile / "config.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        mid = str(data.get("upstream_model") or "").strip() if isinstance(data, dict) else ""
        if mid:
            return {"alias": str(data.get("upstream_alias") or "").strip(), "id": mid}
    except Exception:  # noqa: BLE001
        pass
    return {}


def external_cli_effective_default(
    engine_kind: str,
    *,
    configured: str | None = None,
    configured_explicit: bool = False,
) -> str:
    """The engine's effective default wire model when no override is given.

    ALWAYS a canonical, passable id for the engine — or ``""`` when the
    engine has no safe default (the CLI's own default then applies and no
    ``--model`` flag is built). NEVER the engine label and never a raw,
    unvalidated configured value.

    Resolution chain (first validated hit wins):

    1. ``configured`` — the ACTUAL agent's ``upstream_model``. For claude,
       codex and antigravity accepted ONLY when ``configured_explicit`` (the operator set it in
       the agent's per-agent config — never the merged global/local-LLM
       upstream, never a builtin pin). The session stamp and
       the engine gate both pass the agent's merged cfg, so the stamped
       session model and the spawn wire id agree by construction;
    2. the engine profile's per-agent ``upstream_model`` read from its own
       file only (``~/.sharedagents/agents/<claude|codex|antigravity>/config.json``
       — see :func:`_claude_profile_upstream_model`);
    3. claude: ``ANTHROPIC_MODEL`` env — BEFORE settings.json;
       codex: the model codex itself is configured with (``config.toml``);
    4. claude: ``~/.claude/settings.json`` ``model`` / ``env.ANTHROPIC_MODEL``;
    5. NO pinned fallback for any engine: with no explicit model selection
       the CLI applies its own default and sharedagents builds no ``--model``;
       ACP engines: the ACP server / ``acp_model`` config owns the model.

    The configured-value candidates are ONLY accepted when they validate.
    For claude, validation is the label invariant plus the CLI's own
    authority over model syntax (aliases / ``[1m]`` forms pass); for codex
    it fails closed against the installed codex's own model list. With no discoverable
    catalogue (antigravity without ``external_models.antigravity``) the
    label invariant alone applies — in particular an agent's global/local
    ``upstream_model`` (e.g. a vLLM id) must never ride to a vendor CLI as
    ``--model``.
    """
    if engine_kind in _ACP_ENGINE_KINDS:
        # ACP engines have no safe sharedagents-side default: the model is owned
        # by the ACP server (session default) or the agent's acp_model pin.
        return ""
    if engine_kind == "zcode_cli":
        # No sharedagents-side pin. The model the CLI itself is configured with
        # (``~/.zcode/v2/provider_config.json``) is what runs, so sharedagents
        # builds ``--model`` ONLY from an operator-explicit per-agent value.
        # Stamping the CLI's own default onto argv would re-case a value the
        # CLI already applies and risk rejecting its own default.
        if configured_explicit:
            canonical = validate_external_cli_model(engine_kind, configured)
            if canonical:
                return canonical
        return ""
    if engine_kind == "claude_cli":
        # Only an operator-EXPLICIT per-agent upstream_model
        # seeds the claude chain (``configured_explicit``), then the claude
        # profile's own per-agent file, then env BEFORE settings (1c), then
        # NO pin — the Claude CLI applies its own default.
        if configured_explicit:
            canonical = validate_external_cli_model(engine_kind, configured)
            if canonical:
                return canonical
        profile = _CLI_ENGINE_AGENT_PROFILES.get(engine_kind)
        if profile:
            canonical = validate_external_cli_model(
                engine_kind,
                _claude_profile_upstream_model(profile),
            )
            if canonical:
                return canonical
        for candidate in (_claude_env_model(), _claude_settings_model()):
            canonical = validate_external_cli_model(engine_kind, candidate)
            if canonical:
                return canonical
        return ""
    if engine_kind == "antigravity_cli":
        # Same leak class as claude: the merged global/local-LLM
        # ``upstream_model`` (``qwen3.8-27b-fp8`` after ``sharedagents configure``)
        # is not a Gemini id. Auto means omit ``--model`` so the CLI's own
        # settings pick Gemini 3.x. Only an operator-EXPLICIT per-agent
        # Gemini id may seed the default.
        if configured_explicit:
            canonical = validate_external_cli_model(engine_kind, configured)
            if canonical:
                return canonical
        profile = _CLI_ENGINE_AGENT_PROFILES.get(engine_kind)
        if profile:
            canonical = validate_external_cli_model(
                engine_kind,
                _claude_profile_upstream_model(profile),
            )
            if canonical:
                return canonical
        return ""
    if engine_kind == "codex_cli":
        return _codex_effective_default(configured, configured_explicit)
    canonical = validate_external_cli_model(engine_kind, configured)
    if canonical:
        return canonical
    profile = _CLI_ENGINE_AGENT_PROFILES.get(engine_kind)
    if profile:
        try:
            from sharedagents_agent.agent import AgentConfig

            canonical = validate_external_cli_model(
                engine_kind,
                str(AgentConfig.load(profile).upstream_model or ""),
            )
            if canonical:
                return canonical
        except Exception:  # noqa: BLE001
            pass
    return ""


def _codex_effective_default(configured: str | None, configured_explicit: bool) -> str:
    """Codex default chain: an operator-EXPLICIT per-agent ``upstream_model``,
    then ``~/.sharedagents/agents/codex/config.json``, then the model codex itself
    is configured with (``config.toml``), else ``""`` (no ``--model``).

    No builtin pin exists. Like
    claude, the MERGED agent value is never consulted: without a
    pin it is the global local-LLM upstream (a vLLM id), and while the codex
    list is unknown the gate cannot catch it before ``codex --model``.
    """
    if configured_explicit:
        canonical = validate_external_cli_model("codex_cli", configured)
        if canonical:
            return canonical
    for candidate in (_claude_profile_upstream_model("codex"), _codex_config_model()):
        canonical = validate_external_cli_model("codex_cli", candidate)
        if canonical:
            return canonical
    return ""


def _codex_config_model() -> str:
    """The model codex is configured with: the active ``[profiles.<name>]``
    ``model``, else top-level ``model`` in ``$CODEX_HOME/config.toml``;
    ``""`` when unset or unreadable."""
    try:
        from .openai_codex import _codex_config, _codex_config_profile, _codex_home

        data = _codex_config(_codex_home())
        model = _codex_config_profile(data).get("model") or data.get("model")
        return str(model or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def external_cli_model_refusal(
    model: str,
    engine_kind: str,
    *,
    source: str = "explicit",
    backend_note: str = "",
) -> str:
    """Operator-facing refusal copy for an external engine's model gate hit.

    ``engine_kind`` is the ``EngineKind`` VALUE string (``"claude_cli"`` /
    ``"codex_cli"`` / …). ``source`` names where the refused id came from
    so the message reflects reality instead of
    always blaming parent inheritance:

    * ``"explicit"`` — passed as an explicit ``Agent(model=...)`` override;
    * ``"inherited"`` — propagated from the parent session's model override;
    * ``"picker"`` — selected from the ``/models`` picker;
    * ``"default"`` — the engine profile's configured ``upstream_model``
      (an operator misconfiguration).

    ``backend_note`` is an optional engine-specific symptom sentence (e.g.
    codex's 400 rejection) appended before the remediation advice. For codex
    the valid list is labelled with the installed codex version it belongs to.
    """
    label = engine_kind.replace("_cli", "").replace("_acp", "")
    src_note = {
        "explicit": " It was passed as an explicit model override.",
        "inherited": (
            " It was inherited from the parent session's model override — a"
            f" {label} delegate must never inherit a foreign model."
        ),
        "picker": " It was selected from the /models picker.",
        "default": (f" It comes from the {label} profile's configured upstream_model."),
    }.get(source, "")
    allowlist = sorted(_engine_allowlist(engine_kind))
    valid = ", ".join(allowlist) if allowlist else "(catalogue not configured)"
    valid_label = "valid"
    if engine_kind == "codex_cli":
        try:
            from .codex_catalogue import installed_codex_version

            version = installed_codex_version()
            if version:
                valid_label = f"valid for codex {version}"
        except Exception:  # noqa: BLE001
            pass
    default = external_cli_effective_default(engine_kind)
    default_clause = f" (default: {default})" if default else ""
    return (
        f"[{label} delegate: model '{model}' is not valid for {label}"
        f" ({valid_label}: {valid}).{src_note}{backend_note} Remove the override or set"
        f" upstream_model to a {label} id{default_clause} and retry.]"
    )


def _codex_canonical_model_id(model_id: str | None) -> str | None:
    """Canonical wire slug for ``model_id``, or ``None`` when it is not a
    passable codex model.

    The canonical form is the backend entry's own spelling: a mixed-case
    user value (``"GPT-5.6-SOL"``) validates AND spawns as
    ``--model gpt-5.6-sol`` — the CLI's slug comparison is exact, so the raw
    casing would 400 — while a case-sensitive entry keeps its ORIGINAL
    casing. The spawn paths build ``--model`` from THIS canonical value, never the raw user
    string. Network-free: ``None`` while the installed codex's list is
    unknown (the picker then offers nothing it cannot back up).
    """
    mid = (model_id or "").strip().lower()
    if not mid:
        return ""
    return _codex_cache_model_map().get(mid)


def _is_codex_model_id(model_id: str) -> bool:
    """True ONLY for a model the INSTALLED codex CLI is offered by the
    backend (:mod:`.codex_catalogue`) — a newly added model works as soon as
    that list has it, without a source change, and a dropped one stops
    being offered. Everything else — gpt-4o, gpt-9999, any foreign provider
    id, and the literal ``"codex"`` label — is False. Used to filter picker
    rows; the spawn gate itself is :func:`validate_codex_model`.
    """
    return _codex_canonical_model_id(model_id) is not None


def codex_effective_default_model() -> str:
    """The codex profile's effective default wire model for ``--model``.

    OPTION B: with no explicit override, sharedagents
    PASSES this id as ``--model`` so the session stamp and the spawn agree.
    Resolution (:func:`_codex_effective_default`): the operator's
    ``~/.sharedagents/agents/codex/config.json`` ``upstream_model``, then the
    model codex itself is configured with (``config.toml``). There is NO
    pinned fallback: ``""`` means no ``--model`` and codex applies its own
    built-in default. Also used on a resumed thread with no model of its own
    — codex does NOT keep a thread's model across ``exec resume``.

    A non-empty result is ALWAYS a canonical,
    passable codex id, validated through the same gate as an explicit
    override (:func:`validate_codex_model`); a raw, unvalidated id never
    leaves this function.
    """
    return external_cli_effective_default("codex_cli")


def validate_codex_model(model: str | None) -> str | None:
    """Fail-fast gate for a codex delegate's effective model — canonical form.

    Codex facade over the engine-generic gate
    (:func:`validate_external_cli_model` with ``engine_kind="codex_cli"``).
    Returns the CANONICAL wire slug when ``model`` may be passed to
    ``codex --model`` (the backend entry's own spelling), ``""`` for an
    empty override (no flag), and ``None`` when the model MUST be refused.
    Callers build the operator-facing message with :func:`codex_model_refusal`.

    The incident: a codex_cli delegate inherited the LOCAL parent's model
    override (deepseek-v4-flash) and the ChatGPT backend rejected it with a
    400, burning the whole delegation. Anything the INSTALLED codex is not
    offered (:func:`_codex_gate_lookup`) fails fast BEFORE the CLI spawn.
    The literal ``"codex"`` label (the CLI engine's ``display_model`` name,
    never a real model id) is refused here too, so it can never reach
    ``codex --model``. Only while that list is unknown (offline, nothing
    saved) do other ids pass through for codex itself to judge.
    """
    return validate_external_cli_model("codex_cli", model)


def codex_model_refusal(model: str, *, source: str = "explicit") -> str:
    """Operator-facing refusal copy for a codex model gate hit.

    Codex facade over the engine-generic refusal
    (:func:`external_cli_model_refusal` with ``engine_kind="codex_cli"``).
    ``source`` names where the refused id came from so the message
    reflects reality instead of always blaming
    parent inheritance:

    * ``"explicit"`` — passed as an explicit ``Agent(model=...)`` override;
    * ``"inherited"`` — propagated from the parent session's model override;
    * ``"picker"`` — selected from the ``/models`` picker;
    * ``"default"`` — the codex profile's configured ``upstream_model``
      (an operator misconfiguration).
    """
    return external_cli_model_refusal(
        model,
        "codex_cli",
        source=source,
        backend_note=(" The ChatGPT-backend codex CLI rejects invalid ids with a 400 'model not supported'."),
    )


__all__ = [
    "_LOG",
    "_TTL_S",
    "_cache",
    "ANTHROPIC_MODELS_URL",
    "CLAUDE_OAUTH_TOKEN_URL",
    "CLAUDE_OAUTH_CLIENT_ID",
    "OPENAI_MODELS_URL",
    "GEMINI_MODELS_URL",
    "CODEX_CHATGPT_MODELS_URL",
    "CODEX_RESPONSES_CLIENT_VERSION_FALLBACK",
    "CODEX_RESPONSES_API",
    "CODEX_OAUTH_AUTH",
    "CODEX_TRANSPORT_ALIAS_PREFIX",
    "CODEX_OAUTH_TOKEN_URL",
    "CODEX_OAUTH_CLIENT_ID",
    "_SHIPPED_EXTERNAL_MODELS",
    "_fallback_models",
    "_cli_profile_model_row",
    "_mono",
    "_cache_get",
    "_cache_put",
    "_cache_put_ttl",
    "served_fallback",
    "_set_fallback_served",
    "_http_get_json",
    "_row_with_effort",
    "_codex_cache_model_map",
    "_codex_cache_models",
    "_codex_cache_context_windows",
    "CLI_ENGINE_LABELS",
    "_CLI_ENGINE_PROVIDERS",
    "_CLI_ENGINE_AGENT_PROFILES",
    "_ACP_ENGINE_KINDS",
    "_engine_allowlist",
    "_codex_gate_lookup",
    "_engine_catalog_lookup",
    "validate_external_cli_model",
    "_claude_env_model",
    "_claude_settings_model",
    "_claude_profile_upstream_model",
    "external_cli_effective_default",
    "external_cli_model_refusal",
    "_codex_canonical_model_id",
    "_is_codex_model_id",
    "_codex_effective_default",
    "_codex_config_model",
    "codex_effective_default_model",
    "validate_codex_model",
    "codex_model_refusal",
]
