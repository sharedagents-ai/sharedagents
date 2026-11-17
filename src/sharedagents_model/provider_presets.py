"""Built-in LLM provider presets + the free default template row.

Two things live here:

1. :data:`PROVIDER_PRESETS` — a curated list of well-known,
   **OpenAI-compatible** providers the first-run wizard offers as
   one-pick presets (so a newcomer picks "DeepSeek" instead of
   memorising ``https://api.deepseek.com/v1`` + the right env var).

2. The **free default template row** (NVIDIA Nemotron via OpenRouter),
   kept on every install (never silently dropped) and keyed by the
   user's ``OPENROUTER_API_KEY``. It chats the instant that key is set;
   there is **no bundled key**: a repo-committed shared key violates
   OpenRouter's terms and is revoked upstream. A keyless, GPU-less install is
   handled by the guided-setup path, not a bundled key.

Anthropic is intentionally NOT a preset: sharedagents speaks the OpenAI
``/v1/chat/completions`` shape, and Anthropic's native Messages API is
different. Use Claude models through the OpenRouter preset instead.

(Bundled key removed.)
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse


@dataclass(frozen=True)
class ProviderPreset:
    """One OpenAI-compatible provider the wizard can pre-fill."""

    key: str  # short id, e.g. "deepseek"
    label: str  # display name, e.g. "DeepSeek"
    base_url: str  # OpenAI-compatible base URL (ends in /v1 or /openai)
    api_key_env: str  # conventional env var for the key ("" = local/no key)
    note: str = ""  # one-line hint shown in the picker


# Order matters — this is the wizard's menu order. OpenRouter first
# (it fronts every other provider incl. free models), then the big
# direct providers, then local servers.
PROVIDER_PRESETS: list[ProviderPreset] = [
    ProviderPreset(
        "openrouter",
        "OpenRouter",
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        "1000s of models incl. free ones (Claude, GPT, Llama…)",
    ),
    ProviderPreset(
        "openai",
        "OpenAI",
        "https://api.openai.com/v1",
        "OPENAI_API_KEY",
    ),
    ProviderPreset(
        "deepseek",
        "DeepSeek",
        "https://api.deepseek.com/v1",
        "DEEPSEEK_API_KEY",
    ),
    ProviderPreset(
        "xai",
        "xAI (Grok)",
        "https://api.x.ai/v1",
        "XAI_API_KEY",
    ),
    ProviderPreset(
        "qwen",
        "Qwen (DashScope)",
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "DASHSCOPE_API_KEY",
    ),
    ProviderPreset(
        "groq",
        "Groq",
        "https://api.groq.com/openai/v1",
        "GROQ_API_KEY",
    ),
    ProviderPreset(
        "mistral",
        "Mistral",
        "https://api.mistral.ai/v1",
        "MISTRAL_API_KEY",
    ),
    ProviderPreset(
        "gemini",
        "Google Gemini",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY",
        "OpenAI-compat endpoint",
    ),
    ProviderPreset(
        "cerebras",
        "Cerebras",
        "https://api.cerebras.ai/v1",
        "CEREBRAS_API_KEY",
    ),
    ProviderPreset(
        "together",
        "Together",
        "https://api.together.xyz/v1",
        "TOGETHER_API_KEY",
    ),
    ProviderPreset(
        "moonshot",
        "Moonshot (Kimi)",
        "https://api.moonshot.ai/v1",
        "MOONSHOT_API_KEY",
        "2.8T MoE, 1M context — kimi-k3",
    ),
    ProviderPreset(
        "vllm",
        "Local vLLM",
        "http://127.0.0.1:8000/v1",
        "",
        "self-hosted",
    ),
    ProviderPreset(
        "sglang",
        "Local SGLang",
        "http://127.0.0.1:30000/v1",
        "",
        "self-hosted",
    ),
    ProviderPreset(
        "ollama",
        "Local Ollama",
        "http://127.0.0.1:11434/v1",
        "",
        "self-hosted",
    ),
    ProviderPreset(
        "lmstudio",
        "Local LM Studio",
        "http://127.0.0.1:1234/v1",
        "",
        "self-hosted",
    ),
]


def preset_by_key(key: str) -> ProviderPreset | None:
    key = (key or "").strip().lower()
    for p in PROVIDER_PRESETS:
        if p.key == key:
            return p
    return None


# --------------------------------------------------------------------------
# Bundled free default — the zero-config quickstart
# --------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
DEFAULT_MODEL_ALIAS = "nemotron-free"
DEFAULT_API_KEY_ENV = "OPENROUTER_API_KEY"
DEFAULT_CONTEXT_WINDOW = 1_000_000  # OpenRouter model metadata (2026-05)

# NOTE: there is no bundled shared OpenRouter key. A shared key committed to
# a public repo violates OpenRouter's terms and is auto-revoked by
# secret-scanning or drained by abuse. The Nemotron row below is now a *keyed template*: it chats the
# instant OPENROUTER_API_KEY is set. The keyless GPU-less case is handled by
# the guided-setup path, not a bundled key.
#   Removed: BUNDLED_OPENROUTER_KEY, is_openrouter(), bundled_key_for_row(),
#   and the ``useBundledKey`` row field.


# --------------------------------------------------------------------------
# Compatibility wrapper for the TUI and legacy callers.
def resolve_temperature(model: str, cfg_temperature: float | None = None) -> float | None:
    """Return configured temperature, respecting strict provider requirements."""
    from .sampling import configured_sampling

    return configured_sampling({"temperature": cfg_temperature}, model_id=model).get("temperature")


# Native-agent xAI Grok catalogue (direct api.x.ai, not OpenRouter).
# Keyed by XAI_API_KEY — same pattern as the free Nemotron template — and
# offered ONLY where that key actually resolves: a deployment that runs on
# another provider alone (AWS Bedrock, a local vLLM, …) must never have xAI
# forced into its catalogue.
XAI_BASE_URL = "https://api.x.ai/v1"
XAI_API_KEY_ENV = "XAI_API_KEY"
XAI_HOST = "api.x.ai"
# Header names a credential may ride in as a literal value — mirrors
# ``sharedagents_model.models._AUTH_HEADER_NAMES``, the set ``row_is_hosted()``
# already treats as "this row authenticates". Keep the two in sync.
_AUTH_HEADER_NAMES = frozenset({"authorization", "x-api-key", "api-key"})
_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
XAI_GROK_MODELS: tuple[dict, ...] = (
    {
        "alias": "grok-46",
        "baseUrl": XAI_BASE_URL,
        "id": "grok-4.6",
        "backend": "openai",
        "provider": "xAI",
        "apiKeyEnv": XAI_API_KEY_ENV,
        "contextWindow": 256000,
        "maxTokens": 0,
        "capabilities": ["vision"],
    },
    {
        "alias": "grok-45",
        "baseUrl": XAI_BASE_URL,
        "id": "grok-4.5",
        "backend": "openai",
        "provider": "xAI",
        "apiKeyEnv": XAI_API_KEY_ENV,
        "contextWindow": 256000,
        "maxTokens": 0,
        "capabilities": ["vision"],
    },
    {
        "alias": "grok-43",
        "baseUrl": XAI_BASE_URL,
        "id": "grok-4.3",
        "backend": "openai",
        "provider": "xAI",
        "apiKeyEnv": XAI_API_KEY_ENV,
        "contextWindow": 131072,
        "maxTokens": 0,
        "capabilities": ["vision"],
    },
)


def _key_resolves(name: str, api_keys: Mapping[str, Any] | None) -> bool:
    """True when *name* resolves a non-empty credential on this system.

    Same precedence ``AgentConfig.upstream_headers()`` applies per row: the
    per-agent + shared ``secrets.env`` map first, process env as the fallback.
    An EMPTY map value is the deliberate "block the env fallback" sentinel, so
    it reads as *not* resolving — matching ``audit_api_keys()``'s
    ``agent-local-empty`` status.
    """
    keys = api_keys or {}
    if name in keys:
        return bool(str(keys.get(name) or "").strip())
    return bool(str(os.environ.get(name) or "").strip())


def xai_key_configured(api_keys: Mapping[str, Any] | None = None) -> bool:
    """True when the xAI provider really resolves a key on this system.

    Resolution mirrors the request-time Authorization path
    (``AgentConfig.upstream_headers`` / ``audit_api_keys``):

      1. ``api_keys`` — the per-agent + shared ``secrets.env`` map. An entry
         wins over the process env, and an EMPTY value is the deliberate
         "block the env fallback" sentinel those callers already honour, so it
         reads as *not configured*.
      2. otherwise ``XAI_API_KEY`` from the process env.

    Used to keep the shipped direct-xAI Grok rows out of deployments that run
    on a different provider alone.
    """
    return _key_resolves(XAI_API_KEY_ENV, api_keys)


def _row_endpoint(row: Mapping[str, Any]) -> str:
    """The row's endpoint — canonical ``baseUrl`` or legacy ``endpoint``."""
    return str(row.get("baseUrl") or row.get("endpoint") or "").strip()


def _row_targets_xai(row: Mapping[str, Any]) -> bool:
    """True when the row's endpoint is xAI's own API host.

    Host EQUALITY, never a substring: a look-alike host (``api.x.ai.evil``) or
    a proxied/aggregator endpoint is a different provider and stays untouched.
    """
    try:
        return (urlparse(_row_endpoint(row)).hostname or "").lower() == XAI_HOST
    except ValueError:
        return False


def _row_credential_resolves(
    row: Mapping[str, Any],
    api_keys: Mapping[str, Any] | None,
) -> bool:
    """True when the row itself can authenticate on this system.

    Covers the three ways a row can carry a credential: its ``apiKeyEnv``
    bearer, a ``${VAR}`` reference inside ``headers``, or a literal value in
    an auth header. Any ONE resolving is enough to keep the row — such a row
    is functional even though ``XAI_API_KEY`` is not what feeds it.
    """
    env_name = str(row.get("apiKeyEnv") or row.get("api_key_env") or "").strip()
    if env_name and _key_resolves(env_name, api_keys):
        return True
    headers = row.get("headers")
    if not isinstance(headers, Mapping):
        return False
    for name, value in headers.items():
        if not isinstance(value, str) or not value.strip():
            continue
        if any(_key_resolves(var, api_keys) for var in _ENV_VAR_RE.findall(value)):
            return True
        if str(name).strip().lower() in _AUTH_HEADER_NAMES and "${" not in value:
            return True  # literal credential baked into the row
    return False


def ensure_xai_grok_models(
    models: list,
    *,
    api_keys: Mapping[str, Any] | None = None,
) -> list:
    """Reconcile the shipped xAI Grok rows with this system's xAI availability.

    Native local_llm agents share ``models[]``. Direct xAI rows make Grok
    selectable without going through OpenRouter, but ONLY where the xAI
    provider actually resolves a key (see :func:`xai_key_configured`):

      * key resolves — append every missing shipped row (alias- and
        id-unique); an existing alias wins so an operator rename is never
        clobbered.
      * no key — append nothing AND drop every ``api.x.ai`` row this system
        cannot authenticate: the shipped presets a previous sharedagents forced
        into the catalogue (its setup writes / Models-UI saves persisted
        them) plus operator-customised ones (renamed alias, extra keys) that
        carry no resolving credential of their own. A Bedrock-only — or any
        other single-provider — deployment therefore ends up with no xAI rows
        at all, and gets the shipped rows straight back once a key is
        configured. Rows whose own credential still resolves (an alternative
        env var, a ``${VAR}`` header, a literal auth header) and rows
        pointing anywhere other than api.x.ai are never touched.

    This keeps a Bedrock-only deployment from being forced onto xAI.
    """
    rows = list(models or [])
    if not xai_key_configured(api_keys):
        return [
            row
            for row in rows
            if not (isinstance(row, dict) and _row_targets_xai(row) and not _row_credential_resolves(row, api_keys))
        ]
    seen_alias = {str(m.get("alias") or "") for m in rows if isinstance(m, dict) and m.get("alias")}
    seen_xai_ids: set[str] = {str(m.get("id") or "") for m in rows if isinstance(m, dict) and _row_targets_xai(m)}
    for row in XAI_GROK_MODELS:
        alias = str(row["alias"])
        mid = str(row["id"])
        if alias in seen_alias or mid in seen_xai_ids:
            continue
        rows.append(dict(row))
        seen_alias.add(alias)
        seen_xai_ids.add(mid)
    return rows


def default_model_row() -> dict:
    """The ``models[]`` row for the free Nemotron default (keyed template).

    Present on every install (never silently dropped) but keyed by the user's
    ``OPENROUTER_API_KEY`` — there is no bundled key. Chats immediately once a
    key is set; otherwise the guided-setup path takes over.
    """
    return {
        "alias": DEFAULT_MODEL_ALIAS,
        "baseUrl": DEFAULT_BASE_URL,
        "id": DEFAULT_MODEL,
        "backend": "unknown",
        "contextWindow": DEFAULT_CONTEXT_WINDOW,
        "maxTokens": 0,
        "apiKeyEnv": DEFAULT_API_KEY_ENV,
        "headers": {"X-Title": "sharedagents"},
    }
