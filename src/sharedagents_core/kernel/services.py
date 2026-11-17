"""Unified inference-service config — PRIMARY + BACKUP, all in one place.

Self-hosted inference services (STT/ASR, TTS, embedding, reranking, …) are configured
in ONE place: the sharedagents global config (``~/.sharedagents/config.json``). Local
hosts/IPs are NEVER hardcoded in source and config is NOT scattered across env
vars (operator directives 2026-06-04). Each service has a PRIMARY and an optional
BACKUP endpoint; clients try primary first and fail over to backup.

Config shape (each service the same)::

    "stt": {
        "primary": {"base_url": "http://<host>:<port>/v1", "model": "qwen3-asr-1.7b"},
        "backup":  {"base_url": "https://openrouter.ai/api/v1",
                    "model": "qwen3-asr-1.7b", "api_key_env": "OPENROUTER_API_KEY"}
    },
    "tts": {
        "primary": {"base_url": "http://<host>:<port>/v1", "model": "voice-model"},
        "backup":  {"base_url": "https://openrouter.ai/api/v1",
                    "model": "openai/gpt-4o-mini-tts-2025-12-15",
                    "api_key_env": "OPENROUTER_API_KEY"}
    },
    "embedding": { "primary": {...}, "backup": {...} },
    "reranking": { "primary": {...}, "backup": {...} }

A flat ``{"base_url": ..., "model": ...}`` is also accepted and treated as the
primary (back-compat). ``api_key_env`` names an env var holding a bearer token
(``api_key`` may give the literal token instead); local services usually need none.

**Per-agent override.** Each agent may override the global service config with its
OWN primary+backup: pass the agent's effective config (an ``AgentConfig`` object or
a merged dict) as ``config=`` to :func:`resolve_service`. A service block present on
the agent fully REPLACES the global one (so an agent gets its own primary/backup);
when absent, the global block is used. Standalone callers (e.g. the bundled MCP
servers, which have no agent context) pass nothing and read the global config file.

This module reads the config file directly (stdlib only) so even the minimal,
standalone MCP servers can import it without pulling heavy deps.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

SERVICES = ("stt", "tts", "embedding", "reranking")


def _global_config() -> dict:
    home = os.environ.get("SHAREDAGENTS_HOME") or os.path.expanduser("~/.sharedagents")
    try:
        with open(os.path.join(home, "config.json"), encoding="utf-8") as fh:
            return json.load(fh) or {}
    except (OSError, ValueError):
        return {}


_RESERVED_KEYS = frozenset({"base_url", "model", "api_key", "api_key_env"})


@dataclass
class ServiceEndpoint:
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    label: str = ""  # "primary" | "backup"
    extra: dict = field(default_factory=dict)  # dims/organization/provider/…

    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    def url(self, path: str) -> str:
        return self.base_url.rstrip("/") + "/" + path.lstrip("/")

    def opt(self, key: str, default: Any = None) -> Any:
        """A non-core endpoint field (e.g. ``dims`` for embedding)."""
        return self.extra.get(key, default)


def _endpoint(raw: dict, label: str) -> ServiceEndpoint:
    if not isinstance(raw, dict):
        return ServiceEndpoint(label=label)
    key = ""
    if raw.get("api_key_env"):
        key = os.environ.get(str(raw["api_key_env"]), "")
    elif raw.get("api_key"):
        key = str(raw["api_key"])
    extra = {k: v for k, v in raw.items() if k not in _RESERVED_KEYS}
    return ServiceEndpoint(
        base_url=str(raw.get("base_url") or "").rstrip("/"),
        model=str(raw.get("model") or ""),
        api_key=key,
        label=label,
        extra=extra,
    )


@dataclass
class ServiceConfig:
    name: str
    primary: ServiceEndpoint = field(default_factory=ServiceEndpoint)
    backup: ServiceEndpoint = field(default_factory=ServiceEndpoint)
    endpoints: list[ServiceEndpoint] = field(default_factory=list)

    def candidates(self) -> list[ServiceEndpoint]:
        """Configured endpoints in failover order."""
        if self.endpoints:
            return [e for e in self.endpoints if e.configured]
        return [e for e in (self.primary, self.backup) if e.configured]

    @property
    def configured(self) -> bool:
        return bool(self.candidates())

    def model_mismatch(self) -> str:
        """For services where the BACKUP must produce downstream-compatible
        output (embedding vectors, rerank scores), the backup
        MUST serve the EXACT SAME model id as the primary (operator directive
        2026-06-04). Returns a human-readable mismatch description, or "" when
        consistent / not-both-configured.
        """
        if self.name in ("embedding", "reranking") and self.primary.configured and self.backup.configured:
            pm, bm = self.primary.model, self.backup.model
            if pm and bm and pm != bm:
                return (
                    f"{self.name}: backup model {bm!r} != primary model "
                    f"{pm!r} — backup must use the SAME model for downstream "
                    f"compatibility"
                )
        return ""


def _service_block(name: str, config: Any) -> dict:
    """Pull the ``name`` service block out of ``config``.

    ``config`` may be ``None`` (read the global config file), a merged config
    ``dict`` (agent-effective config), or an ``AgentConfig``-like object exposing
    the block as an attribute. The first non-empty wins; this is how a per-agent
    block overrides the global one (the agent-effective dict already has the
    per-agent value merged over global).

    """
    if config is None:
        config = _global_config()
    if isinstance(config, dict):
        raw = config.get(name)
    else:  # AgentConfig-like object — duck-typed to avoid a circular import
        raw = getattr(config, name, None)
    return raw if isinstance(raw, (dict, list)) else {}


def resolve_service(name: str, config: Any = None) -> ServiceConfig:
    """Resolve a service's primary+backup endpoints or multi-hop chain from unified config.

    Pass ``config`` (an ``AgentConfig`` or merged dict) to honor a per-agent
    override; omit it to read the global ``~/.sharedagents/config.json``.

    """
    raw = _service_block(name, config)
    if isinstance(raw, list):
        endpoints = [
            _endpoint(item, (item.get("label") if isinstance(item, dict) else None) or f"endpoint_{i + 1}")
            for i, item in enumerate(raw)
        ]
        prim = endpoints[0] if endpoints else ServiceEndpoint(label="primary")
        back = endpoints[-1] if len(endpoints) > 1 else ServiceEndpoint(label="backup")
        return ServiceConfig(name, prim, back, endpoints=endpoints)

    if not isinstance(raw, dict):
        return ServiceConfig(name, _endpoint({}, "primary"), ServiceEndpoint(label="backup"))

    if "endpoints" in raw and isinstance(raw["endpoints"], list):
        endpoints = [
            _endpoint(item, (item.get("label") if isinstance(item, dict) else None) or f"endpoint_{i + 1}")
            for i, item in enumerate(raw["endpoints"])
        ]
        prim = endpoints[0] if endpoints else ServiceEndpoint(label="primary")
        back = endpoints[-1] if len(endpoints) > 1 else ServiceEndpoint(label="backup")
        return ServiceConfig(name, prim, back, endpoints=endpoints)

    endpoints = []
    if "primary" in raw and isinstance(raw["primary"], dict):
        endpoints.append(_endpoint(raw["primary"], raw["primary"].get("label") or "primary"))
    if "secondary" in raw and isinstance(raw["secondary"], dict):
        endpoints.append(_endpoint(raw["secondary"], raw["secondary"].get("label") or "secondary"))
    if "tertiary" in raw and isinstance(raw["tertiary"], dict):
        endpoints.append(_endpoint(raw["tertiary"], raw["tertiary"].get("label") or "tertiary"))
    if "fallbacks" in raw and isinstance(raw["fallbacks"], list):
        for i, fb in enumerate(raw["fallbacks"]):
            if isinstance(fb, dict):
                lbl = fb.get("label") or f"fallback_{i + 1}"
                endpoints.append(_endpoint(fb, lbl))
    if "backup" in raw and isinstance(raw["backup"], dict):
        endpoints.append(_endpoint(raw["backup"], raw["backup"].get("label") or "backup"))

    if endpoints:
        prim = endpoints[0]
        back = endpoints[-1] if len(endpoints) > 1 else ServiceEndpoint(label="backup")
        return ServiceConfig(name, prim, back, endpoints=endpoints)

    # flat shape → treat as primary
    prim = _endpoint(raw, "primary")
    return ServiceConfig(name, prim, ServiceEndpoint(label="backup"), endpoints=[prim] if prim.configured else [])


def with_failover(svc: ServiceConfig, attempt: Callable[[ServiceEndpoint], Any]) -> Any:
    """Run ``attempt(endpoint)`` over primary then backup; return the first
    success, or raise with all errors if none succeed / none configured.
    """
    cands = svc.candidates()
    if not cands:
        raise RuntimeError(
            f'{svc.name} not configured: add a "{svc.name}" block (primary/backup) to ~/.sharedagents/config.json'
        )
    errors: list[str] = []
    for ep in cands:
        try:
            return attempt(ep)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{ep.label}({ep.base_url}): {e}")
    raise RuntimeError(f"all {svc.name} endpoints failed — " + " | ".join(errors))


__all__ = [
    "SERVICES",
    "ServiceEndpoint",
    "ServiceConfig",
    "resolve_service",
    "with_failover",
]
