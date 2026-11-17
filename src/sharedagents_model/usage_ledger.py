"""Authoritative per-turn cost ledger.

One row per turn, keyed ``(session_id, turn_id)``. Identical re-finalization
is an idempotent no-op. Live per-round checkpoints and a later write with a
strictly larger billed token SUM upgrade the existing row in place
(last-round occupancy must not permanently undercount the registry). An
explicit provider total may replace an interim token-priced estimate at the
same token count. Spend is derived by SUM, not by mutating counters:

  * goal spend     = SUM(cost_pico) WHERE ancestor_goal_id = ?
  * session spend  = SUM(cost_pico) WHERE root_session_id = ?

A child (delegated) turn writes its OWN row carrying the threaded
``root_session_id`` + ``ancestor_goal_id``, so its cost rolls up to the root
session and ancestor goal automatically. A parent never charges a child's cost.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from sharedagents_core.layout import usage_ledger_db_path
from sharedagents_model.usage_pricing import (
    AUTO_PRICING_FILENAME,
    CanonicalUsage,
    CostResult,
    PricingTable,
    normalize_usage,
    price,
)

_LOG = logging.getLogger("sharedagents.usage_ledger")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS turn_costs (
    session_id           TEXT NOT NULL,
    turn_id              TEXT NOT NULL,
    root_session_id      TEXT,
    ancestor_goal_id     TEXT,
    agent_name           TEXT,
    model                TEXT,
    model_alias          TEXT,
    provider             TEXT,
    input_tokens         INTEGER NOT NULL DEFAULT 0,
    cached_read_tokens   INTEGER NOT NULL DEFAULT 0,
    cached_write_tokens  INTEGER NOT NULL DEFAULT 0,
    thinking_tokens      INTEGER NOT NULL DEFAULT 0,
    output_tokens        INTEGER NOT NULL DEFAULT 0,
    cost_pico            INTEGER NOT NULL DEFAULT 0,
    cost_estimated       INTEGER NOT NULL DEFAULT 0,
    created_at           TEXT NOT NULL,
    PRIMARY KEY (session_id, turn_id)
);
CREATE INDEX IF NOT EXISTS idx_tc_goal ON turn_costs(ancestor_goal_id);
CREATE INDEX IF NOT EXISTS idx_tc_root ON turn_costs(root_session_id);
CREATE TABLE IF NOT EXISTS cost_accumulators (
    scope_key             TEXT PRIMARY KEY,
    exact_pico            TEXT NOT NULL,
    charged_pico          INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS ai_generations (
    generation_id        TEXT PRIMARY KEY,
    session_id           TEXT NOT NULL,
    turn_id              TEXT NOT NULL,
    turn_idx             INTEGER NOT NULL DEFAULT 0,
    round_idx            INTEGER NOT NULL DEFAULT 0,
    agent_name           TEXT NOT NULL DEFAULT '',
    model                TEXT NOT NULL DEFAULT '',
    model_alias          TEXT NOT NULL DEFAULT '',
    provider             TEXT NOT NULL DEFAULT '',
    engine               TEXT NOT NULL DEFAULT '',
    input_tokens         INTEGER NOT NULL DEFAULT 0,
    cached_tokens        INTEGER NOT NULL DEFAULT 0,
    fresh_tokens         INTEGER NOT NULL DEFAULT 0,
    output_tokens        INTEGER NOT NULL DEFAULT 0,
    thinking_tokens      INTEGER NOT NULL DEFAULT 0,
    content_tokens       INTEGER NOT NULL DEFAULT 0,
    total_tokens         INTEGER NOT NULL DEFAULT 0,
    context_window       INTEGER NOT NULL DEFAULT 0,
    ttft_ms              REAL NOT NULL DEFAULT 0,
    ttfc_ms              REAL NOT NULL DEFAULT 0,
    prefill_ms           REAL NOT NULL DEFAULT 0,
    prefill_tps          REAL NOT NULL DEFAULT 0,
    gen_tps              REAL NOT NULL DEFAULT 0,
    wt_ms                REAL NOT NULL DEFAULT 0,
    classify_ms          REAL NOT NULL DEFAULT 0,
    plan_ms              REAL NOT NULL DEFAULT 0,
    tool_events          INTEGER NOT NULL DEFAULT 0,
    cost_usd             REAL NOT NULL DEFAULT 0,
    tokens_estimated     INTEGER NOT NULL DEFAULT 0,
    finish_reason        TEXT NOT NULL DEFAULT '',
    created_at           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_gen_created ON ai_generations(created_at);
CREATE INDEX IF NOT EXISTS idx_gen_session ON ai_generations(session_id);
CREATE INDEX IF NOT EXISTS idx_gen_provider ON ai_generations(provider);
CREATE INDEX IF NOT EXISTS idx_gen_agent ON ai_generations(agent_name);
"""


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utcnow() -> datetime:
    """Aware UTC now — the turn clock for peak/off-peak pricing."""
    return datetime.now(timezone.utc)


_PROVIDER_PLACEHOLDERS = frozenset({"", "unknown", "null", "none", "-", "—"})
_PROVIDER_ALIASES = {
    "vllm": "local-vllm",
    "local vllm": "local-vllm",
    "local-llm": "local-vllm",
    "local_llm": "local-vllm",
    "openai-compat": "local-vllm",
    "configured": "local-vllm",
    "x-ai": "xai",
}
_ENGINE_PROVIDERS = {
    "local_llm": "local-vllm",
    "vllm": "local-vllm",
    "cursor_acp": "cursor",
    "codex_cli": "codex",
    "claude_cli": "anthropic",
    "opencode_acp": "opencode",
    "hermes_acp": "hermes",
    "kimi_acp": "kimi",
    "devin_acp": "devin",
    "antigravity": "antigravity",
}
_AGENT_PROVIDERS = {
    "claude": "anthropic",
    "cursor": "cursor",
    "codex": "codex",
    "kimi": "kimi",
    "hermes": "hermes",
    "devin": "devin",
    "opencode": "opencode",
    "antigravity": "antigravity",
}
_SLASH_VENDORS = frozenset(
    {
        "openrouter",
        "google",
        "x-ai",
        "z-ai",
        "qwen",
        "anthropic",
        "openai",
        "deepseek",
        "moonshotai",
        "moonshot-ai",
        "meta",
        "xiaomi",
        "minimax",
        "nvidia",
        "mistralai",
        "thinkingmachines",
    }
)
_WEAK_PROVIDERS = frozenset({"", "local-vllm", "vllm"})
_DIRECT_DEEPSEEK_ALIASES = frozenset({"dpsk", "dpsk-pro", "deepseek"})
_KNOWN_ALIAS_TO_ID = {
    "dpsk": "deepseek-v4-flash",
    "dpsk-pro": "deepseek-v4-pro",
    "deepseek": "deepseek-v4-flash",
    "local": "qwen3.8-27b-fp8",
    "qwen38-fp8": "qwen3.8-27b-fp8",
    "qwen38-int4": "qwen3.8-27b-int4",
    "qwen38-27b": "qwen3.8-27b",
    "int4": "qwen3.6-27b-awq-bf16-int4",
}
_config_provider_map: dict[str, str] | None = None
_config_alias_map: dict[str, str] | None = None


def _engine_model_labels() -> frozenset[str]:
    """Return labels reserved for engine/provider plumbing.

    Engine labels are metadata, never wire model ids.  Keep this derived from
    the live engine maps so adding an engine cannot reintroduce a literal label
    into the ledger.  The delegate shim map is imported lazily to avoid the
    agent/model import cycle during startup.
    """
    labels = {str(value).strip().lower() for value in _ENGINE_PROVIDERS if str(value).strip()}
    try:
        from sharedagents_agent.agents.delegate_shims import _CLI_ENGINE_LABELS

        labels.update(str(value).strip().lower() for value in _CLI_ENGINE_LABELS.values() if str(value).strip())
        labels.update(str(key).strip().lower() for key in _CLI_ENGINE_LABELS if str(key).strip())
        labels.update(str(key).strip().lower().replace("_", "-") for key in _CLI_ENGINE_LABELS if str(key).strip())
    except Exception:  # noqa: BLE001
        # The model ledger is also used in small standalone tools where the
        # agent package is intentionally unavailable.  The local map still
        # protects every engine known to this module in that case.
        pass
    return frozenset(labels)


def _normalize_provider(value: Any) -> str:
    key = str(value or "").strip().lower().replace("_", "-")
    if key in _PROVIDER_PLACEHOLDERS:
        return ""
    return _PROVIDER_ALIASES.get(key, key)


def _provider_map_from_config() -> dict[str, str]:
    global _config_provider_map
    if _config_provider_map is not None:
        return _config_provider_map
    mapping: dict[str, str] = {}
    try:
        from sharedagents_core.kernel.config import load_global_config

        models = load_global_config().get("models") or []
    except Exception:  # noqa: BLE001
        models = []
    if isinstance(models, list):
        for raw in models:
            if not isinstance(raw, dict):
                continue
            backend = str(raw.get("backend") or "").strip().lower()
            resolved = _normalize_provider(raw.get("provider"))
            if not resolved:
                resolved = "local-vllm" if backend in {"vllm", "local-vllm", ""} else ""
            if not resolved:
                continue
            for key in (raw.get("id"), raw.get("alias")):
                name = str(key or "").strip()
                if name:
                    mapping[name] = resolved
    _config_provider_map = mapping
    return mapping


def _is_direct_deepseek(model: Any, model_alias: Any = "") -> bool:
    """True for the DeepSeek API itself, not an OpenRouter ``deepseek/…`` id."""
    for name in (model, model_alias):
        key = str(name or "").strip().lower()
        if not key or "/" in key:
            continue
        if key in _DIRECT_DEEPSEEK_ALIASES or key.startswith("deepseek-"):
            return True
    return False


_DIRECT_CODEX_MODELS = frozenset(
    {
        "gpt-5.6-sol",
        "gpt-5.6-sol-wm",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.3-codex-spark",
        "gpt-6-astra",
        "codex-auto-review",
        "codex",
    }
)


def _is_direct_codex(model: Any, model_alias: Any = "") -> bool:
    """True for the Codex backend/models, not an OpenRouter ``openai/…`` id."""
    for name in (model, model_alias):
        key = str(name or "").strip().lower()
        if not key:
            continue
        if key.startswith(("codex:", "codex/")):
            return True
        if "/" in key:
            continue
        if key in _DIRECT_CODEX_MODELS or key.startswith("codex-") or "-codex-" in key:
            return True
    return False


def _is_bare_grok(model: Any, model_alias: Any = "") -> bool:
    """True for unsashed ``grok*`` ids, not OpenRouter ``x-ai/grok-*``."""
    for name in (model, model_alias):
        key = str(name or "").strip().lower()
        if not key or "/" in key:
            continue
        if key.startswith("grok"):
            return True
    return False


def _stale_native_cursor_grok(
    *,
    provider: Any,
    engine: Any,
    model: Any,
    model_alias: Any,
    agent_name: Any,
) -> bool:
    """True when an older grok→cursor heuristic stamped Cursor on a native turn.

    Cursor ACP selects ``cursor``; the native worker selects the catalog
    row (xAI, DeepSeek, …). This predicate is only for repairing stored
    rows — live attribution trusts the worker-stamped provider.
    """
    if _normalize_provider(provider) != "cursor":
        return False
    if not _is_bare_grok(model, model_alias):
        return False
    engine_key = str(engine or "").strip().lower()
    if engine_key in {"cursor_acp", "cursor"}:
        return False
    agent = str(agent_name or "").strip().lower()
    return agent != "cursor"


def resolve_generation_provider(
    *,
    provider: Any = "",
    engine: Any = "",
    model: Any = "",
    model_alias: Any = "",
    agent_name: Any = "",
) -> str:
    """Return a concrete provider label. Never empty or ``unknown``.

    The worker-selected catalog ``provider`` (stamped on round metrics) is
    authoritative. Model-name maps run only when that stamp is missing.
    """
    if _is_direct_deepseek(model, model_alias):
        return "deepseek"
    if _is_direct_codex(model, model_alias):
        return "codex"
    explicit = _normalize_provider(provider)
    if explicit and explicit not in _WEAK_PROVIDERS:
        return explicit
    engine_key = str(engine or "").strip().lower()
    if engine_key in _ENGINE_PROVIDERS and engine_key not in {"local_llm", "vllm"}:
        return _ENGINE_PROVIDERS[engine_key]
    cfg = _provider_map_from_config()
    for name in (model, model_alias):
        key = str(name or "").strip()
        if key and key in cfg:
            return cfg[key]
    model_id = str(model or model_alias or "").strip()
    if "/" in model_id:
        vendor = model_id.split("/", 1)[0].strip().lower()
        if vendor == "openrouter":
            return "openrouter"
        if vendor in _SLASH_VENDORS:
            return "openrouter"
    lower = model_id.lower()
    if lower.startswith(("qwen3.", "qwen3-", "qwen38")):
        return "local-vllm"
    if lower.startswith(("gpt-5", "gpt-4", "o3", "o4")):
        agent = str(agent_name or "").strip().lower()
        if agent == "cursor":
            return "cursor"
        if agent in {"codex", "openai"}:
            return "codex" if agent == "codex" else "openai"
        return "openrouter"
    if lower.startswith("claude"):
        return "anthropic"
    if lower.startswith("composer"):
        return "cursor"
    agent = str(agent_name or "").strip().lower()
    if agent in _AGENT_PROVIDERS:
        return _AGENT_PROVIDERS[agent]
    if _is_bare_grok(model, model_alias) or lower.startswith("grok"):
        return "xai"
    if engine_key:
        return _ENGINE_PROVIDERS.get(engine_key) or _normalize_provider(engine_key) or "local-vllm"
    return "local-vllm"


_LABEL_PLACEHOLDERS = frozenset(
    {
        "",
        "unknown",
        "null",
        "none",
        "-",
        "—",
        "n/a",
        "na",
        "undefined",
    }
)
_PROVIDER_AGENTS = {
    "cursor": "cursor",
    "anthropic": "claude",
    "openai": "codex",
    "codex": "codex",
    "opencode": "opencode",
    "kimi": "kimi",
    "hermes": "hermes",
    "devin": "devin",
    "antigravity": "antigravity",
    "deepseek": "jarvis",
    "openrouter": "jarvis",
    "local-vllm": "jarvis",
    "xai": "jarvis",
    "google": "antigravity",
}
_PROVIDER_MODELS = {
    "cursor": "Auto",
    "anthropic": "claude",
    "openai": "gpt",
    "codex": "gpt-5.6-sol",
    "opencode": "opencode",
    "kimi": "kimi",
    "hermes": "hermes",
    "devin": "devin",
    "antigravity": "gemini",
    "deepseek": "deepseek",
    "openrouter": "openrouter",
    "local-vllm": "local-vllm",
    "xai": "grok-4.6",
    "google": "gemini",
}


def _clean_label(value: Any) -> str:
    key = str(value or "").strip()
    if not key or key.lower().replace("_", "-") in _LABEL_PLACEHOLDERS:
        return ""
    return key


def resolve_generation_agent(
    *,
    agent_name: Any = "",
    provider: Any = "",
    engine: Any = "",
    model: Any = "",
    model_alias: Any = "",
) -> str:
    """Return a concrete agent label. Never empty."""
    explicit = _clean_label(agent_name)
    if explicit:
        return "jarvis" if explicit.lower() == "main" else explicit
    prov = resolve_generation_provider(
        provider=provider,
        engine=engine,
        model=model,
        model_alias=model_alias,
        agent_name=agent_name,
    )
    return _PROVIDER_AGENTS.get(prov, "jarvis")


def _alias_to_model_id() -> dict[str, str]:
    """Map display aliases to the wire model id that the ledger must store."""
    global _config_alias_map
    if _config_alias_map is not None:
        return _config_alias_map
    mapping = dict(_KNOWN_ALIAS_TO_ID)
    try:
        from sharedagents_core.kernel.config import load_global_config

        models = load_global_config().get("models") or []
    except Exception:  # noqa: BLE001
        models = []
    if isinstance(models, list):
        for raw in models:
            if not isinstance(raw, dict):
                continue
            alias = str(raw.get("alias") or "").strip()
            model_id = str(raw.get("id") or "").strip()
            if alias and model_id and alias != model_id:
                mapping[alias] = model_id
    try:
        from sharedagents_model.usage_pricing import PricingTable

        for entry in PricingTable.load().entries:
            alias = str(entry.match.get("alias") or "").strip()
            model_id = str(entry.match.get("id") or "").strip()
            if alias and model_id and alias != model_id:
                mapping[alias] = model_id
    except Exception:  # noqa: BLE001
        pass
    _config_alias_map = mapping
    return mapping


def resolve_generation_model(
    *,
    model: Any = "",
    model_alias: Any = "",
    provider: Any = "",
    engine: Any = "",
    agent_name: Any = "",
) -> str:
    """Return the full wire model id. Aliases are display-only."""
    wire = _clean_label(model)
    alias = _clean_label(model_alias)
    # Engine labels are metadata, never wire model ids. Prefer a real model
    # carried as the alias, otherwise use the provider fallback below. This is
    # deliberately derived from the engine maps rather than hardcoding one
    # label, so Codex, Antigravity, and future CLI engines are covered.
    engine_labels = _engine_model_labels()
    if wire.lower() in engine_labels:
        wire = alias if alias.lower() not in engine_labels else ""
    if alias.lower() in engine_labels:
        alias = ""
    aliases = _alias_to_model_id()
    if wire and wire not in aliases:
        return wire
    for candidate in (alias, wire):
        expanded = aliases.get(candidate, "")
        if expanded:
            return expanded
    if wire:
        return wire
    prov = resolve_generation_provider(
        provider=provider,
        engine=engine,
        model=model,
        model_alias=model_alias,
        agent_name=agent_name,
    )
    fallback = _PROVIDER_MODELS.get(prov, prov or "local-vllm")
    resolved_fallback = aliases.get(fallback, fallback)
    return "unknown" if resolved_fallback.lower() in engine_labels else resolved_fallback


def persist_model_alias(
    *,
    model: Any = "",
    model_alias: Any = "",
    resolved: str = "",
) -> str:
    """Keep the short display name when the caller only sent an alias."""
    aliases = _alias_to_model_id()
    incoming_alias = _clean_label(model_alias)
    incoming = _clean_label(model)
    if incoming_alias and incoming_alias != resolved:
        return incoming_alias
    if incoming and incoming in aliases:
        return incoming
    return incoming_alias or resolved


def generation_id(session_id: str, turn_id: str, round_idx: int) -> str:
    """Stable generation key: sha256(``{session_id}:{turn_id}:{round_idx}``)."""
    return hashlib.sha256(f"{session_id}:{turn_id}:{int(round_idx)}".encode()).hexdigest()


def ledger_turn_id(session_id: str, turn_idx: int) -> str:
    """Stable ledger key: sha256(``{session_id}:{turn_idx}``).

    JSONL ``physical_turn_id`` stays a per-attempt UUID for streaming. The
    ledger row must survive re-finalization of the same session turn, so it
    keys off the durable index rather than that UUID.
    """
    return hashlib.sha256(f"{session_id}:{int(turn_idx)}".encode()).hexdigest()


def session_output_looks_cumulative(outputs: list[int]) -> bool:
    """True when a session stored Codex-style running totals instead of deltas.

    Codex ``turn.completed`` usage is thread-cumulative (openai/codex#17539).
    Older writes stored that snapshot on every turn, so SUM(output_tokens)
    counted the same generated text many times.
    """
    if len(outputs) < 3:
        return False
    mx = max(outputs)
    total = sum(outputs)
    if mx < 50_000 or total <= mx * 18 // 10:
        return False
    grows = 0
    pairs = 0
    for prev, cur in zip(outputs, outputs[1:]):
        pairs += 1
        if cur >= prev and cur >= 10_000:
            grows += 1
    return grows >= max(2, (pairs + 2) // 5)


def deflate_session_outputs(outputs: list[int]) -> list[int]:
    """Convert a cumulative session series into per-turn deltas."""
    values = [max(0, int(value or 0)) for value in outputs]
    if not session_output_looks_cumulative(values):
        return values
    prev = 0
    deltas: list[int] = []
    for cur in values:
        delta = cur if prev > 0 and cur < (prev + 1) // 2 else max(0, cur - prev)
        deltas.append(delta)
        prev = cur
    return deltas


class UsageLedger:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path or usage_ledger_db_path())
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.db_path), timeout=30.0)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA busy_timeout=15000")
        return con

    def _ensure_schema(self) -> None:
        with self._connect() as con:
            con.executescript(SCHEMA_SQL)
            con.commit()

    def record(
        self,
        *,
        session_id: str,
        turn_id: str,
        cost: CostResult,
        usage: CanonicalUsage,
        root_session_id: str | None = None,
        ancestor_goal_id: str | None = None,
        agent_name: str = "",
        return_allocated_pico: bool = False,
    ) -> bool | int:
        """Insert one turn's cost row.

        By default returns True if newly inserted (or upgraded from an
        undercounted row) and False for an identical existing
        ``(session_id, turn_id)``. With ``return_allocated_pico=True``, returns
        the integer pico-USD allocated to the row (including the existing
        value on a duplicate) so caller-visible metrics stay identical to the
        authoritative ledger.
        """
        if not session_id or not turn_id:
            return False
        provider = resolve_generation_provider(
            provider=usage.provider,
            model=usage.model,
            model_alias=usage.model_alias,
            agent_name=agent_name,
        )
        model = resolve_generation_model(
            model=usage.model,
            model_alias=usage.model_alias,
            provider=usage.provider,
            agent_name=agent_name,
        )
        alias = persist_model_alias(
            model=usage.model,
            model_alias=usage.model_alias,
            resolved=model,
        )
        agent = resolve_generation_agent(
            agent_name=agent_name,
            provider=usage.provider,
            model=usage.model,
            model_alias=usage.model_alias,
        )
        with self._connect() as con:
            # Serialize the idempotency check, exact-cost carry, turn insert,
            # and accumulator update. An identical duplicate must alter
            # neither the row nor the fractional carry.
            con.execute("BEGIN IMMEDIATE")
            exists = con.execute(
                "SELECT cost_pico, input_tokens, cached_read_tokens, "
                "cached_write_tokens, thinking_tokens, output_tokens, "
                "cost_estimated, model, model_alias, provider, agent_name "
                "FROM turn_costs WHERE session_id=? AND turn_id=?",
                (session_id, turn_id),
            ).fetchone()
            if exists is None and usage.output >= 50_000:
                prior = con.execute(
                    "SELECT output_tokens, input_tokens, cached_read_tokens "
                    "FROM turn_costs WHERE session_id=? AND turn_id!=? "
                    "ORDER BY created_at DESC LIMIT 1",
                    (session_id, turn_id),
                ).fetchone()
                if prior is not None:
                    prev_out = int(prior["output_tokens"] or 0)
                    prev_in = int(prior["input_tokens"] or 0)
                    prev_cr = int(prior["cached_read_tokens"] or 0)
                    if prev_out > 0 and usage.output >= prev_out and usage.cached_read >= max(1_000_000, prev_cr):
                        usage = CanonicalUsage(
                            input=max(0, usage.input - prev_in),
                            cached_read=max(0, usage.cached_read - prev_cr),
                            cached_write=usage.cached_write,
                            thinking=usage.thinking,
                            output=usage.output - prev_out,
                            model=usage.model,
                            model_alias=usage.model_alias,
                            provider=usage.provider,
                            estimated=usage.estimated,
                        )
            if exists is not None:
                stored_prompt = (
                    int(exists["input_tokens"] or 0)
                    + int(exists["cached_read_tokens"] or 0)
                    + int(exists["cached_write_tokens"] or 0)
                )
                incoming_prompt = int(usage.input or 0) + int(usage.cached_read or 0) + int(usage.cached_write or 0)
                stored_gen = int(exists["output_tokens"] or 0) + int(exists["thinking_tokens"] or 0)
                incoming_gen = int(usage.output or 0) + int(usage.thinking or 0)
                usage_split_diff = any(
                    incoming != int(exists[column] or 0)
                    for incoming, column in (
                        (int(usage.input or 0), "input_tokens"),
                        (int(usage.cached_read or 0), "cached_read_tokens"),
                        (int(usage.cached_write or 0), "cached_write_tokens"),
                        (int(usage.thinking or 0), "thinking_tokens"),
                        (int(usage.output or 0), "output_tokens"),
                    )
                )
                usage_grows = incoming_prompt > stored_prompt or incoming_gen > stored_gen
                usage_rebalances = usage_split_diff and incoming_prompt == stored_prompt and incoming_gen == stored_gen
                authoritative_cost = bool(getattr(cost, "authoritative", False))
                estimate_upgrade = bool(int(exists["cost_estimated"] or 0) and not usage.estimated)
                identity_upgrade = any(
                    incoming and incoming != (exists[column] or "")
                    for incoming, column in (
                        (model, "model"),
                        (alias, "model_alias"),
                        (provider, "provider"),
                        (agent, "agent_name"),
                    )
                )
                # Identical or smaller ordinary re-finalization stays a no-op.
                # A strictly larger billed SUM upgrades
                # the row in place so a last-round occupancy write cannot
                # permanently undercount the registry. A same-total split
                # rebalance (for example, a cache hit/miss breakdown arriving
                # after an occupancy-only checkpoint) is also accepted. An
                # explicit provider total may replace an interim estimate
                # when token counts are already equal; ordinary duplicate
                # CostResult objects do not get this escape hatch.
                if (
                    not usage_grows
                    and not usage_rebalances
                    and not authoritative_cost
                    and not estimate_upgrade
                    and not identity_upgrade
                ):
                    con.rollback()
                    return int(exists["cost_pico"]) if return_allocated_pico else False
                old_cost_pico = int(exists["cost_pico"] or 0)
                row_cost_pico = int(cost.total_pico)
                scope_key = (
                    f"goal:{ancestor_goal_id}" if ancestor_goal_id else f"session:{root_session_id or session_id}"
                )
                exact = cost.exact_total_pico
                if exact is None and authoritative_cost:
                    exact = Decimal(row_cost_pico)
                if exact is not None:
                    accumulator = con.execute(
                        "SELECT exact_pico, charged_pico FROM cost_accumulators WHERE scope_key=?",
                        (scope_key,),
                    ).fetchone()
                    if accumulator is not None:
                        prior_exact = Decimal(accumulator["exact_pico"])
                        prior_charged = int(accumulator["charged_pico"])
                        corrected_exact = prior_exact - Decimal(old_cost_pico) + exact
                        corrected_charged = int(corrected_exact.quantize(Decimal(1), rounding=ROUND_HALF_UP))
                        row_cost_pico = corrected_charged - (prior_charged - old_cost_pico)
                        con.execute(
                            "UPDATE cost_accumulators SET exact_pico=?, charged_pico=? WHERE scope_key=?",
                            (str(corrected_exact), corrected_charged, scope_key),
                        )
                # A same-token authoritative correction must not downgrade
                # the already-recorded usage buckets. A growing or rebalance
                # checkpoint replaces them; the final cost can still replace
                # the interim row cost on its own.
                row_usage = usage if (usage_grows or usage_rebalances or estimate_upgrade or identity_upgrade) else None
                con.execute(
                    """
                    UPDATE turn_costs SET
                        input_tokens=?, cached_read_tokens=?,
                        cached_write_tokens=?, thinking_tokens=?,
                        output_tokens=?, cost_pico=?, cost_estimated=?,
                        model=COALESCE(NULLIF(?, ''), model),
                        model_alias=COALESCE(NULLIF(?, ''), model_alias),
                        provider=COALESCE(NULLIF(?, ''), provider),
                        agent_name=COALESCE(NULLIF(?, ''), agent_name)
                    WHERE session_id=? AND turn_id=?
                    """,
                    (
                        (row_usage.input if row_usage is not None else exists["input_tokens"]),
                        (row_usage.cached_read if row_usage is not None else exists["cached_read_tokens"]),
                        (row_usage.cached_write if row_usage is not None else exists["cached_write_tokens"]),
                        (row_usage.thinking if row_usage is not None else exists["thinking_tokens"]),
                        (row_usage.output if row_usage is not None else exists["output_tokens"]),
                        row_cost_pico,
                        1 if cost.estimated else 0,
                        model,
                        alias,
                        provider,
                        agent,
                        session_id,
                        turn_id,
                    ),
                )
                con.commit()
                return row_cost_pico if return_allocated_pico else True

            row_cost_pico = int(cost.total_pico)
            exact = cost.exact_total_pico
            if exact is None and getattr(cost, "authoritative", False):
                exact = Decimal(row_cost_pico)
            scope_key = f"goal:{ancestor_goal_id}" if ancestor_goal_id else f"session:{root_session_id or session_id}"
            if exact is not None:
                accumulator = con.execute(
                    "SELECT exact_pico, charged_pico FROM cost_accumulators WHERE scope_key=?",
                    (scope_key,),
                ).fetchone()
                prior_exact = Decimal(accumulator["exact_pico"]) if accumulator is not None else Decimal(0)
                prior_charged = int(accumulator["charged_pico"]) if accumulator is not None else 0
                cumulative_exact = prior_exact + exact
                cumulative_charged = int(cumulative_exact.quantize(Decimal(1), rounding=ROUND_HALF_UP))
                row_cost_pico = cumulative_charged - prior_charged

            cur = con.execute(
                """
                INSERT OR IGNORE INTO turn_costs (
                    session_id, turn_id, root_session_id, ancestor_goal_id,
                    agent_name, model, model_alias, provider,
                    input_tokens, cached_read_tokens, cached_write_tokens,
                    thinking_tokens, output_tokens, cost_pico, cost_estimated,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    turn_id,
                    root_session_id or session_id,
                    ancestor_goal_id,
                    agent,
                    model,
                    alias,
                    provider,
                    usage.input,
                    usage.cached_read,
                    usage.cached_write,
                    usage.thinking,
                    usage.output,
                    row_cost_pico,
                    1 if cost.estimated else 0,
                    _utcnow_iso(),
                ),
            )
            if exact is not None:
                con.execute(
                    "INSERT INTO cost_accumulators "
                    "(scope_key, exact_pico, charged_pico) VALUES (?, ?, ?) "
                    "ON CONFLICT(scope_key) DO UPDATE SET "
                    "exact_pico=excluded.exact_pico, "
                    "charged_pico=excluded.charged_pico",
                    (scope_key, str(cumulative_exact), cumulative_charged),
                )
            con.commit()
            if return_allocated_pico:
                return row_cost_pico
            return cur.rowcount > 0

    def goal_cost_pico(self, goal_id: str) -> int:
        if not goal_id:
            return 0
        with self._connect() as con:
            row = con.execute(
                "SELECT COALESCE(SUM(cost_pico), 0) AS s FROM turn_costs WHERE ancestor_goal_id = ?",
                (goal_id,),
            ).fetchone()
        return int(row["s"] or 0)

    def session_cost_pico(self, root_session_id: str) -> int:
        if not root_session_id:
            return 0
        with self._connect() as con:
            row = con.execute(
                "SELECT COALESCE(SUM(cost_pico), 0) AS s FROM turn_costs WHERE root_session_id = ?",
                (root_session_id,),
            ).fetchone()
        return int(row["s"] or 0)

    def session_own_cost_pico(self, session_id: str) -> int:
        """Cost of a single session's OWN turns, keyed by ``session_id`` (not the
        rolled-up ``root_session_id``).

        ``session_cost_pico`` sums a whole tree by its true root; this sums just
        the turns a given session ran itself — the "downstream" figure a sub-agent
        chat needs, distinct from the tree total.
        """
        if not session_id:
            return 0
        with self._connect() as con:
            row = con.execute(
                "SELECT COALESCE(SUM(cost_pico), 0) AS s FROM turn_costs WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return int(row["s"] or 0)

    def session_window_cost_pico(
        self,
        session_id: str,
        start_iso: str,
        end_iso: str = "",
        *,
        end_exclusive: bool = False,
    ) -> int | None:
        """Cost of ``session_id``'s own turns recorded in ``[start, end]``.

        One delegation job drives one child session for a bounded time; a
        continued child runs again later under the same session id, so the
        job's figure is its time window, not the session total. Returns
        ``None`` when no row falls in the window (unknown, not zero).
        ``created_at`` is stored as UTC ISO-8601, so string order is time
        order.
        """
        if not session_id or not start_iso:
            return None
        sql = (
            "SELECT COUNT(*) AS n, COALESCE(SUM(cost_pico), 0) AS s "
            "FROM turn_costs WHERE session_id = ? AND created_at >= ?"
        )
        params: list[Any] = [session_id, start_iso]
        if end_iso:
            sql += " AND created_at < ?" if end_exclusive else " AND created_at <= ?"
            params.append(end_iso)
        with self._connect() as con:
            row = con.execute(sql, params).fetchone()
        if not row or not int(row["n"] or 0):
            return None
        return int(row["s"] or 0)

    def session_delegate_cost_pico(self, root_session_id: str) -> int:
        """Expensive-delegate spend rolled up under ``root_session_id``: the sum
        of every turn in the tree that a DESCENDANT (delegate/sub-agent) session
        ran, i.e. the whole-tree total minus the root session's OWN turns.

        A delegate child runs under its own ``session_id`` but rolls its cost up
        to the parent's ``root_session_id`` (delegate_shims._persist_child_cost_
        attribution), so ``root_session_id = root AND
        session_id != root`` is exactly the delegated portion of the tree spend.
        This is the recorded delegate spend the web budget indicator
        surfaces, distinct from ``session_cost_pico`` (whole
        tree) and ``session_own_cost_pico`` (one session's own turns).
        """
        if not root_session_id:
            return 0
        with self._connect() as con:
            row = con.execute(
                "SELECT COALESCE(SUM(cost_pico), 0) AS s FROM turn_costs WHERE root_session_id = ? AND session_id != ?",
                (root_session_id, root_session_id),
            ).fetchone()
        return int(row["s"] or 0)

    def record_generation(
        self,
        *,
        session_id: str,
        turn_id: str,
        metrics: dict[str, Any] | None,
        turn_idx: int = 0,
        round_idx: int = 0,
        agent_name: str = "",
        finish_reason: str = "",
    ) -> str:
        """Upsert one sharedagents-owned generation log row.

        The vLLM router never writes this table. Callers pass whatever the
        engine measured; missing turn-stat fields are filled by
        ``complete_metrics`` (same estimator as the per-turn stats row).
        """
        if not session_id or not turn_id:
            return ""
        from sharedagents_agent.metrics_format import complete_metrics

        raw = dict(metrics or {})

        def _opt_int(value: Any) -> int | None:
            try:
                if value is None:
                    return None
                return int(value)
            except (TypeError, ValueError):
                return None

        def _ms_to_s(value: Any) -> float | None:
            try:
                if value is None:
                    return None
                return float(value) / 1000.0
            except (TypeError, ValueError):
                return None

        filled = complete_metrics(
            raw,
            prompt_chars=_opt_int(raw.get("prompt_chars")),
            thinking_chars=_opt_int(raw.get("thinking_chars")),
            content_chars=_opt_int(raw.get("content_chars")),
            model=str(raw.get("model") or ""),
            context_window=_opt_int(raw.get("context_window")),
            turn_s=_ms_to_s(raw.get("wt_ms")) if raw.get("wt_ms") is not None else raw.get("turn_s"),
            ttft_s=_ms_to_s(raw.get("ttft_ms")) if raw.get("ttft_ms") is not None else raw.get("ttft_s"),
            gen_window_s=raw.get("gen_window_s"),
        )
        gid = generation_id(session_id, turn_id, round_idx)
        now = _utcnow_iso()

        def _i(key: str) -> int:
            try:
                return int(filled.get(key) or 0)
            except (TypeError, ValueError):
                return 0

        def _f(key: str) -> float:
            try:
                return float(filled.get(key) or 0.0)
            except (TypeError, ValueError):
                return 0.0

        engine = str(filled.get("engine") or "").strip()
        model = resolve_generation_model(
            model=filled.get("model"),
            model_alias=filled.get("model_alias"),
            provider=filled.get("provider"),
            engine=engine,
            agent_name=agent_name,
        )
        alias = persist_model_alias(
            model=filled.get("model"),
            model_alias=filled.get("model_alias"),
            resolved=model,
        )
        provider = resolve_generation_provider(
            provider=filled.get("provider"),
            engine=engine,
            model=model,
            model_alias=alias,
            agent_name=agent_name,
        )
        agent = resolve_generation_agent(
            agent_name=agent_name,
            provider=filled.get("provider"),
            engine=engine,
            model=model,
            model_alias=alias,
        )
        cached = _i("prompt_tokens_billed_cached") or _i("prompt_tokens_cached")
        prompt = _i("prompt_tokens_billed") or _i("prompt_tokens")
        cache_write = _i("prompt_tokens_billed_cache_write") or _i("prompt_tokens_cache_write")
        fresh = _i("prompt_tokens_fresh")
        if fresh <= 0 and prompt > 0:
            fresh = max(0, prompt - cached - cache_write)
        if cached <= 0 and cache_write > 0:
            cached = cache_write
        cost = _f("cost_usd")
        if filled.get("cost_usd_override") is not None:
            try:
                cost = float(filled["cost_usd_override"])
            except (TypeError, ValueError):
                pass
        elif cost == 0.0:
            try:
                from sharedagents_model.usage_pricing import normalize_usage, price

                usage = normalize_usage(filled)
                cost = price(
                    usage,
                    get_pricing_table().resolve(usage),
                    when=_utcnow(),
                ).total_usd
            except Exception:  # noqa: BLE001
                pass
        row = (
            gid,
            session_id,
            turn_id,
            int(turn_idx or 0),
            int(round_idx or 0),
            agent,
            model,
            alias,
            provider,
            engine,
            prompt,
            cached,
            fresh,
            _i("completion_tokens"),
            _i("reasoning_tokens"),
            _i("content_tokens"),
            _i("total_tokens"),
            _i("context_window"),
            _f("ttft_ms"),
            _f("ttfc_ms"),
            _f("prefill_ms"),
            _f("prefill_tps"),
            _f("tps"),
            _f("wt_ms"),
            _f("classify_ms"),
            _f("plan_ms"),
            _i("tool_events"),
            cost,
            1 if filled.get("tokens_estimated") else 0,
            str(finish_reason or filled.get("finish_reason") or "").strip(),
            now,
        )
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO ai_generations (
                    generation_id, session_id, turn_id, turn_idx, round_idx,
                    agent_name, model, model_alias, provider, engine,
                    input_tokens, cached_tokens, fresh_tokens, output_tokens,
                    thinking_tokens, content_tokens, total_tokens,
                    context_window, ttft_ms, ttfc_ms, prefill_ms, prefill_tps,
                    gen_tps, wt_ms, classify_ms, plan_ms, tool_events,
                    cost_usd, tokens_estimated, finish_reason, created_at
                ) VALUES (
                    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                )
                ON CONFLICT(generation_id) DO UPDATE SET
                    agent_name=excluded.agent_name,
                    model=excluded.model,
                    model_alias=excluded.model_alias,
                    provider=excluded.provider,
                    engine=excluded.engine,
                    input_tokens=excluded.input_tokens,
                    cached_tokens=excluded.cached_tokens,
                    fresh_tokens=excluded.fresh_tokens,
                    output_tokens=excluded.output_tokens,
                    thinking_tokens=excluded.thinking_tokens,
                    content_tokens=excluded.content_tokens,
                    total_tokens=excluded.total_tokens,
                    context_window=excluded.context_window,
                    ttft_ms=excluded.ttft_ms,
                    ttfc_ms=excluded.ttfc_ms,
                    prefill_ms=excluded.prefill_ms,
                    prefill_tps=excluded.prefill_tps,
                    gen_tps=excluded.gen_tps,
                    wt_ms=excluded.wt_ms,
                    classify_ms=excluded.classify_ms,
                    plan_ms=excluded.plan_ms,
                    tool_events=excluded.tool_events,
                    cost_usd=excluded.cost_usd,
                    tokens_estimated=excluded.tokens_estimated,
                    finish_reason=excluded.finish_reason
                """,
                row,
            )
            con.commit()
        return gid

    def sync_generations_from_turns(self) -> int:
        """Copy billed ``turn_costs`` rows that have no generation log yet.

        ``ai_generations`` only started with the Usage log feature. Production
        already had years of ``turn_costs``; without this copy the Logs tab
        stays empty even though the ledger has thousands of turns.
        Live ``record_generation`` rows (TTFT / tok/s) are left untouched.
        """
        inserted = 0
        with self._connect() as con:
            missing = con.execute(
                """
                SELECT session_id, turn_id, agent_name, model, model_alias,
                       provider, input_tokens, cached_read_tokens,
                       cached_write_tokens, thinking_tokens, output_tokens,
                       cost_pico, cost_estimated, created_at
                FROM turn_costs t
                WHERE NOT EXISTS (
                    SELECT 1 FROM ai_generations g
                    WHERE g.session_id = t.session_id AND g.turn_id = t.turn_id
                )
                """
            ).fetchall()
            if not missing:
                return 0
            rows = []
            for row in missing:
                sid = str(row["session_id"] or "")
                tid = str(row["turn_id"] or "")
                if not sid or not tid:
                    continue
                cached = int(row["cached_read_tokens"] or 0)
                fresh = int(row["input_tokens"] or 0)
                cache_write = int(row["cached_write_tokens"] or 0)
                if cached <= 0 and cache_write > 0:
                    cached = cache_write
                output = int(row["thinking_tokens"] or 0) + int(row["output_tokens"] or 0)
                thinking = int(row["thinking_tokens"] or 0)
                content = int(row["output_tokens"] or 0)
                prompt = fresh + cached
                cost_usd = int(row["cost_pico"] or 0) / 1_000_000_000_000
                rows.append(
                    (
                        generation_id(sid, tid, 0),
                        sid,
                        tid,
                        0,
                        0,
                        str(row["agent_name"] or "").strip(),
                        str(row["model"] or "").strip(),
                        str(row["model_alias"] or "").strip(),
                        resolve_generation_provider(
                            provider=row["provider"],
                            model=row["model"],
                            model_alias=row["model_alias"],
                            agent_name=row["agent_name"],
                        ),
                        "",
                        prompt,
                        cached,
                        fresh,
                        output,
                        thinking,
                        content,
                        prompt + output,
                        0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0,
                        cost_usd,
                        int(row["cost_estimated"] or 0),
                        "ledger",
                        str(row["created_at"] or _utcnow_iso()),
                    )
                )
            if not rows:
                return 0
            con.executemany(
                """
                INSERT OR IGNORE INTO ai_generations (
                    generation_id, session_id, turn_id, turn_idx, round_idx,
                    agent_name, model, model_alias, provider, engine,
                    input_tokens, cached_tokens, fresh_tokens, output_tokens,
                    thinking_tokens, content_tokens, total_tokens,
                    context_window, ttft_ms, ttfc_ms, prefill_ms, prefill_tps,
                    gen_tps, wt_ms, classify_ms, plan_ms, tool_events,
                    cost_usd, tokens_estimated, finish_reason, created_at
                ) VALUES (
                    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                )
                """,
                rows,
            )
            inserted = con.total_changes
            con.commit()
        return inserted

    def repair_generation_providers(self) -> int:
        """Fill empty/unknown ``ai_generations.provider`` from model/engine/agent.

        Also rewrite local_llm grok rows that an older heuristic stamped as
        ``cursor`` — those calls went to api.x.ai.
        """
        updated = 0
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT generation_id, provider, engine, model, model_alias, agent_name
                FROM ai_generations
                WHERE provider IS NULL
                   OR trim(provider) = ''
                   OR lower(provider) IN ('unknown', 'null', 'none', 'local-vllm', 'vllm')
                   OR (lower(provider) IN ('openrouter', 'openai') AND (
                       lower(model) LIKE '%codex%'
                       OR lower(model_alias) LIKE '%codex%'
                       OR lower(model) LIKE 'gpt-5.6%'
                       OR lower(model) LIKE 'gpt-5.5%'
                       OR lower(model) LIKE 'gpt-5.4%'
                   ) AND lower(model) NOT LIKE '%/%')
                   OR (
                       lower(provider) = 'cursor'
                       AND lower(COALESCE(engine, '')) = 'local_llm'
                       AND (
                           lower(COALESCE(model, '')) LIKE 'grok%'
                           OR lower(COALESCE(model_alias, '')) LIKE 'grok%'
                       )
                   )
                """
            ).fetchall()
            payload = []
            for row in rows:
                stored = row["provider"]
                if _stale_native_cursor_grok(
                    provider=stored,
                    engine=row["engine"],
                    model=row["model"],
                    model_alias=row["model_alias"],
                    agent_name=row["agent_name"],
                ):
                    stored = ""
                resolved = resolve_generation_provider(
                    provider=stored,
                    engine=row["engine"],
                    model=row["model"],
                    model_alias=row["model_alias"],
                    agent_name=row["agent_name"],
                )
                if resolved and resolved != str(row["provider"] or ""):
                    payload.append((resolved, row["generation_id"]))
            if payload:
                con.executemany(
                    "UPDATE ai_generations SET provider=? WHERE generation_id=?",
                    payload,
                )
                updated = con.total_changes
                con.commit()
        return updated

    def repair_turn_cost_providers(self) -> int:
        """Rewrite turn_costs grok rows that native agents billed as Cursor."""
        updated = 0
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT session_id, turn_id, provider, model, model_alias, agent_name
                FROM turn_costs
                WHERE lower(provider) = 'cursor'
                  AND (
                      lower(COALESCE(model, '')) LIKE 'grok%'
                      OR lower(COALESCE(model_alias, '')) LIKE 'grok%'
                  )
                """
            ).fetchall()
            payload = []
            for row in rows:
                stored = row["provider"]
                if _stale_native_cursor_grok(
                    provider=stored,
                    engine="",
                    model=row["model"],
                    model_alias=row["model_alias"],
                    agent_name=row["agent_name"],
                ):
                    stored = ""
                resolved = resolve_generation_provider(
                    provider=stored,
                    model=row["model"],
                    model_alias=row["model_alias"],
                    agent_name=row["agent_name"],
                )
                if resolved and resolved != str(row["provider"] or ""):
                    payload.append((resolved, row["session_id"], row["turn_id"]))
            if payload:
                con.executemany(
                    "UPDATE turn_costs SET provider=? WHERE session_id=? AND turn_id=?",
                    payload,
                )
                updated = con.total_changes
                con.commit()
        return updated

    def list_generations(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        provider: str = "",
        agent_name: str = "",
        model: str = "",
    ) -> dict[str, Any]:
        """Newest-first page of sharedagents generation logs."""
        self.sync_generations_from_turns()
        self.repair_generation_providers()
        self.repair_turn_cost_providers()
        limit = min(20, max(1, int(limit)))
        offset = max(0, int(offset))
        where: list[str] = []
        args: list[Any] = []
        if provider:
            where.append("provider = ?")
            args.append(provider)
        if agent_name:
            where.append("agent_name = ?")
            args.append(agent_name)
        if model:
            resolved = resolve_generation_model(model=model, model_alias=model)
            names = [name for name in dict.fromkeys([model, resolved]) if name]
            placeholders = ", ".join("?" * len(names))
            where.append(f"(model IN ({placeholders}) OR model_alias IN ({placeholders}))")
            args.extend(names)
            args.extend(names)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        with self._connect() as con:
            facets = {
                "providers": [
                    str(row[0])
                    for row in con.execute(
                        "SELECT DISTINCT provider FROM ai_generations "
                        "WHERE trim(COALESCE(provider, '')) != '' "
                        "ORDER BY provider COLLATE NOCASE"
                    )
                ],
                "agents": [
                    str(row[0])
                    for row in con.execute(
                        "SELECT DISTINCT agent_name FROM ai_generations "
                        "WHERE trim(COALESCE(agent_name, '')) != '' "
                        "ORDER BY agent_name COLLATE NOCASE"
                    )
                ],
                "models": [
                    str(row[0])
                    for row in con.execute(
                        "SELECT DISTINCT COALESCE(NULLIF(model, ''), model_alias) AS name "
                        "FROM ai_generations "
                        "WHERE trim(COALESCE(NULLIF(model, ''), model_alias, '')) != '' "
                        "ORDER BY name COLLATE NOCASE"
                    )
                ],
            }
            total = con.execute(
                f"SELECT COUNT(*) AS n FROM ai_generations {clause}",
                args,
            ).fetchone()["n"]
            rows = con.execute(
                f"""
                SELECT generation_id, session_id, turn_id, turn_idx, round_idx,
                       agent_name, model, model_alias, provider, engine,
                       input_tokens, cached_tokens, fresh_tokens, output_tokens,
                       thinking_tokens, content_tokens, total_tokens,
                       context_window, ttft_ms, ttfc_ms, prefill_ms, prefill_tps,
                       gen_tps, wt_ms, classify_ms, plan_ms, tool_events,
                       cost_usd, tokens_estimated, finish_reason, created_at
                FROM ai_generations {clause}
                ORDER BY created_at DESC, round_idx DESC
                LIMIT ? OFFSET ?
                """,
                [*args, limit, offset],
            ).fetchall()
        return {
            "total": int(total or 0),
            "limit": limit,
            "offset": offset,
            "filters": facets,
            "generations": [
                {
                    **dict(r),
                    "provider": resolve_generation_provider(
                        provider=r["provider"],
                        engine=r["engine"],
                        model=r["model"],
                        model_alias=r["model_alias"],
                        agent_name=r["agent_name"],
                    ),
                    "agent_name": resolve_generation_agent(
                        agent_name=r["agent_name"],
                        provider=r["provider"],
                        engine=r["engine"],
                        model=r["model"],
                        model_alias=r["model_alias"],
                    ),
                    "model": resolve_generation_model(
                        model=r["model"],
                        model_alias=r["model_alias"],
                        provider=r["provider"],
                        engine=r["engine"],
                        agent_name=r["agent_name"],
                    ),
                    "model_alias": _clean_label(r["model_alias"]),
                }
                for r in rows
            ],
        }

    def goal_estimated(self, goal_id: str) -> bool:
        """True if ANY row contributing to this goal was estimated."""
        if not goal_id:
            return False
        with self._connect() as con:
            row = con.execute(
                "SELECT MAX(cost_estimated) AS e FROM turn_costs WHERE ancestor_goal_id = ?",
                (goal_id,),
            ).fetchone()
        return bool(row["e"])


_ledger: UsageLedger | None = None


def get_usage_ledger(db_path: Path | None = None) -> UsageLedger:
    global _ledger
    if _ledger is None or db_path is not None:
        _ledger = UsageLedger(db_path)
    return _ledger


def _env_pricing_ttl() -> float:
    """TTL window (s) for the pricing-table cache; bad env values default to 60."""
    raw = os.environ.get("SHAREDAGENTS_PRICING_TTL_S", "60") or "60"
    try:
        return float(raw)
    except (TypeError, ValueError):
        _LOG.warning("invalid SHAREDAGENTS_PRICING_TTL_S=%r — defaulting to 60s", raw)
        return 60.0


# Pricing is a hot-editable config file, so the cache carries a TTL: after the
# TTL elapses we re-stat the file and reload only if its mtime changed (cheap —
# no re-parse on an unchanged file). Lets a pricing.json edit take effect within
# the TTL with no process restart. Override the window with SHAREDAGENTS_PRICING_TTL_S
# (0 = reload-check on every call; useful in tests).
_PRICING_TTL_S: float = _env_pricing_ttl()

# key -> {"table": PricingTable, "checked": monotonic, "mtime": float|None}.
# Guarded by _pricing_lock — the gateway is single-threaded asyncio, but
# delegated child agents may run on worker threads, so the check/reload/replace
# must be atomic and never expose a half-updated entry.
_pricing_cache: dict[str, dict[str, Any]] = {}
_pricing_lock = threading.Lock()


def _pricing_fingerprint(path: str | None) -> tuple[float | None, float | None]:
    """``(auto overlay mtime, operator file mtime)``, either side None if absent.

    BOTH files decide the effective table, so both must be watched: the auto
    overlay is rewritten by ``sharedagents_model.pricing_sync`` on a long-lived
    process's behalf, and a change there that nothing re-stats would never be
    picked up. The bundled-asset fallback still isn't watched — only the
    writable layers.
    """
    user_path = Path(path).expanduser() if path else None
    if user_path is None:
        from sharedagents_core.layout import pricing_path

        user_path = pricing_path()
    mtimes: list[float | None] = []
    for candidate in (user_path.with_name(AUTO_PRICING_FILENAME), user_path):
        try:
            mtimes.append(candidate.stat().st_mtime)
        except OSError:
            mtimes.append(None)
    return (mtimes[0], mtimes[1])


def get_pricing_table(
    path: str | None = None,
    *,
    reload: bool = False,
) -> PricingTable:
    """Load (and cache by path) the pricing table, with a TTL+mtime auto-reload.
    ``path`` honors ``AgentConfig.pricing_path``; "" / None uses the default
    location with the bundled-asset fallback. Pass ``reload=True`` to force.

    Fail-closed: a transient bad edit (unparseable pricing.json) keeps the
    last-good table rather than reverting rates mid-edit.
    """
    key = str(path or "")
    now = time.monotonic()
    with _pricing_lock:
        entry = _pricing_cache.get(key)
        if entry is None:
            table = PricingTable.load(path or None)
            _pricing_cache[key] = {
                "table": table,
                "checked": now,
                "fingerprint": _pricing_fingerprint(path),
            }
            return table
        if reload:
            try:
                table = PricingTable.load(path or None, strict=True)
            except Exception as exc:  # noqa: BLE001
                _LOG.warning(
                    "pricing reload failed (%s) — keeping last-good table",
                    exc,
                )
                entry["checked"] = now
                return entry["table"]
            _pricing_cache[key] = {
                "table": table,
                "checked": now,
                "fingerprint": _pricing_fingerprint(path),
            }
            return table
        if now - entry["checked"] < _PRICING_TTL_S:
            return entry["table"]
        # TTL elapsed — re-stat; reload only if either writable layer changed.
        fingerprint = _pricing_fingerprint(path)
        if fingerprint == entry["fingerprint"]:
            entry["checked"] = now
            return entry["table"]
        try:
            table = PricingTable.load(path or None, strict=True)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("pricing reload failed (%s) — keeping last-good table", exc)
            entry["checked"] = now  # back off until the next TTL window
            return entry["table"]
        # Atomic replace — never a half-updated entry.
        _pricing_cache[key] = {"table": table, "checked": now, "fingerprint": fingerprint}
        return table


class CostRecordingError(RuntimeError):
    """A cost-recording failure that MUST NOT be swallowed.

    Raised by :func:`record_turn_cost` only when ``surface_failure=True`` — set
    by the finalizer for a BUDGETED or DELEGATE turn. Swallowing the failure on
    such a turn would leave the authoritative ledger short a row, which
    undercounts the goal budget ``SUM`` (a goal could then drive
    past its real spend) and violates the rule to record the actual
    cost of every expensive delegate call. Ordinary (non-budgeted, non-delegate)
    turns keep the best-effort swallow and never see this exception.
    """


def record_turn_cost(
    *,
    session_id: str,
    turn_id: str,
    metrics: dict[str, Any] | None,
    root_session_id: str | None = None,
    ancestor_goal_id: str | None = None,
    agent_name: str = "",
    table: PricingTable | None = None,
    ledger: UsageLedger | None = None,
    surface_failure: bool = False,
) -> CostResult | None:
    """Single finalizer: normalize a turn's metrics → resolve price → record an
    idempotent ledger row. Returns the CostResult (None on no-op / missing ids).

    Best-effort by default: never raises into an ordinary turn. Stamps
    ``cost_*`` onto ``metrics`` in place so the cost rides the existing metrics
    flow (stats row, transcript).

    Surface/halt on budgeted+delegate turns: when
    ``surface_failure=True`` (the caller marks a BUDGETED or DELEGATE turn), a
    recording failure is RAISED as :class:`CostRecordingError` instead of
    swallowed, so the turn surfaces/halts rather than silently leaving the
    authoritative ledger short a row (which would undercount the goal budget
    ``SUM`` or drop a delegate-spend row). Ordinary turns keep the
    best-effort swallow.
    """
    if not session_id or not turn_id:
        return None
    try:
        tbl = table or get_pricing_table()
        usage = normalize_usage(metrics)
        # Price at the turn's wall clock: entries with a peak/off-peak schedule
        # bill the peak column when the turn ran in a peak
        # window; entries without a schedule ignore ``when`` entirely.
        cost = price(usage, tbl.resolve(usage), when=_utcnow())
        # Cost override: a delegate CLI that reports a total cost but no usable
        # token counts (e.g. claude's ``total_cost_usd`` with usage suppressed).
        # Price it directly instead of from tokens. estimated=False — it's the
        # CLI's own number, not our token estimate.
        override = (metrics or {}).get("cost_usd_override") if isinstance(metrics, dict) else None
        if override is not None:
            try:
                from sharedagents_model.usage_pricing import CostResult, usd_to_pico

                cost = CostResult(
                    total_pico=usd_to_pico(float(override)),
                    estimated=False,
                    model=usage.model,
                    model_alias=usage.model_alias,
                    provider=usage.provider,
                    authoritative=True,
                )
            except (TypeError, ValueError):
                pass
        lg = ledger or get_usage_ledger()
        allocated_pico = lg.record(
            session_id=session_id,
            turn_id=turn_id,
            cost=cost,
            usage=usage,
            root_session_id=root_session_id,
            ancestor_goal_id=ancestor_goal_id,
            agent_name=agent_name,
            return_allocated_pico=True,
        )
        cost.total_pico = int(allocated_pico)
        if isinstance(metrics, dict):
            metrics.update(cost.to_metrics())
        return cost
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("record_turn_cost failed for %s/%s: %s", session_id, turn_id, exc)
        if surface_failure:
            # Budgeted / delegate turn: a swallowed ledger
            # miss corrupts the goal budget SUM / delegate spend record —
            # surface it so the turn halts instead of silently continuing.
            raise CostRecordingError(f"cost recording failed for {session_id}/{turn_id}: {exc}") from exc
        return None


def record_generation(
    *,
    session_id: str,
    turn_id: str,
    metrics: dict[str, Any] | None,
    turn_idx: int = 0,
    round_idx: int = 0,
    agent_name: str = "",
    finish_reason: str = "",
    ledger: UsageLedger | None = None,
) -> str:
    """Write one generation log row from sharedagents (never the vLLM router)."""
    if not session_id or not turn_id:
        return ""
    try:
        lg = ledger or get_usage_ledger()
        return lg.record_generation(
            session_id=session_id,
            turn_id=turn_id,
            metrics=metrics,
            turn_idx=turn_idx,
            round_idx=round_idx,
            agent_name=agent_name,
            finish_reason=finish_reason,
        )
    except Exception as exc:  # noqa: BLE001
        _LOG.warning(
            "record_generation failed for %s/%s/%s: %s",
            session_id,
            turn_id,
            round_idx,
            exc,
        )
        return ""


def list_generations(
    *,
    limit: int = 20,
    offset: int = 0,
    provider: str = "",
    agent_name: str = "",
    model: str = "",
    ledger: UsageLedger | None = None,
) -> dict[str, Any]:
    lg = ledger or get_usage_ledger()
    return lg.list_generations(
        limit=limit,
        offset=offset,
        provider=provider,
        agent_name=agent_name,
        model=model,
    )
