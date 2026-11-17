"""Discover Pi's configured catalogue through its RPC metadata commands."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from sharedagents_orchestrator.engines.pi_rpc_adapter import PiRpcAdapter

from ._common import _cache_get, _cache_put

_LOG = logging.getLogger(__name__)


def _picker_rows(models: Any, current: Any) -> list[dict[str, Any]]:
    if not isinstance(models, list):
        raise ValueError("Pi returned an invalid model catalogue")
    current = current if isinstance(current, dict) else {}
    selected = (current.get("provider"), current.get("id"))
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for model in models:
        if not isinstance(model, dict):
            continue
        provider, model_id = model.get("provider"), model.get("id")
        if not isinstance(provider, str) or not isinstance(model_id, str):
            continue
        provider, model_id = provider.strip(), model_id.strip()
        if not provider or not model_id:
            continue
        qualified = f"{provider}/{model_id}"
        if qualified in seen:
            continue
        seen.add(qualified)
        # Publish only picker metadata, never arbitrary provider config.
        row: dict[str, Any] = {
            "id": qualified,
            "alias": qualified,
            "display": str(model.get("name") or model_id),
            "provider": provider,
            "default": (provider, model_id) == selected,
        }
        for field in ("contextWindow", "maxTokens"):
            value = model.get(field)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                row[field] = value
        rows.append(row)
    return sorted(rows, key=lambda row: not row["default"])


async def _read_catalogue(command: list[str]) -> list[dict[str, Any]]:
    adapter = PiRpcAdapter(command=[*command, "--no-tools"], no_session=True)

    async def read() -> list[dict[str, Any]]:
        response = await adapter.send({"type": "get_available_models"}, wait=True)
        if not response or not response.get("success"):
            raise RuntimeError("Pi model discovery was rejected")
        data = response.get("data")
        if not isinstance(data, dict):
            raise ValueError("Pi returned an invalid model catalogue")
        state = await adapter.send({"type": "get_state"}, wait=True)
        state_data = (state or {}).get("data")
        current = state_data.get("model") if isinstance(state_data, dict) else None
        return _picker_rows(data.get("models"), current)

    try:
        return await asyncio.wait_for(read(), timeout=15.0)
    finally:
        await adapter.close()


def discover_pi_models(
    *,
    force: bool = False,
    command: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Query metadata without prompting a model; forced refresh bypasses cache.

    Engine catalogue callers can already own an event loop. In that case the
    temporary RPC loop runs in a joined thread; no work outlives this call.
    """
    argv = list(command or ["pi"])
    resolved = PiRpcAdapter.resolve_executable(argv[0])
    if resolved:
        argv[0] = resolved
    key = "pi:" + repr(argv)
    if not force:
        cached = _cache_get(key)
        if cached is not None:
            return cached
    try:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            rows = asyncio.run(_read_catalogue(argv))
        else:
            with ThreadPoolExecutor(max_workers=1) as executor:
                rows = executor.submit(lambda: asyncio.run(_read_catalogue(argv))).result()
        _cache_put(key, rows)
        return [dict(row) for row in rows]
    except Exception as exc:
        # RPC errors/stderr can include provider configuration: log only type.
        _LOG.warning("Pi model discovery failed (%s)", type(exc).__name__)
        return []
