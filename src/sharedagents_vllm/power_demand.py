"""Renew power leases by webhook without waiting in the inference request path.

Demand inside the throttle interval is coalesced into one trailing renewal.
Each runtime owns its client and at most one worker per pool; shutdown cancels
the workers, and idle pools stop renewing so their leases expire.
"""

from __future__ import annotations

import asyncio
import os
import time

from .models import Pool, PowerDemand


class Notifier:
    def __init__(self, cfg: PowerDemand | None, state):
        self.cfg = cfg
        self.state = state
        self._client = None
        self._last_sent: dict[str, float] = {}
        self._pending: dict[str, Pool] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    def start(self) -> None:
        if self.cfg is not None:
            import httpx

            # Client/TLS setup is synchronous: do it at startup, never per request.
            try:
                self._client = httpx.AsyncClient(timeout=self.cfg.timeout_seconds, trust_env=False)
            except Exception:  # noqa: BLE001 — optional integration cannot prevent serving
                self.state.bump("power_demand_error:startup")

    async def aclose(self) -> None:
        client, self._client = self._client, None
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._pending.clear()
        self._last_sent.clear()
        if client is not None:
            await client.aclose()

    def note(self, pool: Pool) -> None:
        if self._client is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._pending[pool.model_id] = pool
        if pool.model_id not in self._tasks:
            self._tasks[pool.model_id] = loop.create_task(self._renew(pool.model_id))

    async def _renew(self, model_id: str) -> None:
        try:
            while model_id in self._pending:
                delay = self.cfg.min_interval_seconds - (
                    time.monotonic() - self._last_sent.get(model_id, float("-inf"))
                )
                if delay > 0:
                    await asyncio.sleep(delay)
                pool = self._pending.pop(model_id)
                self._last_sent[model_id] = time.monotonic()
                if not await self._send(pool):
                    self._last_sent.pop(model_id, None)  # retry on the next demand
        finally:
            self._tasks.pop(model_id, None)

    async def _send(self, pool: Pool) -> bool:
        cfg = self.cfg
        body = {
            "lease_id": "vllm-router:" + pool.model_id,
            "caller": "sharedagents_vllm",
            "servers": list(pool.power_servers),
            "ttl_seconds": cfg.ttl_seconds,
            "reason": "requests for " + pool.model_id,
        }
        try:
            token = os.environ.get(cfg.token_env, "")
            # HTTP timeouts bound individual read gaps, not a slow response's
            # total duration. Bound the entire call as well.
            resp = await asyncio.wait_for(
                self._client.post(cfg.url, json=body, headers={"X-Custom-Authorization": token}),
                timeout=cfg.timeout_seconds,
            )
            ok = 200 <= resp.status_code < 300
        except Exception:  # noqa: BLE001 — optional webhook failures stay isolated
            ok = False
        self.state.bump(("power_demand:" if ok else "power_demand_error:") + pool.model_id)
        return ok


def note(pool: Pool | None) -> None:
    """Record demand for ``pool``. Safe to call on every request."""
    if pool is None or not pool.power_servers:
        return
    from .runtime import get

    get().power_demand.note(pool)
