"""FastAPI composition for the local vLLM router."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from .config import default_config_path, load_config, parse_config
from .metrics import render_metrics
from .models import Config, Pool
from .proxy import dispatch_by_model
from .runtime import bind, get

log = logging.getLogger("sharedagents_vllm")


def _own_state(pool: Pool) -> str:
    live = sum(1 for r in pool.replicas if r.healthy)
    if live == 0:
        return "down"
    if all(r.health == "ok" for r in pool.replicas):
        return "ok"
    return "degraded"


def pool_route_state(pools: dict[str, Pool], pool: Pool) -> tuple[str, Pool]:
    """Local serving state of ``pool`` and the pool that actually serves it.

    A forward-only pool (``replicas: []``, used as an alias name) is served by
    its local ``pool:`` fallback hops, one level deep exactly like
    ``dispatch_fallback``. It is up when one of those targets is up, so it is
    not reported down while every request it gets is answered locally.
    External hops do not count: this is the local state.
    """
    if pool.replicas:
        return _own_state(pool), pool
    best: tuple[str, Pool] = ("down", pool)
    for fb in pool.fallbacks:
        target = pools.get(fb.pool or "")
        if target is None or target is pool or not target.replicas:
            continue
        state = _own_state(target)
        if state == "ok":
            return state, target
        if state == "degraded" and best[0] == "down":
            best = (state, target)
    return best


def create_app(
    cfg: Config | None = None,
    *,
    config_path: str | None = None,
    client: Any = None,
    start_health: bool = True,
    environ: dict[str, str] | None = None,
):
    """Build the ASGI app and bind process-global runtime.

    ``client`` / ``start_health=False`` are for tests (inject a fake httpx
    client and skip the background probe loop).
    """
    if cfg is None:
        path = config_path or str(default_config_path())
        cfg = load_config(path)
    rt = bind(cfg, client=client, start_health=start_health, environ=environ)

    @asynccontextmanager
    async def lifespan(app):
        import httpx

        own_client = False
        if rt.client is None:
            rt.client = httpx.AsyncClient(
                trust_env=False,
                follow_redirects=False,
                timeout=httpx.Timeout(
                    cfg.write_pool_timeout_seconds,
                    connect=cfg.connect_timeout_seconds,
                    read=cfg.read_timeout_seconds,
                ),
                limits=httpx.Limits(max_connections=512, max_keepalive_connections=128),
            )
            own_client = True
        health_task = None
        if rt.start_health:
            from .health import health_loop

            health_task = asyncio.create_task(health_loop())
        log.info(
            "sharedagents_vllm loaded %d pools: %s",
            len(cfg.pools),
            ", ".join(f"{m}({len(p.replicas)})" for m, p in cfg.pools.items()),
        )
        try:
            rt.power_demand.start()
            yield
        finally:
            await rt.power_demand.aclose()
            if health_task is not None:
                health_task.cancel()
                try:
                    await health_task
                except asyncio.CancelledError:
                    pass
            if own_client and rt.client is not None:
                await rt.client.aclose()
                rt.client = None

    app = FastAPI(title="sharedagents_vllm", lifespan=lifespan)

    @app.get("/")
    async def root():
        return JSONResponse(
            {
                "service": "sharedagents_vllm",
                "models": "/v1/models",
                "health": "/health",
                "metrics": "/metrics",
            }
        )

    @app.get("/v1/models")
    async def list_models():
        data = []
        pools = get().cfg.pools
        for model_id, pool in pools.items():
            if pool.hidden:
                continue
            healthy = sum(1 for r in pool.replicas if r.healthy)
            route_state, via = pool_route_state(pools, pool)
            data.append(
                {
                    "id": model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "vllm-mux",
                    "kind": pool.kind,
                    "replicas_total": len(pool.replicas),
                    "replicas_healthy": healthy,
                    "replicas": [
                        {
                            "name": r.name,
                            "url": r.url,
                            "healthy": r.healthy,
                            "health": r.health,
                        }
                        for r in pool.replicas
                    ],
                    "route_state": route_state,
                    "served_by": via.model_id,
                }
            )
        return JSONResponse({"object": "list", "data": data})

    @app.get("/health")
    async def health():
        rt = get()
        now = rt.now()
        ok, degraded, down = [], [], []
        replicas_out = []
        any_non_ok = False
        for model_id, pool in rt.cfg.pools.items():
            state, via = pool_route_state(rt.cfg.pools, pool)
            if state == "down":
                down.append(model_id)
            elif state == "degraded":
                live = sum(1 for r in via.replicas if r.healthy)
                label = f"{live}/{len(via.replicas)}"
                if via is not pool:
                    label = f"via {via.model_id} {label}"
                degraded.append(f"{model_id} ({label})")
            else:
                ok.append(model_id)
            for r in pool.replicas:
                if r.health != "ok":
                    any_non_ok = True
                age = (now - r.last_probe_at) if r.last_probe_at else None
                replicas_out.append(
                    {
                        "pool": model_id,
                        "name": r.name,
                        "health": r.health,
                        "failure_class": r.failure_class,
                        "consecutive_failures": r.consecutive_failures,
                        "last_error": r.last_error,
                        "last_probe_age_s": None if age is None else round(age, 3),
                        "last_probe_latency_s": round(r.last_probe_latency_s, 3),
                        "cooldown_until": r.cooldown_until,
                    }
                )
        stuck = rt.state.stuck_count(rt.mono())
        if stuck > 0 or any_non_ok:
            # Never advertise ok while a replica is failed/cooling/degraded or stuck.
            if down and not ok and not degraded:
                status = "down"
            else:
                status = "degraded" if (down or degraded or stuck or any_non_ok) else "ok"
        elif not down:
            status = "ok"
        elif ok or degraded:
            status = "degraded"
        else:
            status = "down"
        if stuck > 0 and status == "ok":
            status = "degraded"
        return JSONResponse(
            {
                "status": status,
                "ok": ok,
                "degraded": degraded,
                "down": down,
                "active_requests": dict(rt.state.active_requests),
                "replicas": replicas_out,
                "stuck_requests": stuck,
                # Cloud hops are part of the status surface: a configured provider
                # with no key silently drops every fallback, which is invisible in
                # the pool/replica view above. Never expose ``api_key`` here.
                "external_providers": [
                    {
                        "id": provider.id,
                        "model": provider.model,
                        "base_url": provider.base_url,
                        "api_key_env": provider.api_key_env,
                        "enabled": provider.enabled,
                        "active_requests": provider.active_requests,
                        "max_active_requests": provider.max_active_requests,
                        "cooldown_remaining_s": round(max(0.0, provider.cooldown_until - now), 3),
                    }
                    for provider in rt.state.ext_providers.values()
                ],
            }
        )

    @app.get("/metrics")
    async def metrics():
        return Response(
            content=render_metrics(),
            media_type="text/plain; version=0.0.4",
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(req: Request):
        return await dispatch_by_model(req, path="/v1/chat/completions", body_kind="json")

    @app.post("/v1/completions")
    async def completions(req: Request):
        return await dispatch_by_model(req, path="/v1/completions", body_kind="json")

    @app.post("/v1/responses")
    async def responses(req: Request):
        return await dispatch_by_model(req, path="/v1/responses", body_kind="json")

    @app.post("/v1/messages")
    async def messages(req: Request):
        return await dispatch_by_model(req, path="/v1/messages", body_kind="json")

    @app.post("/v1/embeddings")
    async def embeddings(req: Request):
        return await dispatch_by_model(req, path="/v1/embeddings", body_kind="json")

    @app.post("/rerank")
    @app.post("/v1/rerank")
    @app.post("/v2/rerank")
    async def rerank(req: Request):
        return await dispatch_by_model(req, path=str(req.url.path), body_kind="json")

    @app.post("/score")
    @app.post("/v1/score")
    async def score(req: Request):
        return await dispatch_by_model(req, path=str(req.url.path), body_kind="json")

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(req: Request):
        return await dispatch_by_model(
            req,
            path="/v1/audio/transcriptions",
            body_kind="multipart",
        )

    @app.post("/v1/audio/translations")
    async def translations(req: Request):
        return await dispatch_by_model(
            req,
            path="/v1/audio/translations",
            body_kind="multipart",
        )

    @app.post("/v1/audio/speech")
    async def speech(req: Request):
        return await dispatch_by_model(req, path="/v1/audio/speech", body_kind="json")

    return app


# Imported for ``parse_config`` re-export convenience in tests.
__all__ = ["create_app", "parse_config"]
