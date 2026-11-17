"""Upstream proxy: replica walk, local pool fallback, then optional cloud."""

from __future__ import annotations

import asyncio
import json
import logging
import time

from . import power_demand
from .models import ExternalProvider, Pool, Replica
from .routing import (
    derive_routing_key,
    key_hash,
    pick_round_robin,
    priority_attempts,
    rank_by_priority,
    rank_replicas,
)
from .runtime import find_pool, get

log = logging.getLogger("sharedagents_vllm")

_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
    "accept-encoding",
}


def _clean_request_headers(req) -> dict[str, str]:
    return {k: v for k, v in req.headers.items() if k.lower() not in _HOP_BY_HOP}


def _clean_response_headers(resp) -> dict[str, str]:
    return {k: v for k, v in resp.headers.items() if k.lower() not in _HOP_BY_HOP}


# Sent downstream while a replica is byte-silent but still generating, so the
# caller's own inactivity timers see a live stream. SSE comment: clients that
# do not know it ignore it. The gateway worker matches the ``sharedagents-mux``
# prefix (``sharedagents_gateway.worker._MUX_KEEPALIVE_PREFIX``).
KEEPALIVE_COMMENT = b": sharedagents-mux keep-alive (upstream generating)\n\n"

# Below this rate a moving counter is not generation: the router's own
# inference probe adds up to 16 tokens every ``inference_probe_seconds``.
_MIN_GENERATION_TPS = 2.0


async def _generation_tokens(replica: Replica) -> float | None:
    """vLLM's cumulative generated-token counter, or None when unavailable."""
    rt = get()
    if rt.client is None:
        return None
    try:
        resp = await rt.client.get(
            f"{replica.url}/metrics",
            timeout=max(0.2, float(rt.cfg.health_timeout_seconds)),
            headers={"X-Router-Probe": "1"},
        )
        if not 200 <= resp.status_code < 300:
            return None
        total: float | None = None
        for line in resp.text.splitlines():
            if line.startswith("vllm:generation_tokens_total"):
                total = (total or 0.0) + float(line.rsplit(" ", 1)[1])
        return total
    except Exception:  # noqa: BLE001
        return None


def _sse_error(message: str) -> bytes:
    """Terminal SSE frame in the OpenAI error envelope the worker surfaces."""
    payload = {"error": {"message": message, "type": "upstream_stall", "code": 504}}
    return f"data: {json.dumps(payload)}\n\n".encode()


async def _open_upstream(
    req,
    upstream_url: str,
    body: bytes | None,
    first_token_timeout: float,
    body_read_timeout: float | None = None,
):
    """Send the request and return once the response headers are in.

    httpx's read timeout also applies between body chunks. When the caller
    paces the body itself it passes ``body_read_timeout`` as that backstop,
    and the wait for the headers is bounded by ``first_token_timeout`` here.
    """
    import httpx

    rt = get()
    assert rt.client is not None
    upstream_req = rt.client.build_request(
        req.method,
        upstream_url,
        headers=_clean_request_headers(req),
        params=dict(req.query_params),
        content=body,
        timeout=httpx.Timeout(
            rt.cfg.write_pool_timeout_seconds,
            connect=rt.cfg.connect_timeout_seconds,
            read=first_token_timeout if body_read_timeout is None else body_read_timeout,
        ),
    )
    send = rt.client.send(upstream_req, stream=True)
    if body_read_timeout is None:
        return await send
    return await asyncio.wait_for(send, timeout=first_token_timeout)


async def respond_from_upstream(
    upstream_resp,
    replica: Replica | None,
    inter_token_timeout: float,
    on_complete,
    first_token_timeout: float | None = None,
):
    from fastapi.responses import Response, StreamingResponse

    content_type = str(upstream_resp.headers.get("content-type", ""))
    is_sse = content_type.startswith("text/event-stream")
    is_stream = is_sse or upstream_resp.headers.get("transfer-encoding") == "chunked"
    rt = get()

    if is_stream:

        async def gen():
            n_chunks = 0
            sidelined = False
            aiter = upstream_resp.aiter_raw()
            # Only a local replica serving SSE can be asked about generation
            # and safely sent comment frames; everything else keeps the plain
            # inter-token timeout (chunked audio must not gain injected bytes).
            probing = is_sse and replica is not None
            tick = min(inter_token_timeout, rt.cfg.generation_probe_seconds) if probing else inter_token_timeout
            # A timed-out read is kept, never cancelled: cancelling an httpx
            # body read closes the upstream connection.
            pending: asyncio.Task | None = None
            last_chunk = last_progress = time.monotonic()
            sample: tuple[float, float] | None = None  # (mono, tokens)
            try:
                while True:
                    if pending is None:
                        pending = asyncio.ensure_future(aiter.__anext__())
                    if n_chunks == 0:
                        await asyncio.wait({pending}, timeout=first_token_timeout)
                    else:
                        await asyncio.wait({pending}, timeout=tick)
                    if pending.done():
                        task, pending = pending, None
                        try:
                            chunk = task.result()
                        except StopAsyncIteration:
                            break
                        n_chunks += 1
                        last_chunk = last_progress = time.monotonic()
                        sample = None
                        yield chunk
                        continue

                    now = time.monotonic()
                    if probing and n_chunks:
                        tokens = await _generation_tokens(replica)
                        if tokens is not None:
                            if sample is not None and (tokens - sample[1] >= _MIN_GENERATION_TPS * (now - sample[0])):
                                last_progress = now
                            sample = (now, tokens)
                    silent = now - last_chunk
                    idle = now - last_progress
                    if n_chunks == 0:
                        reason = f"sent no first token within {first_token_timeout:g}s"
                        if replica is not None:
                            rt.state.mark_unhealthy(replica, "mid-stream before first token")
                            sidelined = True
                    elif idle < inter_token_timeout:
                        if silent < rt.cfg.stream_silence_cap_seconds:
                            if last_progress > last_chunk:
                                yield KEEPALIVE_COMMENT
                            continue
                        reason = (
                            f"sent no output for {silent:.0f}s (silence cap "
                            f"{rt.cfg.stream_silence_cap_seconds:g}s) although its "
                            "engine is still generating"
                        )
                    else:
                        reason = f"inter-token stall (>{inter_token_timeout}s)"
                        if replica is not None:
                            rt.state.mark_unhealthy(replica, reason)
                            sidelined = True
                    if not is_sse:
                        raise asyncio.TimeoutError(reason)
                    name = replica.name if replica is not None else "external provider"
                    message = f"router stream timeout: {name} {reason}"
                    log.warning("%s", message)
                    yield _sse_error(message)
                    break
            except Exception:
                if replica is not None and not sidelined and n_chunks == 0:
                    rt.state.mark_unhealthy(replica, "mid-stream before first token")
                raise
            finally:
                if pending is not None:
                    pending.cancel()
                try:
                    await upstream_resp.aclose()
                finally:
                    if on_complete is not None:
                        on_complete()

        return StreamingResponse(
            gen(),
            status_code=upstream_resp.status_code,
            headers=_clean_response_headers(upstream_resp),
            media_type=upstream_resp.headers.get("content-type"),
        )

    payload = await upstream_resp.aread()
    await upstream_resp.aclose()
    if on_complete is not None:
        on_complete()
    return Response(
        content=payload,
        status_code=upstream_resp.status_code,
        headers=_clean_response_headers(upstream_resp),
        media_type=upstream_resp.headers.get("content-type"),
    )


def _tag_fallback(resp, frm: str, to: str):
    resp.headers["X-Router-Fallback"] = "true"
    resp.headers["X-Router-Fallback-From"] = frm
    resp.headers["X-Router-Fallback-To"] = to
    return resp


def _reject_capacity(model_id: str):
    from fastapi.responses import JSONResponse

    rt = get()
    rt.state.backpressure[model_id] = rt.state.backpressure.get(model_id, 0) + 1
    rt.state.bump(f"backpressure:{model_id}")
    retry_after = rt.cfg.backpressure.retry_after_seconds
    return JSONResponse(
        status_code=rt.cfg.backpressure.reject_status,
        headers={"Retry-After": str(retry_after)},
        content={
            "error": {
                "message": f"all local providers for '{model_id}' at capacity",
                "type": "capacity_exceeded",
                "retry_after": retry_after,
            }
        },
    )


def _disabled_external_hops(model_id: str) -> list[str]:
    """Configured cloud hops that cannot run because their key is missing."""
    rt = get()
    pool = rt.cfg.pools.get(model_id)
    out: list[str] = []
    for hop in (pool.fallbacks if pool is not None else []) or []:
        if not hop.external:
            continue
        provider = rt.state.ext_providers.get(hop.external)
        if provider is None:
            out.append(f"{hop.external} (unknown provider)")
        elif not provider.enabled:
            out.append(f"{provider.id} ({provider.api_key_env or 'api_key_env unset'})")
    return out


def _reject_down(model_id: str, reason: str):
    from fastapi.responses import JSONResponse

    rt = get()
    rt.state.bump(f"fallback_exhausted:{model_id}")
    retry_after = rt.cfg.backpressure.retry_after_seconds
    disabled = _disabled_external_hops(model_id)
    message = f"all local replicas for '{model_id}' are down/wedged and every fallback failed ({reason})"
    error: dict = {
        "message": message,
        "type": "upstream_unavailable",
        "retry_after": retry_after,
    }
    if disabled:
        error["disabled_external_fallbacks"] = disabled
        error["message"] = f"{message}; cloud fallback skipped — no API key for {', '.join(disabled)}"
    return JSONResponse(
        status_code=503,
        headers={"Retry-After": str(retry_after)},
        content={"error": error},
    )


def _visited_plus(visited: frozenset[str] | None, pool_id: str) -> frozenset[str]:
    return (visited or frozenset()) | {pool_id}


async def _read_json_body(req) -> tuple[bytes, dict | None]:
    body = await req.body()
    if not body:
        return body, None
    ctype = req.headers.get("content-type", "")
    if "application/json" not in ctype:
        return body, None
    try:
        return body, json.loads(body)
    except Exception:  # noqa: BLE001
        return body, None


async def dispatch_by_model(req, *, path: str, body_kind: str):
    from fastapi.responses import JSONResponse

    rt = get()
    if body_kind == "json":
        body, parsed = await _read_json_body(req)
        model_id = (parsed or {}).get("model")
        if not model_id:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "`model` field required to route"}},
            )
        pool = find_pool(str(model_id))
        if pool is None:
            return JSONResponse(
                status_code=404,
                content={"error": {"message": f"unknown model '{model_id}'"}},
            )
        power_demand.note(pool)
        # Advisory decision pools own their queue, shed, and outage handling.
        # Single early branch: every other pool's path below is unchanged.
        if pool.kind == "decision":
            from .decision import dispatch_decision

            return await dispatch_decision(
                req,
                pool,
                path=path,
                body=body,
                parsed=parsed,
            )
        key = derive_routing_key(
            {k.lower(): v for k, v in req.headers.items()},
            parsed,
        )
    else:
        body = await req.body()
        parsed = None
        form = await req.form()
        model_id = form.get("model")
        if not model_id:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "`model` field required in form"}},
            )
        pool = find_pool(str(model_id))
        if pool is None:
            return JSONResponse(
                status_code=404,
                content={"error": {"message": f"unknown model '{model_id}'"}},
            )
        power_demand.note(pool)
        key = req.headers.get("x-session-id", "") or ""

    if rt.state.pool_at_capacity(pool):
        rt.state.bump(f"capacity:{pool.model_id}")
        return await dispatch_fallback(
            req,
            pool,
            path=path,
            body=body,
            parsed=parsed,
            key=key,
            reason="capacity",
        )
    if not rt.state.pool_has_healthy(pool):
        rt.state.bump(f"unhealthy:{pool.model_id}")
        if pool.fallbacks:
            return await dispatch_fallback(
                req,
                pool,
                path=path,
                body=body,
                parsed=parsed,
                key=key,
                reason="unhealthy",
            )
        return _reject_down(pool.model_id, "unhealthy")
    return await dispatch_to_pool(
        req,
        pool,
        path=path,
        body=body,
        parsed=parsed,
        key=key,
    )


async def dispatch_to_pool(
    req,
    pool: Pool,
    *,
    path: str,
    body: bytes,
    parsed: dict | None,
    key: str,
    _visited: frozenset[str] | None = None,
    allow_chain: bool = True,
):

    rt = get()
    st = rt.state
    model_id = pool.model_id
    reserved: Replica | None = None
    current_fid: str | None = None

    def _release() -> None:
        nonlocal reserved, current_fid
        if current_fid is not None:
            st.end_inflight(current_fid, reserved)
            current_fid = None
            reserved = None

    try:
        now = time.monotonic()
        st._sweep(now)
        is_new = str(req.headers.get("x-session-new", "")).lower() == "true"
        kh = key_hash(key) if key else ""
        chosen: str | None = None

        if pool.routing == "priority":
            # Fastest host first on every request; full replicas are skipped below.
            candidates = rank_by_priority(pool)
            kh = ""
        elif pool.kind == "llm" and kh:
            ranked = rank_replicas(key, pool)
            healthy_keys = {r.key() for r in ranked if r.healthy}
            prev = None if is_new else st.lookup(kh)
            if prev and prev in healthy_keys:
                candidates = [r for r in ranked if r.key() == prev] + [r for r in ranked if r.key() != prev]
                chosen = prev
            else:
                first = pick_round_robin(pool)
                if first:
                    candidates = [first] + [r for r in ranked if r.key() != first.key()]
                    chosen = first.key()
                else:
                    candidates = ranked
                    chosen = candidates[0].key() if candidates else None
            if chosen:
                st.classify_cache(key, chosen, pool.model_id)
                st.assign(kh, chosen, now)
        else:
            candidates = rank_replicas(key, pool)
            kh = ""

        if not candidates:
            _release()
            if allow_chain and pool.fallbacks:
                return await dispatch_fallback(
                    req,
                    pool,
                    path=path,
                    body=body,
                    parsed=parsed,
                    key=key,
                    reason="unhealthy",
                    _visited=_visited_plus(_visited, model_id),
                )
            return _reject_down(model_id, "unhealthy")

        eff_first = (
            pool.first_token_timeout_seconds
            if pool.first_token_timeout_seconds is not None
            else rt.cfg.first_token_timeout_seconds
        )
        eff_inter = pool.read_timeout_seconds if pool.read_timeout_seconds is not None else rt.cfg.read_timeout_seconds
        eff_open = eff_first if pool.kind == "llm" else max(eff_inter, 120)
        # LLM streams are paced by respond_from_upstream (first token, then
        # generation-aware silence); httpx only backstops a silent body.
        eff_body = max(eff_first, rt.cfg.stream_silence_cap_seconds + eff_inter) if pool.kind == "llm" else None
        last_detail = ""

        # Replicas get the name they serve; fallbacks keep the original body.
        def body_for(replica: Replica) -> bytes:
            served = replica.upstream_model or pool.upstream_model
            if served and parsed is not None and parsed.get("model") != served:
                return json.dumps({**parsed, "model": served}).encode()
            return body

        attempts = priority_attempts(pool) if pool.routing == "priority" else iter(candidates)
        for attempt, replica in enumerate(attempts):
            if replica.max_active > 0 and replica.active >= replica.max_active:
                continue
            # A swept attempt may resume after another request took its slot.
            # Each attempt owns one pool slot through its inflight record.
            if st.pool_at_capacity(pool):
                if allow_chain and pool.fallbacks:
                    return await dispatch_fallback(
                        req,
                        pool,
                        path=path,
                        body=body,
                        parsed=parsed,
                        key=key,
                        reason="capacity",
                        _visited=_visited_plus(_visited, model_id),
                    )
                return _reject_capacity(model_id)
            st.acquire_pool(model_id)
            replica.active += 1
            reserved = replica
            current_fid = st.begin_inflight(
                model_id,
                replica.key(),
                path,
                time.monotonic(),
            )
            try:
                upstream_resp = await _open_upstream(
                    req,
                    f"{replica.url}{path}",
                    body_for(replica),
                    first_token_timeout=eff_open,
                    body_read_timeout=eff_body,
                )
            except Exception as exc:  # noqa: BLE001
                _release()
                st.mark_unhealthy(replica, f"open: {type(exc).__name__}: {exc}")
                last_detail = f"{replica.name}: {exc}"
                log.warning(
                    "[%s] %s open failed (%s), failover %d/%d",
                    pool.kind,
                    replica.name,
                    type(exc).__name__,
                    attempt + 1,
                    len(candidates),
                )
                continue

            if upstream_resp.status_code >= 500:
                try:
                    await upstream_resp.aread()
                except Exception:  # noqa: BLE001
                    pass
                finally:
                    await upstream_resp.aclose()
                _release()
                st.mark_unhealthy(replica, f"http {upstream_resp.status_code}")
                last_detail = f"{replica.name}: http {upstream_resp.status_code}"
                continue

            st.bump(f"route:{pool.model_id}@{replica.name}")
            replica.last_user_success = time.time()
            if kh and replica.key() != chosen:
                st.assign(kh, replica.key(), time.monotonic())

            return await respond_from_upstream(
                upstream_resp,
                replica,
                inter_token_timeout=eff_inter,
                on_complete=_release,
                first_token_timeout=eff_first if pool.kind == "llm" else None,
            )

        _release()
        if allow_chain and pool.fallbacks:
            return await dispatch_fallback(
                req,
                pool,
                path=path,
                body=body,
                parsed=parsed,
                key=key,
                reason="all_failed",
                _visited=_visited_plus(_visited, model_id),
            )
        return _reject_down(model_id, last_detail or "all_failed")
    except BaseException:
        _release()
        raise


async def dispatch_fallback(
    req,
    primary: Pool,
    *,
    path: str,
    body: bytes,
    parsed: dict | None,
    key: str,
    reason: str,
    _visited: frozenset[str] | None = None,
):
    rt = get()
    st = rt.state
    frm = primary.model_id
    visited = _visited_plus(_visited, frm)
    if key:
        st.break_affinity(key_hash(key))

    for fb in primary.fallbacks:
        if fb.pool:
            if fb.pool in visited:
                continue
            target = find_pool(fb.pool)
            if target is None or not target.replicas:
                continue
            power_demand.note(target)  # wanted, even if it is off right now
            if st.pool_at_capacity(target):
                continue
            if not st.pool_has_healthy(target):
                continue
            if parsed is not None:
                rewritten = dict(parsed)
                rewritten["model"] = target.model_id
                new_body = json.dumps(rewritten).encode()
            else:
                rewritten = None
                new_body = body
            label = f"{frm}->{target.model_id}:{reason}"
            st.fallbacks[label] = st.fallbacks.get(label, 0) + 1
            resp = await dispatch_to_pool(
                req,
                target,
                path=path,
                body=new_body,
                parsed=rewritten,
                key=key,
                _visited=visited,
                allow_chain=False,
            )
            if resp.status_code < 500:
                return _tag_fallback(resp, frm, target.model_id)
            continue

        if fb.external:
            provider = st.ext_providers.get(fb.external)
            if provider is None or not st.provider_available(provider):
                continue
            label = f"{frm}->external:{provider.id}:{reason}"
            st.fallbacks[label] = st.fallbacks.get(label, 0) + 1
            resp = await dispatch_external(req, provider, path=path, parsed=parsed)
            if resp is not None:
                return _tag_fallback(resp, frm, f"external:{provider.id}")
            continue

    if reason == "capacity":
        return _reject_capacity(frm)
    return _reject_down(frm, reason)


def join_openai_url(base_url: str, path: str) -> str:
    """Join ``.../v1`` base with an incoming ``/v1/...`` path without doubling."""
    base = base_url.rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    if base.endswith("/v1") and (path == "/v1" or path.startswith("/v1/")):
        path = path[3:] or "/"
    return base + path


async def dispatch_external(req, provider: ExternalProvider, *, path: str, parsed: dict | None):
    """Proxy JSON (or multipart audio) to an OpenAI-compatible cloud provider.

    Returns a Response, or ``None`` so the caller can try the next chain entry.
    """
    import httpx

    # Per-request privacy guard, including requests to otherwise cloud-backed
    # chat pools. LAN-only callers fail closed if local capacity is unavailable.
    if str(req.headers.get("x-router-local-only", "")).lower() in ("1", "true"):
        return None

    rt = get()
    assert rt.client is not None
    upstream_url = join_openai_url(provider.base_url, path)
    headers = _clean_request_headers(req)
    headers["Authorization"] = f"Bearer {provider.api_key}"
    request_kwargs: dict = {}
    if path in ("/v1/audio/transcriptions", "/v1/audio/translations"):
        data: dict[str, str] = {}
        files: dict[str, tuple[str, bytes, str]] = {}
        form = await req.form()
        for name, value in form.multi_items():
            filename = getattr(value, "filename", None)
            if filename is not None:
                files[name] = (
                    filename or "audio",
                    await value.read(),
                    getattr(value, "content_type", None) or "application/octet-stream",
                )
            else:
                data[name] = str(value)
        data["model"] = provider.model
        request_kwargs = {"data": data, "files": files}
        headers.pop("content-type", None)
        headers.pop("Content-Type", None)
    else:
        if parsed is None:
            return None
        out = dict(parsed)
        out["model"] = provider.model
        if out.get("stream"):
            so = dict(out.get("stream_options") or {})
            so.setdefault("include_usage", True)
            out["stream_options"] = so
        request_kwargs = {"content": json.dumps(out).encode()}
        headers["Content-Type"] = "application/json"

    provider.active_requests += 1
    released = False

    def _release() -> None:
        nonlocal released
        if not released:
            released = True
            provider.active_requests = max(0, provider.active_requests - 1)

    try:
        upstream_req = rt.client.build_request(
            req.method,
            upstream_url,
            headers=headers,
            timeout=httpx.Timeout(
                provider.timeout_seconds,
                connect=rt.cfg.connect_timeout_seconds,
                read=provider.timeout_seconds,
            ),
            **request_kwargs,
        )
        upstream_resp = await rt.client.send(upstream_req, stream=True)
    except Exception as exc:  # noqa: BLE001
        _release()
        provider.cooldown_until = time.time() + rt.cfg.backpressure.retry_after_seconds
        log.warning("external %s connect failed: %s", provider.id, type(exc).__name__)
        return None

    if upstream_resp.status_code == 429:
        retry_after = upstream_resp.headers.get("retry-after")
        cd = float(retry_after) if (retry_after or "").isdigit() else rt.cfg.backpressure.retry_after_seconds
        provider.cooldown_until = time.time() + cd
        await upstream_resp.aclose()
        _release()
        rt.state.bump(f"external_429:{provider.id}")
        return None

    if upstream_resp.status_code >= 500:
        try:
            await upstream_resp.aread()
        except Exception:  # noqa: BLE001
            pass
        finally:
            await upstream_resp.aclose()
        _release()
        provider.cooldown_until = time.time() + rt.cfg.backpressure.retry_after_seconds
        rt.state.bump(f"external_5xx:{provider.id}")
        return None

    rt.state.bump(f"route:{provider.id}@{provider.model}")
    return await respond_from_upstream(
        upstream_resp,
        None,
        inter_token_timeout=provider.timeout_seconds,
        on_complete=_release,
    )
