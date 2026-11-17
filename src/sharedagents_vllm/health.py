"""Replica health: /health liveness plus kind-correct inference probes."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Any

from .models import NON_ROUTABLE, Config, Pool, Replica
from .runtime import get

log = logging.getLogger("sharedagents_vllm")

_ASR_PLACEHOLDER = "language None<asr_text>"
_WEDGE_REASONS = frozenset(
    {
        "inference_timeout",
        "inference_hang",
        "stuck_request",
        "health_timeout",
        "inference_shape",
    }
)


def _now() -> float:
    try:
        return get().now()
    except RuntimeError:
        return time.time()


def _health_timeout(cfg: Config):
    import httpx

    t = max(0.2, float(cfg.health_timeout_seconds))
    return httpx.Timeout(connect=min(2.0, t), read=t, write=t, pool=t)


def _infer_timeout(cfg: Config, pool: Pool):
    import httpx

    if pool.kind in ("asr", "tts"):
        t = max(0.2, float(cfg.inference_probe_timeout_asr_tts_seconds))
    else:
        t = max(0.2, float(cfg.inference_probe_timeout_seconds))
    return httpx.Timeout(connect=min(2.0, t), read=t, write=t, pool=t)


def classify_failure(reason: str, *, tcp_up: bool = False, meta_up: bool = False) -> str:
    """Map a probe/request error to down | wedged | overloaded."""
    low = (reason or "").lower()
    if "429" in low or "overloaded" in low:
        return "overloaded"
    if (
        any(
            tok in low
            for tok in (
                "connect",
                "refused",
                "name or service",
                "dns",
                "network unreachable",
                "norouteto",
                "connectionreset",
            )
        )
        and "timeout" not in low
    ):
        return "down"
    if any(tok in reason for tok in _WEDGE_REASONS) or "timeout" in low or "hang" in low:
        return "wedged"
    if tcp_up or meta_up:
        return "wedged"
    if "health_http" in low or "inference_http" in low:
        return "down"
    return "down"


def _enter_failed(replica: Replica, cfg: Config, now: float, failure_class: str) -> None:
    replica.health = "failed"
    replica.failure_class = failure_class
    replica.backoff_k += 1
    exp = cfg.health_backoff_base_seconds * (2 ** max(0, replica.backoff_k - 1))
    delay = min(cfg.health_backoff_cap_seconds, exp)
    replica.cooling_until = now + delay
    replica.cooldown_until = replica.cooling_until
    replica.consecutive_successes = 0


def apply_failure(
    replica: Replica,
    cfg: Config,
    reason: str,
    *,
    now: float | None = None,
    tcp_up: bool = False,
    meta_up: bool = False,
) -> None:
    """Apply a hard or wedge-class failure to the replica state machine."""
    now = _now() if now is None else now
    failure_class = classify_failure(reason, tcp_up=tcp_up, meta_up=meta_up)
    replica.last_error = f"{failure_class}: {reason}"[:200]
    replica.failure_class = failure_class
    replica.consecutive_successes = 0
    replica.consecutive_failures += 1
    replica.last_check = now
    if failure_class == "down":
        if replica.down_since <= 0:
            replica.down_since = now
    else:
        replica.down_since = 0.0

    skip_soft = failure_class == "wedged" and any(
        tok in reason
        for tok in (
            "inference_timeout",
            "inference_hang",
            "stuck_request",
            "health_timeout",
        )
    )
    if replica.health == "ok" and skip_soft:
        _enter_failed(replica, cfg, now, failure_class)
    elif replica.health == "ok":
        replica.health = "degraded"
    elif replica.health == "degraded":
        if replica.consecutive_failures >= cfg.health_fail_threshold:
            _enter_failed(replica, cfg, now, failure_class)
    else:
        _enter_failed(replica, cfg, now, failure_class)

    try:
        get().state.break_affinity_for_replica(replica.key())
        get().state.bump(f"health_fail:{replica.key()}:{failure_class}")
    except RuntimeError:
        pass


def apply_health_only_ok(replica: Replica, cfg: Config, *, now: float | None = None) -> None:
    """A /health 2xx never flips ok.

    After a down streak the host is reachable again: consume the down record
    (one-shot) and move to cooling so admission probes may run. If the streak
    hints at a host reboot, hold the model-load grace first. The grace is set
    exactly once — repeated /health 200s must not extend it forever.
    """
    now = _now() if now is None else now
    replica.last_check = now
    if replica.health not in NON_ROUTABLE:
        return
    if replica.failure_class != "down":
        return
    down_for = (now - replica.down_since) if replica.down_since else 0.0
    replica.down_since = 0.0
    # Host is reachable again; the "down" class no longer applies. The
    # admission probe will re-classify if the engine is actually wedged.
    replica.failure_class = ""
    replica.health = "cooling"
    if down_for >= cfg.host_reboot_hint_seconds:
        grace_until = now + cfg.model_load_grace_seconds
        replica.cooling_until = max(replica.cooling_until, grace_until)
        replica.cooldown_until = replica.cooling_until


def apply_inference_success(
    replica: Replica,
    cfg: Config,
    *,
    now: float | None = None,
) -> None:
    """Kind-correct probe success. Mandatory for re-admission."""
    now = _now() if now is None else now
    if replica.health == "cooling" and now < replica.cooling_until:
        return
    replica.last_error = ""
    replica.last_check = now
    replica.last_probe_at = now
    replica.last_inference_ok = now
    replica.consecutive_failures = 0
    replica.consecutive_successes += 1
    replica.down_since = 0.0

    if replica.health == "degraded":
        if replica.consecutive_successes >= cfg.health_soft_recover_threshold:
            replica.health = "ok"
            replica.failure_class = ""
            replica.backoff_k = 0
            replica.cooling_until = 0.0
            replica.ever_admitted = True
    elif replica.health in NON_ROUTABLE or not replica.ever_admitted:
        need = cfg.health_recover_threshold
        if replica.consecutive_successes >= need:
            replica.health = "ok"
            replica.failure_class = ""
            replica.backoff_k = 0
            replica.cooling_until = 0.0
            replica.cooldown_until = 0.0
            replica.ever_admitted = True
    elif replica.health == "ok":
        replica.failure_class = ""


def probe_model_id(pool: Pool, replica: Replica | None = None) -> str:
    served = replica.upstream_model if replica is not None else None
    return (pool.probe_model or served or pool.upstream_model or pool.model_id).strip()


def apply_decision_failure(
    replica: Replica,
    pool: Pool,
    cfg: Config,
    reason: str,
    *,
    now: float | None = None,
) -> None:
    """Soft failure for advisory decision pools (constraint 3).

    A scorer restart closes the port for ~40 s. The generic ``apply_failure``
    would enter ``failed`` and back the replica off 30 s -> 600 s, so a couple of
    restarts would exclude it for minutes. It would also drop the pool into the
    ``/health`` ``down`` list, letting an *advisory* scorer degrade the router's
    global health signal — which is exactly what must never happen.

    So a decision replica instead goes ``degraded`` (still routable), keeps
    ``backoff_k`` at zero, and caps re-probe cooldown at the pool's outage
    tolerance: long enough not to storm the scorer, short enough to notice
    recovery inside a single restart window.
    """
    now = _now() if now is None else now
    text = str(reason or "inference_fail")
    replica.last_error = text[:200]
    replica.last_check = now
    replica.consecutive_failures += 1
    replica.consecutive_successes = 0
    replica.backoff_k = 0
    replica.health = "degraded"
    if text.startswith("connect"):
        replica.failure_class = "down"
        if not replica.down_since:
            replica.down_since = now
    else:
        replica.failure_class = "wedged"
        replica.down_since = 0.0
    cap = float(getattr(pool, "outage_tolerance_seconds", 0.0) or 45.0)
    replica.cooldown_until = now + cap
    replica.cooling_until = 0.0
    try:
        get().state.break_affinity_for_replica(replica.key())
        get().state.bump(f"health_soft_fail:{replica.key()}")
    except RuntimeError:
        pass


def inference_probe_allowed(pool: Pool, cfg: Config) -> bool:
    if not cfg.inference_probe_enabled:
        return False
    return pool.inference_probe is not False


def should_skip_inference(replica: Replica, cfg: Config, now: float) -> bool:
    if replica.active > 0 and replica.last_user_success > 0:
        if (now - replica.last_user_success) < cfg.inference_probe_seconds:
            return True
    return False


def _stagger_offset(replica: Replica, interval: float) -> float:
    digest = hashlib.blake2b(replica.key().encode(), digest_size=4).digest()
    return (int.from_bytes(digest, "big") % max(1, int(interval * 1000))) / 1000.0


def _llm_success_shape(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    first = choices[0] if isinstance(choices[0], dict) else {}
    msg = first.get("message") if isinstance(first.get("message"), dict) else {}
    content = msg.get("content")
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or first.get("reasoning")
    if isinstance(content, str) and _ASR_PLACEHOLDER in content and not reasoning:
        return False
    if isinstance(content, str) and content.strip():
        return True
    return bool(isinstance(reasoning, str) and reasoning.strip())


def _embed_success_shape(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return False
    first = data[0] if isinstance(data[0], dict) else {}
    emb = first.get("embedding")
    return bool(emb)


def _rerank_success_shape(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        return False
    first = results[0] if isinstance(results[0], dict) else {}
    score = first.get("relevance_score")
    return isinstance(score, (int, float))


def _models_lists(payload: Any, model_id: str) -> bool:
    if not isinstance(payload, dict):
        return False
    data = payload.get("data")
    if not isinstance(data, list):
        return False
    ids = []
    for row in data:
        if isinstance(row, dict) and row.get("id"):
            ids.append(str(row["id"]))
    if not ids:
        return False
    if not model_id:
        return True
    return model_id in ids or any(model_id in i for i in ids)


async def check_http_health(replica: Replica, pool: Pool) -> tuple[bool, str]:
    """GET /health with the dedicated health timeout. Returns (ok, reason)."""
    rt = get()
    client = rt.client
    if client is None:
        return False, "no http client"
    t0 = time.monotonic()
    try:
        r = await client.get(
            f"{replica.url}/health",
            timeout=_health_timeout(rt.cfg),
            headers={"X-Router-Probe": "1"},
        )
        replica.last_probe_latency_s = time.monotonic() - t0
        replica.last_probe_at = rt.now()
        if 200 <= r.status_code < 300:
            return True, ""
        return False, f"health_http {r.status_code}"
    except Exception as exc:  # noqa: BLE001
        replica.last_probe_latency_s = time.monotonic() - t0
        replica.last_probe_at = rt.now()
        name = type(exc).__name__
        if "timeout" in name.lower() or "timeout" in str(exc).lower():
            return False, "health_timeout"
        if "connect" in name.lower() or "refused" in str(exc).lower():
            return False, f"connect:{name}"
        return False, f"{name}: {exc}"[:180]


async def _post_json(url: str, payload: dict, timeout) -> tuple[int | None, Any, str]:
    rt = get()
    client = rt.client
    if client is None:
        return None, None, "no http client"
    try:
        r = await client.post(
            url,
            json=payload,
            timeout=timeout,
            headers={"X-Router-Probe": "1", "Content-Type": "application/json"},
        )
        try:
            body = r.json()
        except Exception:  # noqa: BLE001
            body = None
        return r.status_code, body, ""
    except Exception as exc:  # noqa: BLE001
        name = type(exc).__name__
        if "timeout" in name.lower() or "timeout" in str(exc).lower():
            return None, None, "inference_timeout"
        if "connect" in name.lower() or "refused" in str(exc).lower():
            return None, None, f"connect:{name}"
        return None, None, f"{name}: {exc}"[:180]


async def _get_json(url: str, timeout) -> tuple[int | None, Any, str]:
    rt = get()
    client = rt.client
    if client is None:
        return None, None, "no http client"
    try:
        r = await client.get(url, timeout=timeout, headers={"X-Router-Probe": "1"})
        try:
            body = r.json()
        except Exception:  # noqa: BLE001
            body = None
        return r.status_code, body, ""
    except Exception as exc:  # noqa: BLE001
        name = type(exc).__name__
        if "timeout" in name.lower() or "timeout" in str(exc).lower():
            return None, None, "inference_timeout"
        if "connect" in name.lower() or "refused" in str(exc).lower():
            return None, None, f"connect:{name}"
        return None, None, f"{name}: {exc}"[:180]


async def _probe_by_kind(
    replica: Replica,
    pool: Pool,
    model: str,
    timeout: float,
    cfg: Config,
) -> tuple[bool, str]:
    """Kind-correct probe dispatch. Never chat-probes ASR (REQ-006)."""
    kind = pool.kind
    ok = False
    reason = ""

    if kind == "llm":
        status, body, err = await _post_json(
            f"{replica.url}/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "Reply with exactly: PONG"}],
                "max_tokens": 16,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout,
        )
        if err:
            reason = err
        elif status is None or not (200 <= status < 300):
            reason = f"inference_http {status}"
        elif not _llm_success_shape(body):
            reason = "inference_shape"
        else:
            ok = True
    elif kind == "decision":
        # Advisory scorer: probe with an explicit checkpoint (constraint 4) and
        # classify the shared-503 ``detail`` rather than blanket-failing on the
        # status code (constraint 2). A busy scorer is up, just occupied.
        from .decision import probe_payload, verdict

        status, body, err = await _post_json(
            f"{replica.url}/score",
            probe_payload(model, pool),
            timeout,
        )
        ok, reason = verdict(status, body, err)
    elif kind == "embedding":
        status, body, err = await _post_json(
            f"{replica.url}/v1/embeddings",
            {"model": model, "input": "ping"},
            timeout,
        )
        if err:
            reason = err
        elif status is None or not (200 <= status < 300):
            reason = f"inference_http {status}"
        elif not _embed_success_shape(body):
            reason = "inference_shape"
        else:
            ok = True
    elif kind == "reranker":
        status, body, err = await _post_json(
            f"{replica.url}/v1/rerank",
            {"model": model, "query": "hello", "documents": ["world"]},
            timeout,
        )
        if err:
            reason = err
        elif status is None or not (200 <= status < 300):
            reason = f"inference_http {status}"
        elif not _rerank_success_shape(body):
            reason = "inference_shape"
        else:
            ok = True
    elif kind == "asr":
        h_ok, h_reason = await check_http_health(replica, pool)
        if not h_ok:
            reason = h_reason or "health_http"
        else:
            status, body, err = await _get_json(
                f"{replica.url}/v1/models",
                _health_timeout(cfg),
            )
            if err:
                reason = err
            elif status is None or not (200 <= status < 300):
                reason = f"inference_http {status}"
            elif not _models_lists(body, model):
                reason = "inference_shape"
            else:
                ok = True
    elif kind == "tts":
        status, body, err = await _get_json(
            f"{replica.url}/v1/models",
            _health_timeout(cfg),
        )
        if err:
            reason = err
        elif status is None or not (200 <= status < 300):
            reason = f"inference_http {status}"
        elif body is not None and not _models_lists(body, model) and not _models_lists(body, ""):
            reason = "inference_shape"
        else:
            ok = status is not None and 200 <= status < 300
    else:
        reason = f"unknown kind {kind}"

    return ok, reason


async def check_inference(replica: Replica, pool: Pool) -> tuple[bool, str]:
    """Kind-correct inference (or ASR/TTS metadata bundle). Never chat-probes ASR."""
    rt = get()
    cfg = rt.cfg
    model = probe_model_id(pool, replica)
    timeout = _infer_timeout(cfg, pool)
    t0 = time.monotonic()

    # Track the probe as inflight so vllm_mux_probe_inflight is truthful and
    # probes stay separate from user counters (REQ-010).
    fid = None
    try:
        fid = rt.state.begin_inflight(
            pool.model_id,
            replica.key(),
            "/probe",
            time.monotonic(),
            is_probe=True,
        )
    except Exception:  # noqa: BLE001
        fid = None
    try:
        ok, reason = await _probe_by_kind(replica, pool, model, timeout, cfg)
    finally:
        if fid is not None:
            rt.state.end_inflight(fid)

    replica.last_probe_latency_s = time.monotonic() - t0
    replica.last_probe_at = rt.now()
    try:
        result = "ok" if ok else (reason or "fail")
        get().state.bump_probe(pool.kind, result)
    except RuntimeError:
        pass
    return ok, reason


async def check_replica(replica: Replica, pool: Pool | None = None) -> None:
    """Compatibility wrapper: /health tick, then inference if due."""
    get()
    if pool is None:
        pool = _pool_for(replica)
    if pool is None:
        replica.last_error = "no pool"
        return
    await _health_tick(replica, pool)
    await _inference_tick(replica, pool, force=False)


def _pool_for(replica: Replica) -> Pool | None:
    try:
        rt = get()
    except RuntimeError:
        return None
    for pool in rt.cfg.pools.values():
        if replica in pool.replicas or any(r.key() == replica.key() for r in pool.replicas):
            return pool
    return None


async def _health_tick(replica: Replica, pool: Pool) -> None:
    rt = get()
    cfg = rt.cfg
    now = rt.now()
    # Honour health_check_seconds: the 5s loop tick is for sweeper
    # granularity; /health probes run at the configured interval.
    if replica.last_check > 0 and (now - replica.last_check) < float(cfg.health_check_seconds):
        return
    ok, reason = await check_http_health(replica, pool)
    if ok:
        apply_health_only_ok(replica, cfg, now=now)
        return
    if pool.kind == "decision":
        # Advisory pool: never enter failed/long-backoff on a restart (constraint 3).
        apply_decision_failure(replica, pool, cfg, reason, now=now)
        return
    meta_up = False
    tcp_up = reason in ("health_timeout",) or reason.startswith("health_http")
    apply_failure(replica, cfg, reason, now=now, tcp_up=tcp_up, meta_up=meta_up)


async def _inference_tick(replica: Replica, pool: Pool, *, force: bool) -> None:
    rt = get()
    cfg = rt.cfg
    now = rt.now()
    if not inference_probe_allowed(pool, cfg):
        return
    if replica.failure_class == "down" and replica.health in NON_ROUTABLE:
        return
    if not force:
        if now < replica.next_inference_at:
            return
        if should_skip_inference(replica, cfg, now):
            replica.next_inference_at = now + cfg.inference_probe_seconds
            return
        if replica.health == "cooling" and now < replica.cooling_until:
            return
        if replica.health == "failed" and now < replica.cooling_until:
            return
    ok, reason = await check_inference(replica, pool)
    interval = float(cfg.inference_probe_seconds)
    if replica.health in NON_ROUTABLE and replica.failure_class == "wedged":
        interval = min(
            cfg.health_backoff_cap_seconds,
            max(interval, cfg.health_backoff_base_seconds * (2 ** max(0, replica.backoff_k - 1))),
        )
    replica.next_inference_at = now + interval
    if ok:
        apply_inference_success(replica, cfg, now=now)
    else:
        if pool.kind == "decision":
            # Advisory pool: soft, capped, always-routable failure (constraint 3).
            apply_decision_failure(
                replica,
                pool,
                cfg,
                reason or "inference_fail",
                now=now,
            )
        else:
            meta_up = reason in ("inference_timeout", "inference_shape", "inference_hang")
            apply_failure(replica, cfg, reason or "inference_fail", now=now, meta_up=meta_up)


async def probe_cycle() -> None:
    """One pass: /health all replicas, then due inference probes. Not user traffic."""
    rt = get()
    pairs = [(pool, r) for pool in rt.cfg.pools.values() for r in pool.replicas]
    await asyncio.gather(*(_health_tick(r, p) for p, r in pairs))
    infer = []
    for pool, replica in pairs:
        infer.append(_inference_tick(replica, pool, force=False))
    if infer:
        await asyncio.gather(*infer)
    try:
        rt.state.sweep_stuck(rt.mono())
    except Exception:  # noqa: BLE001
        log.exception("stuck sweeper failed")


async def health_loop() -> None:
    rt = get()
    now = rt.now()
    for pool in rt.cfg.pools.values():
        interval = float(rt.cfg.inference_probe_seconds)
        for replica in pool.replicas:
            # Spread first inference probes across the whole interval so
            # replicas stay desynchronised (REQ-009: no probe storm).
            replica.next_inference_at = now + _stagger_offset(replica, interval)
    await probe_cycle()
    while True:
        await asyncio.sleep(min(5, rt.cfg.health_check_seconds))
        try:
            await probe_cycle()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("health loop iteration failed")
