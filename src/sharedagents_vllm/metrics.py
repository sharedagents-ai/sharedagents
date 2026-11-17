"""Prometheus text using v1 ``vllm_mux_*`` names (Usage-tab compat)."""

from __future__ import annotations

from .runtime import get


def _prom_safe(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def render_metrics() -> str:
    rt = get()
    st = rt.state
    lines: list[str] = []
    lines.append("# HELP vllm_mux_route_total Requests routed by upstream URL/label")
    lines.append("# TYPE vllm_mux_route_total counter")
    for label, count in st.counters.items():
        lines.append(f'vllm_mux_route_total{{upstream="{_prom_safe(label)}"}} {count}')
    lines.append("# HELP vllm_mux_replica_healthy 1 if replica is healthy else 0")
    lines.append("# TYPE vllm_mux_replica_healthy gauge")
    _health_num = {"ok": 3, "degraded": 2, "failed": 1, "cooling": 0}
    for pool in rt.cfg.pools.values():
        for r in pool.replicas:
            lines.append(
                f'vllm_mux_replica_healthy{{model="{_prom_safe(pool.model_id)}",'
                f'name="{_prom_safe(r.name)}",url="{_prom_safe(r.url)}"}} '
                f"{1 if r.healthy else 0}"
            )
    lines.append("# HELP vllm_mux_replica_health Replica state 3=ok 2=degraded 1=failed 0=cooling")
    lines.append("# TYPE vllm_mux_replica_health gauge")
    for pool in rt.cfg.pools.values():
        for r in pool.replicas:
            lines.append(
                f'vllm_mux_replica_health{{model="{_prom_safe(pool.model_id)}",'
                f'name="{_prom_safe(r.name)}",url="{_prom_safe(r.url)}",'
                f'failure_class="{_prom_safe(r.failure_class)}"}} '
                f"{_health_num.get(r.health, 0)}"
            )
    lines.append("# HELP vllm_mux_probe_total Kind-correct probe results")
    lines.append("# TYPE vllm_mux_probe_total counter")
    for label, count in st.probe_total.items():
        kind, _, result = label.partition(":")
        lines.append(f'vllm_mux_probe_total{{kind="{_prom_safe(kind)}",result="{_prom_safe(result)}"}} {count}')
    lines.append("# HELP vllm_mux_stuck_requests In-flight user requests older than stuck_request_seconds")
    lines.append("# TYPE vllm_mux_stuck_requests gauge")
    lines.append(f"vllm_mux_stuck_requests {st.stuck_count()}")
    lines.append("# HELP vllm_mux_probe_inflight In-flight health/inference probes")
    lines.append("# TYPE vllm_mux_probe_inflight gauge")
    lines.append(f"vllm_mux_probe_inflight {st.probe_inflight}")
    lines.append("# HELP vllm_mux_cache_total Routing-affinity cache outcomes")
    lines.append("# TYPE vllm_mux_cache_total counter")
    for (model, replica, reason), count in st.cache.items():
        lines.append(
            f'vllm_mux_cache_total{{model="{_prom_safe(model)}",'
            f'replica="{_prom_safe(replica)}",reason="{_prom_safe(reason)}"}} {count}'
        )
    lines.append("# HELP vllm_mux_affinity_entries Tracked session→replica entries (LRU)")
    lines.append("# TYPE vllm_mux_affinity_entries gauge")
    lines.append(f"vllm_mux_affinity_entries {len(st.affinity)}")
    lines.append("# HELP vllm_mux_session_count Live (non-expired) sessions homed per replica")
    lines.append("# TYPE vllm_mux_session_count gauge")
    all_replicas: set[str] = set()
    for pool in rt.cfg.pools.values():
        for r in pool.replicas:
            all_replicas.add(r.key())
    for rkey in sorted(all_replicas):
        lines.append(f'vllm_mux_session_count{{replica="{_prom_safe(rkey)}"}} {st.session_count.get(rkey, 0)}')
    lines.append("# HELP vllm_mux_active_requests In-flight requests per pool")
    lines.append("# TYPE vllm_mux_active_requests gauge")
    lines.append("# HELP vllm_mux_max_active_requests Configured per-pool concurrency cap (0=unlimited)")
    lines.append("# TYPE vllm_mux_max_active_requests gauge")
    for pool in rt.cfg.pools.values():
        lines.append(
            f'vllm_mux_active_requests{{pool="{_prom_safe(pool.model_id)}"}} {st.active_requests.get(pool.model_id, 0)}'
        )
        lines.append(f'vllm_mux_max_active_requests{{pool="{_prom_safe(pool.model_id)}"}} {pool.max_active_requests}')
    lines.append("# HELP vllm_mux_fallback_total Fallback events by route + reason")
    lines.append("# TYPE vllm_mux_fallback_total counter")
    for label, count in st.fallbacks.items():
        route, _, reason = label.rpartition(":")
        frm, _, to = route.partition("->")
        lines.append(
            f'vllm_mux_fallback_total{{from="{_prom_safe(frm)}",'
            f'to="{_prom_safe(to)}",reason="{_prom_safe(reason)}"}} {count}'
        )
    lines.append("# HELP vllm_mux_backpressure_total 503 rejections (all local options full) per pool")
    lines.append("# TYPE vllm_mux_backpressure_total counter")
    for model, count in st.backpressure.items():
        lines.append(f'vllm_mux_backpressure_total{{pool="{_prom_safe(model)}"}} {count}')
    return "\n".join(lines) + "\n"
