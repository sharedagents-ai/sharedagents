"""Routing key, replica rank, session affinity, and request counters."""

from __future__ import annotations

import hashlib
import time
import uuid
from collections import OrderedDict
from collections.abc import Iterator

from .models import NON_ROUTABLE, Config, ExternalProvider, InFlight, Pool, Replica


def derive_routing_key(headers: dict, body: dict | None) -> str:
    """Stable key: X-Session-Id → user → system+first user → prompt prefix."""
    lowered = {str(k).lower(): v for k, v in headers.items()}
    sid = lowered.get("x-session-id")
    if sid:
        return f"hdr:{sid}"

    if not body:
        return ""

    user = body.get("user")
    if isinstance(user, str) and user:
        return f"usr:{user}"

    msgs = body.get("messages")
    if isinstance(msgs, list) and msgs:
        parts: list[str] = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            content = str(content)[:1024]
            if role == "system":
                parts.append(f"S:{content}")
            elif role == "user":
                parts.append(f"U:{content}")
                if len(parts) >= 2:
                    break
        if parts:
            return "msgs:" + "\n".join(parts)

    prompt = body.get("prompt")
    if isinstance(prompt, str) and prompt:
        return f"prompt:{prompt[:256]}"
    if isinstance(prompt, list) and prompt:
        return f"prompt:{str(prompt[0])[:256]}"

    query = body.get("query")
    if isinstance(query, str) and query:
        return f"query:{query[:256]}"
    return ""


def _h(key: str, replica_url: str) -> bytes:
    return hashlib.blake2b(f"{key}|{replica_url}".encode(), digest_size=8).digest()


def _hrw_sort(replicas: list[Replica], key: str) -> list[Replica]:
    if len(replicas) <= 1:
        return list(replicas)
    return sorted(replicas, key=lambda r: _h(key, r.url), reverse=True)


def rank_replicas(key: str, pool: Pool, *, strict: bool | None = None) -> list[Replica]:
    """ok first (HRW), then degraded. Failed/cooling are never last-ditch when strict."""
    if strict is None:
        try:
            from .runtime import get

            strict = get().cfg.health_strict_routing
        except Exception:  # noqa: BLE001
            strict = True
    ok = [r for r in pool.replicas if r.health == "ok"]
    degraded = [r for r in pool.replicas if r.health == "degraded"]
    ranked = _hrw_sort(ok, key) + _hrw_sort(degraded, key)
    if strict:
        return ranked
    dead = [r for r in pool.replicas if r.health in NON_ROUTABLE]
    return ranked + _hrw_sort(dead, key)


def rank_by_priority(pool: Pool, *, strict: bool | None = None) -> list[Replica]:
    """Config order: ok replicas first, then degraded (then dead when not strict)."""
    if strict is None:
        try:
            from .runtime import get

            strict = get().cfg.health_strict_routing
        except Exception:  # noqa: BLE001
            strict = True
    ok = [r for r in pool.replicas if r.health == "ok"]
    degraded = [r for r in pool.replicas if r.health == "degraded"]
    if strict:
        return ok + degraded
    return ok + degraded + [r for r in pool.replicas if r.health in NON_ROUTABLE]


def priority_attempts(pool: Pool) -> Iterator[Replica]:
    """Recheck live health/capacity after I/O, attempting each host at most once."""
    attempted: set[str] = set()
    while True:
        replica = next(
            (
                r
                for r in rank_by_priority(pool)
                if r.key() not in attempted and (r.max_active <= 0 or r.active < r.max_active)
            ),
            None,
        )
        if replica is None:
            return
        attempted.add(replica.key())
        yield replica


def pick_round_robin(pool: Pool) -> Replica | None:
    """Fill first routable replica in config order that is under max_active."""
    healthy = [r for r in pool.replicas if r.healthy]
    if not healthy:
        return None
    preferred = [r for r in healthy if r.health == "ok"] or healthy
    for r in preferred:
        if r.max_active <= 0 or r.active < r.max_active:
            return r
    return preferred[0]


def key_hash(routing_key: str) -> str:
    if not routing_key:
        return ""
    return hashlib.blake2b(routing_key.encode("utf-8"), digest_size=8).hexdigest()


class State:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.counters: dict[str, int] = {}
        self.affinity: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self.session_count: dict[str, int] = {}
        self.cache: dict[tuple[str, str, str], int] = {}
        self.active_requests: dict[str, int] = {}
        self.fallbacks: dict[str, int] = {}
        self.backpressure: dict[str, int] = {}
        self.ext_providers: dict[str, ExternalProvider] = dict(cfg.external_providers)
        self.inflight: dict[str, InFlight] = {}
        self.probe_inflight: int = 0
        self.probe_total: dict[str, int] = {}
        self.stuck_trips: int = 0
        # replica_key -> mono time of last stuck trip; keeps the /health
        # stuck signal alive after the record is aged out (TTL = stuck_request_seconds).
        self.stuck_recent: dict[str, float] = {}

    def acquire_pool(self, pool_id: str) -> None:
        self.active_requests[pool_id] = self.active_requests.get(pool_id, 0) + 1

    def release_pool(self, pool_id: str) -> None:
        self.active_requests[pool_id] = max(0, self.active_requests.get(pool_id, 0) - 1)

    def pool_at_capacity(self, pool: Pool) -> bool:
        limit = pool.max_active_requests
        if limit <= 0:
            return False
        return self.active_requests.get(pool.model_id, 0) >= limit

    def pool_has_healthy(self, pool: Pool) -> bool:
        return any(r.healthy for r in pool.replicas)

    def begin_inflight(
        self,
        pool_id: str,
        replica_key: str,
        path: str,
        started_mono: float,
        *,
        is_probe: bool = False,
    ) -> str:
        fid = uuid.uuid4().hex
        rec = InFlight(
            id=fid,
            pool_id=pool_id,
            replica_key=replica_key,
            started_mono=started_mono,
            path=path,
            is_probe=is_probe,
        )
        self.inflight[fid] = rec
        if is_probe:
            self.probe_inflight += 1
        return fid

    def end_inflight(self, fid: str | None, replica: Replica | None = None) -> bool:
        """Release a record's replica and pool slots once (probes own neither)."""
        if not fid:
            return False
        rec = self.inflight.pop(fid, None)
        if rec is None:
            return False
        if rec.is_probe:
            self.probe_inflight = max(0, self.probe_inflight - 1)
        else:
            self.release_pool(rec.pool_id)
            if replica is not None:
                replica.active = max(0, replica.active - 1)
        return True

    def stuck_count(self, now_mono: float | None = None) -> int:
        now_mono = time.monotonic() if now_mono is None else now_mono
        limit = self.cfg.stuck_request_seconds
        n = 0
        for rec in self.inflight.values():
            if rec.is_probe:
                continue
            if now_mono - rec.started_mono >= limit:
                n += 1
        # Prune expired recent-trip entries, then count the rest.
        for k in [k for k, t in self.stuck_recent.items() if now_mono - t >= limit]:
            self.stuck_recent.pop(k, None)
        n += len(self.stuck_recent)
        return n

    def sweep_stuck(self, now_mono: float | None = None) -> list[str]:
        now_mono = time.monotonic() if now_mono is None else now_mono
        limit = self.cfg.stuck_request_seconds
        marked: list[str] = []
        replicas = {r.key(): r for pool in self.cfg.pools.values() for r in pool.replicas}
        for rec in list(self.inflight.values()):
            if rec.is_probe:
                continue
            if now_mono - rec.started_mono < limit:
                continue
            # Share the completion path so a late zombie completion cannot
            # release a replacement request's replica or pool slot (REQ-016).
            replica = replicas.get(rec.replica_key)
            self.end_inflight(rec.id, replica)
            if replica is not None:
                from .health import apply_failure

                apply_failure(replica, self.cfg, "stuck_request", tcp_up=True, meta_up=True)
            self.stuck_recent[rec.replica_key] = now_mono
            self.stuck_trips += 1
            marked.append(rec.replica_key)
        return marked

    def bump_probe(self, kind: str, result: str) -> None:
        key = f"{kind}:{result}"
        self.probe_total[key] = self.probe_total.get(key, 0) + 1

    def break_affinity_for_replica(self, replica_key: str) -> None:
        doomed = [kh for kh, (rk, _) in self.affinity.items() if rk == replica_key]
        for kh in doomed:
            self.break_affinity(kh)

    def provider_available(self, provider: ExternalProvider) -> bool:
        if not provider.enabled:
            return False
        if time.time() < provider.cooldown_until:
            return False
        return not (provider.max_active_requests > 0 and provider.active_requests >= provider.max_active_requests)

    def break_affinity(self, kh: str) -> None:
        if not kh:
            return
        prev = self.affinity.pop(kh, None)
        if prev is not None:
            replica, _ = prev
            self.session_count[replica] = max(0, self.session_count.get(replica, 0) - 1)

    def _sweep(self, now: float) -> None:
        cutoff = now - self.cfg.session_ttl_seconds
        while self.affinity:
            kh, (replica, seen) = next(iter(self.affinity.items()))
            if seen >= cutoff:
                break
            self.affinity.popitem(last=False)
            self.session_count[replica] = max(0, self.session_count.get(replica, 0) - 1)

    def lookup(self, kh: str) -> str | None:
        entry = self.affinity.get(kh)
        return entry[0] if entry is not None else None

    def assign(self, kh: str, replica: str, now: float) -> None:
        if not kh:
            return
        prev = self.affinity.get(kh)
        if prev is not None:
            prev_replica, _ = prev
            self.affinity[kh] = (replica, now)
            self.affinity.move_to_end(kh)
            if prev_replica != replica:
                self.session_count[prev_replica] = max(0, self.session_count.get(prev_replica, 0) - 1)
                self.session_count[replica] = self.session_count.get(replica, 0) + 1
            return
        if len(self.affinity) >= self.cfg.affinity_max:
            _old_kh, (old_replica, _) = self.affinity.popitem(last=False)
            self.session_count[old_replica] = max(0, self.session_count.get(old_replica, 0) - 1)
        self.affinity[kh] = (replica, now)
        self.session_count[replica] = self.session_count.get(replica, 0) + 1

    def bump(self, label: str) -> None:
        self.counters[label] = self.counters.get(label, 0) + 1

    def classify_cache(self, routing_key: str, served: str, model: str) -> str:
        if not routing_key:
            self.cache[(model, served, "none")] = self.cache.get((model, served, "none"), 0) + 1
            return "none"
        kh = key_hash(routing_key)
        prev_entry = self.affinity.get(kh)
        prev = prev_entry[0] if prev_entry else None
        reason = "cold" if prev is None else ("hit" if prev == served else "moved")
        self.cache[(model, served, reason)] = self.cache.get((model, served, reason), 0) + 1
        return reason

    def mark_unhealthy(self, replica: Replica, err: str) -> None:
        from .health import apply_failure

        apply_failure(replica, self.cfg, err[:200])
        replica.cooldown_until = max(
            replica.cooldown_until,
            time.time() + self.cfg.failure_cooldown_seconds,
        )
        self.break_affinity_for_replica(replica.key())
