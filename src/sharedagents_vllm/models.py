"""Dataclasses for local vLLM pools, replicas, and optional cloud fallbacks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

HealthState = Literal["ok", "degraded", "failed", "cooling"]
FailureClass = Literal["", "down", "wedged", "overloaded"]

ROUTABLE = frozenset({"ok", "degraded"})
NON_ROUTABLE = frozenset({"failed", "cooling"})

# --- advisory decision pools (kind="decision") -------------------------------
# Declared here, not in decision.py/config.py, so both can share them without
# importing each other (config -> decision -> runtime -> config is a cycle).
# "auto" is deliberately absent: the scorer's workflow auto-detection is off, so auto
# silently routes typed-decisions workflows to the english head with no error.
DECISION_CHECKPOINTS = ("english", "multilingual", "typed-decisions")
DECISION_DEFAULT_CHECKPOINT = "typed-decisions"
# Strictly serial: one lock, one uvicorn worker, ceiling ~1.6 req/s measured.
DECISION_MAX_REQUESTS_PER_SECOND = 1.6
# A restart closes the port for 36-42 s while all three checkpoints load.
DECISION_MIN_OUTAGE_TOLERANCE_SECONDS = 42.0


@dataclass
class InFlight:
    id: str
    pool_id: str
    replica_key: str
    started_mono: float
    path: str
    is_probe: bool = False


@dataclass
class Replica:
    name: str
    url: str  # base URL, no trailing slash
    last_check: float = 0.0
    last_error: str = ""
    cooldown_until: float = 0.0
    max_active: int = 0
    # Served name on this replica; overrides the pool's ``upstream_model`` so
    # one pool can span hosts that serve different builds of the same model.
    upstream_model: str | None = None
    active: int = 0
    health: HealthState = "cooling"
    failure_class: str = ""
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_probe_at: float = 0.0
    last_probe_latency_s: float = 0.0
    last_user_success: float = 0.0
    last_inference_ok: float = 0.0
    backoff_k: int = 0
    down_since: float = 0.0
    cooling_until: float = 0.0
    next_inference_at: float = 0.0
    ever_admitted: bool = False

    def key(self) -> str:
        return f"{self.name}@{self.url}"

    @property
    def healthy(self) -> bool:
        """Routable: ok or degraded. Failed/cooling are never last-ditch."""
        return self.health in ROUTABLE

    @property
    def routable(self) -> bool:
        return self.health in ROUTABLE


@dataclass
class FallbackEntry:
    """One hop: another local pool, or a named ``external_providers`` id."""

    pool: str | None = None
    external: str | None = None


@dataclass
class ExternalProvider:
    """OpenAI-compatible cloud backup. Key comes from ``api_key_env``, never YAML."""

    id: str
    base_url: str
    api_key_env: str
    model: str
    api_key: str = ""
    max_active_requests: int = 10
    timeout_seconds: int = 120
    cost_per_input_token: float = 0.0
    cost_per_output_token: float = 0.0
    active_requests: int = 0
    cooldown_until: float = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


@dataclass
class Backpressure:
    reject_status: int = 503
    retry_after_seconds: int = 5


@dataclass
class Pool:
    model_id: str
    kind: str  # llm | embedding | reranker | asr | tts | decision
    replicas: list[Replica] = field(default_factory=list)
    max_active_requests: int = 0
    fallbacks: list[FallbackEntry] = field(default_factory=list)
    read_timeout_seconds: int | None = None
    first_token_timeout_seconds: int | None = None
    inference_probe: bool | None = None
    probe_model: str | None = None
    # Name the replicas serve when it differs from the pool id, so a pool can
    # carry a short public name (``qwen3.8-flash-next``) for a long served one.
    upstream_model: str | None = None
    # ``affinity`` (default): sessions stick to their replica, new ones fill in
    # config order. ``priority``: every request takes the first replica in
    # config order that is routable and has a free slot (fastest host first).
    routing: str = "affinity"
    # Compatibility routes stay callable without appearing as picker choices.
    hidden: bool = False
    # Servers this pool needs powered on while it gets requests (power_demand webhook).
    power_servers: list[str] = field(default_factory=list)
    # --- advisory decision pools (kind="decision") only -------------------
    # A decision scorer is strictly serial (~1.6 req/s) and sheds rather than queues, so the
    # router owns the queue. See sharedagents_vllm/decision.py for the constraints.
    checkpoint: str = "typed-decisions"
    queue_timeout_seconds: float = 10.0
    queue_max_depth: int = 32
    rate_limit_per_second: float = 1.5
    connect_retry_seconds: float = 2.0
    busy_retry_seconds: float = 3.0
    outage_tolerance_seconds: float = 45.0


@dataclass
class PowerDemand:
    """Webhook that keeps servers on while their model is asked for (see power_demand.py)."""

    url: str
    token_env: str
    ttl_seconds: int = 900
    min_interval_seconds: float = 60.0
    timeout_seconds: float = 5.0


@dataclass
class Config:
    bind: str
    port: int
    health_check_seconds: int
    write_pool_timeout_seconds: int  # httpx write/pool phases, not a total deadline
    connect_timeout_seconds: int
    read_timeout_seconds: int
    first_token_timeout_seconds: int
    failure_cooldown_seconds: int
    session_ttl_seconds: float
    affinity_max: int
    pools: dict[str, Pool]
    backpressure: Backpressure = field(default_factory=Backpressure)
    external_providers: dict[str, ExternalProvider] = field(default_factory=dict)
    power_demand: PowerDemand | None = None
    health_timeout_seconds: int = 3
    inference_probe_seconds: int = 30
    inference_probe_timeout_seconds: int = 30
    inference_probe_timeout_asr_tts_seconds: int = 8
    inference_probe_enabled: bool = True
    health_fail_threshold: int = 2
    health_recover_threshold: int = 2
    health_soft_recover_threshold: int = 3
    health_backoff_base_seconds: int = 30
    health_backoff_cap_seconds: int = 600
    host_reboot_hint_seconds: int = 180
    model_load_grace_seconds: int = 180
    stuck_request_seconds: int = 300
    error_rate_window_seconds: int = 60
    error_rate_fail_ratio: float = 0.5
    health_strict_routing: bool = True
    host_restart_on_wedge: bool = False
    latency_degraded_ms: int = 15000
    # A replica that goes byte-silent mid-stream is asked whether its engine is
    # still generating (vLLM buffers array/object tool arguments whole). Probe
    # cadence, and the longest silence tolerated even while it generates.
    generation_probe_seconds: float = 15.0
    stream_silence_cap_seconds: float = 900.0
