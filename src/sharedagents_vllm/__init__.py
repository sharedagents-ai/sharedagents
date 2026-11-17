"""Local vLLM replica router for sharedagents.

Owns health, session affinity, capacity admission, replica failover,
local pool-to-pool fallback, and optional OpenAI-compatible cloud backup
(``external_providers``) when every local option is down or full.
"""

from __future__ import annotations

from .config import (
    ConfigError,
    default_config_path,
    load_config,
    packaged_example_path,
    parse_config,
)
from .models import (
    Backpressure,
    Config,
    ExternalProvider,
    FallbackEntry,
    InFlight,
    Pool,
    Replica,
)
from .routing import (
    State,
    derive_routing_key,
    pick_round_robin,
    rank_replicas,
)

__all__ = [
    "Backpressure",
    "Config",
    "ConfigError",
    "ExternalProvider",
    "FallbackEntry",
    "InFlight",
    "Pool",
    "Replica",
    "State",
    "create_app",
    "default_config_path",
    "derive_routing_key",
    "load_config",
    "packaged_example_path",
    "parse_config",
    "pick_round_robin",
    "rank_replicas",
]


def create_app(*args, **kwargs):
    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)
