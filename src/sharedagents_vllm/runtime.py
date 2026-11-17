"""Process-global router runtime (config + state + optional httpx client)."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from .config import resolve_external_keys
from .models import Config
from .power_demand import Notifier
from .routing import State


class Runtime:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.state = State(cfg)
        self.power_demand = Notifier(cfg.power_demand, self.state)
        self.client: Any = None
        self.start_health: bool = True
        self.now_fn: Callable[[], float] = time.time
        self.mono_fn: Callable[[], float] = time.monotonic

    def now(self) -> float:
        return self.now_fn()

    def mono(self) -> float:
        return self.mono_fn()


current: Runtime | None = None


def bind(
    cfg: Config,
    *,
    client: Any = None,
    start_health: bool = True,
    environ: dict[str, str] | None = None,
) -> Runtime:
    global current
    resolve_external_keys(cfg, environ)
    rt = Runtime(cfg)
    rt.client = client
    rt.start_health = start_health
    current = rt
    return rt


def get() -> Runtime:
    if current is None:
        raise RuntimeError("sharedagents_vllm runtime is not bound")
    return current


def find_pool(model_id: str):
    rt = get()
    return rt.cfg.pools.get(model_id)
