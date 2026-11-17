"""AI Model Call Telemetry & Metrics Tracker."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationTelemetry:
    """Telemetry payload for a single AI model completion / turn."""

    session_hash: str
    turn_idx: int
    round_idx: int
    agent_name: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cost_usd: float = 0.0
    duration_s: float = 0.0
    speed_tok_s: float = 0.0
    ttft_s: float = 0.0
    overhead_s: float = 0.0
    finish_reason: str = "stop"
    start_row_idx: int = 0
    end_row_idx: int = 0
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_hash": self.session_hash,
            "turn_idx": self.turn_idx,
            "round_idx": self.round_idx,
            "agent_name": self.agent_name,
            "model": self.model,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "cost_usd": self.cost_usd,
            "duration_s": self.duration_s,
            "speed_tok_s": self.speed_tok_s,
            "ttft_s": self.ttft_s,
            "overhead_s": self.overhead_s,
            "finish_reason": self.finish_reason,
            "start_row_idx": self.start_row_idx,
            "end_row_idx": self.end_row_idx,
            "created_at": self.created_at,
        }


def compute_generation_metrics(
    *,
    input_tokens: int,
    output_tokens: int,
    thinking_tokens: int = 0,
    duration_s: float,
    ttft_s: float = 0.0,
    model: str = "",
    provider: str = "",
) -> dict[str, float]:
    """Compute derived metrics: speed (tokens/sec) and cost ($)."""
    dur = max(duration_s, 0.001)
    gen_dur = max(duration_s - ttft_s, 0.001) if ttft_s > 0 else dur
    speed = float(output_tokens) / gen_dur

    # Standard estimation table ($ per 1M tokens)
    pricing = {
        "claude-3-7-sonnet": (3.0, 15.0),
        "claude-3-5-sonnet": (3.0, 15.0),
        "claude-3-5-haiku": (0.8, 4.0),
        "gpt-4o": (2.5, 10.0),
        "gpt-4o-mini": (0.15, 0.6),
        "gemini-2.0-flash": (0.1, 0.4),
        "gemini-2.5-pro": (1.25, 5.0),
    }

    inp_rate, out_rate = pricing.get(model, (0.0, 0.0))
    cost = ((input_tokens * inp_rate) + (output_tokens * out_rate)) / 1_000_000.0

    return {
        "speed_tok_s": round(speed, 2),
        "cost_usd": round(cost, 6),
    }
