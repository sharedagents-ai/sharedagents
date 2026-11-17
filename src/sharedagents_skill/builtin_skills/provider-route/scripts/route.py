#!/usr/bin/env python3
"""Print the live provider-route snapshot (compact text; --json for the full digest)."""

from __future__ import annotations

import json
import math
import os
import sys
from collections.abc import Callable
from typing import Any

# Points of acceptable pace above the lowest dispatchable measured pace.
PACE_BAND = 25

# Static policy, deliberately editable. Match model IDs, never inferred task
# difficulty or engine names alone (one engine can run several model tiers).
CAPABILITY_TIERS = {
    "claude-opus-5-5": 3,
    "gpt-6.1-sol": 3,
    "gemini-4-argon": 3,
    "grok-4.7": 2,
    "glm-5.3": 2,
    "mimo-v2.6-pro": 2,
    "qwen3.8-27b": 1,
}


def capability_tier(model: str) -> int:
    """Known model tier; unknown models have no capability preference.

    Accept provider-qualified IDs and explicit variant suffixes such as
    gemini-4-argon-high and qwen3.8-27b-fp8. Longest match wins.
    """
    model = str(model or "").lower().rsplit("/", 1)[-1]
    for name in sorted(CAPABILITY_TIERS, key=len, reverse=True):
        if model == name or model.startswith(name + "-"):
            return CAPABILITY_TIERS[name]
    return 0


def pace_band() -> float:
    """Environment override is read per digest, including long-lived servers."""
    band = float(os.environ.get("SHAREDAGENTS_PROVIDER_ROUTE_PACE_BAND", PACE_BAND))
    if not math.isfinite(band) or band < 0:
        raise ValueError("SHAREDAGENTS_PROVIDER_ROUTE_PACE_BAND must be finite and nonnegative")
    return band


def rank_engines(
    rows: list[dict[str, Any]],
    *,
    band: float,
    pace_key: Callable[[dict[str, Any]], tuple],
) -> list[dict[str, Any]]:
    """Tier first inside ONE band anchored at the dispatchable minimum.

    Outside the inclusive band, retain the existing pace/reset/capacity order.
    Unknown pace never enters the band. Compare percentage subscriptions only;
    dollar balances retain their existing independent ranking.
    """
    measured = [row for row in rows if row.get("ranking_pace") is not None and row.get("score_unit") == "percent"]
    ceiling = min((row["ranking_pace"] for row in measured), default=-math.inf) + band

    def key(row: dict[str, Any]) -> tuple:
        pace = row.get("ranking_pace")
        if row.get("score_unit") == "percent" and pace is not None and pace <= ceiling:
            return (0, -row.get("capability_tier", 0), pace_key(row))
        return (1, 0, pace_key(row))

    return sorted(rows, key=key)


def main() -> int:
    from sharedagents_web.subscription_limits import routing_digest, routing_digest_text

    args = sys.argv[1:]
    digest = routing_digest(refresh="--refresh" in args)
    print(json.dumps(digest, indent=2) if "--json" in args else routing_digest_text(digest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
