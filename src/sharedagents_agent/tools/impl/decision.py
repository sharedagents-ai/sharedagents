"""Native typed decisions backed by the same catalogue as the Web UI."""

from __future__ import annotations

import json
import time
from typing import Any

from sharedagents_core import decision

from ._utils import ToolResult


def run_decision(args: dict[str, Any]) -> ToolResult:
    started = time.perf_counter()
    try:
        action = args.get("action")
        if action == "list":
            catalog = decision.build_catalog()
            return ToolResult(
                json.dumps(
                    {
                        "models": catalog["models"],
                        "default_model": catalog["default_model"],
                    }
                )
            )
        if action != "score":
            raise decision.DecisionRequestError("action must be list or score")
        payload = {k: args[k] for k in ("model", "state", "questions", "checkpoint") if k in args}
        if not payload.get("model"):
            payload["model"] = decision.build_catalog()["default_model"]
        payload = decision.normalize_score_payload(payload)
        status, body = decision.score_decision(payload)
        body = {**body, "request_latency_ms": round((time.perf_counter() - started) * 1000)}
        return ToolResult(json.dumps(body), ok=status < 400)
    except decision.DecisionRequestError as exc:
        return ToolResult(json.dumps({"error": "invalid_request", "message": str(exc)}), ok=False)
    except Exception:
        # Provider configuration and user context must not leak in tracebacks.
        return ToolResult(
            json.dumps({"error": "decision_unavailable", "message": "Decision service is temporarily unavailable"}),
            ok=False,
        )
