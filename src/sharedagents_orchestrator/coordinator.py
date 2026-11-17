"""Turn & Lifecycle State Coordinator."""

from __future__ import annotations

import logging
from pathlib import Path

from sharedagents_orchestrator.telemetry import GenerationTelemetry, compute_generation_metrics
from sharedagents_session.session_writer import SessionWriter

_log = logging.getLogger(__name__)


class TurnCoordinator:
    """Coordinates turn execution, row bounds, and AI generation logging."""

    def __init__(self, sessions_dir: Path, session_hash: str, *, agent_name: str, model: str):
        self.sessions_dir = Path(sessions_dir)
        self.sid = str(session_hash)
        self.agent_name = agent_name
        self.model = model
        self.writer = SessionWriter(self.sessions_dir, self.sid)

    def start_turn(self, turn_idx: int, user_prompt: str) -> None:
        """Start a new turn."""
        self.writer.turn_started(turn_idx=turn_idx, user=user_prompt)

    def record_generation(
        self,
        *,
        turn_idx: int,
        round_idx: int,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        thinking_tokens: int = 0,
        duration_s: float,
        ttft_s: float = 0.0,
        overhead_s: float = 0.0,
        finish_reason: str = "stop",
        start_row_idx: int = 0,
        end_row_idx: int = 0,
    ) -> GenerationTelemetry:
        """Record and return telemetry for an AI model call."""
        derived = compute_generation_metrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking_tokens=thinking_tokens,
            duration_s=duration_s,
            ttft_s=ttft_s,
            model=self.model,
            provider=provider,
        )

        telemetry = GenerationTelemetry(
            session_hash=self.sid,
            turn_idx=turn_idx,
            round_idx=round_idx,
            agent_name=self.agent_name,
            model=self.model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking_tokens=thinking_tokens,
            cost_usd=derived["cost_usd"],
            duration_s=duration_s,
            speed_tok_s=derived["speed_tok_s"],
            ttft_s=ttft_s,
            overhead_s=overhead_s,
            finish_reason=finish_reason,
            start_row_idx=start_row_idx,
            end_row_idx=end_row_idx,
        )
        return telemetry

    def end_turn(self, turn_idx: int, finish_reason: str = "stop") -> None:
        """Finalize the active turn."""
        self.writer.turn_ended(turn_idx=turn_idx, metrics={"finish_reason": finish_reason})
