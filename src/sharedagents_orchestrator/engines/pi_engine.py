"""Pi RPC implementation of the agent turn contract, owned by the orchestrator."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import time
from collections.abc import AsyncIterator
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sharedagents_agent.agents.engine import (
    AgentTurnEvent,
    Engine,
    EngineKind,
    EngineSessionState,
    HandoffAccepted,
    TextDelta,
    ToolCallResult,
    ToolCallStarted,
    TurnDone,
    TurnError,
    TurnRequest,
    TurnStarted,
)
from sharedagents_agent.delegation_gate import SecretEgressBlocked, guard_egress_payload

from .pi_rpc_adapter import PiRpcAdapter

_log = logging.getLogger(__name__)

# One model call over the whole context; far longer than a control command.
_COMPACT_TIMEOUT_S = 900.0


async def _no_events() -> AsyncIterator[dict[str, Any]]:
    return
    yield  # pragma: no cover - makes this an async generator


async def _compact(adapter: PiRpcAdapter) -> tuple[str, dict[str, Any]]:
    """Run Pi's own ``compact`` RPC command; reply text and usage metrics."""
    response = (
        await adapter.send(
            {"type": "compact"},
            wait=True,
            timeout=_COMPACT_TIMEOUT_S,
        )
        or {}
    )
    if not response.get("success"):
        # Nothing to compact, or already compact: Pi's own wording.
        return str(response.get("error") or "Pi did not compact."), {}
    data = response.get("data")
    data = data if isinstance(data, dict) else {}
    before, after = data.get("tokensBefore"), data.get("estimatedTokensAfter")
    if isinstance(before, int) and isinstance(after, int) and before > 0:
        text = f"Context compacted: {before:,} → {after:,} tokens."
    elif isinstance(before, int) and before > 0:
        text = f"Context compacted ({before:,} tokens before)."
    else:
        text = "Context compacted."
    usage = data.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    counts = {
        key: int(value)
        for key, value in usage.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
    }
    metrics: dict[str, Any] = {}
    if "input" in counts and "output" in counts:
        billed = counts["input"] + counts.get("cacheRead", 0) + counts.get("cacheWrite", 0)
        metrics = {
            "prompt_tokens": billed,
            "prompt_tokens_fresh": counts["input"],
            "prompt_tokens_cached": counts.get("cacheRead", 0),
            "prompt_tokens_cache_write": counts.get("cacheWrite", 0),
            "completion_tokens": counts["output"],
            "tokens_estimated": False,
        }
        if isinstance(after, int) and not isinstance(after, bool) and after > 0:
            # The context meter shows what is left after the compaction;
            # the compaction call itself is billed from the ``_billed`` set.
            metrics.update(
                prompt_tokens_billed=billed,
                prompt_tokens_billed_cached=counts.get("cacheRead", 0),
                prompt_tokens_billed_cache_write=counts.get("cacheWrite", 0),
                prompt_tokens=after,
                prompt_tokens_cached=0,
                prompt_tokens_cache_write=0,
            )
    return text, metrics


class PiRpcEngine(Engine):
    """Run RPC turns with durable Pi sessions and normalized frontend events."""

    kind = EngineKind.PI_RPC

    def __init__(self) -> None:
        self._active: dict[str, PiRpcAdapter] = {}
        self._stopped: set[str] = set()
        self._models: dict[str, str] = {}

    @property
    def name(self) -> str:
        return "pi"

    def is_available(self) -> tuple[bool, str]:
        return PiRpcAdapter().is_available()

    def available_models(self) -> list[dict[str, Any]]:
        from sharedagents_agent.discovery.cache_lock import discover_agent_models

        return discover_agent_models(self.name)

    def context_window(self, model: str | None = None) -> int:
        """Unknown until Pi reports model metadata; accept the session selector."""
        return 0

    def system_prompt_policy(
        self,
        state: EngineSessionState,
        *,
        current_system_prompt_sha256: str,
    ) -> str:
        return "first_turn" if not state.engine_session_id else "every_turn"

    def state_path(self, sessions_dir: Path, sharedagents_sid: str) -> Path:
        return Path(sessions_dir) / f"{sharedagents_sid}.{self.kind.value}.json"

    def read_state(self, sessions_dir: Path, sharedagents_sid: str) -> EngineSessionState:
        state = EngineSessionState(sharedagents_sid, self.kind)
        try:
            data = json.loads(self.state_path(sessions_dir, sharedagents_sid).read_text())
            if isinstance(data, dict):
                for field in fields(state):
                    if field.name not in {"engine", "sharedagents_session_id"}:
                        value = data.get(field.name)
                        if isinstance(value, str):
                            setattr(state, field.name, value)
        except (OSError, ValueError):
            pass
        return state

    def write_state(self, sessions_dir: Path, state: EngineSessionState) -> None:
        path = self.state_path(sessions_dir, state.sharedagents_session_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.pending")
            temporary.write_text(json.dumps(asdict(state)), encoding="utf-8")
            temporary.replace(path)
        except OSError:
            _log.warning("Could not persist Pi engine state")

    def set_config_option(self, sharedagents_sid: str, option_id: str, value: Any) -> None:
        if option_id == "model":
            self._models[sharedagents_sid] = str(value or "")

    def detach_session(self, sharedagents_sid: str) -> None:
        self._models.pop(sharedagents_sid, None)

    @staticmethod
    def _prompt(request: TurnRequest) -> tuple[str, list[dict[str, Any]]]:
        blocks = [request.system_prompt or ""]
        blocks.extend(str(block.get("content") or "") for block in request.preamble)
        if request.handoff is not None and request.handoff.rendered:
            blocks.append(
                "<sharedagents-canonical-handoff>\n" + request.handoff.rendered + "\n</sharedagents-canonical-handoff>"
            )
        images = []
        if request.prompt_parts:
            for part in request.prompt_parts:
                if part.get("type") == "text":
                    blocks.append(str(part.get("text") or ""))
                elif part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    header, separator, data = url.partition(",")
                    if not (separator and header.startswith("data:image/") and header.endswith(";base64")):
                        raise ValueError("Pi images must be inline base64 data URLs")
                    images.append({"type": "image", "data": data, "mimeType": header[5:].split(";")[0]})
                else:
                    raise ValueError("Unsupported Pi prompt content type")
        else:
            blocks.append(request.prompt)
        from sharedagents_agent.agents.external_cli import _foreground_instruction

        return "\n\n".join(block for block in blocks if block) + _foreground_instruction(), images

    async def run_turn(
        self,
        request: TurnRequest,
        state: EngineSessionState,
        *,
        sessions_dir: Path,
        turn_id: str,
        env: dict[str, str] | None = None,
    ) -> AsyncIterator[AgentTurnEvent]:
        turn_start = time.monotonic()
        first_token_at: float | None = None
        last_token_at: float | None = None
        delta_count = thinking_chars = content_chars = tool_events = assistant_messages = 0
        usage_complete = reasoning_complete = True
        model = request.model or self._models.get(state.sharedagents_session_id)
        yield TurnStarted(
            turn_id, state.sharedagents_session_id, state.engine_session_id, bool(state.engine_session_id)
        )
        adapter = None
        outcome = "ok"
        error = ""
        classification = "engine_exception"
        metrics: dict[str, Any] = {}
        try:
            if request.extra.get("_budget_exhausted_manual"):
                classification = "cost_budget_exhausted"
                raise ValueError("Pi cannot run tools after the session budget is exhausted")
            if request.mode not in {"", "auto"}:
                classification = "config"
                raise ValueError("Pi RPC supports only auto mode; ask/plan are unavailable")
            message, images = self._prompt(request)
            session_id = state.engine_session_id or (
                "sharedagents-" + hashlib.sha256(state.sharedagents_session_id.encode()).hexdigest()
            )
            cwd = state.spawn_cwd or request.cwd or os.getcwd()
            # Match the existing CLI/ACP sink guard before spawning a child.
            guard_egress_payload(self.kind, "\n\n".join((message, model or "", str(cwd), session_id)))
            adapter = PiRpcAdapter(
                command=["pi", "--session-id", session_id, "--session-dir", str((Path(sessions_dir) / "pi").resolve())],
                model=model,
                cwd=cwd,
                env=env,
            )
            self._active[turn_id] = adapter
            await adapter.start()
            state.engine_session_id = session_id
            state.spawn_cwd = cwd
            state.priming_status = "in_progress"
            self.write_state(sessions_dir, state)
            engine_command = bool(request.extra.get("_engine_command"))
            if engine_command:
                compact_text, compact_metrics = await _compact(adapter)
                metrics.update(compact_metrics)
                content_chars += len(compact_text)
                yield TextDelta(compact_text)
            async for event in _no_events() if engine_command else adapter.prompt(message, images=images):
                raw = event["raw"]
                kind = event["type"]
                if kind == "response" and event.get("command") == "prompt" and event.get("success"):
                    if request.handoff is not None and not request.extra.get("_handoff_consumed"):
                        request.extra["_handoff_consumed"] = True
                        yield HandoffAccepted(request.handoff.handoff_id, request.handoff.through_seq)
                elif kind in {"text_delta", "thinking_delta", "toolcall_delta"}:
                    chunk = event.get("tool_call", {}).get("delta", "") if kind == "toolcall_delta" else event["chunk"]
                    if chunk:
                        last_token_at = time.monotonic()
                        if first_token_at is None:
                            first_token_at = last_token_at
                        delta_count += 1
                    if kind == "thinking_delta":
                        thinking_chars += len(chunk)
                    elif kind == "text_delta":
                        content_chars += len(chunk)
                    if kind != "toolcall_delta":
                        yield TextDelta(chunk, "thinking" if kind == "thinking_delta" else "content")
                elif kind == "tool_execution_start":
                    tool_events += 1
                    yield ToolCallStarted(
                        str(raw.get("toolCallId") or ""), str(raw.get("toolName") or ""), raw.get("args") or {}
                    )
                elif kind == "tool_execution_end":
                    content = (raw.get("result") or {}).get("content") or []
                    yield ToolCallResult(
                        str(raw.get("toolCallId") or ""),
                        not raw.get("isError", False),
                        "\n".join(str(p.get("text") or "") for p in content),
                    )
                elif kind == "message_end":
                    msg = event.get("message") or {}
                    if msg.get("role") == "assistant":
                        assistant_messages += 1
                        # A retry may supersede an earlier provider failure.
                        reason = msg.get("stopReason")
                        outcome = "aborted" if reason == "aborted" else "error" if reason == "error" else "ok"
                        error = str(msg.get("errorMessage") or "Pi provider failed") if outcome == "error" else ""
                        state.actual_model = str(msg.get("model") or state.actual_model)
                        if msg.get("provider"):
                            metrics["provider"] = str(msg["provider"])
                        metrics["stop_reason"] = reason
                        usage = msg.get("usage")
                        usage = usage if isinstance(usage, dict) else {}
                        counts = {
                            key: int(value)
                            for key, value in usage.items()
                            if isinstance(value, (int, float))
                            and not isinstance(value, bool)
                            and math.isfinite(value)
                            and value >= 0
                        }
                        usage_complete &= "input" in counts and "output" in counts
                        reasoning_complete &= "reasoning" in counts
                        # Pi's input excludes both cache buckets; our prompt total includes them.
                        if "input" in counts:
                            counts["prompt"] = (
                                counts["input"] + counts.get("cacheRead", 0) + counts.get("cacheWrite", 0)
                            )
                        for source, target in (
                            ("prompt", "prompt_tokens"),
                            ("input", "prompt_tokens_fresh"),
                            ("output", "completion_tokens"),
                            ("reasoning", "reasoning_tokens"),
                            ("cacheRead", "prompt_tokens_cached"),
                            ("cacheWrite", "prompt_tokens_cache_write"),
                        ):
                            if source in counts:
                                metrics[target] = metrics.get(target, 0) + counts[source]
        except asyncio.CancelledError:
            outcome = "aborted"
        except SecretEgressBlocked as exc:
            outcome, error = "error", exc.reason
            classification = "secret_egress_blocked"
        except Exception as exc:
            outcome, error = "error", str(exc)
        finally:
            if adapter is not None:
                try:
                    await adapter.close()
                except Exception as exc:
                    outcome, error = "error", f"Pi shutdown failed: {exc}"
            self._active.pop(turn_id, None)
            if turn_id in self._stopped:
                outcome, error = "aborted", ""
            self._stopped.discard(turn_id)
        state.priming_status = "completed" if outcome == "ok" else "failed"
        state.priming_system_prompt_sha256 = request.system_prompt_sha256
        state.last_used_at = datetime.now(timezone.utc).isoformat()
        state.last_error = error
        self.write_state(sessions_dir, state)
        if error:
            yield TurnError(error, classification=classification)
        from sharedagents_agent.metrics_format import complete_metrics

        turn_s = time.monotonic() - turn_start
        reported_model = state.actual_model or model or ""
        metrics.update(engine=self.kind.value, model=reported_model, tool_events=tool_events)
        if "provider" not in metrics and model and "/" in model:
            metrics["provider"] = model.split("/", 1)[0]
        if not reasoning_complete:
            metrics.pop("reasoning_tokens", None)
        if assistant_messages:
            metrics["tokens_estimated"] = not usage_complete or (not reasoning_complete and thinking_chars > 0)
        ttft_s = first_token_at - turn_start if first_token_at is not None and delta_count >= 2 else None
        gen_window_s = (
            last_token_at - first_token_at
            if first_token_at is not None
            and last_token_at is not None
            and delta_count >= 3
            and assistant_messages == 1
            and not tool_events
            else None
        )
        metrics = complete_metrics(
            metrics,
            thinking_chars=thinking_chars,
            content_chars=content_chars,
            turn_s=turn_s,
            ttft_s=ttft_s,
            gen_window_s=gen_window_s,
        )
        if gen_window_s is None or gen_window_s < 0.25:
            # A tool round or terminal blob has no honest single decode window.
            metrics.pop("tps", None)
            if turn_s >= 0.25 and metrics.get("completion_tokens", 0) > 0:
                metrics["tps"] = round(metrics["completion_tokens"] / turn_s, 1)
                metrics["tps_kind"] = "effective"
        yield TurnDone(outcome, state.engine_session_id, metrics)

    async def abort(self, turn_id: str, *, force: bool = False) -> None:
        adapter = self._active.get(turn_id)
        if adapter is not None:
            self._stopped.add(turn_id)
            if force:
                await adapter.close(force=True)
            else:
                await adapter.abort()
