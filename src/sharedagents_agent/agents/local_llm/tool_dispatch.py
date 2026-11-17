"""ToolDispatchMixin — extracted from the ``local_llm`` monolith (behaviour-preserving)."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from sharedagents_agent.tools.registry import is_agent_tool
from sharedagents_agent.tools.widget_adapt import is_ask_user_tool
from sharedagents_core.attachments import tool_image_message

from .state import (
    _DOOM_LOOP_REPEAT_THRESHOLD,
    AgentTurnEvent,
    _build_direct_delegate_preamble,
    _delegate_resume_available,
    _namespace_inner_call_id,
    _now_iso,
    wrap_untrusted,
)

#: Told to the model instead of an image no configured model can look at.
_TOOL_IMAGES_UNVIEWABLE = (
    "[image not shown] This model cannot view images and no vision-capable "
    "model is configured to describe them. Delegate this to a vision-capable "
    "agent, or ask the user what the image shows."
)


def _tool_image_label(name: str, args: Any) -> str:
    """What the image came from, for the caption: the path a tool read."""
    if isinstance(args, dict):
        for key in ("filePath", "file_path", "path"):
            value = args.get(key)
            if isinstance(value, str) and value:
                return value
    return name


def _egress_flag_delta(payload: dict) -> Any:
    """Owner-visible notice for a pre-egress sensitive-data flag.

    Assistant-message text, like a hook's system message: shown at every
    ``/tools`` level and kept in the transcript. Tool-stream text
    (``tool_delta``) only renders with tools set to ``stream``, so a notice
    sent there is invisible at the default setting. An entry of a fan-out is
    told apart by the agent it names, never by its batch index. Never part of
    the tool's result body.
    """
    from sharedagents_agent import delegation_gate

    from ..engine import TextDelta

    text = delegation_gate.sensitive_egress_notice_text(
        tuple(payload.get("classes") or ()),
        str(payload.get("agent") or ""),
    )
    return TextDelta(text=f"\n{text}\n", channel="content")


class ToolDispatchMixin:
    # -----------------------------------------------------------------
    # Tool dispatch
    # -----------------------------------------------------------------

    async def _dispatch_tool_calls(
        self,
        calls: list[dict[str, Any]],
        session: Any,
        messages: list[dict[str, Any]],
        turn: Any,
        round_record: dict[str, Any],
        promoted: set[str],
        *,
        hook_engine: Any = None,
        hook_ctx: dict[str, Any] | None = None,
    ) -> AsyncIterator[AgentTurnEvent]:
        """Run pending tool calls; mutate ``messages`` with results.

        Reference: acp_bridge.py L2370–2549 (parallel batch) and
        L2136–2369 (singleton). Behavior outline:

        1. **Permission gate** — yield a :class:`PermissionRequested`
           event; the Agent layer resolves it against the active
           mode (plan / ask / auto) and writes the decision back into
           ``ev.response``. Denied calls produce a synthetic
           ``tool_call_update`` failed result.
        2. **UserQuestion** — handled by ``_default_run_tool``
           on a name match, routing through ``_ask_user_outbound``
           (auto-mode auto-answers via :func:`_ask_user_auto_answers`).
        3. **Parallel grouping** — :meth:`_build_parallel_groups`
           splits the call list into safe batches (parallel-eligible
           calls grouped, stateful/conflicting ones serialized).
        4. **Dispatch** — for each group, fan out via
           :meth:`_execute_batch_parallel`. The single-call fallback
           uses :meth:`_execute_call_singleton`.
        5. **compact_context intercept** — when the tool name is
           ``compact_context`` (acp_bridge L2223), call
           :meth:`_model_requested_compaction` (a shrink-sized squeeze,
           announced as a visible compaction subagent when it removes
           anything) instead of spawning a runner. The squeezed message
           list MUTATES ``messages`` in place; the tool result reports the
           real outcome.
        6. **delegate budget** — cap external-CLI delegates
           (``delegate_claude`` / ``delegate_codex`` / ``delegate_gemini``)
           at ONE per turn per bridge (acp_bridge L2908–2912). The
           second call returns a synthetic failed tool_call_update
           telling the model to surface the prior reply instead of
           looping. ``delegate_agent`` and ``acp_call`` are NOT capped.
        7. **tool_search promotion** — when a call is ``tool_search``
           and succeeds, parse the selection and add the promoted tool
           names to ``promoted`` so the next round's tools_payload
           includes them (acp_bridge L1780–1798).
        8. **Streaming truncation guard** — if a call's arguments
           carry the ``_arguments_unparsed`` sentinel (worker repair
           gave up), short-circuit with a clean error rather than
           passing garbage to the runner (acp_bridge L2169–2221).

        Yields :class:`ToolCallStarted` / :class:`ToolCallResult` for
        every dispatched call so the gateway can translate them to ACP
        ``tool_call`` / ``tool_call_update`` events. A
        :class:`PermissionRequested` is yielded for every call whose
        ``PreToolUse`` hook didn't already decide — the Agent layer
        resolves it against the permission policy / ASK protocol.
        """
        from ..engine import (
            PermissionRequested,
            TextDelta,
            ToolCallResult,
            ToolCallStarted,
        )

        if not calls:
            return

        hook_ctx = hook_ctx or {}
        round_idx = int(round_record.get("round_idx") or 0)
        allowed_tools = self._runtime_allowed_tools(session)

        def _denied(call_id: str, name: str, reason: str) -> None:
            # Append a synthetic tool-result message so the model sees
            # the denial and can adapt next round. The single model-facing
            # sink for ALL denial/error paths (PreToolUse / PermissionRequest
            # / permission / argument-truncation). ``reason`` can be hook-
            # supplied and echo tool/external text, so it is fenced as
            # untrusted data — the framework's ``[denied]`` label stays
            # outside as the trusted signal. (The ToolCallResult yields on
            # these paths are display-only UI events, not model context.)
            fenced_reason = wrap_untrusted(reason, source=f"gate:{name}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": f"[denied] {fenced_reason}".rstrip(),
                }
            )
            round_record.setdefault("calls", []).append(
                {
                    "id": call_id,
                    "name": name,
                    "ok": False,
                    "denied": reason,
                }
            )

        # --- Gate every call (compact intercept → PreToolUse → policy) ---
        approved: list[dict[str, Any]] = []
        for raw in calls:
            norm = self._normalize_call(raw)
            call_id, name, args = norm["id"], norm["name"], norm["arguments"]

            # Thread the ORIGINAL user task into Agent calls so the dispatcher's
            # single-phase specialist reroute (runners._shim_agent_canonical) can
            # correct a generic misroute (explore/general/build) to the right
            # specialist. The orchestrator's paraphrase to the subagent strips the
            # intent, so the reroute needs the real task. Inert unless the task
            # matches a narrow specialist pattern (subagent_policy.specialist_intent).
            if is_agent_tool(name) and isinstance(args, dict) and "_sharedagents_user_task" not in args:
                _ut = getattr(turn, "user", "")
                if isinstance(_ut, list):
                    _ut = " ".join(str(p.get("text", "") if isinstance(p, dict) else p) for p in _ut)
                _ut = str(_ut or "").strip()
                if _ut:
                    args["_sharedagents_user_task"] = _ut
                    # If the orchestrator fell back to a GENERIC agent, compute the
                    # embedding semantic router's prediction here (async — the shim
                    # can't) and thread it as a hint the dispatcher reroute prefers.
                    # Fully fail-safe: any error leaves no hint and the dispatcher
                    # falls back to its regex detector.
                    _st = str(args.get("subagent_type") or "").strip().lower()
                    if _st in ("explore", "general"):
                        try:
                            from sharedagents_agent.routing_semantic import predict_specialist

                            _hint = await predict_specialist(_ut)
                            if _hint:
                                args["_sharedagents_route_hint"] = _hint
                        except Exception:
                            pass

            # The tool name owns the execution mode. Do not persist the
            # removed model argument (including stale replayed defaults).
            if is_agent_tool(name) and isinstance(args, dict):
                args.pop("background", None)
                runs = args.get("runs")
                for run in runs if isinstance(runs, list) else []:
                    if isinstance(run, dict):
                        run.pop("background", None)

            if name not in allowed_tools:
                # The permission set IS the tool set — a name
                # outside it is simply denied (no promotion, no discovery).
                # On the final handoff round the flag empties allowed_tools, so
                # give a budget-specific reason there instead of the allowlist
                # wording (which the model misreads as a permanent loss).
                if self._max_steps_handoff.get(session.hash):
                    reason = (
                        f"step budget exhausted — tools are disabled for "
                        f"the final round; summarize your progress in text "
                        f"instead of calling {name!r}"
                    )
                else:
                    reason = f"tool {name!r} is not in this agent's allowlist"
                yield ToolCallStarted(call_id=call_id, name=name, arguments=args)
                _denied(call_id, name, reason)
                yield ToolCallResult(
                    call_id=call_id,
                    ok=False,
                    content=f"[denied] {reason}",
                )
                continue

            # 0. Streaming-truncation guard: when the upstream cut off
            #    the tool call's ``arguments`` JSON mid-stream and the
            #    structural repair / parser also failed, ``arguments``
            #    carries a sole ``_arguments_unparsed`` key. Short-
            #    circuit with a clean, model-actionable error rather
            #    than passing garbage to the runner (which would
            #    surface a confusing arg-shape complaint).
            if isinstance(args, dict) and "_arguments_unparsed" in args and len(args) == 1:
                unparsed = str(args.get("_arguments_unparsed") or "")
                tail = unparsed[-160:] if len(unparsed) > 160 else unparsed
                err_msg = (
                    f"tool args malformed: streamed `arguments` JSON for "
                    f"`{name}` was truncated mid-stream and couldn't be "
                    f"auto-repaired ({len(unparsed)} chars; tail: "
                    f"{tail!r}). Retry with a smaller call (fewer items "
                    "or shorter strings) and avoid batching this tool "
                    "with another long-args tool in the same round."
                )
                yield ToolCallStarted(call_id=call_id, name=name, arguments=args)
                _denied(call_id, name, err_msg)
                yield ToolCallResult(
                    call_id=call_id,
                    ok=False,
                    content=f"[arguments truncated] {err_msg}",
                )
                continue

            # 1. compact_context intercept — squeeze in place, no runner.
            if name == "compact_context":
                yield ToolCallStarted(call_id=call_id, name=name, arguments=args)
                squeezed, result_text = self._model_requested_compaction(
                    session,
                    messages,
                )
                messages[:] = squeezed
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": result_text,
                    }
                )
                yield ToolCallResult(
                    call_id=call_id,
                    ok=True,
                    content=result_text,
                )
                continue

            # 1.5. ABSOLUTE: a FORBIDDEN call is an
            #      unappealable deny that runs BEFORE any hook, allowlist, or
            #      mode branch. No PreToolUse / PermissionRequest hook may even
            #      observe it, let alone allow it (``rm -rf /``, a fork bomb, a
            #      gateway self-kill, or a ``kill 0`` own-process-group kill).
            #      This single pre-hook chokepoint is the primary guard; the
            #      later per-hook forbidden re-checks
            #      (``_hook_allow_is_forbidden`` on the hook-allow bypass paths)
            #      are belt-and-braces, not the primary guard.
            if self._effective_criticality_for(name, args) == "forbidden":
                reason = (
                    f"forbidden: {name!r} classifies as FORBIDDEN (blast radius "
                    f"too large to ever allow); it is denied in "
                    f"every mode before any hook runs"
                )
                yield ToolCallStarted(call_id=call_id, name=name, arguments=args)
                _denied(call_id, name, reason)
                yield ToolCallResult(
                    call_id=call_id,
                    ok=False,
                    content=f"[denied: forbidden] {reason}",
                )
                continue

            # 2. PreToolUse hook (Claude Code). deny / exit-2 /
            #    decision:block => skip the call; allow => bypass the
            #    permission policy; ask / no-opinion => fall through.
            #    ``hookSpecificOutput.updatedInput`` lets a hook rewrite
            #    the tool args before dispatch — used by linter-style
            #    plugins that normalise paths, downgrade destructive
            #    flags, or splice safety options into ``Bash`` commands.
            hook_decision = ""
            if hook_engine is not None:
                pre = await self._fire_hook(
                    hook_engine,
                    hook_ctx,
                    "PreToolUse",
                    {"tool_name": name, "tool_input": args},
                )
                for sm in pre.system_messages:
                    yield TextDelta(text=f"[hook] {sm}\n", channel="content")
                if pre.blocked or pre.permission_decision == "deny":
                    reason = pre.block_reason or pre.permission_reason or "blocked by a PreToolUse hook"
                    yield ToolCallStarted(
                        call_id=call_id,
                        name=name,
                        arguments=args,
                    )
                    _denied(call_id, name, reason)
                    yield ToolCallResult(
                        call_id=call_id,
                        ok=False,
                        content=f"[denied by PreToolUse hook] {reason}",
                    )
                    continue
                if pre.updated_input is not None:
                    # Hook rewrote the args. Replace in place — the
                    # downstream criticality gate + runner see the
                    # rewritten dict, so a destructive ``rm -rf`` that
                    # the hook downgraded to ``rm -rfI`` flows through
                    # the rest of the pipeline correctly. Mutate
                    # ``norm["arguments"]`` so the dispatch step
                    # downstream picks the rewritten args up too.
                    args = dict(pre.updated_input)
                    norm["arguments"] = args
                hook_decision = pre.permission_decision  # "" | "allow" | "ask"

                # A PreToolUse hook `allow`
                # short-circuits the permission policy below (the
                # ``if hook_decision != "allow"`` gate), so evaluate()'s
                # forbidden hard-deny would never run and a hook could
                # green-light ``rm -rf /`` / a fork bomb / a gateway
                # self-kill. Re-check with the SAME host-side forbidden
                # classifier and convert the allow into a hard deny when it
                # trips — a hook can never override a FORBIDDEN verdict.
                if hook_decision == "allow" and self._hook_allow_is_forbidden(
                    name,
                    args,
                ):
                    reason = (
                        f"forbidden: a PreToolUse hook allowed {name!r}, but "
                        f"the call classifies as FORBIDDEN (blast radius too "
                        f"large to ever allow); this is an "
                        f"unappealable deny that no hook can override"
                    )
                    yield ToolCallStarted(
                        call_id=call_id,
                        name=name,
                        arguments=args,
                    )
                    _denied(call_id, name, reason)
                    yield ToolCallResult(
                        call_id=call_id,
                        ok=False,
                        content=f"[denied: forbidden] {reason}",
                    )
                    continue

            # 3. Permission policy via PermissionRequested — the Agent
            #    resolves it. Skipped when a PreToolUse hook said allow.
            if hook_decision != "allow":
                import asyncio as _asyncio

                fut: _asyncio.Future[str] = _asyncio.get_event_loop().create_future()
                # Compute effective criticality the cautious way:
                # combine the tool's static kind + a safety heuristic over the
                # command + any legacy/internal claim, then take the most
                # cautious result. Model-facing schemas do not expose a
                # criticality argument.
                effective_crit = self._effective_criticality_for(
                    name,
                    args,
                )
                # OpenCode doom_loop: identical tool+args repeated → ask.
                import hashlib as _hashlib
                import json as _json

                try:
                    _sig = (
                        f"{name}:"
                        f"{_hashlib.sha256(_json.dumps(args, sort_keys=True, default=str).encode()).hexdigest()[:12]}"
                    )
                except Exception:  # noqa: BLE001
                    _sig = str(name)
                _sid = getattr(session, "hash", "") or ""
                _hist = self._doom_loop_sigs.setdefault(_sid, [])
                _hist.append(_sig)
                if len(_hist) >= _DOOM_LOOP_REPEAT_THRESHOLD and _hist[-1] == _hist[-2] == _hist[-3]:
                    # The loop-breaker must NOT override an operator ``allow_always``.
                    # A *rememberable* call (a curated-safe tool at
                    # read_only/mutating criticality) is the only kind that CAN
                    # be under an allow_always, and the engine cannot see the
                    # per-session allowlist here — so it leaves such a call's
                    # criticality untouched and defers to the permission policy,
                    # which owns the allowlist (a remembered signature auto-
                    # allows; an unremembered one is still gated by
                    # the policy). Non-rememberable side-effecting calls
                    # (Bash / delegation / network / destructive / unknown) are
                    # the real wedged-model threat and DO escalate to at least
                    # destructive, forcing a fresh permission decision.
                    #
                    # Never escalate a host-resolved ``read_only`` call
                    # (BashOutput polling, todowrite): that was 36% of live
                    # destructive-gate false positives. Liveness belongs to
                    # the loop-detection/nudge path, not the safety channel.
                    from ..permissions_integration import (
                        PermissionPolicy as _DoomPolicy,
                    )

                    _rememberable = (
                        bool(name)
                        and name not in _DoomPolicy._BASH_TOOLS
                        and effective_crit in ("read_only", "mutating", "network")
                    )
                    if effective_crit != "read_only" and not _rememberable:
                        effective_crit = "destructive"
                # ``PermissionRequest`` hook (Claude Code spec) — fires
                # AFTER the cautious criticality merge and BEFORE the
                # Agent's permission gate. An ECC plugin can:
                #   * ``decision: block`` (exit 2) → outright deny the
                #     tool with the hook's reason as the model-facing
                #     denial body (skips the Agent gate entirely).
                #   * Return ``permissionDecision: allow|deny|ask`` →
                #     overrides the Agent gate's decision.
                #   * Return ``hookSpecificOutput.updatedCriticality``
                #     → mutates the criticality fed to the Agent (e.g.
                #     upgrade ``mutating`` to ``destructive`` after
                #     LLM-judge audit of the command). This is the
                #     primary surface for plugins that want to plug
                #     into ``_effective_criticality_for`` without
                #     forking the engine.
                if hook_engine is not None:
                    perm_hook = await self._fire_hook(
                        hook_engine,
                        hook_ctx,
                        "PermissionRequest",
                        {
                            "tool_name": name,
                            "tool_input": args,
                            "criticality": effective_crit,
                        },
                    )
                    for sm in perm_hook.system_messages:
                        yield TextDelta(
                            text=f"[hook] {sm}\n",
                            channel="content",
                        )
                    if perm_hook.blocked or perm_hook.permission_decision == "deny":
                        reason = (
                            perm_hook.block_reason
                            or perm_hook.permission_reason
                            or "blocked by a PermissionRequest hook"
                        )
                        # Fire PermissionDenied so an audit plugin can
                        # log the hook-side denial too.
                        await self._fire_hook(
                            hook_engine,
                            hook_ctx,
                            "PermissionDenied",
                            {
                                "tool_name": name,
                                "tool_input": args,
                                "criticality": effective_crit,
                                "reason": reason,
                                "source": "hook",
                            },
                        )
                        yield ToolCallStarted(
                            call_id=call_id,
                            name=name,
                            arguments=args,
                        )
                        _denied(call_id, name, reason)
                        yield ToolCallResult(
                            call_id=call_id,
                            ok=False,
                            content=f"[denied by PermissionRequest hook] {reason}",
                        )
                        continue
                    # Most-cautious floor: fold any
                    # ``hookSpecificOutput.updatedCriticality`` into
                    # ``effective_crit`` BEFORE the allow/deny decision below, so
                    # an escalation to ``forbidden`` gates (deny/escalate) the
                    # call even when the composed hook decision is ``allow``. A
                    # hook may only RAISE the severity, never downgrade the host
                    # classifier's (or another hook's) escalation. (Computing the
                    # composed criticality AFTER the allow branch let an
                    # allow+forbidden hook skip the gate.)
                    new_crit = self._extract_updated_criticality(perm_hook)
                    if new_crit:
                        effective_crit = self._most_cautious_criticality(
                            effective_crit,
                            new_crit,
                        )
                    if perm_hook.permission_decision == "allow":
                        # A PermissionRequest
                        # hook `allow` is the second override path that
                        # skips the Agent gate — so it too must never be
                        # able to green-light a FORBIDDEN call. ``effective_crit``
                        # now includes any hook ``updatedCriticality`` escalation
                        # (folded above); if it is forbidden,
                        # hard-deny regardless of the hook.
                        if effective_crit == "forbidden":
                            reason = (
                                f"forbidden: a PermissionRequest hook allowed "
                                f"{name!r}, but the call classifies as FORBIDDEN "
                                f"(blast radius too large to ever allow); "
                                f"this is an unappealable "
                                f"deny that no hook can override"
                            )
                            await self._fire_hook(
                                hook_engine,
                                hook_ctx,
                                "PermissionDenied",
                                {
                                    "tool_name": name,
                                    "tool_input": args,
                                    "criticality": effective_crit,
                                    "reason": reason,
                                    "source": "forbidden_gate",
                                },
                            )
                            yield ToolCallStarted(
                                call_id=call_id,
                                name=name,
                                arguments=args,
                            )
                            _denied(call_id, name, reason)
                            yield ToolCallResult(
                                call_id=call_id,
                                ok=False,
                                content=f"[denied: forbidden] {reason}",
                            )
                            continue
                        # Hook approved unconditionally — skip the
                        # Agent gate, mirror the PreToolUse `allow`
                        # path.
                        approved.append(norm)
                        continue
                perm_ev = PermissionRequested(
                    call_id=call_id,
                    tool_name=name,
                    arguments=args,
                    criticality=effective_crit,
                    response=fut,
                    hook_decision=hook_decision,
                )
                yield perm_ev
                try:
                    decision = await fut
                except Exception:  # noqa: BLE001
                    decision = "deny"
                if str(decision) != "allow":
                    reason = perm_ev.reason or "permission denied"
                    # A denial must not feed the doom-loop history —
                    # otherwise deny→retry→deny self-reinforces into a
                    # guaranteed destructive escalation.
                    if _hist and _hist[-1] == _sig:
                        _hist.pop()
                    # Fire PermissionDenied so an audit plugin sees
                    # the Agent-gate denial too.
                    if hook_engine is not None:
                        await self._fire_hook(
                            hook_engine,
                            hook_ctx,
                            "PermissionDenied",
                            {
                                "tool_name": name,
                                "tool_input": args,
                                "criticality": effective_crit,
                                "reason": reason,
                                "source": "agent_gate",
                            },
                        )
                    yield ToolCallStarted(
                        call_id=call_id,
                        name=name,
                        arguments=args,
                    )
                    _denied(call_id, name, reason)
                    yield ToolCallResult(
                        call_id=call_id,
                        ok=False,
                        content=f"[permission denied] {reason}",
                    )
                    continue

            approved.append(norm)

        if not approved:
            return

        # --- Dispatch the approved calls --------------------------------
        # Parallel-group dispatch: ``_build_parallel_groups`` walks the
        # approved calls and returns groups of indices safe to run
        # concurrently (read-only / NETWORK / state-file-distinct, per
        # ``_call_is_parallel_eligible`` + ``_state_file_key``). Each
        # multi-call group flows through ``_execute_batch_parallel``;
        # singleton groups still use ``_execute_call_singleton``. The
        # batch helper preserves entry order on its per-call queues so
        # appended tool-result messages still match the assistant's
        # ``tool_calls`` order, and a later call's tool result still
        # sees an earlier sibling's result on the next round.
        #
        # ``SHAREDAGENTS_SUBAGENT_INPROCESS=1`` pins every call to the
        # singleton path (regression escape — restores the pre-
        # 2026-05-16 strictly-serial behaviour).
        import os as _os

        if _os.environ.get("SHAREDAGENTS_SUBAGENT_INPROCESS") == "1" or session.hash in getattr(
            self, "_coordination_turns", {}
        ):
            # A todo write must settle before an action can observe it, and
            # the first dispatch must consume the turn before another runs.
            # Fan-out still happens inside the single AgentAsync runs batch.
            groups: list[list[int]] = [[i] for i in range(len(approved))]
        else:
            try:
                from sharedagents_agent.tools import (
                    BUILTIN_TOOLS,
                    PermissionEvaluator,
                    PermissionMode,
                    ToolRegistry,
                )
                from sharedagents_gateway.acp_bridge import (
                    _build_parallel_groups,
                    _call_is_parallel_eligible,
                    _state_file_key,
                )

                evaluator = PermissionEvaluator(PermissionMode.AUTO)
                registry = ToolRegistry(BUILTIN_TOOLS)
                annotated: list[dict[str, Any]] = []
                for entry in approved:
                    nm = entry.get("name")
                    tool = registry.get(nm) if nm else None
                    psafe = False
                    if tool is not None:
                        try:
                            psafe = _call_is_parallel_eligible(
                                tool,
                                entry,
                                evaluator,
                            )
                        except Exception:  # noqa: BLE001
                            psafe = False
                    annotated.append(
                        {
                            "name": nm,
                            "arguments": entry.get("arguments") or {},
                            "parallel_safe": psafe,
                            "state_key": _state_file_key(entry),
                        }
                    )
                groups = _build_parallel_groups(annotated)
            except Exception:  # noqa: BLE001
                # Conservative fallback — one call per group.
                groups = [[i] for i in range(len(approved))]
        # Cancellation is checked BETWEEN groups. Without this, a round
        # whose worker emitted many tool calls kept firing fresh runners
        # for minutes after ESC — the dispatch loop had no cancel check,
        # so the operator's abort (and even a force SIGKILL of the
        # worker) only took effect once every queued call had run. This
        # is the root cause of the "ESC does nothing for 2+ minutes on a
        # 262-tool turn" report.
        #   * hard abort (``_cancelled`` — 2nd ESC / ``/abort``) already
        #     SIGKILL+killpg'd the in-flight runner via ``abort(force)``.
        #   * graceful stop (``_graceful_stop`` — 1st ESC / ``/stop``)
        #     let the just-finished group complete; we just don't start
        #     the next one.
        # Either way the remaining calls are skipped with a synthetic
        # result so each assistant ``tool_call`` keeps a matching tool
        # message (an unmatched tool_call 400s the next upstream round /
        # a later resume).
        sid_for_cancel = getattr(turn, "session_hash", "") or ""
        cancelled_mid_batch = False
        for group in groups:
            if (
                not cancelled_mid_batch
                and sid_for_cancel
                and (self._cancelled.get(sid_for_cancel) or self._graceful_stop.get(sid_for_cancel))
            ):
                cancelled_mid_batch = True
            if cancelled_mid_batch:
                for idx in group:
                    c = approved[idx]
                    _denied(
                        c["id"],
                        c["name"],
                        "turn cancelled before this tool ran",
                    )
                continue
            if len(group) == 1:
                async for ev in self._execute_call_singleton(
                    approved[group[0]],
                    session,
                    round_idx=round_idx,
                    turn=turn,
                    messages=messages,
                    round_record=round_record,
                    hook_engine=hook_engine,
                    hook_ctx=hook_ctx,
                ):
                    yield ev
            else:
                batch = [approved[i] for i in group]
                async for ev in self._execute_batch_parallel(
                    batch,
                    session,
                    round_idx=round_idx,
                    turn=turn,
                    messages=messages,
                    round_record=round_record,
                    hook_engine=hook_engine,
                    hook_ctx=hook_ctx,
                ):
                    yield ev
        # ToolSearch promotion removed — the permission set IS the
        # tool set, so there is nothing to promote after a round.

        # Images returned by this round's tools reach the model in one
        # ``user`` message placed after EVERY tool result of the round, so
        # each assistant ``tool_call`` is still answered by the ``tool``
        # messages right after it. This is the shape a user attachment has,
        # so the image path (native vision, or transcription by a
        # vision-capable model for a text-only one) is the same.
        pending_images = round_record.pop("_tool_images", None) or []
        if pending_images:
            order = {str(c.get("id") or ""): i for i, c in enumerate(approved)}
            pending_images.sort(key=lambda e: order.get(e["call_id"], len(order)))
            image_msg = tool_image_message([(e["name"], e["label"], e["images"]) for e in pending_images])
            if image_msg is not None:
                messages.append(image_msg)

        # ``PostToolBatch`` (Claude Code spec) — fires after the whole
        # round's tool batch completes. Observational: lets a plugin
        # see the per-round tool topology (which tools ran, in what
        # order, with what outcomes) and emit a summary message /
        # tick a global counter / audit the batch shape. Differs from
        # ``PostToolUse`` (per-call) in granularity.
        if hook_engine is not None:
            batch = await self._fire_hook(
                hook_engine,
                hook_ctx,
                "PostToolBatch",
                {
                    "tool_name": ",".join(e["name"] for e in approved),
                    "tool_count": len(approved),
                    "calls": [
                        {
                            "id": e["id"],
                            "name": e["name"],
                            "arguments": e["arguments"],
                        }
                        for e in approved
                    ],
                    "round_idx": round_idx,
                },
            )
            for sm in batch.system_messages:
                yield TextDelta(text=f"[hook] {sm}\n", channel="content")

    def _tool_images_viewable(self, session: Any) -> bool:
        """Can an image a tool returned reach a model that understands it?

        Same decision the user-attachment path makes
        (``acp_bridge._resolve_images_for_model``): the session's model reads
        images itself when its model row declares the ``vision`` capability;
        otherwise the first configured vision-capable model transcribes them.
        Only when neither exists is the image withheld.
        """
        from sharedagents_agent.agent import find_model_with_capability, has_capability

        cfg = getattr(self, "cfg", None)
        try:
            row = cfg.model_row_for_endpoint(
                str(getattr(session, "endpoint", "") or ""),
                str(getattr(session, "model", "") or ""),
            )
        except Exception:  # noqa: BLE001
            row = {}
        if has_capability(row or {}, "vision"):
            return True
        try:
            return find_model_with_capability(cfg, "vision") is not None
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _most_cautious_criticality(*crits: str) -> str:
        """Return the MOST RESTRICTIVE of the given criticality strings.

        The single composition rule for criticality (the most-cautious
        floor): severity only ever
        ratchets UP. Unknown/unranked strings are treated as ``mutating``-level
        so they can neither be silently dropped nor beat a known
        ``destructive``/``forbidden``. Returns ``""`` when no non-empty claim
        is given.
        """
        rank = {
            "read_only": 0,
            "network": 1,
            "mutating": 1,
            "destructive": 2,
            "forbidden": 3,
        }
        real = [(c or "").strip() for c in crits if isinstance(c, str) and c.strip()]
        if not real:
            return ""
        return max(real, key=lambda c: rank.get(c, 1))

    @staticmethod
    def _extract_updated_criticality(perm_hook: Any) -> str:
        """Compose ``hookSpecificOutput.updatedCriticality`` across ALL hooks.

        Returns ``""`` when no hook touched criticality.

        ``HookOutcome`` doesn't carry per-key extension fields (it'd bloat the
        dataclass for every event-specific knob), so the getter walks the raw
        per-hook results from the outcome. Composition is the MOST RESTRICTIVE
        value any hook named — NOT last-writer-wins: no hook may downgrade another hook's escalation,
        consistent with the most-cautious floor. A hook can therefore
        only raise the severity fed to the Agent gate, never lower it.
        """
        if not getattr(perm_hook, "results", None):
            return ""
        found: list[str] = []
        for r in perm_hook.results:
            j = getattr(r, "json_output", None)
            if not isinstance(j, dict):
                continue
            hso = j.get("hookSpecificOutput")
            if not isinstance(hso, dict):
                continue
            uc = hso.get("updatedCriticality")
            if isinstance(uc, str) and uc.strip():
                found.append(uc.strip())
        return ToolDispatchMixin._most_cautious_criticality(*found)

    def _effective_criticality_for(
        self,
        tool_name: str,
        args: dict[str, Any],
    ) -> str:
        """Compute the most-cautious criticality for one tool call.

        Combines three signals — the tool registry's static kind
        (READ / WRITE / EXEC / NETWORK), a safety heuristic over the
        command (for ``Bash``-style execs), and the model's own
        ``args["criticality"]`` claim — and returns the *most
        cautious* of the three so a model can't downgrade
        ``rm -rf /`` past the destructive gate by labeling it
        ``read_only``. Defaults to ``"mutating"`` when the tool
        isn't in the registry.
        """
        if tool_name == "load_tools":
            # Framework-only schema discovery has no external side effects.
            return "read_only"
        try:
            from sharedagents_agent.tools import (
                BUILTIN_TOOLS,
                PermissionEvaluator,
                PermissionMode,
                ToolRegistry,
            )

            # ``BUILTIN_TOOLS`` is a tuple, not a mapping — look up
            # the matching ``Tool`` by name via :class:`ToolRegistry`.
            tool = ToolRegistry(BUILTIN_TOOLS).get(tool_name)
            if tool is None:
                # Belt-and-braces for any entry path that bypasses
                # ``_normalize_call``'s canonicalization: a legacy
                # TitleCase name ("Bash") must still resolve to the
                # canonical lowercase registry id, or the classifier's
                # READ_ONLY verdict for read-only commands (e.g.
                # ``git status``) would be floored to "mutating" and
                # plan mode would hard-deny it. Identity for names
                # that are already canonical.
                tool = ToolRegistry(BUILTIN_TOOLS).get(tool_name.lower())
            if tool is None and tool_name.startswith("mcp__"):
                # Discovered MCP tools (``mcp__<server>__<tool>``) are NOT in
                # BUILTIN_TOOLS. Resolve the criticality KIND by NAME from the
                # annotation cache (registry._mcp_kind_for), which honors the
                # server's readOnlyHint/destructiveHint. This covers BOTH the
                # always-on global servers AND per-session ACP servers (neither
                # is in BUILTIN_TOOLS) and needs no session id — this is a
                # staticmethod. readOnlyHint→READ so a read-only MCP tool gates
                # as READ (else it falls to the pessimistic "mutating" default
                # below and plan mode denies e.g. mcp__web__search);
                # destructiveHint→WRITE; else NETWORK→mutating. Only name
                # + kind matter to _effective_criticality, so a synthetic Tool
                # is sufficient.
                from sharedagents_agent.tools.registry import Tool, _mcp_kind_for

                tool = Tool(
                    name=tool_name,
                    kind=_mcp_kind_for(tool_name),
                    description="",
                    parameters={},
                )
            if tool is None:
                # Unknown tool — pessimistic default.
                return "mutating"
            evaluator = PermissionEvaluator(PermissionMode.AUTO)
            return str(
                evaluator._effective_criticality(tool, args).value,
            )
        except Exception:  # noqa: BLE001
            return "mutating"

    def _hook_allow_is_forbidden(
        self,
        tool_name: str,
        args: dict[str, Any],
    ) -> bool:
        """True iff a hook `allow` on this call must be OVERRIDDEN to a
        hard deny because the call classifies as FORBIDDEN.

        This is ABSOLUTE: a PreToolUse / PermissionRequest
        hook may allow anything BELOW forbidden, but can NEVER green-light
        a FORBIDDEN-classified call (``rm -rf /``, ``rm -rf ~/``, a fork
        bomb, a whole-disk wipe, an sharedagents-gateway self-kill). The engine
        honours a hook `allow` by short-circuiting the permission policy
        WITHOUT emitting a ``PermissionRequested`` event, so
        ``PermissionPolicy.evaluate()``'s forbidden hard-deny
        (``if crit == "forbidden": return "deny"``) never runs on that
        path. This re-applies the SAME host-side forbidden classifier
        (``classify_command`` via :meth:`_effective_criticality_for`) at
        the bypass point so the forbidden verdict stays unappealable — the
        hook can lower friction for non-forbidden calls but cannot un-forbid
        a forbidden one.
        """
        return self._effective_criticality_for(tool_name, args) == "forbidden"

    @staticmethod
    def _normalize_call(raw: dict[str, Any]) -> dict[str, Any]:
        """Coerce a worker / queued tool call into ``{id, name, arguments}``.

        Accepts both the raw OpenAI tool-call shape (``{"id", "function":
        {"name", "arguments"}}``) and the already-normalised
        ``{"id"/"call_id", "name", "arguments"}`` shape, plus the
        ``{"call": ...}`` wrapper the parallel-group helper emits.
        """
        call = raw.get("call") if isinstance(raw.get("call"), dict) else raw
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        call_id = str(call.get("id") or call.get("call_id") or "")
        # Canonicalize legacy TitleCase builtin ids ("Bash" → "bash") at the
        # gate entry — the allowlist composition, hook matchers, registry
        # lookup, and tool runners all key on the canonical lowercase id, but
        # raw call names arrive straight off the wire where the model may
        # emit the legacy spelling. A non-canonical name would otherwise be
        # denied as "not in this agent's allowlist" and, on paths that skip
        # that gate, floor to the pessimistic "mutating" criticality —
        # plan mode then hard-denies even read-only commands like
        # ``git status``. Unknown / MCP / delegate names pass through
        # unchanged (``canonicalize_tool_id`` is identity for non-builtins).
        from sharedagents_agent.tools.registry import canonicalize_tool_id

        name = canonicalize_tool_id(
            str(fn.get("name") or call.get("name") or "?"),
        )
        raw_args = fn.get("arguments")
        if raw_args is None:
            raw_args = call.get("arguments")
        if isinstance(raw_args, str):
            try:
                import json as _json

                args = _json.loads(raw_args or "{}")
            except (ValueError, TypeError):
                args = {"_arguments_unparsed": raw_args}
        elif isinstance(raw_args, dict):
            args = dict(raw_args)
        else:
            args = {}
        return {"id": call_id, "name": name, "arguments": args}

    def _build_parallel_groups(self, calls: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Group calls into safe parallel batches.

        Reference: acp_bridge.py L535–579 (and L499 / L515 for the
        eligibility predicates). Two calls go in the same group iff:

        * Both are read-only / pure-network (per ``Tool.kind``).
        * They don't share a state-file key
          (:func:`_state_file_key` returns distinct keys).
        * Neither call mutates a file the other reads (heuristic on
          ``args.path`` / ``args.file_path``).

        Calls that aren't parallel-eligible become singleton groups so
        the dispatch loop still runs them in order.
        """
        try:
            from sharedagents_gateway.acp_bridge import _build_parallel_groups

            return _build_parallel_groups(calls)
        except ImportError:
            # Conservative fallback — one call per group.
            return [[c] for c in (calls or [])]

    def _auto_complete_matching_todo(
        self,
        session: Any,
        tool_name: str,
        args: dict[str, Any],
    ) -> list[dict[str, Any]] | None:
        """Auto-flip a matching TodoWrite item after a subagent succeeds.

        Reference: SMALL_MODEL_REVIEW.md §9a. Small models routinely
        forget to call TodoWrite to update their own plan items; the
        harness scans the persisted todo list for a ``pending`` /
        ``in_progress`` item whose content matches the spawned
        subagent (by verb + target name) and flips it to ``completed``.

        Returns the updated todo list if a match was found and
        persisted, ``None`` otherwise.
        """
        if not getattr(self.cfg, "auto_complete_todos", True):
            return None
        if not (is_agent_tool(tool_name) or tool_name.startswith("delegate_")):
            return None
        sid = getattr(session, "hash", "") or ""

        if not sid:
            return None
        # Derive the subagent target needle from the tool name + args.
        needle = ""
        if is_agent_tool(tool_name):
            subagent_type = str(args.get("subagent_type") or "").lower()
            if subagent_type in {"general-purpose", "general", "gp"}:
                needle = str(args.get("name") or subagent_type).lower()
            else:
                needle = subagent_type
        elif tool_name.startswith("delegate_"):
            # delegate_claude → "claude", delegate_agent → "agent", etc.
            needle = tool_name[len("delegate_") :].lower()
        if not needle:
            return None
        try:
            from sharedagents_agent.tools.runners import (
                _save_todos,
                load_todos,
                todos_path_for_session,
            )
        except ImportError:
            return None
        # Defensive: some test/CLI cfgs don't carry a sessions_dir
        # (types.SimpleNamespace minimal stubs in test_22 land here).
        # Production cfgs always have it; missing → silently skip
        # auto-complete rather than crash the engine turn.
        sessions_dir = getattr(self.cfg, "sessions_dir", None)
        if sessions_dir is None:
            return None
        path = todos_path_for_session(sessions_dir, sid)
        todos = load_todos(path)
        if not todos:
            return None
        spawn_verbs = ("spawn", "delegate", "subagent")
        for item in todos:
            status = item.get("status") or "pending"
            if status in ("done", "completed"):
                continue
            content = (item.get("content") or "").lower()
            if any(v in content for v in spawn_verbs) and needle in content:
                item["status"] = "completed"
                _save_todos(path, todos)
                return todos
        return None

    async def _execute_call_singleton(
        self,
        entry: dict[str, Any],
        session: Any,
        round_idx: int,
        turn: Any,
        messages: list[dict[str, Any]],
        round_record: dict[str, Any],
        *,
        hook_engine: Any = None,
        hook_ctx: dict[str, Any] | None = None,
    ) -> AsyncIterator[AgentTurnEvent]:
        """Run ONE permission-approved tool call.

        Reference: acp_bridge.py L2136–2369. Mirrors the legacy ALLOW
        branch: streams runner deltas via :class:`TextDelta` keyed on
        the call_id, captures the final result body, appends a
        ``{"role": "tool", "tool_call_id": …, "content": …}`` message
        to ``messages`` so the next round sees the result.

        Mutations: ``messages`` (appended in-place) and
        ``round_record["calls"]`` (one record per call).

        Failure modes captured:

        * Tool runner spawn fails (``TOOL_RUNNER_SPAWN_FAILED`` event).
        * Tool runner aborted mid-stream (``TOOL_RUNNER_ABORT``).
        * Tool runner cancelled (``TOOL_RUNNER_CANCEL``); sets
          ``self._cancelled[sid]`` and re-raises.
        """
        from ..engine import TextDelta, ToolCallResult, ToolCallStarted

        call = entry.get("call") or entry
        call_id = str(call.get("id") or call.get("call_id") or "")
        name = str((call.get("function") or {}).get("name") or call.get("name") or "?")
        raw_args = (call.get("function") or {}).get("arguments") or call.get("arguments") or "{}"
        if isinstance(raw_args, str):
            try:
                import json

                args = json.loads(raw_args)
            except (json.JSONDecodeError, ValueError):
                args = {"_raw": raw_args}
        else:
            args = dict(raw_args or {})

        sid = getattr(session, "hash", "") or ""

        coordination = getattr(self, "_coordination_turns", {}).get(sid)
        if is_agent_tool(name) and isinstance(args, dict):
            # Wave origin and membership are runtime-owned: never trust a
            # model-supplied value (the model-facing join field is
            # ``wave_id``). Only the Jarvis coordinator stamps an origin;
            # batch entries are stripped by the batch runner.
            args.pop("_sharedagents_wave_origin", None)
            args.pop("_sharedagents_wave_id", None)
            if coordination is not None and coordination.origin:
                args["_sharedagents_wave_origin"] = coordination.origin
        if coordination is not None:
            from sharedagents_agent.tools.runners import load_todos, todos_path_for_session

            from .coordination import is_dispatch

            todo_path = todos_path_for_session(self.cfg.sessions_dir, sid)
            open_jobs: list = []
            origins: dict = {}
            if is_dispatch(name, args):
                open_jobs = list(self._open_background_jobs_for(sid))
                if open_jobs:
                    from sharedagents_agent.agents.async_jobs import wave_origin

                    for _job in open_jobs:
                        _wid = str(_job.get("wave_id") or "")
                        if _wid and _wid not in origins:
                            origins[_wid] = wave_origin(_wid)
            reason = coordination.refusal(
                name,
                args,
                has_todos=bool(load_todos(todo_path)),
                open_jobs=open_jobs,
                open_wave_origins=origins,
            )
            if reason:
                yield ToolCallStarted(call_id=call_id, name=name, arguments=args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": f"[workflow refused] {reason}",
                    }
                )
                round_record.setdefault("calls", []).append(
                    {
                        "id": call_id,
                        "name": name,
                        "ok": False,
                        "denied": reason,
                        "bg_handle": False,
                    }
                )
                yield ToolCallResult(call_id=call_id, ok=False, content=reason, outcome="refused")
                return

        # ``SubagentStart`` (Claude Code spec) — fires BEFORE the
        # delegate runner spawns the child agent's turn. Symmetric to
        # SubagentStop (which fires after the child returns). Lets an
        # ECC plugin emit a "child agent started" notification or
        # snapshot the parent state before the cross-agent boundary.
        if hook_engine is not None and name.startswith("delegate_"):
            subagent_name = name[len("delegate_") :] or str(args.get("name") or "agent")
            await self._fire_hook(
                hook_engine,
                hook_ctx or {},
                "SubagentStart",
                {
                    "subagent_name": subagent_name,
                    "tool_name": name,
                    "tool_input": args,
                },
            )

        # ``Elicitation`` (Claude Code spec) — fires BEFORE the
        # AskUserQuestion tool round-trips through the frontend so an
        # ECC plugin can:
        #   * ``decision: block`` (exit 2) → suppress the user
        #     question entirely; the hook's ``reason`` becomes the
        #     synthesised answer body the model gets back (use case:
        #     an MCP elicitation handler that auto-answers from a
        #     remote knowledge base).
        #   * Observe the question list (audit / telemetry).
        # ``ElicitationResult`` fires AFTER the answer comes back so
        # plugins can post-process / log the user's response.
        # Ask-family identity comes from the shared widget dictionary, so a
        # foreign spelling (AskUserQuestion / askuser / ask_user_question) fires
        # the same Elicitation hook round-trip as the native ``question``.
        asks_user = is_ask_user_tool(name)
        elicitation_hook = None
        if hook_engine is not None and asks_user:
            elicitation_hook = await self._fire_hook(
                hook_engine,
                hook_ctx or {},
                "Elicitation",
                {
                    "name": name,
                    "tool_input": args,
                    "questions": args.get("questions") or [],
                },
            )
            for sm in elicitation_hook.system_messages:
                yield TextDelta(
                    text=f"[hook] {sm}\n",
                    channel="content",
                )
            if elicitation_hook.blocked and elicitation_hook.block_reason:
                # Plugin supplied a synthetic answer body — short-
                # circuit the round trip with the hook's reason. This is
                # plugin-authored (not the user), so fence it as untrusted
                # data (same invariant as post_ctx / elicitation context).
                from sharedagents_core.injection_guard import wrap_untrusted

                _blk = wrap_untrusted(
                    elicitation_hook.block_reason,
                    source=f"hook:{name}:elicitation",
                )
                yield ToolCallStarted(
                    call_id=call_id,
                    name=name,
                    arguments=args,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": _blk,
                    }
                )
                yield ToolCallResult(
                    call_id=call_id,
                    ok=True,
                    content=_blk,
                )
                await self._fire_hook(
                    hook_engine,
                    hook_ctx or {},
                    "ElicitationResult",
                    {
                        "name": name,
                        "tool_input": args,
                        "answer": elicitation_hook.block_reason,
                        "source": "hook",
                    },
                )
                return

        yield ToolCallStarted(call_id=call_id, name=name, arguments=args)

        # Stream runner callbacks back through the generator. on_delta
        # carries Claude/Codex/Gemini's prose chunks (keyed on this
        # outer call_id so the TUI mounts a streaming ToolResultMessage
        # under the delegate widget); on_event carries inner
        # tool_use/tool_result blocks (each with the CLI's own call_id,
        # so the TUI mounts native ToolCallMessage/ToolResultMessage
        # for Read/Bash/TodoWrite/etc.). Queue + driver task mirrors
        # _execute_batch_parallel below and ExternalCliEngine.run_turn —
        # async generators can't yield from inside a callback, so we
        # channel through an asyncio.Queue.
        ev_queue: asyncio.Queue[Any] = asyncio.Queue()
        # Per-call cache so a delegate's inner tool ids are namespaced
        # under THIS outer call id (``<call_id>.inner.<raw>``) instead of
        # leaking the child engine's raw ids into the parent session log
        # as if the parent model had emitted them. Mirrors the in-process
        # shim path below. See _namespace_inner_call_id.
        inner_call_ids: dict[str, str] = {}

        async def _on_delta(chunk: str) -> None:
            if chunk:
                await ev_queue.put(
                    TextDelta(
                        text=chunk,
                        channel="tool_delta",
                        call_id=call_id,
                    )
                )

        async def _on_event(payload: dict) -> None:
            kind = (payload or {}).get("kind") or ""
            if kind == "tool_use":
                await ev_queue.put(
                    ToolCallStarted(
                        call_id=_namespace_inner_call_id(
                            call_id,
                            payload.get("id"),
                            inner_call_ids,
                        ),
                        name=str(payload.get("name") or "?"),
                        arguments=dict(payload.get("input") or {}),
                        from_subagent=True,  # child's inner tool — display only, not persisted in parent log
                    )
                )
            elif kind == "tool_result":
                await ev_queue.put(
                    ToolCallResult(
                        call_id=_namespace_inner_call_id(
                            call_id,
                            payload.get("tool_use_id"),
                            inner_call_ids,
                        ),
                        ok=bool(payload.get("ok", True)),
                        content=str(payload.get("text") or ""),
                        from_subagent=True,  # child's inner tool — display only, not persisted in parent log
                    )
                )
            elif kind == "sensitive_egress_flag":
                await ev_queue.put(_egress_flag_delta(payload))

        async def _drive() -> tuple[bool, str, bool, bool, bool, str | None, tuple, tuple]:
            try:
                if name == "load_tools":
                    # Execute only after the normal permission/hook gates, in
                    # batch order. The common result path records it and fires
                    # PostToolUse without spawning an external runner.
                    from sharedagents_agent.mcp_deferral import (
                        load_result_text,
                        requested_servers,
                    )

                    ok, text = load_result_text(
                        requested_servers(args),
                        ToolDispatchMixin._deferred_mcp_servers(
                            self,
                            session,
                            self._runtime_allowed_tools(session),
                        ),
                    )
                    return ok, text, False, False, False, None, (), ()
                return await self._invoke_runner(
                    sid,
                    {"id": call_id, "name": name, "arguments": args},
                    turn,
                    on_delta=_on_delta,
                    on_event=_on_event,
                )
            finally:
                await ev_queue.put(None)  # sentinel

        driver = asyncio.create_task(_drive())

        ok = True
        body = ""
        aborted = False
        truncated = False
        bg_handle = False
        outcome = None
        member_outcomes = ()
        images: tuple = ()
        try:
            while True:
                ev = await ev_queue.get()
                if ev is None:
                    break
                yield ev
            (ok, body, aborted, truncated, bg_handle, outcome, member_outcomes, images) = await driver
        except asyncio.CancelledError:
            if not driver.done():
                driver.cancel()
                try:
                    await driver
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            self._cancelled[sid] = True
            self._log_event("TOOL_RUNNER_CANCEL", session_id=sid, name=name)
            raise
        except Exception as exc:  # noqa: BLE001
            ok, body = False, f"runner error: {exc}"
            bg_handle = False
            self._log_event(
                "TOOL_RUNNER_SPAWN_FAILED",
                session_id=sid,
                name=name,
                error=str(exc),
            )

        if aborted:
            self._log_event(
                "TOOL_RUNNER_ABORT",
                session_id=sid,
                name=name,
            )

        # PostToolUse hooks (Claude Code) — observation + optional
        # context injection. ``additionalContext`` is folded into the
        # tool-result message body so the model actually sees it on the
        # next round (not just the transcript); a blocking PostToolUse
        # hook appends its reason as model-facing feedback.
        post_ctx = ""
        if hook_engine is not None:
            post = await self._fire_hook(
                hook_engine,
                hook_ctx or {},
                "PostToolUse" if ok else "PostToolUseFailure",
                {"tool_name": name, "tool_input": args, "tool_response": body},
            )
            for sm in post.system_messages:
                yield TextDelta(text=f"[hook] {sm}\n", channel="content")
            if post.additional_context:
                post_ctx = post.additional_context
            if post.blocked and post.block_reason:
                post_ctx = (post_ctx + "\n\n" if post_ctx else "") + f"[PostToolUse hook] {post.block_reason}"
            # ``SubagentStop`` (Claude Code) — fires when a child
            # agent's turn completes. sharedagents's "subagents" are the
            # ``delegate_*`` tool family (Claude / Codex / Gemini CLI
            # children + ``delegate_agent`` to a sibling sharedagents
            # agent). The hook receives the child's ``tool_response``
            # body so an ECC plugin can post-process the child's
            # output (lint, route to telemetry, append a
            # cross-agent summary) before the parent sees it.
            if name.startswith("delegate_"):
                subagent_name = name[len("delegate_") :] or str(args.get("name") or "agent")
                sub = await self._fire_hook(
                    hook_engine,
                    hook_ctx or {},
                    "SubagentStop",
                    {
                        "subagent_name": subagent_name,
                        "tool_name": name,
                        "tool_input": args,
                        "tool_response": body,
                        "state": "ok" if ok else "error",
                        "stop_hook_active": False,
                    },
                )
                for sm in sub.system_messages:
                    yield TextDelta(
                        text=f"[hook] {sm}\n",
                        channel="content",
                    )
                if sub.additional_context:
                    post_ctx = (post_ctx + "\n\n" if post_ctx else "") + sub.additional_context
                if sub.blocked and sub.block_reason:
                    post_ctx = (post_ctx + "\n\n" if post_ctx else "") + f"[SubagentStop hook] {sub.block_reason}"
        # Prompt-injection trust boundary: ``body`` is the tool's OWN output
        # (file contents, shell output, web page, search results, MCP server
        # reply, or a sub-agent's text) — untrusted external content. Wrap it
        # in an injection-guard fence so the model treats it as data, never
        # instructions.
        #
        # Exception: an AskUserQuestion / elicitation answer (``asks_user``) is
        # the USER'S OWN input, not external data — fencing it would tell the
        # model to distrust the user's explicit choice. Leave it unfenced.
        #
        # ``post_ctx`` is hook output (PostToolUse / SubagentStop
        # additionalContext, block reasons). Per the hook spec it is
        # informational CONTEXT for the model, not authoritative instruction —
        # and a hook can legitimately echo tool/sub-agent output into it, which
        # is attacker-controllable. So it is ALSO fenced as untrusted data.
        # This makes the invariant exact: the ONLY content outside a fence is
        # the system prompt and the user's own turns.
        #
        # Exception: an AskUserQuestion / elicitation answer (``asks_user``) is
        # the USER'S OWN input, not external data — fencing it would tell the
        # model to distrust the user's explicit choice. Leave it unfenced.
        from sharedagents_core.injection_guard import wrap_untrusted

        if asks_user or (name == "load_tools" and ok):
            # Schema discovery returns framework-owned metadata, also read
            # by the next request to recover successful loads.
            _fenced_body = body
        else:
            _src = f"subagent:{name[len('delegate_') :] or 'agent'}" if name.startswith("delegate_") else f"tool:{name}"
            _fenced_body = wrap_untrusted(body, source=_src)
        _post = wrap_untrusted(post_ctx, source=f"hook:{name}") if post_ctx else ""
        result_body = f"{_fenced_body}\n\n{_post}".strip() if _post else _fenced_body

        # ``ElicitationResult`` (Claude Code spec) — fires AFTER the
        # AskUserQuestion answer comes back so an ECC plugin can
        # post-process / log / route the user's reply. Observational
        # by default; ``additionalContext`` is folded into the tool-
        # result body the model sees so a plugin can add post-hoc
        # interpretation (e.g. "user selected `cancel`; treating as
        # 'do not proceed'").
        if hook_engine is not None and asks_user:
            er = await self._fire_hook(
                hook_engine,
                hook_ctx or {},
                "ElicitationResult",
                {
                    "name": name,
                    "tool_input": args,
                    "answer": body,
                    "ok": ok,
                    "source": "frontend",
                },
            )
            for sm in er.system_messages:
                yield TextDelta(text=f"[hook] {sm}\n", channel="content")
            if er.additional_context:
                # Plugin-supplied interpretation = untrusted hook context →
                # fenced as data (same invariant as post_ctx above).
                _er_ctx = wrap_untrusted(er.additional_context, source=f"hook:{name}:elicitation")
                result_body = f"{result_body}\n\n{_er_ctx}".strip() if result_body else _er_ctx

        if coordination is not None:
            coordination.observe(name, args, ok=ok, background_handle=bg_handle)

        # Images the tool returned (``read`` on a PNG …). A ``tool`` message
        # is text-only on every transport, so they are queued on the round
        # and delivered in ONE ``user`` message after all of this round's
        # tool results (see ``_dispatch_tool_calls``). When no configured
        # model can look at them, the model is told so instead.
        if images:
            if self._tool_images_viewable(session):
                round_record.setdefault("_tool_images", []).append(
                    {
                        "call_id": call_id,
                        "name": name,
                        "label": _tool_image_label(name, args),
                        "images": tuple(images),
                    }
                )
            else:
                images = ()
                result_body = f"{result_body}\n\n{_TOOL_IMAGES_UNVIEWABLE}"

        # Append the result to the message list for the next round.
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": result_body,
            }
        )

        round_record.setdefault("calls", []).append(
            {
                "id": call_id,
                "name": name,
                "ok": ok,
                "truncated": truncated,
                "aborted": aborted,
                "content_size": len(result_body or ""),
                "bg_handle": bool(bg_handle),
            }
        )

        yield ToolCallResult(
            call_id=call_id,
            ok=ok,
            content=result_body,
            outcome=outcome,
            member_outcomes=member_outcomes,
            images=tuple(images),
        )

        # Auto-complete matching TodoWrite item when a subagent
        # (Agent / delegate_*) returns success. Small models forget
        # to call TodoWrite to flip their own plan items; the harness
        # auto-completes the first matching pending/in_progress item.
        # Opt-out via ``auto_complete_todos: false`` on agent config.
        if ok and (is_agent_tool(name) or name.startswith("delegate_")):
            updated = self._auto_complete_matching_todo(session, name, args)
            if updated is not None:
                try:
                    from sharedagents_agent.tools.runners import format_todo_md
                except ImportError:
                    format_todo_md = None  # type: ignore[assignment]
                synth_id = f"auto_todo_{call_id}"
                yield ToolCallStarted(
                    call_id=synth_id,
                    name="todowrite",
                    arguments={"todos": updated},
                )
                if format_todo_md is not None:
                    yield ToolCallResult(
                        call_id=synth_id,
                        ok=True,
                        content=format_todo_md(updated),
                    )
                else:
                    yield ToolCallResult(
                        call_id=synth_id,
                        ok=True,
                        content="[harness] auto-completed matching todo",
                    )

    async def _execute_batch_parallel(
        self,
        entries: list[dict[str, Any]],
        session: Any,
        round_idx: int,
        turn: Any,
        messages: list[dict[str, Any]],
        round_record: dict[str, Any],
        *,
        hook_engine: Any = None,
        hook_ctx: dict[str, Any] | None = None,
    ) -> AsyncIterator[AgentTurnEvent]:
        """Run a batch of parallel-eligible calls concurrently.

        Reference: acp_bridge.py L2370–2549. Fan-out via
        ``asyncio.gather``; each driver task gets its own
        ``on_delta`` callback keyed on the call_id so the streaming
        :class:`TextDelta`s don't interleave incorrectly. Order of
        the appended tool-result messages MATCHES the order of
        ``entries`` so the model sees a deterministic transcript.
        """
        if not entries:
            return

        # Per-entry deterministic queue so the events for different
        # parallel calls don't interleave at the consumer.
        queues: list[asyncio.Queue[Any]] = [asyncio.Queue() for _ in entries]

        async def _drive_one(idx: int, entry: dict[str, Any]) -> None:
            try:
                async for ev in self._execute_call_singleton(
                    entry,
                    session,
                    round_idx,
                    turn,
                    messages,
                    round_record,
                    hook_engine=hook_engine,
                    hook_ctx=hook_ctx,
                ):
                    await queues[idx].put(ev)
            finally:
                await queues[idx].put(None)  # sentinel

        tasks = [asyncio.create_task(_drive_one(i, e)) for i, e in enumerate(entries)]
        try:
            for _idx, q in enumerate(queues):
                while True:
                    ev = await q.get()
                    if ev is None:
                        break
                    yield ev
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
            for t in tasks:
                try:
                    await t
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass

    async def _invoke_runner(
        self,
        session_id: str,
        call: dict[str, Any],
        turn: Any,
        *,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
        on_event: Callable[[dict], Awaitable[None]] | None = None,
    ) -> tuple[bool, str, bool, bool, bool, str | None, tuple, tuple]:
        """Call ``self._tool_runner`` with optional streaming callbacks.

        Reference: acp_bridge.py L2107–2134. Returns
        ``(ok, body, aborted, truncated, background_handle, outcome,
        member_outcomes, images)``. Falls back
        through narrower signatures (no ``on_event``, then no ``on_delta``)
        for test stubs that don't accept them.
        """
        runner = self._tool_runner
        if runner is None:
            try:
                from sharedagents_gateway.acp_bridge import CoreBackend, _default_run_tool
            except ImportError as exc:
                return False, f"no tool runner available: {exc}", False, False, False, None, (), ()
            # ``_default_run_tool`` expects ``(backend, session_id,
            # call, turn, **kw)``. The bridge normally pre-binds
            # ``backend=self`` when it builds the engine (see
            # ``CoreBackend._get_new_agent``). If we got here with no
            # injected runner, the engine was built outside the
            # bridge — wrap with a synthesized CoreBackend-shaped
            # shim so the signature still lines up.
            engine = self
            sess_cache = engine._sessions

            class _RunToolShim:
                def __init__(self) -> None:
                    self.cfg = engine.cfg
                    self.server_handle = None

                def _session_for(self, sid: str) -> Any:
                    return sess_cache.get(sid)

                def _outbound_for(self, _sid: str) -> Any:  # noqa: ARG002
                    return None

                def _get_new_agent(self) -> Any:
                    agent = getattr(engine, "_agent", None)
                    if agent is None:
                        raise RuntimeError("live Agent is not bound to LocalLLMEngine")
                    return agent

                # Reuse the production backend implementations. They depend only
                # on _get_new_agent/_session_for, both supplied above, and keep
                # goal/budget/loop transitions on the live in-process Agent.
                goal = CoreBackend.goal
                goal_budget = CoreBackend.goal_budget
                cron_create = CoreBackend.cron_create
                cron_delete = CoreBackend.cron_delete
                cron_list = CoreBackend.cron_list
                schedule_wakeup = CoreBackend.schedule_wakeup

            shim_backend = _RunToolShim()

            async def _bound(session_id: str, call: dict, turn_state: Any, **kw: Any) -> Any:
                return await _default_run_tool(shim_backend, session_id, call, turn_state, **kw)

            runner = _bound
        if runner is None:
            return False, "no tool runner configured", False, False, False, None, (), ()

        # Try the widest signature first, then progressively narrower
        # ones for test-injected stubs.
        attempts = []
        if on_event is not None:
            attempts.append({"on_delta": on_delta, "on_event": on_event})
        attempts.append({"on_delta": on_delta})
        attempts.append({})
        last_exc: BaseException | None = None
        for kwargs in attempts:
            try:
                result = await runner(session_id, call, turn, **kwargs)
                # Coerce — some runners return a ToolResult, some a tuple.
                if isinstance(result, tuple) and len(result) >= 2:
                    ok = bool(result[0])
                    body = str(result[1] or "")
                    aborted = bool(result[2]) if len(result) > 2 else False
                    truncated = bool(result[3]) if len(result) > 3 else False
                    bg_handle = bool(result[4]) if len(result) > 4 else False
                    return (
                        ok,
                        body,
                        aborted,
                        truncated,
                        bg_handle,
                        result[5] if len(result) > 5 else None,
                        tuple(result[6] or ()) if len(result) > 6 else (),
                        tuple(result[7] or ()) if len(result) > 7 else (),
                    )
                ok = bool(getattr(result, "ok", True))
                body = str(getattr(result, "content", "") or "")
                aborted = bool(getattr(result, "aborted", False))
                truncated = bool(getattr(result, "truncated", False))
                bg_handle = bool(getattr(result, "background_handle", False))
                return (
                    ok,
                    body,
                    aborted,
                    truncated,
                    bg_handle,
                    getattr(result, "outcome", None),
                    tuple(getattr(result, "member_outcomes", ())),
                    tuple(getattr(result, "images", ()) or ()),
                )
            except TypeError as exc:
                last_exc = exc
                continue
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                return False, f"runner raised: {exc}", False, False, False, None, (), ()
        return False, f"no runner signature matched: {last_exc}", False, False, False, None, (), ()

    # -----------------------------------------------------------------
    # Direct-dispatch (slash-command bypass)
    # -----------------------------------------------------------------

    async def _run_direct_tool_call(
        self,
        session: Any,
        turn_id: str,
        direct: dict[str, Any],
    ) -> AsyncIterator[AgentTurnEvent]:
        """Bypass the LLM and invoke a delegate-class tool directly.

        Reference: acp_bridge.py L2551–2837.

        Used by ``/claude``, ``/codex``, ``/gemini``, ``/compact``
        slash commands when the operator types the prompt themselves —
        the local model isn't responsible for the dispatch decision.

        Restricted to tool names matching ``delegate_*`` (so an
        unauthenticated frontend can't smuggle in a write/exec). The
        ``on_event`` path is wired so inner Claude tool blocks fan out
        as proper :class:`ToolCallStarted` / :class:`ToolCallResult`
        events (TodoWrite ticks the plan checklist; Read/Bash render
        as native widgets).

        In the new architecture, ``delegate_<kind>`` tools dispatch
        through ``Agent.run_turn`` on a *child* Agent (engine=CLI),
        so this method shrinks dramatically — the bulk of the legacy
        body moves into ``ExternalCliEngine.run_turn``. What remains
        here is the wire-event translation + per-turn metrics row.
        """
        from .. import delegate_shims
        from ..engine import (
            TextDelta,
            ToolCallResult,
            ToolCallStarted,
            TurnDone,
            TurnError,
        )

        name = str(direct.get("name") or direct.get("tool") or "")
        args = direct.get("arguments") or direct.get("args") or {}
        if not isinstance(args, dict):
            args = {}

        # Whitelist: only delegate_* tools (the slash-command bridges
        # /claude /codex /gemini) and ``compact_context`` (used by
        # /compact) may bypass the LLM. Anything else routes through a
        # normal turn — an unauthenticated frontend can't smuggle a
        # write/exec via this path.
        if name == "compact_context":
            # /compact: deterministic IN-PROCESS compaction — no runner
            # subprocess and, crucially, no synthetic model round. The old
            # path routed /compact through a synthetic LLM turn so the model
            # would emit this tool call; that DEADLOCKED when the context
            # was already at/over the window (the round couldn't fit to
            # emit the call — the exact qwen3.6-27b @ 65536 HTTP 400 the
            # operator was trying to recover from). And the previous body
            # of THIS handler was itself broken: it read
            # ``getattr(session, "messages", [])`` — a field that never
            # existed, so it squeezed an empty list — and threw away
            # ``_maybe_compact``'s return value, persisting nothing. We now
            # build the REAL wire list, summarize the OLD chunk with a
            # foreground compaction subagent (visible in Agent Monitor;
            # in-process so no worker round, abort-aware via
            # ``_await_unless_hard_cancel``; ANY failure falls back to the
            # mechanical squeeze below, which still works at a completely
            # full window), durably trim the persisted tool bodies, clear
            # any stale max_tokens override, and report — stats + semantic
            # memo in the turn body, which replaces the parent context.
            call_id = f"direct-{turn_id[:8]}"
            yield ToolCallStarted(
                call_id=call_id,
                name=name,
                arguments=args,
                is_outer=True,
            )
            child = None
            try:
                from sharedagents_agent.compaction import (
                    spawn_visible_compaction_subagent,
                )

                child = spawn_visible_compaction_subagent(
                    session,
                    engine=self._compaction_engine_label(),
                )
            except Exception:  # noqa: BLE001
                child = None
            try:
                from sharedagents_agent.turn_runtime import (
                    _approx_tokens,
                    _format_compaction_summary,
                    _manual_compact_target,
                    _merge_compaction_summary,
                    compact_messages,
                    compact_session_turns_in_place,
                    prune_session_tool_outputs,
                )

                # Build the SAME message list the next worker request would
                # send (this is the live running context to compact).
                msgs = self._build_messages_simple(session, "", None, [])
                before_tokens = _approx_tokens(msgs)
                removed_est = 0
                audit: dict[str, Any] = {
                    "truncated_tool_call_ids": [],
                    "dropped_tool_call_ids": [],
                    "dropped_turns": 0,
                }
                ctx_window = int(
                    getattr(session, "context_window", 0) or getattr(self.cfg, "upstream_max_context", 0) or 0
                )
                new_messages: list[dict[str, Any]] | None = None
                memo = ""
                if ctx_window > 0:
                    out_budget = self._effective_output_budget(session)
                    # Manual /compact ALWAYS trims (opencode parity): the
                    # target is HALF the current usage (never above the safe
                    # window), so a small context in a huge window still gets
                    # a real squeeze instead of "~0 tokens removed" — see
                    # ``_manual_compact_target`` in sharedagents_agent.compaction.
                    # The AUTO path (_maybe_compact / _compaction_target)
                    # stays threshold-gated on compact_at_frac and is
                    # untouched by this.
                    target = _manual_compact_target(
                        before_tokens,
                        ctx_window,
                        out_budget,
                    )
                    # LLM-first (opencode V2 parity): summarize the OLD chunk
                    # with the hidden compaction agent (in-process, abort-
                    # aware — no synthetic model round, which is what
                    # DEADLOCKED the old /compact at a full window) and
                    # rebuild ``[system, memo, *tail]``. ANY failure —
                    # disabled, no clean split, window too small, summarizer
                    # HTTP/timeout/empty, mid-summary abort — falls back to
                    # the mechanical squeeze below; the turn is NEVER allowed
                    # to break on compaction.
                    try:
                        new_messages, memo = await self._manual_compact_via_summary(
                            session,
                            msgs,
                            sharedagents_sid=getattr(session, "hash", "") or None,
                        )
                    except Exception:  # noqa: BLE001
                        new_messages, memo = None, ""
                    if new_messages is None:
                        _squeezed, removed_est, audit = compact_messages(
                            msgs,
                            target,
                            context_window=ctx_window,
                        )
                        new_messages = _squeezed
                # Durably trim oversized tool-result bodies persisted on the
                # session turns so the saving survives across turns/restarts
                # (the per-round rebuild re-squeezes the wire copy anyway via
                # _maybe_compact, but persisting keeps the on-disk record lean
                # and matches openclaw's in-process /compact semantics).
                removed_chars, _affected = compact_session_turns_in_place(
                    session,
                )
                try:
                    _pruned_c, _ = prune_session_tool_outputs(session, context_window=ctx_window)
                    removed_chars += _pruned_c
                except Exception:  # noqa: BLE001
                    pass
                # Clear any stale reactive max_tokens override the operator is
                # explicitly recovering from — a value pinned too high in a
                # prior overflow keeps re-triggering the 400 (the 14334↔15359
                # oscillation seen in prod). Next turn recomputes if needed.
                try:
                    if getattr(session, "max_tokens_override", None) is not None:
                        session.max_tokens_override = None
                except Exception:  # noqa: BLE001
                    pass
                after_tokens = (
                    _approx_tokens(new_messages) if new_messages is not None else max(0, before_tokens - removed_est)
                )
                removed_est = max(0, before_tokens - after_tokens)
                if removed_est > 0 or removed_chars > 0:
                    stats_line = _format_compaction_summary(
                        removed_est,
                        audit,
                        after_tokens,
                    )
                else:
                    stats_line = "compacted: nothing to trim — the running context is already within budget."
                # The turn body shows BOTH the stats line and the semantic
                # memo (opencode V2 parity): the WebUI stats widget reads
                # ``compaction_last_stats`` and the chat turn renders this
                # ToolCallResult body, so the operator sees what the
                # summarizer kept before the tail.
                body = stats_line + "\n\n" + memo if memo else stats_line
                # Persist compaction stats on the session HEADER so the
                # WebUI collapsed widget can render them (and they survive
                # reloads): bump the manual-compaction count, keep the
                # formatted summary, and record the last-run stats dict.
                # ``compaction_summary`` now holds the SEMANTIC memo (rolled
                # with any prior one via ``_merge_compaction_summary``) when
                # the LLM path produced it, else the stats line — either way
                # the widget has text to show. ``session.save()`` diff-emits
                # a ``header_update`` the projector folds into sessions_meta
                # and the live web/TUI streams carry to the footer.
                # Best-effort: a poisoned writer must not fail the compaction
                # itself.
                try:
                    session.compaction_count = int(getattr(session, "compaction_count", 0) or 0) + 1
                    if memo:
                        session.compaction_summary = _merge_compaction_summary(
                            str(getattr(session, "compaction_summary", "") or ""),
                            memo,
                        )
                    else:
                        session.compaction_summary = body
                    session.compaction_last_stats = {
                        "at": _now_iso(),
                        "before_tokens": before_tokens,
                        "after_tokens": after_tokens,
                        "removed_tokens": removed_est,
                        "removed_chars": removed_chars,
                        "messages_trimmed": (
                            len(audit.get("truncated_tool_call_ids") or [])
                            + len(audit.get("dropped_tool_call_ids") or [])
                            + int(audit.get("dropped_turns") or 0)
                        ),
                        "context_window": ctx_window,
                    }
                    session.save()
                except Exception:  # noqa: BLE001
                    pass
                self._log_event(
                    "COMPACTION_DIRECT",
                    session_id=getattr(session, "hash", ""),
                    removed_tokens_est=removed_est,
                    removed_chars=removed_chars,
                    before_tokens=before_tokens,
                    after_tokens=after_tokens,
                    llm_summary=bool(memo),
                    compaction_count=int(getattr(session, "compaction_count", 0) or 0),
                )
                # A hard ``/abort`` that landed mid-summary was already
                # honoured by ``_await_unless_hard_cancel`` (the summary HTTP
                # call is torn down and the mechanical squeeze completes);
                # mirror the AUTO path's post-compaction checkpoint and end
                # the turn ``aborted`` so the operator's kill is visible.
                aborted = bool(self._cancelled.get(getattr(session, "hash", ""), False))
                if child is not None:
                    child.finish(content=body, aborted=aborted)
                yield ToolCallResult(
                    call_id=call_id,
                    ok=True,
                    content=body,
                )
                yield TurnDone(
                    state="aborted" if aborted else "ok",
                    metrics={"engine": "local_llm", "direct_dispatch": name},
                )
            except Exception as exc:  # noqa: BLE001
                if child is not None:
                    child.finish(content="", aborted=True, error=str(exc))
                yield TurnError(
                    message=str(exc),
                    classification="direct_dispatch_error",
                )
                yield TurnDone(state="error")
            return
        if not name.startswith("delegate_"):
            yield TurnError(
                message=(f"direct-dispatch denied for tool {name!r} — only delegate_* and compact_context allowed"),
                classification="bad_request",
            )
            yield TurnDone(state="error")
            return

        call_id = f"direct-{turn_id[:8]}"
        yield ToolCallStarted(call_id=call_id, name=name, arguments=args)

        shim = delegate_shims.delegate_shim_for_tool(name)
        if shim is None:
            yield ToolCallResult(call_id=call_id, ok=False, content=f"no shim for {name}")
            yield TurnDone(state="error")
            return

        # Build the args actually handed to the shim from a COPY so the
        # ToolCallStarted event above keeps the operator's raw args.
        #
        # Two fixes ride here for the slash-command delegate path:
        #   1. Thread the parent sharedagents session id explicitly. The shim
        #      runs in-process in the gateway, where SHAREDAGENTS_SESSION_ID is
        #      absent — so without this the child CLI session was recreated
        #      fresh every call (no --resume, no parent→child link).
        #   2. Prepend an <sharedagents-context> preamble so the delegate sees
        #      the prior + between-invoke conversation. Guarded: if the
        #      frontend already injected one (the TUI does), skip to avoid
        #      double context.
        shim_args = dict(args)
        shim_args["_sharedagents_parent_session_id"] = getattr(session, "hash", "") or ""
        # Thread the TRUE cost root + ancestor goal from the parent session we
        # hold IN MEMORY (correct agent dir). A CROSS-AGENT child (main→claude)
        # can't look parent_sid up in ITS OWN sessions.db — a different-agent
        # parent isn't there, so it would silently self-root to depth-1.
        # The parent's own cost_root is set when the
        # parent is itself a subagent; else the parent IS the root → use its hash.
        if not shim_args.get("_sharedagents_cost_root_session_id"):
            shim_args["_sharedagents_cost_root_session_id"] = (
                getattr(session, "cost_root_session_id", "") or getattr(session, "hash", "") or ""
            )
        if not shim_args.get("_sharedagents_cost_ancestor_goal_id"):
            from sharedagents_goal.goal_service import session_charged_goal_id

            shim_args["_sharedagents_cost_ancestor_goal_id"] = session_charged_goal_id(session)
        if name in ("delegate_claude", "delegate_codex", "delegate_antigravity"):
            prompt_text = str(shim_args.get("prompt") or "")
            if "<sharedagents-context>" not in prompt_text:
                try:
                    resumable = _delegate_resume_available(session, name)
                    preamble = _build_direct_delegate_preamble(
                        session,
                        name,
                        resumable=resumable,
                    )
                except Exception:  # noqa: BLE001
                    preamble = ""
                if preamble:
                    shim_args["prompt"] = preamble + prompt_text

        buf: list[str] = []
        ev_queue: asyncio.Queue[Any] = asyncio.Queue()
        inner_call_ids: dict[str, str] = {}

        def _inner_call_id(raw: Any) -> str:
            return _namespace_inner_call_id(call_id, raw, inner_call_ids)

        async def _on_delta(chunk: str) -> None:
            if chunk:
                buf.append(chunk)
                await ev_queue.put(
                    TextDelta(
                        text=chunk,
                        channel="tool_delta",
                        call_id=call_id,
                    )
                )

        async def _on_event(payload: dict) -> None:
            kind = (payload or {}).get("kind") or ""
            if kind == "tool_use":
                await ev_queue.put(
                    ToolCallStarted(
                        call_id=_inner_call_id(payload.get("id")),
                        name=str(payload.get("name") or "?"),
                        arguments=dict(payload.get("input") or {}),
                        from_subagent=True,  # child's inner tool — display only, not persisted in parent log
                    )
                )
            elif kind == "tool_result":
                await ev_queue.put(
                    ToolCallResult(
                        call_id=_inner_call_id(payload.get("tool_use_id")),
                        ok=bool(payload.get("ok", True)),
                        content=str(payload.get("text") or ""),
                        from_subagent=True,  # child's inner tool — display only, not persisted in parent log
                    )
                )
            elif kind == "sensitive_egress_flag":
                await ev_queue.put(_egress_flag_delta(payload))

        async def _drive() -> Any:
            try:
                return await shim(
                    shim_args,
                    on_delta=_on_delta,
                    on_event=_on_event,
                )
            finally:
                await ev_queue.put(None)

        driver = asyncio.create_task(_drive())
        started_at = _now_iso()
        try:
            while True:
                ev = await ev_queue.get()
                if ev is None:
                    break
                yield ev
            result = await driver
        except asyncio.CancelledError:
            if not driver.done():
                driver.cancel()
                try:
                    await driver
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            yield ToolCallResult(
                call_id=call_id,
                ok=False,
                content="".join(buf) + "\n[cancelled]",
            )
            yield TurnDone(state="aborted")
            return
        except Exception as exc:  # noqa: BLE001
            yield ToolCallResult(call_id=call_id, ok=False, content=str(exc))
            yield TurnError(message=str(exc), classification="direct_dispatch_error")
            yield TurnDone(state="error")
            return

        body = str(getattr(result, "content", "") or "".join(buf))
        ok = bool(getattr(result, "ok", True))
        # Fence the delegate CLI's reply as untrusted data before it
        # reaches model context. This slash-command dispatch path
        # (/claude //codex /gemini) assembles the SAME external-content
        # class the model-invoked delegate_* path fences; leaving it raw
        # is the D2 injection bypass. Mirror the model-invoked sink.
        from sharedagents_core.injection_guard import wrap_untrusted

        fenced_body = wrap_untrusted(
            body,
            source=f"delegate:{name.replace('delegate_', '') or name}",
        )
        yield ToolCallResult(
            call_id=call_id,
            ok=ok,
            content=fenced_body,
            outcome=getattr(result, "outcome", None),
            member_outcomes=getattr(result, "member_outcomes", ()),
        )
        # The delegate shim surfaces, via this back-channel, BOTH the child
        # CLI's own session id (claude/codex/antigravity resume token) AND
        # the child sharedagents session id (the delegate agent's own session in
        # ``agents/<engine>/sessions/``). Pass them on TurnDone so the
        # PARENT session records the full sharedagents↔delegate link:
        #   * ``engine_session_id`` → Agent folds into the JSONL header +
        #     sessions.db ``sessions_meta.engine_session_id`` (existing).
        #   * the child sharedagents session id + engine id ride in ``metrics``
        #     so the parent TURN record links to the exact delegate session
        #     (queryable from the parent session log, not just the sidecar).
        child_eid = str(shim_args.get("_sharedagents_engine_session_id_out") or "")
        child_sid = str(shim_args.get("_sharedagents_child_session_id_out") or "")
        yield TurnDone(
            state="ok" if ok else "error",
            engine_session_id=child_eid,
            metrics={
                "engine": "local_llm",
                "direct_dispatch": name,
                "delegate_engine": name.replace("delegate_", "") or name,
                "delegate_child_session_id": child_sid,
                "delegate_engine_session_id": child_eid,
                "started_at": started_at,
                "ended_at": _now_iso(),
                "size": len(body),
            },
        )

    def _runtime_allowed_tools(
        self,
        session: Any,
        promoted: set[str] | None = None,  # vestigial: promotion removed
    ) -> set[str]:
        """Effective tool surface for dispatch gating — matches the wire.

        The agent's permission set, narrowed to the tools granted to the Web
        user whose grants the session is bound to — the
        same narrowing ``worker_request._assemble_tools`` applies to the wire.
        """
        from sharedagents_core.access_grants import filter_tool_names
        from sharedagents_session.access_binding import session_access_grants

        allowed = ToolDispatchMixin._agent_allowed_tools(self, session)
        sid = getattr(session, "hash", "") or ""
        sessions_dir = getattr(self.cfg, "sessions_dir", None)
        if sid and sessions_dir:
            grants = session_access_grants(sessions_dir, sid)
            if grants is not None:
                allowed = filter_tool_names(allowed, grants)
        # ``load_tools`` exists exactly when the wire offers it — the
        # agent has MCP servers it reaches only on demand.
        allowed = set(allowed) - {"load_tools"}
        if ToolDispatchMixin._deferred_mcp_servers(self, session, allowed):
            allowed = set(allowed) | {"load_tools"}
        return allowed

    def _deferred_mcp_servers(
        self,
        session: Any,
        allowed: set[str],
    ) -> dict[str, list[str]]:
        """``{server: [tool, …]}`` this agent sees only on demand;
        the same computation ``worker_request._assemble_tools`` applies."""
        from sharedagents_agent.mcp_deferral import deferred_servers
        from sharedagents_agent.tools.registry import acp_session_mcp_tools_sync

        sid = getattr(session, "hash", "") or ""
        try:
            return deferred_servers(
                self.cfg,
                {n for n in allowed if n != "load_tools"},
                session_mcp_names=([t.name for t in acp_session_mcp_tools_sync(sid)] if sid else ()),
            )
        except Exception:  # noqa: BLE001 - never break the gate over the catalog
            return {}

    def _agent_allowed_tools(self, session: Any) -> set[str]:
        """The agent's own tool surface (before any per-user narrowing).

        The agent's permission set (``compose_tool_allowlist``) IS
        the tool set. No core/deferred split, no promotion, no discovery — the
        gate presents and allows exactly what the permissions grant. Keeps the
        max-steps / budget handoff special-cases.
        """
        from sharedagents_agent.agent_profiles import compose_tool_allowlist
        from sharedagents_agent.tools.registry import (
            BUILTIN_TOOLS,
            ToolRegistry,
            acp_session_mcp_tools_sync,
            discovered_mcp_tools_sync,
            rewrite_agent_tools_in_allowlist,
        )

        sid = getattr(session, "hash", "")
        if self._max_steps_handoff.get(sid):
            return set()
        if self._budget_handoff.get(sid):
            # Keep goal_budget available during budget handoff so the agent can
            # raise its own budget and continue — but honor disabled_tools so a
            # blocklisted goal_budget can't slip past the gate (keeps wire==gate:
            # _build_worker_request post-filters disabled from the wire too).
            gb = {"budget"}
            disabled = getattr(self.cfg, "disabled_tools", None)
            if disabled:
                gb -= set(disabled)
            return gb
        session_mcp = [t.name for t in acp_session_mcp_tools_sync(sid)] if sid else []
        allowed = compose_tool_allowlist(self.cfg, session_mcp_names=session_mcp)
        if allowed is None:
            # Full-access agent: every builtin + all discovered/session MCP.
            registry = ToolRegistry(BUILTIN_TOOLS)
            allowed = set(registry.names()) | {t.name for t in discovered_mcp_tools_sync()} | set(session_mcp)
            # Defense-in-depth (this is a permission boundary): compose_tool_
            # allowlist only returns None when there is no blocklist, so this is
            # normally a no-op — but subtract disabled_tools here too so the
            # dispatch gate can never allow a disabled name via the None path.
            disabled = getattr(self.cfg, "disabled_tools", None)
            if disabled:
                allowed -= set(disabled)
        allowed = rewrite_agent_tools_in_allowlist(
            allowed,
            getattr(self.cfg, "delegation_mode", None),
        )
        # Delegation-depth cap: an agent already at the
        # max delegation depth is a LEAF — strip its delegation tools so it does
        # its own work instead of spawning a 4th level. The shared delegate chokepoint
        # (``delegate_shims._drive_child_agent_turn``) refuses the spawn regardless
        # (engine-agnostic hard backstop for CLI/ACP parents); removing the tool
        # here means the local model never sees it and never tries. Depth comes from the env the parent
        # seeds on spawn (SHAREDAGENTS_SANDBOX_AGENT_DEPTH); the root gateway is 0.
        try:
            from sharedagents_agent.sandbox_broker import MAX_DELEGATION_DEPTH

            _depth = int(os.environ.get("SHAREDAGENTS_SANDBOX_AGENT_DEPTH", "0"))
            if _depth >= MAX_DELEGATION_DEPTH:
                allowed = {
                    t
                    for t in allowed
                    if t not in ("AgentSync", "AgentAsync", "BatchAgent") and not t.startswith("delegate_")
                }
        except Exception:
            pass
        return allowed
