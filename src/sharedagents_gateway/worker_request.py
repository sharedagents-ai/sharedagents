"""Worker request assembly + subprocess spawn for :class:`CoreBackend`.

Extracted from ``acp_bridge.py`` (behavior-preserving de-monolith).

* ``_build_worker_request`` assembles a single ``sharedagents_gateway.worker``
  request payload (endpoint, model, messages, model-aware sampling,
  tools, tool_choice), including the vLLM/SGLang-vs-OpenAI-compat knob
  gating and per-row auth headers.
* ``_default_spawn_worker`` spawns the real worker subprocess in its own
  process group and returns ``(TurnState, WorkerEventStream)``.

Re-exported from ``sharedagents_gateway.acp_bridge`` for import stability —
``sharedagents_agent.agents.local_llm`` imports ``_build_worker_request`` /
``_default_spawn_worker`` LAZILY from that path on the live turn loop.

``_default_spawn_worker`` needs three low-level helpers that remain in
``acp_bridge`` (``_blocks_to_user_content``, ``_resolve_images_for_model``,
``_subprocess_event_stream``). It imports them function-locally to keep
the module dependency acyclic (``acp_bridge`` re-exports this module) and
robust to import ordering.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sharedagents_agent.tools import (
    BUILTIN_TOOLS,
    ToolRegistry,
    acp_session_mcp_tools_sync,
    discovered_mcp_tools_sync,
)
from sharedagents_agent.turn_runtime import TurnState
from sharedagents_core import oscompat
from sharedagents_model import wire_replay
from sharedagents_model.bedrock_config import (
    AUTH_API_KEY,
    BEDROCK_API,
    BEDROCK_BACKEND,
    BEDROCK_BASE_SCHEME,
    BEDROCK_BEARER_ENV,
    BedrockConfigError,
    is_bedrock_row,
    validate_bedrock_row,
)
from sharedagents_model.model_parameters import (
    ModelParametersError,
    effort_capability,
    resolve_model_parameters,
)
from sharedagents_model.models import (
    derive_effort_capability,
    for_name,
    merge_overrides,
    resolve_hosted_reasoning_effort,
    resolve_max_tokens,
    row_is_hosted,
    sampling_to_request_fields,
)
from sharedagents_model.sampling import apply_sampling, configured_sampling

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sharedagents_session.session import Session

    from .acp_bridge import CoreBackend, WorkerEventStream


def _build_worker_request(
    backend: CoreBackend,
    sess: Session,
    messages: list[Mapping[str, Any]],
    *,
    include_tools: bool = True,
    allowed_tool_names: set[str] | None = None,
    sampling: dict[str, Any] | None = None,
    endpoint_override: str | None = None,
    model_override: str | None = None,
) -> dict[str, Any]:
    """Assemble a single worker request payload.

    Shape: endpoint, model, messages, model-aware sampling, tools,
    tool_choice. Tool schemas come from :data:`BUILTIN_TOOLS` so the
    worker forwards the OpenAI-shaped list to the upstream and the
    model can emit ``tool_calls``.

    ``allowed_tool_names`` (when given) filters the tools[] payload to
    just those names — the engine passes ``core_tools | promoted`` so
    deferred tool schemas don't ride every round (saves ~30 KB / round
    on a small-model context). When None (legacy callers), every tool
    in the registry ships.

    ``endpoint_override`` / ``model_override`` (when given) substitute
    the session's endpoint/model for THIS payload only — the engine's
    LLM-fallback failover routes a round through the ``role: fallback``
    deployment without mutating the durable session header.
    """
    eff_model = str(model_override or sess.model or "")
    eff_endpoint = str(endpoint_override or sess.endpoint or "")
    cfg = getattr(backend, "cfg", None)
    if cfg is not None:
        if not eff_model or eff_model.lower() in ("auto", "ask", "plan"):
            eff_model = str(getattr(cfg, "upstream_model", "") or eff_model)
        if not eff_endpoint:
            eff_endpoint = str(getattr(cfg, "upstream_endpoint", "") or eff_endpoint)
    mdl = for_name(eff_model)
    sampling_disabled = os.environ.get(
        "SHAREDAGENTS_DISABLE_MODEL_SAMPLING",
        "0",
    ) in ("1", "true", "True")

    # Resolve the configured deployment row before sampling is materialised.
    # Effort needs both the row's backend/capability metadata and the selected
    # model family, while the existing sampler only needed the latter.
    try:
        row = (
            backend.cfg.model_row_for_endpoint(
                eff_endpoint,
                eff_model,
            )
            or {}
        )
    except Exception:  # noqa: BLE001
        row = {}
    row_backend = (
        str(
            row.get("backend") or getattr(sess, "backend", "") or getattr(backend.cfg, "backend", "") or "",
        )
        .strip()
        .lower()
    )
    # ``row_is_hosted`` owns "is this a hosted, authenticated remote API?"
    # (OpenAI-family label / apiKeyEnv / static auth header, minus
    # positively-local signals). The picker's "the wire applies this effort"
    # flag and the transport share this classifier.
    hosted_row = row_is_hosted(row)
    # Legacy single-primary configs often omit ``backend``.  With no auth
    # marker and no label those rows are the local vLLM path used by the
    # existing sampler; explicitly named non-vLLM backends (for example ollama)
    # stay on the generic reasoning_effort fallback.
    local_backend = row_backend in ("vllm", "sglang") or (not hosted_row and not row_backend)
    openai_compat = (not local_backend) and hosted_row
    effort = str(
        getattr(sess, "effort", "") or getattr(backend.cfg, "effort", "") or "",
    ).strip()
    row = {"id": eff_model, "backend": row_backend, **row}
    identities = {str(row.get(k) or "").strip().lower() for k in ("id", "alias", "wire_id", "model_id")}
    if eff_model.strip().lower() not in identities:
        row = {
            k: v
            for k, v in row.items()
            if k
            not in {
                "effort",
                "reasoning",
                "sampling",
                "alias",
                "wire_id",
                "model_id",
                "supported_efforts",
                "supportedEfforts",
                "supported_reasoning_levels",
                "supported_reasoning_efforts",
                "reasoning_efforts",
                "think_efforts",
            }
        }
        row["id"] = eff_model
    mode = str(getattr(sess, "system_mode", "") or getattr(sess, "mode", "") or "auto")
    declaration = resolve_model_parameters(row, mode=mode)
    capability = effort_capability(declaration) if declaration is not None else derive_effort_capability(row)
    # Prefer the vocabulary the provider actually advertised, when this process
    # has learned it: the gateway's boot warm and the picker/`/effort` probes
    # write the discovery cache, and the advertised list is remembered well past
    # that cache's freshness TTL so an expired model list cannot silently turn a
    # confirmed row into an unconfirmed one. Cache read only — never HTTP on the
    # turn path. With nothing learned (a provider that publishes no effort
    # metadata, or a row added after the last probe) the row declaration stands.
    try:
        from sharedagents_model.effort_discovery import discover_effort_capability

        _discovered = discover_effort_capability(row, backend.cfg, probe=False)
        if _discovered is not None:
            _effort = row.get("effort")
            _merged = dict(row)
            _merged["effort"] = {
                **(_effort if isinstance(_effort, dict) else {}),
                **_discovered,
            }
            capability = derive_effort_capability(_merged)
    except ModelParametersError:
        raise
    except Exception as exc:  # noqa: BLE001
        # Never fail a turn over effort resolution — but don't hide a permanent
        # breakage (a bad import, a cache-key change) either.
        log.debug("effort discovery read failed: %s", exc)
    if declaration is not None:
        capability = effort_capability(declaration)

    if sampling_disabled:
        request_fields: dict[str, Any] = {}
        template_kwargs: dict[str, Any] = {}
    else:
        mode = str(getattr(sess, "system_mode", "") or getattr(sess, "mode", "") or "auto")
        base = getattr(mdl, mode, mdl.auto)
        # OpenCode's rule: min(the model's output limit, the global cap). A
        # tuned local family's per-mode budget is only that limit's default;
        # an explicit per-turn ``sampling`` override still wins.
        base = replace(
            base,
            max_completion_tokens=resolve_max_tokens(
                row=row,
                model_id=eff_model,
                default_limit=base.max_completion_tokens,
                cfg=cfg,
            ),
        )
        profile = merge_overrides(base, sampling)
        request_fields, _ = sampling_to_request_fields(profile)
        template_kwargs = {}
        parameters = dict(declaration.get("parameters", {})) if declaration is not None else {}
        for name in ("enable_thinking", "thinking_budget"):
            key = "chat_template_kwargs." + name
            value = parameters.get(key)
            if sampling is not None and name in sampling:
                value = sampling[name]
            if value is not None:
                template_kwargs[name] = value

    # Native Bedrock Converse rows take a SEPARATE, smaller request shape.
    # The worker dispatches on ``api`` before any OpenAI payload assembly,
    # so only the fields the Converse adapter understands ride the wire:
    # route identity, the common sampling fields, tools / tool_choice.
    # No auth headers (credentials live in the child environment — see
    # :func:`worker_env_for_model`), no OpenAI-only extras, no effort /
    # reasoning fields (v1 ships no extended reasoning).
    if is_bedrock_row(row):
        route = validate_bedrock_row(row)
        bedrock_request: dict[str, Any] = {
            "endpoint": eff_endpoint,
            "model": eff_model,
            "messages": wire_replay.sanitize_messages(messages, row),
            "api": BEDROCK_API,
            "backend": BEDROCK_BACKEND,
            "region": route.region,
            "auth": route.auth,
        }
        if route.profile:
            bedrock_request["profile"] = route.profile
        if not sampling_disabled:
            # Only the common inferenceConfig fields Converse accepts.
            for _k in ("max_tokens", "top_p"):
                if request_fields.get(_k) is not None:
                    bedrock_request[_k] = request_fields[_k]
        override = getattr(sess, "max_tokens_override", None)
        if isinstance(override, int) and override > 0:
            bedrock_request["max_tokens"] = override
        resolved = configured_sampling(cfg, row, eff_model, override=sampling, mode=mode)
        for key in ("temperature", "top_p"):
            bedrock_request.pop(key, None)
            if key in resolved:
                bedrock_request[key] = resolved[key]
        if include_tools:
            bedrock_request["tools"], bedrock_request["tool_choice"] = _assemble_tools(
                backend, sess, messages, allowed_tool_names
            )
        return bedrock_request

    # Replayed reasoning is a DeepSeek EXTENSION, not part of the OpenAI
    # message schema every "OpenAI-compatible" host implements. DeepSeek
    # refuses a tool follow-up without it, and strict hosts (Cerebras, Groq)
    # refuse the request WITH it ("property 'messages.N.assistant.
    # reasoning_content' is unsupported", HTTP 400) — so the field is
    # reconciled against what this endpoint is known to accept. The engine's
    # list is never mutated: the same history can be re-sent to another
    # deployment after a failover.
    request: dict[str, Any] = {
        "endpoint": eff_endpoint,
        "model": eff_model,
        "messages": wire_replay.sanitize_messages(messages, row),
    }
    request.update(request_fields)
    # Backend-gate the vLLM/SGLang-only knobs. The sampling PROFILE is chosen
    # by ``for_name(model_id)`` (a substring match), NOT by the deployment's
    # backend — so an external OpenAI-compatible endpoint whose model id happens
    # to contain ``qwen3.6`` / ``gemma4`` (Together, DeepInfra, SiliconFlow, …)
    # would otherwise receive ``top_k`` / ``min_p`` / ``repetition_penalty`` and
    # ``chat_template_kwargs`` — none of which stock OpenAI / DeepSeek /
    # Anthropic-compat servers accept, producing a hard 400.
    #
    # Detecting "hosted OpenAI-compat" by backend NAME alone is not enough: the
    # provider probe (providers.detect_from_models) and the setup wizard only
    # ever emit ``vllm`` / ``sglang`` / ``unknown`` — a DeepSeek/OpenAI/OpenRouter
    # row is ``unknown``, not ``openai`` (the shipped default row is exactly
    # ``backend:"unknown"`` + ``apiKeyEnv`` on OpenRouter). So the reliable
    # signal is AUTH: a local vLLM/SGLang server never needs an api key, so any
    # row that carries an ``apiKeyEnv`` is a hosted OpenAI-compat API. We STRIP
    # when the row is hosted (explicit openai-family backend OR carries an api
    # key) and NOT positively local (``vllm``/``sglang``) — so a local row is
    # never degraded, whatever its backend label. (Reviewers: cursor+claude.)
    if openai_compat:
        # Non-standard chat/completions params rejected by OpenAI-compat APIs.
        for _k in ("top_k", "min_p", "repetition_penalty"):
            request.pop(_k, None)
    # Per-session max_tokens override — set by the engine's reactive
    # overflow squeeze when compact_messages can't trim further and the
    # only way to fit input + output ≤ ctx_window is to give back some
    # output budget. Honoured over the model sampling profile's
    # max_completion_tokens so the retried round actually fits instead
    # of relooping on the same upstream 400. Stays on the session for
    # the rest of the session lifetime — operator can /clear to reset
    # if they later add more context room.
    override = getattr(sess, "max_tokens_override", None)
    if isinstance(override, int) and override > 0:
        request["max_tokens"] = override
    if not sampling_disabled:
        apply_sampling(request, configured_sampling(cfg, row, eff_model, override=sampling, mode=mode))
    extra: dict[str, Any] = {}
    # ``chat_template_kwargs`` (enable_thinking / thinking_budget) and the
    # configured local sampling knobs (e.g. ``repetition_detection``) are
    # vLLM/SGLang-only — never send them to an OpenAI-compatible upstream.
    if not openai_compat and not sampling_disabled:
        if template_kwargs:
            extra["chat_template_kwargs"] = template_kwargs
        for key, value in (declaration or {}).get("parameters", {}).items():
            if (
                key
                in {"dry_multiplier", "dry_base", "dry_allowed_length", "dry_sequence_breakers", "repetition_detection"}
                and value is not None
            ):
                extra[key] = value
    if extra:
        request["extra"] = extra

    # A declaration or advertisement is the only authority for effort. Unknown
    # and stale selections are omitted for every backend, including local ones.
    if not sampling_disabled:
        level = resolve_hosted_reasoning_effort(effort, capability)
        if effort and effort.lower() != "default" and level is None:
            log.warning("selected effort is unavailable for this model; omitting it")
        if level is not None:
            value = capability.get("wire_values", {}).get(level, level)
            wire = capability.get("wire") or "reasoning_effort"
            if wire == "reasoning_effort":
                request[wire] = value
            elif wire == "chat_template_kwargs.thinking_budget":
                if sampling is not None and "thinking_budget" in sampling:
                    value = sampling["thinking_budget"]
                if not local_backend:
                    raise ModelParametersError("thinking_budget effort requires a local vllm/sglang route")
                if value is not None:
                    request.setdefault("extra", {}).setdefault("chat_template_kwargs", {})["thinking_budget"] = value
    speed = str(getattr(sess, "speed", "") or "").strip().lower()
    speed_cap = row.get("speed") if isinstance(row.get("speed"), dict) else None
    speed_wire = str(speed_cap.get("wire") or "").strip() if speed_cap else ""
    speed_values = speed_cap.get("values") if speed_cap else None
    if (
        not sampling_disabled
        and speed
        and speed != "default"
        and speed_wire
        and isinstance(speed_values, list)
        and speed in {str(v).strip().lower() for v in speed_values}
    ):
        request[speed_wire] = speed
        request["_forward_fields"] = list(request.get("_forward_fields", [])) + [speed_wire]
    # Resolve per-row auth + custom headers (OpenRouter Bearer token,
    # X-Title, etc.) from the AgentConfig and ship them on the wire
    # request so the worker subprocess attaches them to the upstream
    # POST. Without this the worker hits the upstream with NO
    # Authorization header — fine for local vLLM rows, but the moment
    # the operator switches to ``kimi`` (OpenRouter) every request
    # comes back HTTP 401. Empty dict for unconfigured rows is safe.
    try:
        upstream_hdrs = backend.cfg.upstream_headers(
            eff_endpoint,
            eff_model,
        )
    except Exception:  # noqa: BLE001
        upstream_hdrs = {}
    # Always include X-Session-Id so the vllm-mux router can maintain
    # prefix-cache stickiness across turns.  On the very first turn of
    # a new session (zero completed turns) we also set X-Session-New
    # so the router can pick the least-loaded replica instead of the
    # HRW-hash winner.
    headers = dict(upstream_hdrs) if upstream_hdrs else {}
    headers["x-session-id"] = sess.hash
    if len(sess.turns) == 0:
        headers["x-session-new"] = "true"
    request["headers"] = headers
    from sharedagents_gateway.codex_responses import (
        CODEX_OAUTH_AUTH,
        CODEX_RESPONSES_API,
        is_codex_responses_endpoint,
    )

    row_api = str(row.get("api") or "").strip()
    row_auth = str(row.get("auth") or "").strip()
    if (
        row_api in (CODEX_RESPONSES_API, "openai-chatgpt-responses")
        or row_auth == CODEX_OAUTH_AUTH
        or is_codex_responses_endpoint(str(request.get("endpoint") or ""))
    ):
        request["api"] = CODEX_RESPONSES_API
        request["auth"] = CODEX_OAUTH_AUTH
        try:
            from sharedagents_agent.discovery.openai_codex import (
                codex_oauth_request_headers,
            )

            oauth_hdrs = codex_oauth_request_headers()
        except Exception:  # noqa: BLE001
            oauth_hdrs = {}
        if oauth_hdrs:
            headers.update(oauth_hdrs)
            request["headers"] = headers
        for _k in (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
            "seed",
        ):
            request.pop(_k, None)
    if include_tools:
        request["tools"], request["tool_choice"] = _assemble_tools(
            backend,
            sess,
            messages,
            allowed_tool_names,
        )
    # Catalog ``provider`` (xAI / DeepSeek / OpenRouter / …) is the
    # deployment the worker actually selected. Stamp it on the request so
    # metrics → ledger can persist it instead of guessing from the model id.
    _catalog_provider = str(row.get("provider") or "").strip()
    if _catalog_provider:
        request["provider"] = _catalog_provider
    return request


def _assemble_tools(
    backend: CoreBackend,
    sess: Session,
    messages: list[Mapping[str, Any]],
    allowed_tool_names: set[str] | None,
) -> tuple[list[dict[str, Any]], Any]:
    """Assemble the ``tools[]`` payload and ``tool_choice`` for one request.

    Shared by the OpenAI-compatible path and the native Bedrock path so
    tool-schema resolution cannot drift. ``tool_choice`` is always ``auto``.
    """
    # Sync cache read — the async sites in `prompt()` /
    # `_run_direct_tool_call` warm the discovery cache before this
    # function (called from `_default_spawn_worker`) runs.
    registry = ToolRegistry(
        BUILTIN_TOOLS + discovered_mcp_tools_sync() + acp_session_mcp_tools_sync(getattr(sess, "hash", "") or "")
    )
    # Present the agent's FULL permission set every turn —
    # no core/deferred split, no promotion, no discovery. The permission
    # set (allowlist + expanded MCP grants, incl. session-scoped MCP) IS
    # the tool set.
    #
    # ``allowed_tool_names`` is honored ONLY as an engine handoff override:
    # when non-None it is the AUTHORITATIVE ceiling for this round (the
    # engine's ``_spawn_worker`` passes {} on a max-steps handoff to empty
    # tools[] and force a text wrap-up, or {"budget"} on a budget
    # handoff). None (the normal case) means "derive from permissions" —
    # the legacy core∪promoted hint is gone, so nothing else passes a set.
    if allowed_tool_names is not None:
        tools = registry.filtered_schemas(set(allowed_tool_names))
    else:
        from sharedagents_agent.agent_profiles import compose_tool_allowlist
        from sharedagents_agent.tools.registry import rewrite_agent_tools_in_allowlist

        session_mcp = [t.name for t in acp_session_mcp_tools_sync(getattr(sess, "hash", "") or "")]
        allowed = compose_tool_allowlist(backend.cfg, session_mcp_names=session_mcp)
        if allowed is None:
            allowed = rewrite_agent_tools_in_allowlist(
                {t["function"]["name"] for t in registry.openai_schemas()},
                getattr(backend.cfg, "delegation_mode", None),
            )
        else:
            allowed = rewrite_agent_tools_in_allowlist(
                set(allowed),
                getattr(backend.cfg, "delegation_mode", None),
            )
        tools = registry.filtered_schemas(allowed)
    # Per-agent blocklist: remove any tools named in
    # ``cfg.disabled_tools`` (e.g. UserQuestion / AskUserQuestion
    # for ACP clients that don't implement session/ask_user).
    disabled = getattr(backend.cfg, "disabled_tools", None)
    if disabled:
        disabled_set = set(disabled)
        tools = [t for t in tools if t["function"]["name"] not in disabled_set]
    # A Web user's session only sees the tools granted to them — the same
    # narrowing the dispatch gate applies.
    from sharedagents_core.access_grants import filter_tool_names
    from sharedagents_session.access_binding import session_access_grants

    sid = getattr(sess, "hash", "") or ""
    sessions_dir = getattr(backend.cfg, "sessions_dir", None)
    if sid and sessions_dir:
        grants = session_access_grants(sessions_dir, sid)
        if grants is not None:
            keep = filter_tool_names({t["function"]["name"] for t in tools}, grants)
            tools = [t for t in tools if t["function"]["name"] in keep]
    # Implicitly reachable MCP servers ship their schemas only once the
    # conversation loaded them (``load_tools`` or a direct call). The handoff
    # override remains a hard ceiling. Leaf agents include load_tools in their
    # ceiling; handoffs omit it and must never gain a synthetic loader.
    if allowed_tool_names is None or "load_tools" in allowed_tool_names:
        from sharedagents_agent import mcp_deferral

        deferred = mcp_deferral.deferred_servers(
            backend.cfg,
            {t["function"]["name"] for t in tools},
            session_mcp_names=[t.name for t in acp_session_mcp_tools_sync(sid)] if sid else (),
        )
        tools = mcp_deferral.present_tools(
            tools,
            deferred,
            mcp_deferral.loaded_servers(
                messages,
                turns=getattr(sess, "turns", ()),
            ),
        )
    return tools, "auto"


def worker_env_for_model(
    cfg: Any,
    row: Mapping[str, Any] | None,
    inherited: Mapping[str, str],
) -> dict[str, str]:
    """Build the child worker environment for ``row`` (pure; no mutation).

    Non-Bedrock rows return a copy of ``inherited`` unchanged. For native
    Bedrock rows the credential source is resolved per the row's explicit
    ``auth`` mode:

    * ``bedrock-api-key`` — the token value is resolved from the per-agent
      ``cfg.api_keys`` map first (an intentional empty value is an explicit
      "no key" and BLOCKS the environment fallback), then from the inherited
      environment under the row's ``apiKeyEnv`` name (default
      ``AWS_BEARER_TOKEN_BEDROCK``). Missing/empty →
      :class:`BedrockConfigError` before any SDK call. The value is set
      ONLY in the returned child env under ``AWS_BEARER_TOKEN_BEDROCK`` —
      the variable botocore reads for the ``bedrock`` signing name.
    * ``aws-iam`` — any ambient ``AWS_BEARER_TOKEN_BEDROCK`` is removed
      from the child env so botocore signs with the normal IAM chain even
      when the gateway process holds a bearer token for another agent's
      API-key row. ``profile`` (when set on the row) is exported as
      ``AWS_PROFILE`` in the child env.

    The gateway's ``os.environ`` is never mutated.

    Traceability: plan/bedrock-native-converse.md (BED-02)
    """
    env = dict(inherited)
    if not is_bedrock_row(row):
        return env
    route = validate_bedrock_row(row)
    if route.profile:
        env["AWS_PROFILE"] = route.profile
    if route.auth == AUTH_API_KEY:
        source = str(row.get("apiKeyEnv") or row.get("api_key_env") or "").strip() or BEDROCK_BEARER_ENV
        api_keys = getattr(cfg, "api_keys", None) or {}
        value = str(api_keys.get(source) or "") if source in api_keys else str(inherited.get(source) or "")
        if not value:
            raise BedrockConfigError(
                f"Bedrock API-key row {route.model_id!r}: credential source "
                f"{source!r} is unset or explicitly blanked — refusing to "
                "fall back to another credential. Set the value in the "
                "agent's secrets.env or the gateway environment."
            )
        env[BEDROCK_BEARER_ENV] = value
    else:
        # aws-iam: force the SigV4 chain by removing any ambient bearer
        # token (botocore prefers bearer for the ``bedrock`` signing name
        # whenever the env var is present).
        env.pop(BEDROCK_BEARER_ENV, None)
    return env


def bedrock_oneshot_request(
    row: Mapping[str, Any],
    messages: list[Mapping[str, Any]],
    *,
    temperature: float | None = None,
    sampling: dict[str, Any] | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Build the worker request for a bounded ONE-SHOT Bedrock call.

    The out-of-band inference paths (image transcription, compaction
    summaries, loop distillation) are not the user's turn: no tools, no
    streaming to the client. The worker dispatches on ``api`` straight to
    the Converse transport, so only route identity, the messages and the
    common sampling fields ride the wire. Credentials NEVER ride the
    request — the child environment carries them
    (:func:`worker_env_for_model`).

    Traceability: plan/bedrock-native-converse.md (BED-03/05)
    """
    route = validate_bedrock_row(row)
    request: dict[str, Any] = {
        "endpoint": f"{BEDROCK_BASE_SCHEME}://{route.region}",
        "model": route.model_id,
        "messages": [dict(m) for m in messages],
        "api": BEDROCK_API,
        "backend": BEDROCK_BACKEND,
        "region": route.region,
        "auth": route.auth,
    }
    if route.profile:
        request["profile"] = route.profile
    if max_tokens is not None:
        request["max_tokens"] = int(max_tokens)
    resolved = dict(sampling or {})
    if temperature is not None:
        resolved["temperature"] = temperature
    resolved = configured_sampling(None, model_id=route.model_id, override=resolved)
    request.update({k: v for k, v in resolved.items() if k in ("temperature", "top_p")})
    return request


async def run_bedrock_oneshot(
    cfg: Any,
    row: Mapping[str, Any],
    messages: list[Mapping[str, Any]],
    *,
    temperature: float | None = None,
    sampling: dict[str, Any] | None = None,
    max_tokens: int | None = None,
    timeout_s: float = 90.0,
) -> tuple[str, dict[str, Any] | None]:
    """Run one bounded Bedrock Converse turn in a worker subprocess.

    Used by the non-turn inference paths when the selected row is native
    Bedrock — they must never POST to ``/chat/completions`` at a
    synthetic ``bedrock://`` endpoint. Returns ``(text, metrics)``;
    raises :class:`RuntimeError` on a worker error, timeout or empty
    output so callers keep their existing mechanical-fallback behavior.

    Traceability: plan/bedrock-native-converse.md (BED-03/04)
    """
    # Function-local: ``acp_bridge`` re-exports this module's helpers, so a
    # module-level import here would be cyclic.
    from .acp_bridge import _run_oneshot_worker

    request = bedrock_oneshot_request(
        row,
        messages,
        temperature=temperature,
        sampling=sampling,
        max_tokens=max_tokens,
    )
    env = worker_env_for_model(cfg, row, os.environ)
    text, metrics = await _run_oneshot_worker(
        request,
        timeout_s=timeout_s,
        env_override=env,
        want_metrics=True,
    )
    return text, metrics


async def _default_spawn_worker(
    backend: CoreBackend,
    session_id: str,
    prompt_blocks: list[Mapping[str, Any]],
    turn_id: str,
    messages: list[Mapping[str, Any]] | None = None,
    *,
    allowed_tool_names: set[str] | None = None,
    nudge_retry_count: int = 0,
    sampling: dict[str, Any] | None = None,
    endpoint_override: str | None = None,
    model_override: str | None = None,
) -> tuple[TurnState, WorkerEventStream]:
    """Spawn a real ``sharedagents_gateway.worker`` subprocess.

    The request includes the assembled message list plus the OpenAI
    tool schemas (from :data:`BUILTIN_TOOLS`) and ``tool_choice="auto"``
    so the model can emit ``tool_calls`` events the bridge dispatches.

    On the first round (``messages is None``) the message list is
    derived from the session's persisted history + the new prompt.
    On follow-up rounds the caller passes the already-augmented
    list (with prior tool_calls / role=tool entries appended).

    ``endpoint_override`` / ``model_override`` (when given) route THIS
    round through an alternate deployment (the engine's LLM-fallback
    failover) without touching the session's durable model/endpoint.
    """
    # These low-level helpers stay in ``acp_bridge``; import them here
    # (function-local) so this module stays acyclic — ``acp_bridge``
    # re-exports ``_build_worker_request`` / ``_default_spawn_worker``.
    from .acp_bridge import (
        _blocks_to_user_content,
        _resolve_images_for_model,
        _subprocess_event_stream,
    )

    sess = backend._session_for(session_id)
    if sess is None:
        raise RuntimeError(f"unknown session: {session_id!r}")

    if messages is None:
        from sharedagents_core.attachment_context import model_blocks, model_user

        user_content = _blocks_to_user_content(model_blocks(prompt_blocks))
        msgs: list[dict[str, Any]] = []
        if sess.system_prompt:
            msgs.append({"role": "system", "content": sess.system_prompt})
        for prior in sess.turns:
            u = model_user(prior.get("user"), prior.get("attachments"))
            if u:
                msgs.append({"role": "user", "content": u})
            c = prior.get("content")
            if c:
                msgs.append({"role": "assistant", "content": c})
        msgs.append({"role": "user", "content": user_content})
    else:
        msgs = list(messages)

    # Make any image content safe for the selected model — transcribe
    # via a vision model when the selected one is text-only, so a
    # ``deepseek-v4-flash``-style upstream doesn't 400 on ``image_url``.
    msgs = await _resolve_images_for_model(
        backend,
        sess,
        msgs,
        endpoint_override=endpoint_override,
        model_override=model_override,
    )

    request = _build_worker_request(
        backend,
        sess,
        msgs,
        allowed_tool_names=allowed_tool_names,
        sampling=sampling,
        endpoint_override=endpoint_override,
        model_override=model_override,
    )
    # Sampler seed rotation on nudge retries: when the engine signals
    # this round is a retry (nudge_retry_count > 0) and a base seed is
    # set, rotate it so the model doesn't reproduce the same loop
    # deterministically. _build_worker_request writes `seed` at top
    # level via _REQUEST_FIELD_NAMES (models.py), so mutating
    # request["seed"] here lands on the wire payload.
    if nudge_retry_count > 0:
        base_seed = request.get("seed")
        if base_seed is not None:
            from sharedagents_agent.turn_runtime import _rotate_seed

            request["seed"] = _rotate_seed(int(base_seed), nudge_retry_count)

    package_parent = str(Path(__file__).resolve().parent.parent)
    env = dict(os.environ)
    # Per-model credential isolation (native Bedrock rows only): resolve
    # the effective deployment row for THIS request and build the child
    # env. API-key rows get their bearer token in the child env only
    # (per-agent secrets.env wins over the process env; an intentional
    # empty sentinel blocks the fallback); IAM rows get any ambient bearer
    # token removed so the SDK signs with the normal IAM chain. The
    # gateway's own os.environ is never mutated.
    try:
        eff_row = (
            backend.cfg.model_row_for_endpoint(
                str(request.get("endpoint") or ""),
                str(request.get("model") or ""),
            )
            or {}
        )
    except Exception:  # noqa: BLE001
        eff_row = {}
    env = worker_env_for_model(backend.cfg, eff_row, env)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = package_parent + (os.pathsep + existing if existing else "")
    env["PYTHONUNBUFFERED"] = "1"
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "sharedagents_gateway.worker",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        limit=16 * 1024 * 1024,
        # Own session / process group. A force-cancel SIGKILLs the whole
        # process group via ``os.killpg`` (TurnState.stop(force=True)); if
        # this subprocess shared the gateway's group, a hard ESC /
        # ``/abort`` would kill the GATEWAY itself (observed:
        # the gateway died and systemd restarted it). ``start_new_session``
        # makes it a session leader so killpg reaches only it and its
        # descendants (the bash/sleep/CLI it spawns).
        start_new_session=not oscompat.IS_WINDOWS,
        **oscompat.detach_kwargs(),
    )
    if proc.stdin is None:
        raise RuntimeError("worker stdin pipe is None")
    proc.stdin.write(json.dumps(request).encode("utf-8"))
    await proc.stdin.drain()
    proc.stdin.close()
    turn = TurnState(turn_id, session_hash=session_id, proc=proc)
    return turn, _subprocess_event_stream(proc)
