"""Response parsing identities and token/context safety profiles.

Optional model parameters and effort policy live in model_parameters.defaults.json
and the operator's model_parameters.json. This module shares their capability
resolution between discovery, commands, picker and request assembly.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from sharedagents_agent.tools.permissions import PermissionMode


def effort_to_reasoning_effort(effort: str) -> str | None:
    """Preserve a selected provider token; absence never invents a value."""
    level = str(effort or "").strip()
    return level if level and level.lower() not in {"default", "auto", "reset", "clear"} else None


def effort_capability_confirmed(capability: Mapping[str, Any]) -> bool:
    return capability.get("source") in {"configured", "shipped", "advertised"}


def resolve_hosted_reasoning_effort(effort: str, capability: Mapping[str, Any]) -> str | None:
    """Return an exact supported selection, or an explicitly configured default.

    Invalid/stale selections are omitted, never clamped or replaced. Provider
    advertisements describe support and do not opt into sending a default.
    """
    if not effort_capability_confirmed(capability):
        return None
    level = effort_to_reasoning_effort(effort)
    if level is None and capability.get("source") != "advertised":
        level = capability.get("default")
    return level if level in capability.get("values", []) else None


def effort_capability_from_metadata(metadata: Mapping[str, Any]) -> dict[str, Any] | None:
    """Preserve advertised tokens, order, and default without manufacturing any."""
    if not isinstance(metadata, Mapping):
        return None
    blocks = [
        metadata.get("reasoning"),
        metadata.get("effort"),
        metadata,
        metadata.get("reasoning_efforts"),
        metadata.get("think_efforts"),
    ]
    for block in blocks:
        if not isinstance(block, Mapping) or block.get("support") is False:
            continue
        raw = next(
            (
                block[k]
                for k in (
                    "supported_efforts",
                    "supportedEfforts",
                    "supported_levels",
                    "supportedLevels",
                    "supported_reasoning_levels",
                    "supported_reasoning_efforts",
                    "valid_efforts",
                )
                if isinstance(block.get(k), (list, tuple))
            ),
            None,
        )
        if raw is None:
            continue
        values = []
        for item in raw:
            token = item.get("effort") if isinstance(item, Mapping) else item
            if isinstance(token, str) and token.strip() and token not in values:
                values.append(token)
        if block.get("mandatory") is True:
            values = [v for v in values if v != "none"]
        default = next(
            (
                block[k]
                for k in ("default_effort", "defaultEffort", "default_level", "defaultLevel", "default_reasoning_level")
                if block.get(k) in values
            ),
            None,
        )
        return {"values": values, "default": default, "wire": "reasoning_effort", "source": "advertised"}
    return None


#: Hostnames that are unambiguously the local machine.
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "::1", ""})
#: Static header names that carry credentials.
_AUTH_HEADER_NAMES = frozenset({"authorization", "x-api-key", "api-key"})


def _local_host(host: str) -> bool:
    h = str(host or "").strip().lower().strip("[]")
    return h in _LOCAL_HOSTS or h.startswith("127.")


def row_is_hosted(row: Mapping[str, Any] | None) -> bool:
    """True when *row* targets a hosted, authenticated remote API.

    Auth is the reliable signal: a bare row (no label, no credentials) is the
    legacy local-vLLM path. Hosted = an OpenAI-family ``backend`` label, an
    ``apiKeyEnv``, or a static auth header. Positively-local signals win: a
    ``vllm``/``sglang`` label, or a loopback endpoint whose only auth is a
    static header (an authenticated local proxy).

    Single classifier shared by the wire gate
    (``sharedagents_gateway.worker_request``), the picker's "the wire applies this
    effort" flag. File declarations decide the vocabulary independently.
    """
    payload = row if isinstance(row, Mapping) else {}
    backend = str(payload.get("backend") or "").strip().lower()
    if backend in ("vllm", "sglang"):
        return False
    if backend in ("openai", "openai-compat", "anthropic", "bedrock"):
        return True
    if str(payload.get("apiKeyEnv") or payload.get("api_key_env") or "").strip():
        return True
    headers = payload.get("headers")
    if isinstance(headers, Mapping) and any(str(name).strip().lower() in _AUTH_HEADER_NAMES for name in headers):
        endpoint = str(payload.get("baseUrl") or payload.get("base_url") or payload.get("endpoint") or "").strip()
        if not endpoint:
            return True
        from urllib.parse import urlparse

        return not _local_host(urlparse(endpoint).hostname or "")
    return False


def bedrock_effort_capability() -> dict[str, Any]:
    """Effort capability for a native Bedrock row — depth is UNSUPPORTED in v1.

    Bedrock extended thinking is off in v1 (its signed ``reasoningContent``
    cannot be safely replayed under sharedagents's OpenAI-shaped history), so the
    picker and ``/effort`` must advertise no levels and the wire must never
    ship a ``reasoning_effort`` token. ``supported: False`` lets the UI mark
    the slider explicitly instead of showing an empty but live-looking one.

    Traceability: plan/bedrock-native-converse.md (BED-06)
    """
    return {
        "values": [],
        "default": "",
        "supported": False,
        "source": "bedrock-unsupported",
    }


def native_effort_capability(row: Mapping[str, Any] | None = None, model_id: str = "") -> dict[str, Any]:
    """Read a file declaration; unknown models have no invented vocabulary."""
    from .bedrock_config import is_bedrock_row
    from .model_parameters import effort_capability, resolve_model_parameters

    payload = row or {}
    if is_bedrock_row(payload):
        return bedrock_effort_capability()
    declaration = resolve_model_parameters(payload, model_id)
    return effort_capability(declaration) if declaration is not None else {"values": [], "source": "unknown"}


def derive_effort_capability(row: Mapping[str, Any], *, mode: str = "auto") -> dict[str, Any]:
    """One capability resolver for picker, commands, discovery and wire."""
    from .bedrock_config import is_bedrock_row
    from .model_parameters import effort_capability, resolve_model_parameters, validate_effort

    if is_bedrock_row(row):
        return bedrock_effort_capability()
    declaration = resolve_model_parameters(row, mode=mode)
    if declaration is not None:
        return effort_capability(declaration)
    override = row.get("effort")
    if "effort" in row and (override is None or isinstance(override, Mapping) and "values" in override):
        capability = validate_effort(override, "model row effort", legacy=True)
        source = override.get("source") if isinstance(override, Mapping) else None
        capability["source"] = source if source in {"advertised", "unknown", "bedrock-unsupported"} else "configured"
        return capability
    return effort_capability_from_metadata(row) or {"values": [], "source": "unknown"}


@dataclass(frozen=True)
class ModeSampling:
    """Per-mode sampling knobs. `None` for any field = let upstream default.

    `enable_thinking` and `thinking_budget` ride inside
    `chat_template_kwargs`; the rest go on the top-level chat/completions
    body. `sampling_to_request_fields()` does the split.
    """

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    max_completion_tokens: int | None = None
    seed: int | None = None
    # Qwen3-family chat-template kwargs (None = unset, let server default).
    enable_thinking: bool | None = None
    thinking_budget: str | int | None = None


# Completion ceilings are token/context safety policy. Optional thinking and
# sampling parameters are read from the declaration, never from these profiles.
_PLAN_DEFAULT = ModeSampling(max_completion_tokens=32768)
_ASK_DEFAULT = ModeSampling(max_completion_tokens=16384)
_AUTO_DEFAULT = ModeSampling(max_completion_tokens=16384)


@dataclass(frozen=True)
class Model:
    """One model's per-mode sampling profile + family identity.

    `name` is matched (case-insensitive) against upstream model ids — exact
    match wins, falling back to substring match against the model name.
    `family` describes the chat-template family ("qwen3", "gemma4", …) for
    callers that want to dispatch on family rather than model.

    `thinking_tags` are the open/close markers the model emits around its
    chain-of-thought when no reasoning parser is active server-side.
    Qwen3 uses `<think>...</think>`; Gemma 4 uses `<thought>...</thought>`.
    Used by the worker's `ThinkSplitter` to route inline reasoning into
    the thinking channel.

    `prefill_thinking_open_tag=True` means the request builder should
    end the prompt with the open tag (e.g. `<thought>\n`) using vLLM's
    `continue_final_message=true` so generation begins inside a thought
    block. This is the documented Gemma 4 thinking pattern.
    """

    name: str
    family: str = ""
    plan: ModeSampling = field(default_factory=lambda: _PLAN_DEFAULT)
    ask: ModeSampling = field(default_factory=lambda: _ASK_DEFAULT)
    auto: ModeSampling = field(default_factory=lambda: _AUTO_DEFAULT)
    thinking_tags: tuple[str, str] = ("<think>", "</think>")
    prefill_thinking_open_tag: bool = False
    repetition_monitor: bool = False
    # Worker post-stream `chars_per_token < floor` ratio guard. Default
    # ``None`` falls back to ``$SHAREDAGENTS_LOOP_CHARS_PER_TOKEN_MIN`` (1.5).
    # Quantization-fragile families (gemma-4 NVFP4) tighten this so a
    # round that emits text at less than the family's typical chars/
    # token rate trips the loop guard one round earlier.
    loop_chars_per_token_min: float | None = None
    # Per-model absolute compaction trigger (in prompt tokens). When
    # set, the bridge's `_maybe_compact` squeezes old tool results
    # before the next round if `prompt_tokens >=
    # compact_at_max_prompt_tokens`, regardless of the agent-level
    # `compact_at_frac × upstream_max_context` threshold. Use this to
    # keep quantization-fragile models inside their reliable operating
    # range (FP8 / NVFP4 attention layers degrade past their
    # documented stable region — Qwen3.6-FP8 collapse-mode is
    # documented past long context per QwenLM/Qwen3.5#115). Default
    # ``None`` defers entirely to the agent-level config.
    compact_at_max_prompt_tokens: int | None = None
    # Vocabulary id of the reasoning-close token. Set only where verified
    # against the served tokenizer: it turns on the worker's token-id
    # reasoning split (``sharedagents_gateway.reasoning_split``) on vLLM.
    reasoning_close_token_id: int | None = None

    def sampling_for(self, mode: PermissionMode) -> ModeSampling:
        """Return the `ModeSampling` for the given permission mode."""
        return getattr(self, mode.value)


# Known models. Append here when a new local model joins your fleet.
#
# Qwen3 family. `presence_penalty=0.0` per the Qwen3.6-27B model card's
# thinking-mode recommendation (the model card reserves
# presence_penalty=1.5 for non-thinking instruct mode — sharedagents uses
# thinking). Repetition protection is handled by DRY sampler +
# server-side `repetition_detection` + the runtime streaming loop
# detector, not by presence_penalty.
# Settings deliberately lenient compared to gemma's (3/20/4): qwen's
# observed failure mode is paragraph-level repeats during reasoning,
# not the token-collapse pattern that hits gemma-4-NVFP4. Larger
# `min_pattern_size` avoids false positives on legitimate tight
# structures (numbered-list scaffolding, repeated tool-result markers,
# code snippets with regular cadence). `min_count=8` requires the
# loop to be unambiguous before firing.

# Qwen3.8 — latest Qwen3 generation. Built on the Qwen3.5
# architectural foundation with gains across coding, professional work,
# research, and long-horizon agentic tasks. Native vision-language
# (images + video), flexible thinking control, `reasoning_effort` and
# `preserve_thinking` support. Sampling knobs carried over from Qwen3.6:
# same Qwen3 family, same thinking-mode behaviour.
QWEN3_8 = Model(
    name="qwen3.8",
    family="qwen3",
    repetition_monitor=True,
    loop_chars_per_token_min=0.8,
    compact_at_max_prompt_tokens=128_000,
    # ``</think>``; checked 2026-10-27 on the Flash-Next, 27B nvfp4 and 27B
    # int4 pools (real closes stream as ``\n`` 198, 248069, ``\n\n`` 271).
    reasoning_close_token_id=248069,
)

QWEN3_6 = Model(
    name="qwen3.6",
    family="qwen3",
    repetition_monitor=True,
    # Post-stream ratio guard floor. Qwen3.6's legitimate dense
    # reasoning (math, code, numbers, single-char tokens) sits at
    # ~1.0–1.2 chars/token. The original 1.5 floor produced false
    # positives on legitimate output cut by max_tokens (finish_reason=
    # "length"). Set at 0.8 so we only flag genuinely degenerate
    # token collapse (<0.5 chars/token) while letting legitimate
    # dense reasoning pass. The env-default fallback is 0.5.
    loop_chars_per_token_min=0.8,
    # Compact context at 128k prompt tokens. Qwen3.6-27B-FP8 advertises
    # 256k native context but the long-context decode cliff (single-
    # digit tok/s past 16k per the vLLM recipe) and the linear-attn
    # FP8 quantization fragility (QwenLM/Qwen3.5#115) make 128k a
    # comfortable ceiling — gives the model 4× the documented sweet
    # spot, well below the size where the SSM-block coherence
    # collapse has been reproduced. The bridge squeezes old tool
    # results before the next round once we cross this; the model
    # keeps the compaction summary in `sess.compaction_summary`.
    compact_at_max_prompt_tokens=128_000,
)

# Gemma 3/4 family. vLLM with `--reasoning-parser gemma4` splits the
# thought channel off into `delta.reasoning_content`; the worker's
# splitter handles any leaked markers as a safety net.
#
# Protocol (per ai.google.dev/gemma/docs/core/prompt-formatting-gemma4
# and vLLM's gemma4 recipe): Gemma 4 wraps thinking in an *asymmetric*
# channel marker — open `<|channel>thought\n`, close `<channel|>` (note
# the `|` swaps sides). Thinking is enabled via the chat-template kwarg
# `enable_thinking=true`, NOT by prefilling an open tag with
# `continue_final_message=true` (we used to do that for `<thought>`,
# which is a different format that does not match Gemma 4's tokenizer
# — vLLM's parser then misclassified the entire response as reasoning
# and the TUI showed an empty bubble because `reasoning_visibility=off`
# is the default).
#
# Sampling: Gemma 4 31B has a model-level repetition bug
# (google-deepmind/gemma#622, vllm-project/vllm#40080) that
# `repetition_penalty` does not fix. Community-validated mitigation for
# agentic / tool-calling use is lower temperature, mild
# frequency/presence penalty, and `min_p=0` (any positive `min_p`
# breaks Gemma 4 tool calling).
#
# AUTO's `max_completion_tokens=2048` is a deliberate tightening from the
# loop-prone 4096 we initially shipped; PLAN uses 3072 so it remains
# strictly larger without restoring that observed failure ceiling. NVFP4
# quantization leaves attention
# weights in BF16 but truncates MLP mantissa, which in practice means
# the model is more likely to slip into a token-collapse loop on long
# planning rounds (we observed 4096-token loops at ~40k prompt
# context on a local Gemma-4-31B NVFP4 MTP route). A 2048 ceiling lets the
# model land a normal answer in one round but caps the per-round blast
# radius when the streaming loop detectors haven't caught the
# collapse yet — pairs with the worker's `chars_per_token < 1.5`
# ratio guard that fires post-stream.
_GEMMA_PLAN = ModeSampling(
    max_completion_tokens=3072,
)

_GEMMA_ASK = replace(
    _GEMMA_PLAN,
    max_completion_tokens=1536,
)

_GEMMA_AUTO = replace(
    _GEMMA_PLAN,
    max_completion_tokens=2048,
)

GEMMA4 = Model(
    name="gemma",
    family="gemma4",
    plan=_GEMMA_PLAN,
    ask=_GEMMA_ASK,
    auto=_GEMMA_AUTO,
    thinking_tags=("<|channel>thought\n", "<channel|>"),
    prefill_thinking_open_tag=False,
    # NVFP4 + MTP slip into token-collapse looping more readily than
    # BF16/FP8 baselines do. Healthy per-round ratios on such routes measure
    # 1.7–3.5 chars/token; 2.0 is comfortably above the noise floor
    # while still being tight enough to catch a collapse round (live
    # measured collapse: 0.37 chars/token).
    loop_chars_per_token_min=2.0,
    repetition_monitor=True,
)

# Truly-generic model profile for upstream APIs (OpenAI, xAI/Grok, Anthropic,
# OpenRouter, etc.) that do NOT support vLLM-specific sampling knobs like
# `presence_penalty`, `min_p`, `top_k`, `repetition_penalty`, or
# `frequency_penalty`. Those fields are left at `None` so
# `sampling_to_request_fields` drops them — the upstream gets its own
# defaults. No `max_completion_tokens` either: a hosted model's output
# ceiling is data, not a number here -- see `resolve_max_tokens` (a
# hardcoded 16,384 once cut a DeepSeek `write` five rounds running).
#
# NOTE: `temperature` defaults to None — sharedagents omits temperature by
# default so upstreams apply their own defaults. The per-turn
# `backend.cfg.temperature` is the authoritative knob when explicitly configured.
_GENERIC_PLAN = ModeSampling()
_GENERIC_ASK = ModeSampling()
_GENERIC_AUTO = ModeSampling()

GENERIC = Model(
    name="generic",
    family="",
    plan=_GENERIC_PLAN,
    ask=_GENERIC_ASK,
    auto=_GENERIC_AUTO,
)

KNOWN_MODELS: tuple[Model, ...] = (QWEN3_8, QWEN3_6, GEMMA4)


def for_name(name: str) -> Model:
    """Resolve a Model by upstream model id. Falls back to `GENERIC`.

    Case-insensitive. Exact name match first; then substring (so
    `qwen3.6-27b-fp8` matches `qwen3.6`). Path-style ids (SGLang's
    `/home/.../Qwen3.6-27B`) are normalized to the basename for matching.
    """
    if not name:
        return GENERIC
    n = name.lower()
    base = n.rsplit("/", 1)[-1]
    for m in KNOWN_MODELS:
        if m.name.lower() == base or m.name.lower() == n:
            return m
    for m in KNOWN_MODELS:
        m_n = m.name.lower()
        if m_n and (m_n in n or m_n in base):
            return m
    return GENERIC


#: OpenCode's global output cap (``OUTPUT_TOKEN_MAX`` in opencode 1.18.33).
#: Raised globally by ``$SHAREDAGENTS_OUTPUT_TOKEN_MAX`` or per agent by
#: ``upstream_max_completion_tokens``.
OUTPUT_TOKEN_MAX_DEFAULT = 32000
#: The limit assumed when nothing reports one (no row, no provider /models,
#: no catalogue entry, e.g. before the first catalogue fetch). OpenCode would
#: send its 32k cap there because it ships the catalogue; without one, 32k
#: would 400 a model that stops at 16,384 (gpt-4o) and has always worked.
UNKNOWN_OUTPUT_LIMIT = 16384


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value or 0)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def output_token_cap(cfg: Any = None) -> int:
    """The global output cap: the agent's ``upstream_max_completion_tokens``,
    else ``$SHAREDAGENTS_OUTPUT_TOKEN_MAX``, else OpenCode's 32,000."""
    for raw in (
        getattr(cfg, "upstream_max_completion_tokens", None),
        os.environ.get("SHAREDAGENTS_OUTPUT_TOKEN_MAX"),
    ):
        value = _positive_int(raw)
        if value is not None:
            return value
    return OUTPUT_TOKEN_MAX_DEFAULT


def row_output_limit(
    row: Mapping[str, Any] | None,
    model_id: str,
) -> int | None:
    """The operator's output limit for the model -- its own row's
    ``maxTokens`` (OpenCode's ``limit.output``) -- or ``None``.

    Only a row whose ``id`` IS ``model_id`` counts. ``model_row_for_endpoint``
    falls back to the first row on the endpoint for a model the config does
    not list -- right for credentials and headers, wrong for a limit that
    belongs to that row's own model.
    """
    return _own_row_value(row, model_id, "maxTokens")


def _own_row_value(
    row: Mapping[str, Any] | None,
    model_id: str,
    key: str,
) -> int | None:
    if not isinstance(row, Mapping):
        return None
    if str(row.get("id") or "").strip() != str(model_id or "").strip():
        return None
    return _positive_int(row.get(key))


def resolve_max_tokens(
    *,
    row: Mapping[str, Any] | None,
    model_id: str,
    default_limit: int | None = None,
    cfg: Any = None,
) -> int:
    """``max_tokens`` for one request -- OpenCode's rule, ``min(limit, cap)``.

    The model's limit, first found wins:

    1. the operator's ``maxTokens`` on the model's own row;
    2. what the provider's ``GET /models`` reported at the last refresh in
       any process (the shared discovery cache) -- fresher than
    3. a provider-reported ``outputLimit`` saved on the row itself;
    4. the models.dev catalogue (:mod:`sharedagents_model.catalogue`);
    5. ``default_limit`` -- the per-mode budget of a tuned local family
       (qwen/gemma on the operator's vLLM, which publishes no limit). Never
       for a hosted row, even when its name matches the family
       (``qwen/qwen3.8-max`` on OpenRouter): same rule as the effort table;
    6. :data:`UNKNOWN_OUTPUT_LIMIT`.

    The request builder and the compaction guard both call this for the
    session's own route; a per-turn ``sampling`` override or a failover route
    is resolved on the wire only.
    """
    from sharedagents_model import catalogue

    limit = (
        row_output_limit(row, model_id)
        or catalogue.discovered_output_limit(row or {}, model_id)
        or _own_row_value(row, model_id, "outputLimit")
        or catalogue.output_limit(row or {}, model_id)
        or (None if row_is_hosted(row) else _positive_int(default_limit))
        or UNKNOWN_OUTPUT_LIMIT
    )
    return min(limit, output_token_cap(cfg))


def needs_loop_detection(name: str) -> bool:
    """Whether the RUNTIME streaming repetition/loop detector should run for
    this model.

    Only the local families that configure server-side ``repetition_detection``
    (qwen3.6 — both ``qwen3.6-27b-fp8`` and ``qwen3.6-27b-awq-bf16-int4`` —
    and gemma) have the documented token/paragraph-repeat failure mode that the
    detector exists to catch. Models that resolve to ``GENERIC`` (e.g.
    ``deepseek-v4-flash``) do NOT loop that way and were getting FALSE POSITIVES
    on legitimate repeated structure in long reasoning, so the monitor stays
    OFF for them.
    """
    return for_name(name).repetition_monitor


# Top-level chat/completions request fields we know about. Anything else
# in `ModeSampling` rides as a chat_template_kwarg.
_REQUEST_FIELD_NAMES: tuple[str, ...] = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
    "seed",
)


def sampling_to_request_fields(s: ModeSampling) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a `ModeSampling` into (top-level request fields, chat_template_kwargs).

    The top-level dict goes onto the chat/completions body; the
    chat_template_kwargs dict goes inside `chat_template_kwargs` so vLLM /
    SGLang forward it to the model's chat template (e.g. Qwen3's
    `enable_thinking` / `thinking_budget`). None values are dropped so we
    only send explicit knobs.
    """
    request_fields: dict[str, Any] = {}
    for k in _REQUEST_FIELD_NAMES:
        v = getattr(s, k, None)
        if v is not None:
            request_fields[k] = v
    if s.max_completion_tokens is not None:
        # OpenAI-compat name for the cap.
        request_fields["max_tokens"] = s.max_completion_tokens

    template_kwargs: dict[str, Any] = {}
    if s.enable_thinking is not None:
        template_kwargs["enable_thinking"] = s.enable_thinking
    if s.thinking_budget is not None:
        template_kwargs["thinking_budget"] = s.thinking_budget

    return request_fields, template_kwargs


def merge_overrides(base: ModeSampling, override: dict[str, Any] | None) -> ModeSampling:
    """Apply an override dict (from a per-turn `sampling` request body) on top of `base`.

    Returns a new `ModeSampling` with override fields replacing base fields
    where the override is non-None. Fields not present in override or set
    to None are inherited from base. Unknown keys in override are ignored
    so callers can't smuggle arbitrary fields through.
    """
    if not override:
        return base
    valid = {f.name for f in base.__dataclass_fields__.values()}
    kwargs = {f: getattr(base, f) for f in valid}
    for k, v in override.items():
        if k in valid and v is not None:
            kwargs[k] = v
    return ModeSampling(**kwargs)
