"""Per-backend differences between OpenAI-compatible LLM servers.

sharedagents targets local inference. Today that means **vLLM** and **SGLang**;
both speak the OpenAI streaming API but their wire output differs in small
but real ways:

- `/v1/models[].owned_by` carries the backend kind ("vllm" / "sglang") —
  used for auto-detection in the wizard.
- The reasoning-token delta key differs: vLLM (Qwen3 build) emits
  `delta.reasoning`, SGLang emits `delta.reasoning_content`. Both servers
  also fall back to inline `<think>...</think>` in `delta.content` if no
  reasoning parser is configured.
- The model id format differs: vLLM exposes a clean name like
  `qwen3.6-27b-fp8`; SGLang exposes the full filesystem path
  (`/srv/models/Qwen3.6-27B`). Each accepts its own id back.
- SGLang's per-chunk delta keeps every key always present (often `null`),
  vLLM's delta only carries the keys that actually changed. The streaming
  router treats `None`/empty equally, so this is a non-issue, but it's
  worth knowing if you `repr()` chunks while debugging.

`Provider` records each backend in one place so callers (worker, runtime,
wizard, frontends) don't sprinkle `if backend == "vllm"` checks. Adding a
new server is a one-line append to `PROVIDERS`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

# Default alias set probed across every backend. The streaming router
# tries each in order; per-provider lists below just reorder for the
# typical-most-likely-first case so we exit the loop one alias earlier.
_DEFAULT_REASONING_KEYS: tuple[str, ...] = (
    "reasoning_content",
    "reasoning",
    "thinking_content",
    "thinking",
)


_DEFAULT_THINKING_ONLY_NUDGE = (
    "[auto-continue] Your previous reply contained reasoning but no "
    "visible answer. Write the final answer to the user now — concise "
    "and direct. Do not start a new chain of thought."
)


@dataclass(frozen=True)
class Provider:
    """One OpenAI-compatible backend kind.

    Provider exists so per-backend differences (delta wire shape,
    reasoning routing, EOS-recovery behaviour) live in ONE place
    instead of being sprinkled through worker / runtime / wizard. Code
    that needs backend-specific behaviour should look it up via
    `by_name(session.backend)` and read attributes/methods here.
    """

    name: str
    """Stable id persisted in `AgentConfig.backend`. Empty = unknown."""

    label: str
    """Human-readable name shown in the wizard / status bar."""

    reasoning_keys: tuple[str, ...] = _DEFAULT_REASONING_KEYS
    """Delta keys that may carry reasoning tokens, in priority order."""

    typical_endpoint_hint: str = ""
    """One-line hint shown next to the wizard's endpoint prompt."""

    has_reasoning_channel: bool = False
    """True if the backend emits a dedicated `reasoning_content` channel
    separate from `content`. SGLang with `--reasoning-parser qwen3` does;
    vLLM without a parser inlines `<think>...</think>` into `content`
    and our worker's splitter handles the routing. Drives whether the
    runtime should consider a "thinking-only stop" (reasoning emitted
    but content empty) a recoverable failure mode."""

    thinking_only_recovery_nudge: str = _DEFAULT_THINKING_ONLY_NUDGE
    """Text to append to the message list when a turn ends with
    reasoning emitted but no content (qwen3 premature-EOS bug or
    SGLang long-CoT budget exhaustion). Empty string disables the
    recovery for this backend; useful if a future backend never has
    this failure mode."""

    def extract_reasoning(self, delta: dict) -> str:
        """Return the first non-empty reasoning value found in `delta`.

        Per-chunk routing: callers pass `chunk["choices"][0]["delta"]`,
        receive the reasoning text (or empty string). Empty / None values
        in any alias are skipped so we never emit empty thinking events.
        """
        for k in self.reasoning_keys:
            v = delta.get(k)
            if v:
                return v
        return ""


# vLLM-specific tuning — the Qwen3.6 build emits `delta.reasoning` (not
# `reasoning_content`), so we probe that first to short-circuit the loop.
# Empirical: gpu-a vLLM ships qwen3.6-27b-fp8 with NO reasoning parser
# wired up, so reasoning + answer both flow through `delta.content` and
# the worker's `ThinkSplitter` does the routing. Marking
# has_reasoning_channel=False reflects the typical config; sites that
# enable a reasoning parser are still handled (the worker probes every
# alias in `reasoning_keys`).
VLLM = Provider(
    name="vllm",
    label="vLLM",
    reasoning_keys=("reasoning", "reasoning_content", "thinking_content", "thinking"),
    typical_endpoint_hint="e.g. http://HOST:8000/v1",
    has_reasoning_channel=False,
)

# SGLang with `--reasoning-parser qwen3` always splits CoT off into
# `reasoning_content`. Empirical (gpu-b sglang, qwen3.6-27b): plain "hi"
# generates 260 reasoning tokens before any content arrives. The
# qwen3 premature-EOS bug fires often during this long CoT — model
# emits EOS mid-think with `finish_reason=stop` and no content. The
# default thinking-only-recovery nudge is a good fit; we keep it as
# the inherited default rather than override.
SGLANG = Provider(
    name="sglang",
    label="SGLang",
    reasoning_keys=("reasoning_content", "reasoning", "thinking_content", "thinking"),
    typical_endpoint_hint="e.g. http://HOST:30000/v1 (sometimes 8001)",
    has_reasoning_channel=True,
)

# Used as a safe fallback when /v1/models doesn't carry a recognized
# `owned_by` (third-party servers, llama.cpp's OpenAI-compat shim, etc.).
UNKNOWN = Provider(
    name="unknown",
    label="OpenAI-compatible",
    has_reasoning_channel=False,
)

PROVIDERS: tuple[Provider, ...] = (VLLM, SGLANG, UNKNOWN)
PROVIDERS_BY_NAME: dict[str, Provider] = {p.name: p for p in PROVIDERS}


def detect_from_models(models_data: Iterable[dict]) -> Provider:
    """Identify provider from a `/v1/models` response's `data` array.

    Inspects the first entry's `owned_by` field. Falls back to UNKNOWN
    when nothing matches — `owned_by` is a short, well-defined string
    that both target backends set explicitly.
    """
    for entry in models_data:
        owned = str(entry.get("owned_by") or "").lower()
        if "vllm" in owned:
            return VLLM
        if "sglang" in owned:
            return SGLANG
    return UNKNOWN


def detect_from_endpoint_sync(
    endpoint: str, *, timeout: float = 5.0, headers: dict | None = None
) -> tuple[Provider, list[dict]]:
    """Probe `<endpoint>/models` synchronously; return (Provider, models_data).

    Sync variant for callers in CLI / wizard contexts that aren't running
    an event loop. Uses `httpx.Client` for parity with the rest of the
    codebase. Raises `httpx.HTTPError` on connection / 4xx / 5xx.
    """
    import httpx

    with httpx.Client(timeout=timeout, headers=headers or {}) as c:
        r = c.get(f"{endpoint.rstrip('/')}/models")
        r.raise_for_status()
        data = r.json().get("data") or []
    return detect_from_models(data), data


async def detect_from_endpoint(
    endpoint: str, *, timeout: float = 5.0, headers: dict | None = None
) -> tuple[Provider, list[dict]]:
    """Async variant of `detect_from_endpoint_sync`."""
    import httpx

    async with httpx.AsyncClient(timeout=timeout, headers=headers or {}) as c:
        r = await c.get(f"{endpoint.rstrip('/')}/models")
        r.raise_for_status()
        data = r.json().get("data") or []
    return detect_from_models(data), data


def by_name(name: str) -> Provider:
    """Look up a provider by its `name` field; falls back to UNKNOWN."""
    return PROVIDERS_BY_NAME.get(name or "", UNKNOWN)


def first_model_id(models_data: Sequence[dict]) -> str:
    """Pick a default model id from a `/v1/models` `data` array.

    Returns the first entry's `id` verbatim — both vLLM (`qwen3.6-27b-fp8`)
    and SGLang (full filesystem path) accept their own ids as the chat
    request's `model` field.
    """
    for entry in models_data:
        mid = entry.get("id")
        if mid:
            return str(mid)
    return ""


def probe_max_context(
    endpoint: str,
    models_data: Sequence[dict] | None = None,
    *,
    timeout: float = 5.0,
    headers: dict | None = None,
) -> int | None:
    """Discover the model's max context length from the upstream.

    Probe order:
        1. `/v1/models[0].max_model_len` — vLLM exposes this directly on
           every model entry. Cheap if `models_data` was already fetched.
        2. `<endpoint>/get_model_info` — SGLang's per-engine endpoint;
           field name is `max_model_len`. Lives one path level above
           `/v1`, so we strip `/v1` before probing.

    Returns the integer when found, `None` when neither source has a
    usable value (caller should fall back to a manual prompt).
    """
    import httpx

    # 1) vLLM-style: `/v1/models` already carries it.
    if models_data:
        for entry in models_data:
            v = entry.get("max_model_len")
            if isinstance(v, int) and v > 0:
                return v
            # Some forks nest it under root or use a different name.
            for alt in ("context_length", "n_ctx", "max_context_length"):
                v2 = entry.get(alt)
                if isinstance(v2, int) and v2 > 0:
                    return v2

    # 2) SGLang-style: `/get_model_info` (sibling of `/v1`).
    base = endpoint.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    try:
        with httpx.Client(timeout=timeout, headers=headers or {}) as c:
            r = c.get(f"{base}/get_model_info")
            r.raise_for_status()
            info = r.json() or {}
    except (httpx.HTTPError, ValueError):
        return None
    for k in ("max_model_len", "context_length", "max_total_tokens", "max_context_length"):
        v = info.get(k)
        if isinstance(v, int) and v > 0:
            return v
    return None
