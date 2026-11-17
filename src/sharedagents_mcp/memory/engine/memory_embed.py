"""Embedding providers for the memory index.

`resolve_embedder()` is the single entry point: reads the active
agent's config + global config + env vars, and returns the configured
`Embedder` (or `None` to fall back to BM25-only). Everything else is
internal.

Provider auto-select chain mirrors OpenClaw's `doctor-memory-search`
priority list (high to low): `openai`, `gemini`, `voyage`, `mistral`,
then local providers (`ollama`, `lmstudio`). Currently `openai` and
`ollama` are wired; adding others is a matter of one new adapter
class each.

Each adapter implements:

    provider     : str    — stable id ("openai" / "ollama" / ...)
    model        : str    — embedding model name
    dims         : int    — vector dimension
    provider_key : str    — SHA-256 over (provider | base_url | org),
                            used to scope `embedding_cache` rows
    embed_query(text)        -> list[float]
    embed_batch(texts)       -> list[list[float]]

`embed_batch` returns vectors aligned with the input order. Failures
raise — `MemoryIndex` catches and falls back to a None vector for
the failed item, so partial-failure batches don't poison the index.

All vectors are L2-normalized before storage so cosine similarity is
equivalent to the inner product (consistent with what `chunks_vec`
and our fallback in-Python cosine assume).
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Any

# OpenAI defaults — match `doctor-memory-search-DA8TSpcF.js` / OpenClaw.
DEFAULT_OPENAI_MODEL: str = "text-embedding-3-small"
DEFAULT_OPENAI_DIMS: int = 1536

# Ollama default — `nomic-embed-text` is the most common local choice.
DEFAULT_OLLAMA_MODEL: str = "nomic-embed-text"
DEFAULT_OLLAMA_DIMS: int = 768


def _normalize(vec: list[float]) -> list[float]:
    """L2-normalize a vector. NaN/Inf inputs collapse to a zero vector
    so we never write garbage into the index.
    """
    cleaned = [x if math.isfinite(x) else 0.0 for x in vec]
    norm = math.sqrt(sum(x * x for x in cleaned))
    if norm < 1e-10:
        return cleaned
    return [x / norm for x in cleaned]


def _provider_key(provider: str, base_url: str = "", organization: str = "") -> str:
    """SHA-256(provider | base_url | organization). Same scope key
    OpenClaw stores in `embedding_cache.provider_key`.
    """
    payload = f"{provider}|{base_url}|{organization}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---- OpenAI ----


class OpenAIEmbedder:
    """OpenAI text-embedding-* adapter via the official SDK.

    The model and dim are bound at construction time so a bound
    `Embedder` instance is internally consistent. Switching models
    requires a new instance (and a re-index, gated on `scope_hash`).
    """

    provider = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_OPENAI_MODEL,
        dims: int = DEFAULT_OPENAI_DIMS,
        base_url: str = "",
        organization: str = "",
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI embedder")
        try:
            from openai import AsyncOpenAI  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError("openai SDK not installed; run `pip install openai`") from e
        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        if organization:
            kwargs["organization"] = organization
        self._client = AsyncOpenAI(**kwargs)
        self.model = model
        self.dims = dims
        self.provider_key = _provider_key(self.provider, base_url, organization)

    async def embed_query(self, text: str) -> list[float]:
        vecs = await self.embed_batch([text])
        return vecs[0] if vecs else []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # OpenAI's batch endpoint accepts up to 2048 items; we cap at
        # 512 because the embedding endpoint has a 300k token-per-batch
        # ceiling and 512 chunks of 400 tokens fits comfortably.
        out: list[list[float]] = []
        for i in range(0, len(texts), 512):
            batch = texts[i : i + 512]
            resp = await self._client.embeddings.create(
                model=self.model,
                input=batch,
            )
            for d in resp.data:
                out.append(_normalize(list(d.embedding)))
        return out


# ---- Ollama ----


class OllamaEmbedder:
    """Local Ollama embedder via the OpenAI-compatible /api/embeddings."""

    provider = "ollama"

    def __init__(
        self,
        *,
        base_url: str,
        model: str = DEFAULT_OLLAMA_MODEL,
        dims: int = DEFAULT_OLLAMA_DIMS,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dims = dims
        self.timeout = timeout
        self.provider_key = _provider_key(self.provider, self.base_url)

    async def embed_query(self, text: str) -> list[float]:
        vecs = await self.embed_batch([text])
        return vecs[0] if vecs else []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        import httpx

        out: list[list[float]] = []
        async with httpx.AsyncClient(timeout=self.timeout) as c:
            # Ollama's /api/embeddings accepts one text at a time — issue
            # a small concurrency batch instead of a bulk call.
            import asyncio

            sem = asyncio.Semaphore(4)

            async def _one(t: str) -> list[float]:
                async with sem:
                    r = await c.post(
                        f"{self.base_url}/api/embeddings",
                        json={"model": self.model, "prompt": t},
                    )
                    r.raise_for_status()
                    data = r.json()
                    vec = data.get("embedding") or []
                    return _normalize([float(x) for x in vec])

            results = await asyncio.gather(*[_one(t) for t in texts])
            out.extend(results)
        return out


# ---- failover (primary + backup) ----


class FailoverEmbedder:
    """Wraps a PRIMARY + BACKUP embedder; tries primary then backup on every
    call. Exposes the PRIMARY's model/dims/provider_key so the index scope is
    STABLE regardless of which endpoint served — which is only safe because the
    backup serves the EXACT SAME model (backup
    models must be identical for downstream compatibility). A backup whose
    declared dims differ is dropped (a mismatched vector space would corrupt the
    index)."""

    provider = "failover"

    def __init__(self, primary: Any, backup: Any | None, log: Any = None) -> None:
        self._primary = primary
        self._backup = backup
        self.model = primary.model
        self.dims = primary.dims
        self.provider_key = primary.provider_key  # stable index scope
        if backup is not None and getattr(backup, "dims", primary.dims) != primary.dims:
            # Same model ⇒ same dims; a mismatch means an inconsistent backup —
            # refuse it rather than risk poisoning the index.
            if log is not None:
                log.warning(
                    "embedding backup dims %s != primary %s — dropping backup (backup must serve the same model)",
                    getattr(backup, "dims", "?"),
                    primary.dims,
                )
            self._backup = None

    async def embed_query(self, text: str) -> list[float]:
        try:
            return await self._primary.embed_query(text)
        except Exception:  # noqa: BLE001
            if self._backup is None:
                raise
            return await self._backup.embed_query(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            return await self._primary.embed_batch(texts)
        except Exception:  # noqa: BLE001
            if self._backup is None:
                raise
            return await self._backup.embed_batch(texts)


def _embedder_from_endpoint(ep: Any, default_dims: int) -> Any | None:
    """Build an OpenAI-compatible embedder from a services ``ServiceEndpoint``
    (a local vLLM instance and OpenRouter both speak the OpenAI embeddings API). ``None`` if
    the endpoint isn't configured."""
    if not ep.configured:
        return None
    dims = int(ep.opt("dims") or default_dims)
    # A local OpenAI-compatible server usually needs no key; pass a sentinel so
    # the SDK doesn't reject an empty string.
    return OpenAIEmbedder(
        api_key=ep.api_key or "sk-no-auth",
        model=ep.model or DEFAULT_OPENAI_MODEL,
        dims=dims,
        base_url=ep.base_url,
        organization=str(ep.opt("organization") or ""),
    )


def _probe_endpoint_dims(ep: Any, fallback: int) -> int:
    """Return the ACTUAL embedding width the endpoint emits.

    Root-cause guard for a silent, total failure mode: when the config omits
    ``dims`` we must NOT assume the OpenAI default (1536). If the endpoint emits
    a different width (e.g. ``qwen3-embedding-8b`` → 4096) the index vector
    table is created at the wrong width and EVERY embedding is dropped by the
    length-guard in :meth:`MemoryIndex._embed_chunks_blocking` — so ``chunks_vec``
    stays empty and semantic search silently degrades to BM25-only until the
    embedding-recovery backfill sees the missing vectors. One tiny probe
    embedding removes the guesswork up front. Falls back to ``fallback`` only if
    the probe itself fails, so a transient endpoint blip never breaks embedder
    construction. Runs whether or not an event loop is already active.
    """
    import asyncio

    async def _probe() -> int:
        emb = _embedder_from_endpoint(ep, fallback)
        if emb is None:
            return fallback
        vec = await emb.embed_query("dimension probe")
        return len(vec) if vec else fallback

    try:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_probe())
        # A loop is already running in this thread — run the probe to completion
        # on a dedicated worker thread with its own loop.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(lambda: asyncio.run(_probe())).result()
    except Exception:  # noqa: BLE001
        return fallback


# ---- factory ----


def _read_codex_api_key() -> str:
    """Pull `OPENAI_API_KEY` from `~/.codex/auth.json`.

    OpenClaw and the OpenAI Codex CLI store the operator's API key in
    `~/.codex/auth.json` when `auth_mode == "apikey"`. Reading it here
    means sharedagents's memory embedder Just Works on machines where the
    user has already authenticated `codex` / OpenClaw without forcing
    them to also export `OPENAI_API_KEY` into their shell. Returns "" if
    the file is missing, malformed, or in OAuth (`chatgpt`) mode.
    """
    import json as _json

    try:
        path = Path.home() / ".codex" / "auth.json"
        if not path.is_file():
            return ""
        data = _json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if not isinstance(data, dict):
        return ""
    if data.get("auth_mode") != "apikey":
        # OAuth tokens live under `tokens.access_token` and rotate
        # frequently; they don't authenticate `api.openai.com/v1/embeddings`,
        # only Codex's bridge. Skip silently.
        return ""
    key = data.get("OPENAI_API_KEY")
    return key if isinstance(key, str) else ""


# Module-level singleton cache. Building an `OpenAIEmbedder` constructs
# an `AsyncOpenAI` httpx client; tearing it down per-search means every
# memory_search pays a cold-TLS handshake (~500-1000ms)
# AND a new asyncio loop creation (the runner uses
# `_run_blocking_coroutine` from inside the gateway's event loop, which
# spawns a fresh thread + loop per call). Reusing the embedder lets the
# httpx client keep its connection pool warm.
#
# Keyed by the (provider, base_url, organization, dims, model) tuple so
# config changes invalidate the cache. The `None` sentinel is also
# cached so repeated "no embedder configured" calls don't re-walk env +
# config files on each search.
_EMBEDDER_CACHE: dict[tuple, Any | None] = {}


def _config_mtime() -> float:
    """`mtime_ns` of the global config file, or 0 if it doesn't exist.

    Folded into the embedder cache key so an operator who edits
    `~/.sharedagents/config.json` (e.g. to switch embedding model) gets a
    fresh embedder on the next lookup — without `clear_embedder_cache`.
    Cheap (one stat call per lookup); the cache key tuple stays small.
    """
    try:
        from sharedagents_core.kernel.config import global_config_path

        p = global_config_path()
        st = p.stat()
    except (OSError, FileNotFoundError):
        return 0.0
    return float(st.st_mtime_ns)


def _embedder_cache_key(
    *,
    provider: str,
    model: str,
    dims: int,
    base_url: str,
    organization: str,
) -> tuple:
    # Include the config-file mtime so post-edit lookups bypass the
    # cache automatically.
    return (provider, model, dims, base_url, organization, _config_mtime())


def resolve_embedder() -> Any | None:
    """Build the configured embedder, or None if disabled / unavailable.

    Cached at module scope — repeated calls return the same instance so
    the underlying httpx connection pool stays warm. Use
    `clear_embedder_cache()` to invalidate (e.g. after a config edit).

    Resolution order (inference-service endpoints resolve
    exclusively through the unified config, never a standalone env var or a
    hardcoded host/IP in source):

    1. The unified `embedding` services block via `sharedagents_core.kernel.services`
       (PRIMARY + BACKUP), then the legacy `memory_embedding` block. The
       `provider` (openai / ollama / none / auto) is read from that config, not
       from a bespoke environment variable.
    2. Global config (`~/.sharedagents/config.json`):
       - `memory_embedding.provider`
       - `memory_embedding.model`
       - `memory_embedding.dims`
       - `memory_embedding.base_url`
       - `memory_embedding.organization`
       - `memory_embedding.api_key` (or env `OPENAI_API_KEY`)
    3. `~/.codex/auth.json` — the same file OpenClaw / `codex` write
       when the user runs `codex login` with an API key. Lets
       embedding Just Work without an explicit env var if the user
       already authenticated those tools.
    4. Use Ollama only when its endpoint is explicitly present in the global
       config. Self-hosted endpoints are never inferred from environment
       variables or source-code host literals.

    Returns None silently when nothing is configured. Memory tools fall
    back to BM25-only — search still works, just less semantically.
    """
    # Provider is sourced from the unified config only: no
    # standalone env override. A configured ``memory_embedding.provider`` of
    # ``none`` disables embedding; anything else (or unset) lets the unified
    # ``embedding`` services block resolve below.
    provider = ""
    try:
        from sharedagents_core.kernel.config import load_global_config as _lgc

        provider = str(((_lgc() or {}).get("memory_embedding") or {}).get("provider") or "").strip().lower()
    except Exception:  # noqa: BLE001
        provider = ""

    # PRIMARY + BACKUP via the unified `embedding` services block (operator
    # directive 2026-06-04). Takes precedence over the legacy `memory_embedding`
    # block when present. Both endpoints are OpenAI-compatible; the backup MUST
    # serve the same model (same vector space) — `FailoverEmbedder` enforces it.
    if provider not in ("none",):
        try:
            from sharedagents_core.kernel.services import resolve_service

            svc = resolve_service("embedding")
        except Exception:  # noqa: BLE001
            svc = None
        if svc is not None and svc.primary.configured:
            import logging as _logging

            _log = _logging.getLogger("sharedagents.memory.embed")
            mismatch = svc.model_mismatch()
            if mismatch:
                _log.warning("%s", mismatch)
            # Never GUESS the embedding width. If the operator
            # declared dims, honour it; otherwise probe the live endpoint rather
            # than defaulting to 1536 — a wrong width silently empties chunks_vec
            # (see _probe_endpoint_dims). One probe per config-mtime (cached).
            _declared_dims = svc.primary.opt("dims")
            if _declared_dims:
                default_dims = int(_declared_dims)
            else:
                default_dims = _probe_endpoint_dims(svc.primary, DEFAULT_OPENAI_DIMS)
                if default_dims != DEFAULT_OPENAI_DIMS:
                    _log.warning(
                        "memory embedding: config omits 'dims'; probed endpoint "
                        "%s emits %d-d vectors (default assumption was %d). Using "
                        "the probed width so the vector index matches — set "
                        "embedding.dims=%d in config to skip the probe.",
                        svc.primary.base_url,
                        default_dims,
                        DEFAULT_OPENAI_DIMS,
                        default_dims,
                    )
            key = (
                "embedding-failover",
                svc.primary.base_url,
                svc.backup.base_url,
                svc.primary.model,
                default_dims,
                _config_mtime(),
            )
            if key in _EMBEDDER_CACHE:
                return _EMBEDDER_CACHE[key]
            try:
                prim = _embedder_from_endpoint(svc.primary, default_dims)
                back = _embedder_from_endpoint(svc.backup, default_dims)
                inst = FailoverEmbedder(prim, back, log=_log) if prim else None
                _EMBEDDER_CACHE[key] = inst
                if inst is not None:
                    return inst
            except (RuntimeError, ValueError) as e:
                _log.warning("embedding failover build failed: %s", e)

    cfg: dict[str, Any] = {}
    try:
        from sharedagents_core.kernel.config import load_global_config

        cfg = (load_global_config() or {}).get("memory_embedding") or {}
    except Exception:  # noqa: BLE001
        cfg = {}

    if not provider:
        provider = str(cfg.get("provider") or "auto").strip().lower()
    if provider == "none":
        return None

    if provider in ("auto", "openai", ""):
        api_key = os.environ.get("OPENAI_API_KEY") or str(cfg.get("api_key") or "") or _read_codex_api_key()
        if api_key or provider == "openai":
            model = str(cfg.get("model") or DEFAULT_OPENAI_MODEL)
            dims = int(cfg.get("dims") or DEFAULT_OPENAI_DIMS)
            base_url = str(cfg.get("base_url") or "")
            organization = str(cfg.get("organization") or "")
            key = _embedder_cache_key(
                provider="openai",
                model=model,
                dims=dims,
                base_url=base_url,
                organization=organization,
            )
            if key in _EMBEDDER_CACHE:
                return _EMBEDDER_CACHE[key]
            try:
                inst = OpenAIEmbedder(
                    api_key=api_key,
                    model=model,
                    dims=dims,
                    base_url=base_url,
                    organization=organization,
                )
                _EMBEDDER_CACHE[key] = inst
                return inst
            except (RuntimeError, ValueError):
                if provider == "openai":
                    return None
                # auto: fall through to Ollama
    if provider in ("auto", "ollama"):
        base_url = str(cfg.get("base_url") or "").strip()
        if not base_url:
            return None
        model = str(cfg.get("model") or DEFAULT_OLLAMA_MODEL)
        dims = int(cfg.get("dims") or DEFAULT_OLLAMA_DIMS)
        key = _embedder_cache_key(
            provider="ollama",
            model=model,
            dims=dims,
            base_url=base_url,
            organization="",
        )
        if key in _EMBEDDER_CACHE:
            return _EMBEDDER_CACHE[key]
        # auto-detect: ping /api/tags; if it answers, Ollama is up.
        if provider == "auto":
            try:
                import httpx

                with httpx.Client(timeout=2.0) as c:
                    r = c.get(f"{base_url.rstrip('/')}/api/tags")
                    if r.status_code != 200:
                        return None
            except Exception:  # noqa: BLE001
                return None
        try:
            inst = OllamaEmbedder(
                base_url=base_url,
                model=model,
                dims=dims,
            )
            _EMBEDDER_CACHE[key] = inst
            return inst
        except Exception:  # noqa: BLE001
            return None
    return None


def clear_embedder_cache() -> None:
    """Drop the cached embedder instances. Call after a config edit
    or when an httpx client has gotten itself into a stuck state."""
    _EMBEDDER_CACHE.clear()


__all__ = [
    "DEFAULT_OLLAMA_DIMS",
    "DEFAULT_OLLAMA_MODEL",
    "DEFAULT_OPENAI_DIMS",
    "DEFAULT_OPENAI_MODEL",
    "OllamaEmbedder",
    "OpenAIEmbedder",
    "FailoverEmbedder",
    "resolve_embedder",
]
