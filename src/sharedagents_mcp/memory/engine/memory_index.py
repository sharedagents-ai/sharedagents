"""SQLite-backed semantic memory index.

Exposes a `MemoryIndex` class that owns one `memory.db` per agent home
and indexes `MEMORY.md` + `memory/*.md` for hybrid (BM25 + vector)
search.

Schema follows OpenClaw's bundled `memory-host-engine-storage` so the
file format is portable across implementations:

    files          — incremental tracking (path, hash, mtime, size)
    chunks         — content + metadata + JSON-serialized embedding
    chunks_fts     — FTS5 lexical index (BM25)
    chunks_vec     — sqlite-vec vec0 virtual table for ANN
    embedding_cache — (provider, model, provider_key, hash) → vector
    meta           — `memory_index_meta_v1` JSON with scopeHash

`scope_hash` is a SHA-256 over the active config (provider, model,
chunk_tokens, chunk_overlap, vector_dims, sources, fts_tokenizer); a
mismatch on boot drops + recreates the index, so swapping embedding
models doesn't poison the cache.
"""

from __future__ import annotations

import asyncio as _asyncio
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import threading as _threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

_log = logging.getLogger("sharedagents.memory.index")

# Leaf recall primitives hoisted to the kernel: the FTS/BM25
# helpers and the snippet cap are shared with session indexing and carry no
# memory dependency, so they live below both. Imported downward here and
# re-exported via __all__ so existing callers (session search) are
# unchanged.
from sharedagents_core.kernel.recall import (  # noqa: E402
    DEFAULT_SNIPPET_CHARS,
    bm25_rank_to_score,
    build_fts_query,
    fts_terms,
)

# ---- constants — match OpenClaw 2026.4.26 (memory-search-BkPKyCZS.js) ----

# Chunking
DEFAULT_CHUNK_TOKENS: int = 400
DEFAULT_CHUNK_OVERLAP: int = 80
# 1 token ≈ 4 chars (OpenClaw uses the same ratio).
CHARS_PER_TOKEN: int = 4

# Hybrid search
DEFAULT_MAX_RESULTS: int = 6
DEFAULT_MIN_SCORE: float = 0.35
DEFAULT_HYBRID_VECTOR_WEIGHT: float = 0.7
DEFAULT_HYBRID_TEXT_WEIGHT: float = 0.3
DEFAULT_HYBRID_CANDIDATE_MULTIPLIER: int = 4
DEFAULT_MMR_LAMBDA: float = 0.7
# Lexical-confidence floor. SQLite BM25 is an *IDF-weighted*
# score: on a tiny corpus (a couple of daily chunks) the IDF of every term
# collapses toward 0, so even a verbatim exact-phrase hit scores ~0.000 and is
# wrongly dropped by the absolute ``min_score`` floor — real recall returns "no
# matches" for content that is provably on disk. Fix: a chunk that lexically
# contains (almost) the whole query is a *confident* match regardless of corpus
# size, so we floor its fused score to ``LEXICAL_CONFIDENT_FLOOR * coverage``
# (coverage = fraction of distinct query terms present). This is corpus-
# independent and precision-safe: it is a MAX, so it never lowers a strong
# large-corpus score, and it only rescues high-coverage lexical matches while
# an explicit high ``min_score`` still wins (the floor is capped below it).
LEXICAL_CONFIDENT_FLOOR: float = 0.6
# Only near-complete lexical matches are rescued; a chunk containing well under
# this fraction of the query's terms stays subject to the ordinary floor.
LEXICAL_COVERAGE_MIN: float = 0.6
DEFAULT_TEMPORAL_DECAY_HALF_LIFE_DAYS: int = 30

# ``DEFAULT_SNIPPET_CHARS`` moved to ``sharedagents_core.kernel.recall`` (imported
# above, re-exported via __all__).

# FTS tokenizer (OpenClaw default — NOT unicode64).
DEFAULT_FTS_TOKENIZER: str = "unicode61"

# Embedding defaults — matches OpenAI provider default.
DEFAULT_EMBEDDING_MODEL: str = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMS: int = 1536

# What `meta.memory_index_meta_v1` stores. Bumped only when the schema
# changes shape in a way that needs migration; scopeHash handles the
# common case (same schema, different config).
META_VERSION: int = 1


# ---- types ----


@dataclass(frozen=True)
class Chunk:
    """One chunk produced by `chunk_markdown`. `id` is SHA-256 of
    (path || start_line || end_line || text); `hash` is SHA-256(text)
    alone — used to look up `embedding_cache` rows so identical chunk
    bodies across files reuse the same embedding.
    """

    id: str
    path: str
    source: str
    start_line: int
    end_line: int
    hash: str
    text: str


@dataclass(frozen=True)
class MemoryHit:
    """One hybrid-search result. `score` is the post-fusion combined
    score; `vector_score` and `text_score` are the per-component scores
    so the model can debug why a chunk ranked where it did. `snippet`
    is the FTS5 snippet when available, else the chunk's full text.
    """

    id: str
    path: str
    source: str
    start_line: int
    end_line: int
    score: float
    vector_score: float
    text_score: float
    snippet: str


class Embedder(Protocol):
    """Embedding provider contract. ``None`` is a legal value:
    callers degrade to FTS-only when no embedder is configured.
    """

    provider: str  # e.g. "openai", "ollama"
    model: str
    dims: int
    provider_key: str

    async def embed_query(self, text: str) -> list[float]: ...
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


# ---- chunk algorithm — matches OpenClaw memory-host-markdown ----


def chunk_markdown(
    text: str,
    *,
    file_path: str,
    source: str = "memory",
    target_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Line-aware sliding-window chunker.

    Each chunk is ≤ ~target_tokens with ≤ ~overlap_tokens of overlap
    carried into the next chunk. Line numbers are preserved for
    attribution. Empty inputs yield zero chunks.
    """
    target_chars = max(1, target_tokens) * CHARS_PER_TOKEN
    overlap_chars = max(0, overlap_tokens) * CHARS_PER_TOKEN
    text = text or ""
    if not text.strip():
        return []
    lines = text.split("\n")
    chunks: list[Chunk] = []
    cur_lines: list[str] = []
    cur_chars = 0
    start_line = 1

    def _emit(end_line: int) -> None:
        if not cur_lines:
            return
        body = "\n".join(cur_lines)
        chunks.append(_make_chunk(file_path, source, body, start_line, end_line))

    for i, line in enumerate(lines, 1):
        cur_lines.append(line)
        cur_chars += len(line) + 1  # +1 for the implicit newline
        if cur_chars < target_chars:
            continue
        _emit(i)
        # There is no next window after the final input line. Keeping an
        # overlap seed here would make the post-loop flush emit the same
        # chunk a second time (notably for newline-terminated files whose
        # final split item is empty).
        if i == len(lines):
            cur_lines = []
            cur_chars = 0
            start_line = i + 1
            continue
        # Carry an overlap tail forward so semantic continuity holds
        # across cuts. Walk from the END until we've accumulated
        # `overlap_chars`, then keep that suffix as the seed of the next
        # chunk. Minimum of 1 line so we never lose a line on the seam.
        if overlap_chars <= 0:
            cur_lines = []
            cur_chars = 0
            start_line = i + 1
            continue
        carry: list[str] = []
        carry_chars = 0
        for ln in reversed(cur_lines):
            if carry_chars >= overlap_chars and carry:
                break
            carry.insert(0, ln)
            carry_chars += len(ln) + 1
        cur_lines = carry
        cur_chars = carry_chars
        start_line = i - len(carry) + 1

    if cur_lines:
        _emit(len(lines))
    return chunks


def _make_chunk(
    file_path: str,
    source: str,
    body: str,
    start: int,
    end: int,
) -> Chunk:
    text_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    chunk_id = hashlib.sha256(f"{file_path}|{start}|{end}|{body}".encode()).hexdigest()
    return Chunk(
        id=chunk_id,
        path=file_path,
        source=source,
        start_line=start,
        end_line=end,
        hash=text_hash,
        text=body,
    )


# ---- BM25 normalization & FTS query construction ----

# Ports `manager-ICKYl3BU.js::buildFtsQuery` and `bm25RankToScore` from
# OpenClaw verbatim — both are user-facing scoring primitives that the
# rest of the fusion code depends on, so behaviour parity matters.

_FTS_TOKEN_RE = re.compile(r"[^\W_]+|[\d_]+", re.UNICODE)


# ``build_fts_query`` and ``bm25_rank_to_score`` moved to
# ``sharedagents_core.kernel.recall`` (imported at the top of this module, re-exported
# via __all__) — leaf recall math shared with session indexing.


# ---- scope_hash: invalidates index when config changes ----


def compute_scope_hash(
    *,
    provider: str,
    model: str,
    vector_dims: int,
    chunk_tokens: int,
    chunk_overlap: int,
    sources: list[str],
    fts_tokenizer: str,
) -> str:
    """SHA-256 over the canonical config. Different scope → drop + rebuild."""
    payload = json.dumps(
        {
            "provider": provider,
            "model": model,
            "vector_dims": int(vector_dims),
            "chunk_tokens": int(chunk_tokens),
            "chunk_overlap": int(chunk_overlap),
            "sources": sorted(sources or ["memory"]),
            "fts_tokenizer": fts_tokenizer,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# Lazy-initialized worker thread + dedicated event loop for the
# "called from inside a running loop" path. Reuses ONE loop across
# every call, so the OpenAI httpx client (which lives on that loop)
# keeps its connection pool warm. Without this, every search call
# spun up a fresh thread → fresh loop → fresh httpx client → cold
# TLS handshake to OpenAI (~500ms each).
_BG_LOOP_LOCK = _threading.Lock()
_BG_LOOP: Any | None = None
_BG_LOOP_THREAD: Any | None = None


def _ensure_bg_loop() -> Any:
    global _BG_LOOP, _BG_LOOP_THREAD
    with _BG_LOOP_LOCK:
        if _BG_LOOP is not None and _BG_LOOP_THREAD is not None and _BG_LOOP_THREAD.is_alive():
            return _BG_LOOP
        loop = _asyncio.new_event_loop()
        t = _threading.Thread(
            target=loop.run_forever,
            daemon=True,
            name="sharedagents-embed-loop",
        )
        t.start()
        _BG_LOOP = loop
        _BG_LOOP_THREAD = t
        return loop


def _run_blocking_coroutine(coro):
    """Execute an async coroutine from a sync caller, regardless of
    whether the calling thread already has a running event loop.

    Two paths:

    - **No running loop** — the common case for the tool-runner
      subprocess and the CLI commands. `asyncio.run` makes a fresh
      loop, runs to completion, closes it.
    - **Running loop** — the calling thread is itself inside an
      `asyncio.run` (e.g. the gateway's agent_runtime loop). We can't
      reuse that loop (different invocation context). Instead of
      spinning up a new thread + loop EVERY call (which trashes the
      OpenAI httpx client and forces a cold TLS handshake each
      search), use a long-lived background loop on a daemon thread
      and `run_coroutine_threadsafe` onto it.
    """
    try:
        _asyncio.get_running_loop()
        # Running loop in the caller — ferry to the persistent bg loop.
        loop = _ensure_bg_loop()
        fut = _asyncio.run_coroutine_threadsafe(coro, loop)
        return fut.result()
    except RuntimeError:
        # No running loop in this thread — fast path.
        return _asyncio.run(coro)


def compute_provider_key(
    *,
    provider: str,
    base_url: str = "",
    organization: str = "",
) -> str:
    """SHA-256 of (provider || base_url || organization).

    Scopes the embedding cache so two openai instances pointing at
    different base URLs don't trample each other's cached vectors.
    Matches the `provider_key` column on OpenClaw's `embedding_cache`.
    """
    payload = f"{provider}|{base_url}|{organization}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---- main class ----


class MemoryIndex:
    """One `memory.db` per agent home. Lazy schema init, lazy reindex.

    Lifecycle:

    - `__init__` opens the DB, runs DDL, attempts to load `sqlite-vec`,
      checks the meta row's scope_hash; if an embedder is configured
      and the hash is absent or mismatched, drops and recreates index
      tables (chunks/chunks_fts/chunks_vec) so the vector dim stays
      aligned with the active embedding model. A BM25-only open
      (``embedder is None``) never rebuilds or rewrites meta — it
      adopts the stored scope so ``memory_get`` / embedder-resolution
      failures cannot wipe a live hybrid index.
    - `index_all()` walks tracked files, re-chunks any that have
      changed, and embeds new chunks if an embedder is wired. Cheap
      when nothing's stale — `os.stat` per file + a hash compare.
    - `search()` runs hybrid search (BM25 + vector when an embedder is
      configured, BM25-only when not). Returns at most `top_k` results
      with `score >= min_score`.

    Thread-safety: NOT shared across threads. Open per call/thread.
    Most callers go through helper functions in `memory_tools` that
    instantiate a fresh `MemoryIndex` for each tool call.
    """

    def __init__(
        self,
        agent_home: Path,
        *,
        embedder: Embedder | None = None,
        chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        fts_tokenizer: str = DEFAULT_FTS_TOKENIZER,
        sources: list[str] | None = None,
    ) -> None:
        self.agent_home = Path(agent_home)
        self.memory_dir = self.agent_home / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.memory_dir / "memory.db"
        self.embedder = embedder
        self.chunk_tokens = chunk_tokens
        self.chunk_overlap = chunk_overlap
        self.fts_tokenizer = fts_tokenizer
        self.sources = list(sources or ["memory"])

        # Resolved at boot. May change if scope_hash mismatch forces a
        # rebuild — keep the fields populated even when there's no
        # embedder so `search()` can still report the active config.
        self.provider = embedder.provider if embedder else "none"
        self.model = embedder.model if embedder else DEFAULT_EMBEDDING_MODEL
        self.vector_dims = embedder.dims if embedder else DEFAULT_EMBEDDING_DIMS
        self.provider_key = embedder.provider_key if embedder else ""

        # Whether sqlite-vec loaded successfully. False → BM25-only.
        self.has_vec0: bool = False
        self._init_db()

    # -- connection helpers -------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Open a fresh connection. WAL for concurrent readers/writer.

        We attempt `enable_load_extension` and load `sqlite-vec` on
        every connection — `vec0` virtual tables are only visible
        when the extension is loaded for THIS connection. A failure
        here flips `self.has_vec0 = False` and the rest of the code
        paths around vector search degrade to FTS-only.
        """
        con = sqlite3.connect(str(self.db_path))
        con.row_factory = sqlite3.Row
        # WAL: lets readers proceed while we're writing; survives
        # crashes better than the default rollback journal.
        try:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA synchronous=NORMAL")
            con.execute("PRAGMA temp_store=MEMORY")
        except sqlite3.DatabaseError:
            pass
        # Load sqlite-vec if available. Caller paths read self.has_vec0
        # to decide whether to use chunks_vec or fall back.
        if self._load_vec0(con):
            self.has_vec0 = True
        return con

    def _load_vec0(self, con: sqlite3.Connection) -> bool:
        """Load `sqlite-vec`. Returns True iff vec0 is now usable."""
        try:
            con.enable_load_extension(True)
        except (AttributeError, sqlite3.NotSupportedError):
            return False
        try:
            import sqlite_vec  # type: ignore[import-not-found]
        except ImportError:
            try:
                con.enable_load_extension(False)
            except Exception:  # noqa: BLE001
                pass
            return False
        try:
            sqlite_vec.load(con)
        except Exception:  # noqa: BLE001
            try:
                con.enable_load_extension(False)
            except Exception:  # noqa: BLE001
                pass
            return False
        try:
            con.enable_load_extension(False)
        except Exception:  # noqa: BLE001
            pass
        return True

    # -- schema + scope_hash ------------------------------------------

    def _init_db(self) -> None:
        con = self._connect()
        try:
            con.executescript(_SCHEMA_NON_VEC)
            if self.has_vec0:
                # Vector dimension is locked at table-create time. If a
                # future scope_hash mismatch needs a different dim,
                # `_rebuild_index_tables` recreates this table.
                con.execute(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0("
                    f"id TEXT PRIMARY KEY, embedding FLOAT[{self.vector_dims}])"
                )
            self._enforce_scope_hash(con)
            con.commit()
        finally:
            con.close()

    def _adopt_stored_scope(self, stored: dict) -> None:
        """Copy durable embedder identity off the meta row.

        Used when this process opened the index without an embedder so
        ``search()`` still reports the real hybrid config and so a
        later write of ``memory_index_meta_v1`` cannot replace a live
        ``scopeHash`` with the BM25-only defaults.
        """
        provider = stored.get("provider")
        if isinstance(provider, str) and provider:
            self.provider = provider
        model = stored.get("model")
        if isinstance(model, str) and model:
            self.model = model
        dims = stored.get("vectorDims")
        if isinstance(dims, int) and dims > 0:
            self.vector_dims = dims
        key = stored.get("providerKey")
        if isinstance(key, str):
            self.provider_key = key
        tokens = stored.get("chunkTokens")
        if isinstance(tokens, int) and tokens > 0:
            self.chunk_tokens = tokens
        overlap = stored.get("chunkOverlap")
        if isinstance(overlap, int) and overlap >= 0:
            self.chunk_overlap = overlap
        tokenizer = stored.get("ftsTokenizer")
        if isinstance(tokenizer, str) and tokenizer:
            self.fts_tokenizer = tokenizer
        sources = stored.get("sources")
        if isinstance(sources, list) and sources:
            self.sources = [str(s) for s in sources]

    def _enforce_scope_hash(self, con: sqlite3.Connection) -> None:
        """Compare configured scope_hash against meta; rebuild on mismatch.

        Opening without an embedder is BM25-only and must not treat the
        placeholder ``provider=none`` / 1536-d defaults as a new scope.
        That mismatch used to DROP ``chunks`` on ``memory_get`` and on
        any search whose embedder failed to resolve.
        """
        row = con.execute("SELECT value FROM meta WHERE key = 'memory_index_meta_v1'").fetchone()
        stored: dict | None = None
        if row is not None:
            try:
                stored = json.loads(row["value"])
            except (TypeError, ValueError):
                stored = None
        if self.embedder is None:
            if isinstance(stored, dict) and stored.get("scopeHash"):
                self._adopt_stored_scope(stored)
            return
        current = compute_scope_hash(
            provider=self.provider,
            model=self.model,
            vector_dims=self.vector_dims,
            chunk_tokens=self.chunk_tokens,
            chunk_overlap=self.chunk_overlap,
            sources=self.sources,
            fts_tokenizer=self.fts_tokenizer,
        )
        if stored is None or stored.get("scopeHash") != current:
            try:
                self._rebuild_index_tables(con)
            except Exception:
                # Partial rebuild — `executescript` may have implicitly
                # committed mid-way and left the schema in a half-state.
                # Do NOT update the meta scope_hash: the next boot still
                # sees the old (or missing) value and re-attempts the
                # rebuild instead of believing the schema is fresh. The
                # failure is not propagated; the retry is the recovery.
                return
        # Always refresh the meta blob so the JSON view is current —
        # but only after a successful rebuild (or no rebuild at all).
        con.execute(
            "INSERT INTO meta(key, value) VALUES('memory_index_meta_v1', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (
                json.dumps(
                    {
                        "version": META_VERSION,
                        "provider": self.provider,
                        "model": self.model,
                        "providerKey": self.provider_key,
                        "sources": self.sources,
                        "scopeHash": current,
                        "chunkTokens": self.chunk_tokens,
                        "chunkOverlap": self.chunk_overlap,
                        "vectorDims": self.vector_dims,
                        "ftsTokenizer": self.fts_tokenizer,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            ),
        )
        con.execute(
            "INSERT INTO meta(key, value) VALUES('last_indexed', ?) ON CONFLICT(key) DO NOTHING",
            (str(int(time.time() * 1000)),),
        )

    def _rebuild_index_tables(self, con: sqlite3.Connection) -> None:
        """Drop + recreate `chunks` / `chunks_fts` / `chunks_vec`.

        `files` and `embedding_cache` are NOT dropped — file-level
        tracking and cached vectors are still valid (the cache is keyed
        by content hash + provider + model, so it tolerates re-embedding
        only the chunks that landed in the new chunk windows).
        """
        # Drop in dependency order: chunks_vec depends on chunks.id;
        # chunks_fts is independent but tied logically.
        if self.has_vec0:
            con.execute("DROP TABLE IF EXISTS chunks_vec")
        con.executescript("DROP TABLE IF EXISTS chunks_fts;\nDROP TABLE IF EXISTS chunks;\nDELETE FROM files;")
        # Recreate (schema script is idempotent — IF NOT EXISTS clauses).
        con.executescript(_SCHEMA_NON_VEC)
        if self.has_vec0:
            con.execute(
                f"CREATE VIRTUAL TABLE chunks_vec USING vec0(id TEXT PRIMARY KEY, embedding FLOAT[{self.vector_dims}])"
            )

    # -- public API ---------------------------------------------------

    def tracked_files(self) -> list[Path]:
        """All `.md` paths under the agent home that we want to index.

        Always includes `MEMORY.md` if it exists. Always includes
        every `memory/*.md`. Skips `memory.db` and any subdirectories.
        """
        out: list[Path] = []
        memory_md = self.agent_home / "MEMORY.md"
        if memory_md.is_file():
            out.append(memory_md)
        if self.memory_dir.is_dir():
            for entry in sorted(self.memory_dir.iterdir()):
                if entry.is_file() and entry.suffix.lower() == ".md":
                    out.append(entry)
        return out

    def needs_reindex(self) -> bool:
        """True if any tracked file is newer than `files.mtime`/`size`/`hash`.

        Cheap O(N tracked files): one `stat` per file, one indexed
        lookup against `files`. Designed to be safe to call every
        heartbeat tick.
        """
        files = self.tracked_files()
        if not files:
            return False
        con = self._connect()
        try:
            for f in files:
                rel = self._rel_path(f)
                row = con.execute(
                    "SELECT mtime, size, hash FROM files WHERE path = ?",
                    (rel,),
                ).fetchone()
                try:
                    st = f.stat()
                except OSError:
                    continue
                if row is None:
                    return True
                stored_mtime = float(row["mtime"] or 0)
                stored_size = int(row["size"] or 0)
                # mtime stored as ms; compare with ~1ms tolerance.
                cur_mtime_ms = int(st.st_mtime * 1000)
                if abs(int(stored_mtime) - cur_mtime_ms) > 1 or int(st.st_size) != stored_size:
                    return True
                if self._path_needs_embedding_backfill(con, rel):
                    return True
            # Catch deletions: any tracked rows that no longer exist on disk.
            on_disk = {self._rel_path(f) for f in files}
            return any(r["path"] not in on_disk for r in con.execute("SELECT path FROM files"))
        finally:
            con.close()

    def _path_needs_embedding_backfill(
        self,
        con: sqlite3.Connection,
        rel_path: str,
    ) -> bool:
        """Return whether an active embedder can repair this path.

        A temporary embedding outage may legitimately re-index changed text in
        BM25-only mode. Those chunks carry ``[]`` and have no ``chunks_vec``
        row. Once the embedder is available again, file mtime/hash checks alone
        consider the path current, so explicitly treat incomplete embeddings
        as drift and let ``index_all()`` refill them from cache or the provider.
        """
        if self.embedder is None:
            return False
        missing_json = con.execute(
            "SELECT 1 FROM chunks WHERE path = ? AND (embedding IS NULL OR embedding = '' OR embedding = '[]') LIMIT 1",
            (rel_path,),
        ).fetchone()
        if missing_json is not None:
            return True
        if not self.has_vec0:
            return False
        missing_vec = con.execute(
            "SELECT 1 FROM chunks AS c WHERE c.path = ? "
            "AND NOT EXISTS (SELECT 1 FROM chunks_vec AS v WHERE v.id = c.id) "
            "LIMIT 1",
            (rel_path,),
        ).fetchone()
        return missing_vec is not None

    def index_all(self) -> dict:
        """Re-index every tracked file that has drifted.

        Returns a stats dict: `{indexed_files, indexed_chunks,
        skipped_unchanged, deleted_files, embedded_chunks,
        cache_hits, duration_s}`. Embedding-related fields are zero
        when no `embedder` is configured.
        """
        started = time.time()
        files = self.tracked_files()
        stats: dict[str, int] = {
            "indexed_files": 0,
            "indexed_chunks": 0,
            "skipped_unchanged": 0,
            "deleted_files": 0,
            "embedded_chunks": 0,
            "cache_hits": 0,
        }
        con = self._connect()
        try:
            on_disk_rel: set[str] = set()
            for f in files:
                rel = self._rel_path(f)
                on_disk_rel.add(rel)
                changed = self._index_one_locked(con, f, rel, stats)
                if changed:
                    stats["indexed_files"] += 1
                else:
                    stats["skipped_unchanged"] += 1
            # Drop chunks/files for paths that no longer exist on disk.
            for r in list(con.execute("SELECT path FROM files")):
                if r["path"] in on_disk_rel:
                    continue
                if self.has_vec0:
                    con.execute(
                        "DELETE FROM chunks_vec WHERE id IN (SELECT id FROM chunks WHERE path = ?)",
                        (r["path"],),
                    )
                con.execute(
                    "DELETE FROM chunks_fts WHERE id IN (SELECT id FROM chunks WHERE path = ?)",
                    (r["path"],),
                )
                con.execute("DELETE FROM chunks WHERE path = ?", (r["path"],))
                con.execute("DELETE FROM files  WHERE path = ?", (r["path"],))
                stats["deleted_files"] += 1
            con.execute(
                "INSERT INTO meta(key, value) VALUES('last_indexed', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(int(time.time() * 1000)),),
            )
            con.commit()
        finally:
            con.close()
        return {**stats, "duration_s": round(time.time() - started, 3)}

    def _rel_path(self, abs_path: Path) -> str:
        """Path relative to agent_home, with forward slashes (matches OpenClaw)."""
        try:
            rel = abs_path.resolve().relative_to(self.agent_home.resolve())
        except ValueError:
            rel = abs_path
        return str(rel).replace(os.sep, "/")

    def _index_one_locked(
        self,
        con: sqlite3.Connection,
        abs_path: Path,
        rel_path: str,
        stats: dict[str, int],
    ) -> bool:
        """Index a single file inside an open connection.

        Returns True iff the file was (re-)indexed, False iff it was
        unchanged and skipped. Caller is responsible for the commit.
        """
        try:
            body = abs_path.read_text(encoding="utf-8", errors="replace")
            st = abs_path.stat()
        except OSError:
            return False
        size = int(st.st_size)
        mtime_ms = int(st.st_mtime * 1000)
        file_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()

        row = con.execute(
            "SELECT hash, size, mtime FROM files WHERE path = ?",
            (rel_path,),
        ).fetchone()
        if row is not None and row["hash"] == file_hash and not self._path_needs_embedding_backfill(con, rel_path):
            return False  # no actual change

        # Re-chunk. The `source` tag on each chunk drives the
        # source-filter at search time. For MemoryIndex this is "memory".
        primary_source = self.sources[0] if self.sources else "memory"
        chunks = chunk_markdown(
            body,
            file_path=rel_path,
            source=primary_source,
            target_tokens=self.chunk_tokens,
            overlap_tokens=self.chunk_overlap,
        )
        # Replace ALL chunks for this path. Simplest correct strategy —
        # incremental diffing of chunks would require a stable id that
        # survives small edits, which our SHA-256(path+lines+text) id
        # explicitly does not.
        if self.has_vec0:
            # Drop the dependent vec0 rows BEFORE the chunks delete so we
            # can still SELECT chunks.id; chunks_vec rows are keyed by the
            # same id but stored in a separate virtual table.
            con.execute(
                "DELETE FROM chunks_vec WHERE id IN (SELECT id FROM chunks WHERE path = ?)",
                (rel_path,),
            )
        # Mirror the chunks delete into FTS — must come BEFORE the chunks
        # delete so `id IN (SELECT ...)` still matches.
        con.execute(
            "DELETE FROM chunks_fts WHERE id IN (SELECT id FROM chunks WHERE path = ?)",
            (rel_path,),
        )
        con.execute("DELETE FROM chunks WHERE path = ?", (rel_path,))

        now_ms = int(time.time() * 1000)
        for ch in chunks:
            embedding_json = "[]"
            embedding_vec: list[float] | None = None
            if self.embedder is not None:
                # Embedding cache lookup (skip the API call entirely
                # when content + provider + model match a prior run).
                cached = self._read_embedding_cache(con, ch.hash)
                if cached is not None:
                    embedding_vec = cached
                    stats["cache_hits"] += 1
                # Otherwise the caller's batch path will fill it in; we
                # still write the chunk row now with an empty vector
                # placeholder, and the embed pass below upserts the
                # vector once available.
            if embedding_vec is not None:
                embedding_json = json.dumps(embedding_vec)
            con.execute(
                "INSERT OR REPLACE INTO chunks(id, path, source, start_line, "
                "end_line, hash, model, text, embedding, updated_at) VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ch.id,
                    ch.path,
                    ch.source,
                    ch.start_line,
                    ch.end_line,
                    ch.hash,
                    self.model,
                    ch.text,
                    embedding_json,
                    now_ms,
                ),
            )
            con.execute(
                "INSERT INTO chunks_fts("
                "text, id, path, source, model, start_line, end_line"
                ") VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    ch.text,
                    ch.id,
                    ch.path,
                    ch.source,
                    self.model,
                    ch.start_line,
                    ch.end_line,
                ),
            )
            if embedding_vec is not None and self.has_vec0:
                self._upsert_vec(con, ch.id, embedding_vec)

        # If embedder is configured, embed any chunks that didn't have a
        # cache hit. Done in a single batch so providers that support
        # batched calls keep their throughput.
        if self.embedder is not None and chunks:
            to_embed: list[Chunk] = []
            for ch in chunks:
                row2 = con.execute(
                    "SELECT embedding FROM chunks WHERE id = ?",
                    (ch.id,),
                ).fetchone()
                if row2 is None:
                    continue
                if (row2["embedding"] or "[]") == "[]":
                    to_embed.append(ch)
            if to_embed:
                vectors = self._embed_chunks_blocking(to_embed)
                for ch, vec in zip(to_embed, vectors):
                    if vec is None:
                        continue
                    embedding_json = json.dumps(vec)
                    con.execute(
                        "UPDATE chunks SET embedding = ? WHERE id = ?",
                        (embedding_json, ch.id),
                    )
                    if self.has_vec0:
                        self._upsert_vec(con, ch.id, vec)
                    self._write_embedding_cache(con, ch.hash, vec)
                    stats["embedded_chunks"] += 1

        # Track the file's stat snapshot so `needs_reindex` sees us as up-to-date.
        con.execute(
            "INSERT INTO files(path, source, hash, mtime, size) VALUES "
            "(?, ?, ?, ?, ?) "
            "ON CONFLICT(path) DO UPDATE SET hash = excluded.hash, "
            "mtime = excluded.mtime, size = excluded.size",
            (rel_path, primary_source, file_hash, mtime_ms, size),
        )
        stats["indexed_chunks"] += len(chunks)
        return True

    def _upsert_vec(
        self,
        con: sqlite3.Connection,
        chunk_id: str,
        vec: list[float],
    ) -> None:
        """Upsert one row into `chunks_vec`. JSON-array form is what
        sqlite-vec accepts as the bind value for a FLOAT[] column.

        ``vec0`` virtual tables do not implement SQLite's normal
        ``INSERT OR REPLACE`` conflict handling, so replacement must be
        expressed explicitly as delete-then-insert.
        """
        if not self.has_vec0:
            return
        con.execute(
            "DELETE FROM chunks_vec WHERE id = ?",
            (chunk_id,),
        )
        con.execute(
            "INSERT INTO chunks_vec(id, embedding) VALUES (?, ?)",
            (chunk_id, json.dumps(vec)),
        )

    # ---- embedding cache --------------------------------------------

    def _read_embedding_cache(
        self,
        con: sqlite3.Connection,
        content_hash: str,
    ) -> list[float] | None:
        if self.embedder is None:
            return None
        row = con.execute(
            "SELECT embedding, dims FROM embedding_cache WHERE "
            "provider = ? AND model = ? AND provider_key = ? AND hash = ?",
            (self.provider, self.model, self.provider_key, content_hash),
        ).fetchone()
        if row is None:
            return None
        try:
            vec = json.loads(row["embedding"])
        except (TypeError, ValueError):
            return None
        if not isinstance(vec, list) or len(vec) != self.vector_dims:
            return None
        return [float(x) for x in vec]

    def _write_embedding_cache(
        self,
        con: sqlite3.Connection,
        content_hash: str,
        vec: list[float],
    ) -> None:
        if self.embedder is None:
            return
        con.execute(
            "INSERT OR REPLACE INTO embedding_cache("
            "provider, model, provider_key, hash, embedding, dims, updated_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                self.provider,
                self.model,
                self.provider_key,
                content_hash,
                json.dumps(vec),
                self.vector_dims,
                int(time.time() * 1000),
            ),
        )

    def _embed_chunks_blocking(
        self,
        chunks: list[Chunk],
    ) -> list[list[float] | None]:
        """Synchronous wrapper around the (async) embedder.

        Callers come from two contexts:

        - The tool runner subprocess (`run_memory_search` → `index_all`).
          That's a fresh thread with no running loop, so a plain
          `asyncio.run` works.
        - Tests / heartbeat / any in-process caller that's itself inside
          an `asyncio.run` block. `asyncio.run` would raise there with
          "loop is already running", so we offload to a worker thread.

        `_run_blocking_coroutine` does the right thing in both cases.
        """
        if self.embedder is None:
            return [None] * len(chunks)
        try:
            vecs = _run_blocking_coroutine(self.embedder.embed_batch([c.text for c in chunks]))
        except Exception:  # noqa: BLE001
            return [None] * len(chunks)
        out: list[list[float] | None] = []
        emitted_dim = 0
        for vec in vecs:
            if not isinstance(vec, list) or len(vec) != self.vector_dims:
                if isinstance(vec, list) and vec:
                    emitted_dim = len(vec)
                out.append(None)
            else:
                out.append([float(x) for x in vec])
        # Never fail silently: if the endpoint emits a consistent width that
        # differs from the index's vector table, EVERY embedding is dropped here
        # and semantic search silently becomes BM25-only. Surface it loudly (once
        # per index) so a dims misconfiguration is diagnosable instead of a
        # mystery "search returns nothing". The permanent fix is that the
        # embedder now probes the endpoint width when config omits `dims`
        # (memory_embed._probe_endpoint_dims); this guard is the last-resort alarm.
        if emitted_dim and not getattr(self, "_warned_dim_drop", False):
            self._warned_dim_drop = True
            _log.warning(
                "memory index %s: embedder emitted %d-d vectors but the index "
                "vector table is %d-d — ALL vectors dropped, semantic search is "
                "BM25-only. Set embedding.dims=%d (endpoint's real width) so the "
                "scope rebuilds the vec table to match.",
                getattr(self, "db_path", "?"),
                emitted_dim,
                self.vector_dims,
                emitted_dim,
            )
        return out

    # ---- search -----------------------------------------------------

    def search(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_MAX_RESULTS,
        min_score: float = DEFAULT_MIN_SCORE,
        relevance_floor: float = 0.0,
        vector_weight: float = DEFAULT_HYBRID_VECTOR_WEIGHT,
        text_weight: float = DEFAULT_HYBRID_TEXT_WEIGHT,
        candidate_multiplier: int = DEFAULT_HYBRID_CANDIDATE_MULTIPLIER,
        mmr_enabled: bool = False,
        mmr_lambda: float = DEFAULT_MMR_LAMBDA,
        sources: list[str] | None = None,
    ) -> list[MemoryHit]:
        """Hybrid search. Falls back to FTS-only when no embedder is
        configured or `chunks_vec` is unavailable.

        Two complementary filters control how many results come back:

        - `top_k`            — hard cap on result count (default 6).
        - `min_score`        — absolute floor on the fused score
                               (default 0.35). Drops candidates below
                               this regardless of how many we already
                               have.
        - `relevance_floor`  — relative-to-best floor as a fraction of
                               the top hit's score, in [0, 1]. Default
                               0 = off. Set to 0.5 to drop everything
                               that scores below 50% of the best hit
                               — useful when "top 6" returns one
                               great match plus 5 weak ones and the
                               weak ones are noise.

        Caller can clamp the source list to e.g. `["memory"]` to
        exclude session transcripts.
        """
        query = (query or "").strip()
        if not query:
            return []
        candidate_k = max(1, candidate_multiplier) * max(1, top_k)
        sources = list(sources or self.sources or ["memory"])
        # Clamp relevance_floor to [0, 1] so a misconfigured caller
        # (negative or >1) doesn't accidentally drop everything or
        # surface noise.
        relevance_floor = max(0.0, min(1.0, float(relevance_floor)))

        con = self._connect()
        try:
            text_results = self._fts_search(con, query, candidate_k, sources)
            vec_results: list[dict] = []
            if self.embedder is not None and self.has_vec0:
                vec_results = self._vector_search(con, query, candidate_k, sources)
            return self._fuse(
                text_results,
                vec_results,
                top_k=top_k,
                min_score=min_score,
                relevance_floor=relevance_floor,
                vector_weight=vector_weight,
                text_weight=text_weight,
                mmr_enabled=mmr_enabled,
                mmr_lambda=mmr_lambda,
            )
        finally:
            con.close()

    def _fts_search(
        self,
        con: sqlite3.Connection,
        query: str,
        candidate_k: int,
        sources: list[str],
    ) -> list[dict]:
        terms = fts_terms(query)
        if not terms:
            return []
        # AND is the precision-primary pass: every query term must be present.
        # If it matches nothing, the query almost always carried a term the
        # corpus doesn't contain (a stop-word the model appended, a typo) — so
        # fall back to OR to *recall* the chunk that holds the most query
        # terms. Term coverage (below) then keeps OR honest. Without this
        # fallback, a natural-language question like "what is <fact>?" returns
        # zero rows because "what" is absent from the daily.
        rows = self._fts_rows(con, build_fts_query(query, mode="and"), candidate_k)
        if not rows:
            rows = self._fts_rows(con, build_fts_query(query, mode="or"), candidate_k)
        n_terms = len(terms)
        out: list[dict] = []
        for r in rows:
            if r["source"] not in sources:
                continue
            # Term coverage: fraction of the query's distinct terms that
            # actually appear in the chunk text. Corpus-independent evidence of
            # match quality that BM25's IDF weighting throws away on small
            # corpora. Measured on the full chunk text with the same
            # tokenization build_fts_query uses.
            chunk_terms = set(fts_terms(r["text"] or ""))
            present = sum(1 for t in terms if t in chunk_terms)
            coverage = (present / n_terms) if n_terms else 0.0
            out.append(
                {
                    "id": r["id"],
                    "path": r["path"],
                    "source": r["source"],
                    "start_line": int(r["start_line"]),
                    "end_line": int(r["end_line"]),
                    "snippet": r["snippet"] or "",
                    "text_score": bm25_rank_to_score(float(r["rank"])),
                    "coverage": coverage,
                }
            )
            if len(out) >= candidate_k:
                break
        return out

    def _fts_rows(
        self,
        con: sqlite3.Connection,
        match: str | None,
        candidate_k: int,
    ) -> list[sqlite3.Row]:
        """Run one FTS5 MATCH pass. Returns rows ordered by bm25 (best first),
        carrying the full chunk ``text`` so the caller can score term coverage.
        Filtering on ``source`` is done in Python (see caller) so bm25's
        internal ordering is preserved."""
        if match is None:
            return []
        return con.execute(
            "SELECT id, path, source, start_line, end_line, text, "
            "       snippet(chunks_fts, 0, '', '', '…', 64) AS snippet, "
            "       bm25(chunks_fts) AS rank "
            "FROM chunks_fts WHERE chunks_fts MATCH ? "
            "ORDER BY rank LIMIT ?",
            (match, candidate_k * 2),
        ).fetchall()

    def _vector_search(
        self,
        con: sqlite3.Connection,
        query: str,
        candidate_k: int,
        sources: list[str],
    ) -> list[dict]:
        if self.embedder is None:
            return []
        try:
            qvec = _run_blocking_coroutine(self.embedder.embed_query(query))
        except Exception:  # noqa: BLE001
            return []
        if not isinstance(qvec, list) or len(qvec) != self.vector_dims:
            return []
        # `vec0` exposes a KNN MATCH operator; we filter on source via a
        # join against `chunks` after the vec MATCH so it stays inside
        # the ANN index.
        rows = con.execute(
            "SELECT v.id AS id, v.distance AS distance, "
            "       c.path AS path, c.source AS source, "
            "       c.start_line AS start_line, c.end_line AS end_line, "
            "       c.text AS text "
            "FROM chunks_vec AS v JOIN chunks AS c ON c.id = v.id "
            "WHERE v.embedding MATCH ? AND k = ? "
            "ORDER BY v.distance",
            (json.dumps(qvec), candidate_k * 2),
        ).fetchall()
        out: list[dict] = []
        for r in rows:
            if r["source"] not in sources:
                continue
            # vec0 returns euclidean / cosine distance depending on column
            # type — for normalized vectors it's 2*(1 - cos_sim). Convert
            # to a [0, 1] similarity score.
            distance = float(r["distance"])
            similarity = max(0.0, 1.0 - distance / 2.0)
            text = r["text"] or ""
            snippet = text[:DEFAULT_SNIPPET_CHARS]
            out.append(
                {
                    "id": r["id"],
                    "path": r["path"],
                    "source": r["source"],
                    "start_line": int(r["start_line"]),
                    "end_line": int(r["end_line"]),
                    "snippet": snippet,
                    "vector_score": similarity,
                }
            )
            if len(out) >= candidate_k:
                break
        return out

    def _fuse(
        self,
        text_hits: list[dict],
        vec_hits: list[dict],
        *,
        top_k: int,
        min_score: float,
        relevance_floor: float = 0.0,
        vector_weight: float,
        text_weight: float,
        mmr_enabled: bool,
        mmr_lambda: float,
    ) -> list[MemoryHit]:
        # Renormalize the weights based on which modalities actually
        # produced candidates. Without this, FTS-only mode multiplies
        # every text match by 0.3 (the default text_weight) and trips
        # the min_score=0.35 floor on perfectly-good matches. Same for
        # vector-only when FTS returns empty (e.g. punctuation-only
        # query that the tokenizer drops). Mirrors what OpenClaw does
        # when one of the indexes is disabled.
        active_vec = bool(vec_hits) and self.embedder is not None
        active_txt = bool(text_hits)
        if active_vec and active_txt:
            total_w = max(1e-9, vector_weight + text_weight)
            vw = vector_weight / total_w
            tw = text_weight / total_w
        elif active_vec:
            vw, tw = 1.0, 0.0
        else:
            vw, tw = 0.0, 1.0

        by_id: dict[str, dict] = {}
        for r in vec_hits:
            by_id[r["id"]] = {
                "id": r["id"],
                "path": r["path"],
                "source": r["source"],
                "start_line": r["start_line"],
                "end_line": r["end_line"],
                "snippet": r["snippet"],
                "vector_score": float(r.get("vector_score") or 0.0),
                "text_score": 0.0,
                "coverage": 0.0,
            }
        for r in text_hits:
            existing = by_id.get(r["id"])
            if existing is not None:
                existing["text_score"] = float(r.get("text_score") or 0.0)
                existing["coverage"] = float(r.get("coverage") or 0.0)
                # FTS snippet is more informative than the raw chunk text
                # we used as a placeholder on the vec side.
                if r.get("snippet"):
                    existing["snippet"] = r["snippet"]
            else:
                by_id[r["id"]] = {
                    "id": r["id"],
                    "path": r["path"],
                    "source": r["source"],
                    "start_line": r["start_line"],
                    "end_line": r["end_line"],
                    "snippet": r["snippet"],
                    "vector_score": 0.0,
                    "text_score": float(r.get("text_score") or 0.0),
                    "coverage": float(r.get("coverage") or 0.0),
                }

        scored: list[dict] = []
        for entry in by_id.values():
            score = vw * entry["vector_score"] + tw * entry["text_score"]
            # Lexical-confidence floor: a chunk that
            # lexically contains (almost) the whole query is a confident match
            # even when BM25's IDF has collapsed to ~0 on a small corpus. Floor
            # its score by coverage so an exact-phrase hit clears the default
            # ``min_score`` instead of being dropped at ~0.000. ``max`` means a
            # strong large-corpus score is never lowered; the floor stays below
            # an explicit high ``min_score`` so precision-tuned callers still
            # filter it out downstream.
            coverage = float(entry.get("coverage") or 0.0)
            if coverage >= LEXICAL_COVERAGE_MIN:
                score = max(score, LEXICAL_CONFIDENT_FLOOR * coverage)
            scored.append({**entry, "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)

        if mmr_enabled and len(scored) > 1:
            scored = self._apply_mmr(scored, mmr_lambda)

        # Apply the relative-to-best filter on top of `min_score`.
        # `relevance_floor` is a fraction of the top hit's score —
        # 0.5 means "drop anything below 50% of the best hit." The
        # absolute `min_score` and the relative `relevance_floor`
        # compose as a logical AND.
        rel_cutoff = 0.0
        if scored and relevance_floor > 0:
            rel_cutoff = relevance_floor * float(scored[0]["score"])

        out: list[MemoryHit] = []
        for entry in scored:
            if entry["score"] < min_score:
                continue
            if entry["score"] < rel_cutoff:
                continue
            out.append(
                MemoryHit(
                    id=entry["id"],
                    path=entry["path"],
                    source=entry["source"],
                    start_line=entry["start_line"],
                    end_line=entry["end_line"],
                    score=float(entry["score"]),
                    vector_score=float(entry["vector_score"]),
                    text_score=float(entry["text_score"]),
                    snippet=entry["snippet"][:DEFAULT_SNIPPET_CHARS],
                )
            )
            if len(out) >= top_k:
                break
        return out

    def _apply_mmr(self, scored: list[dict], lambda_: float) -> list[dict]:
        """Token-overlap MMR over snippets.

        OpenClaw's manager.ts uses a token-set cache + Jaccard-like
        similarity. Since we don't have the candidate vectors handy
        for every entry once we've serialized them, the snippet
        overlap is a pragmatic stand-in that runs in pure Python.
        """
        if not scored:
            return []
        lambda_ = max(0.0, min(1.0, lambda_))
        token_cache: dict[str, set[str]] = {}

        def toks(entry: dict) -> set[str]:
            cached = token_cache.get(entry["id"])
            if cached is not None:
                return cached
            text = (entry.get("snippet") or "").lower()
            tokens = set(re.findall(r"[\w]{2,}", text, flags=re.UNICODE))
            token_cache[entry["id"]] = tokens
            return tokens

        remaining = list(scored)
        selected: list[dict] = []
        while remaining:
            if not selected:
                # Best raw score first.
                best = remaining.pop(0)
                selected.append(best)
                continue
            best_idx = 0
            best_mmr = -math.inf
            for i, cand in enumerate(remaining):
                cand_tokens = toks(cand)
                max_sim = 0.0
                for s in selected:
                    s_tokens = toks(s)
                    if not cand_tokens or not s_tokens:
                        continue
                    inter = len(cand_tokens & s_tokens)
                    union = len(cand_tokens | s_tokens)
                    sim = inter / union if union else 0.0
                    if sim > max_sim:
                        max_sim = sim
                mmr = lambda_ * cand["score"] - (1 - lambda_) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i
            selected.append(remaining.pop(best_idx))
        return selected

    # ---- direct file access -----------------------------------------

    def get_file(
        self,
        rel_path: str,
        *,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        """Read `agent_home/<rel_path>` with optional 1-based line range.

        Caller-facing privacy gate is enforced ABOVE this — by the time
        we're here we trust `rel_path`. We still refuse `..` traversal
        and absolute paths defensively.
        """
        rel_path = rel_path.strip().lstrip("/")
        if not rel_path or ".." in rel_path.split("/"):
            raise ValueError(f"invalid memory path: {rel_path!r}")
        candidate = (self.agent_home / rel_path).resolve()
        try:
            candidate.relative_to(self.agent_home.resolve())
        except ValueError:
            raise ValueError(f"path escapes agent home: {rel_path!r}") from None
        if not candidate.is_file():
            raise FileNotFoundError(f"no such memory file: {rel_path}")
        body = candidate.read_text(encoding="utf-8", errors="replace")
        if start_line is None and end_line is None:
            return body
        lines = body.split("\n")
        s = max(1, int(start_line or 1))
        e = max(s, int(end_line or len(lines)))
        return "\n".join(lines[s - 1 : e])

    def stats(self) -> dict:
        """One-shot dump of index health for `sharedagents memory status`."""
        con = self._connect()
        try:
            chunk_count = con.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]
            file_count = con.execute("SELECT count(*) AS n FROM files").fetchone()["n"]
            cache_count = con.execute("SELECT count(*) AS n FROM embedding_cache").fetchone()["n"]
            embedded_count = con.execute(
                "SELECT count(*) AS n FROM chunks WHERE embedding IS NOT NULL AND embedding != '' AND embedding != '[]'"
            ).fetchone()["n"]
            vector_count = 0
            if self.has_vec0:
                vector_count = con.execute("SELECT count(*) AS n FROM chunks_vec").fetchone()["n"]
            row = con.execute("SELECT value FROM meta WHERE key = 'memory_index_meta_v1'").fetchone()
            meta = json.loads(row["value"]) if row else {}
            row2 = con.execute("SELECT value FROM meta WHERE key = 'last_indexed'").fetchone()
            last = int(row2["value"]) if row2 else 0
        finally:
            con.close()
        try:
            db_size = self.db_path.stat().st_size
        except OSError:
            db_size = 0
        return {
            "db_path": str(self.db_path),
            "db_size_bytes": db_size,
            "has_vec0": self.has_vec0,
            "chunk_count": chunk_count,
            "embedded_count": embedded_count,
            "vector_count": vector_count,
            "file_count": file_count,
            "cache_count": cache_count,
            "last_indexed_ms": last,
            "meta": meta,
        }

    def clear(self) -> None:
        """Drop chunks/FTS/vec/files — what `sharedagents memory index --force`
        triggers. **Keeps `embedding_cache`**: it's keyed by
        `(provider, model, provider_key, hash)`, so identical chunk text
        reappears with the same hash and skips the API. Dropping it
        here would force every `--force` rebuild to re-embed, which is
        what every operator paying for OpenAI tokens specifically does
        NOT want.
        """
        con = self._connect()
        try:
            if self.has_vec0:
                con.execute("DELETE FROM chunks_vec")
            con.executescript("DELETE FROM chunks_fts;\nDELETE FROM chunks;\nDELETE FROM files;")
            con.commit()
        finally:
            con.close()


# ---- DDL ----------------------------------------------------------------

# `chunks_vec` is created separately (per-connection, after sqlite-vec
# loads, with the dim baked in). Everything else is pure SQLite.

_SCHEMA_NON_VEC = """
CREATE TABLE IF NOT EXISTS files (
  path     TEXT PRIMARY KEY,
  source   TEXT NOT NULL DEFAULT 'memory',
  hash     TEXT NOT NULL,
  mtime    INTEGER NOT NULL,
  size     INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
  id          TEXT PRIMARY KEY,
  path        TEXT NOT NULL,
  source      TEXT NOT NULL DEFAULT 'memory',
  start_line  INTEGER NOT NULL,
  end_line    INTEGER NOT NULL,
  hash        TEXT NOT NULL,
  model       TEXT NOT NULL,
  text        TEXT NOT NULL,
  embedding   TEXT NOT NULL,
  updated_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_path   ON chunks(path);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);

-- FTS5 in regular (non-content) mode. Matches OpenClaw's bundled
-- schema. We DON'T use AFTER INSERT/DELETE triggers — FTS5's `delete`
-- pseudo-command requires the *exact* column tuple at delete time,
-- which is fragile across `INSERT OR REPLACE` (rowid churn). Instead,
-- the Python upsert path keeps the two tables in lock-step.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  text,
  id         UNINDEXED,
  path       UNINDEXED,
  source     UNINDEXED,
  model      UNINDEXED,
  start_line UNINDEXED,
  end_line   UNINDEXED,
  tokenize='unicode61'
);

CREATE TABLE IF NOT EXISTS embedding_cache (
  provider     TEXT NOT NULL,
  model        TEXT NOT NULL,
  provider_key TEXT NOT NULL,
  hash         TEXT NOT NULL,
  embedding    TEXT NOT NULL,
  dims         INTEGER,
  updated_at   INTEGER NOT NULL,
  PRIMARY KEY (provider, model, provider_key, hash)
);
CREATE INDEX IF NOT EXISTS idx_embedding_cache_updated_at ON embedding_cache(updated_at);

CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""


__all__ = [
    "Chunk",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_CHUNK_TOKENS",
    "DEFAULT_EMBEDDING_DIMS",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_HYBRID_CANDIDATE_MULTIPLIER",
    "DEFAULT_HYBRID_TEXT_WEIGHT",
    "DEFAULT_HYBRID_VECTOR_WEIGHT",
    "DEFAULT_MAX_RESULTS",
    "DEFAULT_MIN_SCORE",
    "DEFAULT_MMR_LAMBDA",
    "DEFAULT_SNIPPET_CHARS",
    "DEFAULT_TEMPORAL_DECAY_HALF_LIFE_DAYS",
    "Embedder",
    "MemoryHit",
    "MemoryIndex",
    "bm25_rank_to_score",
    "build_fts_query",
    "chunk_markdown",
    "compute_provider_key",
    "compute_scope_hash",
]
