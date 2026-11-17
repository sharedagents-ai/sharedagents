"""Memory index **admin** CLI — ``sharedagents memory {status,search,index,prune,synth}``.

This is low-level operator tooling that administers the on-disk memory index DB
directly (stats / raw-SQL prune / rebuild / forced synth). Unlike the model and
the agent hot path (which reach memory ONLY over the ``mcp__memory__*`` wire), a DB-admin tool legitimately uses the engine in-process — you
do not administer a database through its own application's RPC. Keeping it here,
INSIDE ``sharedagents_mcp``, is what lets ``sharedagents_tui`` stay engine-free while the
operator keeps a full-power local admin surface.
"""

from __future__ import annotations

import argparse
import sys


def cmd_memory(args: argparse.Namespace) -> int:
    """Dispatch `sharedagents memory {status,search,index,prune}`. See `--help`.

    Each subcommand opens a fresh `MemoryIndex` against the named agent's
    home directory (`~/.sharedagents/agents/<name>/`). `index --force` drops
    the entire DB first; otherwise we run incremental indexing — only
    re-chunk files whose hash has changed.
    """
    import json as _json

    from sharedagents_agent import agent as _agent_mod
    from sharedagents_mcp.memory.engine.memory_index import MemoryIndex

    op = getattr(args, "op", None)
    name = getattr(args, "agent", "jarvis") or "jarvis"
    home = _agent_mod.agents_root() / name
    if not home.is_dir():
        print(
            f"error: agent {name!r} has no home directory ({home}). Run `sharedagents agent init` first.",
            file=sys.stderr,
        )
        return 2

    embedder = None
    try:
        from sharedagents_mcp.memory.engine.memory_embed import resolve_embedder

        embedder = resolve_embedder()
    except Exception:  # noqa: BLE001
        embedder = None

    if op == "status":
        idx = MemoryIndex(home, embedder=embedder)
        s = idx.stats()
        needs_index = idx.needs_reindex()
        meta = s.get("meta") or {}
        print(f"agent      : {name}")
        print(f"db         : {s['db_path']} ({s['db_size_bytes']:,} bytes)")
        print(f"vec0       : {'loaded' if s['has_vec0'] else 'NOT LOADED (FTS-only)'}")
        print(f"chunks     : {s['chunk_count']:,}")
        print(f"embeddings : {s['embedded_count']:,}/{s['chunk_count']:,}")
        print(f"vectors    : {s['vector_count']:,}/{s['chunk_count']:,}")
        print(f"files      : {s['file_count']:,}")
        print(f"cache rows : {s['cache_count']:,}")
        print(f"needs index: {'yes' if needs_index else 'no'}")
        last_ms = s.get("last_indexed_ms") or 0
        if last_ms:
            import datetime as _dt

            ts = _dt.datetime.fromtimestamp(last_ms / 1000).isoformat(timespec="seconds")
            print(f"last index : {ts}")
        else:
            print("last index : never")
        print(f"provider   : {meta.get('provider', 'none')}")
        print(f"model      : {meta.get('model', '-')}")
        print(f"vec dims   : {meta.get('vectorDims', '-')}")
        print(f"chunk_tok  : {meta.get('chunkTokens', '-')}  overlap: {meta.get('chunkOverlap', '-')}")
        print(f"sources    : {','.join(meta.get('sources', ['memory']))}")
        if scope := meta.get("scopeHash"):
            print(f"scopeHash  : {scope[:24]}...")
        # Show tracked files
        tracked = idx.tracked_files()
        if tracked:
            print(f"\ntracked ({len(tracked)}):")
            for p in tracked[:20]:
                try:
                    size = p.stat().st_size
                    print(f"  {p.relative_to(home)}  ({size:,} bytes)")
                except OSError:
                    print(f"  {p.relative_to(home)}  (?)")
            if len(tracked) > 20:
                print(f"  ... and {len(tracked) - 20} more")
        return 0

    if op == "search":
        idx = MemoryIndex(home, embedder=embedder)
        if idx.needs_reindex():
            idx.index_all()
        hits = idx.search(
            args.query,
            top_k=int(args.top_k or 6),
            min_score=float(args.min_score or 0.35),
            relevance_floor=float(getattr(args, "relevance_floor", 0.0) or 0.0),
        )
        if args.json:
            print(
                _json.dumps(
                    [
                        {
                            "id": h.id,
                            "path": h.path,
                            "source": h.source,
                            "start_line": h.start_line,
                            "end_line": h.end_line,
                            "score": h.score,
                            "vector_score": h.vector_score,
                            "text_score": h.text_score,
                            "snippet": h.snippet,
                        }
                        for h in hits
                    ],
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0
        if not hits:
            print(f"no matches for {args.query!r} (min_score={args.min_score})")
            return 0
        for i, h in enumerate(hits, 1):
            print(f"\n--- {i}. score {h.score:.3f}  vec {h.vector_score:.3f}  text {h.text_score:.3f}")
            print(f"    {h.path}:{h.start_line}-{h.end_line}")
            print(f"    {h.snippet[:300]!r}")
        return 0

    if op == "index":
        idx = MemoryIndex(home, embedder=embedder)
        if getattr(args, "reembed", False):
            if embedder is None:
                print(
                    "error: --reembed requires a configured embedder "
                    "(set OPENAI_API_KEY, run `codex login`, or configure "
                    "memory_embedding in ~/.sharedagents/config.json).",
                    file=sys.stderr,
                )
                return 2
            print("dropping chunks/FTS/vec/files AND embedding cache...")
            idx.clear()
            con = idx._connect()
            try:
                con.execute("DELETE FROM embedding_cache")
                con.commit()
            finally:
                con.close()
        elif args.force:
            print("dropping chunks/FTS/vec/files (cache survives)...")
            idx.clear()
        stats = idx.index_all()
        print(
            f"indexed:  files={stats['indexed_files']}  "
            f"chunks={stats['indexed_chunks']}  "
            f"skipped={stats['skipped_unchanged']}  "
            f"deleted={stats['deleted_files']}  "
            f"embedded={stats['embedded_chunks']}  "
            f"cache_hits={stats['cache_hits']}  "
            f"t={stats['duration_s']}s"
        )
        return 0

    if op == "prune":
        import datetime as _dt
        import re as _re
        import sqlite3 as _sql

        days = int(args.older_than_days or 365)
        cutoff = _dt.date.today() - _dt.timedelta(days=days)
        idx = MemoryIndex(home, embedder=embedder)
        # Match `memory/YYYY-MM-DD*.md` paths and compare to cutoff.
        date_re = _re.compile(r"^memory/(\d{4})-(\d{2})-(\d{2})")
        con = _sql.connect(str(idx.db_path))
        try:
            paths_to_prune: list[str] = []
            for r in con.execute("SELECT DISTINCT path FROM files"):
                p = r[0]
                m = date_re.match(p)
                if not m:
                    continue
                try:
                    d = _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                except ValueError:
                    continue
                if d < cutoff:
                    paths_to_prune.append(p)
            if not paths_to_prune:
                print(f"nothing older than {cutoff.isoformat()} to prune.")
                return 0
            print(f"pruning {len(paths_to_prune)} files older than {cutoff.isoformat()}:")
            for p in paths_to_prune[:20]:
                print(f"  {p}")
            if len(paths_to_prune) > 20:
                print(f"  ... and {len(paths_to_prune) - 20} more")
            for p in paths_to_prune:
                if idx.has_vec0:
                    con.execute(
                        "DELETE FROM chunks_vec WHERE id IN (SELECT id FROM chunks WHERE path = ?)",
                        (p,),
                    )
                con.execute("DELETE FROM chunks_fts WHERE id IN (SELECT id FROM chunks WHERE path = ?)", (p,))
                con.execute("DELETE FROM chunks WHERE path = ?", (p,))
                con.execute("DELETE FROM files  WHERE path = ?", (p,))
            con.commit()
        finally:
            con.close()
        print(f"pruned {len(paths_to_prune)} files.")
        return 0

    if op == "synth":
        # Force a memory-synth pass: read MEMORY.md + recent dailies,
        # ask the configured upstream to rewrite MEMORY.md, persist
        # atomically. ``--force`` bypasses the "no daily newer than
        # MEMORY.md" guard so an operator can re-run on demand. The
        # agent normally runs this on the ``daily_appended`` event;
        # this CLI is the manual escape hatch (and the test_07
        # forced-synth path).
        import asyncio as _asyncio

        from sharedagents_agent.agent import AgentConfig as _Acfg
        from sharedagents_mcp.memory.engine.memory_synth import synth_memory as _synth

        try:
            cfg = _Acfg.load(name)
        except FileNotFoundError:
            print(
                f"error: agent {name!r} not configured. Run `sharedagents agent init {name}` first.",
                file=sys.stderr,
            )
            return 2
        # Resolve the model in this priority order:
        #   1. CLI override (--model X)
        #   2. cfg.memory_flush_model (synth-specific override)
        #   3. cfg.upstream_model (primary chat model)
        endpoint = getattr(args, "endpoint", None) or cfg.upstream_endpoint
        model = getattr(args, "model", None) or cfg.memory_flush_model or cfg.upstream_model or ""
        if not endpoint:
            print(
                "error: no upstream endpoint configured. Set one via "
                "`sharedagents configure --endpoint <url>` or pass --endpoint.",
                file=sys.stderr,
            )
            return 2
        # Empty model is fine — vLLM/SGLang routes to the endpoint's
        # default. For OpenAI-style routers (OpenRouter) pass --model.
        timeout_s = float(getattr(args, "timeout", None) or 60.0)
        force = bool(getattr(args, "force", False))
        stats = _asyncio.run(
            _synth(
                home,
                upstream_endpoint=endpoint,
                upstream_model=model,
                timeout_s=timeout_s,
                force=force,
                cfg=cfg,
            )
        )
        if getattr(args, "json", False):
            print(_json.dumps(stats, indent=2, ensure_ascii=False))
            return 0 if stats.get("ok") else 1
        ok = bool(stats.get("ok"))
        changed = bool(stats.get("changed"))
        reason = stats.get("reason") or ""
        err = stats.get("error") or ""
        dur = stats.get("duration_s")
        head = "✓" if ok else "✗"
        line = f"{head} memory synth — agent {name}"
        if dur is not None:
            line += f"  ({dur}s)"
        print(line)
        if ok:
            if changed:
                print(f"  MEMORY.md rewritten — {home / 'MEMORY.md'}")
            elif reason:
                print(f"  no-op: {reason}")
            else:
                print("  no-op")
        else:
            print(f"  error: {err}")
            preview = stats.get("preview")
            if preview:
                print(f"  preview: {preview!r}")
        return 0 if ok else 1

    print(f"unknown memory op: {op}", file=sys.stderr)
    return 2
