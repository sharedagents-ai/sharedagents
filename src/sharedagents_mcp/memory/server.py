"""sharedagents memory MCP server.

Re-exposes the full memory surface — ``memory_search`` (hybrid search over
MEMORY.md + dailies), ``memory_get`` (read a memory file / line range by path),
``session_search`` (FTS5 search over saved session turns), and the ingestion
tool ``remember`` (add/replace/remove daily entries) — as
``mcp__memory__search`` / ``mcp__memory__get`` / ``mcp__memory__session_search`` /
``mcp__memory__remember``.

Each handler calls the memory **engine** (``sharedagents_mcp.memory.engine.*`` and
``sharedagents_session.session_index``) DIRECTLY — never ``sharedagents_agent.tools.runners``.
That breaks the ``server → runners → engine`` recursion (server-side handlers
live under ``sharedagents_mcp.memory``): a first-party
package that reaches memory only over this wire must not depend, even
transitively through the server it spawns, on the heavy ``sharedagents_agent`` stack.
The path-traversal / privacy gates and env-driven home resolution live inside
the engine functions, so they are unchanged; this module reimplements no gate

**Security — trusted chat context.** The four model-visible tools resolve their
privacy key (``chat_id``) and ``workspace_origin`` from a per-call argument
INJECTED BY TRUSTED FIRST-PARTY CODE (the tool runner or the wire client), else
the spawn env. ``chat_id`` / ``workspace_origin`` are absent from ``TOOL_DEFS``,
so the MODEL cannot supply them — the runner strips/overwrites any model-emitted
value with the turn's trusted session id before dispatch. This is what lets a
PERSISTENT per-home server serve many chats correctly (each call carries the
current turn's trusted chat) without the model being able to forge a ``chat_id``.
The internal dispatch-only ops are likewise reached only by trusted
first-party code via ``sharedagents_mcp.memory.client``.

The read-only tools declare ``readOnlyHint``; ``remember`` is mutating.
Run: ``python -m sharedagents_mcp.memory.server`` (stdio; configured under
``mcpServers.memory``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sharedagents_mcp._stdio_server import ToolError, run_stdio_server

# 64 KiB cap per tool result — matches the runner's MAX_OUTPUT_BYTES so wire
# output is byte-identical to the pre-migration in-process path.
_MAX_OUTPUT_BYTES = 64 * 1024


def _truncate(text: str) -> str:
    enc = text.encode("utf-8")
    if len(enc) <= _MAX_OUTPUT_BYTES:
        return text
    head = enc[:_MAX_OUTPUT_BYTES].decode("utf-8", "ignore")
    return head + f"\n\n[... truncated, {len(enc) - _MAX_OUTPUT_BYTES} more bytes ...]"


def _origin(arguments: dict[str, Any] | None = None) -> str:
    # A per-call ``workspace_origin`` may be injected by TRUSTED first-party code
    # (the tool runner / wire client) — it is NOT in the model-visible TOOL_DEFS,
    # so the model cannot supply it. Falls back to the spawn env.
    if arguments:
        v = arguments.get("workspace_origin")
        if isinstance(v, str) and v:
            return v
    return os.environ.get("SHAREDAGENTS_WORKSPACE_ORIGIN", "")


def _chat_id(arguments: dict[str, Any] | None = None) -> str:
    # TRUSTED: a per-call ``chat_id`` may be injected by the tool runner /
    # wire client (both first-party); it is absent from TOOL_DEFS so the
    # MODEL cannot forge it. Falls back to the spawn env (the one-shot
    # path). See the security note in the module docstring.
    if arguments:
        v = arguments.get("chat_id")
        if isinstance(v, str) and v:
            return v
    return os.environ.get("SHAREDAGENTS_SESSION_ID") or ""


@dataclass(frozen=True)
class _RememberCfgShim:
    """The subset of AgentConfig that ``memory_write.remember`` reads — a thin
    stand-in so the server need not load the full config from disk."""

    name: str
    home: Path
    remember_synthesizer_model: str = ""
    memory_flush_model: str = ""


def _search(arguments: dict[str, Any]) -> str:
    from sharedagents_mcp.memory.engine import memory_tools as _mt

    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ToolError("query required")
    top_k = arguments.get("top_k")
    min_score = arguments.get("min_score")
    relevance_floor = arguments.get("relevance_floor")
    body = _mt.memory_search(
        query,
        top_k=int(top_k) if isinstance(top_k, (int, float)) else 6,
        min_score=float(min_score) if isinstance(min_score, (int, float)) else 0.35,
        relevance_floor=float(relevance_floor) if isinstance(relevance_floor, (int, float)) else 0.0,
        workspace_origin=_origin(arguments),
        chat_id=_chat_id(arguments),
    )
    if body.startswith("memory_search failed") or body.startswith("memory_search refused"):
        raise ToolError(body)
    return _truncate(body)


def _get(arguments: dict[str, Any]) -> str:
    from sharedagents_mcp.memory.engine import memory_tools as _mt

    file_path = arguments.get("file_path")
    if not isinstance(file_path, str) or not file_path:
        raise ToolError("file_path required")
    start_line = arguments.get("start_line")
    end_line = arguments.get("end_line")
    body = _mt.memory_get(
        file_path,
        start_line=int(start_line) if isinstance(start_line, (int, float)) else None,
        end_line=int(end_line) if isinstance(end_line, (int, float)) else None,
        workspace_origin=_origin(arguments),
        chat_id=_chat_id(arguments),
    )
    if body.startswith("memory_get failed") or body.startswith("memory_get refused"):
        raise ToolError(body)
    return _truncate(body)


def _session_search(arguments: dict[str, Any]) -> str:
    from sharedagents_session import session_index as _si

    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ToolError("query required")
    top_k = arguments.get("top_k")
    min_score = arguments.get("min_score")
    body = _si.session_search(
        query,
        top_k=int(top_k) if isinstance(top_k, (int, float)) else 6,
        min_score=float(min_score) if isinstance(min_score, (int, float)) else 0.35,
        workspace_origin=_origin(arguments),
        chat_id=_chat_id(arguments),
    )
    if body.startswith("session_search failed") or body.startswith("session_search refused"):
        raise ToolError(body)
    return _truncate(body)


async def _remember(arguments: dict[str, Any]) -> str:
    from sharedagents_mcp.memory.engine import memory_write as _mw

    action = arguments.get("action")
    if not isinstance(action, str) or action not in ("add", "replace", "remove"):
        raise ToolError("action ∈ {add, replace, remove} required")
    home_str = os.environ.get("SHAREDAGENTS_AGENT_HOME") or ""
    if not home_str:
        raise ToolError("SHAREDAGENTS_AGENT_HOME not set in env")
    content = arguments.get("content")
    old_text = arguments.get("old_text")
    category = arguments.get("category")
    cfg = _RememberCfgShim(
        name=os.environ.get("SHAREDAGENTS_AGENT_NAME") or "jarvis",
        home=Path(home_str),
        remember_synthesizer_model=os.environ.get("SHAREDAGENTS_REMEMBER_SYNTH_MODEL", ""),
        memory_flush_model=os.environ.get("SHAREDAGENTS_MEMORY_FLUSH_MODEL", ""),
    )
    result = await _mw.remember(
        action=action,
        content=content if isinstance(content, str) else "",
        old_text=old_text if isinstance(old_text, str) else "",
        category=category if isinstance(category, str) else "",
        cfg=cfg,
        workspace_origin=_origin(arguments),
        chat_id=_chat_id(arguments),
    )
    body = result.summary
    if result.error:
        body += f"\nerror: {result.error}"
    if not result.ok:
        raise ToolError(_truncate(body))
    return _truncate(body)


# ---------------------------------------------------------------------------
# Internal, dispatch-only ops (phase 2).
#
# These are NOT in ``TOOL_DEFS`` — the stdio scaffold's ``tools/call`` dispatches
# on ``TOOL_DISPATCH`` while ``tools/list`` returns only ``TOOL_DEFS``, so these
# ops are callable over the wire yet INVISIBLE to the model (the model's tool
# grant is unchanged). Only trusted first-party code — via
# ``sharedagents_mcp.memory.client`` — reaches them, so they MAY take per-call chat
# context as arguments (the model cannot forge it here because it cannot invoke
# them). They return JSON strings; the client decodes.
# ---------------------------------------------------------------------------


def _home() -> Path:
    home = os.environ.get("SHAREDAGENTS_AGENT_HOME") or ""
    if not home:
        raise ToolError("SHAREDAGENTS_AGENT_HOME not set in env")
    return Path(home)


def _embedder_info(arguments: dict[str, Any]) -> str:
    """Embedder identity (for the client's centroid-cache key), or ok=False when
    no embedder is configured. Cheap — no embedding performed."""
    from sharedagents_mcp.memory.engine.memory_embed import resolve_embedder

    emb = resolve_embedder()
    if emb is None:
        return json.dumps({"ok": False})
    return json.dumps(
        {
            "ok": True,
            "provider": getattr(emb, "provider", ""),
            "provider_key": getattr(emb, "provider_key", ""),
            "model": getattr(emb, "model", ""),
            "dims": int(getattr(emb, "dims", 0) or 0),
        }
    )


async def _embed(arguments: dict[str, Any]) -> str:
    """Embed ``texts`` server-side using the WARM (module-memoized) embedder.
    ``mode`` = 'query' (per-text query embedding) or 'batch' (corpus). Returns
    ``{ok, vectors}``; ok=False (empty vectors) when no embedder is configured —
    the caller falls back to its non-semantic path (fail-safe)."""
    from sharedagents_mcp.memory.engine.memory_embed import resolve_embedder

    emb = resolve_embedder()
    if emb is None:
        return json.dumps({"ok": False, "vectors": []})
    texts = arguments.get("texts")
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        raise ToolError("texts: list[str] required")
    mode = arguments.get("mode") or "batch"
    if mode == "query":
        vectors = [await emb.embed_query(t) for t in texts]
    else:
        vectors = await emb.embed_batch(texts)
    return json.dumps({"ok": True, "vectors": vectors})


def _provider_status(arguments: dict[str, Any]) -> str:
    """Whether a memory provider is active for this home, and its name."""
    from sharedagents_mcp.memory.engine.memory_manager import get_active_provider

    prov = get_active_provider(_home())
    if prov is None:
        return json.dumps({"ok": True, "active": False, "name": ""})
    return json.dumps({"ok": True, "active": True, "name": getattr(prov, "name", "")})


async def _observe_turn(arguments: dict[str, Any]) -> str:
    """Best-effort provider turn-observation hook. Forwards a finished turn to
    the active memory provider's ``observe_turn`` if one is subscribed for this
    home. Today no concrete provider is registered (the base class is abstract),
    so this is a no-op — but keeping the seam server-side is what lets the agent
    reach it over the wire instead of importing the engine."""
    from sharedagents_mcp.memory.engine.memory_manager import get_active_provider

    prov = get_active_provider(_home())
    if prov is None or not hasattr(prov, "observe_turn"):
        return json.dumps({"ok": True, "observed": False})
    res = prov.observe_turn(
        arguments.get("cfg"),
        arguments.get("session"),
        arguments.get("turn"),
    )
    if hasattr(res, "__await__"):
        await res
    return json.dumps({"ok": True, "observed": True})


async def _prefetch(arguments: dict[str, Any]) -> str:
    """Per-turn auto-recall block for ``query`` (or '' when the privacy gate
    refuses / no provider / recall-mode tools / nothing recalled). ``chat_id`` /
    ``workspace_origin`` are per-CALL (trusted caller supplies the turn's chat)."""
    from sharedagents_mcp.memory.engine.memory_manager import prefetch_for_query

    query = arguments.get("query")
    if not isinstance(query, str):
        raise ToolError("query required")
    chat_id = arguments.get("chat_id")
    workspace_origin = arguments.get("workspace_origin")
    block = await prefetch_for_query(
        home=_home(),
        query=query,
        workspace_origin=workspace_origin if isinstance(workspace_origin, str) else _origin(),
        chat_id=chat_id if isinstance(chat_id, str) else _chat_id(),
    )
    return json.dumps({"ok": True, "block": block or ""})


def _index_rebuild(arguments: dict[str, Any]) -> str:
    """Best-effort rebuild of the local memory embedding index (index_all).
    When ``if_needed`` is set (the heartbeat's periodic tick), skips entirely
    unless ``needs_reindex`` — preserving the cheap no-op on unchanged corpora."""
    from sharedagents_mcp.memory.engine.memory_embed import resolve_embedder
    from sharedagents_mcp.memory.engine.memory_index import MemoryIndex

    try:
        embedder = resolve_embedder()
    except Exception:  # noqa: BLE001
        embedder = None
    idx = MemoryIndex(_home(), embedder=embedder)
    if arguments.get("if_needed") and not idx.needs_reindex():
        return json.dumps({"ok": True, "skipped": True})
    stats = idx.index_all()
    return json.dumps({"ok": True, "stats": stats}, default=str)


async def _subscribe(arguments: dict[str, Any]) -> str:
    """Activate/refresh this home's memory provider server-side (idempotent).
    Puts the provider INSIDE the memory server, where prefetch/observe_turn/
    provider_status already look. No concrete provider is registered today, so
    this is a no-op by default. ``memory_provider`` / ``memory_recall_mode`` are
    the only cfg fields the engine reads."""
    from sharedagents_mcp.memory.engine.memory_manager import subscribe_for_agent

    class _Cfg:
        home = _home()
        memory_provider = arguments.get("memory_provider", "")
        memory_recall_mode = arguments.get("memory_recall_mode", "")

    prov = await subscribe_for_agent(
        home=_home(),
        cfg=_Cfg(),
        workspace_origin=arguments.get("workspace_origin") or _origin(),
    )
    return json.dumps({"ok": True, "active": prov is not None})


def _index_search(arguments: dict[str, Any]) -> str:
    """Hybrid search returning structured hits (for the TUI memory browser)."""
    from sharedagents_mcp.memory.engine.memory_embed import resolve_embedder
    from sharedagents_mcp.memory.engine.memory_index import MemoryIndex

    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ToolError("query required")
    top_k = arguments.get("top_k")
    min_score = arguments.get("min_score")
    try:
        embedder = resolve_embedder()
    except Exception:  # noqa: BLE001
        embedder = None
    idx = MemoryIndex(_home(), embedder=embedder)
    hits = idx.search(
        query,
        top_k=int(top_k) if isinstance(top_k, (int, float)) else 6,
        min_score=float(min_score) if isinstance(min_score, (int, float)) else 0.0,
    )
    out = [
        {
            "id": h.id,
            "path": h.path,
            "source": h.source,
            "start_line": h.start_line,
            "end_line": h.end_line,
            "score": h.score,
            "vector_score": h.vector_score,
            "text_score": h.text_score,
            "snippet": getattr(h, "snippet", ""),
        }
        for h in hits
    ]
    return json.dumps({"ok": True, "hits": out}, default=str)


async def _synth(arguments: dict[str, Any]) -> str:
    """Synthesize MEMORY.md from recent dailies. Upstream endpoint/model come
    from the trusted caller (per-call) or env fallback."""
    from sharedagents_mcp.memory.engine.memory_synth import synth_memory

    endpoint = arguments.get("upstream_endpoint") or os.environ.get("SHAREDAGENTS_UPSTREAM_ENDPOINT", "")
    model = arguments.get("upstream_model") or os.environ.get("SHAREDAGENTS_UPSTREAM_MODEL", "")
    force = bool(arguments.get("force", False))
    stats = await synth_memory(
        _home(),
        upstream_endpoint=str(endpoint),
        upstream_model=str(model),
        force=force,
    )
    return json.dumps({"ok": True, "stats": stats}, default=str)


TOOL_DEFS: list[dict[str, Any]] = [
    {
        "name": "search",
        "description": (
            "Hybrid semantic search (BM25 + vector) over MEMORY.md and daily logs "
            "(memory/*.md). Use when you need to recall past context, decisions, "
            "lessons, or project state. Returns ranked chunks with file+line references. "
            "The index is refreshed incrementally by the heartbeat; stale for a few "
            "seconds at most."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Max results to return (1-15, default 6).",
                },
                "min_score": {
                    "type": "number",
                    "description": (
                        "Absolute floor on the fused score (0-1, default 0.25). "
                        "Lower to widen recall on a small corpus."
                    ),
                },
            },
            "required": ["query"],
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "get",
        "description": (
            "Read a memory file (or line range) by path. Use after a search "
            "narrows down the location, or to read MEMORY.md / a specific daily "
            "log directly. Path is relative to the agent's home directory; "
            "absolute paths and '..' traversal are refused."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": ("Memory file path, e.g. 'MEMORY.md' or 'memory/2026-04-27.md'."),
                },
                "start_line": {
                    "type": "integer",
                    "description": "Optional 1-based start line.",
                },
                "end_line": {
                    "type": "integer",
                    "description": "Optional 1-based end line (inclusive).",
                },
            },
            "required": ["file_path"],
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "session_search",
        "description": (
            "Search past saved sessions for matching turns using FTS5. Use when "
            "the user references a previous conversation ('did we discuss…', "
            "'remind me what I said about…') or when you need to recover context "
            "from a prior session. Returns ranked hits with session title, date, "
            "turn index, and a snippet. Saved transcripts are shared across all "
            "native agents in this single-owner runtime. Empty result is normal "
            "for fresh agents."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Max results to return (1-20, default 6).",
                },
                "min_score": {
                    "type": "number",
                    "description": (
                        "Absolute floor on the BM25 score (0-1, default 0.35). Lower to widen recall on a small corpus."
                    ),
                },
            },
            "required": ["query"],
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "remember",
        "description": (
            "Write to your persistent memory. Use to save durable context you'll "
            "need across sessions — decisions, lessons, project state, user "
            "preferences. Writes to daily logs (memory/YYYY-MM-DD.md); the "
            "background MEMORY.md synthesizer curates durable principles from "
            "these dailies. Actions: 'add' (append a bullet), 'replace' (swap "
            "matching text), 'remove' (delete matching text). Category is "
            "optional (ProjectState / Decision / Lesson / Context / UserPreference)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "replace", "remove"],
                    "description": "add (append bullet), replace (swap old_text → content), remove (delete matching old_text).",
                },
                "content": {
                    "type": "string",
                    "description": "The text to add or the replacement text (for add/replace).",
                },
                "old_text": {
                    "type": "string",
                    "description": "The exact text to replace/remove (for replace/remove).",
                },
                "category": {
                    "type": "string",
                    "description": "Optional category label (ProjectState, Decision, Lesson, Context, UserPreference).",
                },
            },
            "required": ["action"],
        },
    },
]

# Model-visible tools (also in TOOL_DEFS → tools/list) + internal dispatch-only
# ops (NOT in TOOL_DEFS → invisible to the model; trusted first-party code only).
TOOL_DISPATCH = {
    "search": _search,
    "get": _get,
    "session_search": _session_search,
    "remember": _remember,
    # internal (dispatch-only):
    "embedder_info": _embedder_info,
    "embed": _embed,
    "provider_status": _provider_status,
    "subscribe": _subscribe,
    "observe_turn": _observe_turn,
    "prefetch": _prefetch,
    "index_rebuild": _index_rebuild,
    "index_search": _index_search,
    "synth": _synth,
}


if __name__ == "__main__":
    run_stdio_server("memory", "1.0.0", TOOL_DEFS, TOOL_DISPATCH)
