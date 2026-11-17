"""Persistent MCP **wire** client for the builtin memory capability.

Wire-only; bounded, never blocks a turn.

This module is the ONLY sanctioned way for first-party packages
(``sharedagents_core`` / ``sharedagents_agent`` / ``sharedagents_tui`` / ``sharedagents_telegram``)
to reach memory: every operation crosses the ``mcp__memory__*`` JSON-RPC wire to
``python -m sharedagents_mcp.memory.server``. It imports **no** memory-engine symbol
(``sharedagents_mcp.memory.engine.*``) — that boundary is what keeps memory wire-only and
what ``tests/test_mcp021_memory_wire_boundary.py`` enforces.

Design (bounded, never blocks a turn):

* **One warm connection.** A single dedicated *owner task* enters the
  ``stdio_client`` + ``ClientSession`` contexts ONCE and services every call
  from a request queue. The MCP SDK's anyio transport requires enter/call/exit
  to happen in the same task; the owner-task model guarantees that while still
  letting many callers (in the same loop) await results concurrently.
* **No cold-spawn on the hot path.** Callers only enqueue a request and await a
  future under a deadline. If the connection is down, the owner task reconnects
  with bounded backoff in the background while the caller's deadline elapses and
  the fail-safe default is returned — a turn is never blocked on a spawn.
* **Hard deadline + fail-safe.** Hot-path methods (``prefetch``, ``embed``,
  search/get) return an empty/None default on timeout or any transport error
  rather than raising; user-invoked tools surface a bounded explicit error.
* **Instrumented.** ``spawn_count`` counts real server spawns so a persistence
  test can assert warm calls reuse the same child (zero new spawns).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sharedagents_mcp.mcp_client import MCPServerConfig, _flatten_content, _mcp_attr, mcp_session

# Hot-path request deadline (every hot-path request has a 250 ms
# maximum, covering the complete request/response wait).
HOT_DEADLINE_S: float = float(os.environ.get("SHAREDAGENTS_MEMORY_WIRE_DEADLINE_S", "0.25"))
# A non-hot op (index rebuild, synth, embed-batch of a large corpus) may take
# longer; it is still bounded, just not to the hot deadline.
SLOW_DEADLINE_S: float = float(os.environ.get("SHAREDAGENTS_MEMORY_WIRE_SLOW_DEADLINE_S", "30.0"))
# Corpus batch-embed (routing centroids / skill index) is a ONE-TIME cached cost
# on the turn path — bounded to a few seconds so a stalled wire fails safe
# quickly, but longer than the 250 ms hot deadline a real endpoint batch needs.
EMBED_BATCH_DEADLINE_S: float = float(os.environ.get("SHAREDAGENTS_MEMORY_WIRE_EMBED_DEADLINE_S", "4.0"))
# Bounded reconnect backoff schedule (seconds). Never a cold spawn on the hot
# path — this runs only inside the owner task.
_BACKOFF_S = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)


def _mark_tool_schema_known(sess: Any, name: str) -> None:
    """Stop the MCP SDK re-listing tools before each internal-op call.

    The memory wire's internal ops (``observe_turn``, ``prefetch``,
    ``embed``, ...) are dispatch-only by design: they are absent from
    ``tools/list`` and declare no output schema. mcp>=2 treats an unlisted
    tool as a stale cache and runs a full ``list_tools`` round trip (then
    logs a warning) before every such call, which doubles hot-path latency.
    Recording "no output schema" up front keeps each call one round trip.
    """
    schemas = getattr(sess, "_tool_output_schemas", None)
    if isinstance(schemas, dict):
        schemas.setdefault(name, None)


class MemoryWireError(RuntimeError):
    """A user-invoked memory tool failed over the wire (bounded, explicit).

    Hot-path *auto* operations never raise this — they fail safe to an empty
    default. It is raised only for model-/user-invoked tools where a silent
    empty result would be wrong.
    """


_SHUTDOWN = object()


@dataclass
class _Req:
    name: str
    arguments: dict[str, Any]
    deadline_s: float
    fut: asyncio.Future[tuple[bool, str]]
    cancelled: bool = False  # set when the caller timed out; _serve skips it
    cancel_event: asyncio.Event | None = None  # set on caller timeout to abort in-flight


class MemoryWireClient:
    """A persistent wire client bound to one memory server (one agent home)."""

    def __init__(self, server: MCPServerConfig) -> None:
        self._server = server
        self._q: asyncio.Queue[_Req | object] = asyncio.Queue()
        self._owner: asyncio.Task | None = None
        self._ready = asyncio.Event()
        self._closing = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self.spawn_count = 0  # real server spawns (persistence-test instrument)

    # -- lifecycle ------------------------------------------------------------

    def _ensure_owner(self) -> None:
        """Start the owner task lazily on the calling loop. Idempotent."""
        loop = asyncio.get_event_loop()
        if self._owner is not None and self._loop is loop and not self._owner.done():
            return
        # (Re)bind to this loop — a fresh loop (e.g. a new ``asyncio.run``) needs
        # its own owner task and queue.
        self._loop = loop
        self._q = asyncio.Queue()
        self._ready = asyncio.Event()
        self._closing = False
        self._owner = loop.create_task(self._run_owner())

    async def _run_owner(self) -> None:
        """Own the warm session; service the queue; reconnect with backoff."""
        attempt = 0
        while not self._closing:
            try:
                async with mcp_session(self._server, timeout_s=SLOW_DEADLINE_S) as sess:
                    self.spawn_count += 1
                    attempt = 0
                    self._ready.set()
                    await self._serve(sess)
                    return  # clean shutdown (_SHUTDOWN dequeued)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — any transport failure → reconnect
                self._ready.clear()
                if self._closing:
                    return
                await asyncio.sleep(_BACKOFF_S[min(attempt, len(_BACKOFF_S) - 1)])
                attempt += 1

    async def _serve(self, sess: Any) -> None:
        """Dispatch queued requests on the warm session until shutdown."""
        while not self._closing:
            item = await self._q.get()
            if item is _SHUTDOWN:
                return
            req: _Req = item  # type: ignore[assignment]
            if req.cancelled or req.fut.done():
                # The caller timed out and walked away BEFORE we dequeued — do
                # NOT run the stale call (that caused respawn/churn) and do NOT
                # treat it as a transport failure.
                continue
            _mark_tool_schema_known(sess, req.name)
            call_task = asyncio.ensure_future(
                sess.call_tool(req.name, arguments=req.arguments),
            )
            cancel_task = asyncio.ensure_future(req.cancel_event.wait()) if req.cancel_event else None
            waits = {call_task} | ({cancel_task} if cancel_task else set())
            try:
                done, _pending = await asyncio.wait(
                    waits,
                    timeout=req.deadline_s,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if call_task not in done:
                    # Caller cancelled OR the serve-side deadline elapsed: ABORT
                    # the in-flight call so its side effect does not execute after
                    # the caller gave up. Not a transport
                    # failure — do not reconnect.
                    call_task.cancel()
                    if cancel_task:
                        cancel_task.cancel()
                    if not req.cancelled and not req.fut.done():
                        req.fut.set_result((False, "memory wire timeout"))
                    continue
                if cancel_task:
                    cancel_task.cancel()
                result = call_task.result()
                is_error = bool(_mcp_attr(result, "isError", False))
                body = _flatten_content(getattr(result, "content", None))
                if not req.cancelled and not req.fut.done():
                    req.fut.set_result((not is_error, body))
            except Exception as exc:  # noqa: BLE001
                if cancel_task:
                    cancel_task.cancel()
                if req.cancelled:
                    # Caller already gone; a slow/errored stale call is not a
                    # transport failure — don't reconnect on its account.
                    continue
                # Propagate to the caller AND bubble to the owner so a broken
                # transport (EOF/child death) triggers reconnect.
                if not req.fut.done():
                    req.fut.set_exception(exc)
                raise

    async def aclose(self) -> None:
        self._closing = True
        try:
            self._q.put_nowait(_SHUTDOWN)
        except Exception:  # noqa: BLE001
            pass
        if self._owner is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._owner), timeout=2.0)
            except Exception:  # noqa: BLE001
                self._owner.cancel()

    # -- call primitive -------------------------------------------------------

    async def _call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        deadline_s: float,
    ) -> tuple[bool, str]:
        """Enqueue a wire call and await its result under ``deadline_s``.

        Returns ``(ok, body)``. On timeout / not-ready / transport error returns
        ``(False, <reason>)`` — the caller decides fail-safe vs. explicit error.
        Never spawns synchronously on this path.
        """
        self._ensure_owner()
        fut: asyncio.Future[tuple[bool, str]] = self._loop.create_future()  # type: ignore[union-attr]
        req = _Req(name, arguments, deadline_s, fut, cancel_event=asyncio.Event())
        self._q.put_nowait(req)
        try:
            return await asyncio.wait_for(asyncio.shield(fut), timeout=deadline_s)
        except asyncio.TimeoutError:
            # Mark the request dead so _serve skips it if still queued, and signal
            # cancel_event so _serve stops WAITING on an in-flight call — not a
            # transport failure, so no respawn on its account.
            self._mark_cancelled(req, fut)
            return (False, "memory wire timeout")
        except asyncio.CancelledError:
            # The CALLER was cancelled. CancelledError is BaseException, so the
            # broad ``except Exception`` below never runs — mark the request dead
            # here too so a queued call is skipped and we stop waiting on an
            # in-flight one, then re-raise to honour cancellation.
            self._mark_cancelled(req, fut)
            raise
        except Exception as exc:  # noqa: BLE001
            self._mark_cancelled(req, fut)
            return (False, f"memory wire error: {exc}")

    @staticmethod
    def _mark_cancelled(req: _Req, fut: asyncio.Future) -> None:
        # NOTE: the stdio server (sharedagents_mcp/_stdio_server.py) runs handlers
        # synchronously to completion and implements NO cancellation protocol, so
        # a request already being processed by the server WILL finish there — for
        # READS this is harmless (the result is discarded), and MUTATIONS
        # (``remember``) use the 30s slow deadline so a caller timeout mid-write
        # is rare and, when it happens, applying the requested write is
        # acceptable. What we CAN and DO guarantee: a still-QUEUED request is
        # never started, and an in-flight one is no longer waited on / never
        # respawns the session.
        req.cancelled = True
        if req.cancel_event is not None:
            req.cancel_event.set()
        if not fut.done():
            fut.cancel()

    # -- typed operations: existing model-visible tools -----------------------
    #
    # These four map to the tools the memory server already exposes in
    # ``tools/list`` (search/get/session_search/remember). Hot-path *auto*
    # reads fail safe to an empty string; the mutating ``remember`` and
    # user-invoked reads surface a bounded :class:`MemoryWireError`.

    async def search(
        self, query: str, *, top_k: int | None = None, min_score: float | None = None, hot: bool = True
    ) -> str:
        args: dict[str, Any] = {"query": query}
        if top_k is not None:
            args["top_k"] = top_k
        if min_score is not None:
            args["min_score"] = min_score
        ok, body = await self._call(
            "search",
            args,
            deadline_s=HOT_DEADLINE_S if hot else SLOW_DEADLINE_S,
        )
        return body if ok else ""

    async def get(self, file_path: str, *, start_line: int | None = None, end_line: int | None = None) -> str:
        args: dict[str, Any] = {"file_path": file_path}
        if start_line is not None:
            args["start_line"] = start_line
        if end_line is not None:
            args["end_line"] = end_line
        ok, body = await self._call("get", args, deadline_s=SLOW_DEADLINE_S)
        if not ok:
            raise MemoryWireError(body)
        return body

    async def session_search(self, query: str, *, top_k: int | None = None, min_score: float | None = None) -> str:
        args: dict[str, Any] = {"query": query}
        if top_k is not None:
            args["top_k"] = top_k
        if min_score is not None:
            args["min_score"] = min_score
        ok, body = await self._call("session_search", args, deadline_s=SLOW_DEADLINE_S)
        return body if ok else ""

    async def remember(
        self, *, action: str, content: str | None = None, old_text: str | None = None, category: str | None = None
    ) -> str:
        args: dict[str, Any] = {"action": action}
        for k, v in (("content", content), ("old_text", old_text), ("category", category)):
            if v is not None:
                args[k] = v
        ok, body = await self._call("remember", args, deadline_s=SLOW_DEADLINE_S)
        if not ok:
            raise MemoryWireError(body)
        return body

    # -- typed operations: internal ops (dispatch-only, invisible to model) ----

    def _decode(self, ok: bool, body: str) -> dict[str, Any]:
        if not ok:
            return {"ok": False}
        try:
            obj = json.loads(body)
            return obj if isinstance(obj, dict) else {"ok": False}
        except (ValueError, TypeError):
            return {"ok": False}

    async def embed(
        self, texts: list[str], *, mode: str = "batch", deadline_s: float | None = None
    ) -> list[list[float]]:
        # Turn-critical (routing + skill auto-invoke): a single query embed is
        # HOT-bounded (250 ms); a one-time corpus batch is bounded to a few
        # seconds. Never the 30 s slow deadline — a stalled wire must fail safe.
        if deadline_s is None:
            deadline_s = HOT_DEADLINE_S if mode == "query" else EMBED_BATCH_DEADLINE_S
        ok, body = await self._call("embed", {"texts": texts, "mode": mode}, deadline_s=deadline_s)
        obj = self._decode(ok, body)
        vecs = obj.get("vectors") if obj.get("ok") else None
        return vecs if isinstance(vecs, list) else []

    async def embedder(self) -> WireEmbedder | None:
        """A wire-backed embedder proxy, or None when no embedder is configured
        server-side (caller falls back to its non-semantic path)."""
        # HOT deadline — this is awaited on the turn path (routing/skill preamble).
        ok, body = await self._call("embedder_info", {}, deadline_s=HOT_DEADLINE_S)
        info = self._decode(ok, body)
        if not info.get("ok"):
            return None
        return WireEmbedder(
            self,
            provider_key=str(info.get("provider_key", "")),
            model=str(info.get("model", "")),
            dims=int(info.get("dims", 0) or 0),
            provider=str(info.get("provider", "")),
        )

    async def provider_status(self) -> dict[str, Any]:
        ok, body = await self._call("provider_status", {}, deadline_s=HOT_DEADLINE_S)
        return self._decode(ok, body)

    async def observe_turn(self, *, cfg: Any = None, session: Any = None, turn: Any = None) -> dict[str, Any]:
        """Best-effort provider turn-observation over the wire (no-op when no
        provider is subscribed). Never raises on the caller's path."""
        ok, body = await self._call(
            "observe_turn",
            {"cfg": cfg, "session": session, "turn": turn},
            deadline_s=SLOW_DEADLINE_S,
        )
        return self._decode(ok, body)

    async def prefetch(self, query: str, *, chat_id: str = "", workspace_origin: str = "") -> str:
        """Per-turn auto-recall block (hot path, fail-safe to '')."""
        ok, body = await self._call(
            "prefetch",
            {"query": query, "chat_id": chat_id, "workspace_origin": workspace_origin},
            deadline_s=HOT_DEADLINE_S,
        )
        obj = self._decode(ok, body)
        block = obj.get("block") if obj.get("ok") else ""
        return block if isinstance(block, str) else ""

    async def index_rebuild(self, *, if_needed: bool = False) -> dict[str, Any]:
        ok, body = await self._call(
            "index_rebuild",
            {"if_needed": if_needed},
            deadline_s=SLOW_DEADLINE_S,
        )
        return self._decode(ok, body)

    async def subscribe(
        self, *, memory_provider: str = "", memory_recall_mode: str = "", workspace_origin: str = ""
    ) -> dict[str, Any]:
        """Activate this home's memory provider server-side (idempotent; no-op by
        default). Used by the heartbeat so the provider lives in the server."""
        ok, body = await self._call(
            "subscribe",
            {
                "memory_provider": memory_provider,
                "memory_recall_mode": memory_recall_mode,
                "workspace_origin": workspace_origin,
            },
            deadline_s=SLOW_DEADLINE_S,
        )
        return self._decode(ok, body)

    async def index_search(
        self, query: str, *, top_k: int | None = None, min_score: float | None = None
    ) -> list[dict[str, Any]]:
        args: dict[str, Any] = {"query": query}
        if top_k is not None:
            args["top_k"] = top_k
        if min_score is not None:
            args["min_score"] = min_score
        ok, body = await self._call("index_search", args, deadline_s=SLOW_DEADLINE_S)
        obj = self._decode(ok, body)
        hits = obj.get("hits") if obj.get("ok") else None
        return hits if isinstance(hits, list) else []

    async def synth(
        self, *, upstream_endpoint: str = "", upstream_model: str = "", force: bool = False
    ) -> dict[str, Any]:
        ok, body = await self._call(
            "synth",
            {"upstream_endpoint": upstream_endpoint, "upstream_model": upstream_model, "force": force},
            deadline_s=SLOW_DEADLINE_S,
        )
        return self._decode(ok, body)


class WireEmbedder:
    """A drop-in replacement for the in-process embedder object, backed by the
    memory server's dispatch-only ``embed`` op. Exposes the same surface the
    callers read: ``provider_key`` / ``model`` / ``dims`` (sync attrs, for the
    centroid-cache key) and async ``embed_query`` / ``embed_batch``."""

    def __init__(
        self, client: MemoryWireClient, *, provider_key: str, model: str, dims: int, provider: str = ""
    ) -> None:
        self._client = client
        self.provider_key = provider_key
        self.model = model
        self.dims = dims
        self.provider = provider

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return await self._client.embed(texts, mode="batch")

    async def embed_query(self, text: str) -> list[float]:
        vecs = await self._client.embed([text], mode="query")
        return vecs[0] if vecs else []


# ---- server config + process-wide singleton --------------------------------

# Env var carrying the agent home into the spawned memory server. The engine's
# ``_resolve_agent_home`` reads ``SHAREDAGENTS_AGENT_HOME`` first, so the wire path
# resolves the same home the in-process path did and the privacy/path gates are
# unchanged. (Per-CHAT context — chat_id / workspace_origin — is per-call, not
# per-connection; it must be threaded as call arguments, not spawn env, once a
# single warm per-home connection serves multiple chats.)
_HOME_ENV = "SHAREDAGENTS_AGENT_HOME"


def memory_server_config(home: str | os.PathLike[str] | None = None) -> MCPServerConfig:
    """Build the :class:`MCPServerConfig` for ``python -m sharedagents_mcp.memory.server``
    bound to ``home`` (the agent's memory root). Same interpreter, same
    home-resolution env the in-process path used, so the wire path is
    behavior-identical."""
    env: dict[str, str] = {}
    if home is not None:
        env[_HOME_ENV] = str(Path(home))
    # Guarantee the child imports the SAME sharedagents_mcp as us, even when this
    # package is run from a source tree that isn't pip-installed on the child's
    # default path (tests, dev checkouts). Harmless when it IS installed — the
    # installed copy still wins for every OTHER package, and this path resolves
    # to the very package we were imported from. Without this the child dies
    # with ModuleNotFoundError, and because every hot-path op fails safe the
    # memory subsystem degrades SILENTLY to empty recall.
    #
    # Same idiom as the other first-party ``sys.executable -m sharedagents_*`` spawn
    # sites (sharedagents_gateway.worker_request, sharedagents_agent.agents.delegate_shims).
    # NOTE: mcp_session spawns with ``{**os.environ, **server.env}`` — server.env
    # OVERRIDES the inherited environment, so fold any inherited PYTHONPATH into
    # the value here rather than relying on the child to inherit it.
    package_parent = str(Path(__file__).resolve().parents[2])
    inherited = os.environ.get("PYTHONPATH", "")
    env["PYTHONPATH"] = package_parent + (os.pathsep + inherited if inherited else "")
    return MCPServerConfig(
        name="memory",
        transport="stdio",
        command=sys.executable,
        args=("-m", "sharedagents_mcp.memory.server"),
        env=env,
    )


# Warm clients keyed by (home, event-loop id). A single MemoryWireClient owns an
# owner task bound to ONE loop; a process may run several loops (the gateway's
# async loop for async callers, plus the background bridge loop below for sync
# callers), so each (home, loop) gets its own warm client + server — no thrashing
# rebind between loops.
_SINGLETON: dict[tuple[str, int], MemoryWireClient] = {}
_SINGLETON_LOCK = threading.Lock()


def get_memory_client(home: str | os.PathLike[str] | None = None) -> MemoryWireClient:
    """Warm client for ``home`` on the CURRENT event loop (spawned once per
    (home, loop), reused). Callers in first-party packages use this instead of
    importing the engine. Thread-safe: async callers and the sync-bridge thread
    may both create clients concurrently."""
    try:
        loop_id = id(asyncio.get_running_loop())
    except RuntimeError:
        loop_id = 0
    key = (str(Path(home)) if home is not None else "", loop_id)
    with _SINGLETON_LOCK:
        client = _SINGLETON.get(key)
        if client is None:
            client = MemoryWireClient(memory_server_config(home))
            _SINGLETON[key] = client
        return client


# ---- sync bridge: a persistent background loop for SYNC callers -------------
#
# agent.py's recall/index/observe paths are sync methods. Calling ``asyncio.run``
# per invocation would spawn a fresh memory server EVERY turn (the wire client must
# stay persistent). Instead we run one daemon thread with a
# long-lived event loop; sync callers submit coroutines to it, so the per-home
# ``MemoryWireClient`` bound to that loop stays warm (one server, reused).


class _BackgroundLoop:
    """A daemon thread owning a persistent asyncio loop for sync callers."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread = None
        self._lock = threading.Lock()

    def _ensure(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is not None and self._thread is not None and self._thread.is_alive():
                return self._loop
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._loop.run_forever,
                name="sharedagents-memory-wire",
                daemon=True,
            )
            self._thread.start()
            return self._loop

    def run(self, coro, *, timeout: float):
        """Submit ``coro`` to the persistent loop and block up to ``timeout``."""
        loop = self._ensure()
        fut = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            return fut.result(timeout=timeout)
        except Exception:  # noqa: BLE001 — includes concurrent.futures.TimeoutError
            fut.cancel()
            raise


_BG = _BackgroundLoop()


def prefetch_sync(
    home: str | os.PathLike[str] | None, query: str, *, chat_id: str = "", workspace_origin: str = ""
) -> str:
    """Sync per-turn recall over the WARM wire (persistent background loop —
    NOT a per-call server spawn). Fail-safe to '' on any error/timeout."""
    try:
        return (
            _BG.run(
                get_memory_client(home).prefetch(
                    query,
                    chat_id=chat_id,
                    workspace_origin=workspace_origin,
                ),
                timeout=HOT_DEADLINE_S + 0.25,
            )
            or ""
        )
    except Exception:  # noqa: BLE001
        return ""


def index_rebuild_sync(home: str | os.PathLike[str] | None, *, if_needed: bool = False) -> dict[str, Any]:
    """Sync best-effort index rebuild over the warm wire."""
    try:
        return _BG.run(
            get_memory_client(home).index_rebuild(if_needed=if_needed),
            timeout=SLOW_DEADLINE_S,
        )
    except Exception:  # noqa: BLE001
        return {"ok": False}


def observe_turn_sync(home: str | os.PathLike[str] | None, *, turn: Any = None) -> None:
    """Sync best-effort provider turn-observation over the warm wire (no-op by
    default). Never raises."""
    try:
        _BG.run(get_memory_client(home).observe_turn(turn=turn), timeout=SLOW_DEADLINE_S)
    except Exception:  # noqa: BLE001
        pass


def subscribe_sync(
    home: str | os.PathLike[str] | None,
    *,
    memory_provider: str = "",
    memory_recall_mode: str = "",
    workspace_origin: str = "",
) -> dict[str, Any]:
    """Sync provider subscribe over the warm wire. Runs on the SAME background
    loop / server that ``prefetch_sync`` uses, so the provider it activates lives
    where recall looks (cross-loop registry unification)."""
    try:
        return _BG.run(
            get_memory_client(home).subscribe(
                memory_provider=memory_provider,
                memory_recall_mode=memory_recall_mode,
                workspace_origin=workspace_origin,
            ),
            timeout=SLOW_DEADLINE_S,
        )
    except Exception:  # noqa: BLE001
        return {"ok": False}


def synth_sync(
    home: str | os.PathLike[str] | None, *, upstream_endpoint: str = "", upstream_model: str = ""
) -> dict[str, Any]:
    """Sync memory synth over the warm wire (same background server)."""
    try:
        return _BG.run(
            get_memory_client(home).synth(
                upstream_endpoint=upstream_endpoint,
                upstream_model=upstream_model,
            ),
            timeout=SLOW_DEADLINE_S,
        )
    except Exception:  # noqa: BLE001
        return {"ok": False}
