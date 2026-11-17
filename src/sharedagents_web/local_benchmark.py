"""Local inference benchmarks for the Models → vLLM tab.

Measures prefill (TTFT) and decode throughput against the LLM replicas in
``vllm-router.yaml``. Two passes per run:

* ``0`` context — a minimal prompt, so the numbers are the model's baseline.
* one operator-selected context length (default 65536 tokens).

All selected replicas run the same pass concurrently; results stream to the
browser over SSE as each replica finishes. Only operator-configured replica
URLs are ever contacted — the client sends target ids, never URLs (no SSRF
surface). One run at a time, because overlapping runs would distort the
throughput they are meant to measure.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sharedagents_core import oscompat
from sharedagents_web.router_models import list_router_models

DEFAULT_CONTEXT_TOKENS = 65536
DEFAULT_MAX_OUTPUT_TOKENS = 256
MIN_OUTPUT_TOKENS = 16
MAX_CONTEXT_TOKENS = 262144
MAX_TARGETS = 32
REQUEST_TIMEOUT_SECONDS = 900

# English filler tokenizes at ~4.3 chars/token with the Qwen tokenizer; this
# lands the generated prompt near the requested context (61k actual at the old
# 4.0 rate). The exact count is always taken from the response usage.
_CHARS_PER_TOKEN = 4.3
_FILLER = (
    "The quick brown fox jumps over the lazy dog while benchmark probes "
    "measure prefill and decode throughput on local inference."
)
_INSTRUCTION = "Respond with one short sentence about the colour of the sky. Do not repeat or summarise the context."


class BenchmarkError(ValueError):
    """Invalid target, context, or run state."""


@dataclass
class BenchmarkTarget:
    id: str
    name: str
    url: str
    pool: str
    host: str
    model: str
    max_active: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "url": self.url,
            "pool": self.pool,
            "host": self.host,
            "model": self.model,
            "max_active": self.max_active,
        }


def _host(url: str) -> str:
    text = str(url or "")
    text = text.split("://", 1)[-1]
    return text.split("/", 1)[0]


def _pid_alive(pid: Any) -> bool:
    try:
        value = int(pid)
    except (TypeError, ValueError):
        return False
    if value <= 0:
        return False
    try:
        os.kill(value, 0)
    except OSError:
        return False
    return True


def list_targets(path=None) -> list[BenchmarkTarget]:
    """Every configured LLM replica, deduplicated by ``name@url``."""
    listed = list_router_models(path)
    seen: set[str] = set()
    targets: list[BenchmarkTarget] = []
    for pool in listed.get("models") or []:
        if pool.get("kind") != "llm":
            continue
        pool_model = str(pool.get("upstream_model") or "").strip()
        for replica in pool.get("replicas") or []:
            url = str(replica.get("url") or "").rstrip("/")
            name = str(replica.get("name") or "").strip()
            if not url or not name:
                continue
            key = f"{name}@{url}"
            if key in seen:
                continue
            seen.add(key)
            targets.append(
                BenchmarkTarget(
                    id=key,
                    name=name,
                    url=url,
                    pool=str(pool.get("id") or ""),
                    host=_host(url),
                    model=str(replica.get("upstream_model") or "").strip() or pool_model or str(pool.get("id") or ""),
                    max_active=int(replica.get("max_active") or 0),
                )
            )
    return targets


def build_prompt(context_tokens: int, marker: str = "") -> str:
    """Deterministic filler sized to roughly ``context_tokens`` tokens.

    Each block carries a counter and the run marker, so repeated text cannot
    be collapsed into one trivial token run or served wholly from a previous
    run's prefix cache.
    """
    if context_tokens <= 0:
        return _INSTRUCTION
    target_chars = max(1, int(context_tokens) * _CHARS_PER_TOKEN)
    head = f"Benchmark context {marker or 'run'}.\n\n"
    blocks: list[str] = []
    size = 0
    index = 0
    while size < target_chars:
        block = f"Section {index:07d}. {_FILLER}\n"
        blocks.append(block)
        size += len(block)
        index += 1
    return head + "".join(blocks) + "\n" + _INSTRUCTION


def _estimate_tokens(text: str) -> int:
    return max(1, len(str(text or "")) // _CHARS_PER_TOKEN)


def _usage_counts(payload: dict[str, Any] | None) -> tuple[int | None, int | None]:
    if not isinstance(payload, dict):
        return None, None
    try:
        prompt = int(payload.get("prompt_tokens"))
    except (TypeError, ValueError):
        prompt = None
    try:
        completion = int(payload.get("completion_tokens"))
    except (TypeError, ValueError):
        completion = None
    return prompt, completion


def _stream_text(obj: dict[str, Any]) -> str:
    """Reasoning or content text from one streamed chunk (never tool calls)."""
    for choice in obj.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if isinstance(delta, dict):
            text = delta.get("content") or delta.get("reasoning_content") or ""
            if isinstance(text, str) and text:
                return text
        message = choice.get("message")
        if isinstance(message, dict):
            text = message.get("content") or ""
            if isinstance(text, str) and text:
                return text
        text = choice.get("text")
        if isinstance(text, str) and text:
            return text
    return ""


def measure_endpoint(
    target: BenchmarkTarget,
    *,
    context_tokens: int,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    marker: str = "",
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """One streaming completion; prefill = time to first generated token.

    Returns a metrics dict with ``error`` set instead of raising, so one dead
    replica cannot abort the rest of the simultaneous run.
    """
    import httpx

    prompt = build_prompt(context_tokens, marker)
    payload = {
        "model": target.model or target.pool,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(max_output_tokens),
        "temperature": 0,
        "stream": True,
        # Force a full-length decode. Without this the model often stops after
        # 40-60 tokens, and vLLM coalesces chunks, so a handful of timestamps
        # over a ~30 ms window produced nonsense four-digit "tok/s" numbers.
        "ignore_eos": True,
        "stream_options": {"include_usage": True},
    }
    result: dict[str, Any] = {
        "target": target.id,
        "phase": int(context_tokens),
        "prompt_tokens": None,
        "completion_tokens": None,
        "prefill_ms": None,
        "prefill_tps": None,
        "gen_ms": None,
        "gen_tps": None,
        "ttft_ms": None,
        "total_ms": None,
        "error": "",
    }
    payload_without_options = {key: value for key, value in payload.items() if key != "stream_options"}
    payload_minimal = {key: value for key, value in payload_without_options.items() if key != "ignore_eos"}
    variants = (payload, payload_without_options, payload_minimal)
    response = None
    started = 0.0
    first_token: float | None = None
    last_token: float | None = None
    chunks = 0
    usage: dict[str, Any] | None = None
    try:
        # Some older OpenAI-compatible servers reject ``stream_options`` or
        # ``ignore_eos``; retry without them so the benchmark still reports.
        for attempt, body in enumerate(variants):
            started = time.perf_counter()
            first_token = last_token = None
            chunks = 0
            usage = None
            with httpx.stream(
                "POST",
                target.url + "/v1/chat/completions",
                json=body,
                timeout=httpx.Timeout(connect=10.0, read=timeout, write=60.0, pool=10.0),
            ) as response:
                if response.status_code >= 400:
                    detail = response.read().decode("utf-8", "replace")[:300]
                    if attempt < len(variants) - 1 and response.status_code == 400:
                        continue
                    result["error"] = f"HTTP {response.status_code}: {detail}"
                    return result
                for line in response.iter_lines():
                    raw = str(line or "").strip()
                    if not raw:
                        continue
                    data = raw[5:].strip() if raw.startswith("data:") else raw
                    if not data or data == "[DONE]":
                        continue
                    try:
                        obj = json.loads(data)
                    except ValueError:
                        continue
                    if isinstance(obj, dict):
                        if isinstance(obj.get("usage"), dict):
                            usage = obj["usage"]
                        if _stream_text(obj):
                            now = time.perf_counter()
                            if first_token is None:
                                first_token = now
                            last_token = now
                            chunks += 1
            break
        if response is not None and response.status_code >= 400:
            return result
    except Exception as exc:  # noqa: BLE001 — reported as data
        result["error"] = f"{type(exc).__name__}: {exc}"[:400]
        return result

    total = time.perf_counter()
    usage_prompt, usage_completion = _usage_counts(usage)
    prompt_tokens = usage_prompt if usage_prompt is not None else _estimate_tokens(prompt)
    completion_tokens = usage_completion if usage_completion is not None else max(0, chunks)
    result["prompt_tokens"] = prompt_tokens
    result["completion_tokens"] = completion_tokens
    result["total_ms"] = round((total - started) * 1000.0, 1)
    if first_token is not None:
        ttft = max(first_token - started, 0.000001)
        result["ttft_ms"] = round(ttft * 1000.0, 1)
        result["prefill_ms"] = result["ttft_ms"]
        result["prefill_tps"] = round(prompt_tokens / ttft, 1)
        decode_s = (last_token - first_token) if last_token is not None else 0.0
        result["gen_ms"] = round(decode_s * 1000.0, 1)
        # A handful of coalesced chunks over a few tens of milliseconds cannot
        # describe decode throughput; require a real observation window.
        if decode_s >= 0.05 and completion_tokens > 1:
            result["gen_tps"] = round(completion_tokens / decode_s, 1)
    return result


@dataclass
class RunStore:
    """Run snapshots + event logs on disk.

    gunicorn serves this app with multiple workers, so an in-process manager
    would hand the POST to one worker and the SSE reconnect to another. Files
    (atomic snapshot replace + append-only JSONL events) are shared by every
    worker; an ``flock`` on ``run.lock`` is the one-run-at-a-time guard and is
    released automatically if the owning worker dies.
    """

    def __init__(self, root: Path | None = None) -> None:
        if root is None:
            override = os.environ.get("SHAREDAGENTS_BENCHMARK_DIR")
            if override:
                root = Path(override)
            else:
                home = Path(os.environ.get("SHAREDAGENTS_HOME") or (Path.home() / ".sharedagents")).expanduser()
                root = home / "local-benchmarks"
        self.root = Path(root).expanduser()

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str, suffix: str) -> Path:
        return self.root / f"{run_id}{suffix}"

    def try_lock(self) -> int | None:
        self.ensure()
        fd = os.open(str(self.root / "run.lock"), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            locked = oscompat.try_file_lock(fd)
        except OSError:
            locked = False
        if not locked:
            os.close(fd)
            return None
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode("ascii"))
        return fd

    @staticmethod
    def unlock(fd: int | None) -> None:
        if fd is None:
            return
        try:
            oscompat.file_unlock(fd)
        finally:
            os.close(fd)

    def write_snapshot(self, run_id: str, snapshot: dict[str, Any]) -> None:
        self.ensure()
        target = self._path(run_id, ".json")
        tmp = self.root / f".{run_id}.{os.getpid()}.{threading.get_ident()}.tmp"
        tmp.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, target)

    def read_snapshot(self, run_id: str) -> dict[str, Any] | None:
        try:
            raw = self._path(run_id, ".json").read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        if data.get("status") == "running" and not _pid_alive(data.get("owner_pid")):
            # The worker that owned this run died mid-flight; present it as
            # failed instead of a run that looks live forever.
            data = {**data, "status": "error", "error": "benchmark worker exited"}
        return data

    def append_event(self, run_id: str, event: dict[str, Any]) -> None:
        self.ensure()
        with open(self._path(run_id, ".jsonl"), "ab") as handle:
            handle.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))
            handle.flush()

    def read_events(self, run_id: str, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
        try:
            with open(self._path(run_id, ".jsonl"), "rb") as handle:
                handle.seek(offset)
                chunk = handle.read()
        except OSError:
            return [], offset
        # Only consume complete lines: a poll can race an in-flight append and
        # must not drop (or half-parse) the partial tail.
        end = chunk.rfind(b"\n")
        if end < 0:
            return [], offset
        complete = chunk[: end + 1]
        events: list[dict[str, Any]] = []
        for line in complete.splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue
            if isinstance(data, dict):
                events.append(data)
        return events, offset + len(complete)

    def set_pointer(self, name: str, run_id: str) -> None:
        self.ensure()
        target = self.root / name
        tmp = self.root / f".{name}.{os.getpid()}.tmp"
        tmp.write_text(run_id, encoding="ascii")
        os.replace(tmp, target)

    def clear_pointer(self, name: str) -> None:
        try:
            (self.root / name).unlink()
        except OSError:
            pass

    def read_pointer(self, name: str) -> str | None:
        try:
            text = (self.root / name).read_text(encoding="ascii").strip()
        except OSError:
            return None
        return text or None

    def prune(self, keep: int = 12) -> None:
        try:
            files = sorted(
                self.root.glob("*.json"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            return
        for stale in files[keep:]:
            for suffix in (".json", ".jsonl"):
                try:
                    stale.with_suffix(suffix).unlink()
                except OSError:
                    pass


@dataclass
class BenchmarkRun:
    id: str
    context_tokens: int
    max_output_tokens: int
    targets: list[BenchmarkTarget]
    store: RunStore
    status: str = "running"
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def phases(self) -> list[int]:
        phases = [0]
        if self.context_tokens > 0:
            phases.append(int(self.context_tokens))
        return phases

    def publish(self, event: dict[str, Any]) -> None:
        with self._lock:
            self.events.append(event)
        self.store.append_event(self.id, event)
        self.store.write_snapshot(self.id, self.snapshot())

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "context_tokens": self.context_tokens,
            "max_output_tokens": self.max_output_tokens,
            "phases": self.phases,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "owner_pid": os.getpid(),
            "targets": [target.as_dict() for target in self.targets],
            "results": list(self.results.values()),
        }


class BenchmarkManager:
    """Starts one run and keeps it available for polling/SSE reconnects."""

    def __init__(self, store: RunStore | None = None) -> None:
        self._lock = threading.Lock()
        self.store = store or RunStore()
        self._current: BenchmarkRun | None = None
        self._last: BenchmarkRun | None = None
        self._lock_fd: int | None = None

    @property
    def current(self) -> BenchmarkRun | None:
        with self._lock:
            return self._current

    def snapshot(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            for run in (self._current, self._last):
                if run is not None and run.id == run_id:
                    return run.snapshot()
        return self.store.read_snapshot(run_id)

    def last_snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            run = self._current
        if run is not None:
            return run.snapshot()
        # A run may be owned by another gunicorn worker: its ``current``
        # pointer is the truth, even when this worker finished an older run.
        current_id = self.store.read_pointer("current")
        if current_id:
            snapshot = self.store.read_snapshot(current_id)
            if snapshot is not None and snapshot.get("status") == "running":
                return snapshot
        with self._lock:
            run = self._last
        if run is not None:
            return run.snapshot()
        for pointer in ("last", "current"):
            run_id = self.store.read_pointer(pointer)
            if run_id:
                snapshot = self.store.read_snapshot(run_id)
                if snapshot is not None:
                    return snapshot
        return None

    def start(
        self,
        *,
        target_ids: list[str] | None,
        context_tokens: int,
        max_output_tokens: int,
        path=None,
    ) -> BenchmarkRun:
        context_tokens = _validate_context(context_tokens)
        max_output_tokens = _validate_max_output(max_output_tokens)
        available = list_targets(path)
        by_id = {target.id: target for target in available}
        if not by_id:
            raise BenchmarkError("No local LLM replicas are configured")
        if target_ids:
            unknown = [tid for tid in target_ids if tid not in by_id]
            if unknown:
                raise BenchmarkError("Unknown benchmark target(s): " + ", ".join(sorted(unknown)[:5]))
            selected = [by_id[tid] for tid in target_ids]
        else:
            selected = list(by_id.values())
        if len(selected) > MAX_TARGETS:
            raise BenchmarkError(f"Select at most {MAX_TARGETS} replicas per run")
        with self._lock:
            if self._current is not None and self._current.status == "running":
                raise BenchmarkError("A benchmark run is already in progress")
            if self._lock_fd is None:
                self._lock_fd = self.store.try_lock()
            if self._lock_fd is None:
                raise BenchmarkError("A benchmark run is already in progress")
            run = BenchmarkRun(
                id=uuid.uuid4().hex[:12],
                context_tokens=context_tokens,
                max_output_tokens=max_output_tokens,
                targets=selected,
                store=self.store,
            )
            self._current = run
        self.store.prune()
        self.store.clear_pointer("current")
        self.store.set_pointer("current", run.id)
        run.publish(
            {
                "type": "start",
                "run": run.snapshot(),
            }
        )
        thread = threading.Thread(
            target=self._work,
            args=(run,),
            name="local-benchmark",
            daemon=True,
        )
        thread.start()
        return run

    def _work(self, run: BenchmarkRun) -> None:
        try:
            for phase in run.phases:
                run.publish({"type": "phase", "phase": phase})
                workers = max(1, min(len(run.targets), 16))
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {}
                    for target in run.targets:
                        run.publish(
                            {
                                "type": "target",
                                "target": target.id,
                                "phase": phase,
                                "status": "running",
                            }
                        )
                        futures[
                            pool.submit(
                                measure_endpoint,
                                target,
                                context_tokens=phase,
                                max_output_tokens=run.max_output_tokens,
                                marker=run.id,
                            )
                        ] = target
                    for future in as_completed(futures):
                        target = futures[future]
                        try:
                            metrics = future.result()
                        except Exception as exc:  # noqa: BLE001
                            metrics = {
                                "target": target.id,
                                "phase": phase,
                                "error": f"{type(exc).__name__}: {exc}"[:400],
                            }
                        key = f"{phase}:{target.id}"
                        run.results[key] = metrics
                        run.publish(
                            {
                                "type": "result",
                                "target": target.id,
                                "phase": phase,
                                "status": "error" if metrics.get("error") else "done",
                                "metrics": metrics,
                            }
                        )
            run.status = "done"
        except Exception as exc:  # noqa: BLE001 — surfaced to the UI
            run.status = "error"
            run.publish(
                {
                    "type": "error",
                    "error": f"{type(exc).__name__}: {exc}"[:400],
                }
            )
        finally:
            run.finished_at = time.time()
            run.publish({"type": "done", "run": run.snapshot()})
            self.store.set_pointer("last", run.id)
            self.store.clear_pointer("current")
            with self._lock:
                if self._current is run:
                    self._last = run
                    self._current = None
                fd, self._lock_fd = self._lock_fd, None
            self.store.unlock(fd)


def _validate_context(value: Any) -> int:
    try:
        tokens = int(value)
    except (TypeError, ValueError) as exc:
        raise BenchmarkError("context tokens must be an integer") from exc
    if tokens < 0:
        raise BenchmarkError("context tokens cannot be negative")
    if tokens > MAX_CONTEXT_TOKENS:
        raise BenchmarkError(f"context tokens cannot exceed {MAX_CONTEXT_TOKENS}")
    if 0 < tokens < 256:
        raise BenchmarkError("context tokens must be 0 or at least 256")
    return tokens


def _validate_max_output(value: Any) -> int:
    try:
        tokens = int(value)
    except (TypeError, ValueError) as exc:
        raise BenchmarkError("generation tokens must be an integer") from exc
    if tokens < MIN_OUTPUT_TOKENS or tokens > 2048:
        raise BenchmarkError(f"generation tokens must be between {MIN_OUTPUT_TOKENS} and 2048")
    return tokens


MANAGER = BenchmarkManager()
