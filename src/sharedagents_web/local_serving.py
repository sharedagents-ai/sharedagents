"""Live local-router / vLLM serving snapshot for the Models page.

Reads mux health that ``router_status_snapshot`` already gathers, then adds
process units and log tails so the vLLM pane can show running/waiting counts,
routed-request counters, and the operator-facing vLLM logs without a second
Prometheus scrape.
"""

from __future__ import annotations

import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sharedagents_core.layout import sharedagents_root
from sharedagents_web.router_status import model_category, router_status_snapshot

_LOG_TAIL_BYTES = 180_000
_LOG_TAIL_LINES = 250
_VLLM_UNITS = ("sharedagents-vllm.service",)
_JOURNAL_LINES = 180
_FINISHED_RE = re.compile(
    r"Finished request\s+(?P<id>\S+):\s+"
    r"(?:generated\s+)?(?P<tokens>\d+)\s+tokens"
    r"(?:[^\d]+(?P<secs>[\d.]+)\s*s)?"
    r"(?:[^\d]+(?P<tps>[\d.]+)\s*tok(?:ens)?/s)?",
    re.IGNORECASE,
)
_ENGINE_STATS_RE = re.compile(
    r"Avg prompt throughput:\s*(?P<prompt>[\d.]+)\s*tokens/s,\s*"
    r"Avg generation throughput:\s*(?P<gen>[\d.]+)\s*tokens/s,\s*"
    r"Running:\s*(?P<running>\d+)\s*reqs"
    r"(?:,\s*Swapped:\s*\d+\s*reqs)?"
    r",\s*Waiting:\s*(?P<waiting>\d+)\s*reqs,\s*"
    r"GPU KV cache usage:\s*(?P<cache>[\d.]+)%",
    re.IGNORECASE,
)
_LINE_STAMP_RE = re.compile(r"(?P<stamp>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def vllm_server_log_path() -> Path:
    env = os.environ.get("SHAREDAGENTS_VLLM_LOG")
    if env:
        return Path(env).expanduser()
    return sharedagents_root() / "vllm-server.log"


def tail_text(path: Path, *, max_bytes: int = _LOG_TAIL_BYTES, max_lines: int = _LOG_TAIL_LINES) -> str:
    """Last ``max_lines`` of a file, capped at ``max_bytes``."""
    try:
        size = path.stat().st_size
    except OSError:
        return ""
    start = max(0, size - max_bytes)
    try:
        with path.open("rb") as handle:
            handle.seek(start)
            blob = handle.read()
    except OSError:
        return ""
    text = blob.decode("utf-8", errors="replace")
    if start > 0:
        nl = text.find("\n")
        if nl >= 0:
            text = text[nl + 1 :]
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    return "\n".join(lines)


def _journal_tail(unit: str) -> str:
    try:
        proc = subprocess.run(
            [
                "journalctl",
                "--user",
                "-u",
                unit,
                "-n",
                str(_JOURNAL_LINES),
                "--no-pager",
                "-o",
                "short-iso",
            ],
            capture_output=True,
            text=True,
            timeout=2.5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired, subprocess.SubprocessError):
        return ""
    if proc.returncode != 0:
        return (proc.stderr or proc.stdout or "").strip()
    return (proc.stdout or "").strip()


def _unit_status(unit: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "unit": unit,
        "active": "unknown",
        "sub": "",
        "pid": None,
        "available": False,
        "error": None,
    }
    try:
        proc = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                unit,
                "--property=LoadState,ActiveState,SubState,MainPID,NRestarts,FragmentPath",
                "--no-pager",
            ],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired, subprocess.SubprocessError) as exc:
        row["error"] = str(exc) or exc.__class__.__name__
        return row
    if proc.returncode != 0:
        row["error"] = (proc.stderr or proc.stdout or "systemctl show failed").strip()
        return row
    parsed: dict[str, str] = {}
    for line in (proc.stdout or "").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key.strip()] = value.strip()
    if parsed.get("LoadState") == "not-found":
        row["error"] = "unit not installed"
        return row
    row["available"] = True
    row["active"] = parsed.get("ActiveState") or "unknown"
    row["sub"] = parsed.get("SubState") or ""
    try:
        pid = int(parsed.get("MainPID") or 0)
    except (TypeError, ValueError):
        pid = 0
    row["pid"] = pid or None
    try:
        row["restarts"] = int(parsed.get("NRestarts") or 0)
    except (TypeError, ValueError):
        row["restarts"] = None
    row["path"] = parsed.get("FragmentPath") or ""
    return row


def _line_stamp(line: str) -> str | None:
    match = _LINE_STAMP_RE.search(line)
    return match.group("stamp") if match else None


def _float_or_none(raw: str | None) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def parse_vllm_recent_turns(
    text: str,
    *,
    max_finished: int = 20,
    max_stats: int = 8,
) -> dict[str, list[dict[str, Any]]]:
    """Pull last finished requests and engine-stat snapshots out of a log tail."""
    finished: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []
    for line in str(text or "").splitlines():
        fin = _FINISHED_RE.search(line)
        if fin:
            finished.append(
                {
                    "kind": "finished",
                    "id": (fin.group("id") or "").rstrip(".,"),
                    "tokens": int(fin.group("tokens")),
                    "seconds": _float_or_none(fin.group("secs")),
                    "tokens_per_sec": _float_or_none(fin.group("tps")),
                    "stamp": _line_stamp(line),
                    "line": line.strip(),
                }
            )
            continue
        eng = _ENGINE_STATS_RE.search(line)
        if not eng:
            continue
        stats.append(
            {
                "kind": "engine",
                "prompt_tokens_per_sec": _float_or_none(eng.group("prompt")),
                "tokens_per_sec": _float_or_none(eng.group("gen")),
                "running": int(eng.group("running")),
                "waiting": int(eng.group("waiting")),
                "gpu_cache_pct": _float_or_none(eng.group("cache")),
                "stamp": _line_stamp(line),
                "line": line.strip(),
            }
        )
    return {
        "finished": finished[-max_finished:],
        "engine_stats": stats[-max_stats:],
    }


def collect_vllm_logs() -> list[dict[str, Any]]:
    """File tail + journal for the local vLLM unit. Missing sources stay empty."""
    rows: list[dict[str, Any]] = []
    path = vllm_server_log_path()
    mtime = None
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except OSError:
        mtime = None
    rows.append(
        {
            "id": "vllm-server",
            "title": "vLLM server log",
            "path": str(path),
            "mtime": mtime,
            "text": tail_text(path) if path.is_file() else "",
            "missing": not path.is_file(),
        }
    )
    for unit in _VLLM_UNITS:
        rows.append(
            {
                "id": "journal-" + unit,
                "title": unit,
                "path": "journalctl --user -u " + unit,
                "mtime": None,
                "text": _journal_tail(unit),
                "missing": False,
            }
        )
    return rows


def collect_vllm_processes() -> list[dict[str, Any]]:
    return [_unit_status(unit) for unit in _VLLM_UNITS]


def is_llm_kind(kind: str = "", name: str = "") -> bool:
    """True for chat/completion pools; false for decision/embed/rerank/asr/tts."""
    return model_category(kind, name) == "llm"


def _mux_calls_for_pool(routes: list[dict[str, Any]], pool: str) -> float:
    needle = str(pool or "").strip().lower()
    if not needle:
        return 0.0
    total = 0.0
    for row in routes:
        if not isinstance(row, dict):
            continue
        upstream = str(row.get("upstream") or "").lower()
        if "route:" not in upstream:
            continue
        if needle not in upstream:
            continue
        try:
            total += float(row.get("value") or 0)
        except (TypeError, ValueError):
            continue
    return total


def group_other_models(
    replicas: list[dict[str, Any]],
    mux: dict[str, Any] | None = None,
    *,
    category: str = "other",
) -> list[dict[str, Any]]:
    """Status + call counts for non-LLM pools of one ``category`` (decision / other)."""
    routes = list((mux or {}).get("route_total") or [])
    by_pool: dict[str, dict[str, Any]] = {}
    for row in replicas:
        row_category = row.get("category") or model_category(
            row.get("kind") or "",
            row.get("pool") or row.get("name") or "",
        )
        if row_category != category:
            continue
        pool = str(row.get("pool") or row.get("name") or "other")
        slot = by_pool.setdefault(
            pool,
            {
                "id": pool,
                "kind": str(row.get("kind") or "other"),
                "category": category,
                "healthy": False,
                "replicas": 0,
                "replicas_up": 0,
                "running": 0,
                "calls": 0,
            },
        )
        slot["replicas"] += 1
        if row.get("healthy"):
            slot["healthy"] = True
            slot["replicas_up"] += 1
        if isinstance(row.get("running"), (int, float)):
            slot["running"] += int(row["running"])
    for slot in by_pool.values():
        slot["calls"] = _mux_calls_for_pool(routes, slot["id"])
    return list(by_pool.values())


def _flatten_replicas(fleet: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for server in fleet.get("servers") or []:
        if not isinstance(server, dict):
            continue
        for model in server.get("models") or []:
            if not isinstance(model, dict):
                continue
            rows.append(
                {
                    "server": server.get("label") or server.get("id") or "",
                    "host": server.get("host") or "",
                    "hostname": server.get("hostname") or "",
                    "gpus": list(server.get("gpus") or []),
                    "pool": model.get("id") or "",
                    "kind": model.get("kind") or "llm",
                    "category": model.get("category") or model_category(model.get("kind") or "", model.get("id") or ""),
                    "name": model.get("name") or "",
                    "url": model.get("url") or "",
                    "healthy": bool(model.get("healthy")),
                    "running": model.get("active"),
                    "waiting": model.get("waiting"),
                    "gpu_cache_pct": model.get("gpu_cache_pct"),
                    "max_active": model.get("max_active"),
                    "max_context": model.get("max_context"),
                    "tokens_per_sec": model.get("tokens_per_sec"),
                }
            )
    return rows


def _mux_counters(routers: list[dict[str, Any]]) -> dict[str, Any]:
    routes: list[dict[str, Any]] = []
    backpressure: list[dict[str, Any]] = []
    active: dict[str, float] = {}
    for router in routers:
        metrics = router.get("metrics") or {}
        for entry in metrics.get("route_total") or []:
            if not isinstance(entry, dict):
                continue
            labels = entry.get("labels") or {}
            routes.append(
                {
                    "upstream": labels.get("upstream") or labels.get("pool") or "",
                    "value": entry.get("value"),
                    "root": router.get("root") or "",
                }
            )
        for entry in metrics.get("backpressure_total") or []:
            if not isinstance(entry, dict):
                continue
            labels = entry.get("labels") or {}
            backpressure.append(
                {
                    "pool": labels.get("pool") or "",
                    "value": entry.get("value"),
                    "root": router.get("root") or "",
                }
            )
        for entry in metrics.get("active_requests") or []:
            if not isinstance(entry, dict):
                continue
            labels = entry.get("labels") or {}
            pool = str(labels.get("pool") or "")
            if not pool:
                continue
            try:
                active[pool] = float(entry.get("value") or 0)
            except (TypeError, ValueError):
                continue
    return {
        "route_total": routes,
        "backpressure_total": backpressure,
        "pool_active": active,
    }


def vllm_serving_snapshot() -> dict[str, Any]:
    """Full ``/api/local/vllm`` payload."""
    status = router_status_snapshot()
    fleet = status.get("fleet") or {}
    routers = status.get("routers") or []
    replicas = _flatten_replicas(fleet if isinstance(fleet, dict) else {})
    mux = _mux_counters(routers if isinstance(routers, list) else [])
    logs = collect_vllm_logs()
    recent = parse_vllm_recent_turns("\n".join(str(row.get("text") or "") for row in logs))
    llm_replicas = [row for row in replicas if row["category"] == "llm"]
    other_models = group_other_models(replicas, mux)
    decision_models = group_other_models(replicas, mux, category="decision")
    running = 0
    waiting = 0
    capacity = 0
    llm_up = 0
    for row in llm_replicas:
        if isinstance(row.get("running"), (int, float)):
            running += int(row["running"])
        if isinstance(row.get("waiting"), (int, float)):
            waiting += int(row["waiting"])
        if isinstance(row.get("max_active"), (int, float)):
            capacity += int(row["max_active"])
        if row.get("healthy"):
            llm_up += 1
    # Resolve exact pool ids against the same backend categories used by the
    # replica tables. Scorers may have neutral names; hidden/forward-only pools
    # have topology rows even when they have no advertised replicas.
    categories = {
        row["id"]: row.get("category") or model_category(row.get("kind") or "", row["id"])
        for row in fleet.get("routes") or []
    }
    categories.update({row["pool"]: row["category"] for row in replicas})
    external_ids = {row["id"] for row in fleet.get("externals") or []}

    def _llm_mux_row(row: dict[str, Any]) -> bool:
        upstream = str(row.get("upstream") or "")
        if not upstream.startswith("route:"):
            return False
        pool = upstream[len("route:") :].partition("@")[0]
        if pool in external_ids or pool.startswith("openrouter"):
            return False
        return categories.get(pool, model_category("", pool)) == "llm"

    llm_mux = {
        "route_total": [row for row in (mux.get("route_total") or []) if isinstance(row, dict) and _llm_mux_row(row)],
        "backpressure_total": mux.get("backpressure_total") or [],
        "pool_active": mux.get("pool_active") or {},
    }
    return {
        "fetched_at": _utc_now(),
        "fleet": fleet,
        "routers": routers,
        "replicas": replicas,
        "llm_replicas": llm_replicas,
        "other_models": other_models,
        "decision_models": decision_models,
        "mux": llm_mux,
        "running": running,
        "waiting": waiting,
        "capacity": capacity or None,
        "llm_replicas_healthy": llm_up,
        "llm_replicas_total": len(llm_replicas),
        "recent_turns": recent,
        "logs": logs,
        "processes": collect_vllm_processes(),
    }
