"""Live vLLM router status for the Usage tab.

Pure read + bounded HTTP: each router root is derived from the configured
model ``baseUrl`` values (never hardcoded), configured ``backend=vllm``
models are grouped by root, and ``{root}/health``, ``{root}/v1/models`` and
``{root}/metrics`` are fetched live with a 2-second timeout. Every endpoint
degrades independently — a failure marks only that endpoint unavailable with
an error message and never raises, so the ledger data on the Usage tab is
always preserved.
"""

from __future__ import annotations

import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from sharedagents_core.kernel.config import load_global_config

#: Per-endpoint fetch timeout in seconds (req 6).
ROUTER_TIMEOUT = 2.0

_HEALTH_FIELDS = (
    "status",
    "ok",
    "degraded",
    "down",
    "active_requests",
    "external_providers",
)

#: Response key -> exact Prometheus metric name (req 10).
_METRIC_NAMES = {
    "replica_healthy": "vllm_mux_replica_healthy",
    "active_requests": "vllm_mux_active_requests",
    "max_active_requests": "vllm_mux_max_active_requests",
    "route_total": "vllm_mux_route_total",
    "backpressure_total": "vllm_mux_backpressure_total",
}
_METRIC_TO_KEY = {name: key for key, name in _METRIC_NAMES.items()}

_METRIC_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\s*(\{[^}]*\})?\s+(\S+)(?:\s+\S+)?\s*$")
_LABEL_PAIR = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


_NON_LLM_KINDS = frozenset(
    {
        "embedding",
        "embed",
        "reranker",
        "asr",
        "tts",
        "whisper",
        "speech",
    }
)
_NON_LLM_NAME = ("embed", "rerank", "asr", "tts", "whisper")


def model_category(kind: str = "", name: str = "") -> str:
    """``llm``, ``decision`` or ``other`` for a router pool; the page groups by it.

    Decision pools are scorers, not chat models: they get their own
    group instead of falling into LLM counts.
    """
    key = str(kind or "").strip().lower()
    blob = f"{key} {name or ''}".lower()
    if key == "decision" or (not key and "decision" in blob):
        return "decision"
    if key in _NON_LLM_KINDS or any(token in blob for token in _NON_LLM_NAME):
        return "other"
    return "llm"


def derive_router_root(base_url: str) -> str:
    """Router root for a configured model baseUrl (req 2).

    Strip trailing slashes, then remove exactly one trailing ``/v1`` if
    present; otherwise use the baseUrl unchanged.
    """
    root = str(base_url or "").rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return root


def configured_vllm_models(config: dict | None = None) -> list[dict[str, Any]]:
    """Configured ``backend=vllm`` models as ``{name, baseUrl, root}`` rows.

    ``config`` defaults to a fresh read of the global config so every
    snapshot is live (no cache). Legacy rows use ``endpoint`` instead of
    ``baseUrl`` (same normalization as ``sharedagents_agent.agent``).
    """
    if config is None:
        config = load_global_config()
    rows = config.get("models") if isinstance(config, dict) else None
    result: list[dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("backend") or "").strip().lower() != "vllm":
            continue
        base_url = str(row.get("baseUrl") or row.get("endpoint") or "").strip()
        if not base_url:
            continue
        name = str(row.get("alias") or "").strip() or str(row.get("id") or "").strip()
        if not name:
            continue
        result.append({"name": name, "baseUrl": base_url, "root": derive_router_root(base_url)})
    return result


def group_models_by_root(rows: list[dict[str, Any]]) -> list[tuple[str, list[str]]]:
    """Group model rows by derived router root (first-appearance order)."""
    order: list[str] = []
    names: dict[str, list[str]] = {}
    for row in rows:
        root = row["root"]
        if root not in names:
            names[root] = []
            order.append(root)
        names[root].append(row["name"])
    return [(root, names[root]) for root in order]


def parse_health(payload: Any) -> dict[str, Any] | None:
    """/health JSON -> the six health fields, absent keys as None (req 8)."""
    if not isinstance(payload, dict):
        return None
    return {field: payload.get(field) for field in _HEALTH_FIELDS}


def parse_models(payload: Any) -> list | None:
    """/v1/models -> the model/replica list verbatim, or None (req 9)."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    return None


def _parse_labels(label_blob: str | None) -> dict[str, str]:
    if not label_blob:
        return {}
    inner = label_blob.strip()
    if inner.startswith("{") and inner.endswith("}"):
        inner = inner[1:-1]
    return dict(_LABEL_PAIR.findall(inner))


def parse_metrics(text: str) -> dict[str, list | None]:
    """Prometheus text -> the five vllm_mux_* groups (reqs 10-11).

    Each entry is ``{"labels": {...}, "value": float}``; absent metric
    names come back as None. ``replica_healthy`` entries additionally
    carry ``healthy`` (value > 0). Raises ValueError when the body
    contains data lines that parse into no samples at all (req 7:
    invalid data is a per-endpoint failure, not an empty result).
    """
    buckets: dict[str, list] = {key: [] for key in _METRIC_NAMES}
    saw_data_line = False
    saw_sample = False
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        saw_data_line = True
        match = _METRIC_LINE.match(line)
        if not match:
            continue
        name, label_blob, value_token = match.group(1), match.group(2), match.group(3)
        try:
            value = float(value_token)
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        saw_sample = True
        key = _METRIC_TO_KEY.get(name)
        if key is None:
            continue
        entry: dict[str, Any] = {"labels": _parse_labels(label_blob), "value": value}
        if key == "replica_healthy":
            entry["healthy"] = value > 0
        buckets[key].append(entry)
    if saw_data_line and not saw_sample:
        raise ValueError("invalid Prometheus text: no parseable samples")
    return {key: (entries or None) for key, entries in buckets.items()}


def _error_message(exc: Exception, timeout: float) -> str:
    if isinstance(exc, requests.Timeout):
        return f"timeout after {timeout:g}s"
    return str(exc) or exc.__class__.__name__


def _fetch_json(url: str, timeout: float) -> Any:
    resp = requests.get(url, timeout=timeout)
    if resp.status_code // 100 != 2:
        raise ValueError(f"{resp.status_code} {resp.reason or ''}".strip())
    return resp.json()


def _fetch_text(url: str, timeout: float) -> str:
    resp = requests.get(url, timeout=timeout)
    if resp.status_code // 100 != 2:
        raise ValueError(f"{resp.status_code} {resp.reason or ''}".strip())
    return resp.text


def fetch_router_endpoints(root: str, timeout: float = ROUTER_TIMEOUT) -> dict[str, Any]:
    """Live-fetch /health, /v1/models and /metrics for one root (reqs 6-7).

    Each endpoint is wrapped in its own try/except so one failure never
    masks the others. Returns the ``health`` / ``models_endpoint`` /
    ``metrics`` blocks of a router-status entry.
    """
    health: dict[str, Any] = {"available": False, "error": None}
    health.update(dict.fromkeys(_HEALTH_FIELDS))
    try:
        parsed = parse_health(_fetch_json(f"{root}/health", timeout))
        if parsed is None:
            raise ValueError("invalid JSON: /health is not an object")
        health.update(parsed)
        health["available"] = True
    except Exception as exc:  # noqa: BLE001 — per-endpoint isolation (req 7)
        health["error"] = _error_message(exc, timeout)

    models_endpoint: dict[str, Any] = {"available": False, "error": None, "data": None}
    try:
        data = parse_models(_fetch_json(f"{root}/v1/models", timeout))
        if data is None:
            raise ValueError("invalid JSON: /v1/models has no model list")
        models_endpoint["data"] = data
        models_endpoint["available"] = True
    except Exception as exc:  # noqa: BLE001 — per-endpoint isolation (req 7)
        models_endpoint["error"] = _error_message(exc, timeout)

    metrics: dict[str, Any] = {"available": False, "error": None}
    metrics.update(dict.fromkeys(_METRIC_NAMES))
    try:
        metrics.update(parse_metrics(_fetch_text(f"{root}/metrics", timeout)))
        metrics["available"] = True
    except Exception as exc:  # noqa: BLE001 — per-endpoint isolation (req 7)
        metrics["error"] = _error_message(exc, timeout)

    return {"health": health, "models_endpoint": models_endpoint, "metrics": metrics}


def hardware_inventory_path() -> Path:
    """Optional host inventory: ``SHAREDAGENTS_HARDWARE_INVENTORY``, else the sharedagents home."""
    env = os.environ.get("SHAREDAGENTS_HARDWARE_INVENTORY")
    if env:
        return Path(env).expanduser()
    from sharedagents_core.layout import sharedagents_root

    return sharedagents_root() / "hardware_inventory.json"


def _vram_gb(raw: Any) -> float | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        amount = float(text.split()[0])
    except (TypeError, ValueError, IndexError):
        return None
    unit = text.lower()
    if "gib" in unit or "gb" in unit:
        return round(amount)
    if "mib" in unit or "mb" in unit:
        return round(amount / 1024)
    return round(amount)


def _short_gpu_name(name: str) -> str:
    text = re.sub(r"^NVIDIA\s+(GeForce\s+)?", "", str(name or "").strip())
    text = re.sub(r"\s+(Blackwell|Ada Lovelace|Hopper|Ampere)\b.*$", "", text, flags=re.I)
    text = re.sub(r"\s+Workstation Edition$", "", text, flags=re.I)
    return text or str(name or "GPU")


def estimate_llm_tps(gpus: list[dict[str, Any]]) -> float | None:
    """Rough decode-capacity hint from the GPUs on the box (one serving replica)."""
    names = " ".join(str(gpu.get("name") or "") for gpu in gpus).upper()
    if "5090" in names:
        return 55.0
    if "PRO 6000" in names or "RTX 6000" in names:
        return 70.0
    if "4090" in names:
        return 45.0
    if "3090" in names:
        return 28.0
    if "3060" in names:
        return 12.0
    return None


def parse_compute_gpus(entry: dict[str, Any] | None) -> list[dict[str, Any]]:
    """NVIDIA compute GPUs only — skip iGPU / BMC display adapters."""
    gpus: list[dict[str, Any]] = []
    for raw in (entry or {}).get("gpu") or []:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "")
        kind = str(raw.get("type") or "").upper()
        upper = name.upper()
        if "ASPEED" in upper or "RAPHAEL" in upper:
            continue
        if kind not in ("NVIDIA", "") and "NVIDIA" not in upper and "RTX" not in upper:
            continue
        if "NVIDIA" not in upper and "RTX" not in upper and kind != "NVIDIA":
            continue
        gpus.append(
            {
                "name": _short_gpu_name(name),
                "vram_gb": _vram_gb(raw.get("memory")),
            }
        )
    return gpus


def load_hardware_by_ip(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Map IPv4 -> {hostname, label, gpus} from the host inventory.

    ``label`` is the entry's own ``label`` when it has one, else the short
    hostname (domain dropped).
    """
    inventory = path or hardware_inventory_path()
    try:
        rows = json.loads(inventory.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(rows, list):
        return {}
    by_ip: dict[str, dict[str, Any]] = {}
    for entry in rows:
        if not isinstance(entry, dict):
            continue
        hostname = str(entry.get("hostname") or "").strip()
        label = str(entry.get("label") or "").strip() or hostname.split(".", 1)[0]
        host = {
            "hostname": hostname,
            "label": label,
            "gpus": parse_compute_gpus(entry),
        }
        for nic in entry.get("network") or []:
            if not isinstance(nic, dict):
                continue
            for ip in nic.get("ipv4") or []:
                ip_s = str(ip or "").strip()
                if ip_s:
                    by_ip[ip_s] = host
    return by_ip


def _mux_yaml_path(path: Path | None = None) -> Path | None:
    if path is not None:
        return Path(path)
    try:
        from sharedagents_vllm.config import default_config_path

        return default_config_path()
    except Exception:  # noqa: BLE001
        return None


def _load_mux_raw(path: Path | None = None) -> dict[str, Any]:
    resolved = _mux_yaml_path(path)
    if resolved is None:
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    try:
        raw = yaml.safe_load(Path(resolved).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _external_row(eid: str, edata: dict[str, Any]) -> dict[str, Any]:
    base = str(edata.get("baseUrl") or edata.get("base_url") or "").strip()
    try:
        max_active = int(edata.get("max_active_requests") or 0) or None
    except (TypeError, ValueError):
        max_active = None
    try:
        timeout = int(edata.get("timeout_seconds") or 0) or None
    except (TypeError, ValueError):
        timeout = None
    return {
        "kind": "external",
        "id": str(eid),
        "model": str(edata.get("model") or "").strip(),
        "host": host_from_url(base),
        "base_url": base,
        "max_active": max_active,
        "timeout_seconds": timeout,
    }


def load_mux_topology(path: Path | None = None) -> dict[str, Any]:
    """Pools, replica caps, fallback hops, and external providers (no secrets)."""
    raw = _load_mux_raw(path)
    externals: dict[str, dict[str, Any]] = {}
    for eid, edata in (raw.get("external_providers") or {}).items():
        if isinstance(edata, dict) and str(eid).strip():
            externals[str(eid)] = _external_row(str(eid), edata)
    caps: dict[str, dict[str, Any]] = {}
    pools: dict[str, dict[str, Any]] = {}
    for pool_id, pdata in (raw.get("pools") or {}).items():
        if not isinstance(pdata, dict):
            continue
        try:
            pool_max = int(pdata.get("max_active_requests") or 0)
        except (TypeError, ValueError):
            pool_max = 0
        kind = str(pdata.get("kind") or "")
        hops: list[dict[str, Any]] = []
        for fb in pdata.get("fallbacks") or []:
            if not isinstance(fb, dict):
                continue
            pool_hop = str(fb.get("pool") or "").strip()
            ext_hop = str(fb.get("external") or "").strip()
            if pool_hop:
                hops.append({"kind": "pool", "id": pool_hop})
            elif ext_hop:
                hops.append(
                    dict(
                        externals.get(ext_hop)
                        or {
                            "kind": "external",
                            "id": ext_hop,
                            "model": "",
                            "host": "",
                            "base_url": "",
                            "max_active": None,
                            "timeout_seconds": None,
                        }
                    )
                )
        pools[str(pool_id)] = {
            "id": str(pool_id),
            "kind": kind,
            "max_active": pool_max or None,
            "fallbacks": hops,
        }
        for replica in pdata.get("replicas") or []:
            if not isinstance(replica, dict):
                continue
            name = str(replica.get("name") or "").strip()
            url = str(replica.get("url") or "").rstrip("/")
            if not name or not url:
                continue
            caps[f"{name}@{url}"] = {
                "max_active": int(replica.get("max_active") or 0),
                "pool_max_active": pool_max,
                "kind": kind,
                "pool": str(pool_id),
            }
    return {
        "caps": caps,
        "pools": pools,
        "externals": list(externals.values()),
    }


def load_replica_caps(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Replica concurrency caps from vllm-router.yaml, keyed ``name@url``."""
    return load_mux_topology(path).get("caps") or {}


def model_context_windows(config: dict | None = None) -> dict[str, int]:
    if config is None:
        config = load_global_config()
    windows: dict[str, int] = {}
    for row in (config or {}).get("models") or []:
        if not isinstance(row, dict):
            continue
        try:
            window = int(row.get("contextWindow") or 0)
        except (TypeError, ValueError):
            window = 0
        if window <= 0:
            continue
        for key in (row.get("id"), row.get("alias")):
            name = str(key or "").strip()
            if name:
                windows[name] = window
    return windows


def observed_tps_by_model(limit: int = 200) -> dict[str, float]:
    """Recent measured generation tok/s from the sharedagents generation log."""
    try:
        from sharedagents_model.usage_ledger import list_generations

        rows = list_generations(limit=limit).get("generations") or []
    except Exception:  # noqa: BLE001
        return {}
    buckets: dict[str, list[float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            tps = float(row.get("gen_tps") or 0)
        except (TypeError, ValueError):
            continue
        if tps <= 0:
            continue
        for key in (row.get("model"), row.get("model_alias")):
            name = str(key or "").strip()
            if name:
                buckets.setdefault(name, []).append(tps)
    return {name: sum(vals) / len(vals) for name, vals in buckets.items() if vals}


def host_from_url(url: str) -> str:
    return (urlparse(str(url or "")).hostname or "").strip()


def _lookup_tps(model_id: str, observed: dict[str, float]) -> float | None:
    if model_id in observed:
        return round(observed[model_id], 1)
    compact = model_id.replace(".", "").replace("-", "")
    for key, value in observed.items():
        if key.replace(".", "").replace("-", "") == compact:
            return round(value, 1)
    return None


_REPLICA_METRIC_ALIASES = (
    ("running", ("num_requests_running",)),
    ("waiting", ("num_requests_waiting",)),
    ("gpu_cache_pct", ("gpu_cache_usage_perc", "kv_cache_usage_perc")),
    ("tokens_per_sec", ("avg_generation_throughput_toks_per_s",)),
)


def _metric_basename(name: str) -> str:
    text = str(name or "").strip()
    if text.startswith("vllm:"):
        text = text[len("vllm:") :]
    elif text.startswith("vllm_"):
        text = text[len("vllm_") :]
    return text


def parse_vllm_replica_metrics(text: str) -> dict[str, Any]:
    """Prometheus text from a vLLM replica -> running/waiting/cache/tok/s."""
    found: dict[str, float] = {}
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _METRIC_LINE.match(line)
        if not match:
            continue
        key = _metric_basename(match.group(1))
        try:
            value = float(match.group(3))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        found[key] = value
    out: dict[str, Any] = {
        "running": None,
        "waiting": None,
        "gpu_cache_pct": None,
        "tokens_per_sec": None,
    }
    for dest, aliases in _REPLICA_METRIC_ALIASES:
        for alias in aliases:
            if alias not in found:
                continue
            value = found[alias]
            if dest in ("running", "waiting"):
                out[dest] = int(value)
            elif dest == "gpu_cache_pct":
                out[dest] = round(value * 100.0, 1) if 0.0 <= value <= 1.0 else round(value, 1)
            elif value > 0:
                out[dest] = round(value, 1)
            break
    return out


def _probe_replica(url: str, timeout: float) -> dict[str, Any]:
    """Healthy-replica only: context window + live vLLM serving gauges."""
    base = str(url or "").rstrip("/")
    out: dict[str, Any] = {
        "max_context": None,
        "running": None,
        "waiting": None,
        "gpu_cache_pct": None,
        "tokens_per_sec": None,
    }
    try:
        models = parse_models(_fetch_json(f"{base}/v1/models", timeout))
        if models:
            window = models[0].get("max_model_len") if isinstance(models[0], dict) else None
            if window:
                out["max_context"] = int(window)
    except Exception:  # noqa: BLE001
        pass
    try:
        text = _fetch_text(f"{base}/metrics", timeout)
    except Exception:  # noqa: BLE001
        return out
    out.update(parse_vllm_replica_metrics(text))
    return out


def _metric_value(metrics: dict[str, Any], key: str, label: str, label_value: str) -> float | None:
    entries = metrics.get(key)
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        labels = entry.get("labels") or {}
        if str(labels.get(label) or "") == label_value:
            try:
                return float(entry.get("value"))
            except (TypeError, ValueError):
                return None
    return None


def _annotate_fallback_health(
    hops: list[dict[str, Any]],
    pool_health: dict[str, bool],
    provider_health: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Mark each hop usable/unusable from live router data.

    Pool hops use the replica roll-up; external hops use the provider's
    ``enabled`` state from ``/health`` — a configured cloud provider with no
    API key can never serve a fallback. Rows stay unannotated when the router
    does not report the provider, so an older router renders as before.
    """
    providers = provider_health or {}
    out = []
    for hop in hops:
        row = dict(hop)
        kind = row.get("kind")
        if kind == "pool":
            hid = str(row.get("id") or "")
            if hid in pool_health:
                row["healthy"] = pool_health[hid]
        elif kind == "external":
            info = providers.get(str(row.get("id") or ""))
            if info is not None:
                row["enabled"] = bool(info.get("enabled"))
                cooling = _cooldown_seconds(info) > 0
                row["healthy"] = row["enabled"] and not cooling
                row["api_key_env"] = row.get("api_key_env") or info.get("api_key_env") or ""
                row["model"] = row.get("model") or info.get("model") or ""
                if not row["enabled"]:
                    env = row["api_key_env"]
                    row["disabled_reason"] = f"no API key ({env})" if env else "no API key"
                elif cooling:
                    row["disabled_reason"] = f"cooling {int(_cooldown_seconds(info))}s"
        out.append(row)
    return out


def _cooldown_seconds(info: dict[str, Any]) -> float:
    """Provider cooldown from a /health row; unparseable or absent -> 0."""
    try:
        return float(info.get("cooldown_remaining_s") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _provider_health(routers: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """External provider id -> its ``/health`` row (enabled, api_key_env, model)."""
    out: dict[str, dict[str, Any]] = {}
    for router in routers:
        payload = router.get("health") or {}
        for row in payload.get("external_providers") or []:
            if not isinstance(row, dict):
                continue
            pid = str(row.get("id") or "")
            if pid:
                out[pid] = row
    return out


def _annotate_externals(externals: list[Any], provider_health: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """External provider cards carry the live ``enabled`` flag when reported."""
    out: list[dict[str, Any]] = []
    for row in externals:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        info = provider_health.get(str(item.get("id") or ""))
        if info is not None:
            item["enabled"] = bool(info.get("enabled"))
            item["api_key_env"] = item.get("api_key_env") or info.get("api_key_env") or ""
        out.append(item)
    return out


def build_fleet(
    routers: list[dict[str, Any]],
    *,
    inventory: dict[str, dict[str, Any]] | None = None,
    replica_caps: dict[str, dict[str, Any]] | None = None,
    context_windows: dict[str, int] | None = None,
    observed_tps: dict[str, float] | None = None,
    replica_probes: dict[str, dict[str, Any]] | None = None,
    mux_topology: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Server-centric view: GPUs, models, concurrency, context, tok/s, fallbacks."""
    inventory = inventory if inventory is not None else load_hardware_by_ip()
    topology = mux_topology if mux_topology is not None else load_mux_topology()
    replica_caps = replica_caps if replica_caps is not None else (topology.get("caps") or {})
    pool_meta = topology.get("pools") or {}
    context_windows = context_windows if context_windows is not None else model_context_windows()
    observed_tps = observed_tps if observed_tps is not None else observed_tps_by_model()

    servers: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    replicas_total = 0
    replicas_healthy = 0
    active_total = 0
    waiting_total = 0
    max_total = 0
    pool_health: dict[str, bool] = {}
    provider_health = _provider_health(routers)

    for router in routers:
        health = router.get("health") or {}
        metrics = router.get("metrics") or {}
        pool_active = health.get("active_requests") if isinstance(health.get("active_requests"), dict) else {}
        models_data = (router.get("models_endpoint") or {}).get("data") or []
        if not isinstance(models_data, list):
            continue
        for model in models_data:
            if not isinstance(model, dict):
                continue
            model_id = str(model.get("id") or "").strip()
            kind = str(model.get("kind") or (pool_meta.get(model_id) or {}).get("kind") or "")
            replicas = model.get("replicas") or []
            if not isinstance(replicas, list):
                continue
            # The router reports forward-only (alias) pools through the pool
            # that serves them; older routers only expose their replicas.
            route_state = model.get("route_state")
            if route_state in ("ok", "degraded", "down"):
                pool_health[model_id] = route_state != "down"
            else:
                pool_health[model_id] = any(isinstance(r, dict) and r.get("healthy") for r in replicas)
            pool_max = _metric_value(metrics, "max_active_requests", "pool", model_id)
            for replica in replicas:
                if not isinstance(replica, dict):
                    continue
                name = str(replica.get("name") or "").strip()
                url = str(replica.get("url") or "").rstrip("/")
                if not url:
                    continue
                host = host_from_url(url)
                healthy = bool(replica.get("healthy"))
                replicas_total += 1
                if healthy:
                    replicas_healthy += 1
                cap = replica_caps.get(f"{name}@{url}") or {}
                max_active = int(cap.get("max_active") or 0)
                if max_active <= 0 and pool_max:
                    max_active = int(pool_max)
                probe = (replica_probes or {}).get(url) or {}
                running = probe.get("running")
                if running is None and isinstance(pool_active, dict) and len(replicas) == 1:
                    try:
                        running = int(pool_active.get(model_id) or 0)
                    except (TypeError, ValueError):
                        running = 0
                if isinstance(running, (int, float)):
                    active_total += int(running)
                if isinstance(probe.get("waiting"), (int, float)):
                    waiting_total += int(probe.get("waiting") or 0)
                max_total += max_active
                context = probe.get("max_context") or context_windows.get(model_id)
                tps = probe.get("tokens_per_sec")
                if tps is None:
                    tps = _lookup_tps(model_id, observed_tps)
                if tps is None and (cap.get("kind") or kind) == "llm":
                    hw = inventory.get(host) or {}
                    tps = estimate_llm_tps(list(hw.get("gpus") or []))
                hops = list((pool_meta.get(model_id) or {}).get("fallbacks") or [])
                waiting = probe.get("waiting")
                model_row = {
                    "id": model_id,
                    "kind": cap.get("kind") or kind or "llm",
                    "category": model_category(cap.get("kind") or kind, model_id),
                    "healthy": healthy,
                    "url": url,
                    "name": name,
                    "max_context": int(context) if context else None,
                    "active": int(running) if running is not None else None,
                    "waiting": int(waiting) if waiting is not None else None,
                    "gpu_cache_pct": probe.get("gpu_cache_pct"),
                    "max_active": max_active or None,
                    "pool_max_active": (
                        int(cap.get("pool_max_active") or 0)
                        or ((pool_meta.get(model_id) or {}).get("max_active"))
                        or (int(pool_max) if pool_max else None)
                    )
                    or None,
                    "tokens_per_sec": tps,
                    "fallbacks": hops,
                }
                server = servers.get(host)
                if server is None:
                    hw = inventory.get(host) or {}
                    label = str(hw.get("label") or name or host or "server").upper()
                    server = {
                        "id": name or host,
                        "label": label,
                        "hostname": hw.get("hostname") or "",
                        "host": host,
                        "gpus": list(hw.get("gpus") or []),
                        "healthy_replicas": 0,
                        "replicas": 0,
                        "models": [],
                    }
                    servers[host] = server
                    order.append(host)
                server["replicas"] += 1
                if healthy:
                    server["healthy_replicas"] += 1
                server["models"].append(model_row)

    nodes = []
    for host in order:
        server = servers[host]
        if server["healthy_replicas"] == server["replicas"] and server["replicas"]:
            state = "ok"
        elif server["healthy_replicas"] > 0:
            state = "degraded"
        else:
            state = "down"
        server["state"] = state
        nodes.append(server)

    routes = []
    for pool_id, meta in pool_meta.items():
        routes.append(
            {
                "id": pool_id,
                "kind": meta.get("kind") or "llm",
                "category": model_category(meta.get("kind") or "", pool_id),
                "max_active": meta.get("max_active"),
                "healthy": pool_health.get(pool_id),
                "fallbacks": _annotate_fallback_health(list(meta.get("fallbacks") or []), pool_health, provider_health),
            }
        )
    for server in nodes:
        for model in server.get("models") or []:
            model["fallbacks"] = _annotate_fallback_health(
                list(model.get("fallbacks") or []), pool_health, provider_health
            )

    cluster_status = "ok"
    if replicas_total and replicas_healthy == 0:
        cluster_status = "down"
    elif replicas_healthy < replicas_total:
        cluster_status = "degraded"
    if routers:
        reported = str(((routers[0].get("health") or {}).get("status")) or "")
        if reported in ("ok", "degraded", "down"):
            cluster_status = reported

    return {
        "status": cluster_status,
        "servers_total": len(nodes),
        "replicas_healthy": replicas_healthy,
        "replicas_total": replicas_total,
        "active_requests": active_total,
        "waiting_requests": waiting_total,
        "max_active_requests": max_total,
        "routes": routes,
        "externals": _annotate_externals(topology.get("externals") or [], provider_health),
        "servers": nodes,
    }


def _probe_healthy_replicas(routers: list[dict[str, Any]], timeout: float) -> dict[str, dict[str, Any]]:
    urls: list[str] = []
    seen: set[str] = set()
    for router in routers:
        for model in (router.get("models_endpoint") or {}).get("data") or []:
            if not isinstance(model, dict):
                continue
            for replica in model.get("replicas") or []:
                if not isinstance(replica, dict) or not replica.get("healthy"):
                    continue
                url = str(replica.get("url") or "").rstrip("/")
                if url and url not in seen:
                    seen.add(url)
                    urls.append(url)
    if not urls:
        return {}
    probes: dict[str, dict[str, Any]] = {}
    workers = min(6, len(urls))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_probe_replica, url, min(timeout, 1.2)): url for url in urls}
        for future in as_completed(futures):
            url = futures[future]
            try:
                probes[url] = future.result()
            except Exception:  # noqa: BLE001
                probes[url] = {
                    "max_context": None,
                    "running": None,
                    "waiting": None,
                    "gpu_cache_pct": None,
                    "tokens_per_sec": None,
                }
    return probes


def router_status_snapshot(config: dict | None = None, timeout: float = ROUTER_TIMEOUT) -> dict[str, Any]:
    """Full ``/api/usage/router-status`` payload (req 5).

    Fresh config read + fresh fetches on every call — no persistent cache.
    """
    if config is None:
        config = load_global_config()
    routers = []
    for root, names in group_models_by_root(configured_vllm_models(config)):
        endpoints = fetch_router_endpoints(root, timeout=timeout)
        routers.append(
            {
                "root": root,
                "models": names,
                "health": endpoints["health"],
                "models_endpoint": endpoints["models_endpoint"],
                "metrics": endpoints["metrics"],
            }
        )
    probes = _probe_healthy_replicas(routers, timeout=timeout)
    fleet = build_fleet(
        routers,
        context_windows=model_context_windows(config),
        replica_probes=probes,
        mux_topology=load_mux_topology(),  # operator YAML only; ships empty
    )
    return {"routers": routers, "fleet": fleet}
