"""Where a model's output limit comes from when its config row sets none.

``max_tokens`` follows OpenCode's rule (:func:`sharedagents_model.models.resolve_max_tokens`):
``min(model output limit, cap)``. Two sources feed the limit here, both read
from files only -- never HTTP on the turn path:

* :func:`discovered_output_limit` -- what the provider's own ``GET /models``
  reported at the last native refresh, in any process (the shared
  ``discovered_models.json``);
* :func:`output_limit` -- the models.dev catalogue OpenCode uses.
  :func:`refresh` (the gateway's boot warm and native model refresh) keeps a
  slim copy under the sharedagents root. That copy also carries each model's list
  rate (shape 2, :func:`model_cost`), which ``sharedagents_model.pricing_sync``
  turns into the machine-managed pricing overlay.

Rows map to a catalogue provider by data, not by a table here: the row's API
host against the provider's published ``api`` URL (narrowed by URL path and
provider name when several providers share a host), else the row's provider
name, else a provider id that is a label of the host (``api.openai.com``).
Local endpoints (loopback, private, single-label hosts) never match -- the
catalogue lists LM Studio and friends on ``127.0.0.1``.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import tempfile
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

_LOG = logging.getLogger("sharedagents.model.catalogue")

DEFAULT_URL = "https://models.dev/api.json"
#: Overrides the source; an empty value disables the catalogue (air-gapped
#: hosts, tests).
URL_ENV = "SHAREDAGENTS_MODEL_CATALOGUE_URL"
#: Output limits rarely change; a day-old copy is fresh enough.
MAX_AGE_S = 24 * 3600
#: Bumped whenever the slim layout changes in a way an older copy cannot
#: satisfy. Shape 2 added the per-model ``cost`` block; a copy without it is
#: treated as stale so the next refresh rewrites it (otherwise a host that
#: refreshed before this existed would look "fresh" while carrying no rates).
SHAPE = 2
#: models.dev is ~5 MB; refuse anything absurd, and never hang a refresh.
MAX_BYTES = 32 * 1024 * 1024
MAX_FETCH_S = 60.0

_NATIVE_CACHE_KEY = "__native__"  # sharedagents_agent.discovery.cache_lock.NATIVE_CACHE_KEY
_NON_ALNUM = re.compile(r"[^a-z0-9]")
_memo: dict[str, tuple[tuple[int, int], Any]] = {}


def _root() -> Path:
    from sharedagents_core.kernel.layout import sharedagents_root

    return sharedagents_root()


def catalogue_path() -> Path:
    return _root() / "model_catalogue.json"


def _source_url() -> str:
    raw = os.environ.get(URL_ENV)
    return DEFAULT_URL if raw is None else raw.strip()


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _non_negative_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number < 0:  # NaN or negative
        return None
    return number


def _url(value: Any) -> str:
    """Scheme and host are case-insensitive; the path is not (two tenants on
    one gateway can differ only by path case)."""
    text = str(value or "").strip().rstrip("/")
    parts = urlparse(text)
    if not parts.scheme or not parts.netloc:
        return text
    return (
        parts._replace(
            scheme=parts.scheme.lower(),
            netloc=parts.netloc.lower(),
        )
        .geturl()
        .rstrip("/")
    )


def _slim_cost(value: Any) -> dict[str, float] | None:
    """The catalogue's per-model list rate, under the pricing table's names.

    models.dev publishes ``{input, output, cache_read, cache_write}`` -- the
    same buckets ``sharedagents_model.usage_pricing`` bills -- so this only has to
    reject the nonsense (missing, non-numeric, negative, NaN) and rename.
    """
    if not isinstance(value, dict):
        return None
    out: dict[str, float] = {}
    for src, dest in (
        ("input", "input"),
        ("output", "output"),
        ("cache_read", "cached_read"),
        ("cache_write", "cached_write"),
    ):
        rate = _non_negative_float(value.get(src))
        if rate is not None:
            out[dest] = rate
    return out or None


def _slim(data: Any) -> dict[str, Any] | None:
    """Keep what a lookup needs: provider name, API URL, model output limits
    and list cost (shape 2)."""
    if not isinstance(data, dict):
        return None
    providers: dict[str, Any] = {}
    for pid, entry in data.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("models"), dict):
            continue
        models: dict[str, Any] = {}
        for mid, model in entry["models"].items():
            if not isinstance(model, dict):
                continue
            limit = model.get("limit")
            output = _positive_int(limit.get("output")) if isinstance(limit, dict) else None
            cost = _slim_cost(model.get("cost"))
            if not output and not cost:
                continue
            kept: dict[str, Any] = {}
            if output:
                kept["output"] = output
            if cost:
                kept["cost"] = cost
            models[str(mid)] = kept
        if not models:
            continue
        api = _url(entry.get("api")) if isinstance(entry.get("api"), str) else ""
        providers[str(pid)] = {
            "name": str(entry.get("name") or ""),
            "api": api,
            "host": (urlparse(api).hostname or "") if api else "",
            "models": models,
        }
    return {"shape": SHAPE, "providers": providers} if providers else None


def _read_json(path: Path) -> Any:
    """Parse ``path``, re-reading only when the file changes."""
    try:
        stat = path.stat()
    except OSError:
        return None
    key = (stat.st_mtime_ns, stat.st_size)
    hit = _memo.get(str(path))
    if hit is not None and hit[0] == key:
        return hit[1]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = None
    _memo[str(path)] = (key, data)
    return data


def _providers() -> dict[str, Any]:
    data = _read_json(catalogue_path())
    providers = data.get("providers") if isinstance(data, dict) else None
    return providers if isinstance(providers, dict) else {}


def _fresh(path: Path, url: str) -> bool:
    """A usable copy of THIS source, younger than :data:`MAX_AGE_S`."""
    try:
        if time.time() - path.stat().st_mtime >= MAX_AGE_S:
            return False
    except OSError:
        return False
    data = _read_json(path)
    return (
        isinstance(data, dict)
        and data.get("source") == url
        and data.get("shape") == SHAPE
        and isinstance(data.get("providers"), dict)
        and bool(data["providers"])
    )


def _stream(url: str, timeout: float, deadline: float) -> bytes:
    chunks: list[bytes] = []
    size = 0
    with (
        httpx.Client(
            timeout=timeout,
            headers={"User-Agent": "sharedagents", "Accept": "application/json"},
        ) as client,
        client.stream("GET", url) as resp,
    ):
        resp.raise_for_status()
        for chunk in resp.iter_bytes():
            size += len(chunk)
            if size > MAX_BYTES:
                raise ValueError(f"catalogue larger than {MAX_BYTES} bytes")
            if time.monotonic() > deadline:
                raise TimeoutError(f"catalogue download over {MAX_FETCH_S}s")
            chunks.append(chunk)
    return b"".join(chunks)


def _download(url: str, timeout: float) -> bytes:
    """The body, or an error once :data:`MAX_FETCH_S` has passed in total.

    httpx timeouts are per read, so a server trickling its headers never
    trips them; the fetch runs on a daemon thread that is abandoned at the
    deadline (it only ever writes its own result slot).
    """
    deadline = time.monotonic() + MAX_FETCH_S
    result: dict[str, Any] = {}

    def run() -> None:
        try:
            result["body"] = _stream(url, timeout, deadline)
        except BaseException as exc:  # noqa: BLE001 -- re-raised below
            result["error"] = exc

    worker = threading.Thread(target=run, name="sharedagents-catalogue-fetch", daemon=True)
    worker.start()
    worker.join(MAX_FETCH_S)
    if worker.is_alive():
        raise TimeoutError(f"catalogue download over {MAX_FETCH_S}s")
    if "error" in result:
        raise result["error"]
    return result["body"]


def refresh(*, force: bool = False, timeout: float = 20.0) -> bool:
    """Fetch the catalogue into the local copy unless a fresh one is in place.

    True when a usable copy is in place afterwards. A failed fetch keeps the
    previous copy -- lookups never check its age.
    """
    url = _source_url()
    if not url:
        return False
    path = catalogue_path()
    if not force and _fresh(path, url):
        return True
    try:
        slim = _slim(json.loads(_download(url, timeout)))
    except Exception as exc:  # noqa: BLE001 -- network, HTTP status, size, JSON
        _LOG.info("model catalogue refresh failed (%s): %s", url, exc)
        return False
    if slim is None:
        _LOG.info("model catalogue refresh: no output limits in %s", url)
        return False
    slim["source"] = url
    slim["fetched_at"] = int(time.time())
    tmp: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=".model_catalogue.",
            suffix=".tmp",
        )
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(slim, fh, separators=(",", ":"))
        os.replace(tmp, path)
        tmp = None
    except OSError as exc:
        _LOG.warning("model catalogue: could not write %s: %s", path, exc)
        return False
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    return True


def _norm(text: str) -> str:
    return _NON_ALNUM.sub("", (text or "").lower())


def _is_local(host: str) -> bool:
    if not host or host == "localhost" or "." not in host:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private or ip.is_link_local


def _by_name(pids: list[str], name: str, providers: dict[str, Any]) -> list[str]:
    return [pid for pid in pids if name and name in (_norm(pid), _norm(str(providers[pid].get("name") or "")))]


def _candidates(row: Mapping[str, Any], providers: dict[str, Any]) -> list[str]:
    base_url = _url(row.get("baseUrl") or row.get("endpoint"))
    base = urlparse(base_url)
    if base.scheme == "bedrock":
        return ["amazon-bedrock"] if "amazon-bedrock" in providers else []
    host = base.hostname or ""
    if _is_local(host):
        return []
    name = _norm(str(row.get("provider") or ""))
    by_host = sorted(pid for pid, p in providers.items() if p.get("host") == host)
    if len(by_host) > 1:
        # Several providers share the host (zai / zai-coding-plan): the
        # row's own API path, then its provider name, pick among them.
        by_path = [
            pid
            for pid in by_host
            if providers[pid].get("api")
            and (base_url == providers[pid]["api"] or base_url.startswith(providers[pid]["api"] + "/"))
        ]
        by_host = by_path or by_host
        if len(by_host) > 1:
            by_host = _by_name(by_host, name, providers) or by_host
    if by_host:
        return by_host
    by_name = _by_name(sorted(providers), name, providers)
    if by_name:
        return by_name
    labels = set(host.split("."))
    return sorted(pid for pid in providers if pid in labels)


def _slim_model(providers: dict[str, Any], pid: str, model_id: str) -> dict[str, Any]:
    """The slim entry for one model on one provider, or ``{}``."""
    models = (providers.get(pid) or {}).get("models")
    entry = models.get(model_id) if isinstance(models, dict) else None
    return entry if isinstance(entry, dict) else {}


def output_limit(row: Mapping[str, Any], model_id: str) -> int | None:
    """The catalogue's output limit for ``model_id`` on ``row``'s provider.

    When the row still maps to several providers that list the model with
    different limits, the smallest wins: a refused request costs a turn.
    """
    if not model_id or not _source_url():
        return None
    providers = _providers()
    found = [
        limit
        for pid in (_candidates(row, providers) if providers else ())
        if (limit := _positive_int(_slim_model(providers, pid, model_id).get("output")))
    ]
    return min(found) if found else None


def providers_for_row(row: Mapping[str, Any], providers: dict[str, Any]) -> list[str]:
    """Catalogue provider ids a discovered row could belong to.

    :func:`_candidates` maps by API host, which every EXTERNAL-engine row
    lacks: those carry ``provider`` as a plain string and no endpoint at all,
    so they fall back to a name match. The fallback is deliberately gated on
    having no endpoint whatsoever -- a row WITH an endpoint that mapped to
    nothing stays unmapped, so a local OpenAI-compatible server can never be
    priced at a remote provider's rates."""
    if not providers:
        return []
    found = _candidates(row, providers)
    if found:
        return found
    if _url(row.get("baseUrl") or row.get("endpoint")):
        return []
    return _by_name(sorted(providers), _norm(str(row.get("provider") or "")), providers)


def slim_providers() -> dict[str, Any]:
    """The local slim catalogue's providers (shape 2), or ``{}``.

    Each model maps to ``{"output": <int>, "cost": {...}}``. Empty when the
    source is disabled, no copy has been written yet, or the copy predates
    :data:`SHAPE`."""
    if not _source_url():
        return {}
    return _providers()


def model_cost(pid: str, model_id: str) -> dict[str, float] | None:
    """The catalogue's list rate for ``model_id`` on provider ``pid``.

    Buckets are already the pricing table's names (``input``, ``output``,
    ``cached_read``, ``cached_write``). ``None`` when the catalogue is
    disabled, stale, or simply does not carry this model.
    """
    if not model_id or not pid or not _source_url():
        return None
    providers = _providers()
    cost = _slim_model(providers, pid, model_id).get("cost")
    return dict(cost) if isinstance(cost, dict) else None


def catalogue_revision() -> tuple[int | None, str]:
    """``(fetched_at, source)`` of the local copy, for row provenance."""
    data = _read_json(catalogue_path())
    if not isinstance(data, dict):
        return None, ""
    fetched = data.get("fetched_at")
    return (
        int(fetched) if isinstance(fetched, (int, float)) and not isinstance(fetched, bool) else None,
        str(data.get("source") or ""),
    )


def discovered_output_limit(row: Mapping[str, Any], model_id: str) -> int | None:
    """The output limit the provider's own ``GET /models`` reported for
    ``model_id`` at ``row``'s endpoint, from the shared native discovery cache
    (a refresh run by the web process reaches the long-lived gateway here)."""
    endpoint = _url(row.get("baseUrl") or row.get("endpoint"))
    if not endpoint or not model_id:
        return None
    data = _read_json(_root() / "discovered_models.json")
    rows = data.get(_NATIVE_CACHE_KEY) if isinstance(data, dict) else None
    for cached in rows if isinstance(rows, list) else ():
        if (
            isinstance(cached, dict)
            and str(cached.get("id") or "").strip() == model_id
            and _url(cached.get("baseUrl") or cached.get("endpoint")) == endpoint
        ):
            return _positive_int(cached.get("outputLimit"))
    return None
