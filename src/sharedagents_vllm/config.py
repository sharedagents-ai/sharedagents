"""Load and validate router YAML.

Literal ``api_key`` values are rejected. Cloud backup uses ``api_key_env``
names resolved from the process environment at bind time.
"""

from __future__ import annotations

import logging
import math
import os
import re
from pathlib import Path
from typing import Any

from .models import (
    DECISION_CHECKPOINTS,
    DECISION_DEFAULT_CHECKPOINT,
    DECISION_MAX_REQUESTS_PER_SECOND,
    DECISION_MIN_OUTAGE_TOLERANCE_SECONDS,
    Backpressure,
    Config,
    ExternalProvider,
    FallbackEntry,
    Pool,
    PowerDemand,
    Replica,
)

log = logging.getLogger("sharedagents_vllm")
_warned_request_timeout_alias = False


class ConfigError(ValueError):
    """Router YAML is invalid."""


def _walk_no_literal_keys(obj: Any, path: str = "") -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            here = f"{path}.{key}" if path else str(key)
            if key == "api_key":
                raise ConfigError(f"forbidden {here!r}: put the env-var name in api_key_env, never a secret in YAML")
            _walk_no_literal_keys(value, here)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _walk_no_literal_keys(item, f"{path}[{i}]")


def _load_optional_env_file(path: Path, dest: dict[str, str]) -> None:
    """Merge KEY=value lines into dest. Existing dest keys win."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or key in dest:
            continue
        dest[key] = value.strip().strip('"').strip("'")


def resolve_external_keys(cfg: Config, environ: dict[str, str] | None = None) -> None:
    """Fill ``ExternalProvider.api_key`` from ``api_key_env``. Missing env disables.

    When ``environ`` is omitted, the process env is merged with optional
    ``$SHAREDAGENTS_HOME/secrets.env`` (same pattern as the old mux ``.secrets.env``).
    Tests pass an explicit mapping and skip the file.
    """
    if environ is not None:
        env = dict(environ)
    else:
        env = dict(os.environ)
        home = os.environ.get("SHAREDAGENTS_HOME") or os.path.expanduser("~/.sharedagents")
        _load_optional_env_file(Path(home) / "secrets.env", env)
    for provider in cfg.external_providers.values():
        name = (provider.api_key_env or "").strip()
        provider.api_key = str(env.get(name) or "").strip() if name else ""

    # A missing key is the one fallback failure nothing else reports: the hop is
    # simply skipped and the caller sees a local 503. Say it out loud, once.
    missing = [
        f"{provider.id} ({provider.api_key_env or 'api_key_env unset'})"
        for provider in cfg.external_providers.values()
        if not provider.api_key
    ]
    if missing:
        log.warning(
            "external fallback disabled — no API key in the environment for: %s. "
            "Put the key in $SHAREDAGENTS_HOME/secrets.env (or the unit's "
            "EnvironmentFile) and restart.",
            ", ".join(missing),
        )


def parse_config(raw: dict[str, Any] | None) -> Config:
    """Build a Config from an already-loaded mapping (tests use this)."""
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")
    _walk_no_literal_keys(raw)

    pools: dict[str, Pool] = {}
    raw_pools = raw.get("pools") or {}
    if not isinstance(raw_pools, dict):
        raise ConfigError("pools must be a mapping")
    for model_id, pdata in raw_pools.items():
        if not isinstance(pdata, dict):
            raise ConfigError(f"pool {model_id!r} must be a mapping")
        kind = str(pdata.get("kind") or "").strip()
        if not kind:
            raise ConfigError(f"pool {model_id!r} needs kind")
        reps = []
        for rdata in pdata.get("replicas") or []:
            if not isinstance(rdata, dict):
                raise ConfigError(f"pool {model_id!r} replica must be a mapping")
            name = str(rdata.get("name") or "").strip()
            url = str(rdata.get("url") or "").rstrip("/")
            if not name or not url:
                raise ConfigError(f"pool {model_id!r} replica needs name and url")
            _rum = rdata.get("upstream_model")
            reps.append(
                Replica(
                    name=name,
                    url=url,
                    max_active=int(rdata.get("max_active", 0) or 0),
                    upstream_model=(str(_rum).strip() or None) if _rum is not None else None,
                )
            )
        _ip = pdata.get("inference_probe")
        _pm = pdata.get("probe_model")
        fbs: list[FallbackEntry] = []
        for fb in pdata.get("fallbacks") or []:
            if not isinstance(fb, dict):
                raise ConfigError(f"pool {model_id!r} fallback must be a mapping")
            pool = str(fb.get("pool") or "").strip() or None
            external = str(fb.get("external") or "").strip() or None
            if bool(pool) == bool(external):
                raise ConfigError(
                    f"pool {model_id!r} fallback must be exactly one of "
                    "{{pool: <id>}} or {{external: <provider>}}"
                )
            fbs.append(FallbackEntry(pool=pool, external=external))
        _rt = pdata.get("read_timeout_seconds")
        _dec: dict[str, Any] = {}
        if kind == "decision":
            # Never let private decision payloads reach a hosted fallback or DNS
            # alias. Requiring a literal, non-public address also avoids DNS
            # rebinding — ``is_global`` is the property that actually matters
            # here, and stating it as a predicate keeps any concrete network
            # range out of the shipped source.
            import ipaddress
            from urllib.parse import urlsplit

            if fbs:
                raise ConfigError("decision pools must not have fallbacks")
            for replica in reps:
                url = urlsplit(replica.url)
                try:
                    addr = ipaddress.ip_address(url.hostname or "")
                except ValueError as exc:
                    raise ConfigError("decision replicas require literal LAN IPs") from exc
                if (
                    url.scheme != "http"
                    or url.username
                    or url.password
                    or url.query
                    or url.fragment
                    or url.path not in ("", "/")
                    or addr.is_global
                ):
                    raise ConfigError("decision replicas must be LAN-only HTTP endpoints")
            _dec = _decision_settings(str(model_id), pdata)
        _um = pdata.get("upstream_model")
        upstream_model = (str(_um).strip() or None) if _um is not None else None
        if kind == "decision" and (upstream_model or any(r.upstream_model for r in reps)):
            raise ConfigError("decision pools must not set upstream_model")
        routing = str(pdata.get("routing") or "affinity").strip()
        if routing not in ("affinity", "priority"):
            raise ConfigError(f"pool {model_id!r} routing must be affinity or priority")
        _ftt = pdata.get("first_token_timeout_seconds")
        _probe = pdata.get("inference_probe")
        _pm = pdata.get("probe_model")
        pools[str(model_id)] = Pool(
            model_id=str(model_id),
            kind=kind,
            replicas=reps,
            max_active_requests=int(pdata.get("max_active_requests", 0) or 0),
            fallbacks=fbs,
            read_timeout_seconds=(int(_rt) if _rt is not None else None),
            first_token_timeout_seconds=(int(_ftt) if _ftt is not None else None),
            inference_probe=(_as_bool(_probe) if _probe is not None else None),
            probe_model=(str(_pm).strip() or None) if _pm is not None else None,
            upstream_model=upstream_model,
            routing=routing,
            hidden=_as_bool(pdata.get("hidden", False)),
            power_servers=_power_servers(str(model_id), pdata.get("power_servers")),
            **_dec,
        )

    # Advisory decision pools are never another pool's fallback target: an
    # advisory scorer must not become a hidden dependency of a chat/embed/ASR
    # route, and it must never be reached by failover traffic.
    for _mid, _pool in pools.items():
        for _fb in _pool.fallbacks:
            if _fb.pool and _fb.pool in pools and pools[_fb.pool].kind == "decision":
                raise ConfigError(
                    f"pool {_mid!r} must not fall back to decision pool "
                    f"{_fb.pool!r}: advisory decision pools are never fallback "
                    "targets"
                )

    ext: dict[str, ExternalProvider] = {}
    raw_ext = raw.get("external_providers") or {}
    if raw_ext and not isinstance(raw_ext, dict):
        raise ConfigError("external_providers must be a mapping")
    for pid, edata in (raw_ext or {}).items():
        if not isinstance(edata, dict):
            raise ConfigError(f"external provider {pid!r} must be a mapping")
        base = str(edata.get("base_url") or "").rstrip("/")
        model = str(edata.get("model") or "").strip()
        if not base or not model:
            raise ConfigError(f"external provider {pid!r} needs base_url and model")
        ext[str(pid)] = ExternalProvider(
            id=str(pid),
            base_url=base,
            api_key_env=str(edata.get("api_key_env") or ""),
            model=model,
            max_active_requests=int(edata.get("max_active_requests", 10) or 0),
            timeout_seconds=int(edata.get("timeout_seconds", 120)),
            cost_per_input_token=float(edata.get("cost_per_input_token", 0.0) or 0.0),
            cost_per_output_token=float(edata.get("cost_per_output_token", 0.0) or 0.0),
        )

    power_demand = _power_demand(raw.get("power_demand"))
    if any(p.power_servers for p in pools.values()) and power_demand is None:
        raise ConfigError("pools with power_servers need a power_demand webhook")

    bp_raw = raw.get("backpressure") or {}
    if not isinstance(bp_raw, dict):
        raise ConfigError("backpressure must be a mapping")
    global _warned_request_timeout_alias
    if "request_timeout_seconds" in raw and not _warned_request_timeout_alias:
        log.warning(
            "request_timeout_seconds is deprecated; use write_pool_timeout_seconds "
            "(write/pool only, not an overall request deadline). "
            "write_pool_timeout_seconds takes precedence when both keys are set."
        )
        _warned_request_timeout_alias = True
    return Config(
        bind=str(raw.get("bind", "0.0.0.0")),
        port=int(raw.get("port", 8000)),
        health_check_seconds=int(raw.get("health_check_seconds", 10)),
        write_pool_timeout_seconds=int(raw.get("write_pool_timeout_seconds", raw.get("request_timeout_seconds", 3600))),
        connect_timeout_seconds=int(raw.get("connect_timeout_seconds", 5)),
        read_timeout_seconds=int(raw.get("read_timeout_seconds", 120)),
        first_token_timeout_seconds=int(raw.get("first_token_timeout_seconds", 300)),
        failure_cooldown_seconds=int(raw.get("failure_cooldown_seconds", 120)),
        session_ttl_seconds=float(raw.get("session_ttl_seconds", 900)),
        affinity_max=int(raw.get("affinity_max", 50000)),
        pools=pools,
        backpressure=Backpressure(
            reject_status=int(bp_raw.get("reject_status", 503)),
            retry_after_seconds=int(bp_raw.get("retry_after_seconds", 5)),
        ),
        external_providers=ext,
        power_demand=power_demand,
        health_timeout_seconds=int(raw.get("health_timeout_seconds", 3)),
        inference_probe_seconds=int(raw.get("inference_probe_seconds", 30)),
        inference_probe_timeout_seconds=int(raw.get("inference_probe_timeout_seconds", 30)),
        inference_probe_timeout_asr_tts_seconds=int(raw.get("inference_probe_timeout_asr_tts_seconds", 8)),
        inference_probe_enabled=_as_bool(raw.get("inference_probe_enabled"), True),
        health_fail_threshold=int(raw.get("health_fail_threshold", 2)),
        health_recover_threshold=int(raw.get("health_recover_threshold", 2)),
        health_soft_recover_threshold=int(raw.get("health_soft_recover_threshold", 3)),
        health_backoff_base_seconds=int(raw.get("health_backoff_base_seconds", 30)),
        health_backoff_cap_seconds=int(raw.get("health_backoff_cap_seconds", 600)),
        host_reboot_hint_seconds=int(raw.get("host_reboot_hint_seconds", 180)),
        model_load_grace_seconds=int(raw.get("model_load_grace_seconds", 180)),
        stuck_request_seconds=int(raw.get("stuck_request_seconds", 300)),
        error_rate_window_seconds=int(raw.get("error_rate_window_seconds", 60)),
        error_rate_fail_ratio=float(raw.get("error_rate_fail_ratio", 0.5)),
        health_strict_routing=_as_bool(raw.get("health_strict_routing"), True),
        host_restart_on_wedge=_as_bool(raw.get("host_restart_on_wedge"), False),
        latency_degraded_ms=int(raw.get("latency_degraded_ms", 15000)),
        generation_probe_seconds=float(raw.get("generation_probe_seconds", 15)),
        stream_silence_cap_seconds=float(raw.get("stream_silence_cap_seconds", 900)),
    )


def _decision_settings(model_id: str, pdata: dict[str, Any]) -> dict[str, Any]:
    """Advisory decision-pool tuning, validated against the gap-test constraints.

    See ``sharedagents_vllm/decision.py`` for why each bound exists. The defaults are
    the measured envelope of a serial scorer, so a minimal pool row is
    already correct.
    """
    checkpoint = str(pdata.get("checkpoint", DECISION_DEFAULT_CHECKPOINT) or "").strip()
    if checkpoint not in DECISION_CHECKPOINTS:
        raise ConfigError(
            f"pool {model_id!r} checkpoint must be one of "
            f"{', '.join(DECISION_CHECKPOINTS)} — 'auto' is not allowed because "
            "the scorer's workflow auto-detection is off, so 'auto' silently selects "
            "the english head for typed-decisions workflows"
        )

    settings: dict[str, Any] = {
        "checkpoint": checkpoint,
        "queue_timeout_seconds": float(pdata.get("queue_timeout_seconds", 10.0)),
        "queue_max_depth": int(pdata.get("queue_max_depth", 32)),
        "rate_limit_per_second": float(pdata.get("rate_limit_per_second", 1.5)),
        "connect_retry_seconds": float(pdata.get("connect_retry_seconds", 2.0)),
        "busy_retry_seconds": float(pdata.get("busy_retry_seconds", 3.0)),
        "outage_tolerance_seconds": float(pdata.get("outage_tolerance_seconds", 45.0)),
    }

    rate = settings["rate_limit_per_second"]
    if rate <= 0:
        raise ConfigError(f"pool {model_id!r} rate_limit_per_second must be > 0")
    if rate > DECISION_MAX_REQUESTS_PER_SECOND:
        raise ConfigError(
            f"pool {model_id!r} rate_limit_per_second={rate} exceeds the measured "
            f"ceiling of {DECISION_MAX_REQUESTS_PER_SECOND} req/s; the scorer is "
            "strictly serial and SHEDS (503) anything above it"
        )
    if settings["queue_max_depth"] < 1:
        raise ConfigError(f"pool {model_id!r} queue_max_depth must be >= 1")
    if settings["queue_timeout_seconds"] <= 0:
        raise ConfigError(f"pool {model_id!r} queue_timeout_seconds must be > 0")
    tolerance = settings["outage_tolerance_seconds"]
    if tolerance < DECISION_MIN_OUTAGE_TOLERANCE_SECONDS:
        raise ConfigError(
            f"pool {model_id!r} outage_tolerance_seconds={tolerance} is below "
            f"{DECISION_MIN_OUTAGE_TOLERANCE_SECONDS}s; a restart closes the port "
            "for 36-42 s while all three checkpoints load, so a shorter window "
            "would treat a normal restart as a real outage"
        )
    return settings


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def load_config(path: str | Path) -> Config:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ConfigError("PyYAML is required to load router YAML; install sharedagents[vllm-router]") from exc
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config {p}: {exc}") from exc
    raw = yaml.safe_load(text)
    return parse_config(raw)


def packaged_example_path() -> Path:
    """Shipped empty template. Not a live fallback — copy it, then edit."""
    return Path(__file__).resolve().parent / "data" / "vllm-router.example.yaml"


def default_config_path() -> Path:
    """Operator YAML only: ``$SHAREDAGENTS_VLLM_CONFIG`` or ``$SHAREDAGENTS_HOME/vllm-router.yaml``.

    Missing files stay missing. The packaged example is an empty template
    (see ``packaged_example_path``), never substituted as live topology.
    """
    env = os.environ.get("SHAREDAGENTS_VLLM_CONFIG")
    if env:
        return Path(env)
    home = os.environ.get("SHAREDAGENTS_HOME") or os.path.expanduser("~/.sharedagents")
    return Path(home) / "vllm-router.yaml"


def _power_servers(model_id: str, value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        raise ConfigError(f"pool {model_id!r} power_servers must be a list of server names")
    return [v.strip() for v in value]


def _power_demand(value: Any) -> PowerDemand | None:
    """``power_demand: {url, token_env, ttl_seconds: 900, min_interval_seconds: 60}``.
    The token is read from the named environment variable, never from this file."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ConfigError("power_demand must be a mapping")
    if value.keys() - {"url", "token_env", "ttl_seconds", "min_interval_seconds", "timeout_seconds"}:
        raise ConfigError("power_demand accepts only url, token_env and timing settings; never inline tokens")
    url, token_env = value.get("url"), value.get("token_env")
    if not isinstance(url, str) or not isinstance(token_env, str):
        raise ConfigError("power_demand needs an http(s) url and token_env")
    url, token_env = url.strip(), token_env.strip()
    from urllib.parse import urlsplit

    try:
        parsed = urlsplit(url)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port == 0
            or any(c.isspace() for c in url)
        ):
            raise ValueError
    except ValueError as exc:
        raise ConfigError("power_demand url must be an http(s) endpoint without credentials") from exc
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token_env) is None:
        raise ConfigError("power_demand token_env must be an environment variable name")
    numbers = {}
    for name, default in (("ttl_seconds", 900), ("min_interval_seconds", 60), ("timeout_seconds", 5)):
        raw = value.get(name, default)
        try:
            number = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ConfigError(f"power_demand {name} must be a finite number") from exc
        if isinstance(raw, bool) or not math.isfinite(number):
            raise ConfigError(f"power_demand {name} must be a finite number")
        numbers[name] = number
    ttl, interval, timeout = (numbers[n] for n in ("ttl_seconds", "min_interval_seconds", "timeout_seconds"))
    if not ttl.is_integer() or not 1 <= ttl <= 86400 or interval < 0 or timeout <= 0:
        raise ConfigError("power_demand needs integer ttl_seconds 1..86400, interval >= 0 and timeout > 0")
    # A previous response can occupy the sole worker for a full timeout after
    # the lease was accepted; the next renewal needs its own timeout as well.
    if max(interval, timeout) + timeout >= ttl:
        raise ConfigError(
            "power_demand max(min_interval_seconds, timeout_seconds) + timeout_seconds must be below ttl_seconds"
        )
    return PowerDemand(
        url=url, token_env=token_env, ttl_seconds=int(ttl), min_interval_seconds=interval, timeout_seconds=timeout
    )
