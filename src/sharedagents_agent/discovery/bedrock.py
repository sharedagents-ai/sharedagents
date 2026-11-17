"""Discover native Bedrock routes from ambient AWS credentials.

Only catalogue reads are made. No inference, credential persistence, process-wide
AWS environment mutation, or implicit instance metadata probing is performed.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from sharedagents_model.bedrock_config import (
    AUTH_API_KEY,
    AUTH_IAM,
    BEDROCK_API,
    BEDROCK_BEARER_ENV,
    is_valid_bedrock_region,
    validate_bedrock_row,
)

_LOG = logging.getLogger(__name__)
DEFAULT_REGION = "sa-east-1"
DISCOVERED_BY = "bedrock-environment"
_CACHE_TTL = 300.0
_FAILURE_TTL = 60.0
_LOCK = threading.Lock()
_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_ENV_KEYS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    BEDROCK_BEARER_ENV,
    "AWS_PROFILE",
    "AWS_DEFAULT_PROFILE",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "AWS_CONFIG_FILE",
    "AWS_SHARED_CREDENTIALS_FILE",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_ROLE_ARN",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
)


def _credential_context() -> str | None:
    """Cheap gate and in-memory cache identity; never contact EC2 metadata."""
    env = os.environ
    files = [
        Path(env.get("AWS_CONFIG_FILE") or "~/.aws/config").expanduser(),
        Path(env.get("AWS_SHARED_CREDENTIALS_FILE") or "~/.aws/credentials").expanduser(),
    ]
    stamps = []
    for path in files:
        try:
            stat = path.stat()
            stamps.append((str(path), stat.st_mtime_ns, stat.st_size))
        except OSError:
            pass
    enabled = (
        env.get(BEDROCK_BEARER_ENV)
        or (env.get("AWS_ACCESS_KEY_ID") and env.get("AWS_SECRET_ACCESS_KEY"))
        or env.get("AWS_PROFILE")
        or env.get("AWS_DEFAULT_PROFILE")
        or (env.get("AWS_WEB_IDENTITY_TOKEN_FILE") and env.get("AWS_ROLE_ARN"))
        or env.get("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI")
        or env.get("AWS_CONTAINER_CREDENTIALS_FULL_URI")
        or stamps
    )
    if not enabled:
        return None
    # The digest remains in memory only; credentials never enter rows or logs.
    raw = repr(([env.get(key, "") for key in _ENV_KEYS], stamps))
    return hashlib.sha256(raw.encode()).hexdigest()


def _client() -> tuple[Any, str, str]:
    import boto3
    import botocore
    from botocore.config import Config

    for sdk in (boto3, botocore):
        version = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", sdk.__version__)
        if not version or not ((1, 43, 101) <= tuple(map(int, version.groups())) < (2, 0, 0)):
            raise RuntimeError("Bedrock discovery requires boto3/botocore >=1.43.101,<2")
    session = boto3.Session()
    region = next(
        (
            str(value).strip()
            for value in (
                os.environ.get("AWS_REGION"),
                os.environ.get("AWS_DEFAULT_REGION"),
                session.region_name,
                DEFAULT_REGION,
            )
            if value and str(value).strip()
        ),
        DEFAULT_REGION,
    )
    if not is_valid_bedrock_region(region):
        raise ValueError("Invalid AWS region for Bedrock discovery")
    auth = AUTH_API_KEY if os.environ.get(BEDROCK_BEARER_ENV) else AUTH_IAM
    if auth == AUTH_IAM:
        # Do not probe instance metadata merely because ~/.aws/config exists.
        # Explicit container/web-identity/profile providers remain SDK-managed.
        resolver = session._session.get_component("credential_provider")
        resolver.remove("iam-role")
        if session.get_credentials() is None:
            raise RuntimeError("No AWS credentials for Bedrock discovery")
    return (
        session.client(
            "bedrock",
            region_name=region,
            config=Config(
                signature_version="bearer" if auth == AUTH_API_KEY else "v4",
                connect_timeout=2,
                read_timeout=3,
                retries={"total_max_attempts": 1},
            ),
        ),
        region,
        auth,
    )


def _text_model(model: dict[str, Any]) -> bool:
    return (
        "TEXT" in (model.get("inputModalities") or [])
        and "TEXT" in (model.get("outputModalities") or [])
        and model.get("responseStreamingSupported") is True
        and (model.get("modelLifecycle") or {}).get("status") == "ACTIVE"
    )


def _row(model_id: str, name: str, region: str, auth: str) -> dict[str, Any]:
    # The AWS catalogue does not publish context/output limits. The context
    # window is a conservative operating budget, not a claim about the model.
    # No ``maxTokens``: on a row it is a live output cap (the operator sets
    # it), and a placeholder would clamp every discovered model's replies.
    row: dict[str, Any] = {
        "alias": f"bedrock-{model_id}",
        "id": model_id,
        "display": name,
        "api": BEDROCK_API,
        "backend": "bedrock",
        "provider": "bedrock",
        "baseUrl": f"bedrock://{region}",
        "region": region,
        "auth": auth,
        "contextWindow": 4096,
        "discoveredBy": DISCOVERED_BY,
    }
    if auth == AUTH_API_KEY:
        row["apiKeyEnv"] = BEDROCK_BEARER_ENV
    validate_bedrock_row(row)
    return row


def _list_models(client: Any, region: str, auth: str) -> list[dict[str, Any]]:
    summaries = client.list_foundation_models(byOutputModality="TEXT").get("modelSummaries", [])
    text_models = {m["modelId"]: m for m in summaries if m.get("modelId") and _text_model(m)}
    rows = {
        mid: _row(mid, m.get("modelName") or mid, region, auth)
        for mid, m in text_models.items()
        if "ON_DEMAND" in (m.get("inferenceTypesSupported") or [])
    }
    # A profile can be the only invocable route for a model. Keep it even
    # when its foundation model is not advertised for direct on-demand use.
    token = None
    seen_tokens: set[str] = set()
    try:
        for _ in range(20):  # Bounded even if an upstream pagination bug loops.
            page = client.list_inference_profiles(maxResults=1000, **({"nextToken": token} if token else {}))
            for profile in page.get("inferenceProfileSummaries", []):
                model_ids = [
                    str(m.get("modelArn") or "").split("foundation-model/", 1)[-1] for m in profile.get("models", [])
                ]
                if (
                    profile.get("status") != "ACTIVE"
                    or not model_ids
                    or not all(mid in text_models for mid in model_ids)
                ):
                    continue
                mid = profile.get("inferenceProfileId")
                if mid:
                    rows[mid] = _row(mid, profile.get("inferenceProfileName") or mid, region, auth)
            token = page.get("nextToken")
            if not token or token in seen_tokens:
                break
            seen_tokens.add(token)
    except Exception as exc:  # A list-profile denial must not hide direct models.
        _LOG.info("Bedrock inference profile discovery unavailable (%s)", type(exc).__name__)
    return list(rows.values())


def discover_bedrock_models(*, force: bool = False) -> list[dict[str, Any]]:
    """Short-lived process cache shared by native agents; optional SDK stays lazy."""
    key = _credential_context()
    if key is None:
        return []
    with _LOCK:
        now = time.monotonic()
        cached = _CACHE.get(key)
        if not force and cached and now < cached[0]:
            return copy.deepcopy(cached[1])
        client = None
        ttl = _CACHE_TTL
        try:
            client, region, auth = _client()
            rows = _list_models(client, region, auth)
        except Exception as exc:
            # Never log exception bodies: SDK/provider errors can contain secrets.
            _LOG.warning(
                "Bedrock discovery unavailable (%s); check AWS credentials, "
                "catalogue permissions and the sharedagents[bedrock] extra",
                type(exc).__name__,
            )
            rows = []
            ttl = _FAILURE_TTL
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:  # Closing idle connections must not discard rows.
                    pass
        _CACHE.clear()
        _CACHE[key] = (time.monotonic() + ttl, rows)
        return copy.deepcopy(rows)


def bedrock_credentials_present() -> bool:
    """Whether ambient AWS credentials exist (a file/env check, no network)."""
    return _credential_context() is not None


def cached_bedrock_models() -> list[dict[str, Any]]:
    """The last Bedrock catalogue this process read, expired or not.

    Never contacts AWS: the picker summary uses it to name a Bedrock default
    model; the provider's own request refreshes the catalogue.
    """
    key = _credential_context()
    if key is None:
        return []
    with _LOCK:
        cached = _CACHE.get(key)
        return copy.deepcopy(cached[1]) if cached else []


def sync_bedrock_models_into_cfg(cfg: Any, *, force: bool = False) -> list[dict[str, Any]]:
    """Merge at load/reload/refresh, preserving operator rows and other engines."""
    existing = list(getattr(cfg, "models", None) or [])
    if str(getattr(cfg, "engine", "") or "local_llm").strip() != "local_llm":
        return existing
    existing = [r for r in existing if r.get("discoveredBy") != DISCOVERED_BY]
    extra = discover_bedrock_models(force=force)
    seen = {(r.get("baseUrl"), r.get("id")) for r in existing}
    aliases = {r.get("alias") for r in existing}
    for row in extra:
        if (row["baseUrl"], row["id"]) in seen:
            continue
        # An explicit per-agent bearer override belongs to a different account
        # or blocks ambient credentials. Its catalogue must be configured there.
        keys = getattr(cfg, "api_keys", {}) or {}
        if (
            row["auth"] == AUTH_API_KEY
            and BEDROCK_BEARER_ENV in keys
            and keys[BEDROCK_BEARER_ENV] != os.environ.get(BEDROCK_BEARER_ENV)
        ):
            continue
        base = row["alias"]
        suffix = 2
        while row["alias"] in aliases:
            row["alias"] = f"{base}-{suffix}"
            suffix += 1
        aliases.add(row["alias"])
        seen.add((row["baseUrl"], row["id"]))
        existing.append(row)
    cfg.models = existing
    return existing
