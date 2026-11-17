"""Auto-detect prepaid API keys and probe vendor balance endpoints.

Only providers with a local key are probed. Keys are read from the process
environment, the runtime root's ``.env`` (``$SHAREDAGENTS_HOME``-aware via
``sharedagents_root()``), and ``/etc/environment``; values are never
written to the snapshot.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from sharedagents_core.layout import sharedagents_root
from sharedagents_web.subscription_limits import _as_float, _provider, _window

HttpFn = Callable[[str, str, dict[str, str], bytes | None, float], tuple[int, dict[str, Any]]]

_HTTP_TIMEOUT_S = 12.0
_BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
_XAI_MGMT_ENV = ("XAI_MANAGEMENT_API_KEY", "XAI_MGMT_KEY", "XAI_MANAGEMENT_KEY")


def _env_files() -> tuple[Path, ...]:
    """Env-file fallback list, resolved lazily so an isolated home is honored.

    ``sharedagents_root()`` is the single source of truth for the runtime-tree
    root (``$SHAREDAGENTS_AGENTS_HOME`` > ``$SHAREDAGENTS_HOME`` > ``~/.sharedagents``), so
    an isolated dev instance reads ITS OWN ``.env`` instead of silently
    loading the operator's. ``/etc/environment`` stays the system-wide
    fallback. Computed per call (not at import time) so tests and late
    ``SHAREDAGENTS_HOME`` changes are respected.
    """
    return (sharedagents_root() / ".env", Path("/etc/environment"))


# Detected from the same env names the first-run provider presets already use.
API_PROVIDERS: tuple[dict[str, Any], ...] = (
    {
        "id": "openrouter",
        "name": "OpenRouter",
        "env": ("OPENROUTER_API_KEY",),
        "kind": "balance",
    },
    {
        "id": "deepseek",
        "name": "DeepSeek",
        "env": ("DEEPSEEK_API_KEY",),
        "kind": "balance",
    },
    {
        "id": "kimi",
        "name": "Kimi",
        "env": ("KIMI_API_KEY", "MOONSHOT_API_KEY"),
        "kind": "balance",
    },
    {
        "id": "xai",
        "name": "xAI",
        "env": ("XAI_API_KEY",) + _XAI_MGMT_ENV,
        "kind": "balance",
    },
    {
        "id": "cerebras",
        "name": "Cerebras",
        "env": ("CEREBRAS_API_KEY",),
        "kind": "balance",
    },
)


def detected_api_keys() -> dict[str, str]:
    """Return {provider_id: env_var_name} for every configured API key.

    Values are not returned.
    """
    found: dict[str, str] = {}
    for spec in API_PROVIDERS:
        for name in spec["env"]:
            if _env(name):
                found[str(spec["id"])] = name
                break
    return found


def catalog_api_providers() -> list[dict[str, str]]:
    """Return display rows for every API provider that has a local key."""
    rows: list[dict[str, str]] = []
    for spec in API_PROVIDERS:
        if not any(_env(name) for name in spec["env"]):
            continue
        rows.append(
            {
                "id": str(spec["id"]),
                "name": str(spec["name"]),
                "kind": str(spec["kind"]),
            }
        )
    return rows


def probe_api_balances(*, http: HttpFn | None = None) -> list[dict[str, Any]]:
    """Probe only providers that have a local key, in parallel."""
    specs = [spec for spec in API_PROVIDERS if any(_env(name) for name in spec["env"])]
    if not specs:
        return []
    if len(specs) == 1:
        return [_probe_spec(specs[0], http)]
    cards: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(8, len(specs))) as pool:
        futures = [pool.submit(_probe_spec, spec, http) for spec in specs]
        cards = [future.result() for future in futures]
    return cards


def probe_api_provider(provider_id: str, *, http: HttpFn | None = None) -> dict[str, Any]:
    """Probe a single prepaid API provider."""
    for spec in API_PROVIDERS:
        if str(spec["id"]) == provider_id:
            if not any(_env(name) for name in spec["env"]):
                return _provider(provider_id, plan="API", error=f"{provider_id} API key is not configured")
            return _probe_spec(spec, http)
    return _provider(provider_id, error="Unknown API provider")


def parse_openrouter_credits(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    total = _as_float(data.get("total_credits"))
    used = _as_float(data.get("total_usage"))
    remaining = None
    if total is not None and used is not None:
        remaining = round(total - used, 6)
    percent = None
    if total:
        percent = round(min(100.0, max(0.0, (used or 0.0) / total * 100.0)), 1)
    return _provider(
        "openrouter",
        plan="Pay-as-you-go",
        windows=[
            _window(
                "credits",
                "Available",
                percent,
                None,
                used=used,
                limit=total,
                remaining=remaining,
                unit="USD",
            )
        ],
    )


def parse_deepseek_balance(payload: dict[str, Any]) -> dict[str, Any]:
    infos = payload.get("balance_infos") if isinstance(payload.get("balance_infos"), list) else []
    usd = next((row for row in infos if isinstance(row, dict) and row.get("currency") == "USD"), None)
    row = usd or (infos[0] if infos and isinstance(infos[0], dict) else {})
    remaining = _as_float(row.get("total_balance"))
    topped = _as_float(row.get("topped_up_balance"))
    granted = _as_float(row.get("granted_balance"))
    return _provider(
        "deepseek",
        plan="API",
        windows=[
            _window(
                "balance",
                "Balance",
                None,
                None,
                used=None,
                limit=None,
                remaining=remaining,
                unit="USD",
            ),
            _window(
                "topped_up",
                "Topped up",
                None,
                None,
                remaining=topped,
                unit="USD",
            ),
            _window(
                "granted",
                "Granted",
                None,
                None,
                remaining=granted,
                unit="USD",
            ),
        ],
    )


def parse_kimi_balance(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    available = _as_float(data.get("available_balance"))
    cash = _as_float(data.get("cash_balance"))
    voucher = _as_float(data.get("voucher_balance"))
    return _provider(
        "kimi",
        plan="API",
        windows=[
            _window("available", "Available", None, None, remaining=available, unit="USD"),
            _window("cash", "Cash", None, None, remaining=cash, unit="USD"),
            _window("voucher", "Voucher", None, None, remaining=voucher, unit="USD"),
        ],
    )


def parse_xai_prepaid(payload: dict[str, Any], *, unsettled_usage: float = 0.0) -> dict[str, Any]:
    """Combine the prepaid ledger with usage not yet posted to that ledger."""
    total = payload.get("total") if isinstance(payload.get("total"), dict) else {}
    remaining = _cents_to_usd_abs(total.get("val"))
    if remaining is None:
        remaining = _first_usd(
            payload.get("remaining"),
            payload.get("balance"),
            payload.get("prepaid_balance"),
        )
    used = 0.0
    purchased = 0.0
    changes = payload.get("changes") if isinstance(payload.get("changes"), list) else []
    for change in changes:
        if not isinstance(change, dict):
            continue
        origin = str(change.get("changeOrigin") or change.get("change_origin") or "")
        amount = change.get("amount") if isinstance(change.get("amount"), dict) else {}
        usd = _cents_to_usd_abs(amount.get("val") if amount else change.get("amount"))
        if usd is None:
            continue
        if origin == "SPEND":
            used += usd
        elif origin in {"PURCHASE", "AUTO_PURCHASE", "REFUND", "MANUAL"}:
            purchased += usd
    if remaining is not None:
        remaining = max(0.0, round(remaining - unsettled_usage, 2))
    used = round(used + unsettled_usage, 2) if used or unsettled_usage else None
    purchased = round(purchased, 2) if purchased else None
    if remaining is None and used is None and purchased is None:
        raise ValueError("xAI prepaid payload had no dollar amounts")
    limit = None
    percent = None
    if remaining is not None and used is not None:
        limit = round(remaining + used, 2)
        if limit:
            percent = round(min(100.0, max(0.0, used / limit * 100.0)), 1)
    windows = [
        _window(
            "credits",
            "Available",
            percent,
            None,
            used=used,
            limit=limit,
            remaining=remaining,
            unit="USD",
        )
    ]
    if purchased is not None:
        windows.append(_window("purchased", "Purchased", None, None, remaining=purchased, unit="USD"))
    return _provider("xai", plan="Prepaid", windows=windows)


def parse_cerebras_balance(payload: dict[str, Any]) -> dict[str, Any]:
    remaining = _extract_usd(payload)
    if remaining is None:
        raise ValueError("Cerebras payload had no dollar balance")
    used = _first_usd(
        payload.get("used"),
        payload.get("spent"),
        payload.get("usage"),
        (payload.get("data") or {}).get("used") if isinstance(payload.get("data"), dict) else None,
    )
    limit = _first_usd(
        payload.get("limit"),
        payload.get("total"),
        (payload.get("data") or {}).get("limit") if isinstance(payload.get("data"), dict) else None,
    )
    percent = None
    if used is not None and limit:
        percent = round(min(100.0, max(0.0, used / limit * 100.0)), 1)
    return _provider(
        "cerebras",
        plan="Pay-as-you-go",
        windows=[
            _window(
                "credits",
                "Available",
                percent,
                None,
                used=used,
                limit=limit,
                remaining=remaining,
                unit="USD",
            )
        ],
    )


def _probe_spec(spec: dict[str, Any], http: HttpFn | None) -> dict[str, Any]:
    key = ""
    for name in spec["env"]:
        key = _env(name)
        if key:
            break
    try:
        return _fetch(spec, key, http)
    except Exception as exc:  # noqa: BLE001
        return _provider(str(spec["id"]), plan="API", error=str(exc) or exc.__class__.__name__)


def _fetch(spec: dict[str, Any], key: str, http: HttpFn | None) -> dict[str, Any]:
    provider_id = str(spec["id"])
    if provider_id == "openrouter":
        status, payload = _http(http, "GET", "https://openrouter.ai/api/v1/credits", _headers(key), None)
        if status >= 400:
            raise RuntimeError(f"OpenRouter credits HTTP {status}")
        return parse_openrouter_credits(payload)
    if provider_id == "deepseek":
        status, payload = _http(http, "GET", "https://api.deepseek.com/user/balance", _headers(key), None)
        if status >= 400:
            raise RuntimeError(f"DeepSeek balance HTTP {status}")
        return parse_deepseek_balance(payload)
    if provider_id == "kimi":
        status, payload = _http(http, "GET", "https://api.moonshot.ai/v1/users/me/balance", _headers(key), None)
        if status == 401:
            status, payload = _http(http, "GET", "https://api.moonshot.cn/v1/users/me/balance", _headers(key), None)
        if status >= 400:
            raise RuntimeError(f"Kimi balance HTTP {status}")
        return parse_kimi_balance(payload)
    if provider_id == "xai":
        return _fetch_xai(http)
    if provider_id == "cerebras":
        return _fetch_cerebras(key, http)
    raise RuntimeError(f"Unknown API provider {provider_id}")


def _fetch_xai(http: HttpFn | None) -> dict[str, Any]:
    inference = _env("XAI_API_KEY")
    management = ""
    for name in _XAI_MGMT_ENV:
        management = _env(name)
        if management:
            break
    team_id = (_env("XAI_TEAM_ID") or "").strip()
    blocked = False
    if inference:
        status, payload = _http(http, "GET", "https://api.x.ai/v1/api-key", _headers(inference), None)
        if status >= 400:
            raise RuntimeError(f"xAI api-key HTTP {status}")
        blocked = bool(payload.get("team_blocked") or payload.get("api_key_blocked"))
        team_id = team_id or str(payload.get("team_id") or "").strip()
    if management and team_id:
        status, payload = _http(
            http,
            "GET",
            f"https://management-api.x.ai/v1/billing/teams/{team_id}/prepaid/balance",
            _headers(management),
            None,
        )
        if status >= 400:
            raise RuntimeError(f"xAI prepaid HTTP {status}")
        unsettled_usage = _xai_unsettled_usage(http, management, team_id, payload)
        card = parse_xai_prepaid(payload, unsettled_usage=unsettled_usage)
        if blocked:
            card["note"] = "Team/key is blocked."
        return card
    if inference:
        note = "Prepaid $ needs an xAI management key (Settings → Management Keys)."
        if blocked:
            note += " Team/key is blocked."
        return _signed_in_card("xai", "API", note)
    raise RuntimeError("xAI API key is not configured")


def _xai_unsettled_usage(http: HttpFn | None, management: str, team_id: str, balance: dict[str, Any]) -> float:
    """Prepaid spend the ledger has not posted as a SPEND row yet.

    xAI posts each month's SPEND days after the month closes, so the ledger
    total lags the console. The open month comes from the invoice preview
    (``prepaidCreditsUsed``, the figure the console subtracts); a closed month
    whose SPEND has not posted yet is read from the usage API.
    """
    status, preview = _http(
        http,
        "GET",
        f"https://management-api.x.ai/v1/billing/teams/{team_id}/postpaid/invoice/preview",
        _headers(management),
        None,
    )
    if status >= 400:
        raise RuntimeError(f"xAI invoice preview HTTP {status}")
    cycle = preview.get("billingCycle") if isinstance(preview.get("billingCycle"), dict) else {}
    invoice = preview.get("coreInvoice") if isinstance(preview.get("coreInvoice"), dict) else {}
    used = invoice.get("prepaidCreditsUsed")
    open_used = _cents_to_usd_abs(used.get("val") if isinstance(used, dict) else used)
    try:
        open_month = datetime(int(cycle["year"]), int(cycle["month"]), 1, tzinfo=timezone.utc)
    except (KeyError, TypeError, ValueError):
        open_month = None
    if open_used is None or open_month is None:
        raise ValueError("xAI invoice preview has no prepaid usage for the open month")

    start = _xai_first_unposted_time(balance, open_month)
    if start is None:
        return 0.0
    if start >= open_month:
        return open_used
    closed = _xai_usage_between(http, management, team_id, start, open_month)
    return max(0.0, round(open_used + closed, 2))


def _xai_first_unposted_time(balance: dict[str, Any], open_month: datetime) -> datetime | None:
    """Start of the first month not covered by a posted SPEND (or the first credit).

    Returns ``None`` when a posted SPEND already covers the open month: that
    usage is in the ledger total already, so nothing is unsettled.
    """
    changes = balance.get("changes") if isinstance(balance.get("changes"), list) else []
    rows = [row for row in changes if isinstance(row, dict)]
    spends = [row for row in rows if (row.get("changeOrigin") or row.get("change_origin")) == "SPEND"]
    if spends:
        try:
            year, month = max((int(row["spendBpKeyYear"]), int(row["spendBpKeyMonth"])) for row in spends)
        except (KeyError, TypeError, ValueError):
            raise ValueError("xAI prepaid SPEND has no billing month") from None
        if (year, month) >= (open_month.year, open_month.month):
            return None
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
        return datetime(year, month, 1, tzinfo=timezone.utc)
    dated = [str(row.get("createTime") or row.get("createTs") or "") for row in rows]
    dated = [value for value in dated if value]
    if not dated:
        raise ValueError("xAI prepaid ledger has no dated changes to reconcile with usage")
    return datetime.fromisoformat(min(dated).replace("Z", "+00:00")).astimezone(timezone.utc)


def _xai_usage_between(http: HttpFn | None, management: str, team_id: str, start: datetime, end: datetime) -> float:
    request = {
        "analyticsRequest": {
            "timeRange": {
                "startTime": start.strftime("%Y-%m-%d %H:%M:%S"),
                "endTime": end.strftime("%Y-%m-%d %H:%M:%S"),
                "timezone": "Etc/GMT",
            },
            "timeUnit": "TIME_UNIT_DAY",
            "values": [{"name": "usd", "aggregation": "AGGREGATION_SUM"}],
            "groupBy": [],
            "filters": [],
        }
    }
    status, usage = _http(
        http,
        "POST",
        f"https://management-api.x.ai/v1/billing/teams/{team_id}/usage",
        {**_headers(management), "Content-Type": "application/json"},
        json.dumps(request).encode("utf-8"),
    )
    if status >= 400:
        raise RuntimeError(f"xAI usage HTTP {status}")
    series = usage.get("timeSeries")
    if usage.get("limitReached") or not isinstance(series, list):
        raise ValueError("xAI usage response is incomplete")
    amount = 0.0
    for row in series:
        if not isinstance(row, dict) or not isinstance(row.get("dataPoints"), list):
            raise ValueError("xAI usage response is malformed")
        for point in row["dataPoints"]:
            values = point.get("values") if isinstance(point, dict) else None
            if not isinstance(values, list):
                raise ValueError("xAI usage response is malformed")
            for value in values:
                usd = _as_float(value)
                if usd is None:
                    raise ValueError("xAI usage response has an invalid amount")
                amount += usd
    return max(0.0, round(amount, 2))


def _fetch_cerebras(key: str, http: HttpFn | None) -> dict[str, Any]:
    """Validate the inference key. Prepaid $ is console-only (no public API)."""
    status, _models = _http(http, "GET", "https://api.cerebras.ai/v1/models", _headers(key, browser=True), None)
    if status >= 400:
        raise RuntimeError(f"Cerebras models HTTP {status}")
    return _console_card(
        "cerebras",
        "Console",
        "No public credit-balance API. Remaining $ is on the Cloud Console Overview tab.",
        "https://cloud.cerebras.ai",
    )


def _signed_in_card(provider_id: str, plan: str, note: str) -> dict[str, Any]:
    card = _provider(
        provider_id,
        plan=plan,
        windows=[_window("key", "API key", None, None, remaining=None)],
    )
    card["note"] = note
    return card


def _console_card(provider_id: str, plan: str, note: str, console_url: str) -> dict[str, Any]:
    card = _signed_in_card(provider_id, plan, note)
    card["console_url"] = console_url
    return card


def _headers(key: str, *, browser: bool = False) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
    }
    if browser:
        headers["User-Agent"] = _BROWSER_UA
    return headers


def _env(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if value:
        return value
    prefix = name + "="
    for path in _env_files():
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#") or not stripped.startswith(prefix):
                continue
            return stripped.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _http(
    http: HttpFn | None,
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None,
) -> tuple[int, dict[str, Any]]:
    if http is not None:
        return http(method, url, headers, body, _HTTP_TIMEOUT_S)
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=_HTTP_TIMEOUT_S) as response:
            payload = json.loads(response.read().decode("utf-8") or "{}")
            return int(response.status), payload if isinstance(payload, dict) else {}
    except HTTPError as exc:
        raw = exc.read() if exc.fp else b""
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except Exception:  # noqa: BLE001
            payload = {}
        return int(exc.code), payload if isinstance(payload, dict) else {}


def _cents_to_usd_abs(value: Any) -> float | None:
    number = _as_float(value)
    if number is None:
        return None
    return round(abs(number) / 100.0, 2)


def _first_usd(*values: Any) -> float | None:
    for value in values:
        if isinstance(value, dict):
            nested = _first_usd(
                value.get("val"),
                value.get("amount"),
                value.get("usd"),
                value.get("remaining"),
                value.get("balance"),
            )
            if nested is not None:
                return nested
            continue
        number = _as_float(value)
        if number is not None:
            return number
    return None


def _extract_usd(payload: dict[str, Any]) -> float | None:
    if not isinstance(payload, dict):
        return None
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    return _first_usd(
        data.get("current_balance"),
        data.get("currentBalance"),
        data.get("available_balance"),
        data.get("availableBalance"),
        data.get("remaining_credits"),
        data.get("remainingCredits"),
        data.get("credits_remaining"),
        data.get("balance"),
        data.get("credits"),
        data.get("remaining"),
        payload.get("current_balance"),
        payload.get("balance"),
        payload.get("credits"),
        payload.get("remaining"),
    )
