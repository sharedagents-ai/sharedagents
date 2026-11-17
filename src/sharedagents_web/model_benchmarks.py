"""Artificial Analysis benchmark data for the WebUI "Models" tab.

Data source
-----------
``GET https://artificialanalysis.ai/api/v2/language/models/free`` — the Free
tier of the Artificial Analysis API, authenticated with an ``x-api-key``
header resolved from ``ARTIFICIAL_ANALYSIS_API_KEY`` (process env → the
runtime root's ``.env`` (``$SHAREDAGENTS_HOME``-aware via ``sharedagents_root()``) →
``/etc/environment``, the same precedence and plain manual line read as
:mod:`sharedagents_web.api_balances`). A machine without a usable key gets a
structured ``no_api_key`` payload from :func:`fetch_benchmarks` — never an
exception, never a 500.

Licence / attribution
---------------------
The Free tier is "internal use only with attribution" and the site ToS forbids
scraping, so this module only ever talks to the documented JSON API — it never
fetches, renders, or headless-browses an artificialanalysis.ai page. Every
payload carries :data:`ATTRIBUTION` (``"Data: Artificial Analysis"``) and
:data:`ATTRIBUTION_URL` so the UI can render the required credit.

Data flow / request budget
--------------------------
:func:`fetch_benchmarks` is cache-first: the normalised payload is stored in
``usage_root()/benchmarks.json`` (override ``SHAREDAGENTS_BENCHMARKS_CACHE``) with a
6-hour TTL (override ``SHAREDAGENTS_BENCHMARKS_TTL_S``). The network is touched only
when the cache is missing/expired or the caller explicitly asks for a refresh —
there is deliberately no per-render or per-poll fetch. The live API paginates
200 rows per page (653 rows / 4 pages at the time of writing, and the count
grows), so one refresh costs one request per page; :func:`_fetch_all_pages`
walks ``?page=N`` until ``pagination.has_more`` is false, with a generous
safety cap rather than a hardcoded page count.

``SHAREDAGENTS_BENCHMARKS_FIXTURE=1`` short-circuits the whole thing with a
deterministic AA-shaped fixture that touches no network and writes no cache
(tests and headed verification use it — see :func:`fixture_payload`).

Null metrics
------------
Every metric field is independently nullable, and on the live 653-row
population they are *very* nullable (speed ~50%, cost-per-task ~78%,
price ~33% at the time of writing). Nulls are therefore preserved as ``None``
end-to-end — never coerced to ``0``, never sorted as lowest — and the
intermediate containers themselves may be null
(``"artificial_analysis_intelligence_index_cost": null``), so every level of
the walk goes through :func:`_mapping`, which returns ``{}`` for anything that
is not a mapping. Charts render only rows that carry the charted metric and
report the skipped count.

Units are never mixed: ``cost_per_task_usd`` is USD per Intelligence-Index
*task* while ``usd_per_1m_in`` / ``usd_per_1m_out`` are USD per *million
tokens*. They are separate fields and are not plotted on a shared axis.

Enabled models
--------------
"Enabled" means "sharedagents can select this model today": the union of the global
catalogue (``~/.sharedagents/config.json`` → ``models[]``), the per-engine
``external_models`` rows (unioned with the shipped catalogues via
``sharedagents_agent.discovery._common._fallback_models``), and the discovery cache
(``~/.sharedagents/discovered_models.json`` via
``sharedagents_agent.discovery.cache_lock.read_all_cached_models``). All three are
read-only here; unreadable files degrade to "no enabled models" rather than an
 error. ``enabled`` is re-derived on every read (cache hits included) so the
 catalogue can change without waiting out the TTL.

Local deployments
-----------------
``local`` means "sharedagents serves this model from this machine" and is discovered
on every request from disk — never from a list in this file (see
:func:`local_models` for the four sources and the positive signal a row must
carry to qualify). A local entry is a DEPLOYMENT (``qwen3.8-27b-fp8`` on the
local router) while Artificial Analysis benchmarks a MODEL (``Qwen3.8 27B``),
so the two catalogues frequently do not meet: every local row that finds no AA
partner is reported in ``local_unmatched`` instead of being silently dropped.

Intelligence vs cost Pareto
---------------------------
``pareto`` / ``on_frontier`` / ``default_selection`` are all derived here, per
request, from the payload itself: the frontier is exactly the non-dominated set
of the models that report BOTH metrics (a point is dominated when another point
costs no more AND scores no less, with at least one of the two strict), and the
default selection follows the priority order documented in
:func:`select_default` — local deployments first, then frontier points by
intelligence, then the models nearest the frontier in normalised
``(log10 cost, intelligence)`` space, capped by
:data:`DEFAULT_SELECTION_CAP`. The browser renders what this module returns; it
never re-derives membership.

Provider colours
----------------
Colours come from :mod:`sharedagents_web.provider_palette`, which owns the only
provider→colour table in the tree and exposes the single resolver
:func:`sharedagents_web.provider_palette.provider_color`. Artificial Analysis
publishes no provider colour at all (verified against the OpenAPI
spec and the live Free payload — see that module's docstring), so there is no
API branch to take today; the API-shaped hook is gone rather than left in as a
lookup that can only ever return nothing. What this module does is ship the
resolved ``colors`` map in the payload, so every chart, legend and table on the
page paints the same provider the same colour — the browser never invents one.
"""

from __future__ import annotations

import ipaddress
import json
import math
import os
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from sharedagents_core.layout import sharedagents_root, usage_root
from sharedagents_web.provider_palette import provider_color_map
from sharedagents_web.subscription_limits import _as_float

AA_ENDPOINT = "https://artificialanalysis.ai/api/v2/language/models/free"
ATTRIBUTION = "Data: Artificial Analysis"
ATTRIBUTION_URL = "https://artificialanalysis.ai/"
SIGNUP_URL = "https://artificialanalysis.ai/login"
API_KEY_ENV = "ARTIFICIAL_ANALYSIS_API_KEY"

_CACHE_NAME = "benchmarks.json"
# Bundled snapshot: shipped in the wheel so a fresh install with no AA key and
# no cache still renders the Models page. Regenerated by
# scripts/refresh_models_snapshot.py before each release.
_SNAPSHOT_NAME = "models_snapshot.json"
SNAPSHOT_ENV = "SHAREDAGENTS_MODELS_SNAPSHOT"
_CACHE_TTL_S = 21600.0
_HTTP_TIMEOUT_S = 12.0
_PAGE_SIZE = 200
_MAX_PAGES = 25
_USER_AGENT = "sharedagents-webui/1.0 (internal-use benchmark dashboard)"


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


def env_file_hint() -> str:
    """Operator-facing path of the primary env file, resolved live.

    Derived from :func:`_env_files` (so ``$SHAREDAGENTS_HOME`` /
    ``$SHAREDAGENTS_AGENTS_HOME`` are honored) instead of a hardcoded default;
    a path under the user's home keeps the short ``~/…`` form, so with no
    override the message reads exactly as it always did.
    """
    path = _env_files()[0]
    home = Path.home()
    if path.is_relative_to(home):
        return f"~/{path.relative_to(home)}"
    return str(path)


# ``(method, url, headers, body, timeout) -> (status, payload, headers)``.
# Mirrors the injectable HTTP shape used by ``api_balances``/`subscription_limits``
# with response headers added so the tier / rate-limit headers can be read.
HttpFn = Callable[
    [str, str, dict[str, str], bytes | None, float],
    tuple[int, dict[str, Any], dict[str, str]],
]

_PUNCT_RE = re.compile(r"[^0-9a-z]+")
# The ordered digit runs of a text (``"Muse Spark 1.2"`` -> ``"12"``). Order
# matters: ``"GPT-4.5"`` is ``45`` while ``"gpt-5.4"`` is ``54`` — different
# models, so the comparison must not become an unordered digit bag.
_DIGIT_RE = re.compile(r"\d+")
# ``+`` is a version suffix, not punctuation: ``Command A+`` is a different model
# from ``Command A``, so it folds onto a real token instead of being erased.
_PLUS_TOKEN = "plus"
# Trailing " (…)" qualifier groups, e.g. "Claude Opus 5 (Adaptive Reasoning,
# Xhigh Effort)" or "GLM-4.5V (Non-reasoning)".
_QUALIFIER_RE = re.compile(r"\s*\(([^()]*)\)\s*$")
# Engine prefixes an alias may carry that the catalogue id would not
# ("or-claude-opus5" is the OpenRouter switch key for "claude-opus-5").
_ALIAS_PREFIXES = (
    "or-",
    "openrouter-",
    "agy-",
    "antigravity-",
    "cursor-",
    "codex-",
    "hermes-",
    "kimi-",
    "openai-",
    "anthropic-",
    "google-",
    "gemini-",
    "moonshot-",
    "moonshotai-",
    "deepseek-",
    "xai-",
)
# Effort / reasoning variants AA appends to a slug when the name carries a
# parenthetical the slug spells out with hyphens. Deliberately narrow: ordinary
# model-name words ("flash", "pro", "medium" as in "Mistral Medium") are NOT
# here, and a token is only ever peeled when it does not occur in the model's
# own name (see ``_slug_keys``).
_VARIANT_SLUG_TOKENS = frozenset(
    {
        "adaptive",
        "reasoning",
        "non-reasoning",
        "nonreasoning",
        "thinking",
        "effort",
        "high",
        "xhigh",
        "minimal",
        "none",
        "default",
    }
)


class BenchmarkError(Exception):
    """Structured fetch failure; carries the machine-readable error payload."""

    def __init__(self, code: str, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        payload["status"] = self.status
        if self.code in {"no_api_key", "rate_limited"}:
            payload["env_var"] = API_KEY_ENV
            payload["env_file"] = env_file_hint()
            payload["signup_url"] = SIGNUP_URL
        return payload


# ---------------------------------------------------------------------------
# Environment / configuration
# ---------------------------------------------------------------------------


def fixture_enabled() -> bool:
    """True when the deterministic offline fixture replaces the network."""
    raw = (os.environ.get("SHAREDAGENTS_BENCHMARKS_FIXTURE") or "").strip()
    return raw.lower() in {"1", "true", "yes", "on"}


def cache_path() -> Path:
    """Benchmark cache location (env override, else ``usage_root()``)."""
    override = (os.environ.get("SHAREDAGENTS_BENCHMARKS_CACHE") or "").strip()
    return Path(override).expanduser() if override else (usage_root() / _CACHE_NAME)


def snapshot_path() -> Path:
    """Bundled snapshot location (env override, else the wheel's ``data/`` dir).

    The snapshot ships inside the ``sharedagents_web`` package so a fresh install
    with no AA key and no cache still has a models table to render.
    """
    override = (os.environ.get(SNAPSHOT_ENV) or "").strip()
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parent / "data" / _SNAPSHOT_NAME


def _load_snapshot() -> dict[str, Any] | None:
    """Read the bundled snapshot, or ``None`` when it is absent/corrupt.

    This is the fresh-install fallback: it is only consulted when there is no
    AA key and no cache, so a missing or malformed snapshot degrades to the
    structured ``no_api_key`` error rather than a 500.
    """
    path = snapshot_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        return None
    return data


def cache_ttl_s() -> float:
    """Cache TTL in seconds (default 6h; ``SHAREDAGENTS_BENCHMARKS_TTL_S`` overrides)."""
    raw = (os.environ.get("SHAREDAGENTS_BENCHMARKS_TTL_S") or "").strip()
    if not raw:
        return _CACHE_TTL_S
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _CACHE_TTL_S
    return value if value > 0 else _CACHE_TTL_S


def _env(name: str) -> str:
    """Resolve ``name`` from the process env, then :func:`_env_files`.

    Same precedence and parsing as :func:`sharedagents_web.api_balances._env`: plain
    ``NAME=`` lines, surrounding quotes stripped, ``export NAME=`` NOT matched.
    """
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


def api_key() -> str:
    """The Artificial Analysis API key, or ``""`` when none is configured."""
    return _env(API_KEY_ENV)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _json_dict(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def header_value(headers: Any, name: str) -> str | None:
    """Case-insensitive header lookup.

    The live API emits ``X-Aa-Tier`` / ``X-Ratelimit-Remaining`` (note the
    casing). ``urllib``'s ``HTTPMessage`` is case-insensitive but
    ``dict(resp.headers)`` is not, so every read goes through this helper.
    """
    if headers is None:
        return None
    getter = getattr(headers, "get", None)
    if callable(getter):
        try:
            if hasattr(headers, "get_all"):  # HTTPMessage: already case-insensitive
                value = getter(name)
                if value is not None:
                    return str(value)
        except Exception:  # noqa: BLE001 - never fail on a header probe
            pass
    target = name.strip().lower()
    try:
        items: Iterable[tuple[Any, Any]] = headers.items()
    except AttributeError:
        return None
    for key, value in items:
        if str(key).strip().lower() == target:
            return None if value is None else str(value)
    return None


def _http(
    http: HttpFn | None,
    method: str,
    url: str,
    headers: dict[str, str],
) -> tuple[int, dict[str, Any], dict[str, str]]:
    """One request; returns ``(status, payload, headers)``.

    The response headers are flattened into a plain dict and always read back
    through :func:`header_value` (case-insensitive), because the server's
    ``X-Aa-Tier`` / ``X-Ratelimit-*`` casing does not survive ``dict()``.
    """
    if http is not None:
        return http(method, url, headers, None, _HTTP_TIMEOUT_S)
    request = Request(url, headers=headers, method=method)
    try:
        with urlopen(request, timeout=_HTTP_TIMEOUT_S) as response:
            raw = response.read().decode("utf-8", errors="replace")
            flat = {str(k): str(v) for k, v in response.headers.items()}
            return int(response.status), _json_dict(raw), flat
    except HTTPError as exc:
        raw = exc.read() if exc.fp else b""
        flat = {str(k): str(v) for k, v in (exc.headers or {}).items()}
        return int(exc.code), _json_dict(raw.decode("utf-8", errors="replace")), flat
    except URLError as exc:
        raise BenchmarkError(
            "network_error",
            f"Artificial Analysis API unreachable: {exc.reason}",
        ) from exc
    except (TimeoutError, OSError) as exc:
        raise BenchmarkError(
            "network_error",
            f"Artificial Analysis API unreachable: {exc}",
        ) from exc


def _headers(key: str) -> dict[str, str]:
    return {
        "x-api-key": key,
        "Accept": "application/json",
        "User-Agent": _USER_AGENT,
    }


def _int_or_none(value: str | None) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _fetch_all_pages(key: str, http: HttpFn | None) -> dict[str, Any]:
    """Walk every page of the Free-tier model list (200 rows per page).

    Stops when ``pagination.has_more`` is false — never on a hardcoded page
    count. ``_MAX_PAGES`` is only a runaway guard (25 pages = 5000 rows).
    """
    rows: list[Any] = []
    meta: dict[str, Any] = {"tier": None, "intelligence_index_version": None}
    rate_limit: dict[str, Any] = {"limit": None, "remaining": None, "reset": None}
    page = 1
    while page <= _MAX_PAGES:
        status, payload, headers = _http(http, "GET", f"{AA_ENDPOINT}?page={page}", _headers(key))
        if status == 401:
            raise BenchmarkError(
                "no_api_key",
                f"Artificial Analysis rejected the configured {API_KEY_ENV} (HTTP 401).",
                status=status,
            )
        if status == 429:
            raise BenchmarkError(
                "rate_limited",
                "Artificial Analysis rate limit hit (HTTP 429); the daily budget resets at 00:00 UTC.",
                status=status,
            )
        if status != 200:
            raise BenchmarkError(
                "http_error",
                f"Artificial Analysis API returned HTTP {status}.",
                status=status,
            )
        data = payload.get("data")
        if not isinstance(data, list):
            raise BenchmarkError(
                "bad_payload",
                "Artificial Analysis API response carried no model list.",
                status=status,
            )
        rows.extend(data)
        tier = payload.get("tier") or header_value(headers, "X-Aa-Tier")
        if tier:
            meta["tier"] = tier
        if payload.get("intelligence_index_version"):
            meta["intelligence_index_version"] = payload["intelligence_index_version"]
        for header, field in (
            ("X-Ratelimit-Limit", "limit"),
            ("X-Ratelimit-Remaining", "remaining"),
        ):
            value = _int_or_none(header_value(headers, header))
            if value is not None:
                rate_limit[field] = value
        reset = header_value(headers, "X-Ratelimit-Reset")
        if reset:
            rate_limit["reset"] = reset
        pagination = _mapping(payload.get("pagination"))
        if not pagination.get("has_more"):
            break
        page += 1
    meta["data"] = rows
    meta["pages_fetched"] = page
    meta["rate_limit"] = rate_limit
    return meta


# ---------------------------------------------------------------------------
# Name normalisation + enabled-model matching
# ---------------------------------------------------------------------------


def _mapping(value: Any) -> dict[str, Any]:
    """``value`` when it is a mapping, else ``{}`` (null-tolerant walk)."""
    return value if isinstance(value, dict) else {}


def _fold_plus(value: str) -> str:
    """``"Command A+"`` -> ``"Command A plus"`` — ``+`` becomes a real token."""
    return str(value or "").replace("+", f" {_PLUS_TOKEN} ")


def normalize_name(value: Any) -> str:
    """Case/punctuation/whitespace-insensitive comparison key.

    ``"GPT-5.6-Luna"``, ``"gpt-5.6-luna"`` and ``"GPT 5.6 Luna"`` all fold to
    ``"gpt56luna"``. A trailing parenthetical qualifier group is removed too
    (``"Kimi K3 (low)"`` -> ``"kimi k3"``) so a bare sharedagents ``display`` can
    still meet a qualified AA row.

    A literal ``+`` is NOT erased: it folds onto the token ``plus``, so
    ``"Command A+"`` (``commandaplus``) can never collapse onto ``"Command A"``
    (``commanda``). That collapse was the audit's first confirmed false positive
    — a 2026 model labelled enabled off a 2025 catalogue row.
    """
    text = _fold_plus(_bare_name(str(value or "")))
    return _PUNCT_RE.sub("", text.strip().lower())


def _significant(value: Any) -> tuple[str, bool]:
    """The version signature of a name/id/slug: ``(digit order, carries "+")``.

    Separators are dropped but the digit *order* is kept, so ``"Muse Spark 1.2"``,
    ``"muse-spark-1-2"`` and ``"meta/muse-spark-1.2"`` all sign as ``"12"`` while
    ``"GPT-4.5"`` (``45``) and ``"gpt-5.4"`` (``54``) do not. Two texts with
    different signatures are not the same model version, and this is the guard
    that keeps a dated or ``+``-suffixed AA row from claiming a catalogue row
    that describes a different revision of the family.
    """
    text = str(value or "")
    return ("".join(_DIGIT_RE.findall(text)), "+" in text)


def _bare_name(value: str) -> str:
    """Drop trailing ``(...)`` qualifier groups, repeatedly."""
    text = str(value or "").strip()
    while True:
        stripped = _QUALIFIER_RE.sub("", text).strip()
        if stripped == text or not stripped:
            return stripped
        text = stripped


def qualifier(value: Any) -> str | None:
    """The trailing parenthetical text of a label, if any.

    ``"Claude Opus 5 (Adaptive Reasoning, High Effort)"`` ->
    ``"Adaptive Reasoning, High Effort"``; used as the chart label suffix so
    two AA variants of one sharedagents model stay distinguishable.
    """
    text = str(value or "").strip()
    out: list[str] = []
    while True:
        match = _QUALIFIER_RE.search(text)
        if not match:
            break
        out.insert(0, match.group(1).strip())
        text = text[: match.start()].strip()
    joined = "; ".join(part for part in out if part)
    return joined or None


def _strip_provider(value: str) -> str:
    """``"anthropic/claude-opus-5"`` -> ``"claude-opus-5"``."""
    text = str(value or "").strip()
    if "/" in text:
        return text.rsplit("/", 1)[-1].strip()
    return text


# ``(normalised key, the text the key was derived from)``. The provenance is what
# lets the version guard ask "was this claim licensed by the AA text that actually
# produced the key?" instead of by some other field of the row.
Key = tuple[str, str]


def _key_sources(value: Any) -> tuple[Key, ...]:
    """Ordered ``(key, source)`` candidates for a catalogue/AA text field.

    The source is the first form that produced the key, so a key taken from a
    qualified name keeps the qualifier in its signature (``"DeepSeek R1 (Jan
    '25)"`` signs as ``125`` even though the key drops the parenthetical).
    """
    text = str(value or "").strip()
    if not text:
        return ()
    out: list[Key] = []
    for candidate in (text, _strip_provider(text), _bare_name(text)):
        key = normalize_name(candidate)
        if key and all(key != seen for seen, _ in out):
            out.append((key, candidate))
    return tuple(out)


def _keys(value: Any) -> tuple[str, ...]:
    """Ordered normalised candidates for a catalogue field (full, then bare)."""
    return tuple(key for key, _ in _key_sources(value))


def _alias_key_sources(value: Any) -> tuple[Key, ...]:
    """Alias candidates: the alias itself plus **prefix-stripped** forms only.

    Deliberately narrower than :func:`_key_sources`: the audit found this step
    1-for-1 wrong on the live population, because it peeled the engine prefix
    into an unconstrained guess — ``or-muse-spark`` met AA's bare ``muse-spark``
    slug while the catalogue row it points at is ``meta/muse-spark-1.2``, a
    *different* release. No provider/bare re-derivation happens here, and the
    version guard in :func:`_version_compatible` still has to agree.
    """
    text = str(value or "").strip()
    if not text:
        return ()
    variants = [text]
    current = text
    for _ in range(2):
        lowered = current.lower()
        stripped = False
        for prefix in _ALIAS_PREFIXES:
            if lowered.startswith(prefix) and len(current) > len(prefix):
                current = current[len(prefix) :]
                variants.append(current)
                stripped = True
                break
        if not stripped:
            break
    out: list[Key] = []
    for variant in variants:
        key = normalize_name(variant)
        if key and all(key != seen for seen, _ in out):
            out.append((key, variant))
    return tuple(out)


def _slug_key_sources(slug: Any, name: Any = None) -> tuple[Key, ...]:
    """AA slug candidates, incl. the qualifier-derived bare form, with sources.

    AA spells a parenthetical qualifier out with hyphens in the slug
    (``"Claude Opus 5 (Adaptive Reasoning, High Effort)"`` ->
    ``claude-opus-5-high``). Trailing slug tokens that also appear in the
    qualifier — or in the known effort/reasoning vocabulary — are peeled off so
    the variant rows of one model fold onto the model's own key.
    """
    text = str(slug or "").strip()
    if not text:
        return _key_sources(name)
    out: list[Key] = []
    qualifier_tokens = set(_PUNCT_RE.split(_fold_plus(str(qualifier(name) or "").lower())))
    qualifier_tokens.discard("")
    name_tokens = set(_PUNCT_RE.split(_fold_plus(_bare_name(str(name or "")).lower())))
    name_tokens.discard("")
    bare = _bare_name(text)
    if bare != text:
        out.extend(_key_sources(bare))
    tokens = [token for token in re.split(r"[-_]", text) if token]
    while len(tokens) > 1:
        tail = tokens[-1].lower()
        if tail in qualifier_tokens or (tail in _VARIANT_SLUG_TOKENS and tail not in name_tokens):
            tokens.pop()
        else:
            break
    candidate = "-".join(tokens)
    if candidate and candidate != text:
        out.extend(_key_sources(candidate))
    out.extend(_key_sources(text))
    deduped: list[Key] = []
    for key, source in out:
        if key and all(key != seen for seen, _ in deduped):
            deduped.append((key, source))
    return tuple(deduped)


def _slug_keys(slug: Any, name: Any = None) -> tuple[str, ...]:
    """AA slug candidates, incl. the qualifier-derived bare form."""
    return tuple(key for key, _ in _slug_key_sources(slug, name))


def _token_key(value: Any) -> str | None:
    """Order-insensitive key: the version signature plus the sorted tokens.

    ``"claude-4-5-haiku"`` and ``"claude-haiku-4.5"`` fold to the same key (same
    token multiset, same digits in the same order), which is the audit's false
    negative: sharedagents's ``anthropic/claude-haiku-4.5`` never met AA's
    ``claude-4-5-haiku`` rows because the normalized token order differed.
    ``"GPT-4.5"`` and ``"gpt-5.4"`` do NOT fold together (``45`` vs ``54``), and
    a ``+`` stays a token. Fewer than two tokens is too weak to be evidence of
    anything, so those return ``None``.
    """
    text = _fold_plus(str(value or "").lower())
    tokens = sorted(token for token in _PUNCT_RE.split(text) if token)
    if len(tokens) < 2:
        return None
    digits, plus = _significant(value)
    return ("+" if plus else "") + digits + "|" + "|".join(tokens)


def _token_sources(value: Any) -> tuple[Key, ...]:
    """``(token key, source)`` candidates for one catalogue/AA text field.

    The recorded source is always the *whole* field text, so a token key that
    ignores a parenthetical qualifier is still licensed only by the field's full
    version signature — the same convention as :func:`_key_sources`, and the
    reason a dated AA row cannot be re-enabled through this softer step.
    """
    text = str(value or "").strip()
    if not text:
        return ()
    out: list[Key] = []
    for candidate in (text, _strip_provider(text), _bare_name(text)):
        key = _token_key(candidate)
        if key and all(key != seen for seen, _ in out):
            out.append((key, text))
    return tuple(out)


def _global_config() -> dict[str, Any]:
    """Read-only ``~/.sharedagents/config.json``; ``{}`` when absent/unreadable."""
    try:
        from sharedagents_core.kernel.config import load_global_config

        cfg = load_global_config()
    except Exception:  # noqa: BLE001 - operator config must never break the page
        return {}
    return cfg if isinstance(cfg, dict) else {}


def _configured_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """``models[]`` from the global catalogue, legacy keys normalised."""
    rows = cfg.get("models")
    if not isinstance(rows, list):
        return []
    raw = [row for row in rows if isinstance(row, dict)]
    try:
        from sharedagents_agent.agent import _normalize_models

        return [row for row in _normalize_models(raw) if isinstance(row, dict)]
    except Exception:  # noqa: BLE001 - e.g. duplicate-alias ValueError
        return [dict(row) for row in raw]


def _shipped_external() -> dict[str, Any]:
    """Shipped per-engine external catalogue (``{}`` if the import fails)."""
    try:
        from sharedagents_agent.discovery._common import _SHIPPED_EXTERNAL_MODELS

        return dict(_SHIPPED_EXTERNAL_MODELS)
    except Exception:  # noqa: BLE001
        return {}


def _external_rows(cfg: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """``(engine, row)`` for every configured/shipped external model row."""
    engines: list[str] = []
    configured = _mapping(cfg.get("external_models"))
    for engine in configured:
        key = str(engine).strip()
        if key and key not in engines:
            engines.append(key)
    shipped = _shipped_external()
    for engine in shipped:
        key = str(engine).strip()
        if key and key not in engines:
            engines.append(key)
    out: list[tuple[str, dict[str, Any]]] = []
    for engine in engines:
        rows: list[dict[str, Any]] | None = None
        try:
            from sharedagents_agent.discovery._common import _fallback_models

            rows = [row for row in _fallback_models(engine) if isinstance(row, dict)]
        except Exception:  # noqa: BLE001
            rows = None
        if rows is None:
            rows = [row for row in (configured.get(engine) or []) if isinstance(row, dict)] + [
                dict(row) for row in shipped.get(engine, ()) or ()
            ]
        for row in rows:
            out.append((engine, row))
    return out


def _discovered_rows() -> list[tuple[str, dict[str, Any]]]:
    """``(agent, row)`` from the discovery cache; read-only, never raised."""
    try:
        from sharedagents_agent.discovery.cache_lock import read_all_cached_models

        data = read_all_cached_models()
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(data, dict):
        return []
    out: list[tuple[str, dict[str, Any]]] = []
    for agent in sorted(data):
        rows = data[agent]
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict):
                out.append((str(agent), row))
    return out


def enabled_rows() -> list[dict[str, Any]]:
    """Every model sharedagents can select today, as ``{alias,id,display,source}``.

    Read-only union, in precedence order (first writer wins per normalised id):
    global ``models[]`` → ``external_models``/shipped engines → discovery cache.
    Missing or unreadable sources degrade to "no enabled models".
    """
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_aliases: set[str] = set()

    def add(row: dict[str, Any], source: str) -> None:
        if not isinstance(row, dict):
            return
        alias = str(row.get("alias") or "").strip()
        model_id = str(row.get("id") or row.get("model") or "").strip()
        display = str(row.get("display") or "").strip()
        if not model_id and not alias:
            return
        id_key = normalize_name(model_id) or normalize_name(alias)
        alias_key = alias.lower()
        if id_key and id_key in seen_ids:
            return
        if alias_key and alias_key in seen_aliases:
            return
        if id_key:
            seen_ids.add(id_key)
        if alias_key:
            seen_aliases.add(alias_key)
        rows.append(
            {
                "alias": alias,
                "id": model_id,
                "display": display or model_id or alias,
                "source": source,
            }
        )

    cfg = _global_config()
    for row in _configured_rows(cfg):
        add(row, "config")
    for engine, row in _external_rows(cfg):
        add(row, f"external:{engine}")
    for agent, row in _discovered_rows():
        add(row, f"discovered:{agent}")
    return rows


# ---------------------------------------------------------------------------
# Local deployments — "sharedagents serves this model from this machine"
# ---------------------------------------------------------------------------
#
# Everything below is derived from disk on every request; no model name,
# provider or alias is written into this module. The four sources are exactly
# the ones this repo keeps authoritative, and a row only qualifies when it
# carries a POSITIVE local signal — otherwise a cloud row with no endpoint at
# all (the engine catalogues are full of those) would be mislabelled as local.
#
# The sources:
#   * ``~/.sharedagents/config.json`` ``models[]`` — typically a few
#     ``backend: vllm`` rows with a loopback ``baseUrl`` among many hosted
#     OpenAI-family rows.
#   * ``~/.sharedagents/discovered_models.json`` — per-engine buckets. The
#     ``__native__`` bucket is mostly hosted rows with remote base URLs, which
#     is why the bucket NAME is not treated as proof: the verified predicate is
#     applied to every row of every bucket, ``__native__`` included.
#   * ``~/.sharedagents/vllm-router.yaml`` — the local vLLM mux list; its ``pools``
#     are locally-served deployments (``external_providers`` are remote
#     fallbacks and are deliberately NOT local).
#   * ``~/.sharedagents/agents/*/config.json`` ``upstream_model`` pins — included
#     only when the SAME row proves the route is local (loopback
#     ``upstream_endpoint`` or a ``vllm``/``sglang`` backend). A pin on a
#     hosted CLI engine (``codex``, ``antigravity``) has no local endpoint in
#     the data, so it is not claimed as local.
#
# Every reader degrades to "nothing local" rather than raising: an unreadable
# file, a malformed YAML, a missing PyYAML, or a lock that cannot be taken must
# never break the page.

#: ``backend`` labels that mean "served by a local inference server". Mirrors
#: the positively-local branch of ``sharedagents_model.models.row_is_hosted``.
LOCAL_BACKEND_LABELS = frozenset({"vllm", "sglang", "ollama", "lmstudio"})
_LAN_HOST_SUFFIXES = (".local", ".lan", ".internal", ".home", ".home.arpa", ".localdomain")
_ENDPOINT_FIELDS = (
    "baseUrl",
    "base_url",
    "endpoint",
    "url",
    "upstream_endpoint",
    "openai_base_url",
)
#: The four disk sources, in precedence order. Reported to the UI as-is so
#: "why is this local?" has an answer that points at a file.
LOCAL_SOURCES = ("config", "discovered", "vllm-router", "agent-pin")


def _endpoint_url(row: dict[str, Any]) -> str:
    """First non-empty endpoint field of ``row`` (``""`` when it has none)."""
    for field in _ENDPOINT_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _endpoint_host(value: Any) -> str:
    """Extract a host from a URL or a scheme-less host[:port][/path].

    Bare IPv6 addresses are accepted too. No DNS lookup is needed to render
    the page; malformed values and socket paths provide no host evidence.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        if "://" not in text and not text.startswith("//"):
            authority = re.split(r"[/?#]", text, maxsplit=1)[0]
            try:
                return str(ipaddress.ip_address(authority)).lower()
            except ValueError:
                pass
            if text.startswith(("/", ".", "~")):
                return ""
            text = "//" + text
        parsed = urlparse(text)
        if parsed.scheme in {"unix", "http+unix", "https+unix"}:
            return ""
        return (parsed.hostname or "").strip().lower().rstrip(".")
    except ValueError:
        return ""


def _is_loopback_endpoint(value: Any) -> bool:
    """True for localhost names, loopback IPs and wildcard bind addresses.

    A missing/unparseable endpoint is not evidence of localness. Parse IPs
    rather than accepting any DNS name that happens to start with ``127.``.
    """
    host = _endpoint_host(value)
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    return addr.is_loopback or addr.is_unspecified


def _is_remote_endpoint(value: Any) -> bool:
    """True for globally routable IPs and public-looking DNS names.

    Bare hostnames and LAN suffixes retain the backend's local signal. An
    arbitrary fully-qualified name may use split DNS; its actual network is
    unknowable here, so router pools remain the authoritative local inventory.
    """
    host = _endpoint_host(value)
    if not host or _is_loopback_endpoint(value):
        return False
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return "." in host and not host.endswith(_LAN_HOST_SUFFIXES)
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    # is_global also excludes shared address space (e.g. Tailscale's CGNAT
    # addresses), which is neither is_private nor a hosted public endpoint.
    return addr.is_global and not addr.is_multicast


def row_is_local(row: Any) -> bool:
    """True when an sharedagents model row is served locally, by positive signal only.

    A public endpoint overrides the backend label. Otherwise a local inference
    backend or loopback endpoint is a positive signal. LAN addresses alone do
    not prove that a model is locally served, nor does a missing endpoint.
    """
    if not isinstance(row, dict):
        return False
    endpoint = _endpoint_url(row)
    # A public endpoint wins over a stale label: an agent can keep
    # ``backend: vllm`` after it moves to a hosted API.
    if _is_remote_endpoint(endpoint):
        return False
    backend = str(row.get("backend") or "").strip().lower()
    if backend in LOCAL_BACKEND_LABELS:
        return True
    return _is_loopback_endpoint(endpoint)


def _local_row(row: dict[str, Any], source: str) -> dict[str, Any]:
    """One local deployment, normalised to ``{alias,id,display,source}``."""
    alias = str(row.get("alias") or "").strip()
    model_id = str(row.get("id") or row.get("model") or "").strip()
    display = str(row.get("display") or row.get("display_name") or "").strip()
    return {
        "alias": alias,
        "id": model_id,
        "display": display or model_id or alias,
        "source": source,
    }


def _local_config_rows() -> list[dict[str, Any]]:
    """``~/.sharedagents/config.json`` ``models[]`` rows that are served locally."""
    out: list[dict[str, Any]] = []
    for row in _configured_rows(_global_config()):
        if row_is_local(row):
            out.append(_local_row(row, "config"))
    return out


def _local_discovered_rows() -> list[dict[str, Any]]:
    """Discovery-cache rows that are served locally, every bucket included.

    ``__native__`` is the native family's catalogue and is read through its own
    ``NATIVE_CACHE_KEY`` entry (``read_all_cached_models`` fans that bucket out
    under every native agent's name, which would mislabel the provenance here).
    The bucket name is NOT proof of localness — 476 of its 482 rows on this
    fleet are hosted ``backend: openai`` rows — so the verified
    :func:`row_is_local` predicate decides, and the bucket that produced a row is
    kept in ``source`` so the UI can say where it came from.
    """
    try:
        from sharedagents_agent.discovery.cache_lock import (
            NATIVE_CACHE_KEY,
            get_cached_models,
            read_all_cached_models,
        )

        native = [(NATIVE_CACHE_KEY, row) for row in get_cached_models(NATIVE_CACHE_KEY)]
        buckets = [
            (str(bucket), row)
            for bucket, rows in (read_all_cached_models() or {}).items()
            for row in rows
            if isinstance(row, dict)
        ]
    except Exception:  # noqa: BLE001 - a broken/unreadable cache is "nothing local"
        return []
    out: list[dict[str, Any]] = []
    for bucket, row in native + buckets:
        if row_is_local(row):
            out.append(_local_row(row, f"discovered:{bucket}"))
    return out


def _local_router_rows() -> list[dict[str, Any]] | None:
    """Pool ids of the local vLLM mux list (``$SHAREDAGENTS_VLLM_CONFIG`` or the operator YAML).

    Read through the router's own loader when it is importable, so this module
    never hand-parses the topology. ``external_providers`` are remote last hops
    and are deliberately excluded: only ``pools`` are locally served.
    ``None`` means no usable router config; ``[]`` means a router with no public
    LLM pools, which must not revive stale rows from other disk sources.
    """
    try:
        from sharedagents_vllm.config import default_config_path, load_config

        path = default_config_path()
        if not path.is_file():
            return None
        cfg = load_config(path)
    except Exception:  # noqa: BLE001 - a missing file / missing PyYAML is not an error
        return None
    pools = getattr(cfg, "pools", None)
    if not isinstance(pools, Mapping):
        return None
    out: list[dict[str, Any]] = []
    for model_id, pool in pools.items():
        if getattr(pool, "hidden", False):
            continue
        if str(getattr(pool, "kind", "") or "") != "llm":
            continue
        name = str(model_id or "").strip()
        if not name:
            continue
        out.append(
            {
                "alias": name,
                "id": name,
                "display": name,
                "source": "vllm-router",
                "kind": str(getattr(pool, "kind", "") or ""),
            }
        )
    return out


def _local_agent_rows() -> list[dict[str, Any]]:
    """Per-agent ``upstream_model`` pins whose route the data proves is local.

    The pin alone is not enough: the same config must place the model on a
    loopback endpoint or label it with a local backend. A hosted CLI pin (the
    ``codex`` / ``antigravity`` style) therefore never shows up here.
    """
    out: list[dict[str, Any]] = []
    agents_root = sharedagents_root() / "agents"
    try:
        entries = sorted(agents_root.iterdir(), key=lambda path: path.name)
    except OSError:
        return []
    for entry in entries:
        path = entry / "config.json"
        try:
            if not path.is_file():
                continue
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        cfg = raw if isinstance(raw, dict) else {}
        model_id = str(cfg.get("upstream_model") or "").strip()
        if not model_id:
            continue
        row = {
            "alias": str(cfg.get("upstream_alias") or "").strip() or model_id,
            "id": model_id,
            "display": str(cfg.get("upstream_alias") or "").strip() or model_id,
            "backend": cfg.get("backend"),
            "baseUrl": cfg.get("upstream_endpoint"),
        }
        if row_is_local(row):
            out.append(_local_row(row, f"agent-pin:{entry.name}"))
    return out


def _non_llm_router_ids() -> set[str]:
    """Normalised ids of router pools whose kind is not ``llm``."""
    try:
        from sharedagents_vllm.config import default_config_path, load_config

        path = default_config_path()
        if not path.is_file():
            return set()
        pools = getattr(load_config(path), "pools", None) or {}
    except Exception:  # noqa: BLE001 - no router config means nothing to exclude
        return set()
    return {
        normalize_name(model_id)
        for model_id, pool in pools.items()
        if str(getattr(pool, "kind", "") or "") != "llm" and normalize_name(model_id)
    }


def local_models() -> list[dict[str, Any]]:
    """Every model sharedagents serves from this machine, discovered per call.

    With a usable router config: exactly its non-hidden ``llm`` pools, carrying
    ``source: vllm-router``, in config order with configured aliases attached.
    Otherwise: the union of config, discovery and agent pins, each carrying its
    ``source``. That union is deduplicated on the normalised wire id (falling
    back to the alias), first source wins — the configured row is authoritative
    and the discovery cache is downstream of it. Returns ``[]`` when nothing
    on disk says "local", and never raises.
    """
    # When this machine runs the router, the models it serves ARE the local
    # models: its public LLM names, nothing from caches or agent pins.
    served = _local_router_rows()
    if served is not None:
        configured = _local_config_rows()
        by_id: dict[str, tuple[int, dict[str, Any]]] = {}
        for n, row in enumerate(configured):
            by_id.setdefault(str(row.get("id") or ""), (n, row))
        by_normalized: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for model_id, entry in by_id.items():
            by_normalized.setdefault(normalize_name(model_id), []).append(entry)
        served_counts: dict[str, int] = {}
        for row in served:
            key = normalize_name(row["id"])
            served_counts[key] = served_counts.get(key, 0) + 1
        ordered = []
        for row in served:
            # Router ids are distinct wire routes even when punctuation/case
            # normalization collapses them. Only guess when both sides agree
            # on one candidate; an exact id always keeps its own config alias.
            match = by_id.get(row["id"])
            key = normalize_name(row["id"])
            candidates = by_normalized.get(key, [])
            if match is None and key and served_counts[key] == 1 and len(candidates) == 1:
                match = candidates[0]
            position, config_row = match if match is not None else (len(configured), {})
            ordered.append((position, {**row, "alias": config_row.get("alias") or row["alias"]}))
        return [row for _, row in sorted(ordered, key=lambda item: item[0])]
    rows: list[dict[str, Any]] = []
    # The page lists language models only: speech, decision and other router
    # kinds are dropped by id from every source.
    seen: set[str] = set(_non_llm_router_ids())
    for source_rows in (
        _local_config_rows(),
        _local_discovered_rows(),
        _local_agent_rows(),
    ):
        for row in source_rows:
            key = normalize_name(row.get("id")) or str(row.get("alias") or "").lower()
            if not key or key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def _apply_local(
    models: Sequence[dict[str, Any]],
    local: Sequence[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Mark every AA row sharedagents serves locally; report the rest of the locals.

    The join is the SAME cascade as :func:`match_enabled`, run with the local
    deployments as the catalogue and the AA rows as the thing to match: exact
    normalised wire id ↔ AA slug/name first, then the alias, then the reordered
    token set, each still version-guarded. A local deployment that finds no AA
    row (the common case for a quantised build — AA benchmarks a MODEL, this
    fleet runs a DEPLOYMENT of it) is returned in ``unmatched`` so the UI can
    say "running locally, not benchmarked by Artificial Analysis" instead of
    dropping it.
    """
    catalogue = list(local) if local is not None else local_models()
    steps = _steps(catalogue) if catalogue else None
    matched: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in models:
        item = dict(row)
        hit = _match_in(steps, item.get("name"), item.get("slug")) if steps else None
        item["local"] = hit is not None
        item["local_deployment"] = (
            {
                "alias": hit.get("alias"),
                "id": hit.get("id"),
                "display": hit.get("display"),
                "source": hit.get("source"),
            }
            if hit
            else None
        )
        if hit:
            matched.add(_local_identity(hit))
        out.append(item)
    unmatched: list[dict[str, Any]] = []
    for row in catalogue:
        if _local_identity(row) in matched:
            continue
        unmatched.append(
            {
                "alias": str(row.get("alias") or ""),
                "id": str(row.get("id") or ""),
                "display": str(row.get("display") or ""),
                "source": str(row.get("source") or ""),
                "kind": str(row.get("kind") or ""),
            }
        )
    return out, unmatched


def _local_identity(row: Mapping[str, Any]) -> str:
    """The deployment identity, preserving distinct public router routes."""
    if row.get("source") == "vllm-router":
        return str(row.get("id") or row.get("alias") or "")
    return normalize_name(row.get("id")) or str(row.get("alias") or "").lower()


def _index(
    rows: Sequence[dict[str, Any]],
    field: str,
) -> dict[str, list[tuple[dict[str, Any], int]]]:
    """Key -> every ``(row, insertion order)`` indexed under it.

    First-writer-wins is deliberately gone: one key may legitimately be shared
    (an operator alias and a discovered id normalize alike), so every row stays
    in the bucket and :func:`_match_in` picks the most specific claim.
    """
    index: dict[str, list[tuple[dict[str, Any], int]]] = {}
    for order, row in enumerate(rows):
        if field == "token":
            pairs = [
                pair for text in (row.get("id"), row.get("alias"), row.get("display")) for pair in _token_sources(text)
            ]
        elif field == "alias":
            pairs = _alias_key_sources(row.get("alias"))
        else:
            pairs = _key_sources(row.get(field))
        for key, _source in pairs:
            bucket = index.setdefault(key, [])
            if all(seen_row is not row for seen_row, _ in bucket):
                bucket.append((row, order))
    return index


_STEP_ORDER = ("id_slug", "id_name", "display_name", "alias_slug", "token_set")
# Steps whose evidence is the model's own wire identifier. `display_name` is an
# operator-written label and the last two re-derive a key, so the payload marks
# those claims `match_exact: false`.
_EXACT_STEPS = frozenset({"id_slug", "id_name"})


def _steps(rows: Sequence[dict[str, Any]]):
    """The documented cascade: ``(label, index, aa_field)`` in priority order.

    1. exact normalised sharedagents ``id`` vs the AA slug (full, provider-stripped,
       and qualifier-stripped variants)
    2. normalised sharedagents ``id`` vs the AA name
    3. normalised sharedagents ``display`` vs the AA name
    4. alias-derived guess (engine prefix stripped, nothing else) vs the AA slug
    5. same-token-multiset fallback: the AA slug/name re-ordered — e.g.
       ``claude-4-5-haiku`` meeting ``claude-haiku-4.5`` — with the version
       signature still required to match exactly
    """
    ids = _index(rows, "id")
    displays = _index(rows, "display")
    aliases = _index(rows, "alias")
    tokens = _index(rows, "token")
    return (
        ("id_slug", ids, "slug"),
        ("id_name", ids, "name"),
        ("display_name", displays, "name"),
        ("alias_slug", aliases, "slug_exact"),
        ("token_set", tokens, "token"),
    )


def _targets(name: Any, slug: Any) -> dict[str, tuple[Key, ...]]:
    """The AA-side key candidates per cascade field, each with its provenance."""
    slug_text = str(slug or "").strip()
    name_text = str(name or "").strip()
    tokens: list[Key] = []
    for text in (slug_text, name_text):
        for key, source in _token_sources(text):
            if all(key != seen for seen, _ in tokens):
                tokens.append((key, source))
    return {
        "slug": _slug_key_sources(slug_text, name_text),
        "slug_exact": _key_sources(slug_text) or _key_sources(name_text),
        "name": _key_sources(name_text),
        "token": tuple(tokens),
    }


def _version_compatible(aa_text: Any, row: dict[str, Any]) -> bool:
    """The AA text and the catalogue row's wire ``id`` must agree on versions.

    ``Command A+`` signs ``("", True)`` while ``cohere/command-a`` signs
    ``("", False)``; AA's bare ``Muse Spark`` signs ``("", False)`` while
    ``meta/muse-spark-1.2`` signs ``("12", False)`` — that alias may no longer
    claim that row. The catalogue id is authoritative: a version that only shows
    up in an alias or a display cannot license the claim, so a dated AA row
    cannot claim a versionless catalogue row either (or the reverse).
    """
    return _significant(aa_text) == _significant(row.get("id"))


def _specificity(key: str) -> int:
    """How much version information a matched key consumed (``…-0613`` > ``…``)."""
    return len("".join(_DIGIT_RE.findall(key)))


def _match_in(steps, name: Any, slug: Any) -> dict[str, Any] | None:
    """The most specific candidate across the cascade, or ``None``.

    Candidates that fail :func:`_version_compatible` are dropped, then the
    survivors are ranked by cascade step, by the version information the matched
    key carries, by the AA key order and finally by catalogue insertion order —
    deterministic, and independent of which key happened to be written into the
    index first. So a dated catalogue row (``…-0613``, ``…-0902``) takes
    precedence over a versionless sibling that both keys could reach.
    """
    if not steps:
        return None
    targets = _targets(name, slug)
    slug_keys = {key for key, _ in targets["slug"]}
    best: tuple[Any, ...] | None = None
    found: dict[str, Any] | None = None
    for order, (label, index, field) in enumerate(steps):
        for key_rank, (key, source) in enumerate(targets.get(field, ())):
            for row, row_rank in index.get(key, ()):
                if not _version_compatible(source, row):
                    continue
                rank = (order, -_specificity(key), key_rank, row_rank)
                if best is not None and rank >= best:
                    continue
                best = rank
                hit = dict(row)
                hit["match"] = label
                hit["match_key"] = key
                hit["exact"] = label in _EXACT_STEPS and key in slug_keys
                found = hit
    return found


def match_enabled(
    name: Any,
    slug: Any,
    rows: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Match one Artificial Analysis row to an sharedagents model, else ``None``.

    This is a deliberate HEURISTIC join, because the two catalogues share no
    key: AA publishes ``name``/``slug``/``model_creator.name``, while sharedagents
    stores an operator-chosen ``alias`` (``or-kimi-k3``), a wire ``id``
    (``moonshotai/kimi-k3``) and sometimes a ``display`` — which is
    frequently empty. The cascade in :func:`_steps` is ordered
    most-specific-first and each step is a normalised equality test, so it is
    deterministic: with the same catalogue the same AA row always resolves the
    same way. A row that matches nothing simply stays ``enabled: false`` —
    this function never raises.

    Every AA row of a family matches independently: AA publishes one row per
    provider-side effort variant (``claude-opus-5-high``, ``claude-opus-5-low``,
    …), all of which are the SAME sharedagents model. They are therefore ALL marked
    enabled (never silently collapsed to one), and the qualifier
    (:func:`qualifier`) is carried on the normalised row so the chart labels
    keep the effort/level that explains their differing intelligence/speed/cost.

    The returned row carries ``match`` (the cascade step that matched) and
    ``exact`` (True only for a ``id_slug``/``id_name`` hit that the AA *slug*
    licenses — a soft claim from a display, an alias guess or the reordered
    token fallback is False), which :func:`_apply_enabled` surfaces to the UI as
    ``match_step`` / ``match_exact``.
    """
    catalogue = list(rows) if rows is not None else enabled_rows()
    if not catalogue:
        return None
    return _match_in(_steps(catalogue), name, slug)


def _apply_enabled(models: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    catalogue = enabled_rows()
    steps = _steps(catalogue) if catalogue else None
    out: list[dict[str, Any]] = []
    for row in models:
        item = dict(row)
        hit = _match_in(steps, item.get("name"), item.get("slug")) if steps else None
        item["enabled"] = hit is not None
        item["sharedagents"] = (
            {
                "alias": hit.get("alias"),
                "id": hit.get("id"),
                "display": hit.get("display"),
            }
            if hit
            else None
        )
        # The audit's point: a 92%-exact slug join and a soft alias guess must not
        # look identical in the payload, so the UI can render them differently.
        item["match_step"] = hit.get("match") if hit else None
        item["match_exact"] = bool(hit and hit.get("exact"))
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def _normalize_model(row: dict[str, Any]) -> dict[str, Any]:
    """One AA row -> the stable chart row used by the UI.

    Each metric is read through :func:`_mapping` at every level (the
    intermediate cost container is itself null on many rows) and kept as
    ``None`` when absent — no zero-coercion, so a chart can skip it honestly.
    """
    evaluations = _mapping(row.get("evaluations"))
    cost_block = _mapping(row.get("artificial_analysis_intelligence_index_cost"))
    cost_per_task = _mapping(cost_block.get("cost_per_task"))
    performance = _mapping(row.get("performance"))
    pricing = _mapping(row.get("pricing"))
    creator = _mapping(row.get("model_creator"))
    name = str(row.get("name") or row.get("slug") or "").strip()
    return {
        "aa_id": str(row.get("id") or ""),
        "name": name,
        "slug": str(row.get("slug") or "").strip(),
        "provider": str(creator.get("name") or "").strip(),
        "release_date": row.get("release_date") or None,
        "variant": qualifier(name),
        "intelligence": _as_float(evaluations.get("artificial_analysis_intelligence_index")),
        "speed_tps": _as_float(performance.get("median_output_tokens_per_second")),
        "cost_per_task_usd": _as_float(cost_per_task.get("total_cost")),
        "usd_per_1m_in": _as_float(pricing.get("price_1m_input_tokens")),
        "usd_per_1m_out": _as_float(pricing.get("price_1m_output_tokens")),
    }


# ---------------------------------------------------------------------------
# Intelligence vs cost Pareto, family freshness, default selection
# ---------------------------------------------------------------------------
#
# All of this is computed per request from the payload in hand — there is no
# model name, provider or index anywhere in it. The numbers below are the only
# fixed values: a cap, a log floor, and two quantiles.

#: Hard cap on the models the Pareto chart shows before any user interaction.
DEFAULT_SELECTION_CAP = 20
#: Cost axis is ``log10``; a genuine ``$0.00`` (AA publishes a few) has no log,
#: so it is plotted at this floor and labelled as such rather than dropped.
LOG_COST_FLOOR = 1e-4
#: The "most attractive quadrant" split: the upper-left quadrant of the plotted
#: cloud, i.e. the cheap half × the smart half. AA's own bounds are not
#: published (the site must not be scraped), so the region is derived from the
#: points actually on the chart instead of being a hardcoded rectangle.
QUADRANT_X_QUANTILE = 0.5
QUADRANT_Y_QUANTILE = 0.5
#: Pareto chart axes.
PARETO_X_FIELD = "cost_per_task_usd"
PARETO_Y_FIELD = "intelligence"
#: Family grouping: a token that is a parameter count ("27b", "1.5b", "8k")
#: identifies the deployment and stays in the family key; every other digit is
#: a version and is dropped, so "Gemini 3.8 Flash" and "Gemini 3.7 Flash" are
#: one family while "Llama 5 70B" and "Llama 5 8B" are not.
_SIZE_TOKEN_RE = re.compile(r"^\d+(?:[.,]\d+)?[bkm]$")
_TOKEN_SPLIT_RE = re.compile(r"[^0-9a-z.]+")
_NON_LETTER_RE = re.compile(r"[^a-z]+")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def has_chart_metrics(row: Any) -> bool:
    """True when a row reports BOTH the Pareto axes (the scatter's requirement)."""
    if not isinstance(row, dict):
        return False
    return _is_number(row.get(PARETO_X_FIELD)) and _is_number(row.get(PARETO_Y_FIELD))


def assign_keys(models: Sequence[dict[str, Any]]) -> None:
    """Stamp every row with the stable selection ``key`` used by the UI.

    AA slugs are unique in the live population (653 rows, 653 distinct slugs,
    2026-09-19) but that is not guaranteed by the API, so a collision falls back
    to ``slug#name`` and then to an ordinal — the key is always present, always
    unique, and never a model-name constant.
    """
    seen: set[str] = set()
    for index, row in enumerate(models):
        base = str(row.get("slug") or "").strip() or normalize_name(row.get("name"))
        base = base or f"model-{index}"
        key = base
        if key in seen:
            key = f"{base}#{normalize_name(row.get('name')) or index}"
        if key in seen:
            key = f"{key}#{index}"
        seen.add(key)
        row["key"] = key


def pareto_frontier(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The non-dominated rows, ordered by cost ascending.

    A row is dominated when another row costs no more AND scores no less, with at
    least one of the two strict — so two rows with identical cost and identical
    intelligence dominate nothing, and both stay on the frontier (AA publishes
    exactly that shape: one row per effort variant). Rows missing either metric
    are not eligible and are never plotted at ``0``.

    Implementation is the ascending-cost sweep with a running best intelligence:
    within one cost, the highest intelligence wins and ties all survive, and a
    later cost only joins when it strictly beats the best intelligence so far.
    """
    points = sorted(
        (row for row in rows if has_chart_metrics(row)),
        key=lambda row: (float(row[PARETO_X_FIELD]), -float(row[PARETO_Y_FIELD])),
    )
    frontier: list[dict[str, Any]] = []
    best: float | None = None
    index = 0
    while index < len(points):
        cost = float(points[index][PARETO_X_FIELD])
        group: list[dict[str, Any]] = []
        while index < len(points) and float(points[index][PARETO_X_FIELD]) == cost:
            group.append(points[index])
            index += 1
        top = max(float(row[PARETO_Y_FIELD]) for row in group)
        if best is None or top > best:
            frontier.extend(row for row in group if float(row[PARETO_Y_FIELD]) == top)
            best = top
    return frontier


def _quantile(values: Sequence[float], fraction: float) -> float | None:
    """Linear-interpolation quantile of a non-empty numeric list."""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, float(fraction))) * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def quadrant_bounds(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The "most attractive quadrant" RULE, plus the population's own corner.

    The shaded region is the upper-left quadrant of the cloud that is actually
    plotted: its right edge is the :data:`QUADRANT_X_QUANTILE` quantile of the
    plotted cost-per-task values and its floor is the
    :data:`QUADRANT_Y_QUANTILE` quantile of the plotted Intelligence Index
    values. The page applies this rule to the points it draws, so the region
    always covers the cheap-and-smart part of what is on screen — it is never an
    empty corner and never a hardcoded rectangle. AA's own bounds are not
    published by the API and their site must not be scraped, which is exactly
    why the region is derived from the data in hand.

    ``population`` carries the same quantiles over every model that reports both
    metrics, for the UI's explanation text (not for drawing).
    """
    costs = [float(row[PARETO_X_FIELD]) for row in rows if has_chart_metrics(row)]
    scores = [float(row[PARETO_Y_FIELD]) for row in rows if has_chart_metrics(row)]
    return {
        "x_quantile": QUADRANT_X_QUANTILE,
        "y_quantile": QUADRANT_Y_QUANTILE,
        "x_floor": LOG_COST_FLOOR,
        "basis": (
            "upper-left quadrant of the plotted cloud: cost q"
            f"{QUADRANT_X_QUANTILE:g} and intelligence q{QUADRANT_Y_QUANTILE:g} "
            "of the points on the chart"
        ),
        "population": {
            "x_max": _quantile(costs, QUADRANT_X_QUANTILE),
            "y_min": _quantile(scores, QUADRANT_Y_QUANTILE),
            "model_count": len(costs),
        },
    }


def family_key(value: Any) -> str:
    """The model *lineage* a name belongs to, with versions stripped out.

    ``"Gemini 3.8 Flash (high)"`` → ``"gemini flash"``;
    ``"Claude Opus 4.8 (Adaptive Reasoning, Max Effort)"`` → ``"claude opus"``;
    ``"Llama 5 70B (Non-reasoning)"`` → ``"llama 70b"`` (the parameter count is
    identity, not a version). A name with nothing but digits falls back to its
    own normalised form, so unrelated numeric names never share a family.
    """
    text = _fold_plus(_bare_name(str(value or ""))).lower()
    parts: list[str] = []
    for token in _TOKEN_SPLIT_RE.split(text):
        if not token:
            continue
        if _SIZE_TOKEN_RE.match(token):
            parts.append(token)
            continue
        letters = _NON_LETTER_RE.sub("", token)
        if letters:
            parts.append(letters)
    key = " ".join(parts).strip()
    return key or normalize_name(value) or str(value or "").strip().lower()


def version_tuple(value: Any) -> tuple[int, ...]:
    """The ordered digits of a name as integers (``"GPT-5.6 Luna"`` → ``(5, 6)``)."""
    return tuple(int(part) for part in _DIGIT_RE.findall(str(value or "")))


def release_ordinal(value: Any) -> float:
    """``release_date`` as a sortable number; ``0.0`` when absent/unparseable."""
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _model_rank(rows: Sequence[dict[str, Any]]) -> tuple[float, tuple[int, ...], float]:
    """Freshness of one model (its rows share a bare name): newest first."""
    date = max((release_ordinal(row.get("release_date")) for row in rows), default=0.0)
    version = max((version_tuple(row.get("name")) for row in rows), default=())
    best_score = max(
        (float(row[PARETO_Y_FIELD]) for row in rows if has_chart_metrics(row)),
        default=0.0,
    )
    return (date, version, best_score)


def latest_per_family(rows: Sequence[dict[str, Any]]) -> set[str]:
    """Keys of the rows that survive the "latest per family" filter.

    Two levels, so that obsolete generations go and effort variants stay:

    1. group by lineage (:func:`family_key`), then by model (the qualifier-free
       name — every effort/reasoning spelling of one model, which is how AA
       publishes them);
    2. keep the newest model of the lineage by ``(release_date, version tokens,
       best intelligence)``, and within it every row tied on
       ``(release_date, version tokens)``.

    So ``Gemini 3.7 Flash`` is dropped once ``Gemini 3.8 Flash`` exists (and
    ``Claude Opus 4.6`` once ``Claude Opus 5`` exists), while the six effort
    variants of one model all survive — they are one model at six price points,
    not six generations. The version tokens are the documented fallback for a
    row with no ``release_date`` at all.
    """
    families: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        lineage = family_key(row.get("name") or row.get("slug"))
        model = _bare_name(str(row.get("name") or row.get("slug") or "")).lower()
        families.setdefault(lineage, {}).setdefault(model, []).append(row)
    keep: set[str] = set()
    for models in families.values():
        best_model: tuple[tuple[float, tuple[int, ...], float], str] | None = None
        for name, group in models.items():
            rank = _model_rank(group)
            if best_model is None or (rank, name) > best_model:
                best_model = (rank, name)
        if best_model is None:
            continue
        winner = models[best_model[1]]
        within = max((release_ordinal(row.get("release_date")), version_tuple(row.get("name"))) for row in winner)
        for row in winner:
            if (release_ordinal(row.get("release_date")), version_tuple(row.get("name"))) == within:
                key = row.get("key")
                if isinstance(key, str) and key:
                    keep.add(key)
    return keep


def frontier_distances(
    rows: Sequence[dict[str, Any]],
    frontier: Sequence[dict[str, Any]],
) -> dict[str, float]:
    """Normalised distance from each row to the nearest frontier point.

    Both axes are scaled to ``[0, 1]`` over the population in ``rows`` — cost
    through ``log10`` (clamped at :data:`LOG_COST_FLOOR`), intelligence linearly
    — then the Euclidean distance to the closest frontier point is taken. This is
    the "nearest the Pareto frontier" ranking of :func:`select_default`; a row on
    the frontier scores ``0.0``.
    """
    points = [row for row in rows if has_chart_metrics(row)]
    out: dict[str, float] = {}
    if not points or not frontier:
        return out
    logs = [math.log10(max(float(row[PARETO_X_FIELD]), LOG_COST_FLOOR)) for row in points]
    scores = [float(row[PARETO_Y_FIELD]) for row in points]
    log_span = (max(logs) - min(logs)) or 1.0
    score_span = (max(scores) - min(scores)) or 1.0

    def scaled(row: dict[str, Any]) -> tuple[float, float]:
        x = (math.log10(max(float(row[PARETO_X_FIELD]), LOG_COST_FLOOR)) - min(logs)) / log_span
        y = (float(row[PARETO_Y_FIELD]) - min(scores)) / score_span
        return (x, y)

    frontier_xy = [scaled(row) for row in frontier]
    for row in points:
        key = row.get("key")
        if not isinstance(key, str) or not key:
            continue
        x, y = scaled(row)
        out[key] = min(math.hypot(x - fx, y - fy) for fx, fy in frontier_xy)
    return out


def select_default(
    rows: Sequence[dict[str, Any]],
    frontier: Sequence[dict[str, Any]] | None = None,
    *,
    cap: int = DEFAULT_SELECTION_CAP,
) -> list[dict[str, Any]]:
    """The models the Pareto chart opens with — at most ``cap``, computed live.

    Priority order (documented here and surfaced in the UI's "why these models"
    affordance):

    1. **local** — every deployment sharedagents serves from this machine that AA
       benchmarks. Included because we run it: never filtered out by
       "Latest per family" and never outranked by a frontier point;
    2. **frontier** — the non-dominated points, highest intelligence first;
    3. **nearest** — the remaining candidates, closest to the frontier in
       normalised ``(log10 cost, intelligence)`` space, then by intelligence and
       cost.

    The cap is a constant; the MEMBERSHIP is not — change the payload (or the
    catalogue on disk) and this list changes with it. If more deployments are
    local than the cap allows, the highest-intelligence local ones win, because
    "at most 10" is the hard rule. Returns ``[]`` for an empty population.
    """
    pool = [row for row in rows if has_chart_metrics(row)]
    if not pool:
        return []
    points = list(frontier) if frontier is not None else pareto_frontier(pool)
    distances = frontier_distances(pool, points)
    chosen: list[tuple[dict[str, Any], str]] = []
    taken: set[str] = set()

    def take(row: dict[str, Any], reason: str) -> None:
        key = row.get("key")
        if not isinstance(key, str) or not key or key in taken:
            return
        taken.add(key)
        chosen.append((row, reason))

    for row in sorted(
        (row for row in pool if row.get("local")),
        key=lambda row: (-float(row[PARETO_Y_FIELD]), float(row[PARETO_X_FIELD]), row.get("key") or ""),
    ):
        take(row, "local")
    for row in sorted(
        points,
        key=lambda row: (-float(row[PARETO_Y_FIELD]), float(row[PARETO_X_FIELD]), row.get("key") or ""),
    ):
        take(row, "frontier")
    for row in sorted(
        pool,
        key=lambda row: (
            distances.get(str(row.get("key") or ""), float("inf")),
            -float(row[PARETO_Y_FIELD]),
            float(row[PARETO_X_FIELD]),
            row.get("key") or "",
        ),
    ):
        take(row, "nearest")

    out: list[dict[str, Any]] = []
    for row, reason in chosen[: max(0, int(cap))]:
        key = str(row.get("key") or "")
        out.append(
            {
                "key": key,
                "slug": str(row.get("slug") or ""),
                "name": str(row.get("name") or ""),
                "provider": str(row.get("provider") or ""),
                "reason": reason,
                "distance": round(distances.get(key, 0.0), 6),
                "local": bool(row.get("local")),
                "on_frontier": any(row is point for point in points),
                "cost_per_task_usd": row.get(PARETO_X_FIELD),
                "intelligence": row.get(PARETO_Y_FIELD),
            }
        )
    return out


def _pareto_rows(
    frontier: Sequence[dict[str, Any]],
    *,
    keys: set[str],
) -> list[dict[str, Any]]:
    """The frontier as the chart consumes it: coordinates, identity, cost asc."""
    out: list[dict[str, Any]] = []
    for row in frontier:
        key = str(row.get("key") or "")
        if key not in keys:
            continue
        out.append(
            {
                "key": key,
                "slug": str(row.get("slug") or ""),
                "name": str(row.get("name") or ""),
                "provider": str(row.get("provider") or ""),
                "cost_per_task_usd": row.get(PARETO_X_FIELD),
                "intelligence": row.get(PARETO_Y_FIELD),
                "local": bool(row.get("local")),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Payload assembly
# ---------------------------------------------------------------------------


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _base_payload() -> dict[str, Any]:
    """Every payload carries the same top-level contract, incl. attribution.

    The Pareto/selection keys are part of that contract, so a cache hit, a live
    fetch, the fixture and an error payload all hand the page the same shape —
    the browser never has to guess whether a field is missing or empty.
    """
    return {
        "updated_at": _iso_now(),
        "source": "cache",
        "tier": None,
        "intelligence_index_version": None,
        # Provenance for the bundled-snapshot fallback; None for live/cache/
        # fixture/error so the UI only shows it when source == "snapshot".
        "snapshot_date": None,
        "attribution": ATTRIBUTION,
        "attribution_url": ATTRIBUTION_URL,
        "pages_fetched": 0,
        "rate_limit": None,
        "models": [],
        "model_count": 0,
        "enabled_count": 0,
        "error": None,
        # Intelligence vs cost Pareto (requirement: computed per request).
        "pareto": [],
        "pareto_count": 0,
        "pareto_all": [],
        "quadrant": None,
        "default_selection": [],
        "default_selection_all": [],
        "default_selection_cap": DEFAULT_SELECTION_CAP,
        "default_selection_basis": (
            "local deployments first, then Pareto-frontier points by "
            "intelligence, then the models nearest the frontier in "
            "(log10 cost, intelligence) space"
        ),
        # Locally-served deployments, discovered from disk on every call.
        "local": [],
        "local_count": 0,
        "local_unmatched": [],
        "local_unmatched_count": 0,
        "local_sources": [],
        # Presentation data resolved server-side so every chart agrees.
        "provider_colors": {},
        "provider_ids": {},
        "colors": {},
    }


def _finalize(data: dict[str, Any], *, source: str) -> dict[str, Any]:
    """Overlay live-derived fields on a snapshot: enabled, local, Pareto, selection.

    Every derived field here is recomputed on every read — cache hits included —
    because none of it is a property of the snapshot: ``enabled`` follows the
    sharedagents catalogue, ``local`` follows what this machine is serving right now,
    and the Pareto/selection fields follow the metrics in the payload.
    """
    payload = _base_payload()
    payload.update({key: value for key, value in data.items() if key in payload})
    models = [row for row in (data.get("models") or []) if isinstance(row, dict)]
    models = _apply_enabled(models)
    # One disk read per request: the same local catalogue both joins the payload
    # and answers "what are we running that AA does not benchmark?".
    local_rows = local_models()
    models, local_unmatched = _apply_local(models, local_rows)
    assign_keys(models)
    # "Latest per family" is a UI toggle, so both pools are prepared here: the
    # browser picks a variant, it never re-derives which rows are the newest.
    fresh = latest_per_family(models)
    for row in models:
        row["family_latest"] = str(row.get("key") or "") in fresh
    pool = [row for row in models if has_chart_metrics(row)]
    filtered = [row for row in pool if row.get("family_latest") or row.get("local")]
    frontier = pareto_frontier(filtered) if filtered else []
    frontier_all = pareto_frontier(pool) if pool else []
    frontier_keys = {str(row.get("key") or "") for row in frontier}
    frontier_all_keys = {str(row.get("key") or "") for row in frontier_all}
    for row in models:
        key = str(row.get("key") or "")
        row["on_frontier"] = key in frontier_keys
        row["on_frontier_all"] = key in frontier_all_keys

    payload["models"] = models
    payload["source"] = source
    payload["attribution"] = ATTRIBUTION
    payload["attribution_url"] = ATTRIBUTION_URL
    payload["model_count"] = len(models)
    payload["enabled_count"] = sum(1 for row in models if row.get("enabled"))
    payload["pareto"] = _pareto_rows(frontier, keys=frontier_keys)
    payload["pareto_all"] = _pareto_rows(frontier_all, keys=frontier_all_keys)
    payload["pareto_count"] = len(payload["pareto"])
    payload["quadrant"] = quadrant_bounds(pool)
    payload["default_selection"] = select_default(filtered, frontier)
    payload["default_selection_all"] = select_default(pool, frontier_all)
    matched_keys: dict[str, str] = {}
    for row in models:
        if row.get("local"):
            matched_keys[_local_identity(_mapping(row.get("local_deployment")))] = str(row.get("key") or "")
    payload["local"] = [
        {
            "alias": str(row.get("alias") or ""),
            "id": str(row.get("id") or ""),
            "display": str(row.get("display") or ""),
            "source": str(row.get("source") or ""),
            "matched": _local_identity(row) in matched_keys,
            "key": matched_keys.get(_local_identity(row)),
        }
        for row in local_rows
    ]
    payload["local_count"] = len(payload["local"])
    payload["local_matched_count"] = len(matched_keys)
    payload["local_unmatched"] = [
        {
            "alias": str(row.get("alias") or ""),
            "id": str(row.get("id") or ""),
            "display": str(row.get("display") or ""),
            "source": str(row.get("source") or ""),
            "kind": str(row.get("kind") or ""),
        }
        for row in local_unmatched
    ]
    payload["local_unmatched_count"] = len(payload["local_unmatched"])
    payload["local_sources"] = list(LOCAL_SOURCES)
    provider_pairs = [
        (row.get("provider"), (data.get("provider_ids") or {}).get(str(row.get("provider") or ""))) for row in models
    ]
    payload["provider_colors"] = dict(data.get("provider_colors") or {})
    payload["provider_ids"] = dict(data.get("provider_ids") or {})
    payload["colors"] = provider_color_map(provider_pairs)
    return payload


def _empty_payload(error: dict[str, Any]) -> dict[str, Any]:
    payload = _base_payload()
    payload["source"] = "error"
    payload["error"] = error
    return payload


def _creator_ids(raw: dict[str, Any]) -> dict[str, str]:
    """``{creator name: creator id}`` from the raw AA rows.

    The id is the stable half of ``model_creator`` (a UUID that survives a
    published rename), and it is what the palette's deterministic fallback hashes
    when the curated table does not know the lab. Carried with the cached
    payload so a cache read colours exactly like the live fetch did.
    """
    out: dict[str, str] = {}
    for row in raw.get("data") or []:
        if not isinstance(row, dict):
            continue
        creator = _mapping(row.get("model_creator"))
        name = str(creator.get("name") or "").strip()
        creator_id = str(creator.get("id") or "").strip()
        if name and creator_id and name not in out:
            out[name] = creator_id
    return out


def _normalize_raw(raw: dict[str, Any]) -> dict[str, Any]:
    """AA-shaped ``{data:[…], pagination:…}`` -> pre-finalize payload."""
    return {
        "updated_at": raw.get("updated_at") or _iso_now(),
        "tier": raw.get("tier"),
        "intelligence_index_version": raw.get("intelligence_index_version"),
        "pages_fetched": raw.get("pages_fetched"),
        "rate_limit": raw.get("rate_limit"),
        "provider_ids": _creator_ids(raw),
        "models": [_normalize_model(row) for row in raw.get("data", []) if isinstance(row, dict)],
    }


# ---------------------------------------------------------------------------
# Fixture (offline)
# ---------------------------------------------------------------------------


def _fixture_row(
    *,
    aa_id: str,
    name: str,
    slug: str,
    creator: str,
    release: str | None,
    intelligence: float | None,
    speed: float | None,
    cost: float | None,
    price_in: float | None = 1.0,
    price_out: float | None = 4.0,
    coding: float | None = None,
    agentic: float | None = None,
) -> dict[str, Any]:
    """AA-shaped row; ``cost=None`` emits the container itself as null."""
    return {
        "id": aa_id,
        "name": name,
        "slug": slug,
        "release_date": release,
        "model_creator": {"id": creator.lower(), "name": creator, "slug": creator.lower()},
        "evaluations": {
            "artificial_analysis_intelligence_index": intelligence,
            "artificial_analysis_coding_index": coding,
            "artificial_analysis_agentic_index": agentic,
        },
        "artificial_analysis_intelligence_index_cost": (
            None
            if cost is None
            else {
                "total_cost": cost,
                "cost_per_task": {"total_cost": cost},
            }
        ),
        "pricing": {
            "price_1m_input_tokens": price_in,
            "price_1m_output_tokens": price_out,
            "price_1m_cache_hit_tokens": None,
            "price_1m_cache_write_tokens": None,
        },
        "performance": {
            "median_output_tokens_per_second": speed,
            "median_time_to_first_token_seconds": 0.5 if speed else None,
            "median_end_to_end_response_time_seconds": 8.0 if speed else None,
        },
    }


def fixture_payload(now: str | None = None) -> dict[str, Any]:
    """Deterministic AA-shaped payload — no network, no cache write.

    Covers the shapes the live endpoint actually returns: all-metrics-present,
    null intelligence, null speed, null cost (with the intermediate
    ``artificial_analysis_intelligence_index_cost`` container itself null),
    all-null, multiple effort variants of one family, and one model sharedagents has
    enabled (``Claude Opus 5`` / ``claude-opus-5``, which is in the shipped
    catalogue on every machine).
    """
    rows = [
        _fixture_row(
            aa_id="fixture-0001",
            name="Claude Opus 5 (Adaptive Reasoning, High Effort)",
            slug="claude-opus-5-high",
            creator="Anthropic",
            release="2026-05-14",
            intelligence=62.4,
            speed=48.1,
            cost=1.24,
            price_in=5.0,
            price_out=25.0,
            coding=58.2,
            agentic=54.0,
        ),
        _fixture_row(
            aa_id="fixture-0002",
            name="Claude Opus 5 (Adaptive Reasoning, Xhigh Effort)",
            slug="claude-opus-5-xhigh",
            creator="Anthropic",
            release="2026-05-14",
            intelligence=64.9,
            speed=27.6,
            cost=2.87,
            price_in=5.0,
            price_out=25.0,
            coding=60.1,
            agentic=55.8,
        ),
        _fixture_row(
            aa_id="fixture-0003",
            name="Claude Opus 4.6 (Non-reasoning, High Effort)",
            slug="claude-opus-4-6-non-reasoning-high",
            creator="Anthropic",
            release="2026-02-03",
            intelligence=54.1,
            speed=71.3,
            cost=0.41,
            price_in=3.0,
            price_out=15.0,
            coding=49.7,
            agentic=46.2,
        ),
        _fixture_row(
            aa_id="fixture-0004",
            name="Kimi K3 (low)",
            slug="kimi-k3-low",
            creator="Moonshot AI",
            release="2026-06-21",
            intelligence=58.8,
            speed=63.4,
            cost=None,
            price_in=0.6,
            price_out=2.4,
            coding=55.1,
            agentic=51.0,
        ),
        _fixture_row(
            aa_id="fixture-0005",
            name="GPT-5.6 Luna",
            slug="gpt-5-6-luna",
            creator="OpenAI",
            release="2026-07-02",
            intelligence=60.2,
            speed=88.5,
            cost=0.32,
            price_in=1.25,
            price_out=10.0,
            coding=57.9,
            agentic=56.4,
        ),
        _fixture_row(
            aa_id="fixture-0006",
            name="Gemini 3.8 Flash (high)",
            slug="gemini-3-8-flash-high",
            creator="Google",
            release="2026-08-19",
            intelligence=None,
            speed=142.7,
            cost=0.11,
            price_in=0.4,
            price_out=2.5,
        ),
        _fixture_row(
            aa_id="fixture-0007",
            name="DeepSeek V4 Flash",
            slug="deepseek-v4-flash",
            creator="DeepSeek",
            release="2026-04-11",
            intelligence=47.3,
            speed=55.0,
            cost=None,
            price_in=0.28,
            price_out=0.42,
        ),
        _fixture_row(
            aa_id="fixture-0008",
            name="Qwen3.8 27B",
            slug="qwen3-8-27b",
            creator="Alibaba",
            release="2026-03-27",
            intelligence=44.6,
            speed=None,
            cost=0.09,
            price_in=0.1,
            price_out=0.3,
        ),
        _fixture_row(
            aa_id="fixture-0009",
            name="Grok 4.6",
            slug="grok-4-6",
            creator="xAI",
            release="2026-06-05",
            intelligence=57.0,
            speed=74.2,
            cost=0.58,
            price_in=2.0,
            price_out=8.0,
        ),
        _fixture_row(
            aa_id="fixture-0010",
            name="Mistral Medium 3.5",
            slug="mistral-medium-3-5",
            creator="Mistral AI",
            release="2026-05-30",
            intelligence=None,
            speed=None,
            cost=None,
            price_in=None,
            price_out=None,
        ),
        _fixture_row(
            aa_id="fixture-0011",
            name="Llama 5 70B (Non-reasoning)",
            slug="llama-5-70b-non-reasoning",
            creator="Meta",
            release="2026-01-22",
            intelligence=39.8,
            speed=96.4,
            cost=0.14,
            price_in=0.2,
            price_out=0.6,
        ),
    ]
    total_pages = 1
    return {
        "tier": "free",
        "intelligence_index_version": "2.0",
        "pagination": {
            "page": 1,
            "page_size": _PAGE_SIZE,
            "total_pages": total_pages,
            "has_more": False,
        },
        "data": rows,
        "pages_fetched": 1,
        "rate_limit": {"limit": 100, "remaining": 100, "reset": None},
        "updated_at": now or _iso_now(),
    }


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_cache() -> dict[str, Any] | None:
    return _read_json(cache_path()) or None


def _write_cache(data: dict[str, Any]) -> None:
    """Persist the snapshot.

    ``enabled`` / ``sharedagents`` are deliberately NOT cached — they are re-derived
    from the live catalogue on every read, so enabling a model shows up
    immediately instead of after the TTL. ``rate_limit`` IS cached: it is the
    upstream budget as of ``updated_at`` (which the UI displays next to the
    counts), and the alternative is an empty budget figure on every cache read.
    """
    path = cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    except OSError:
        return


def _read_fresh_cache() -> dict[str, Any] | None:
    data = _read_cache()
    if data is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(data.get("updated_at")).replace("Z", "+00:00"))
        age = time.time() - parsed.timestamp()
    except (TypeError, ValueError, OSError):
        return None
    if age < 0 or age > cache_ttl_s():
        return None
    if not isinstance(data.get("models"), list):
        return None
    return data


def offline_models() -> list[dict[str, Any]]:
    """Normalised benchmark rows already on this machine; never the network.

    Fixture (when enabled) → the saved cache at any age → the bundled
    snapshot. For callers on a model-facing path (the delegation options
    list) that may show a public score but must never wait on the API.
    """
    if fixture_enabled():
        data: dict[str, Any] | None = _normalize_raw(fixture_payload())
    else:
        data = _read_cache()
        if not data or not isinstance(data.get("models"), list):
            data = _load_snapshot()
    models = (data or {}).get("models")
    return [row for row in models if isinstance(row, dict)] if isinstance(models, list) else []


def intelligence_by_model(models: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Public intelligence index per catalogue model id, from :func:`offline_models`.

    ``models`` are catalogue rows (``id`` / ``alias`` / ``display``); the join
    is the same heuristic cascade the Models tab uses. A model with several
    benchmark rows (effort variants) takes the most exact match, then the
    highest score. Models the public catalogue does not list are absent.
    """
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in models:
        if not isinstance(raw, Mapping):
            continue
        model_id = str(raw.get("id") or "").strip()
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        rows.append(
            {
                "id": model_id,
                "alias": str(raw.get("alias") or "").strip(),
                "display": str(raw.get("display") or "").strip(),
            }
        )
    if not rows:
        return {}
    # A benchmark can describe several configured deployments. Joining the
    # whole deployment catalogue at once chooses just one winning row and
    # drops its siblings, so match each distinct model id independently.
    catalogue = offline_models()
    scores: dict[str, float] = {}
    for row in rows:
        steps = _steps([row])
        best: tuple[tuple[Any, ...], float] | None = None
        for bench in catalogue:
            score = bench.get("intelligence")
            if not _is_number(score):
                continue
            hit = _match_in(steps, bench.get("name"), bench.get("slug"))
            if hit is None:
                continue
            step = hit.get("match")
            rank = (
                not hit.get("exact"),
                _STEP_ORDER.index(step) if step in _STEP_ORDER else len(_STEP_ORDER),
                -float(score),
            )
            if best is None or rank < best[0]:
                best = (rank, float(score))
        if best is not None:
            scores[row["id"]] = best[1]
    return scores


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def fetch_benchmarks(
    *,
    refresh: bool = False,
    http: HttpFn | None = None,
) -> dict[str, Any]:
    """Cache-first benchmark payload for the Models tab.

    Order: fixture (``SHAREDAGENTS_BENCHMARKS_FIXTURE=1``, no network, no cache
    write) → fresh cache (unless ``refresh``) → live API keyed by
    ``ARTIFICIAL_ANALYSIS_API_KEY``. A missing key, a 401, a 429, an HTTP
    error, or an unreachable host all come back as a structured ``error`` on an
    otherwise complete payload — this never raises and never returns a 500.

    Fresh-install fallback: with no key AND no cache on disk, the bundled
    snapshot (``data/models_snapshot.json``) is served with
    ``source="snapshot"`` and a ``snapshot_date`` so the page still renders.
    The live path is untouched — a key present always wins.

    ``enabled`` / ``sharedagents`` are re-derived on every call, so a cache hit
    still reflects the catalogue as it is right now.
    """
    if fixture_enabled():
        return _finalize(_normalize_raw(fixture_payload()), source="fixture")

    if not refresh:
        cached = _read_fresh_cache()
        if cached is not None:
            return _finalize(cached, source="cache")

    key = api_key()
    if not key:
        # Fresh-install fallback: no key AND no cache on disk -> serve the
        # bundled snapshot so the Models page still renders, with provenance
        # (source="snapshot", snapshot_date) for the Source KPI. A cache file
        # present means a prior run already populated data, so we keep the
        # existing no_api_key error in that case. A missing/corrupt snapshot
        # degrades to the same structured error rather than a 500.
        if not cache_path().is_file():
            snapshot = _load_snapshot()
            if snapshot is not None:
                return _finalize(snapshot, source="snapshot")
        return _empty_payload(
            {
                "code": "no_api_key",
                "message": (
                    f"No {API_KEY_ENV} configured for the web server "
                    f"(checked the process env, {env_file_hint()}, and /etc/environment)."
                ),
                "status": None,
                "env_var": API_KEY_ENV,
                "env_file": env_file_hint(),
                "signup_url": SIGNUP_URL,
            }
        )

    try:
        raw = _fetch_all_pages(key, http)
    except BenchmarkError as exc:
        return _empty_payload(exc.as_dict())
    except Exception as exc:  # noqa: BLE001 - never let a fetch reach the route
        return _empty_payload(
            {
                "code": "unexpected_error",
                "message": f"Benchmark fetch failed: {type(exc).__name__}",
                "status": None,
            }
        )

    data = _normalize_raw(raw)
    _write_cache(data)
    return _finalize(data, source="live")
