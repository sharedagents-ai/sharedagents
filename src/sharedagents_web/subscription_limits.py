"""Live subscription-quota snapshots for Codex, Claude, Cursor, AGY, and ZCode.

Each probe reuses the credential the matching CLI already stored. There is no
supported public subscriber API for these five plans; the endpoints below are
the same undocumented quota surfaces the CLIs and community monitors call.

Sources (2026-09 / 2026-10):
- Codex: ChatGPT ``/backend-api/wham/usage`` (Bearer from ``~/.codex/auth.json``)
- Claude: ``GET /api/oauth/usage`` (OAuth from ``~/.claude/.credentials.json``)
- Cursor: ``DashboardService/GetCurrentPeriodUsage`` (``~/.config/cursor/auth.json``)
- AGY: Cloud Code ``retrieveUserQuotaSummary`` (``~/.gemini/antigravity-cli/…``)
- ZCode: Z.AI ``GET /api/monitor/usage/quota/limit`` (Bearer ``ZCODE_API_KEY``,
  the GLM Coding Plan key the zcode CLI provisions — the same quota surface
  z.ai's own account dashboard and community monitors read. The CLI's stored
  copy in ``~/.zcode/v2/credentials.json`` is encrypted with the CLI's own
  keychain, so the probe takes the key the operator exported). The exported
  key is NOT what runs turns: ``zcode`` dispatches on its own OAuth login in
  that same store, so a stale export blanks the meters while the engine keeps
  serving. ``_zcode_cli_login`` reads that distinction (presence only, never
  the ciphertext) so a working engine is never reported as offline.
- MiMo: no usage API exists (Xiaomi documents the console progress bar only),
  so ``mimo_plan`` ESTIMATES the Token Plan's credits from sharedagents's own token
  ledger (``turn_costs``) at Xiaomi's published per-token credit rates.
- Devin: the ``cog_`` key cannot read quota (the CLI's Connect RPC rejects
  it), so ``_fetch_devin`` validates it against ``GET api.devin.ai/v3/self``
  (org id), then calls the CLI's own ``GetUserStatus`` RPC with the credential
  the devin CLI stored (``~/.local/share/devin/credentials.toml``) — a live
  reading on every refresh, like the other providers. Without that credential
  or when the RPC fails it falls back to the ``GetUserStatusResponse`` the
  CLI caches after each session (``~/.cache/devin/cli/user_status.*.bin``),
  stamped with the cache's own age so an old copy is never paced as live.

Routing (``routing_digest``) treats each vendor as a SET OF LANES, not one
number: every window is classified by model family (claude / gpt / gemini) and
window kind (short / long / spend), and each lane gets a computed access verdict
— whether the engine's real dispatch gate accepts a model of that family, and
whether the engine's own pinned default would actually spend that lane.
"""

from __future__ import annotations

import base64
import calendar
import copy
import json
import logging
import math
import os
import re
import selectors
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from sharedagents_core import oscompat
from sharedagents_core.layout import agents_root, sharedagents_root, usage_root

_LOG = logging.getLogger(__name__)

PROVIDER_IDS = ("codex", "claude", "cursor", "agy", "zcode", "mimo", "opencode", "devin")
PROVIDER_AGENTS = {
    "codex": "codex",
    "claude": "claude",
    "cursor": "cursor",
    "agy": "antigravity",
    "kimi": "kimi",
    "zcode": "zcode",
    "mimo": "mimo",
    "opencode": "opencode",
    "devin": "devin",
}
PROVIDER_NAMES = {
    "codex": "Codex",
    "claude": "Claude",
    "cursor": "Cursor",
    "agy": "Antigravity",
    "openrouter": "OpenRouter",
    "deepseek": "DeepSeek",
    "kimi": "Kimi",
    "xai": "xAI",
    "cerebras": "Cerebras",
    "zcode": "ZCode (Z.AI)",
    "mimo": "MiMo (Xiaomi)",
    "opencode": "OpenCode",
    "devin": "Devin",
}

#: Claude ``limits[]`` kinds that restate a top-level quota bucket under another
#: name (``session`` is the 5-hour window); skipped when that bucket was parsed.
_CLAUDE_LIMIT_BUCKETS = {"session": "five_hour", "weekly_all": "seven_day"}

#: Model families a quota lane can serve.
MODEL_FAMILIES: tuple[str, ...] = ("gemini", "claude", "gpt")

#: Canonical ordering for a lane that meters several families at once.
_FAMILY_ORDER: tuple[str, ...] = ("claude", "gpt", "gemini", "other")

#: Label / id / model-slug keywords per family — a substring scan, so AGY's
#: combined group ("Claude and GPT models · Weekly") matches BOTH ``claude`` and
#: ``gpt`` instead of silently picking one, and Claude's model-scoped "Fable"
#: bucket still lands on ``claude``. Deliberately NOT keyed on the window ``id``:
#: the Codex probe ids its primary window ``five_hour`` while the vendor meter is
#: weekly (``limit_window_seconds: 604800``, ``secondary_window: null``).
_FAMILY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("gemini", ("gemini", "gemma", "google", "palm")),
    ("claude", ("claude", "anthropic", "sonnet", "opus", "haiku", "fable")),
    ("gpt", ("gpt", "chatgpt", "openai", "codex", "oss-")),
)

#: Fallback used only when no keyword matched: the family that vendor's own CLI
#: ships by default. Cursor owns its own models and ZCode runs GLM, so their
#: lanes fall to ``other``.
_VENDOR_DEFAULT_FAMILY: dict[str, str] = {
    "codex": "gpt",
    "claude": "claude",
    "agy": "gemini",
    "cursor": "other",
    "zcode": "other",
    "mimo": "other",
    "opencode": "other",
    "devin": "other",
}

#: A window at or under a day is burst ("short") work; anything longer is
#: sustained ("long") work. Money rows are "spend".
_SHORT_WINDOW_MAX_S = 86_400.0
_LONG_WINDOW_TOKENS = ("week", "seven", "7d", "month", "28-day")
_SHORT_WINDOW_TOKENS = ("5-hour", "5h", "five_hour", "session", "hour", "day", "rolling")

_CACHE_NAME = "subscriptions.json"
_CACHE_TTL_S = 300.0
PACE_FRESHNESS_S = 2 * _CACHE_TTL_S
#: The pace pill is drawn 0–100% wide: 100% = on track to use the limit exactly
#: by its reset. Values above 100% fill the pill and read as "runs out early".
PACE_SCALE_MAX = 100.0
#: Used % is compared with the share of the period already passed, but never
#: with less than this share, so a burst in the first minutes after a reset is
#: not extrapolated over the whole period.
PACE_MIN_ELAPSED = 0.10
#: A limit with less than this percent left is hit: the plan cannot serve now.
LIMIT_HIT_REMAINING = 1.0
HORIZON_LABELS = {"short": "5-hour", "medium": "Weekly", "long": "Monthly"}
HORIZON_PERIOD_HOURS = {"short": 5.0, "medium": 168.0, "long": 720.0}
PACE_REASON_TEXT = {
    "stale_data": "data older than 10 min — press Update",
    "cached_data": "live reading unavailable — showing the CLI's cached copy",
    "unpinned": "no model pinned",
    "stale_reset": "reset time passed — press Update",
    "no_reset": "no reset time reported",
    "no_remaining": "no remaining % reported",
}
_DOMINANT_PACE_REASONS = (
    "stale_data",
    "unpinned",
    "stale_reset",
    "no_reset",
    "no_remaining",
)
_PACE_WINDOW_KEYS = (
    "pace_used_percent",
    "pace_elapsed_percent",
    "pace_period_hours",
    "pace_hours_to_reset",
    "pace_status",
    "pace_reason",
    "pace_horizon",
)
_PACE_ROW_KEYS = (
    "pace_used_percent",
    "pace_status",
    "pace_reason",
    "pace_basis",
    "pace_partial",
    "pace_horizon",
    "pace_resets_at",
    "pace_limit_hit",
    "pace_use_next",
    "pace_pools",
    "pace_engine_limit_hit",
    "pace_engine_pool",
    "pace_engine_resets_at",
)
_HTTP_TIMEOUT_S = 12.0
_MAX_BODY = 2 * 1024 * 1024

_FIXTURE_GATE: dict[str, dict[str, Any]] = {
    "standard": {
        "codex": {
            "kind": "codex_cli",
            "pin": "gpt-6-astra",
            "catalogue": {"gpt": ["gpt-6-astra", "gpt-5.6-luna"]},
        },
        "claude": {
            "kind": "claude_cli",
            "pin": "claude-opus-4-6",
            "catalogue": {"claude": ["claude-opus-4-6", "claude-fable-5"]},
        },
        "agy": {
            "kind": "antigravity_cli",
            "pin": "gemini-3.7-flash-medium",
            "catalogue": {
                "gemini": ["gemini-3.7-flash-medium", "gemini-3.8-flash-high"],
                "claude": ["claude-opus-4-6-thinking", "claude-sonnet-4-6"],
                "gpt": ["gpt-oss-120b-medium"],
            },
        },
        "cursor": {"kind": "cursor_acp", "pin": "", "catalogue": {}},
        "zcode": {
            "kind": "zcode_cli",
            "pin": "glm-5.3",
            "catalogue": {"other": ["glm-5.3", "glm-5.3-flash"]},
        },
        "opencode": {
            "kind": "opencode_acp",
            "pin": "opencode-go/grok-4.7",
            "catalogue": {"other": ["opencode-go/grok-4.7", "opencode-go/qwen3.8-flash"]},
        },
    }
}
_FIXTURE_GATE["pace-edge"] = {
    key: ({**value, "pin": ""} if key == "agy" else dict(value)) for key, value in _FIXTURE_GATE["standard"].items()
}

HttpFn = Callable[[str, str, dict[str, str], bytes | None, float], tuple[int, dict[str, Any]]]


class ProviderUnavailable(ValueError):
    """A provider id is known but has no local credential on this machine."""


def fixture_enabled() -> bool:
    raw = (os.environ.get("SHAREDAGENTS_SUBSCRIPTION_FIXTURE") or "").strip()
    return raw.lower() in {"1", "true", "yes", "on"}


def cache_path() -> Path:
    override = (os.environ.get("SHAREDAGENTS_SUBSCRIPTION_CACHE") or "").strip()
    return Path(override).expanduser() if override else (usage_root() / _CACHE_NAME)


_PACE_HISTORY_NAME = "pace_history.json"
#: Windows the Usage history charts offer, newest first in time order.
PACE_HISTORY_PERIODS = {"6h": 6 * 3600, "24h": 86400, "7d": 7 * 86400, "30d": 30 * 86400}
PACE_HISTORY_DEFAULT_PERIOD = "24h"
#: Most points one chart window returns; denser windows keep the last point
#: of each equal time slice.
PACE_HISTORY_MAX_POINTS = 240
#: Storage: the newest day at the sampler's 15-minute slots, one point per
#: hour before that, nothing older than the longest window. The hard cap only
#: guards a runaway writer.
_PACE_FULL_RES_S = 86400
_PACE_THIN_BUCKET_S = 3600
#: The sampler records one point per 15-minute slot, and the newest day is
#: stored at that grid: a later reading in the same slot replaces the earlier.
PACE_SAMPLE_S = 15 * 60
_PACE_SAMPLER_LOCK_NAME = "pace_sampler.lock"
_pace_sampler_lock = threading.Lock()
_pace_sampler_started = False
_pace_history_lock = threading.Lock()
_PACE_RETENTION_S = max(PACE_HISTORY_PERIODS.values())
_PACE_HISTORY_HARD_CAP = 5000
_FIXTURE_PACE_HISTORY: list[dict[str, Any]] = []


def _reset_pace_after_fork() -> None:
    # Threads do not survive fork, and a vanished thread may have held a lock.
    global _pace_sampler_lock, _pace_sampler_started, _pace_history_lock
    _pace_sampler_lock = threading.Lock()
    _pace_history_lock = threading.Lock()
    _pace_sampler_started = False


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_pace_after_fork)


def _operator_home() -> Path:
    # HOME can be changed by tests; an explicit path to the account's real
    # tree must still be protected. Windows has no pwd module.
    try:
        import pwd

        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except (ImportError, KeyError):
        return Path.home()


_LIVE_SHAREDAGENTS_ROOT = _operator_home() / ".sharedagents"


def pace_history_path() -> Path:
    override = (os.environ.get("SHAREDAGENTS_PACE_HISTORY_PATH") or "").strip()
    return Path(override).expanduser() if override else (usage_root() / _PACE_HISTORY_NAME)


def _test_run_on_live_file(path: Path) -> bool:
    """True while pytest runs and ``path`` is inside the operator's ~/.sharedagents.

    Tests that did not redirect ``SHAREDAGENTS_HOME`` appended their synthetic
    snapshots (stamped 2026-09-30, Claude at 720%) to the live pace history,
    and the Usage chart drew them as spikes. A test run never writes there.
    """
    # PYTEST_CURRENT_TEST is absent during collection. The loaded runner also
    # protects module-level calls; the environment marker covers child processes.
    if not os.environ.get("PYTEST_CURRENT_TEST") and "pytest" not in sys.modules:
        return False
    try:
        return path.resolve().is_relative_to(_LIVE_SHAREDAGENTS_ROOT.resolve())
    except (OSError, RuntimeError):
        return True


def _drop_backdated(history: list[Any]) -> list[dict[str, Any]]:
    """Keep points in time order: one stamped before an earlier point is a stray."""
    kept: list[dict[str, Any]] = []
    newest: float | None = None
    for entry in history:
        if not isinstance(entry, dict):
            continue
        stamp = _parse_iso_epoch(entry.get("timestamp"))
        if stamp is None or (newest is not None and stamp < newest):
            continue
        newest = stamp
        kept.append(entry)
    return kept


def get_pace_history() -> dict[str, Any]:
    """Read saved pace and API balance history."""
    if fixture_enabled() and not os.environ.get("SHAREDAGENTS_PACE_HISTORY_PATH"):
        return {"history": _drop_backdated(_FIXTURE_PACE_HISTORY)}
    path = pace_history_path()
    try:
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            data = json.loads(text)
            if isinstance(data, dict) and isinstance(data.get("history"), list):
                return dict(data, history=_drop_backdated(data["history"]))
            if isinstance(data, list):
                return {"history": _drop_backdated(data)}
    except Exception:
        _LOG.debug("failed to read pace history", exc_info=True)
    return {"history": []}


@contextmanager
def _pace_history_writer(path: Path, *, in_memory: bool) -> Iterator[None]:
    # Every writer (web, gateway and routing) must lock the entire read/modify/
    # replace. This is separate from the sampler election: snapshot() writes
    # history while holding that election lock.
    with _pace_history_lock:
        if in_memory:
            yield
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path.with_name(path.name + ".lock"), "a+b") as handle, oscompat.file_lock(handle):
                yield


def record_pace_history(snapshot: dict[str, Any]) -> None:
    """Save a snapshot point of subscription pace and API balance history."""
    if not isinstance(snapshot, dict):
        return
    stamp = str(snapshot.get("updated_at") or _iso_now())
    stamp_epoch = _parse_iso_epoch(stamp)
    if stamp_epoch is None:
        return
    providers = snapshot.get("providers") or []
    if not providers:
        return

    sub_entries: list[dict[str, Any]] = []
    api_entries: list[dict[str, Any]] = []

    for p in providers:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("id") or "")
        name = str(p.get("name") or pid)
        kind = str(p.get("kind") or ("subscription" if pid in PROVIDER_IDS else "balance"))

        if kind == "subscription":
            pace_used = p.get("pace_used_percent")
            sub_entries.append(
                {
                    "id": pid,
                    "name": name,
                    "pace_used_percent": pace_used,
                    "pace_status": p.get("pace_status"),
                    "pace_label": p.get("pace_label"),
                    "pace_limit_hit": bool(p.get("pace_limit_hit")),
                }
            )
        else:
            windows = p.get("windows") or []
            rem = None
            used = None
            for w in windows:
                if isinstance(w, dict):
                    if w.get("remaining") is not None and rem is None:
                        try:
                            rem = float(w["remaining"])
                        except (TypeError, ValueError):
                            pass
                    if w.get("used") is not None and used is None:
                        try:
                            used = float(w["used"])
                        except (TypeError, ValueError):
                            pass
            api_entries.append(
                {
                    "id": pid,
                    "name": name,
                    "remaining": rem,
                    "used": used,
                    "plan": p.get("plan"),
                }
            )

    entry = {
        "timestamp": stamp,
        "subscriptions": sub_entries,
        "api": api_entries,
    }

    in_memory = fixture_enabled() and not os.environ.get("SHAREDAGENTS_PACE_HISTORY_PATH")
    path = pace_history_path()
    if not in_memory and _test_run_on_live_file(path):
        _LOG.debug("test run: not writing pace history to %s", path)
        return
    try:
        with _pace_history_writer(path, in_memory=in_memory):
            before = list(get_pace_history().get("history", []))
            current = list(before)
            last_epoch = _parse_iso_epoch(current[-1].get("timestamp")) if current else None
            if last_epoch == stamp_epoch:
                current[-1] = entry
            elif last_epoch is not None and stamp_epoch < last_epoch:
                # An older snapshot arriving late (or a synthetic one) would draw
                # the chart backwards in time.
                return
            else:
                current.append(entry)
            compacted = _compact_history(current)[-_PACE_HISTORY_HARD_CAP:]
            if compacted == before:
                return  # a cached snapshot read again: nothing new to write
            current = compacted

            if in_memory:
                _FIXTURE_PACE_HISTORY[:] = current
                return

            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps({"history": current}, indent=2) + "\n", encoding="utf-8")
            tmp_path.replace(path)
    except Exception:
        _LOG.debug("failed to write pace history", exc_info=True)


def _last_per_slice(rows: list[dict[str, Any]], *, start: float, width: float) -> list[dict[str, Any]]:
    """Keep the last point of each ``width``-second slice counted from ``start``."""
    kept: list[dict[str, Any]] = []
    last_slice = None
    for entry in rows:
        stamp = _parse_iso_epoch(entry.get("timestamp"))
        if stamp is None:
            continue
        index = int((stamp - start) // width)
        if kept and index == last_slice:
            kept[-1] = entry
        else:
            kept.append(entry)
        last_slice = index
    return kept


def _compact_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retention for stored history (time-ordered input): the newest day as
    the last point of each 15-minute slot, one point per clock hour before
    that, nothing older than the longest chart window."""
    if not history:
        return []
    newest = _parse_iso_epoch(history[-1].get("timestamp"))
    if newest is None:
        return list(history)
    full_from = newest - _PACE_FULL_RES_S
    older = [
        entry
        for entry in history
        if newest - _PACE_RETENTION_S <= (_parse_iso_epoch(entry.get("timestamp")) or 0) < full_from
    ]
    recent = [entry for entry in history if (_parse_iso_epoch(entry.get("timestamp")) or 0) >= full_from]
    return _last_per_slice(older, start=0, width=_PACE_THIN_BUCKET_S) + _last_per_slice(
        recent, start=0, width=PACE_SAMPLE_S
    )


def sample_pace_once(*, now: float | None = None) -> bool:
    """Refresh every vendor and record this 15-minute slot's pace point,
    unless the slot already has one. True when this call recorded.

    Each web worker runs a sampler; the file lock lets one refresh per slot
    and the other finds the slot filled. Failures are swallowed: the next
    slot tries again.
    """
    if fixture_enabled():
        return False
    now = _pace_now() if now is None else now
    slot = int(now // PACE_SAMPLE_S)

    def slot_filled() -> bool:
        history = get_pace_history().get("history", [])
        last = _parse_iso_epoch(history[-1].get("timestamp")) if history else None
        return last is not None and int(last // PACE_SAMPLE_S) >= slot

    lock_path = pace_history_path().with_name(_PACE_SAMPLER_LOCK_NAME)
    if _test_run_on_live_file(lock_path):
        return False
    try:
        if slot_filled():
            return False
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a+b") as handle:
            if not oscompat.try_file_lock(handle):
                return False
            try:
                if slot_filled():
                    return False
                snapshot(refresh=True)
                return slot_filled()  # recording can fail silently (e.g. disk full)
            finally:
                oscompat.file_unlock(handle)
    except Exception:
        _LOG.debug("pace sample failed", exc_info=True)
        return False


def _seconds_to_next_slot(now: float) -> float:
    return PACE_SAMPLE_S - (now % PACE_SAMPLE_S)


def ensure_pace_sampler() -> bool:
    """Start the 15-minute pace sampler once per process. True if running.

    Without it a point is only saved when the Usage page, a routing decision
    or a card refresh reads usage, which left hours without points. Off under
    pytest and with SHAREDAGENTS_PACE_SAMPLER=0.
    """
    global _pace_sampler_started
    if (
        fixture_enabled()
        or "pytest" in sys.modules
        or os.environ.get("PYTEST_CURRENT_TEST")
        or os.environ.get("SHAREDAGENTS_PACE_SAMPLER", "").strip().lower() in {"0", "false", "no", "off"}
    ):
        return False
    with _pace_sampler_lock:
        if _pace_sampler_started:
            return True

        def _loop() -> None:
            # A restart mid-slot fills that slot. Protect the first sample too:
            # a transient failure must not leave the worker without a sampler.
            while True:
                try:
                    sample_pace_once()
                except Exception:  # noqa: BLE001 — a sample failure is never fatal
                    _LOG.debug("pace sample failed", exc_info=True)
                # Always sleep, including after a failed write/refresh. Recompute
                # from wall time after resume/clock steps; never replay missed slots.
                time.sleep(_seconds_to_next_slot(_pace_now()) + 5)

        threading.Thread(target=_loop, name="pace-sampler", daemon=True).start()
        _pace_sampler_started = True
        return True


def pace_history_window(period: str | None = None) -> dict[str, Any]:
    """History for one chart window, at most ``PACE_HISTORY_MAX_POINTS`` points.

    The window ends at the newest recorded point, not the wall clock, so a
    page opened after the recorder was idle still shows its latest data (the
    axis carries the real times). An unknown period falls back to the default.
    """
    key = period if period in PACE_HISTORY_PERIODS else PACE_HISTORY_DEFAULT_PERIOD
    span = PACE_HISTORY_PERIODS[key]
    out: dict[str, Any] = {"period": key, "periods": list(PACE_HISTORY_PERIODS), "history": []}
    history = get_pace_history().get("history", [])
    newest = _parse_iso_epoch(history[-1].get("timestamp")) if history else None
    if newest is None:
        return out
    start = newest - span
    rows = [entry for entry in history if (_parse_iso_epoch(entry.get("timestamp")) or 0) >= start]
    if len(rows) > PACE_HISTORY_MAX_POINTS:
        # MAX - 1 slices of the span, so the newest point closes the last one.
        rows = _last_per_slice(rows, start=start, width=span / (PACE_HISTORY_MAX_POINTS - 1))
    out.update({"from": _epoch_iso(start), "to": _epoch_iso(newest), "history": rows})
    return out


def _pace_now() -> float:
    return time.time()


def _parse_iso_epoch(value: Any) -> float | None:
    if value is None or value == "":
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError, OSError):
        return None


def _round_dp(value: float, places: int) -> float:
    quant = Decimal("1").scaleb(-places)
    return float(Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP))


def _fixture_gate_entry(provider_id: str) -> dict[str, Any]:
    table = _FIXTURE_GATE.get(fixture_scenario_name()) or _FIXTURE_GATE["standard"]
    entry = table.get(provider_id) or {}
    return entry if isinstance(entry, dict) else {}


def is_included_allowance_meter(window: dict[str, Any]) -> bool:
    """True for included plan money that expires at a vendor-reported reset."""
    unit = str(window.get("unit") or "").lower()
    if unit != "usd":
        return False
    if window.get("remaining_percent") is None:
        return False
    return bool(window.get("resets_at"))


def pace_horizon(window: dict[str, Any]) -> str:
    """``short`` (5-hour), ``medium`` (weekly), ``long`` (monthly / included spend)."""
    kind = window_kind(window)
    seconds = _as_float(window.get("window_seconds"))
    if kind == "short":
        return "short"
    if kind == "long":
        if seconds is not None and seconds > 10 * 86_400.0:
            return "long"
        return "medium"
    return "long"


def horizon_period_hours(window: dict[str, Any], horizon: str | None) -> float:
    seconds = _as_float(window.get("window_seconds"))
    if seconds is not None and seconds > 0:
        return seconds / 3600.0
    return float(HORIZON_PERIOD_HOURS.get(horizon or "", 168.0))


def normalized_used_percent(remaining_percent: float, hours_left: float, period_hours: float) -> tuple[float, float]:
    """``(pace %, elapsed %)`` for one limit: used % over the share of time passed.

    The pace is the share of the limit this rate of use reaches by the reset:
    100% = on track to use it exactly by then, under 100% = part of it expires
    unused, over 100% = it runs out before the reset. The elapsed share is
    floored at ``PACE_MIN_ELAPSED`` so a burst right after a reset does not read
    as a trend; a reset further out than one period reads as a fresh period.
    """
    pace, elapsed = _unrounded_pace(remaining_percent, hours_left, period_hours)
    return _round_dp(pace, 1), _round_dp(elapsed * 100.0, 1)


def _unrounded_pace(remaining_percent: float, hours_left: float, period_hours: float) -> tuple[float, float]:
    """``(pace %, elapsed share)`` before rounding."""
    used = max(0.0, min(100.0, 100.0 - float(remaining_percent)))
    elapsed = 1.0 - max(0.0, min(1.0, float(hours_left) / float(period_hours)))
    return used / max(elapsed, PACE_MIN_ELAPSED), elapsed


def pace_scale_max(cards: list[Any]) -> float:
    del cards
    return PACE_SCALE_MAX


def format_pace_percent(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{int(_round_dp(float(value), 0))}%"


def format_runway(hours: float | None) -> str:
    if hours is None:
        return "—"
    if hours <= 0:
        return "0h"
    minutes = int(hours * 60.0)
    if minutes < 1:
        return "<1m"
    if minutes < 60:
        return f"{minutes}m"
    if minutes < 2880:
        hrs, mins = divmod(minutes, 60)
        return f"{hrs}h {mins}m" if mins else f"{hrs}h"
    days, rem = divmod(minutes, 1440)
    hrs = rem // 60
    return f"{days}d {hrs}h" if hrs else f"{days}d"


def pace_eligibility(
    card: dict[str, Any],
    window: dict[str, Any],
    lane: dict[str, Any],
    *,
    pin: str,
) -> tuple[str | None, str | None]:
    card_id = str(card.get("id") or "")
    if card_id not in PROVIDER_IDS:
        return "not_applicable", "balance"
    if window_kind(window) == "spend" and not is_included_allowance_meter(window):
        return "not_applicable", "spend_pool"
    if not lane.get("reachable") and card_id not in _MODEL_RAIL_PROVIDERS:
        return "not_spendable", "unreachable"
    families = [family for family in (lane.get("model_families") or []) if family in MODEL_FAMILIES]
    if not families:
        return None, None
    if not pin:
        return "unknown", "unpinned"
    if not lane.get("spendable_now"):
        return "not_spendable", "pin_locked"
    return None, None


#: Subscriptions with no vendor CLI: the plan is spent by a native agent whose
#: own config pins one of the plan's models (``_native_plan_engine``). Without
#: that agent no lane is "reachable", but the allowance is still a timed quota,
#: so pace is measured either way.
_MODEL_RAIL_PROVIDERS = frozenset({"mimo"})


def _window_pace_result(
    *,
    status: str,
    reason: str | None,
    horizon: str | None,
    period_hours: float | None,
    hours_out: float | None,
    used: float | None = None,
    elapsed: float | None = None,
) -> dict[str, Any]:
    return {
        "pace_used_percent": used,
        "pace_elapsed_percent": elapsed,
        "pace_period_hours": period_hours,
        "pace_hours_to_reset": hours_out,
        "pace_status": status,
        "pace_reason": reason,
        "pace_horizon": horizon,
    }


def window_pace(
    card: dict[str, Any],
    window: dict[str, Any],
    lane: dict[str, Any],
    *,
    pin: str,
    now: float,
) -> dict[str, Any]:
    """Pace of one limit (``normalized_used_percent``), or why it has none.

    Precedence: not applicable / not spendable / unpinned, then stale data, a
    missing remaining %, a reset that already passed, a used-up limit (even
    without a reset time), and last a missing reset time.
    """
    reset_epoch = _parse_iso_epoch(window.get("resets_at"))
    hours = (reset_epoch - now) / 3600.0 if reset_epoch is not None and reset_epoch > now else None
    horizon = pace_horizon(window)
    period = horizon_period_hours(window, horizon)
    base: dict[str, Any] = {
        "horizon": horizon,
        "period_hours": _round_dp(period, 4) if period else None,
        "hours_out": _round_dp(hours, 2) if hours is not None else None,
    }
    remaining_raw = window.get("remaining_percent")
    remaining = float(remaining_raw) if remaining_raw is not None else None
    status, reason = pace_eligibility(card, window, lane, pin=pin)
    if status is not None:
        return _window_pace_result(status=status, reason=reason, **base)
    updated = _parse_iso_epoch(card.get("updated_at"))
    if updated is None or (now - updated) > PACE_FRESHNESS_S:
        return _window_pace_result(status="unknown", reason="stale_data", **base)
    if card.get("id") == "devin" and card.get("reading_source") == "cli_cache":
        return _window_pace_result(status="unknown", reason="cached_data", **base)
    if remaining is None:
        return _window_pace_result(status="unknown", reason="no_remaining", **base)
    if reset_epoch is not None and reset_epoch <= now:
        return _window_pace_result(status="unknown", reason="stale_reset", **{**base, "hours_out": None})
    if hours is None:
        if remaining == 0:
            return _window_pace_result(status="exhausted", reason=None, **base)
        return _window_pace_result(status="unknown", reason="no_reset", **base)
    used, elapsed = normalized_used_percent(remaining, hours, period)
    return _window_pace_result(
        status="exhausted" if remaining == 0 else "ok",
        reason=None,
        used=used,
        elapsed=elapsed,
        **base,
    )


def _is_allowance_share(window: dict[str, Any]) -> bool:
    """Cursor ``Auto`` / ``API``: vendor percentages INSIDE the included allowance.

    They carry no reset or window length of their own and are shares of the
    plan's included spend, so the plan's pace comes from that one meter rather
    than from three overlapping readings of the same money.
    """
    unit = str(window.get("unit") or "").lower()
    return unit not in {"usd", "credits"} and not window.get("resets_at") and window.get("window_seconds") is None


def _dominant_pace_reason(reasons: list[str | None]) -> str | None:
    present = {reason for reason in reasons if reason}
    for reason in _DOMINANT_PACE_REASONS:
        if reason in present:
            return reason
    return next((reason for reason in reasons if reason), None)


def _partial_labels(windows: list[dict[str, Any]]) -> str:
    names = [str(window.get("label") or window.get("id") or "") for window in windows]
    extra = len(names) - 2
    if extra > 0:
        return f"{names[0]}, {names[1]} +{extra}"
    return ", ".join(names)


def _card_pace_fields(
    status: str,
    reason: str | None,
    *,
    used: float | None = None,
    basis: int | None = None,
    partial: bool = False,
    horizon: str | None = None,
    resets_at: str | None = None,
    limit_hit: bool = False,
    label: str = "—",
    note: str | None = None,
    pools: list[dict[str, Any]] | None = None,
    lanes: list[dict[str, Any]] | None = None,
    remaining: float | None = None,
    engine_hit: bool = False,
    engine_pool: str | None = None,
    engine_resets_at: str | None = None,
) -> dict[str, Any]:
    return {
        "pace_used_percent": used,
        "pace_status": status,
        "pace_reason": reason,
        "pace_basis": basis,
        "pace_partial": partial,
        "pace_horizon": horizon,
        "pace_resets_at": resets_at,
        "pace_limit_hit": bool(limit_hit),
        "pace_use_next": False,
        "pace_label": label,
        "pace_note": note,
        # Every limit with its own pace and whether it counted (``_pace_lanes``),
        # and the same rolled up per named pool (``_pace_pools``).
        "pace_lanes": lanes or [],
        "pace_pools": pools or [],
        # Least left on any counted limit of a pool that still has room.
        "pace_remaining_percent": remaining,
        # The agent's own model cannot run: a used-up pool belongs to its pinned
        # model (``model_scope`` / ``model_slug``), or every pool is used up. Routing reads this,
        # not the plan's pace: the plan can be healthy while its agent is not.
        "pace_engine_limit_hit": bool(engine_hit),
        "pace_engine_pool": engine_pool,
        "pace_engine_resets_at": engine_resets_at,
    }


#: Why a limit is not in the plan's pace, as shown on the card.
PACE_EXCLUDED_TEXT = {
    "pin_locked": "the pinned model cannot spend it",
    "unreachable": "no agent can reach it",
    "spend_pool": "pay-as-you-go spend, not a timed limit",
    "allowance_share": "a share of the included allowance",
    "balance": "prepaid balance",
    **PACE_REASON_TEXT,
}


def _used_text(window: dict[str, Any]) -> str:
    return format_pace_percent(100.0 - float(window.get("remaining_percent") or 0.0))


def _limit_line(window: dict[str, Any], paced: dict[str, Any]) -> str:
    """One tooltip line showing how a limit's pace was derived."""
    label = str(window.get("label") or window.get("id") or "Limit")
    pace = paced.get("pace_used_percent")
    if pace is None:
        return f"{label} {_used_text(window)} used · no reset time reported"
    elapsed = paced.get("pace_elapsed_percent")
    passed = f"{format_pace_percent(elapsed)} of the period passed"
    floor = PACE_MIN_ELAPSED * 100.0
    if elapsed is not None and elapsed < floor:
        passed += f" (counted as {format_pace_percent(floor)})"
    return f"{label} {_used_text(window)} used · {passed} → {format_pace_percent(pace)}"


def _hit_line(window: dict[str, Any], paced: dict[str, Any]) -> str:
    hours = paced.get("pace_hours_to_reset")
    when = f"resets in {format_runway(hours)}" if hours is not None else "reset time unknown"
    return f"{window.get('label') or window.get('id')} limit hit · {when}"


def _period_hours(paced: dict[str, Any]) -> float:
    return _as_float(paced.get("pace_period_hours")) or 0.0


_PaceRow = tuple[int, dict[str, Any], dict[str, Any]]


def _pool_name(window: dict[str, Any]) -> str:
    return str(window.get("pool") or "").strip()


def _limit_hit(row: _PaceRow, *, now: float) -> bool:
    remaining = _as_float(row[1].get("remaining_percent"))
    reset = _parse_iso_epoch(row[1].get("resets_at"))
    explicit = bool(row[1].get("limit_hit")) or str(row[1].get("status") or "").lower() in {
        "exhausted",
        "limit hit",
        "individual quota reached",
    }
    return (explicit or (remaining is not None and remaining < LIMIT_HIT_REMAINING)) and (reset is None or reset > now)


def _pinned_pool_hits(
    windows: list[dict[str, Any]],
    *,
    pin: str,
    now: float,
) -> list[dict[str, Any]]:
    """Mandatory model limits apply even when access/pace says pin_locked.

    Match the actual pin, never every exhausted reserve pool on the card. A
    family-wide Gemini pool having headroom cannot override an Argon limit.
    This check is independent of pace measurement, freshness and gate access.
    """
    pin = str(pin or "").lower()
    if not pin:
        return []
    hits = []
    for window in windows:
        scope = str(window.get("model_scope") or "").lower()
        slug = str(window.get("model_slug") or "").lower()
        exact = {slug, str(window.get("limit_name") or "").lower()} - {""}
        matches = (bool(scope) and scope in pin) or (bool(slug) and pin in exact)
        if matches and _limit_hit((0, window, {}), now=now):
            hits.append(window)
    return hits


def _reset_epoch(row: _PaceRow) -> float:
    reset = _parse_iso_epoch(row[1].get("resets_at"))
    return reset if reset is not None else math.inf


def _fastest_row(rows: list[_PaceRow]) -> _PaceRow:
    """Highest pace; on a tie the longer limit, then the one that resets first."""
    return max(
        rows,
        key=lambda row: (
            float(row[2].get("pace_used_percent") or 0.0),
            _period_hours(row[2]),
            -_reset_epoch(row),
            -row[0],
        ),
    )


def _pool_groups(measured: list[_PaceRow]) -> dict[str, list[_PaceRow]]:
    """Counted limits by pool; limits without a pool name form one pool."""
    groups: dict[str, list[_PaceRow]] = {}
    for row in measured:
        groups.setdefault(_pool_name(row[1]), []).append(row)
    return groups


def _used_up(rows: list[_PaceRow], *, now: float) -> bool:
    """A pool is used up while any of its limits is hit."""
    return any(_limit_hit(row, now=now) for row in rows)


def _reopening_row(rows: list[_PaceRow], *, now: float) -> _PaceRow:
    """The hit limit of a used-up pool that resets last: the pool opens then."""
    return max((row for row in rows if _limit_hit(row, now=now)), key=lambda row: (_reset_epoch(row), -row[0]))


def _raw_lane_pace(window: dict[str, Any], paced: dict[str, Any], *, now: float) -> tuple[float | None, float | None]:
    """``(pace %, elapsed %)`` of a limit whatever its eligibility, else Nones.

    A limit left out of the plan (another model's pool, say) still shows how
    fast it is being used, so nothing on the card is silently dropped. Money
    that is not a timed limit (pay-as-you-go spend, a balance, a share of the
    allowance) has no pace.
    """
    if paced.get("pace_used_percent") is not None:
        return paced["pace_used_percent"], paced.get("pace_elapsed_percent")
    if paced.get("pace_status") == "not_applicable":
        return None, None
    remaining = _as_float(window.get("remaining_percent"))
    reset = _parse_iso_epoch(window.get("resets_at"))
    if remaining is None or reset is None or reset <= now:
        return None, None
    period = horizon_period_hours(window, paced.get("pace_horizon") or pace_horizon(window))
    return normalized_used_percent(remaining, (reset - now) / 3600.0, period)


def _pace_lanes(
    windows: list[dict[str, Any]],
    paces: list[dict[str, Any]],
    measured: list[_PaceRow],
    eligible: list[_PaceRow],
    basis: _PaceRow | None,
    *,
    now: float,
) -> list[dict[str, Any]]:
    """Every limit of the plan with its own pace, whether it counted, and why not.

    All lanes of a used-up pool read "Limit hit" (``pace_label``), while
    ``limit_hit`` records each limit's own state. Used-up pools do not set the
    plan's pace, since the plan keeps serving from its other pools.
    """
    counted = {row[0] for row in measured}
    eligible_indices = {row[0] for row in eligible}
    row_groups = _pool_groups(eligible)
    hit_pools = {name for name, rows in row_groups.items() if _used_up(rows, now=now)}
    # Named pools excluded by the pin still show their availability. An
    # unnamed model-specific reserve must not use up the agent's general pool.
    excluded_groups = _pool_groups(
        [
            (index, window, paced)
            for index, (window, paced) in enumerate(zip(windows, paces))
            if index not in eligible_indices and _pool_name(window) and paced.get("pace_status") != "not_applicable"
        ]
    )
    excluded_hit = {name for name, rows in excluded_groups.items() if _used_up(rows, now=now)}
    lanes = []
    for index, (window, paced) in enumerate(zip(windows, paces)):
        pace, elapsed = _raw_lane_pace(window, paced, now=now)
        is_counted = index in counted
        hit = paced.get("pace_status") != "not_applicable" and _limit_hit((index, window, paced), now=now)
        pool_hit = hit or _pool_name(window) in (hit_pools if index in eligible_indices else excluded_hit)
        window["pace_limit_hit"] = pool_hit
        window["pace_lane_label"] = "Limit hit" if pool_hit else None
        lanes.append(
            {
                "id": window.get("id"),
                "label": window.get("label"),
                "pool": _pool_name(window) or None,
                "horizon": paced.get("pace_horizon"),
                "remaining_percent": window.get("remaining_percent"),
                "elapsed_percent": elapsed,
                "pace_used_percent": pace,
                "resets_at": window.get("resets_at"),
                "hours_to_reset": paced.get("pace_hours_to_reset"),
                "limit_hit": hit,
                "pace_label": "Limit hit" if pool_hit else format_pace_percent(pace),
                "counted": is_counted,
                "pool_used_up": pool_hit,
                "binding": basis is not None and index == basis[0],
                "status": paced.get("pace_status"),
                "reason": None if is_counted else paced.get("pace_reason"),
            }
        )
    return lanes


def _pace_pools(lanes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per named pool: its binding limit's pace, and why if not counted.

    A pool's pace is its own binding limit (hit first, else the highest pace)
    among the limits that counted; a pool with none counted shows its
    fastest-used limit with ``counted`` false and the reason.
    """
    names = list(dict.fromkeys(lane["pool"] for lane in lanes if lane.get("pool")))
    pools = []
    for name in names:
        rows = [lane for lane in lanes if lane.get("pool") == name]
        counted = [lane for lane in rows if lane["counted"]]
        # A hit can be known without a measured projection (missing reset or
        # stale data). It still determines the pool's availability.
        scope = [lane for lane in rows if lane["counted"] or lane["pool_used_up"]] or rows
        top = max(
            scope,
            key=lambda lane: (
                bool(lane["limit_hit"]),
                float(lane["pace_used_percent"] if lane["pace_used_percent"] is not None else -1.0),
            ),
        )
        remaining = [value for lane in scope if (value := _as_float(lane.get("remaining_percent"))) is not None]
        pools.append(
            {
                "label": name,
                "pace_used_percent": top["pace_used_percent"],
                "binding_lane": top["id"],
                "remaining_percent": min(remaining) if remaining else None,
                "resets_at": top["resets_at"],
                "limit_hit": any(lane["limit_hit"] for lane in scope),
                "binding": any(lane["binding"] for lane in rows),
                "counted": bool(counted),
                "reason": None if counted else next((lane["reason"] for lane in rows if lane.get("reason")), None),
            }
        )
    return pools


def _excluded_line(lane: dict[str, Any]) -> str:
    reason = str(lane.get("reason") or "")
    why = PACE_EXCLUDED_TEXT.get(reason, reason or "not measured")
    left = format_pace_percent(_as_float(lane.get("remaining_percent")))
    pace = lane.get("pace_used_percent")
    at = f" · pace {format_pace_percent(pace)}" if pace is not None else ""
    return f"{lane.get('label') or lane.get('id')}: {left} left{at} · not counted ({why})"


def card_pace(card: dict[str, Any], *, now: float) -> dict[str, Any]:
    """One pace for the plan: the fastest limit among the pools with room left.

    Every limit — 5-hour, weekly, monthly, every model pool — has its own pace:
    the used % over the share of its period already passed
    (``normalized_used_percent``), i.e. how much of it this rate of use reaches
    by its reset. All of them are listed in ``pace_lanes`` and per pool in
    ``pace_pools``; a limit this agent cannot spend (another model's pool), or
    one that cannot be paced, is listed with ``counted`` false and the reason.

    A limit with under ``LIMIT_HIT_REMAINING`` percent left is hit, and its
    pool is used up until that limit resets. A used-up pool reads "Limit hit"
    on its own lanes but does not set the plan's pace: the plan keeps serving
    from its other pools, so its pace is the highest pace among the pools that
    still have room (limits are never summed or averaged). The plan reads
    "Limit hit" (``exhausted``) only when every pool is used up.

    Whether the plan's AGENT can run is separate (``pace_engine_limit_hit``):
    a used-up pool that belongs to the agent's pinned model (``model_scope`` / ``model_slug``,
    e.g. the Gemini 4 Argon pool under an Argon pin) stops the agent even while
    the plan has room. Routing reads that, not the plan's pace.
    """
    if not card.get("ok"):
        return _card_pace_fields("unknown", "card_error")
    card_id = str(card.get("id") or "")
    if card_id not in PROVIDER_IDS:
        return _card_pace_fields("not_applicable", "balance", note="prepaid API — no timed allowance")
    pin = _engine_default_model(_engine_kind_for(card_id))
    windows = [row for row in (card.get("windows") or []) if isinstance(row, dict)]
    cache: dict[str, Any] = {}
    lanes = [_model_lane(window, card_id, cache) for window in windows]
    allowance = next((window for window in windows if is_included_allowance_meter(window)), None)
    paces = [window_pace(card, window, lane, pin=pin, now=now) for window, lane in zip(windows, lanes)]
    measured: list[_PaceRow] = []
    unknown: list[_PaceRow] = []
    shares: list[dict[str, Any]] = []
    for index, (window, paced) in enumerate(zip(windows, paces)):
        if (
            paced["pace_status"] in {"ok", "exhausted", "unknown"}
            and allowance is not None
            and _is_allowance_share(window)
        ):
            # Cursor Auto / API: percentages inside the included allowance with
            # no reset of their own, so the allowance meter is the limit.
            shares.append(window)
            paced = {
                **paced,
                "pace_status": "not_applicable",
                "pace_reason": "allowance_share",
                "pace_used_percent": None,
                "pace_elapsed_percent": None,
            }
            paces[index] = paced
        for key in _PACE_WINDOW_KEYS:
            window[key] = paced[key]
        status = paced["pace_status"]
        if status in {"ok", "exhausted"}:
            measured.append((index, window, paced))
        elif status == "unknown":
            unknown.append((index, window, paced))
    # Availability is independent of whether a projection can be calculated.
    # Keep stale/no-reset limits in their pools: a known hit still blocks until
    # reset, and an unpaced pool with headroom is not proof of plan exhaustion.
    eligible = [
        (index, window, paced)
        for index, (window, paced, lane) in enumerate(zip(windows, paces, lanes))
        if paced["pace_status"] != "not_applicable" and pace_eligibility(card, window, lane, pin=pin)[0] is None
    ]
    groups = _pool_groups(eligible)
    used_up = frozenset(name for name, rows in groups.items() if _used_up(rows, now=now))
    open_rows = [row for row in measured if _pool_name(row[1]) not in used_up]
    hit = [row for row in eligible if _limit_hit(row, now=now)]
    plan_hit = bool(groups) and len(used_up) == len(groups)
    if plan_hit:
        # Nothing left anywhere: the plan serves again when its first pool opens.
        basis = min(
            (_reopening_row(rows, now=now) for rows in groups.values()),
            key=lambda row: (_reset_epoch(row), row[0]),
        )
    else:
        basis = _fastest_row(open_rows) if open_rows else None
    # Eligibility already established that these scopes/exact models match
    # the pin. Both representations tie a pool to the agent's model.
    scoped = {_pool_name(row[1]) for row in eligible if row[1].get("model_scope") or row[1].get("model_slug")}
    blocked = sorted(name for name in scoped if name in used_up)
    pinned_hits = _pinned_pool_hits(windows, pin=pin, now=now)
    blocked = sorted(
        set(blocked) | {_pool_name(window) or str(window.get("label") or window.get("id")) for window in pinned_hits}
    )
    engine_hit = plan_hit or bool(blocked)
    engine_resets_at = None
    reopening = [_reopening_row(groups[name], now=now) for name in blocked if name in groups]
    reopening.extend((0, window, {}) for window in pinned_hits)
    if plan_hit and basis is not None:
        reopening.append(basis)
    if reopening:
        engine_resets_at = max(reopening, key=_reset_epoch)[1].get("resets_at")
    pace_lanes = _pace_lanes(windows, paces, measured, eligible, basis, now=now)
    pools = _pace_pools(pace_lanes)
    if basis is None:
        fields = {
            "lanes": pace_lanes,
            "pools": pools,
            "engine_hit": engine_hit,
            "engine_pool": ", ".join(blocked) or None,
            "engine_resets_at": engine_resets_at,
        }
        if not unknown:
            return _card_pace_fields(
                "not_spendable",
                "no_spendable_window",
                note="no limit this agent can spend",
                **fields,
            )
        reason = _dominant_pace_reason([row[2].get("pace_reason") for row in unknown])
        text = PACE_REASON_TEXT.get(reason or "", reason or "unknown")
        return _card_pace_fields("unknown", reason, note=f"pace unknown: {text}", **fields)
    index, window, paced = basis
    used = paced.get("pace_used_percent")
    remaining = min(
        (
            value
            for row in eligible
            if (plan_hit or _pool_name(row[1]) not in used_up)
            if (value := _as_float(row[1].get("remaining_percent"))) is not None
        ),
        default=None,
    )
    lines = []
    if plan_hit:
        lines.append(
            f"Limit hit: every pool is used up · {_hit_line(window, paced)} — the plan cannot serve until then"
        )
        lines.extend(_hit_line(row[1], row[2]) for row in hit if row is not basis)
    else:
        lines.append(f"Sets the pace: {_limit_line(window, paced)}")
        for row in sorted(
            (row for row in open_rows if row is not basis),
            key=lambda row: (-float(row[2].get("pace_used_percent") or 0.0), row[0]),
        ):
            lines.append(_limit_line(row[1], row[2]))
        for row in hit:
            if _pool_name(row[1]) in blocked:
                why = "the agent's pinned model uses this pool and cannot run until then"
            else:
                why = "not in the pace while used up; the plan still has room elsewhere"
            lines.append(f"{_hit_line(row[1], row[2])} — {why}")
    for row in measured:
        if _pool_name(row[1]) in used_up and not _limit_hit(row, now=now):
            lines.append(f"{_limit_line(row[1], row[2])} · its pool is used up")
    if shares and allowance is not None:
        lines.append(
            f"{_partial_labels(shares)}: shares of {allowance.get('label') or 'the plan'}, not counted separately"
        )
    for lane in pace_lanes:
        if not lane["counted"] and lane["reason"] not in {"allowance_share", None}:
            if lane["status"] != "unknown":
                lines.append(_excluded_line(lane))
    partial = bool(unknown)
    reason = None
    if partial:
        reason = _dominant_pace_reason([row[2].get("pace_reason") for row in unknown])
        text = PACE_REASON_TEXT.get(reason or "", reason or "unknown")
        matches = [row[1] for row in unknown if row[2].get("pace_reason") == reason] or [row[1] for row in unknown]
        lines.append(f"not counted: {text} ({_partial_labels(matches)})")
    cached_devin = card_id == "devin" and card.get("reading_source") == "cli_cache"
    return _card_pace_fields(
        "unknown" if cached_devin else ("exhausted" if plan_hit else "ok"),
        paced.get("pace_reason") if cached_devin else reason,
        used=used,
        basis=index,
        partial=partial,
        horizon=paced.get("pace_horizon"),
        resets_at=window.get("resets_at"),
        limit_hit=plan_hit,
        label="—" if cached_devin else ("Limit hit" if plan_hit else format_pace_percent(used)),
        note="\n".join(lines),
        pools=pools,
        lanes=pace_lanes,
        remaining=_round_dp(remaining, 1) if remaining is not None else None,
        engine_hit=engine_hit,
        engine_pool=", ".join(blocked) or None,
        engine_resets_at=engine_resets_at,
    )


def _pace_rankable(card: Any) -> bool:
    """A plan that can be "use next": measured pace, no limit hit, and its agent can run."""
    pace = _as_float(card.get("pace_used_percent")) if isinstance(card, dict) else None
    return (
        isinstance(card, dict)
        and str(card.get("id") or "") in PROVIDER_IDS
        and card.get("pace_status") == "ok"
        and not card.get("pace_limit_hit")
        and not card.get("pace_engine_limit_hit")
        and pace is not None
        and math.isfinite(pace)
        and pace >= 0
    )


def _pace_rank_key(card: dict[str, Any]) -> tuple[float, float]:
    """Lowest pace first; on a tie, the plan whose limit resets sooner."""
    used = _as_float(card.get("pace_used_percent"))
    reset = _parse_iso_epoch(card.get("pace_resets_at"))
    return (
        _round_dp(used, 1) if used is not None else math.inf,
        reset if reset is not None else math.inf,
    )


def _mark_use_next(cards: list[Any]) -> None:
    for card in cards or []:
        if isinstance(card, dict):
            card["pace_use_next"] = False
    eligible = [card for card in cards or [] if _pace_rankable(card)]
    if eligible:
        winner = min(eligible, key=lambda card: (*_pace_rank_key(card), str(card.get("id") or "")))
        winner["pace_use_next"] = True


def _display_sort_key(card: dict[str, Any], index: int) -> tuple[Any, ...]:
    if str(card.get("id") or "") not in PROVIDER_IDS:
        return (1, 0, 0.0, 0.0, index)
    if _pace_rankable(card):
        return (0, 0, *_pace_rank_key(card), index)
    group = {"ok": 1, "exhausted": 1, "unknown": 2, "not_spendable": 3}.get(str(card.get("pace_status") or ""), 4)
    used = _as_float(card.get("pace_used_percent"))
    return (0, group, used if used is not None else math.inf, math.inf, index)


def _sort_snapshot_providers(providers: list[Any]) -> list[Any]:
    indexed = [(index, card) for index, card in enumerate(providers or [])]
    indexed.sort(
        key=lambda item: _display_sort_key(item[1], item[0]) if isinstance(item[1], dict) else (1, 0, 0.0, 0.0, item[0])
    )
    return [card for _, card in indexed]


def _attach_pace_to_card(card: dict[str, Any], *, now: float | None = None) -> dict[str, Any]:
    if now is None:
        now = _parse_iso_epoch(card.get("updated_at")) if fixture_enabled() else None
        now = now if now is not None else _pace_now()
    card.update(card_pace(card, now=now))
    return card


def _attach_pace_to_snapshot(data: dict[str, Any], *, now: float | None = None) -> dict[str, Any]:
    if now is None:
        now = _parse_iso_epoch(data.get("updated_at")) if fixture_enabled() else None
        now = now if now is not None else _pace_now()
    cards = [card for card in (data.get("providers") or []) if isinstance(card, dict)]
    for card in cards:
        _attach_pace_to_card(card, now=now)
    _mark_use_next(cards)
    data["providers"] = _sort_snapshot_providers(data.get("providers") or [])
    data["pace_scale_max"] = pace_scale_max(data.get("providers") or [])
    return data


def _snapshot_with_history(data: dict[str, Any]) -> dict[str, Any]:
    """Pace the snapshot the page is served, save it as a history point and
    attach the history. Only the real snapshot paths call this: working out
    pace never writes anything on its own."""
    data = _attach_pace_to_snapshot(data)
    record_pace_history(data)
    # The default chart window only: the full store is up to 30 days.
    data["pace_history"] = pace_history_window()["history"]
    return data


def fixture_scenario_name() -> str:
    raw = (os.environ.get("SHAREDAGENTS_SUBSCRIPTION_FIXTURE_SCENARIO") or "").strip()
    return "pace-edge" if raw == "pace-edge" else "standard"


def _stamp_plus(stamp: str, *, days: float = 0, hours: float = 0, minutes: float = 0) -> str:
    parsed = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    return (
        (parsed + timedelta(days=days, hours=hours, minutes=minutes))
        .astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def fixture_snapshot(now: str | None = None) -> dict[str, Any]:
    """Deterministic four-provider payload for Docker / headed verify."""
    stamp = now or _iso_now()
    five_hour = _stamp_plus(stamp, hours=5)
    weekly = _stamp_plus(stamp, days=7)
    billing = _stamp_plus(stamp, days=16)
    providers = [
        _provider(
            "codex",
            plan="Plus",
            updated_at=stamp,
            windows=[
                _window("five_hour", "5-hour", 24.0, five_hour),
                _window("seven_day", "Weekly", 11.0, weekly),
            ],
        ),
        _provider(
            "claude",
            plan="Max",
            updated_at=stamp,
            windows=[
                _window("five_hour", "5-hour", 41.0, five_hour),
                _window("seven_day", "Weekly", 18.0, weekly),
            ],
        ),
        _provider(
            "cursor",
            plan="Pro",
            updated_at=stamp,
            windows=[
                _window("plan", "Included spend", 63.0, billing, used=126.0, limit=200.0, unit="USD"),
                _window("auto", "Auto", 55.0, None),
                _window("api", "API", 12.0, None),
            ],
        ),
        _provider(
            "agy",
            plan="Google",
            updated_at=stamp,
            windows=[
                _window("gemini_five_hour", "Gemini · 5-hour", 37.0, five_hour),
                _window("gemini_weekly", "Gemini · Weekly", 22.0, weekly),
                _window("claude_gpt_five_hour", "Claude/GPT · 5-hour", 8.0, five_hour),
                _window("claude_gpt_weekly", "Claude/GPT · Weekly", 5.0, weekly),
            ],
        ),
        _provider(
            "zcode",
            plan="Lite",
            updated_at=stamp,
            windows=[
                _window(
                    "five_hour",
                    "5-hour",
                    12.0,
                    five_hour,
                    used=243.0,
                    limit=2000.0,
                    remaining=1756.0,
                ),
                _window(
                    "seven_day",
                    "Weekly",
                    3.0,
                    weekly,
                    used=321.0,
                    limit=10000.0,
                    remaining=9678.0,
                ),
            ],
        ),
        _provider(
            "openrouter",
            plan="Pay-as-you-go",
            updated_at=stamp,
            windows=[
                _window("credits", "Available", 7.4, None, used=92.6, limit=100.0, remaining=7.4, unit="USD"),
            ],
        ),
        _provider(
            "deepseek",
            plan="API",
            updated_at=stamp,
            windows=[
                _window("balance", "Balance", None, None, remaining=12.15, unit="USD"),
            ],
        ),
        _provider(
            "xai",
            plan="Prepaid",
            updated_at=stamp,
            windows=[
                _window(
                    "credits",
                    "Available",
                    9.6,
                    None,
                    used=1.93,
                    limit=20.11,
                    remaining=18.18,
                    unit="USD",
                )
            ],
        ),
        _provider(
            "cerebras",
            plan="Pay-as-you-go",
            updated_at=stamp,
            windows=[
                _window("credits", "Available", None, None, remaining=14.32, unit="USD"),
            ],
        ),
    ]
    if fixture_scenario_name() == "pace-edge":
        by_id = {str(row.get("id") or ""): row for row in providers}
        by_id["codex"] = _provider(
            "codex",
            plan="Plus",
            updated_at=stamp,
            windows=[
                _window(
                    "five_hour",
                    "Weekly",
                    97.0,
                    _stamp_plus(stamp, minutes=6),
                    window_seconds=604_800.0,
                ),
                _window(
                    "additional_gpt_reserve",
                    "gpt-reserve · Weekly",
                    0.0,
                    _stamp_plus(stamp, days=7),
                    window_seconds=604_800.0,
                    model_slug="gpt-5.6-luna",
                    limit_name="gpt-reserve",
                ),
                _window(
                    "credits",
                    "Credits",
                    40.0,
                    _stamp_plus(stamp, days=30),
                    used=40.0,
                    limit=100.0,
                    remaining=60.0,
                    unit="credits",
                ),
            ],
        )
        by_id["claude"] = _provider(
            "claude",
            plan="Max",
            updated_at=stamp,
            windows=[
                _window("five_hour", "5-hour", 20.0, _stamp_plus(stamp, hours=-1)),
                _window("seven_day", "Weekly", 100.0, _stamp_plus(stamp, hours=24)),
                _window(
                    "weekly_scoped",
                    "Fable",
                    0.0,
                    _stamp_plus(stamp, days=3),
                    window_seconds=604_800.0,
                    model_scope="Fable",
                ),
                _window(
                    "extra_usage",
                    "Extra credits",
                    30.0,
                    None,
                    used=15.0,
                    limit=50.0,
                    unit="credits",
                ),
            ],
        )
        by_id["cursor"] = _provider(
            "cursor",
            plan="Pro",
            updated_at=stamp,
            windows=[
                _window(
                    "plan",
                    "Included spend",
                    80.0,
                    _stamp_plus(stamp, days=10),
                    used=160.0,
                    limit=200.0,
                    unit="USD",
                ),
                _window("auto", "Auto", 50.0, None),
                _window("api", "API", 0.0, None),
                _window(
                    "spend_limit",
                    "Spend limit",
                    10.0,
                    None,
                    used=5.0,
                    limit=50.0,
                    unit="USD",
                ),
            ],
        )
        providers = [by_id[str(row.get("id") or "")] for row in providers if str(row.get("id") or "") in by_id]
    return {
        "updated_at": stamp,
        "source": "fixture",
        "providers": providers,
    }


def empty_provider(provider_id: str, error: str | None = None) -> dict[str, Any]:
    return _provider(provider_id, error=error)


def normalize_percent(value: Any) -> float | None:
    """Clamp a vendor 0–100 percent.

    Codex, Claude, and Cursor all report 0–100, so ``1`` is 1% — reading values
    up to 1 as 0–1 fractions turned a fresh week's 1% into "100% used".
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(min(max(number, 0.0), 100.0), 1)


def parse_codex_payload(payload: dict[str, Any]) -> dict[str, Any]:
    snapshot = _codex_snapshot(payload)
    plan = (
        str(
            snapshot.get("planType") or payload.get("planType") or payload.get("plan_type") or payload.get("plan") or ""
        ).strip()
        or None
    )
    windows: list[dict[str, Any]] = []
    primary = _codex_window_dict(snapshot, payload, "primary")
    secondary = _codex_window_dict(snapshot, payload, "secondary")
    if primary:
        windows.append(
            _window(
                "five_hour",
                _codex_window_label(
                    primary.get("windowDurationMins") or _seconds_to_minutes(primary.get("limit_window_seconds")),
                    "5-hour",
                ),
                normalize_percent(_first_present(primary.get("usedPercent"), primary.get("used_percent"))),
                _epoch_or_iso(primary.get("resetsAt") or primary.get("resets_at") or primary.get("reset_at")),
                window_seconds=_codex_window_seconds(primary),
            )
        )
    if secondary:
        windows.append(
            _window(
                "seven_day",
                _codex_window_label(
                    secondary.get("windowDurationMins") or _seconds_to_minutes(secondary.get("limit_window_seconds")),
                    "Weekly",
                ),
                normalize_percent(_first_present(secondary.get("usedPercent"), secondary.get("used_percent"))),
                _epoch_or_iso(secondary.get("resetsAt") or secondary.get("resets_at") or secondary.get("reset_at")),
                window_seconds=_codex_window_seconds(secondary),
            )
        )
    # ``additional_rate_limits[]`` carries the meters the primary window hides:
    # live a weekly ``gpt-reserve`` meter at 0% used tied to
    # ``gpt-5.6-luna`` — an idle second GPT lane the old parser dropped entirely.
    windows.extend(_codex_additional_windows(payload, snapshot))
    credits = (
        payload.get("credits")
        if isinstance(payload.get("credits"), dict)
        else (snapshot.get("credits") if isinstance(snapshot.get("credits"), dict) else {})
    )
    if credits and (credits.get("has_credits") or credits.get("hasCredits") or credits.get("limit")):
        used = _as_float(credits.get("used"))
        limit = _as_float(credits.get("limit") or credits.get("individualLimit"))
        remaining = _as_float(credits.get("balance") or credits.get("remaining"))
        percent = normalize_percent(credits.get("usedPercent") or credits.get("used_percent"))
        if percent is None and used is not None and limit:
            percent = round(min(100.0, used / limit * 100.0), 1)
        if percent is None and remaining is not None and limit:
            percent = round(min(100.0, max(0.0, (1.0 - remaining / limit) * 100.0)), 1)
        if percent is not None or (used is not None and limit is not None):
            windows.append(
                _window(
                    "credits",
                    "Credits",
                    percent,
                    _epoch_or_iso(credits.get("resetsAt") or credits.get("resets_at")),
                    used=used,
                    limit=limit,
                    remaining=remaining,
                    unit="credits",
                )
            )
    if not windows:
        raise ValueError("Codex usage payload had no rate-limit windows")
    return _provider("codex", plan=plan, windows=windows)


def parse_claude_payload(payload: dict[str, Any]) -> dict[str, Any]:
    plan = str(payload.get("plan") or payload.get("subscriptionType") or "").strip() or None
    windows: list[dict[str, Any]] = []
    rate_limits = payload.get("rate_limits") if isinstance(payload.get("rate_limits"), dict) else {}
    buckets = {
        "five_hour": ("5-hour", payload.get("five_hour") or rate_limits.get("five_hour"), 18_000.0, None),
        "seven_day": ("Weekly", payload.get("seven_day") or rate_limits.get("seven_day"), 604_800.0, None),
        "seven_day_sonnet": ("Weekly · Sonnet", payload.get("seven_day_sonnet"), 604_800.0, "Sonnet"),
        "seven_day_opus": ("Weekly · Opus", payload.get("seven_day_opus"), 604_800.0, "Opus"),
    }
    for window_id, (label, raw, seconds, scope) in buckets.items():
        if not isinstance(raw, dict):
            continue
        percent = normalize_percent(
            _first_present(
                raw.get("utilization"),
                raw.get("used_percentage"),
                raw.get("used_percent"),
                raw.get("percent"),
            )
        )
        if percent is None:
            continue
        windows.append(
            _window(
                window_id,
                label,
                percent,
                _epoch_or_iso(raw.get("resets_at")),
                window_seconds=seconds,
                model_scope=scope,
            )
        )
    extra = payload.get("extra_usage") if isinstance(payload.get("extra_usage"), dict) else {}
    if extra.get("is_enabled"):
        used = _as_float(extra.get("used_credits"))
        limit = _as_float(extra.get("monthly_limit"))
        percent = normalize_percent(extra.get("utilization"))
        if percent is None and used is not None and limit:
            percent = round(min(100.0, used / limit * 100.0), 1)
        windows.append(
            _window(
                "extra_usage",
                "Extra credits",
                percent,
                None,
                used=used,
                limit=limit,
                unit="credits",
            )
        )
    limits = payload.get("limits")
    if isinstance(limits, list):
        bucket_ids = {row["id"] for row in windows}
        for row in limits:
            if not isinstance(row, dict):
                continue
            kind = str(row.get("kind") or "limit")
            if _CLAUDE_LIMIT_BUCKETS.get(kind) in bucket_ids:
                continue
            label = {
                "session": "Session",
                "weekly_all": "Weekly",
                "weekly_scoped": str((row.get("scope") or {}).get("model", {}).get("display_name") or "Weekly scoped"),
            }.get(kind, kind.replace("_", " ").title())
            percent = normalize_percent(_first_present(row.get("percent"), row.get("utilization")))
            if percent is None:
                continue
            windows.append(
                _window(
                    kind,
                    label,
                    percent,
                    _epoch_or_iso(row.get("resets_at")),
                    window_seconds=18_000.0 if kind == "session" else 604_800.0,
                    # A model-scoped bucket meters ONE model, so it is only
                    # spendable while the agent's own pin is that model.
                    model_scope=label if kind == "weekly_scoped" else None,
                )
            )
    # Model-scoped buckets the vendor meters OUTSIDE ``limits[]``. Live
    # that is ``nimbus_quill`` at 0% used. The vendor advertises
    # neither a window length nor a scope for it, so it is reported as a lane
    # while its kind stays a guess (``window_kind_source == "default"``) — the
    # routing digest never advertises it as burst or sustained capacity, and its
    # scope keeps it out of the ranking unless the agent pin is that model.
    for window_id, label in (("nimbus_quill", "Nimbus Quill"),):
        raw = payload.get(window_id)
        if not isinstance(raw, dict):
            continue
        percent = normalize_percent(
            _first_present(raw.get("utilization"), raw.get("used_percentage"), raw.get("used_percent"))
        )
        if percent is None:
            continue
        windows.append(
            _window(
                window_id,
                label,
                percent,
                _epoch_or_iso(raw.get("resets_at") or raw.get("resetsAt")),
                model_scope=label,
            )
        )
    if not windows:
        raise ValueError("Claude usage payload had no quota windows")
    return _provider("claude", plan=plan, windows=windows)


def parse_cursor_payload(payload: dict[str, Any], plan_info: dict[str, Any] | None = None) -> dict[str, Any]:
    plan_usage = payload.get("planUsage") if isinstance(payload.get("planUsage"), dict) else {}
    spend = payload.get("spendLimitUsage") if isinstance(payload.get("spendLimitUsage"), dict) else {}
    info = (
        (plan_info or {}).get("planInfo") if isinstance((plan_info or {}).get("planInfo"), dict) else (plan_info or {})
    )
    plan = str(info.get("planName") or payload.get("planName") or payload.get("membershipType") or "").strip() or None
    windows: list[dict[str, Any]] = []
    total = _cents_to_usd(plan_usage.get("totalSpend"))
    limit = _cents_to_usd(plan_usage.get("limit") or info.get("includedAmountCents"))
    percent = normalize_percent(plan_usage.get("totalPercentUsed"))
    if percent is None and total is not None and limit:
        percent = round(min(100.0, total / limit * 100.0), 1)
    cycle_end = _epoch_or_iso(payload.get("billingCycleEnd") or info.get("billingCycleEnd"))
    cycle_start = _parse_iso_epoch(_epoch_or_iso(payload.get("billingCycleStart") or info.get("billingCycleStart")))
    cycle_end_epoch = _parse_iso_epoch(cycle_end)
    # The vendor's own billing cycle is the period of the included allowance
    # (months are 28–31 days, so a fixed 30-day guess skews the pace).
    cycle_seconds = (
        cycle_end_epoch - cycle_start
        if cycle_start is not None and cycle_end_epoch is not None and cycle_end_epoch > cycle_start
        else None
    )
    if percent is not None or total is not None:
        windows.append(
            _window(
                "plan",
                "Included spend",
                percent,
                cycle_end,
                used=total,
                limit=limit,
                unit="USD",
                window_seconds=cycle_seconds,
            )
        )
    # Auto and API are two allowances with their own meters; both follow the
    # billing cycle, and ``plan`` above is the vendor's total across them.
    pool_dollars = _cursor_pool_dollars(plan_usage)
    for pool_id, pool_label, key in (
        ("auto", "Auto", "autoPercentUsed"),
        ("api", "API", "apiPercentUsed"),
    ):
        pool_used = normalize_percent(plan_usage.get(key))
        if pool_used is not None:
            windows.append(
                _window(
                    pool_id,
                    pool_label,
                    pool_used,
                    cycle_end,
                    window_seconds=cycle_seconds,
                    pool=pool_label,
                    pool_limit_usd=pool_dollars.get(pool_id),
                )
            )
    pooled_limit = _cents_to_usd(spend.get("pooledLimit"))
    pooled_used = _cents_to_usd(spend.get("pooledUsed"))
    if pooled_limit or pooled_used:
        pooled_pct = None
        if pooled_used is not None and pooled_limit:
            pooled_pct = round(min(100.0, pooled_used / pooled_limit * 100.0), 1)
        windows.append(
            _window(
                "spend_limit",
                "Spend limit",
                pooled_pct,
                None,
                used=pooled_used,
                limit=pooled_limit,
                remaining=_cents_to_usd(spend.get("pooledRemaining")),
                unit="USD",
            )
        )
    if not windows:
        raise ValueError("Cursor usage payload had no plan spend")
    return _provider("cursor", plan=plan, windows=windows)


def _cursor_pool_dollars(plan_usage: dict[str, Any]) -> dict[str, float]:
    """Dollar size of the Auto and API pools, solved from the blended total.

    Cursor reports each pool as a percentage only, plus ``totalPercentUsed``
    over ``totalSpend``. That total is dollar-weighted across the two pools, so:

    * both pools together = ``totalSpend / totalPercentUsed``;
    * Auto's share of it = ``(total% - api%) / (auto% - api%)``, API the rest.

    Observed: $456.02 at 36.336% is $1,255; Auto 38.0017% and API 0%
    give $1,200 and $55. Nothing can be solved before anything is spent, or
    while both pools read the same percent — the result is then empty. The
    amounts are reported on the pool windows only; pace never weighs pools
    (``card_pace`` takes the binding limit).
    """
    spend = _cents_to_usd(plan_usage.get("totalSpend"))
    total = _as_float(plan_usage.get("totalPercentUsed"))
    auto = _as_float(plan_usage.get("autoPercentUsed"))
    api = _as_float(plan_usage.get("apiPercentUsed"))
    if not spend or spend <= 0 or not total or total <= 0 or auto is None or api is None:
        return {}
    if abs(auto - api) < 1e-9:
        return {}
    share = (total - api) / (auto - api)
    if not 0.0 < share < 1.0:
        return {}
    both = spend / (total / 100.0)
    auto_dollars = round(both * share, 2)
    return {"auto": auto_dollars, "api": round(both - auto_dollars, 2)}


def parse_agy_payload(payload: dict[str, Any]) -> dict[str, Any]:
    groups = _agy_groups(payload)
    windows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for group in groups:
        name = str(group.get("displayName") or group.get("name") or "Models").strip()
        lowered = name.lower()
        # A group named after its "models" is a shared pool. Any other group is
        # one model's own pool ("Gemini 4 Argon"): it gets its own lane id so it
        # cannot collide with the shared Gemini lanes, and a model scope, so
        # routing only counts it when the agent is pinned to that model.
        shared = "models" in lowered
        if shared and "gemini" in lowered:
            slug = "gemini"
        elif shared and ("claude" in lowered or "gpt" in lowered):
            slug = "claude_gpt"
        else:
            slug = _slug(name)
        model_scope = None if shared else re.sub(r"[^a-z0-9.]+", "-", lowered).strip("-")
        buckets = group.get("buckets") if isinstance(group.get("buckets"), list) else []
        for bucket in buckets:
            if not isinstance(bucket, dict):
                continue
            remaining = _valid_agy_remaining(bucket)
            if remaining is None:
                continue
            used_percent = round(max(0.0, min(100.0, (1.0 - remaining) * 100.0)), 1)
            period = _agy_period(bucket)
            window_id = f"{slug}_{period}"
            if window_id in seen_ids:
                # Two groups resolving to the same lane id would collide; the
                # first group owns the lane and the later one is not advertised
                # twice (usage.js keys rows by window id).
                continue
            seen_ids.add(window_id)
            label = f"{name} · {'Weekly' if period == 'weekly' else '5-hour'}"
            windows.append(
                _window(
                    window_id,
                    label,
                    used_percent,
                    _epoch_or_iso(bucket.get("resetTime") or bucket.get("reset_time") or bucket.get("resetAt")),
                    window_seconds=604_800.0 if period == "weekly" else 18_000.0,
                    model_scope=model_scope or None,
                    pool=name,
                )
            )
    if not windows:
        raise ValueError("Antigravity quota payload had no remainingFraction buckets")
    return _provider("agy", plan="Google", windows=windows)


#: Z.AI quota ``unit`` codes → (window id, label, length in seconds). Unit
#: meanings are not documented by Z.AI; 3 and 6 were established from the
#: vendor's own frontend and community monitors (pi-glm-usage): 3 is the
#: 5-hour prompt quota, 6 the weekly one. A ``nextResetTime`` roughly one
#: week out on live data confirms 6 as weekly.
_ZCODE_UNIT_WINDOWS: dict[int, tuple[str, str, float]] = {
    3: ("five_hour", "5-hour", 18_000.0),
    6: ("seven_day", "Weekly", 604_800.0),
}

_ZCODE_PLAN_NAMES = {"lite": "Lite", "pro": "Pro", "max": "Max"}


def parse_zcode_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Z.AI ``/api/monitor/usage/quota/limit`` → provider card.

    ``data.limits[]`` rows carry the plan's meters: ``percentage`` is the
    vendor's own used percent, ``usage`` the window's credit limit,
    ``currentValue`` the credits used, ``remaining`` the credits left, and
    ``nextResetTime`` a millisecond epoch. ``data.level`` is the plan tier.
    """
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    level = str(data.get("level") or "").strip().lower()
    plan = _ZCODE_PLAN_NAMES.get(level, level.title() if level else None)
    windows: list[dict[str, Any]] = []
    rows = data.get("limits") if isinstance(data.get("limits"), list) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("type") or "") != "CREDIT_LIMIT":
            continue
        try:
            unit = int(row.get("unit"))
        except (TypeError, ValueError):
            continue
        window_id, label, seconds = _ZCODE_UNIT_WINDOWS.get(unit, (f"unit_{unit}", f"Unit {unit}", 0.0))
        percent = normalize_percent(row.get("percentage"))
        used = _as_float(row.get("currentValue"))
        limit = _as_float(row.get("usage"))
        remaining = _as_float(row.get("remaining"))
        if percent is None and used is not None and limit:
            percent = round(min(100.0, used / limit * 100.0), 1)
        if percent is None:
            continue
        # No ``unit``: these credits are the plan's TIMED allowance (the way
        # Claude's 5-hour prompts are), not a prepaid pool — a unit would
        # classify the window as ``spend`` and exempt it from pace. The raw
        # credit amounts still ride ``used``/``limit`` for the row display.
        windows.append(
            _window(
                window_id,
                label,
                percent,
                _epoch_or_iso(row.get("nextResetTime")),
                used=used,
                limit=limit,
                remaining=remaining,
                window_seconds=seconds or None,
            )
        )
    if not windows:
        raise ValueError("ZCode usage payload had no quota limits")
    return _provider("zcode", plan=plan, windows=windows)


_OPENCODE_WINDOWS: dict[str, tuple[str, str, float]] = {
    "rolling": ("rolling", "Rolling", 18_000.0),
    "rolling_usage": ("rolling", "Rolling", 18_000.0),
    "five_hour": ("rolling", "Rolling", 18_000.0),
    "weekly": ("weekly", "Weekly", 604_800.0),
    "weekly_usage": ("weekly", "Weekly", 604_800.0),
    "seven_day": ("weekly", "Weekly", 604_800.0),
    "monthly": ("monthly", "Monthly", 2_592_000.0),
    "monthly_usage": ("monthly", "Monthly", 2_592_000.0),
    "thirty_day": ("monthly", "Monthly", 2_592_000.0),
}


def _month_seconds_before(resets_at: str | None) -> float | None:
    """Length of the calendar month that ends at ``resets_at`` (28–31 days)."""
    try:
        end = datetime.fromisoformat(str(resets_at or "").replace("Z", "+00:00"))
    except ValueError:
        return None
    year, month = (end.year, end.month - 1) if end.month > 1 else (end.year - 1, 12)
    days = calendar.monthrange(year, month)[1]
    start = end.replace(year=year, month=month, day=min(end.day, days))
    return (end - start).total_seconds()


def parse_opencode_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """OpenCode Go ``/zen/go/v1/usage`` → provider card.

    Handles rolling (5-hour), weekly (7-day), and monthly (30-day) quota windows,
    as well as any extra credit or balance pool.
    """
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    usage_dict = data.get("usage") if isinstance(data.get("usage"), dict) else data
    plan_raw = str(data.get("plan") or payload.get("plan") or "Go").strip()
    if plan_raw.lower() in {"go", "opencode-go", "opencode_go"}:
        plan = "Go"
    elif plan_raw.lower() in {"go_plus", "go-plus", "goplus"}:
        plan = "Go Plus"
    else:
        plan = plan_raw.title() if plan_raw else "Go"

    windows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for key, spec in _OPENCODE_WINDOWS.items():
        window_id, label, seconds = spec
        if window_id in seen_ids:
            continue
        row = usage_dict.get(key)
        if not isinstance(row, dict):
            continue
        seen_ids.add(window_id)

        percent = normalize_percent(
            _first_present(
                row.get("percent"),
                row.get("percentage"),
                row.get("usedPercent"),
                row.get("used_percent"),
            )
        )
        used = _as_float(row.get("used") or row.get("currentValue"))
        limit = _as_float(row.get("limit") or row.get("usage"))
        remaining = _as_float(row.get("remaining") or row.get("balance"))
        if percent is None and used is not None and limit:
            percent = round(min(100.0, max(0.0, used / limit * 100.0)), 1)
        if percent is None and remaining is not None and limit:
            percent = round(min(100.0, max(0.0, (1.0 - remaining / limit) * 100.0)), 1)
        if percent is None and remaining is None and used is None:
            continue

        resets_at = _epoch_or_iso(
            row.get("resetsAt") or row.get("resets_at") or row.get("reset_at") or row.get("nextResetTime")
        )
        if window_id == "monthly":
            seconds = _month_seconds_before(resets_at) or seconds
        windows.append(
            _window(
                window_id,
                label,
                percent,
                resets_at,
                used=used,
                limit=limit,
                remaining=remaining,
                window_seconds=seconds,
            )
        )

    # Optional credits / pay-as-you-go balance (e.g. Zen balance or Extra Usage)
    credits = data.get("credits") or data.get("extra_usage") or payload.get("credits") or payload.get("extra_usage")
    if isinstance(credits, dict) and (credits.get("limit") or credits.get("balance") or credits.get("remaining")):
        rem = _as_float(credits.get("balance") or credits.get("remaining"))
        lim = _as_float(credits.get("limit"))
        windows.append(
            _window(
                "credits",
                "Credits",
                None,
                None,
                remaining=rem,
                limit=lim,
                unit="USD",
            )
        )

    if not windows:
        raise ValueError("OpenCode usage payload had no rate-limit windows")
    return _provider("opencode", plan=plan, windows=windows)


def detected_subscription_providers() -> list[dict[str, str]]:
    """Return subscription engines that have a local CLI/OAuth credential."""
    rows: list[dict[str, str]] = []
    for provider_id in PROVIDER_IDS:
        if not subscription_available(provider_id):
            continue
        rows.append(
            {
                "id": provider_id,
                "name": PROVIDER_NAMES[provider_id],
                "kind": "subscription",
            }
        )
    return rows


def discovered_provider_rows() -> list[dict[str, str]]:
    """Live catalog: signed-in engines plus prepaid APIs that have a local key."""
    if fixture_enabled():
        rows: list[dict[str, str]] = []
        for row in fixture_snapshot().get("providers") or []:
            provider_id = str(row.get("id") or "")
            if not provider_id:
                continue
            rows.append(
                {
                    "id": provider_id,
                    "name": str(row.get("name") or PROVIDER_NAMES.get(provider_id, provider_id)),
                    "kind": "subscription" if provider_id in PROVIDER_IDS else "balance",
                }
            )
        return rows
    from sharedagents_web.api_balances import catalog_api_providers

    return detected_subscription_providers() + catalog_api_providers()


def subscription_available(provider_id: str) -> bool:
    """True when this machine has a reusable credential for the engine."""
    if provider_id == "codex":
        return bool(_codex_token()[0])
    if provider_id == "claude":
        return _claude_has_credential()
    if provider_id == "cursor":
        return bool(_cursor_token())
    if provider_id == "agy":
        return _agy_has_credential()
    if provider_id == "zcode":
        return _zcode_has_credential()
    if provider_id == "mimo":
        return _mimo_has_credential()
    if provider_id == "opencode":
        return _opencode_has_credential()
    if provider_id == "devin":
        return _devin_has_credential()
    return False


def catalog() -> dict[str, Any]:
    """Immediate provider list for per-card loading widgets."""
    return {
        "providers": discovered_provider_rows(),
        "source": "fixture" if fixture_enabled() else "live",
    }


def probe_provider(provider_id: str, *, refresh: bool = False, http: HttpFn | None = None) -> dict[str, Any]:
    """Probe one provider and merge it into the cached snapshot."""
    provider_id = str(provider_id or "").strip()
    if not provider_id:
        raise ValueError("provider_id is required")
    if fixture_enabled():
        data = fixture_snapshot()
        card = next(
            (row for row in data["providers"] if row.get("id") == provider_id),
            empty_provider(provider_id, error="Unknown provider"),
        )
        _attach_pace_to_card(card, now=_parse_iso_epoch(data.get("updated_at")))
        return {"updated_at": data.get("updated_at") or _iso_now(), "source": "fixture", "provider": card}
    available = {row["id"] for row in discovered_provider_rows()}
    if provider_id not in available:
        raise ProviderUnavailable(f"{provider_id} is not available on this machine")
    if not refresh:
        cached = _read_fresh_cache()
        if cached is not None:
            for row in cached.get("providers") or []:
                if isinstance(row, dict) and row.get("id") == provider_id:
                    card = dict(row)
                    if not _cached_card_usable(card):
                        break
                    if not card.get("updated_at"):
                        card["updated_at"] = cached.get("updated_at") or _iso_now()
                    _attach_pace_to_card(card)
                    return {
                        "updated_at": card["updated_at"],
                        "source": cached.get("source") or "live",
                        "provider": card,
                    }
    if provider_id in PROVIDER_IDS:
        card = _probe(provider_id, http=http)
    else:
        from sharedagents_web.api_balances import probe_api_provider

        card = probe_api_provider(provider_id, http=http)
    stamp = _iso_now()
    if not card.get("updated_at"):
        card["updated_at"] = stamp
    _merge_cached_provider(card)
    _attach_pace_to_card(card)
    return {"updated_at": card["updated_at"], "source": "live", "provider": card}


def snapshot(*, refresh: bool = False, http: HttpFn | None = None) -> dict[str, Any]:
    """Return cached, fixture, or freshly probed provider cards."""
    if fixture_enabled():
        data = fixture_snapshot()
        return _snapshot_with_history(data)
    cached = None if refresh else _read_fresh_cache()
    if cached is not None:
        return _snapshot_with_history(cached)
    from concurrent.futures import ThreadPoolExecutor

    from sharedagents_web.api_balances import probe_api_provider

    ids = [row["id"] for row in discovered_provider_rows()]

    def _one(provider_id: str) -> dict[str, Any]:
        if provider_id in PROVIDER_IDS:
            return _probe(provider_id, http=http)
        return probe_api_provider(provider_id, http=http)

    if not ids:
        cards = []
    elif len(ids) == 1:
        cards = [_one(ids[0])]
    else:
        with ThreadPoolExecutor(max_workers=min(8, len(ids))) as pool:
            cards = list(pool.map(_one, ids))
    stamp = _iso_now()
    for card in cards:
        if isinstance(card, dict) and not card.get("updated_at"):
            card["updated_at"] = stamp
    data = {
        "updated_at": stamp,
        "source": "live",
        "providers": cards,
    }
    _write_cache(data)
    return _snapshot_with_history(data)


def _probe(provider_id: str, *, http: HttpFn | None = None) -> dict[str, Any]:
    try:
        if provider_id == "codex":
            return _fetch_codex(http)
        if provider_id == "claude":
            return _fetch_claude(http)
        if provider_id == "cursor":
            return _fetch_cursor(http)
        if provider_id == "agy":
            return _fetch_agy(http)
        if provider_id == "zcode":
            return _fetch_zcode(http)
        if provider_id == "mimo":
            return _fetch_mimo()
        if provider_id == "opencode":
            return _fetch_opencode(http)
        if provider_id == "devin":
            return _fetch_devin(http)
    except Exception as exc:  # noqa: BLE001
        return empty_provider(provider_id, error=str(exc) or exc.__class__.__name__)
    return empty_provider(provider_id, error="Unknown provider")


def _fetch_codex(http: HttpFn | None) -> dict[str, Any]:
    token, account_id = _codex_token()
    if not token:
        raise RuntimeError("Codex is not signed in (~/.codex/auth.json)")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    if account_id:
        headers["ChatGPT-Account-Id"] = account_id
    status, payload = _http(http, "GET", "https://chatgpt.com/backend-api/wham/usage", headers, None)
    if status >= 400:
        raise RuntimeError(f"Codex usage HTTP {status}")
    return parse_codex_payload(payload)


def _fetch_claude(http: HttpFn | None) -> dict[str, Any]:
    token = _claude_token()
    if not token:
        raise RuntimeError("Claude is not signed in (~/.claude/.credentials.json)")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "anthropic-beta": "oauth-2025-04-20",
        "anthropic-version": "2023-06-01",
    }
    status, payload = _http(http, "GET", "https://api.anthropic.com/api/oauth/usage", headers, None)
    if status == 401:
        token = _claude_token(refresh=True)
        if not token:
            raise RuntimeError("Claude usage HTTP 401")
        headers["Authorization"] = f"Bearer {token}"
        status, payload = _http(http, "GET", "https://api.anthropic.com/api/oauth/usage", headers, None)
    if status >= 400:
        raise RuntimeError(f"Claude usage HTTP {status}")
    card = parse_claude_payload(payload)
    if not card.get("plan"):
        card["plan"] = _claude_plan_name()
    return card


def _fetch_cursor(http: HttpFn | None) -> dict[str, Any]:
    token = _cursor_token()
    if not token:
        raise RuntimeError("Cursor is not signed in (~/.config/cursor/auth.json)")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Connect-Protocol-Version": "1",
    }
    usage_status, usage = _http(
        http,
        "POST",
        "https://api2.cursor.sh/aiserver.v1.DashboardService/GetCurrentPeriodUsage",
        headers,
        b"{}",
    )
    if usage_status >= 400:
        raise RuntimeError(f"Cursor usage HTTP {usage_status}")
    plan_info: dict[str, Any] = {}
    try:
        _status, plan_info = _http(
            http,
            "POST",
            "https://api2.cursor.sh/aiserver.v1.DashboardService/GetPlanInfo",
            headers,
            b"{}",
        )
    except Exception:  # noqa: BLE001
        plan_info = {}
    return parse_cursor_payload(usage, plan_info if isinstance(plan_info, dict) else {})


_AGY_QUOTA_URL = "https://daily-cloudcode-pa.googleapis.com/v1internal:retrieveUserQuotaSummary"
#: Per-model quota read. Since 2026-10-16 the summary above lists only the
#: "Claude and GPT models" group; the Gemini pool the CLI spends is reported
#: here as one bucket per Gemini model id, all sharing one fraction and reset.
_AGY_MODEL_QUOTA_URL = "https://daily-cloudcode-pa.googleapis.com/v1internal:retrieveUserQuota"
_AGY_GEMINI_POOL = "Gemini Models"
_AGY_USER_AGENT = "antigravity/1.11.3 Linux/x86_64"
#: Quota reads per probe. The backend serves the *same* bucket+reset from two
#: accounting states roughly 0.5-0.6pp apart (measured live: remainingFraction
#: 0.3702 / 0.3764 for gemini-weekly, ~50/50 across 54 reads), so a single read
#: is a coin flip between the fresh and the optimistic lagging state. A few
#: reads almost always surface the fresh one, and the probe runs at most once
#: per cache TTL, so the extra ~0.2s round trips are cheap.
_AGY_QUOTA_ATTEMPTS = 3


def _fetch_agy(http: HttpFn | None) -> dict[str, Any]:
    token = _agy_token()
    if not token:
        raise RuntimeError("Antigravity is not signed in (~/.gemini/antigravity-cli)")
    # The CLI read (~3.5s) runs alongside the RPC reads instead of after them.
    cli_usage = _agy_cli_future()

    def read(url: str, access: str) -> tuple[int, dict[str, Any]]:
        return _http(
            http,
            "POST",
            url,
            {
                "Authorization": f"Bearer {access}",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": _AGY_USER_AGENT,
            },
            b"{}",
        )

    def summary(access: str) -> tuple[int, dict[str, Any]]:
        return read(_AGY_QUOTA_URL, access)

    status, payload = summary(token)
    if status == 401:
        token = _agy_token(refresh=True)
        if not token:
            raise RuntimeError("Antigravity quota HTTP 401")
        status, payload = summary(token)
    if status >= 400:
        hint = ""
        if status == 401 and not _agy_client_secret():
            hint = " (token expired; set AGY_OAUTH_CLIENT_SECRET to enable refresh)"
        raise RuntimeError(f"Antigravity quota HTTP {status}{hint}")
    for extra in _agy_extra_samples(summary, token, _AGY_QUOTA_ATTEMPTS - 1):
        payload = _merge_agy_samples(payload, extra)
    if not any("gemini" in _agy_group_key(group).lower() for group in _agy_groups(payload)):
        gemini = _agy_gemini_group(
            _agy_extra_samples(lambda access: read(_AGY_MODEL_QUOTA_URL, access), token, _AGY_QUOTA_ATTEMPTS)
        )
        if gemini is not None:
            payload = {**payload, "groups": [gemini, *_agy_groups(payload)]}
        else:
            _LOG.warning(
                "Antigravity Gemini quota unavailable: summary has no Gemini group "
                "and per-model reads yielded no usable Gemini bucket; keeping summary lanes"
            )
    try:
        cli = cli_usage.result(timeout=_AGY_CLI_TIMEOUT_S)
        payload = _merge_agy_cli_groups(payload, cli)
    except Exception as exc:  # noqa: BLE001 — malformed optional data must not lose RPC lanes
        _LOG.warning("Antigravity CLI quota read failed (%s)", type(exc).__name__)
    return parse_agy_payload(payload)


#: The agy CLI's own quota screen as JSON. agy >= 1.1.11 answers this read-only
#: slash command locally: no agent turn, no quota spent, no conversation left
#: behind. It is the only reader that sees the pools the Cloud Code RPCs omit for
#: this client: the separate "Gemini 4 Argon" weekly pool
#: and the "Gemini Models" weekly bucket.
_AGY_CLI_USAGE_ARGS = ("-p", "/usage", "--output-format", "json")
_AGY_CLI_TIMEOUT_S = 30.0
_AGY_CLI_MAX_OUTPUT = 1024 * 1024
_AGY_CLI_LOCK = threading.Lock()
# Created on first use, and again after the PID changes. sharedagents-web builds the
# app in the gunicorn master and then forks gthread workers; an import-time
# executor that has already started its idle thread is inherited as a dead
# worker + a still-raised idle semaphore, so submit() never runs.
_AGY_CLI_POOL: ThreadPoolExecutor | None = None
_AGY_CLI_POOL_PID = 0
_AGY_CLI_PENDING: Future | None = None


def _agy_cli_pool() -> ThreadPoolExecutor:
    """Return the per-process one-worker pool, creating it after fork if needed."""
    global _AGY_CLI_POOL, _AGY_CLI_POOL_PID, _AGY_CLI_PENDING
    pid = os.getpid()
    if _AGY_CLI_POOL is None or pid != _AGY_CLI_POOL_PID:
        _AGY_CLI_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agy-quota")
        _AGY_CLI_POOL_PID = pid
        _AGY_CLI_PENDING = None
    return _AGY_CLI_POOL


def _agy_cli_future() -> Future:
    """Share an in-flight CLI read across concurrent cache misses/refreshes.

    No queue of CLI subprocesses accumulates, including when an RPC read fails
    before its caller consumes this future. The reader has its own deadline.
    """
    global _AGY_CLI_PENDING
    with _AGY_CLI_LOCK:
        pool = _agy_cli_pool()
        if _AGY_CLI_PENDING is None or _AGY_CLI_PENDING.done():
            _AGY_CLI_PENDING = pool.submit(_agy_cli_usage)
        return _AGY_CLI_PENDING


def _agy_cli_output(args: list[str], *, deadline: float, limit: int) -> bytes | None:
    """Read bounded stdout before a shared deadline; discard stderr entirely.

    Use a selectable pipe so neither a noisy child nor an inherited pipe can
    defeat the deadline. Platforms without selectable subprocess pipes fail
    open to the RPC lanes. No child output or exception text is logged.
    """
    proc = subprocess.Popen(
        args,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=str(Path.home()),
    )
    try:
        assert proc.stdout is not None
        output = bytearray()
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise subprocess.TimeoutExpired(args, _AGY_CLI_TIMEOUT_S)
                chunk = os.read(proc.stdout.fileno(), min(65536, limit + 1 - len(output)))
                if not chunk:
                    break
                output.extend(chunk)
                if len(output) > limit:
                    _LOG.warning("Antigravity CLI quota output exceeded size limit")
                    return None
        status = proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        if status != 0:
            _LOG.warning("Antigravity CLI quota read exited %s", status)
            return None
        return bytes(output)
    finally:
        # Reap only the child we launched; never signal an interactive CLI.
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close()
        proc.wait(timeout=1.0)


def _agy_cli_usage() -> dict[str, Any]:
    """``{"groups": [...]}`` from ``agy -p /usage``; ``{}`` when it is unavailable.

    Best-effort by design: a missing binary, a timeout, a non-zero exit or an
    unexpected shape all leave the RPC lanes as they are.
    """
    binary = shutil.which("agy") or shutil.which("antigravity")
    if not binary:
        return {}
    try:
        deadline = time.monotonic() + _AGY_CLI_TIMEOUT_S
        version = _agy_cli_output([binary, "--version"], deadline=deadline, limit=4096)
        # Older versions can send slash commands to an agent. Unknown version
        # formats also fail open, before any prompt is passed to the binary.
        match = re.fullmatch(rb"(\d+)\.(\d+)\.(\d+)\s*", version or b"")
        if not match or tuple(int(part) for part in match.groups()) < (1, 1, 11):
            return {}
        raw = _agy_cli_output(
            [binary, *_AGY_CLI_USAGE_ARGS],
            deadline=deadline,
            limit=_AGY_CLI_MAX_OUTPUT,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _LOG.warning("Antigravity CLI quota read failed (%s)", type(exc).__name__)
        return {}
    if raw is None:
        return {}
    try:
        payload = json.loads(raw.decode("utf-8") or "{}")
    except (ValueError, RecursionError):
        return {}
    command = payload.get("command") if isinstance(payload, dict) else None
    if (
        not isinstance(command, dict)
        or command.get("name") != "usage"
        or payload.get("status") != "SUCCESS"
        or payload.get("conversation_id")
        or payload.get("num_turns", 0) != 0
    ):
        return {}
    data = command.get("data") if isinstance(command, dict) else None
    groups = data.get("groups") if isinstance(data, dict) else None
    if not isinstance(groups, list):
        return {}
    return {"groups": [group for group in groups if isinstance(group, dict)]}


def _merge_agy_cli_groups(payload: dict[str, Any], cli: dict[str, Any]) -> dict[str, Any]:
    """Add the CLI's buckets for every (pool, period) the RPC reads did not measure.

    Lanes the RPCs already measured keep the RPC reading, which is the
    more-consumed of several samples. Matching is on pool name and period, not
    bucket id, because the two sources id the same meter differently.
    """
    cli_groups = _agy_groups(cli) if isinstance(cli, dict) else []
    if not cli_groups:
        return payload
    groups = [dict(group) for group in _agy_groups(payload)]
    by_name = {_agy_group_key(group).strip().casefold(): group for group in groups}
    for cli_group in cli_groups:
        name = _agy_group_key(cli_group).strip()
        cli_buckets = cli_group.get("buckets")
        if not name or not isinstance(cli_buckets, list):
            continue
        group = by_name.get(name.casefold())
        if group is None:
            group = {"displayName": name, "buckets": []}
            groups.append(group)
            by_name[name.casefold()] = group
        buckets = [row for row in group.get("buckets") or [] if isinstance(row, dict)]
        measured = {_agy_period(row) for row in buckets if _valid_agy_remaining(row) is not None}
        for bucket in cli_buckets:
            fraction = _valid_agy_remaining(bucket)
            if fraction is None or not 0 <= fraction <= 1:
                continue
            period = _agy_period(bucket)
            if period not in measured:
                buckets.append(bucket)
                measured.add(period)
        group["buckets"] = buckets
    return {**payload, "groups": groups}


def _agy_gemini_group(samples: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Gemini pool as a summary group, built from per-model quota reads.

    Every Gemini model id carries the same pool fraction and reset, so the
    pool is the most-consumed reading across models and samples. Buckets
    without a reset time (tab completion models) are not metered windows.
    """
    picked: dict[str, Any] | None = None
    for sample in samples:
        rows = sample.get("buckets") if isinstance(sample.get("buckets"), list) else []
        for bucket in rows:
            if not isinstance(bucket, dict):
                continue
            if not str(bucket.get("modelId") or "").lower().startswith("gemini"):
                continue
            if not _agy_reset_stamp(bucket) or _agy_remaining_fraction(bucket) is None:
                continue
            picked = bucket if picked is None else _more_consumed_agy_bucket(picked, bucket)
    if picked is None:
        return None
    return {
        "displayName": _AGY_GEMINI_POOL,
        "buckets": [
            {
                "bucketId": "gemini-5h",
                "window": "5h",
                "resetTime": _agy_reset_stamp(picked),
                "remainingFraction": _agy_remaining_fraction(picked),
            }
        ],
    }


def _agy_extra_samples(
    summary: Callable[[str], tuple[int, dict[str, Any]]], token: str, count: int
) -> list[dict[str, Any]]:
    """Extra quota reads for the same request.

    Best-effort by design: the first sample already succeeded, so a failed or
    empty extra sample is skipped (``_merge_agy_samples`` keeps the first).
    """
    samples: list[dict[str, Any]] = []
    for _ in range(max(0, count)):
        try:
            status, payload = summary(token)
        except Exception:  # noqa: BLE001 — extra samples are optional
            break
        if status >= 400 or not isinstance(payload, dict):
            continue
        samples.append(payload)
    return samples


def _merge_agy_samples(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    """Keep the more-consumed reading per bucket across several quota samples.

    ``retrieveUserQuotaSummary`` returns the same bucket + ``resetTime`` from two
    accounting states ~0.5-0.6pp apart (measured live: gemini-weekly read back as
    remainingFraction 0.3702 and 0.3764, with a near 50/50 split over 54 reads),
    so a single read is a coin flip between the fresh ledger and a lagging one.
    Reporting the lagging read is what made the Usage card say "37% left" while
    the ``agy`` CLI screen at the same time said "36% remaining".

    Quota is a decreasing budget, so the lower remaining fraction is the value to
    trust: it never overstates what is left, it does not flicker between the two
    states on every refresh, and it is what routing decisions want.
    """
    readings: dict[tuple[str, str], dict[str, Any]] = {}
    for payload in (primary, secondary):
        if not isinstance(payload, dict):
            continue
        for group in _agy_groups(payload):
            group_key = _agy_group_key(group)
            raw_buckets = group.get("buckets")
            if not isinstance(raw_buckets, list):
                continue
            for bucket in raw_buckets:
                if not isinstance(bucket, dict):
                    continue
                key = (group_key, _agy_bucket_key(bucket))
                current = readings.get(key)
                readings[key] = bucket if current is None else _more_consumed_agy_bucket(current, bucket)
    for group in _agy_groups(primary):
        group_key = _agy_group_key(group)
        raw_buckets = group.get("buckets")
        if not isinstance(raw_buckets, list):
            continue
        for index, bucket in enumerate(raw_buckets):
            if not isinstance(bucket, dict):
                continue
            picked = readings.get((group_key, _agy_bucket_key(bucket)))
            if picked is not None and _agy_remaining_fraction(picked) is not None:
                raw_buckets[index] = picked
    return primary


def _agy_group_key(group: dict[str, Any]) -> str:
    return str(group.get("displayName") or group.get("name") or "")


def _agy_bucket_key(bucket: dict[str, Any]) -> str:
    return str(bucket.get("bucketId") or bucket.get("id") or bucket.get("displayName") or bucket.get("name") or "")


def _more_consumed_agy_bucket(current: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Pick the reading from the accounting state that has consumed more."""
    current_remaining = _agy_remaining_fraction(current)
    candidate_remaining = _agy_remaining_fraction(candidate)
    if candidate_remaining is None:
        return current
    if current_remaining is None:
        return candidate
    current_reset = _agy_reset_stamp(current)
    candidate_reset = _agy_reset_stamp(candidate)
    if current_reset and candidate_reset and current_reset != candidate_reset:
        # A quota window rolled over between the samples: keep the newer window.
        return candidate if candidate_reset > current_reset else current
    return candidate if candidate_remaining < current_remaining else current


def _agy_reset_stamp(bucket: dict[str, Any]) -> str:
    return str(bucket.get("resetTime") or bucket.get("reset_time") or bucket.get("resetAt") or "")


_ZCODE_QUOTA_URL = "https://api.z.ai/api/monitor/usage/quota/limit"
#: The lane the zcode CLI actually dispatches on. When the quota monitor above
#: loses the account's coding-plan record (it answers with an error body —
#: HTTP 200 with ``{"code":500,"msg":"当前用户不存在coding plan"}``
#: after the Nov 1 renewal rollover — and no limit rows), a minimal completion
#: here proves the plan still serves model traffic, and the card stays
#: "unmeasured" instead of "offline".
_ZCODE_LIVENESS_URL = "https://api.z.ai/api/anthropic/v1/messages"
_ZCODE_LIVENESS_MODEL = "GLM-5.3"
_ZCODE_LIVENESS_ANTHROPIC_VERSION = "2023-06-01"

#: Where ``zcode login`` keeps the account's own Z.AI credentials. Every value
#: in it is CLI-keychain ciphertext (``enc:v1:…``), so sharedagents can prove a
#: login EXISTS without ever being able to read the token — exactly the
#: evidence needed to tell "engine offline" apart from "exported key stale".
_ZCODE_CLI_STORE = Path("v2") / "credentials.json"
_ZCODE_CLI_LOGIN_ENTRY = "oauth:zai:access_token"
#: Same relocation the zcode delegate honours for its own reads.
_ZCODE_HOME_ENV = ("ZCODE_HOME", "ZCODE_STORAGE_DIR")


def _zcode_cli_store_path() -> Path:
    for var in _ZCODE_HOME_ENV:
        raw = (os.environ.get(var) or "").strip()
        if raw:
            return Path(raw).expanduser() / _ZCODE_CLI_STORE
    return Path.home() / ".zcode" / _ZCODE_CLI_STORE


def _zcode_cli_login() -> bool:
    """True when the zcode CLI holds its own Z.AI login.

    This is the credential the engine ACTUALLY dispatches on (see
    :mod:`sharedagents_agent.delegates.zcode`), independent of the coding-plan key
    the operator exports for the meters. Only the access-token entry's
    presence is inspected — the value is ciphertext sharedagents never decrypts.
    """
    token = _read_json(_zcode_cli_store_path()).get(_ZCODE_CLI_LOGIN_ENTRY)
    return isinstance(token, str) and bool(token.strip())


def _zcode_unmeasured_card(note: str) -> dict[str, Any]:
    """A ZCode engine that serves turns but exposes no readable meter.

    Same contract as the live-probe card in :func:`_fetch_zcode`: the 0%-used
    5-hour window keeps the pace machinery honest (it reads like a fresh plan)
    and a genuinely spent window still announces itself on dispatch. The
    difference is the evidence — a CLI login rather than a paid-for completion.
    """
    return _provider(
        "zcode",
        plan="Unmeasured (CLI login)",
        windows=[
            _window(
                "five_hour",
                "5-hour · unmeasured",
                0.0,
                None,
                window_seconds=_ZCODE_UNIT_WINDOWS[3][2],
            )
        ],
        note=note,
    )


def _zcode_api_key() -> str:
    """The GLM Coding Plan key the zcode CLI provisions.

    The CLI's own store (``~/.zcode/v2/credentials.json``) encrypts it with the
    CLI's keychain, so the probe reads the exported ``ZCODE_API_KEY`` — the
    same variable the zcode delegate passes through to its subprocesses
    """
    return (os.environ.get("ZCODE_API_KEY") or "").strip()


def _zcode_has_credential() -> bool:
    """True when this machine can run ZCode at all.

    Two independent ways in: the exported coding-plan key (which the meters
    need) or the CLI's own Z.AI login (which is what actually serves turns).
    Listing the engine on the Usage page when only the login exists is the
    point — without it a working ZCode has no card at all.
    """
    return bool(_zcode_api_key()) or _zcode_cli_login()


#: Z.AI frames its error text as ``[<code>][<text>][<request id>]``.
_ZCODE_ERROR_FRAME = re.compile(r"^\[\d+\]\[(?P<text>[^\]]*)\]")
_ZCODE_KEY_EXPIRED = "ZCode usage HTTP 401 (key expired — re-export ZCODE_API_KEY)"


def _zcode_refusal_message(payload: dict[str, Any]) -> str:
    """The lane's own refusal text, without the code/request-id frame.

    Observed (plan lapsed): HTTP 429 with ``{"error": {"code": "1113",
    "message": "[1113][Insufficient balance or no resource package. Please
    recharge.][<request id>]"}}``.
    """
    err = payload.get("error") if isinstance(payload, dict) else None
    message = err.get("message") if isinstance(err, dict) else None
    if not isinstance(message, str):
        return ""
    message = message.strip()
    framed = _ZCODE_ERROR_FRAME.match(message)
    if framed:
        message = framed.group("text").strip()
    return message[:200]


def _zcode_lane_check(http: HttpFn | None, key: str) -> tuple[int | None, str]:
    """A one-token completion on the coding-plan lane the CLI dispatches on.

    This is the engine's actual dispatch lane, so a 200 here is ground truth
    that the plan still serves model traffic even when the quota monitor
    cannot find the plan record. It costs 1-2 tokens and runs only on the
    probe's error path (at most once per cache TTL).

    Returns ``(status, refusal)``: ``status`` is ``None`` when the call never
    got an HTTP answer; ``refusal`` is the vendor's message when the lane
    turns the request down (``""`` when it gave none).
    """
    body = json.dumps(
        {
            "model": _ZCODE_LIVENESS_MODEL,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "ping"}],
        }
    ).encode("utf-8")
    try:
        status, payload = _http(
            http,
            "POST",
            _ZCODE_LIVENESS_URL,
            {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "anthropic-version": _ZCODE_LIVENESS_ANTHROPIC_VERSION,
            },
            body,
        )
    except Exception:  # noqa: BLE001 — network-level failure = lane not proven alive
        return None, ""
    if status == 200:
        return status, ""
    return status, _zcode_refusal_message(payload)


def _fetch_zcode(http: HttpFn | None) -> dict[str, Any]:
    key = _zcode_api_key()
    if not key:
        if _zcode_cli_login():
            # The meters need the exported key; turns do not. Say so instead of
            # pretending the engine is down (the card used to vanish entirely,
            # because ``_zcode_has_credential`` gated the provider row on it).
            return _zcode_unmeasured_card(
                "ZCode runs on the zcode CLI's own Z.AI login, which is "
                "present, so the engine is not offline — but no "
                "``ZCODE_API_KEY`` is exported, so Z.AI's meters cannot be "
                "read. Export the coding-plan key `zcode login` provisions to "
                "see the windows."
            )
        raise RuntimeError(
            "ZCode has no exported key (set ZCODE_API_KEY to the GLM Coding Plan key `zcode login` provisions)"
        )
    card: dict[str, Any] | None = None
    error: Exception | None = None
    status, payload = _http(
        http,
        "GET",
        _ZCODE_QUOTA_URL,
        {"Authorization": f"Bearer {key}", "Accept": "application/json"},
        None,
    )
    if status == 401:
        error = RuntimeError(_ZCODE_KEY_EXPIRED)
    elif status >= 400:
        error = RuntimeError(f"ZCode usage HTTP {status}")
    else:
        try:
            card = parse_zcode_payload(payload)
        except ValueError as exc:
            # No limit rows: either the payload genuinely has no meters, or
            # the monitor has lost the account's coding-plan record.
            error = exc
    if card is not None:
        return card
    assert error is not None, "probe failed without an error and without a card"
    lane_status, refusal = _zcode_lane_check(http, key)
    if lane_status == 200:
        # The plan still serves model traffic on its own lane, so the quota
        # surface's silence is a vendor-side record loss, not exhaustion:
        # report an unmeasured-but-OK card instead of taking a live engine
        # offline. The 0%-used 5-hour window keeps the pace machinery honest
        # (it reads like a fresh plan), and a genuinely spent window still
        # announces itself on dispatch.
        return _provider(
            "zcode",
            plan="Unmeasured (live probe)",
            windows=[
                _window(
                    "five_hour",
                    "5-hour · unmeasured",
                    0.0,
                    None,
                    window_seconds=_ZCODE_UNIT_WINDOWS[3][2],
                )
            ],
            note=(
                "Z.AI's quota monitor reports no coding-plan record for this "
                "key; this card is a live probe of the coding-plan lane "
                "(a minimal completion succeeded), so usage is unmeasured. "
                "If the window is actually spent the engine will say so on "
                "dispatch."
            ),
        )
    if lane_status == 401:
        if _zcode_cli_login():
            # The key the METERS read is dead; the key the ENGINE uses is the
            # CLI's own login, which is present and serves turns. In practice
            # the exported key drew the monitor's 身份验证失败 error
            # body and a 401 on the lane while ``zcode -p`` answered normally.
            # Taking that engine offline would be a lie, so keep the card up
            # and name the one thing that is actually broken.
            return _zcode_unmeasured_card(
                "The zcode CLI signs in with its own Z.AI login and still "
                "serves turns, so the engine is not offline — only the "
                "separately exported ``ZCODE_API_KEY`` is rejected (HTTP 401), "
                "which is what the meters read. Re-export it from the Z.AI "
                "console (the coding-plan key ``zcode login`` provisions) to "
                "restore the windows."
            )
        # The lane rejects the key itself. Its body is Z.AI's localised
        # "authentication failed" (live: ``{"error": {"message": "身份验证失败。",
        # "type": "1000"}}``) and the monitor may even answer 200 with the same
        # failure as an error body, so name the key and how to fix it. With no
        # CLI login behind it there is genuinely nothing left to run turns on.
        raise RuntimeError(_ZCODE_KEY_EXPIRED)
    if refusal:
        # The dispatch lane's own answer is the real reason (the
        # plan lapsed — "Insufficient balance or no resource package"); the
        # monitor's error only says it found no meters.
        raise RuntimeError(f"ZCode coding plan not serving: {refusal}")
    raise error


_OPENCODE_QUOTA_URL = "https://opencode.ai/zen/go/v1/usage"
#: An OpenCode Console login holds a session, not an API key; the key endpoint
#: rejects it, and the same payload is served for it under the inference base.
_OPENCODE_CONSOLE_QUOTA_URL = "https://opencode.ai/inference/go/v1/usage"
#: opencode.ai refuses urllib's default User-Agent (Cloudflare error 1010).
_OPENCODE_USER_AGENT = "sharedagents-usage"


def _opencode_token() -> str:
    """The OpenCode API key (Bearer token).

    Read from environment variables (``OPENCODE_API_KEY``, ``OPENCODE_GO_API_KEY``)
    or the local CLI store (``~/.local/share/opencode/auth.json``).
    """
    env_token = (os.environ.get("OPENCODE_API_KEY") or os.environ.get("OPENCODE_GO_API_KEY") or "").strip()
    if env_token:
        return env_token

    for candidate in (
        Path.home() / ".local" / "share" / "opencode" / "auth.json",
        Path.home() / ".config" / "opencode" / "auth.json",
    ):
        if not candidate.is_file():
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        for key in ("opencode-go", "opencode", "go"):
            entry = data.get(key)
            if isinstance(entry, dict):
                token = entry.get("key") or entry.get("access") or entry.get("token") or entry.get("access_token")
                if token and isinstance(token, str):
                    return token.strip()
        top_token = data.get("key") or data.get("access") or data.get("token") or data.get("access_token")
        if top_token and isinstance(top_token, str):
            return top_token.strip()
    return ""


def _opencode_v2_console_credential(con: sqlite3.Connection) -> tuple[str, str]:
    """``(access token, org id)`` from opencode v2's ``credential`` table.

    v2 (``opencode auth login``) stores the console login as a JSON value
    ``{"type": "oauth", "access", "refresh", "expires", "metadata": {"orgID",
    ...}}`` under ``integration_id = 'opencode'``; the active row wins.
    """
    try:
        rows = con.execute(
            "SELECT value FROM credential WHERE integration_id = 'opencode' "
            "ORDER BY coalesce(active, 0) DESC, time_updated DESC"
        ).fetchall()
    except sqlite3.Error:  # pre-v2 database: no credential table
        return "", ""
    for (raw,) in rows:
        try:
            value = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if not isinstance(value, dict) or value.get("type") != "oauth":
            continue
        token = value.get("access")
        if not isinstance(token, str) or not token.strip():
            continue
        meta = value.get("metadata")
        org = meta.get("orgID") if isinstance(meta, dict) else ""
        return token.strip(), str(org or "").strip()
    return "", ""


def _opencode_v1_console_account(con: sqlite3.Connection) -> tuple[str, str]:
    """``(session token, org id)`` from opencode v1's ``account`` tables."""
    try:
        row = con.execute(
            "SELECT a.access_token, s.active_org_id FROM account_state s "
            "JOIN account a ON a.id = s.active_account_id LIMIT 1"
        ).fetchone()
    except sqlite3.Error:
        return "", ""
    if not row or not isinstance(row[0], str) or not row[0].strip():
        return "", ""
    return row[0].strip(), str(row[1] or "").strip()


def _opencode_console_session() -> tuple[str, str]:
    """``(session token, org id)`` of the active OpenCode Console login.

    The CLI keeps console logins in its own database
    (``~/.local/share/opencode/opencode.db``) and renews them itself, so this
    only reads them. v2's ``credential`` table comes first: v2 leaves the v1
    ``account`` row untouched, so that token lapses in place (v1
    session expired while the v2 credential was valid until 11-17). The v1
    tables remain the fallback for older installs.
    """
    path = Path.home() / ".local" / "share" / "opencode" / "opencode.db"
    if not path.is_file():
        return "", ""
    try:
        con = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=2.0)
        try:
            session = _opencode_v2_console_credential(con)
            if not session[0]:
                session = _opencode_v1_console_account(con)
        finally:
            con.close()
    except sqlite3.Error:
        return "", ""
    return session


def _opencode_has_credential() -> bool:
    return bool(_opencode_token() or _opencode_console_session()[0])


def _fetch_opencode(http: HttpFn | None) -> dict[str, Any]:
    headers = {"Accept": "application/json", "User-Agent": _OPENCODE_USER_AGENT}
    token = _opencode_token()
    url = _OPENCODE_QUOTA_URL
    console = False
    if not token:
        token, org = _opencode_console_session()
        url = _OPENCODE_CONSOLE_QUOTA_URL
        console = True
        if org:
            headers["x-opencode-org-id"] = org
    if not token:
        raise RuntimeError("OpenCode is not signed in (run `opencode auth login` or set OPENCODE_API_KEY)")
    headers["Authorization"] = f"Bearer {token}"
    status, payload = _http(http, "GET", url, headers, None)
    if status == 401 and console:
        raise RuntimeError("OpenCode usage HTTP 401 (console session expired; run `opencode auth login`)")
    if status == 401:
        raise RuntimeError("OpenCode usage HTTP 401 (API key invalid or expired)")
    if status >= 400:
        raise RuntimeError(f"OpenCode usage HTTP {status}")
    return parse_opencode_payload(payload)


def _mimo_env(name: str) -> str:
    from sharedagents_web.api_balances import _env

    return _env(name)


def _mimo_has_credential() -> bool:
    """The MiMo Token Plan key is present (the plan this card meters)."""
    from sharedagents_web import mimo_plan

    return bool(_mimo_env(mimo_plan.PLAN_KEY_ENV))


def _fetch_mimo() -> dict[str, Any]:
    """Token Plan card ESTIMATED from the local usage ledger (no vendor API)."""
    from sharedagents_web import mimo_plan

    try:
        cfg = mimo_plan.plan_config()
        est = mimo_plan.estimate_usage(cfg)
    except mimo_plan.MimoPlanNotConfigured as exc:
        raise RuntimeError(str(exc)) from None
    used, limit = float(est["used"]), float(est["limit"])
    percent = round(min(100.0, max(0.0, used / limit * 100.0)), 1)
    seconds = (est["end"] - est["start"]).total_seconds()
    plan = str(cfg.get("plan") or "").strip() or "Token Plan"
    return _provider(
        "mimo",
        plan=f"{plan} · estimate",
        windows=[
            _window(
                "plan_monthly",
                "Plan · Monthly (est.)",
                percent,
                mimo_plan.iso_z(est["end"]),
                used=used,
                limit=limit,
                remaining=max(0.0, limit - used),
                window_seconds=seconds,
            )
        ],
    )


# --- Devin -----------------------------------------------------------------
#
# There is no PAT-accessible quota REST endpoint: the webapp's Usage & Limits
# page needs a browser session, and the CLI's Connect RPC
# (``exa.seat_management_pb.SeatManagementService/GetUserStatus`` on
# ``server.codeium.com``) rejects the ``cog_`` key. What the key CAN do is
# ``GET https://api.devin.ai/v3/self`` (also the org id), and what the CLI
# itself reads is the ``GetUserStatusResponse`` it caches after every agent
# session under ``~/.cache/devin/cli/user_status.<identity>.bin`` — a JSON
# wrapper around a base64 protobuf whose ``user_status`` message carries
# ``daily_quota_remaining_percent``{14}, ``weekly_quota_remaining_percent``{15},
# ``overage_balance_micros``{8} (int64, ``-1`` = no on-demand pool),
# ``daily_quota_reset_at_unix``{17}, and ``weekly_quota_reset_at_unix``{18}.
# Field numbers verified against the installed CLI's schema tables and a real
# cache file (2026-10); they are positional, so a CLI update may renumber them
# — the parser scans for the shape instead of trusting one path.
#
# The cache changes only while a devin session runs, so on its own it is
# almost always older than the pace freshness limit and Devin could never be
# ranked. The same RPC accepts the key the CLI stores at login, so each refresh
# reads the quota live (``_fetch_devin_live``) and the cache is the fallback.

_DEVIN_SELF_URL = "https://api.devin.ai/v3/self"
#: The CLI only refreshes its cache while a session runs; past this age the
#: numbers no longer describe the current quota window, so the card errors
#: instead of showing stale usage as if it were live.
_DEVIN_CACHE_MAX_AGE_S = 48 * 3600.0
_DEVIN_CACHE_SCAN = 4  # freshest cache files to try (one per CLI identity)
_DEVIN_CACHE_MAX_BYTES = 1_000_000  # bounded cache scan: max 1 MB per file


def _devin_api_key() -> str:
    """The Cognition ``cog_`` key the ``devin`` agent spawns with.

    Environment first, then the same ``secrets.env`` layers the runtime loads
    for the devin agent (per-agent override over the shared file).
    """
    for name in ("DEVIN_API_KEY", "WINDSURF_API_KEY"):
        key = (os.environ.get(name) or "").strip()
        if key:
            return key
    home = sharedagents_root()
    for rel in ("agents/devin/secrets.env", "secrets.env"):
        path = home / rel
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#") or "=" not in stripped:
                continue
            name, _, value = stripped.partition("=")
            if name.strip() in ("DEVIN_API_KEY", "WINDSURF_API_KEY"):
                value = value.strip().strip('"').strip("'")
                if value:
                    return value
    return ""


def _devin_has_credential() -> bool:
    return bool(_devin_api_key())


#: The CLI's own quota RPC. It takes the key the devin CLI stores at login,
#: not the ``cog_`` key, and answers with the same ``GetUserStatusResponse``
#: the CLI caches (proto3 JSON here: camelCase names, int64 as strings).
_DEVIN_STATUS_RPC = "/exa.seat_management_pb.SeatManagementService/GetUserStatus"
_DEVIN_DEFAULT_API_SERVER = "https://server.codeium.com"
#: The RPC rejects a request without client metadata; these identify the
#: caller generically and carry no account data.
_DEVIN_CLIENT_NAME = "devin"
_DEVIN_CLIENT_VERSION = "1.0.0"


def _devin_credentials_path() -> Path:
    override = (os.environ.get("SHAREDAGENTS_DEVIN_CREDENTIALS") or "").strip()
    if override:
        return Path(override)
    data_home = (os.environ.get("XDG_DATA_HOME") or "").strip()
    base = Path(data_home) if data_home else Path.home() / ".local" / "share"
    return base / "devin" / "credentials.toml"


def _devin_cli_credential() -> tuple[str, str]:
    """``(key, api server)`` the devin CLI stored at login, or ``("", "")``.

    ``credentials.toml`` holds flat ``name = "value"`` pairs; reading them
    line by line avoids needing ``tomllib`` on Python 3.10.
    """
    path = _devin_credentials_path()
    try:
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 64_000:
            return "", ""
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "", ""
    values: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "[")) or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        values[name.strip()] = value.strip().strip('"').strip("'")
    key = values.get("windsurf_api_key", "")
    server = values.get("api_server_url", "") or _DEVIN_DEFAULT_API_SERVER
    if not server.startswith("https://"):
        return "", ""
    return key, server.rstrip("/")


def _pb_fields(buf: bytes) -> list[tuple[int, int, Any]]:
    """Minimal protobuf wire scan: ``[(field number, wire type, value)]``.

    Raises ``ValueError`` on truncated or malformed input so a corrupt cache
    file surfaces as a probe error instead of half-parsed numbers.
    """
    out: list[tuple[int, int, Any]] = []
    i = 0
    end = len(buf)

    def varint() -> int:
        nonlocal i
        value = 0
        shift = 0
        while True:
            if i >= end:
                raise ValueError("truncated varint")
            if shift > 63:
                raise ValueError("varint shift overflow")
            byte = buf[i]
            i += 1
            value |= (byte & 0x7F) << shift
            shift += 7
            if not byte & 0x80:
                return value

    while i < end:
        key = varint()
        field, wire = key >> 3, key & 7
        if wire == 0:
            value = varint()
            if value >= 1 << 63:  # int64 sentinels (overage -1) are signed
                value -= 1 << 64
            out.append((field, wire, value))
        elif wire == 2:
            length = varint()
            if i + length > end:
                raise ValueError("truncated length-delimited field")
            out.append((field, wire, buf[i : i + length]))
            i += length
        elif wire == 5:
            if i + 4 > end:
                raise ValueError("truncated fixed32")
            out.append((field, wire, buf[i : i + 4]))
            i += 4
        elif wire == 1:
            if i + 8 > end:
                raise ValueError("truncated fixed64")
            out.append((field, wire, buf[i : i + 8]))
            i += 8
        else:
            raise ValueError(f"unsupported wire type {wire}")
    return out


def _pb_ints(fields: list[tuple[int, int, Any]]) -> dict[int, int]:
    return {field: value for field, wire, value in fields if wire == 0 and isinstance(value, int)}


def _pb_sub_messages(fields: list[tuple[int, int, Any]]) -> list[bytes]:
    out: list[bytes] = []
    for _field, wire, value in fields:
        if wire != 2 or not isinstance(value, (bytes, bytearray)):
            continue
        try:
            _pb_fields(value)
        except ValueError:
            continue  # a string, not a nested message
        out.append(value)
    return out


def _pb_org_ids(buf: bytes, depth: int = 0) -> set[str]:
    """Every organization identifier anywhere in the payload (identity check)."""
    found: set[str] = set()
    if depth > 6:
        return found
    try:
        fields = _pb_fields(buf)
    except ValueError:
        return found
    for _field, wire, value in fields:
        if wire != 2 or not isinstance(value, (bytes, bytearray)):
            continue
        text = None
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError:
            pass
        if text is not None and 3 <= len(text) <= 64 and " " not in text:
            if text.startswith("org-") or text.startswith("org_"):
                found.add(text)
        found |= _pb_org_ids(value, depth + 1)
    return found


def _devin_quota_message(payload: bytes, max_depth: int = 3) -> bytes | None:
    """The ``user_status`` sub-message, located by shape.

    It is the nested message holding at least one plausible reset epoch
    {17, 18}; quota ints {14, 15} can both be omitted when zero in proto3.
    Scanning keeps working if the response
    wrapper renumbers outer fields or nests them deeper.
    """
    queue: list[tuple[bytes, int]] = [(payload, 0)]
    while queue:
        chunk, depth = queue.pop(0)
        try:
            fields = _pb_fields(chunk)
        except ValueError:
            continue
        ints = _pb_ints(fields)
        resets = [ints[r] for r in (17, 18) if r in ints]
        if any(1_500_000_000 <= r <= 2_500_000_000 for r in resets):
            return chunk
        if depth < max_depth:
            for sub in _pb_sub_messages(fields):
                queue.append((sub, depth + 1))
    return None


def _devin_plan_name(status: bytes) -> str | None:
    """Plan label from the plan-info sub-message (short printable string)."""
    for chunk in _pb_sub_messages(_pb_fields(status)):
        for _field, wire, value in _pb_fields(chunk):
            if wire != 2 or not isinstance(value, (bytes, bytearray)):
                continue
            try:
                text = value.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if 0 < len(text) <= 40 and text.isprintable() and not text.startswith("org-"):
                return text
    return None


def _devin_int(value: Any) -> int | None:
    """An int from a protobuf varint or a proto3-JSON number / int64 string."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"-?\d{1,20}", value.strip()):
        return int(value.strip())
    return None


def _devin_card(
    *,
    daily_remaining: int | None,
    weekly_remaining: int | None,
    daily_reset: int | None,
    weekly_reset: int | None,
    overage_micros: int | None,
    plan: str | None,
    updated_at: str | None,
    now: float,
) -> dict[str, Any]:
    """Provider card from the ``GetUserStatusResponse`` quota fields.

    proto3 does not encode a zero: a window whose reset time is present but
    whose remaining percent is absent has 0% left (used up), not "no data".
    A window whose reset already passed has no current reading (None).
    """
    windows: list[dict[str, Any]] = []
    expired_windows: list[str] = []
    for window_id, label, remaining, reset_epoch, seconds in (
        ("daily", "Daily", daily_remaining, daily_reset, 86_400.0),
        ("weekly", "Weekly", weekly_remaining, weekly_reset, 604_800.0),
    ):
        if remaining is None and reset_epoch is not None and reset_epoch > 0:
            remaining = 0
        if remaining is None or not 0 <= remaining <= 100:
            continue
        resets_at = _epoch_or_iso(reset_epoch) if reset_epoch else None
        if reset_epoch is not None and reset_epoch <= now:
            used_pct = None
            expired_windows.append(label)
        else:
            used_pct = round(100.0 - remaining, 1)
        windows.append(
            _window(
                window_id,
                label,
                used_pct,
                resets_at,
                window_seconds=seconds,
            )
        )
    if overage_micros is not None and overage_micros >= 0:
        windows.append(
            _window(
                "overage",
                "On-demand balance",
                None,
                None,
                remaining=round(overage_micros / 1_000_000.0, 2),
                unit="USD",
            )
        )
    if not windows:
        raise ValueError("Devin status payload had no quota windows")
    note = None
    if expired_windows:
        expired_names = " and ".join(expired_windows)
        note = f"{expired_names} quota reset has passed — run devin CLI to refresh"
    return _provider(
        "devin",
        plan=plan,
        windows=windows,
        updated_at=updated_at,
        note=note,
    )


def parse_devin_status(wrapper: dict[str, Any], *, now: float | None = None) -> dict[str, Any]:
    """One ``user_status.*.bin`` JSON wrapper → provider card.

    The wrapper's ``fetched_at_secs`` becomes the card's ``updated_at`` so the
    UI shows the data's real age, not the probe's clock.
    If a quota window's reset timestamp has already passed, its cached usage is
    treated as expired: ``used_percent`` and ``remaining_percent`` are set to None
    so stale or exhausted usage is never displayed as current after a reset.
    """
    if now is None:
        now = time.time()
    raw = wrapper.get("payload")
    if not isinstance(raw, str) or not raw:
        raise ValueError("Devin status cache has no payload")
    try:
        payload = base64.b64decode(raw, validate=False)
    except (ValueError, TypeError) as exc:
        raise ValueError("Devin status cache payload is not base64") from exc
    status = _devin_quota_message(payload)
    if status is None:
        raise ValueError("Devin status payload had no quota message")
    ints = _pb_ints(_pb_fields(status))
    fetched = wrapper.get("fetched_at_secs")
    updated_at = _epoch_or_iso(fetched) if isinstance(fetched, (int, float)) and not isinstance(fetched, bool) else None
    card = _devin_card(
        daily_remaining=ints.get(14),
        weekly_remaining=ints.get(15),
        daily_reset=ints.get(17),
        weekly_reset=ints.get(18),
        overage_micros=ints.get(8),
        plan=_devin_plan_name(status),
        updated_at=updated_at,
        now=now,
    )
    # Even a recent session cache is a fallback, not a successful live read.
    # Preserve its real timestamp while keeping it out of pace recommendations.
    card["reading_source"] = "cli_cache"
    return card


def parse_devin_user_status(payload: dict[str, Any], *, now: float | None = None) -> tuple[dict[str, Any], str]:
    """Live ``GetUserStatus`` JSON → ``(provider card, organization id)``.

    The card is stamped with ``now``: the numbers were read at this moment.
    """
    if now is None:
        now = time.time()
    user = payload.get("userStatus") if isinstance(payload, dict) else None
    plan_status = user.get("planStatus") if isinstance(user, dict) else None
    if not isinstance(plan_status, dict):
        raise ValueError("Devin user status had no plan status")
    plan_info = plan_status.get("planInfo")
    if not isinstance(plan_info, dict):
        plan_info = payload.get("planInfo") if isinstance(payload.get("planInfo"), dict) else {}
    devin_info = plan_info.get("devinInfo") if isinstance(plan_info.get("devinInfo"), dict) else {}
    plan_name = plan_info.get("planName")
    plan = plan_name.strip() if isinstance(plan_name, str) and plan_name.strip() else None
    card = _devin_card(
        daily_remaining=_devin_int(plan_status.get("dailyQuotaRemainingPercent")),
        weekly_remaining=_devin_int(plan_status.get("weeklyQuotaRemainingPercent")),
        daily_reset=_devin_int(plan_status.get("dailyQuotaResetAtUnix")),
        weekly_reset=_devin_int(plan_status.get("weeklyQuotaResetAtUnix")),
        overage_micros=_devin_int(plan_status.get("overageBalanceMicros")),
        plan=plan,
        updated_at=_epoch_or_iso(int(now)),
        now=now,
    )
    return card, str(devin_info.get("orgId") or "").strip()


def _devin_same_org(found: set[str] | str, org_id: str) -> bool:
    def norm(value: str) -> str:
        return value.lower().removeprefix("org-").removeprefix("org_")

    candidates = {found} if isinstance(found, str) else found
    return any(o == org_id or norm(o) == norm(org_id) for o in candidates if o)


def _fetch_devin_live(http: HttpFn | None, *, org_id: str, now: float) -> dict[str, Any]:
    """Read the quota now through the CLI's own RPC; raise with the reason."""
    key, server = _devin_cli_credential()
    if not key:
        raise LookupError("no devin CLI login")
    body = json.dumps(
        {
            "metadata": {
                "apiKey": key,
                "ideName": _DEVIN_CLIENT_NAME,
                "ideVersion": _DEVIN_CLIENT_VERSION,
                "extensionVersion": _DEVIN_CLIENT_VERSION,
            }
        }
    ).encode("utf-8")
    status, payload = _http(
        http,
        "POST",
        server + _DEVIN_STATUS_RPC,
        {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Connect-Protocol-Version": "1",
        },
        body,
    )
    if status >= 400:
        raise RuntimeError(f"GetUserStatus HTTP {status}")
    card, live_org = parse_devin_user_status(payload, now=now)
    if org_id and not _devin_same_org(live_org, org_id):
        raise RuntimeError("CLI login belongs to another organization")
    if not any(w.get("used_percent") is not None or w.get("remaining") is not None for w in card.get("windows", [])):
        raise RuntimeError("GetUserStatus reported no current quota window")
    return card


def _devin_cache_files() -> list[Path]:
    override = (os.environ.get("SHAREDAGENTS_DEVIN_STATUS_CACHE") or "").strip()
    if override:
        p = Path(override)
        return [p] if p.is_file() and not p.is_symlink() else []
    base = Path.home() / ".cache" / "devin" / "cli"
    try:
        candidates = [p for p in base.glob("user_status.*.bin") if p.is_file() and not p.is_symlink()]
        rows = sorted(
            candidates,
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
    except OSError:
        return []
    return rows


def _fetch_devin(http: HttpFn | None) -> dict[str, Any]:
    key = _devin_api_key()
    if not key:
        raise RuntimeError("Devin key missing (set DEVIN_API_KEY/WINDSURF_API_KEY or agents/devin/secrets.env)")
    status, self_payload = _http(
        http,
        "GET",
        _DEVIN_SELF_URL,
        {"Authorization": f"Bearer {key}", "Accept": "application/json"},
        None,
    )
    if status == 401:
        raise RuntimeError("Devin /v3/self HTTP 401 (key invalid or expired)")
    if status >= 400:
        raise RuntimeError(f"Devin /v3/self HTTP {status}")
    org_id = str(
        self_payload.get("primary_billing_org_id")
        or self_payload.get("org_id")
        or self_payload.get("devin_sessions_org_id")
        or ""
    ).strip()
    reasons: list[str] = []
    now = time.time()
    live_failed = False
    try:
        return _fetch_devin_live(http, org_id=org_id, now=now)
    except LookupError as exc:
        reasons.append(str(exc))
    except (OSError, RuntimeError, ValueError) as exc:
        live_failed = True
        reasons.append(f"live read failed: {exc}")
    for path in _devin_cache_files()[:_DEVIN_CACHE_SCAN]:
        try:
            st = path.stat()
            if st.st_size > _DEVIN_CACHE_MAX_BYTES:
                reasons.append(f"{path.name}: file exceeds size limit")
                continue
            wrapper = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            reasons.append(f"{path.name}: unreadable ({exc.__class__.__name__})")
            continue
        if not isinstance(wrapper, dict):
            reasons.append(f"{path.name}: not a JSON object")
            continue
        fetched = wrapper.get("fetched_at_secs")
        if not isinstance(fetched, (int, float)) or isinstance(fetched, bool):
            reasons.append(f"{path.name}: no fetched_at_secs")
            continue
        if now - fetched > _DEVIN_CACHE_MAX_AGE_S:
            reasons.append(f"{path.name}: older than 48 h")
            continue
        try:
            card = parse_devin_status(wrapper, now=now)
        except ValueError as exc:
            reasons.append(f"{path.name}: {exc}")
            continue
        if org_id:
            payload_bytes = base64.b64decode(wrapper.get("payload") or "")
            if not _devin_same_org(_pb_org_ids(payload_bytes), org_id):
                reasons.append(f"{path.name}: another organization")
                continue
        # Verify cache has at least one usable quota or balance window
        has_active_window = any(
            w.get("used_percent") is not None or w.get("remaining") is not None for w in card.get("windows", [])
        )
        if not has_active_window:
            reasons.append(f"{path.name}: all quota windows expired past reset")
            continue
        if live_failed:
            # The cache is only as fresh as the last devin session; its own
            # timestamp keeps pace "unknown" once it is older than the
            # freshness limit, and the note says why the reading is old.
            fallback = "Live reading unavailable — showing the devin CLI's last cached copy"
            card["note"] = f"{fallback}. {card['note']}" if card.get("note") else fallback
        return card
    detail = "; ".join(reasons[:3]) or "no devin CLI cache found"
    raise RuntimeError(
        f"Devin usage unavailable ({detail}) — sign in with `devin auth login` "
        "or run a devin agent session, then press Update"
    )


def _http(
    http: HttpFn | None,
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None,
) -> tuple[int, dict[str, Any]]:
    if http is not None:
        return http(method, url, headers, body, _HTTP_TIMEOUT_S)
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_S) as response:
            raw = response.read(_MAX_BODY + 1)
            if len(raw) > _MAX_BODY:
                raise RuntimeError("usage response exceeded size limit")
            payload = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(payload, dict):
                raise RuntimeError("usage response was not a JSON object")
            return int(response.status), payload
    except urllib.error.HTTPError as exc:
        raw = exc.read(_MAX_BODY) if exc.fp else b""
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except Exception:  # noqa: BLE001
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        return int(exc.code), payload


def _codex_token() -> tuple[str, str]:
    path = _codex_home() / "auth.json"
    data = _read_json(path)
    tokens = data.get("tokens") if isinstance(data.get("tokens"), dict) else {}
    return (
        str(tokens.get("access_token") or data.get("access_token") or "").strip(),
        str(tokens.get("account_id") or data.get("account_id") or "").strip(),
    )


def _claude_token(*, refresh: bool = False) -> str:
    try:
        from sharedagents_agent.discovery.anthropic import (
            _claude_token_expired,
            _read_claude_oauth_record,
            _refresh_claude_oauth_token,
        )
    except Exception:  # noqa: BLE001
        _claude_token_expired = None  # type: ignore[assignment]
        _read_claude_oauth_record = None  # type: ignore[assignment]
        _refresh_claude_oauth_token = None  # type: ignore[assignment]
    if _read_claude_oauth_record is not None:
        record = _read_claude_oauth_record()
        token = str(record.get("accessToken") or "").strip()
        expired = bool(_claude_token_expired and _claude_token_expired(record))
        if (refresh or expired or not token) and _refresh_claude_oauth_token is not None:
            token = str(_refresh_claude_oauth_token(record) or token or "").strip()
        return token
    env = (os.environ.get("CLAUDE_CONFIG_DIR") or "").strip()
    root = Path(env) if env else (Path.home() / ".claude")
    data = _read_json(root / ".credentials.json")
    oauth = data.get("claudeAiOauth") if isinstance(data.get("claudeAiOauth"), dict) else {}
    return str(oauth.get("accessToken") or "").strip()


def _claude_plan_name() -> str | None:
    env = (os.environ.get("CLAUDE_CONFIG_DIR") or "").strip()
    root = Path(env) if env else (Path.home() / ".claude")
    data = _read_json(root / ".credentials.json")
    oauth = data.get("claudeAiOauth") if isinstance(data.get("claudeAiOauth"), dict) else {}
    plan = str(oauth.get("subscriptionType") or "").strip()
    tier = str(oauth.get("rateLimitTier") or "").strip()
    if plan and "max" in tier.lower():
        return "Max"
    return plan.title() if plan else None


def _claude_has_credential() -> bool:
    try:
        from sharedagents_agent.discovery.anthropic import _read_claude_oauth_record
    except Exception:  # noqa: BLE001
        _read_claude_oauth_record = None  # type: ignore[assignment]
    if _read_claude_oauth_record is not None:
        record = _read_claude_oauth_record() or {}
        if record.get("accessToken") or record.get("refreshToken"):
            return True
    env = (os.environ.get("CLAUDE_CONFIG_DIR") or "").strip()
    root = Path(env) if env else (Path.home() / ".claude")
    data = _read_json(root / ".credentials.json")
    oauth = data.get("claudeAiOauth") if isinstance(data.get("claudeAiOauth"), dict) else {}
    return bool(str(oauth.get("accessToken") or oauth.get("refreshToken") or "").strip())


def _agy_has_credential() -> bool:
    env = (os.environ.get("ANTIGRAVITY_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip()
    if env.startswith("ya29"):
        return True
    token = _agy_oauth_record()
    return bool(str(token.get("access_token") or token.get("refresh_token") or "").strip())


def _agy_keyring_record() -> dict[str, Any]:
    """Read the CLI's OS credential store without exposing tokens in argv/logs.

    Keep the keyring backend in a bounded subprocess: a locked or unavailable
    desktop credential store must not hold the web request open indefinitely.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "keyring", "get", "gemini", "antigravity"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5.0,
        )
        if proc.returncode != 0:
            return {}
        data = json.loads(proc.stdout or "{}")
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _agy_oauth_record() -> dict[str, Any]:
    """One token reader shared by sign-in detection and the quota probe."""
    for path in (
        Path.home() / ".gemini" / "antigravity-cli" / "antigravity-oauth-token",
        Path.home() / ".antigravity" / "auth.json",
        Path.home() / ".config" / "antigravity" / "auth.json",
    ):
        data = _read_json(path)
        token = data.get("token") if isinstance(data.get("token"), dict) else data
        if str(token.get("access_token") or token.get("refresh_token") or "").strip():
            return token
    data = _agy_keyring_record()
    return data.get("token") if isinstance(data.get("token"), dict) else data


def _cursor_token() -> str:
    env = (os.environ.get("CURSOR_API_KEY") or "").strip()
    if env:
        return env
    data = _read_json(Path.home() / ".config" / "cursor" / "auth.json")
    return str(data.get("accessToken") or data.get("token") or "").strip()


# Public Antigravity installed-app client id (it appears in every OAuth consent
# URL). The matching client secret is never shipped in source: token refresh
# reads it from AGY_OAUTH_CLIENT_SECRET and is skipped when that is unset.
_AGY_OAUTH_CLIENT_ID = "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com"


def _agy_client_secret() -> str:
    return (os.environ.get("AGY_OAUTH_CLIENT_SECRET") or "").strip()


def _agy_token(*, refresh: bool = False) -> str:
    env = (os.environ.get("ANTIGRAVITY_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip()
    if env and env.startswith("ya29"):
        return env
    token = _agy_oauth_record()
    access = str(token.get("access_token") or "").strip()
    if access and not refresh:
        expiry = str(token.get("expiry") or "")
        if expiry:
            try:
                parsed = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                if parsed.timestamp() - time.time() > 60:
                    return access
            except ValueError:
                return access
        else:
            return access
    refreshed = _refresh_agy_access(str(token.get("refresh_token") or ""))
    return refreshed or access


def _refresh_agy_access(refresh_token: str) -> str:
    client_secret = _agy_client_secret()
    if not refresh_token or not client_secret:
        return ""
    body = urllib.parse.urlencode(
        {
            "client_id": os.environ.get("AGY_OAUTH_CLIENT_ID") or _AGY_OAUTH_CLIENT_ID,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }
    ).encode("utf-8")
    try:
        status, payload = _http(
            None,
            "POST",
            "https://oauth2.googleapis.com/token",
            {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
            body,
        )
    except Exception:  # noqa: BLE001
        return ""
    if status >= 400:
        return ""
    return str(payload.get("access_token") or "").strip()


def _codex_home() -> Path:
    env = (os.environ.get("CODEX_HOME") or "").strip()
    return Path(env) if env else (Path.home() / ".codex")


def _codex_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    by_id = payload.get("rateLimitsByLimitId")
    if isinstance(by_id, dict) and isinstance(by_id.get("codex"), dict):
        return by_id["codex"]
    if isinstance(payload.get("rateLimits"), dict):
        return payload["rateLimits"]
    if isinstance(payload.get("rate_limit"), dict):
        return payload["rate_limit"]
    return payload


def _codex_window_dict(
    snapshot: dict[str, Any],
    payload: dict[str, Any],
    which: str,
) -> dict[str, Any]:
    rate = payload.get("rate_limit") if isinstance(payload.get("rate_limit"), dict) else {}
    candidates = (
        snapshot.get(which),
        snapshot.get(f"{which}_window"),
        rate.get(which),
        rate.get(f"{which}_window"),
    )
    for row in candidates:
        if isinstance(row, dict) and (row.get("usedPercent") is not None or row.get("used_percent") is not None):
            return row
    return {}


def _seconds_to_minutes(value: Any) -> int | None:
    seconds = _as_float(value)
    if seconds is None or seconds <= 0:
        return None
    return int(round(seconds / 60.0))


def _codex_window_label(minutes: Any, fallback: str) -> str:
    try:
        value = int(minutes)
    except (TypeError, ValueError):
        return fallback
    if value == 300:
        return "5-hour"
    if value == 10080:
        return "Weekly"
    if value > 0 and value % 1440 == 0:
        return f"{value // 1440}-day"
    if value > 0 and value % 60 == 0:
        return f"{value // 60}-hour"
    return fallback


def _codex_window_seconds(row: dict[str, Any]) -> float | None:
    """The vendor's window LENGTH for a Codex meter, in seconds.

    Live the primary meter reports ``limit_window_seconds: 604800``
    — weekly — while the parser ids that window ``five_hour``. Routing must
    classify from this field, not from the id.
    """
    seconds = _as_float(_first_present(row.get("limit_window_seconds"), row.get("limitWindowSeconds")))
    if seconds is not None and seconds > 0:
        return seconds
    minutes = _as_float(_first_present(row.get("windowDurationMins"), row.get("window_duration_mins")))
    if minutes is not None and minutes > 0:
        return minutes * 60.0
    return None


def _codex_additional_windows(payload: dict[str, Any], snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """``additional_rate_limits[]`` as extra quota windows.

    The primary ``rate_limit`` object hides every extra meter the plan carries.
    Live ChatGPT returns one: ``limit_name: "gpt-reserve"``, weekly,
    0% used, ``normal_model_slug: "gpt-5.6-luna"`` — a second, idle GPT lane the
    old parser dropped, which made routing believe GPT had only the 76%-used
    weekly meter. Emitted under a distinct id so the primary window (and every
    existing consumer of it) is untouched.
    """
    windows: list[dict[str, Any]] = []
    for row in _codex_additional_limits(payload, snapshot):
        rate = row.get("rate_limit") if isinstance(row.get("rate_limit"), dict) else {}
        primary = rate.get("primary_window") if isinstance(rate.get("primary_window"), dict) else {}
        if not primary and isinstance(row.get("primary_window"), dict):
            primary = row["primary_window"]
        if not primary:
            continue
        percent = normalize_percent(_first_present(primary.get("usedPercent"), primary.get("used_percent")))
        if percent is None and (rate.get("limit_reached") or rate.get("allowed") is False):
            percent = 100.0
        if percent is None:
            continue
        seconds = _codex_window_seconds(primary)
        name = str(_first_present(row.get("limit_name"), row.get("limitName"), "additional") or "additional").strip()
        slug = str(_first_present(row.get("normal_model_slug"), row.get("model_slug"), "") or "").strip()
        windows.append(
            _window(
                f"additional_{_slug(name)}",
                f"{name} · {_codex_window_label(_seconds_to_minutes(seconds), 'Weekly')}",
                percent,
                _epoch_or_iso(
                    _first_present(
                        primary.get("reset_at"),
                        primary.get("resetsAt"),
                        primary.get("resets_at"),
                    )
                ),
                window_seconds=seconds,
                model_slug=slug or None,
                limit_name=name or None,
            )
        )
    return windows


def _codex_additional_limits(payload: dict[str, Any], snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """``additional_rate_limits`` rows, wherever the payload nests them."""
    for source in (payload, snapshot):
        if not isinstance(source, dict):
            continue
        rows = source.get("additional_rate_limits") or source.get("additionalRateLimits")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def _agy_groups(payload: dict[str, Any]) -> list[dict[str, Any]]:
    direct = payload.get("groups")
    if isinstance(direct, list):
        return [row for row in direct if isinstance(row, dict)]
    for key in ("response", "summary"):
        wrapper = payload.get(key)
        if isinstance(wrapper, dict) and isinstance(wrapper.get("groups"), list):
            return [row for row in wrapper["groups"] if isinstance(row, dict)]
    return []


def _agy_remaining_fraction(bucket: dict[str, Any]) -> float | None:
    for key in ("remainingFraction", "remaining_fraction"):
        value = _as_float(bucket.get(key))
        if value is not None:
            return value
    remaining = bucket.get("remaining")
    if isinstance(remaining, dict):
        for key in ("remainingFraction", "remaining_fraction"):
            value = _as_float(remaining.get(key))
            if value is not None:
                return value
        if str(remaining.get("case") or "") == "remainingFraction":
            return _as_float(remaining.get("value"))
    return None


def _valid_agy_remaining(bucket: dict[str, Any]) -> float | None:
    """The bucket's remaining fraction, when it is a usable number.

    ``_as_float`` accepts ``inf``/``nan`` (Python's JSON parser reads those too),
    but a remaining fraction outside ``0..1`` has no meaning, so callers that
    turn it into a percent must reject it the same way the CLI merge does.
    """
    value = _agy_remaining_fraction(bucket)
    if value is not None and math.isfinite(value) and 0.0 <= value <= 1.0:
        return value
    return None


def _agy_period(bucket: dict[str, Any]) -> str:
    label = " ".join(str(bucket.get(key) or "") for key in ("bucketId", "id", "displayName", "name", "window")).lower()
    if "week" in label or "7d" in label or "seven" in label:
        return "weekly"
    return "five_hour"


def _provider(
    provider_id: str,
    *,
    plan: str | None = None,
    windows: list[dict[str, Any]] | None = None,
    error: str | None = None,
    updated_at: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    res = {
        "id": provider_id,
        "name": PROVIDER_NAMES.get(provider_id, provider_id.replace("_", " ").title()),
        "plan": plan,
        "ok": error is None and bool(windows),
        "error": error,
        "windows": windows or [],
        "updated_at": updated_at or _iso_now(),
    }
    if note is not None:
        res["note"] = note
    return res


def _window(
    window_id: str,
    label: str,
    used_percent: float | None,
    resets_at: str | None,
    *,
    used: float | None = None,
    limit: float | None = None,
    remaining: float | None = None,
    unit: str | None = None,
    window_seconds: float | None = None,
    model_slug: str | None = None,
    limit_name: str | None = None,
    model_scope: str | None = None,
    pool: str | None = None,
    pool_limit_usd: float | None = None,
) -> dict[str, Any]:
    remaining_percent = None if used_percent is None else round(max(0.0, 100.0 - used_percent), 1)
    row = {
        "id": window_id,
        "label": label,
        "used_percent": used_percent,
        "remaining_percent": remaining_percent,
        "resets_at": resets_at,
        "used": used,
        "limit": limit,
        "remaining": remaining,
        "unit": unit,
        # Routing metadata (additive): the vendor's window LENGTH in seconds when
        # it advertises one, the model id a reserve meter is tied to, the vendor's
        # own name for the meter, and the model a model-SCOPED bucket belongs to
        # (Claude ``Fable`` / ``Opus`` / ``Sonnet`` weekly buckets). The routing
        # digest classifies lanes from these instead of trusting ``id``.
        "window_seconds": window_seconds,
        "model_slug": model_slug,
        "limit_name": limit_name,
        "model_scope": model_scope,
    }
    # A plan can hold several separately metered allowances (Cursor ``Auto`` and
    # ``API``, Antigravity's model groups). ``pool`` names the one this limit
    # belongs to; a plan with a single allowance leaves it out.
    # ``pool_limit_usd`` is that pool's dollar size when the vendor reports the
    # limit as a percentage and the size had to be worked out.
    if pool:
        row["pool"] = pool
        if pool_limit_usd:
            row["pool_limit_usd"] = pool_limit_usd
    return row


def _cached_card_usable(card: dict[str, Any]) -> bool:
    """Reject a stale xAI key-only card once prepaid dollars can be fetched."""
    if card.get("id") != "xai":
        return True
    for window in card.get("windows") or []:
        if (
            isinstance(window, dict)
            and str(window.get("unit") or "").lower() == "usd"
            and window.get("remaining") is not None
        ):
            return True
    return False


def _read_fresh_cache() -> dict[str, Any] | None:
    data = _read_cache()
    if data is None:
        return None
    if str(data.get("source") or "") == "fixture":
        return None
    stamp = data.get("updated_at")
    try:
        parsed = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
        age = time.time() - parsed.timestamp()
    except (TypeError, ValueError, OSError):
        return None
    if age < 0 or age > _CACHE_TTL_S:
        return None
    providers = data.get("providers")
    if not isinstance(providers, list):
        return None
    allowed = {row["id"] for row in discovered_provider_rows()}
    kept = [row for row in providers if isinstance(row, dict) and row.get("id") in allowed]
    if {row.get("id") for row in kept} != allowed:
        return None
    if any(not _cached_card_usable(row) for row in kept):
        return None
    data["providers"] = kept
    return data


def _read_cache() -> dict[str, Any] | None:
    path = cache_path()
    data = _read_json(path)
    return data or None


def _write_cache(data: dict[str, Any]) -> None:
    path = cache_path()
    if _test_run_on_live_file(path):
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    except OSError:
        return


def _merge_cached_provider(card: dict[str, Any]) -> None:
    cached = _read_cache() or {
        "updated_at": _iso_now(),
        "source": "live",
        "providers": [],
    }
    if str(cached.get("source") or "") == "fixture":
        cached = {
            "updated_at": _iso_now(),
            "source": "live",
            "providers": [],
        }
    allowed = {row["id"] for row in discovered_provider_rows()}
    providers = [row for row in (cached.get("providers") or []) if isinstance(row, dict) and row.get("id") in allowed]
    replaced = False
    for index, row in enumerate(providers):
        if row.get("id") == card.get("id"):
            providers[index] = card
            replaced = True
            break
    if not replaced and card.get("id") in allowed:
        providers.append(card)
    cached["providers"] = providers
    cached["updated_at"] = _iso_now()
    cached["source"] = "live"
    _write_cache(cached)
    # The cache keeps raw cards and pace is worked out on read; saving the raw
    # cache recorded a point with every plan's pace empty.
    record_pace_history(_attach_pace_to_snapshot(copy.deepcopy(cached)))


def window_model_families(
    label: str,
    *,
    window_id: str = "",
    provider_id: str = "",
    model_slug: str = "",
) -> tuple[str, ...]:
    """Model families one quota lane can serve, in ``_FAMILY_ORDER``.

    Label → id → slug keyword scan first, so AGY's combined group ("Claude and
    GPT models · Weekly") reports BOTH ``claude`` and ``gpt`` instead of one,
    and Claude's model-scoped "Fable" bucket still lands on ``claude``. An
    unknown label degrades to the vendor default (Codex → gpt, Claude → claude,
    AGY → gemini, Cursor → other) and never raises.
    """
    haystack = " ".join(str(value or "") for value in (label, window_id, model_slug)).lower()
    found = {family for family, keywords in _FAMILY_KEYWORDS if any(keyword in haystack for keyword in keywords)}
    if found:
        return tuple(family for family in _FAMILY_ORDER if family in found)
    return (_VENDOR_DEFAULT_FAMILY.get(provider_id, "other"),)


def classify_model_family(
    label: str,
    *,
    window_id: str = "",
    provider_id: str = "",
    model_slug: str = "",
) -> str:
    """``gemini`` / ``claude`` / ``gpt`` / ``other`` for one lane label.

    ``"+"``-joined when one lane meters several families at once (AGY's
    combined Claude+GPT group), e.g. ``"claude+gpt"``.
    """
    return "+".join(window_model_families(label, window_id=window_id, provider_id=provider_id, model_slug=model_slug))


def _window_kind_detail(window: dict[str, Any]) -> tuple[str, str]:
    """``(window_kind, source)`` for one window.

    Window LENGTH wins when the vendor advertises it, because vendor ids lie:
    ``parse_codex_payload`` ids the primary GPT window ``five_hour`` while the
    meter is weekly (``limit_window_seconds: 604800``). Labels are the second
    signal, and an unclassifiable window degrades to ``"short"`` with source
    ``"default"`` — the digest never promises sustained capacity it could not
    establish.
    """
    unit = str(window.get("unit") or "").lower()
    if unit in {"usd", "credits"}:
        return "spend", "label"
    seconds = _as_float(window.get("window_seconds"))
    if seconds is not None and seconds > 0:
        return ("short" if seconds <= _SHORT_WINDOW_MAX_S else "long"), "seconds"
    text = f"{window.get('label') or ''} {window.get('id') or ''}".lower()
    if any(token in text for token in _LONG_WINDOW_TOKENS):
        return "long", "label"
    match = re.search(r"(\d+)\s*-?\s*day", text)
    if match:
        return ("long" if int(match.group(1)) >= 3 else "short"), "label"
    if any(token in text for token in _SHORT_WINDOW_TOKENS):
        return "short", "label"
    return "short", "default"


def window_kind(window: dict[str, Any]) -> str:
    """``"short"`` (burst) / ``"long"`` (sustained) / ``"spend"`` for a window."""
    return _window_kind_detail(window)[0]


def window_kind_source(window: dict[str, Any]) -> str:
    """``"seconds"`` / ``"label"`` / ``"default"`` — how ``window_kind`` was decided."""
    return _window_kind_detail(window)[1]


def _engine_kind_for(provider_id: str) -> str:
    """Engine kind (``antigravity_cli`` …) for a subscription provider id.

    Resolved from the external registry (``short_name`` → ``kind``) rather than a
    hardcoded map, so a renamed agent or a new engine flows through.
    """
    if fixture_enabled():
        return str(_fixture_gate_entry(provider_id).get("kind") or "")
    agent = PROVIDER_AGENTS.get(provider_id)
    if not agent:
        return ""
    if provider_id in _MODEL_RAIL_PROVIDERS:
        return _native_plan_engine(provider_id)
    try:
        from sharedagents_agent.external_registry import EXTERNAL_REGISTRY
    except Exception:  # noqa: BLE001 — sharedagents_agent is an optional dependency here
        return ""
    for spec in EXTERNAL_REGISTRY.values():
        if str(getattr(spec, "short_name", "")) == agent:
            return str(getattr(getattr(spec, "kind", ""), "value", "") or "")
    return ""


def _agent_config(agent: str) -> dict[str, Any]:
    """The agent's own on-disk ``config.json`` (``{}`` when absent or unreadable)."""
    try:
        config = json.loads((agents_root() / agent / "config.json").read_text())
    except (OSError, ValueError):
        return {}
    return config if isinstance(config, dict) else {}


def _plan_models(provider_id: str) -> dict[str, str]:
    """``{model id: base URL}`` of the configured models a model-rail plan serves."""
    if provider_id != "mimo":
        return {}
    from sharedagents_web import mimo_plan

    try:
        return mimo_plan.plan_models()
    except Exception:  # noqa: BLE001 — unreadable global config: no plan models
        return {}


def _native_plan_engine(provider_id: str) -> str:
    """Engine of the native agent that spends a model-rail plan (``""`` when none).

    The agent counts only while its own config pins a plan model on the plan's
    host: repointed to another model or endpoint, its turns stop drawing on the
    plan and the lane goes back to unreachable.
    """
    config = _agent_config(PROVIDER_AGENTS[provider_id])
    plan_models = _plan_models(provider_id)
    model = str(config.get("upstream_model") or "")
    if model not in plan_models:
        return ""
    endpoint = str(config.get("upstream_endpoint") or "").rstrip("/")
    if endpoint and endpoint != plan_models[model]:
        return ""
    return str(config.get("engine") or "")


def _engine_config_model_key(engine_kind: str) -> str:
    """The agent-config key that fixes an engine's model.

    ``upstream_model`` for the external-CLI engines and ``acp_model`` for the ACP
    engines (cursor / opencode / kimi / hermes), which never read a per-call
    model at all. Read from the registry, not hardcoded.
    """
    try:
        from sharedagents_agent.external_registry import EXTERNAL_REGISTRY
    except Exception:  # noqa: BLE001
        return "upstream_model"
    for spec in EXTERNAL_REGISTRY.values():
        if str(getattr(getattr(spec, "kind", ""), "value", "")) == engine_kind:
            return "acp_model" if str(getattr(spec, "adapter", "")) == "acp" else "upstream_model"
    return "upstream_model"


def _engine_default_model(engine_kind: str) -> str:
    """The model this engine's own config pins it to (``""`` when unpinned).

    Read through sharedagents's real resolver so the answer matches what a dispatch
    would pass on the wire (operator ``upstream_model``/``acp_model`` →
    ``ANTHROPIC_MODEL`` → ``~/.claude/settings.json`` → …, see
    ``discovery/_common.py`` ``external_cli_effective_default``).
    """
    if not engine_kind:
        return ""
    if fixture_enabled():
        for _provider_id, entry in (_FIXTURE_GATE.get(fixture_scenario_name()) or _FIXTURE_GATE["standard"]).items():
            if str(entry.get("kind") or "") == engine_kind:
                return str(entry.get("pin") or "")
        return ""
    try:
        from sharedagents_agent.discovery._common import external_cli_effective_default

        return str(external_cli_effective_default(engine_kind) or "").strip()
    except Exception:  # noqa: BLE001 — optional dependency / no local config
        return ""


def _validate_model_id(provider_id: str, model_id: str) -> bool:
    """True when the engine's REAL dispatch gate accepts *model_id*.

    Delegates to the same validators a dispatch uses — ``validate_codex_model``
    for Codex, ``validate_external_cli_model`` for the other external engines —
    so reachability is computed from live config instead of being assumed. A
    Gemini-only catalogue therefore yields ``False`` while an operator catalogue
    that lists ``claude-opus-4-6-thinking`` yields ``True``.
    """
    engine_kind = _engine_kind_for(provider_id)
    if not engine_kind or not model_id:
        return False
    if fixture_enabled():
        accepted: list[str] = []
        for ids in (_fixture_gate_entry(provider_id).get("catalogue") or {}).values():
            accepted.extend(ids)
        return model_id in accepted
    try:
        from sharedagents_agent.discovery import _common as discovery
    except Exception:  # noqa: BLE001 — optional dependency
        return False
    try:
        if provider_id == "codex":
            accepted = discovery.validate_codex_model(model_id)
        else:
            accepted = discovery.validate_external_cli_model(engine_kind, model_id)
    except Exception:  # noqa: BLE001 — gate raised: treat as unreachable
        return False
    return bool(str(accepted or "").strip())


def _catalogue_ids(provider_id: str, family: str, cache: dict[str, Any] | None = None) -> list[str]:
    """Catalogue model ids of *family* for a provider (operator ∪ shipped rows).

    The catalogue is the same union the dispatch gate itself consults
    (``_engine_catalog_lookup``), so this asks "which ids could ever spend this
    lane?" without re-implementing any family rule — the vendor validators still
    decide.
    """
    key = f"catalogue:{provider_id}:{family}"
    if cache is not None and key in cache:
        return list(cache[key])
    if fixture_enabled():
        ids = list((_fixture_gate_entry(provider_id).get("catalogue") or {}).get(family) or [])
        if cache is not None:
            cache[key] = list(ids)
        return ids
    ids: list[str] = []
    engine_kind = _engine_kind_for(provider_id)
    try:
        from sharedagents_agent.discovery._common import _engine_catalog_lookup

        rows = _engine_catalog_lookup(engine_kind)
    except Exception:  # noqa: BLE001 — optional dependency
        rows = {}
    for model_id in rows.values():
        model_id = str(model_id or "").strip()
        if model_id and family in window_model_families(model_id, provider_id=provider_id):
            ids.append(model_id)
    ids = sorted(dict.fromkeys(ids))
    if cache is not None:
        cache[key] = list(ids)
    return ids


def _lane_status(window: dict[str, Any]) -> str:
    """``ready`` / ``exhausted`` / ``unknown`` for one lane."""
    percent = _as_float(window.get("remaining_percent"))
    unit = str(window.get("unit") or "").lower()
    usd = _as_float(window.get("remaining")) if unit in {"usd", "credits"} else None
    if percent is None and usd is None:
        return "unknown"
    if (percent is not None and percent > 0) or (usd is not None and usd > 0):
        return "ready"
    return "exhausted"


def _unreachable_lane_unlock(provider_id: str) -> str:
    """What would make a family-less lane dispatchable, or why nothing will."""
    display = PROVIDER_NAMES.get(provider_id, provider_id)
    if provider_id in _MODEL_RAIL_PROVIDERS:
        config_path = agents_root() / PROVIDER_AGENTS[provider_id] / "config.json"
        models = ", ".join(sorted(_plan_models(provider_id))) or "one of the plan's models"
        return (
            f"{display} is a subscription with no agent spending it — set "
            f"{config_path} upstream_model to a plan model ({models})"
        )
    return f"{display} is a prepaid balance rail — pick it as a paid model, not through AgentAsync"


def _lane_access(
    provider_id: str,
    families: tuple[str, ...],
    *,
    model_slug: str = "",
    limit_name: str = "",
    model_scope: str = "",
    cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Computed access verdict for one lane: reachable? spendable right now?

    Nothing is hardcoded per vendor. The ids that could spend the lane come from
    the engine's own catalogue, the engine's real dispatch validator decides
    whether they are accepted, and the engine's resolved default model decides
    whether a default dispatch would actually land there. Both facts matter:
    "an id validates" and "a dispatch would use it" are different, and only the
    second one makes a lane a routing target.

    A lane that carries no model family has no gate that could fail: the vendor
    itself decides which model a dispatch uses, so a reachable one is spendable
    as it stands.
    """
    display = PROVIDER_NAMES.get(provider_id, provider_id)
    engine_kind = _engine_kind_for(provider_id)
    lane_families = [family for family in families if family in MODEL_FAMILIES]
    pin = _engine_default_model(engine_kind)
    pin_families = set(window_model_families(pin, provider_id=provider_id)) if pin else set()
    if not lane_families:
        # Vendor-owned lane (Cursor ``Auto`` / ``API``, balance rails): the vendor
        # picks the model, so there is no model-family gate for the pin check
        # below to fail. A reachable lane therefore IS spendable. Reporting ``[]``
        # here made ``spendable_now`` permanently false, which demoted every
        # family-less roster engine to ``reserve_only`` with a model-pin hint it
        # does not have and hid it from ``available_agents``, while the Usage page
        # badged the very same lane "USE NEXT".
        # ``other`` is the vendor-owned sentinel ``window_model_families`` already
        # uses for those lanes and is deliberately not a ``MODEL_FAMILIES``
        # member, so it never becomes a ``family_capacity`` key.
        reachable = bool(engine_kind)
        return {
            "reachable": reachable,
            "spendable_families": list(families or ("other",)) if reachable else [],
            "requires_model_pin": None,
            "reason": (
                "vendor-owned lane — no model-family gate"
                if reachable
                else f"{display} is not a dispatchable roster engine on this machine"
            ),
            "unlock": None if reachable else _unreachable_lane_unlock(provider_id),
        }
    # Which concrete model ids could spend this lane? A reserve meter that names
    # ITS OWN model (``gpt-reserve`` → ``gpt-5.6-luna``) is judged on that id
    # alone: if the gate rejects it, no other vendor model magically spends that
    # meter. Family lanes without a slug fall back to the engine's catalogue.
    lane_ids = [candidate for candidate in (str(model_slug or "").strip(), str(limit_name or "").strip()) if candidate]
    if lane_ids:
        candidates = list(dict.fromkeys(lane_ids))
    else:
        candidates = []
        for family in lane_families:
            for model_id in _catalogue_ids(provider_id, family, cache):
                if model_id not in candidates:
                    candidates.append(model_id)
    accepted = sorted(candidate for candidate in candidates if _validate_model_id(provider_id, candidate))
    if not accepted:
        return {
            "reachable": False,
            "spendable_families": [],
            "requires_model_pin": None,
            "reason": (
                f"{display} dispatch gate rejects every {'/'.join(lane_families)} model id that "
                f"could spend this lane ({', '.join(candidates) or 'no candidate id'}) — view-only"
            ),
            "unlock": (
                f"{display} cannot select a {'/'.join(lane_families)} model for this meter at all "
                f"(gate: discovery/_common.py validate_external_cli_model / validate_codex_model)"
            ),
        }
    config_key = _engine_config_model_key(engine_kind)
    agent = PROVIDER_AGENTS.get(provider_id) or provider_id
    config_path = agents_root() / agent / "config.json"
    hint = model_slug if model_slug in accepted else accepted[0]
    # A lane is only spendable when the engine's pinned default is (a) of a
    # family the lane meters, (b) the exact model a reserve meter is tied to
    # (``model_slug``), and (c) inside a model-scoped bucket (``model_scope``).
    unmet: list[str] = []
    if not (pin_families & set(lane_families)):
        unmet.append(f"its resolved default is {pin or 'unset'}")
    if model_slug and pin not in {model_slug, limit_name}:
        unmet.append(f"this lane needs model {model_slug}")
    if model_scope and model_scope.lower() not in str(pin or "").lower():
        scope_ids = [
            model_id
            for model_id in _catalogue_ids(provider_id, lane_families[0], cache)
            if model_scope.lower() in model_id.lower()
        ]
        hint = scope_ids[0] if scope_ids else model_scope
        unmet.append(f"this lane is model-scoped to {model_scope} (pin {pin or 'unset'})")
    if not unmet:
        return {
            "reachable": True,
            "spendable_families": list(lane_families),
            "requires_model_pin": None,
            "reason": (
                f"{display} can select {'/'.join(lane_families)} models and is pinned to "
                f"{pin or 'its own default'} — lane is spendable now"
            ),
            "unlock": None,
        }
    return {
        "reachable": True,
        "spendable_families": [],
        "requires_model_pin": hint,
        "reason": (
            f"{display} can select {'/'.join(lane_families)} models ({hint} validates) but "
            f"{'; '.join(unmet)}, so a default dispatch never spends this lane"
        ),
        "unlock": f"set {config_path} {config_key} to e.g. {hint}",
    }


def _model_lane(window: dict[str, Any], provider_id: str, cache: dict[str, Any] | None = None) -> dict[str, Any]:
    """One routing lane: a quota window plus family, kind and computed access."""
    label = str(window.get("label") or window.get("id") or "window")
    model_slug = str(window.get("model_slug") or "").strip()
    limit_name = str(window.get("limit_name") or "").strip()
    model_scope = str(window.get("model_scope") or "").strip()
    families = window_model_families(
        label,
        window_id=str(window.get("id") or ""),
        provider_id=provider_id,
        model_slug=model_slug,
    )
    kind, kind_source = _window_kind_detail(window)
    access = _lane_access(
        provider_id,
        families,
        model_slug=model_slug,
        limit_name=limit_name,
        model_scope=model_scope,
        cache=cache,
    )
    unit = str(window.get("unit") or "").lower()
    return {
        "id": str(window.get("id") or ""),
        "label": label,
        "provider": provider_id,
        "agent": PROVIDER_AGENTS.get(provider_id),
        "model_family": "+".join(families),
        "model_families": list(families),
        "window_kind": kind,
        "window_kind_source": kind_source,
        "used_percent": window.get("used_percent"),
        "remaining_percent": window.get("remaining_percent"),
        "remaining_usd": _as_float(window.get("remaining")) if unit in {"usd", "credits"} else None,
        "resets_at": window.get("resets_at"),
        "status": _lane_status(window),
        "reachable": bool(access["reachable"]),
        "spendable_now": bool(access["spendable_families"]),
        "spendable_families": list(access["spendable_families"]),
        "requires_model_pin": access["requires_model_pin"],
        "unlock": access["unlock"],
        "model_slug": model_slug or None,
        "model_scope": model_scope or None,
        "reason": access["reason"],
    }


def _lane_rank_key(lane: dict[str, Any]) -> tuple[int, str, str]:
    """Deterministic lane order: rounded remaining desc, then provider, then id.

    Rounded to whole points on purpose: AGY reports the same bucket from two
    accounting states ~0.5pp apart (``_merge_agy_samples`` keeps the more
    consumed read), so ordering on decimals would decide lanes by noise. The
    provider/lane-id tie-break keeps the result stable instead.
    """
    remaining = _as_float(lane.get("remaining_percent")) or 0.0
    return (-int(round(remaining)), str(lane.get("provider") or ""), str(lane.get("id") or ""))


def _best_lane(lanes: list[dict[str, Any]], *, kinds: tuple[str, ...] = ("long", "short")) -> dict[str, Any] | None:
    """Looseest lane a default dispatch can actually spend (``None`` when none).

    Two deliberate filters:

    * ``spendable_now`` only. A lane the owning agent's pinned model cannot reach
      (AGY's Claude+GPT group while AGY is pinned to Gemini) must never make a
      provider look like spare capacity.
    * sustained before burst. The first kind with a usable lane wins, so a
      provider is judged on the window that supports real work rather than on a
      short lane that refills in an hour — and on the stabler of the two meters,
      since AGY reports the same bucket from two accounting states ~0.5pp apart.

    Lanes whose window length could not be established (``window_kind_source ==
    "default"``) never rank: an idle bucket whose window is unknown must not
    decide an ordering.
    """
    for kind in kinds:
        usable = [
            lane
            for lane in lanes
            if lane.get("spendable_now")
            and lane.get("status") == "ready"
            and lane.get("window_kind") == kind
            and lane.get("window_kind_source") != "default"
            and _as_float(lane.get("remaining_percent")) is not None
        ]
        if usable:
            return sorted(usable, key=_lane_rank_key)[0]
    return None


def _all_lanes_spent(lanes: list[dict[str, Any]]) -> bool:
    """True only when EVERY measured lane is out of budget.

    This is vendor-wide exhaustion, independent of reachability. The routing
    digest also marks an agent exhausted when its mandatory pinned pool is
    spent, even if the vendor has another healthy pool. Unmeasured lanes are
    ignored, and a card with nothing measured is not called exhausted.
    """
    measured = [lane for lane in lanes if lane.get("status") in {"ready", "exhausted"}]
    return bool(measured) and all(lane.get("status") == "exhausted" for lane in measured)


def _spendable_ready_lanes(lanes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Lanes a DEFAULT dispatch can spend right now.

    ``reachable`` alone is not enough — that is the leak this guard closes. A
    pin-locked lane (AGY's ``Claude and GPT models`` meters while that agent is
    pinned to Gemini, Codex's ``gpt-reserve`` while Codex is pinned to
    ``gpt-6-astra``) is reachable and can sit at 0% used, but a default dispatch
    never spends it, so it must not make a vendor look dispatchable.
    """
    return [lane for lane in lanes if lane.get("spendable_now") and lane.get("status") == "ready"]


def _pin_locked_lanes(lanes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reachable lanes that still carry budget but need a model-pin change.

    These are the ``reserve`` lanes of ``family_capacity``: real quota, idle,
    and out of reach for a default dispatch until the owner repoints the owning
    agent's model. They are reported, never routed on.
    """
    return [
        lane
        for lane in lanes
        if lane.get("reachable") and not lane.get("spendable_now") and lane.get("status") == "ready"
    ]


def _capacity_entry(lane: dict[str, Any], family: str) -> dict[str, Any]:
    """The public shape of one family-capacity answer."""
    return {
        "family": family,
        "window_kind": lane.get("window_kind"),
        "provider": lane.get("provider"),
        "agent": lane.get("agent"),
        "lane_id": lane.get("id"),
        "lane_label": lane.get("label"),
        "remaining_percent": lane.get("remaining_percent"),
        "resets_at": lane.get("resets_at"),
        "reachable": bool(lane.get("reachable")),
        "spendable_now": bool(lane.get("spendable_now")),
        "requires_model_pin": lane.get("requires_model_pin"),
        "model_scope": lane.get("model_scope"),
        "unlock": lane.get("unlock"),
        "reason": lane.get("reason"),
        "pace_used_percent": lane.get("pace_used_percent"),
    }


def _build_family_capacity(lanes: list[dict[str, Any]]) -> dict[str, Any]:
    """Best spendable lane per model family and window kind.

    Answers the question the old single-number digest could not: "I need a
    Claude / GPT / Gemini model — which subscription still has room?".
    ``short`` is the burst (≤1-day) answer, ``long`` the sustained (weekly)
    answer. A lane is advertised only when a default dispatch can actually spend
    it (``spendable_now``) and its window kind was established
    (``window_kind_source`` != ``default``); everything else stays visible but
    inert — ``reserve`` for lanes that only need the owning agent's model pin
    changed, ``unreachable`` for lanes the dispatch gate cannot select at all.
    Money lanes are excluded: they are balance rails, not model capacity.
    """
    capacity: dict[str, Any] = {}
    for family in MODEL_FAMILIES:
        entry: dict[str, Any] = {
            "short": None,
            "long": None,
            "reserve": [],
            "unreachable": [],
        }
        for lane in sorted(
            (lane for lane in lanes if family in (lane.get("model_families") or [])),
            key=_lane_rank_key,
        ):
            summary = _capacity_entry(lane, family)
            if not lane.get("reachable"):
                entry["unreachable"].append(summary)
                continue
            if not lane.get("spendable_now"):
                entry["reserve"].append(summary)
                continue
            if lane.get("window_kind") not in {"short", "long"}:
                continue
            if lane.get("window_kind_source") == "default" or lane.get("status") != "ready":
                continue
            if entry[lane["window_kind"]] is None:
                entry[lane["window_kind"]] = summary
        capacity[family] = entry
    return capacity


#: Provider ``state`` values that mean "a default dispatch can spend this NOW".
#: They are only ever produced for ``status == "ready"`` rows, so the
#: ``available_agents`` gate reads one field instead of re-deriving lane access.
_DISPATCHABLE_STATES = ("spendable", "balance")


def _provider_state(
    status: str,
    *,
    spendable: list[dict[str, Any]],
    pin_locked: list[dict[str, Any]],
    score: float | None,
    score_unit: str,
) -> str:
    """Why a vendor is — or is not — a routing target, in one word.

    ``status`` stays the CONSTRAINT verdict (``ready`` still means "this vendor
    has budget somewhere", so a vendor whose only budget is pin-locked still
    reads ``ready``). ``state`` is the DISPATCH verdict and is what
    ``available_agents`` reads: a vendor whose only remaining budget sits in a
    reserve lane is ``reserve_only`` and must not be advertised as available,
    however reachable that lane is.
    """
    if status != "ready":
        return status
    if spendable:
        return "spendable"
    if score_unit == "usd" and score:
        return "balance"
    if pin_locked:
        return "reserve_only"
    return "unselectable"


def _reserve_only_entry(row: dict[str, Any]) -> dict[str, Any]:
    """The digest's explicit entry for an idle, pin-locked vendor.

    Such a vendor stays OUT of ``available_agents`` — Jarvis reads that list as
    "what I may dispatch" — but dropping it silently would hide real quota, so it
    is reported here with the pin change that would unlock each idle lane.
    """
    lanes = _pin_locked_lanes(row.get("lanes") or [])
    notes = [
        str(note)
        for note in dict.fromkeys(lane.get("unlock") or lane.get("requires_model_pin") or "" for lane in lanes)
        if note
    ]
    return {
        "id": row.get("id"),
        "agent": row.get("agent"),
        "state": "reserve_only",
        "note": "; ".join(notes) or "owner must repoint the owning agent's model pin to spend this lane",
        "lanes": [
            {
                "id": lane.get("id"),
                "label": lane.get("label"),
                "model_family": lane.get("model_family"),
                "remaining_percent": lane.get("remaining_percent"),
                "requires_model_pin": lane.get("requires_model_pin"),
                "unlock": lane.get("unlock"),
            }
            for lane in lanes
        ],
    }


def _engine_rank_key(row: dict[str, Any]) -> tuple[int, float, float, str]:
    """Measured subscription pace first, then reset; capacity only if unknown.

    Rank the dispatchable subset itself. The Usage page's global winner may
    be pin-locked, so its badge cannot select a fallback for the agent.
    """
    if _pace_rankable(row):
        return (0, *_pace_rank_key(row), str(row.get("id") or ""))
    return (1, -int(round(float(row.get("score") or 0.0))), math.inf, str(row.get("id") or ""))


_CLI_QUOTA_NAME = "cli_quota.json"
_CLI_QUOTA_LOCK = threading.Lock()
_RESET_IN_RE = re.compile(
    r"resets in\s+(?:(\d+)\s*d\s*)?(?:(\d+)\s*h\s*)?(?:(\d+)\s*m\s*)?(?:(\d+)\s*s)?",
    re.IGNORECASE,
)
_RESET_AT_RE = re.compile(
    r"(?:will reset at|resets at)\s+"
    r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?)",
    re.IGNORECASE,
)


def cli_quota_path() -> Path:
    """One record per agent next to the subscription cache. Not a second probe."""
    return usage_root() / _CLI_QUOTA_NAME


def _epoch_iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_cli_quota_reset(text: str, *, now: float) -> str | None:
    """ISO reset from a CLI quota error, or None when the text has no future reset.

    Relative ("Resets in 167h", "166h45m42s") is measured from ``now``.
    Absolute ("will reset at 2026-10-02 12:55:22") is read as UTC when the
    vendor omits a zone. A reset that has already passed is not a block.
    """
    raw = str(text or "")
    relative = _RESET_IN_RE.search(raw)
    if relative and any(relative.groups()):
        days, hours, minutes, seconds = (int(part or 0) for part in relative.groups())
        span = days * 86400 + hours * 3600 + minutes * 60 + seconds
        if span > 0:
            return _epoch_iso(now + span)
    absolute = _RESET_AT_RE.search(raw)
    if not absolute:
        return None
    stamp = absolute.group(1).strip().replace(" ", "T")
    if not stamp.endswith("Z") and not re.search(r"[+-]\d{2}:?\d{2}$", stamp):
        stamp += "Z"
    parsed = _parse_iso_epoch(stamp)
    if parsed is None or parsed <= now:
        return None
    return _epoch_iso(parsed)


def _read_cli_quota() -> dict[str, Any]:
    data = _read_json(cli_quota_path())
    rows = data.get("agents") if isinstance(data.get("agents"), dict) else data
    return rows if isinstance(rows, dict) else {}


def _write_cli_quota(rows: dict[str, Any]) -> None:
    path = cli_quota_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"agents": rows}, indent=2) + "\n"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def note_cli_quota_exhaustion(agent: str, text: str, *, now: float | None = None) -> str | None:
    """Remember one engine's CLI quota wall until the reset named in ``text``.

    A later error may extend the reset. It never shortens one that is still
    ahead, and a message with no future reset does not clear the record.
    Returns the ISO reset that is now in force, or None when nothing was kept.
    """
    agent = str(agent or "").strip().lower()
    if not agent:
        return None
    moment = time.time() if now is None else float(now)
    reset = parse_cli_quota_reset(text, now=moment)
    if reset is None:
        return None
    with _CLI_QUOTA_LOCK:
        rows = _read_cli_quota()
        current = rows.get(agent) if isinstance(rows.get(agent), dict) else {}
        current_epoch = _parse_iso_epoch(current.get("resets_at"))
        new_epoch = _parse_iso_epoch(reset)
        if current_epoch is not None and new_epoch is not None and current_epoch >= new_epoch:
            return str(current.get("resets_at") or "")
        rows[agent] = {
            "resets_at": reset,
            "recorded_at": _epoch_iso(moment),
            "detail": str(text or "")[:240],
        }
        try:
            _write_cli_quota(rows)
        except OSError:
            return None
    return reset


def active_cli_quota(
    agent: str,
    *,
    now: float,
    rows: dict[str, Any] | None = None,
) -> dict[str, str] | None:
    """The blocker while this agent's recorded CLI reset is still ahead."""
    agent = str(agent or "").strip().lower()
    if not agent:
        return None
    try:
        table = _read_cli_quota() if rows is None else rows
        row = table.get(agent)
    except OSError:
        return None
    if not isinstance(row, dict):
        return None
    reset = str(row.get("resets_at") or "")
    epoch = _parse_iso_epoch(reset)
    if epoch is None or epoch <= now:
        return None
    shown = reset[5:16] or "unknown"
    return {"resets_at": reset, "blocker": f"CLI quota exhausted, resets {shown}"}


#: How far a cached CLI wall's reset may drift from the reset the vendor
#: reports for a tracked limit and still be recognized as that limit
#: surfacing. The countdown in the CLI error ("Resets in 121h40m0s") is
#: parsed when the error is noted, so the two absolute times can differ by
#: the CLI-to-note delay and the countdown's own rounding.
_CLI_QUOTA_ATTRIBUTION_S = 900.0


#: Pace-lane reasons for a limit this agent cannot spend. A measurement
#: failure (``stale_data``, ``no_reset``, ``unpinned``) is not one of these:
#: an unmeasured lane cannot explain a wall away.
_CLI_QUOTA_UNSPENDABLE_REASONS = frozenset({"pin_locked", "unreachable"})


def _cli_quota_reset_close(reset_at: Any, wall: float) -> bool:
    reset = _parse_iso_epoch(reset_at)
    return reset is not None and abs(reset - wall) <= _CLI_QUOTA_ATTRIBUTION_S


def _cli_quota_dead_known_limit(lane: dict[str, Any]) -> bool:
    """True when this pace lane is a limit the card already shows as hit.

    A counted hit is a pool the agent can spend and pool analysis has already
    judged. An uncounted hit explains a wall only when the lane was excluded
    because the agent cannot spend it (``pin_locked`` / ``unreachable``), which
    is the 2026-10-19 Argon case. A hit whose reason is a measurement failure
    does not.
    """
    if not lane.get("limit_hit"):
        return False
    if lane.get("counted"):
        return True
    return str(lane.get("reason") or "") in _CLI_QUOTA_UNSPENDABLE_REASONS


def cli_quota_explained_by_pools(
    row: dict[str, Any],
    lanes: list[dict[str, Any]] | None,
    cli_quota: dict[str, str],
) -> bool:
    """True when the cached CLI wall is a limit the card already shows as hit.

    ``note_cli_quota_exhaustion`` remembers an engine-wide CLI error such as
    Antigravity's "Individual quota reached", which the vendor does not
    attribute to a pool. Routing consults that cache only after pool analysis
    found no blocker. The cache then yields only when every pace lane resetting
    within the attribution window is a limit the card itself shows limit-hit
    (a counted pool, or an uncounted ``pin_locked`` / ``unreachable`` one) and
    at least one such lane matches. That is the wall being a pool the card has
    already judged, so the documented rule stands: the agent drops only when
    every counted pool is reached, and a pool it cannot spend stays ``not
    counted``.

    An open lane in that window vetoes attribution. Headroom on a limit that
    happens to reset nearby is not evidence the CLI failure was that limit, and
    a genuine account wall (the 2026-10-16 cache) must keep blocking. So does
    any wall while pace is unmeasured, any wall that matches no pace lane, and
    any malformed lane: a bad record keeps the block rather than failing the
    digest. Lanes without a ``counted`` key are ignored, so a raw vendor window
    cannot explain a wall.
    """
    try:
        if not isinstance(row, dict) or not isinstance(cli_quota, dict):
            return False
        if not row.get("ok") or str(row.get("pace_status") or "") != "ok":
            return False
        wall = _parse_iso_epoch(cli_quota.get("resets_at"))
        if wall is None or not isinstance(lanes, list):
            return False
        saw_dead = False
        for lane in lanes:
            if not isinstance(lane, dict) or "counted" not in lane:
                continue
            if not _cli_quota_reset_close(lane.get("resets_at"), wall):
                continue
            if not _cli_quota_dead_known_limit(lane):
                return False
            saw_dead = True
        return saw_dead
    except Exception:
        # A malformed lane must not take down routing. Keeping the block is
        # the safe side of a bad record.
        return False


def routing_blocker(row: dict[str, Any]) -> str | None:
    """One shared exhaustion explanation for the digest and dispatch refusal."""
    if row.get("pace_engine_limit_hit") and row.get("pace_engine_pool"):
        reset = str(row.get("pace_engine_resets_at") or "")[5:16] or "unknown"
        return f"pinned pool {row['pace_engine_pool']} exhausted, resets {reset}"
    if row.get("pace_limit_hit") or row.get("exhausted"):
        reset = str(row.get("pace_engine_resets_at") or row.get("pace_resets_at") or "")[5:16] or "unknown"
        return f"all usable pools exhausted, resets {reset}"
    if row.get("pace_engine_limit_hit"):
        return "pinned model pool exhausted"
    return None


def routing_digest(*, refresh: bool = False) -> dict[str, Any]:
    """Rank usable subscriptions by pace, exposing model pins for task fit.

    Lane-aware: a provider is no longer one number. Each row keeps
    its historical fields (``remaining_percent`` is still the BINDING/tightest
    window so the Usage card, ``static/usage.js`` and the digest agree on the
    constraint, with unchanged types) and adds ``lanes[]`` — every quota window
    classified by model family, window kind and computed access — plus
    ``best_lane`` (looseest lane a default dispatch can spend), ``exhausted``
    (all measured lanes spent OR the agent's pinned model pool is spent),
    ``dispatch_blocker``, ``state`` and ``score_unit``. Vendor-wide quota can
    remain healthy while the pinned model cannot run.

    ``status`` remains the CONSTRAINT verdict (``ready`` = this vendor has budget
    somewhere, so a pin-locked vendor still reads ``ready``); ``state`` is the
    DISPATCH verdict and is what ``available_agents`` reads. ``available_agents``
    lists only vendors a default dispatch can spend RIGHT NOW — a spendable lane
    (``state: spendable``) or a measured balance (``state: balance``). Reachable
    is not enough: a vendor whose only remaining budget sits in reserve lanes is
    ``reserve_only``, and one whose lanes the dispatch gate cannot select at all
    is ``unselectable``. Both stay out of ``available_agents``, so idle-but-locked
    quota can never be mistaken for capacity Jarvis may dispatch.

    ``reserve_only`` reports those pin-locked vendors explicitly (id, agent,
    ``note`` naming the config/pin change that unlocks them, and the idle lanes
    with their remaining percent). They are deliberately NOT dropped from the
    payload: the quota is real and the owner decides whether to repoint a pin.

    ``recommended_agent`` prefers capability tiers within PACE_BAND points of
    the lowest dispatchable measured subscription pace, then pace/reset (the
    Usage page's global winner can be unavailable). ``pace_preferred_agents``
    lists usable subscriptions strictly below the 100% target, in priority
    order. The caller filters these by the task's required intelligence,
    tools and grants before considering a local model. ``pace_candidates``
    exposes the resolved engine/model so a role or vendor name is not used as
    a proxy for intelligence. This is a routing preference, not permission to
    exceed budgets or create work simply to consume quota.

    Pace is the fastest limit among the plan's pools with room left
    (``card_pace``): every limit the agent can spend (5-hour, weekly, monthly,
    every model pool) is paced as used % over the share of its period already
    passed. An agent drops out of ``available_agents`` when its own model
    cannot run (``pace_engine_limit_hit``: a used-up pool scoped to its pinned
    model) or every pool is used up (``pace_limit_hit``). ``pace_lanes`` /
    ``pace_pools`` list every limit and pool with its own pace and state.
    Without measured pace, external recommendations fall back to the best
    ``score``.

    Scoring rule: ``score`` is what a provider can actually SERVE, not its worst
    lane — the best SPENDABLE lane's remaining percent for percent-metered
    subscriptions (sustained lanes first, burst as fallback), the clamped
    remaining balance for dollar-metered providers, 10.0 for console-only keys,
    0.0 when exhausted/offline, and ``None`` (unit ``none``) when there is
    nothing a default dispatch can spend. It stays 0-100 and monotone in that
    capacity, and ``score_unit`` marks the number's unit because percent
    and dollar scores are not comparable: ``recommended_agent`` only ever ranks
    same-unit rows, with the other unit reported in ``recommended_by_unit``.

    ``family_capacity`` maps each model family (``claude`` / ``gpt`` /
    ``gemini``) to its best lane per window kind plus the lanes that exist but
    cannot be spent as-is (``reserve``: needs the owning agent's own model pin;
    ``unreachable``: the dispatch gate cannot select that family at all). All of
    it is computed against live config — no vendor is hardcoded as reachable or
    unreachable.
    """
    # The skill script owns the editable routing policy. Import by module name
    # because the packaged skill's directory contains a hyphen.
    from importlib import import_module

    policy = import_module("sharedagents_skill.builtin_skills.provider-route.scripts.route")
    band = policy.pace_band()
    data = snapshot(refresh=refresh)
    now = _pace_now()
    cli_quota_rows = _read_cli_quota()
    cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    all_lanes: list[dict[str, Any]] = []
    for card in data.get("providers") or []:
        if not isinstance(card, dict) or not card.get("id"):
            continue
        provider_id = str(card["id"])
        windows = [row for row in (card.get("windows") or []) if isinstance(row, dict)]
        lanes = [_model_lane(window, provider_id, cache) for window in windows]
        for window, lane in zip(windows, lanes):
            for key in _PACE_WINDOW_KEYS:
                lane[key] = window.get(key)
        all_lanes.extend(lanes)
        remaining_percent, remaining_usd, status = _card_remaining(card, lanes=lanes)
        best_lane = _best_lane(lanes)
        score, score_unit = _route_score(status, best_lane, remaining_usd)
        state = _provider_state(
            status,
            spendable=_spendable_ready_lanes(lanes),
            pin_locked=_pin_locked_lanes(lanes),
            score=score,
            score_unit=score_unit,
        )
        row = {
            "id": provider_id,
            "name": card.get("name") or PROVIDER_NAMES.get(provider_id, provider_id),
            "agent": PROVIDER_AGENTS.get(provider_id),
            "kind": "subscription" if provider_id in PROVIDER_IDS else "balance",
            "ok": bool(card.get("ok")),
            "plan": card.get("plan"),
            "status": status,
            "state": state,
            "remaining_percent": remaining_percent,
            "remaining_usd": remaining_usd,
            "score": score,
            "score_unit": score_unit,
            "exhausted": _all_lanes_spent(lanes),
            "best_lane": best_lane,
            "lanes": lanes,
            "updated_at": card.get("updated_at"),
        }
        for key in _PACE_ROW_KEYS:
            row[key] = card.get(key)
        engine = _engine_kind_for(provider_id)
        configured_model = ""
        if not fixture_enabled() and row["agent"]:
            config = _agent_config(row["agent"])
            configured_model = str(config.get("upstream_model") or config.get("acp_model") or "")
        model = configured_model if provider_id in _MODEL_RAIL_PROVIDERS else _engine_default_model(engine)
        # The wire resolver returns "" for every ACP engine, and for a CLI
        # whose validator rejected the id. The config pin is still what the
        # operator selected; tier ranking already falls back to it.
        pin = model or configured_model
        # Defend against stale/excluded pace fields as well as fresh snapshots.
        hits = _pinned_pool_hits(windows, pin=pin, now=now)
        if hits:
            row["pace_engine_limit_hit"] = True
            row["pace_engine_pool"] = ", ".join(
                sorted({_pool_name(window) or str(window.get("label") or window.get("id")) for window in hits})
            )
            row["pace_engine_resets_at"] = max(
                hits,
                key=lambda window: _parse_iso_epoch(window.get("resets_at")) or math.inf,
            ).get("resets_at")
        row["dispatch_blocker"] = routing_blocker(row)
        if not row["dispatch_blocker"] and row.get("agent"):
            # A CLI quota wall (individual quota, usage limit) may be an
            # untracked account wall: the reset recorded at classification
            # then stays in force until it passes, even while every percent
            # pool still has headroom. It yields only when every pace lane
            # resetting at that moment is a limit the card already shows
            # limit-hit (the record equalled the dead, pin-locked
            # Argon weekly pool and dropped a Flash-pinned agent the Usage
            # page still called USE NEXT). An open lane that close keeps the
            # block — headroom does not explain a CLI failure away.
            cli_quota = active_cli_quota(row["agent"], now=now, rows=cli_quota_rows)
            if cli_quota and not cli_quota_explained_by_pools(
                row,
                card.get("pace_lanes") or [],
                cli_quota,
            ):
                row["dispatch_blocker"] = cli_quota["blocker"]
        if row["dispatch_blocker"]:
            row["state"] = "exhausted"
            row["exhausted"] = True
        row.update(
            engine=engine,
            model=model,
            configured_model=configured_model,
            capability_tier=policy.capability_tier(model or configured_model),
        )
        row["ranking_pace"] = row.get("pace_used_percent") if _pace_rankable(row) else None
        rows.append(row)
    engines = [
        row
        for row in rows
        # ``state`` is only a dispatchable word when ``status == "ready"``, so
        # this gate is "has something a default dispatch can spend right now" —
        # reachable-but-pin-locked lanes no longer qualify.
        if row["agent"]
        and row["state"] in _DISPATCHABLE_STATES
        and not row.get("pace_limit_hit")
        and not row.get("pace_engine_limit_hit")
    ]
    engines = policy.rank_engines(engines, band=band, pace_key=_engine_rank_key)
    percent_engines = [row for row in engines if row["score_unit"] == "percent"]
    usd_engines = [row for row in engines if row["score_unit"] == "usd"]
    primary = percent_engines or usd_engines
    recommended = primary[0] if primary else None
    recommended_percent = percent_engines[0] if percent_engines else None
    preferred = [row["agent"] for row in engines if _pace_rankable(row) and 0 <= float(row["pace_used_percent"]) < 100]
    candidates = []
    for row in engines:
        # Some adapters leave model selection to the child session handshake.
        # Expose the agent-owned selection separately; it is useful for task
        # fit but must not be misrepresented as a validated runtime model.
        candidates.append(
            {
                "agent": row["agent"],
                "engine": row["engine"],
                # A native agent sends its own pin on the wire: no CLI default applies.
                "model": row["model"],
                "configured_model": row["configured_model"],
                "capability_tier": row["capability_tier"],
                "pace_used_percent": row.get("pace_used_percent") if _pace_rankable(row) else None,
                "prefer_before_local": row["agent"] in preferred,
            }
        )
    exhausted_providers = {row["id"] for row in rows if row["dispatch_blocker"]}
    return {
        "updated_at": data.get("updated_at") or _iso_now(),
        "source": data.get("source") or "live",
        "recommended_agent": recommended["agent"] if recommended else None,
        "available_agents": [row["agent"] for row in engines],
        "pace_target_percent": 100.0,
        "pace_band": band,
        "pace_preferred_agents": preferred,
        "pace_candidates": candidates,
        "recommended_by_unit": {
            "percent": recommended_percent["agent"] if recommended_percent else None,
            "usd": usd_engines[0]["agent"] if usd_engines else None,
        },
        "reserve_only": [_reserve_only_entry(row) for row in rows if row["state"] == "reserve_only"],
        "family_capacity": _build_family_capacity(
            [lane for lane in all_lanes if lane["provider"] not in exhausted_providers]
        ),
        "providers": rows,
    }


def _fmt_num(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return "?"


def _pool_note(pool: dict[str, Any], pools: list[dict[str, Any]]) -> str:
    """One pool of a plan for the snapshot's note column."""
    del pools
    label = str(pool.get("label") or "pool")
    if not pool.get("counted"):
        reason = f": {pool['reason']}" if pool.get("reason") else ""
        return f"{label} {_fmt_num(pool.get('remaining_percent'))}% left, not counted{reason}"
    flag = " (sets the pace)" if pool.get("binding") else ""
    if pool.get("limit_hit"):
        return f"{label} limit hit{flag}"
    return f"{label} pace {_fmt_num(pool.get('pace_used_percent'))}{flag}"


def routing_digest_text(digest: dict[str, Any]) -> str:
    """Compact routing snapshot for the ``provider-route`` skill.

    The full :func:`routing_digest` carries every provider's every lane (tens of
    KB of JSON); the model needs only who to route to and why. One short line
    per dispatchable agent (engine, resolved or configured model, pace, reset),
    what is blocked, prepaid balances, reserve-only quota and the best lane per
    model family. Same facts and ordering as the digest, a few hundred tokens.
    """
    rows = [r for r in (digest.get("providers") or []) if isinstance(r, dict)]
    by_agent = {r.get("agent"): r for r in rows if r.get("agent")}
    available = list(digest.get("available_agents") or [])
    preferred = list(digest.get("pace_preferred_agents") or [])
    lines = [
        f"Snapshot {str(digest.get('updated_at') or '')[:16]}Z ({digest.get('source') or 'live'}); "
        f"pace target {_fmt_num(digest.get('pace_target_percent') or 100)}%.",
        f"Below target, tier first within {_fmt_num(digest.get('pace_band', 25))} pace points "
        "(use before local): " + (", ".join(preferred) or "none"),
        f"Recommended external: {digest.get('recommended_agent') or 'none'}. "
        f"Dispatchable: {', '.join(available) or 'none'}.",
    ]
    winner = next((r for r in rows if r.get("pace_use_next")), None)
    if winner is not None:
        wname = winner.get("agent") or winner.get("id")
        note = "" if winner.get("agent") in available else " (not dispatchable)"
        lines.append(f"Usage-page pace winner: {wname}{note}.")
    lines.append("agent | engine | model | pace% | resets | note")
    for cand in digest.get("pace_candidates") or []:
        agent = cand.get("agent")
        row = by_agent.get(agent) or {}
        model = cand.get("model") or (f"cfg:{cand['configured_model']}" if cand.get("configured_model") else "?")
        pace = cand.get("pace_used_percent")
        notes = []
        if not cand.get("prefer_before_local"):
            notes.append("not below target")
        if row.get("pace_limit_hit"):
            notes.append("limit hit")
        elif row.get("pace_engine_limit_hit"):
            notes.append(f"model pool {row.get('pace_engine_pool') or '?'} limit hit")
        pools = row.get("pace_pools") or []
        if pools:
            notes.append("pools: " + "; ".join(_pool_note(pool, pools) for pool in pools))
        lines.append(
            f"{agent} | {cand.get('engine') or '?'} | {model} | "
            f"{_fmt_num(pace) if pace is not None else '?'} | "
            f"{str(row.get('pace_resets_at') or '')[5:10] or '?'} | {', '.join(notes)}"
        )
    blocked = [
        f"{r['agent']} " + (r.get("dispatch_blocker") or routing_blocker(r) or str(r.get("state") or r.get("status")))
        for r in rows
        if r.get("agent") and r["agent"] not in available
    ]
    if blocked:
        lines.append("Not dispatchable: " + ", ".join(blocked) + ".")
        lines.append(f"Next eligible alternative: {digest.get('recommended_agent') or 'none'}.")
    balances = [
        f"{r.get('id')} ${_fmt_num(r.get('remaining_usd'), 2)}"
        for r in rows
        if r.get("kind") == "balance" and r.get("remaining_usd") is not None
    ]
    if balances:
        lines.append("Prepaid USD (not expiring quota): " + ", ".join(balances) + ".")
    for entry in digest.get("reserve_only") or []:
        lines.append(
            f"Reserve only (locked): {entry.get('agent') or entry.get('id')}: {str(entry.get('note') or '')[:90]}"
        )
    capacity = []
    for family, entry in (digest.get("family_capacity") or {}).items():
        parts = []
        for kind in ("short", "long"):
            lane = (entry or {}).get(kind)
            if lane:
                parts.append(f"{kind} {lane.get('agent')} {_fmt_num(lane.get('remaining_percent'))}%")
        if parts:
            capacity.append(f"{family}: " + ", ".join(parts))
    if capacity:
        lines.append("Best spendable lane by family: " + "; ".join(capacity) + ".")
    return "\n".join(lines)


def _binding_window(windows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the window that actually binds the provider's remaining quota.

    Mirrors ``bindingWindow`` in ``static/usage.js`` so the card and the routing
    digest cannot disagree about the CONSTRAINT — this is what a provider row's
    ``remaining_percent`` still reports. It is deliberately NOT the routing
    ranking any more: ranking reads the looseest lane a default dispatch can
    spend (``_best_lane``), because reducing a provider to its tightest window
    scored a Claude Max at 28% while its weekly lane had 73% free.

    Window order is a vendor detail - Claude lists an idle 5-hour window first -
    so window #0 is not the constraint: percentage rows bind on the lowest
    ``remaining_percent``. Dollar rows are slices of one balance (Balance /
    Topped up / Granted, Available / Cash / Voucher), so they take the first row
    carrying a balance instead - min() would pick a 0.00 voucher row and wrongly
    mark a funded provider exhausted.
    """
    binding: dict[str, Any] | None = None
    lowest = math.inf
    for window in windows:
        value = window.get("remaining_percent")
        if value is None:
            continue
        try:
            remain = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(remain) and remain < lowest:
            lowest = remain
            binding = window
    usd = next(
        (
            window
            for window in windows
            if str(window.get("unit") or "").lower() == "usd" and window.get("remaining") is not None
        ),
        None,
    )
    if usd is not None:
        return usd
    if binding is not None:
        return binding
    return windows[0] if windows else {}


def _card_remaining(
    card: dict[str, Any], *, lanes: list[dict[str, Any]] | None = None
) -> tuple[float | None, float | None, str]:
    """``(binding remaining %, binding remaining USD, status)`` for one card.

    ``remaining_percent`` / ``remaining_usd`` stay the BINDING (tightest) window
    so the Usage card, ``static/usage.js`` ``bindingWindow`` and the digest keep
    agreeing on what constrains the vendor. Once paced, the least left on any
    limit the agent can spend (``pace_remaining_percent``) is used instead, so a
    pool another model owns cannot read as this agent's constraint.

    ``status`` is the routing verdict and is now lane-aware: ``exhausted`` means
    no lane a default dispatch can spend has budget left, so a vendor with one
    dead lane and one healthy lane is no longer reported dead (the old code read
    only ``_binding_window``, i.e. the single tightest window). Ranking is not
    derived from this field — ``score``/``best_lane`` in ``routing_digest`` carry
    "what can this provider actually serve".
    """
    windows = [row for row in (card.get("windows") or []) if isinstance(row, dict)]
    if not card.get("ok"):
        return None, None, "offline"
    binding = _binding_window(windows)
    remaining_percent = float(binding["remaining_percent"]) if binding.get("remaining_percent") is not None else None
    remaining_usd = (
        float(binding["remaining"])
        if str(binding.get("unit") or "").lower() == "usd" and binding.get("remaining") is not None
        else None
    )
    pooled = _as_float(card.get("pace_remaining_percent"))
    if pooled is not None:
        remaining_percent = pooled
    if lanes is None:
        provider_id = str(card.get("id") or "")
        lanes = [_model_lane(window, provider_id) for window in windows]
    reachable = [lane for lane in lanes if lane.get("reachable")]
    if any(lane.get("status") == "ready" for lane in reachable):
        return remaining_percent, remaining_usd, "ready"
    if reachable and all(lane.get("status") == "exhausted" for lane in reachable):
        return remaining_percent, remaining_usd, "exhausted"
    if card.get("console_url") and remaining_percent is None and remaining_usd is None:
        return None, None, "console"
    if remaining_percent is None and remaining_usd is None:
        return None, None, "available"
    return remaining_percent, remaining_usd, "ready"


def _route_score(
    status: str,
    best_lane: dict[str, Any] | None,
    remaining_usd: float | None,
) -> tuple[float | None, str]:
    """``(score, score_unit)`` — the ranking key behind ``recommended_agent``.

    Rule (0-100, monotone in what a provider can actually SERVE):

    * ``percent`` — the best SPENDABLE lane's remaining percent, sustained
      (``long``) lanes first and burst (``short``) lanes as the fallback. Ranking
      on that instead of the tightest window is the fix for a Claude Max whose
      5-hour lane is 72% used while its weekly lane has 73% free: that provider
      was scored 28.0 as if nearly spent. Lanes whose window length could not be
      established never score, so an idle bucket the vendor does not describe
      (Claude ``nimbus_quill``) cannot top the ranking at 100.0.
    * ``usd`` — the clamped remaining balance, unchanged in meaning from before
      (balance cards have no percentage).
    * ``console`` → 10.0, ``exhausted``/``offline`` → 0.0, all with unit
      ``unknown``.
    * nothing a default dispatch can spend (no spendable lane, no measured
      balance) → ``None`` with unit ``none``. There is no honest number for a
      vendor whose only remaining quota is pin-locked, so none is emitted —
      the old ``50.0`` placeholder read as "mid-table capacity" for a vendor
      that in fact could not be dispatched at all.

    ``score_unit`` exists because percent and dollar scores are NOT comparable:
    ``routing_digest`` ranks same-unit rows only, so a $14 balance can no longer
    outrank a 28%-remaining subscription.
    """
    if status in {"exhausted", "offline"}:
        return 0.0, "unknown"
    if status == "console":
        return 10.0, "unknown"
    if best_lane is not None and best_lane.get("remaining_percent") is not None:
        return float(best_lane["remaining_percent"]), "percent"
    if remaining_usd is not None:
        return min(100.0, max(0.0, float(remaining_usd))), "usd"
    return None, "none"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return data if isinstance(data, dict) else {}


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _epoch_or_iso(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value > 1_000_000_000_000:
            value = value / 1000.0
        try:
            return (
                datetime.fromtimestamp(float(value), tz=timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )
        except (OSError, OverflowError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except ValueError:
        return text


def _cents_to_usd(value: Any) -> float | None:
    number = _as_float(value)
    if number is None:
        return None
    return round(number / 100.0, 2)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _slug(value: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")
    return out or "group"
