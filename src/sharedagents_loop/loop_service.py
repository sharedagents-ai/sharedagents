"""Loop service — parsing, creation, lifecycle, command dispatch, tools.

Behavioral mirror of the reference ``/loop`` semantics (clean-room; the
behavioral spec lives in ``/tmp/loop_report.md`` and the implementation
plan in ``plan/2026-08-06-loops-tab.md``):

* interval parsing with ``Ns``/``Nm``/``Nh``/``Nd`` suffixes and
  ``every <N> <unit>`` clauses (1-minute minimum granularity);
* interval → cron conversion with clean-interval rounding;
* deterministic per-loop first-fire jitter
  (``min(hash(id) * 0.5 * period, 30 min)``);
* no age-based expiry and a 50-job ceiling shared with cron jobs;
* the cron-tool family: ``CronCreate``/``CronDelete``/``CronList`` and
  ``ScheduleWakeup`` (delay clamped to 60–3600 s, rounded up to the
  next whole minute; ``stop=True`` cancels pending wakeups);
* ``loop.md`` override resolution (project-local then config dir,
  25,000-byte truncation with a warning line);
* kill-switch: ``SHAREDAGENTS_DISABLE_CRON=1`` disables command + tools.
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sharedagents_core.layout import sharedagents_root
from sharedagents_loop.loop_registry import (
    MAX_LOOP_BUDGET_USD,
    MAX_LOOP_PROMPT_BYTES,
    STATUS_RUNNING,
    LoopFilters,
    LoopRecord,
    LoopRegistry,
    TransitionResult,
    first_fire_jitter_s,  # noqa: F401 — re-exported: /loop tool surface consumes it from here
    get_loop_registry,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default cadence when no interval is given (10 minutes).
DEFAULT_INTERVAL_S = 600

#: loop.md size cap (reference behavior truncates at 25,000 bytes).
LOOP_MD_MAX_BYTES = 25_000

#: Durable prompt cap; unlike loop.md, oversize prompts are refused rather
#: than truncated because silently changing an instruction is unsafe.
LOOP_PROMPT_MAX_BYTES = MAX_LOOP_PROMPT_BYTES

#: ScheduleWakeup clamp window (seconds).
_WAKEUP_MIN_S = 60
_WAKEUP_MAX_S = 3600

#: Clean intervals used when rounding uneven requests.
_CLEAN_MINUTES = (1, 2, 5, 10, 15, 20, 30)
#: Minute values kept exactly (quarter-hour aligned).
_KEEP_MINUTES = (1, 2, 5, 10, 15, 20, 30, 45)
_CLEAN_HOURS = (1, 2, 3, 4, 6, 8, 12, 24)
_CLEAN_DAYS = (1, 2, 3, 4, 7, 14, 28)

_INTERVAL_RE = re.compile(r"^(\d+)([smhd])?$")
_EVERY_CLAUSE_RE = re.compile(
    r"\bevery\s+(\d+)\s*(minutes?|mins?|hours?|hrs?|days?|s|m|h|d)?\b",
    re.IGNORECASE,
)

#: Map "every N <word>" unit words onto the suffix alphabet.
_EVERY_UNIT_MAP = {
    "s": "s",
    "m": "m",
    "h": "h",
    "d": "d",
    "minute": "m",
    "minutes": "m",
    "min": "m",
    "mins": "m",
    "hour": "h",
    "hours": "h",
    "hr": "h",
    "hrs": "h",
    "day": "d",
    "days": "d",
}

#: Kill-switch env var (the reference tooling's disable switch equivalent).
DISABLE_CRON_ENV = "SHAREDAGENTS_DISABLE_CRON"


class IntervalParseError(ValueError):
    """Raised when an interval token cannot be turned into a schedule."""


@dataclass
class LoopResult:
    """Outcome of a loop-creation attempt (mirrors the goals result type)."""

    ok: bool
    message: str
    loop: LoopRecord | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "message": self.message,
            "loop": self.loop.to_dict() if self.loop else None,
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# Interval parsing / cron conversion
# ---------------------------------------------------------------------------


def _clean_round(value: int, options: tuple[int, ...]) -> int:
    """Round ``value`` to the nearest entry of ``options`` (ties → lower)."""
    best = options[0]
    for opt in options:
        if abs(opt - value) < abs(best - value):
            best = opt
    return best


def interval_to_cron(interval_s: int) -> str:
    """Convert a whole-second interval into a 5-field cron expression."""
    if interval_s < 60:
        raise IntervalParseError("minimum interval is 1 minute")
    minutes = interval_s // 60
    if interval_s % 3600 == 0:
        hours = interval_s // 3600
        if hours <= 23:
            return f"0 */{hours} * * *"
    if interval_s % 86400 == 0:
        days = interval_s // 86400
        if days <= 28:
            return f"0 0 */{days} * *"
    if minutes <= 59:
        return f"*/{minutes} * * * *"
    raise IntervalParseError("minute intervals must be 1-59 (use hours instead)")


def period_label(interval_s: int) -> str:
    """Human label for a whole-second interval ("every 30m")."""
    if interval_s < 60:
        return f"every {interval_s}s"
    if interval_s < 3600:
        return f"every {interval_s // 60}m"
    if interval_s < 86400:
        if interval_s % 3600 == 0:
            return f"every {interval_s // 3600}h"
        return f"every {interval_s // 60}m"
    if interval_s % 86400 == 0:
        return f"every {interval_s // 86400}d"
    return f"every {interval_s // 3600}h"


def parse_interval(text: str) -> tuple[int, str, str, str | None]:
    """Parse an interval token into ``(seconds, cron, label, note)``.

    Accepts ``<N>`` (bare seconds, 60+), ``<N>s``, ``<N>m``, ``<N>h``,
    ``<N>d``, and ``every <N> <unit>``
    clauses. Uneven intervals round to the nearest clean interval, and
    the rounding is reported via the returned ``note``.
    """
    text = (text or "").strip().lower()
    if not text:
        raise IntervalParseError("empty interval")
    every = _EVERY_CLAUSE_RE.fullmatch(text)
    if every:
        unit = _EVERY_UNIT_MAP.get((every.group(2) or "m").lower(), "m")
        text = f"{every.group(1)}{unit}"
    m = _INTERVAL_RE.match(text)
    if not m:
        raise IntervalParseError(f"invalid interval: {text!r}")
    value = int(m.group(1))
    unit = m.group(2)
    if value <= 0:
        raise IntervalParseError("minimum interval is 1 minute")
    if unit in (None, "s"):  # bare and suffixed seconds
        if value < 60:
            raise IntervalParseError("minimum interval is 1 minute")
        rounded_minutes = (value + 59) // 60
        interval_s, cron, label, note = parse_interval(f"{rounded_minutes}m")
        if interval_s != value:
            note = f"rounded {value}s to {label}"
        return interval_s, cron, label, note
    elif unit == "m":
        if value >= 24 * 60:
            raise IntervalParseError("day intervals must be 1-28 (use a cron expression)")
        if value <= 59:
            if value in _KEEP_MINUTES:
                interval_s = value * 60
                return interval_s, interval_to_cron(interval_s), period_label(interval_s), None
            rounded = _clean_round(value, _CLEAN_MINUTES + (60,))
            if rounded == 60:
                return 3600, "0 */1 * * *", "every 1h", (f"rounded {value}m to every 1h")
            return (
                rounded * 60,
                interval_to_cron(rounded * 60),
                period_label(rounded * 60),
                f"rounded {value}m to {period_label(rounded * 60)}",
            )
        hours = _clean_round(value / 60, _CLEAN_HOURS)
        interval_s = hours * 3600
        note = None if interval_s == value * 60 else f"rounded {value}m to {period_label(interval_s)}"
        return interval_s, interval_to_cron(interval_s), period_label(interval_s), note
    elif unit == "h":
        if value > 23:
            raise IntervalParseError("hour intervals must be 1-23 (use days instead)")
        interval_s = value * 3600
    else:  # unit == "d"
        if value > 28:
            raise IntervalParseError("day intervals must be 1-28 (use a cron expression)")
        interval_s = value * 86400
    return interval_s, interval_to_cron(interval_s), period_label(interval_s), None


def human_schedule_for_cron(cron: str) -> str:
    """Label for a cron expression: friendly for interval-derived exprs."""
    m = re.match(r"^\*/(\d+) \* \* \* \*$", cron)
    if m:
        return f"every {m.group(1)}m"
    m = re.match(r"^0 \*/(\d+) \* \* \*$", cron)
    if m:
        return f"every {m.group(1)}h"
    m = re.match(r"^0 0 \*/(\d+) \* \*$", cron)
    if m:
        return f"every {m.group(1)}d"
    m = re.match(r"^(\d{1,2}) (\d{1,2}) \* \* \*$", cron)
    if m:
        # Daily wall-clock time: `MM HH * * *` → "daily at HH:MM".
        return f"daily at {int(m.group(2)):02d}:{int(m.group(1)):02d}"
    return f"cron: {cron}"


def parse_time_of_day(text: str) -> tuple[str, str]:
    """Parse a daily wall-clock time ``HH:MM`` (24-hour) into a daily cron
    expression and its human label.

    Returns ``(cron, label)`` — e.g. ``("30 9 * * *", "daily at 09:30")``.
    """
    text = (text or "").strip()
    m = re.fullmatch(r"(\d{1,2}):(\d{1,2})", text)
    if not m:
        raise IntervalParseError("invalid time (use 24-hour HH:MM, e.g. 09:30)")
    hour, minute = int(m.group(1)), int(m.group(2))
    if hour > 23 or minute > 59:
        raise IntervalParseError("invalid time (use 24-hour HH:MM, e.g. 09:30)")
    cron = f"{minute} {hour} * * *"
    return cron, human_schedule_for_cron(cron)


def cron_disabled() -> bool:
    """Kill-switch: ``SHAREDAGENTS_DISABLE_CRON=1`` disables command + tools."""
    return os.environ.get(DISABLE_CRON_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def cron_disabled_message() -> str:
    """Canonical kill-switch refusal text (single source of truth)."""
    return f"scheduled work is disabled ({DISABLE_CRON_ENV}=1)"


# ---------------------------------------------------------------------------
# loop.md override
# ---------------------------------------------------------------------------


def load_loop_md(cwd: Path | None = None) -> str | None:
    """Resolve the loop.md override: project-local then config dir.

    Lookup order mirrors the reference behavior adapted to sharedagents's
    config conventions: ``<cwd>/.sharedagents/loop.md`` then
    ``<sharedagents_root()>/loop.md``. Empty/whitespace files are ignored;
    contents are truncated at 25,000 bytes with a warning line.
    """
    candidates: list[Path] = []
    if cwd:
        candidates.append(Path(cwd) / ".sharedagents" / "loop.md")
    candidates.append(sharedagents_root() / "loop.md")
    for path in candidates:
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not raw.strip():
            continue
        data = raw.encode("utf-8")
        if len(data) > LOOP_MD_MAX_BYTES:
            data = data[:LOOP_MD_MAX_BYTES]
            raw = (
                data.decode("utf-8", errors="replace") + f"\n\nWARNING: loop.md was truncated at "
                f"{LOOP_MD_MAX_BYTES} bytes — the tail is missing.\n"
            )
        return raw
    return None


def build_loop_prompt(
    loop: LoopRecord,
    *,
    loop_md: str | None,
    first_fire: bool,
) -> str:
    """Assemble the prompt fired into the session for one loop run.

    The stored prompt_text is the body; loop.md is injected in full on
    the first fire (or after a content change) and as a short reminder
    afterwards. All text here is original sharedagents phrasing.
    """
    head = "[scheduled loop · auto-run · your instructions follow]\n" if first_fire else "[scheduled loop · auto-run]\n"
    body = str(loop.prompt_text or "").strip()
    if loop_md:
        if first_fire:
            body += f"\n\n--- loop.md (full) ---\n{loop_md}"
        else:
            body += (
                "\n\n--- loop.md reminder ---\nloop.md is available in context; re-read it if the situation changed.\n"
            )
    return f"{head}{body}"


def parse_fresh_session_each_run(raw: Any) -> bool:
    """Opt-in only: mint a new session per tick iff the operator said so.

    ``bool("false")`` is True in Python, so JSON/form leftovers such as
    ``"false"``, ``"0"``, ``[]``, or any other truthy junk must not enable
    minting. Only an explicit yes (``True``, ``1``, ``"true"``/``"yes"``/
    ``"on"``/``"1"``) turns the flag on.
    """
    if raw is True:
        return True
    if raw is False or raw is None:
        return False
    if isinstance(raw, bool):
        return False
    if isinstance(raw, (int, float)):
        return raw == 1
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return False


def parse_loop_budget_usd(raw: Any) -> float | None:
    """Parse an optional per-run USD budget. Empty/None means unset."""
    if raw is None or raw is False:
        return None
    if isinstance(raw, str) and not raw.strip():
        return None
    if isinstance(raw, bool):
        raise ValueError("budget_usd must be a number")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("budget_usd must be a number") from exc
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError("budget_usd must be a finite number")
    if value <= 0:
        return None
    if value > MAX_LOOP_BUDGET_USD:
        raise ValueError(f"budget_usd must be ≤ {MAX_LOOP_BUDGET_USD:g}")
    return value


def _infer_origin_from_rebound(
    rebound_ids: list[str],
    *,
    sessions_dir: Any = None,
) -> str:
    """Authoritative origin = nearest parent of the oldest rebound.

    Empty when that parent cannot be read (missing row / db / error).
    Never walks to the root and never persists a guess.
    """
    if not rebound_ids:
        return ""
    oldest = str(rebound_ids[0] or "").strip()
    if not oldest:
        return ""
    try:
        from sharedagents_session.session_index import ancestor_ids

        chain = ancestor_ids(oldest, sessions_dir=sessions_dir)
    except Exception:  # noqa: BLE001 — unverified; do not guess
        return ""
    if chain is None or not chain:
        return ""
    return str(chain[0] or "").strip()


def resolve_loop_origin(
    rec: LoopRecord,
    fallback_sid: str,
    *,
    sessions_dir: Any = None,
    registry: Any = None,
) -> str:
    """Immutable session the loop was first bound to.

    Fresh-session ticks parent here so they stay siblings under one root
    instead of a linked list that evades the 3-level delegation cap.

    For a legacy loop with a blank ``origin_session_id``, the origin is
    the nearest parent of the oldest ``session_rebound`` (the first bound
    session). If that parent cannot be read authoritatively, this returns
    ``""`` and does not persist a guess.
    """
    stored = str(getattr(rec, "origin_session_id", "") or "").strip()
    if stored:
        return stored
    reg = registry if registry is not None else _registry()
    try:
        rebound = list(reg.rebound_session_ids(rec.loop_id))
    except Exception:  # noqa: BLE001 — unverified; do not guess or persist
        return ""
    inferred = _infer_origin_from_rebound(rebound, sessions_dir=sessions_dir)
    if not inferred and not rebound:
        # Successful empty history — current bound session is first-bound.
        inferred = str(getattr(rec, "session_id", "") or fallback_sid or "").strip()
    if not inferred:
        return ""
    try:
        persisted = reg.ensure_origin_session_id(rec.loop_id, inferred)
        if persisted:
            return persisted
    except Exception:  # noqa: BLE001 — do not invent a stored origin
        pass
    return inferred


def flatten_fresh_session_parent_chains(
    *,
    registry: Any = None,
    sessions_dir: Any = None,
) -> dict[str, int]:
    """Rewrite loop-minted parents so every rebound is a sibling of the origin.

    Idempotent. Uses ``session_rebound`` events as the minted-session set so
    a real ``Agent`` child of a tick is left alone. Returns
    ``{loops, rewritten}``.
    """
    from sharedagents_session.session_index import ancestor_ids, set_session_parent

    reg = registry if registry is not None else _registry()
    loops = list(reg.list_loops())
    rewritten = 0
    touched_loops = 0
    for rec in loops:
        rebound = list(reg.rebound_session_ids(rec.loop_id))
        if not rebound:
            stored = str(getattr(rec, "origin_session_id", "") or "").strip()
            if not stored:
                current = str(rec.session_id or "").strip()
                if current:
                    reg.ensure_origin_session_id(rec.loop_id, current)
            continue
        origin = resolve_loop_origin(
            rec,
            rebound[0],
            sessions_dir=sessions_dir,
            registry=reg,
        )
        if not origin:
            continue
        chain = {origin, *rebound, str(rec.session_id or "")}
        loop_rewritten = 0
        for sid in rebound:
            if not sid or sid == origin:
                continue
            parents = ancestor_ids(sid, sessions_dir=sessions_dir)
            if parents is None:
                continue
            current_parent = parents[0] if parents else ""
            if current_parent == origin:
                continue
            if not current_parent:
                # Already an independent root session (post-2026-09-06
                # "new session each run" mints without a parent) — never
                # re-parent it under the origin.
                continue
            if current_parent not in chain:
                continue
            if set_session_parent(
                sid,
                origin,
                sessions_dir=sessions_dir,
                only_if_parent=current_parent,
            ):
                loop_rewritten += 1
        if loop_rewritten:
            touched_loops += 1
            rewritten += loop_rewritten
    return {"loops": touched_loops, "rewritten": rewritten}


def _bound_session_model(
    agent: Any,
    sid: str,
    *,
    sessions_dir: Any = None,
) -> str:
    """Model stamped on the loop's bound session (``""`` when unknown).

    A per-run mint is a NEW conversation continuing the loop's bound one,
    so it must inherit the model the operator picked on that session
    (``/model`` / the chat picker). Without inheritance every fresh run
    silently falls back to the agent's configured default
    ``upstream_model`` — the "I picked qwen, runs go to deepseek" gap.
    CLI-engine labels ("claude"/"codex"/"antigravity") are engine names,
    not model ids, and never propagate (mirrors
    ``delegate_shims._CLI_ENGINE_LABELS``).
    """
    model = ""
    get_session = getattr(agent, "get_session", None)
    if callable(get_session):
        try:
            sess = get_session(sid)
        except Exception:  # noqa: BLE001 — mint must never fail on a header read
            sess = None
        if sess is not None:
            model = str(getattr(sess, "model", "") or "").strip()
    if not model:
        # A rebound session exists in sessions.db even before its JSONL
        # does — read the header row directly (the same helper the
        # delegation model-inheritance path uses).
        try:
            from sharedagents_agent.agents.delegate_shims import (
                _parent_session_model_and_agent,
            )

            model, _name = _parent_session_model_and_agent(sessions_dir, sid)
            model = str(model or "").strip()
        except Exception:  # noqa: BLE001 — the agent default is the fallback
            model = ""
    if model:
        try:
            from sharedagents_agent.agents.delegate_shims import _CLI_ENGINE_LABELS

            if model.lower() in {v.lower() for v in _CLI_ENGINE_LABELS.values()}:
                return ""
        except Exception:  # noqa: BLE001
            pass
        if model.lower() in {"auto", "ask", "plan"}:
            return ""
        models = getattr(getattr(agent, "cfg", None), "models", None)
        if models:
            hit = False
            for item in models:
                md = dict(item) if isinstance(item, dict) else {}
                if model in (
                    str(md.get("alias") or ""),
                    str(md.get("id") or ""),
                ):
                    hit = True
                    break
            if not hit:
                return ""
    return model


def prepare_loop_fire(agent: Any, rec: LoopRecord) -> tuple[str, str]:
    """Mint a fresh session and/or restamp per-run budget before a fire.

    Returns ``(session_id, error)``. An empty error means the fire session
    is ready. Existing loops with both options unset are a no-op.

    A minted tick is an INDEPENDENT session: no ``parent_session_id`` at
    all. Parenting ticks (to the previous tick or to the loop origin)
    made the agent monitor render every run as a "subagent" child —
    "new session each run" must mint a standalone conversation instead.
    Run lineage lives in the loop registry (``session_rebound`` events +
    ``origin_session_id``), and the tick inherits the model stamped on
    the bound session so runs follow ``/model`` picks.
    """
    sid = str(getattr(rec, "session_id", "") or "")
    if not sid:
        return "", "missing_session"
    mint = parse_fresh_session_each_run(
        getattr(rec, "fresh_session_each_run", False),
    )
    if mint and int(getattr(rec, "run_count", 0) or 0) > 0:
        create = getattr(agent, "create_session", None)
        if not callable(create):
            return sid, "fresh_session_unavailable"
        sessions_dir = getattr(agent, "sessions_dir", None)
        origin = resolve_loop_origin(rec, sid, sessions_dir=sessions_dir)
        from sharedagents_agent.agents.session_lifecycle import loop_mint_context

        with loop_mint_context(origin or sid) as mint:
            bound_model = _bound_session_model(
                agent,
                sid,
                sessions_dir=sessions_dir,
            )
            model_kwargs: dict[str, str] = {"model": bound_model} if bound_model else {}
            try:
                new = create(**model_kwargs)
            except TypeError:
                try:
                    new = create(workspace_cwd="", **model_kwargs)
                except TypeError:
                    # A create_session without the model keyword — keep the
                    # agent default model (pre-inheritance behavior).
                    try:
                        new = create()
                    except TypeError:
                        new = create(workspace_cwd="")
            except Exception as exc:  # noqa: BLE001
                return sid, f"fresh_session_failed:{exc}"
            new_sid = str(getattr(new, "hash", None) or new or "").strip()
            if not new_sid:
                return sid, "fresh_session_failed:empty_id"
            mint.register_tick(new_sid)
            rebound = _registry().rebind_session(rec.loop_id, new_sid)
            sid = str(rebound.session_id) if rebound is not None else new_sid

    budget = getattr(rec, "budget_usd", None)
    if budget is None:
        return sid, ""
    try:
        usd = float(budget)
    except (TypeError, ValueError):
        return sid, "invalid_loop_budget"
    if usd <= 0:
        return sid, ""
    get_session = getattr(agent, "get_session", None)
    sess = get_session(sid) if callable(get_session) else None
    if sess is None:
        return sid, "budget_session_not_found"
    try:
        from sharedagents_goal.goal_service import (
            session_budget_snapshot,
            set_session_budget,
        )

        spent = float(session_budget_snapshot(sess).get("spent_usd") or 0.0)
        cap = spent + usd
        result = set_session_budget(agent, sess, cap)
    except Exception as exc:  # noqa: BLE001
        return sid, f"budget_stamp_failed:{exc}"
    if not getattr(result, "ok", False):
        return sid, str(getattr(result, "message", "") or "budget_stamp_failed")
    return sid, ""


# ---------------------------------------------------------------------------
# Loop lifecycle
# ---------------------------------------------------------------------------


def _registry() -> LoopRegistry:
    return get_loop_registry()


def create_loop_for_session(
    agent: Any,
    sess: Any,
    *,
    prompt_text: str,
    interval_s: int | None = None,
    cron: str | None = None,
    created_by: str | None = None,
    actor_email: str = "",
    fire_immediately: bool = False,
    fresh_session_each_run: bool = False,
    budget_usd: float | None = None,
) -> LoopResult:
    """Create a loop bound to a session and kick the local supervisor.

    The supervise kick only happens when already on an event loop —
    WSGI threads must not churn the supervisor lease (mirrors the goals
    event-loop guard).

    ``fire_immediately`` is the web-create contract: the
    first run is anchored at the creation instant so the operator sees it
    start right away. The ``/loop`` command keeps the interval + jitter
    cadence (default False).
    """
    if cron_disabled():
        return LoopResult(False, cron_disabled_message())
    session_id = getattr(sess, "hash", None) or str(sess)
    agent_name = getattr(agent, "name", "main")
    human = (
        human_schedule_for_cron(cron) if cron else period_label(interval_s) if interval_s is not None else "model-paced"
    )
    try:
        loop = _registry().create_loop(
            session_id=session_id,
            agent_name=agent_name,
            prompt_text=prompt_text,
            interval_s=interval_s,
            cron=cron,
            human_schedule=human,
            created_by=created_by,
            actor_email=actor_email,
            fire_immediately=fire_immediately,
            fresh_session_each_run=parse_fresh_session_each_run(
                fresh_session_each_run,
            ),
            budget_usd=budget_usd,
        )
    except LookupError as exc:
        return LoopResult(False, str(exc))
    except ValueError as exc:
        return LoopResult(False, str(exc))
    schedule_label = period_label(interval_s) if interval_s else "model-paced"
    message = f"loop scheduled {schedule_label} (no automatic expiry) — id {loop.loop_id}"
    if cron:
        message = f"loop scheduled {human} (no automatic expiry) — id {loop.loop_id}"
    result = LoopResult(True, message, loop)
    supervisor = getattr(agent, "loop_supervisor", None)
    if supervisor is not None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return result
        try:
            asyncio.ensure_future(supervisor.supervise_loop(loop.loop_id, source="service"))
        except Exception:
            pass
    return result


def pause_loop(agent: Any, loop_id: str) -> TransitionResult:
    result = _registry().pause_loop(loop_id, reason="user")
    _cancel_supervisor(agent, loop_id)
    return result


def resume_loop(agent: Any, loop_id: str) -> TransitionResult:
    if cron_disabled():
        return TransitionResult(
            False,
            cron_disabled_message(),
            _registry().get(loop_id),
        )
    result = _registry().resume_loop(loop_id)
    if result.ok:
        _kick_supervisor(agent, loop_id)
    return result


def archive_loop(agent: Any, loop_id: str) -> TransitionResult:
    result = _registry().archive_loop(loop_id, reason="user")
    _cancel_supervisor(agent, loop_id)
    return result


def delete_loop(agent: Any, loop_id: str) -> TransitionResult:
    """Permanently delete a loop (record + run/event history + wakeups).

    2026-09-07 (owner contract): the terminal action is DELETE, never
    archive — ``/loop stop``/``/loop delete`` and the board's Delete button
    all land here. The supervisor drive task is cancelled; if it is
    mid-cycle it exits at its next checkpoint when the row is gone.
    """
    if not _registry().delete_loop(loop_id):
        return TransitionResult(False, f"loop not found: {loop_id}")
    _cancel_supervisor(agent, loop_id)
    return TransitionResult(True, "deleted")


def _kick_supervisor(agent: Any, loop_id: str) -> None:
    supervisor = getattr(agent, "loop_supervisor", None)
    if supervisor is None:
        return
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    try:
        asyncio.ensure_future(supervisor.supervise_loop(loop_id, source="service"))
    except Exception:
        pass


def _cancel_supervisor(agent: Any, loop_id: str) -> None:
    supervisor = getattr(agent, "loop_supervisor", None)
    if supervisor is None:
        return
    try:
        # The registry transition already happened — only tear down the
        # drive task + lease, don't double-pause.
        supervisor.cancel(loop_id, pause=False)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Command dispatch (/loop)
# ---------------------------------------------------------------------------

_USAGE = (
    "usage: /loop [interval] <prompt>  — e.g. /loop 5m check the deploy, "
    "/loop every 30m check the deploy; "
    "/loop list | /loop pause [id] | /loop resume [id] | /loop delete [id] "
    "(stop/archive/clear/… are aliases of delete)"
)

#: ``stop`` verb family — mirrors the /goal clear aliases
#: (clear|stop|reset|off|none|done, goal_service.parse_goal_subcommand) so
#: a user who types any of them gets the same terminal action. The terminal
#: action is DELETE (2026-09-07 owner contract): the record, its run/event
#: history and pending wakeups are removed — never archived.
_LOOP_STOP_ALIASES = ("stop", "clear", "reset", "off", "none", "done")

# Shared with the WebUI, which stamps creators only on creation commands.
LOOP_CONTROL_SUBCOMMANDS = frozenset(("list", "pause", "resume", "delete", "archive", *_LOOP_STOP_ALIASES))


def _most_recent_loop_for_session(
    session_id: str,
    *,
    status: str | None = None,
) -> LoopRecord | None:
    loops = _registry().list_loops(
        LoopFilters(session_id=session_id, status=status),
    )
    return loops[0] if loops else None


async def execute_loop_command(agent: Any, sess: Any, arg: str) -> dict[str, Any]:
    """Single shared entry point for the ``/loop`` chat command.

    Returns ``{"ok", "message", "loop", "note", "loops"}``; ``loops`` is
    populated by the ``list`` subcommand.
    """
    if cron_disabled():
        return {
            "ok": False,
            "message": (f"scheduled work is disabled ({DISABLE_CRON_ENV}=1) — unset it to use /loop"),
            "loop": None,
            "note": None,
            "loops": [],
        }
    session_id = getattr(sess, "hash", None) or str(sess)
    text = (arg or "").strip()
    if not text:
        return {
            "ok": False,
            "message": _USAGE,
            "loop": None,
            "note": None,
            "loops": [],
        }
    tokens = text.split(None, 1)
    head = tokens[0].lower()

    # Subcommands (list / pause / resume / archive / stop).
    if head == "list":
        loops = _registry().list_loops(LoopFilters(session_id=session_id))
        return {
            "ok": True,
            "message": f"{len(loops)} loop(s) on this session",
            "loop": None,
            "note": None,
            "loops": [item.to_dict() for item in loops],
        }
    if head in LOOP_CONTROL_SUBCOMMANDS:
        target = tokens[1].strip() if len(tokens) > 1 else ""
        loop = None
        loop = _registry().get(target) if target else _most_recent_loop_for_session(session_id)
        if loop is None:
            return {
                "ok": False,
                "message": f"no loop found to {head}" + (f": {target}" if target else " on this session"),
                "loop": None,
                "note": None,
                "loops": [],
            }
        if loop.session_id != session_id:
            return {
                "ok": False,
                "message": f"loop not found for this session: {loop.loop_id}",
                "loop": None,
                "note": None,
                "loops": [],
            }
        if head in _LOOP_STOP_ALIASES or head in ("delete", "archive"):
            # Terminal verb family is DELETE: the loop record,
            # its run/event history and pending wakeups are removed — never
            # archived. Same lifecycle as the board's Delete button.
            result = delete_loop(agent, loop.loop_id)
            message = "deleted" if result.ok else result.message
        elif head == "pause":
            result = pause_loop(agent, loop.loop_id)
            message = result.message
        elif head == "resume":
            result = resume_loop(agent, loop.loop_id)
            message = result.message
        else:
            result = delete_loop(agent, loop.loop_id)
            message = result.message
        return {
            "ok": result.ok,
            "message": message,
            "loop": result.loop.to_dict() if result.loop else None,
            "note": None,
            "loops": [],
        }

    # Creation: leading interval token, trailing "every N unit" clause,
    # then default 10m. The rest is the prompt body.
    prompt = text
    interval_s: int | None = None
    cron: str | None = None
    note: str | None = None
    first_token = tokens[0]
    if _INTERVAL_RE.match(first_token) and len(tokens) > 1:
        try:
            interval_s, cron, _label, note = parse_interval(first_token)
        except IntervalParseError as exc:
            return {
                "ok": False,
                "message": str(exc),
                "loop": None,
                "note": None,
                "loops": [],
            }
        prompt = tokens[1]
    else:
        every = _EVERY_CLAUSE_RE.search(text)
        if every:
            unit = _EVERY_UNIT_MAP.get((every.group(2) or "m").lower(), "m")
            try:
                interval_s, cron, _label, note = parse_interval(f"{every.group(1)}{unit}")
            except IntervalParseError as exc:
                return {
                    "ok": False,
                    "message": str(exc),
                    "loop": None,
                    "note": None,
                    "loops": [],
                }
            prompt = text[: every.start()].rstrip() + " " + text[every.end() :].strip()
            prompt = prompt.strip()
        else:
            interval_s = DEFAULT_INTERVAL_S

    if not prompt:
        return {
            "ok": False,
            "message": _USAGE,
            "loop": None,
            "note": None,
            "loops": [],
        }
    result = create_loop_for_session(
        agent,
        sess,
        prompt_text=prompt,
        interval_s=interval_s,
        cron=cron,
        created_by="chat",
    )
    return {
        "ok": result.ok,
        "message": result.message + (f" ({note})" if note else ""),
        "loop": result.loop.to_dict() if result.loop else None,
        "note": note,
        "loops": [],
    }


# ---------------------------------------------------------------------------
# Gateway-side handlers (web + control endpoint)
# ---------------------------------------------------------------------------


def run_loop_action(agent: Any, loop_id: str, action: str) -> dict[str, Any]:
    """Handle a web/control endpoint loop action (pause/resume/archive)."""
    loop = _registry().get(loop_id)
    if loop is None:
        return {"ok": False, "message": f"loop not found: {loop_id}"}
    agent_name = str(getattr(agent, "name", "") or "")
    if agent_name and loop.agent_name != agent_name:
        return {
            "ok": False,
            "message": f"loop is owned by agent {loop.agent_name}",
            "loop": loop.to_dict(),
        }
    if action == "pause":
        result = pause_loop(agent, loop_id)
    elif action == "resume":
        result = resume_loop(agent, loop_id)
    elif action == "archive":
        result = archive_loop(agent, loop_id)
    elif action == "delete":
        result = delete_loop(agent, loop_id)
    else:
        return {"ok": False, "message": f"unknown loop action: {action}"}
    return {
        "ok": result.ok,
        "message": result.message,
        "loop": result.loop.to_dict() if result.loop else None,
    }


def stop_loops_for_session(
    agent: Any,
    session_id: str,
    *,
    reason: str = "aborted",
) -> int:
    """Halt every loop rooted on a session and cancel its wakeups.

    Never raises (mirrors ``stop_goals_for_session``). Called from the
    cancel paths so an operator stop ends the loop, not just the
    in-flight turn.
    """
    stopped = 0
    try:
        for loop in _registry().list_loops(LoopFilters(session_id=session_id, status=STATUS_RUNNING)):
            _registry().pause_loop(loop.loop_id, reason=reason)
            _cancel_supervisor(agent, loop.loop_id)
            stopped += 1
    except Exception:
        pass
    try:
        _registry().cancel_wakeups_for_session(session_id)
    except Exception:
        pass
    # Cron jobs bound to the session stop too: paused (not deleted), so they
    # neither fire nor re-arm at boot until the user prompts the session
    # again (``resume_jobs_for_session``).
    try:
        for job_id in _registry().pause_jobs_for_session(
            session_id,
            reason=reason,
        ):
            _cancel_supervisor(agent, f"job:{job_id}")
    except Exception:
        pass
    return stopped


def resume_jobs_for_session(agent: Any, session_id: str) -> list[str]:
    """Re-arm the cron jobs a Stop paused on ``session_id``.

    Called when the user prompts the session again. Never raises.
    """
    if not session_id:
        return []
    try:
        resumed = _registry().resume_jobs_for_session(session_id)
    except Exception:
        return []
    for job_id in resumed:
        _kick_supervisor(agent, f"job:{job_id}")
    return resumed


# ---------------------------------------------------------------------------
# Cron-tool family (CronCreate / CronDelete / CronList / ScheduleWakeup)
# ---------------------------------------------------------------------------


def cron_create(
    agent: Any,
    session_id: str,
    *,
    cron: str,
    prompt: str,
    recurring: bool = True,
    durable: bool = False,
) -> dict[str, Any]:
    """``CronCreate`` backend: durable schedule that fires ``prompt``."""
    if cron_disabled():
        return {
            "ok": False,
            "message": cron_disabled_message(),
        }
    try:
        job = _registry().create_job(
            session_id=session_id,
            agent_name=getattr(agent, "name", "main"),
            cron=cron,
            human_schedule=human_schedule_for_cron(cron),
            prompt=prompt,
            recurring=recurring,
            durable=durable,
            created_by="model",
        )
    except LookupError as exc:
        return {"ok": False, "message": str(exc)}
    except ValueError as exc:
        return {"ok": False, "message": str(exc)}
    _kick_supervisor(agent, f"job:{job['job_id']}")
    return {"ok": True, "job": job}


def cron_delete(
    agent: Any,
    job_id: str,
    *,
    session_id: str | None,
) -> dict[str, Any]:
    """``CronDelete`` backend."""
    if cron_disabled():
        return {
            "ok": False,
            "message": cron_disabled_message(),
        }
    job = _registry().get_job(job_id)
    if not session_id:
        return {"ok": False, "message": "session_id is required"}
    if job is None or str(job.get("session_id") or "") != session_id:
        return {
            "ok": False,
            "message": f"job not found for this session: {job_id}",
        }
    agent_name = str(getattr(agent, "name", "") or "")
    if agent_name and str(job.get("agent_name") or "") != agent_name:
        return {
            "ok": False,
            "message": f"job not found for this agent: {job_id}",
        }
    supervisor = getattr(agent, "loop_supervisor", None)
    if supervisor is not None:
        try:
            supervisor.cancel(f"job:{job_id}")
        except Exception:
            pass
    deleted = _registry().delete_job(job_id, reason="user")
    return {
        "ok": deleted,
        "message": "job deleted" if deleted else f"job not found: {job_id}",
    }


def cron_list(agent: Any, session_id: str | None = None) -> dict[str, Any]:
    """``CronList`` backend."""
    if cron_disabled():
        return {
            "ok": False,
            "message": cron_disabled_message(),
            "jobs": [],
        }
    jobs = _registry().list_jobs()
    if session_id:
        jobs = [j for j in jobs if j["session_id"] == session_id]
    return {"ok": True, "jobs": jobs}


def schedule_wakeup(
    agent: Any,
    session_id: str,
    *,
    delay_seconds: int,
    stop: bool = False,
) -> dict[str, Any]:
    """``ScheduleWakeup`` backend.

    Delays are clamped to 60–3600 s and rounded up to the next whole
    minute. ``stop=True`` cancels every pending wakeup for the session
    (a recurring cron/loop is NOT cancelled — reference behavior).
    """
    if cron_disabled():
        return {
            "ok": False,
            "message": cron_disabled_message(),
            "scheduledFor": None,
            "clampedDelaySeconds": None,
            "wasClamped": False,
            "stopped": False,
            "cancelledWakeups": 0,
        }
    if stop:
        cancelled = _registry().cancel_wakeups_for_session(session_id)
        return {
            "ok": True,
            "scheduledFor": None,
            "clampedDelaySeconds": 0,
            "wasClamped": False,
            "stopped": True,
            "cancelledWakeups": cancelled,
        }
    try:
        raw = int(delay_seconds)
    except (TypeError, ValueError, OverflowError):
        raw = _WAKEUP_MIN_S
    was_clamped = raw < _WAKEUP_MIN_S or raw > _WAKEUP_MAX_S
    clamped = max(_WAKEUP_MIN_S, min(_WAKEUP_MAX_S, raw))
    # Round UP to the next whole minute.
    rounded = ((clamped + 59) // 60) * 60
    import datetime as dt

    scheduled_for = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=rounded)).isoformat(timespec="seconds")
    loop_id = None
    # Bind to the session's most recent loop, if any — the wakeup then
    # re-arms that loop (model-paced mode) when it is still running.
    recent = _most_recent_loop_for_session(
        session_id,
        status=STATUS_RUNNING,
    )
    if recent is not None:
        loop_id = recent.loop_id
        _registry().cancel_wakeups_for_loop(loop_id)
        _registry().rearm_from_wakeup(loop_id, scheduled_for)
    _registry().add_wakeup(
        session_id=session_id,
        agent_name=getattr(agent, "name", "main"),
        scheduled_for=scheduled_for,
        loop_id=loop_id,
    )
    return {
        "ok": True,
        "scheduledFor": scheduled_for,
        "clampedDelaySeconds": rounded,
        "wasClamped": was_clamped,
        "stopped": False,
        "cancelledWakeups": 0,
    }
