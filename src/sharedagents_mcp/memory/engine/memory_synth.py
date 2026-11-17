"""Background memory synthesizer.

Owns MEMORY.md. The agent never writes to it directly — it writes raw
observations to `memory/YYYY-MM-DD.md` via the `remember` tool, and
THIS module periodically reads those dailies + the current MEMORY.md
and rewrites MEMORY.md with the distilled general principles.

Trigger: subscribes to `daily_appended` events on the process-wide
event bus, debounced by `quiet_seconds` (default 60s) — a burst of
remember calls coalesces into one synthesizer run.

The synthesizer is per-agent (each agent owns its own memory home), so
the subscriber is registered ONCE per agent at startup with the
agent-specific home/upstream pinned in a closure.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx

from sharedagents_core.kernel.event_bus import get_bus, register_reset_hook
from sharedagents_core.kernel.prompt_pair import load_prompt_pair
from sharedagents_model.sampling import apply_sampling, configured_sampling, sampling_row

_log = logging.getLogger(__name__)


# Prompt body lives in `<home>/MEMORY-SYNTH.md` (operator-editable),
# loaded fresh on every synth run via ``load_prompt_pair``. The file
# uses two ``<!-- system -->`` / ``<!-- user -->`` sections; the user
# section may contain ``{memory}``, ``{dailies}``, ``{n_dailies}``
# placeholders that this module fills in. If the file is missing or
# either section is empty, we skip synthesis rather than fall back to
# a hardcoded prompt — the operator has explicit control.


# Most recent N dailies fed into one synthesis run. Older dailies have
# usually had their durable parts already lifted; rescanning them every
# tick is wasted tokens. Tunable per agent if needed.
_DEFAULT_DAILIES_WINDOW: int = 7

# Per-daily input cap so a runaway agent that floods one day doesn't
# blow the synth prompt.
_PER_DAILY_CHAR_CAP: int = 32_000


def _read_recent_dailies(home: Path, *, n: int) -> list[tuple[str, str]]:
    """Return the most recent N daily files as (basename, body) pairs,
    newest first. Capped per-file at `_PER_DAILY_CHAR_CAP`.
    """
    d = home / "memory"
    if not d.is_dir():
        return []
    files = sorted(
        (p for p in d.glob("*.md") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:n]
    out: list[tuple[str, str]] = []
    for p in files:
        try:
            body = p.read_text(encoding="utf-8")
        except OSError:
            continue
        if len(body) > _PER_DAILY_CHAR_CAP:
            body = body[:_PER_DAILY_CHAR_CAP] + f"\n\n[... truncated at {_PER_DAILY_CHAR_CAP} chars ...]"
        out.append((p.name, body))
    return out


async def synth_memory(
    home: Path,
    *,
    upstream_endpoint: str,
    upstream_model: str,
    n_dailies: int = _DEFAULT_DAILIES_WINDOW,
    timeout_s: float = 60.0,
    force: bool = False,
    cfg: Any = None,
) -> dict[str, Any]:
    """Read MEMORY.md + recent dailies, run the synthesizer, rewrite
    MEMORY.md atomically. Returns a stats dict.

    ``force=True`` bypasses the "no daily newer than MEMORY.md" early
    return so an operator (or ``sharedagents memory synth --force``) can
    re-run synthesis without first touching a daily file.

    Failure modes:
      - LLM unreachable / non-200 → return {"ok": False, "error": ...},
        don't touch MEMORY.md.
      - Output doesn't contain the required section markers → reject,
        don't touch MEMORY.md.
      - Output identical to input → return {"changed": False}.
    """
    started = time.monotonic()
    memory_path = home / "MEMORY.md"
    memory_body = memory_path.read_text(encoding="utf-8") if memory_path.is_file() else ""
    memory_mtime = memory_path.stat().st_mtime if memory_path.is_file() else 0.0

    # Early return: if no daily is newer than MEMORY.md, nothing to
    # synthesize. This avoids an LLM call on every heartbeat tick when
    # no new content has landed. The boot-time sweep in agent_runtime
    # handles the case where dailies were written while the agent was
    # down. ``force=True`` bypasses this guard.
    d = home / "memory"
    if not force and d.is_dir() and memory_mtime > 0:
        newest_daily = max(
            (p.stat().st_mtime for p in d.glob("*.md") if p.is_file()),
            default=0.0,
        )
        if newest_daily <= memory_mtime:
            return {
                "ok": True,
                "changed": False,
                "reason": "no daily newer than MEMORY.md",
                "duration_s": round(time.monotonic() - started, 3),
            }
    elif not d.is_dir():
        return {
            "ok": True,
            "changed": False,
            "reason": "no daily logs to synthesize from",
            "duration_s": round(time.monotonic() - started, 3),
        }

    dailies = _read_recent_dailies(home, n=n_dailies)
    if not dailies:
        return {
            "ok": True,
            "changed": False,
            "reason": "no daily logs to synthesize from",
            "duration_s": round(time.monotonic() - started, 3),
        }

    dailies_block = "\n\n".join(f"## {name}\n\n{body}" for name, body in dailies)

    # Prompt: agent-home override, else shipped runtime template.
    override = home / "MEMORY-SYNTH.md"
    if override.is_file():
        prompt_pair = load_prompt_pair(override)
    else:
        from sharedagents_core.kernel.config import memory_synth_template_text
        from sharedagents_core.kernel.prompt_pair import parse_prompt_pair_text

        prompt_pair = parse_prompt_pair_text(memory_synth_template_text())
    system_msg = (prompt_pair.get("system") or "").strip()
    user_template = (prompt_pair.get("user") or "").strip()
    if not system_msg or not user_template:
        return {
            "ok": False,
            "error": "MEMORY-SYNTH.md missing or has no system/user sections",
            "duration_s": round(time.monotonic() - started, 3),
        }
    try:
        user_msg = user_template.format(
            memory=memory_body or "_(empty — fresh agent)_",
            dailies=dailies_block,
            n_dailies=len(dailies),
        )
    except (KeyError, IndexError) as e:
        return {
            "ok": False,
            "error": f"MEMORY-SYNTH.md template missing placeholder: {e}",
            "duration_s": round(time.monotonic() - started, 3),
        }

    payload = {
        "model": upstream_model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 4096,
        # Disable thinking for Qwen3-style reasoning models during synthesis.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if cfg is None:
        from sharedagents_core.kernel.config import load_global_config

        cfg = load_global_config()
    apply_sampling(
        payload,
        configured_sampling(
            cfg,
            sampling_row(cfg, upstream_endpoint, upstream_model),
            upstream_model,
            purpose="memory_synth",
        ),
    )
    url = upstream_endpoint.rstrip("/") + "/chat/completions"

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as c:
            r = await c.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"http: {type(e).__name__}: {e}",
            "duration_s": round(time.monotonic() - started, 3),
        }

    out_text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    out_text = out_text.strip()
    if not out_text:
        return {
            "ok": False,
            "error": "empty synth output",
            "duration_s": round(time.monotonic() - started, 3),
        }
    if "<!-- sharedagents:section" not in out_text:
        return {
            "ok": False,
            "error": "synth output missing section markers; refusing to write",
            "duration_s": round(time.monotonic() - started, 3),
            "preview": out_text[:300],
        }
    if not out_text.endswith("\n"):
        out_text += "\n"

    if out_text == memory_body:
        return {
            "ok": True,
            "changed": False,
            "duration_s": round(time.monotonic() - started, 3),
            "n_dailies": len(dailies),
        }

    # Atomic-rename write.
    tmp = memory_path.with_name(f".{memory_path.name}.tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        f.write(out_text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, memory_path)

    return {
        "ok": True,
        "changed": True,
        "duration_s": round(time.monotonic() - started, 3),
        "n_dailies": len(dailies),
        "before_chars": len(memory_body),
        "after_chars": len(out_text),
    }


# ---- event-bus subscriber ------------------------------------------

# Per-agent registration state so subscribe_for_agent() is idempotent
# across multiple calls. Keyed by agent home path.
_REGISTERED: set[str] = set()

# Inverted dependency: the kernel event bus no longer imports this
# module to clear ``_REGISTERED`` in ``reset_bus_for_tests``. Instead we hand it
# a reset callback here, at import time, so the bus stays a leaf and knows
# nothing about the memory engine.
register_reset_hook(_REGISTERED.clear)


def subscribe_for_agent(
    *,
    home: Path,
    upstream_endpoint: str,
    upstream_model: str,
    quiet_seconds: float = 60.0,
) -> bool:
    """Wire `daily_appended` → `synth_memory(home, ...)` on the
    process-wide event bus, debounced by `quiet_seconds`.

    Returns True if newly subscribed, False if this home was already
    wired. Safe to call multiple times.
    """
    key = str(home.resolve())
    if key in _REGISTERED:
        return False

    async def handler(payload: dict[str, Any]) -> None:
        # Filter to events about THIS agent's home — the bus is shared.
        ev_home = (payload or {}).get("home", "")
        try:
            if ev_home and Path(ev_home).resolve() != home.resolve():
                return
        except Exception:  # noqa: BLE001
            return
        try:
            stats = await synth_memory(
                home,
                upstream_endpoint=upstream_endpoint,
                upstream_model=upstream_model,
            )
            _log.info(
                "memory_synth: %s home=%s",
                "changed" if stats.get("changed") else stats.get("error") or "unchanged",
                home,
            )
        except Exception:  # noqa: BLE001
            _log.exception("memory_synth handler crashed for %s", home)

    bus = get_bus()
    bus.subscribe(
        "daily_appended",
        handler,
        quiet_seconds=quiet_seconds,
        name=f"memory_synth[{home.name}]",
    )
    _REGISTERED.add(key)
    return True


__all__ = [
    "subscribe_for_agent",
    "synth_memory",
]
