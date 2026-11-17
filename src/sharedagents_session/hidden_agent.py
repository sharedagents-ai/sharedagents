"""Shared prompt/build/ensure helpers for hidden runtime agents."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

HiddenAgentCaller = Callable[[list[dict[str, str]]], Awaitable[str]]

_FALLBACK_SYSTEM = {
    "title": (
        "You are a title generator. Output one bare 2-6 word title in the same language as the user's opening request."
    ),
    "teacher": (
        "Evaluate whether the supplied session goal is complete and return only the requested strict JSON verdict."
    ),
    "summary": (
        "Recap the supplied finished task in 2-5 sentences and return only "
        "the requested strict JSON object (summary, verdict, follow_up, "
        "confidence)."
    ),
}


def load_hidden_agent_system(profile_name: str) -> str:
    """Load a bundled hidden profile prompt without consulting stale homes."""
    name = str(profile_name or "").strip().lower()
    try:
        from sharedagents_agent.agent_profiles import (
            load_profile_agents_md,
            profile_agents_bundle_root,
            resolve_profile,
        )

        profile = resolve_profile(name)
        if profile is not None:
            body = load_profile_agents_md(
                profile_agents_bundle_root() / name,
                profile,
            ).strip()
            if body:
                return body
    except Exception:  # noqa: BLE001
        pass
    return _FALLBACK_SYSTEM.get(name, "")


def build_hidden_agent_title_messages(
    user_text: Any,
    summary: str = "",
) -> list[dict[str, str]]:
    """Build the hidden title agent's opening-turn prompt."""
    from sharedagents_session.session_title import opening_prompt_text

    parts: list[str] = []
    blurb = str(summary or "").strip()
    if blurb:
        parts.append("Background — what has happened so far:\n" + blurb)
    parts.append("Generate a title for this conversation:\n" + opening_prompt_text(user_text))
    return [
        {"role": "system", "content": load_hidden_agent_system("title")},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def _turn_text(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(
            str(part.get("text") or "").strip()
            for part in value
            if isinstance(part, dict)
            and str(part.get("type") or "").lower() == "text"
            and str(part.get("text") or "").strip()
        )
    return str(value or "").strip()


async def ensure_hidden_agent_value(
    session: Any,
    caller: HiddenAgentCaller,
    *,
    messages: list[dict[str, str]],
    attribute: str,
    cleaner: Callable[[Any], str],
    persist: Callable[[], None] | None = None,
) -> bool:
    """Call one hidden agent and persist a changed cleaned attribute."""
    try:
        raw = await caller(messages)
    except Exception:  # noqa: BLE001 — hidden agents never fail a user turn
        return False
    value = cleaner(raw)
    current = str(getattr(session, attribute, "") or "").strip()
    if not value or value == current:
        return False
    setattr(session, attribute, value)
    saver = persist if persist is not None else getattr(session, "save", None)
    if callable(saver):
        try:
            saver()
        except Exception:  # noqa: BLE001
            pass
    return True


__all__ = [
    "HiddenAgentCaller",
    "build_hidden_agent_title_messages",
    "ensure_hidden_agent_value",
    "load_hidden_agent_system",
]
