"""Hardcoded per-engine mode catalogues and most-autonomous defaults.

Picker ids are the native CLI/ACP values (or sharedagents ``auto``/``ask``/``plan``
for CLI engines). Discovery may add extra rows; it never replaces this
catalogue and never collapses Cursor ``ask`` into ``plan``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

# Engine → (id, label, description). First row is the most autonomous default.
ACP_MODE_CATALOGUE: dict[str, tuple[dict[str, str], ...]] = {
    "opencode_acp": (
        {"id": "build", "name": "Build", "description": "OpenCode build agent; permission is allow-all"},
        {"id": "plan", "name": "Plan", "description": "OpenCode plan agent"},
    ),
    "cursor_acp": (
        {"id": "agent", "name": "Agent", "description": "Full agent capabilities with tool access"},
        {"id": "plan", "name": "Plan", "description": "Read-only mode for planning before implementation"},
        {"id": "ask", "name": "Ask", "description": "Q&A style explanations (read-only)"},
    ),
    "kimi_acp": (
        {"id": "auto", "name": "Auto"},
        {"id": "plan", "name": "Plan"},
        {"id": "default", "name": "Default"},
    ),
    "hermes_acp": (
        {"id": "dont_ask", "name": "Don't Ask", "description": "Most autonomous — do not prompt"},
        {"id": "accept_edits", "name": "Accept Edits", "description": "Auto-approve workspace edits"},
        {"id": "default", "name": "Default", "description": "Prompt for dangerous operations"},
    ),
    "devin_acp": (
        {"id": "bypass", "name": "Bypass Permissions", "description": "Auto-approve all tool calls"},
        {"id": "accept-edits", "name": "Code", "description": "Write and edit code; auto-approve workspace edits"},
        {"id": "ask", "name": "Ask", "description": "Answer questions without code changes"},
        {"id": "plan", "name": "Plan", "description": "Plan changes before implementing"},
    ),
    "qwen_acp": (
        {"id": "auto", "name": "Auto", "description": "LLM classifier auto-approves safe actions"},
        {"id": "plan", "name": "Plan", "description": "Analyze only, do not modify files or execute commands"},
        {"id": "default", "name": "Default", "description": "Require approval for file edits or shell commands"},
        {"id": "yolo", "name": "Yolo", "description": "Automatically approve all tools"},
    ),
    "claude_cli": (
        {"id": "auto", "name": "Allow all", "description": "--permission-mode bypassPermissions"},
        {"id": "ask", "name": "Ask", "description": "--permission-mode default"},
        {"id": "plan", "name": "Plan", "description": "--permission-mode plan"},
    ),
    "codex_cli": (
        {"id": "auto", "name": "Allow all", "description": "--dangerously-bypass-approvals-and-sandbox"},
        {"id": "ask", "name": "Ask", "description": "workspace-write + approval on-request"},
        {"id": "plan", "name": "Plan", "description": "read-only sandbox, never ask"},
    ),
    "antigravity_cli": (
        {"id": "auto", "name": "Allow all", "description": "--dangerously-skip-permissions"},
        {"id": "ask", "name": "Ask", "description": "CLI default — prompt for tools"},
        {"id": "plan", "name": "Plan", "description": "--mode plan --sandbox"},
    ),
    "zcode_cli": (
        {"id": "yolo", "name": "Allow all", "description": "--mode yolo"},
        {"id": "edit", "name": "Edit", "description": "--mode edit — auto-apply edits, prompt elsewhere"},
        {"id": "build", "name": "Build", "description": "--mode build — the CLI's most conservative mode"},
    ),
}

# Most autonomous requested id per engine (must be in that engine's catalogue).
DEFAULT_REQUESTED_MODE_BY_ENGINE: dict[str, str] = {
    "cursor_acp": "agent",
    "opencode_acp": "build",
    "kimi_acp": "auto",
    "hermes_acp": "dont_ask",
    "devin_acp": "bypass",
    "qwen_acp": "auto",
    "claude_cli": "auto",
    "codex_cli": "auto",
    "antigravity_cli": "auto",
    "zcode_cli": "yolo",
    "acp": "auto",
    "local_llm": "auto",
}

# Headless CLIs still have plan/ask/auto — they are no longer auto-only.
CLI_AUTO_ONLY_ENGINES: frozenset[str] = frozenset()

_PLAN_IDS = frozenset({"plan", "architect"})
_AUTO_IDS = frozenset(
    {
        "auto",
        "agent",
        "build",
        "code",
        "accept_edits",
        "acceptedits",
        "bypasspermissions",
        "dont_ask",
        "dontask",
        "yolo",
        "allow_all",
        "allow-all",
        "bypass",
    }
)
_ASK_IDS = frozenset({"ask", "default"})


def classify_native_mode(native: Mapping[str, Any] | str) -> str | None:
    """Return ``plan``, ``ask``, ``auto``, or ``None``.

    Cursor ``ask`` stays ``ask``. Read-only descriptions still classify as plan.
    """
    if isinstance(native, str):
        nid = native
        name = ""
        desc = ""
    else:
        nid = str(native.get("id") or native.get("value") or "").strip()
        name = str(native.get("name") or "").strip()
        desc = str(native.get("description") or "").strip()
    key = nid.lower().replace("-", "_")
    blob = f"{key} {name.lower()} {desc.lower()}"
    if key in _ASK_IDS and "read-only" not in blob and "readonly" not in blob:
        if key == "ask":
            return "ask"
        if key == "default" and "read-only" not in blob:
            return "ask" if "prompt" in blob or "dangerous" in blob else None
    if key in _PLAN_IDS or "plan" in key or "architect" in blob:
        return "plan"
    if "read-only" in blob or "readonly" in blob or "read only" in blob:
        return "plan"
    if key in _AUTO_IDS or any(token in key for token in ("agent", "build", "code", "auto", "dont_ask")):
        return "auto"
    if key == "ask":
        return "ask"
    return None


def extract_native_modes(payload: Mapping[str, Any] | None) -> list[dict[str, str]]:
    """Pull mode rows from a ``session/new`` or ``session/load`` result."""
    if not isinstance(payload, Mapping):
        return []
    found: list[dict[str, str]] = []
    seen: set[str] = set()

    def _add(item: Mapping[str, Any] | None) -> None:
        if not isinstance(item, Mapping):
            return
        nid = str(item.get("id") or item.get("value") or "").strip()
        if not nid or nid in seen:
            return
        seen.add(nid)
        found.append(
            {
                "id": nid,
                "name": str(item.get("name") or nid),
                "description": str(item.get("description") or ""),
            }
        )

    modes = payload.get("modes")
    if isinstance(modes, Mapping):
        for item in modes.get("availableModes") or modes.get("available_modes") or []:
            if isinstance(item, Mapping):
                _add(item)
    for opt in payload.get("configOptions") or payload.get("config_options") or []:
        if not isinstance(opt, Mapping):
            continue
        category = str(opt.get("category") or "")
        oid = str(opt.get("id") or opt.get("configId") or "")
        if category != "mode" and oid not in {"mode", "agent"}:
            continue
        for item in opt.get("options") or []:
            if isinstance(item, Mapping):
                _add(item)
    return found


def collapse_to_plan_auto(
    native_modes: Iterable[Mapping[str, Any] | str],
) -> dict[str, str]:
    """Legacy helper: map ``plan`` / ``auto`` / ``ask`` onto one native id each."""
    mapping: dict[str, str] = {}
    for item in native_modes:
        bucket = classify_native_mode(item)
        if bucket is None or bucket in mapping:
            continue
        if isinstance(item, str):
            mapping[bucket] = item
        else:
            mapping[bucket] = str(item.get("id") or item.get("value") or "")
    return {k: v for k, v in mapping.items() if v}


def catalogue_rows(engine: str) -> tuple[dict[str, str], ...]:
    return ACP_MODE_CATALOGUE.get(engine, ())


def advertised_requested_modes(
    engine: str,
    *,
    discovered: Iterable[Mapping[str, Any] | str] | None = None,
) -> tuple[str, ...]:
    """Mode ids the frontend may send for ``engine`` (catalogue first)."""
    if engine in {"", "local_llm"}:
        return ("auto", "ask")
    rows = list(catalogue_rows(engine))
    ids = [str(row["id"]) for row in rows if row.get("id")]
    if discovered:
        seen = set(ids)
        for item in discovered:
            nid = item if isinstance(item, str) else str(item.get("id") or item.get("value") or "").strip()
            if nid and nid not in seen:
                ids.append(nid)
                seen.add(nid)
    return tuple(ids) if ids else ("auto",)


def native_modes_for_engine(
    engine: str,
    *,
    discovered: Iterable[Mapping[str, Any] | str] | None = None,
) -> dict[str, str]:
    """requested picker id → native wire id (identity for catalogue rows)."""
    mapping: dict[str, str] = {}
    for row in catalogue_rows(engine):
        nid = str(row.get("id") or "")
        if nid:
            mapping[nid] = nid
    # Legacy aliases so older callers that send plan/auto still resolve.
    if engine == "opencode_acp":
        mapping.setdefault("auto", "build")
        mapping.setdefault("ask", "build")
    elif engine == "cursor_acp":
        mapping.setdefault("auto", "agent")
    elif engine == "hermes_acp":
        mapping.setdefault("auto", "dont_ask")
        mapping.setdefault("ask", "default")
        mapping.setdefault("plan", "default")
    elif engine == "devin_acp":
        mapping.setdefault("auto", "bypass")
    elif engine == "qwen_acp":
        # Qwen's approval-gated wire id is ``default``; sharedagents's ask
        # picker id resolves there (mirrors the spec mode_map).
        mapping.setdefault("ask", "default")
    if discovered:
        for item in discovered:
            nid = item if isinstance(item, str) else str(item.get("id") or item.get("value") or "").strip()
            if nid:
                mapping.setdefault(nid, nid)
    return mapping


def default_requested_mode_for_engine(
    engine: str,
    *,
    discovered: Iterable[Mapping[str, Any] | str] | None = None,
) -> str:
    """Most autonomous advertised mode."""
    pinned = DEFAULT_REQUESTED_MODE_BY_ENGINE.get(engine, "auto")
    advertised = advertised_requested_modes(engine, discovered=discovered)
    if pinned in advertised:
        return pinned
    if "auto" in advertised:
        return "auto"
    if advertised:
        return advertised[0]
    return pinned


def native_id_for_mode(
    engine: str,
    mode: str,
    *,
    discovered: Iterable[Mapping[str, Any] | str] | None = None,
) -> str:
    """Wire id to pass on CLI/ACP for a picker/session mode."""
    wanted = (mode or "").strip()
    mapping = native_modes_for_engine(engine, discovered=discovered)
    if wanted in mapping:
        return mapping[wanted]
    return wanted or default_requested_mode_for_engine(engine, discovered=discovered)


__all__ = [
    "ACP_MODE_CATALOGUE",
    "CLI_AUTO_ONLY_ENGINES",
    "DEFAULT_REQUESTED_MODE_BY_ENGINE",
    "advertised_requested_modes",
    "catalogue_rows",
    "classify_native_mode",
    "collapse_to_plan_auto",
    "default_requested_mode_for_engine",
    "extract_native_modes",
    "native_id_for_mode",
    "native_modes_for_engine",
]
