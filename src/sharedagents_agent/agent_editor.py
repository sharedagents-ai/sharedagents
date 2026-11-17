"""Agent Builder — the read/write surface for agent definitions.

The Builder (web ``/agents/builder``) reads and edits what the runtime
actually consumes: the per-agent ``config.json`` (raw JSON, merged with
global + built-in defaults the same way :meth:`AgentConfig.load` does)
and the operator-editable ``AGENTS.md`` that ``load_system_prompt``
folds into the system prompt.

Why a dedicated module and NOT ``AgentConfig.save()``: ``save()`` is a
*lean, profile-derived* writer — for profile agents it re-derives the
per-agent file from the profile scaffold and silently DROPS operator keys
(``save()`` at ``agent.py:1524``; see the ``_GLOBAL_FIELDS`` carve-outs at
``agent.py:182`` for the historical drop-hazard class). Every mutation
here is therefore a **surgical raw-JSON merge** over the operator's file
plus an **atomic ``os.replace``** — operator-owned fields that the patch
does not name are byte-preserved, and a crash mid-write can never leave a
truncated config (the file either has the old or the new full content).

Concurrency: every mutating call accepts an optional ``sha`` (SHA-256 of
the current file bytes). A stale/missing-vs-mismatched guard returns a
409 with the CURRENT file + sha so the client can re-merge — the lost-
update protection for a UI that loaded state earlier.

Scope (v1, per the Agent Builder design):
  * PATCH-able per-agent keys: ``permission_mode``, ``tools_allowlist``,
    ``tools_allowlist_strict``, ``skill_allowlist``,
    ``skill_allowlist_strict``, ``disabled_tools``, ``temperature``,
    ``effort``, ``models``, and ``upstream_model``.  A catalog-backed
    ``upstream_model`` selection atomically derives the matching provider
    route fields (endpoint / alias / backend / context); those route fields
    are NOT independently patchable. ``write_roots`` / ``write_denyroots`` /
    ``delegates_allowlist`` are READ-ONLY in v1.
  * ``AGENTS.md`` writes: 64 KiB cap, reject the untrusted-data fence
    markers (an AGENTS.md that ships its own fences would confuse the
    injection-guard rendering), atomic replace.
  * ``upstream_model`` is editable for EVERY agent class: the Model drop-down is always
    populated from the effective catalogue and a selection persists the
    full provider route exactly as for local agents. Engine/profile-
    managed agents keep ``model_managed_by`` INFORMATIONAL (the UI shows
    their managed default as a note) and fall back to
    that default when no per-agent override is set — an override set here
    wins for new sessions (per-agent file merges last). Catalogue
    membership validation (unknown/ambiguous → 422) applies to ALL
    classes; there is no ``model_override_unsupported`` rejection.
  * Effective permissions ALWAYS come from the runtime's own resolver
    (``compose_tool_allowlist`` / ``discover_skills``) — never re-derived
    in the web UI.
  * Every mutation is written to an append-only JSONL audit log.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from sharedagents_agent.agent import AgentConfig, agents_root
from sharedagents_agent.agents.engine import EngineKind
from sharedagents_core.config_io import config_write_lock

#: Keys a PATCH may write in v1. Everything else is rejected (422) so a
#: mistyped key can never silently no-op or — worse — reach a structural
#: field through the merge.
PATCHABLE_KEYS: frozenset[str] = frozenset(
    {
        "description",
        "permission_mode",
        "tools_allowlist",
        "tools_allowlist_strict",
        "skill_allowlist",
        "skill_allowlist_strict",
        "disabled_tools",
        # Per-agent list of disabled HARDCODED system-prompt section ids (the
        # Agent Builder's enable/disable cards). A list of section
        # identifiers; ``[]`` (or null, normalized to ``[]``) re-enables every
        # section. Only the toggle state is ever written — never a composed
        # prompt string.
        "disabled_prompt_sections",
        "temperature",
        "effort",
        "models",
        # Per-agent primary-model override (``cfg.upstream_model`` — the id the
        # runtime sends on the wire and uses to pick the persona/identity line).
        # The Model drop-down in the Agent Builder writes this; ``_UNSET``
        # (and JSON ``null`` — both mean "inherit global") removes the override
        # so the agent falls back to the global model. A JSON null is NEVER
        # persisted: it would shadow the global and starve the mandatory-model
        # guard on new sessions.
        "upstream_model",
        # Per-agent delegation-mode pin for first-party Agent calls
        # ``background`` / ``foreground``
        # pin the execution mode (the runtime overrides the model's explicit
        # value with a visible note); ``default`` explicitly opts back into the
        # unpoliced legacy semantics; null/_UNSET removes the key (fall back to
        # the profile default — jarvis ⇒ background, build/plan ⇒ foreground).
        "delegation_mode",
    }
)

# A model catalog row is a provider/model deployment, not merely a wire id.
# These flat AgentConfig fields are materialized as one atomic consequence of
# selecting ``upstream_model``.  They deliberately remain OUT of
# PATCHABLE_KEYS: callers may choose a configured row but cannot smuggle an
# arbitrary endpoint/backend through the Builder.
_MODEL_ROUTE_FIELDS: tuple[str, ...] = (
    "upstream_endpoint",
    "upstream_alias",
    "backend",
    "upstream_max_context",
)

# The effective ``models[]`` rows may carry provider credentials and custom
# headers.  The Builder needs picker metadata only.  Keep this projection
# explicit/fail-closed so a future secret-shaped row field is not serialized
# merely because AgentConfig learned about it.
_PUBLIC_MODEL_FIELDS: tuple[str, ...] = (
    "alias",
    "id",
    "backend",
    "provider",
    "contextWindow",
    "maxTokens",
)

#: Permission-mode values the runtime understands (MODE-*.md / turn gating).
PERMISSION_MODES: frozenset[str] = frozenset({"auto", "ask", "plan"})

#: Delegation-mode values the runtime understands: ``background``/``foreground``
#: pin the mode for first-party Agent calls; ``default`` is the explicit
#: unpoliced reset.
DELEGATION_MODES: frozenset[str] = frozenset({"background", "foreground", "default"})

#: Sentinel meaning "remove this key from the per-agent file" (falls back
#: to the global config / built-in defaults on the next load).
UNSET = "_UNSET"

#: Agent name must be a plain directory component — no slashes, no dots
#: that could path-escape, no spaces. Mirrors the CLI scaffold's naming.
_VALID_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")

#: AGENTS.md write cap (plan: 64 KiB). The runtime context loader caps
#: reads at 1 MiB per file; the Builder's edit surface is deliberately
#: smaller so a paste-bomb cannot bloat every session's prompt.
MAX_AGENTS_MD_BYTES = 64 * 1024

_AUDIT_SERVICE = "agent-editor"
_AUDIT_PATH = "logs/agent-editor.jsonl"


class AgentEditorError(Exception):
    """A domain error with an HTTP status + wire code for the gateway endpoint."""

    def __init__(self, status: int, code: str, message: str, *, extra: dict | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.extra = extra or {}


# ---------------------------------------------------------------------------
# sha / naming / audit helpers
# ---------------------------------------------------------------------------


def _file_sha(path: Path) -> str:
    """SHA-256 of the file's exact bytes (or of b"" when absent)."""
    try:
        blob = path.read_bytes()
    except OSError:
        blob = b""
    return hashlib.sha256(blob).hexdigest()


def validate_agent_name(name: str) -> None:
    """Reject names that are not safe directory components (404 vs 422:
    a name that can never be an agent is a bad request, not a miss)."""
    if not isinstance(name, str) or not _VALID_NAME_RE.match(name):
        raise AgentEditorError(422, "invalid_agent_name", f"invalid agent name: {name!r}")


def _agent_config_path(name: str) -> Path:
    validate_agent_name(name)
    return agents_root() / name / "config.json"


def _audit(kind: str, msg: str, *, agent: str, k: dict | None = None) -> None:
    """Append one JSONL audit line (best-effort — audit must never crash a save)."""
    try:
        from sharedagents_core import jsonlog

        root = agents_root().parent
        writer = jsonlog.get_writer(
            root / _AUDIT_PATH,
            service=_AUDIT_SERVICE,
        )
        writer.write(agent=agent, level="info", msg=msg, kind=kind, k=k or {})
    except Exception:  # noqa: BLE001 — audit is best-effort by contract
        pass


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write ``data`` as pretty JSON via tmp-file + ``os.replace``.

    ``os.replace`` is atomic on POSIX: readers (the gateway's mtime
    watcher, ``AgentConfig.load``) see either the old or the new file,
    never a torn write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    with config_write_lock(path):
        tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(blob)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def list_agent_defs() -> list[dict]:
    """Summary row per configured agent (a directory with a config.json)."""
    out: list[dict] = []
    root = agents_root()
    if not root.is_dir():
        return out
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        cfg_path = child / "config.json"
        if not cfg_path.is_file():
            continue
        try:
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = {}
        agents_md = child / "AGENTS.md"
        out.append(
            {
                "name": child.name,
                "agent_profile": str(raw.get("agent_profile") or "").strip() or None,
                "description": str(raw.get("description") or "").strip() or None,
                "permission_mode": str(raw.get("permission_mode") or "auto"),
                "engine": str(raw.get("engine") or "").strip() or None,
                "config_sha": _file_sha(cfg_path),
                "has_agents_md": agents_md.is_file(),
            }
        )
    return out


def load_agent_config(name: str) -> AgentConfig:
    """``AgentConfig.load`` with a Builder-shaped 404 for unknown agents."""
    validate_agent_name(name)
    cfg_path = agents_root() / name / "config.json"
    if not cfg_path.is_file():
        raise AgentEditorError(404, "unknown_agent", f"agent {name!r} is not configured")
    try:
        return AgentConfig.load(name)
    except FileNotFoundError as exc:
        raise AgentEditorError(404, "unknown_agent", f"agent {name!r} is not configured") from exc


def effective_tools(cfg: AgentConfig) -> tuple[list[str], bool]:
    """The tools this agent may see, per the runtime's OWN resolver.

    Uses ``compose_tool_allowlist`` — the exact function the wire
    ``tools[]``, the dispatch gate, and the prompt consume — and expands
    a ``None`` (unrestricted) result to the concrete registry names so
    the UI can render a list. NEVER reimplement the allowlist logic here.
    """
    from sharedagents_agent.agent_profiles import compose_tool_allowlist
    from sharedagents_agent.tools.registry import (
        BUILTIN_TOOLS,
        ToolRegistry,
        discovered_mcp_tools_sync,
    )

    allowed = compose_tool_allowlist(cfg)
    if allowed is None:
        registry = ToolRegistry(list(BUILTIN_TOOLS) + list(discovered_mcp_tools_sync()))
        return sorted(registry.names()), True
    return sorted(allowed), False


def effective_skills(cfg: AgentConfig) -> tuple[list[str], bool]:
    """Effective skill set: discovered skills filtered by the allowlist,
    using the runtime's own discovery + filter (``discover_skills``)."""
    from sharedagents_skill.skills import discover_skills

    home = Path(getattr(cfg, "home", "") or "")
    try:
        found = discover_skills(
            home,
            allowlist=list(cfg.skill_allowlist) if cfg.skill_allowlist is not None else None,
            allowlist_strict=bool(cfg.skill_allowlist_strict),
        )
    except Exception:  # noqa: BLE001 — permissions card must degrade, not 500
        return [], True
    unrestricted = cfg.skill_allowlist is None and not cfg.skill_allowlist_strict
    return [s.name for s in found], unrestricted


def available_skills(cfg: AgentConfig) -> list[str]:
    """Installed, globally enabled skills before this agent's allowlist.

    The Builder uses this catalog to let an operator restore a skill that is
    currently filtered out. It is display data only; effective access remains
    the result of :func:`effective_skills` and the runtime discovery path.
    """
    from sharedagents_skill.skills import discover_skills

    home = Path(getattr(cfg, "home", "") or "")
    try:
        return [
            skill.name
            for skill in discover_skills(
                home,
                allowlist=None,
                allowlist_strict=False,
            )
        ]
    except Exception:  # noqa: BLE001 — catalog failure must not break Builder
        return []


def model_owner(cfg: object) -> str | None:
    """Who owns this agent's DEFAULT model — INFORMATIONAL only.

    It does not gate the Builder's model selector; it only drives the UI's
    managed-default note so the operator sees what a cleared override falls
    back to:

       * ``"engine"`` — the agent runs on an external engine (anything other
         than ``local_llm`` per :class:`EngineKind`: claude_cli / codex_cli /
         antigravity_cli / cursor_acp / opencode_acp / kimi_acp / hermes_acp /
         devin_acp / qwen_acp /
         acp). Those
         engines resolve the DEFAULT model via their OWN catalogs when no
         per-agent ``upstream_model`` override is set. Mirrors the runtime's
         egress classification (``EngineModeRegistry``: LOCAL iff engine is
         ``local_llm``, everything else FOREIGN). An empty engine string is
         the pre-engine legacy default and counts as local — same
         normalization the factory applies.
       * ``"profile"`` — the agent is profile-managed (``agent_profile``
         resolves to a known profile; the profile bypasses the persona stack
         in ``context.py:257``). The profile does NOT set ``upstream_model``
         (``apply_profile_to_config`` touches only allowlists +
         ``permission_mode``), so the merged per-agent override still wins
         for new sessions; ``"profile"`` here only describes the default
         source when no override is set. Matches the exact predicate
         ``AgentConfig.load`` uses to apply the profile (``agent.py:1728`` —
         ``resolve_profile`` of the ``agent_profile`` field).
       * ``None`` — a LOCAL NON-PROFILE agent: the global/legacy default
         applies when no override is set.

       ``model_owner`` is the single source of truth for the UI flag
       (``read_definition`` → ``model_managed_by``), but the PATCH path no
       longer consults it — membership validation (``validate_model_membership``)
       is the only model gate and applies to every class.
    """
    engine = str(getattr(cfg, "engine", "") or "").strip()
    if engine and engine != EngineKind.LOCAL_LLM.value:
        return "engine"
    from sharedagents_agent.agent_profiles import resolve_profile_for_config

    if resolve_profile_for_config(cfg) is not None:
        return "profile"
    return None


def _public_model_catalog(rows: Any) -> list[dict[str, Any]]:
    """Return the secret-safe picker projection of effective model rows."""
    out: list[dict[str, Any]] = []
    for raw in rows or []:
        if not isinstance(raw, dict):
            continue
        row: dict[str, Any] = {}
        for key in _PUBLIC_MODEL_FIELDS:
            value = raw.get(key)
            if isinstance(value, str):
                value = value.strip()
                if value:
                    row[key] = value
            elif key in {"contextWindow", "maxTokens"} and (
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
            ):
                row[key] = value
        # Rows without an id cannot be selected and would create a misleading
        # blank option.  The runtime's /models command likewise surfaces them
        # as configuration errors rather than selectable deployments.
        if row.get("id"):
            out.append(row)
    return out


def read_definition(name: str) -> dict:
    """Full definition for one agent: raw on-disk values + the effective
    (merged) values the runtime resolves, including effective permissions.

    ``effective_*`` fields are computed by the backend resolvers
    (``compose_tool_allowlist`` / ``discover_skills``) — the web UI only
    renders what this endpoint returns.
    """
    validate_agent_name(name)
    cfg_path = _agent_config_path(name)
    if not cfg_path.is_file():
        raise AgentEditorError(404, "unknown_agent", f"agent {name!r} is not configured")
    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AgentEditorError(
            422, "config_unreadable", f"agent {name!r} config.json is not valid JSON: {exc}"
        ) from exc
    cfg = load_agent_config(name)
    eff_tools, unrestricted = effective_tools(cfg)
    eff_skills, all_skills = effective_skills(cfg)
    skill_catalog = available_skills(cfg)
    # Foreign engines own their catalogue out-of-band. After load()
    # strips inherited global ``models[]``, Builder must surface the
    # engine menu (cursor / opencode / claude) rather than an empty
    # list or leftover OpenRouter rows. Operator-owned per-agent
    # ``models[]`` still wins.
    from sharedagents_command.commands import engine_model_rows

    engine_rows = engine_model_rows(cfg)
    own_models = [m for m in (getattr(cfg, "models", None) or []) if isinstance(m, dict)]
    if engine_rows is not None and not own_models:
        catalog = _public_model_catalog(engine_rows)
        if getattr(cfg, "_upstream_model_is_per_agent", False):
            eff_model = str(getattr(cfg, "upstream_model", "") or "").strip()
        else:
            default = next(
                (r for r in engine_rows if isinstance(r, dict) and r.get("default")),
                engine_rows[0] if engine_rows else None,
            )
            eff_model = str((default or {}).get("id") or "")
    else:
        catalog = _public_model_catalog(own_models)
        try:
            eff_model = (
                str(getattr(cfg, "upstream_model", "") or "").strip()
                or str((cfg._primary_model_row() or {}).get("id") or "").strip()
            )
        except Exception:  # noqa: BLE001 — model id is display data, never fatal
            eff_model = ""
    return {
        "name": name,
        "agent_profile": getattr(cfg, "agent_profile", None),
        "description": raw.get("description") if "description" in raw else (getattr(cfg, "description", None) or ""),
        "engine": getattr(cfg, "engine", None),
        "permission_mode": getattr(cfg, "permission_mode", "auto"),
        "temperature": getattr(cfg, "temperature", None),
        "effort": getattr(cfg, "effort", None),
        "models": catalog,
        "model": eff_model,
        # Server-authoritative model-ownership flag: "engine" | "profile" |
        # null. INFORMATIONAL only — the UI shows the
        # managed default as a note but the drop-down stays enabled and
        # populated for every class; the PATCH path no longer gates on it.
        "model_managed_by": model_owner(cfg),
        "system": getattr(cfg, "system", None),
        # Raw operator-owned values (what PATCH edits; None = not set on disk).
        "tools_allowlist": raw.get("tools_allowlist"),
        "tools_allowlist_strict": bool(raw.get("tools_allowlist_strict", False)),
        "skill_allowlist": raw.get("skill_allowlist"),
        "skill_allowlist_strict": bool(raw.get("skill_allowlist_strict", False)),
        "disabled_tools": raw.get("disabled_tools"),
        # Per-agent disabled hardcoded prompt sections — raw on-disk
        # value; absent key means every section is enabled.
        "disabled_prompt_sections": raw.get("disabled_prompt_sections"),
        "delegation_mode": raw.get("delegation_mode"),
        "on_disk_keys": sorted(raw.keys()),
        "config_sha": _file_sha(cfg_path),
        # Effective permissions (backend-resolved).
        "effective_tools": eff_tools,
        "effective_tools_unrestricted": unrestricted,
        "effective_skills": eff_skills,
        "effective_skills_unrestricted": all_skills,
        # Full pre-agent-filter catalog for the checkbox editor. This list is
        # informational; ``effective_skills`` remains backend-authoritative.
        "available_skills": skill_catalog,
    }


def read_agents_md(name: str) -> dict:
    """The operator-editable AGENTS.md body + sha for one agent."""
    validate_agent_name(name)
    cfg_path = _agent_config_path(name)
    if not cfg_path.is_file():
        raise AgentEditorError(404, "unknown_agent", f"agent {name!r} is not configured")
    path = agents_root() / name / "AGENTS.md"
    if path.is_file():
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise AgentEditorError(422, "agents_md_unreadable", f"AGENTS.md is not readable: {exc}") from exc
    else:
        content = ""
    return {
        "name": name,
        "content": content,
        "sha": _file_sha(path),
        "exists": path.is_file(),
        "bytes": len(content.encode("utf-8")),
        "max_bytes": MAX_AGENTS_MD_BYTES,
    }


# ---------------------------------------------------------------------------
# Validation (shared by PATCH and the dry-run validate endpoint)
# ---------------------------------------------------------------------------


def validate_patch(patch: Any) -> dict[str, Any]:
    """Validate + normalize a PATCH body. Returns the cleaned patch.

    Raises ``AgentEditorError`` (422) with a per-key detail map when any
    key is unknown or a value has the wrong shape. Mirrors the profile
    system's allowlist normalization (``apply_profile_to_config``): a
    null allowlist means UNRESTRICTED (strict forced off), a non-null
    list without an explicit strict flag becomes strict.

    ``upstream_model`` semantics: ``null`` means "inherit global" — the
    same as ``"_UNSET"`` — and is normalized to the sentinel so the key
    is REMOVED from the per-agent file (never persisted as JSON null,
    which would shadow the global and break the mandatory-model guard on
    new sessions). The API accepts null; it just means inherit.
    """
    if not isinstance(patch, dict):
        raise AgentEditorError(422, "bad_patch", "patch body must be a JSON object")
    unknown = sorted(set(patch) - PATCHABLE_KEYS)
    if unknown:
        raise AgentEditorError(
            422,
            "unknown_key",
            f"unsupported key(s): {', '.join(unknown)}",
            extra={"unknown_keys": unknown},
        )
    out: dict[str, Any] = {}
    errors: dict[str, str] = {}

    def _str_list(key: str) -> None:
        val = patch[key]
        if val is None:
            out[key] = None
            return
        if isinstance(val, str) and val == UNSET:
            out[key] = UNSET
            return
        if not isinstance(val, list) or not all(isinstance(x, str) and x.strip() for x in val):
            errors[key] = 'must be a list of non-empty strings, null, or "_UNSET"'
            return
        out[key] = [x.strip() for x in val]

    def _tool_list(key: str) -> None:
        """Validate tool ids without requiring MCP servers to be online.

        Builtins are checked against the real registry (legacy TitleCase ids
        canonicalize there).  MCP exact names and server grants are validated
        structurally rather than against the discovery cache: a configured
        server may be cold/unreachable while the operator edits its grant.
        """
        _str_list(key)
        value = out.get(key)
        if not isinstance(value, list):
            return
        from sharedagents_agent.tools.registry import (
            BUILTIN_TOOLS,
            ToolRegistry,
            canonicalize_tool_id,
        )

        builtins = set(ToolRegistry(BUILTIN_TOOLS).names())
        normalized: list[str] = []
        invalid: list[str] = []
        for token in value:
            canonical = canonicalize_tool_id(token)
            if canonical in builtins:
                normalized.append(canonical)
                continue
            if token == "mcp__*":
                normalized.append(token)
                continue
            if token.startswith("mcp__"):
                rest = token[len("mcp__") :]
                server, sep, tool = rest.rpartition("__")
                if sep and server and tool and not any(ch.isspace() or ord(ch) < 32 for ch in token):
                    normalized.append(token)
                    continue
            invalid.append(token)
        if invalid:
            errors[key] = "unknown or invalid tool id(s): " + ", ".join(invalid)
            out.pop(key, None)
        else:
            out[key] = normalized

    if "description" in patch:
        v = patch["description"]
        if v is None or (isinstance(v, str) and v == UNSET):
            out["description"] = UNSET
        elif isinstance(v, str):
            out["description"] = v.strip()
        else:
            errors["description"] = 'must be a string, null, or "_UNSET"'
    if "permission_mode" in patch:
        v = patch["permission_mode"]
        if v not in PERMISSION_MODES:
            errors["permission_mode"] = f"must be one of {sorted(PERMISSION_MODES)}"
        else:
            out["permission_mode"] = v
    if "delegation_mode" in patch:
        v = patch["delegation_mode"]
        if v is None or v == UNSET:
            # Remove the key → the applied profile's default applies again.
            out["delegation_mode"] = UNSET
        elif v in DELEGATION_MODES:
            out["delegation_mode"] = v
        else:
            errors["delegation_mode"] = f"must be one of {sorted(DELEGATION_MODES)} or null (inherit profile default)"
    if "tools_allowlist" in patch:
        _tool_list("tools_allowlist")
    if "tools_allowlist_strict" in patch:
        v = patch["tools_allowlist_strict"]
        if not isinstance(v, bool):
            errors["tools_allowlist_strict"] = "must be a boolean"
        else:
            out["tools_allowlist_strict"] = v
    if "skill_allowlist" in patch:
        _str_list("skill_allowlist")
    if "skill_allowlist_strict" in patch:
        v = patch["skill_allowlist_strict"]
        if not isinstance(v, bool):
            errors["skill_allowlist_strict"] = "must be a boolean"
        else:
            out["skill_allowlist_strict"] = v
    if "disabled_tools" in patch:
        _tool_list("disabled_tools")
    if "disabled_prompt_sections" in patch:
        # null / _UNSET removes the key (absent ⇒ every section enabled);
        # a list must be UNIQUE non-empty ids, each ≤ 128 chars — ids are
        # slugs of section titles, so a stray duplicate or whitespace-only
        # entry would silently disable nothing (or the wrong section).
        v = patch["disabled_prompt_sections"]
        if v is None or v == UNSET:
            out["disabled_prompt_sections"] = UNSET
        elif (
            isinstance(v, list)
            and all(isinstance(x, str) and x.strip() for x in v)
            and len({x.strip() for x in v}) == len(v)
            and all(len(x.strip()) <= 128 for x in v)
        ):
            out["disabled_prompt_sections"] = [x.strip() for x in v]
        else:
            errors["disabled_prompt_sections"] = (
                'must be a list of unique non-empty section ids (each <= 128 chars), null, or "_UNSET"'
            )
    if "temperature" in patch:
        v = patch["temperature"]
        if v is None:
            out["temperature"] = None
        elif isinstance(v, bool) or not isinstance(v, (int, float)) or not (0.0 <= float(v) <= 2.0):
            errors["temperature"] = "must be a number in [0, 2] or null"
        else:
            out["temperature"] = float(v)
    if "effort" in patch:
        v = patch["effort"]
        if v is None or (isinstance(v, str) and v == UNSET):
            out["effort"] = UNSET
        elif isinstance(v, str) and v.strip():
            out["effort"] = v.strip()
        else:
            errors["effort"] = 'must be a non-empty string, null, or "_UNSET"'
    if "models" in patch:
        v = patch["models"]
        if v is None:
            out["models"] = None
        elif isinstance(v, list) and all(isinstance(x, dict) for x in v):
            out["models"] = v
        else:
            errors["models"] = "must be a list of model objects or null"
    if "upstream_model" in patch:
        v = patch["upstream_model"]
        if v is None or (isinstance(v, str) and v == UNSET):
            # JSON null = "inherit global" (same as _UNSET). Normalize to
            # the sentinel so the key is REMOVED on write — never persist a
            # JSON null, which would shadow the global upstream_model and
            # break the mandatory-model guard on new sessions. The API keeps
            # accepting null; it just means inherit, exactly like _UNSET.
            out["upstream_model"] = UNSET
        elif isinstance(v, str) and v.strip():
            out["upstream_model"] = v.strip()
        else:
            errors["upstream_model"] = 'must be a non-empty string, null, or "_UNSET"'

    if errors:
        raise AgentEditorError(
            422,
            "invalid_patch",
            "invalid patch values",
            extra={"field_errors": errors},
        )

    # Allowlist normalization (mirrors apply_profile_to_config, so the
    # Builder's writes resolve identically to a profile-managed config):
    #   null list            → UNRESTRICTED, strict forced off
    #   non-null list + no explicit strict → EXACT list, strict on
    for key in ("tools_allowlist", "skill_allowlist"):
        if key not in out:
            continue
        strict_key = f"{key}_strict"
        if out[key] is None:
            if strict_key not in out:
                out[strict_key] = False
        elif strict_key not in out:
            out[strict_key] = True
    return out


# ---------------------------------------------------------------------------
# Writes (raw-JSON merge + atomic replace; sha-guarded)
# ---------------------------------------------------------------------------


def validate_model_membership(name: str, cleaned: dict[str, Any]) -> None:
    """Reject an ``upstream_model`` override that names an unknown model.

    The Model drop-down is config-driven from the runtime's own catalog —
    the agent's effective ``models[]`` rows (merged global + per-agent),
    exactly what ``resolve_model`` / ``_primary_model_row`` / the TUI
    picker enumerate. A non-empty override must be one of those row ids;
    ``None`` / ``_UNSET`` (inherit global) and agents WITHOUT a configured
    ``models[]`` catalog (flat single-primary configs) are always accepted
    (documented, not a security boundary — matching the TUI picker).

    The check applies to EVERY agent class:
    engine-managed and profile-managed agents may set a
    per-agent override too — the merged ``upstream_model`` wins for new
    sessions (per-agent file merges last in ``AgentConfig.load``), and
    without an override they fall back to their managed default exactly as
    before. ``model_managed_by`` is informational only.

    Validation runs against the PROSPECTIVE state, not just the current
    file: when the same patch also replaces ``models[]``, the NEW list is
    the membership catalog, so a combined replace-models + pick-model
    patch is judged on what the agent WOULD have (an id outside the new
    list is rejected). Removing the catalog (``models`` → null/[]) while
    setting a non-empty override is rejected outright — a deliberate
    catalog-escape that would leave the override dangling with no
    membership anchor.
    """
    if "upstream_model" not in cleaned:
        return
    value = cleaned["upstream_model"]
    if value is None or value == UNSET or not isinstance(value, str):
        # null / _UNSET = "inherit global" — never a model claim, so it is
        # accepted for EVERY agent class (clearing a stale override is
        # harmless; the engine/profile simply falls back to its default).
        return
    cfg = load_agent_config(name)
    if "models" in cleaned:
        new_models = cleaned["models"]
        if new_models is None or new_models == []:
            raise AgentEditorError(
                422,
                "model_catalog_removed",
                f"cannot remove the models catalog while setting "
                f"upstream_model {value!r} — clear the override "
                f'(null or "_UNSET") or keep the catalog',
            )
        rows = [m for m in new_models if isinstance(m, dict)]
    else:
        own_models = [m for m in (getattr(cfg, "models", None) or []) if isinstance(m, dict)]
        if own_models:
            try:
                from sharedagents_agent.discovery.cache_lock import native_models_with_cache

                rows = native_models_with_cache(name, own_models)
            except Exception:  # noqa: BLE001
                rows = own_models
        else:
            from sharedagents_command.commands import engine_model_rows

            erows = engine_model_rows(cfg)
            rows = [m for m in erows if isinstance(m, dict)] if erows is not None else []
    matched_rows = [
        row
        for row in rows
        if isinstance(row, dict)
        and (str(row.get("id") or "").strip() == value or str(row.get("alias") or "").strip() == value)
    ]
    if rows and not matched_rows:
        known = [str(m.get("id") or m.get("alias") or "") for m in rows]
        known = [mid for mid in known if mid]
        raise AgentEditorError(
            422,
            "unknown_model",
            f"model {value!r} is not configured for agent {name!r} — available: {', '.join(known)}",
            extra={"model": value, "available_models": known},
        )
    if len(matched_rows) > 1:
        aliases = [str(row.get("alias") or "") for row in matched_rows]
        raise AgentEditorError(
            422,
            "ambiguous_model",
            f"model {value!r} appears in multiple configured provider rows; "
            "model ids must uniquely identify a Builder selection",
            extra={"model": value, "aliases": aliases},
        )


def _model_route_patch(name: str, cleaned: dict[str, Any]) -> dict[str, Any]:
    """Derive flat provider routing fields for a model selection.

    ``AgentConfig`` keeps the active provider in flat fields while ``models[]``
    is the deployment catalogue.  Persisting only the chosen row's ``id``
    creates a split-brain default (model B sent to provider A).  This helper
    resolves the prospective catalogue row server-side and expands the one
    operator selection into the complete route in the SAME atomic raw-JSON
    merge.  Catalog-less legacy configs retain their historical id-only
    behavior because no authoritative route metadata exists to derive.

    ``_UNSET`` means inherit global and therefore removes the whole per-agent
    route override, not just the model id.
    """
    if "upstream_model" not in cleaned:
        return {}
    value = cleaned["upstream_model"]
    if value is None or value == UNSET:
        return dict.fromkeys(_MODEL_ROUTE_FIELDS, UNSET)
    if not isinstance(value, str):
        return {}

    cfg = load_agent_config(name)
    if "models" in cleaned:
        prospective = cleaned["models"]
        rows = prospective if isinstance(prospective, list) else []
    else:
        own_models = [m for m in (getattr(cfg, "models", None) or []) if isinstance(m, dict)]
        if own_models:
            try:
                from sharedagents_agent.discovery.cache_lock import native_models_with_cache

                rows = native_models_with_cache(name, own_models)
            except Exception:  # noqa: BLE001
                rows = own_models
        else:
            from sharedagents_command.commands import engine_model_rows

            erows = engine_model_rows(cfg)
            rows = [m for m in erows if isinstance(m, dict)] if erows is not None else []

    matches = [
        row
        for row in rows
        if isinstance(row, dict)
        and (
            str(row.get("id") or row.get("model") or "").strip() == value
            or str(row.get("alias") or "").strip() == value
        )
    ]
    if not matches:
        # Membership validation already rejects a miss when the catalogue has
        # ids.  No match here therefore means a catalog-less legacy config.
        return {}
    if len(matches) > 1:
        aliases = [str(row.get("alias") or "") for row in matches]
        raise AgentEditorError(
            422,
            "ambiguous_model",
            f"model {value!r} appears in multiple configured provider rows; "
            "model ids must uniquely identify a Builder selection",
            extra={"model": value, "aliases": aliases},
        )

    row = matches[0]
    route: dict[str, Any] = {}
    endpoint = str(row.get("baseUrl") or row.get("endpoint") or "").strip()
    alias = str(row.get("alias") or "").strip()
    backend = str(row.get("backend") or "").strip()
    context = row.get("contextWindow", row.get("max_context"))
    if endpoint:
        route["upstream_endpoint"] = endpoint.rstrip("/")
    if alias:
        route["upstream_alias"] = alias
    if backend:
        route["backend"] = backend
    if isinstance(context, int) and not isinstance(context, bool) and context >= 0:
        route["upstream_max_context"] = context
    return route


def patch_config(
    name: str,
    patch: dict[str, Any],
    *,
    sha: str | None = None,
) -> dict:
    """Surgically merge *patch* into the per-agent config.json and persist
    atomically. Operator keys NOT named in *patch* are byte-preserved.

    ``sha`` guards against lost updates: when supplied and different from
    the current file's SHA-256, raises 409 carrying the CURRENT file so
    the client can re-merge. ``UNSET`` removes a key (fall back to
    global/builtin). Returns the new config_sha + normalized patch.
    """
    validate_agent_name(name)
    path = _agent_config_path(name)
    if not path.is_file():
        raise AgentEditorError(404, "unknown_agent", f"agent {name!r} is not configured")
    current_sha = _file_sha(path)
    if sha is not None and sha != current_sha:
        try:
            current_raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            current_raw = {}
        raise AgentEditorError(
            409,
            "config_changed",
            "config changed since it was loaded — re-merge and retry",
            extra={"config_sha": current_sha, "current": current_raw},
        )
    cleaned = validate_patch(patch)
    validate_model_membership(name, cleaned)
    # Expand the selected catalog row only AFTER membership/ownership checks.
    # Derived fields are not client-patchable and land in the same os.replace.
    cleaned.update(_model_route_patch(name, cleaned))
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AgentEditorError(
            422, "config_unreadable", f"agent {name!r} config.json is not valid JSON: {exc}"
        ) from exc
    for key, value in cleaned.items():
        if value == UNSET:
            raw.pop(key, None)
        else:
            raw[key] = value
    _atomic_write_json(path, raw)
    new_sha = _file_sha(path)
    _audit(
        "config.patch",
        f"patched config keys: {sorted(cleaned)}",
        agent=name,
        k={"keys": sorted(cleaned), "config_sha": new_sha},
    )
    return {"ok": True, "config_sha": new_sha, "applied_keys": sorted(cleaned)}


def write_agents_md(
    name: str,
    content: str,
    *,
    sha: str | None = None,
) -> dict:
    """Persist the operator-editable AGENTS.md (atomic replace, sha-guarded).

    Rejects content over the 64 KiB cap and content carrying the
    untrusted-data fence markers — AGENTS.md is TRUSTED input, and a file
    that ships its own fences would be indistinguishable from (and could
    pre-empt) the runtime's own injection-guard fencing.
    """
    validate_agent_name(name)
    cfg_path = _agent_config_path(name)
    if not cfg_path.is_file():
        raise AgentEditorError(404, "unknown_agent", f"agent {name!r} is not configured")
    if not isinstance(content, str):
        raise AgentEditorError(422, "bad_content", "content must be a string")
    blob = content.encode("utf-8")
    if len(blob) > MAX_AGENTS_MD_BYTES:
        raise AgentEditorError(
            413,
            "agents_md_too_large",
            f"AGENTS.md exceeds the {MAX_AGENTS_MD_BYTES}-byte cap ({len(blob)} bytes)",
        )
    from sharedagents_core.kernel.injection_guard import UNTRUSTED_CLOSE, UNTRUSTED_OPEN

    if UNTRUSTED_OPEN in content or UNTRUSTED_CLOSE in content:
        raise AgentEditorError(
            422,
            "agents_md_fences_rejected",
            "AGENTS.md must not contain the untrusted-data fence markers",
        )
    path = agents_root() / name / "AGENTS.md"
    current_sha = _file_sha(path)
    if sha is not None and sha != current_sha:
        raise AgentEditorError(
            409,
            "agents_md_changed",
            "AGENTS.md changed since it was loaded — re-merge and retry",
            extra={"agents_md_sha": current_sha},
        )
    _atomic_write_text(path, content)
    new_sha = _file_sha(path)
    _audit("agents_md.write", "wrote AGENTS.md", agent=name, k={"bytes": len(blob), "agents_md_sha": new_sha})
    return {"ok": True, "agents_md_sha": new_sha, "bytes": len(blob)}


# ---------------------------------------------------------------------------
# System-prompt render (hardcoded vs AGENTS.md split)
# ---------------------------------------------------------------------------


def render_system_prompt(name: str) -> dict:
    """The COMPLETE system prompt the agent would send to the model, split
    into annotated sections by source: ``hardcoded`` (runtime code /
    shipped templates), ``agents_md`` (the operator-editable file),
    ``system`` (the config ``system`` field), ``dynamic`` (below the cache
    boundary — MEMORY.md etc.).

    Composed with the SAME builders ``_compose_system_prompt`` uses, in
    the SAME order, for the same (mode, cwd, origin) inputs. As a drift
    guard the assembled text is compared byte-for-byte against the real
    ``_compose_system_prompt`` output: if they ever diverge (e.g. a future
    composition change this module did not mirror), the response degrades
    to a single unannotated block with ``exact=false`` rather than lying
    about which part is which.

    Every section carries a stable ``id`` (slug of its title; duplicate
    slugs get ``-2``, ``-3``, … in render order) and a ``disabled`` flag
    (hardcoded sections only): the Agent Builder's enable/disable cards
    toggle ``cfg.disabled_prompt_sections``, and the composed
    prompt the agent actually sends is the canonical sections MINUS the
    disabled hardcoded ones. ``disabled_prompt_sections`` (top level) is
    the agent's persisted list, echoed back for the UI.
    """
    cfg = load_agent_config(name)
    from sharedagents_agent.context import load_system_prompt_parts
    from sharedagents_agent.prompt_build import (
        _compose_system_prompt,
        _model_family_and_id,
        _profile_system_prompt,
        assign_prompt_part_ids,
        dynamic_prompt_parts,
        filter_disabled_prompt_parts,
        head_prompt_parts,
        tail_prompt_parts,
    )
    from sharedagents_agent.subagent_policy import profile_skips_memory
    from sharedagents_agent.system_prompt import compose_with_cache_boundary

    mode = str(getattr(cfg, "permission_mode", "") or "auto")
    if mode not in ("auto", "plan"):
        mode = "auto"

    home = Path(getattr(cfg, "home", "") or "")
    # Same engine-kind signal the runtime composer derives (cfg.engine
    # holds the EngineKind value string) so the mirror stays byte-exact.
    engine_kind = str(getattr(cfg, "engine", "") or "")
    model_family, model_id = _model_family_and_id(cfg)
    skip_memory = profile_skips_memory(cfg)

    # 1+2. Canonical sections — the EXACT shared builders the runtime
    #      composer uses (head → .md stack → tail → dynamic), so the
    #      mirror cannot drift from what the agent actually sends.
    head_parts = head_prompt_parts(cfg)
    md_parts = load_system_prompt_parts(
        str(getattr(cfg, "system", "") or ""),
        root=home,
        mode=mode,
        model_family=model_family,
        model_id=model_id,
        profile_system=_profile_system_prompt(cfg),
        agent_name=cfg.name,
        engine_kind=engine_kind,
        record_breakdown=False,  # builder views must not pollute /context
    )
    tail_parts = tail_prompt_parts(cfg)
    # No live cwd / heartbeat origin in the builder — same inputs the
    # runtime gets for a plain session (WORKSPACE-CONTEXT from the
    # ``SOURCE_FOLDER`` env only, then MEMORY.md).
    dynamic = dynamic_prompt_parts(cfg, skip_memory=skip_memory)

    # 3. Stable ids + persisted toggle state. Ids are assigned over the
    #    FULL canonical list (render order) so a disabled section keeps
    #    the id the operator toggled, and duplicates get -2/-3 suffixes
    #    deterministically.
    all_parts = head_parts + md_parts + tail_parts + dynamic
    assign_prompt_part_ids(all_parts)
    disabled = {str(s).strip() for s in (getattr(cfg, "disabled_prompt_sections", None) or []) if str(s).strip()}
    for p in all_parts:
        p["disabled"] = p["source"] == "hardcoded" and p["id"] in disabled

    # 4. Assemble the prompt the agent WOULD send — canonical sections
    #    minus the disabled hardcoded ones (only hardcoded is filterable;
    #    agents_md / system / dynamic are never touched). The join
    #    sequence mirrors ``_compose_system_prompt`` exactly:
    #    head → md stack → skills block → MCP block (each appended to the
    #    FULL stable, preserving byte order), then cache-boundary + dynamic.
    live_head = filter_disabled_prompt_parts(head_parts, disabled)
    live_md = filter_disabled_prompt_parts(md_parts, disabled)
    live_tail = filter_disabled_prompt_parts(tail_parts, disabled)
    stable_head = "\n\n".join(p["text"] for p in live_head if p["text"] and p["text"].strip())
    md_stable = "\n\n".join(p["text"] for p in live_md)
    if stable_head.strip() and md_stable.strip():
        stable = f"{stable_head}\n\n{md_stable}"
    elif not stable_head.strip():
        stable = md_stable
    else:
        stable = stable_head
    for t in live_tail:
        stable = f"{stable}\n\n{t['text']}" if stable.strip() else t["text"]
    dynamic_sections = [d["text"] for d in dynamic]
    assembled = compose_with_cache_boundary([stable], dynamic_sections)

    real = _compose_system_prompt(cfg, mode, skip_memory=False)
    exact = assembled == real
    if not exact:
        # Degrade: never misattribute. Fall back to the real text unannotated.
        return {
            "agent": name,
            "mode": mode,
            "exact": False,
            "full": real,
            "chars": len(real),
            "sections": [
                {"source": "unknown", "title": "System prompt", "id": "system-prompt", "disabled": False, "text": real}
            ],
            "disabled_prompt_sections": sorted(disabled),
            "agents_md_sha": _file_sha(home / "AGENTS.md"),
        }
    all_sections = all_parts
    if dynamic:
        # The cache boundary marker separates stable from dynamic in the
        # assembled text. Carry it at the top of the first dynamic section
        # so the UI shows where the boundary sits; note the real composer
        # glues it with single "\n" on each side while sections join with
        # "\n\n" — tests normalize that one known format difference.
        all_sections[-len(dynamic)] = {
            **all_sections[-len(dynamic)],
            "text": f"<!-- SHAREDAGENTS_CACHE_BOUNDARY -->\n\n{all_sections[-len(dynamic)]['text']}",
        }
    return {
        "agent": name,
        "mode": mode,
        "exact": True,
        "full": assembled,
        "chars": len(assembled),
        "sections": all_sections,
        "disabled_prompt_sections": sorted(disabled),
        "agents_md_sha": _file_sha(home / "AGENTS.md"),
    }


__all__ = [
    "AgentEditorError",
    "PATCHABLE_KEYS",
    "UNSET",
    "validate_agent_name",
    "list_agent_defs",
    "model_owner",
    "available_skills",
    "read_definition",
    "read_agents_md",
    "validate_patch",
    "validate_model_membership",
    "patch_config",
    "write_agents_md",
    "render_system_prompt",
]
