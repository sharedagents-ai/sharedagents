"""Factory — build the right :class:`Agent` for a configured name.

The single entry point. Every caller that wants to drive an agent
(gateway, ``/agent`` slash command, ``delegate_agent`` tool) imports
:func:`load_agent` from here. Reading ``~/.sharedagents/agents/<name>/config.json``
and branching on the ``engine`` field is the factory's only
responsibility — no policy lives in the engines themselves.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

from . import default_scaffolds
from .agent import Agent
from .engine import Engine, EngineKind

_LOG = logging.getLogger("sharedagents.agents.factory")


# Map EngineKind → (lazy import path, class name). Lazy so a typo in
# one engine module doesn't break loading agents of a different kind,
# and so we don't pay the import cost for engines an operator never
# uses.
_ENGINE_CLASSES: dict[EngineKind, tuple[str, str]] = {
    EngineKind.LOCAL_LLM: ("sharedagents_agent.agents.local_llm", "LocalLLMEngine"),
    EngineKind.ACP: ("sharedagents_agent.agents.acp_engine", "AcpEngine"),
}


def resolve_engine_kind(raw: str | None) -> EngineKind:
    """Parse a config string into :class:`EngineKind`.

    Tolerates ``None`` / empty string (treated as
    :attr:`EngineKind.LOCAL_LLM` for back-compat with pre-refactor
    configs). Unknown values raise ``ValueError``.
    """
    if not raw:
        return EngineKind.LOCAL_LLM
    raw = str(raw).strip().lower()
    if not raw:
        return EngineKind.LOCAL_LLM
    try:
        return EngineKind(raw)
    except ValueError as exc:
        valid = ", ".join(k.value for k in EngineKind)
        raise ValueError(f"unknown engine kind: {raw!r}. Valid: {valid}") from exc


def _engine_for(kind: EngineKind, cfg: object) -> Engine:
    """Instantiate the right Engine subclass for ``kind``.

    LocalLLMEngine takes the AgentConfig (reads upstream/sampling/
    compaction from it on every turn); the generic ACP engine takes it
    too (reads its ``acp_command`` / model / auth from the agent's
    config). The native-CLI engines take no args.
    """
    if kind in _ENGINE_CLASSES:
        module_path, class_name = _ENGINE_CLASSES[kind]
        adapter = "acp" if kind is EngineKind.ACP else "local"
    else:
        from sharedagents_agent.external_registry import spec_for

        spec = spec_for(kind)
        if not spec.engine_module or not spec.engine_class:
            raise ValueError(f"no engine adapter registered for {kind.value!r}")
        module_path, class_name = spec.engine_module, spec.engine_class
        adapter = spec.adapter
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if kind is EngineKind.LOCAL_LLM:
        return cls(cfg)
    if adapter == "acp":
        return cls(cfg, engine_kind=kind)
    return cls()


def load_agent(name: str, *, config: Any | None = None) -> Agent:
    """Construct the :class:`Agent` for the named directory.

    1. ``AgentConfig.load(name)`` reads
       ``~/.sharedagents/agents/<name>/config.json``. Callers that already own
       the authoritative in-memory config may pass it via ``config``; this
       keeps construction inside the same entry point without rereading a
       potentially stale file.
    2. :func:`resolve_engine_kind` parses ``cfg.engine``.
    3. The matching :class:`Engine` subclass is instantiated.
    4. ``Agent(cfg=cfg, engine=engine)`` is returned.

    Does NOT check CLI availability — :meth:`Engine.is_available` is
    consulted lazily on the first turn so an agent with a missing
    CLI can still be listed in ``/agents`` (with a friendly status).

    Raises ``FileNotFoundError`` when no agent dir / config.json
    exists for ``name``; ``ValueError`` for an unknown engine kind.
    """
    from sharedagents_agent.agent import AgentConfig

    cfg = config if config is not None else AgentConfig.load(name)
    kind = resolve_engine_kind(getattr(cfg, "engine", "") or "")
    engine = _engine_for(kind, cfg)
    return Agent(cfg=cfg, engine=engine)


def ensure_default_external_agents() -> list[str]:
    """Auto-scaffold ``agents/claude/``, ``agents/codex/``, ``agents/antigravity/``.

    Called by the gateway at startup (and by the test suite). For each
    of the three external engines:

    1. If ``~/.sharedagents/agents/<short_name>/config.json`` already
       exists, skip (operator may have customized it).
    2. If the corresponding CLI is missing from PATH, skip — never
       half-scaffold a dir.
    3. Otherwise, write the default scaffold via
       :func:`default_scaffolds.write_scaffold`.

    Returns the list of newly-created agent short names. Idempotent
    on every boot.
    """
    from sharedagents_agent.agent import agents_root
    from sharedagents_agent.tools.runners import _which_cli

    root = agents_root()
    root.mkdir(parents=True, exist_ok=True)
    default_scaffolds.promote_generic_devin_config(root / "devin")
    created: list[str] = []

    for kind_str in default_scaffolds.EXTERNAL_AGENT_KINDS:
        short = default_scaffolds.short_name_for(kind_str)
        details = default_scaffolds._KIND_DETAILS[kind_str]
        cli_cmd = details["cli_command"]
        home = root / short
        if (home / "config.json").exists():
            continue
        if _which_cli(cli_cmd) is None:
            _LOG.info(
                "AGENT_SCAFFOLD_SKIP cli=%s reason=not_on_path",
                cli_cmd,
            )
            continue
        default_scaffolds.write_scaffold(home, kind_str)
        created.append(short)
        _LOG.info(
            "AGENT_SCAFFOLD_CREATED kind=%s home=%s",
            kind_str,
            home,
        )

    return created
