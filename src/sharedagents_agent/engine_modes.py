"""Server-authoritative agent/engine mode capability registry.

Frontends consume this module's projection and never maintain private mode
maps.  The registry deliberately distinguishes behavioral Plan modes from
permission-only restrictions.
"""

from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class BehaviorKind(str, enum.Enum):
    PROFILE_PLAN = "profile_plan"
    PROFILE_BUILD = "profile_build"
    NATIVE_PLAN = "native_plan"
    NATIVE_BUILD = "native_build"
    NATIVE_ASK = "native_ask"
    PERMISSION_ONLY = "permission_only"


class ControlKind(str, enum.Enum):
    PROFILE = "profile"
    CLI_FLAGS = "cli_flags"
    ACP_MODE = "acp_mode"
    ACP_CONFIG = "acp_config"
    NATIVE_RUNNER = "native_runner"


class EgressClass(str, enum.Enum):
    LOCAL = "local"
    CONFIGURED_PRIVATE = "configured_private"
    FOREIGN = "foreign"


@dataclass(frozen=True, slots=True)
class ModeCapability:
    requested_mode_id: str
    label: str
    behavior_kind: BehaviorKind
    control_kind: ControlKind
    native_value: str
    permission_mode: str
    native_behavior: bool
    supported: bool = True
    unsupported_code: str = ""
    unsupported_reason: str = ""

    def public_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["behavior_kind"] = self.behavior_kind.value
        result["control_kind"] = self.control_kind.value
        result["id"] = self.requested_mode_id
        return result


@dataclass(frozen=True, slots=True)
class AgentModeCapabilities:
    agent_name: str
    engine_kind: str
    default_mode_id: str
    modes: tuple[ModeCapability, ...]
    capability_epoch: str
    egress_class: EgressClass

    def public_dict(self) -> dict[str, Any]:
        available = any(mode.supported for mode in self.modes)
        unavailable = next((mode for mode in self.modes if not mode.supported), None)
        return {
            "agent": self.agent_name,
            "engine": self.engine_kind,
            "available": available,
            "unavailableCode": (unavailable.unsupported_code if unavailable is not None and not available else ""),
            "unavailableReason": (unavailable.unsupported_reason if unavailable is not None and not available else ""),
            "defaultModeId": self.default_mode_id,
            "default_mode_id": self.default_mode_id,
            "availableModes": [mode.public_dict() for mode in self.modes],
            "modes": [mode.public_dict() for mode in self.modes],
            "capabilityEpoch": self.capability_epoch,
            "egressClass": self.egress_class.value,
        }

    def mode(self, requested_mode_id: str | None) -> ModeCapability | None:
        wanted = requested_mode_id or self.default_mode_id
        return next(
            (mode for mode in self.modes if mode.requested_mode_id == wanted),
            None,
        )


def _mode(
    mode_id: str,
    label: str,
    behavior: BehaviorKind,
    control: ControlKind,
    native_value: str,
    permission: str,
    *,
    native: bool,
    supported: bool = True,
    code: str = "",
    reason: str = "",
) -> ModeCapability:
    return ModeCapability(
        mode_id,
        label,
        behavior,
        control,
        native_value,
        permission,
        native,
        supported,
        code,
        reason,
    )


def _configured_acp_command(
    agent_config: Any,
    default: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Return the command the ACP adapter will actually launch.

    ``AgentConfig`` deliberately does not retain arbitrary ``acp_*`` keys, so
    production discovery reads the canonical config file.  The attribute seam
    keeps lightweight callers/tests deterministic without weakening that
    production path.
    """
    configured = getattr(agent_config, "acp_command", None)
    if isinstance(configured, str) and configured.strip():
        return tuple(configured.split())
    if isinstance(configured, (list, tuple)) and configured:
        return tuple(str(item) for item in configured)

    config_path = getattr(agent_config, "config_path", None)
    if config_path is None:
        home = getattr(agent_config, "home", None)
        if home is not None:
            config_path = Path(home) / "config.json"
    if config_path is not None:
        try:
            data = json.loads(Path(config_path).read_text(encoding="utf-8"))
            raw = data.get("acp_command") if isinstance(data, dict) else None
            if isinstance(raw, str) and raw.strip():
                return tuple(raw.split())
            if isinstance(raw, list) and raw:
                return tuple(str(item) for item in raw)
        except (OSError, TypeError, ValueError):
            pass
    return default


def _engine_availability(
    engine: str,
    agent_config: Any,
) -> tuple[bool, str, str]:
    """Read-only launchability probe using execution's canonical resolver."""
    cli = ""
    if engine == "claude_cli":
        cli = "claude"
    elif engine == "codex_cli":
        cli = "codex"
    elif engine == "antigravity_cli":
        cli = "antigravity"
    elif engine == "zcode_cli":
        cli = "zcode"
    elif engine == "pi_rpc":
        cli = "pi"
    elif engine == "cursor_acp":
        command = _configured_acp_command(agent_config, ("cursor-agent", "acp"))
        cli = command[0] if command else ""
    elif engine == "opencode_acp":
        command = _configured_acp_command(agent_config, ("opencode", "acp"))
        cli = command[0] if command else ""
    elif engine == "hermes_acp":
        command = _configured_acp_command(agent_config, ("hermes", "acp"))
        cli = command[0] if command else ""
    elif engine == "kimi_acp":
        command = _configured_acp_command(agent_config, ("kimi", "acp"))
        cli = command[0] if command else ""
    elif engine == "devin_acp":
        command = _configured_acp_command(agent_config, ("devin", "acp"))
        cli = command[0] if command else ""
    elif engine == "qwen_acp":
        command = _configured_acp_command(agent_config, ("qwen", "--acp"))
        cli = command[0] if command else ""
    elif engine == "acp":
        command = _configured_acp_command(agent_config)
        if not command:
            return (
                False,
                "agent_unavailable",
                "ACP adapter has no acp_command configured",
            )
        cli = command[0]
    else:
        return True, "", ""

    from sharedagents_agent.tools.runners import _which_cli

    if cli and _which_cli(cli):
        return True, "", ""
    return (
        False,
        "agent_unavailable",
        f"required adapter CLI `{cli or engine}` is not installed or launchable",
    )


def _with_availability(
    modes: tuple[ModeCapability, ...],
    availability: tuple[bool, str, str],
) -> tuple[ModeCapability, ...]:
    available, code, reason = availability
    if available:
        return modes
    return tuple(
        ModeCapability(
            requested_mode_id=mode.requested_mode_id,
            label=mode.label,
            behavior_kind=mode.behavior_kind,
            control_kind=mode.control_kind,
            native_value=mode.native_value,
            permission_mode=mode.permission_mode,
            native_behavior=mode.native_behavior,
            supported=False,
            unsupported_code=code,
            unsupported_reason=reason,
        )
        for mode in modes
    )


class EngineModeRegistry:
    """Resolve one honest mode matrix for a configured agent."""

    def discover(self, agent_name: str, agent_config: Any) -> AgentModeCapabilities:
        engine = str(getattr(agent_config, "engine", "") or "local_llm")
        default = str(getattr(agent_config, "permission_mode", "") or "auto")
        modes: tuple[ModeCapability, ...]
        name = str(agent_name or "").strip().lower()
        conversational = {"jarvis", "plan", "build", "research"}
        non_conversational = {
            "explore",
            "general",
            "docs",
            "compaction",
            "title",
            "summary",
            # review is auto-only like the other delegation-only agents: it has
            # no mutating reach beyond ``review/`` so an ``ask`` tier adds
            # nothing. This set gates the MODE MATRIX only — it is NOT the WebUI
            # visibility list (``sharedagents_web.agent_visibility``); ``review``
            # must stay absent from THAT one or it vanishes from the picker.
            "review",
        }

        if engine == "local_llm" and name in conversational:
            modes = (
                _mode(
                    "auto", "Allow all", BehaviorKind.PROFILE_BUILD, ControlKind.PROFILE, "auto", "auto", native=False
                ),
                _mode("ask", "Ask", BehaviorKind.NATIVE_ASK, ControlKind.PROFILE, "ask", "ask", native=False),
            )
            default = "auto"
        elif engine == "local_llm" and name in non_conversational:
            modes = (
                _mode(
                    "auto", "Allow all", BehaviorKind.PROFILE_BUILD, ControlKind.PROFILE, "auto", "auto", native=False
                ),
            )
            default = "auto"
        elif engine == "pi_rpc":
            modes = (
                _mode(
                    "auto",
                    "Allow all",
                    BehaviorKind.NATIVE_BUILD,
                    ControlKind.NATIVE_RUNNER,
                    "auto",
                    "auto",
                    native=True,
                ),
            )
            default = "auto"
        elif engine in {
            "cursor_acp",
            "opencode_acp",
            "kimi_acp",
            "hermes_acp",
            "devin_acp",
            "qwen_acp",
            "acp",
            "claude_cli",
            "codex_cli",
            "antigravity_cli",
            "zcode_cli",
        }:
            from sharedagents_agent.acp_mode_dict import (
                catalogue_rows,
                default_requested_mode_for_engine,
            )

            discovered = getattr(agent_config, "discovered_acp_modes", None)
            control = (
                ControlKind.ACP_CONFIG
                if engine in {"opencode_acp", "devin_acp"}
                else ControlKind.CLI_FLAGS
                if engine.endswith("_cli")
                else ControlKind.ACP_MODE
            )
            perm_for = {
                "plan": "plan",
                "ask": "ask",
                "default": "ask",
                "auto": "auto",
                "build": "auto",
                "agent": "auto",
                "dont_ask": "auto",
                "accept_edits": "auto",
                "accept-edits": "auto",
                "bypass": "auto",
            }
            rows: list[ModeCapability] = []
            seen: set[str] = set()
            for item in catalogue_rows(engine):
                nid = str(item.get("id") or "")
                if not nid or nid in seen:
                    continue
                seen.add(nid)
                behavior = (
                    BehaviorKind.NATIVE_PLAN
                    if nid == "plan"
                    else BehaviorKind.NATIVE_ASK
                    if nid in {"ask", "default"}
                    else BehaviorKind.NATIVE_BUILD
                )
                rows.append(
                    _mode(
                        nid,
                        str(item.get("name") or nid),
                        behavior,
                        control,
                        nid,
                        perm_for.get(nid, "auto"),
                        native=True,
                    )
                )
            if discovered:
                for item in discovered:
                    nid = item if isinstance(item, str) else str(item.get("id") or item.get("value") or "").strip()
                    label = nid if isinstance(item, str) else str(item.get("name") or nid)
                    if not nid or nid in seen:
                        continue
                    seen.add(nid)
                    rows.append(
                        _mode(
                            nid,
                            label,
                            BehaviorKind.NATIVE_BUILD,
                            control,
                            nid,
                            perm_for.get(nid, "auto"),
                            native=True,
                        )
                    )
            if not rows:
                rows.append(
                    _mode(
                        "auto",
                        "Allow all",
                        BehaviorKind.NATIVE_BUILD,
                        control,
                        "auto",
                        "auto",
                        native=True,
                    )
                )
            modes = tuple(rows)
            default = default_requested_mode_for_engine(
                engine,
                discovered=discovered,
            )
        else:
            modes = (
                _mode(
                    "auto", "Allow all", BehaviorKind.PROFILE_BUILD, ControlKind.PROFILE, "auto", "auto", native=False
                ),
            )
            default = "auto"

        modes = _with_availability(modes, _engine_availability(engine, agent_config))
        if not any(mode.requested_mode_id == default and mode.supported for mode in modes):
            default = next((mode.requested_mode_id for mode in modes if mode.supported), default)
        epoch_payload = {
            "agent": agent_name,
            "engine": engine,
            "default": default,
            "modes": [mode.public_dict() for mode in modes],
        }
        epoch = hashlib.sha256(json.dumps(epoch_payload, sort_keys=True).encode()).hexdigest()
        return AgentModeCapabilities(
            agent_name,
            engine,
            default,
            modes,
            epoch,
            EgressClass.LOCAL if engine == "local_llm" else EgressClass.FOREIGN,
        )


__all__ = [
    "AgentModeCapabilities",
    "BehaviorKind",
    "ControlKind",
    "EgressClass",
    "EngineModeRegistry",
    "ModeCapability",
]
