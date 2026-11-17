"""Per-agent defaulting for the experimental background ``Agent`` mode.

The gateway is a multi-agent process, so its process environment is not an
authoritative place to read an agent-specific switch.  The gateway stamps the
value from the *acting* agent's ``<home>/.env`` onto each tool-runner
subprocess; this module owns that small parsing/stamping contract.

Delegation-mode policy is CONFIGURABLE per agent,
not hardcoded: the per-agent ``config.json`` ``delegation_mode`` field pins
the execution mode for every first-party ``Agent`` call; the applied
profile's ``delegation_mode`` default is the scaffold baseline; everything
else keeps the legacy unpoliced semantics (explicit boolean, then ``.env``,
then foreground).
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, MutableMapping
from pathlib import Path

ENV_NAME = "SHAREDAGENTS_EXPERIMENTAL_BACKGROUND_SUBAGENTS"
_VALID = frozenset({"true", "false"})

# --- per-agent delegation-mode policy ------------
# The execution mode for first-party ``Agent`` calls is CONFIGURABLE per
# agent — NOT hardcoded name sets:
#   1. per-agent ``<home>/config.json`` → ``delegation_mode``
#      (``"background"`` | ``"foreground"`` | ``"default"``) — the operator
#      override, editable via the Agent Builder; ``"default"`` (or absent)
#      means unpoliced;
#   2. the applied profile's ``delegation_mode`` default (an AgentProfile
#      field; plugin agents may declare it in frontmatter) — scaffolded
#      jarvis ⇒ background, build/plan ⇒ foreground;
#   3. legacy unpoliced semantics: explicit ``background`` boolean wins, else
#      the per-agent .env default, else foreground.
# A pinned mode overrides an explicit ``background`` value the model passes,
# with a visible coercion diagnostic. ``runs`` fan-out stays concurrent in
# both pinned modes.
VALID_DELEGATION_MODES: frozenset[str] = frozenset({"background", "foreground", "default"})

_BG_COERCION_NOTE = (
    "[delegation policy: this agent is configured background-only "
    "(delegation_mode=background) — explicit background=false was overridden "
    "to background]"
)
_FG_COERCION_NOTE = (
    "[delegation policy: this agent is configured foreground-only "
    "(delegation_mode=foreground) — explicit background=true was overridden "
    "to foreground]"
)


def _config_delegation_mode(home: str) -> tuple[str | None, str]:
    """Read ``delegation_mode`` from an agent's ``config.json``.

    Returns ``(mode, diagnostic)``: mode is ``"background"``/``"foreground"``
    when pinned, ``"default"`` when the operator EXPLICITLY reset the pin
    (suppresses the profile default — unpoliced), and ``None`` when absent.
    A malformed value is diagnosed and ignored — fail safe: fall through to
    the profile default / legacy resolution.
    """
    if not home:
        return None, ""
    path = Path(home) / "config.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, ""
    raw = data.get("delegation_mode")
    if raw is None or str(raw).strip() == "":
        return None, ""
    value = str(raw).strip().lower()
    if value == "default":
        return "default", ""
    if value in ("background", "foreground"):
        return value, ""
    return None, (
        "[configuration diagnostic: `delegation_mode` must be exactly "
        f"`background`, `foreground`, or `default`; got {raw!r}. Ignoring the "
        "field and falling back to the profile default / legacy mode.]"
    )


def _profile_delegation_mode(name: str) -> str | None:
    """The acting profile's scaffolded delegation-mode default, if any."""
    from sharedagents_agent.agent_profiles import _ALIASES, resolve_profile

    canonical = _ALIASES.get(name, name)
    prof = resolve_profile(canonical)
    if prof is None:
        return None
    return getattr(prof, "delegation_mode", None) or None


def policed_delegation_mode(
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Return ``"background"`` / ``"foreground"`` for a policed caller, else ``None``.

    Resolution mirrors the write gate: the acting agent's home config
    (``$SHAREDAGENTS_AGENT_HOME`` → ``config.json`` ``delegation_mode``) first,
    then the applied profile's default (``$SHAREDAGENTS_AGENT_PROFILE``, falling
    back to ``$SHAREDAGENTS_AGENT_NAME``, canonicalized via the profile alias
    map). An unknown identity is unpoliced. Per-agent config WINS over the
    profile default (operator override).
    """
    source = os.environ if environ is None else environ
    mode, _diag = _config_delegation_mode((source.get("SHAREDAGENTS_AGENT_HOME") or "").strip())
    if mode == "default":
        # Explicit operator reset — the profile default is suppressed too.
        return None
    if mode is not None:
        return mode
    caller = (
        (source.get("SHAREDAGENTS_AGENT_PROFILE") or "").strip()
        or (source.get("SHAREDAGENTS_AGENT_NAME") or "").strip()
    ).lower()
    if not caller:
        return None
    return _profile_delegation_mode(caller)


_CONTROL_ACTIONS = frozenset({"list", "abort", "steer"})


def apply_delegation_background_to_args(
    args: dict,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Rewrite Agent args so displayed/persisted ``background`` matches execution.

    Control actions (list/abort/steer) drop ``background`` — it is not a
    dispatch, and Luna filling the schema default ``false`` made the Web UI
    look like Jarvis waited inline. Dispatch calls get ``background`` set
    to the policed value so an explicit ``background=false`` from a
    background-pinned caller still shows the actual background run.

    Returns the coercion diagnostic (empty when nothing was overridden).
    """
    if not isinstance(args, dict):
        return ""
    action = str(args.get("action") or "dispatch").strip().lower()
    if action in _CONTROL_ACTIONS:
        args.pop("background", None)
        return ""
    explicit = args.get("background") if "background" in args else None
    wants, diag = resolve_background_wants(explicit=explicit, environ=environ)
    args["background"] = wants
    if diag:
        args["_sharedagents_background_config_diagnostic"] = diag
    return diag


def resolve_background_wants(
    *,
    explicit: bool | None,
    environ: Mapping[str, str] | None = None,
) -> tuple[bool, str]:
    """Resolve the execution mode for one first-party ``Agent`` call.

    Precedence (highest first):
      1. per-agent ``delegation_mode`` config (``config.json`` — operator
         override, Agent Builder editable);
      2. the applied profile's delegation-mode default (scaffolded jarvis ⇒
         background, build/plan ⇒ foreground; plugin frontmatter);
      3. explicit ``background`` boolean;
      4. per-agent ``SHAREDAGENTS_EXPERIMENTAL_BACKGROUND_SUBAGENTS`` default;
      5. foreground.

    Returns ``(wants_background, diagnostic)``. The diagnostic is non-empty
    when the runtime overrode an explicit value (visible coercion), ignored a
    malformed ``delegation_mode`` config value, or a malformed .env value
    failed safe to foreground.
    """
    source = os.environ if environ is None else environ
    mode, cfg_diag = _config_delegation_mode((source.get("SHAREDAGENTS_AGENT_HOME") or "").strip())
    if mode == "default":
        # Explicit operator reset — unpoliced; the profile default is
        # suppressed (the operator asked for the legacy semantics).
        mode = None
    elif mode is None:
        # Absent or malformed config — the profile default may apply.
        caller = (
            (source.get("SHAREDAGENTS_AGENT_PROFILE") or "").strip()
            or (source.get("SHAREDAGENTS_AGENT_NAME") or "").strip()
        ).lower()
        if caller:
            mode = _profile_delegation_mode(caller)
    if mode == "background":
        if explicit is False:
            return True, _BG_COERCION_NOTE
        return True, (cfg_diag or "")
    if mode == "foreground":
        if explicit is True:
            return False, _FG_COERCION_NOTE
        return False, (cfg_diag or "")
    # Unpoliced — legacy semantics.
    if explicit is not None:
        return explicit is True, (cfg_diag or "")
    wants, diag = parse_default(environ)
    if cfg_diag:
        diag = f"{cfg_diag} {diag}".strip()
    return wants, diag


def parse_default(
    environ: Mapping[str, str] | None = None,
) -> tuple[bool, str]:
    """Return ``(background_by_default, diagnostic)``.

    Only the exact lower-case values ``true`` and ``false`` are accepted.
    Missing means foreground.  Invalid input fails safe to foreground and
    returns a model/user-visible diagnostic for the dispatch path to attach to
    the foreground result.
    """
    source = os.environ if environ is None else environ
    raw = source.get(ENV_NAME)
    if raw is None or raw == "":
        return False, ""
    if raw == "true":
        return True, ""
    if raw == "false":
        return False, ""
    return False, (
        f"[configuration diagnostic: {ENV_NAME} must be exactly `true` or "
        f"`false`; got {raw!r}. Failing safe to foreground execution.]"
    )


def _value_from_env_file(home: Path) -> str | None:
    """Read the last assignment for :data:`ENV_NAME` from ``home/.env``."""
    path = Path(home) / ".env"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    found: str | None = None
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() != ENV_NAME:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        found = value
    return found


def stamp_for_agent(
    env: MutableMapping[str, str],
    agent_home: Path,
) -> None:
    """Stamp the acting agent's switch into a child-process environment.

    The inherited process-wide value is removed first.  This is the isolation
    boundary that prevents Jarvis's enabled default from leaking into Build,
    Plan, or another profile hosted by the same gateway.
    """
    env.pop(ENV_NAME, None)
    raw = _value_from_env_file(Path(agent_home))
    if raw is not None:
        env[ENV_NAME] = raw


def ensure_default_in_env_file(agent_home: Path, *, enabled: bool) -> bool:
    """Add the switch when absent, preserving all existing file content.

    A valid operator assignment is authoritative and is never replaced.
    Invalid assignments are also left untouched so boot can diagnose them
    visibly instead of silently rewriting operator configuration.  Returns
    ``True`` only when the file was created or extended.
    """
    home = Path(agent_home)
    path = home / ".env"
    try:
        current = path.read_text(encoding="utf-8") if path.exists() else ""
    except OSError:
        return False
    if _value_from_env_file(home) is not None:
        return False
    home.mkdir(parents=True, exist_ok=True)
    prefix = current
    if prefix and not prefix.endswith("\n"):
        prefix += "\n"
    value = "true" if enabled else "false"
    path.write_text(f"{prefix}{ENV_NAME}={value}\n", encoding="utf-8")
    return True


__all__ = [
    "ENV_NAME",
    "VALID_DELEGATION_MODES",
    "ensure_default_in_env_file",
    "parse_default",
    "policed_delegation_mode",
    "resolve_background_wants",
    "apply_delegation_background_to_args",
    "stamp_for_agent",
]
