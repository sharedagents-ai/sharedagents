"""Single registry for all built-in external agent integrations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from sharedagents_agent.agents.engine import EngineKind


@dataclass(frozen=True, slots=True)
class ScaffoldKindDetails:
    """On-disk scaffold + detect/install row for one external kind."""

    short_name: str
    display_name: str
    cli_command: str
    install_hint: str
    install_cmd: str


@dataclass(frozen=True, slots=True)
class ScaffoldTemplates:
    """Parameterized content shared by every external-agent scaffold."""

    soul: str
    tools: str

    def render(self, spec: ExternalAgentSpec) -> tuple[str, str]:
        values = {
            "display_name": spec.display_name,
            "cli_command": spec.cli_command,
            "engine": spec.kind.value,
        }
        return self.soul.format(**values), self.tools.format(**values)


DEFAULT_SCAFFOLD_TEMPLATES = ScaffoldTemplates(
    soul="""# SOUL — {display_name} as a sharedagents agent

You are **{display_name}**, driven through the `{cli_command}` integration
registered as `{engine}`. Resume the engine's durable session when available;
do not restate prior turns.

Work as a terminal-native coding agent: inspect the repository, make the
smallest correct change, run relevant verification, and continue until the
acceptance criteria are met or a concrete blocker remains. Keep foreground
work alive until the result is ready; background-only work is not a completed
turn.

Respect project instructions, configured permissions, and the engine's native
tool surface. Ask before destructive operations. Write only inside the named
repository, prefer concise technical output, and report the verification run.
Use sibling delegation only when the handoff explicitly needs another engine.

## ⛔ Protected `main` & Local Checkout Sync (owner ruling 2026-09-16)

Remote `main` is protected and can ONLY be modified via a Pull Request merge; never push directly to `origin/main`. The local `main` checkout may ONLY be synchronized to remote `main` (e.g. `git pull --ff-only origin main` or `git reset --hard origin/main`); never make code edits, staging, commits, or local merges of feature branches on `main`. All work must be developed in an isolated task branch worktree (`.worktrees/<branch>`) and merged via PR. Instructions to "merge to main" mean ensuring the PR is merged on the remote forge, then fast-forward syncing local `main` — never running `git merge` locally on the `main` checkout.
""",
    tools="""# TOOLS — what {display_name} has access to

You run through `{cli_command}` (`{engine}`) and use that engine's native file,
search, terminal, web, agent, skill, and MCP capabilities when advertised.
sharedagents supplies the session transport and permission boundary; never invent
tools or bypass a denied operation.
""",
)


@dataclass(frozen=True, slots=True)
class ExternalAgentSpec:
    """One built-in external engine row in :data:`EXTERNAL_REGISTRY`."""

    kind: EngineKind
    short_name: str
    display_name: str
    cli_command: str
    adapter: Literal["cli", "acp", "rpc"]
    install_hint: str
    install_cmd: str
    provider: str | None = None
    delegate_runner_name: str | None = None
    supports_on_event: bool = False
    install_url: str | None = None
    default_acp_command: tuple[str, ...] | None = None
    mode_map: dict[str, str] | None = None
    default_auth: str | None = None
    default_model: str | None = None
    # Operator-facing selector alias when the engine owns a default that is
    # not a wire model id (for example Antigravity's ``auto`` selection).
    default_model_alias: str | None = None
    default_context_window: int | None = None
    discover_models_name: str | None = None
    discover_models_module: str | None = None
    engine_module: str | None = None
    engine_class: str | None = None
    session_id_key: str | None = None
    # The engine's OWN slash command that compacts its context, sent bare as
    # the whole prompt. ``None``: the engine has no such command, so the text
    # would reach the model as an ordinary request.
    compact_command: str | None = None
    scaffold_templates: ScaffoldTemplates = DEFAULT_SCAFFOLD_TEMPLATES


_ACP_ENGINE = "sharedagents_agent.agents.acp_engine"

_SPECS = (
    ExternalAgentSpec(
        kind=EngineKind.PI_RPC,
        short_name="pi",
        display_name="Pi",
        cli_command="pi",
        adapter="rpc",
        install_hint="https://pi.dev/docs/latest/quickstart",
        install_cmd="npm install -g --ignore-scripts @earendil-works/pi-coding-agent",
        discover_models_name="discover_pi_models",
        discover_models_module="sharedagents_agent.discovery.pi",
        engine_module="sharedagents_orchestrator.engines.pi_engine",
        engine_class="PiRpcEngine",
        compact_command="/compact",
    ),
    ExternalAgentSpec(
        kind=EngineKind.CLAUDE_CLI,
        short_name="claude",
        display_name="Claude Code",
        cli_command="claude",
        adapter="cli",
        install_hint="https://claude.com/code",
        install_cmd="npm install -g @anthropic-ai/claude-code",
        provider="anthropic",
        delegate_runner_name="run_delegate_claude_async",
        supports_on_event=True,
        install_url="https://claude.com/code",
        default_context_window=200_000,
        discover_models_name="discover_anthropic_models",
        discover_models_module="sharedagents_agent.discovery.anthropic",
        engine_module="sharedagents_agent.agents.claude_engine",
        engine_class="ClaudeCliEngine",
        session_id_key="claude_session_id",
        compact_command="/compact",
    ),
    ExternalAgentSpec(
        kind=EngineKind.CODEX_CLI,
        short_name="codex",
        display_name="OpenAI Codex",
        cli_command="codex",
        adapter="cli",
        install_hint="https://github.com/openai/codex",
        install_cmd="npm install -g @openai/codex",
        provider="openai",
        delegate_runner_name="run_delegate_codex_async",
        supports_on_event=True,
        install_url="https://github.com/openai/codex",
        default_context_window=200_000,
        discover_models_name="discover_openai_models",
        discover_models_module="sharedagents_agent.discovery.openai_compat",
        engine_module="sharedagents_agent.agents.codex_engine",
        engine_class="CodexCliEngine",
        session_id_key="codex_thread_id",
    ),
    ExternalAgentSpec(
        kind=EngineKind.CURSOR_ACP,
        short_name="cursor",
        display_name="Cursor",
        cli_command="cursor-agent",
        adapter="acp",
        install_hint="curl https://cursor.com/install -fsS | bash  (then: cursor-agent login)",
        install_cmd="curl https://cursor.com/install -fsS | bash",
        provider="cursor",
        default_acp_command=("cursor-agent", "acp"),
        mode_map={"auto": "agent", "plan": "plan"},
        default_auth="cursor_login",
        default_model_alias="auto",
        default_context_window=1_000_000,
        discover_models_name="discover_cursor_models",
        discover_models_module="sharedagents_agent.discovery.acp_list",
        engine_module=_ACP_ENGINE,
        engine_class="AcpEngine",
    ),
    ExternalAgentSpec(
        kind=EngineKind.OPENCODE_ACP,
        short_name="opencode",
        display_name="OpenCode",
        cli_command="opencode",
        adapter="acp",
        install_hint="https://opencode.ai",
        install_cmd="curl -fsSL https://opencode.ai/install | bash",
        provider="opencode",
        default_acp_command=("opencode", "acp"),
        compact_command="/compact",
        default_auth="",
        default_context_window=200_000,
        discover_models_name="discover_opencode_models",
        discover_models_module="sharedagents_agent.discovery.acp_list",
        engine_module=_ACP_ENGINE,
        engine_class="AcpEngine",
    ),
    ExternalAgentSpec(
        kind=EngineKind.KIMI_ACP,
        short_name="kimi",
        display_name="Kimi Code",
        cli_command="kimi",
        adapter="acp",
        install_hint="https://kimi.ai",
        install_cmd="npm install -g @moonshot-ai/kimi-code",
        provider="kimi",
        default_acp_command=("kimi", "acp"),
        compact_command="/compact",
        mode_map={"auto": "auto", "plan": "plan"},
        default_auth="login",
        default_model="moonshot-ai/kimi-k3",
        default_context_window=1_000_000,
        discover_models_name="discover_kimi_models",
        discover_models_module="sharedagents_agent.discovery.acp_list",
        engine_module=_ACP_ENGINE,
        engine_class="AcpEngine",
    ),
    ExternalAgentSpec(
        kind=EngineKind.HERMES_ACP,
        short_name="hermes",
        display_name="Hermes Agent",
        cli_command="hermes",
        adapter="acp",
        install_hint="https://hermes-agent.nousresearch.com/docs",
        install_cmd="uv tool install hermes-agent[all]",
        provider="hermes",
        default_acp_command=("hermes", "acp"),
        compact_command="/compress",
        mode_map={"auto": "accept_edits", "plan": "default", "ask": "default"},
        default_auth="",
        default_model="anthropic:claude-sonnet-5",
        default_context_window=1_000_000,
        discover_models_name="discover_hermes_models",
        discover_models_module="sharedagents_agent.discovery.acp_list",
        engine_module=_ACP_ENGINE,
        engine_class="AcpEngine",
    ),
    ExternalAgentSpec(
        kind=EngineKind.DEVIN_ACP,
        short_name="devin",
        display_name="Devin",
        cli_command="devin",
        adapter="acp",
        install_hint=("curl -fsSL https://cli.devin.ai/install.sh | bash  (then: devin auth login)"),
        install_cmd="curl -fsSL https://cli.devin.ai/install.sh | bash",
        provider="devin",
        install_url="https://docs.devin.ai/cli",
        default_acp_command=("devin", "acp"),
        compact_command="/compact",
        # ``session/set_mode`` is a no-op. ``auto`` is sharedagents's allow-all
        # picker id; Devin's matching wire id is ``bypass``.
        mode_map={
            "auto": "bypass",
            "plan": "plan",
            "ask": "ask",
            "accept-edits": "accept-edits",
        },
        default_auth="",
        discover_models_name="discover_devin_models",
        discover_models_module="sharedagents_agent.discovery.acp_list",
        engine_module=_ACP_ENGINE,
        engine_class="AcpEngine",
    ),
    ExternalAgentSpec(
        kind=EngineKind.QWEN_ACP,
        short_name="qwen",
        display_name="Qwen Code",
        cli_command="qwen",
        adapter="acp",
        install_hint=("https://qwenlm.github.io/qwen-code-docs/  (routes live in ~/.qwen/settings.json)"),
        install_cmd="npm install -g @qwen-code/qwen-code",
        provider="qwen",
        install_url="https://github.com/QwenLM/qwen-code",
        default_acp_command=("qwen", "--acp"),
        compact_command="/compact",
        # Native ACP modes (``session/set_mode``): plan / default /
        # auto-edit / auto / yolo. sharedagents's allow-all picker id is
        # ``auto``; the wire id matches. Model route is a config-option
        # select (``model``), so the picker id passes through unchanged.
        mode_map={"auto": "auto", "plan": "plan", "ask": "default"},
        # Auth is ``~/.qwen/settings.json`` (security.auth.selectedType +
        # modelProviders openai routes); no CLI login step to run.
        default_auth="",
        default_model="qwen3.8-27b",
        default_context_window=262_144,
        discover_models_name="discover_qwen_models",
        discover_models_module="sharedagents_agent.discovery.acp_list",
        engine_module=_ACP_ENGINE,
        engine_class="AcpEngine",
    ),
    ExternalAgentSpec(
        kind=EngineKind.ANTIGRAVITY_CLI,
        short_name="antigravity",
        display_name="Google Antigravity",
        cli_command="antigravity",
        adapter="cli",
        install_hint="curl -fsSL https://antigravity.google/cli/install.sh | bash",
        install_cmd="curl -fsSL https://antigravity.google/cli/install.sh | bash",
        provider="antigravity",
        delegate_runner_name="run_delegate_antigravity_async",
        supports_on_event=True,
        install_url="https://antigravity.google/docs/cli/install/",
        default_context_window=1_000_000,
        discover_models_name="discover_gemini_models",
        discover_models_module="sharedagents_agent.discovery.google",
        default_model_alias="auto",
        engine_module="sharedagents_agent.agents.antigravity_engine",
        engine_class="AntigravityCliEngine",
        session_id_key="antigravity_session_id",
    ),
    ExternalAgentSpec(
        kind=EngineKind.ZCODE_CLI,
        short_name="zcode",
        display_name="ZCode (Z.AI GLM)",
        cli_command="zcode",
        adapter="cli",
        install_hint="npm install -g zcode-app-cli  (then: zcode login)",
        install_cmd="npm install -g zcode-app-cli",
        provider="zai",
        delegate_runner_name="run_delegate_zcode_async",
        supports_on_event=True,
        install_url="https://z.ai",
        default_context_window=200_000,
        discover_models_name="discover_zcode_models",
        discover_models_module="sharedagents_agent.discovery.zcode",
        engine_module="sharedagents_agent.agents.zcode_engine",
        engine_class="ZCodeCliEngine",
        session_id_key="zcode_session_id",
    ),
)

EXTERNAL_REGISTRY: dict[EngineKind, ExternalAgentSpec] = {spec.kind: spec for spec in _SPECS}
EXTERNAL_AGENT_KINDS: tuple[str, ...] = tuple(spec.kind.value for spec in _SPECS)
# ``acp`` is the user-configurable generic adapter rather than a built-in
# vendor row, but it still owns the model dimension and must follow the same
# external ``/model`` routing rules. Keep this set derived from the registry so
# adding a built-in external row does not require another command-layer edit.
EXTERNAL_ENGINE_VALUES: frozenset[str] = frozenset((EngineKind.ACP.value, *EXTERNAL_AGENT_KINDS))
CLI_DELEGATE_SHORT_NAMES: frozenset[str] = frozenset(spec.short_name for spec in _SPECS if spec.adapter == "cli")


def spec_for(engine_kind: str | Any) -> ExternalAgentSpec:
    kind = engine_kind if isinstance(engine_kind, EngineKind) else EngineKind(str(engine_kind).strip().lower())
    try:
        return EXTERNAL_REGISTRY[kind]
    except KeyError as exc:
        valid = ", ".join(k.value for k in EXTERNAL_REGISTRY)
        raise ValueError(f"unknown external engine kind: {engine_kind!r}. Valid: {valid}") from exc


def spec_for_short_name(short_name: str) -> ExternalAgentSpec:
    raw = str(short_name or "").strip().lower()
    for spec in EXTERNAL_REGISTRY.values():
        if spec.short_name == raw:
            return spec
    raise ValueError(f"unknown external agent short name: {short_name!r}")


def compact_command_for(engine_kind: str | Any) -> str | None:
    """The engine's own compact command, or ``None`` when it has none."""
    try:
        return spec_for(engine_kind).compact_command
    except ValueError:
        return None


def is_compact_command(text: str) -> bool:
    """True when ``text`` is, on its own, some engine's compact command."""
    command = str(text or "").strip()
    return bool(command) and any(spec.compact_command == command for spec in _SPECS)


def compact_unavailable_notice(
    engine_kind: str | Any,
    *,
    has_native_context: bool,
) -> str:
    """Why a compact request cannot go to this engine; ``""`` when it can.

    Sending the command anyway hands it to the model as an ordinary
    request, which it answers (or researches with tools) at full cost
    without compacting anything.
    """
    try:
        spec = spec_for(engine_kind)
    except ValueError:
        return ""
    if not spec.compact_command:
        return f"{spec.display_name} cannot compact on request, so its context was left as it is."
    if not has_native_context:
        return (
            f"{spec.display_name} has nothing of its own to compact in this "
            "conversation right now. Send it a message first."
        )
    return ""


def is_external_engine_kind(engine_kind: str | Any) -> bool:
    """Return whether ``engine_kind`` uses an external model transport.

    The generic ``acp`` adapter has no :class:`ExternalAgentSpec`, so callers
    that need to classify all external engines should use this predicate
    instead of probing ``EXTERNAL_REGISTRY`` directly.
    """
    raw = getattr(engine_kind, "value", engine_kind)
    return str(raw or "").strip().lower() in EXTERNAL_ENGINE_VALUES


def kind_details_for(engine_kind: str | Any) -> ScaffoldKindDetails:
    spec = spec_for(engine_kind)
    return ScaffoldKindDetails(
        short_name=spec.short_name,
        display_name=spec.display_name,
        cli_command=spec.cli_command,
        install_hint=spec.install_hint,
        install_cmd=spec.install_cmd,
    )


def short_name_for(engine_kind: str | Any) -> str:
    return spec_for(engine_kind).short_name


def delegate_runner_for_short_name(short_name: str) -> Callable[..., Any]:
    spec = spec_for_short_name(short_name)
    if spec.adapter != "cli" or not spec.delegate_runner_name:
        raise ValueError(f"{short_name!r} is not a CLI delegate agent")
    import sharedagents_agent.tools.runners as runners

    fn = getattr(runners, spec.delegate_runner_name, None)
    if fn is None:
        raise RuntimeError(f"delegate runner {spec.delegate_runner_name!r} not found for {short_name!r}")
    return fn


__all__ = [
    "CLI_DELEGATE_SHORT_NAMES",
    "EXTERNAL_AGENT_KINDS",
    "EXTERNAL_ENGINE_VALUES",
    "EXTERNAL_REGISTRY",
    "ExternalAgentSpec",
    "ScaffoldTemplates",
    "DEFAULT_SCAFFOLD_TEMPLATES",
    "ScaffoldKindDetails",
    "delegate_runner_for_short_name",
    "kind_details_for",
    "is_external_engine_kind",
    "short_name_for",
    "spec_for",
    "spec_for_short_name",
]
