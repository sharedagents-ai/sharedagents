"""CLI subcommands for managing agent configuration.

Wired into the top-level ``sharedagents`` parser as the ``agent`` subcommand:

    sharedagents agent init [name]                  # scaffold ~/.sharedagents/agents/<name>/
    sharedagents agent init NAME --engine cursor_acp --model auto
                                               # custom specialist on an external engine
    sharedagents agent list                         # list configured agents + state
    sharedagents agent status [name]                # gateway-pid + ACP initialize probe
    sharedagents agent health [name]                # turn-health stats
    sharedagents agent heartbeat <op> [name]        # heartbeat scheduler config
    sharedagents agent rollback-backup <op> [name]  # snapshot maintenance

Per-agent daemon lifecycle (``start``/``stop``/``restart``/``serve``/
``logs``) is intentionally absent: every agent runs as a stdio child
of the single sharedagents gateway, which is started by systemd
(``sharedagents-gateway.service``). The agent surface here is config-only;
the gateway owns process lifecycle.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

from sharedagents_agent.agent import (
    AgentConfig,
    agents_root,
    derive_upstream_alias,
    global_config_path,
    list_agents,
    load_global_config,
    save_global_config,
)
from sharedagents_agent.agents.default_scaffolds import ensure_voice_agent

# ---- internals ----


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (PermissionError, ProcessLookupError):
        return False


def _say(msg: str) -> None:
    """Write a status line to stderr so it never pollutes -p stdout."""
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


_ECC_VENV_DEPS = ("pyyaml", "python-pptx", "websockets")
_ECC_VENV_SENTINEL_NAME = ".sharedagents-deps-installed"


def _install_ecc_bundle(global_cfg: dict) -> None:
    """Clone `everything-claude-code` into ``~/.sharedagents/ecc/`` and wire
    it into ``skills_dirs`` + ``commands_dirs``. Optional first-run
    wizard step gated by an explicit y/N (default N).

    Performs the smallest install that gets the bundle usable:

    1. ``git clone`` the upstream repo into ``~/.sharedagents/ecc/``
       (keeps ``.git`` so ``git pull`` updates work). Skipped when
       the directory already exists — a partial / manual clone is
       treated as authoritative.
    2. Create a venv at ``~/.sharedagents/ecc/skills/.venv`` using the
       running interpreter (``python -m venv``); installs the three
       deps the ECC skill helpers need (``pyyaml``, ``python-pptx``,
       ``websockets``) and marks success with a sentinel file. A
       previous failed install (venv dir exists but sentinel
       missing) is re-attempted; a fresh dependency added later by
       a sharedagents release can be installed by deleting the sentinel.
    3. ALWAYS reconciles ``skills_dirs`` + ``commands_dirs`` against
       what's actually on disk so a re-run wires existing dirs even
       when the clone was skipped — a manual or partial install no
       longer looks installed-but-not-wired.

    Hard failures raise; the caller downgrades to a warning so the
    wizard doesn't abort.
    """
    import subprocess

    ecc_root = Path.home() / ".sharedagents" / "ecc"
    ecc_root.parent.mkdir(parents=True, exist_ok=True)
    if not ecc_root.exists():
        _say(f"  cloning ECC → {ecc_root} …")
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/affaan-m/everything-claude-code.git",
                str(ecc_root),
            ],
            check=True,
        )
    else:
        _say(f"  ECC dir exists at {ecc_root} — reconciling config without re-cloning")

    venv_root = ecc_root / "skills" / ".venv"
    sentinel = venv_root / _ECC_VENV_SENTINEL_NAME
    if not sentinel.exists():
        try:
            import venv as _venv

            if not venv_root.exists():
                _say(f"  creating skill venv → {venv_root} …")
                _venv.create(str(venv_root), with_pip=True, clear=False)
            else:
                _say(f"  reusing existing venv at {venv_root} (sentinel missing — repairing deps)")
            pip = venv_root / "bin" / "pip"
            if pip.exists():
                rc = subprocess.run(
                    [str(pip), "install", "--quiet", *_ECC_VENV_DEPS],
                    check=False,
                ).returncode
                if rc == 0:
                    sentinel.write_text(
                        f"deps={','.join(_ECC_VENV_DEPS)}\n",
                        encoding="utf-8",
                    )
                else:
                    _say(f"  ⚠ pip install returned {rc} — sentinel NOT written; re-run will retry")
            else:
                _say(f"  ⚠ no pip at {pip} — skills with stdlib-only scripts still work")
        except Exception as exc:  # noqa: BLE001
            _say(f"  ⚠ ECC skill venv setup failed: {exc} (re-run to retry)")

    skills_dir = ecc_root / "skills"
    cmds_dir = ecc_root / "commands"
    legacy_cmds_dir = ecc_root / "legacy-command-shims" / "commands"
    sd = list(global_cfg.get("skills_dirs") or [])
    if str(skills_dir) not in sd and skills_dir.is_dir():
        sd.append(str(skills_dir))
        global_cfg["skills_dirs"] = sd
    cd = list(global_cfg.get("commands_dirs") or [])
    for entry in (cmds_dir, legacy_cmds_dir):
        if entry.is_dir() and str(entry) not in cd:
            cd.append(str(entry))
    if cd:
        global_cfg["commands_dirs"] = cd
    _say(f"  ✓ ECC ready → {ecc_root}")


def _parse_agents_md_name(home: Path) -> str:
    """Extract the operator's name from ``<home>/AGENTS.md`` if pre-filled."""
    p = home / "AGENTS.md"
    if not p.exists():
        return ""
    try:
        body = p.read_text(encoding="utf-8")
    except OSError:
        return ""
    import re as _re

    m = _re.search(r"^- \*\*Name:\*\*\s*(.*)$", body, flags=_re.MULTILINE)
    raw = m.group(1).strip() if m else ""
    if not raw:
        return ""
    lowered = raw.lower().strip("() ")
    if lowered in ("set on first run", "your name", "operator"):
        return ""
    if raw.startswith("(") and raw.endswith(")"):
        return ""
    return raw


_parse_user_md_name = _parse_agents_md_name


def _ask_bool(ask_fn, prompt: str, default: bool) -> bool:
    """Yes/no prompt with explicit `[Y/n]` or `[y/N]` framing."""
    suffix = "[Y/n]" if default else "[y/N]"
    raw = ask_fn(f"  {prompt} {suffix}: ").strip().lower()
    if not raw:
        return default
    if raw in ("y", "yes", "1", "true", "on"):
        return True
    if raw in ("n", "no", "0", "false", "off"):
        return False
    return default


def _offer_gateway_service(ask) -> None:
    """Final wizard step — get the gateway running so setup ends "up".

    The whole point of the wizard is ``pip install sharedagents`` → answer
    prompts → you're running. Without this the operator lands on a
    bare ``sharedagents gateway is not running`` pointer and has to go
    hunting for ``systemctl`` incantations.

    Config is already saved by the time this runs, so this only
    affects whether the gateway is LIVE when the wizard returns. It
    NEVER raises and NEVER blocks the return: on any failure or Ctrl+C
    it degrades to a one-line "start it yourself" pointer.

    First-run is deliberately **sudo-free**: it never installs a system
    service. The gateway is started in the background in user space.
    Survive-reboot via systemd is an explicit opt-in the operator can
    run later (`sharedagents gateway install-service`), not part of setup.

    Resolution:
      1. Gateway already up (systemd unit or CLI-started) → say so.
      2. Otherwise → start the gateway in the BACKGROUND (user space,
         no sudo, no systemd) so this session is up and running.

    ``ask`` is accepted for call-site symmetry with the rest of the
    wizard but is unused now that there's no service prompt.
    """
    from sharedagents_gateway import gateway_cli as gwcli

    _say("")
    _say("  ── Service ──")

    # (1) Already running? _running() tracks the CLI pidfile; a
    # systemd-managed gateway won't show there, so also probe /health.
    try:
        running, pid = gwcli._running()
    except Exception:  # noqa: BLE001
        running, pid = False, None
    if running:
        _say(f"  ✓ sharedagents gateway already running (pid={pid})")
        return
    gw = gwcli._gateway_config()
    url = f"http://{gw['host']}:{gw['port']}"
    try:
        if httpx.get(f"{url}/health", timeout=1.0).status_code == 200:
            _say(f"  ✓ sharedagents gateway already running at {url}")
            return
    except httpx.HTTPError:
        pass

    try:
        # First-run NEVER prompts for sudo. We always bring the gateway
        # up in user space (no root, no systemd) so `pip install sharedagents`
        # → wizard → running needs zero privileges. The survive-reboot
        # path stays available as an explicit, opt-in command
        # (`sharedagents gateway install-service`, systemd) for operators who
        # actually want a boot service.
        res = gwcli.start_gateway_background()
        if res["ok"]:
            if res["already_running"]:
                _say(f"  ✓ gateway already running (pid={res['pid']})")
            else:
                _say(f"  ✓ gateway started (pid={res['pid']}) → {res['url']}")
                if gwcli.systemd_available():
                    _say(
                        "    (background run — won't survive a reboot. For a "
                        "boot service, run `sharedagents gateway install-service`.)"
                    )
        else:
            _say(f"  couldn't start the gateway automatically: {res['error']}")
            _say(
                "    start it yourself with `sharedagents gateway start` "
                "(or `sharedagents gateway install-service` for a boot service)."
            )
    except (EOFError, KeyboardInterrupt):
        _say("")
        _say(
            "  skipped service setup — start the gateway later with "
            "`sharedagents gateway install-service` or `sharedagents gateway start`."
        )
    except Exception as e:  # noqa: BLE001
        _say(f"  service setup hit an error ({e}); start the gateway with `sharedagents gateway start`.")


def _collect_bedrock_row(ask) -> dict | None:
    """Collect one native Amazon Bedrock model row (Converse API).

    Distinct from the OpenAI-compatible provider presets: Bedrock is not an
    HTTP ``/v1`` endpoint, so this branch never probes the network, never
    auto-discovers a model id, and records only the NAME of a bearer-token
    environment variable — never the token value. The row is validated with
    ``validate_bedrock_row`` before it is returned (fail closed).

    Traceability: plan/bedrock-native-converse.md (BED-01, BED-06)
    """
    from sharedagents_model.bedrock_config import (
        AUTH_API_KEY,
        AUTH_IAM,
        AUTH_VALID,
        BEDROCK_API,
        BEDROCK_BACKEND,
        BEDROCK_BASE_SCHEME,
        BEDROCK_BEARER_ENV,
        BedrockConfigError,
        is_valid_bedrock_region,
        validate_bedrock_row,
    )

    _say("  Native Amazon Bedrock (Converse / ConverseStream — no HTTP /v1).")

    # Auth: IAM recommended (this server resolves AWS env credentials) and shown
    # as the default, but the operator must actively choose — never auto-picked.
    auth = (
        ask("  auth [aws-iam] (aws-iam = IAM role/profile; bedrock-api-key = bearer token): ").strip().lower()
        or AUTH_IAM
    )
    while auth not in AUTH_VALID:
        _say(f"    unrecognised auth {auth!r}; choose one of {list(AUTH_VALID)}")
        auth = ask("  auth [aws-iam]: ").strip().lower() or AUTH_IAM

    while True:
        region = ask("  region (e.g. us-east-1): ").strip()
        if not region:
            _say("    region is required for Bedrock — try again, or Ctrl+C.")
            continue
        if not is_valid_bedrock_region(region):
            _say(f"    invalid AWS region {region!r}; expected e.g. us-east-1.")
            continue
        break

    while True:
        model_id = ask("  model id (foundation id or inference-profile ARN): ").strip()
        if model_id:
            break
        _say("    model id is required — no auto-discovery for Bedrock.")

    profile = ""
    if auth == AUTH_IAM:
        profile = ask("  AWS profile [blank = default credential chain]: ").strip()

    api_key_env = ""
    if auth == AUTH_API_KEY:
        api_key_env = ask(f"  apiKeyEnv [{BEDROCK_BEARER_ENV}]: ").strip() or BEDROCK_BEARER_ENV
        # Only the variable NAME is stored; the token value is never prompted
        # for, printed, or written to config.json.
        _say(
            f"    the token value is never stored here — export {api_key_env} "
            "(or put it in the agent's secrets.env) at runtime."
        )

    ctx_default = 200000
    ctx = ctx_default
    raw = ask(f"  contextWindow [{ctx_default}]: ").strip()
    if raw:
        try:
            parsed = int(raw)
            if parsed <= 0:
                raise ValueError
            ctx = parsed
        except ValueError:
            _say(f"    invalid number {raw!r}; using {ctx_default}")
            ctx = ctx_default

    alias_default = model_id.split(":")[0].rsplit("/", 1)[-1].replace(".", "-") or "bedrock"
    alias = ask(f"  alias (short name for /model) [{alias_default}]: ").strip() or alias_default

    row: dict = {
        "alias": alias,
        "api": BEDROCK_API,
        "backend": BEDROCK_BACKEND,
        "provider": BEDROCK_BACKEND,
        "baseUrl": f"{BEDROCK_BASE_SCHEME}://{region}",
        "region": region,
        "id": model_id,
        "auth": auth,
        "contextWindow": ctx,
        "maxTokens": 0,
    }
    if profile:
        row["profile"] = profile
    if api_key_env:
        row["apiKeyEnv"] = api_key_env
    try:
        validate_bedrock_row(row)
    except BedrockConfigError as e:
        _say(f"    {e}")
        return None
    return row


def setup_main_wizard(*, reconfigure: bool = False) -> AgentConfig | None:
    """Interactive setup for the `jarvis` orchestrator — same flow first-run and
    re-run. (`main` was retired; jarvis is the sole orchestrator. The
    function name is kept for its stable CLI entry point.)

    `reconfigure=False` (first-run): empty defaults; called automatically
    when `~/.sharedagents/agents/jarvis/config.json` is missing.

    `reconfigure=True` (re-run): pre-fills every prompt from the existing
    config so a `sharedagents configure` invocation is a quick walk-through
    where Enter keeps each value. If the agent isn't yet configured we
    fall through to first-run behaviour.

    Walks the user through every AgentConfig variable:

    1. Model deployments (alias/baseUrl/id/backend/contextWindow).
    2. Operator name (pre-fills AGENTS.md).
    3. Brave Search API key (optional, powers `web_search` tool).
    4. Telegram bot token + allowed chat_ids (optional).
    5. Local agent port (default 8700, free-port fallback).
    6. Advanced: temperature, enable_thinking, effort, permission_mode,
       compact_at_frac, host, heartbeat_enabled, heartbeat_interval_s.
       Gated behind a single Y/N so first-run users don't have to slog
       through them.

    Returns the saved config, or None if the user aborted.
    """
    if not sys.stdin.isatty():
        _say(
            "agent 'jarvis' is not configured and stdin is not a tty.\n"
            "run `sharedagents agent init` interactively first."
        )
        return None

    import getpass

    import httpx as _httpx

    from sharedagents_agent.agent import personalize_user_md
    from sharedagents_model import providers as prov

    defaults = AgentConfig(name="jarvis", agent_profile="jarvis")

    # When `reconfigure=True` and a config exists on disk, every prompt
    # below pre-fills from it. When the user calls `sharedagents configure`
    # before the first-run install, fall through to empty defaults.
    existing_cfg: AgentConfig | None = None
    if reconfigure:
        try:
            existing_cfg = AgentConfig.load("jarvis")
        except FileNotFoundError:
            existing_cfg = None
    is_rerun = existing_cfg is not None

    default_username = ""
    if is_rerun:
        default_username = _parse_user_md_name(existing_cfg.home)
    if not default_username:
        try:
            default_username = getpass.getuser()
        except Exception:  # noqa: BLE001
            default_username = os.environ.get("USER") or ""

    _say("")
    _say(f"  sharedagents — {'reconfigure' if is_rerun else 'first-run setup'}")
    _say(f"  global config (LLM endpoint / model / backend) → {global_config_path()}")
    _say(f"  per-agent config (port + identity)             → {agents_root() / 'jarvis' / 'config.json'}")
    _say("  Ctrl+C / EOF aborts at any point; no file is written until the end.")
    _say("")

    try:
        # input()'s prompt goes to stdout, so write the prompt to stderr first
        # and read with a blank input() prompt — keeps stdout clean even when
        # the wizard is run inside a stdout-piped context.
        def ask(prompt: str) -> str:
            sys.stderr.write(prompt)
            sys.stderr.flush()
            return input("")

        # ----- 1. Collect models (one or more) -----
        _say("  Backend hints:")
        _say(f"    - vLLM    {prov.VLLM.typical_endpoint_hint}")
        _say(f"    - SGLang  {prov.SGLANG.typical_endpoint_hint}")
        _say("")

        endpoints: list[dict] = []

        def collect_one_endpoint():
            """Collect a single model entry (alias + baseUrl + id +
            backend + contextWindow). Returns a dict ready for
            `models[]`. The wizard loops on this so the operator can
            add as many model deployments as they want.
            """
            from sharedagents_model import provider_presets as pp

            # Provider picker — pick a known OpenAI-compatible provider
            # (pre-fills baseUrl + apiKeyEnv) or 'c' to type a custom URL.
            _say("  Pick a provider:")
            for i, p in enumerate(pp.PROVIDER_PRESETS, 1):
                tail = f"key: ${p.api_key_env}" if p.api_key_env else "local, no key"
                extra = f" — {p.note}" if p.note else ""
                _say(f"    {i:>2}. {p.label:<18} {p.base_url}  ({tail}){extra}")
            _say("     c. Custom OpenAI-compatible endpoint (type the URL)")
            _say("     b. Amazon Bedrock (native Converse API)")
            preset = None
            sel = ask("  provider [c]: ").strip().lower()
            if sel in ("b", "bedrock"):
                while True:
                    bedrock_row = _collect_bedrock_row(ask)
                    if bedrock_row is not None:
                        return bedrock_row
                    _say("    let's try the Bedrock configuration again.")
            if sel.isdigit() and 1 <= int(sel) <= len(pp.PROVIDER_PRESETS):
                preset = pp.PROVIDER_PRESETS[int(sel) - 1]
            elif sel and sel not in ("c", "custom"):
                preset = pp.preset_by_key(sel)  # also accept the key name
                if preset is None:
                    _say(
                        f"    unknown provider {sel!r} — falling back to custom; "
                        "type the OpenAI-compatible baseUrl below."
                    )

            endpoint = ""
            if preset is not None:
                endpoint = preset.base_url.rstrip("/")
                _say(f"    → {preset.label}: {endpoint}")
            while not endpoint:
                raw = ask("  baseUrl: ").strip()
                if not raw:
                    _say("    (required — try again, or Ctrl+C to abort)")
                    continue
                endpoint = raw.rstrip("/")
                if not endpoint.endswith("/v1") and not endpoint.endswith("/openai"):
                    guess = endpoint + "/v1"
                    _say(f"    (no /v1 suffix; will try {guess})")
                    endpoint = guess

            api_key_env_default = (
                preset.api_key_env
                if preset is not None
                else ("OPENROUTER_API_KEY" if "openrouter.ai" in endpoint else "")
            )
            api_key_env = ask(f"  apiKeyEnv [{api_key_env_default or 'blank = none'}]: ").strip() or api_key_env_default
            # Env-key adopt notice (spec-level onboarding UX; plan/fresh-install-onboarding.md
            # §5.2). The secret itself is never stored — only the var name — and is resolved
            # from the environment at request time. So if the chosen provider's key is already
            # exported, tell the user it's adopted (masked, mirroring the Brave-key flow below);
            # otherwise point them at how to set it. This is what makes a box that already has
            # e.g. DEEPSEEK_API_KEY set "just work" without a paste.
            if api_key_env:
                _present = os.environ.get(api_key_env)
                if _present:
                    _mask = _present[:6] + "…" if len(_present) > 6 else "(set)"
                    _say(f"    ✓ detected {api_key_env}={_mask} in the environment — this model will use it")
                else:
                    _say(
                        f"    (no {api_key_env} in the environment yet — export it to authenticate; re-run `sharedagents configure` anytime)"
                    )
            headers: dict[str, str] = {}
            if "openrouter.ai" in endpoint:
                headers["HTTP-Referer"] = ask("  HTTP-Referer [blank = omit]: ").strip()
                headers["X-Title"] = ask("  X-Title [sharedagents]: ").strip() or "sharedagents"
                headers = {k: v for k, v in headers.items() if v}
            probe_headers = dict(headers)
            if api_key_env and os.environ.get(api_key_env):
                probe_headers.setdefault("Authorization", f"Bearer {os.environ[api_key_env]}")

            # Probe + detect backend
            provider = prov.UNKNOWN
            models_data: list[dict] = []
            try:
                provider, models_data = prov.detect_from_endpoint_sync(endpoint, timeout=5.0, headers=probe_headers)
                if provider is prov.UNKNOWN:
                    _say(
                        f"    probed {endpoint}/models — reachable but `owned_by` "
                        f"didn't match vllm/sglang; treating as generic OpenAI-compat."
                    )
                else:
                    _say(f"    detected backend: {provider.label} ✓")
            except _httpx.HTTPError as e:
                _say(f"    couldn't reach {endpoint}/models: {e}")
                _say("    (continuing with manual backend)")

            choice = (
                ask(f"  backend [{provider.name or 'unknown'}] (enter vllm / sglang / unknown to override): ")
                .strip()
                .lower()
                or provider.name
            )
            backend = prov.by_name(choice)

            # Alias — short user-facing handle for `/model <alias>`.
            alias_default = derive_upstream_alias(endpoint)
            alias_prompt = (
                f"  alias (short name for /model) [{alias_default}]: "
                if alias_default
                else "  alias (short name for /model): "
            )
            alias = ask(alias_prompt).strip() or alias_default

            # Model id (the value sent in chat-completions `model` field).
            # Model id is REQUIRED — auto-discovery via `/v1/models`
            # has been removed because it silently bound sessions to
            # whichever model the upstream happened to expose first,
            # which drifted as operators reloaded vLLM/SGLang. Prompt
            # until the operator types something non-empty.
            # The model id shall ALWAYS be operator-
            # supplied, never silently auto-bound to whatever /v1/models
            # happened to return first (that drifts as operators reload
            # vLLM/SGLang). The discovered ids are shown as a convenience
            # list to copy from, but an empty answer re-prompts — it never
            # falls back to a discovered default.
            if models_data:
                _say("    available model ids:")
                for m in models_data[:5]:
                    _say(f"      - {m.get('id')}")
                if len(models_data) > 5:
                    _say(f"      … and {len(models_data) - 5} more")
            prompt = "  id (required): "
            while True:
                ep_model = ask(prompt).strip()
                if ep_model:
                    break
                _say("    model id is required — type one of the listed ids (no auto-discover).")

            # contextWindow — try vLLM's `max_model_len` then SGLang's
            # `/get_model_info`. Prompt with the detected value (or
            # 96000 fallback) so the operator can override but doesn't
            # have to remember the number on a clean install.
            detected_ctx = prov.probe_max_context(endpoint, models_data, headers=probe_headers)
            if detected_ctx:
                _say(f"    detected contextWindow: {detected_ctx:,} tokens ✓")
                ctx_default = detected_ctx
            else:
                _say("    couldn't auto-detect contextWindow (no max_model_len exposed)")
                ctx_default = 96000
            raw_ctx = ask(f"  contextWindow [{ctx_default}]: ").strip()
            try:
                ep_max_ctx = int(raw_ctx) if raw_ctx else ctx_default
                if ep_max_ctx <= 0:
                    raise ValueError
            except ValueError:
                _say(f"    invalid number {raw_ctx!r}; using {ctx_default}")
                ep_max_ctx = ctx_default

            # OpenClaw-aligned shape: each model deployment gets
            # `alias` (sharedagents-specific user handle), `baseUrl` (the
            # OpenAI-compat endpoint), `id` (model id), `backend`
            # (vllm/sglang/unknown — sharedagents concept), `contextWindow`
            # and `maxTokens` (the model's output limit; requests send
            # min(maxTokens, cap); 0 = discover it).
            row = {
                "alias": alias,
                "baseUrl": endpoint,
                "id": ep_model,
                "backend": backend.name,
                "contextWindow": ep_max_ctx,
                "maxTokens": 0,  # operator can fill in later if desired
            }
            if api_key_env:
                row["apiKeyEnv"] = api_key_env
            if headers:
                row["headers"] = headers
            return row

        # On a re-run with existing models, offer to keep them as-is.
        # The model collection sub-wizard probes /v1/models per row, which
        # is slow and unnecessary if you're just here to update telegram.
        if is_rerun and existing_cfg.models:
            _say("  Currently configured models:")
            for i, m in enumerate(existing_cfg.models, 1):
                _say(
                    f"    {i}. alias={m.get('alias', '')!r:<10} "
                    f"baseUrl={m.get('baseUrl', '')} "
                    f"id={m.get('id', '') or '<auto>'} "
                    f"backend={m.get('backend', '') or '<unknown>'} "
                    f"ctx={m.get('contextWindow', 0):,}"
                )
            keep = _ask_bool(ask, "keep these models as-is?", default=True)
            if keep:
                endpoints = [dict(m) for m in existing_cfg.models]
            else:
                # Destructive path: "no" silently dropped ALL prior rows
                # on the wizard's previous shape — an operator who fat-
                # fingered the question lost minutes-to-hours of model
                # collection (real incident). Make the
                # destructive answer require an explicit "wipe" token so
                # an accidental keystroke can't drop the operator's model
                # configuration. The typed-confirmation
                # gate is UNIFORM — it fires whether one row or many would
                # be lost, never gated on a >=2 row-count threshold. A
                # single fat-fingered "n" at n=1 loses the operator's only
                # configured model: structurally identical exposure, so it
                # gets the identical gate.
                _say(f"    ⚠ this will REMOVE all {len(existing_cfg.models)} currently-configured models.")
                confirm = ask("    type 'wipe' to confirm (anything else keeps them): ").strip().lower()
                if confirm != "wipe":
                    _say("    keeping existing models.")
                    endpoints = [dict(m) for m in existing_cfg.models]

        # Collect at least one model deployment, then keep asking until
        # the operator says they're done. No artificial cap — `models[]`
        # is just a JSON list, and the wizard doesn't display indices so
        # the prompt feels the same on the first or the tenth row.
        # `while True` (not `while not endpoints`) so "add another? yes"
        # actually loops — the previous condition exited as soon as the
        # first row landed.
        while True:
            if endpoints and not _ask_bool(ask, "add another model?", default=False):
                break
            ep = collect_one_endpoint()
            endpoints.append(ep)
            _say(f"    → alias={ep['alias']}, baseUrl={ep['baseUrl']}")

        # If the user kept existing rows, still let them add more.
        if is_rerun and existing_cfg and existing_cfg.models and endpoints == [dict(m) for m in existing_cfg.models]:
            while _ask_bool(ask, "add another model deployment?", default=False):
                ep = collect_one_endpoint()
                endpoints.append(ep)
                _say(f"    → alias={ep['alias']}, baseUrl={ep['baseUrl']}")

        # The first row is the primary (used when no --model flag).
        primary = endpoints[0]
        endpoint = primary["baseUrl"]
        alias = primary["alias"]
        model = primary.get("id", "")
        provider = prov.by_name(primary.get("backend", "unknown"))
        primary_max_context = primary.get("contextWindow") or 96000

        # ----- 4. Your name (pre-fills AGENTS.md) -----
        name_prompt = (
            f"  your name (for AGENTS.md) [{default_username}]: "
            if default_username
            else "  your name (for AGENTS.md, blank = leave empty): "
        )
        user_name = ask(name_prompt).strip() or default_username

        # ----- 5. Web search (Brave) — optional -----
        # Brave Search powers the `web_search` tool. Skipping is fine —
        # the tool will surface a friendly "key not configured" error if
        # the model invokes it, and other tools are unaffected. Default
        # is the env var if set, else whatever's in the global config.
        existing = load_global_config()
        # A persisted key wins over a transient ``BRAVE_API_KEY`` env var so
        # reconfigure masks/keeps the value the operator actually saved (and
        # "Enter to keep" can't silently overwrite it with an env value). Env
        # only prefills on first-run, when no key is stored yet.
        brave_default = str(existing.get("brave_api_key") or "") or os.environ.get("BRAVE_API_KEY", "")
        if brave_default:
            # Mask the key in the prompt — show only the first few chars
            # so the user knows it's there without dumping the secret in
            # their terminal scrollback.
            mask = brave_default[:6] + "…" if len(brave_default) > 6 else "(set)"
            brave_prompt = f"  Brave Search API key (web_search tool) [{mask}, Enter to keep]: "
        else:
            brave_prompt = (
                "  Brave Search API key (web_search tool, blank to skip; get one at https://brave.com/search/api): "
            )
        brave_key = ask(brave_prompt).strip() or brave_default

        # ----- 6. Telegram — optional -----
        # Telegram fields live in ``~/.sharedagents/telegram/config.json``
        # — the bot is its own service, not part of any one agent. On
        # both first-run and re-run we pre-fill from there. Old
        # per-agent ``telegram_bot_token`` keys are still honored as a
        # one-time migration so users coming from the coupled layout
        # don't have to retype their token.
        from sharedagents_telegram.config import TelegramConfig as _TgCfg

        _tg_cfg = _TgCfg.load()
        telegram_bot_token = _tg_cfg.bot_token or (
            str(getattr(existing_cfg, "telegram_bot_token", "") or "") if is_rerun else ""
        )
        telegram_allowed_chat_ids = list(_tg_cfg.allowed_chat_ids) or (
            list(getattr(existing_cfg, "telegram_allowed_chat_ids", []) or []) if is_rerun else []
        )

        if telegram_bot_token:
            # Mask the token in the prompt — show only the first/last
            # few chars so the user knows it's there.
            mask = telegram_bot_token[:8] + "…" + telegram_bot_token[-4:]
            tg_want = ask(f"  Telegram bot token [{mask}, Enter to keep]: ").strip()
            if tg_want:
                telegram_bot_token = tg_want
            # Ask about chat_ids if we have a token.
            if telegram_allowed_chat_ids:
                _say(f"    current allowed chat_ids: {telegram_allowed_chat_ids}")
                chat_ids_input = ask("  add more chat_ids (comma-separated, blank to keep current): ").strip()
                if chat_ids_input:
                    try:
                        new_ids = [int(x.strip()) for x in chat_ids_input.split(",") if x.strip()]
                        telegram_allowed_chat_ids = list(set(telegram_allowed_chat_ids + new_ids))
                    except ValueError:
                        _say("    invalid input; keeping current chat_ids")
            else:
                _say("")
                _say("    You'll need to whitelist your Telegram chat_id. The easiest way:")
                _say("      1. Start the agent:  sharedagents agent start")
                _say("      2. Send any message to your bot")
                _say("      3. Check the log:     tail ~/.sharedagents/agents/jarvis/logs/telegram.log")
                _say("         (it will print your chat_id)")
                _say("      4. Re-run this wizard or edit config.json to add it")
                _say("")
        else:
            tg_want = ask("  set up Telegram bot? [y/N] (get a token from @BotFather): ").strip().lower()
            if tg_want in ("y", "yes"):
                _say("")
                _say("    1. Message @BotFather on Telegram and /newbot")
                _say("    2. Follow prompts (name + username)")
                _say("    3. Copy the HTTP API token it gives you")
                _say("")
                while not telegram_bot_token:
                    telegram_bot_token = ask("  Telegram bot token (from @BotFather): ").strip()
                    if not telegram_bot_token:
                        _say("    (required — try again, or Ctrl+C to abort)")
                        continue
                    # Quick format sanity check: BotFather tokens are
                    # <numeric>:<alphanumeric> — reject obviously wrong input.
                    if ":" not in telegram_bot_token:
                        _say("    (doesn't look like a valid token — expected <number>:<string>)")
                        telegram_bot_token = ""

                _say("")
                _say("    After setup, you'll need to whitelist your Telegram chat_id. The easiest way:")
                _say("      1. Start the agent:  sharedagents agent start")
                _say("      2. Send any message to your bot")
                _say("      3. Check the log:     tail ~/.sharedagents/agents/jarvis/logs/telegram.log")
                _say("         (it will print your chat_id)")
                _say("      4. Re-run this wizard or edit config.json to add it")
                _say("")
            else:
                _say("    (skip — you can always add it later with `sharedagents configure` or editing config.json)")
                _say("")

        # ----- 7. Port -----
        # No prompt. Every agent is mounted at ``/acp/<name>`` on the gateway
        # port, so a per-agent
        # `port` is vestigial — asking for it just confused operators
        # into thinking they'd moved the gateway (they hadn't; the
        # gateway reads the `gateway` block of the global config, not
        # `cfg.port`). Keep the field at its default so retired
        # daemon-path readers still get a sane value; to actually move
        # the listener, edit `gateway` in ~/.sharedagents/config.json.
        port = existing_cfg.port if is_rerun else defaults.port

        # ----- 8. Advanced settings -----
        # Gated behind a single Y/N so first-run users don't slog
        # through six more prompts. On re-run the prompt offers to
        # review them; on first-run the defaults are sane.
        adv_label = "review" if is_rerun else "configure"
        show_advanced = _ask_bool(
            ask,
            f"{adv_label} advanced settings (temperature, thinking, "
            "effort, permission mode, compaction, heartbeat, host)?",
            default=False if not is_rerun else False,
        )

        # Defaults: existing config when re-running, dataclass defaults otherwise.
        adv_src = existing_cfg if is_rerun else defaults
        # Temperature is nullable: None means "omit the param, let the
        # upstream default". Preserve an existing null on re-run instead
        # of crashing in float(None) or snapping it back to a number.
        _temp_src = getattr(adv_src, "temperature", defaults.temperature)
        temperature = None if _temp_src is None else float(_temp_src)
        enable_thinking = bool(getattr(adv_src, "enable_thinking", defaults.enable_thinking))
        effort = str(getattr(adv_src, "effort", getattr(defaults, "effort", "")) or "").strip()
        permission_mode = str(getattr(adv_src, "permission_mode", defaults.permission_mode))
        compact_at_frac = float(getattr(adv_src, "compact_at_frac", defaults.compact_at_frac))
        compact_prune_on_load = bool(getattr(adv_src, "compact_prune_on_load", defaults.compact_prune_on_load))
        host = str(getattr(adv_src, "host", defaults.host))
        heartbeat_enabled = bool(getattr(adv_src, "heartbeat_enabled", defaults.heartbeat_enabled))
        heartbeat_interval_s = int(getattr(adv_src, "heartbeat_interval_s", defaults.heartbeat_interval_s))
        # Optional `# SYSTEM` block appended after the .md context. Most
        # users leave it empty.
        system_extra = str(getattr(adv_src, "system", defaults.system))
        description = str(getattr(adv_src, "description", "The default sharedagents agent."))

        if show_advanced:
            _say("")
            _say("  ── Advanced ──")

            # Temperature ('null'/'none' → omit the param, upstream default)
            temp_disp = "null" if temperature is None else str(temperature)
            raw = ask(f"  temperature [{temp_disp}] (or 'null'): ").strip()
            if raw:
                if raw.lower() in ("null", "none"):
                    temperature = None
                else:
                    try:
                        temperature = float(raw)
                    except ValueError:
                        _say(f"    invalid float; keeping {temp_disp}")

            # enable_thinking (Qwen3 chat-template flag)
            enable_thinking = _ask_bool(
                ask,
                "enable_thinking (chat_template_kwargs.enable_thinking for Qwen3-family)?",
                default=enable_thinking,
            )

            raw = ask(f"  effort [{effort or 'default'}] (provider level / default): ").strip()
            if raw:
                from sharedagents_command.commands import normalize_effort

                level = normalize_effort(raw)
                effort = "" if level == "default" else level

            # Permission mode
            raw = ask(f"  permission_mode [{permission_mode}] (plan / ask / auto): ").strip().lower()
            if raw in ("plan", "ask", "auto"):
                permission_mode = raw
            elif raw:
                _say(f"    unrecognised; keeping {permission_mode}")

            # Proactive compaction trigger
            raw = ask(
                f"  compact_at_frac [{compact_at_frac}] (0.85 = squeeze early; 1.0 = wait for overflow): "
            ).strip()
            if raw:
                try:
                    val = float(raw)
                    if 0.0 < val <= 1.0:
                        compact_at_frac = val
                    else:
                        _say(f"    out of range (0..1]; keeping {compact_at_frac}")
                except ValueError:
                    _say(f"    invalid float; keeping {compact_at_frac}")

            # Load-time tool-result pruning (OpenCode ``compaction.prune``).
            from sharedagents_agent.compaction import PRUNE_CACHE_WARNING

            _say(f"  WARNING: {PRUNE_CACHE_WARNING}")
            compact_prune_on_load = _ask_bool(
                ask,
                "compact_prune_on_load (erase old tool results on every session load)?",
                default=compact_prune_on_load,
            )

            # Host (loopback by default — change only if you intend to
            # expose the gateway to the network).
            raw = ask(f"  host [{host}]: ").strip()
            if raw:
                host = raw

            # Heartbeat
            heartbeat_enabled = _ask_bool(
                ask,
                "heartbeat_enabled (background tick into a dedicated session)?",
                default=heartbeat_enabled,
            )
            if heartbeat_enabled:
                raw = ask(f"  heartbeat_interval_s [{heartbeat_interval_s}] (seconds): ").strip()
                if raw:
                    try:
                        val = int(raw)
                        if val > 0:
                            heartbeat_interval_s = val
                        else:
                            _say(f"    must be > 0; keeping {heartbeat_interval_s}")
                    except ValueError:
                        _say(f"    invalid int; keeping {heartbeat_interval_s}")

            # System extra (`# SYSTEM` appended block). Show the current
            # value compressed to one line so the user can decide if they
            # want to clear/replace it. Leave editing the multi-line body
            # to a real editor — the prompt accepts a one-line value or
            # blank-to-keep / "-" to clear.
            sys_show = system_extra.replace("\n", " ⏎ ")[:60]
            sys_label = sys_show or "<empty>"
            raw = ask(f"  system extra (# SYSTEM block) [{sys_label}, Enter to keep, '-' to clear, or new text]: ")
            if raw == "-":
                system_extra = ""
            elif raw:
                system_extra = raw

            # Description (one-liner shown in `agent list`).
            raw = ask(f"  description [{description}]: ").strip()
            if raw:
                description = raw
            _say("")
    except (EOFError, KeyboardInterrupt):
        _say("\n  setup canceled — nothing written.")
        return None

    # Global config: shared across every agent. Preserve any existing
    # global fields the wizard didn't touch (e.g. permission_mode set by
    # editing the JSON manually or by a future wizard pass).
    glob = load_global_config()
    glob.update(
        {
            "upstream_endpoint": endpoint,
            "upstream_model": model,
            "upstream_alias": alias,
            "backend": primary.get("backend") or provider.name,
            # OpenClaw-aligned model deployments. Each row carries its own
            # `contextWindow`, used by the runtime's compaction guard
            # per-session (falls back to `upstream_max_context` for legacy
            # / single-primary configs).
            "models": endpoints,
            # Kept as a top-level fallback for callers that haven't been
            # ported to per-session contextWindow yet (chat_completions
            # passthrough, control_status snapshot).
            "upstream_max_context": primary_max_context,
            # Advanced fields — written even when the user skipped the
            # advanced block, so the saved config matches what was effective
            # at wizard time. Existing values (or dataclass defaults) round-
            # trip cleanly when the user just hits Enter through the prompts.
            "temperature": temperature,
            "effort": effort,
            "enable_thinking": enable_thinking,
            "permission_mode": permission_mode,
            "compact_at_frac": compact_at_frac,
            "compact_prune_on_load": compact_prune_on_load,
            "host": host,
        }
    )
    # Explicitly drop the legacy alias if a previous wizard run wrote it.
    glob.pop("upstream_endpoints", None)
    if brave_key:
        glob["brave_api_key"] = brave_key

    # Default MCP servers (youtube + markitdown) + the skills map —
    # shipped enabled out of the box. Shared with the zero-config
    # quickstart via `_seed_mcp_and_skills` so both first-run paths
    # produce the same baseline. Never clobbers operator config: MCP
    # only-if-absent, skills via setdefault (an explicit `enabled: false`
    # the operator set by hand survives — they keep a kill switch).
    _seed_mcp_and_skills(glob)

    # ----- Teacher-model supervision (optional) -----
    # A larger model that supervises this (small, local) agent: it reviews
    # every session at least once and at failure events, injects
    # continuation prompts to keep the small model working until the task
    # is actually done, and escalates to a stronger model in-place when
    # guidance stalls.
    _teacher_enabled = False
    _say("")
    _say("  ── Teacher model (optional) ──")
    _say(
        "  A larger model that supervises this agent and keeps it working "
        "until the task is actually done (catches premature 'done', "
        "reasoning loops, drift, and stuck-on-error)."
    )
    if _ask_bool(ask, "enable teacher-model supervision?", default=False):
        _say("  Configure the TEACHER endpoint (a stronger model, e.g. a remote DeepSeek / cloud model):")
        try:
            teacher_row = collect_one_endpoint()
            teacher_row["role"] = "teacher"
            # Replace any prior teacher row, then append.
            glob["models"] = [m for m in (glob.get("models") or []) if m.get("role") != "teacher"] + [teacher_row]
            _say("  Supervise on:")
            _say("    1. turn end only (recommended)")
            _say("    2. turn end + failures (premature / loop / tool)")
            _say("    3. + periodic during long tasks (catches drift)")
            mode = ask("  choice [1]: ").strip() or "1"
            triggers = ["Stop", "PrematureTurnEnd", "ReasoningLoopDetected"]
            if mode == "1":
                triggers = ["Stop"]
            if mode in ("2", "3"):
                triggers += ["StopFailure", "SessionError", "PostToolUseFailure"]
            periodic = 5 if mode == "3" else 0
            try:
                max_intv = int(ask("  max guidance steps per task [6]: ").strip() or "6")
            except ValueError:
                max_intv = 6
            escalate = _ask_bool(
                ask,
                "auto-escalate to a stronger model if the small model stays stuck?",
                default=True,
            )
            glob["teacher"] = {
                "enabled": True,
                "triggers": triggers,
                "idle_seconds": 0,
                "periodic_minutes": periodic,
                "max_interventions": max_intv,
                "min_reviews_per_session": 1,
                "stall_limit": 2,
                "tool_failure_threshold": 5,
                "no_result_round_limit": 20,
                "min_confidence": 0.6,
                "timeout_seconds": 45,
                "deep_tier": "auto",
                "escalation": "auto" if escalate else "off",
                "escalation_targets": ["teacher"],
                "restore_model_after_escalation": True,
                "fallback_endpoint": None,
                "on_unreachable": "warn",
            }
            _teacher_enabled = True
            _say("  ✓ teacher model configured (hooks + scan installed after save)")
        except (KeyboardInterrupt, EOFError):
            raise
        except Exception as exc:  # noqa: BLE001
            _say(f"  ⚠ teacher setup skipped: {exc}")

    # Optional ECC (everything-claude-code) bundle install — 228+
    # Claude Code skills covering instincts, memory ops, security
    # scanning, continuous learning, research workflows. Opt-in by
    # design: a fresh user shouldn't be forced to clone a multi-MB
    # third-party repo + create a Python venv before they can chat.
    # Operators who want it can also run ``sharedagents configure`` later
    # OR install manually (clone to ``~/.sharedagents/ecc/``, append
    # ``~/.sharedagents/ecc/skills`` to ``skills_dirs`` in the global
    # config). Default is NO so the wizard stays fast.
    ecc_skills_dir = Path.home() / ".sharedagents" / "ecc" / "skills"
    if ecc_skills_dir.is_dir():
        # Already-present install (manual clone, prior wizard run, or
        # sharedagents configure). Reconcile config so a partial install
        # still gets wired into skills_dirs / commands_dirs.
        try:
            _install_ecc_bundle(glob)
        except Exception as exc:  # noqa: BLE001
            _say(f"  ⚠ ECC reconcile failed: {exc}")
    else:
        _say("")
        _say("  Optional: everything-claude-code (ECC) skill bundle")
        _say(
            "  ~228 Claude Code skills (instincts, memory, security "
            "scanning, …). Adds ~50 MB on disk + a Python venv for "
            "skill scripts. Skip for a leaner install — you can run "
            "the install later via `sharedagents configure` or git-clone "
            "manually."
        )
        choice = ask("  install ECC now? [y/N]: ").strip().lower()
        if choice in ("y", "yes"):
            try:
                _install_ecc_bundle(glob)
            except Exception as exc:  # noqa: BLE001
                _say(f"  ⚠ ECC install failed: {exc}")
                _say("    skipping — you can retry via `sharedagents configure`")

    save_global_config(glob)
    if brave_key:
        _say(f"  ✓ saved global config + Brave key → {global_config_path()}")
    else:
        _say(f"  ✓ saved global config → {global_config_path()}")

    # Teacher supervision is fully NATIVE — there is nothing to install or wire
    # (no hooks, no scripts, no cron). The engine reads the `teacher` policy +
    # the role=="teacher" model row straight from config at runtime.

    # Per-agent config: only what differs between agents (name,
    # description, port, system extra, heartbeat). Telegram fields go
    # to the standalone bot config below.
    cfg = AgentConfig(
        name="jarvis",
        agent_profile="jarvis",
        description=description,
        port=port,
        system=system_extra,
        heartbeat_enabled=heartbeat_enabled,
        heartbeat_interval_s=heartbeat_interval_s,
    )
    cfg.save()
    # Persist the bot-wide Telegram config to its own home. The agent
    # gateway no longer reads these fields; the standalone Telegram
    # service does.
    if telegram_bot_token or telegram_allowed_chat_ids:
        from sharedagents_telegram.config import TelegramConfig as _TgCfg

        tg_cfg = _TgCfg.load()
        if telegram_bot_token:
            tg_cfg.bot_token = telegram_bot_token
        if telegram_allowed_chat_ids:
            tg_cfg.allowed_chat_ids = list(telegram_allowed_chat_ids)
        if not tg_cfg.default_agent:
            tg_cfg.default_agent = "jarvis"
        tg_cfg.save()
        _say(f"  ✓ saved telegram config → {tg_cfg.config_path}")
    ensure_voice_agent()  # scaffolds jarvis AGENTS.md / MEMORY.md
    if user_name and personalize_user_md(cfg, user_name):
        _say(f"  ✓ pre-filled AGENTS.md with name={user_name!r}")
    _say(f"  ✓ saved jarvis orchestrator  → {cfg.config_path}")

    # Offer to wire up external coding agents (Claude Code / Codex / Cursor /
    # Antigravity) as sharedagents delegates — detect what's installed, quick-install
    # the rest, and scaffold each. Best-effort: never blocks the wizard.
    _say("")
    _say("  ── Coding agents ──")
    try:
        setup_external_agents(emit=_say)
    except Exception:  # noqa: BLE001
        _say("  (skipped external-agent setup; run `sharedagents agents` later)")

    # Final step: get the gateway live so the wizard ends "up and
    # running" instead of dumping a `systemctl` to-do on the operator.
    # `ask` is still in scope here (try/except doesn't open a new
    # scope) and the wizard only reaches this point on a clean run.
    _offer_gateway_service(ask)

    # Re-load so the returned cfg reflects the merged (global + per-agent) view.
    return AgentConfig.load("jarvis")


def _seed_mcp_and_skills(glob: dict) -> None:
    """Seed default MCP servers + the skills map into ``glob`` in place.

    Shared by the first-run wizard and the zero-config quickstart so both
    produce the same baseline. Never clobbers operator config: MCP
    defaults only when absent; skills via ``setdefault`` (an explicit
    ``enabled: false`` survives). ``sys.executable`` is the sharedagents venv,
    so the bundled MCP servers import with no PYTHONPATH wrangling.
    """
    import importlib.util as _ilu

    _venv_python = sys.executable
    if not (glob.get("mcpServers") or glob.get("mcp_servers")):
        # Only seed a bundled MCP server when its (optional, `full`-extra)
        # deps are importable — a lean `pip install sharedagents` shouldn't wire
        # up servers that can't spawn. `pip install sharedagents[full]` brings
        # these in; re-running `sharedagents configure` then seeds them.
        servers: dict = {}
        if _ilu.find_spec("yt_dlp") and _ilu.find_spec("youtube_transcript_api"):
            servers["youtube"] = {
                "command": _venv_python,
                "args": ["-m", "sharedagents_mcp.youtube_mcp_server"],
                "env": {},
            }
        if _ilu.find_spec("markitdown_mcp"):
            servers["markitdown"] = {
                "command": _venv_python,
                "args": ["-m", "markitdown_mcp"],
                "env": {},
            }
        if servers:
            glob["mcpServers"] = servers
            glob.pop("mcp_servers", None)
    from sharedagents_skill.skills import discover_skills as _discover_skills

    _skills_map = glob.get("skills")
    if not isinstance(_skills_map, dict):
        _skills_map = {}
    for _s in _discover_skills(include_disabled=True):
        _skills_map.setdefault(_s.name, {"enabled": True})
    if _skills_map:
        glob["skills"] = _skills_map


def _quickstart_status_lines(key_present: bool) -> list[str]:
    """First-run messaging for the free-Nemotron template row.

    A keyless template is NEVER presented as ready-to-chat.
    When the OpenRouter key is present the row works and we say so; when it is
    absent we show actionable remediation (set a key / ``sharedagents configure``)
    so turn one gives guidance, not a cryptic upstream 401. Pure function
    (no I/O) so the outcome is unit-testable at the seam.
    """
    from sharedagents_model import provider_presets as pp

    if key_present:
        return [f"    ({pp.DEFAULT_API_KEY_ENV} detected in the environment — you can chat now)"]
    return [
        "    ⚠ not yet chattable — this row is a keyed template with no key set.",
        "      To start chatting, either:",
        "        • export OPENROUTER_API_KEY=…   (free key: https://openrouter.ai/keys), or",
        "        • run `sharedagents configure`        (pick any provider, or a local model)",
    ]


def _write_quickstart_config() -> AgentConfig | None:
    """Non-interactive first-run: seed the free NVIDIA Nemotron default row
    (OpenRouter, keyed by ``OPENROUTER_API_KEY`` — there is no bundled key).

    The row is always kept. Whether the user can chat immediately
    depends on a key: if ``OPENROUTER_API_KEY`` is already set the row works
    at once; otherwise this function presents it as a *keyed template* and
    steers the user to set a key or run ``sharedagents configure`` — it never
    reports a keyless template as ready-to-chat, so turn
    one surfaces actionable guidance instead of a cryptic upstream 401.
    """
    from sharedagents_model import provider_presets as pp

    glob = load_global_config()
    # The free Nemotron model is ALWAYS kept (added even to a partial pre-existing
    # config). Only become the ACTIVE upstream when none is set — never clobber
    # an operator's existing active model.
    fresh = not glob.get("models")
    glob["models"] = _ensure_free_model(list(glob.get("models") or []))
    if not glob.get("upstream_model"):
        glob.update(
            {
                "upstream_endpoint": pp.DEFAULT_BASE_URL,
                "upstream_model": pp.DEFAULT_MODEL,
                "upstream_alias": pp.DEFAULT_MODEL_ALIAS,
                "backend": "unknown",
                "upstream_max_context": pp.DEFAULT_CONTEXT_WINDOW,
            }
        )
        glob.pop("upstream_endpoints", None)
    _seed_mcp_and_skills(glob)
    save_global_config(glob)
    _say("")
    if fresh:
        _say(f"  ✓ free Nemotron default saved → {global_config_path()}")
        for _line in _quickstart_status_lines(bool(os.environ.get(pp.DEFAULT_API_KEY_ENV))):
            _say(_line)
    else:
        _say(f"  ✓ kept existing LLM config; created jarvis orchestrator → {global_config_path()}")
    cfg = AgentConfig(name="jarvis", description="sharedagents jarvis orchestrator", agent_profile="jarvis")
    cfg.save()
    ensure_voice_agent()  # scaffolds jarvis
    _say(f"  ✓ saved jarvis orchestrator → {cfg.config_path}")
    _offer_gateway_service(lambda *_a, **_k: "")
    return AgentConfig.load("jarvis")


def _env_llm_endpoint_and_model() -> tuple[str, str] | None:
    """Return ``(endpoint, model)`` from a scripted / Docker env, or None.

    Official names win; the shorter ``LLM_*`` aliases match the isolated
    container deployment so ``pip install sharedagents`` + those two env
    vars is a complete non-interactive first run.
    """
    endpoint = (os.environ.get("SHAREDAGENTS_UPSTREAM_ENDPOINT") or os.environ.get("LLM_ENDPOINT") or "").strip()
    model = (os.environ.get("SHAREDAGENTS_UPSTREAM_MODEL") or os.environ.get("LLM_MODEL") or "").strip()
    if endpoint and model:
        return endpoint, model
    return None


def _upsert_models_row(models: list, *, model_id: str, base_url: str) -> list:
    """Insert or update the operator's local/OpenAI-compatible model row.

    The configured model is the picker default (index 0 + ``default: true``).
    The always-kept free cloud row stays in the list, but it must not steal
    first-chat routing after ``configure --endpoint/--model``.
    """
    out: list = []
    found = None
    for raw in models:
        if not isinstance(raw, dict):
            out.append(raw)
            continue
        row = dict(raw)
        if str(row.get("id") or "") == model_id:
            row["baseUrl"] = base_url
            row.setdefault("provider", "Local vLLM")
            row.setdefault("backend", "vllm")
            row["default"] = True
            found = row
        else:
            row.pop("default", None)
            out.append(row)
    if found is None:
        found = {
            "id": model_id,
            "baseUrl": base_url,
            "provider": "Local vLLM",
            "backend": "vllm",
            "default": True,
        }
    return [found, *out]


def apply_endpoint_model_config(
    endpoint: str,
    model: str,
    *,
    extras: dict | None = None,
    start_gateway: bool = True,
) -> AgentConfig | None:
    """Non-interactive bootstrap against a known OpenAI-compatible endpoint.

    This is the scripted ``pip install sharedagents`` path: ``sharedagents configure
    --endpoint URL --model NAME`` (or first-run with
    ``SHAREDAGENTS_UPSTREAM_ENDPOINT`` / ``LLM_ENDPOINT`` set). Writes the
    global config (active upstream + ``models[]`` + MCP/skills), scaffolds
    jarvis, and optionally starts the paired stack so the next
    ``sharedagents -p`` / web turn can talk to the model.
    """
    endpoint = (endpoint or "").strip()
    model = (model or "").strip()
    if not endpoint or not model:
        raise ValueError("endpoint and model are both required")

    glob = load_global_config()
    glob.update(
        {
            "upstream_endpoint": endpoint,
            "upstream_model": model,
            "upstream_alias": derive_upstream_alias(endpoint),
            "backend": glob.get("backend") or "vllm",
        }
    )
    if extras:
        for key, value in extras.items():
            if value is not None:
                glob[key] = value
    if not glob.get("upstream_max_context"):
        glob["upstream_max_context"] = 256000
    glob["models"] = _ensure_free_model(
        _upsert_models_row(list(glob.get("models") or []), model_id=model, base_url=endpoint)
    )
    glob.pop("upstream_endpoints", None)
    _seed_mcp_and_skills(glob)
    path = save_global_config(glob)
    _say(f"  ✓ endpoint {endpoint} model {model} → {path}")

    ensure_voice_agent()
    cfg = AgentConfig.load("jarvis")
    _say(f"  ✓ saved jarvis orchestrator → {cfg.config_path}")
    if start_gateway:
        _offer_gateway_service(lambda *_a, **_k: "")
    return cfg


def _ensure_free_model(models: list) -> list:
    """Guarantee the free cloud model is present in ``models`` — it is ALWAYS
    kept (unless the operator explicitly removes it). Returns the list with the
    free row prepended when it was absent, unchanged otherwise.
    """
    from urllib.parse import urlparse

    from sharedagents_model import provider_presets as pp

    def _is_openrouter(url: str) -> bool:
        # Inlined after provider_presets.is_openrouter was removed with the
        # bundled key. Host-equality, not substring,
        # so a look-alike host can't masquerade as the free default row.
        try:
            return (urlparse(url or "").hostname or "").lower() == "openrouter.ai"
        except ValueError:
            return False

    has_free = any(
        isinstance(m, dict) and str(m.get("id")) == pp.DEFAULT_MODEL and _is_openrouter(str(m.get("baseUrl") or ""))
        for m in models
    )
    # Append, never prepend: a just-configured local/vLLM row must remain
    # the picker default. Only the free row's existence is required.
    out = list(models) if has_free else [*models, pp.default_model_row()]
    # Direct-xAI Grok rows ride the same "only where the provider really
    # resolves a key" gate as AgentConfig.load — the setup write must never
    # force xAI onto a deployment that runs on another provider alone.
    return pp.ensure_xai_grok_models(out, api_keys=_shared_secret_keys())


def _shared_secret_keys() -> dict[str, str]:
    """The install-wide ``~/.sharedagents/secrets.env`` key map (empty on failure).

    The CLI writes the GLOBAL config, so only the SHARED secrets layer is
    consulted (``name=""`` → no per-agent file); process env stays the
    fallback inside ``xai_key_configured()``. A bad/unreadable secrets file
    must never break first-run, so it degrades to env-only.
    """
    try:
        from sharedagents_agent.agent import _load_agent_secrets, sharedagents_root

        return _load_agent_secrets(sharedagents_root(), "")
    except Exception:  # noqa: BLE001 - setup continues on env keys alone
        return {}


def _write_local_gpu_config(setup) -> AgentConfig | None:
    """Attach ``main`` to a local model on the detected GPU(s), KEEPING the free
    cloud model alongside it (always-kept default).

    ``setup`` is a :class:`sharedagents_model.local_gpu.LocalSetup`. The local server
    row (no API key, 127.0.0.1) becomes the active upstream; the free model
    stays in ``models[]`` as a fallback. Only sets the active upstream on a
    fresh config (never clobbers an existing one)."""
    row = setup.row
    glob = load_global_config()
    fresh = not glob.get("models")
    models = _ensure_free_model(list(glob.get("models") or []))
    if not any(m.get("baseUrl") == row["baseUrl"] and m.get("id") == row["id"] for m in models):
        models.append(row)
    glob["models"] = models
    if fresh:
        glob.update(
            {
                "upstream_endpoint": row["baseUrl"],
                "upstream_model": row["id"],
                "upstream_alias": row["alias"],
                "backend": row["backend"],
                "upstream_max_context": row["contextWindow"],
            }
        )
        glob.pop("upstream_endpoints", None)
    _seed_mcp_and_skills(glob)
    save_global_config(glob)
    verb = "provisioned" if setup.provisioned else "found running"
    _say("")
    _say(f"  ✓ detected {setup.profile.short_name()} — using your local GPU")
    if fresh:
        _say(f"  ✓ {verb} local model '{row['id']}' ({setup.server.backend}) → {row['baseUrl']}")
        _say(f"    saved → {global_config_path()}  (change anytime with `sharedagents configure`)")
    else:
        _say(f"  ✓ kept existing LLM config; created jarvis orchestrator → {global_config_path()}")
    cfg = AgentConfig(name="jarvis", description="sharedagents jarvis orchestrator", agent_profile="jarvis")
    cfg.save()
    ensure_voice_agent()
    _say(f"  ✓ saved jarvis orchestrator → {cfg.config_path}")
    _offer_gateway_service(lambda *_a, **_k: "")
    return AgentConfig.load("jarvis")


def _add_local_provision(profile, choice, model_id: str):
    """Kick off a local model install in a DETACHED background shell and return
    the (pending) local ``models[]`` row to add ALONGSIDE the free model.

    The free cloud model stays the active upstream — the local row is just
    registered so the user can ``/model``-switch to it once it's serving. The
    install runs as a parallel shell logging to ``~/.sharedagents/provision.log``,
    watchable via ``sharedagents monitor``. Returns ``(row, started)``.
    """
    import subprocess

    from sharedagents_model import local_gpu as lg

    base_url = f"http://127.0.0.1:{lg._provision_port()}/v1"
    served = model_id.split("/")[-1].lower()
    cw = min(choice.context_window, 32_768)
    row = {
        "alias": served,
        "baseUrl": base_url,
        "id": served,
        "backend": "vllm",
        "contextWindow": cw,
        "maxTokens": 0,
        "apiKeyEnv": "",
        "headers": {"X-Title": "sharedagents"},
    }
    log = os.path.expanduser("~/.sharedagents/provision.log")
    # Write the marker BEFORE spawning: the worker re-derives what to install
    # but clears the marker on success / sets state=error on failure, so it must
    # never race a not-yet-written marker (which would let the parent overwrite
    # the worker's error with a stale "provisioning" → a stuck indicator).
    lg.write_provision_marker(model_id, base_url, served)
    # Tell the worker EXACTLY which model to install (the recommended best, or a
    # repo the user typed). The worker re-derives via _provision_model_id, which
    # honors SHAREDAGENTS_PROVISION_MODEL — so this pins the wizard's choice and keeps
    # the marker, the models[] row, and the actual download all in agreement.
    env = dict(os.environ)
    env["SHAREDAGENTS_PROVISION_MODEL"] = model_id
    try:
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "ab") as fh:
            subprocess.Popen(
                [sys.executable, "-m", "sharedagents_model.provision_worker"],
                stdout=fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=env,
            )
    except (OSError, subprocess.SubprocessError) as exc:
        # Nothing started → undo the marker so the user never sees an unusable
        # local model in /model and the TUI/monitor don't poll forever.
        lg.clear_provision_marker()
        _say(f"  ✗ couldn't start the local install: {exc}")
        return None, False
    return row, True


def _ask_local_model(profile, recommended_id: str) -> str | None:
    """Opt-in local-GPU model prompt.

    Recommends sharedagents's best model for the detected GPU (``recommended_id``) but
    lets the user install ANY HuggingFace repo instead — pressing Enter / ``y``
    takes the recommended best, typing a repo id (``owner/name``) installs that,
    ``n`` skips. Returns the model id to install, or None to skip. The launcher
    re-derives the serving recipe (tool/reasoning parser) from whatever model is
    chosen, so a user-supplied model still serves correctly. Non-interactive →
    None (never installs).
    """
    if not (getattr(sys, "stdin", None) and sys.stdin.isatty()):
        return None
    _say(f"  Found {profile.short_name()}.")
    _say("  sharedagents can install vLLM (if needed) + download a local model in a")
    _say("  background shell while you keep chatting on the free model")
    _say("  (watch with `sharedagents monitor`; switch with `/model` when it's ready).")
    _say(f"    ★ recommended best model for this GPU:  {recommended_id}")
    try:
        ans = input(
            "    Install?  [Y]=recommended · type a HuggingFace repo id for a different model · [n]=skip:  "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        return None
    low = ans.lower()
    if low in ("n", "no"):
        return None
    if low in ("", "y", "yes"):
        return recommended_id
    # Anything else → a user-supplied model id (HF repo or local tag). A plain
    # word with no "/" is almost certainly a typo, not a repo — fall back to the
    # recommended model rather than launch a doomed download.
    if "/" not in ans:
        _say(f"    '{ans}' isn't a HuggingFace repo id (owner/name) — installing the recommended model instead.")
        return recommended_id
    _say(f"    → installing your model:  {ans}")
    return ans


def _confirm_found_model(server) -> bool:
    """Ask whether to adopt an ALREADY-serving local model as the active default.

    The running model is the PRE-FILLED default: pressing Enter (or ``y``)
    adopts it as the active upstream; ``n`` keeps the free cloud model. Either
    way the local model is registered in ``models[]`` and switchable via
    ``/model`` — this only decides which one is *active* out of the gate.

    Non-interactive or EOF → ``True`` (accept the pre-filled default), matching
    the always-up first-run contract (a piped/headless launch never blocks)."""
    if not (getattr(sys, "stdin", None) and sys.stdin.isatty()):
        return True
    _say(f"  Found a local model already serving: {server.model_id}")
    _say(f"    at {server.base_url}")
    try:
        ans = input("    Use it as your default model?  [Y]=yes · [n]=keep the free model:  ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return True
    return ans not in ("n", "no")


def _guided_first_run() -> AgentConfig | None:
    """Interactive first-run wizard.

    The free cloud model is the default and is ALWAYS kept (chat works
    immediately). Then, opt-in (never automatic), offer to set up a local GPU
    model — which installs vLLM + downloads weights in a PARALLEL background
    shell while the user keeps chatting on the free model, switchable with
    ``/model`` once it's serving. Finally, offer coding agents + the gateway.
    """
    from sharedagents_model import local_gpu as lg, provider_presets as pp

    glob = load_global_config()
    # Whether a real active upstream pre-existed BEFORE the free-model default
    # below sets one. Used so adopting a found local model overrides only the
    # free default we're about to write, never an operator's existing choice.
    had_upstream = bool(glob.get("upstream_model"))
    # 1) Free cloud model — the always-kept default. Added if missing (even for
    #    a partial pre-existing config); active upstream only when none is set.
    models = _ensure_free_model(list(glob.get("models") or []))
    if not glob.get("upstream_model"):
        glob.update(
            {
                "upstream_endpoint": pp.DEFAULT_BASE_URL,
                "upstream_model": pp.DEFAULT_MODEL,
                "upstream_alias": pp.DEFAULT_MODEL_ALIAS,
                "backend": "unknown",
                "upstream_max_context": pp.DEFAULT_CONTEXT_WINDOW,
            }
        )
        glob.pop("upstream_endpoints", None)
    _say("")
    _say("  sharedagents first-run setup")
    # The free row is always added, but it is a keyed
    # template — only report "chat now" when a key actually resolves. Both first-run
    # paths (this interactive one and _write_quickstart_config) share the single
    # tested _quickstart_status_lines() decision so neither can drift.
    _say(f"  ✓ free model row added — {pp.DEFAULT_MODEL_ALIAS} (always kept)")
    for _line in _quickstart_status_lines(bool(os.environ.get(pp.DEFAULT_API_KEY_ENV))):
        _say(_line)

    # 2) Local GPU model — OPT-IN. Never installs without an explicit yes.
    provision_started = False
    try:
        profile = lg.gpu_profile()
        if profile is not None:
            server = lg.probe_local_server()
            if server is not None:
                cw = server.max_model_len or 32_768
                row = lg._model_row(server, cw)
                if not any(m.get("baseUrl") == row["baseUrl"] and m.get("id") == row["id"] for m in models):
                    models.append(row)
                # The found model is the pre-filled default — let the user adopt
                # it (Enter) or keep the free model (n). Only override the free
                # default we just set; never clobber a pre-existing upstream.
                if _confirm_found_model(server) and not had_upstream:
                    glob.update(
                        {
                            "upstream_endpoint": row["baseUrl"],
                            "upstream_model": row["id"],
                            "upstream_alias": row["alias"],
                            "backend": row["backend"],
                            "upstream_max_context": row["contextWindow"],
                        }
                    )
                    glob.pop("upstream_endpoints", None)
                    _say(f"  ✓ using local model '{row['id']}' as your default → {row['baseUrl']}")
                else:
                    _say(f"  ✓ local model '{row['id']}' added — switch anytime with `/model`")
            else:
                choice = lg.select_model_for_profile(profile)
                recommended = lg._provision_model_id(choice, profile) if choice else None
                model_id = _ask_local_model(profile, recommended) if recommended else None
                if model_id:
                    row, provision_started = _add_local_provision(profile, choice, model_id)
                    if (
                        provision_started
                        and row
                        and not any(m.get("baseUrl") == row["baseUrl"] and m.get("id") == row["id"] for m in models)
                    ):
                        models.append(row)
    except Exception:  # GPU detection must never break first-run
        pass

    glob["models"] = models
    _seed_mcp_and_skills(glob)
    save_global_config(glob)
    cfg = AgentConfig(name="jarvis", description="sharedagents jarvis orchestrator", agent_profile="jarvis")
    cfg.save()
    ensure_voice_agent()
    _say(f"  ✓ saved jarvis orchestrator → {cfg.config_path}")

    # 3) Coding agents — opt-in (setup_external_agents prompts per agent).
    _say("")
    _say("  ── Coding agents ──")
    try:
        setup_external_agents(emit=_say)
    except Exception:  # noqa: BLE001
        _say("  (skipped — run `sharedagents agents` later)")

    # 4) Gateway up so the wizard ends running.
    _offer_gateway_service(lambda *_a, **_k: "")

    if provision_started:
        _say("")
        _say("  ⏳ Your local model is installing in a background shell.")
        _say("     Watch:  sharedagents monitor")
        _say("     When it's ready, switch to it with:  /model")
    return AgentConfig.load("jarvis")


def _first_run_fork() -> AgentConfig | None:
    """First run: interactive → the guided wizard (free model default + opt-in
    local GPU model + coding agents). Non-interactive (piped/Docker) → attach to
    an already-running local server if present, else the free cloud default.
    NEVER installs anything without an explicit prompt.
    """
    interactive = bool(getattr(sys, "stdin", None)) and sys.stdin.isatty()
    if interactive:
        return _guided_first_run()
    # Scripted / Docker: a caller that already named a reachable upstream
    # (``sharedagents configure --endpoint/--model`` or the matching env vars)
    # must not fall through to the keyless OpenRouter template.
    env_pair = _env_llm_endpoint_and_model()
    if env_pair is not None:
        return apply_endpoint_model_config(*env_pair, start_gateway=True)
    # Non-interactive: never prompt, never install. Attach to a running local
    # server if there is one (not an install); otherwise the free cloud model.
    try:
        from sharedagents_model import local_gpu as lg

        profile = lg.gpu_profile()
        if profile is not None:
            setup = lg.attach_setup(profile)
            if setup is not None:
                return _write_local_gpu_config(setup)
    except Exception:  # detection must never break first-run
        pass
    return _write_quickstart_config()


def ensure_main_configured() -> AgentConfig | None:
    """Make sure the ``main`` agent is configured on disk.

    If not configured, run the first-run fork: zero-config free-Nemotron
    quickstart or the provider wizard. Never spawns the gateway here —
    callers (TUI self-bootstrap / ``_offer_gateway_service``) own that.
    """
    try:
        return AgentConfig.load("jarvis")
    except FileNotFoundError:
        pass  # expected on first run — fall through so any first-run error
        # surfaces on its own, not chained off this FileNotFoundError
    return _first_run_fork()


def gateway_alive() -> bool:
    """True iff the sharedagents gateway is reachable.

    Primary signal: ``$SHAREDAGENTS_HOME/gateway.pid`` → live pid.
    Fallback: ``GET /health`` against the configured bind URL —
    covers a missing pidfile while the listener is otherwise
    healthy.
    """
    from .gateway import _gateway_config, _gateway_pidfile

    pidfile = _gateway_pidfile()
    if pidfile.is_file():
        try:
            pid = int(pidfile.read_text().strip())
            if _is_alive(pid):
                return True
        except (OSError, ValueError):
            pass
    gw = _gateway_config()
    try:
        r = httpx.get(f"http://{gw['host']}:{gw['port']}/health", timeout=1.0)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


# ---- commands ----


def cmd_init(args: argparse.Namespace) -> int:
    # external-engine agents via write_custom_external_scaffold; else a
    # local_llm config dir.
    name = args.name or "jarvis"
    engine = (getattr(args, "engine", None) or "").strip().lower()
    model = (getattr(args, "model", None) or "").strip()
    description = (getattr(args, "description", None) or "").strip()

    if name == "jarvis":
        if engine:
            print(
                "jarvis is the local orchestrator — omit --engine "
                "(or pick a different agent name for a custom specialist)",
                file=sys.stderr,
            )
            return 2
        ensure_voice_agent()
        cfg = AgentConfig.load("jarvis")
        print(f"agent {cfg.name} ready at {cfg.home}")
        return 0

    home = agents_root() / name
    if (home / "config.json").exists():
        print(f"agent {name} already configured at {home}")
        return 0

    if engine:
        from sharedagents_agent.agents import default_scaffolds as ds
        from sharedagents_agent.agents.engine import EngineKind
        from sharedagents_agent.agents.factory import resolve_engine_kind

        # Accept both kind values (cursor_acp) and short names (cursor).
        kind: EngineKind | None = None
        try:
            kind = resolve_engine_kind(engine)
        except ValueError:
            for k, details in ds._KIND_DETAILS.items():
                if details.get("short_name") == engine:
                    kind = EngineKind(k)
                    break
        if kind is None:
            valid = ", ".join(f"{d['short_name']}|{k}" for k, d in ds._KIND_DETAILS.items())
            print(
                f"unknown engine {engine!r}. Use an external kind or short name: {valid}, or acp.",
                file=sys.stderr,
            )
            return 2
        if kind.value == "local_llm":
            print(
                "--engine local_llm is the default; omit --engine for a "
                "local agent, or pass an external kind "
                "(cursor_acp, claude_cli, codex_cli, antigravity_cli, "
                "opencode_acp, kimi_acp, hermes_acp, devin_acp, qwen_acp, acp).",
                file=sys.stderr,
            )
            return 2
        written = ds.write_custom_external_scaffold(
            home,
            kind.value,
            name=name,
            model=model or "auto",
            description=description or f"{name} — custom agent on {kind.value}",
        )
        print(f"agent {name} scaffolded at {home} (engine={kind.value}, model={model or 'auto'})")
        if written:
            print(f"  wrote: {', '.join(written)}")
        print("  Restart the gateway to mount it:  sharedagents gateway restart")
        return 0

    cfg = AgentConfig(
        name=name,
        description=description or f"sharedagents agent: {name}",
    )
    cfg.save()
    print(f"agent {name} scaffolded at {cfg.home} (engine=local_llm)")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List configured agents + their gateway-mount status.

    Every agent is a stdio child of the sharedagents gateway, mounted at
    ``/acp/<name>``. ``running`` means the gateway is alive AND
    answers ``initialize`` for that mount; ``stopped`` means the
    gateway is down (no agent is reachable until the gateway boots).
    """
    names = list_agents()
    if not names:
        print("no agents configured. run `sharedagents agent init` to scaffold the default `jarvis` orchestrator.")
        return 0

    from .gateway import _gateway_config, _gateway_pidfile

    gw = _gateway_config()
    gw_pidfile = _gateway_pidfile()
    gw_pid: int | None = None
    if gw_pidfile.is_file():
        try:
            candidate = int(gw_pidfile.read_text().strip())
            if _is_alive(candidate):
                gw_pid = candidate
        except (OSError, ValueError):
            gw_pid = None
    gw_alive = gateway_alive()
    gw_pid_label = str(gw_pid) if gw_pid is not None else "-"
    gateway_base = f"http://{gw['host']}:{gw['port']}"
    headers: dict[str, str] = {}
    if gw.get("auth_token"):
        headers["Authorization"] = f"Bearer {gw['auth_token']}"

    print(f"{'NAME':<14} {'STATE':<10} {'GW_PID':<8} {'MOUNT'}")
    for name in names:
        if not gw_alive:
            print(f"{name:<14} {'stopped':<10} {'-':<8} {gateway_base}/acp/{name}")
            continue
        state = "unhealthy"
        try:
            r = httpx.post(
                f"{gateway_base}/acp/{name}/acp",
                content=json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {"protocolVersion": 1, "clientCapabilities": {}},
                    }
                ).encode("utf-8"),
                headers={**headers, "Content-Type": "application/json"},
                timeout=2.0,
            )
            if r.status_code == 200:
                state = "running"
        except httpx.HTTPError:
            pass
        print(f"{name:<14} {state:<10} {gw_pid_label:<8} {gateway_base}/acp/{name}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Print agent status by hitting the gateway.

    Every agent is mounted at ``/acp/<name>`` on the north gateway. We probe
    ``initialize`` against that URL to surface the agent's ``agentInfo`` and
    ``agentCapabilities``. Pid/uptime fields refer to the gateway
    itself, not a per-agent process — there isn't one.
    """
    name = args.name or "jarvis"
    try:
        cfg = AgentConfig.load(name)
    except FileNotFoundError:
        sys.stderr.write(f"agent {name!r} not configured.\n")
        return 2

    from .gateway import _gateway_config, _gateway_pidfile

    gw = _gateway_config()
    gw_pidfile = _gateway_pidfile()
    gw_pid: int | None = None
    if gw_pidfile.is_file():
        try:
            candidate = int(gw_pidfile.read_text().strip())
            if _is_alive(candidate):
                gw_pid = candidate
        except (OSError, ValueError):
            gw_pid = None
    # If pidfile is missing / stale, fall back to the HTTP /health
    # probe (covers a systemd-managed gateway whose pidfile got
    # raced — the listener is still healthy).
    if gw_pid is None and not gateway_alive():
        print(f"agent {name}: gateway not running")
        print(f"  home: {cfg.home}")
        return 1

    try:
        uptime_s = max(0.0, time.time() - gw_pidfile.stat().st_mtime)
    except OSError:
        uptime_s = 0.0
    gateway_base = f"http://{gw['host']}:{gw['port']}"
    agent_url = f"{gateway_base}/acp/{name}"
    out: dict = {
        "agent": name,
        "gateway_pid": gw_pid,
        "gateway_url": gateway_base,
        "uptime_s": round(uptime_s, 1),
        "url": agent_url,
        "acp_url": f"{agent_url}/acp",
        "home": str(cfg.home),
        "upstream_endpoint": getattr(cfg, "upstream_endpoint", ""),
        "upstream_model": getattr(cfg, "upstream_model", ""),
    }
    headers = {"Content-Type": "application/json"}
    if gw.get("auth_token"):
        headers["Authorization"] = f"Bearer {gw['auth_token']}"
    try:
        r = httpx.post(
            f"{agent_url}/acp",
            content=json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": 1,
                        "clientCapabilities": {},
                    },
                }
            ).encode("utf-8"),
            headers=headers,
            timeout=3.0,
        )
        if r.status_code == 200:
            body = r.json()
            result = body.get("result") if isinstance(body, dict) else None
            if isinstance(result, dict):
                out["agent_info"] = result.get("agentInfo")
                out["agent_capabilities"] = result.get("agentCapabilities")
                out["protocol_version"] = result.get("protocolVersion")
            out["acp_ok"] = True
        else:
            out["acp_ok"] = False
            out["acp_status_code"] = r.status_code
    except httpx.HTTPError as e:
        out["acp_ok"] = False
        out["acp_error"] = str(e)
    print(json.dumps(out, indent=2))
    return 0 if out.get("acp_ok") else 1


def _rollback_backup_dirs(home) -> list:
    """Return rollback-backup directories sorted oldest-first by name.
    Names are timestamps (`YYYYMMDDTHHMMSSZ`) so lexical sort = chrono."""
    root = home / ".rollback-backup"
    if not root.exists():
        return []
    return sorted(
        [p for p in root.iterdir() if p.is_dir()],
        key=lambda p: p.name,
    )


def _du(path) -> int:
    """Recursive byte size of `path`. Best-effort — silently skips
    files we can't stat. Used by `rollback-backup list` so an operator
    can decide whether a `clean` is worth running."""
    total = 0
    for sub in path.rglob("*"):
        try:
            if sub.is_file():
                total += sub.stat().st_size
        except OSError:
            continue
    return total


def cmd_rollback_backup_list(args: argparse.Namespace) -> int:
    name = args.name or "jarvis"
    try:
        cfg = AgentConfig.load(name)
    except FileNotFoundError:
        sys.stderr.write(f"agent {name!r} not configured.\n")
        return 2
    dirs = _rollback_backup_dirs(cfg.home)
    if not dirs:
        print(f"agent {name}: no rollback-backup snapshots in {cfg.home}/.rollback-backup")
        return 0
    print(f"agent {name}: {len(dirs)} snapshot(s)")
    for p in dirs:
        n_files = sum(1 for _ in p.iterdir() if _.is_file())
        print(f"  {p.name}  {n_files} files  {_du(p)}b  ({p})")
    return 0


def cmd_rollback_backup_clean(args: argparse.Namespace) -> int:
    """Keep the `--keep` most-recent snapshots, delete the rest."""
    import shutil

    name = args.name or "jarvis"
    try:
        cfg = AgentConfig.load(name)
    except FileNotFoundError:
        sys.stderr.write(f"agent {name!r} not configured.\n")
        return 2
    keep = max(0, int(args.keep))
    dirs = _rollback_backup_dirs(cfg.home)
    if len(dirs) <= keep:
        print(f"agent {name}: {len(dirs)} snapshot(s) — ≤ keep={keep}, nothing to delete")
        return 0
    to_delete = dirs[: len(dirs) - keep]
    for p in to_delete:
        try:
            shutil.rmtree(p)
            print(f"  removed {p.name}")
        except OSError as e:
            sys.stderr.write(f"  failed to remove {p}: {e}\n")
    print(f"agent {name}: kept {keep} most-recent, removed {len(to_delete)} older snapshot(s)")
    return 0


def _toggle_heartbeat(args: argparse.Namespace, enabled: bool) -> int:
    """Mutate `heartbeat_enabled` in the per-agent config. Effect is
    deferred to the next agent restart — the scheduler is wired up at
    `serve()` boot, not on a config-watch loop, so there's no
    middle-of-the-tick toggle that could leave a partial heartbeat
    in flight.
    """
    name = args.name or "jarvis"
    try:
        cfg = AgentConfig.load(name)
    except FileNotFoundError:
        sys.stderr.write(f"agent {name!r} not configured.\n")
        return 2
    cfg.heartbeat_enabled = enabled
    interval = getattr(args, "interval_s", None)
    if interval is not None and interval > 0:
        cfg.heartbeat_interval_s = int(interval)
    cfg.save()
    state = "enabled" if enabled else "disabled"
    print(
        f"agent {name}: heartbeat {state}  "
        f"(interval={cfg.heartbeat_interval_s}s).  "
        f"Restart agent for change to take effect:\n"
        f"  sharedagents agent restart {name}"
    )
    return 0


def cmd_heartbeat_enable(args: argparse.Namespace) -> int:
    return _toggle_heartbeat(args, enabled=True)


def cmd_heartbeat_disable(args: argparse.Namespace) -> int:
    return _toggle_heartbeat(args, enabled=False)


def cmd_heartbeat_status(args: argparse.Namespace) -> int:
    """Print scheduler config + persisted state. No HTTP call — works
    even when the gateway is stopped.
    """
    name = args.name or "jarvis"
    try:
        cfg = AgentConfig.load(name)
    except FileNotFoundError:
        sys.stderr.write(f"agent {name!r} not configured.\n")
        return 2
    from sharedagents_gateway.heartbeat import (
        is_heartbeat_content_effectively_empty,
        load_state,
        read_heartbeat_md,
        state_path,
    )

    print(f"agent: {name}")
    print(f"heartbeat_enabled: {cfg.heartbeat_enabled}")
    print(f"heartbeat_interval_s: {cfg.heartbeat_interval_s}")
    state = load_state(cfg.home)
    sp = state_path(cfg.home)
    print(f"state file: {sp} ({'present' if sp.exists() else 'absent'})")
    if state:
        for k in ("session_id", "created_at", "last_run_at"):
            if state.get(k):
                print(f"  {k}: {state[k]}")
        progress = state.get("session_progress") or {}
        if progress:
            print(f"  session_progress entries: {len(progress)}")
    body = read_heartbeat_md(cfg.home)
    if body is None:
        print("HEARTBEAT.md: missing")
    else:
        empty = is_heartbeat_content_effectively_empty(body)
        marker = "EFFECTIVELY EMPTY (ticks SKIP)" if empty else "non-empty (ticks RUN)"
        print(f"HEARTBEAT.md: {len(body)} chars — {marker}")
    return 0


def cmd_heartbeat_fire(args: argparse.Namespace) -> int:
    """Deprecated — the in-process heartbeat scheduler is retired.

    Periodic / scheduled-job work belongs to an external scheduler
    that drives ACP ``session/prompt`` against the running agent.
    """
    name = args.name or "jarvis"
    sys.stderr.write(
        f"agent {name}: `heartbeat fire` is deprecated — the in-process "
        f"heartbeat scheduler is retired. Drive periodic work through "
        f"ACP from an external scheduler. No tick fired.\n"
    )
    return 1


def cmd_health(args: argparse.Namespace) -> int:
    """Report turn-health stats from the agent's saved sessions.

    Aggregates the `failure_modes` field that `_record_failure_mode`
    populates on every turn — `loop_detected`, `empty_stop`,
    `thinking_only_stop`, `length_continue`, `tool_loop_cap`,
    `compaction_summarised`, `forced_plan_pivot`, etc. Without this,
    the only way to tell whether a runtime change is moving the needle
    is to grep session JSONs by hand.

    Defaults to the last 50 turns across the most-recently-modified
    sessions in the agent's `sessions/` directory. Outputs a small
    table (mode, count, % of turns) plus a per-mode list of recent
    session IDs so an operator can drill into one.
    """
    name = args.name or "jarvis"
    try:
        cfg = AgentConfig.load(name)
    except FileNotFoundError:
        sys.stderr.write(f"agent {name!r} not configured.\n")
        return 2
    sessions_dir = cfg.sessions_dir
    if not sessions_dir.exists():
        print(f"(no sessions dir at {sessions_dir})")
        return 0
    # turn-level failure signals live in sessions.db ``turn_events``
    # (loop / error / nudge / distill / finish), projected from the
    # JSONL event logs. Refresh the index, then aggregate.
    from sharedagents_session.session_index import SessionIndex, connect_sessions_db

    idx = SessionIndex(sessions_dir)
    idx.refresh()
    limit = max(1, int(args.limit or 50))
    _FAIL_KINDS = ("loop", "error", "nudge", "distill", "tool_result_fail")
    counts: dict[str, int] = {}
    sample_sids: dict[str, list[str]] = {}
    turns_seen = 0
    con = connect_sessions_db(idx.db_path)
    try:
        row = con.execute(
            "SELECT COALESCE(SUM(CASE WHEN last_turn_idx>=0 THEN last_turn_idx+1 ELSE 0 END),0) n FROM sessions_meta"
        ).fetchone()
        turns_seen = int(row["n"]) if row else 0
        ph = ",".join("?" for _ in _FAIL_KINDS)
        cur = con.execute(
            f"SELECT kind, session_hash FROM turn_events WHERE kind IN ({ph}) ORDER BY ts DESC LIMIT ?",
            (*_FAIL_KINDS, limit * 20),
        )
        for r in cur.fetchall():
            k = r["kind"]
            counts[k] = counts.get(k, 0) + 1
            lst = sample_sids.setdefault(k, [])
            sid = r["session_hash"]
            if sid not in lst and len(lst) < 5:
                lst.append(sid)
    finally:
        con.close()
    if turns_seen == 0:
        print("(no turns found)")
        return 0
    total_fail = sum(counts.values())
    print(f"agent {name}: turn health ({turns_seen} turn(s) indexed)")
    print(f"  failure/recovery events: {total_fail}")
    print()
    if not counts:
        print("  (clean — no recovery paths fired)")
        return 0
    width = max(len(k) for k in counts) + 2
    print(f"  {'mode':<{width}} count  % of turns  recent sessions")
    for mode, cnt in sorted(counts.items(), key=lambda kv: -kv[1]):
        pct = 100 * cnt / turns_seen
        sids = ",".join(s[:12] for s in sample_sids.get(mode, []))
        print(f"  {mode:<{width}} {cnt:>5}  {pct:>9.1f}%  {sids}")
    return 0


# ---- external coding agents: detect + quick-install + configure ----


def _ask_yn(prompt: str, default_yes: bool = True) -> bool:
    suffix = "[Y/n]" if default_yes else "[y/N]"
    try:
        sys.stderr.write(f"  {prompt} {suffix} ")
        sys.stderr.flush()
        ans = input("").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return default_yes
    if not ans:
        return default_yes
    return ans in ("y", "yes")


def _ensure_user_npm_prefix(em) -> str | None:
    """Make ``npm install -g`` work without root by pointing npm at a
    user-writable prefix when its current one isn't writable.

    Returns the bin dir (``<prefix>/bin``) the installer should prepend
    to PATH so ``shutil.which(<cli>)`` sees the freshly installed
    binary, or ``None`` when npm isn't on PATH.

    On most non-root desktops + every fresh Docker image, npm defaults
    to ``/usr/local`` (or ``/usr``) and ``npm install -g`` EACCES's. We
    set ``prefix=$HOME/.npm-global`` in the user's npmrc — same shape as
    the well-known sudo-free workaround — and never touch system
    config. If the existing prefix IS writable, we leave it alone.
    """
    import os
    import shutil
    import subprocess
    from pathlib import Path

    npm = shutil.which("npm")
    if not npm:
        return None

    def _bin(prefix: str) -> str:
        return str(Path(prefix) / "bin")

    try:
        cp = subprocess.run(  # noqa: S603
            [npm, "config", "get", "prefix"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        current = (cp.stdout or "").strip() if cp.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        current = ""

    # If the current prefix dir (or its parent) is user-writable, npm
    # will already work. We still report its bin dir so the caller can
    # PATH-prepend.
    if current and current not in ("undefined", "null"):
        probe = Path(current)
        if probe.exists() and os.access(probe, os.W_OK):
            return _bin(current)
        if not probe.exists():
            parent = probe.parent
            if parent.exists() and os.access(parent, os.W_OK):
                return _bin(current)

    # Switch to ~/.npm-global (sudo-free, predictable, picked up by the
    # gateway unit's ``_npm_prefix_bins`` fallback list).
    target = Path.home() / ".npm-global"
    target.mkdir(parents=True, exist_ok=True)
    (target / "bin").mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(  # noqa: S603
            [npm, "config", "set", "prefix", str(target)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        em(f"  warning: could not set npm prefix to {target}: {exc}")
        return None
    em(f"  configured npm prefix → {target} (sudo-free global installs)")
    return _bin(str(target))


def _maybe_refresh_gateway_unit(em) -> None:
    """Re-render + reload the user-instance gateway unit so the new
    install location lands on the gateway service's PATH.

    Best-effort. Skipped when no unit is installed (the gateway will
    pick up the new PATH the next time the user runs ``sharedagents gateway
    install-service``). Errors are surfaced as one-liners — never fatal
    to the wizard.
    """
    import subprocess
    from pathlib import Path

    try:
        from sharedagents_gateway import gateway_cli as gw
    except ImportError:
        return

    unit_path = Path.home() / ".config" / "systemd" / "user" / "sharedagents-gateway.service"
    if not unit_path.exists():
        return  # nothing to refresh

    try:
        new_unit = gw.render_gateway_unit()
    except Exception as exc:  # noqa: BLE001
        em(f"  warning: could not re-render gateway unit: {exc}")
        return
    if unit_path.read_text(encoding="utf-8") == new_unit:
        return  # PATH already current

    try:
        unit_path.write_text(new_unit, encoding="utf-8")
    except OSError as exc:
        em(f"  warning: could not write {unit_path}: {exc}")
        return
    em(f"  refreshed {unit_path.name} (PATH now includes new CLI install dirs)")

    # daemon-reload + restart so the gateway picks up the new PATH now,
    # not on next manual restart. Silent on success — only emit on
    # failure (the user can run `sharedagents gateway restart` themselves).
    try:
        subprocess.run(  # noqa: S603, S607
            ["systemctl", "--user", "daemon-reload"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        cp = subprocess.run(  # noqa: S603, S607
            ["systemctl", "--user", "restart", "sharedagents-gateway.service"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if cp.returncode == 0:
            em("  restarted sharedagents-gateway.service")
        else:
            em(
                f"  note: gateway restart returned {cp.returncode}; "
                "run `sharedagents gateway restart` to pick up the new PATH"
            )
    except (OSError, subprocess.SubprocessError) as exc:
        em(f"  note: could not restart gateway ({exc}); run `sharedagents gateway restart` to pick up the new PATH")


def setup_external_agents(*, auto: bool = False, do_setup: bool = True, emit=None) -> bool:
    """Detect external coding-agent CLIs (Claude Code / Codex / Cursor /
    Antigravity), print status, and — unless ``do_setup`` is False — quick-install
    the missing ones and configure each as an sharedagents delegate. ``auto`` runs
    non-interactively (install + configure all). ``emit`` is the output sink
    (defaults to stderr ``_say`` for the wizard; the CLI passes ``print``).
    Returns True if any agent was newly configured.

    Shared by ``sharedagents agents`` and the first-run wizard.
    """
    import os
    import shutil
    import subprocess

    from sharedagents_agent.agent import agents_root
    from sharedagents_agent.agents import default_scaffolds as ds

    em = emit or _say
    rows = ds.detect_external_agents()
    em("External coding agents (drive them via `/<name> <task>` or as delegates):")
    for r in rows:
        inst = "✓ installed" if r["installed"] else "✗ not installed"
        conf = "✓ configured" if r["configured"] else "— not configured"
        em(f"  {r['display_name']:<18} {r['cli_command']:<14} {inst:<16} {conf}")

    if not do_setup:
        return False

    interactive = auto or (bool(getattr(sys, "stdin", None)) and sys.stdin.isatty())
    if not interactive:
        em("  (run `sharedagents agents` in a terminal, or with --yes, to install/configure)")
        return False

    # ``npm install -g`` needs a sudo-free prefix, but probing/creating it is
    # an install side effect — it must run ONLY after an opt-in is accepted, so
    # a declined install runs nothing. Prepare it lazily on first confirmed
    # install and cache the result for the rest of the loop.
    install_env = os.environ.copy()
    _prep = {"done": False}

    def _prepare_install_env() -> None:
        if _prep["done"]:
            return
        _prep["done"] = True
        npm_bin = _ensure_user_npm_prefix(em)
        if npm_bin and npm_bin not in install_env.get("PATH", "").split(os.pathsep):
            install_env["PATH"] = npm_bin + os.pathsep + install_env.get("PATH", "")

    changed = False
    installed_any = False
    for r in rows:
        if r["configured"]:
            continue
        name, disp = r["short_name"], r["display_name"]
        if not r["installed"]:
            cmd = r["install_cmd"]
            if not cmd:
                em(f"  {disp}: install manually → {r['install_hint']}")
                continue
            if not (auto or _ask_yn(f"{disp} not installed. Install it now?  ({cmd})", default_yes=False)):
                continue
            # Opt-in accepted → now (and only now) run the npm-prefix side effect.
            _prepare_install_env()
            em(f"  installing {disp} … ({cmd})")
            try:
                rc = subprocess.run(cmd, shell=True, env=install_env).returncode  # noqa: S602
            except (OSError, subprocess.SubprocessError) as exc:
                rc = 1
                em(f"  ✗ could not run installer: {exc}")
            # Re-probe with the install env's PATH so a fresh
            # ~/.npm-global/bin/<cli> isn't invisible to this lookup.
            cli_path = shutil.which(r["cli_command"], path=install_env.get("PATH"))
            if rc != 0 or cli_path is None:
                em(f"  ✗ {disp} install failed — install manually: {r['install_hint']}")
                continue
            r["installed"] = True
            installed_any = True
            em(f"  ✓ {disp} installed at {cli_path}")
        # Installed (now or already) but not configured → scaffold the agent.
        if r["installed"] and (auto or _ask_yn(f"Configure sharedagents agent '{name}'?")):
            ds.write_scaffold(agents_root() / name, r["kind"])
            changed = True
            em(f"  ✓ configured '{name}' — use it with `/{name} <task>`")

    if installed_any:
        # New CLIs landed; the running gateway's PATH may be stale.
        # Re-render + restart the user-instance unit so the gateway can
        # actually spawn them.
        _maybe_refresh_gateway_unit(em)
    if changed:
        em("  Restart the gateway to mount the new agent(s):  sharedagents gateway restart")
    return changed


def cmd_agents(args: argparse.Namespace) -> int:
    """`sharedagents agents` — detect / quick-install / configure external coding
    agents. ``--list`` just prints status; ``--yes`` is non-interactive."""
    setup_external_agents(
        auto=bool(getattr(args, "yes", False)),
        do_setup=not bool(getattr(args, "list", False)),
        emit=print,
    )
    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    """`sharedagents monitor` — watch the background local-model install (the shell
    started by the first-run wizard). Live mode shows the download/load progress
    bar until the model is serving (or fails); ``--once`` prints one snapshot
    plus a tail of the vLLM server log."""
    import time

    from sharedagents_model import local_gpu as lg

    vlog = os.path.expanduser("~/.sharedagents/vllm-server.log")
    once = bool(getattr(args, "once", False))

    def render() -> bool:
        st = lg.provision_status()
        if st is None:
            print("No local-model provisioning in progress.")
            return False
        if st.get("server_up"):
            print(f"✓ local model ready: {st.get('served')} → {st.get('base_url')}")
            print("  switch to it with:  /model")
            return False
        if st.get("phase") == "error":
            print(f"✗ provisioning failed: {st.get('error') or 'unknown'}")
            print(f"  see {vlog}")
            return False
        sys.stdout.write("\r" + (st.get("bar") or "  provisioning…").ljust(72))
        sys.stdout.flush()
        return True

    if once:
        render()
        sys.stdout.write("\n")
        tail = lg._log_tail(vlog, 4000)
        if tail:
            print("--- vLLM log (tail) ---")
            print("\n".join(tail.splitlines()[-15:]))
        return 0
    print("Watching local-model provisioning (Ctrl+C to stop)…")
    try:
        while render():
            time.sleep(2.0)
        sys.stdout.write("\n")
    except KeyboardInterrupt:
        sys.stdout.write("\n")
    return 0


# ---- argparse wiring ----


def add_monitor_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "monitor",
        help="Watch the background local-model install (download bar + log)",
    )
    p.add_argument("--once", action="store_true", help="Print one snapshot + log tail instead of watching live")
    p.set_defaults(_dispatch=cmd_monitor)


def add_agents_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "agents",
        help="Detect, quick-install, and configure external coding agents (Claude Code / Codex / Cursor / Antigravity)",
    )
    p.add_argument("--list", "-l", action="store_true", help="Just print detection status; don't install/configure")
    p.add_argument("--yes", "-y", action="store_true", help="Non-interactive: install + configure everything known")
    p.set_defaults(_dispatch=cmd_agents)


def cmd_orchestrator_serve(args: argparse.Namespace) -> int:  # noqa: ARG001
    """``sharedagents orchestrator serve`` — run the SOUTH orchestrator service.

    Owns the agents / SessionWriter / SandboxBroker; turns run here. The north
    gateway (``sharedagents gateway serve --role gateway``) proxies commands to it
    over ACP-HTTP and serves reads from the shared filesystem (§13 split).
    """
    from .gateway import serve_orchestrator_main

    return serve_orchestrator_main()


def cmd_orchestrator(args: argparse.Namespace) -> int:
    """``sharedagents orchestrator {status,all}`` — cross-agent live view.

    Reads the unified ``sessions.db`` once and prints one row per
    session, grouped by agent. ``status`` filters on the live state
    set; ``all`` returns every session regardless of state.
    """
    from sharedagents_agent.orchestrator import (
        all_sessions,
        list_live_sessions,
        render_table,
    )

    op = getattr(args, "op", "status") or "status"
    if op == "status":
        print(render_table(list_live_sessions()))
        return 0
    if op == "all":
        print(render_table(all_sessions(limit=int(getattr(args, "limit", 100)))))
        return 0
    print(f"unknown orchestrator subcommand: {op!r}", file=sys.stderr)
    return 2


def cmd_migrate_layout(args: argparse.Namespace) -> int:
    """``sharedagents migrate-layout`` — archive + delete legacy per-agent
    sessions/logs into the unified ``~/.sharedagents/`` layout."""
    from sharedagents_gateway.migrate_layout import migrate

    report = migrate()
    if report["agents_migrated"] == 0:
        print("nothing to migrate — layout already unified")
        return 0
    print(f"migrated legacy data for {report['agents_migrated']} agent(s)")
    print(f"backup: {report['tarball']}")
    return 0


def add_migrate_layout_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "migrate-layout",
        help="Archive + delete legacy per-agent sessions/logs once; "
        "the new unified layout puts every agent's data in "
        "~/.sharedagents/sessions and ~/.sharedagents/logs.",
    )
    p.set_defaults(_dispatch=cmd_migrate_layout)


def add_orchestrator_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "orchestrator",
        help="Cross-agent live view of every session's state in the shared sessions.db.",
    )
    p.set_defaults(_dispatch=lambda args: p.print_help() or 0)
    inner = p.add_subparsers(dest="op")
    p_status = inner.add_parser(
        "status",
        help="List sessions currently in a live state",
    )
    p_status.set_defaults(_dispatch=cmd_orchestrator)
    p_all = inner.add_parser(
        "all",
        help="List every session (newest first)",
    )
    p_all.add_argument("--limit", "-n", type=int, default=100, help="how many sessions to return (default: 100)")
    p_all.set_defaults(_dispatch=cmd_orchestrator)
    p_serve = inner.add_parser(
        "serve",
        help="Run the SOUTH orchestrator service (§13 split): owns the agents "
        "/ SessionWriter / sandbox; turns run here. Binds loopback :8099.",
    )
    p_serve.set_defaults(_dispatch=cmd_orchestrator_serve)


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "agent",
        help=(
            "Agent config: init / list / status / health / heartbeat / "
            "rollback-backup. Process lifecycle lives on `sharedagents "
            "gateway` — the gateway hosts every agent as a stdio child."
        ),
    )
    p.set_defaults(_dispatch=lambda args: p.print_help() or 0)
    inner = p.add_subparsers(dest="op")
    inner.required = True

    def _name(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("name", nargs="?", default="jarvis", help="agent name (default: jarvis)")

    p_init = inner.add_parser(
        "init",
        help=(
            "Scaffold a new agent dir. With --engine, build a custom "
            "specialist on an external engine "
            "(e.g. researcher --engine cursor_acp --model auto)."
        ),
    )
    _name(p_init)
    p_init.add_argument(
        "--engine",
        default="",
        help=(
            "External engine: kind (cursor_acp, claude_cli, …) or short "
            "name (cursor, claude, …). Omit for a local_llm agent."
        ),
    )
    p_init.add_argument(
        "--model",
        default="",
        help=(
            "Default model for ACP engines (stored as acp_model). "
            "Use 'auto' (default when --engine is set) for the engine's "
            "own Auto/default picker; or a concrete id like composer-2.5."
        ),
    )
    p_init.add_argument(
        "--description",
        default="",
        help="One-line description shown in /agents and the picker.",
    )
    p_init.set_defaults(_dispatch=cmd_init)

    p_list = inner.add_parser("list", help="List configured agents + gateway-mount status")
    p_list.set_defaults(_dispatch=cmd_list)

    p_stat = inner.add_parser("status", help="Probe gateway and the agent's ACP mount")
    _name(p_stat)
    p_stat.set_defaults(_dispatch=cmd_status)

    p_health = inner.add_parser(
        "health",
        help="Report failure-mode rates across recent turns",
    )
    _name(p_health)
    p_health.add_argument(
        "--limit",
        "-n",
        type=int,
        default=50,
        help="how many recent turns to scan (default: 50)",
    )
    p_health.set_defaults(_dispatch=cmd_health)

    # Rollback-backup management. `scaffold_context_files(overwrite=True)`
    # snapshots the agent home into `<home>/.rollback-backup/<ts>/`
    # before rewriting templates. This subtree accumulates over time;
    # `sharedagents agent rollback-backup {list,clean}` keeps it bounded.
    p_rb = inner.add_parser(
        "rollback-backup",
        help="List or clean .rollback-backup snapshots in the agent home",
    )
    rb_inner = p_rb.add_subparsers(dest="rb_op")
    p_rb.set_defaults(_dispatch=lambda args: p_rb.print_help() or 0)
    p_rb_list = rb_inner.add_parser("list", help="List existing snapshots")
    _name(p_rb_list)
    p_rb_list.set_defaults(_dispatch=cmd_rollback_backup_list)
    p_rb_clean = rb_inner.add_parser("clean", help="Delete snapshots (keeps the most-recent --keep, default 3)")
    _name(p_rb_clean)
    p_rb_clean.add_argument("--keep", type=int, default=3, help="Number of most-recent snapshots to retain (default 3)")
    p_rb_clean.set_defaults(_dispatch=cmd_rollback_backup_clean)

    # Heartbeat lifecycle: enable/disable/status/fire. The scheduler
    # reads `cfg.heartbeat_enabled` at boot, so enable/disable mutate
    # the per-agent config and require an `agent restart` to take
    # effect. `status` is read-only; `fire` POSTs a single tick into
    # the running gateway via HTTP.
    p_hb = inner.add_parser(
        "heartbeat",
        help="Manage heartbeat scheduler (enable/disable/status/fire)",
    )
    hb_inner = p_hb.add_subparsers(dest="hb_op")
    p_hb.set_defaults(_dispatch=lambda args: p_hb.print_help() or 0)
    for hb_op, hb_fn in (
        ("enable", cmd_heartbeat_enable),
        ("disable", cmd_heartbeat_disable),
        ("status", cmd_heartbeat_status),
        ("fire", cmd_heartbeat_fire),
    ):
        sp = hb_inner.add_parser(hb_op, help=f"{hb_op} the heartbeat scheduler")
        _name(sp)
        sp.set_defaults(_dispatch=hb_fn)
    # `enable` accepts an optional `--interval-s` so the user can tune
    # cadence in one shot without hand-editing config.json.
    for sp in (hb_inner.choices.get("enable"), hb_inner.choices.get("disable")):
        if sp is not None:
            sp.add_argument(
                "--interval-s",
                type=int,
                default=None,
                help="Set heartbeat_interval_s (default 1800).",
            )


def dispatch(args: argparse.Namespace) -> int:
    fn = getattr(args, "_dispatch", None)
    if fn is None:
        return 1
    return fn(args)
