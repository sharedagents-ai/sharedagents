"""Provider-specific model discovery implementation."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ._common import (
    _LOG,
    CLI_ENGINE_LABELS,
    _cache_get,
    _cache_put,
    _fallback_models,
)

_CURSOR_LIST_LINE_RE = re.compile(r"^(\S+)\s+-\s+(.+)$")


def _resolve_opencode_binary(
    command: list[str] | None = None,
) -> str | None:
    """Resolve the OpenCode executable from its configured ACP command."""
    exe = str((command or ["opencode"])[0] or "opencode").strip()
    try:
        from sharedagents_agent.tools.runners import _which_cli

        return _which_cli(exe)
    except Exception:  # noqa: BLE001
        return shutil.which(exe)


def _read_opencode_default_model() -> str:
    """Read only OpenCode's top-level default model from JSON/JSONC config.

    ``opencode debug config`` includes provider credentials, so discovery must
    not invoke or log it merely to identify the default picker row.
    """
    configured = str(os.environ.get("OPENCODE_CONFIG") or "").strip()
    config_root = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
    paths = ([Path(configured)] if configured else []) + [
        config_root / "opencode" / "opencode.jsonc",
        config_root / "opencode" / "opencode.json",
    ]
    pattern = re.compile(r'^\s*["\']?model["\']?\s*:\s*["\']([^"\']+)["\']')
    candidates: list[tuple[int, str]] = []
    for path in paths:
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                match = pattern.match(line)
                if match:
                    candidates.append((len(line) - len(line.lstrip()), match.group(1)))
        except OSError:
            continue
        if candidates:
            return min(candidates, key=lambda item: item[0])[1].strip()
    return ""


def _parse_opencode_models(
    text: str,
    *,
    default_model: str = "",
) -> list[dict[str, Any]]:
    """Parse provider-qualified IDs emitted by ``opencode models``."""
    seen: set[str] = set()
    ids: list[str] = []
    pattern = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[^\s/][^\s]*$")
    for raw in (text or "").splitlines():
        model_id = raw.strip()
        if not pattern.fullmatch(model_id) or model_id in seen:
            continue
        seen.add(model_id)
        ids.append(model_id)
    default = (default_model or "").strip()
    if default in seen:
        ids = [default, *[model_id for model_id in ids if model_id != default]]
    return [
        {
            "alias": model_id,
            "id": model_id,
            "display": model_id.split("/", 1)[1],
            "provider": model_id.split("/", 1)[0],
            "default": model_id == default,
        }
        for model_id in ids
    ]


def discover_opencode_models(
    *,
    force: bool = False,
    command: list[str] | None = None,
    run: Callable[[list[str]], str] | None = None,
) -> list[dict[str, Any]]:
    """Return OpenCode's provider-aware model catalogue.

    OpenCode's own ``models`` command applies its provider configuration and
    credentials. Results use the shared one-hour cache; orchestrator startup
    calls this with ``force=True`` so every service restart refreshes the list.
    """
    try:
        binary = _resolve_opencode_binary(command)
        key = f"opencode:{binary or 'opencode'}"
        if not force:
            cached = _cache_get(key)
            if cached is not None:
                return cached

        rows: list[dict[str, Any]] = []
        if binary:
            argv = [binary, "models"]
            if force:
                argv.append("--refresh")
            try:
                if run is not None:
                    output = run(argv)
                else:
                    from sharedagents_agent.delegate_env import build_third_party_spawn_env

                    proc = subprocess.run(
                        argv,
                        capture_output=True,
                        text=True,
                        timeout=15.0,
                        check=False,
                        env=build_third_party_spawn_env(),
                    )
                    output = proc.stdout if proc.returncode == 0 else ""
                rows = _parse_opencode_models(
                    output,
                    default_model=_read_opencode_default_model(),
                )
            except Exception as exc:  # noqa: BLE001
                _LOG.info("opencode model discovery failed: %s", exc)
                rows = []
        if not rows:
            rows = []
        _cache_put(key, rows)
        return [dict(row) for row in rows]
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("discover_opencode_models failed: %s", exc)
        return []


def _resolve_cursor_binary(command: list[str] | None = None) -> str | None:
    """Resolve the ``cursor-agent`` binary from an ACP command argv."""
    exe = str((command or ["cursor-agent"])[0] or "cursor-agent").strip()
    if not exe:
        exe = "cursor-agent"
    found = shutil.which(exe)
    if found:
        return found
    cand = os.path.expanduser(f"~/.local/bin/{exe}")
    return cand if os.path.exists(cand) else None


def _cursor_context_window(mid: str, label: str) -> int:
    """Context window for a Cursor catalogue row.

    Prefer an encoded ``context=`` / ``1M`` marker from the id/label; otherwise
    Composer stays at 200k and every other Cursor model defaults to 1M.
    """
    from sharedagents_agent.agents.acp_engine import _parse_model_context

    ctx = _parse_model_context(f"{mid} {label}")
    if ctx:
        return ctx
    if "composer" in mid.lower() or "composer" in label.lower():
        return 200_000
    return 1_000_000


def _enrich_cursor_model_row(row: dict[str, Any]) -> dict[str, Any]:
    """Ensure picker rows carry provider + contextWindow for Web/TUI badges."""
    out = dict(row)
    mid = str(out.get("id") or out.get("alias") or "").strip()
    label = str(out.get("display") or mid)
    if not out.get("provider"):
        out["provider"] = "cursor"
    if not int(out.get("contextWindow") or 0):
        out["contextWindow"] = _cursor_context_window(mid, label)
    return out


def _parse_cursor_list_models(text: str) -> tuple[list[dict[str, Any]], str]:
    """Parse ``cursor-agent --list-models`` stdout.

    Returns ``(rows, active_model_id)`` where ``active_model_id`` prefers the
    row tagged ``(current)``, else ``(default)``, else ``""``.
    """
    rows: list[dict[str, Any]] = []
    current = ""
    default = ""
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("Available models"):
            continue
        if line.startswith("Tip:"):
            break
        m = _CURSOR_LIST_LINE_RE.match(line)
        if not m:
            continue
        mid = m.group(1).strip()
        label = m.group(2).strip()
        if not mid:
            continue
        # Cursor tags the active row as ``(current)``, ``(default)``, or the
        # combined ``(current, default)`` — strip the whole trailing marker.
        marker = re.search(r"\s*\(([^)]*)\)\s*$", label)
        if marker:
            parts = {p.strip().lower() for p in marker.group(1).split(",") if p.strip()}
            if "current" in parts:
                current = mid
            if "default" in parts:
                default = mid
            if parts & {"current", "default"}:
                label = label[: marker.start()].rstrip()
        label = label.strip() or mid
        rows.append(
            _enrich_cursor_model_row(
                {
                    "alias": mid,
                    "id": mid,
                    "display": label,
                    "contextWindow": _cursor_context_window(mid, label),
                }
            )
        )
    active = current or default
    return rows, active


def _run_cursor_list_models(
    argv: list[str],
    *,
    env: dict[str, str],
    timeout: float = 8.0,
) -> str:
    """Capture ``cursor-agent --list-models`` stdout without pipe truncation.

    Cursor's CLI truncates ``--list-models`` at ~8KiB when stdout is a pipe
    (``subprocess`` ``capture_output`` / ``PIPE``). Redirecting to a real file
    returns the full catalogue (same as a shell ``>file`` redirect).
    """
    with tempfile.TemporaryDirectory(prefix="sharedagents-cursor-models-") as tmp:
        path = Path(tmp) / "list-models.txt"
        with path.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(
                argv,
                stdout=fh,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                check=False,
                env=env,
            )
        if proc.returncode != 0:
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return ""


def discover_cursor_models(
    *,
    force: bool = False,
    command: list[str] | None = None,
    run: Callable[[list[str]], str] | None = None,
) -> list[dict[str, Any]]:
    """Return Cursor's live model catalogue as picker rows.

    Runs ``cursor-agent --list-models`` (or ``command[0] --list-models`` when
    the agent overrides ``acp_command``). Returns empty on missing binary or failed probe.
    Memoised for :data:`_TTL_S`.
    """
    try:
        bin_path = _resolve_cursor_binary(command)
        key = f"cursor:{bin_path or 'cursor-agent'}"
        if not force:
            cached = _cache_get(key)
            if cached is not None:
                return cached

        rows: list[dict[str, Any]] = []
        active = ""
        if bin_path:
            argv = [bin_path, "--list-models"]
            try:
                if run is not None:
                    out = run(argv)
                else:
                    # Cursor-agent is a THIRD-PARTY vendor
                    # CLI. This probe must NOT inherit the full parent os.environ
                    # (infra secrets an agentic CLI could exfiltrate); reduce to the
                    # least-privilege allowlist. Discovery carries no
                    # per-agent api_keys — HOME is allowlisted so cursor-agent still
                    # reaches its own stored credential — so no overrides.
                    from sharedagents_agent.delegate_env import build_third_party_spawn_env

                    out = _run_cursor_list_models(argv, env=build_third_party_spawn_env())
                rows, active = _parse_cursor_list_models(out)
            except Exception as exc:  # noqa: BLE001
                _LOG.info("cursor model discovery failed: %s", exc)
                rows = []
        if not rows:
            rows = [_enrich_cursor_model_row(r) for r in _fallback_models("cursor")]
            active = rows[0]["id"] if rows else ""
        else:
            rows = [_enrich_cursor_model_row(r) for r in rows]
        active = rows[0]["id"] if rows else ""
        # Cache the active id alongside rows so cold-start callers can read it
        # without re-parsing or spawning twice.
        for r in rows:
            if r["id"] == active:
                r["_cursor_active"] = True
        _cache_put(key, rows)
        _cache_put(f"{key}:active", [{"id": active}] if active else [])
        return [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("discover_cursor_models failed: %s", exc)
        return []


def discover_cursor_current_model(
    *,
    force: bool = False,
    command: list[str] | None = None,
    run: Callable[[list[str]], str] | None = None,
) -> str:
    """Return the model Cursor reports as ``(current)`` or ``(default)``.

    Thin wrapper over :func:`discover_cursor_models` — shares the same cache.
    """
    try:
        bin_path = _resolve_cursor_binary(command)
        key = f"cursor:{bin_path or 'cursor-agent'}:active"
        if not force:
            cached = _cache_get(key)
            if cached is not None and cached:
                return str(cached[0].get("id") or "")
        rows = discover_cursor_models(force=force, command=command, run=run)
        bin_path = _resolve_cursor_binary(command)
        key = f"cursor:{bin_path or 'cursor-agent'}"
        cached = _cache_get(key) or []
        for r in cached:
            if r.get("_cursor_active"):
                return str(r.get("id") or "")
        return str(rows[0]["id"]) if rows else ""
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("discover_cursor_current_model failed: %s", exc)
        return ""


_KIMI_MODEL_HEADER = re.compile(
    r"^\[models\.(?:\"([^\"]+)\"|'([^']+)'|([^\]]+))\]\s*$",
)


def _kimi_config_model_ids(config_text: str) -> list[str]:
    """Model table keys from ``~/.kimi-code/config.toml`` (ids only, no secrets)."""
    ids: list[str] = []
    seen: set[str] = set()
    for line in (config_text or "").splitlines():
        m = _KIMI_MODEL_HEADER.match(line.strip())
        if not m:
            continue
        mid = (m.group(1) or m.group(2) or m.group(3) or "").strip()
        if not mid or mid.lower() in CLI_ENGINE_LABELS:
            continue
        key = mid.lower()
        if key in seen:
            continue
        seen.add(key)
        ids.append(mid)
    return ids


def _hermes_config_default_model(config_text: str) -> str:
    """``model.default`` from Hermes YAML, if present."""
    in_model = False
    for raw in (config_text or "").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith("model:") and not line.startswith(" "):
            in_model = True
            continue
        if in_model and not line.startswith((" ", "\t")):
            in_model = False
        if in_model:
            m = re.match(r"^\s+default:\s*(.+)$", line)
            if m:
                return m.group(1).strip().strip("\"'")
    return ""


def _row(alias: str, mid: str, display: str, provider: str, ctx: int = 1_000_000) -> dict[str, Any]:
    return {
        "alias": alias,
        "id": mid,
        "display": display,
        "provider": provider,
        "backend": provider,
        "contextWindow": ctx,
    }


def discover_kimi_models(*, force: bool = False) -> list[dict[str, Any]]:
    """Picker rows for ``kimi acp``: config.toml catalogue ∪ shipped K3.

    Never empty: a missing CLI/config still yields ``moonshot-ai/kimi-k3``.
    """
    try:
        key = "kimi:catalogue"
        if not force:
            cached = _cache_get(key)
            if cached is not None:
                return cached
        rows = []
        cfg_path = Path.home() / ".kimi-code" / "config.toml"
        try:
            ids = _kimi_config_model_ids(cfg_path.read_text(encoding="utf-8"))
        except OSError:
            ids = []
        seen: set[str] = set()
        for mid in ids:
            if mid.lower() in seen:
                continue
            seen.add(mid.lower())
            short = mid.rsplit("/", 1)[-1]
            rows.append(_row(short, mid, short, "kimi"))
        _cache_put(key, rows)
        return [dict(r) for r in rows]
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("discover_kimi_models failed: %s", exc)
        return []


def parse_devin_models_list(payload: Any) -> list[dict[str, Any]]:
    """Pull picker rows out of ``devin models list --format json``.

    Accepts a flat model list or ``{"families": [{"models": [...]}]}``.
    A dict that itself contains a ``models`` list is a family header, not
    a model.
    """
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(mid: str, name: str) -> None:
        mid = mid.strip()
        if not mid or mid.lower() in seen or mid.lower() in CLI_ENGINE_LABELS:
            return
        seen.add(mid.lower())
        label = (name or mid).strip() or mid
        rows.append(_row(label, mid, label, "devin"))

    def walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        children = node.get("models")
        if isinstance(children, list):
            walk(children)
        else:
            mid = node.get("id") or node.get("slug") or node.get("model_id") or node.get("modelId")
            name = node.get("name") or node.get("display_name") or node.get("displayName") or node.get("display")
            if isinstance(mid, str):
                add(mid, name if isinstance(name, str) else mid)
        for key in ("families", "data", "groups", "items"):
            child = node.get(key)
            if isinstance(child, (list, dict)) and key != "models":
                walk(child)

    walk(payload)
    return rows


def parse_devin_acp_model_options(options: Any) -> list[dict[str, Any]]:
    """Picker rows from a Devin ACP ``session/new`` config option ``model``.

    Each option is ``{"value", "name"}``. ``currentValue`` is the account's
    selected model and is marked ``default``. The payload has no context
    window, so rows carry 0 (unknown) instead of a guessed size.
    """
    if not isinstance(options, list):
        return []
    model: Any = None
    for opt in options:
        if isinstance(opt, dict) and str(opt.get("id") or "") == "model":
            model = opt
            break
    if not isinstance(model, dict):
        return []
    current = str(model.get("currentValue") or "").strip()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in model.get("options") or []:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("value") or item.get("id") or "").strip()
        if not mid or mid.lower() in seen or mid.lower() in CLI_ENGINE_LABELS:
            continue
        seen.add(mid.lower())
        name = str(item.get("name") or item.get("label") or mid).strip() or mid
        row = _row(mid, mid, name, "devin", 0)
        if current and mid == current:
            row["default"] = True
        rows.append(row)
    return rows


def _devin_spawn_overrides() -> dict[str, str]:
    """The Devin ACP key, from the agent secret file or the process env.

    ``devin models list`` ignores this variable. The ACP server does not.
    """
    key = str(os.environ.get("WINDSURF_API_KEY") or "").strip()
    if not key:
        try:
            from sharedagents_agent.agent import _load_agent_secrets, sharedagents_root

            loaded = _load_agent_secrets(sharedagents_root(), "devin")
            key = str((loaded or {}).get("WINDSURF_API_KEY") or "").strip()
        except Exception:  # noqa: BLE001
            key = ""
    return {"WINDSURF_API_KEY": key} if key else {}


def _devin_acp_model_options(bin_path: str, overrides: dict[str, str]) -> list[Any]:
    """Open one Devin ACP session and return its ``configOptions``.

    The process is killed as soon as ``session/new`` returns. Discovery is
    synchronous (the web picker calls it from Flask); if a loop is already
    running, the probe runs on a worker thread.
    """
    import asyncio

    from sharedagents_acp.client.spawn import spawn_stdio_agent
    from sharedagents_agent.delegate_env import build_third_party_spawn_env

    async def probe() -> list[Any]:
        env = build_third_party_spawn_env(overrides or None)
        client, proc = await spawn_stdio_agent(
            [bin_path, "acp"],
            cwd=tempfile.gettempdir(),
            env=env,
            inherit_parent_env=False,
        )
        try:
            await asyncio.wait_for(client.initialize(), timeout=20)
            await asyncio.wait_for(
                client.session_new(cwd=tempfile.gettempdir()),
                timeout=30,
            )
            return list(client.last_session_config_options or [])
        finally:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()

    try:
        asyncio.get_running_loop()
        running = True
    except RuntimeError:
        running = False
    if not running:
        return asyncio.run(probe())
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(probe())).result(timeout=40)


def discover_devin_models(
    *,
    force: bool = False,
    command: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Picker rows for the Devin account catalogue.

    ``devin models list`` is used when the CLI has a credentials file.
    That command stays "Not logged in" for a ``WINDSURF_API_KEY``-only
    install, so discovery then reads the ``model`` config option from one
    ACP ``session/new``. Empty results are not cached.
    """
    try:
        exe = str((command or ["devin"])[0] or "devin").strip() or "devin"
        key = f"devin:{exe}"
        if not force:
            cached = _cache_get(key)
            if cached is not None:
                return cached
        try:
            from sharedagents_agent.tools.runners import _which_cli

            bin_path = _which_cli(exe)
        except Exception:  # noqa: BLE001
            bin_path = shutil.which(exe)
        if not bin_path:
            return []
        from sharedagents_agent.delegate_env import build_third_party_spawn_env

        overrides = _devin_spawn_overrides()
        proc = subprocess.run(
            [bin_path, "models", "list", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
            env=build_third_party_spawn_env(overrides or None),
        )
        rows: list[dict[str, Any]] = []
        if proc.returncode == 0 and (proc.stdout or "").strip():
            try:
                rows = parse_devin_models_list(json.loads(proc.stdout))
            except json.JSONDecodeError:
                rows = []
        if not rows:
            try:
                rows = parse_devin_acp_model_options(
                    _devin_acp_model_options(bin_path, overrides),
                )
            except Exception as exc:  # noqa: BLE001
                _LOG.info("devin ACP model probe failed: %s", exc)
                rows = []
        if not rows:
            return []
        _cache_put(key, rows)
        return [dict(row) for row in rows]
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("discover_devin_models failed: %s", exc)
        return []


def discover_hermes_models(*, force: bool = False) -> list[dict[str, Any]]:
    """Picker rows for ``hermes acp`` via live config/agent discovery."""
    try:
        key = "hermes:catalogue"
        if not force:
            cached = _cache_get(key)
            if cached is not None:
                return cached
        rows = []
        cfg_path = Path.home() / ".hermes" / "config.yaml"
        try:
            default = _hermes_config_default_model(
                cfg_path.read_text(encoding="utf-8"),
            )
        except OSError:
            default = ""
        if default and default.lower() not in CLI_ENGINE_LABELS:
            rows.append(_row(default, default, default, "hermes"))
        _cache_put(key, rows)
        return [dict(r) for r in rows]
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("discover_hermes_models failed: %s", exc)
        return []


_QWEN_DEFAULT_MODEL = "qwen3.8-27b"
_QWEN_DEFAULT_CONTEXT = 262_144


def _qwen_settings_model_rows(settings: dict[str, Any]) -> list[dict[str, Any]]:
    """Picker rows from ``~/.qwen/settings.json`` ``modelProviders``.

    Current schema (settings ``$version`` 4): ``modelProviders`` maps a
    provider id to a bare ``ModelConfig[]`` array; each entry carries
    ``id`` plus optional ``name``/``description`` and
    ``generationConfig.contextWindowSize``. Wrapped preview shapes
    (``{"protocol": …, "models": […]}``) are skipped — qwen-code skips
    them too. Duplicate ids across providers collapse on first sight.
    """
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    providers = settings.get("modelProviders")
    if isinstance(providers, dict):
        for models in providers.values():
            if not isinstance(models, list):
                continue  # legacy wrapped preview shape — skipped upstream too
            for entry in models:
                if not isinstance(entry, dict):
                    continue
                mid = str(entry.get("id") or "").strip()
                if not mid or mid.lower() in seen or mid.lower() in CLI_ENGINE_LABELS:
                    continue
                seen.add(mid.lower())
                label = str(entry.get("name") or mid).strip() or mid
                gen = entry.get("generationConfig")
                ctx = _QWEN_DEFAULT_CONTEXT
                if isinstance(gen, dict):
                    raw_ctx = gen.get("contextWindowSize")
                    if isinstance(raw_ctx, (int, float)) and raw_ctx > 0:
                        ctx = int(raw_ctx)
                rows.append(_row(label, mid, label, "qwen", ctx))
    return rows


def discover_qwen_models(*, force: bool = False) -> list[dict[str, Any]]:
    """Picker rows for ``qwen --acp``: ``~/.qwen/settings.json`` catalogue.

    Never empty: a missing settings file still yields the registry default
    ``qwen3.8-27b``.
    """
    try:
        key = "qwen:settings"
        if not force:
            cached = _cache_get(key)
            if cached is not None:
                return cached
        cfg_path = Path.home() / ".qwen" / "settings.json"
        try:
            settings = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            settings = {}
        rows = _qwen_settings_model_rows(settings) if isinstance(settings, dict) else []
        if not rows:
            rows.append(
                _row(_QWEN_DEFAULT_MODEL, _QWEN_DEFAULT_MODEL, _QWEN_DEFAULT_MODEL, "qwen", _QWEN_DEFAULT_CONTEXT)
            )
        _cache_put(key, rows)
        return [dict(r) for r in rows]
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("discover_qwen_models failed: %s", exc)
        return []


__all__ = [
    "_CURSOR_LIST_LINE_RE",
    "_resolve_opencode_binary",
    "_read_opencode_default_model",
    "_parse_opencode_models",
    "discover_opencode_models",
    "_resolve_cursor_binary",
    "_parse_cursor_list_models",
    "discover_cursor_models",
    "discover_cursor_current_model",
    "_KIMI_MODEL_HEADER",
    "_kimi_config_model_ids",
    "_hermes_config_default_model",
    "_row",
    "discover_kimi_models",
    "discover_hermes_models",
    "_qwen_settings_model_rows",
    "discover_qwen_models",
    "parse_devin_models_list",
    "parse_devin_acp_model_options",
    "discover_devin_models",
]
