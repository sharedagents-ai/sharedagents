"""Plugin bundles — Claude Code–compatible.

A *plugin* is a directory shipping any combination of skills, file
commands, agents, and hooks as one unit. sharedagents plugins live at
``~/.sharedagents/plugins/<name>/`` and may carry any of:

```
~/.sharedagents/plugins/<name>/
    .claude-plugin/plugin.json   # optional manifest (CC format)
    plugin.json                  # alternative manifest location
    skills/<name>/SKILL.md       # any number of skills
    commands/<name>.md           # any number of file commands
    agents/<name>.md             # any number of subagents
    hooks/hooks.json             # CC-shape hook config
    hooks.json                   # alternative hooks location
```

Discovery automatically picks up `skills/`, `commands/`, `agents/`,
and `hooks/hooks.json` from every enabled plugin — operators don't
need to register the constituent dirs by hand.

## State

The filesystem is the source of truth — there is no plugins.json
registry. A plugin directory with a ``.disabled`` suffix is loaded as
"installed but inactive" (no discovery contribution); ``enable``
renames it back. ``install`` writes a small ``.sharedagents-plugin.json``
inside the plugin dir recording the install source (git URL or local
path) and ref, used by ``sharedagents plugin list`` and future updates.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

_DISABLED_SUFFIX = ".disabled"
_STATE_FILE = ".sharedagents-plugin.json"


def plugins_root() -> Path:
    """Where installed plugins live (one subdir per plugin).

    Honors ``$SHAREDAGENTS_HOME`` so the test suite (and any sandboxed
    run) writes into the sandbox tree rather than the operator's
    real ``~/.sharedagents``. Mirrors ``sharedagents_agent.agent.sharedagents_root``
    resolution.
    """
    home = os.environ.get("SHAREDAGENTS_HOME")
    base = Path(home).expanduser() if home else Path.home() / ".sharedagents"
    return base / "plugins"


# --------------------------------------------------------------------------
# Plugin record
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Plugin:
    name: str
    path: Path
    description: str = ""
    version: str = ""
    source: str = ""
    ref: str = ""
    enabled: bool = True

    @property
    def skills_dir(self) -> Path | None:
        d = self.path / "skills"
        return d if d.is_dir() else None

    @property
    def commands_dir(self) -> Path | None:
        d = self.path / "commands"
        return d if d.is_dir() else None

    @property
    def agents_dir(self) -> Path | None:
        d = self.path / "agents"
        return d if d.is_dir() else None

    @property
    def hooks_file(self) -> Path | None:
        for cand in (self.path / "hooks" / "hooks.json", self.path / "hooks.json"):
            if cand.is_file():
                return cand
        return None

    @property
    def mcp_file(self) -> Path | None:
        """``.mcp.json`` / ``mcp.json`` — a Claude-Code-style MCP bundle.

        Carries a ``{"mcpServers": {...}}`` object whose rows match
        ``sharedagents_mcp.mcp_client.MCPServerConfig.from_dict``. The
        dotfile is canonical; the plain form is accepted for
        ergonomics.
        """
        for cand in (
            self.path / ".claude-plugin" / "mcp.json",
            self.path / ".mcp.json",
            self.path / "mcp.json",
        ):
            if cand.is_file():
                return cand
        return None

    def mcp_servers(self) -> dict[str, Any]:
        """Parse this plugin's ``mcpServers`` block (or ``{}``)."""
        f = self.mcp_file
        if f is None:
            return {}
        data = _read_json(f)
        raw = data.get("mcpServers") or data.get("mcp_servers") or data
        return raw if isinstance(raw, dict) else {}

    def components(self) -> dict[str, int]:
        """Quick summary for ``/plugin list``: count of each component type."""
        out: dict[str, int] = {}
        if self.skills_dir:
            n = sum(
                1
                for d in self.skills_dir.iterdir()
                if d.is_dir() and ((d / "SKILL.md").is_file() or (d / "skill.md").is_file())
            )
            if n:
                out["skills"] = n
        if self.commands_dir:
            n = sum(1 for f in self.commands_dir.glob("*.md"))
            if n:
                out["commands"] = n
        if self.agents_dir:
            n = sum(1 for f in self.agents_dir.glob("*.md"))
            if n:
                out["agents"] = n
        if self.hooks_file:
            out["hooks"] = 1
        mcp_n = len(self.mcp_servers())
        if mcp_n:
            out["mcpServers"] = mcp_n
        return out


# --------------------------------------------------------------------------
# Manifest / state helpers
# --------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) or {}
    except (OSError, json.JSONDecodeError):
        return {}


def _parse_manifest(plugin_dir: Path) -> dict[str, Any]:
    """Read the optional CC-format ``plugin.json`` manifest."""
    for cand in (plugin_dir / ".claude-plugin" / "plugin.json", plugin_dir / "plugin.json"):
        if cand.is_file():
            return _read_json(cand)
    return {}


def _load_state(plugin_dir: Path) -> dict[str, Any]:
    state = plugin_dir / _STATE_FILE
    return _read_json(state) if state.is_file() else {}


def _save_state(plugin_dir: Path, state: dict[str, Any]) -> None:
    (plugin_dir / _STATE_FILE).write_text(
        json.dumps(state, indent=2) + "\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------
# Listing / lookup
# --------------------------------------------------------------------------


def _claude_code_plugin_dirs() -> list[Path]:
    """Yield every plugin install dir inside Claude Code's user cache.

    Claude Code installs plugins at
    ``~/.claude/plugins/cache/<marketplace>/<plugin>/<version>/`` and
    records the active install in ``~/.claude/plugins/installed_plugins.json``.
    Walking the cache directly (rather than the JSON manifest) catches
    plugins that the JSON missed — and the JSON's ``installPath`` is
    the same path we'd walk to anyway, so there's no benefit to
    reading it twice.

    Each returned path is a plugin bundle root (the dir that contains
    ``.claude-plugin/plugin.json`` or ``plugin.json``). Missing
    ``~/.claude/plugins/cache`` returns ``[]`` so an operator who only
    uses sharedagents isn't penalised.
    """
    cache = Path.home() / ".claude" / "plugins" / "cache"
    if not cache.is_dir():
        return []
    out: list[Path] = []
    for marketplace in sorted(cache.iterdir()):
        if not marketplace.is_dir() or marketplace.name.startswith("."):
            continue
        for plugin in sorted(marketplace.iterdir()):
            if not plugin.is_dir() or plugin.name.startswith("."):
                continue
            # Pick the lexicographically-last version dir (newest by
            # semver-prefix when versions are real, or "unknown" as
            # the single fallback Claude Code writes for git-pinned
            # installs without a version string).
            versions = [v for v in sorted(plugin.iterdir()) if v.is_dir() and not v.name.startswith(".")]
            if not versions:
                continue
            out.append(versions[-1])
    return out


def list_plugins() -> list[Plugin]:
    """Every installed plugin (enabled + disabled), sorted by name.

    Discovers plugins from two roots, in this precedence order:

    1. ``~/.sharedagents/plugins/<name>/`` — sharedagents's own install path
       (``/plugin install`` writes here).
    2. ``~/.claude/plugins/cache/<marketplace>/<plugin>/<version>/`` —
       Claude Code's install path. Lets a plugin installed by
       ``claude /plugin install ...`` appear in sharedagents's
       ``/plugin list`` automatically, satisfying the single-format
       compatibility contract.

    Duplicates (same plugin name in both roots) keep the sharedagents
    install — operator's own copy wins over a marketplace cache.
    """
    out: list[Plugin] = []
    seen_names: set[str] = set()

    def _record(child: Path) -> None:
        if not child.is_dir() or child.name.startswith("."):
            return
        disabled = child.name.endswith(_DISABLED_SUFFIX)
        canonical = child.name[: -len(_DISABLED_SUFFIX)] if disabled else child.name
        manifest = _parse_manifest(child)
        state = _load_state(child)
        name = str(manifest.get("name") or canonical)
        if name in seen_names:
            return
        seen_names.add(name)
        out.append(
            Plugin(
                name=name,
                path=child,
                description=str(manifest.get("description") or ""),
                version=str(manifest.get("version") or ""),
                source=str(state.get("source") or ""),
                ref=str(state.get("ref") or ""),
                enabled=not disabled,
            )
        )

    root = plugins_root()
    if root.is_dir():
        for child in sorted(root.iterdir()):
            _record(child)
    for child in _claude_code_plugin_dirs():
        _record(child)
    out.sort(key=lambda p: p.name)
    return out


def find_plugin(name: str) -> Plugin | None:
    """Case-insensitive lookup by manifest name OR directory name."""
    if not name:
        return None
    want = name.strip().lower()
    for p in list_plugins():
        if p.name.lower() == want or p.path.name.lower() == want:
            return p
        # Match the canonical name even when disabled.
        canon = p.path.name[: -len(_DISABLED_SUFFIX)] if p.path.name.endswith(_DISABLED_SUFFIX) else p.path.name
        if canon.lower() == want:
            return p
    return None


# --------------------------------------------------------------------------
# Discovery hooks — called from skills.py / file_commands.py
# --------------------------------------------------------------------------


def plugin_skill_dirs() -> list[Path]:
    """``skills/`` dirs from every ENABLED plugin (for the skill search chain)."""
    return [p.skills_dir for p in list_plugins() if p.enabled and p.skills_dir is not None]


def plugin_command_dirs() -> list[Path]:
    """``commands/`` dirs from every ENABLED plugin."""
    return [p.commands_dir for p in list_plugins() if p.enabled and p.commands_dir is not None]


def plugin_agent_files() -> list[Path]:
    """``agents/*.md`` files from every ENABLED plugin, in plugin-discovery
    order (so the first plugin defining an agent name wins, matching the
    filesystem-precedence of ``list_plugins``).

    These are what :func:`sharedagents_agent.agent_profiles.resolve_profile`
    consults so a plugin-contributed subagent is reachable by the Agent-tool
    subagent resolver with no manual per-component registration — the same
    auto-contribution contract the skill/command/hook/mcp dirs already have.
    """
    out: list[Path] = []
    for p in list_plugins():
        if not p.enabled or p.agents_dir is None:
            continue
        out.extend(sorted(p.agents_dir.glob("*.md")))
    return out


def plugin_mcp_servers() -> dict[str, Any]:
    """Merged ``mcpServers`` block contributed by every ENABLED plugin.

    Conflicts go to the plugin whose name sorts first (deterministic).
    Callers wanting an override should list the server explicitly in
    ``~/.sharedagents/config.json``'s ``mcpServers`` map — user config wins
    via ``load_mcp_servers`` (which calls this helper and overlays the
    user map on top).
    """
    out: dict[str, Any] = {}
    for p in list_plugins():
        if not p.enabled:
            continue
        for name, row in p.mcp_servers().items():
            out.setdefault(str(name), row)
    return out


# --------------------------------------------------------------------------
# Marketplace search
# --------------------------------------------------------------------------

# Built-in curated registry — enough to make ``/plugin search`` useful
# out of the box, even without network access. Operators can extend it
# by dropping ``~/.sharedagents/plugin-marketplace.json`` shaped like
# ``{"plugins": [{"name": ..., "source": ..., "description": ...,
# "tags": [...], "components": [...], "ref": "main"}]}``.
_BUILTIN_MARKETPLACE: list[dict[str, Any]] = [
    {
        "name": "everything-claude-code",
        "source": "https://github.com/anthropics/claude-code",
        "description": (
            "Anthropic's reference Claude Code plugin — ships the core "
            "slash commands, hooks, and skills you see in Claude Code."
        ),
        "tags": ["claude-code", "commands", "hooks", "skills", "official"],
        "components": ["commands", "hooks", "skills"],
    },
    {
        "name": "superpowers",
        "source": "https://github.com/anthropics/claude-code-plugins",
        "description": (
            "Skills bundle covering brainstorming, TDD, debugging, "
            "frontend-design, mcp-builder, writing-plans, and more."
        ),
        "tags": ["skills", "official"],
        "components": ["skills"],
    },
    {
        "name": "mcp-github",
        "source": "https://github.com/modelcontextprotocol/servers",
        "description": (
            "Reference MCP servers from modelcontextprotocol/servers — "
            "github, slack, postgres, filesystem, brave-search."
        ),
        "tags": ["mcp", "github", "official"],
        "components": ["mcpServers"],
    },
    {
        "name": "mcp-filesystem",
        "source": "https://github.com/modelcontextprotocol/servers",
        "description": ("Filesystem MCP server — read/write/list files over MCP."),
        "tags": ["mcp", "filesystem"],
        "components": ["mcpServers"],
    },
    {
        "name": "sharedagents-hello",
        "source": "https://github.com/sharedagents/sharedagents-hello-plugin",
        "description": (
            "Example sharedagents plugin: hello-world skill + /hello command. Template when authoring a new plugin."
        ),
        "tags": ["example", "skill", "command"],
        "components": ["skills", "commands"],
    },
]


def _load_user_marketplace() -> list[dict[str, Any]]:
    """Optional user registry at ``~/.sharedagents/plugin-marketplace.json``.

    Shape: ``{"plugins": [...]}`` — same fields as the built-in rows.
    Missing or malformed file → empty list (a typo shouldn't wedge
    ``/plugin search``).
    """
    home = os.environ.get("SHAREDAGENTS_HOME")
    base = Path(home).expanduser() if home else Path.home() / ".sharedagents"
    p = base / "plugin-marketplace.json"
    if not p.is_file():
        return []
    data = _read_json(p)
    rows = data.get("plugins") or data.get("entries") or []
    return rows if isinstance(rows, list) else []


@dataclass(frozen=True)
class MarketplaceEntry:
    name: str
    source: str
    description: str = ""
    ref: str = ""
    tags: tuple[str, ...] = ()
    components: tuple[str, ...] = ()
    score: int = 0

    def install_hint(self) -> str:
        ref = f" --ref {self.ref}" if self.ref else ""
        return f"/plugin install {self.name}{ref}"


def _row_to_entry(row: dict[str, Any], score: int = 0) -> MarketplaceEntry:
    return MarketplaceEntry(
        name=str(row.get("name") or ""),
        source=str(row.get("source") or row.get("url") or ""),
        description=str(row.get("description") or ""),
        ref=str(row.get("ref") or ""),
        tags=tuple(str(t) for t in (row.get("tags") or [])),
        components=tuple(str(c) for c in (row.get("components") or [])),
        score=score,
    )


def marketplace_entries() -> list[MarketplaceEntry]:
    """Union of built-in + user-supplied marketplace rows (no filter)."""
    rows = list(_BUILTIN_MARKETPLACE) + list(_load_user_marketplace())
    return [_row_to_entry(r) for r in rows if (r.get("name") or "").strip()]


def search(query: str, *, limit: int = 25) -> list[MarketplaceEntry]:
    """Substring-rank marketplace entries by ``query``.

    Tokens (whitespace-split) match case-insensitively against
    ``name`` (weight 3), ``tags`` (weight 2), ``description`` /
    ``components`` (weight 1). Empty query returns every entry
    sorted by name. Results truncated to ``limit``.
    """
    entries = marketplace_entries()
    q = (query or "").strip().lower()
    if not q:
        return sorted(entries, key=lambda e: e.name)[:limit]
    tokens = [t for t in q.split() if t]
    scored: list[MarketplaceEntry] = []
    for e in entries:
        hay_name = e.name.lower()
        hay_tags = " ".join(e.tags).lower()
        hay_comp = " ".join(e.components).lower()
        hay_desc = e.description.lower()
        s = 0
        for tok in tokens:
            if tok in hay_name:
                s += 3
            if tok in hay_tags:
                s += 2
            if tok in hay_comp:
                s += 1
            if tok in hay_desc:
                s += 1
        if s > 0:
            scored.append(
                MarketplaceEntry(
                    name=e.name,
                    source=e.source,
                    description=e.description,
                    ref=e.ref,
                    tags=e.tags,
                    components=e.components,
                    score=s,
                )
            )
    scored.sort(key=lambda e: (-e.score, e.name))
    return scored[:limit]


def resolve_marketplace_name(source: str) -> MarketplaceEntry | None:
    """Return the marketplace row matching ``source`` by name, or None.

    A bare token (no ``://``, no path separator, no ``.``) is treated
    as a marketplace name; an exact case-insensitive match wins.
    URL-ish / path-ish strings return ``None`` (the caller passes
    them through to git clone / local copy).
    """
    s = (source or "").strip()
    if not s:
        return None
    if "://" in s or "/" in s or s.endswith(".git") or s.startswith("."):
        return None
    want = s.lower()
    for e in marketplace_entries():
        if e.name.lower() == want:
            return e
    return None


# --------------------------------------------------------------------------
# Lifecycle: install / enable / disable / remove
# --------------------------------------------------------------------------


def _derive_name(source: str) -> str:
    base = source.rstrip("/").split("/")[-1]
    if base.endswith(".git"):
        base = base[:-4]
    return base.strip().lower().replace(" ", "-")


def install(
    source: str,
    *,
    name: str | None = None,
    ref: str | None = None,
) -> Plugin:
    """Install a plugin from a marketplace name, git URL, or local dir.

    A bare token matching a marketplace entry resolves to that
    entry's ``source`` / ``ref`` (see ``resolve_marketplace_name``);
    a local path is copied (``shutil.copytree``); anything else is
    cloned with ``git clone --depth 1`` (and ``--branch <ref>`` when
    provided). Writes a small ``.sharedagents-plugin.json`` state file so
    ``list`` can show where each plugin came from.
    """
    src = source.strip()
    if not src:
        raise ValueError("source URL or path required")
    market = resolve_marketplace_name(src)
    if market is not None:
        src = market.source
        if not ref and market.ref:
            ref = market.ref
        if name is None:
            name = market.name
    root = plugins_root()
    root.mkdir(parents=True, exist_ok=True)
    nm = name or _derive_name(src)
    if not nm:
        raise ValueError(f"couldn't derive a name from {source!r}")
    dst = root / nm
    if dst.exists() or (root / f"{nm}{_DISABLED_SUFFIX}").exists():
        raise FileExistsError(f"plugin already installed: {nm}")

    src_path = Path(src).expanduser()
    if src_path.exists() and src_path.is_dir():
        shutil.copytree(src_path, dst)
    else:
        cmd = ["git", "clone", "--depth", "1"]
        if ref:
            cmd += ["--branch", ref]
        cmd += [src, str(dst)]
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except subprocess.CalledProcessError as e:
            shutil.rmtree(dst, ignore_errors=True)
            tail = (e.stderr or e.stdout or "").strip().splitlines()[-3:]
            raise RuntimeError("git clone failed: " + "\n".join(tail or [str(e)])) from e
        except FileNotFoundError as e:
            raise RuntimeError("git not on PATH; cannot install from a URL") from e
    _save_state(dst, {"source": src, "ref": ref or "", "installed_at": int(time.time())})
    p = find_plugin(nm)
    if p is None:
        raise RuntimeError(f"installed at {dst} but failed to load")
    return p


def disable(name: str) -> Plugin:
    """Disable an installed plugin — renames its dir with a ``.disabled`` suffix."""
    p = find_plugin(name)
    if p is None:
        raise FileNotFoundError(f"plugin not installed: {name}")
    if not p.enabled:
        return p
    new_path = p.path.with_name(p.path.name + _DISABLED_SUFFIX)
    if new_path.exists():
        raise FileExistsError(f"name conflict: {new_path}")
    p.path.rename(new_path)
    out = find_plugin(name)
    return out if out is not None else p


def enable(name: str) -> Plugin:
    """Re-enable a disabled plugin."""
    p = find_plugin(name)
    if p is None:
        raise FileNotFoundError(f"plugin not installed: {name}")
    if p.enabled:
        return p
    target = p.path.with_name(p.path.name[: -len(_DISABLED_SUFFIX)])
    if target.exists():
        raise FileExistsError(f"name conflict: {target}")
    p.path.rename(target)
    out = find_plugin(name)
    return out if out is not None else p


def remove(name: str, *, purge: bool = False) -> Path:
    """Uninstall a plugin.

    ``purge=False`` (default) moves the dir into ``plugins/.trash/`` so
    an accidental remove is recoverable. ``purge=True`` rm-rf's it.
    Returns the new path on disk (or the deleted path).
    """
    p = find_plugin(name)
    if p is None:
        raise FileNotFoundError(f"plugin not installed: {name}")
    if purge:
        shutil.rmtree(p.path)
        return p.path
    trash = plugins_root() / ".trash"
    trash.mkdir(parents=True, exist_ok=True)
    target = trash / f"{p.path.name}-{int(time.time())}"
    p.path.rename(target)
    return target
