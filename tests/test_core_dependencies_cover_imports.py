"""A lean `pip install sharedagents` must start: every third-party module the core
packages import at module level has to be a declared core dependency."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10; pytest depends on tomli there
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
CORE_PACKAGES = (
    "sharedagents",
    "sharedagents_core",
    "sharedagents_tui",
    "sharedagents_agent",
    "sharedagents_gateway",
    "sharedagents_model",
)
IMPORT_NAME = {"sqlite-vec": "sqlite_vec", "agent-client-protocol": "acp", "sse-starlette": "sse_starlette"}


def _declared_import_names() -> set[str]:
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    names = set()
    for spec in deps:
        dist = re.split(r"[<>=!~\[;\s]", spec, maxsplit=1)[0].lower()
        names.add(IMPORT_NAME.get(dist, dist.replace("-", "_")))
    return names


def test_core_packages_import_only_declared_core_dependencies_at_module_level() -> None:
    declared = _declared_import_names()
    first_party = {p.name for p in (ROOT / "src").iterdir() if p.is_dir()}
    undeclared: dict[str, list[str]] = {}
    for package in CORE_PACKAGES:
        for path in sorted((ROOT / "src" / package).rglob("*.py")):
            for node in ast.parse(path.read_text(encoding="utf-8")).body:
                if isinstance(node, ast.Import):
                    modules = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    modules = [node.module.split(".")[0]]
                else:
                    continue
                for module in modules:
                    if module not in sys.stdlib_module_names | first_party | declared:
                        undeclared.setdefault(module, []).append(str(path.relative_to(ROOT / "src")))
    assert undeclared == {}
