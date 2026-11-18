"""Native research execution, ontology inspection and explicit local-agent setup."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path


def configure_research(home, *, backend_python=None):
    """Explicit setup preserves all unrelated model, endpoint and operator settings."""
    from sharedagents_agent.agent_profiles import RESEARCH, profile_agents_template_path

    home = Path(home)
    if backend_python and os.path.dirname(str(backend_python)):
        backend_python = os.path.abspath(os.path.expanduser(str(backend_python)))
    path = home / "config.json"
    config = json.loads(path.read_text()) if path.exists() else {"name": "research"}
    if config.get("engine", "local_llm") != "local_llm":
        raise ValueError("research ontology setup requires the native local_llm engine")
    before = dict(config)
    config.update(
        agent_profile="research",
        tools_allowlist=list(RESEARCH.tools_allowlist),
        tools_allowlist_strict=True,
        skill_allowlist=[],
        skill_allowlist_strict=True,
        disabled_tools=list(dict.fromkeys([*(config.get("disabled_tools") or []), "AgentSync", "AgentAsync"])),
    )
    home.mkdir(parents=True, exist_ok=True)
    # Install these explicit setup changes only after the SDK can open its store.
    from .storage_client import open_runtime_ontology

    root = Path(os.environ.get("SHAREDAGENTS_ONTOLOGY_ROOT") or home.parent.parent / "ontology")
    open_runtime_ontology(root, "operator", python=backend_python).status()
    updates = []
    if backend_python:
        global_path = Path(os.environ.get("SHAREDAGENTS_HOME") or home.parent.parent) / "config.json"
        global_config = json.loads(global_path.read_text()) if global_path.exists() else {}
        ontology = global_config.get("ontology", {})
        if not isinstance(ontology, dict):
            raise ValueError("ontology configuration must be an object")
        global_config["ontology"] = dict(ontology, python=str(backend_python))
        updates.append((global_path, json.dumps(global_config, indent=2) + "\n"))
    prompt = profile_agents_template_path("research").read_text()
    updates.extend(((path, json.dumps(config, indent=2) + "\n"), (home / "AGENTS.md", prompt)))
    for target, body in updates:
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = target.with_name(target.name + ".ontology-new")
        fd = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(body)
        os.replace(staged, target)
    return {"configured": True, "changed_config_keys": sorted(k for k in config if config.get(k) != before.get(k))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("research", help="Run native research and await ontology publication")
    run.add_argument("question")
    run.add_argument("--agent", default="research")
    run.add_argument("--source", action="append", default=[])
    run.add_argument("--parallel", type=int, default=8)
    setup = sub.add_parser("setup", help="Replace a local research prompt/tools with the installed ontology workflow")
    setup.add_argument("--agent-home", required=True, type=Path)
    setup.add_argument(
        "--backend-python", help="Existing interpreter with the SharedData SDK; saved in runtime configuration"
    )
    inspect = sub.add_parser("inspect", help="Inspect an explicitly scoped SharedData ontology")
    inspect.add_argument("--root", required=True, type=Path)
    inspect.add_argument("--scope", default="operator")
    inspect.add_argument("--query")
    args = parser.parse_args()
    if args.command == "setup":
        value = configure_research(args.agent_home, backend_python=args.backend_python)
    elif args.command == "inspect":
        from .storage_client import open_runtime_ontology

        store = open_runtime_ontology(args.root, args.scope, create=False)
        value = store.search(args.query) if args.query else store.status()
    else:
        from sharedagents_agent.agents.factory import load_agent

        from .coordinator import research_many

        agent = load_agent(args.agent)
        briefs = [args.question + "\nResearch this source: " + url for url in args.source] or [args.question]
        value = asyncio.run(research_many(agent, briefs, parallel=args.parallel))
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return 0 if all(row["state"] == "ok" for row in value) else 1
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
