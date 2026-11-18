"""Explicit SDK interpreter selection; no alternate data store or model calls."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .ontology import encoded, open_ontology


def configuration():
    override = os.environ.get("SHAREDAGENTS_ONTOLOGY_CONFIG")
    if override:
        block = json.loads(Path(override).read_text())
        if not isinstance(block, dict):
            raise ValueError("ontology configuration must be an object")
        return block
    home = Path(os.environ.get("SHAREDAGENTS_HOME", str(Path.home() / ".sharedagents")))
    try:
        block = json.loads((home / "config.json").read_text()).get("ontology", {})
    except FileNotFoundError:
        block = {}
    if not isinstance(block, dict):
        raise ValueError("ontology configuration must be an object")
    return block


class OntologyProcess:
    """One awaited application operation in a configured SharedData interpreter."""

    def __init__(self, python, root, scope, create):
        executable = os.path.expanduser(str(python))
        # Preserve a virtual environment's interpreter symlink. realpath() would
        # silently select the base interpreter and lose its installed SDK.
        self.python = os.path.abspath(executable) if os.path.dirname(executable) else executable
        self.root, self.scope, self.create = str(root), scope, create

    def __getattr__(self, action):
        if action not in {
            "begin",
            "check_open",
            "capture",
            "extract",
            "finish",
            "search",
            "get",
            "describe",
            "status",
            "timeline",
            "export",
            "publications",
            "fact_evidence",
            "model_get",
            "model_list",
            "model_put",
            "model_review",
            "model_retire",
            "model_history",
            "model_search",
            "model_embed",
            "model_retrieve",
            "model_path",
            "model_coverage",
            "model_export",
            "model_import",
            "compile_catalog",
            "retrieved_tables",
            "record_query",
            "query_catalog",
            "model_put_many",
            "table_context",
            "compilation_status",
            "apply_rules",
            "model_analytics",
            "evidence_chunks",
            "index_evidence",
            "evidence_search",
        }:
            raise AttributeError(action)

        def invoke(*args, **kwargs):
            request = {
                "root": self.root,
                "scope": self.scope,
                "create": self.create,
                "action": action,
                "args": args,
                "kwargs": kwargs,
            }
            env = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL", "SYSTEMROOT") if key in os.environ}
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
            try:
                proc = subprocess.run(
                    [self.python, "-m", "sharedagents_research.storage_worker"],
                    input=encoded(request),
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=30,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("SharedData worker operation timed out") from exc
            if proc.returncode:
                raise RuntimeError(f"SharedData worker exited with status {proc.returncode}")
            try:
                response = json.loads(proc.stdout)
            except ValueError as exc:
                raise RuntimeError("SharedData worker did not return a valid application response") from exc
            if not response.get("ok"):
                error = response.get("error", "SharedData operation failed")
                if response.get("error_type") == "FileNotFoundError":
                    raise FileNotFoundError(error)
                if response.get("error_type") == "PermissionError":
                    raise PermissionError(error)
                raise ValueError(error)
            return response["value"]

        return invoke


def open_runtime_ontology(root, scope, *, create=True, python=None):
    selected = python or os.environ.get("SHAREDAGENTS_ONTOLOGY_PYTHON") or configuration().get("python")
    if selected:
        return OntologyProcess(selected, root, scope, create)
    return open_ontology(root, scope, create=create)
