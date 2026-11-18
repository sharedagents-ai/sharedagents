"""Data-defined ontology workspaces and reviewer permissions."""

from __future__ import annotations

from .storage_client import configuration

LEVELS = {"viewer": 1, "editor": 2, "reviewer": 3, "owner": 4}


def authorization(actor, *, operator=False, workspace=None, operation="read"):
    if not isinstance(actor, str) or not actor.strip():
        raise PermissionError("authenticated ontology identity required")
    actor = actor.strip().casefold()
    config = configuration()
    defaults = config.get("default_workspaces", {})
    definitions = config.get("workspaces", {})
    if not isinstance(defaults, dict) or not isinstance(definitions, dict):
        raise PermissionError("ontology workspace configuration is malformed")
    if workspace is None:
        workspace = defaults.get("operator" if operator else actor)
    if workspace:
        if not isinstance(workspace, str) or not workspace or len(workspace) > 100:
            raise PermissionError("invalid ontology workspace")
        definition = definitions.get(workspace, {})
        if not isinstance(definition, dict) or not isinstance(definition.get("members", {}), dict):
            raise PermissionError("ontology membership configuration is malformed")
        members = definition.get("members", {})
        role = "owner" if operator else members.get(actor, "")
        scope = "workspace:" + workspace
    else:
        role = "owner"
        scope = "operator" if operator else "user:" + actor
    needed = {"read": 1, "write": 2, "review": 3, "manage": 4}.get(operation)
    if needed is None or LEVELS.get(role, 0) < needed:
        raise PermissionError("ontology workspace does not grant " + operation)
    return {"scope": scope, "actor": "operator" if operator else "user:" + actor, "role": role, "workspace": workspace}
