"""Registry-driven model discovery for external agents."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from typing import Any


def _accepts_keyword(fn: Callable[..., Any], name: str) -> bool:
    """Whether a registered discoverer accepts a keyword argument.

    Discovery hooks intentionally keep their vendor-specific signatures. The
    registry dispatcher may still receive generic context such as an ACP
    command, so forward that context based on the hook contract rather than
    branching on a vendor function name.
    """
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        # Some extension callables do not expose an inspectable signature. A
        # registered hook is trusted to implement the dispatcher contract.
        return True
    parameter = signature.parameters.get(name)
    if parameter is not None and parameter.kind is not inspect.Parameter.POSITIONAL_ONLY:
        return True
    return any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())


def discover_models(
    engine_kind: str | Any,
    *,
    force: bool = False,
    command: list[str] | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Discover picker rows for a built-in external ``EngineKind``."""
    from sharedagents_agent.external_registry import spec_for

    spec = spec_for(engine_kind)
    name = spec.discover_models_name
    module_name = spec.discover_models_module
    if not name or not module_name:
        raise ValueError(f"no discover_models hook for {engine_kind!r}")
    module = importlib.import_module(module_name)
    fn: Callable[..., list[dict[str, Any]]] = getattr(module, name)
    call_kwargs = dict(kwargs)
    if force and _accepts_keyword(fn, "force"):
        call_kwargs["force"] = force
    if command is not None and _accepts_keyword(fn, "command"):
        call_kwargs["command"] = command
    return fn(**call_kwargs)


__all__ = ["discover_models"]
