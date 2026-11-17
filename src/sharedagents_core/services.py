"""Re-export shim — the stdlib-only service resolver lives in the kernel leaf.

The unified primary/backup service resolver lives in the stdlib-only leaf
``sharedagents_core.kernel.services`` (Phase A cycle-break: the memory engine imports
``resolve_service`` DOWNWARD from the kernel, and the kernel-leaf invariant
forbids any kernel module importing back up into ``sharedagents_core``). This module
preserves the historical import path ``from sharedagents_core.services import
resolve_service`` for every existing caller.

*This* module's own import set must stay **stdlib-only**, so
a standalone bundled MCP server can import it without pulling in the rest of
``sharedagents_core``. It is: the re-export is performed lazily through ``importlib``
(a stdlib module) via a module-level ``__getattr__`` (PEP 562), so the only
static imports here are ``__future__`` and ``importlib``. Resolving a name pulls
in only the stdlib-only kernel leaf — never any heavy dependency — which is
exactly the cheap-import property that matters here.
"""

from __future__ import annotations

import importlib as _importlib

# The stdlib-only implementation. Kept as a string (not a static import) so this
# module's own import set stays stdlib-only.
_IMPL = "sharedagents_core.kernel.services"

__all__ = [
    "SERVICES",
    "ServiceEndpoint",
    "ServiceConfig",
    "resolve_service",
    "with_failover",
]


def __getattr__(name: str):  # PEP 562 — lazy re-export from the kernel leaf.
    return getattr(_importlib.import_module(_IMPL), name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
