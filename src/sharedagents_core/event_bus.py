"""Re-export shim — the event bus now lives in ``sharedagents_core.kernel``.

The in-process debounced event bus was moved down into the leaf package
``sharedagents_core.kernel.event_bus`` so the memory engine (which borrows
``get_bus``) imports it downward without an import-time cycle. Its one back-edge
to the memory engine — a lazy import inside ``reset_bus_for_tests`` — was
inverted to a ``register_reset_hook`` registry so the kernel stays a leaf.

This module preserves the historical import path
``from sharedagents_core.event_bus import get_bus`` (and ``EventBus``,
``reset_bus_for_tests``) for every existing caller and test.
"""

from __future__ import annotations

from sharedagents_core.kernel.event_bus import (  # noqa: F401
    EventBus,
    EventHandler,
    __all__,
    get_bus,
    register_reset_hook,
    reset_bus_for_tests,
)
