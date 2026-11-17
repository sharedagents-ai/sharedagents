"""Re-export shim — the injection guard now lives in ``sharedagents_core.kernel``.

The structural prompt-injection defense (``wrap_untrusted`` and friends) was
moved down into the leaf package ``sharedagents_core.kernel.injection_guard`` so the
memory engine can import the SAME audited guard without creating an import-time
cycle across the core↔memory (and, in Phase B, core↔mcp) package seam.

This module preserves the historical import path
``from sharedagents_core.injection_guard import wrap_untrusted`` for every existing
caller (``local_llm``, ``turn_runtime``, ``sharedagents_telegram``, the memory
manager, the injection-guard test). It re-exports the kernel module's public
API by identity — there is exactly one ``wrap_untrusted`` object, one
per-process nonce, one scrub. Do NOT re-implement anything here.
"""

from __future__ import annotations

# Re-export the whole public surface by identity (not a copy), including the
# module-level constants, so ``sharedagents_core.injection_guard.<name>`` resolves
# for module-attribute callers (``ig.UNTRUSTED_OPEN``) as well as imports.
from sharedagents_core.kernel.injection_guard import (  # noqa: F401
    UNTRUSTED_CLOSE,
    UNTRUSTED_OPEN,
    __all__,
    current_nonce,
    defang_control_tokens,
    injection_defense_prompt_section,
    is_untrusted_envelope,
    neutralize,
    reseal_fence,
    wrap_untrusted,
)
