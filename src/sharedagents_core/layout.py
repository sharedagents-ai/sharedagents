"""Re-export shim — on-disk layout now lives in ``sharedagents_core.kernel``.

The path math moved down into the leaf package ``sharedagents_core.kernel.layout``
so the memory engine can import ``sessions_root`` (and siblings) downward
without an import-time cycle. This module preserves the historical import path
``from sharedagents_core.layout import sessions_root`` for every existing caller.

Every public accessor is re-exported by identity below (an explicit list, not
just ``*`` — the kernel module's ``__all__`` intentionally omits a few helpers
that callers still import by name).
"""

from __future__ import annotations

from sharedagents_core.kernel.layout import (  # noqa: F401
    agent_config_dir,
    agents_root,
    current_agent_name,
    gateway_logfile,
    goals_db_path,
    goals_root,
    heartbeat_logfile,
    logs_root,
    loops_db_path,
    loops_root,
    pricing_auto_path,
    pricing_path,
    sessions_db_path,
    sessions_root,
    sharedagents_root,
    telegram_logfile,
    tui_logfile,
    usage_ledger_db_path,
    usage_root,
    voice_models_root,
)

__all__ = [
    "sharedagents_root",
    "agents_root",
    "agent_config_dir",
    "sessions_root",
    "sessions_db_path",
    "goals_root",
    "goals_db_path",
    "loops_root",
    "loops_db_path",
    "usage_root",
    "usage_ledger_db_path",
    "voice_models_root",
    "pricing_path",
    "pricing_auto_path",
    "logs_root",
    "gateway_logfile",
    "heartbeat_logfile",
    "tui_logfile",
    "telegram_logfile",
    "current_agent_name",
]
