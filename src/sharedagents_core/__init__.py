"""sharedagents-core — the transport-agnostic shared library.

This package is the home for everything below the ACP wire and below
the daemon: model dispatch, sessions, tools, memory, skills,
permissions, providers. It imports nothing from ``sharedagents_acp``,
``sharedagents_gateway``, ``sharedagents_tui``, or ``sharedagents_telegram`` — the
dependency arrow points one way only.
"""

from __future__ import annotations

__version__ = "1.0.1"

__all__ = ["__version__"]
