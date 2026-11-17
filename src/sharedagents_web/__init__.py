"""sharedagents_web — the browser chat frontend for sharedagents agents.

Ships inside the ``sharedagents`` wheel: ``server.py`` plus the ``templates/``
and ``static/`` asset trees ride along as package data, so
``pip install sharedagents[web]`` is enough to run the UI (no separate repo
checkout). Launch via the ``sharedagents-web`` console script,
``python -m sharedagents_web``, or ``sharedagents web start``.
"""

from __future__ import annotations

from pathlib import Path

# Absolute paths to the bundled asset trees. Prefer these over cwd-relative
# lookups so the server works identically from an installed wheel and an
# editable checkout.
PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"
TEMPLATES_DIR = PACKAGE_DIR / "templates"

__all__ = ["PACKAGE_DIR", "STATIC_DIR", "TEMPLATES_DIR"]
