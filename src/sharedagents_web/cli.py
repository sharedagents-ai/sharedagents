"""Console entry for ``sharedagents-web`` / ``python -m sharedagents_web``.

Mint or reuse ``SHAREDAGENTS_WEB_SECRET`` *before* importing ``server`` —
that module fail-fasts at import if the secret is missing, which
blocked a bare ``pip install sharedagents[web]`` + ``sharedagents-web``.
"""

from __future__ import annotations

import os


def main() -> None:
    from sharedagents_core.launch import component_stdio, ensure_utf8_mode

    ensure_utf8_mode()
    component_stdio("web")
    if not os.environ.get("SHAREDAGENTS_WEB_SECRET"):
        from sharedagents_gateway.web_cli import _ensure_web_secret

        os.environ["SHAREDAGENTS_WEB_SECRET"] = _ensure_web_secret()
    from sharedagents_web.server import main as server_main

    server_main()
