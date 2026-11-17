"""``python -m sharedagents_gateway`` entrypoint.

Delegates to :func:`sharedagents_gateway.cli.main` so the module-as-script
form is identical to the eventual ``sharedagents-agent`` console script.
"""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
