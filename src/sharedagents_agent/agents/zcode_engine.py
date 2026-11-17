"""``ZCodeCliEngine`` — wraps the Z.AI ``zcode`` CLI.

Class-shape facade. :meth:`delegate_runner` returns
``runners.run_delegate_zcode_async``; the
:class:`ExternalCliEngine.run_turn` driver translates that runner's
``on_delta`` / ``on_event`` callbacks into :data:`AgentTurnEvent`
instances. The whole CLI subprocess pipeline (argv build, ``--resume``
session sidecar, tree kill on cancel, ``--json`` envelope parsing,
``ZCODE_API_KEY`` per-delegate passthrough) lives in the runner.

Auth: the CLI's own Z.AI OAuth login (``zcode login`` →
``~/.zcode/v2/credentials.json``), covering the GLM Coding Plan quota —
no sharedagents-side token.
"""

from .engine import EngineKind
from .external_cli import ExternalCliEngine


class ZCodeCliEngine(ExternalCliEngine):
    """Run a turn through ``zcode -p <prompt> --json``."""

    kind = EngineKind.ZCODE_CLI
