"""``CodexCliEngine`` — wraps the ``codex exec`` CLI.

Class-shape facade. :meth:`delegate_runner` returns
``runners.run_delegate_codex_async``; the
:class:`ExternalCliEngine.run_turn` driver translates that runner's
``on_delta`` callbacks into :data:`AgentTurnEvent` instances. The
CLI subprocess pipeline (``codex exec resume``, ``-c`` permission
flags, ``thread.started`` parsing, ``-c sandbox_mode=…`` overrides)
lives in the legacy runner.
"""

from .engine import EngineKind
from .external_cli import ExternalCliEngine


class CodexCliEngine(ExternalCliEngine):
    """Run a turn through ``codex exec --json``."""

    kind = EngineKind.CODEX_CLI
