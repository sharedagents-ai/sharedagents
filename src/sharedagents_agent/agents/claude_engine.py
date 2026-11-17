"""``ClaudeCliEngine`` — wraps the ``claude -p`` CLI.

Class-shape facade. :meth:`delegate_runner` returns
``runners.run_delegate_claude_async``; the
:class:`ExternalCliEngine.run_turn` driver translates that runner's
``on_delta`` / ``on_event`` callbacks into :data:`AgentTurnEvent`
instances. The whole CLI subprocess pipeline (stdin priming,
``--resume``, group-kill on cancel, stale-resume retry, structured
logging, truncation, ``--include-partial-messages``, inner-tool
``on_event`` surface) lives in the legacy runner.
"""

from .engine import EngineKind
from .external_cli import ExternalCliEngine


class ClaudeCliEngine(ExternalCliEngine):
    """Run a turn through ``claude -p --output-format stream-json``."""

    kind = EngineKind.CLAUDE_CLI
