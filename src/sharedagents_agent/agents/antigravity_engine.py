"""``AntigravityCliEngine`` — wraps the ``antigravity`` CLI.

Class-shape facade. :meth:`delegate_runner` returns
``runners.run_delegate_antigravity_async``; the
:class:`ExternalCliEngine.run_turn` driver translates that runner's
``on_delta`` callbacks into :data:`AgentTurnEvent` instances. The
CLI subprocess pipeline (``antigravity -p`` first turn / ``--resume``
follow-up, ``--model``, ``--mode``, ``--dangerously-skip-permissions``,
``init`` / ``message`` / ``tool_use`` / ``tool_result`` event
parsing) lives in the legacy runner.
"""

from .engine import EngineKind
from .external_cli import ExternalCliEngine


class AntigravityCliEngine(ExternalCliEngine):
    """Run a turn through ``antigravity -p --output-format stream-json``."""

    kind = EngineKind.ANTIGRAVITY_CLI
