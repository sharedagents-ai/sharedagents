"""Background model-provisioning worker.

Spawned detached by the first-run wizard (``sharedagents_gateway.agent_cli
._start_background_provision``) so the TUI can come up immediately and render a
live download progress bar instead of blocking on a multi-GB download. It runs
the blocking provision (install vLLM if needed → download weights → launch the
server), then clears the marker the TUI polls (see
:func:`sharedagents_model.local_gpu.provision_status`). On failure it marks the
marker so the TUI can surface the error and fall back to a cloud model.

The global config + ``main`` agent are already written by the wizard pointing
at the local endpoint this worker brings up, so there's nothing to persist
here — just serve the model.
"""

from __future__ import annotations

import sys

from sharedagents_model import local_gpu as lg


def main() -> int:
    """Provision the local model; mark the marker on every failure exit."""
    profile = lg.gpu_profile()
    if profile is None:
        lg.mark_provision_error("no NVIDIA GPU detected")
        return 1
    choice = lg.select_model_for_profile(profile)
    if choice is None:
        lg.mark_provision_error("no known model for this GPU profile")
        return 1
    try:
        server = lg.provision_model(choice, profile)
    except Exception as exc:  # never crash silently — record for the TUI
        lg.mark_provision_error(f"provisioning crashed: {exc}")
        return 1
    if server is None:
        lg.mark_provision_error("vLLM provisioning failed (see vllm-server.log)")
        return 1
    lg.clear_provision_marker()
    return 0


if __name__ == "__main__":
    sys.exit(main())
