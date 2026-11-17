"""Stdlib deadline watcher for Windows shells that outlive their launcher.

Invoked with the shell PID and its absolute deadline. The monitor continues to
own the shell directly, so manual cancellation still reaches that process.
Best effort: no birth-identity check or descendant kill; no Job Objects.
"""

from __future__ import annotations

import sys
import time

from sharedagents_core import oscompat


def run(pid: int, deadline_unix: float) -> int:
    # Honor the deadline recorded before spawn, then use monotonic time so
    # subsequent wall-clock adjustments cannot extend the running command.
    deadline = time.monotonic() + max(0.0, deadline_unix - time.time())
    while oscompat.process_alive(pid):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            try:
                oscompat.kill_process(pid, force=True)
            except ProcessLookupError:
                pass
            return 124
        time.sleep(min(0.1, remaining))
    return 0


if __name__ == "__main__":
    raise SystemExit(run(int(sys.argv[1]), float(sys.argv[2])))
