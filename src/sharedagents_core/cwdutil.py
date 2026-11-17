"""Dead-cwd-safe working-directory helper.

A long-lived sharedagents gateway can outlive the directory it was launched from —
e.g. it was started inside a git worktree that was later removed, leaving its
process cwd pointing at an unlinked inode. From then on bare ``os.getcwd()``
raises ``FileNotFoundError [Errno 2] No such file or directory``.

That bare call sits on the subagent-spawn path (``cwd=spawn_cwd or
os.getcwd()`` when a delegate is dispatched with no explicit cwd), so a dead
gateway cwd makes EVERY ``codex`` / ``claude`` / ``antigravity`` runner abort
with ``runner raised: [Errno 2] No such file or directory`` — even though the
spawn itself tolerates an inherited dead cwd (``create_subprocess_exec(cwd=None)``
runs fine). The fix is to never call bare ``os.getcwd()`` on that path: degrade
to a real existing directory instead.
"""

from __future__ import annotations

import os
import tempfile


def safe_getcwd() -> str:
    """``os.getcwd()`` that never raises on a deleted process cwd.

    Returns the live cwd when it resolves; otherwise the first existing
    directory among ``$HOME``, the system temp dir, and ``/`` (always a real,
    existing path). Use this anywhere a dead cwd must degrade rather than abort
    — subprocess spawn cwd, relative-path bases, error-logging context.
    """
    try:
        return os.getcwd()
    except OSError:
        for cand in (os.path.expanduser("~"), tempfile.gettempdir(), "/"):
            try:
                if cand and os.path.isdir(cand):
                    return cand
            except OSError:
                continue
        return "/"
