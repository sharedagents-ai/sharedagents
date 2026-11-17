"""Per-agent boot helpers.

The legacy per-agent daemon (``sharedagents agent serve <name>``) has been
retired. Every agent runs as a stdio child of the sharedagents gateway
(``sharedagents_gateway.gateway``), which is started by systemd
(``sharedagents-gateway.service``).

What's left in this module are the small pre-boot helpers shared by
the gateway's stdio entrypoint (:mod:`sharedagents_gateway.acp_stdio_main`):

* :func:`_load_agent_env_file` — merges ``<home>/.env`` into
  ``os.environ`` to dodge systemd ``EnvironmentFile=`` quirks.
* :func:`_autodetect_display_env` — picks ``DISPLAY`` /
  ``XAUTHORITY`` for headed-browser tools (the scraper agent's
  Chromium) when the launching shell didn't set one.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _autodetect_display_env() -> tuple[str, str | None] | None:
    """Return ``(DISPLAY, XAUTHORITY|None)`` for the first running X
    session this user can attach to, or ``None`` if no usable session
    exists.

    Why: the scraper agent runs ``--headed`` Chromium by default, but
    ``sharedagents`` is often launched from a tmux pane the user reattached
    over SSH after their X session was already up. That shell has no
    ``DISPLAY`` env, so the scraper's Chromium hangs forever waiting
    for an X server. Operators were having to ``export DISPLAY=:10``
    by hand or hand-edit ``~/.sharedagents/agents/scraper/.env``. Auto-pick
    the first viable session — every X server on the box has a
    socket at ``/tmp/.X11-unix/X<N>``, and a running ``Xorg`` /
    ``Xvnc`` / ``Xvfb`` process with ``:N`` in its argv.

    Selection rule (deterministic, not magic):
    1. Walk ``/tmp/.X11-unix/X<N>`` in ascending numeric order.
    2. For each, look for any ``Xorg`` / ``Xvnc`` / ``Xvfb`` /
       ``Xrdp*`` process owned by the current uid whose argv
       carries ``:<N>``.
    3. First match wins. Pair with ``$HOME/.Xauthority`` if it
       exists (most X servers stamp one); skip ``XAUTHORITY``
       otherwise so chromium falls back to anonymous.

    Skips ``:99`` (Xvfb headless commonly used by CI / gdm test
    seats) and ``:1024+`` (Chrome Remote Desktop's
    sandbox-allocated range, usually unauthenticated for the
    user's runtime).
    """
    try:
        sockets = sorted(int(p.name[1:]) for p in Path("/tmp/.X11-unix").glob("X*") if p.name[1:].isdigit())
    except (OSError, ValueError):
        return None
    if not sockets:
        return None

    my_uid = os.getuid()
    candidates: list[tuple[int, list[str]]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            st = entry.stat()
            if st.st_uid != my_uid:
                continue
            cmdline = (entry / "cmdline").read_bytes().split(b"\x00")
            cmd = [c.decode("utf-8", "replace") for c in cmdline if c]
        except (OSError, FileNotFoundError):
            continue
        if not cmd:
            continue
        exe = os.path.basename(cmd[0])
        if exe.startswith(("Xorg", "Xvnc", "Xvfb", "Xrdp")):
            candidates.append((int(entry.name), cmd))

    for display_num in sockets:
        if display_num == 99 or display_num >= 1024:
            continue
        token = f":{display_num}"
        if not any(token in argv for _, argv in candidates):
            continue
        xauth = os.path.expanduser("~/.Xauthority")
        return (token, xauth if os.path.isfile(xauth) else None)
    return None


def _load_agent_env_file(home: Path) -> int:
    """Merge ``<home>/.env`` into ``os.environ`` at boot.

    systemd's ``EnvironmentFile=`` directive is the canonical
    delivery mechanism for per-agent secrets (OpenRouter key,
    Telegram tokens, etc.) but its parser silently drops files
    that contain dollar-sign expansions resolving against unset
    variables — a real failure mode we hit when ``/etc/environment``
    declared ``LD_LIBRARY_PATH=...:$LD_LIBRARY_PATH`` ahead of the
    per-agent ``.env`` and the second file's variables didn't make
    it through.

    Read the file ourselves so the agent process picks up the
    secrets regardless of systemd's parsing quirks. Sets vars that
    aren't already present (or are present but empty) in
    ``os.environ`` — systemd's broken parse path appears to leave
    the key declared with an empty value, which we treat the same
    as missing. Anything the operator put on the unit's
    ``Environment=`` line with a non-empty value wins.

    Returns the count of vars added so the caller can log it.
    """
    env_path = home / ".env"
    if not env_path.is_file():
        return 0
    added = 0
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if not key:
                continue
            existing = os.environ.get(key)
            if existing:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            os.environ[key] = value
            added += 1
    except OSError as e:
        sys.stderr.write(f"failed to read {env_path}: {e}\n")
    return added


__all__ = ["_autodetect_display_env", "_load_agent_env_file"]
