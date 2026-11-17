"""``sharedagents_gateway`` CLI argparse skeleton.

Ships only the *surface* today: each named subcommand maps to a stub
that prints ``"<name>: not yet implemented"`` and returns 0, so smoke
tests can exercise the dispatcher end-to-end. The actual command
implementations live in ``sharedagents_gateway.agent_cli`` (config-only
surface) and ``sharedagents_gateway.gateway_cli`` (process lifecycle).

Subcommand list (config-only — no per-agent daemon lifecycle here):

    init   list   status

A new subcommand here is a public API change: don't add one without
updating the CLI tests.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

# Subcommand list. Order is preserved through ``--help`` output, so
# any reordering is a public-surface change.
SUBCOMMANDS: tuple[str, ...] = (
    "init",
    "list",
    "status",
)

# Stdio ACP entry point. Intentionally NOT in :data:`SUBCOMMANDS` —
# ``acp-stdio`` is REAL behavior (it runs the per-agent
# :class:`sharedagents_agent.agents.gateway_shim.AgentACPBackend` over a
# stdio transport via ``sharedagents_acp.server.serve_stdio``); ACP clients
# spawn it as a subprocess and humans typically don't invoke it
# directly, so it stays out of ``--help``'s named-subcommand list.
EXTRA_SUBCOMMANDS: tuple[str, ...] = (
    "acp-stdio",
    "gateway-serve",
)


def _stub_message(name: str) -> str:
    """The exact stub line every subcommand prints today.

    Centralized so tests can assert on the same substring used at
    runtime. The real implementations live in
    :mod:`sharedagents_gateway.agent_cli`.
    """
    return f"{name}: not yet implemented"


def _make_stub(name: str) -> Callable[[argparse.Namespace], int]:
    """Build the ``_cmd_<name>`` stub for subcommand ``name``.

    Each stub writes the canonical message to stdout and returns 0
    so the CLI smoke tests can run unmodified.
    """

    def _cmd(args: argparse.Namespace) -> int:  # noqa: ARG001
        print(_stub_message(name))
        return 0

    _cmd.__name__ = f"_cmd_{name}"
    return _cmd


# Bind one stub per subcommand under the canonical ``_cmd_<name>``
# attribute name so future tests can monkeypatch them by attribute.
_cmd_init = _make_stub("init")
_cmd_list = _make_stub("list")
_cmd_status = _make_stub("status")


def _cmd_acp_stdio(args: argparse.Namespace) -> int:
    """Run the stdio ACP server bound to ``args.agent_name``.

    Loads the agent's :class:`~sharedagents_agent.agent.AgentConfig`,
    constructs a :class:`~sharedagents_gateway.acp_bridge.CoreBackend`, and
    serves it over stdin/stdout via
    :func:`sharedagents_acp.server.serve_stdio` until EOF. Imports happen
    lazily so the rest of this module's stubs (which other tests
    exercise) don't pay the SDK / asyncio cost on import.
    """
    import asyncio

    from sharedagents_agent.agent import AgentConfig

    from .acp_stdio_main import serve_acp_stdio
    from .runtime import _autodetect_display_env, _load_agent_env_file

    name = getattr(args, "agent_name", None) or "jarvis"
    cfg = AgentConfig.load(name)
    # Stdio children are spawned by the gateway and may not have the
    # agent's secrets in their parent's env — without this load the
    # child's ``cfg.upstream_headers()`` returns no Authorization for
    # OpenRouter-shaped rows and OpenRouter responds HTTP 401.
    _load_agent_env_file(cfg.home)
    # Auto-pin DISPLAY if nothing in env / .env set one. Same code
    # path the gateway runs — gives the scraper agent's headed
    # Chromium an X server to attach to even when launched from a
    # tmux pane that has no DISPLAY (typical for SSH-reattached
    # sessions). The .env override still wins when present.
    import os as _os

    if not _os.environ.get("DISPLAY"):
        picked = _autodetect_display_env()
        if picked is not None:
            display, xauth = picked
            _os.environ["DISPLAY"] = display
            if xauth:
                _os.environ.setdefault("XAUTHORITY", xauth)
    return asyncio.run(serve_acp_stdio(cfg))


def _cmd_gateway_serve(args: argparse.Namespace) -> int:  # noqa: ARG001
    """Start the paired stack (orchestrator then gateway) in the background."""
    from .gateway_cli import start_gateway_background

    result = start_gateway_background()
    if not result["ok"]:
        sys.stderr.write(f"{result['error']}\n")
        return 1
    return 0


_DISPATCH: dict[str, Callable[[argparse.Namespace], int]] = {
    "init": _cmd_init,
    "list": _cmd_list,
    "status": _cmd_status,
}

# Extra (non-stub) subcommands. Looked up by ``main()`` before
# argparse runs so they don't pollute ``--help`` (the smoke tests
# assert the named-subcommand set is exactly :data:`SUBCOMMANDS`).
_EXTRA_DISPATCH: dict[str, Callable[[argparse.Namespace], int]] = {
    "acp-stdio": _cmd_acp_stdio,
    "gateway-serve": _cmd_gateway_serve,
}


def build_parser() -> argparse.ArgumentParser:
    """Construct the ``sharedagents_gateway`` argparse skeleton.

    Kept separate from :func:`main` so tests can inspect the parser
    (e.g. assert subcommand presence) without spawning a subprocess.
    """
    parser = argparse.ArgumentParser(
        prog="sharedagents_gateway",
        description=(
            "sharedagents-agent daemon CLI scaffold. Most named "
            "subcommands are stubs; the real lifecycle lives in "
            "sharedagents_gateway.agent_cli."
        ),
    )
    sub = parser.add_subparsers(dest="op", metavar="<subcommand>")
    for name in SUBCOMMANDS:
        sp = sub.add_parser(name, help=f"{name} (stub)")
        sp.set_defaults(_dispatch=_DISPATCH[name])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m sharedagents_gateway`` and the eventual
    ``sharedagents-agent`` console script.

    Returns the exit code so callers (including tests) can assert on
    it without reading stderr. ``argv=None`` defers to ``sys.argv[1:]``
    just like ``argparse.parse_args`` does on its own.

    The :data:`EXTRA_SUBCOMMANDS` set is dispatched BEFORE argparse so
    those entries don't show up in ``--help``'s subcommand list (the
    CLI smoke tests gate ``--help`` choices against
    :data:`SUBCOMMANDS` exactly). They're discoverable via the
    ``sharedagents_agent_<name>`` exported callables and via direct invocation
    (``python -m sharedagents_gateway acp-stdio main``).
    """
    args_list = list(argv) if argv is not None else list(sys.argv[1:])
    if args_list and args_list[0] in EXTRA_SUBCOMMANDS:
        name = args_list[0]
        ns = argparse.Namespace()
        # Preserve any positional args after the subcommand for the
        # extra-cmd dispatcher (e.g. acp-stdio takes an optional agent
        # name).
        rest = args_list[1:]
        if name == "acp-stdio":
            ns.agent_name = rest[0] if rest else "jarvis"
            return int(_cmd_acp_stdio(ns))
        if name == "gateway-serve":
            return int(_cmd_gateway_serve(ns))
        if name == "self-improver":
            sys.stderr.write("self-improver has been removed from sharedagents\n")
            return 1
    parser = build_parser()
    args = parser.parse_args(args_list)
    dispatch: Callable[[argparse.Namespace], int] | None = getattr(args, "_dispatch", None)
    if dispatch is None:
        parser.print_help(sys.stderr)
        return 0
    return int(dispatch(args))


__all__ = [
    "SUBCOMMANDS",
    "build_parser",
    "main",
    "_cmd_init",
    "_cmd_list",
    "_cmd_status",
    "_cmd_acp_stdio",
    "_cmd_gateway_serve",
]
