"""Run the sharedagents stack: orchestrator, gateway and WebUI together.

    python -m sharedagents.serve            # foreground; Ctrl+C stops everything
    python -m sharedagents.serve --detach   # background; survives this shell exiting
    python -m sharedagents.serve status     # one PASS/FAIL line per component
    python -m sharedagents.serve stop

Also installed as the ``sharedagents-serve`` console script, and reachable as
``python -m sharedagents``. The WebUI needs the ``[web]`` extra
(``pip install "sharedagents[web]"``); ``--no-web`` runs only the two backend roles.

`serve` refuses to start a component whose port is already in use, so it can
never adopt (or half-adopt) a stack it did not launch. Use `status` and `stop`
to inspect or clear an existing one.
"""

from __future__ import annotations

import argparse
import sys

from sharedagents_core.launch import ensure_utf8_mode
from sharedagents_gateway import stack

ACTIONS = ("serve", "stop", "status")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sharedagents-serve",
        description="Run the sharedagents stack: orchestrator (8099), gateway (8100), WebUI (8110).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "action",
        nargs="?",
        default="serve",
        choices=ACTIONS,
        help="serve (default): run the stack; stop: shut it down; status: probe it",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help="start the components detached and return, instead of running in the foreground",
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="manage only the orchestrator and gateway (skips the [web] extra)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:  # Command line only; in-process callers keep their process.
        ensure_utf8_mode()
    args = build_parser().parse_args(argv)
    try:
        return stack.run(args.action, detach=args.detach, no_web=args.no_web)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"FAIL {args.action}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
