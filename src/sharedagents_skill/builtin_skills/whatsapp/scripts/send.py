#!/usr/bin/env python3
"""Queue a WhatsApp line and/or files for the running channel.

send.py [--to <label>] [--file <abs path>]… [text]
"""

from __future__ import annotations

import argparse
import sys

from sharedagents_whatsapp.outbound import enqueue_send


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Queue a WhatsApp line and/or files for the running channel.",
    )
    parser.add_argument(
        "--to",
        metavar="LABEL",
        default=None,
        help="send-allowlisted label (may contain spaces), JID or phone; default: the linked self-chat",
    )
    parser.add_argument(
        "--file",
        dest="files",
        action="append",
        metavar="PATH",
        default=[],
        help="absolute path of a file to attach (repeatable)",
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="message text (tokens joined with single spaces)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    out = enqueue_send(" ".join(args.text), dest=args.to or "self", files=args.files)
    sys.stdout.write(out)
    if "Refused" in out or "No text" in out or "Not linked" in out or "not running" in out:
        return 2 if "Refused" in out else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
