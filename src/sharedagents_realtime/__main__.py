"""``python -m sharedagents_realtime [--host H] [--port P]``"""

from __future__ import annotations

import argparse

from .server import main

parser = argparse.ArgumentParser(prog="sharedagents_realtime", description="sharedagents realtime voice server")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8123)
args = parser.parse_args()
main(host=args.host, port=args.port)
