"""CLI: ``sharedagents-vllm --config PATH``."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from .config import default_config_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sharedagents-vllm")
    parser.add_argument(
        "--config",
        default=os.environ.get("SHAREDAGENTS_VLLM_CONFIG") or str(default_config_path()),
        help="local-only router YAML (no cloud keys)",
    )
    parser.add_argument("--host", default=None, help="override bind address")
    parser.add_argument("--port", type=int, default=None, help="override port")
    parser.add_argument(
        "--log-level",
        default=os.environ.get("SHAREDAGENTS_VLLM_LOG_LEVEL", "info"),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s sharedagents_vllm %(message)s",
    )

    try:
        from .app import create_app
        from .config import load_config
    except ImportError as exc:
        print(exc, file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    if args.host:
        cfg.bind = args.host
    if args.port is not None:
        cfg.port = args.port
    os.environ["SHAREDAGENTS_VLLM_CONFIG"] = str(args.config)
    app = create_app(cfg)
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required; install sharedagents[vllm-router]", file=sys.stderr)
        return 2
    uvicorn.run(app, host=cfg.bind, port=cfg.port, log_level=args.log_level)
    return 0
