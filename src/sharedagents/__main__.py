"""``python -m sharedagents`` — same entry point as ``python -m sharedagents.serve``."""

from sharedagents.serve import main

if __name__ == "__main__":
    raise SystemExit(main())
