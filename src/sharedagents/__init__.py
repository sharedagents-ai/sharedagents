"""sharedagents — cross-package entry points for an installed sharedagents.

The components live in their own top-level packages (``sharedagents_gateway``,
``sharedagents_web``, ``sharedagents_tui``, …). This package holds only the glue that has
to know about all of them at once, so it stays tiny:

    python -m sharedagents.serve      # run the stack (also the sharedagents-serve script)
    python -m sharedagents            # same entry point
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sharedagents")
except PackageNotFoundError:  # a source tree without installed metadata
    __version__ = "0.0.0"

__all__ = ["__version__"]
