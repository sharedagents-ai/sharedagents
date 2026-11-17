"""Shared fixtures for the pricing tests.

Deliberately minimal and NOT autouse: this repo had no ``conftest.py`` before,
so anything global declared here would leak into every other test.

The two env defaults below run before any test module imports, which is what
lets web tests import ``sharedagents_web.server`` directly in any order: the
server refuses to boot without a secret, and never touches the operator's real
``~/.sharedagents`` because the home is redirected to a temp dir when unset.
"""

from __future__ import annotations

import os
import tempfile

import pytest

os.environ.setdefault("SHAREDAGENTS_WEB_SECRET", "pytest-web-secret")
# One home per process: pytest-xdist workers inherit the controller's
# environment, so a home picked there would be shared by every worker (same
# web DB, auth store and kv) and tests would race each other. A home the
# caller set explicitly is kept.
_picked = os.environ.get("_SHAREDAGENTS_PYTEST_HOME")
if not os.environ.get("SHAREDAGENTS_HOME") or os.environ["SHAREDAGENTS_HOME"] == _picked:
    os.environ["SHAREDAGENTS_HOME"] = os.environ["_SHAREDAGENTS_PYTEST_HOME"] = tempfile.mkdtemp(prefix="pytest-web-")


@pytest.fixture
def pricing_home(tmp_path, monkeypatch):
    """An isolated sharedagents tree, so pricing tests never touch ``~/.sharedagents``.

    Yields the root path. ``sharedagents_root()`` reads the environment on every
    call (it is not import-cached), so setting the variables is enough.
    ``SHAREDAGENTS_AGENTS_HOME`` must go: it outranks ``SHAREDAGENTS_HOME``.
    """
    root = tmp_path / "sharedagents"
    root.mkdir()
    monkeypatch.setenv("SHAREDAGENTS_HOME", str(root))
    monkeypatch.delenv("SHAREDAGENTS_AGENTS_HOME", raising=False)
    monkeypatch.setenv("SHAREDAGENTS_PRICING_AUTOSYNC", "1")
    return root


def pytest_collection_modifyitems(items):
    """Every ``*_browser.py`` test drives Chromium: mark it ``browser`` so CI's
    ``-m "not browser"`` deselects it even where the module forgot to."""
    for item in items:
        if item.path.name.endswith("_browser.py"):
            item.add_marker(pytest.mark.browser)
