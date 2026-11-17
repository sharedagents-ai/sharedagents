"""Session-mutating slash commands must materialize a session in a fresh chat.

Regression: only ``/goal`` was gated, so ``/mode``, ``/effort``,
``/teacher``, ``/budget``, ``/loop`` and ``/model <alias>`` typed into a New Chat went out
session-less and surfaced ``RuntimeError: session not found: ''``.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

COMMANDS_JS = Path(__file__).resolve().parents[1] / "src" / "sharedagents_web" / "static" / "commands.js"
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_command_requires_session_vectors() -> None:
    src = COMMANDS_JS.read_text(encoding="utf-8")
    fn = re.search(r"function commandRequiresSession\(text\) \{.*?\n\}", src, re.S)
    assert fn, "commandRequiresSession not found"
    cases = {
        "/goal ship it": True,
        "/goal": True,
        "/mode": True,
        "/mode ask": True,
        "/effort high": True,
        "/teacher on": True,
        "/budget 5": True,
        "/loop 5m check": True,
        "/models": False,
        "/model": False,
        "/model dpsk": True,
        "/modelx": False,
        "/help": False,
        "/status": False,
        "/reasoning on": False,
        "/title hi": False,
        "hello": False,
    }
    script = (
        fn.group(0)
        + f"\nconst c={json.dumps(cases)};"
        + (
            "process.stdout.write(JSON.stringify(Object.fromEntries(Object.keys(c).map(k=>[k,commandRequiresSession(k)]))));"
        )
    )
    out = subprocess.run([NODE, "-e", script], capture_output=True, text=True, check=True).stdout
    assert json.loads(out) == cases
