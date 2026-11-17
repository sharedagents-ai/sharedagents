"""Every page script under ``sharedagents_web/static`` parses.

A syntax error takes a whole page down (the Users tab shipped one: an edit
dropped a function header and the script stopped at load), and nothing but a
browser run would notice — so check them all with ``node --check``.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "src" / "sharedagents_web" / "static"
NODE = shutil.which("node")
SCRIPTS = sorted(p for p in STATIC.rglob("*.js") if "vendor" not in p.relative_to(STATIC).parts)


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_every_static_script_parses():
    assert any(p.name == "users.js" for p in SCRIPTS), "static scripts not found"
    failures = {}
    for script in SCRIPTS:
        done = subprocess.run([NODE, "--check", str(script)], capture_output=True, text=True, timeout=30)
        if done.returncode != 0:
            failures[str(script.relative_to(STATIC))] = done.stderr.strip()
    assert not failures, failures
