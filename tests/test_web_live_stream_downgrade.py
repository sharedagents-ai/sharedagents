"""Client must not auto-downgrade SSE→poll after a session has already framed.

A cancel-induced EventSource drop on a live conversation used to treat the
reconnect as a cold open (``_sawFrame`` is per-connection) and flip the other
tab to 4s REST polling.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "src" / "sharedagents_web" / "static"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node not installed")


def _run_node(js_code: str) -> dict:
    assert NODE is not None
    res = subprocess.run([NODE, "-e", js_code], capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, f"Node script error: {res.stderr}\n{res.stdout}"
    return json.loads(res.stdout)


def test_turn_ended_retrying_does_not_relight_via_microtask():
    src = (STATIC / "streaming.js").read_text()
    assert "queueMicrotask(() => _keepRetryingUI" not in src
    assert "const keepRetrying" in src
    assert "finishTurnChrome" in src


@needs_node
def test_already_framed_session_does_not_auto_downgrade():
    sessions_js = json.dumps(str(STATIC / "sessions.js"))
    script = f"""
    const fs = require('fs');
    const src = fs.readFileSync({sessions_js}, 'utf8');
    const start = src.indexOf('function liveStreamAlreadyFramed');
    const end = src.indexOf('function _sseForcedOff');
    if (start < 0 || end < 0 || end <= start) throw new Error('helper block missing');
    const state = {{ _liveStreamFramedSid: '' }};
    function liveTransport() {{ return 'sse'; }}
    eval(src.slice(start, end));
    const cold = _shouldAutoDowngradeLiveTransport('s1', false);
    _markLiveStreamFramed('s1');
    const afterFrame = _shouldAutoDowngradeLiveTransport('s1', false);
    const thisConnFramed = _shouldAutoDowngradeLiveTransport('s1', true);
    const otherSid = _shouldAutoDowngradeLiveTransport('s2', false);
    process.stdout.write(JSON.stringify({{
      cold, afterFrame, thisConnFramed, otherSid,
      framedSid: state._liveStreamFramedSid,
    }}));
    """
    out = _run_node(script)
    assert out["cold"] is True
    assert out["afterFrame"] is False
    assert out["thisConnFramed"] is False
    assert out["otherSid"] is True
    assert out["framedSid"] == "s1"
