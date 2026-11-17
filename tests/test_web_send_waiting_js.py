"""send.js ``sessionHasActiveWork`` for ``waiting`` rows (JS vector tests).

``waiting`` means the parent has no own turn left (live_turn_idx is NULL) —
it belongs in the idle-state set, where the snapshot's own counts decide
whether anything abortable remains. A wedged row (lost clear event, or a
hidden recap side-call counted by an older backend) sits at ``waiting`` with
zero live descendants and zero open jobs: the composer must render plain
Send, not an armed Stop on a session whose last turn already succeeded.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "src" / "sharedagents_web" / "static"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node not installed")


def _run_node_script(js_code: str) -> dict:
    assert NODE is not None
    res = subprocess.run([NODE, "-e", js_code], capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, f"Node script error: {res.stderr}"
    return json.loads(res.stdout)


def _composer_harness(body: str) -> dict:
    """Run ``body`` against the real send.js with a minimal composer DOM."""
    send_js = json.dumps(str(STATIC / "send.js"))
    script = f"""
    const fs = require('fs'), vm = require('vm');
    const sendSrc = fs.readFileSync({send_js}, 'utf8');
    function makeBtn() {{
      return {{
        classList: {{ _s: new Set(),
          toggle(c, on) {{ on ? this._s.add(c) : this._s.delete(c); }},
          contains(c) {{ return this._s.has(c); }} }},
        style: {{}}, title: '', hidden: true, disabled: false,
        setAttribute() {{}}, querySelector() {{ return {{ style: {{}} }}; }},
      }};
    }}
    const sendBtn = makeBtn(), treeBtn = makeBtn();
    const ctx = vm.createContext({{
      state: {{
        currentSessionId: 'sess_1', isStreaming: false, treeLive: false,
        monitorReadOnly: false, pastedImages: [], attachedTextFiles: [],
        askUserCards: {{}},
        sessions: {{
          sess_1: {{
            state: 'idle', session_state: 'idle',
            activity: {{ phase: 'idle', own_live: true, tree_busy: false,
                        capabilities: {{ can_stop: false, can_send: true }} }},
            _liveSnapshot: {{ session_state: 'idle', tree_busy: false,
                              own_live: false, descendants_running: false,
                              active_descendant_count: 0 }},
          }},
        }},
      }},
      messageInput: {{ value: '' }},
      sendBtn,
      document: {{ getElementById: (id) => id === 'tree-stop-btn' ? treeBtn : null }},
      statusText: {{ textContent: '' }},
      renderStatusSegments: () => {{}},
      updateAgentMonAccordion: () => {{}},
      _stopStreamingWatchdog: () => {{}},
      console,
    }});
    vm.runInContext(sendSrc, ctx);
    vm.runInContext(`
      function composerHasDraft() {{ return !!messageInput.value.trim(); }}
      function treeBusy() {{ return !!(state.isStreaming || state.treeLive); }}
      function askUserHasPendingCard() {{ return false; }}
      function composerWillQueue() {{ return false; }}
      function composerWillSteerBusy() {{ return false; }}
      function updateSendState() {{ updateSendButton(); }}
    `, ctx);
    {body}
    """
    return _run_node_script(script)


@needs_node
def test_wedged_waiting_with_no_live_descendants_shows_send_not_stop():
    """``waiting`` + zero counted work is a wedge — Stop could kill nothing."""
    out = _composer_harness("""
    Object.assign(ctx.state, { treeLive: true });
    Object.assign(ctx.state.sessions.sess_1, {
      state: 'waiting', session_state: 'waiting', tree_busy: true,
      activity: { phase: 'idle', own_live: false, tree_busy: true,
                  capabilities: { can_stop: true, can_send: false } },
      _liveSnapshot: { session_state: 'waiting', tree_busy: true,
                       own_live: false, descendants_running: false,
                       active_descendant_count: 0, open_job_count: 0 },
    });
    vm.runInContext('updateSendButton()', ctx);
    process.stdout.write(JSON.stringify({
      active: vm.runInContext('sessionHasActiveWork()', ctx),
      sendStop: sendBtn.classList.contains('stop-btn'),
    }));
    """)
    assert out == {"active": False, "sendStop": False}


@needs_node
def test_waiting_with_live_descendants_keeps_stop():
    """A real waiting-on-subagents row still arms the tree Stop."""
    out = _composer_harness("""
    Object.assign(ctx.state.sessions.sess_1, {
      state: 'waiting', session_state: 'waiting', tree_busy: true,
      _liveSnapshot: { session_state: 'waiting', tree_busy: true,
                       descendants_running: true, active_descendant_count: 1 },
    });
    vm.runInContext('updateSendButton()', ctx);
    process.stdout.write(JSON.stringify({
      active: vm.runInContext('sessionHasActiveWork()', ctx),
      sendStop: sendBtn.classList.contains('stop-btn'),
    }));
    """)
    assert out == {"active": True, "sendStop": True}


@needs_node
def test_waiting_with_an_open_job_keeps_stop():
    """A dispatched-but-not-yet-indexed child (open job) is real work."""
    out = _composer_harness("""
    Object.assign(ctx.state.sessions.sess_1, {
      state: 'waiting', session_state: 'waiting', tree_busy: true,
      _liveSnapshot: { session_state: 'waiting', tree_busy: true,
                       descendants_running: false,
                       active_descendant_count: 0, open_job_count: 1 },
    });
    process.stdout.write(JSON.stringify({
      active: vm.runInContext('sessionHasActiveWork()', ctx),
    }));
    """)
    assert out == {"active": True}
