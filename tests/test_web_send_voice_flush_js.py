"""Send while the mic is armed must finish dictation, then send.

Bug (owner report): pressing Send while push-to-talk was recording did not
send — the composer was either still empty (the click fell through to the
empty-composer Stop path / no-op) or held only a half-committed transcript,
while the mic kept appending text afterwards. ``sendMessage`` now drains the
armed mic FIRST: recording → stopRecording (final chunk + whole-recording
review) → the transcript lands in the composer → only then is the
Send/Queue/Steer/Stop decision made. A second click during the drain is a
no-op, not a double submit.

Drives the REAL ``voice.js`` + ``attachments.js`` + ``send.js`` in node's
``vm`` (same convention as ``test_voice_silent_final_chunk_js.py``): the ASR
fetch stub answers "dictated message", the prompt POST is captured, and the
assertions read the posted prompt text — the user-visible property.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "src" / "sharedagents_web" / "static"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node not installed")

HARNESS = r"""
const fs = require('fs'), vm = require('vm');
const input = JSON.parse(process.argv[1]);
const noop = () => {};
const posts = [];      // /api/transcribe calls
const prompts = [];    // /api/sessions/<sid>/prompt calls
const cancels = [];    // cancelStreaming calls (must stay empty here)

function makeBtn() {
  return {
    classList: { _s: new Set(),
      toggle(c, on) { on ? this._s.add(c) : this._s.delete(c); },
      add(c) { this._s.add(c); },
      contains(c) { return this._s.has(c); } },
    style: {}, title: '', hidden: false, disabled: false,
    setAttribute() {}, querySelector() { return { style: {} }; },
  };
}
const sendBtn = makeBtn();
const composer = {
  value: '', style: {}, hidden: false,
  dispatchEvent: () => {}, focus: () => {}, setSelectionRange: () => {},
};

const ctx = vm.createContext({
  state: {
    currentSessionId: 'sess', isStreaming: false, treeLive: false,
    monitorReadOnly: false, queueSubmitting: false, promptSubmitting: false,
    voiceSendFlushing: false, pendingPrompt: null, pendingCancel: null,
    pastedImages: [], attachedTextFiles: [], pendingImageReads: [],
    pendingTextReads: [], askUserCards: {}, msgCache: [],
    modelSheet: { phase: 'CLOSED', effective: {} },
    modelOptionsByAgent: {}, currentAgent: 'jarvis', me: 'me',
    liveSid: 'sess', liveSource: { readyState: 1 },
    audioState: 'idle',
    sessions: {
      sess: { state: 'idle', session_state: 'idle',
        activity: { phase: 'idle', own_live: false, tree_busy: false,
                    capabilities: { can_stop: false, can_send: true } },
        _liveSnapshot: { session_state: 'idle', tree_busy: false,
                         own_live: false, descendants_running: false,
                         active_descendant_count: 0, open_job_count: 0 } },
    },
  },
  BASE: '',
  composer,
  messageInput: composer,
  sendBtn, micBtn: null, pastePreview: { hidden: true, innerHTML: '' },
  statusText: { textContent: '' },
  _turn: null,
  document: {
    addEventListener: noop,
    getElementById: (id) => (id === 'message-input' ? composer : null),
    querySelector: () => null, querySelectorAll: () => [],
    documentElement: { style: {} },
    createElement: () => ({ style: {}, classList: { add: noop, toggle: noop, remove: noop } }),
  },
  window: { addEventListener: noop, App: {} },
  localStorage: { getItem: () => null, setItem: noop, removeItem: noop },
  sessionStorage: { getItem: () => null, setItem: noop },
  navigator: {},
  crypto: { randomUUID: () => 'req-' + Math.random().toString(16).slice(2) },
  setTimeout, clearTimeout,
  setInterval: noop, clearInterval: noop, requestAnimationFrame: noop,
  console,
  Event: function (t) { this.type = t; },
  EventSource: { CLOSED: 2 },
  Blob: function (parts, opts) {
    this.parts = parts || [];
    this.type = (opts && opts.type) || '';
    let size = 0;
    for (const p of this.parts) size += (p && (p.byteLength || p.size)) || 0;
    this.size = size;
  },
  FormData: function () {
    this.entries = [];
    this.append = (k, v, n) => { this.entries.push({ k, v, n }); };
  },
  fetch: (url, opts) => {
    posts.push(url);
    const respond = () => Promise.resolve({
      ok: true, status: 200,
      json: () => Promise.resolve({ text: 'dictated message', language: 'en' }),
    });
    if (!input.transcribeDelay) return respond();
    return new Promise((r) => setTimeout(() => r(respond()), input.transcribeDelay));
  },
  // ── stubs for the downstream send path ──
  apiPost: (url, payload) => {
    prompts.push({ url, prompt: payload && payload.prompt });
    return Promise.resolve({ accepted: true, start_offset: 0 });
  },
  escapeHtml: String, escapeAttr: String,
  appendSystemMessage: noop, appendMessage: () => ({ _el: true }),
  viewerAuthor: () => 'me',
  ensureModelSheetState: noop, setModelSheetPhase: noop,
  hideSlashMenu: noop, slashGoalSetPrompt: () => '',
  commandRequiresSession: () => false, createSession: async () => true,
  runCommand: async () => ({}), switchFailureMessage: (e) => String(e),
  cloneModelSelection: () => ({}), firstIntentForSession: () => ({}),
  firstTurnInitialModel: async () => '', modelToApplyBeforeTurn: () => '',
  visToApplyBeforeTurn: () => [], modelApplyCommand: () => '',
  applyAgentSwitch: async () => true, applySessionState: () => false,
  openLiveStream: noop, refreshSessionState: async () => ({}),
  renderHarnessWidget: noop, renderSessionList: noop,
  renderStatusSegments: noop, renderSessionStatusBar: noop,
  updateHeader: noop, syncTeacherState: noop,
  updateAgentMonAccordion: noop, updateRealtimeButton: noop,
  _turnIndicatorStart: noop, _turnIndicatorMark: noop,
  _turnIndicatorTick: noop, _turnIndicatorStop: noop,
  isUnusedWebPlaceholder: () => false, selectedPickerAgent: () => 'jarvis',
  pickerModelAlias: () => '', isLocalLlmAgent: () => false,
  isModeSentinel: () => false, uiSseIsOpen: () => true,
  sessionGwHash: (sid) => sid,
  loadSessionList: noop, loadTopicList: noop,
  getComputedStyle: () => ({ getPropertyValue: () => '' }),
});
for (const f of ['attachments.js', 'send.js', 'voice.js'])
  vm.runInContext(fs.readFileSync(input.static + '/' + f, 'utf8'), ctx, { filename: f });
ctx.cancelStreaming = () => { cancels.push(1); return Promise.resolve(); };
// The 5-min inactivity watchdog is irrelevant here and would hold the loop.
ctx._startStreamingWatchdog = noop;
ctx.__out = { posts, prompts, cancels, sendBtn, composer };
(async () => {
  const vec = fs.readFileSync(input.vec, 'utf8');
  const extra = await vm.runInContext('(async () => {' + vec + '\n})()', ctx);
  process.stdout.write(JSON.stringify({
    posts, prompts, cancels: cancels.length,
    audioState: ctx.state.audioState, composer: composer.value, extra,
  }));
})().catch((e) => { process.stderr.write(String((e && e.stack) || e)); process.exitCode = 1; });
"""


def _run(vectors: str, **kwargs) -> dict:
    assert NODE is not None, "node is required for JS vector tests"
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="rtxsend-"))
    try:
        vec = tmp / "vec.js"
        vec.write_text(vectors, encoding="utf-8")
        done = subprocess.run(
            [NODE, "-e", HARNESS, json.dumps({"static": str(STATIC), "vec": str(vec), **kwargs})],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert done.returncode == 0, done.stderr
        return json.loads(done.stdout)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# One buffered speech chunk — the same setup the voice tests use. The final
# flush transcribes it as "dictated message".
_ARM = """
state.audioState = 'recording';
state.chunkAccumulator = [new Float32Array(1600).fill(0.5)];
state.silenceSamples = 0; state.speechDetected = true;
state.chunkSequence = 0; state.nextExpectedSeq = 1;
state.receivedChunks = new Map(); state.pendingChunkTranscriptions = [];
state.chunksInFlight = 0; state.mediaChunks = []; state.recordingBlocks = [];
state.recordingSamples = 0; state.voiceLanguage = '';
state.voiceBaseValue = ''; state.voiceDraftValue = ''; state.voiceUserEdited = false;
state.recorder = null; state.recordingStream = null;
composer.value = '';
"""


@needs_node
def test_send_while_recording_flushes_the_transcript_then_posts():
    """Mid-recording Send = stop dictation + send what was said."""
    out = _run(_ARM + "return await sendMessage();")
    assert out["extra"] is True
    assert out["audioState"] == "idle"
    assert out["posts"] == ["/api/transcribe"]  # the final chunk went out
    assert len(out["prompts"]) == 1
    assert out["prompts"][0]["prompt"] == "dictated message"
    assert out["prompts"][0]["url"] == "/api/sessions/sess/prompt"
    assert out["cancels"] == 0  # never reached the Stop path


@needs_node
def test_a_second_click_during_the_flush_does_not_double_submit():
    out = _run(
        _ARM
        + """
        const first = sendMessage();
        const second = await sendMessage();  // must land inside the drain window
        const firstResult = await first;
        return { first: firstResult, second };
        """,
        transcribeDelay=150,
    )
    assert out["extra"]["first"] is True
    assert out["extra"]["second"] is False
    assert len(out["prompts"]) == 1


@needs_node
def test_send_while_mic_is_processing_waits_for_the_drain():
    """Mic button already stopped recording: Send waits out the in-flight
    drain instead of posting the half-committed draft."""
    out = _run(
        _ARM
        + """
        const stop = stopRecording();          // flips audioState to 'processing'
        const result = await sendMessage();    // waits out the drain, then sends
        await stop;
        return result;
        """,
        transcribeDelay=80,
    )
    assert out["extra"] is True
    assert out["audioState"] == "idle"
    assert len(out["prompts"]) == 1
    assert out["prompts"][0]["prompt"] == "dictated message"


@needs_node
def test_send_with_idle_mic_never_touches_the_recorder():
    """Control: an ordinary typed send takes the old path unchanged."""
    out = _run("""
    composer.value = 'typed text';
    return await sendMessage();
    """)
    assert out["extra"] is True
    assert out["posts"] == []  # no transcription call
    assert len(out["prompts"]) == 1
    assert out["prompts"][0]["prompt"] == "typed text"


@needs_node
def test_flush_timeout_bails_instead_of_posting_a_half_transcript():
    """A still-running drain must not release Send: the half transcript would
    post and late chunks would land in the cleared composer (or an empty draft
    + busy tree would hit cancelStreaming). On timeout sendMessage bails —
    draft preserved, nothing posted — and the drain keeps writing into the
    intact composer so the NEXT click submits the completed transcript."""
    out = _run(
        _ARM
        + """
        state.voiceFlushTimeoutMs = 30;          // bound below the ASR delay
        const first = await sendMessage();       // must bail, not send
        const bailedFlushing = state.voiceSendFlushing;
        // The abandoned drain still completes into the composer.
        const deadline = Date.now() + 3000;
        while (state.audioState === 'processing' && Date.now() < deadline) {
          await new Promise((r) => setTimeout(r, 25));
        }
        const afterDrain = composer.value;
        const second = await sendMessage();      // now sends the whole transcript
        return { first, second, bailedFlushing, afterDrain };
        """,
        transcribeDelay=200,
    )
    assert out["extra"]["first"] is False
    assert out["extra"]["bailedFlushing"] is False  # guard released on bail
    assert out["extra"]["afterDrain"] == "dictated message"
    assert out["extra"]["second"] is True
    assert out["cancels"] == 0  # never hit Stop path
    assert len(out["prompts"]) == 1  # one post, on click 2
    assert out["prompts"][0]["prompt"] == "dictated message"
