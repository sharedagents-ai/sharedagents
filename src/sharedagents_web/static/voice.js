/* voice.js — extracted from app.js (de-monolith Task 9, behavior-preserving).
   Classic script sharing the global / window.App scope; loaded BEFORE
   app.js per index.html. No references were rewritten. */

const VOICE_TARGET_RATE = 16000;
const VOICE_MIN_CHUNK_SAMPLES = 1600; // 100ms
const VOICE_PREROLL_SAMPLES = 8000; // 500ms
const VOICE_SILENCE_SAMPLES = 14400; // 900ms: a breath or comma pause is not a chunk end
const VOICE_MAX_CHUNK_SAMPLES = 240000; // 15s
const VOICE_OVERLAP_SAMPLES = 12000; // 750ms hard-cap overlap
// Whole-recording review keeps the recording's audio for one final pass; past
// this length the live chunk draft stands on its own.
const VOICE_REVIEW_MAX_SAMPLES = 16000 * 600; // 10 min
const VOICE_RMS_SPEECH = 0.02;
// Below THIS a frame is digital silence, not quiet audio: ~20x under the VAD
// speech trigger above, ~30x above the 16-bit quantisation floor. The VAD
// threshold decides when to close a chunk — it is NOT a "safe to delete" line
// (a soft trailing consonant sits under it), which is why trimming uses this
// one instead. See _dropTrailingSilence.
const VOICE_RMS_SILENCE = 0.001;
const VOICE_GAP_MARK = "\u2026";
const VOICE_WORKLET = `
class VoiceTap extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) this.port.postMessage(ch.slice(0));
    return true;
  }
}
registerProcessor("voice-tap", VoiceTap);
`;

async function toggleRecording() {
  if (state.audioState === "recording") {
    await stopRecording();
  } else if (state.audioState === "idle") {
    await startRecording();
  }
  // ignore if 'processing' or 'unsupported'
}

function _voiceInputEl() {
  return (
    document.getElementById("message-input") ||
    (typeof messageInput !== "undefined" ? messageInput : null)
  );
}

function _setVoiceStageVisible(visible) {
  const stage = document.getElementById("voice-live-stage");
  const canvas = document.getElementById("audio-visualizer");
  const draft = document.querySelector(".composer-draft");
  if (stage) stage.hidden = !visible;
  if (canvas) canvas.hidden = !visible;
  if (draft) draft.classList.toggle("recording", !!visible);
}

function _renderVoiceLiveCaption() {
  const cap = document.getElementById("voice-live-caption");
  const textEl = cap && cap.querySelector(".voice-caption-text");
  const pendingEl = cap && cap.querySelector(".voice-caption-pending");
  const input = _voiceInputEl();
  if (textEl) textEl.textContent = ((input && input.value) || "").trim();
  if (pendingEl) pendingEl.hidden = !(state.chunksInFlight > 0);
}

/** Encode Float32Array mono samples into a standard 16-bit PCM WAV Blob. */
function _encodeWav(samples, sampleRate = 16000) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  function writeString(offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // Mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

/** Downsample Float32Array buffer to 16kHz via box averaging. */
function _downsampleBuffer(buffer, inputRate, targetRate = 16000) {
  // ScriptProcessor channel buffers may be reused after the callback. Retain
  // our own samples for both the live chunk and the whole-recording pass.
  if (targetRate === inputRate) return buffer.slice();
  const ratio = inputRate / targetRate;
  const newLength = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLength);
  let offsetResult = 0;
  let offsetBuffer = 0;
  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
    let accum = 0;
    let count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
      accum += buffer[i];
      count++;
    }
    result[offsetResult] = count > 0 ? accum / count : buffer[offsetBuffer];
    offsetResult++;
    offsetBuffer = nextOffsetBuffer;
  }
  return result;
}

function _processVoiceFrame(inputData, inputRate) {
  if (state.audioState !== "recording") return;
  const downsampled = _downsampleBuffer(inputData, inputRate, VOICE_TARGET_RATE);
  if (state.recordingBlocks) {
    state.recordingSamples = (state.recordingSamples || 0) + downsampled.length;
    if (state.recordingSamples <= VOICE_REVIEW_MAX_SAMPLES) {
      state.recordingBlocks.push(downsampled);
    } else {
      state.recordingBlocks = null; // too long to review; keep the chunk draft
    }
  }

  let sumSq = 0;
  for (let i = 0; i < downsampled.length; i++) {
    sumSq += downsampled[i] * downsampled[i];
  }
  const rms = Math.sqrt(sumSq / downsampled.length);

  if (rms > VOICE_RMS_SPEECH) {
    state.speechDetected = true;
    state.silenceSamples = 0;
  } else {
    state.silenceSamples = (state.silenceSamples || 0) + downsampled.length;
  }

  if (!state.chunkAccumulator) state.chunkAccumulator = [];
  state.chunkAccumulator.push(downsampled);

  let totalSamples = 0;
  for (let i = 0; i < state.chunkAccumulator.length; i++) {
    totalSamples += state.chunkAccumulator[i].length;
  }

  if (!state.speechDetected) {
    while (state.chunkAccumulator.length > 1 && totalSamples > VOICE_PREROLL_SAMPLES) {
      const removed = state.chunkAccumulator.shift();
      totalSamples -= removed.length;
    }
    return;
  }

  if (state.silenceSamples >= VOICE_SILENCE_SAMPLES) {
    _emitVoiceChunk(false, "silence");
  } else if (totalSamples >= VOICE_MAX_CHUNK_SAMPLES) {
    _emitVoiceChunk(false, "max");
  }
}

/** RMS of one accumulator frame — the same measure _processVoiceFrame gates on. */
function _frameRms(block) {
  if (!block || block.length === 0) return 0;
  let sumSq = 0;
  for (let i = 0; i < block.length; i++) sumSq += block[i] * block[i];
  return Math.sqrt(sumSq / block.length);
}

/**
 * Drop a chunk's trailing silence before it is encoded.
 *
 * A "silence"-triggered chunk ends with the whole VOICE_SILENCE_SAMPLES window
 * that closed it, and the stop flush ends with the preroll's quietest frames —
 * bare silence that the ASR model turns into invented filler/punctuation.
 *
 * The cut line is VOICE_RMS_SILENCE (digital silence), NOT VOICE_RMS_SPEECH:
 * VAD's speech threshold only decides when a chunk is closed, and treating
 * everything under it as deletable silently cut real audio — soft final
 * phonemes (a quiet trailing "s"/"t") and quiet trailing words. Frames at or
 * below VOICE_RMS_SILENCE carry nothing at all, so trimming them loses no
 * speech.
 *
 * Returns [] when the whole buffer is at or below that floor (there is nothing
 * to transcribe), otherwise the blocks to encode — never cut before the last
 * frame that carries audio and never shrunk below VOICE_MIN_CHUNK_SAMPLES.
 */
function _dropTrailingSilence(blocks) {
  let lastAudio = -1;
  for (let i = blocks.length - 1; i >= 0; i--) {
    if (_frameRms(blocks[i]) > VOICE_RMS_SILENCE) {
      lastAudio = i;
      break;
    }
  }
  if (lastAudio < 0) return [];

  let end = lastAudio + 1;
  let kept = 0;
  for (let i = 0; i < end; i++) kept += blocks[i].length;
  while (end < blocks.length && kept < VOICE_MIN_CHUNK_SAMPLES) {
    kept += blocks[end].length;
    end++;
  }
  return end >= blocks.length ? blocks : blocks.slice(0, end);
}

function _emitVoiceChunk(isFinal = false, reason = "silence") {
  if (!state.chunkAccumulator || state.chunkAccumulator.length === 0) return;
  const totalLength = state.chunkAccumulator.reduce((sum, b) => sum + b.length, 0);
  if (totalLength < VOICE_MIN_CHUNK_SAMPLES && !isFinal) return;

  const blocks = _dropTrailingSilence(state.chunkAccumulator);
  const sendLength = blocks.reduce((sum, b) => sum + b.length, 0);
  const merged = new Float32Array(sendLength);
  let offset = 0;
  for (const block of blocks) {
    merged.set(block, offset);
    offset += block.length;
  }

  const keepOverlap =
    reason === "max" && !isFinal && merged.length > VOICE_OVERLAP_SAMPLES;
  state.chunkAccumulator = keepOverlap
    ? [merged.slice(merged.length - VOICE_OVERLAP_SAMPLES)]
    : [];
  state.silenceSamples = 0;
  if (reason !== "max") state.speechDetected = false;

  // Never send a chunk that carries no audio — in particular the stop flush's
  // residual buffer, which is normally the preroll's pure silence. qwen3-asr
  // decodes silent audio as CJK filler ("嗯。") and _flushCommittedTranscript
  // appends whatever comes back to the composer verbatim, so a silent request
  // leaves a stray "嗯。" in the draft. The gate is the AUDIO in the buffer, not
  // the VAD's speech flag: a quiet-but-real trailing fragment (speechDetected
  // false, every frame under VOICE_RMS_SPEECH) still goes out, and the
  // server-side guard in /api/transcribe rejects what is actually silence.
  if (blocks.length === 0) return;

  const wavBlob = _encodeWav(merged, VOICE_TARGET_RATE);
  const seq = ++state.chunkSequence;

  const promise = _sendChunkForTranscription(wavBlob, seq);
  if (state.pendingChunkTranscriptions) {
    state.pendingChunkTranscriptions.push(promise);
  }
}

function _markChunkResult(seq, text) {
  if (!state.receivedChunks) return;
  state.receivedChunks.set(seq, text);
  _flushCommittedTranscript();
}

async function _sendChunkForTranscription(wavBlob, seq) {
  state.chunksInFlight = (state.chunksInFlight || 0) + 1;
  _renderVoiceLiveCaption();
  try {
    const formData = new FormData();
    formData.append("audio", wavBlob, `chunk_${seq}.wav`);
    // Echo the language the server decoded an earlier chunk in, so a short
    // chunk of the same recording cannot flip language. The server decides
    // whether to honour it.
    if (state.voiceLanguage) formData.append("language", state.voiceLanguage);

    const resp = await fetch(BASE + "/api/transcribe", {
      method: "POST",
      body: formData,
    });
    if (!resp.ok) {
      _markChunkResult(seq, VOICE_GAP_MARK);
      return;
    }
    const data = await resp.json();
    if (data.language && !state.voiceLanguage) state.voiceLanguage = data.language;
    _markChunkResult(seq, (data.text || "").trim());
  } catch (err) {
    console.warn(`Chunk ${seq} transcription error:`, err);
    _markChunkResult(seq, VOICE_GAP_MARK);
  } finally {
    state.chunksInFlight = Math.max(0, (state.chunksInFlight || 1) - 1);
    _renderVoiceLiveCaption();
  }
}

function _flushCommittedTranscript() {
  const input = _voiceInputEl();
  if (!input || !state.receivedChunks) return;
  // Latch edits BEFORE incorporating late replies. Updating voiceDraftValue
  // after an append must never make a user's edit eligible for replacement.
  if (state.voiceDraftValue !== undefined && input.value !== state.voiceDraftValue) {
    state.voiceUserEdited = true;
  }

  while (state.receivedChunks.has(state.nextExpectedSeq)) {
    const nextText = state.receivedChunks.get(state.nextExpectedSeq);
    state.receivedChunks.delete(state.nextExpectedSeq);
    state.nextExpectedSeq++;

    if (nextText) {
      if (input.value && !input.value.endsWith(" ") && nextText !== VOICE_GAP_MARK) {
        input.value += " ";
      }
      input.value += nextText;
      input.dispatchEvent(new Event("input", { bubbles: true }));
    }
  }
  state.voiceDraftValue = input.value;
  _renderVoiceLiveCaption();
}

/**
 * One transcription of the WHOLE recording, replacing the chunk-by-chunk
 * draft. Chunks are decoded alone, so a word split at a boundary or a chunk
 * misheard without its sentence stays wrong; the full audio is decoded with
 * its context (and its language detected over all of it). The draft stays
 * when the user typed meanwhile, the pass fails, or it returns nothing.
 */
async function _reviewWholeRecording() {
  const blocks = state.recordingBlocks;
  state.recordingBlocks = null;
  const input = _voiceInputEl();
  if (!blocks || !input || (state.chunkSequence || 0) < 2) return;
  if (state.voiceUserEdited || input.value !== state.voiceDraftValue) return;
  const draftValue = input.value;
  const kept = _dropTrailingSilence(blocks);
  if (kept.length === 0) return;
  const total = kept.reduce((sum, b) => sum + b.length, 0);
  const merged = new Float32Array(total);
  let offset = 0;
  for (const block of kept) {
    merged.set(block, offset);
    offset += block.length;
  }
  try {
    const formData = new FormData();
    formData.append("audio", _encodeWav(merged, VOICE_TARGET_RATE), "recording.wav");
    formData.append("final", "1");
    formData.append("draft", draftValue);
    formData.append("prefix", state.voiceBaseValue || "");
    const resp = await fetch(BASE + "/api/transcribe", { method: "POST", body: formData });
    if (!resp.ok) return;
    const text = ((await resp.json()).text || "").trim();
    if (!text || state.voiceUserEdited || input.value !== draftValue) return;
    const base = state.voiceBaseValue || "";
    input.value = base + (base && !base.endsWith(" ") ? " " : "") + text;
    state.voiceDraftValue = input.value;
    input.dispatchEvent(new Event("input", { bubbles: true }));
  } catch (err) {
    console.warn("Whole-recording review failed:", err);
  }
}

/** Test hook: commit one live caption chunk through the real sequence queue. */
function __voiceCommitTestChunk(text) {
  if (!state.receivedChunks) {
    state.receivedChunks = new Map();
    state.nextExpectedSeq = 1;
  }
  if (!state.nextExpectedSeq) state.nextExpectedSeq = 1;
  state.chunksInFlight = Math.max(1, state.chunksInFlight || 0);
  state.receivedChunks.set(state.nextExpectedSeq, text);
  _flushCommittedTranscript();
}

async function startRecording() {
  if (state.monitorReadOnly) return;
  if (state.audioState !== "idle") return;
  if (state.isStreaming) return; // don't start recording while a turn runs
  if (state.realtimeState === "connecting" || state.realtimeState === "live")
    return; // realtime owns the mic

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    state.recordingStream = stream;
    state.audioState = "recording";
    updateMicButton();

    // Chunking state
    state.chunkAccumulator = [];
    state.silenceSamples = 0;
    state.speechDetected = false;
    state.chunkSequence = 0;
    state.nextExpectedSeq = 1;
    state.receivedChunks = new Map();
    state.pendingChunkTranscriptions = [];
    state.chunksInFlight = 0;
    state.mediaChunks = [];
    state.recordingBlocks = [];
    state.recordingSamples = 0;
    state.voiceLanguage = "";
    const draftInput = _voiceInputEl();
    state.voiceBaseValue = draftInput ? draftInput.value : "";
    state.voiceDraftValue = state.voiceBaseValue;
    state.voiceUserEdited = false;

    await startAudioVisualizer(stream); // live waveform, captions, and VAD
  } catch (err) {
    stopAudioVisualizer();
    state.audioState =
      err.name === "NotAllowedError" || err.name === "PermissionDeniedError"
        ? "idle"
        : err.name === "NotFoundError"
          ? "unsupported"
          : "idle";
    if (state.audioState === "unsupported") micBtn.disabled = true;
    updateMicButton();
    const msg =
      err.name === "NotAllowedError"
        ? "Microphone access denied. Please allow microphone access in your browser settings."
        : err.name === "NotFoundError"
          ? "No microphone found."
          : `Microphone error: ${err.message}`;
    const status = $("#status-text");
    if (status) {
      status.textContent = `🎤 ${msg}`;
      status.style.color = "";
      setTimeout(() => {
        if (status.textContent && status.textContent.startsWith("🎤")) {
          status.textContent = "";
        }
      }, 5000);
    }
  }
}

async function stopRecording() {
  if (state.audioState === "processing") return;
  if (state.audioState !== "recording") {
    if (state.recordingStream) {
      state.recordingStream.getTracks().forEach((t) => t.stop());
      state.recordingStream = null;
    }
    state.audioState = "idle";
    updateMicButton();
    return;
  }

  state.audioState = "processing";
  updateMicButton();

  // 1. Flush residual PCM, or the whole-file MediaRecorder fallback.
  if (state.recorder && state.recorder.state !== "inactive") {
    await new Promise((resolve) => {
      state.recorder.onstop = resolve;
      try {
        state.recorder.stop();
      } catch (_) {
        resolve();
      }
    });
    const blob = new Blob(state.mediaChunks || [], {
      type: state.recorder.mimeType || "audio/webm",
    });
    state.recorder = null;
    state.mediaChunks = [];
    if (blob.size > 0) {
      const seq = ++state.chunkSequence;
      const promise = _sendChunkForTranscription(blob, seq);
      if (state.pendingChunkTranscriptions) {
        state.pendingChunkTranscriptions.push(promise);
      }
    }
  } else {
    // Flush the residual PCM. _emitVoiceChunk sends it only when it carries
    // audio: the stop click's usual residual (the preroll's pure silence) is
    // dropped WITHOUT a request, so the silent final chunk that made the ASR
    // model answer "嗯。" can't happen. Audio is what decides this, not
    // state.speechDetected — a quiet trailing fragment word (every frame under
    // the VAD's speech threshold) must still be transcribed.
    _emitVoiceChunk(true, "final");
  }

  // 2. Disconnect audio processor & hide the live stage (draft stays visible)
  stopAudioVisualizer();
  if (state.recordingStream) {
    state.recordingStream.getTracks().forEach((t) => t.stop());
    state.recordingStream = null;
  }

  // 3. Wait for any in-flight chunk requests. Already-committed seqs are
  // visible in the composer; this only covers the residual / late replies.
  if (state.pendingChunkTranscriptions && state.pendingChunkTranscriptions.length > 0) {
    await Promise.allSettled(state.pendingChunkTranscriptions);
    _flushCommittedTranscript();
  }

  // 4. Review the whole recording once and replace the chunk draft.
  await _reviewWholeRecording();

  state.audioState = "idle";
  updateMicButton();
  const input = _voiceInputEl();
  if (input) {
    input.style.display = "";
    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
  }
}

async function _attachVoiceProcessor(audioCtx, source) {
  const inputRate = audioCtx.sampleRate;
  const silenceGain = audioCtx.createGain();
  silenceGain.gain.value = 0;
  state.silenceGainNode = silenceGain;

  if (audioCtx.audioWorklet) {
    try {
      const url = URL.createObjectURL(
        new Blob([VOICE_WORKLET], { type: "application/javascript" }),
      );
      try {
        await audioCtx.audioWorklet.addModule(url);
      } finally {
        URL.revokeObjectURL(url);
      }
      const node = new AudioWorkletNode(audioCtx, "voice-tap");
      node.port.onmessage = (e) => _processVoiceFrame(e.data, inputRate);
      source.connect(node);
      node.connect(silenceGain);
      silenceGain.connect(audioCtx.destination);
      state.audioProcessor = node;
      return true;
    } catch (err) {
      console.warn("Voice AudioWorklet unavailable:", err);
    }
  }

  if (audioCtx.createScriptProcessor) {
    const processor = audioCtx.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = (e) => {
      _processVoiceFrame(e.inputBuffer.getChannelData(0), inputRate);
    };
    source.connect(processor);
    processor.connect(silenceGain);
    silenceGain.connect(audioCtx.destination);
    state.audioProcessor = processor;
    return true;
  }
  return false;
}

function _startMediaRecorderFallback(stream) {
  if (typeof MediaRecorder === "undefined") return false;
  try {
    const rec = new MediaRecorder(stream);
    state.mediaChunks = [];
    rec.ondataavailable = (e) => {
      if (e.data && e.data.size) state.mediaChunks.push(e.data);
    };
    rec.start();
    state.recorder = rec;
    return true;
  } catch (err) {
    console.warn("MediaRecorder fallback unavailable:", err);
    return false;
  }
}

/** Start a real-time waveform + live captions. The composer textarea stays visible. */
async function startAudioVisualizer(stream) {
  const canvas = document.getElementById("audio-visualizer");
  if (!canvas) return;

  try {
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === "suspended") {
      try { await audioCtx.resume(); } catch (_) {}
    }
    const source = audioCtx.createMediaStreamSource(stream);
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 256;
    analyser.smoothingTimeConstant = 0.8;
    source.connect(analyser);

    state.audioContext = audioCtx;
    state.audioSourceNode = source;
    state.analyserNode = analyser;

    const hooked = await _attachVoiceProcessor(audioCtx, source);
    if (!hooked && state.audioState === "recording") {
      _startMediaRecorderFallback(stream);
    }

    // Waveform + caption overlay above the still-visible textarea.
    const input = _voiceInputEl();
    if (input) input.style.display = "";
    _setVoiceStageVisible(true);
    _renderVoiceLiveCaption();
    const stage = document.getElementById("voice-live-stage") || canvas;
    const rect = stage.getBoundingClientRect();
    canvas.width = rect.width * (window.devicePixelRatio || 1);
    canvas.height = rect.height * (window.devicePixelRatio || 1);

    const ctx = canvas.getContext("2d");
    ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
    const W = rect.width;
    const H = rect.height;
    // Resolve theme colours to real values — the canvas API cannot parse a
    // CSS ``var(...)`` string, so read them off the document once up front.
    const cs = getComputedStyle(document.documentElement);
    const colBg =
      (cs.getPropertyValue("--bg-secondary") || "#16162a").trim() || "#16162a";
    const colBorder =
      (cs.getPropertyValue("--border-color") || "#2a2a3e").trim() || "#2a2a3e";
    const colAccent =
      (cs.getPropertyValue("--accent") || "#4fc3f7").trim() || "#4fc3f7";

    function draw() {
      // Run while push-to-talk records OR a realtime session is connecting/live
      // (onMicStreamStart fires during 'connecting', before 'live' is set).
      const rtActive =
        state.realtimeState === "connecting" || state.realtimeState === "live";
      if (state.audioState !== "recording" && !rtActive) return;

      const data = new Uint8Array(analyser.frequencyBinCount);
      analyser.getByteTimeDomainData(data);

      ctx.fillStyle = colBg;
      ctx.fillRect(0, 0, W, H);

      // Center line (silence reference)
      ctx.strokeStyle = colBorder;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, H / 2);
      ctx.lineTo(W, H / 2);
      ctx.stroke();

      // Waveform path
      ctx.strokeStyle = colAccent;
      ctx.lineWidth = 2;
      ctx.beginPath();

      const slice = W / data.length;
      for (let i = 0; i < data.length; i++) {
        const x = i * slice;
        const y = (data[i] / 128) * (H / 2);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Fill under the waveform (gradient)
      ctx.lineTo(W, H / 2);
      ctx.closePath();
      const grad = ctx.createLinearGradient(0, 0, 0, H);
      grad.addColorStop(0, "rgba(79, 195, 247, 0.15)");
      grad.addColorStop(1, "rgba(79, 195, 247, 0)");
      ctx.fillStyle = grad;
      ctx.fill();

      requestAnimationFrame(draw);
    }

    draw();
  } catch (err) {
    console.warn("Audio visualizer unavailable:", err);
    const input = _voiceInputEl();
    if (input) input.style.display = "";
    _setVoiceStageVisible(true);
    if (state.audioState === "recording") _startMediaRecorderFallback(stream);
  }
}

/** Stop the waveform display and clean up the AudioContext. */
function stopAudioVisualizer() {
  _setVoiceStageVisible(false);
  const pending = document.querySelector(".voice-caption-pending");
  if (pending) pending.hidden = true;
  const input = _voiceInputEl();
  if (input) input.style.display = "";

  if (state.audioProcessor) {
    try {
      state.audioProcessor.disconnect();
      state.audioProcessor.onaudioprocess = null;
    } catch (_) {}
    state.audioProcessor = null;
  }
  if (state.silenceGainNode) {
    try {
      state.silenceGainNode.disconnect();
    } catch (_) {}
    state.silenceGainNode = null;
  }
  if (state.audioSourceNode) {
    try {
      state.audioSourceNode.disconnect();
    } catch (_) {}
    state.audioSourceNode = null;
  }
  if (state.analyserNode) {
    try {
      state.analyserNode.disconnect();
    } catch (_) {}
    state.analyserNode = null;
  }

  if (state.audioContext && state.audioContext.state !== "closed") {
    state.audioContext.close().catch(() => {});
  }
  state.audioContext = null;
}

function updateMicButton() {
  if (!micBtn) return;
  micBtn.className = "mic-btn";
  const micIcon = micBtn.querySelector(".mic-icon");
  const stopIcon = micBtn.querySelector(".mic-stop-icon");
  if (micIcon) micIcon.style.display = "";
  if (stopIcon) stopIcon.style.display = "none";
  if (state.monitorReadOnly) {
    micBtn.disabled = true;
    micBtn.title = "Read-only subagent session";
    if (typeof updateRealtimeButton === "function") updateRealtimeButton();
    return;
  }

  switch (state.audioState) {
    case "recording":
      micBtn.classList.add("recording");
      micBtn.title = state.chunksInFlight
        ? "Stop recording — transcribing live"
        : "Stop recording";
      micBtn.setAttribute("aria-label", "Stop recording");
      if (micIcon) micIcon.style.display = "none";
      if (stopIcon) stopIcon.style.display = "";
      break;
    case "processing":
      micBtn.classList.add("processing");
      micBtn.title = "Transcribing…";
      micBtn.disabled = true;
      break;
    case "unsupported":
      micBtn.disabled = true;
      micBtn.title = "Microphone not available";
      break;
    default: // idle
      // Push-to-talk and realtime voice share the same mic; only one runs at a
      // time, so disable the mic while a realtime session is connecting/live.
      if (
        state.realtimeState !== "idle" &&
        state.realtimeState !== "unsupported"
      ) {
        micBtn.disabled = true;
        micBtn.title = "Realtime voice is active";
      } else {
        micBtn.title = state.isStreaming
          ? "Wait for the current turn to finish"
          : "Record audio message";
        micBtn.disabled = !!state.isStreaming;
      }
      break;
  }
  // Keep the realtime button's enabled state in sync (mic owns the device while
  // recording/processing). Guarded — updateRealtimeButton never calls back here.
  if (typeof updateRealtimeButton === "function") updateRealtimeButton();
}

// ══════════════════════════════════════════════════════════════════════
// Realtime voice (inline) — the rightmost input-bar button. Reuses the tested
// realtime.js engine (transport / audio / wake) but renders the live
// conversation through the SAME chat widgets as everything else: user + bot
// turns become normal bubbles via appendMessage()/setAssistantContent(), and the
// mic waveform reuses the existing #audio-visualizer canvas. The engine streams
// to the standalone realtime server (SHAREDAGENTS_REALTIME_URL); only rendering is
// swapped via the hooks passed to SharedagentsRealtime.configure().
// ══════════════════════════════════════════════════════════════════════

/** Resolve the realtime server URL: injected meta, else derive <host>:8123. */
