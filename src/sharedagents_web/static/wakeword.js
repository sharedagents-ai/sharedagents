/* Client-side wake-word engine (openWakeWord, ONNX, 100% in-browser).
 *
 * Runs the openWakeWord pipeline locally so the realtime client only streams
 * upstream AFTER the wake word fires — the room audio never leaves the device
 * until you address Jarvis. Pure WASM inference via onnxruntime-web; no audio
 * and no model call ever touches the network.
 *
 * Pipeline (per 1280-sample / 80 ms frame @ 16 kHz), faithfully ported from the
 * reference openwakeword_wasm implementation — the magic constants below are
 * load-bearing (model-trained), do not "tidy" them:
 *   1. melspectrogram.onnx: [1,1280] → 5×32 mels; transform x/10 + 2.
 *   2. embedding_model.onnx: a 76-frame mel window [1,76,32,1] → 96-d embedding;
 *      the mel buffer slides forward 8 frames per embedding.
 *   3. <keyword>.onnx: the last 16 embeddings [1,16,96] → wake score.
 *   4. silero_vad.onnx gates detections to actual speech (+ a 12-frame hangover
 *      so the wake score — which peaks AFTER the VAD — stays inside the window).
 *
 * The engine runs its OWN 16 kHz AudioContext (the models are 16 kHz-trained)
 * fed from the SAME MediaStream the realtime client already opened at 24 kHz, so
 * there is a single mic permission and a single capture — each context resamples
 * the shared track to its own rate.
 *
 * Global (UMD): window.WakeWordEngine. Requires onnxruntime-web's `ort` global
 * to be loaded first.
 */
"use strict";

(function (global) {
  const ort = global.ort;

  const FRAME = 1280; // 80 ms @ 16 kHz — the model's fixed frame size
  const SR = 16000;
  const MEL_WINDOW = 76; // mel frames per embedding
  const MEL_HOP = 8; // mel frames advanced per embedding
  const EMBED_DIM = 96;
  const EMBED_WINDOW = 16; // embeddings per wake score
  const VAD_HANGOVER = 12; // frames speech stays "active" after VAD drops

  const MODEL_FILE = {
    hey_jarvis: "hey_jarvis_v0.1.onnx",
    hey_mycroft: "hey_mycroft_v0.1.onnx",
    hey_rhasspy: "hey_rhasspy_v0.1.onnx",
    alexa: "alexa_v0.1.onnx",
  };

  // Inlined AudioWorklet: re-chunks the resampled mic into exact 1280-sample
  // frames and posts each to the main thread for inference.
  const WORKLET = `
class WakeTap extends AudioWorkletProcessor {
  constructor() { super(); this.buf = new Float32Array(1280); this.pos = 0; }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) for (let i = 0; i < ch.length; i++) {
      this.buf[this.pos++] = ch[i];
      if (this.pos === 1280) { this.port.postMessage(this.buf.slice(0)); this.pos = 0; }
    }
    return true;
  }
}
registerProcessor("wake-tap", WakeTap);`;

  class WakeWordEngine {
    constructor(opts = {}) {
      this.keyword = opts.keyword || "hey_jarvis";
      this.modelFile = opts.modelFile || MODEL_FILE[this.keyword] || MODEL_FILE.hey_jarvis;
      this.modelBase = (opts.modelBase || "models").replace(/\/+$/, "");
      this.ortWasmPath = opts.ortWasmPath || null;
      // Default matches RealtimeConfig.wake_threshold (0.28). openWakeWord's
      // stock demos use ~0.5; that is too deaf for room/laptop mics.
      this.threshold = opts.threshold != null ? opts.threshold : 0.28;
      this.cooldownMs = opts.cooldownMs != null ? opts.cooldownMs : 1200;
      // VAD gate is OFF by default: requiring Silero `speech` simultaneously was
      // suppressing real wake peaks (they land just after the phrase, often past
      // the VAD window) — the #1 sensitivity killer. The server-side wake_word
      // TEXT gate is the precision filter, so the client can afford to be
      // sensitive (a rare false trigger just streams briefly, then the server
      // drops it). When on, the gate uses a LOWER VAD threshold + LONGER hangover.
      this.vadGate = !!opts.vadGate;
      this.vadThreshold = opts.vadThreshold != null ? opts.vadThreshold : 0.35;
      this.hangover = opts.hangover != null ? opts.hangover : 20; // frames (~1.6 s)
      this.debug = !!opts.debug;

      this._handlers = { ready: new Set(), detect: new Set(), error: new Set(), score: new Set() };
      this._loaded = false;
      this._ctx = null;
      this._worklet = null;
      this._source = null;
      this._stream = null;
      this._ownStream = false;
      this._queue = Promise.resolve();
      this._pending = 0;   // frames queued-but-not-yet-scored (backlog guard)
      this._dropped = 0;
      this._cooling = false;
      this._resetState();
    }

    on(event, handler) {
      if (this._handlers[event]) this._handlers[event].add(handler);
      return () => this._handlers[event] && this._handlers[event].delete(handler);
    }
    _emit(event, payload) {
      for (const h of Array.from(this._handlers[event] || [])) {
        try { h(payload); } catch (e) { /* listener errors must not break the pipeline */ }
      }
    }
    _log(...a) { if (this.debug) console.debug("[wake]", ...a); }

    async load() {
      if (this._loaded) return;
      if (!ort) throw new Error("onnxruntime-web (ort) not loaded");
      if (this.ortWasmPath) ort.env.wasm.wasmPaths = this.ortWasmPath;
      // Single-thread: we are not cross-origin isolated (no COOP/COEP), so
      // SharedArrayBuffer is unavailable — force 1 thread rather than let ORT
      // probe and warn. The models are tiny; WASM-SIMD CPU is plenty.
      ort.env.wasm.numThreads = 1;
      const so = { executionProviders: ["wasm"] };
      const url = (f) => `${this.modelBase}/${f}`;
      try {
        this._melspec = await ort.InferenceSession.create(url("melspectrogram.onnx"), so);
        this._embed = await ort.InferenceSession.create(url("embedding_model.onnx"), so);
        this._vad = await ort.InferenceSession.create(url("silero_vad.onnx"), so);
        this._kw = await ort.InferenceSession.create(url(this.modelFile), so);
      } catch (e) {
        throw new Error("wake model load failed: " + (e && e.message ? e.message : e));
      }
      this._loaded = true;
      this._emit("ready");
      this._log("models loaded", this.keyword);
    }

    // Start listening. Pass an existing MediaStream to reuse it; otherwise the
    // engine opens its OWN capture with noiseSuppression/autoGainControl OFF —
    // openWakeWord is trained on near-raw speech, and the realtime client's
    // NS/AGC-processed stream measurably hurts detection. EC stays on so wake
    // still works while Jarvis is speaking (barge-in).
    async start({ mediaStream } = {}) {
      if (!this._loaded) throw new Error("call load() before start()");
      if (this._worklet) return;
      this._resetState();
      if (mediaStream) {
        this._stream = mediaStream;
        this._ownStream = false;
      } else {
        this._stream = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: true, noiseSuppression: false, autoGainControl: false },
        });
        this._ownStream = true;
      }
      // Prefer a 16 kHz context (models are 16 kHz-trained). Some desktop
      // browsers ignore the requested rate and stick to hardware 44.1/48 kHz —
      // keep that context and resample each worklet frame to SR before
      // inference (throwing here used to kill wake on PC → silent PTT with no
      // embedded hold-to-talk control).
      this._ctx = new AudioContext({ sampleRate: SR });
      this._nativeRate = this._ctx.sampleRate;
      this._resampleAcc = []; // native-rate samples awaiting resample→FRAME
      if (Math.abs(this._nativeRate - SR) > 1) {
        this._log(
          "AudioContext sampleRate",
          this._nativeRate,
          "!=",
          SR,
          "; resampling to 16 kHz",
        );
      }
      // A second AudioContext created after the page's first gesture can start
      // suspended on mobile → the worklet never runs → detection silently dies.
      // Sticky activation (the connect click) permits this resume.
      try { await this._ctx.resume(); } catch (e) {}
      this._source = this._ctx.createMediaStreamSource(this._stream);
      const blobUrl = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
      await this._ctx.audioWorklet.addModule(blobUrl);
      this._worklet = new AudioWorkletNode(this._ctx, "wake-tap");
      this._worklet.port.onmessage = (e) => {
        const chunk = e.data;
        // Backlog guard: on a slow CPU the 4-model inference can fall behind
        // real time. Dropping too aggressively (old limit=3) skipped the wake
        // score peak itself — the phrase was spoken, never scored. Allow a
        // deeper queue so peaks still land; scoring stays near-live enough.
        if (this._pending > 8) { this._dropped++; return; }
        const frames = this._nativeFramesToModel(chunk);
        for (const frame of frames) {
          this._pending++;
          this._queue = this._queue
            .then(() => this._process(frame))
            .catch((err) => this._emit("error", err))
            .finally(() => { this._pending--; });
        }
      };
      this._source.connect(this._worklet);
      // The worklet must be pulled by the graph to run; route to destination
      // through a zero gain so it never produces audible output.
      const mute = this._ctx.createGain();
      mute.gain.value = 0;
      this._worklet.connect(mute);
      mute.connect(this._ctx.destination);
      this._log("listening");
    }

    async stop() {
      if (this._worklet) { this._worklet.port.onmessage = null; try { this._worklet.disconnect(); } catch (e) {} this._worklet = null; }
      if (this._source) { try { this._source.disconnect(); } catch (e) {} this._source = null; }
      if (this._ctx && this._ctx.state !== "closed") { try { await this._ctx.close(); } catch (e) {} }
      this._ctx = null;
      // Only stop tracks we opened ourselves — never the realtime client's stream.
      if (this._ownStream && this._stream) this._stream.getTracks().forEach((t) => t.stop());
      this._stream = null;
      this._cooling = false;
    }

    // Pause/resume without tearing down models or the mic (e.g. while the tab is
    // backgrounded, to save battery on phones).
    async pause() { if (this._ctx && this._ctx.state === "running") await this._ctx.suspend(); }
    async resume() { if (this._ctx && this._ctx.state === "suspended") await this._ctx.resume(); }

    get loaded() { return this._loaded; }
    get running() { return !!this._worklet; }

    _resetState() {
      this._melBuf = [];
      this._embedBuf = [];
      for (let i = 0; i < EMBED_WINDOW; i++) this._embedBuf.push(new Float32Array(EMBED_DIM));
      const shape = [2, 1, 64];
      this._vadH = new ort.Tensor("float32", new Float32Array(128), shape);
      this._vadC = new ort.Tensor("float32", new Float32Array(128), shape);
      this._speech = false;
      this._hangover = 0;
      this._pending = 0;
      // Rolling peak so a single dropped/borderline frame after the true max
      // can still trip detection (openWakeWord scores are spiky).
      this._scorePeak = 0;
      this._scorePeakAge = 0;
      this._resampleAcc = [];
    }

    /** Convert a worklet chunk at the AudioContext native rate into zero or
     *  more exact FRAME-sized buffers at the model rate (16 kHz). */
    _nativeFramesToModel(chunk) {
      if (!chunk || !chunk.length) return [];
      const native = this._nativeRate || SR;
      if (Math.abs(native - SR) < 1) {
        // Worklet already posts FRAME-sized buffers at 16 kHz.
        return [chunk];
      }
      // Accumulate native samples, then peel off enough for one 80 ms model frame.
      const acc = this._resampleAcc || (this._resampleAcc = []);
      for (let i = 0; i < chunk.length; i++) acc.push(chunk[i]);
      const needNative = Math.ceil(FRAME * (native / SR));
      const out = [];
      while (acc.length >= needNative) {
        const slice = acc.splice(0, needNative);
        const frame = new Float32Array(FRAME);
        const ratio = native / SR;
        for (let i = 0; i < FRAME; i++) {
          const src = i * ratio;
          const i0 = Math.floor(src);
          const i1 = Math.min(i0 + 1, slice.length - 1);
          const t = src - i0;
          frame[i] = slice[i0] * (1 - t) + slice[i1] * t;
        }
        out.push(frame);
      }
      return out;
    }

    async _runVad(chunk) {
      try {
        const input = new ort.Tensor("float32", chunk, [1, chunk.length]);
        const sr = new ort.Tensor("int64", [BigInt(SR)], []);
        const r = await this._vad.run({ input, sr, h: this._vadH, c: this._vadC });
        this._vadH = r.hn; this._vadC = r.cn;
        return r.output.data[0] > this.vadThreshold;
      } catch (e) { this._emit("error", e); return false; }
    }

    async _process(chunk) {
      // VAD is only computed when it actually gates (saves an inference/frame).
      // With a LONG hangover so a peak landing just after the phrase still counts.
      if (this.vadGate) {
        const voiced = await this._runVad(chunk);
        if (voiced) { this._speech = true; this._hangover = this.hangover; }
        else if (this._speech && --this._hangover <= 0) this._speech = false;
      }

      // Stage 1: frame → melspectrogram (5 frames of 32), normalized.
      const melIn = new ort.Tensor("float32", chunk, [1, FRAME]);
      const melOut = await this._melspec.run({ [this._melspec.inputNames[0]]: melIn });
      const mel = melOut[this._melspec.outputNames[0]].data;
      for (let j = 0; j < mel.length; j++) mel[j] = mel[j] / 10.0 + 2.0;
      for (let j = 0; j < 5; j++) this._melBuf.push(new Float32Array(mel.subarray(j * 32, (j + 1) * 32)));

      // Stage 2+3: slide a 76-frame mel window → embedding → 16-embedding window
      // → wake score, advancing the mel buffer by 8 each step.
      while (this._melBuf.length >= MEL_WINDOW) {
        const flatMel = new Float32Array(MEL_WINDOW * 32);
        for (let j = 0; j < MEL_WINDOW; j++) flatMel.set(this._melBuf[j], j * 32);
        const embOut = await this._embed.run({
          [this._embed.inputNames[0]]: new ort.Tensor("float32", flatMel, [1, MEL_WINDOW, 32, 1]),
        });
        this._embedBuf.shift();
        this._embedBuf.push(new Float32Array(embOut[this._embed.outputNames[0]].data));

        const flatEmb = new Float32Array(EMBED_WINDOW * EMBED_DIM);
        for (let j = 0; j < EMBED_WINDOW; j++) flatEmb.set(this._embedBuf[j], j * EMBED_DIM);
        const kwOut = await this._kw.run({
          [this._kw.inputNames[0]]: new ort.Tensor("float32", flatEmb, [1, EMBED_WINDOW, EMBED_DIM]),
        });
        const score = kwOut[this._kw.outputNames[0]].data[0];
        this._emit("score", { keyword: this.keyword, score }); // for debug overlays

        // Peak-hold (~400 ms of embedding steps): fire if the current score OR
        // a very recent peak clears the threshold while the signal is still
        // elevated. Catches "Hey Jarvis" peaks that land between frames or get
        // slightly attenuated by resampling.
        if (score >= this._scorePeak) {
          this._scorePeak = score;
          this._scorePeakAge = 0;
        } else {
          this._scorePeakAge++;
          if (this._scorePeakAge > 5) { // ~5 embedding hops
            this._scorePeak = score;
            this._scorePeakAge = 0;
          }
        }
        const gateOk = !this.vadGate || this._speech;
        const hit =
          score > this.threshold ||
          (this._scorePeak > this.threshold && score > this.threshold * 0.55);
        if (hit && gateOk && !this._cooling) {
          this._cooling = true;
          this._log("DETECT", this.keyword, score, "peak", this._scorePeak);
          this._emit("detect", {
            keyword: this.keyword,
            score,
            peak: this._scorePeak,
          });
          this._scorePeak = 0;
          this._scorePeakAge = 0;
          setTimeout(() => { this._cooling = false; }, this.cooldownMs);
        }
        this._melBuf.splice(0, MEL_HOP);
      }
    }
  }

  WakeWordEngine.MODEL_FILE = MODEL_FILE;
  global.WakeWordEngine = WakeWordEngine;
})(typeof window !== "undefined" ? window : this);
