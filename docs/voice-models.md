# Voice wake-word models

The realtime voice client supports hands-free wake word ("Hey Jarvis" by
default) with an in-browser [openWakeWord](https://github.com/dscripka/openWakeWord)
pipeline running under onnxruntime-web — audio never leaves the device
until the wake word fires.

## Licensing

openWakeWord's code is Apache-2.0, but its pre-trained models are licensed
**CC BY-NC-SA 4.0** (non-commercial, share-alike). They cannot be
distributed inside this MIT-licensed package, so they are downloaded on
demand instead, and using them is your own non-commercial-licence decision.
The MIT-licensed `silero_vad.onnx` does ship bundled; the rest are fetched
per machine.

## Installing the models

```bash
sharedagents web fetch-voice-models
```

This downloads the upstream openWakeWord v0.5.1 release assets
(`https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/`) into
`<sharedagents home>/voice-models/` — `~/.sharedagents/voice-models/`, or
`$SHAREDAGENTS_HOME/voice-models/` when that is set (`--dir` overrides).
Every file is checked against a SHA-256 pinned in
`src/sharedagents_web/voice_models.py`; a mismatch is deleted and never
installed. Run it as the user the web server runs as. It fetches:

- `melspectrogram.onnx` and `embedding_model.onnx` — shared feature models
- `hey_jarvis_v0.1.onnx` — the default wake word

Other upstream wake words are optional:

```bash
sharedagents web fetch-voice-models --keyword hey_mycroft_v0.1 --keyword alexa_v0.1
sharedagents web fetch-voice-models --list
```

The web server resolves `/static/models/<file>` from
`<sharedagents home>/voice-models/` first, then from the bundled package
assets, so no extra wiring is needed. Files are looked up per request: no
restart is needed after a fetch, just reload the page.

## When the models are missing

Nothing breaks: `/static/models/<file>` returns 404, the browser's wake
engine fails to load, and voice falls back to push-to-talk (hold to talk).
The chat shows:

```text
wake init failed: wake model load failed: … — run `sharedagents web fetch-voice-models` on the web host — hands-free disabled; hold to talk
```

## Choosing another wake word

The browser model is picked by `realtime.client_wake.keyword` in
`~/.sharedagents/config.json`: `hey_jarvis` (default), `hey_mycroft`,
`hey_rhasspy` or `alexa`. Fetch the matching model with `--keyword`. This
changes only the in-browser detector; the server's protocol words
(`realtime.wake_word`, default `jarvis`) are configured separately. Set
`realtime.client_wake.enabled` to `false` to turn client-side wake off and
use push-to-talk only.
