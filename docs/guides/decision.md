# Decision models

The Decision page starts with empty context and questions. **Load sample** fills
an example; **Run decision** submits it. Plain text becomes a JSON object with a
`text` field; JSON objects and arrays retain their structure. The page displays
Sending, Running, and Done/Error, plus total request latency in milliseconds.
Running means the server accepted the request and is resolving/scoring it; it
does not report a provider's internal queue or token progress.

Discovery probes every configured provider, so its result is cached: the page
serves the last known catalogue immediately while a single background refresh
runs at most once a minute. The header's refresh control (or `?refresh=1` on
the catalog API) re-probes synchronously, as does the first request after a
restart. Score requests resolve their target from the same cache instead of
re-probing providers on every call.

**Change model** opens the decision-model catalogue in a popup. In Questions,
**Choose agent & model** opens the same card catalogue and saved preferences as
chat; Save commits the selection, while Discard leaves it unchanged. After adding
context, **Suggest questions** asks that agent/model to draft up to three typed
questions and appends them to the editable cards (up to eight total). Existing
questions are preserved. Review the suggestions before running the decision.
The request runs in a tracked session with the selected model and normal agent
permissions. **Open suggestion session** exposes progress or any request for
attention. Invalid replies leave the questions unchanged and offer a retry.

The native `decision` tool shares the page's model discovery and validation:

```json
{"action": "list"}
```

```json
{
  "action": "score",
  "state": "I was charged twice. Please help.",
  "questions": {
    "billing": {"type": "noul", "instructions": "Is this a billing issue?"}
  }
}
```

Omit `model` to use the page's default OpenRouter Jev model, or pass a catalogue
ID returned by `list`. No arbitrary URL is accepted. Credentials come from the
selected provider's configured environment variable or shared secret store.
Configured decision models and local scorers are discovered through the same
registry as the page; adding providers does not require changing this tool.

Questions support `choice` (2–16 label/description pairs in `criteria`), `score`
(2–16 ordered descriptions), and `noul` (yes/no probability). Up to eight named
questions and 32 KiB per request are supported. `request_latency_ms` includes
discovery and scoring. The tool is a network operation under existing tool
permissions; hosted models can incur provider charges. Results never execute
the selected action. Restricted profiles must explicitly allow `decision`.

The HTTP scoring endpoint continues to return ordinary JSON by default. Clients
requesting `Accept: application/x-ndjson` receive a `running` line followed by
`done` or `error` with a `result` object. After streaming starts, errors are in
the final event; preflight validation still uses HTTP 422.
