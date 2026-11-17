# Pi RPC engine

Pi runs through the orchestrator's `PiRpcEngine` and `PiRpcAdapter`.
The agent registry exposes it as `pi_rpc`, with the default agent name `pi`.
Existing factory-based gateway and delegation paths load that engine; no ACP
wrapper is required.

Install and authenticate Pi using its [quickstart](https://pi.dev/docs/latest/quickstart).
The existing external-agent scaffolder detects the `pi` executable and creates
the default agent on its next normal invocation. For a custom agent, use
`"engine": "pi_rpc"` in its config. This integration does not install Pi,
change provider credentials, or restart the gateway.

Use a current Pi release supporting `--mode rpc`, `--session-id`, and
`agent_settled`. The CLI's configured provider/model applies unless a model is
explicitly selected in sharedagents. The model selector accepts Pi's `provider/model`
syntax. Authentication stays in Pi's native configuration and environment.

The Model tab's Refresh button queries Pi's `get_available_models` and
`get_state` metadata commands and updates the shared model catalogue. Rows use
provider-qualified IDs, so identical model names from different providers stay
distinct. The same catalogue supplies the engine's model list. Discovery uses
an ephemeral RPC process with tools and session persistence disabled; it sends
no model prompt. A forced refresh bypasses both catalogue caches.

Each turn starts a managed RPC subprocess and resumes the same durable Pi
session using its recorded working directory. Native session files live under
the sharedagents sessions directory's `pi/` folder. The normal engine sidecar records
the session identifier and completion state. Composed context and the canonical
handoff travel in the prompt; acceptance advances the handoff watermark, and
completion waits for both prompt acceptance and `agent_settled`.

Turn details include the reported model/provider, token totals (including
cache reads and writes), wall time, and observed first-token latency when Pi
streams multiple chunks. A progressive single response shows generation
throughput; tool rounds and single-blob responses show whole-turn effective
throughput. Missing output usage and reasoning splits use the shared metrics
estimator. These metrics apply to new turns; existing turn records are unchanged.

Only `auto` mode is supported. Requests for `ask` or `plan` fail before launch.
Manual turns after session budget exhaustion also fail before launch.
The existing external-engine preflight and outbound-secret checks also apply
to Pi. Inline base64 images are supported; other image locations fail explicitly.

RPC errors and premature EOF are turn failures. Stdout uses chunked LF framing
with a configurable 16 MiB record limit; oversized records fail explicitly.
Stderr is drained continuously, retaining only the last 64 KiB in memory.
No stderr content is automatically copied into logs or chat.

Protocol references: [RPC](https://pi.dev/docs/latest/rpc),
[CLI](https://pi.dev/docs/latest/cli),
[events](https://pi.dev/docs/latest/json).
