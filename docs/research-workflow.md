# Knowledge-store research

The native `research` profile uses typed tools from a configured `knowledge`
MCP server. The store is supplied by SharedData's research workflow build;
SharedData is not imported by the sharedagents client or added to the lean core's
dependencies. Install/run the backend in an interpreter with its own required
dependencies and storage configuration.

## Layers

SharedData is the data layer: it stores sources and model outputs, validates
them against its claim contract and ontology, resolves identity, ranks evidence,
binds market series, measures event reactions and resolves forecasts from
prices. It never calls a model or writes an answer. The agent the user chooses
is the intelligence layer: it extracts, verifies, judges, synthesises and
forecasts on its own model route, and writes back only through the server's
typed actions (each recorded with actor and model). The schema, retrieval
statuses and action list are documented in SharedData's `docs/ontology.md`.

Any MCP-capable agent can be that layer. A native profile gets the tools by
listing `mcp__knowledge__<tool>` names in its allowlist (the bundled `research`
profile does). Claude Code and Codex CLI mount the same stdio server with their
own MCP configuration.

## Connection

Configure the server under `mcpServers` in the operator's runtime configuration:

```json
{
  "mcpServers": {
    "knowledge": {
      "command": "python",
      "args": ["-m", "SharedData.ResearchMCP"],
      "env": {
        "SHAREDDATA_RESEARCH_NAME": "RESEARCH"
      }
    }
  }
}
```

Choose the interpreter and database folders at runtime. `DATABASE_FOLDER` must
be exported to the server process (and `DATABASE_FOLDERS` when needed). No path,
host, credential or model route belongs in the package. This local stdio
connection is governed by OS/runtime configuration; it creates no HTTP service.

The server opens read-only unless its authorized runtime environment sets
`SHAREDDATA_RESEARCH_WRITABLE=1`. Read-only mode supports retrieval and existing
lineage, but cannot create runs/outcomes. Prepare retrieval indexes on the
writer before starting readers. Writes are serialized per operation across
backend processes; reasoning between calls holds no writer lease.

## Native profile

The bundled profile is registered/scaffolded through the normal profile system,
exposed as a native agent in the picker, and advertised in the delegation
roster. Existing operator-owned profile files
and model settings follow the usual preservation rules. No daemon restart,
live profile rewrite or model selection is part of this code change.

The profile has an explicit allowlist of `mcp__knowledge__...` tools. It has no
shell/file-write tools, external web lookup, delegates, training, budget grants,
promotion or deployment. Its model route comes from operator configuration.
The normal MCP permission system still classifies backend read-only annotations
and mutating tools. Tool schemas remain subject to the existing discovery and
allowlist gates.

Tools cover the schema (`ontology`), objects (`find_objects`, `get_object`),
ranked retrieval (`query`, with relevance tags, the subject's own statements
and an event study of the related market series), market functions
(`price_at`, `event_reaction`, `market_impact`), lineage (questions, runs,
forecasts, outcomes) and the ingestion actions (`register_sources`,
`submit_extractions`, `submit_verdicts`, `build`, link decisions,
`bind_series`). Actions and `resolve_forecast_from_market` need a writable
server.

Research checks observed coverage and entity identity, freezes evidence in a
run, and records cited facts separately from conditional inferences. Scenarios
use explicit roles, capacity, power, contracts, costs, dependency exposure and
assumptions. Missing coverage leads to abstention or a stated limitation.
Forecasts have future horizons/resolution rules and later evidence-backed
outcomes. Proposals do not ingest or verify data. Source text remains untrusted
content. General external web research remains available through `general`.

## Foreground application API

```python
from sharedagents_research import KnowledgeClient, ResearchWorkflow


async def research():
    async with KnowledgeClient.connect() as client:
        return await ResearchWorkflow(client).run(
            "sanctions",
            related_to="energy",
            last_hours=24,
            model="verbatim",
            prompt_version="verbatim-v1",
            request_id="energy-research-1",
        )
```

The connection loads the configured `knowledge` server or accepts an explicit
`MCPServerConfig`. One foreground MCP session owns the workflow and cleanup.
No task is detached. The returned object contains question/run/result IDs, the
evidence digest and the stored structured answer.

Without a synthesis callback the run stays open: the workflow returns
`status: "awaiting_synthesis"` with the run ID, the evidence digest, the
evidence status (`linked`, `topic_only`, `no_match`) and match counts. The
calling agent reads the pinned evidence and records its own answer with
`finish_run`. With `respond(snapshot)`, a synchronous or awaited callback returns:

```json
{
  "status": "answered",
  "facts": [{"text": "Supported statement", "claims": ["claim-id"]}],
  "inferences": [{"text": "Conditional implication", "assumptions": ["Explicit assumption"]}],
  "limitations": ["Observed coverage does not certify complete ingestion."]
}
```

The callback owns its model authorization and cost/deadline limits. There is no
fallback provider or hidden model call. With a callback configured, no-coverage runs abstain without calling
it; without a callback they stay open for the agent. Retrying a request that already finished returns its stored result without
calling synthesis again. Unknown interrupted model calls are not automatically
retried; use the backend's budgeted experiments for paid comparisons.

The callback interface requires complete frozen context within its size limit
and fails explicitly otherwise. Interactive agent use can page
`get_evidence`/`get_source_text` instead. Backend refusals, invalid JSON and size
failures propagate as `ResearchError`; no alternate storage is silently created.

Model/prompt versions, grouped evaluation suites, representative samples,
session-wide spending ledgers, validation/held-out comparisons and reviewed
training exports are managed by `KnowledgeBase.research` on the backend. Its
optimizer records proposals; acceptance and production deployment remain
separate operator actions. This client grants no paid budget and starts no
training or ingestion job.
