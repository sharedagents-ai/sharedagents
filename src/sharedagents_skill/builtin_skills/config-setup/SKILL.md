---
name: config-setup
description: "Guide a user through setting up or reconfiguring sharedagents by conversation. Use when the user asks to configure models, endpoints, agents, MCP/ACP servers, Telegram, gateway networking/auth, permission modes, memory, skills, persona files, or to make sharedagents work on a fresh machine without manually editing JSON."
---

# Config Setup

Use this skill when the operator wants sharedagents configured by talking to
the agent. Your job is to turn their intent into small, reviewable edits
to the right files, then validate the result.

## First Rules

- Prefer the built-in wizard for a blank install: `sharedagents agent init` or
  `sharedagents configure`. Use direct file edits when the user wants a
  specific change, the wizard cannot express it, or a prior config must
  be repaired.
- Never print secrets in the chat. If a token/key is already present,
  refer to the env var or say it is set. Do not paste its value.
- Preserve unrelated config keys. Load JSON, make a minimal patch, and
  keep user-managed fields intact.
- Ask one focused question only when a required value is missing. Good
  questions ask for an endpoint URL, model id, auth env var name,
  gateway exposure choice, or agent name.
- Prefer config-watch reload. Model / `upstream_*` / `models[]` edits to
  `~/.sharedagents/config.json` apply without a process bounce. Persona file
  edits apply on the next turn.
- Auto mode matches OpenCode ``--auto``: do not ask for tool permission.
  Prefer config-watch reload. On systemd, `sharedagents gateway restart`
  bounces NORTH only; `--stack` also bounces the orchestrator. Never
  SIGTERM the orchestrator. You may mention impact/rollback in the
  reply; do not stop for a `question` or permission card.

## Files

Global runtime config:

```text
~/.sharedagents/config.json
```

Owns model deployments and shared runtime settings:

- `models[]`: canonical model list.
- `upstream_endpoint`, `upstream_model`, `upstream_alias`, `backend`:
  fallback/primary model fields, normally matching `models[0]`.
- `upstream_max_context`, `temperature`, `enable_thinking`,
  `permission_mode`, `compact_at_frac`.
- `gateway`: listener host/port/auth and child process limits.
- `mcpServers` or `mcp_servers`: MCP server definitions.
- `acp_agents`: remote ACP agent definitions.
- `brave_api_key`: legacy/direct Brave key; prefer `BRAVE_API_KEY` in
  the environment when possible.

Per-agent config:

```text
~/.sharedagents/agents/<name>/config.json
```

Owns agent identity and behavior:

- `name`, `description`, `engine`, `permission_mode`.
- `system`: optional extra `# SYSTEM` block appended after markdown
  context files.
- `heartbeat_enabled`, `heartbeat_interval_s`.
- `memory_search_enabled`, `memory_flush_enabled`,
  `memory_flush_model`, `remember_synthesizer_model`.
- `hooks_file`, `tools_allowlist`, `skill_auto_invoke`.
- Per-agent overrides for global model fields are allowed, but avoid
  them unless the user explicitly wants that agent to use a different
  default model.

Per-agent context:

```text
~/.sharedagents/agents/<name>/SOUL.md
~/.sharedagents/agents/<name>/USER.md
~/.sharedagents/agents/<name>/TOOLS.md
~/.sharedagents/agents/<name>/AGENTS.md
~/.sharedagents/agents/<name>/MEMORY.md
~/.sharedagents/agents/<name>/HEARTBEAT.md
```

Use these markdown files for persona, operator preferences, local tool
notes, sibling-agent guidance, durable memory, and heartbeat behavior.
Do not cram long prose into `config.json.system` when a context file is
the natural home.

Telegram config:

```text
~/.sharedagents/telegram/config.json
```

Telegram is standalone. Do not add bot tokens to per-agent config.

Standalone TUI config:

```text
~/.config/sharedagents/config.json
```

This is only for the standalone TUI endpoint aliases. It is not where
the gateway reads model deployments. Do not edit it when configuring an
agent model unless the user specifically asks about standalone TUI
endpoint aliases.

## Model Row Shape

Use the canonical `models[]` shape:

```json
{
  "alias": "local",
  "baseUrl": "http://host:8001/v1",
  "id": "model-id",
  "backend": "vllm",
  "contextWindow": 96000,
  "maxTokens": 0,
  "apiKeyEnv": "OPENROUTER_API_KEY",
  "headers": {
    "HTTP-Referer": "https://example.com",
    "X-Title": "sharedagents"
  },
  "capabilities": ["vision"]
}
```

Notes:

- `alias` is the short handle for `/model <alias>`.
- `baseUrl` must be an OpenAI-compatible base ending in `/v1`.
- `id` is the exact model id sent to chat completions.
- `backend` is usually `vllm`, `sglang`, or `unknown`.
- `contextWindow` is total input plus output context.
- `maxTokens` is the model's output limit, like OpenCode's `limit.output`.
  Each response is sent `max_tokens = min(limit, cap)` (reasoning counts
  against it). The cap is 32,000 like OpenCode's; raise it per agent with
  `upstream_max_completion_tokens` or globally with `SHAREDAGENTS_OUTPUT_TOKEN_MAX`.
  `0` means discover the limit: the provider's own `/models` at refresh,
  else the models.dev catalogue, else just the cap.
- `apiKeyEnv` names an environment variable. Do not store bearer keys
  directly in `models[]`.
- Use `capabilities: ["vision"]` only when the model endpoint really
  supports images.

### Native Amazon Bedrock row

Bedrock does **not** use an OpenAI-compatible `baseUrl`. Opt in with the
explicit `api` marker; `baseUrl` is a `bedrock://REGION` display URL and the
AWS SDK derives the real endpoint from `region`. Install the optional SDK
(`pip install 'sharedagents[bedrock]'`) first.

```json
{
  "alias": "bedrock-primary",
  "api": "bedrock-converse",
  "backend": "bedrock",
  "provider": "bedrock",
  "baseUrl": "bedrock://us-east-1",
  "region": "us-east-1",
  "id": "<real foundation id or inference-profile ARN>",
  "auth": "aws-iam",
  "contextWindow": 200000,
  "maxTokens": 0
}
```

- `auth` is `aws-iam` (boto3 chain, recommended) or `bedrock-api-key`
  (bearer token; add `"apiKeyEnv": "AWS_BEARER_TOKEN_BEDROCK"`). Never
  store the token value in the row.
- Never auto-discover the model id; the operator supplies a real one.
- Do not probe `bedrock://.../models`, run a paid call from `doctor --health`,
  or add reasoning/effort settings — Bedrock depth is unsupported in v1.
- Grant `bedrock:InvokeModelWithResponseStream` (and `bedrock:InvokeModel`
  for one-shot paths) and enable model access in the target region.

After changing the primary model, keep top-level fallback fields in sync
with `models[0]`:

```json
{
  "upstream_endpoint": "<models[0].baseUrl>",
  "upstream_model": "<models[0].id>",
  "upstream_alias": "<models[0].alias>",
  "backend": "<models[0].backend>",
  "upstream_max_context": <models[0].contextWindow>
}
```

## Workflow

1. Inspect current state:

```bash
sharedagents agent list
sharedagents gateway status
```

Read only the relevant config/context files. If a file contains secrets,
summarize fields and mask values.

2. Decide the edit path:

- Blank or broad setup: run `sharedagents agent init` or `sharedagents configure`
  if interactive use is possible.
- Narrow change: patch JSON or markdown directly.
- New external CLI agent: let sharedagents scaffold it when possible, then
  edit `SOUL.md`, `TOOLS.md`, `AGENTS.md`, and `config.json`.

3. Patch safely:

- Validate JSON before and after editing.
- Keep unknown keys.
- Use canonical `models[]`; remove legacy `upstream_endpoints` only when
  rewriting model config.
- For gateway exposed beyond localhost, make sure an auth token is set.
- For MCP servers, prefer commands available in the gateway environment
  and include only required env vars.

4. Validate:

```bash
python -m json.tool ~/.sharedagents/config.json >/dev/null
python -m json.tool ~/.sharedagents/agents/<name>/config.json >/dev/null
sharedagents agent status <name>
sharedagents gateway status
```

When a model endpoint changed, probe it without exposing secrets:

```bash
curl -sS http://host:8001/v1/models | head
```

For authenticated providers, use the configured env var:

```bash
curl -sS -H "Authorization: Bearer ${OPENROUTER_API_KEY}" https://openrouter.ai/api/v1/models | head
```

5. Reload:

- Probe `sharedagents gateway status` and the model endpoint after a config
  write. Config-watch reloads `models[]` / `upstream_*` without a bounce.
- If a process bounce is still required, prefer `sharedagents gateway restart`
  (NORTH only on systemd) over `--stack`. Never SIGTERM the orchestrator.
- `sharedagents telegram restart` after Telegram config changes, if that
  command/service exists.
- No restart is usually needed for ordinary next-turn context file
  changes, but starting a fresh session may be needed for external CLI
  engines that cache their initial injected prompt.

## Common Requests

### Add a local vLLM/SGLang model

Ask for `baseUrl`, `id`, alias, backend, and context window if missing.
Append a `models[]` row and decide whether it should become primary. If
primary, move it to index 0 and sync top-level fallback fields.

### Add an API provider

Ask for provider base URL, model id, alias, and the env var name holding
the key. Add `apiKeyEnv`, not the key value. For OpenRouter, add
`headers.X-Title: sharedagents` and `HTTP-Referer` only if the user provides
one.

### Create or tune an agent

Use `~/.sharedagents/agents/<name>/config.json` for structural settings and
the markdown files for behavior. Keep role/personality in `SOUL.md`,
operator facts in `USER.md`, local tool instructions in `TOOLS.md`,
sibling routing in `AGENTS.md`, and recurring background behavior in
`HEARTBEAT.md`.

### Enable skill auto-invoke

Patch the agent config:

```json
{
  "skill_auto_invoke": {
    "enabled": true,
    "threshold": 0.6,
    "max_skills": 3
  }
}
```

Tell the user this silently no-ops if embedding support is unavailable.

### Expose the gateway on the network

Patch the global `gateway` block, not the per-agent `host`:

```json
{
  "gateway": {
    "host": "0.0.0.0",
    "port": 8700,
    "auth_token": "<token>"
  }
}
```

Use an existing token if present. If none exists, generate a strong
random token and report only that it was written; do not paste it unless
the user explicitly asks for it.

### Configure MCP

Patch `mcpServers` in `~/.sharedagents/config.json` unless the existing file
uses `mcp_servers`. A server entry is:

```json
{
  "command": "command-name",
  "args": ["arg1", "arg2"],
  "env": {
    "ENV_VAR": "value"
  }
}
```

After restart, the server's tools appear directly in the agent's tool list
as `mcp__<server>__<tool>` (or use the relevant status/log command) to
confirm the server spawns.

## Final Response

Report:

- Files changed.
- What setting changed, with secrets masked.
- Validation commands run and their result.
- Whether a restart was performed or is still needed.
