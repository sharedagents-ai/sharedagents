# Configuration and runtime files

Source code and runtime state are separate. A checkout contains reusable code,
default assets, examples, and tests. A running installation stores its settings,
credentials, sessions, and memory in its runtime home.

## Runtime home

The default home is `~/.sharedagents`. Set `SHAREDAGENTS_HOME` to choose another
location. The older `SHAREDAGENTS_AGENTS_HOME` variable points to an exact
agents directory and takes precedence; avoid combining the two in new setups.

```text
<runtime-home>/
  config.json                Shared model and service configuration
  secrets.env                Shared named secrets
  agents/<name>/             Agent configuration, prompts, and memory
  sessions/                  Session ledgers and derived indexes
  logs/                      Runtime logs
```

Channels and optional services add their own files under this home. The
WhatsApp database, browser authentication data, session logs, memory, and
backups can contain sensitive information. They do not belong in Git.

## Model declarations

The first-run wizard is the easiest way to create a working configuration.
Global agent model declarations live in `<runtime-home>/config.json`, in the
`models` array. A model row uses `alias`, `baseUrl`, and `id`; a hosted provider
can name its credential environment variable with `apiKeyEnv`.

The [local model example](../../examples/config.local-model.json) shows a
generic row. Replace its model ID with an ID actually served by your local
endpoint. Merge the row into your configuration rather than replacing the
whole file. The example contains no credentials and is not a complete
production configuration.

The standalone TUI also has an endpoint settings file under
`$XDG_CONFIG_HOME/sharedagents/config.json` (or `~/.config/sharedagents/config.json`).
Its endpoint aliases are a different schema from the global agent model
registry. Add agent model rows to the runtime home, not to the TUI settings file.

## Secrets

Named secrets are read from the shared `secrets.env`, per-agent
`agents/<name>/secrets.env`, and supported process environment settings. Agent
secrets override shared values. Use the WebUI's secret editor or the documented
provider environment variables; do not put secret values into model examples,
prompts, issue reports, or tracked JSON.

The repository's `.env.example` is a blank template for container environment
variables. Copy it to a private `.env` only for local use. An environment file
is executable shell input if sourced: use a trusted file and never source an
unreviewed attachment.

## Further configuration

- [Sampling precedence](../guides/sampling.md) and [model parameter declarations](../guides/model-parameters.md)
- [Amazon Bedrock credentials and models](../guides/bedrock.md)
- [Forge integration](../guides/forge.md)
- [Voice models and storage](../guides/voice-models.md)
- [Security policy and proxy configuration](../../SECURITY.md)
