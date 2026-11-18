# Configuration examples

These files contain generic examples, not production settings. They use
loopback or reserved example addresses and contain no credential values.

- `config.local-model.json`: one model declaration for an OpenAI-compatible
  server on loopback. Replace the model ID with the served ID and merge the
  declaration into your runtime configuration.
- The repository-root `.env.example`: blank provider variables for a local
  container environment file. Fill only the variables you need in a private
  copy outside version control.

Use the first-run wizard for a complete initial configuration. See
[configuration](../docs/getting-started/configuration.md) for the distinction
between runtime agent settings and standalone TUI endpoint settings.
