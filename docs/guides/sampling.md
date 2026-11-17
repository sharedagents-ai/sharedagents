# Sampling configuration

Sampling is minimal by default: no temperature, top_p, top_k, min_p,
repetition_penalty, presence_penalty, frequency_penalty or seed is sent unless
configured in a model declaration or an explicit sampling layer. Providers
apply their own defaults to omitted fields. Model effort lists and optional
parameters are loaded from the packaged `model_parameters.defaults.json` and
your `model_parameters.json` beside `config.json`. A matching operator entry
replaces the shipped entry. See [model parameter configuration](model-parameters.md)
for the schema, aliases, null/omission rules, and reload behavior.

Add these blocks to the global config, merging with your existing settings:

```json
{
  "sampling": {"temperature": 0.3, "top_p": 0.9},
  "side_sampling": {
    "title": {"temperature": 0.5},
    "distill": {"temperature": 0.2, "presence_penalty": 1.0},
    "compaction": {"temperature": 0.1}
  }
}
```

Precedence, lowest to highest: model-file parameters → legacy top-level
`temperature` → `sampling`
for main turns, or `side_sampling[purpose]` for side calls → the selected
`models[]` row's `sampling` → per-turn `sampling`.
Side calls do not inherit global `sampling`. A higher-layer `null` removes
an inherited value; for example, `"sampling": {"temperature": null}` on one
model row opts that row out of a global temperature. Zero is an explicit value.
Unknown keys, booleans and invalid numeric values are ignored.

The side-call purposes are `distill`, `compaction`, `title`, `teacher`,
`goal_review`, `command_validator`, `vision`, `memory_synth`, `intake` and
`x_search`. Bedrock only sends supported temperature/top_p values; the Codex
Responses transport strips unsupported sampling fields. Explicit hosted-row
settings (including top_k) are sent, so choose fields your provider supports.

To opt a Gemma deployment into its former profile, add this property to that
existing model row:

```json
{
  "sampling": {
    "temperature": 0.4,
    "top_p": 0.9,
    "top_k": 20,
    "min_p": 0.0,
    "frequency_penalty": 0.4,
    "presence_penalty": 0.3,
    "repetition_penalty": 1.05
  }
}
```

Exact shipped local Qwen declarations retain the DRY sampler and repetition
detector. Additional served ids/aliases must be declared explicitly. These
extras can be edited in a model declaration's `parameters` map:

```json
{
  "dry_multiplier": 0.8,
  "dry_base": 1.75,
  "dry_allowed_length": 4,
  "dry_sequence_breakers": ["\n", ".", "!", "?", ":", "—", "```"],
  "repetition_detection": {
    "max_pattern_size": 80,
    "min_pattern_size": 4,
    "min_count": 5
  }
}
```

Optional Qwen tuning belongs in its model row's `sampling`, for example
`{"presence_penalty": 1.0, "min_p": 0.05}` when supported by the deployment.
This replaces the old `SHAREDAGENTS_AUTO_PRESENCE_PENALTY`,
`SHAREDAGENTS_AUTO_MIN_P` and `SHAREDAGENTS_DISTILL_PRESENCE_PENALTY` tuning knobs
(use `side_sampling.distill.presence_penalty` for the latter).

The `/temperature` command, Agent Builder and setup wizard continue writing
the legacy top-level temperature alias; clear it with `null` to use upstream
defaults. A value in `sampling.temperature` takes precedence over that alias.
Raw realtime chat separately accepts `realtime.temperature` as a number or
`null`, and omits it by default.

When migrating an existing configuration, review its top-level temperature
alongside the sampling blocks; omitted values use provider defaults.
