# Model effort and optional request parameters

SharedAgents reads `model_parameters.json` beside the runtime's `config.json`.
The usual SharedAgents home environment overrides apply. The installed package
also supplies `sharedagents_model/data/model_parameters.defaults.json`.
Installation and upgrades never write the operator file.

Copy an entry from the shipped file, or add a new entry:

```json
{
  "version": 1,
  "models": [
    {
      "provider": "vllm",
      "model": "example-model",
      "aliases": ["example-served-alias"],
      "effort": {
        "values": ["low", "medium", "custom-depth"],
        "wire": "reasoning_effort"
      },
      "parameters": {
        "temperature": 0,
        "chat_template_kwargs.enable_thinking": false
      }
    }
  ]
}
```

The picker and request builder use the same declaration. Tokens retain their
spelling and order. Adding a model, alias, or token does not require Python edits.

## Matching and precedence

An exact operator entry wins over an exact shipped entry. It replaces the entire
entry, including aliases, effort defaults, parameters, and mode overrides. Keys
removed from the operator entry are not restored from the shipped file. An empty
`models` list overrides no models.

Match `provider` and the model id or an explicitly listed alias. Id matching is
case-insensitive; effort tokens are case-sensitive. There is no family, substring,
or punctuation-based alias inference. A row's explicit `provider` is used first,
then its local backend identity, then an exact recognized provider hostname (or
its backend/owner when the hostname is unknown). This distinguishes hosted
providers that share the same OpenAI-compatible transport.

When neither file matches, an existing model-row `effort` declaration or provider
advertisement can supply a capability. For compatibility, a row's existing
`effort.values` declaration without `wire` uses `reasoning_effort`. A file entry
always wins over row metadata and cached discovery. Provider discovery runs only
in the existing discovery paths, never synchronously during request assembly.

When an operator replaces a shipped entry, repeat any aliases to retain them.
Undeclared quantized ids and served aliases no longer inherit a model-family
policy. Add their exact ids to your operator file.

## Omission and selection

- A missing `effort`, `effort: null`, or `effort: {"values": []}` on a matched entry
  disables effort. Discovery cannot restore it.
- A populated effort requires `values` and `wire`. `values` contains unique,
  nonempty strings. `default`, when set, must be in the list.
- Without an explicit selection or a configured default, no effort is sent.
  Provider-advertised defaults are informational and do not create a selection.
- To opt into a default, add `"default": "medium"` to the effort object. Remove it
  or set it to null to restore omission. `/effort default` clears the session
  override; an explicitly configured agent effort still takes precedence over
  the model-file default.
- An explicit selection outside the current list is omitted with a diagnostic.
  It is never clamped to a stronger level or replaced by the configured default.
  Stored preferences are not rewritten when their token becomes unavailable.
- The effort tokens `default`, `auto`, `reset`, and `clear` are reserved for
  clearing overrides. Other tokens do not need to belong to a Python enum.
- Missing/null optional parameter values are omitted. Valid zero and false values
  are retained. Empty template containers are not emitted.

Other explicitly configured sampling follows the existing order above the file's
base parameters: legacy temperature, global/purpose sampling, model-row sampling,
and per-turn sampling. A null in a sampling map removes a lower-layer value.
Token/context safety ceilings, messages, tools, authentication, and routing remain
separate from this optional-parameter configuration.

## Supported destinations and values

Standard effort uses `wire: "reasoning_effort"`, emitted once at the top level.
Local vLLM/SGLang numeric-budget control uses
`wire: "chat_template_kwargs.thinking_budget"` and requires a complete mapping:

```json
{
  "values": ["quick", "deep"],
  "wire": "chat_template_kwargs.thinking_budget",
  "wire_values": {"quick": 2048, "deep": 8192}
}
```

A standard effort may also declare a complete `wire_values` mapping of displayed
tokens to string wire tokens. No family-based conversion is applied.

`parameters` accepts existing sampling fields: `temperature`, `top_p`, `top_k`,
`min_p`, `repetition_penalty`, `presence_penalty`, `frequency_penalty`, and `seed`.
Local entries additionally accept `chat_template_kwargs.enable_thinking`,
`chat_template_kwargs.thinking_budget`, `dry_multiplier`, `dry_base`,
`dry_allowed_length`, `dry_sequence_breakers`, and `repetition_detection` (positive
integer `min_pattern_size`, `max_pattern_size`, and `min_count`). A parameter may
not collide with the effort destination. Native Bedrock effort remains unsupported;
existing native transport restrictions remain in effect.

Optional `modes` can override `parameters` and the effort `default` for `plan`,
`ask`, or `auto`. These overlays apply within the winning entry; a null removes a
base value. They do not inherit mode settings from the replaced shipped entry.

## Reload and errors

Each process loads a validated snapshot at initialization/first use. Apply edits
through the normal application restart; request assembly does not poll files.
Embedding applications can call `clear_model_parameters_cache()` at their normal
configuration reload boundary. All participating processes need to reload.

A present unreadable, malformed, or invalid file raises `ModelParametersError`
with the file/entry/key location. It does not silently restore shipped defaults.
Unknown keys, duplicate model/alias selectors, unsupported destinations, invalid
numeric values, and out-of-list defaults are errors. Errors do not echo complete
configuration values. Keep credentials in the existing secret configuration.

## Migration from Python policy

| Previous policy | File declaration / new behavior |
|---|---|
| Qwen3.8 local effort fields | Shipped local entries declare `low`, `medium`, `xhigh`, default `medium`, top-level effort |
| Qwen3.6 local budget ladder | Shipped `wire_values`: 2048 / 8192 / 16384 |
| Gemma local budget ladder | Shipped `wire_values`: 512 / 1024 / 2048 |
| DeepSeek host-wide cap | Exact shipped model entries declare `low`, `high`, default `high`; operator entries can replace them |
| Kimi model-name temperature override | Provider-scoped shipped parameter, overridable/removable like other settings |
| Implicit agent effort `max` | No implicit agent selection; existing explicit agent configuration remains authoritative |
| Unknown local id passing a session token through | Omit effort unless configured or advertised |
| Unsupported effort clamped/defaulted | Omit with a diagnostic |

The shipped file also holds the existing local optional repetition/template
parameters. Model-specific response parsing and token/context safety limits remain
in code. No checkpoint, serving-image, or service configuration change is required
by the source fix itself.
