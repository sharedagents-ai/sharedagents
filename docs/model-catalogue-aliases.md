# Short model names and compatibility routes

A router pool's key is its public model name. Set `upstream_model` when the
replica serves a different name. Existing names can remain as forward-only
pools with `hidden: true`:

```yaml
pools:
  short-model:
    kind: llm
    upstream_model: vendor/long-served-model
    replicas:
      - {name: gpu-host, url: http://127.0.0.1:8010}
  vendor/long-served-model:
    kind: llm
    hidden: true
    replicas: []
    fallbacks:
      - pool: short-model
```

`hidden` only removes a route from `/v1/models` and the Models page's router
catalogue. It does not restrict access: requests using the old name still work.
Configured model rows should use the public name and the router endpoint;
their `display` field controls the picker label. Refresh discovery after a
rename to replace cached catalogue entries. Existing client aliases can stay
unchanged.

The Models page lists all local deployments above the benchmark charts.
Deployments without an Artificial Analysis match remain visible with an
explicit missing-benchmark label; no intelligence or cost metrics are invented.
