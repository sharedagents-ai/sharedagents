# Memory index maintenance

Normal maintenance is automatic: `remember` indexes immediately after a
successful write, memory search lazily indexes detected drift before querying,
and the gateway heartbeat asks the persistent memory service to reindex only
when needed. The Markdown files remain the source of truth; `memory.db` is a
rebuildable search index.

Use the operator CLI when checking or repairing an agent explicitly:

```bash
# Show file/chunk/vector counts, index scope, and whether drift is pending.
sharedagents memory status --agent <agent-name>

# Safe incremental pass: changed files and missing vectors only.
sharedagents memory index --agent <agent-name>

# Rebuild chunks/FTS/vectors, retaining reusable embedding-cache rows.
sharedagents memory index --agent <agent-name> --force

# Also discard the cache and call the configured embedding provider again.
sharedagents memory index --agent <agent-name> --reembed
```

Start with the incremental command. `--force` rebuilds only the derived index;
it does not delete the Markdown memories. `--reembed` is the expensive recovery
path for a changed model or poisoned cache and requires a working embedding
provider.
