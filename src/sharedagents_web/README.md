# SharedAgents WebUI

The browser interface is packaged as `sharedagents_web`, including its templates
and static assets. It shares gateway conversations with the terminal and
messaging interfaces. Install it with `python -m pip install "sharedagents[web]"`.

The WebUI defaults to `127.0.0.1:8110` and proxies the gateway at
`SHAREDAGENTS_GATEWAY_URL` (default `http://127.0.0.1:8100`). Native browser
accounts and revocable session cookies are the primary authentication path.
External authentication is an optional, explicit deployment setting. Access
checks apply to API routes and conversation ownership.

Web session/authentication state lives under `<runtime-home>/web/` by default;
`SHAREDAGENTS_WEB_DB` can override the database location. Canonical conversation
history is managed by the gateway/session packages. All of this runtime state
is private and belongs outside the source tree.

## Run

For a configured runtime, use the foreground stack launcher:

```bash
python -m sharedagents.serve
```

For a standalone WebUI process, use `sharedagents-web` or
`python -m sharedagents_web`. Those entry points generate or reuse a random
`SHAREDAGENTS_WEB_SECRET` before importing the server. Create a browser account
with `sharedagents-web --create-user user@example.com`; the command prompts
for its password.

The `sharedagents web` subcommands also manage the web process:

```bash
sharedagents web start
sharedagents web status
sharedagents web logs
```

`stop`, `restart`, and `install-service` are lifecycle operations for an existing
installation. Inspect the target instance and its service manager before using
them. See [deployment](../../docs/guides/deployment.md) and
[security](../../SECURITY.md) for container networking, native login, proxy trust,
and the limited loopback debug mode.

## Public URL

For a reverse-proxied installation, set `SHAREDAGENTS_WEB_BASE_URL` to the
browser-facing origin and path, for example `https://example.com/assistant`.
This URL is used for authenticated artifact links. Its path also supplies
`SHAREDAGENTS_WEB_BASE_PATH` when the latter is omitted.

## Feature surface & slash commands

### Incoming attachments

The Web UI saves the original bytes of every accepted upload to the server's
`~/Downloads` folder. Set `SHAREDAGENTS_DOWNLOADS_DIR` to choose another default.
Regular accounts use separate `users/<identity-hash>` subfolders. Files are
private, and duplicate names receive a unique suffix rather than overwriting
existing files. All file types and empty files are accepted, up to 25 MB each.
Reverse proxies must allow at least 30 MB request bodies so multipart overhead
does not reduce the advertised per-file limit.

Users can specify another location in their message and ask the agent to save
or copy the uploaded files there. API clients can send multipart `file` and optional
`destination_dir` fields to `POST /api/attachments`. Relative destinations
are resolved below the default folder. Only
administrators may select destinations outside their download folder.

The response includes `path`, `name`, and `size`. The chat uploader does not
extract document text or place whole datasets in the prompt. It gives the
agent saved paths and asks it to inspect the schema and a small Python sample
before choosing an analysis. For CSV/TSV, the bounded sampler is available as
`python -m sharedagents_core.attachment_sampling <path> ...`; it includes initial
records so metadata rows preceding the headers remain visible.

Clients that explicitly need conversion can send `extract_text=true`, or use
the legacy `POST /api/extract` endpoint. Conversion failures still preserve the
original. Image previews may be resized for the model; the original upload is
preserved in full. Uploading a file does not execute it.

All sharedagents slash commands run through the gateway's shared command runtime
(`sharedagents_command.command_runtime`) — the web frontend carries **zero** command
logic. Type `/` for the command palette/autocomplete; results render
generically by `kind` (text / keyvals / list / table / checklist / board /
detail), and `actions` (buttons) echo a command back to the backend. Chat and
slash commands share one persistent gateway session (via the
`X-Sharedagents-Session-Id` header), so `/context`, `/history`, `/todo`, `/mode`,
`/model`, etc. reflect the conversation you're in.

### Declared capability boundary (web)

`mode=ask` (interactive per-tool permission prompts and the `question` tool)
is available on the web frontend via permission cards and
`POST /api/sessions/<sid>/answer`. Unknown frontends still fail closed.
`plan` and `auto` work on every frontend. The full per-frontend capability
matrix is advertised at `GET /control/v1/{agent}/commands` (`capabilities`
field) and defined in `sharedagents_command.command_runtime.FRONTEND_CAPABILITIES`.

Interactive *panels* that exist as rich modals in the TUI (the live monitor
log-tail) are surfaced on the web as the same data rendered to `list` with
action buttons — the underlying operations are all reachable; the modal-level
interaction is TUI-specific presentation.

## Optional hardware inventory

The Models router view reads `hardware_inventory.json` from `SHAREDAGENTS_HOME`
(default `~/.sharedagents`). Set `SHAREDAGENTS_HARDWARE_INVENTORY` in the sharedagents `.env`
to use another path. A missing or unreadable inventory leaves router discovery available without
inventory GPU details. Each host entry may provide a `label`; otherwise its
short hostname is used. Keep inventory files and deployment labels outside
the repository.
