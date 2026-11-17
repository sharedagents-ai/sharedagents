# sharedagents-web

A lightweight, **localhost-only** OpenWebUI-style browser frontend for the
sharedagents agents — the same conversations as the sharedagents TUI, in a browser.

- Binds `127.0.0.1:8110` only (non-loopback `SHAREDAGENTS_WEB_HOST` is rejected).
- Auth is fail-closed via the dashboard verify-token; sessions are scoped
  per user. Loads **zero** external/CDN resources (all JS/CSS vendored under
  `static/vendor/`), with a strict Content-Security-Policy.
- Proxies to the sharedagents gateway at `SHAREDAGENTS_GATEWAY_URL` (default
  `http://127.0.0.1:8100`); sessions persist under
  `SHAREDAGENTS_SESSION_DIR` (default `~/.sharedagents/sessions`).

## Run

Managed via the `sharedagents web` CLI (mirrors `sharedagents gateway` / `sharedagents telegram`):

```bash
sharedagents web start            # daemonize (loopback :8110); bootstraps web/venv + SHAREDAGENTS_WEB_SECRET on first run
sharedagents web status           # pid / uptime / url
sharedagents web logs             # tail -f ~/.sharedagents/logs/web.log
sharedagents web restart
sharedagents web stop
sharedagents web install-service  # install + enable + start the sharedagents-web systemd USER unit (boot-persistent via linger)

sharedagents web start --debug            # SHAREDAGENTS_WEB_DEBUG=1 — skip dashboard-JWT auth (local use)
sharedagents web install-service --debug  # bake debug (no-auth) into the systemd unit
```

`start` auto-creates `web/venv` (flask + requests) and generates/persists
`SHAREDAGENTS_WEB_SECRET` to `~/.sharedagents/.env` (also read by the systemd unit's
`EnvironmentFile`), so sessions survive restarts. The unit
(`~/.config/systemd/user/sharedagents-web.service`) orders itself `After` the
gateway. Bind is **loopback-only** `127.0.0.1:8110`.

Or run it standalone (what the CLI does under the hood):

```bash
cd web
python -m venv venv && venv/bin/pip install -r requirements.txt   # first time
SHAREDAGENTS_WEB_SECRET=$(openssl rand -hex 32) venv/bin/python server.py
```

Then open http://127.0.0.1:8110/ (intended to be embedded as a dashboard
iframe). `venv/` is gitignored.

For a reverse-proxied deployment, set `SHAREDAGENTS_WEB_BASE_URL` to the canonical
browser-facing origin and path, for example:

```bash
SHAREDAGENTS_WEB_BASE_URL=https://example.com/sharedagentsweb
```

The URL is used by the Web UI and authenticated artifact API to generate
attachment links. Its path also supplies `SHAREDAGENTS_WEB_BASE_PATH` when that
variable is omitted; setting the path explicitly remains supported.

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
(`sharedagents_core.command_runtime`) — the web frontend carries **zero** command
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
to use another path. Installations using the former `~/src/hardware_inventory.json`
location must set this override or move the file into their sharedagents home.
A missing or unreadable inventory leaves router discovery available without
inventory GPU details. Each host entry may provide a `label`; otherwise its
short hostname is used. Keep inventory files and deployment labels outside
the repository.
