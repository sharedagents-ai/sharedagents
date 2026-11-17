# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-11-17

First public SharedAgents platform release.

### Changed

- Refresh the bundled models snapshot from live Artificial Analysis data.
- API balances use compact expandable cards with provider logos and a fill for
  the reported credit remaining. Usage tabs are ordered Usage, API,
  Subscriptions, Prices, Logs.
- The Dev tab forge client is generic: configure it with `FORGE_KIND`,
  `FORGE_URL`, `FORGE_TOKEN` and optional `FORGE_API_URL` /
  `FORGE_USER_AGENT` (Gitea or GitHub). The older `GITEA_*` / `GITHUB_*`
  names still work. See `docs/forge.md`.
- `sharedagents web fetch-voice-models` refuses any wake-word model without
  a pinned SHA-256; the optional `alexa`, `hey_mycroft` and `hey_rhasspy`
  models are now pinned too. No web restart is needed after a fetch.
- Dev-flow Stage-0 criteria IDs are now `REQ-SDLC-0001` … `REQ-SDLC-0010`.
  `POST /api/dev/flows` rejects criteria that use the former prefix; flows
  already stored keep their recorded IDs.

### Security

- Source distributions use neutral ownership metadata instead of the build
  machine's account and group names.
- Failed web logins are throttled per client IP and per account: repeated
  misses get `429` with `Retry-After` and an escalating lockout. Behind a
  reverse proxy, `X-Forwarded-For` is honoured only from peers listed in
  `SHAREDAGENTS_WEB_TRUSTED_PROXIES` (comma-separated IPs).

### Fixed

- Aborted chat turns keep their Turn details disclosure and available metrics;
  the aborted status no longer adds a separate command-output message.
- The Users admin listing reports each account's last sign-in again
  (`last_seen` was always 0).
- With `FORGE_TOKEN` set and no `FORGE_URL`, the forge client follows a
  legacy `GITEA_BASE` (or `GITHUB_BASE`) before defaulting to GitHub, so a
  Gitea token is never sent to `api.github.com`.
- An explicit `FORGE_URL` decides the forge backend: it is never paired with
  the other forge's legacy token (`GITHUB_TOKEN` with a Gitea URL or
  `GITEA_TOKEN` with github.com); the Dev tab reports the missing token
  instead.

### Documentation

- `docs/voice-models.md` covers the wake-word model source, licence split
  (Apache-2.0 code, CC BY-NC-SA 4.0 models), storage under
  `SHAREDAGENTS_HOME`, the push-to-talk fallback when models are missing, and
  the real `realtime.client_wake.*` settings; the README `[realtime]` extra
  points to the fetch step.

### Added

- Multi-agent orchestrator with a turn loop, multi-modal engine adapters
  (CLI, ACP, A2A) and tool dispatch.
- Terminal TUI (Textual), browser WebUI (Sessions, Usage, Agent Monitor,
  Tasks, Loops, Topics, Dev, Goals, Voice, Agent Builder tabs).
- Edge gateway: auth, ingress queue with fast ACK, REST cache, SSE streaming.
- Single-file `<sid>.jsonl` session ledger with SQLite cache.
- Telegram and WhatsApp (linked-device self-chat) channels.
- Realtime voice client with in-browser wake word and push-to-talk.
- Bundled MCP servers (memory, search, filesystem, terminal); memory is a
  builtin MCP capability backed by Markdown + sqlite-vec.
- Optional extras: `full` (doc/media tools), `web`, `stt`, `bedrock`,
  `whatsapp`, `vllm-router`.
- Optional local vLLM replica router (`sharedagents-vllm`) with health
  probes, affinity routing and advisory decision pools.
- Deliberate Windows support (`.bat` launchers, CRLF via `.gitattributes`).
