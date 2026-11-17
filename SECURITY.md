# Security Policy

SharedAgents is an agent platform with real capabilities: it can run shell
commands, read and write files, drive a browser, and send messages on
Telegram and WhatsApp. A vulnerability here can have an outsized blast
radius, so please report issues privately rather than opening a public
issue.

## Reporting a vulnerability

- **GitHub (preferred):** use the "Report a vulnerability" button under the
  [Security tab](https://github.com/sharedagents-ai/sharedagents/security)
  to file a private advisory.
- **Email:** admin@sharedagents.ai

Please include: the affected version/commit, a description of the impact,
reproduction steps or a proof of concept, and any suggested mitigation.
We aim to acknowledge reports within 72 hours and to coordinate disclosure
with the reporter before a fix is released.

## Scope

Things we treat as vulnerabilities:

- Authentication or authorization bypass in the gateway, web UI or channels
- Remote code execution, command/path injection or sandbox escapes not
  reachable by design
- Token, credential or session-material disclosure (logs, API responses,
  artifacts)
- Privilege escalation between users, or between an agent and its operator

Things that are **expected behavior**, not vulnerabilities:

- The agent can execute commands and edit files — that is the product.
  Deploy it for users you trust, and use per-agent permission modes,
  folder grants and channel allowlists to scope what it may do.
- Prompt injection from content the agent reads. Treat untrusted content
  as untrusted; enable the allowlists and restricted permission modes.

## Deployment notes

- Bind services to loopback unless you have configured authentication —
  the default gateway, orchestrator and web addresses are `127.0.0.1`.
- Never commit `~/.sharedagents/` contents (config, `secrets.env`,
  WhatsApp `neonize.sqlite3`, session logs). The repo ships guards
  (`scripts/check_forbidden_patterns.py`, `tests/test_no_local_references.py`)
  that keep deployment identity out of the tree; run them before a release.
- `SHAREDAGENTS_WEB_SECRET` is required and must be random — the web server
  refuses to start without it.
- Web login is throttled. After 5 failed attempts from one client IP, or
  against one account, that IP or account is blocked for 30 seconds; each
  further miss doubles the block, up to 15 minutes. Blocked requests get
  `429` with `Retry-After` and the password is not checked. A successful
  login clears the account's count. Each web worker keeps its own counters
  in memory (`SHAREDAGENTS_WEB_WORKERS`, default 2), so a client can make
  up to 5 misses per worker before a block, a block holds only on the
  worker that set it, and a restart resets the counters.
- Behind a reverse proxy, set `SHAREDAGENTS_WEB_TRUSTED_PROXIES`
  (comma-separated exact IPs, no CIDR ranges) to the peer address the web
  server actually sees for proxied requests. Under Docker compose with the
  default userland proxy that is usually the project network's gateway, not
  the proxy's own host address; find it with
  `docker network inspect <project>_default -f '{{(index .IPAM.Config 0).Gateway}}'`
  or from the peer address in the web access log after one request.
  Throttling then uses the last `X-Forwarded-For` hop, so exactly
  one trusted proxy hop is supported. The header is ignored from any other
  peer, and `SHAREDAGENTS_WEB_TRUST_PROXY` alone does not trust it. Without
  the setting, every client behind the proxy shares one IP counter.
- Do not expose a WebUI through a public tunnel without an access policy
  (for example Cloudflare Access) in front of it.

## Supported versions

Only the latest release receives security fixes.
